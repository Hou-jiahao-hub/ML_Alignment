"""Single-helix fit to local measurements and its Newton-decrement gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .gauge import (
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    unpack_gauge_fixed_track_parameters,
)
from .gauss_newton import (
    GAUSS_NEWTON_INITIAL_DAMPING,
    GAUSS_NEWTON_MAX_ITERS,
    GAUSS_NEWTON_MIN_DAMPING,
    GAUSS_NEWTON_TOLERANCE,
    _merge_gauss_newton_diagnostics,
)
from .trajectory import intersect_plane

MEASUREMENT_VARIANCE_FLOOR = 1.0e-8
MEASUREMENT_NEWTON_DECREMENT_STATIONARY = 1.0e-6
MEASUREMENT_FIT_CHUNK_ROWS = 2048


@dataclass(frozen=True)
class MeasurementFitInputs:
    """Tensors required by the production single-helix measurement fit."""

    measured: torch.Tensor
    variance: torch.Tensor
    component_mask: torch.Tensor
    centers: torch.Tensor
    rotations: torch.Tensor

    def __post_init__(self) -> None:
        if self.measured.ndim != 3 or self.measured.shape[-1] != 2:
            raise ValueError("measured must have shape [tracks, hits, 2].")
        if self.variance.shape != self.measured.shape:
            raise ValueError("variance must match measured shape.")
        if self.component_mask.shape != self.measured.shape:
            raise ValueError("component_mask must match measured shape.")
        expected_centers = (*self.measured.shape[:2], 3)
        if self.centers.shape != expected_centers:
            raise ValueError("centers must have shape [tracks, hits, 3].")
        expected_rotations = (*self.measured.shape[:2], 3, 3)
        if self.rotations.shape != expected_rotations:
            raise ValueError("rotations must have shape [tracks, hits, 3, 3].")
        if self.component_mask.dtype != torch.bool:
            raise ValueError("component_mask must be boolean.")
        if self.measured.shape[0] == 0:
            raise ValueError("At least one measurement-fit row is required.")
        if not bool(self.component_mask.any(dim=(1, 2)).all()):
            raise ValueError("Every measurement-fit row needs an active component.")

    def slice(self, start: int, stop: int) -> "MeasurementFitInputs":
        return MeasurementFitInputs(
            measured=self.measured[start:stop],
            variance=self.variance[start:stop],
            component_mask=self.component_mask[start:stop],
            centers=self.centers[start:stop],
            rotations=self.rotations[start:stop],
        )

    def to(
        self, device: str | torch.device, dtype: torch.dtype
    ) -> "MeasurementFitInputs":
        return MeasurementFitInputs(
            measured=self.measured.to(device=device, dtype=dtype),
            variance=self.variance.to(device=device, dtype=dtype),
            component_mask=self.component_mask.to(device=device),
            centers=self.centers.to(device=device, dtype=dtype),
            rotations=self.rotations.to(device=device, dtype=dtype),
        )


def measurement_local_prediction(
    parameters: torch.Tensor,
    inputs: MeasurementFitInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one single helix per row onto each measurement surface."""
    if parameters.shape != (inputs.measured.shape[0], 6):
        raise ValueError("parameters must have shape [tracks, 6].")
    expanded = parameters[:, None, :].expand(-1, inputs.measured.shape[1], -1)
    _path, intersections, valid = intersect_plane(
        expanded, inputs.centers, inputs.rotations[..., :, 2], n_newton=12
    )
    local3 = torch.matmul(
        inputs.rotations.transpose(-1, -2),
        (intersections - inputs.centers).unsqueeze(-1),
    ).squeeze(-1)
    return local3[..., :2], valid


def measurement_standardized_residual(
    theta: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inputs: MeasurementFitInputs,
) -> torch.Tensor:
    """Return flattened whitened residuals in the ``F = 0.5 chi2`` gauge."""
    return _measurement_standardized_residual_tensors(
        theta,
        frame,
        inputs.measured,
        inputs.variance,
        inputs.component_mask,
        inputs.centers,
        inputs.rotations,
    )


def _measurement_standardized_residual_tensors(
    theta: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    measured: torch.Tensor,
    variance: torch.Tensor,
    component_mask: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    parameters = unpack_gauge_fixed_track_parameters(theta, frame)
    expanded = parameters[:, None, :].expand(-1, measured.shape[1], -1)
    _path, intersections, valid = intersect_plane(
        expanded, centers, rotations[..., :, 2], n_newton=12
    )
    local3 = torch.matmul(
        rotations.transpose(-1, -2),
        (intersections - centers).unsqueeze(-1),
    ).squeeze(-1)
    residual = (measured - local3[..., :2]) / variance.clamp_min(
        MEASUREMENT_VARIANCE_FLOOR
    ).sqrt()
    active = component_mask & valid.unsqueeze(-1)
    return torch.where(active, residual, torch.zeros_like(residual)).flatten(1)


def _single_measurement_residual(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    measured: torch.Tensor,
    variance: torch.Tensor,
    component_mask: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    return _measurement_standardized_residual_tensors(
        theta.unsqueeze(0),
        (origin.unsqueeze(0), axis0.unsqueeze(0), axis1.unsqueeze(0)),
        measured.unsqueeze(0),
        variance.unsqueeze(0),
        component_mask.unsqueeze(0),
        centers.unsqueeze(0),
        rotations.unsqueeze(0),
    ).squeeze(0)


# Forward mode: five parameters against 2H residuals, so five JVPs replace 2H
# VJPs that would each backpropagate through the unrolled plane-intersection
# Newton iterations.
_BATCHED_MEASUREMENT_RESIDUAL_JACOBIAN = torch.func.vmap(
    torch.func.jacfwd(_single_measurement_residual, argnums=0)
)


def measurement_residual_jacobian(
    theta: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    inputs: MeasurementFitInputs,
) -> torch.Tensor:
    """Differentiate the production measurement residual row by row."""
    return _BATCHED_MEASUREMENT_RESIDUAL_JACOBIAN(
        theta,
        *frame,
        inputs.measured,
        inputs.variance,
        inputs.component_mask,
        inputs.centers,
        inputs.rotations,
    )


def _require_well_constrained_measurement_rows(
    parameters: torch.Tensor,
    inputs: MeasurementFitInputs,
    jacobian: torch.Tensor,
    stage: str,
) -> None:
    """Reject rows for which LM damping would manufacture a false solution."""
    finite_jacobian = torch.isfinite(jacobian).all(dim=(1, 2))
    if not bool(finite_jacobian.all()):
        count = int((~finite_jacobian).sum())
        raise FloatingPointError(
            f"Non-finite {stage} measurement Jacobian on {count} rows."
        )

    _local, intersection_valid = measurement_local_prediction(parameters, inputs)
    active = inputs.component_mask & intersection_valid.unsqueeze(-1)
    active_count = active.sum(dim=(1, 2))
    no_valid = active_count == 0
    if bool(no_valid.any()):
        raise ValueError(
            f"{stage.capitalize()} measurement fit has {int(no_valid.sum())} rows "
            "with no valid measurement components."
        )

    n_parameters = jacobian.shape[-1]
    rank = torch.linalg.matrix_rank(jacobian)
    underconstrained = (active_count < n_parameters) | (rank < n_parameters)
    if bool(underconstrained.any()):
        raise ValueError(
            f"{stage.capitalize()} measurement fit has "
            f"{int(underconstrained.sum())} underconstrained rows; each row "
            f"requires rank {n_parameters} in the unregularized Jacobian."
        )


def _fit_helix_measurements_gauss_newton_block(
    initial: torch.Tensor,
    inputs: MeasurementFitInputs,
    max_iters: int,
    damping: float,
    tolerance: float,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    frame = reference_gauge_frame(initial)
    theta = pack_gauge_fixed_track_parameters(initial).clone()

    def residual_of(values: torch.Tensor) -> torch.Tensor:
        return measurement_standardized_residual(values, frame, inputs)

    def jacobian_of(values: torch.Tensor) -> torch.Tensor:
        return measurement_residual_jacobian(values, frame, inputs)

    residual = residual_of(theta)
    current = 0.5 * residual.square().sum(dim=1)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Non-finite initial Gauss-Newton objective.")
    initial_jacobian = jacobian_of(theta)
    _require_well_constrained_measurement_rows(
        initial, inputs, initial_jacobian, "initial"
    )
    initial_objective = float(current.mean())
    identity = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)
    level = torch.full(
        (theta.shape[0], 1, 1),
        float(damping),
        dtype=theta.dtype,
        device=theta.device,
    )
    converged = False
    iterations = 0
    change = 0.0
    gradient = torch.zeros_like(theta)
    for iterations in range(1, max_iters + 1):
        jacobian = initial_jacobian if iterations == 1 else jacobian_of(theta)
        hessian = jacobian.transpose(1, 2) @ jacobian
        gradient = (
            jacobian.transpose(1, 2) @ residual.unsqueeze(-1)
        ).squeeze(-1)
        scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
        regularized = hessian + level * scale[:, None, None] * identity
        step = torch.linalg.solve(regularized, -gradient.unsqueeze(-1)).squeeze(-1)
        candidate = theta + step
        candidate_residual = residual_of(candidate)
        candidate_objective = 0.5 * candidate_residual.square().sum(dim=1)
        improved = candidate_objective < current
        theta = torch.where(improved.unsqueeze(-1), candidate, theta)
        # The next iteration's residual is the accepted candidate's, already known.
        residual = torch.where(improved.unsqueeze(-1), candidate_residual, residual)
        level = torch.where(
            improved[:, None, None],
            (level * 0.1).clamp_min(GAUSS_NEWTON_MIN_DAMPING),
            level * 10.0,
        )
        decrease = (current - candidate_objective).clamp_min(0.0) / current.abs().clamp_min(1.0)
        current = torch.where(improved, candidate_objective, current)
        change = float(decrease.max())
        if change < tolerance:
            converged = True
            break
    fitted = unpack_gauge_fixed_track_parameters(theta, frame).detach()
    return fitted, {
        "converged": bool(converged),
        "iterations": int(iterations),
        "initial_objective": initial_objective,
        "final_objective": float(current.mean()),
        "max_relative_decrease": float(change),
        "max_gradient_norm": float(gradient.norm(dim=1).max()),
    }


def fit_helix_measurements_gauss_newton(
    initial: torch.Tensor,
    inputs: MeasurementFitInputs,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
    chunk_rows: int = MEASUREMENT_FIT_CHUNK_ROWS,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict]:
    """Fit independent single helices to local measurements in row chunks."""
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial must have shape [tracks, 6].")
    if initial.shape[0] != inputs.measured.shape[0]:
        raise ValueError("initial and measurement inputs must have matching rows.")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    initial = initial.to(device=device, dtype=dtype)
    inputs = inputs.to(device=device, dtype=dtype)
    step = initial.shape[0] if chunk_rows <= 0 else chunk_rows
    fitted_blocks: list[torch.Tensor] = []
    diagnostics_blocks: list[dict[str, float | int | bool]] = []
    weights: list[int] = []
    for start in range(0, initial.shape[0], max(step, 1)):
        stop = min(start + max(step, 1), initial.shape[0])
        fitted, diagnostics = _fit_helix_measurements_gauss_newton_block(
            initial[start:stop],
            inputs.slice(start, stop),
            max_iters=max_iters,
            damping=damping,
            tolerance=tolerance,
        )
        fitted_blocks.append(fitted)
        diagnostics_blocks.append(diagnostics)
        weights.append(stop - start)
    fitted = torch.cat(fitted_blocks, dim=0)
    decrement = measurement_newton_decrement(
        fitted, inputs, chunk_rows=chunk_rows, device=device, dtype=dtype
    )
    return fitted, {
        "solver": "gauss-newton",
        "convergence": _merge_gauss_newton_diagnostics(
            diagnostics_blocks, weights
        ),
        "stationarity": measurement_stationarity_report(decrement),
    }


def measurement_newton_decrement(
    parameters: torch.Tensor,
    inputs: MeasurementFitInputs,
    chunk_rows: int = MEASUREMENT_FIT_CHUNK_ROWS,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return per-row ``g^T H^-1 g`` for the single-helix measurement fit."""
    if parameters.shape != (inputs.measured.shape[0], 6):
        raise ValueError("parameters must have shape [tracks, 6].")
    parameters = parameters.to(device=device, dtype=dtype)
    inputs = inputs.to(device=device, dtype=dtype)
    step = parameters.shape[0] if chunk_rows <= 0 else chunk_rows
    pieces: list[torch.Tensor] = []
    for start in range(0, parameters.shape[0], max(step, 1)):
        stop = min(start + max(step, 1), parameters.shape[0])
        block_parameters = parameters[start:stop]
        block_inputs = inputs.slice(start, stop)
        frame = reference_gauge_frame(block_parameters)
        theta = pack_gauge_fixed_track_parameters(block_parameters)
        residual = measurement_standardized_residual(theta, frame, block_inputs)
        jacobian = measurement_residual_jacobian(theta, frame, block_inputs)
        _require_well_constrained_measurement_rows(
            block_parameters, block_inputs, jacobian, "stationarity"
        )
        hessian = jacobian.transpose(1, 2) @ jacobian
        gradient = (
            jacobian.transpose(1, 2) @ residual.unsqueeze(-1)
        ).squeeze(-1)
        scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
        identity = torch.eye(
            theta.shape[1], dtype=theta.dtype, device=theta.device
        )
        regularized = (
            hessian
            + GAUSS_NEWTON_MIN_DAMPING * scale[:, None, None] * identity
        )
        step_vector = torch.linalg.solve(
            regularized, gradient.unsqueeze(-1)
        ).squeeze(-1)
        pieces.append((gradient * step_vector).sum(dim=1).detach())
    return torch.cat(pieces, dim=0)


def measurement_stationarity_report(
    decrement: torch.Tensor,
    threshold: float = MEASUREMENT_NEWTON_DECREMENT_STATIONARY,
) -> dict[str, float | int]:
    """Summarize measurement-fit Newton decrements for the production gate."""
    finite = decrement[torch.isfinite(decrement)]

    def quantile(value: float) -> float:
        return float(finite.quantile(value)) if finite.numel() else float("nan")

    return {
        "stationary_threshold": float(threshold),
        "non_stationary_row_fraction": (
            float((decrement > threshold).to(torch.float64).mean())
            if decrement.numel()
            else 0.0
        ),
        "n_non_finite": int(decrement.numel() - finite.numel()),
        "newton_decrement_p50": quantile(0.50),
        "newton_decrement_p90": quantile(0.90),
        "newton_decrement_p99": quantile(0.99),
        "newton_decrement_max": (
            float(finite.max()) if finite.numel() else float("nan")
        ),
    }
