"""ACTS-independent differentiable six-parameter helix mathematics.

The physical parameter order is ``(x0, y0, z0, psi0, tan_lambda, rho)``.
Path length is transverse arc length in millimetres and ``rho`` is signed
curvature in inverse millimetres.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


CURVATURE_SCALE = 1.0e-4
S_MAX = 1200.0
PLANE_PARALLEL_EPS = 1.0e-6
PLANE_RESIDUAL_TOLERANCE = 1.0e-3
CLOSEST_POINT_CURVATURE_EPS = 1.0e-9
GAUSS_NEWTON_MAX_ITERS = 60
GAUSS_NEWTON_TOLERANCE = 1.0e-6
GAUSS_NEWTON_INITIAL_DAMPING = 1.0e-8
GAUSS_NEWTON_MIN_DAMPING = 1.0e-12
GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE = 1.0e-12
POINT_FIT_MAX_NON_STATIONARY_ROW_FRACTION = 1.0e-3
MEASUREMENT_VARIANCE_FLOOR = 1.0e-8
MEASUREMENT_NEWTON_DECREMENT_STATIONARY = 1.0e-6
MEASUREMENT_FIT_CHUNK_ROWS = 2048


def _sinc(value: torch.Tensor) -> torch.Tensor:
    """Return ``sin(value) / value`` with a differentiable zero limit."""
    return torch.sinc(value / torch.pi)


def helix_point(track: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Evaluate signed-curvature helices at transverse arc length ``s``."""
    x0, y0, z0, psi0, tanl, rho = track.unbind(-1)
    angle = rho * s
    half_sinc = _sinc(angle / 2.0)
    x = x0 + s * torch.cos(psi0 + angle / 2.0) * half_sinc
    y = y0 + s * torch.sin(psi0 + angle / 2.0) * half_sinc
    z = z0 + tanl * s
    return torch.stack((x, y, z), dim=-1)


def helix_tangent(track: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Differentiate :func:`helix_point` with respect to transverse arc length."""
    _x0, _y0, _z0, psi0, tanl, rho = track.unbind(-1)
    psi = psi0 + rho * s
    return torch.stack(
        (torch.cos(psi), torch.sin(psi), tanl * torch.ones_like(s)), dim=-1
    )


def accept_newton_candidate(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    valid: torch.Tensor,
    candidate_ok: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept feasible Newton steps without reviving an invalid path."""
    updated_valid = valid & candidate_ok
    return torch.where(updated_valid, candidate, previous), updated_valid


def closest_helix_points(
    track: torch.Tensor,
    points: torch.Tensor,
    n_newton: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return closest points on each helix to reference global points."""
    delta = points - track[..., :3]
    s = delta[..., 0] * torch.cos(track[..., 3]) + delta[..., 1] * torch.sin(
        track[..., 3]
    )
    valid = torch.isfinite(s)
    s = torch.where(valid, s, torch.zeros_like(s))
    for _ in range(n_newton):
        position = helix_point(track, s)
        tangent = helix_tangent(track, s)
        displacement = position - points
        rho = track[..., 5]
        psi = track[..., 3] + rho * s
        second = torch.stack(
            (-rho * torch.sin(psi), rho * torch.cos(psi), torch.zeros_like(rho)),
            dim=-1,
        )
        first_derivative = torch.sum(displacement * tangent, dim=-1)
        second_derivative = torch.sum(
            tangent * tangent + displacement * second, dim=-1
        )
        safe = torch.where(
            second_derivative.abs() > CLOSEST_POINT_CURVATURE_EPS,
            second_derivative,
            torch.where(
                second_derivative >= 0,
                torch.full_like(second_derivative, CLOSEST_POINT_CURVATURE_EPS),
                torch.full_like(second_derivative, -CLOSEST_POINT_CURVATURE_EPS),
            ),
        )
        candidate = s - first_derivative / safe
        candidate_ok = torch.isfinite(candidate) & (candidate.abs() <= S_MAX)
        s, valid = accept_newton_candidate(s, candidate, valid, candidate_ok)
    fitted = helix_point(track, s)
    valid = valid & torch.isfinite(fitted).all(dim=-1)
    fitted = torch.where(valid.unsqueeze(-1), fitted, track[..., :3])
    return fitted, s, valid


def intersect_plane(
    track: torch.Tensor,
    center: torch.Tensor,
    normal: torch.Tensor,
    n_newton: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect helices with planes and return ``(s, point, valid_mask)``."""
    tangent0 = helix_tangent(track, torch.zeros_like(track[..., 0]))
    denominator0 = torch.sum(tangent0 * normal, dim=-1)
    straight_valid = denominator0.abs() >= PLANE_PARALLEL_EPS
    safe_denominator = torch.where(
        straight_valid,
        denominator0,
        torch.where(
            denominator0 >= 0,
            torch.ones_like(denominator0),
            -torch.ones_like(denominator0),
        ),
    )
    s = torch.sum((center - track[..., :3]) * normal, dim=-1) / safe_denominator
    tangent_at_initial = helix_tangent(track, s)
    initially_valid = (
        straight_valid
        & torch.isfinite(s)
        & (
            torch.sum(tangent_at_initial * normal, dim=-1).abs()
            >= PLANE_PARALLEL_EPS
        )
    )
    s = torch.where(initially_valid, s, torch.zeros_like(s))
    iteration_valid = initially_valid
    for _ in range(n_newton):
        point = helix_point(track, s)
        tangent = helix_tangent(track, s)
        residual = torch.sum((point - center) * normal, dim=-1)
        derivative = torch.sum(tangent * normal, dim=-1)
        derivative_safe = torch.where(
            derivative.abs() >= PLANE_PARALLEL_EPS,
            derivative,
            torch.where(
                derivative >= 0,
                torch.full_like(derivative, PLANE_PARALLEL_EPS),
                torch.full_like(derivative, -PLANE_PARALLEL_EPS),
            ),
        )
        candidate = s - residual / derivative_safe
        candidate_ok = (
            torch.isfinite(candidate)
            & (candidate.abs() <= S_MAX)
            & (derivative.abs() >= PLANE_PARALLEL_EPS)
        )
        s, iteration_valid = accept_newton_candidate(
            s, candidate, iteration_valid, candidate_ok
        )
    point = helix_point(track, s)
    final_residual = torch.sum((point - center) * normal, dim=-1).abs()
    finite = torch.isfinite(s) & torch.isfinite(point).all(dim=-1)
    valid = (
        iteration_valid
        & finite
        & (s.abs() <= S_MAX)
        & (final_residual <= PLANE_RESIDUAL_TOLERANCE)
    )
    point = torch.where(
        valid.unsqueeze(-1), point, helix_point(track, torch.zeros_like(s))
    )
    s = torch.where(valid, s, torch.zeros_like(s))
    return s, point, valid


def split_track_parameters(
    track_parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split physical helices into linear parameters and scaled curvature."""
    return track_parameters[..., :5], track_parameters[..., 5:6] / CURVATURE_SCALE


def assemble_track_parameters(
    linear: torch.Tensor, rho_scaled: torch.Tensor
) -> torch.Tensor:
    """Assemble physical helix parameters from optimizer representation."""
    return torch.cat((linear, rho_scaled * CURVATURE_SCALE), dim=-1)


def reference_gauge_frame(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix the redundant helix origin to a transverse reference plane."""
    direction = helix_tangent(reference, torch.zeros_like(reference[:, 0]))
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    helper = torch.zeros_like(direction)
    helper[:, 2] = 1.0
    degenerate = direction[:, 2].abs() > 1.0 - 1.0e-6
    helper[degenerate] = torch.tensor(
        [1.0, 0.0, 0.0], dtype=helper.dtype, device=helper.device
    )
    axis0 = torch.cross(helper, direction, dim=-1)
    axis0 = axis0 / axis0.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    axis1 = torch.cross(direction, axis0, dim=-1)
    return reference[:, :3].clone(), axis0, axis1


def pack_gauge_fixed_track_parameters(reference: torch.Tensor) -> torch.Tensor:
    """Pack a helix into the five observable parameters of a fixed gauge."""
    _linear, rho_scaled = split_track_parameters(reference)
    offsets = torch.zeros(
        (reference.shape[0], 2), dtype=reference.dtype, device=reference.device
    )
    return torch.cat((offsets, reference[:, 3:5], rho_scaled), dim=1)


def unpack_gauge_fixed_track_parameters(
    theta: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Recover physical helix parameters from a fixed-gauge representation."""
    origin, axis0, axis1 = frame
    position = origin + theta[:, 0:1] * axis0 + theta[:, 1:2] * axis1
    return assemble_track_parameters(
        torch.cat((position, theta[:, 2:4]), dim=1), theta[:, 4:5]
    )


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


_BATCHED_MEASUREMENT_RESIDUAL_JACOBIAN = torch.func.vmap(
    torch.func.jacrev(_single_measurement_residual, argnums=0)
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


def damped_gauss_newton(
    theta: torch.Tensor,
    residual_of: Callable[[torch.Tensor], torch.Tensor],
    jacobian_of: Callable[[torch.Tensor], torch.Tensor],
    candidate_ok_of: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Minimise independent row residuals with the measurement-refit LM rule."""
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    current = 0.5 * residual_of(theta).square().sum(dim=1)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Non-finite initial Gauss-Newton objective.")
    initial = float(current.mean())
    identity = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)
    level = torch.full(
        (theta.shape[0], 1, 1),
        float(damping),
        dtype=theta.dtype,
        device=theta.device,
    )
    iterations = 0
    change = 0.0
    for iterations in range(1, max_iters + 1):
        residual = residual_of(theta)
        jacobian = jacobian_of(theta)
        hessian = jacobian.transpose(1, 2) @ jacobian
        gradient = (jacobian.transpose(1, 2) @ residual.unsqueeze(-1)).squeeze(-1)
        scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
        stationary_hessian = (
            hessian + GAUSS_NEWTON_MIN_DAMPING * scale[:, None, None] * identity
        )
        decrement = (
            gradient.unsqueeze(1)
            @ torch.linalg.solve(stationary_hessian, gradient.unsqueeze(-1))
        ).flatten()
        relative_decrement = decrement / current.clamp_min(
            GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE
        )
        stationary = torch.isfinite(decrement) & (
            (decrement <= GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE)
            | (relative_decrement <= tolerance)
        )
        if bool(stationary.all()):
            break

        regularized = hessian + level * scale[:, None, None] * identity
        step = torch.linalg.solve(regularized, -gradient.unsqueeze(-1)).squeeze(-1)
        candidate = theta + step
        candidate_objective = 0.5 * residual_of(candidate).square().sum(dim=1)
        candidate_ok = (
            torch.ones_like(stationary)
            if candidate_ok_of is None
            else candidate_ok_of(candidate)
        )
        active = ~stationary
        improved = active & candidate_ok & torch.isfinite(candidate_objective) & (
            candidate_objective < current
        )
        theta = torch.where(improved.unsqueeze(-1), candidate, theta)
        next_level = torch.where(
            improved[:, None, None],
            (level * 0.1).clamp_min(GAUSS_NEWTON_MIN_DAMPING),
            level * 10.0,
        )
        level = torch.where(active[:, None, None], next_level, level)
        proposed_decrease = (current - candidate_objective).clamp_min(
            0.0
        ) / current.abs().clamp_min(1.0)
        decrease = torch.where(
            improved, proposed_decrease, torch.zeros_like(proposed_decrease)
        )
        current = torch.where(improved, candidate_objective, current)
        change = float(decrease.max())

    residual = residual_of(theta)
    jacobian = jacobian_of(theta)
    hessian = jacobian.transpose(1, 2) @ jacobian
    gradient = (jacobian.transpose(1, 2) @ residual.unsqueeze(-1)).squeeze(-1)
    scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
    stationary_hessian = (
        hessian + GAUSS_NEWTON_MIN_DAMPING * scale[:, None, None] * identity
    )
    decrement = (
        gradient.unsqueeze(1)
        @ torch.linalg.solve(stationary_hessian, gradient.unsqueeze(-1))
    ).flatten()
    relative_decrement = decrement / current.clamp_min(
        GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE
    )
    finite = torch.isfinite(decrement) & torch.isfinite(relative_decrement)
    stationary = finite & (
        (decrement <= GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE)
        | (relative_decrement <= tolerance)
    )
    return theta, {
        "converged": bool(stationary.all()),
        "iterations": int(iterations),
        "initial_objective": initial,
        "final_objective": float(current.mean()),
        "max_relative_decrease": float(change),
        "max_gradient_norm": float(gradient.norm(dim=1).max()),
        "relative_newton_decrement_max": float(
            relative_decrement[finite].max() if finite.any() else float("nan")
        ),
        "non_stationary_row_fraction": float((~stationary).double().mean()),
        "n_non_finite": int((~finite).sum()),
    }


def _merge_gauss_newton_diagnostics(
    blocks: list[dict[str, float | int | bool]], weights: list[int]
) -> dict[str, float | int | bool]:
    total = float(sum(weights))
    return {
        "converged": all(bool(block["converged"]) for block in blocks),
        "iterations": max(int(block["iterations"]) for block in blocks),
        "initial_objective": sum(
            float(block["initial_objective"]) * weight
            for block, weight in zip(blocks, weights)
        ) / total,
        "final_objective": sum(
            float(block["final_objective"]) * weight
            for block, weight in zip(blocks, weights)
        ) / total,
        "max_relative_decrease": max(
            float(block["max_relative_decrease"]) for block in blocks
        ),
        "max_gradient_norm": max(
            float(block["max_gradient_norm"]) for block in blocks
        ),
        "n_chunks": len(blocks),
    }


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

    current = 0.5 * residual_of(theta).square().sum(dim=1)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Non-finite initial Gauss-Newton objective.")
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
        jacobian = jacobian_of(theta)
        residual = residual_of(theta)
        hessian = jacobian.transpose(1, 2) @ jacobian
        gradient = (
            jacobian.transpose(1, 2) @ residual.unsqueeze(-1)
        ).squeeze(-1)
        scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
        regularized = hessian + level * scale[:, None, None] * identity
        step = torch.linalg.solve(regularized, -gradient.unsqueeze(-1)).squeeze(-1)
        candidate = theta + step
        candidate_objective = 0.5 * residual_of(candidate).square().sum(dim=1)
        improved = candidate_objective < current
        theta = torch.where(improved.unsqueeze(-1), candidate, theta)
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


def _single_point_fit_residual(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    reference = unpack_gauge_fixed_track_parameters(
        theta.unsqueeze(0),
        (origin.unsqueeze(0), axis0.unsqueeze(0), axis1.unsqueeze(0)),
    ).squeeze(0)
    expanded = reference.expand(points.shape[0], -1)
    with torch.no_grad():
        _fitted, path, valid = closest_helix_points(expanded.detach(), points)
    predicted = helix_point(expanded, path)
    effective = mask & valid
    residual = torch.where(
        effective.unsqueeze(-1), predicted - points, torch.zeros_like(points)
    )
    return residual.flatten()


def _single_point_fit_effective_mask(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    reference = unpack_gauge_fixed_track_parameters(
        theta.unsqueeze(0),
        (origin.unsqueeze(0), axis0.unsqueeze(0), axis1.unsqueeze(0)),
    ).squeeze(0)
    with torch.no_grad():
        _fitted, _path, valid = closest_helix_points(
            reference.expand(points.shape[0], -1), points
        )
    return mask & valid


_BATCHED_POINT_FIT_JACOBIAN = torch.func.vmap(
    torch.func.jacrev(_single_point_fit_residual, argnums=0)
)


def _fit_helix_points_gauss_newton_block(
    initial: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Fit per-track helix seeds to global points by damped Gauss-Newton."""
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial must have shape [tracks, 6].")
    if points.shape != (initial.shape[0], mask.shape[1], 3):
        raise ValueError("points must have shape [tracks, hits, 3].")
    if mask.shape != points.shape[:2]:
        raise ValueError("mask must have shape [tracks, hits].")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every helix seed needs at least one fitted point.")

    frame = reference_gauge_frame(initial)
    theta = pack_gauge_fixed_track_parameters(initial)

    def residual_of(values: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(_single_point_fit_residual)(values, *frame, points, mask)

    def jacobian_of(values: torch.Tensor) -> torch.Tensor:
        return _BATCHED_POINT_FIT_JACOBIAN(values, *frame, points, mask)

    initial_effective = torch.func.vmap(_single_point_fit_effective_mask)(
        theta, *frame, points, mask
    )

    def candidate_ok_of(values: torch.Tensor) -> torch.Tensor:
        candidate_effective = torch.func.vmap(_single_point_fit_effective_mask)(
            values, *frame, points, mask
        )
        return ((~initial_effective) | candidate_effective).all(dim=1)

    fitted_theta, stats = damped_gauss_newton(
        theta,
        residual_of,
        jacobian_of,
        candidate_ok_of=candidate_ok_of,
        max_iters=max_iters,
        damping=damping,
        tolerance=tolerance,
    )
    fitted = unpack_gauge_fixed_track_parameters(fitted_theta, frame).detach()
    return fitted, {"solver": "gauss-newton", **stats}


def fit_helix_points_gauss_newton(
    initial: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
    chunk_rows: int = 2048,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Fit point seeds with the measurement-refit LM strategy in row chunks."""
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial must have shape [tracks, 6].")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [tracks, hits].")
    if mask.shape != points.shape[:2]:
        raise ValueError("mask must have shape [tracks, hits].")
    if points.ndim != 3 or points.shape != (initial.shape[0], mask.shape[1], 3):
        raise ValueError("points must have shape [tracks, hits, 3].")
    if initial.shape[0] == 0:
        raise ValueError("At least one helix seed is required.")

    step = initial.shape[0] if chunk_rows <= 0 else chunk_rows
    fitted_blocks: list[torch.Tensor] = []
    stats_blocks: list[dict[str, float | int | bool | str]] = []
    weights: list[int] = []
    for start in range(0, initial.shape[0], max(step, 1)):
        stop = min(start + max(step, 1), initial.shape[0])
        fitted, stats = _fit_helix_points_gauss_newton_block(
            initial[start:stop],
            points[start:stop],
            mask[start:stop],
            max_iters=max_iters,
            damping=damping,
            tolerance=tolerance,
        )
        fitted_blocks.append(fitted)
        stats_blocks.append(stats)
        weights.append(stop - start)

    total = float(sum(weights))
    diagnostics: dict[str, float | int | bool | str] = {
        "solver": "gauss-newton",
        "converged": all(bool(stats["converged"]) for stats in stats_blocks),
        "iterations": max(int(stats["iterations"]) for stats in stats_blocks),
        "initial_objective": sum(
            float(stats["initial_objective"]) * weight
            for stats, weight in zip(stats_blocks, weights)
        ) / total,
        "final_objective": sum(
            float(stats["final_objective"]) * weight
            for stats, weight in zip(stats_blocks, weights)
        ) / total,
        "max_relative_decrease": max(
            float(stats["max_relative_decrease"]) for stats in stats_blocks
        ),
        "max_gradient_norm": max(
            float(stats["max_gradient_norm"]) for stats in stats_blocks
        ),
        "relative_newton_decrement_max": max(
            float(stats["relative_newton_decrement_max"]) for stats in stats_blocks
        ),
        "non_stationary_row_fraction": sum(
            float(stats["non_stationary_row_fraction"]) * weight
            for stats, weight in zip(stats_blocks, weights)
        ) / total,
        "n_non_finite": sum(int(stats["n_non_finite"]) for stats in stats_blocks),
        "n_chunks": len(stats_blocks),
    }
    return torch.cat(fitted_blocks, dim=0), diagnostics


def require_stationary_helix_point_fit(
    diagnostics: dict,
    max_non_stationary_row_fraction: float = (
        POINT_FIT_MAX_NON_STATIONARY_ROW_FRACTION
    ),
) -> None:
    """Fail closed when a point seed solver returns budget-truncated rows."""
    n_non_finite = int(diagnostics["n_non_finite"])
    if n_non_finite:
        raise ValueError(
            "SMT helix seed has non-finite Newton decrements: "
            f"{n_non_finite} rows."
        )
    fraction = float(diagnostics["non_stationary_row_fraction"])
    if fraction > max_non_stationary_row_fraction:
        raise ValueError(
            "SMT helix seed is non-stationary: "
            f"{fraction:.4%} of rows exceed the Newton-decrement criterion "
            f"(allowed {max_non_stationary_row_fraction:.4%})."
        )


_BROKEN_HELIX_COMPAT_NAMES = {
    "HIGHLAND_ENERGY_GEV",
    "HIGHLAND_LOG_COEFFICIENT",
    "KINK_THETA_EPS",
    "MUON_MASS_GEV",
    "assign_hit_segments",
    "broken_helix_local_path",
    "broken_helix_segments",
    "broken_helix_track_parameters",
    "highland_kink_sigma",
    "node_arc_lengths",
    "segment_origin_path",
}


def __getattr__(name: str):
    """Lazily preserve historical imports of experimental broken-helix symbols."""
    if name not in _BROKEN_HELIX_COMPAT_NAMES:
        raise AttributeError(name)
    try:
        from . import broken_helix_model
    except ImportError:  # Direct script execution.
        import broken_helix_model
    value = getattr(broken_helix_model, name)
    globals()[name] = value
    return value
