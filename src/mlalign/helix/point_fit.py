"""Single-helix fit to global SMT points, the seed of every track refit."""

from __future__ import annotations

import torch

from .gauge import (
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    unpack_gauge_fixed_track_parameters,
)
from .gauss_newton import (
    GAUSS_NEWTON_INITIAL_DAMPING,
    GAUSS_NEWTON_MAX_ITERS,
    GAUSS_NEWTON_TOLERANCE,
    damped_gauss_newton,
)
from .trajectory import closest_helix_points, helix_point

POINT_FIT_MAX_NON_STATIONARY_ROW_FRACTION = 1.0e-3


def _single_point_fit_residual_and_mask(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = unpack_gauge_fixed_track_parameters(
        theta.unsqueeze(0),
        (origin.unsqueeze(0), axis0.unsqueeze(0), axis1.unsqueeze(0)),
    ).squeeze(0)
    expanded = reference.expand(points.shape[0], -1)
    # Differentiate the closest-point path too: holding it fixed keeps the
    # gradient but inflates J^T J along the tangent, so seed steps crawl.
    _fitted, path, valid = closest_helix_points(expanded, points)
    predicted = helix_point(expanded, path)
    effective = mask & valid
    residual = torch.where(
        effective.unsqueeze(-1), predicted - points, torch.zeros_like(points)
    )
    return residual.flatten(), effective


def _single_point_fit_residual(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return _single_point_fit_residual_and_mask(
        theta, origin, axis0, axis1, points, mask
    )[0]


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
    torch.func.jacfwd(_single_point_fit_residual, argnums=0)
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

    def row_inputs(rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (*(part[rows] for part in frame), points[rows], mask[rows])

    def residual_of(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(_single_point_fit_residual)(values, *row_inputs(rows))

    def jacobian_of(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return _BATCHED_POINT_FIT_JACOBIAN(values, *row_inputs(rows))

    initial_effective = torch.func.vmap(_single_point_fit_effective_mask)(
        theta, *frame, points, mask
    )

    def candidate_of(
        values: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # One closest-point solve yields both the residual and the validity
        # that decides whether the candidate may be accepted.
        residual, candidate_effective = torch.func.vmap(
            _single_point_fit_residual_and_mask
        )(values, *row_inputs(rows))
        ok = ((~initial_effective[rows]) | candidate_effective).all(dim=1)
        return residual, ok

    fitted_theta, stats = damped_gauss_newton(
        theta,
        residual_of,
        jacobian_of,
        candidate_of=candidate_of,
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
