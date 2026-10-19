"""Row-independent damped Gauss-Newton/LM shared by the helix fits."""

from __future__ import annotations

from collections.abc import Callable

import torch

GAUSS_NEWTON_MAX_ITERS = 60
GAUSS_NEWTON_TOLERANCE = 1.0e-6
GAUSS_NEWTON_INITIAL_DAMPING = 1.0e-8
GAUSS_NEWTON_MIN_DAMPING = 1.0e-12
GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE = 1.0e-12
RowFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _stationary_newton_decrement(
    jacobian: torch.Tensor, residual: torch.Tensor, identity: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(hessian, gradient, scale, g^T H^-1 g)`` at minimal damping."""
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
    return hessian, gradient, scale, decrement


def damped_gauss_newton(
    theta: torch.Tensor,
    residual_of: RowFunction,
    jacobian_of: RowFunction,
    candidate_of: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        | None
    ) = None,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Minimise independent row residuals with the measurement-refit LM rule.

    Every callback takes ``(values, rows)`` and evaluates only the listed rows;
    ``candidate_of`` returns the candidate residual and its feasibility in one
    pass.  A stationary row is never stepped again, so its decrement can only
    repeat: it leaves the active set with its final diagnostics recorded, and
    the remaining rows follow the full-batch trajectory unchanged.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    if candidate_of is None:
        def candidate_of(values, rows):
            return residual_of(values, rows), torch.ones_like(rows, dtype=torch.bool)

    theta = theta.clone()
    rows = torch.arange(theta.shape[0])
    residual = residual_of(theta, rows)
    current = 0.5 * residual.square().sum(dim=1)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Non-finite initial Gauss-Newton objective.")
    initial = float(current.mean())
    identity = torch.eye(theta.shape[1], dtype=theta.dtype)
    level = torch.full(
        (theta.shape[0], 1, 1),
        float(damping),
        dtype=theta.dtype,
    )
    decrement = torch.full_like(current, float("nan"))
    relative_decrement = torch.full_like(current, float("nan"))
    gradient_norm = torch.full_like(current, float("nan"))
    iterations = 0
    change = 0.0
    stepped = False
    for iterations in range(1, max_iters + 1):
        values = theta[rows]
        row_residual = residual[rows]
        row_current = current[rows]
        hessian, gradient, scale, row_decrement = _stationary_newton_decrement(
            jacobian_of(values, rows), row_residual, identity
        )
        row_relative = row_decrement / row_current.clamp_min(
            GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE
        )
        decrement[rows] = row_decrement
        relative_decrement[rows] = row_relative
        gradient_norm[rows] = gradient.norm(dim=1)
        stationary = torch.isfinite(row_decrement) & (
            (row_decrement <= GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE)
            | (row_relative <= tolerance)
        )
        if bool(stationary.all()):
            stepped = False
            break

        active = ~stationary
        rows = rows[active]
        values = values[active]
        row_residual = row_residual[active]
        row_current = row_current[active]
        row_level = level[rows]
        regularized = (
            hessian[active] + row_level * scale[active][:, None, None] * identity
        )
        step = torch.linalg.solve(
            regularized, -gradient[active].unsqueeze(-1)
        ).squeeze(-1)
        candidate = values + step
        candidate_residual, candidate_ok = candidate_of(candidate, rows)
        candidate_objective = 0.5 * candidate_residual.square().sum(dim=1)
        improved = candidate_ok & torch.isfinite(candidate_objective) & (
            candidate_objective < row_current
        )
        theta[rows] = torch.where(improved.unsqueeze(-1), candidate, values)
        residual[rows] = torch.where(
            improved.unsqueeze(-1), candidate_residual, row_residual
        )
        level[rows] = torch.where(
            improved[:, None, None],
            (row_level * 0.1).clamp_min(GAUSS_NEWTON_MIN_DAMPING),
            row_level * 10.0,
        )
        proposed_decrease = (row_current - candidate_objective).clamp_min(
            0.0
        ) / row_current.abs().clamp_min(1.0)
        decrease = torch.where(
            improved, proposed_decrease, torch.zeros_like(proposed_decrease)
        )
        current[rows] = torch.where(improved, candidate_objective, row_current)
        change = float(decrease.max())
        stepped = True

    if stepped:
        # Budget exhausted: only the rows stepped last have stale diagnostics.
        _hessian, gradient, _scale, row_decrement = _stationary_newton_decrement(
            jacobian_of(theta[rows], rows), residual[rows], identity
        )
        decrement[rows] = row_decrement
        relative_decrement[rows] = row_decrement / current[rows].clamp_min(
            GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE
        )
        gradient_norm[rows] = gradient.norm(dim=1)
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
        "max_gradient_norm": float(gradient_norm.max()),
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
