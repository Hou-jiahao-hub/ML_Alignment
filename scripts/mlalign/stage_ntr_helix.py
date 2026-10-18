"""Stage B -- NTR-helix: non-target-refitted helices, fixed-track block-LM geometry."""

from __future__ import annotations

import torch

from .block_lm_contract import (
    BLOCK_LM_DECREMENT_STATIONARY,
    BLOCK_LM_MAX_DAMPING_RETRIES,
    _block_lm_finalize_record,
    _block_lm_new_record,
    _block_lm_run_summary,
)
from .box_constraints import _block_lm_free_dof_mask
from .conventions import DEFAULT_ACTIVE_DOF, EPS_COVARIANCE
from .helix_init import seed_helices
from .joint_residuals import _sensor_residual_data, _target_alignment_evaluation

#: Block-LM budget and per-iteration step caps.  A sensor can only travel
#: ``cap * accepted_steps``, so these three constants jointly decide whether a
#: misalignment is reachable at all.  The former values (20 iterations,
#: 1e-3 rad) were calibrated for a geometry that two surface warm starts had
#: already pre-corrected: against the raw injected 0.030 rad they need at least
#: 30 accepted steps, which 20 iterations cannot supply.  Measured on the
#: 20260901 joint-only runs, that stranded 14 of 2268 sensors at 1-6 capped
#: steps regardless of the iteration budget.  At 5e-3 rad the same run
#: reproduced the warm-start pipeline's geometry and its ACTS acceptance to
#: within 1e-7, and converged in 13 iterations.
LM_MAX_ITERS = 60
LM_INITIAL_DAMPING = 1e-3
FISHER_EIGEN_FLOOR = 1e-4
LM_MAX_STEP_T = 0.1
LM_MAX_STEP_R = 5e-3
BLOCK_LM_MAX_DAMPING = 1.0e12
BLOCK_LM_MIN_DAMPING = 1.0e-12


def regularize_fisher_matrix(fisher, damping, eigen_floor_ratio):
    """Regularize a normalized Fisher matrix and report weak-mode diagnostics."""
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("Fisher matrix must be square")
    if damping < 0 or eigen_floor_ratio < 0:
        raise ValueError("LM damping and Fisher eigen floor must be non-negative")
    symmetric = 0.5 * (fisher + fisher.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues[0].detach().cpu())
    maximum = float(eigenvalues[-1].detach().cpu())
    floor = max(maximum, 0.0) * float(eigen_floor_ratio)
    regularization = max(0.0, floor - minimum) + float(damping) * max(maximum, 1.0)
    regularized = symmetric + torch.eye(
        len(symmetric), device=symmetric.device, dtype=symmetric.dtype
    ) * regularization
    condition = float("inf") if minimum <= 0 else maximum / minimum
    return regularized, {
        "min_eigenvalue": minimum,
        "max_eigenvalue": maximum,
        "condition_number": condition,
        "eigen_floor": floor,
        "regularization": regularization,
    }


def _block_lm_fixed_chi2(residuals, covariance, fixed_mask):
    """Weighted residual sum of squares on a *fixed* comparison set.

    The acceptance test has to hold the component set still.  Scoring a
    candidate on its own mask lets a step that kills the hardest intersections
    look like an improvement, because the mean is taken over whatever survived.
    """
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    return torch.sum(torch.where(fixed_mask, weighted, torch.zeros_like(weighted)))


def _block_lm_retry_damping(damping, initial_damping, retry_number):
    """Raise damping on a retry without losing the pre-registered absolute ladder.

    Accepted steps may have halved a sensor's damping as far as
    ``BLOCK_LM_MIN_DAMPING``.  The retry number therefore anchors the ladder at
    the production initial damping instead of merely multiplying that tiny
    carried value eight times.
    """
    anchor = max(float(initial_damping), LM_INITIAL_DAMPING)
    return min(
        max(float(damping) * 10.0, anchor * (10.0 ** int(retry_number))),
        BLOCK_LM_MAX_DAMPING,
    )


def _block_lm_decrement(fisher, gradient, free_mask, eigen_floor_ratio):
    """Newton decrement ``g^T H^-1 g`` on the free set, plus rank diagnostics.

    Solved through the eigendecomposition rather than by regularizing and
    calling ``solve``: any floor added to ``H`` can only shrink the decrement
    (``(H + eI)^-1 <= H^-1``), which would turn "not converged" into "converged"
    -- the failure this contract exists to remove.

    When a direction is weaker than the Fisher floor its curvature is supplied
    by the floor, not by the data, so it is dropped from the sum.  The remaining
    value is then a *lower bound* on the true decrement and cannot be used to
    claim convergence; ``lambda2_is_lower_bound`` says so, and
    ``null_gradient_max`` reports how much gradient is left sitting in the
    dropped directions.
    """
    free_fisher = fisher[free_mask][:, free_mask]
    free_gradient = gradient[free_mask]
    n_free = int(free_gradient.numel())
    if n_free == 0:
        # Every DOF sits on a bound with its gradient pointing outwards: the
        # projected gradient is zero, so this is a KKT point of the box problem.
        return {
            "lambda2": 0.0,
            "n_free": 0,
            "rank_deficient": False,
            "lambda2_is_lower_bound": False,
            "min_eigenvalue_free": float("nan"),
            "max_eigenvalue_free": float("nan"),
            "eigen_floor_free": float("nan"),
            "null_gradient_max": 0.0,
        }
    symmetric = 0.5 * (free_fisher + free_fisher.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    minimum = float(eigenvalues[0].detach().cpu())
    maximum = float(eigenvalues[-1].detach().cpu())
    floor = max(maximum, 0.0) * float(eigen_floor_ratio)
    weak = eigenvalues <= floor
    projection = eigenvectors.transpose(-1, -2) @ free_gradient
    strong = ~weak
    lambda2 = (
        float((projection[strong].square() / eigenvalues[strong]).sum().detach().cpu())
        if bool(strong.any())
        else 0.0
    )
    return {
        "lambda2": lambda2,
        "n_free": n_free,
        "rank_deficient": bool(weak.any()) or minimum <= 0.0,
        "lambda2_is_lower_bound": bool(weak.any()),
        "min_eigenvalue_free": minimum,
        "max_eigenvalue_free": maximum,
        "eigen_floor_free": floor,
        "null_gradient_max": (
            float(projection[weak].abs().max().detach().cpu()) if bool(weak.any()) else 0.0
        ),
    }


def _block_lm_linearized_block(
    tracks, current, sensor_index, batches, geometry, active_index, physical_scales, fixed_mask
):
    """One Gauss-Newton linearisation of a sensor block on a fixed component set.

    ``standardized`` returns ``r / sigma`` restricted to ``fixed_mask``, so
    ``r^T r`` is the chi2 of that set, ``gradient`` is grad F and ``fisher`` is
    the Gauss-Newton Hessian, both in the ``F = 0.5 * chi2`` gauge and scaled to
    dimensionless units by ``physical_scales``.

    Damping does not enter here, which is what makes retrying a rejected step
    cheap: the whole ladder reuses one Jacobian.
    """

    def standardized(active_values):
        row = current.scatter(0, active_index, active_values)
        values, variances, _mask, _intersection_valid = _sensor_residual_data(
            tracks, row, sensor_index, batches, geometry
        )
        return (values / variances.clamp_min(EPS_COVARIANCE).sqrt())[fixed_mask]

    active_values = current[active_index].detach().requires_grad_(True)
    residual_vector = standardized(active_values).detach()
    jacobian = torch.autograd.functional.jacobian(
        standardized, active_values, vectorize=True
    ).detach()
    normalized_jacobian = jacobian * physical_scales
    fisher = normalized_jacobian.transpose(0, 1) @ normalized_jacobian
    gradient = normalized_jacobian.transpose(0, 1) @ residual_vector
    return residual_vector, fisher, gradient


def run_block_lm_alignment(
    batches,
    geometry,
    config,
    active_dof=DEFAULT_ACTIVE_DOF,
    initial_track_params=None,
):
    """Fit target geometry with fixed non-target-refitted helices using block LM.

    Each target sensor is its own block: ``_sensor_residual_data`` selects hits
    by sensor id alone and the helices are frozen, so a block's residuals depend
    only on its own alignment row.  The blocks are therefore exactly decoupled,
    which is why one that reaches a stationary point can be retired for good.

    A block stops for one of four reasons, reported rather than collapsed into a
    flag: it is stationary (a free-set Newton decrement per component under
    ``BLOCK_LM_DECREMENT_STATIONARY``, or a KKT point of the box constraints),
    the iteration or damping budget ran out, the Fisher matrix is rank deficient
    so the decrement is only a lower bound, or no damping keeps the candidate's
    intersections alive.

    A candidate is accepted only if it lowers the chi2 **on the component set
    the current point has**, and only if it keeps every one of those components.
    Scoring each side on its own mask is what lets a step that kills the hardest
    intersections buy a lower mean with less information.  One consequence worth
    knowing: since accepted candidates may add components, the fixed set grows,
    and ``history`` compares objectives with different numbers of terms in them.
    It is not guaranteed monotone, and a rise in it is not by itself a bug.

    This fixes the convergence contract.  It is not a physics change: if the
    geometry it produces matches the baseline, the claim is that the contract
    holds, not that accuracy improved.
    """
    use_cuda = bool(config.gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    if initial_track_params is None:
        initial_track_params = seed_helices(batches, device=device, dtype=dtype)
    tracks = initial_track_params.to(device=device, dtype=dtype).detach()
    active = tuple(int(index) for index in active_dof)
    if not active:
        raise ValueError("Block LM requires at least one active alignment DOF")
    if any(index < 0 or index >= 6 for index in active):
        raise ValueError("Block LM active DOF indices must be in [0, 5]")
    if config.lm_max_iters < 1:
        raise ValueError("lm_max_iters must be at least 1")
    n_surfaces = len(geometry.centers)
    alignment = torch.zeros((n_surfaces, 6), device=device, dtype=dtype)
    active_index = torch.tensor(active, device=device, dtype=torch.long)
    physical_scales = torch.tensor(
        [1.0, 1.0, 1.0, 0.01, 0.01, 0.01], device=device, dtype=dtype
    )[active_index]
    initial_loss, initial_invalid, initial_residuals, initial_mask = (
        _target_alignment_evaluation(tracks, alignment, batches, geometry)
    )
    if initial_invalid > config.joint_max_invalid_fraction:
        raise ValueError(
            "Initial block-LM helices exceed the invalid-intersection limit: "
            f"{initial_invalid:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    best_loss = float(initial_loss.detach().cpu())
    best_iteration = -1
    history = []
    checkpoint_history = [(-1, best_loss)]
    invalid_history = [initial_invalid]
    fisher_diagnostics = []
    stopped_early = False
    target_sensors = torch.nonzero(geometry.target_mask, as_tuple=False).flatten().tolist()
    per_dof_limit = torch.tensor(
        [config.lm_max_step_t if index < 3 else config.lm_max_step_r for index in active],
        device=device,
        dtype=dtype,
    )
    bound_limit = torch.tensor(
        [config.joint_clamp_t if index < 3 else config.joint_clamp_r for index in active],
        device=device,
        dtype=dtype,
    )
    audit = {index: _block_lm_new_record(index, config) for index in target_sensors}
    # One damping per sensor, carried across outer iterations.  A shared scalar
    # lets a sensor that accepted a step halve the damping of a sensor that did
    # not, which is the opposite of what a trust region is for.
    damping_by_sensor = {index: float(config.lm_initial_damping) for index in target_sensors}
    active_sensors = list(target_sensors)
    stop_reason = "iteration_budget"
    for iteration in range(config.lm_max_iters):
        if not active_sensors:
            stop_reason = "all_blocks_terminal"
            break
        still_active = []
        for sensor_index in active_sensors:
            record = audit[sensor_index]
            current = alignment[sensor_index].detach()
            current_data = _sensor_residual_data(
                tracks, current, sensor_index, batches, geometry
            )
            if current_data is None:
                continue
            residuals, covariance, component_mask, current_valid = current_data
            # The comparison set for this visit.  Because an accepted candidate
            # must keep every component the current point has, this can only
            # grow -- which is also why ``history`` is not guaranteed monotone:
            # a larger set is a different objective, with more terms in it.
            fixed_mask = component_mask.detach()
            if not fixed_mask.any():
                # The sensor has target hits (otherwise current_data is None),
                # but none currently intersect.  This is an intersection
                # failure, not absence of data, and must remain in the status
                # denominator.
                record["n_components_start"] = 0
                record["n_components_end"] = 0
                record["had_invalid_rejection"] = True
                record["retired_at_iteration"] = iteration
                continue
            n_components = int(fixed_mask.sum())
            if record["n_components_start"] is None:
                record["n_components_start"] = n_components
            record["n_components_end"] = n_components
            current_invalid = 1.0 - float(current_valid.float().mean().detach().cpu())

            residual_vector, fisher, gradient = _block_lm_linearized_block(
                tracks, current, sensor_index, batches, geometry,
                active_index, physical_scales, fixed_mask,
            )
            chi2_before = float(residual_vector.square().sum().detach().cpu())

            free_mask = _block_lm_free_dof_mask(
                current[active_index].detach(), gradient, -bound_limit, bound_limit
            )
            report = _block_lm_decrement(
                fisher, gradient, free_mask, config.fisher_eigen_floor
            )
            n_free = report["n_free"]
            per_component = (
                report["lambda2"] / n_components if n_components else float("nan")
            )
            # An empty free set means every active DOF sits on a bound with its
            # gradient pointing outwards: the projected gradient vanishes, so
            # this is a KKT point of the box-constrained problem.
            stationary = n_free == 0 or per_component <= BLOCK_LM_DECREMENT_STATIONARY
            record.update({
                "lambda2": report["lambda2"],
                "lambda2_per_component": per_component,
                "lambda2_is_lower_bound": report["lambda2_is_lower_bound"],
                "n_free": n_free,
                "frozen_dof": len(active) - n_free,
                "boundary": n_free < len(active),
                "rank_deficient": report["rank_deficient"],
                "stationary": stationary,
                "min_eigenvalue_free": report["min_eigenvalue_free"],
                "null_gradient_max": report["null_gradient_max"],
            })
            if report["rank_deficient"]:
                # Reported, not retired.  The decrement is a lower bound here,
                # so it cannot certify convergence -- but stopping these blocks
                # outright would change the geometry wholesale and make "the
                # contract was fixed" indistinguishable from "the answer moved".
                pass
            elif stationary:
                record["retired_at_iteration"] = iteration
                continue

            accepted = False
            ladder_lost = False
            ladder_invalid = False
            for attempt in range(1 + BLOCK_LM_MAX_DAMPING_RETRIES):
                damping = damping_by_sensor[sensor_index]
                regularized, diagnostics = regularize_fisher_matrix(
                    fisher,
                    damping=damping,
                    eigen_floor_ratio=config.fisher_eigen_floor,
                )
                lstsq_fallback = False
                try:
                    normalized_step = -torch.linalg.solve(regularized, gradient)
                except RuntimeError:
                    lstsq_fallback = True
                    normalized_step = -torch.linalg.lstsq(
                        regularized, gradient[:, None]
                    ).solution[:, 0]
                physical_step = normalized_step * physical_scales
                # The step cap is a trust region, not a feasible-set boundary:
                # it never enters the KKT test, but a capped step is no longer
                # the LM step, so it is counted separately.
                step_clipped = bool((physical_step.abs() > per_dof_limit).any())
                physical_step = torch.maximum(
                    torch.minimum(physical_step, per_dof_limit), -per_dof_limit
                )
                candidate = current.clone()
                candidate[active_index] += physical_step
                candidate[:3].clamp_(-config.joint_clamp_t, config.joint_clamp_t)
                candidate[3:].clamp_(-config.joint_clamp_r, config.joint_clamp_r)
                (
                    candidate_residuals,
                    candidate_covariance,
                    candidate_mask,
                    candidate_valid,
                ) = _sensor_residual_data(
                    tracks, candidate, sensor_index, batches, geometry
                )
                # Both sides scored on the same set.  Scoring the candidate on
                # its own mask is how a step that kills the hardest
                # intersections buys a lower mean with less information.
                chi2_after = _block_lm_fixed_chi2(
                    candidate_residuals, candidate_covariance, fixed_mask
                )
                lost = bool((fixed_mask & ~candidate_mask).any())
                sensor_invalid = 1.0 - float(
                    candidate_valid.float().mean().detach().cpu()
                )
                # Anchored on where this sensor already is: the run-level limit
                # is checked globally at entry, so a sensor that starts out
                # degraded would otherwise be frozen for reasons that have
                # nothing to do with the step under test.
                invalid_ok = sensor_invalid <= max(
                    current_invalid, config.joint_max_invalid_fraction
                )
                ok = bool(
                    (not lost)
                    and invalid_ok
                    and torch.isfinite(chi2_after)
                    and float(chi2_after) < chi2_before
                )
                reject_reason = None
                if not ok:
                    reject_reason = (
                        "component_loss" if lost
                        else "invalid_fraction" if not invalid_ok
                        else "no_decrease"
                    )
                    ladder_lost = ladder_lost or lost
                    ladder_invalid = ladder_invalid or not invalid_ok
                    if lost:
                        record["component_loss_rejections"] += 1
                    if not invalid_ok:
                        record["invalid_fraction_rejections"] += 1
                record["evaluations"] += 1
                record["clipped_steps"] += int(step_clipped)
                record["lstsq_fallbacks"] += int(lstsq_fallback)
                record["final_damping"] = float(damping)
                record["max_damping_seen"] = max(record["max_damping_seen"], float(damping))
                record["min_damping_seen"] = min(record["min_damping_seen"], float(damping))
                diagnostics.update({
                    "iteration": iteration,
                    "sensor_index": int(sensor_index),
                    "accepted": ok,
                    "attempt": attempt,
                    "damping": float(damping),
                    "step_rms": float(
                        torch.sqrt(torch.mean(physical_step.square())).detach().cpu()
                    ),
                    "step_max": float(physical_step.abs().max().detach().cpu()),
                    "step_clipped": step_clipped,
                    "lstsq_fallback": lstsq_fallback,
                    "gradient_norm": float(gradient.norm().detach().cpu()),
                    "lambda2": report["lambda2"],
                    "lambda2_per_component": per_component,
                    "lambda2_is_lower_bound": report["lambda2_is_lower_bound"],
                    "n_free": n_free,
                    "rank_deficient": report["rank_deficient"],
                    "null_gradient_max": report["null_gradient_max"],
                    "chi2_fixed_before": chi2_before,
                    "chi2_fixed_after": float(chi2_after.detach().cpu()),
                    "n_components_fixed": n_components,
                    "components_before": n_components,
                    "components_after": int(candidate_mask.sum()),
                    "component_loss": lost,
                    "reject_reason": reject_reason,
                    "sensor_invalid_fraction": sensor_invalid,
                    "loss_before": chi2_before / n_components,
                    "loss_after": float(chi2_after.detach().cpu()) / n_components,
                })
                fisher_diagnostics.append(diagnostics)
                if ok:
                    alignment[sensor_index] = candidate
                    record["accepted_steps"] += 1
                    record["accepted_component_deltas"].append(
                        int(candidate_mask.sum()) - n_components
                    )
                    damping_by_sensor[sensor_index] = max(
                        damping * 0.5, BLOCK_LM_MIN_DAMPING
                    )
                    accepted = True
                    break
                if damping >= BLOCK_LM_MAX_DAMPING:
                    break
                # Written back, so the raised damping is used by the next solve
                # whether that is this ladder's retry or the next iteration.
                record["retries"] += 1
                damping_by_sensor[sensor_index] = _block_lm_retry_damping(
                    damping,
                    initial_damping=config.lm_initial_damping,
                    retry_number=attempt + 1,
                )
            if accepted:
                still_active.append(sensor_index)
            else:
                # No feasible descent step at this point under any damping.
                record["had_component_loss"] = ladder_lost
                record["had_invalid_rejection"] = ladder_invalid
                if not (ladder_lost or ladder_invalid):
                    record["exhausted_reason"] = "damping_retries"
                record["retired_at_iteration"] = iteration
        active_sensors = still_active
        evaluated_loss, invalid_fraction, _residuals, _mask = _target_alignment_evaluation(
            tracks, alignment, batches, geometry
        )
        evaluated_value = float(evaluated_loss.detach().cpu())
        history.append(evaluated_value)
        checkpoint_history.append((iteration, evaluated_value))
        invalid_history.append(invalid_fraction)
        if evaluated_value < best_loss:
            best_loss = evaluated_value
            best_iteration = iteration
    stopped_early = stop_reason == "all_blocks_terminal"
    blocks = [_block_lm_finalize_record(audit[index]) for index in target_sensors]
    block_lm_summary = _block_lm_run_summary(blocks, "contract-fixed", stop_reason, config)
    final_loss, final_invalid, final_residuals, final_mask = _target_alignment_evaluation(
        tracks, alignment, batches, geometry
    )
    if final_invalid > config.joint_max_invalid_fraction:
        raise ValueError(
            "Block-LM result exceeds the invalid-intersection limit: "
            f"{final_invalid:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    return {
        "solver": "block-lm",
        "predicted_deltas": alignment.detach().cpu().double().numpy(),
        "track_params": tracks.detach().cpu().double().numpy(),
        "final_residuals": final_residuals.detach().cpu(),
        "final_component_mask": final_mask.detach().cpu(),
        "initial_residuals": initial_residuals.detach().cpu(),
        "initial_component_mask": initial_mask.detach().cpu(),
        "history": history,
        "checkpoint_history": checkpoint_history,
        "best_total_loss": best_loss,
        "last_total_loss": float(final_loss.detach().cpu()),
        "best_iteration": best_iteration,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "initial_total_loss": float(initial_loss.detach().cpu()),
        "initial_invalid_fraction": initial_invalid,
        "invalid_fraction": final_invalid,
        "invalid_history": invalid_history,
        "fisher_diagnostics": fisher_diagnostics,
        "block_lm_summary": block_lm_summary,
        "block_lm_blocks": blocks,
    }
