"""Non-target helix initialization: SMT point seed and measurement refit."""

from __future__ import annotations

import torch

from .conventions import EPS_COVARIANCE
from .helix.measurement_fit import (
    MeasurementFitInputs,
    fit_helix_measurements_gauss_newton,
    measurement_newton_decrement,
    measurement_stationarity_report,
)
from .helix.point_fit import fit_helix_points_gauss_newton
from .helix.trajectory import closest_helix_points, intersect_plane
from .track_batches import _non_target_fit_mask

#: Solver for the joint initialization's measurement refit.  Iteration 18 measured
#: the former Adam path on real data: Newton decrement p90 = 4.17e5 against a
#: threshold of 1e-6, with 100% of rows non-stationary after the full
#: 200-iteration budget.  The same objective solved by Gauss-Newton comes back
#: stationary.  The Adam path was removed on 2026-10-18; reproducing a run made
#: before Iteration 19 requires checking out commit 97cd01c.
JOINT_REFIT_SOLVER_GAUSS_NEWTON = "gauss-newton"

#: Pre-registration for the joint refit (independent of the retired Gate 4
#: ``PREREGISTRATION_VERSION``).  Bump it whenever a constant below changes meaning.
#: v1: Iteration 19, the Adam -> Gauss-Newton swap.
JOINT_REFIT_PREREGISTRATION_VERSION = 1

#: Acceptance criterion for the Gauss-Newton refit, fixed before the swap was run.
#: The per-row threshold itself is not redefined here: it is the pre-registered
#: ``NEWTON_DECREMENT_STATIONARY = 1e-6`` in the ``F = 0.5 * chi2`` gauge.  This is
#: only how many rows may miss it, and the value is carried over by analogy from
#: ``MAX_RANK_DEFICIENT_ROW_FRACTION`` -- a handful of pathological rows does not
#: invalidate a fit, above that the fit is not usable.  It was set from that
#: precedent, not from looking at what the solver produced.
JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION = 1.0e-3


def seed_helices(
    batches,
    n_iter=200,
    dtype=torch.float64,
    return_diagnostics=False,
):
    """Fit per-track helix warm starts to ACTS smoothed points with LM."""
    points = batches.g_smt.to(dtype=dtype)
    mask = _non_target_fit_mask(batches)
    n_tracks, _n_hits, _ = points.shape
    first_index = mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(n_tracks)
    first = points[rows, first_index]
    direction = batches.p_smt.to(dtype=dtype)[rows, first_index]
    transverse = torch.linalg.norm(direction[:, :2], dim=1).clamp(min=1e-6)
    psi = torch.atan2(direction[:, 1], direction[:, 0])
    tanl = direction[:, 2] / transverse
    initial = torch.cat(
        (
            first,
            psi[:, None],
            tanl[:, None],
            torch.zeros((n_tracks, 1), dtype=dtype),
        ),
        dim=1,
    )
    fitted, diagnostics = fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=n_iter
    )
    return (fitted, diagnostics) if return_diagnostics else fitted


def smt_seed_statistics(batches, track_parameters):
    """Summarize the non-target SMT point fit used only for initialization."""
    points = batches.g_smt.to(track_parameters.dtype)
    fit_mask = _non_target_fit_mask(batches)
    expanded = track_parameters[:, None, :].expand(-1, points.shape[1], -1)
    fitted, _path, valid = closest_helix_points(expanded, points)
    effective = fit_mask & valid
    if not effective.any():
        raise ValueError("No valid non-target SMT points remain after helix seeding.")
    squared_distance = torch.sum((fitted - points).square(), dim=-1)
    loss = torch.sum(torch.where(effective, squared_distance, torch.zeros_like(squared_distance)))
    loss = loss / effective.sum()
    invalid_fraction = 1.0 - float(valid[fit_mask].float().mean().cpu())
    return {
        "loss": float(loss.cpu()),
        "n_points": int(effective.sum().cpu()),
        "invalid_fraction": invalid_fraction,
    }


def _measurement_refit_objective(
    track_parameters,
    batches,
    geometry,
):
    """Evaluate non-target measurement chi2 on fixed working geometry."""
    measured = batches.measured_local.to(track_parameters.dtype)
    covariance = batches.covariances.to(track_parameters.dtype)
    sensor_idx = batches.sensor_idx
    fit_mask = _non_target_fit_mask(batches)
    loc1_mask = batches.loc1_mask
    n_hits = measured.shape[1]
    tracks = track_parameters[:, None, :].expand(-1, n_hits, -1)
    centers = geometry.centers.to(track_parameters.dtype)[sensor_idx]
    rotations = geometry.rotations.to(track_parameters.dtype)[sensor_idx]
    _path, intersections, intersection_valid = intersect_plane(
        tracks, centers, rotations[..., :, 2]
    )
    local = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)[..., :2]
    residuals = measured - local
    valid_hits = fit_mask & intersection_valid
    component_mask = torch.stack((valid_hits, valid_hits & loc1_mask), dim=-1)
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    loss = torch.sum(torch.where(component_mask, weighted, torch.zeros_like(weighted)))
    loss = loss / component_mask.sum().clamp(min=1)
    invalid_fraction = 1.0 - float(
        intersection_valid[fit_mask].float().mean().detach().cpu()
    )
    return loss, invalid_fraction, component_mask


def _measurement_fit_inputs_from_joint(batches, geometry, parameters):
    """Build the ACTS-independent single-helix measurement-fit tensors."""
    fit_mask = _non_target_fit_mask(batches)
    if not bool(fit_mask.any(dim=1).all()):
        raise ValueError("Every joint track needs at least one non-target hit.")
    dtype = parameters.dtype
    sensor_idx = batches.sensor_idx
    centers = geometry.centers.to(dtype=dtype)[sensor_idx]
    rotations = geometry.rotations.to(dtype=dtype)[sensor_idx]
    loc1_mask = batches.loc1_mask
    component_mask = torch.stack((fit_mask, fit_mask & loc1_mask), dim=-1)
    return MeasurementFitInputs(
        measured=batches.measured_local.to(dtype=dtype),
        variance=batches.covariances.to(dtype=dtype),
        component_mask=component_mask,
        centers=centers,
        rotations=rotations,
    )


def measure_joint_refit_stationarity(
    batches,
    geometry,
    parameters,
    dtype=torch.float64,
):
    """Per-row Newton decrement of the joint measurement refit.

    Turns "the optimiser may have stopped on budget" into a number.  Measurement
    only: it reads the parameters and never changes them.

    The shared helix model evaluates the unnormalised ``F = 0.5 * chi2`` gauge.
    ``_measurement_refit_objective`` divides by the component count, which would
    scale every decrement and make the pre-registered threshold meaningless.

    Runs in float64 on the CPU by default: a decrement compared against 1e-6 is
    below float32 resolution.
    """
    measured = _measurement_fit_inputs_from_joint(batches, geometry, parameters)
    decrement = measurement_newton_decrement(
        parameters, measured, dtype=dtype
    )
    return measurement_stationarity_report(decrement)


def _refit_helices_gauss_newton(batches, geometry, initial):
    """Solve the same objective, and refuse to return a solution that is not one.

    Always float64: the pre-registered stationarity threshold is 1e-6, which is
    below float32 resolution.  The caller casts the result back to its own dtype.
    """
    measured = _measurement_fit_inputs_from_joint(batches, geometry, initial)
    fitted, solver_stats = fit_helix_measurements_gauss_newton(
        initial,
        measured,
        dtype=torch.float64,
    )
    report = solver_stats["stationarity"]
    if report["n_non_finite"]:
        raise FloatingPointError(
            "Gauss-Newton joint refit produced non-finite Newton decrements on "
            f"{report['n_non_finite']} rows."
        )
    fraction = float(report["non_stationary_row_fraction"])
    if fraction > JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION:
        raise ValueError(
            "Gauss-Newton joint refit did not reach the pre-registered stationary "
            f"point: {fraction:.3%} of rows exceed a Newton decrement of "
            f"{report['stationary_threshold']:.3g} (limit "
            f"{JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION:.3%}, p90 "
            f"{report['newton_decrement_p90']:.6g})."
        )
    return fitted, {
        "convergence": solver_stats["convergence"],
        "stationarity_gate": report,
    }


def fit_helices_to_measurements(
    batches,
    geometry,
    initial_parameters,
    dtype=torch.float64,
    max_invalid_fraction=0.05,
    stationarity=True,
):
    """Refit helices using only non-target local measurements and fixed surfaces.

    ``stationarity`` adds a read-only Newton-decrement measurement of the seed and
    of the refit output.  It does not touch the returned parameters: the seed
    figure is what separates "the optimiser converged" from "the seed was already
    close".  It is a diagnostic, and switching it off does not switch off the
    pre-registered acceptance criterion the Gauss-Newton refit is held to -- that
    one reads the decrement the solver computes for itself.  On the float64 CPU
    path the refit figure is that same gate report, reused.
    """
    initial = initial_parameters.to(dtype=dtype)
    with torch.no_grad():
        initial_loss_tensor, initial_invalid, _initial_mask = _measurement_refit_objective(
            initial, batches, geometry
        )
    initial_loss = float(initial_loss_tensor.cpu())
    if initial_invalid > max_invalid_fraction:
        raise ValueError(
            "Initial non-target measurement refit exceeds invalid-intersection limit: "
            f"{initial_invalid:.2%} > {max_invalid_fraction:.2%}."
        )
    fitted, solver_stats = _refit_helices_gauss_newton(
        batches, geometry, initial
    )
    fitted = fitted.to(dtype=dtype)
    with torch.no_grad():
        evaluated_loss, final_invalid, component_mask = _measurement_refit_objective(
            fitted, batches, geometry
        )
        final_loss = float(evaluated_loss.cpu())
    if final_invalid > max_invalid_fraction:
        raise ValueError(
            "Non-target measurement refit intersection failure rate exceeds limit: "
            f"{final_invalid:.2%} > {max_invalid_fraction:.2%}."
        )
    stats = {
        "solver": JOINT_REFIT_SOLVER_GAUSS_NEWTON,
        "joint_refit_preregistration_version": JOINT_REFIT_PREREGISTRATION_VERSION,
        "chi2_per_component": final_loss,
        "initial_chi2_per_component": initial_loss,
        "n_components": int(component_mask.sum().cpu()),
        "invalid_fraction": final_invalid,
        "parameter_updates": {
            "position_rms": float(torch.sqrt(torch.mean((fitted[:, :3] - initial[:, :3]).square())).cpu()),
            "direction_rms": float(torch.sqrt(torch.mean((fitted[:, 3:5] - initial[:, 3:5]).square())).cpu()),
            "rho_rms": float(torch.sqrt(torch.mean((fitted[:, 5] - initial[:, 5]).square())).cpu()),
        },
    }
    stats.update(solver_stats)
    if stationarity:
        # The Gauss-Newton gate already evaluated the same float64 decrement on
        # these exact parameters; recomputing it costs a full Jacobian pass.  A
        # cast to float32 changes the parameters, so only then is the refit
        # report measured again.
        reuse_gate = fitted.dtype == torch.float64
        stats["stationarity"] = {
            "seed": measure_joint_refit_stationarity(batches, geometry, initial),
            "refit": (
                dict(stats["stationarity_gate"])
                if reuse_gate
                else measure_joint_refit_stationarity(batches, geometry, fitted)
            ),
        }
    return fitted, stats
