"""Pre-registered block-LM convergence contract: thresholds, terminal states, audit."""

from __future__ import annotations

import math

#: Pre-registration for the block-LM convergence contract, kept separate from
#: ``JOINT_REFIT_PREREGISTRATION_VERSION`` above and from the Gate 4 block's
#: ``PREREGISTRATION_VERSION``: those govern the measurement refit and the
#: validation verdicts, neither of which changes here.
#: v1: the L1 audit -- per-sensor damping, retry after rejection, fixed
#: comparison set, and a stationarity/KKT stopping test.
#: v2: review hardening -- undefined truth comparisons fail closed, sensors
#: with hits but no valid intersections remain visible, the retry ladder keeps
#: its absolute scale after accepted-step damping reductions, and any least-
#: squares fallback is sticky rank-deficiency evidence.
BLOCK_LM_PREREGISTRATION_VERSION = 2

#: Stationarity threshold for one sensor block, normalized by component count:
#: a block is stationary when ``lambda2 / n_components <=`` this value, where
#: ``lambda2 = g_free^T H_free^-1 g_free`` is the Newton decrement restricted to
#: the DOFs the box constraints leave free.  ``standardized()`` returns
#: ``r / sigma``, so ``r^T r = chi2``, ``g = J^T r = grad F`` and ``H = J^T J``:
#: the gauge is ``F = 0.5 * chi2``, the same one ``NEWTON_DECREMENT_STATIONARY``
#: (validate_helix_model.py) was frozen in, and the value is carried over from
#: it.  The caliber differs -- that constant is per fitted track row, this one is
#: per measurement component -- so the two are frozen independently and neither
#: may be adjusted to make the other pass.
#: One deliberate divergence: ``newton_decrement`` regularizes before solving,
#: which can only shrink the decrement.  The check here adds no regularization
#: at all, because a decrement biased low is exactly how a non-stationary block
#: gets reported as converged.
#: This threshold is evidence of convergence, not evidence of accuracy.  A block
#: that meets it has no descent left to offer under its own quadratic model; it
#: says nothing about how close the geometry is to truth.
BLOCK_LM_DECREMENT_STATIONARY = 1.0e-6

#: How many times a rejected candidate may be retried at ten-fold damping within
#: one outer iteration.  From ``LM_INITIAL_DAMPING = 1e-3`` eight rungs reach
#: 1e5, where the step is ~1e-8 of the Gauss-Newton step and its direction has
#: collapsed onto steepest descent -- for a smooth objective that descends unless
#: the gradient already vanishes, which the stationarity test catches first.
#: Still no descent after eight rungs therefore falsifies smoothness (the
#: intersection solver's validity branches and its ``|s| <= 1200 mm`` cut) and is
#: worth reporting rather than spending more budget on.  Retries reuse the
#: Jacobian, so a full ladder costs a fraction of one extra block visit.
BLOCK_LM_MAX_DAMPING_RETRIES = 8

#: The four terminal states a sensor block can end a block-LM run in, plus the
#: internal ``no_data`` case for sensors that carry no target hit at all.  A run
#: reports counts and fractions over these; per-sensor detail stays in memory.
BLOCK_LM_STATUS_CONVERGED = "converged"
BLOCK_LM_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
BLOCK_LM_STATUS_RANK_DEFICIENT = "rank_deficient"
BLOCK_LM_STATUS_INTERSECTION_FAILED = "intersection_failed"
BLOCK_LM_STATUS_NO_DATA = "no_data"
BLOCK_LM_TERMINAL_STATUSES = (
    BLOCK_LM_STATUS_CONVERGED,
    BLOCK_LM_STATUS_BUDGET_EXHAUSTED,
    BLOCK_LM_STATUS_RANK_DEFICIENT,
    BLOCK_LM_STATUS_INTERSECTION_FAILED,
)


def _block_lm_new_record(sensor_index, config):
    """Blank audit record for one sensor block."""
    return {
        "sensor_index": int(sensor_index),
        "evaluations": 0,
        "accepted_steps": 0,
        "accepted_component_deltas": [],
        "retries": 0,
        "clipped_steps": 0,
        "lstsq_fallbacks": 0,
        "component_loss_rejections": 0,
        "invalid_fraction_rejections": 0,
        "had_component_loss": False,
        "had_invalid_rejection": False,
        "final_damping": float(config.lm_initial_damping),
        "max_damping_seen": float(config.lm_initial_damping),
        "min_damping_seen": float(config.lm_initial_damping),
        "exhausted_reason": "outer_iters",
        "retired_at_iteration": None,
        "n_components_start": None,
        "n_components_end": None,
        "lambda2": None,
        "lambda2_per_component": None,
        "lambda2_is_lower_bound": False,
        "n_free": None,
        "frozen_dof": 0,
        "boundary": False,
        "rank_deficient": False,
        "stationary": False,
        "min_eigenvalue_free": None,
        "null_gradient_max": None,
        "chi2_fixed": None,
    }


def _block_lm_finalize_record(record):
    """Stamp the terminal state onto a block record once the run has stopped."""
    record["status"] = _block_lm_block_status(record)
    return record


def _block_lm_block_status(record):
    """Terminal state of one block, in the pre-registered priority order.

    ``intersection_failed`` outranks ``rank_deficient`` because a candidate that
    cannot keep its own intersections says nothing about curvature, and
    ``rank_deficient`` outranks ``converged`` because a decrement computed with
    weak directions dropped is a lower bound -- passing a threshold with a
    statistic biased low is precisely the false convergence this contract
    removes.
    """
    if record.get("n_components_end") is None:
        return BLOCK_LM_STATUS_NO_DATA
    if record.get("had_component_loss") or record.get("had_invalid_rejection"):
        return BLOCK_LM_STATUS_INTERSECTION_FAILED
    if record.get("rank_deficient") or record.get("lstsq_fallbacks", 0) > 0:
        return BLOCK_LM_STATUS_RANK_DEFICIENT
    if record.get("stationary"):
        return BLOCK_LM_STATUS_CONVERGED
    return BLOCK_LM_STATUS_BUDGET_EXHAUSTED


def _block_lm_run_summary(blocks, arm, stop_reason, config):
    """Run-level roll-up of the block terminal states.

    Deliberately scalar: with a couple of thousand target sensors, per-sensor
    rows would grow the summary by an order of magnitude while saying nothing
    the counts and worst cases do not.  ``block_lm_blocks`` keeps the detail in
    memory for tests and ad-hoc analysis.
    """
    with_data = [item for item in blocks if item["status"] != BLOCK_LM_STATUS_NO_DATA]
    denominator = len(with_data)
    counts = {name: 0 for name in BLOCK_LM_TERMINAL_STATUSES}
    for item in with_data:
        counts[item["status"]] += 1
    # Boundary blocks stop at a KKT point with lambda2 identically zero; letting
    # them into the quantiles would drag the distribution down for free.
    interior = [
        item["lambda2_per_component"]
        for item in with_data
        if not item["boundary"] and item["lambda2_per_component"] is not None
    ]
    finite = sorted(value for value in interior if math.isfinite(value))

    def quantile(fraction):
        if not finite:
            return float("nan")
        position = fraction * (len(finite) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        weight = position - low
        return finite[low] * (1.0 - weight) + finite[high] * weight

    accepted_deltas = [
        value for item in with_data for value in item["accepted_component_deltas"]
    ]
    span_deltas = [item["n_components_end"] - item["n_components_start"] for item in with_data]
    damping_high = [item["max_damping_seen"] for item in with_data]
    damping_low = [item["min_damping_seen"] for item in with_data]
    eigenvalues = [
        item["min_eigenvalue_free"]
        for item in with_data
        if item["min_eigenvalue_free"] is not None
        and math.isfinite(item["min_eigenvalue_free"])
    ]
    null_gradients = [
        item["null_gradient_max"]
        for item in with_data
        if item["null_gradient_max"] is not None
        and math.isfinite(item["null_gradient_max"])
    ]
    clipped = sum(item["clipped_steps"] for item in with_data)
    evaluations = sum(item["evaluations"] for item in with_data)
    return {
        "claim_scope": "convergence-contract-only",
        "arm": arm,
        "preregistration_version": BLOCK_LM_PREREGISTRATION_VERSION,
        "stationary_threshold": BLOCK_LM_DECREMENT_STATIONARY,
        "max_damping_retries": BLOCK_LM_MAX_DAMPING_RETRIES,
        "stop_reason": stop_reason,
        "blocks_total": len(blocks),
        "blocks_without_data": len(blocks) - denominator,
        "status_counts": counts,
        "status_fractions": {
            name: (counts[name] / denominator if denominator else 0.0)
            for name in BLOCK_LM_TERMINAL_STATUSES
        },
        "converged_detail": {
            "interior": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_CONVERGED and not item["boundary"]
            ),
            "boundary": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_CONVERGED and item["boundary"]
            ),
        },
        "budget_exhausted_detail": {
            "outer_iters": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_BUDGET_EXHAUSTED
                and item["exhausted_reason"] == "outer_iters"
            ),
            "damping_retries": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_BUDGET_EXHAUSTED
                and item["exhausted_reason"] == "damping_retries"
            ),
        },
        "decrement_per_component": {
            "interior_only": True,
            "n_blocks": len(interior),
            "n_non_finite": len(interior) - len(finite),
            "non_stationary_block_fraction": (
                sum(1 for value in interior if not (value <= BLOCK_LM_DECREMENT_STATIONARY))
                / len(interior)
                if interior
                else 0.0
            ),
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p99": quantile(0.99),
            "max": finite[-1] if finite else float("nan"),
        },
        "component_guard": {
            "accepted_steps": len(accepted_deltas),
            "min_components_delta_accepted": min(accepted_deltas) if accepted_deltas else 0,
            "min_components_end_minus_start": min(span_deltas) if span_deltas else 0,
            "component_loss_rejections": sum(
                item["component_loss_rejections"] for item in with_data
            ),
            "invalid_fraction_rejections": sum(
                item["invalid_fraction_rejections"] for item in with_data
            ),
        },
        "damping": {
            "blocks_with_retry": sum(1 for item in with_data if item["retries"] > 0),
            "retries_total": sum(item["retries"] for item in with_data),
            "max_damping_reached": (
                max(damping_high) if damping_high else float(config.lm_initial_damping)
            ),
            "min_damping_reached": (
                min(damping_low) if damping_low else float(config.lm_initial_damping)
            ),
        },
        "trust_region": {
            "clipped_steps": clipped,
            "clipped_step_fraction": (clipped / evaluations if evaluations else 0.0),
        },
        "boundary": {
            "blocks_with_active_bound": sum(1 for item in with_data if item["boundary"]),
            "free_set_empty_blocks": sum(1 for item in with_data if item["n_free"] == 0),
            "frozen_dof_total": sum(item["frozen_dof"] for item in with_data),
        },
        "rank": {
            "lstsq_fallbacks": sum(item["lstsq_fallbacks"] for item in with_data),
            "min_eigenvalue_min": min(eigenvalues) if eigenvalues else float("nan"),
            "floor_limited_blocks": sum(
                1 for item in with_data if item["lambda2_is_lower_bound"]
            ),
            "null_gradient_max": max(null_gradients) if null_gradients else float("nan"),
        },
    }
