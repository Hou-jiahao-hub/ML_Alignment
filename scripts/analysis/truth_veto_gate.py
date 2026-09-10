"""Post-hoc truth-distance veto gate for simulation-only candidate acceptance.

POST-HOC ONLY -- NEVER A SELECTION INPUT.

This module answers exactly one question, once, after chi2/likelihood-only
calibration has already selected a single candidate geometry: does that
candidate's truth distance justify calling it an improvement?  It exists
because a real, already-measured failure mode makes chi2 alone untrustworthy:
folding target-layer hits into a track refit was observed to make truth
distance roughly 9.9x worse while chi2 could look better (see
``docs/history/ml_optimization_log.md`` and
``docs/handoff/2026-09-07-alignment-methodology-roadmap.md`` L3).

Nothing computed here may be:
  - read back into a solver's accept/reject or convergence loop,
  - used to choose a hyperparameter (damping, active DOF, lambda_l2, ...),
  - used as an early-stopping signal,
  - evaluated across multiple candidates to pick a "winner".

Any of the above is a pre-registered methodology violation: truth is only
available in simulation, so folding it into calibration would select for a
rule that cannot transfer to real data, where no truth exists.
``evaluate_truth_veto``'s mandatory ``after_candidate_selection=True``
argument exists to make this explicit at every call site; it is not a real
feature switch -- passing anything else raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

#: Upper-tail bound of a two-sided 95% CI, matching the convention already in
#: use by ``validate_helix_model.paired_bootstrap_ci`` and
#: ``scripts/analysis/compare_truth_rms.py`` for "paired 95% CI upper bound".
DEFAULT_CI_QUANTILE = 0.975
DEFAULT_N_BOOTSTRAP = 2000
#: Primary metric: candidate must be strictly better (ratio < 1.0).
PRIMARY_CI_UPPER_LIMIT = 1.0
#: Required (secondary) metrics: candidate must not be meaningfully worse.
REQUIRED_CI_UPPER_LIMIT = 1.01


def _per_event_sums(
    distance: np.ndarray, event_ids: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event sum of squared distance and hit count, over valid hits only."""
    distance = np.asarray(distance, dtype=np.float64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    if distance.shape != event_ids.shape or distance.shape != valid.shape:
        raise ValueError("distance, event_ids and valid must share one shape")
    flat_distance = distance.reshape(-1)
    flat_events = event_ids.reshape(-1)
    flat_valid = valid.reshape(-1)
    selected_events = flat_events[flat_valid]
    selected_sq = flat_distance[flat_valid] ** 2
    if selected_events.size == 0:
        raise ValueError("no valid hits to aggregate")
    order = np.argsort(selected_events, kind="stable")
    unique_events, start_index, counts = np.unique(
        selected_events[order], return_index=True, return_counts=True
    )
    sums = np.add.reduceat(selected_sq[order], start_index)
    return unique_events, sums, counts.astype(np.float64)


def paired_truth_distance_ratio_ci(
    candidate_distance,
    baseline_distance,
    event_ids,
    candidate_valid,
    baseline_valid,
    *,
    bootstrap_seed: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_quantile: float = DEFAULT_CI_QUANTILE,
) -> dict[str, Any]:
    """Event-block bootstrap paired ratio RMS(candidate) / RMS(baseline).

    Only hits valid in *both* arms are compared (``candidate_valid &
    baseline_valid``), matching the pairing convention already used by
    ``scripts/analysis/measure_all_hit_refit.py::paired_block``.  Resamples
    whole events with replacement (mirroring
    ``reconstruction_acceptance.build_acceptance_baseline``'s own bootstrap:
    per-event sum-of-squares and counts, resampled event indices, quantile of
    the resampled ratio distribution) so hits from the same track stay
    together under resampling.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.5 < ci_quantile < 1.0:
        raise ValueError("ci_quantile must be an upper-tail bound in (0.5, 1.0)")
    candidate_distance = np.asarray(candidate_distance, dtype=np.float64)
    baseline_distance = np.asarray(baseline_distance, dtype=np.float64)
    if candidate_distance.shape != baseline_distance.shape:
        raise ValueError(
            "candidate and baseline distances are not paired on the same "
            "events/hits: shapes differ "
            f"({candidate_distance.shape} vs {baseline_distance.shape})"
        )
    shared_valid = np.asarray(candidate_valid, dtype=bool) & np.asarray(baseline_valid, dtype=bool)

    # Fail closed on corrupt inputs up front, rather than letting a NaN/Inf
    # quietly propagate into a "finite-looking" ci_upper later (a resample
    # that happens to avoid the bad hit could otherwise look like a clean
    # pass).  Only hits that actually enter the aggregation are checked, so a
    # non-finite distance on a hit already excluded by shared_valid is inert.
    for label, distance in (("candidate", candidate_distance), ("baseline", baseline_distance)):
        selected = distance.reshape(-1)[shared_valid.reshape(-1)]
        if selected.size and not np.isfinite(selected).all():
            raise ValueError(
                f"{label}_distance contains non-finite values on hits valid in "
                "both arms; refusing to compute a truth-distance ratio from them"
            )

    events_c, sums_c, counts_c = _per_event_sums(candidate_distance, event_ids, shared_valid)
    events_b, sums_b, counts_b = _per_event_sums(baseline_distance, event_ids, shared_valid)
    if events_c.size < 2:
        raise ValueError("need at least two distinct events to bootstrap by event")

    def _rms(sum_sq, count):
        return np.sqrt(sum_sq / count)

    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_rms = float(_rms(sums_b.sum(), counts_c.sum()))
    if not (math.isfinite(baseline_rms) and baseline_rms > 0):
        raise ValueError(
            "overall baseline RMS is zero or non-finite; a truth-distance "
            "ratio against it is undefined"
        )
    nominal_ratio = float(_rms(sums_c.sum(), counts_c.sum()) / baseline_rms)

    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, events_c.size, size=(n_bootstrap, events_c.size))
    resampled_counts = counts_c[draws].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # A resampled event-block landing on all-zero-baseline events divides
        # by zero here; that is exactly what the finiteness check below must
        # catch, not a computation bug to warn about.
        resampled_ratio = _rms(sums_c[draws].sum(axis=1), resampled_counts) / _rms(
            sums_b[draws].sum(axis=1), resampled_counts
        )
    n_finite = int(np.isfinite(resampled_ratio).sum())
    if n_finite < n_bootstrap:
        raise ValueError(
            f"{n_bootstrap - n_finite} of {n_bootstrap} bootstrap resamples were "
            "non-finite (a resampled event-block hit zero baseline RMS); "
            "refusing to report a ci_upper silently computed from the "
            "remaining subset"
        )
    ci_upper = float(np.quantile(resampled_ratio, ci_quantile, method="higher"))

    return {
        "n_events": int(events_c.size),
        "n_hits": int(counts_c.sum()),
        "nominal_ratio": nominal_ratio,
        "ci_upper": ci_upper,
        "ci_quantile": ci_quantile,
        "n_finite_resamples": n_finite,
        "bootstrap": {"seed": bootstrap_seed, "replicates": n_bootstrap},
    }


@dataclass(frozen=True)
class TruthMetricSpec:
    """One pre-registered truth-distance check inside a veto verdict."""

    name: str
    ci_upper_limit: float
    strict: bool  # True: ci_upper must be < limit (primary); False: <= limit (required)


def primary_metric(name: str, ci_upper_limit: float = PRIMARY_CI_UPPER_LIMIT) -> TruthMetricSpec:
    return TruthMetricSpec(name=name, ci_upper_limit=ci_upper_limit, strict=True)


def required_metric(name: str, ci_upper_limit: float = REQUIRED_CI_UPPER_LIMIT) -> TruthMetricSpec:
    return TruthMetricSpec(name=name, ci_upper_limit=ci_upper_limit, strict=False)


def evaluate_truth_veto(
    metric_results: Mapping[str, Mapping[str, Any]],
    metric_specs: Sequence[TruthMetricSpec],
    *,
    after_candidate_selection: bool,
) -> dict[str, Any]:
    """Pre-registered pass/fail rule: primary CI-upper < 1.0, every other required
    metric's CI-upper <= 1.01.  Each metric is judged independently -- there is
    no combined statistic, so a single failing metric fails the whole
    candidate with no cross-layer or cross-component averaging.
    """
    if after_candidate_selection is not True:
        raise ValueError(
            "evaluate_truth_veto must only run after candidate selection; "
            "pass after_candidate_selection=True at the one call site that "
            "checks an already-selected candidate, never inside a selection "
            "or tuning loop"
        )
    if not metric_specs:
        raise ValueError("at least one metric spec is required")
    if not any(spec.strict for spec in metric_specs):
        raise ValueError("metric_specs must include at least one primary metric")
    names = [spec.name for spec in metric_specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate metric names in metric_specs")

    failed: list[str] = []
    evaluated: dict[str, Any] = {}
    for spec in metric_specs:
        if spec.name not in metric_results:
            raise KeyError(f"missing truth metric result for required spec {spec.name!r}")
        metric_result = metric_results[spec.name]
        ci_upper = metric_result["ci_upper"]
        if not math.isfinite(ci_upper):
            raise ValueError(
                f"truth metric {spec.name!r} has a non-finite ci_upper; cannot "
                "judge a veto from it"
            )
        nominal_ratio = metric_result.get("nominal_ratio")
        if nominal_ratio is not None and not math.isfinite(nominal_ratio):
            raise ValueError(
                f"truth metric {spec.name!r} has a non-finite nominal_ratio; "
                "cannot judge a veto from it"
            )
        n_finite_resamples = metric_result.get("n_finite_resamples")
        replicates = metric_result.get("bootstrap", {}).get("replicates")
        if (
            n_finite_resamples is not None
            and replicates is not None
            and n_finite_resamples != replicates
        ):
            raise ValueError(
                f"truth metric {spec.name!r} has only {n_finite_resamples} of "
                f"{replicates} finite bootstrap resamples; refusing to judge a "
                "veto from a partial resample"
            )
        ok = (ci_upper < spec.ci_upper_limit) if spec.strict else (ci_upper <= spec.ci_upper_limit)
        evaluated[spec.name] = {
            "ci_upper": ci_upper,
            "limit": spec.ci_upper_limit,
            "strict": spec.strict,
            "passed": bool(ok),
        }
        if not ok:
            failed.append(spec.name)

    return {
        "passed": not failed,
        "failed_metrics": failed,
        "metrics": evaluated,
        "claim_scope": "post_hoc_truth_veto_only",
    }
