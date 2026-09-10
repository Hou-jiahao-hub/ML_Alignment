import numpy as np
import pytest


def _module():
    import importlib

    return importlib.import_module("scripts.analysis.truth_veto_gate")


def test_paired_ratio_ci_matches_manual_bootstrap_on_toy_events():
    m = _module()
    # 4 events, 2 hits each; candidate distance is exactly half the baseline.
    event_ids = np.repeat([0, 1, 2, 3], 2)
    baseline = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    candidate = baseline * 0.5
    valid = np.ones_like(baseline, dtype=bool)

    result = m.paired_truth_distance_ratio_ci(
        candidate, baseline, event_ids, valid, valid, bootstrap_seed=1, n_bootstrap=200
    )

    assert result["nominal_ratio"] == pytest.approx(0.5)
    assert result["n_events"] == 4
    assert result["n_hits"] == 8


def test_paired_ratio_ci_is_reproducible_given_seed():
    m = _module()
    event_ids = np.repeat(np.arange(6), 3)
    baseline = np.linspace(1.0, 2.0, 18)
    candidate = baseline * 0.9
    valid = np.ones_like(baseline, dtype=bool)

    first = m.paired_truth_distance_ratio_ci(
        candidate, baseline, event_ids, valid, valid, bootstrap_seed=42, n_bootstrap=500
    )
    second = m.paired_truth_distance_ratio_ci(
        candidate, baseline, event_ids, valid, valid, bootstrap_seed=42, n_bootstrap=500
    )

    assert first == second


def test_paired_ratio_ci_only_uses_hits_valid_in_both_arms():
    m = _module()
    # 3 events x 2 hits; one hit in event 1 is invalid only in the baseline
    # arm.  It must be dropped from *both* arms, not just the baseline one,
    # so the two arms stay paired on an identical hit set.
    event_ids = np.repeat([0, 1, 2], 2)
    baseline = np.array([1.0, 1.0, 1.0, 999.0, 1.0, 1.0])
    candidate = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    candidate_valid = np.ones(6, dtype=bool)
    baseline_valid = np.array([True, True, True, False, True, True])

    result = m.paired_truth_distance_ratio_ci(
        candidate, baseline, event_ids, candidate_valid, baseline_valid, bootstrap_seed=1
    )

    assert result["n_hits"] == 5
    assert result["nominal_ratio"] == pytest.approx(2.0)


def test_paired_ratio_ci_rejects_mismatched_shapes_between_arms():
    m = _module()
    event_ids = np.array([0, 0, 1, 1])
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    candidate = np.array([1.0, 1.0, 1.0])
    valid = np.array([True, True, True, True])

    with pytest.raises(ValueError, match="not paired"):
        m.paired_truth_distance_ratio_ci(
            candidate, baseline, event_ids, valid[:3], valid, bootstrap_seed=1
        )


def test_paired_ratio_ci_raises_on_nonfinite_candidate_distance_over_valid_hits():
    m = _module()
    event_ids = np.array([0, 0, 1, 1])
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    candidate = np.array([1.0, np.nan, 1.0, 1.0])
    valid = np.array([True, True, True, True])

    with pytest.raises(ValueError, match="non-finite"):
        m.paired_truth_distance_ratio_ci(
            candidate, baseline, event_ids, valid, valid, bootstrap_seed=1
        )


def test_paired_ratio_ci_ignores_nonfinite_candidate_distance_on_hits_invalid_in_either_arm():
    m = _module()
    # The non-finite entry sits on a hit that is invalid in the *baseline* arm,
    # so it is excluded from both arms by the shared_valid mask and must not
    # trip the finiteness guard at all.
    event_ids = np.array([0, 0, 1, 1])
    baseline = np.array([1.0, 1.0, 1.0, 1.0])
    candidate = np.array([1.0, np.nan, 1.0, 1.0])
    candidate_valid = np.array([True, True, True, True])
    baseline_valid = np.array([True, False, True, True])

    result = m.paired_truth_distance_ratio_ci(
        candidate, baseline, event_ids, candidate_valid, baseline_valid, bootstrap_seed=1
    )
    assert result["n_hits"] == 3


def test_paired_ratio_ci_raises_on_nonfinite_baseline_distance_over_valid_hits():
    m = _module()
    event_ids = np.array([0, 0, 1, 1])
    baseline = np.array([1.0, 1.0, np.inf, 1.0])
    candidate = np.array([1.0, 1.0, 1.0, 1.0])
    valid = np.array([True, True, True, True])

    with pytest.raises(ValueError, match="non-finite"):
        m.paired_truth_distance_ratio_ci(
            candidate, baseline, event_ids, valid, valid, bootstrap_seed=1
        )


def test_paired_ratio_ci_raises_on_zero_overall_baseline_rms():
    m = _module()
    event_ids = np.array([0, 0, 1, 1])
    baseline = np.array([0.0, 0.0, 0.0, 0.0])
    candidate = np.array([1.0, 1.0, 1.0, 1.0])
    valid = np.array([True, True, True, True])

    with pytest.raises(ValueError, match="baseline RMS"):
        m.paired_truth_distance_ratio_ci(
            candidate, baseline, event_ids, valid, valid, bootstrap_seed=1
        )


def test_paired_ratio_ci_raises_when_any_bootstrap_resample_is_nonfinite():
    m = _module()
    # Event 0's baseline distance is exactly zero; a resample that draws only
    # event 0 divides by a zero baseline RMS for that replicate.  The overall
    # (all-events) baseline RMS is still non-zero, so this must fail inside
    # the bootstrap-finiteness guard specifically, not the overall-RMS guard.
    event_ids = np.array([0, 0, 1, 1])
    baseline = np.array([0.0, 0.0, 2.0, 2.0])
    candidate = np.array([1.0, 1.0, 1.0, 1.0])
    valid = np.array([True, True, True, True])

    with pytest.raises(ValueError, match="non-finite"):
        m.paired_truth_distance_ratio_ci(
            candidate, baseline, event_ids, valid, valid, bootstrap_seed=1, n_bootstrap=200
        )


def test_paired_ratio_ci_rejects_single_event():
    m = _module()
    event_ids = np.array([0, 0, 0])
    baseline = np.array([1.0, 1.0, 1.0])
    candidate = np.array([1.0, 1.0, 1.0])
    valid = np.array([True, True, True])

    with pytest.raises(ValueError, match="at least two"):
        m.paired_truth_distance_ratio_ci(
            candidate, baseline, event_ids, valid, valid, bootstrap_seed=1
        )


def _ci(ci_upper, *, nominal_ratio=None, n_finite_resamples=2000, replicates=2000):
    return {
        "ci_upper": ci_upper,
        "nominal_ratio": ci_upper if nominal_ratio is None else nominal_ratio,
        "n_finite_resamples": n_finite_resamples,
        "bootstrap": {"seed": 1, "replicates": replicates},
    }


def test_evaluate_truth_veto_passes_when_all_ci_uppers_within_registered_limits():
    m = _module()
    specs = [m.primary_metric("target_ratio"), m.required_metric("non_target_ratio")]
    results = {"target_ratio": _ci(0.99), "non_target_ratio": _ci(1.005)}

    verdict = m.evaluate_truth_veto(results, specs, after_candidate_selection=True)

    assert verdict["passed"] is True
    assert verdict["failed_metrics"] == []


def test_evaluate_truth_veto_fails_when_primary_ci_upper_equals_one():
    m = _module()
    specs = [m.primary_metric("target_ratio")]
    results = {"target_ratio": _ci(1.0)}

    verdict = m.evaluate_truth_veto(results, specs, after_candidate_selection=True)

    assert verdict["passed"] is False
    assert verdict["failed_metrics"] == ["target_ratio"]


def test_evaluate_truth_veto_fails_when_a_single_required_metric_exceeds_1_01_even_if_primary_passes():
    m = _module()
    specs = [
        m.primary_metric("target_ratio"),
        m.required_metric("tail_ratio"),
        m.required_metric("non_target_ratio"),
    ]
    results = {
        "target_ratio": _ci(0.5),  # primary looks great
        "tail_ratio": _ci(1.02),  # this alone must kill the candidate
        "non_target_ratio": _ci(1.0),
    }

    verdict = m.evaluate_truth_veto(results, specs, after_candidate_selection=True)

    assert verdict["passed"] is False
    assert verdict["failed_metrics"] == ["tail_ratio"]
    assert verdict["metrics"]["target_ratio"]["passed"] is True


def test_evaluate_truth_veto_requires_after_candidate_selection_latch():
    m = _module()
    specs = [m.primary_metric("target_ratio")]
    results = {"target_ratio": _ci(0.5)}

    with pytest.raises(ValueError, match="after_candidate_selection"):
        m.evaluate_truth_veto(results, specs, after_candidate_selection=False)


def test_evaluate_truth_veto_requires_at_least_one_primary_metric():
    m = _module()
    specs = [m.required_metric("tail_ratio")]
    results = {"tail_ratio": _ci(1.0)}

    with pytest.raises(ValueError, match="primary metric"):
        m.evaluate_truth_veto(results, specs, after_candidate_selection=True)


def test_evaluate_truth_veto_raises_on_missing_metric_result():
    m = _module()
    specs = [m.primary_metric("target_ratio")]

    with pytest.raises(KeyError):
        m.evaluate_truth_veto({}, specs, after_candidate_selection=True)


def test_evaluate_truth_veto_raises_on_nonfinite_ci_upper():
    m = _module()
    specs = [m.primary_metric("target_ratio")]
    results = {"target_ratio": _ci(float("nan"))}

    with pytest.raises(ValueError, match="non-finite ci_upper"):
        m.evaluate_truth_veto(results, specs, after_candidate_selection=True)


def test_evaluate_truth_veto_raises_on_nonfinite_nominal_ratio():
    m = _module()
    specs = [m.primary_metric("target_ratio")]
    results = {"target_ratio": _ci(0.5, nominal_ratio=float("inf"))}

    with pytest.raises(ValueError, match="non-finite nominal_ratio"):
        m.evaluate_truth_veto(results, specs, after_candidate_selection=True)


def test_evaluate_truth_veto_raises_on_incomplete_bootstrap_resamples():
    m = _module()
    specs = [m.primary_metric("target_ratio")]
    results = {"target_ratio": _ci(0.5, n_finite_resamples=1990, replicates=2000)}

    with pytest.raises(ValueError, match="partial resample"):
        m.evaluate_truth_veto(results, specs, after_candidate_selection=True)


def test_evaluate_truth_veto_raises_on_duplicate_metric_names():
    m = _module()
    specs = [m.primary_metric("target_ratio"), m.required_metric("target_ratio")]
    results = {"target_ratio": _ci(0.5)}

    with pytest.raises(ValueError, match="duplicate"):
        m.evaluate_truth_veto(results, specs, after_candidate_selection=True)
