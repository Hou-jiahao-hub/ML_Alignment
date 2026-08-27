import pytest

from write_experiment_summary import build_summary, joint_solver_effective


def _joint(best_iteration=19, correction_max_abs=1.5):
    return {"best_iteration": best_iteration, "correction_max_abs": correction_max_abs}


def _acceptance(passed):
    return {"passed": passed, "failed_components": [] if passed else ["17:8:LOC0"]}


def _summary(**overrides):
    kwargs = dict(
        run_tag="trial-test",
        joint=_joint(),
        joint_adopted=True,
        final_acceptance=_acceptance(True),
        calibration_acceptance=_acceptance(True),
        acceptance_baseline="baseline.json",
        final_geometry="geom.json",
        final_trackstates="states.root",
        final_tracksummary="summary.root",
        polish_attempts=0,
        selection_stop_reason="joint_acceptance_passed",
        event_split_seed=6006,
        blind_validation_skipped=False,
    )
    kwargs.update(overrides)
    return build_summary(**kwargs)


def test_selection_stop_reason_survives_into_the_summary():
    """trial-007 lost `joint_not_improved` because finalize ignored the argument."""
    summary = _summary(
        joint_adopted=False,
        selection_stop_reason="joint_not_improved",
        final_acceptance=_acceptance(False),
    )
    assert summary["selection_stop_reason"] == "joint_not_improved"
    assert summary["stop_reason"] == "blind_validation_failed"


def test_a_reverted_joint_is_not_effective_even_when_the_solver_moved():
    """The Iteration 12 pass condition must not be satisfiable by a discarded joint."""
    joint = _joint(best_iteration=19, correction_max_abs=1.5)
    assert joint_solver_effective(joint) is True

    summary = _summary(joint=joint, joint_adopted=False, selection_stop_reason="joint_not_improved")
    assert summary["joint_solver_effective"] is True
    assert summary["joint_adopted"] is False
    assert summary["joint_effective"] is False
    # acceptance passed, but the joint contributed nothing to the final geometry
    assert summary["passed"] is False


def test_an_adopted_and_effective_joint_passes_when_acceptance_passes():
    summary = _summary(joint_adopted=True, final_acceptance=_acceptance(True))
    assert summary["joint_effective"] is True
    assert summary["passed"] is True
    assert summary["stop_reason"] == "blind_validation_passed"


def test_a_no_op_joint_solver_is_not_effective():
    summary = _summary(joint=_joint(best_iteration=-1, correction_max_abs=0.0))
    assert summary["joint_solver_effective"] is False
    assert summary["joint_effective"] is False
    assert summary["passed"] is False


def test_skipping_blind_validation_leaves_the_verdict_undecided():
    summary = _summary(blind_validation_skipped=True, final_acceptance=None)
    assert summary["passed"] is None
    assert summary["stop_reason"] == "stopped_before_blind_validation"
    assert summary["blind_validation_skipped"] is True
    assert summary["final_acceptance"] is None
    assert summary["workflow"].endswith("calibration-only")
    # the calibration evidence is still recorded
    assert summary["calibration_acceptance"] == _acceptance(True)


def test_a_full_run_keeps_the_blind_validation_workflow_name():
    summary = _summary()
    assert summary["blind_validation_skipped"] is False
    assert summary["workflow"].endswith("blind-validation")


def test_final_acceptance_is_required_unless_validation_is_skipped():
    with pytest.raises(ValueError):
        _summary(final_acceptance=None)
