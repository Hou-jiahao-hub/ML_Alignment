import hashlib
import json

import pytest

from write_experiment_summary import WORKFLOW, build_summary, joint_solver_effective


def _joint(best_iteration=19, correction_max_abs=1.5):
    return {"best_iteration": best_iteration, "correction_max_abs": correction_max_abs}


PROFILE_SHA256 = "ab" * 32
PROFILE = {
    "schema_version": "3.1",
    "selection_version": "full_event_bootstrap_v1",
    "event_identity": {"nominal_events": 4, "event_key_sha256": "cd" * 32},
}


def _acceptance(passed):
    return {
        "passed": passed,
        "failed_components": [] if passed else ["17:8:LOC0"],
        "schema_version": "3.1",
        "profile_identity_verified": True,
        "threshold_profile_sha256": PROFILE_SHA256,
        "profile_type": "diagnostic_only",
        "verdict_eligible": False,
    }


def _summary(**overrides):
    kwargs = dict(
        run_tag="trial-test",
        joint=_joint(),
        joint_adopted=True,
        final_acceptance=_acceptance(True),
        acceptance_baseline="baseline.json",
        acceptance_profile=PROFILE,
        acceptance_baseline_sha256=PROFILE_SHA256,
        final_geometry="geom.json",
        final_trackstates="states.root",
        final_tracksummary="summary.root",
        polish_attempts=0,
        selection_stop_reason="joint_acceptance_passed",
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
    assert summary["stop_reason"] == "acceptance_failed"


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
    assert summary["stop_reason"] == "acceptance_passed"


def test_a_no_op_joint_solver_is_not_effective():
    summary = _summary(joint=_joint(best_iteration=-1, correction_max_abs=0.0))
    assert summary["joint_solver_effective"] is False
    assert summary["joint_effective"] is False
    assert summary["passed"] is False


def test_summary_never_claims_a_hash_split():
    """The summary must not report partitions the run never used."""
    summary = _summary()
    assert "event_split" not in summary
    assert "calibration_acceptance" not in summary
    assert summary["workflow"] == WORKFLOW
    assert summary["acceptance_scope"] == "diagnostic_only"
    assert summary["acceptance_baseline_sha256"] == PROFILE_SHA256


@pytest.mark.parametrize(
    ("acceptance", "profile", "profile_sha256", "message"),
    [
        ({"passed": True}, PROFILE, PROFILE_SHA256, "schema_version"),
        (
            {**_acceptance(True), "profile_identity_verified": False},
            PROFILE,
            PROFILE_SHA256,
            "profile_identity_verified",
        ),
        (
            {**_acceptance(True), "threshold_profile_sha256": "ef" * 32},
            PROFILE,
            PROFILE_SHA256,
            "hash",
        ),
        (
            {**_acceptance(True), "profile_type": "validation"},
            PROFILE,
            PROFILE_SHA256,
            "profile_type",
        ),
        (
            {**_acceptance(True), "verdict_eligible": True},
            PROFILE,
            PROFILE_SHA256,
            "verdict_eligible",
        ),
        (_acceptance(True), {**PROFILE, "schema_version": "2.0"}, PROFILE_SHA256, "schema"),
        (
            _acceptance(True),
            {**PROFILE, "selection_version": "acceptance_v2"},
            PROFILE_SHA256,
            "selection_version",
        ),
    ],
)
def test_summary_fails_closed_without_the_frozen_verified_profile(
    acceptance, profile, profile_sha256, message
):
    with pytest.raises(ValueError, match=message):
        _summary(
            final_acceptance=acceptance,
            acceptance_profile=profile,
            acceptance_baseline_sha256=profile_sha256,
        )


def test_main_hashes_the_baseline_file_and_records_it(tmp_path):
    from write_experiment_summary import main

    profile_path = tmp_path / "baseline.json"
    profile_path.write_text(json.dumps(PROFILE))
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    acceptance = {**_acceptance(True), "threshold_profile_sha256": profile_sha256}
    joint_path = tmp_path / "joint.json"
    acceptance_path = tmp_path / "acceptance.json"
    output = tmp_path / "summary.json"
    joint_path.write_text(json.dumps(_joint()))
    acceptance_path.write_text(json.dumps(acceptance))

    status = main(
        [
            "--run-tag", "trial-test",
            "--joint-summary", str(joint_path),
            "--joint-adopted", "1",
            "--final-acceptance", str(acceptance_path),
            "--acceptance-baseline", str(profile_path),
            "--final-geometry", "geom.json",
            "--final-trackstates", "states.root",
            "--final-tracksummary", "summary.root",
            "--polish-attempts", "0",
            "--selection-stop-reason", "joint_acceptance_passed",
            "--output", str(output),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text())["acceptance_baseline_sha256"] == profile_sha256


def test_final_acceptance_is_required():
    kwargs = dict(
        run_tag="trial-test",
        joint=_joint(),
        joint_adopted=True,
        acceptance_baseline="baseline.json",
        acceptance_profile=PROFILE,
        acceptance_baseline_sha256=PROFILE_SHA256,
        final_geometry="geom.json",
        final_trackstates="states.root",
        final_tracksummary="summary.root",
        polish_attempts=0,
        selection_stop_reason="joint_acceptance_passed",
    )
    with pytest.raises(TypeError):
        build_summary(**kwargs)
