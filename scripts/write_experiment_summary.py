from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 4
ACCEPTANCE_SCHEMA_VERSION = "3.1"
ACCEPTANCE_SELECTION_VERSION = "full_event_bootstrap_v1"

WORKFLOW = "schur-lm_full-events"

TARGET_LAYERS = ["17:8", "24:4"]
ACTIVE_DOF = [0, 1, 5]


def joint_solver_effective(joint: Mapping[str, Any]) -> bool:
    """Whether the joint solver produced a non-zero update at all.

    This is a statement about the solver, not about the geometry that the
    workflow ended up selecting.  A joint stage can be solver-effective and
    still be discarded by the acceptance guard.
    """
    return bool(joint.get("best_iteration", -1) >= 0 and joint.get("correction_max_abs", 0) > 0)


def build_summary(
    *,
    run_tag: str,
    joint: Mapping[str, Any],
    joint_adopted: bool,
    final_acceptance: Mapping[str, Any],
    acceptance_baseline: str,
    acceptance_profile: Mapping[str, Any],
    acceptance_baseline_sha256: str,
    final_geometry: str,
    final_trackstates: str,
    final_tracksummary: str,
    selection_stop_reason: str,
) -> dict[str, Any]:
    if final_acceptance.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError("final acceptance schema_version is missing or unsupported")
    if not final_acceptance.get("profile_identity_verified"):
        raise ValueError("final acceptance profile_identity_verified must be true")
    if final_acceptance.get("profile_type") != "diagnostic_only":
        raise ValueError("final acceptance profile_type must be diagnostic_only")
    if final_acceptance.get("verdict_eligible") is not False:
        raise ValueError("final acceptance verdict_eligible must be false")
    if final_acceptance.get("threshold_profile_sha256") != acceptance_baseline_sha256:
        raise ValueError("final acceptance threshold profile hash does not match baseline")
    if acceptance_profile.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
        raise ValueError("acceptance baseline schema is unsupported")
    if acceptance_profile.get("selection_version") != ACCEPTANCE_SELECTION_VERSION:
        raise ValueError("acceptance baseline selection_version is unsupported")
    identity = acceptance_profile.get("event_identity")
    if not isinstance(identity, Mapping) or not identity.get("event_key_sha256"):
        raise ValueError("acceptance baseline event identity is missing")
    solver_effective = joint_solver_effective(joint)
    # The Iteration 12 pass condition reads "joint has a non-zero effective
    # update".  A joint stage the guard reverted contributes nothing to the
    # final geometry, so it cannot satisfy that condition.
    effective = solver_effective and joint_adopted
    passed = bool(final_acceptance["passed"] and effective)
    stop_reason = "acceptance_passed" if passed else "acceptance_failed"

    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "workflow": WORKFLOW,
        "target_layers": list(TARGET_LAYERS),
        "active_dof": list(ACTIVE_DOF),
        # There is no held-out partition any more: every stage, including the
        # final acceptance check below, runs on the full nominal event set.
        "acceptance_scope": "diagnostic_only",
        "acceptance_baseline": acceptance_baseline,
        "acceptance_baseline_sha256": acceptance_baseline_sha256,
        "final_geometry": final_geometry,
        "final_trackstates": final_trackstates,
        "final_tracksummary": final_tracksummary,
        "schur_lm": dict(joint),
        "joint_effective": effective,
        "joint_solver_effective": solver_effective,
        "joint_adopted": joint_adopted,
        "final_acceptance": dict(final_acceptance),
        "selection_stop_reason": selection_stop_reason,
        "stop_reason": stop_reason,
        "diagnostics": {"truth_and_nominal": "diagnostic_only"},
        "passed": passed,
    }


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def _load_with_sha256(path: str) -> tuple[Any, str]:
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the experiment summary for one run.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--joint-summary", required=True)
    parser.add_argument("--joint-adopted", choices=("0", "1"), required=True)
    parser.add_argument("--final-acceptance", required=True)
    parser.add_argument("--acceptance-baseline", required=True)
    parser.add_argument("--final-geometry", required=True)
    parser.add_argument("--final-trackstates", required=True)
    parser.add_argument("--final-tracksummary", required=True)
    parser.add_argument("--selection-stop-reason", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    acceptance_profile, acceptance_baseline_sha256 = _load_with_sha256(
        args.acceptance_baseline
    )

    summary = build_summary(
        run_tag=args.run_tag,
        joint=_load(args.joint_summary),
        joint_adopted=args.joint_adopted == "1",
        final_acceptance=_load(args.final_acceptance),
        acceptance_baseline=args.acceptance_baseline,
        acceptance_profile=acceptance_profile,
        acceptance_baseline_sha256=acceptance_baseline_sha256,
        final_geometry=args.final_geometry,
        final_trackstates=args.final_trackstates,
        final_tracksummary=args.final_tracksummary,
        selection_stop_reason=args.selection_stop_reason,
    )
    Path(args.output).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
