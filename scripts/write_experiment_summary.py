from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 2

WORKFLOW_FULL = "block-lm_surface-polish2_blind-validation"
WORKFLOW_CALIBRATION_ONLY = "block-lm_surface-polish2_calibration-only"

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
    final_acceptance: Mapping[str, Any] | None,
    calibration_acceptance: Mapping[str, Any],
    acceptance_baseline: str,
    final_geometry: str,
    final_trackstates: str,
    final_tracksummary: str,
    polish_attempts: int,
    selection_stop_reason: str,
    event_split_seed: int,
    blind_validation_skipped: bool,
) -> dict[str, Any]:
    solver_effective = joint_solver_effective(joint)
    # The Iteration 12 pass condition reads "joint has a non-zero effective
    # update".  A joint stage the guard reverted contributes nothing to the
    # final geometry, so it cannot satisfy that condition.
    effective = solver_effective and joint_adopted

    if blind_validation_skipped:
        passed: bool | None = None
        stop_reason = "stopped_before_blind_validation"
        workflow = WORKFLOW_CALIBRATION_ONLY
        acceptance: Mapping[str, Any] | None = None
    else:
        if final_acceptance is None:
            raise ValueError("final_acceptance is required unless blind validation is skipped")
        passed = bool(final_acceptance["passed"] and effective)
        stop_reason = "blind_validation_passed" if passed else "blind_validation_failed"
        workflow = WORKFLOW_FULL
        acceptance = final_acceptance

    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "workflow": workflow,
        "target_layers": list(TARGET_LAYERS),
        "active_dof": list(ACTIVE_DOF),
        "event_split": {
            "seed": event_split_seed,
            "train": 80,
            "calibration": 10,
            "validation": 10,
        },
        "acceptance_baseline": acceptance_baseline,
        "final_geometry": final_geometry,
        "final_trackstates": final_trackstates,
        "final_tracksummary": final_tracksummary,
        "joint_free": dict(joint),
        "joint_effective": effective,
        "joint_solver_effective": solver_effective,
        "joint_adopted": joint_adopted,
        "calibration_acceptance": dict(calibration_acceptance),
        "final_acceptance": dict(acceptance) if acceptance is not None else None,
        "polish_attempts": polish_attempts,
        "selection_stop_reason": selection_stop_reason,
        "stop_reason": stop_reason,
        "blind_validation_skipped": blind_validation_skipped,
        "diagnostics": {"truth_and_nominal": "diagnostic_only"},
        "passed": passed,
    }


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text())


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write the experiment summary for one run.")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--joint-summary", required=True)
    parser.add_argument("--joint-adopted", choices=("0", "1"), required=True)
    parser.add_argument("--calibration-acceptance", required=True)
    parser.add_argument("--final-acceptance", default=None)
    parser.add_argument("--acceptance-baseline", required=True)
    parser.add_argument("--final-geometry", required=True)
    parser.add_argument("--final-trackstates", required=True)
    parser.add_argument("--final-tracksummary", required=True)
    parser.add_argument("--polish-attempts", type=int, required=True)
    parser.add_argument("--selection-stop-reason", required=True)
    parser.add_argument("--event-split-seed", type=int, required=True)
    parser.add_argument("--skip-blind-validation", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    if args.skip_blind_validation:
        final_acceptance = None
    else:
        if args.final_acceptance is None:
            raise SystemExit("--final-acceptance is required unless --skip-blind-validation")
        final_acceptance = _load(args.final_acceptance)

    summary = build_summary(
        run_tag=args.run_tag,
        joint=_load(args.joint_summary),
        joint_adopted=args.joint_adopted == "1",
        final_acceptance=final_acceptance,
        calibration_acceptance=_load(args.calibration_acceptance),
        acceptance_baseline=args.acceptance_baseline,
        final_geometry=args.final_geometry,
        final_trackstates=args.final_trackstates,
        final_tracksummary=args.final_tracksummary,
        polish_attempts=args.polish_attempts,
        selection_stop_reason=args.selection_stop_reason,
        event_split_seed=args.event_split_seed,
        blind_validation_skipped=args.skip_blind_validation,
    )
    Path(args.output).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
