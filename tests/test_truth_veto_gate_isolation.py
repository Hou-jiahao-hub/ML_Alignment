"""Operational regression: truth_veto_gate must never feed back into calibration.

Mirrors how AGENTS.md's ACTS-import boundary is enforced -- the L3 gate is a
post-hoc check only (see scripts/analysis/truth_veto_gate.py's module
docstring); nothing in the solver or acceptance/calibration path may import
or reference it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_REFERENCE_FILES = (
    "scripts/ml_alignment_autograd.py",
    "scripts/calibrate_acceptance.py",
    "scripts/reconstruction_acceptance.py",
    "scripts/run_surface_helix_experiment.sh",
)


def test_solver_and_calibration_modules_never_reference_truth_veto_gate():
    offenders = []
    for relative in FORBIDDEN_REFERENCE_FILES:
        path = REPO_ROOT / relative
        text = path.read_text(encoding="utf-8")
        if "truth_veto_gate" in text:
            offenders.append(relative)
    assert offenders == []
