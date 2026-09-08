from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_geometry(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "ID": "vol=17|lay=8|sen=1",
                    "Translation": [0.0, 0.0, 0.0],
                    "RotationMatrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_compare_truth_rms_invalidates_an_undefined_ratio(tmp_path):
    """A zero baseline RMS has no ratio and must never satisfy non-inferiority."""
    geometry = tmp_path / "geometry.json"
    report = tmp_path / "report.json"
    _write_geometry(geometry)
    script = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "compare_truth_rms.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--misaligned",
            str(geometry),
            "--nominal",
            str(geometry),
            "--baseline",
            str(geometry),
            "--candidate",
            str(geometry),
            "--target-layers",
            "17:8",
            "--resamples",
            "20",
            "--json",
            str(report),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"] == "invalidated"
    assert payload["layers"]["17:8"]["dof"]["tx"]["verdict"] == "n/a"
    assert payload["undefined"]
