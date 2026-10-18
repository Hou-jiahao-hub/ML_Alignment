"""Pinned constants and artifact layout of the pT=100 GeV ACTS-vs-Schur study.

ACTS-free so the ACTS adapter, the Schur driver and the comparison share one
definition. Constants are read at call time so a smoke run can patch them.
"""
from __future__ import annotations

from pathlib import Path

from alignment_artifact_layout import artifact_filename

PT100_ALIGNMENT_RUN = "pt100-align-20261015"
PT100_ALIGNMENT_EVENTS = 20_000
PT100_MISALIGNMENT_SEED = 20261015
# runAlignment's historical cap; smoke runs lower it (each iteration is a dense
# ~13608x13608 inverse + full-pivot LU, ~13 min single-threaded).
PT100_ACTS_MAX_ITERATIONS = 100
PT100_SOURCE_RUN = "material-helix-20261014-r2"
PT100_SOURCE_GROUP = "g3_pt100_material_on"
PT100_STAGES = ("acts-alignment", "misaligned", "aligned-acts", "aligned-schur")
SCHUR_ARMS = ("schur_single_thread", "schur_multi_thread")


def pt100_alignment_paths(data_root, target_dir, run_id):
    root = Path(data_root)
    recon = root / "reconstruction" / run_id
    alignment = root / "alignment" / run_id / target_dir
    experiment = root / "experiments" / "pt100_alignment" / target_dir / run_id
    return {
        "simulation": root / "simulation" / PT100_SOURCE_RUN / PT100_SOURCE_GROUP,
        "nominal_source": (root / "reconstruction" / "nominal" / target_dir
                           / PT100_SOURCE_RUN / PT100_SOURCE_GROUP),
        "alignment": alignment,
        "misalignment": alignment / artifact_filename("misalignment_applied", ".json"),
        "acts_geometry": alignment / artifact_filename("aligned_transforms", ".json"),
        "experiment": experiment,
        **{arm: experiment / arm for arm in SCHUR_ARMS},
        "schur_geometry": (experiment / "schur_single_thread"
                           / f"aligned_transforms_{run_id}_schur_lm.json"),
        "reconstruction": recon,
        "nominal": recon / "nominal" / target_dir,
        "misaligned": recon / "misaligned" / target_dir,
        "aligned-acts": recon / "aligned-acts" / target_dir,
        "aligned-schur": recon / "aligned-schur" / target_dir,
        "comparison": experiment / "comparison",
    }
