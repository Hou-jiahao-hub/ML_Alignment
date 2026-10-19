"""Compact JSON summary of a solver run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_alignment_summary(result, output_path):
    """Write compact solver observability without duplicating hit-sized arrays."""
    retained_keys = (
        "output_geom",
        "n_tracks",
        "n_hits",
        "initialization_stats",
        "stage_layer_chi2",
        "target_metrics",
        "truth_metrics",
        "best_total_loss",
        "last_total_loss",
        "best_iteration",
        "stopped_early",
        "initial_total_loss",
        "initial_invalid_fraction",
        "invalid_fraction",
        "solver",
        "stop_reason",
        "schur_lm_summary",
    )
    summary = {key: _json_ready(result[key]) for key in retained_keys if key in result}
    if "predicted_deltas" in result:
        corrections = np.asarray(result["predicted_deltas"], dtype=float)
        summary["correction_max_abs"] = (
            float(np.max(np.abs(corrections))) if corrections.size else 0.0
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination
