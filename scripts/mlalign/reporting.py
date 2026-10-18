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
        "block_lm_summary",
        "schur_lm_summary",
    )
    summary = {key: _json_ready(result[key]) for key in retained_keys if key in result}
    if "predicted_deltas" in result:
        corrections = np.asarray(result["predicted_deltas"], dtype=float)
        summary["correction_max_abs"] = (
            float(np.max(np.abs(corrections))) if corrections.size else 0.0
        )
    fisher = result.get("fisher_diagnostics")
    if fisher is not None:
        finite_conditions = [
            float(item["condition_number"])
            for item in fisher
            if math.isfinite(float(item.get("condition_number", float("inf"))))
        ]
        regularizations = [float(item.get("regularization", 0.0)) for item in fisher]
        accepted_steps = sum(bool(item.get("accepted")) for item in fisher)
        summary["fisher_summary"] = {
            "evaluations": len(fisher),
            "accepted_steps": accepted_steps,
            "rejected_steps": len(fisher) - accepted_steps,
            "damping_retries": sum(int(item.get("attempt", 0) or 0) for item in fisher),
            "lstsq_fallbacks": sum(bool(item.get("lstsq_fallback")) for item in fisher),
            "clipped_steps": sum(bool(item.get("step_clipped")) for item in fisher),
            "finite_condition_max": max(finite_conditions) if finite_conditions else None,
            "regularization_max": max(regularizations) if regularizations else None,
        }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination
