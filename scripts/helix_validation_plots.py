"""Residual plots for the nominal helix-model validation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HISTOGRAM_BINS = 60
PREDICTION_MODES = (
    "interpolation",
    "forward_extrapolation",
    "backward_extrapolation",
)


def _finite(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def fitted_residual_components(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return finite fitted residual components from valid intersections."""
    required = {
        "fitted_intersection_valid",
        "fitted_measurement_residual_LOC0",
        "fitted_measurement_residual_LOC1",
        "fitted_dg_x",
        "fitted_dg_y",
        "fitted_dg_z",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Comparison table is missing columns: {', '.join(missing)}")

    valid = frame.loc[frame["fitted_intersection_valid"].astype(bool)]
    global_xyz = valid[["fitted_dg_x", "fitted_dg_y", "fitted_dg_z"]].to_numpy(
        dtype=float
    )
    finite_global_rows = np.isfinite(global_xyz).all(axis=1)
    return {
        "local0": _finite(valid["fitted_measurement_residual_LOC0"]),
        "local1": _finite(valid["fitted_measurement_residual_LOC1"]),
        "global_x": _finite(valid["fitted_dg_x"]),
        "global_y": _finite(valid["fitted_dg_y"]),
        "global_z": _finite(valid["fitted_dg_z"]),
        "global_norm": np.linalg.norm(global_xyz[finite_global_rows], axis=1),
    }


def _statistics(values: np.ndarray) -> str:
    if values.size == 0:
        return "No finite residuals"
    mean = float(np.mean(values))
    rms = float(np.sqrt(np.mean(np.square(values))))
    return f"N={values.size}\nmean={mean:.4g} mm\nRMS={rms:.4g} mm"


def _draw_histogram(axis, values: np.ndarray, title: str, xlabel: str) -> None:
    if values.size == 0:
        axis.text(
            0.5,
            0.5,
            "No finite fitted residuals",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        axis.hist(values, bins=HISTOGRAM_BINS, color="#2563eb", alpha=0.82)
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.text(
        0.97,
        0.95,
        _statistics(values),
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=9,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Entries")
    axis.grid(alpha=0.2)


def _scope_caption(frame: pd.DataFrame, title: str) -> str:
    total = len(frame)
    valid = int(frame["fitted_intersection_valid"].astype(bool).sum()) if total else 0
    failure_rate = 1.0 - valid / total if total else 1.0
    return f"{title} — fitted helix (valid {valid}/{total}, failure {failure_rate:.2%})"


def _write_local_plot(frame: pd.DataFrame, path: Path, title: str) -> None:
    components = fitted_residual_components(frame)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    try:
        _draw_histogram(
            axes[0], components["local0"], "Local residual LOC0", "ΔLOC0 [mm]"
        )
        _draw_histogram(
            axes[1], components["local1"], "Local residual LOC1", "ΔLOC1 [mm]"
        )
        figure.suptitle(_scope_caption(frame, title))
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def _write_global_plot(frame: pd.DataFrame, path: Path, title: str) -> None:
    components = fitted_residual_components(frame)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    try:
        panels = (
            ("global_x", "Global residual x", "Δx [mm]"),
            ("global_y", "Global residual y", "Δy [mm]"),
            ("global_z", "Global residual z", "Δz [mm]"),
            ("global_norm", "Global residual magnitude", "|Δr| [mm]"),
        )
        for axis, (key, panel_title, xlabel) in zip(axes.flat, panels):
            _draw_histogram(axis, components[key], panel_title, xlabel)
        figure.suptitle(_scope_caption(frame, title))
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def _plot_scopes(
    frame: pd.DataFrame,
    holdout_layers: tuple[tuple[int, int], ...],
) -> list[tuple[str, str, pd.DataFrame]]:
    scopes = [
        ("fit", "Fit surfaces", frame.loc[frame["split"] == "fit"]),
        (
            "holdout_overall",
            "Holdout surfaces (overall)",
            frame.loc[frame["split"] == "holdout"],
        ),
    ]
    for mode in PREDICTION_MODES:
        selected = frame.loc[
            (frame["split"] == "holdout")
            & (frame["prediction_mode"] == mode)
        ]
        scopes.append(
            (
                f"holdout_{mode}",
                f"Holdout surfaces ({mode})",
                selected,
            )
        )
    for volume, layer in holdout_layers:
        selected = frame.loc[
            (frame["split"] == "holdout")
            & (frame["volume_id"] == volume)
            & (frame["layer_id"] == layer)
        ]
        scopes.append(
            (
                f"holdout_vol{volume}_lay{layer}",
                f"Holdout layer {volume}:{layer}",
                selected,
            )
        )
        for mode in PREDICTION_MODES:
            mode_selected = selected.loc[selected["prediction_mode"] == mode]
            scopes.append(
                (
                    f"holdout_vol{volume}_lay{layer}_{mode}",
                    f"Holdout layer {volume}:{layer} ({mode})",
                    mode_selected,
                )
            )
    return scopes


def write_residual_plots(
    frame: pd.DataFrame,
    output_dir: str | Path,
    holdout_layers: tuple[tuple[int, int], ...],
) -> list[Path]:
    """Write local and global fitted-residual PNGs for validation scopes."""
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for slug, title, selected in _plot_scopes(frame, holdout_layers):
        local_path = plot_dir / f"{slug}_local_residuals.png"
        global_path = plot_dir / f"{slug}_global_residuals.png"
        _write_local_plot(selected, local_path, title)
        _write_global_plot(selected, global_path, title)
        paths.extend((local_path, global_path))
    return paths
