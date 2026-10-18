"""Compare nominal, misaligned, and aligned detector-transform geometries.

For each canonical ``(volume, layer, sensor)`` ID, misaligned and aligned
differences are expressed in the same nominal-surface local frame.  The script
is ACTS-independent and has no dependency on an alignment solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from experiment_manifest import (
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from project_config import absolute_project_path, add_config_argument, load_project_config


DOF_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
DOF_UNITS = ("mm", "mm", "mm", "rad", "rad", "rad")
DEFAULT_ZERO_TOLERANCE = 1e-12
ROTATION_TOLERANCE = 1e-6
CSV_FIELDS = (
    "canonical_id",
    "volume_id",
    "layer_id",
    "sensor_id",
    "dof",
    "unit",
    "misaligned_difference",
    "misaligned_abs_difference",
    "aligned_difference",
    "aligned_abs_difference",
    "recovery_percent",
)
ID_PATTERN = re.compile(
    r"^vol=(\d+)\|lay=(\d+)\|sen=(\d+)"
    r"(?:\|[A-Za-z][A-Za-z0-9_]*=[^|]+)*$"
)


@dataclass(frozen=True)
class SurfaceTransform:
    source_id: str
    translation: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class HistogramSeries:
    """Values and shared binning for one layer and one plotted DOF."""

    volume_id: int
    layer_id: int
    dof: str
    unit: str
    misaligned: np.ndarray
    aligned: np.ndarray
    bin_edges: np.ndarray
    misaligned_rms: float
    aligned_rms: float
    schur: np.ndarray | None = None
    schur_rms: float | None = None


def surface_key(surface_id: str) -> tuple[int, int, int]:
    """Return the canonical geometry key, ignoring fields such as ``ext``."""
    match = ID_PATTERN.fullmatch(surface_id)
    if match is None:
        raise ValueError(
            f"Surface ID has no vol/lay/sen fields: {surface_id!r}"
        )
    return tuple(map(int, match.groups()))


def _validated_transform(item: object, *, label: str, index: int) -> tuple[
    tuple[int, int, int], SurfaceTransform
]:
    if not isinstance(item, dict):
        raise ValueError(f"{label} geometry item {index} must be an object")
    source_id = item.get("ID")
    if not isinstance(source_id, str):
        raise ValueError(f"{label} geometry item {index} has no string ID")
    key = surface_key(source_id)

    try:
        translation = np.asarray(item["Translation"], dtype=np.float64)
        rotation = np.asarray(item["RotationMatrix"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} surface {source_id!r} has invalid transform data"
        ) from error
    if translation.shape != (3,):
        raise ValueError(
            f"{label} surface {source_id!r} Translation must have 3 values"
        )
    if rotation.shape != (3, 3):
        raise ValueError(
            f"{label} surface {source_id!r} RotationMatrix must be 3x3"
        )
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} surface {source_id!r} contains non-finite values")
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=ROTATION_TOLERANCE,
    ) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=ROTATION_TOLERANCE
    ):
        raise ValueError(
            f"{label} surface {source_id!r} RotationMatrix is not a proper rotation"
        )
    return key, SurfaceTransform(source_id, translation, rotation)


def load_geometry(
    path: str | Path, *, label: str
) -> dict[tuple[int, int, int], SurfaceTransform]:
    """Load and validate a detector-transform JSON file."""
    geometry_path = Path(path)
    try:
        with geometry_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} geometry is not valid JSON: {geometry_path}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} geometry must be a non-empty JSON list")

    transforms: dict[tuple[int, int, int], SurfaceTransform] = {}
    for index, item in enumerate(payload):
        key, transform = _validated_transform(item, label=label, index=index)
        if key in transforms:
            raise ValueError(
                f"{label} geometry contains duplicate canonical surface ID {key}"
            )
        transforms[key] = transform
    return transforms


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation using the project's ``Rz @ Ry @ Rx`` convention."""
    ry = -math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0)))
    cos_ry = math.cos(ry)
    if abs(cos_ry) <= 1e-12:
        raise ValueError(
            "Relative rotation is at gimbal lock; local rx and rz differences "
            "cannot be reported uniquely"
        )
    rx = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    rz = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return rx, ry, rz


def nominal_frame_difference(
    nominal: SurfaceTransform, target: SurfaceTransform
) -> np.ndarray:
    """Return the nominal-local 6-DOF displacement from nominal to target."""
    translation = nominal.rotation.T @ (target.translation - nominal.translation)
    relative_rotation = nominal.rotation.T @ target.rotation
    rotation = rotation_matrix_to_euler(relative_rotation)
    return np.concatenate((translation, np.asarray(rotation, dtype=np.float64)))


def compare_geometries(
    nominal: dict[tuple[int, int, int], SurfaceTransform],
    misaligned: dict[tuple[int, int, int], SurfaceTransform],
    aligned: dict[tuple[int, int, int], SurfaceTransform],
    *,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> list[dict[str, object]]:
    """Build long-form per-surface, per-DOF comparison rows."""
    if zero_tolerance < 0 or not math.isfinite(zero_tolerance):
        raise ValueError("zero_tolerance must be a finite non-negative value")
    misaligned_ids = set(misaligned)
    aligned_ids = set(aligned)
    if misaligned_ids != aligned_ids:
        missing_aligned = sorted(misaligned_ids - aligned_ids)
        missing_misaligned = sorted(aligned_ids - misaligned_ids)
        raise ValueError(
            "Misaligned and aligned geometry ID sets differ: "
            f"missing from aligned={missing_aligned[:5]}, "
            f"missing from misaligned={missing_misaligned[:5]}"
        )
    missing_nominal = sorted(misaligned_ids - set(nominal))
    if missing_nominal:
        raise ValueError(
            "Compared surface IDs are absent from nominal geometry: "
            f"{missing_nominal[:5]}"
        )

    rows: list[dict[str, object]] = []
    for volume, layer, sensor in sorted(misaligned_ids):
        key = (volume, layer, sensor)
        misaligned_difference = nominal_frame_difference(nominal[key], misaligned[key])
        aligned_difference = nominal_frame_difference(nominal[key], aligned[key])
        for index, (dof, unit) in enumerate(zip(DOF_NAMES, DOF_UNITS, strict=True)):
            initial = float(misaligned_difference[index])
            remaining = float(aligned_difference[index])
            initial_absolute = abs(initial)
            remaining_absolute = abs(remaining)
            recovery = (
                None
                if initial_absolute <= zero_tolerance
                else 100.0 * (1.0 - remaining_absolute / initial_absolute)
            )
            rows.append(
                {
                    "canonical_id": f"vol={volume}|lay={layer}|sen={sensor}",
                    "volume_id": volume,
                    "layer_id": layer,
                    "sensor_id": sensor,
                    "dof": dof,
                    "unit": unit,
                    "misaligned_difference": initial,
                    "misaligned_abs_difference": initial_absolute,
                    "aligned_difference": remaining,
                    "aligned_abs_difference": remaining_absolute,
                    "recovery_percent": recovery,
                }
            )
    return rows


def compare_geometry_files(
    nominal_path: str | Path,
    misaligned_path: str | Path,
    aligned_path: str | Path,
    *,
    zero_tolerance: float = DEFAULT_ZERO_TOLERANCE,
) -> list[dict[str, object]]:
    """Load three geometry files and return their long-form comparison."""
    nominal = load_geometry(nominal_path, label="Nominal")
    misaligned = load_geometry(misaligned_path, label="Misaligned")
    aligned = load_geometry(aligned_path, label="Aligned")
    return compare_geometries(
        nominal,
        misaligned,
        aligned,
        zero_tolerance=zero_tolerance,
    )


def _difference_statistics(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "max_absolute": float(np.max(np.abs(array))),
    }


def _dof_statistics(rows: list[dict[str, object]]) -> dict[str, object]:
    recovery = np.asarray(
        [row["recovery_percent"] for row in rows if row["recovery_percent"] is not None],
        dtype=np.float64,
    )
    recovery_statistics: dict[str, object] = {
        "valid_count": int(recovery.size),
        "undefined_count": len(rows) - int(recovery.size),
        "mean": float(np.mean(recovery)) if recovery.size else None,
        "median": float(np.median(recovery)) if recovery.size else None,
        "minimum": float(np.min(recovery)) if recovery.size else None,
        "maximum": float(np.max(recovery)) if recovery.size else None,
    }
    return {
        "unit": rows[0]["unit"],
        "misaligned_difference": _difference_statistics(
            float(row["misaligned_difference"]) for row in rows
        ),
        "aligned_difference": _difference_statistics(
            float(row["aligned_difference"]) for row in rows
        ),
        "recovery_percent": recovery_statistics,
    }


def _group_by_dof(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        dof: _dof_statistics([row for row in rows if row["dof"] == dof])
        for dof in DOF_NAMES
        if any(row["dof"] == dof for row in rows)
    }


def build_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate comparison rows overall and by detector layer."""
    surface_ids = {
        (row["volume_id"], row["layer_id"], row["sensor_id"]) for row in rows
    }
    layer_keys = sorted({(row["volume_id"], row["layer_id"]) for row in rows})
    by_layer: dict[str, object] = {}
    for volume, layer in layer_keys:
        layer_rows = [
            row
            for row in rows
            if row["volume_id"] == volume and row["layer_id"] == layer
        ]
        by_layer[f"vol={volume}|lay={layer}"] = {
            "surface_count": len(
                {
                    (row["volume_id"], row["layer_id"], row["sensor_id"])
                    for row in layer_rows
                }
            ),
            "dof": _group_by_dof(layer_rows),
        }
    return {
        "schema_version": 2,
        "surface_count": len(surface_ids),
        "comparison_row_count": len(rows),
        "conventions": {
            "canonical_id": "(volume_id, layer_id, sensor_id)",
            "coordinate_frame": "nominal geometry local frame",
            "difference_direction": "nominal -> misaligned/aligned",
            "rotation_order": "Rz @ Ry @ Rx",
            "translation_unit": "mm",
            "rotation_unit": "rad",
            "recovery_formula": "100 * (1 - abs(aligned_difference) / abs(misaligned_difference))",
            "undefined_recovery": "abs(misaligned_difference) <= zero_tolerance",
        },
        "overall": _group_by_dof(rows),
        "by_layer": by_layer,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def write_reports(
    rows: list[dict[str, object]], output_dir: str | Path
) -> tuple[Path, Path, dict[str, object]]:
    """Write the detailed CSV and schema-versioned JSON summary."""
    destination = Path(output_dir)
    csv_path = destination / "geometry_comparison.csv"
    summary_path = destination / "geometry_comparison_summary.json"

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(csv_path, csv_buffer.getvalue())

    summary = build_summary(rows)
    _atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return csv_path, summary_path, summary


def build_histogram_series(
    rows: list[dict[str, object]],
    *,
    active_dof: Iterable[int],
    bins: int = 50,
    schur_rows: list[dict[str, object]] | None = None,
) -> list[HistogramSeries]:
    """Prepare per-layer values with shared bins across all supplied geometries."""
    if bins <= 0:
        raise ValueError("bins must be a positive integer")

    active_indices = tuple(active_dof)
    if any(not isinstance(index, int) or index not in range(len(DOF_NAMES)) for index in active_indices):
        raise ValueError("active_dof indices must be integers from 0 through 5")

    schur_by_key = None
    if schur_rows is not None:
        def row_key(row):
            return (row["volume_id"], row["layer_id"], row["sensor_id"], row["dof"])

        primary_by_key = {row_key(row): row for row in rows}
        schur_by_key = {row_key(row): row for row in schur_rows}
        if len(primary_by_key) != len(rows) or len(schur_by_key) != len(schur_rows):
            raise ValueError("Comparison rows contain duplicate surface/DOF IDs")
        if primary_by_key.keys() != schur_by_key.keys():
            raise ValueError("ACTS and Schur comparison surface/DOF ID sets differ")
        for key, row in primary_by_key.items():
            other = schur_by_key[key]
            if (row["unit"] != other["unit"]
                    or row["misaligned_difference"] != other["misaligned_difference"]):
                raise ValueError("ACTS and Schur comparisons have different baselines")

    layer_keys = sorted(
        {(int(row["volume_id"]), int(row["layer_id"])) for row in rows}
    )
    series: list[HistogramSeries] = []
    for volume_id, layer_id in layer_keys:
        for dof_index in active_indices:
            dof = DOF_NAMES[dof_index]
            dof_rows = [
                row
                for row in rows
                if int(row["volume_id"]) == volume_id
                and int(row["layer_id"]) == layer_id
                and row["dof"] == dof
            ]
            if not dof_rows:
                continue
            misaligned = np.asarray(
                [row["misaligned_difference"] for row in dof_rows],
                dtype=np.float64,
            )
            aligned = np.asarray(
                [row["aligned_difference"] for row in dof_rows],
                dtype=np.float64,
            )
            schur = (
                None if schur_by_key is None else np.asarray(
                    [schur_by_key[row_key(row)]["aligned_difference"] for row in dof_rows],
                    dtype=np.float64,
                )
            )
            combined = np.concatenate(
                (misaligned, aligned) if schur is None else (misaligned, aligned, schur)
            )
            if not np.all(np.isfinite(combined)):
                raise ValueError(
                    f"Histogram values for vol={volume_id}|lay={layer_id}|{dof} "
                    "contain non-finite values"
                )
            series.append(
                HistogramSeries(
                    volume_id=volume_id,
                    layer_id=layer_id,
                    dof=dof,
                    unit=str(dof_rows[0]["unit"]),
                    misaligned=misaligned,
                    aligned=aligned,
                    bin_edges=np.histogram_bin_edges(combined, bins=bins),
                    misaligned_rms=float(np.sqrt(np.mean(np.square(misaligned)))),
                    aligned_rms=float(np.sqrt(np.mean(np.square(aligned)))),
                    schur=schur,
                    schur_rms=(
                        None if schur is None else float(np.sqrt(np.mean(np.square(schur))))
                    ),
                )
            )
    return series


def write_difference_histograms(
    rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    active_dof: Iterable[int],
    schur_rows: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    """Write one nominal-frame difference histogram PNG for each layer."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series = build_histogram_series(rows, active_dof=active_dof, schur_rows=schur_rows)
    layer_keys = sorted({(item.volume_id, item.layer_id) for item in series})
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for volume_id, layer_id in layer_keys:
        layer_series = [
            item
            for item in series
            if item.volume_id == volume_id and item.layer_id == layer_id
        ]
        column_count = min(3, len(layer_series))
        panel_rows = math.ceil(len(layer_series) / column_count)
        separate_detail = schur_rows is not None
        row_count = 2 * panel_rows if separate_detail else panel_rows
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(5.2 * column_count,
                     3.8 * row_count + 1.6 if separate_detail else 4.0 * row_count),
            squeeze=False,
        )
        try:
            main_axes = list(axes[:panel_rows].flat)
            detail_axes = list(axes[panel_rows:].flat)
            for index, (axis, item) in enumerate(zip(main_axes, layer_series, strict=False)):
                aligned_color = "tab:green" if item.schur is None else "tab:blue"
                aligned_label = "Nominal vs Aligned" if item.schur is None else "Nominal vs ACTS Aligned"
                axis.hist(
                    item.misaligned,
                    bins=item.bin_edges,
                    histtype="step",
                    linewidth=1.8,
                    color="tab:red",
                    label=(
                        "Nominal vs Misaligned\n"
                        f"RMS={item.misaligned_rms:.4g} {item.unit}"
                    ),
                )
                axis.hist(
                    item.aligned,
                    bins=item.bin_edges,
                    histtype="step",
                    linewidth=1.8,
                    color=aligned_color,
                    label=(
                        f"{aligned_label}\n"
                        f"RMS={item.aligned_rms:.4g} {item.unit}"
                    ),
                )
                if item.schur is not None:
                    axis.hist(
                        item.schur,
                        bins=item.bin_edges,
                        histtype="step",
                        linewidth=1.8,
                        color="tab:brown",
                        linestyle="--",
                        label=f"Nominal vs Schur-LM\nRMS={item.schur_rms:.4g} {item.unit}",
                    )
                axis.axvline(
                    0.0,
                    color="0.45",
                    linestyle="--",
                    linewidth=1.0,
                    label="_nolegend_",
                )
                axis.set_title(f"{item.dof}: full range" if separate_detail else item.dof)
                axis.set_xlabel(f"Nominal-frame {item.dof} difference [{item.unit}]")
                axis.set_ylabel("Surface count")
                axis.grid(alpha=0.2)
                if not separate_detail:
                    axis.legend(loc="upper right")

                aligned_bin_edges = np.histogram_bin_edges(
                    item.aligned if item.schur is None else np.concatenate((item.aligned, item.schur)),
                    bins=50,
                )
                detail_axis = (
                    detail_axes[index] if separate_detail
                    else axis.inset_axes([0.07, 0.5, 0.42, 0.42])
                )
                detail_axis.hist(
                    item.aligned,
                    bins=aligned_bin_edges,
                    histtype="step",
                    linewidth=1.5,
                    color=aligned_color,
                )
                if item.schur is not None:
                    detail_axis.hist(
                        item.schur,
                        bins=aligned_bin_edges,
                        histtype="step",
                        linewidth=1.5,
                        color="tab:brown",
                        linestyle="--",
                    )
                detail_axis.axvline(
                    0.0,
                    color="0.45",
                    linestyle="--",
                    linewidth=0.8,
                )
                detail_axis.set_xlim(aligned_bin_edges[0], aligned_bin_edges[-1])
                if separate_detail:
                    detail_axis.set_title(f"{item.dof}: ACTS / Schur-LM detail")
                    detail_axis.set_xlabel(f"Nominal-frame {item.dof} difference [{item.unit}]")
                    detail_axis.set_ylabel("Surface count")
                else:
                    detail_axis.set_title("Aligned detail", fontsize=8)
                    detail_axis.tick_params(axis="both", labelsize=7)
                detail_axis.grid(alpha=0.2)
                if not separate_detail:
                    axis.indicate_inset_zoom(detail_axis, edgecolor="0.4", alpha=0.6)
            for axis in main_axes[len(layer_series):] + detail_axes[len(layer_series):]:
                axis.set_visible(False)

            figure.suptitle(
                f"Geometry differences: Vol{volume_id} Lay{layer_id}"
            )
            if separate_detail:
                from matplotlib.lines import Line2D

                figure.legend(
                    handles=[
                        Line2D([], [], color="tab:red", linewidth=1.8,
                               label="Nominal vs Misaligned"),
                        Line2D([], [], color="tab:blue", linewidth=1.8,
                               label="Nominal vs ACTS Aligned"),
                        Line2D([], [], color="tab:brown", linewidth=1.8, linestyle="--",
                               label="Nominal vs Schur-LM"),
                    ],
                    loc="upper center", bbox_to_anchor=(0.5, 0.945),
                    ncol=3, frameon=False,
                )
                figure.tight_layout(rect=(0.0, 0.23, 1.0, 0.9), h_pad=2.5, w_pad=2.0)
                table_axis = figure.add_axes([0.18, 0.025, 0.76, 0.14])
                table_axis.set_axis_off()
                table_axis.set_title("RMS over all surfaces", fontsize=11, pad=7)
                table = table_axis.table(
                    cellText=[
                        [f"{item.misaligned_rms:.6g}" for item in layer_series],
                        [f"{item.aligned_rms:.6g}" for item in layer_series],
                        [f"{item.schur_rms:.6g}" for item in layer_series],
                    ],
                    rowLabels=["Misaligned", "ACTS Aligned", "Schur-LM"],
                    colLabels=[f"{item.dof} [{item.unit}]" for item in layer_series],
                    cellLoc="center", rowLoc="left", bbox=[0.0, 0.0, 1.0, 1.0],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                for (row, _column), cell in table.get_celld().items():
                    cell.set_edgecolor("0.8")
                    cell.set_linewidth(0.6)
                    if row == 0:
                        cell.set_facecolor("0.95")
            else:
                figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            image_path = (
                plot_dir
                / f"geometry_difference_hist_vol{volume_id}_lay{layer_id}.png"
            )
            figure.savefig(image_path, dpi=160, bbox_inches="tight")
        finally:
            plt.close(figure)

        outputs[f"histogram_vol{volume_id}_lay{layer_id}"] = image_path
    return outputs


def write_rms_bar_chart(
    rows: list[dict[str, object]],
    schur_rows: list[dict[str, object]],
    output_dir: str | Path,
    *,
    active_dof: Iterable[int],
) -> dict[str, Path]:
    """Plot paired all-surface RMS bars in um/mrad and export their values."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series = build_histogram_series(rows, active_dof=active_dof, schur_rows=schur_rows)
    dofs = list(dict.fromkeys(item.dof for item in series))
    if not dofs:
        raise ValueError("RMS bar chart requires at least one active DOF with data")
    columns = min(3, len(dofs))
    row_count = math.ceil(len(dofs) / columns)
    destination = Path(output_dir)
    plot_dir = destination / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    image_path = plot_dir / "geometry_rms_bars.png"
    csv_path = destination / "geometry_rms_bar_values.csv"
    records = []
    figure, axes = plt.subplots(
        row_count, columns, figsize=(5.4 * columns, 5.3 * row_count), squeeze=False,
    )
    try:
        for axis, dof in zip(axes.flat, dofs, strict=False):
            items = [item for item in series if item.dof == dof]
            positions = np.arange(len(items), dtype=np.float64)
            unit = "um" if items[0].unit == "mm" else "mrad"
            colors = ("tab:brown", "tab:blue")
            methods = ("Schur-LM", "ACTS Alignment")
            for method, color, offset in zip(methods, colors, (-0.2, 0.2), strict=True):
                values = [
                    1000.0 * (item.schur_rms if method == "Schur-LM" else item.aligned_rms)
                    for item in items
                ]
                bars = axis.bar(positions + offset, values, width=0.36,
                                color=color, label=method, zorder=3)
                axis.bar_label(bars, labels=[f"{value:.6g}" for value in values],
                               padding=4, fontsize=10)
            for item in items:
                for method, rms in (("Schur-LM", item.schur_rms),
                                    ("ACTS Alignment", item.aligned_rms)):
                    records.append({
                        "volume_id": item.volume_id, "layer_id": item.layer_id,
                        "dof": dof, "method": method, "unit": unit,
                        "surface_count": len(item.aligned), "rms": 1000.0 * rms,
                    })
            maximum = max(max(item.aligned_rms, item.schur_rms) for item in items) * 1000.0
            axis.set_ylim(0.0, 1.22 * maximum if maximum > 0.0 else 1.0)
            axis.set_xticks(positions, [f"{item.volume_id}:{item.layer_id}" for item in items])
            axis.set_title(dof, fontsize=13)
            axis.set_ylabel(f"Geometry error RMS [{'μm' if unit == 'um' else unit}]")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
        for axis in list(axes.flat)[len(dofs):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
                      ncol=2, frameon=False)
        figure.suptitle("Geometry error RMS | nominal local frame | lower is better", fontsize=14)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.82), w_pad=2.5)
        figure.savefig(image_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=(
        "volume_id", "layer_id", "dof", "method", "unit", "surface_count", "rms",
    ))
    writer.writeheader()
    writer.writerows(records)
    _atomic_write_text(csv_path, buffer.getvalue())
    return {"rms_bar_chart": image_path, "rms_bar_values_csv": csv_path}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare geometry JSON files in a shared nominal-local 6-DOF frame."
        )
    )
    add_config_argument(parser)
    parser.add_argument("--nominal", required=True, type=absolute_project_path)
    parser.add_argument("--misaligned", required=True, type=absolute_project_path)
    parser.add_argument("--aligned", required=True, type=absolute_project_path)
    parser.add_argument(
        "--schur-aligned", type=absolute_project_path,
        help="Optional Schur-LM geometry; --aligned then denotes ACTS Alignment.",
    )
    parser.add_argument(
        "--rms-bar-chart", action="store_true",
        help="Also plot paired geometry RMS bars in um/mrad; requires --schur-aligned.",
    )
    parser.add_argument("--output-dir", required=True, type=absolute_project_path)
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=DEFAULT_ZERO_TOLERANCE,
        help="Absolute initial-difference threshold below which recovery is undefined.",
    )
    return parser


def _manifest_metrics(summary: dict[str, object]) -> dict[str, object]:
    overall = summary["overall"]
    return {
        "surface_count": summary["surface_count"],
        "comparison_row_count": summary["comparison_row_count"],
        "mean_recovery_percent": {
            dof: metrics["recovery_percent"]["mean"]
            for dof, metrics in overall.items()
        },
        "undefined_recovery_count": {
            dof: metrics["recovery_percent"]["undefined_count"]
            for dof, metrics in overall.items()
        },
    }


def _failed_input_record(path: str | Path) -> dict[str, object]:
    """Record as much provenance as possible when normal manifest setup fails."""
    input_path = Path(path).resolve()
    record: dict[str, object] = {"path": str(input_path)}
    if not input_path.exists():
        record["error"] = "file not found"
        return record
    if not input_path.is_file():
        record["error"] = "not a regular file"
        return record
    try:
        digest = hashlib.sha256()
        with input_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        record["bytes"] = input_path.stat().st_size
        record["sha256"] = digest.hexdigest()
    except OSError as error:
        record["error"] = f"unreadable: {error}"
    return record


def _current_git_commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_preflight_failure_manifest(
    output_dir: Path,
    *,
    command: list[str],
    inputs: dict[str, str],
    config,
    requested_config: str | None,
    error: Exception,
) -> Path:
    """Write a failed manifest even when config loading or input hashing fails."""
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    configuration = (
        config.as_dict()
        if config is not None
        else {
            "repository_root": str(repository_root),
            "requested_config": requested_config,
            "load_error": str(error),
        }
    )
    manifest = {
        "schema_version": 1,
        "run_id": output_dir.name or "geometry-comparison",
        "created_at": timestamp,
        "completed_at": timestamp,
        "status": "failed",
        "git_commit": _current_git_commit(repository_root),
        "command": command,
        "configuration": configuration,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "inputs": {name: _failed_input_record(path) for name, path in inputs.items()},
        "outputs": {},
        "metrics": {"error": str(error), "failure_phase": "preflight"},
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_argument_parser().parse_args(raw_arguments)
    output_dir = Path(args.output_dir)
    inputs = {
        "nominal": args.nominal,
        "misaligned": args.misaligned,
        "aligned": args.aligned,
    }
    if args.schur_aligned is not None:
        inputs["schur_aligned"] = args.schur_aligned
    command = [sys.executable, str(Path(__file__).resolve()), *raw_arguments]
    config = None
    try:
        config = load_project_config(args.config)
        manifest_path = write_experiment_manifest(
            output_dir,
            run_id=output_dir.name or "geometry-comparison",
            command=command,
            config=config,
            inputs=inputs,
            status="running",
        )
    except Exception as error:
        try:
            _write_preflight_failure_manifest(
                output_dir,
                command=command,
                inputs=inputs,
                config=config,
                requested_config=args.config,
                error=error,
            )
        except Exception as manifest_error:
            print(
                f"Could not write failed manifest: {manifest_error}", file=sys.stderr
            )
        print(f"Geometry comparison failed before manifest creation: {error}", file=sys.stderr)
        return 1

    try:
        if args.rms_bar_chart and args.schur_aligned is None:
            raise ValueError("--rms-bar-chart requires --schur-aligned")
        rows = compare_geometry_files(
            args.nominal,
            args.misaligned,
            args.aligned,
            zero_tolerance=args.zero_tolerance,
        )
        schur_rows = None
        extra_outputs = {}
        extra_metrics = {}
        if args.schur_aligned is not None:
            schur_rows = compare_geometry_files(
                args.nominal, args.misaligned, args.schur_aligned,
                zero_tolerance=args.zero_tolerance,
            )
            schur_csv, schur_summary_path, schur_summary = write_reports(
                schur_rows, output_dir / "schur"
            )
            extra_outputs = {
                "schur_comparison_csv": schur_csv,
                "schur_summary_json": schur_summary_path,
            }
            extra_metrics = {
                "schur": _manifest_metrics(schur_summary),
                "source_files": {
                    name: _failed_input_record(Path(__file__).resolve().with_name(filename))
                    for name, filename in (
                        ("compare_geometries", "compare_geometries.py"),
                        ("project_config", "project_config.py"),
                        ("experiment_manifest", "experiment_manifest.py"),
                    )
                },
            }
        csv_path, summary_path, summary = write_reports(rows, output_dir)
        histogram_paths = write_difference_histograms(
            rows,
            output_dir,
            active_dof=config.active_dof,
            schur_rows=schur_rows,
        )
        if args.rms_bar_chart:
            extra_outputs.update(write_rms_bar_chart(
                rows, schur_rows, output_dir, active_dof=config.active_dof,
            ))
        finalize_experiment_manifest(
            manifest_path,
            status="completed",
            outputs={
                "comparison_csv": csv_path,
                "summary_json": summary_path,
                **histogram_paths,
                **extra_outputs,
            },
            metrics={**_manifest_metrics(summary), **extra_metrics},
        )
    except Exception as error:
        finalize_experiment_manifest(
            manifest_path,
            status="failed",
            metrics={"error": str(error)},
        )
        print(f"Geometry comparison failed: {error}", file=sys.stderr)
        return 1

    print(f"Compared {summary['surface_count']} surfaces ({len(rows)} DOF rows).")
    print(f"Detailed CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
