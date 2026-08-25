from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "compare_geometries.py"
    assert module_path.exists(), "compare_geometries.py has not been implemented"
    spec = importlib.util.spec_from_file_location("compare_geometries", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _surface(surface_id, translation=(0.0, 0.0, 0.0), rotation=None):
    return {
        "ID": surface_id,
        "Translation": list(translation),
        "RotationMatrix": np.eye(3).tolist() if rotation is None else rotation,
    }


def _write_geometry(path, surfaces):
    path.write_text(json.dumps(surfaces), encoding="utf-8")
    return path


def _axis_rotation(axis, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    if axis == "x":
        return [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    if axis == "y":
        return [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    return [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]


def test_comparison_reports_nominal_frame_differences_and_error_reduction(tmp_path):
    m = load_module()
    nominal_rotation = _axis_rotation("z", math.pi / 2.0)
    misaligned_rotation = (
        np.asarray(nominal_rotation) @ np.asarray(_axis_rotation("z", 0.2))
    ).tolist()
    aligned_rotation = (
        np.asarray(nominal_rotation) @ np.asarray(_axis_rotation("z", 0.05))
    ).tolist()
    nominal = _write_geometry(
        tmp_path / "nominal.json",
        [
            _surface(
                "vol=17|lay=8|sen=1|ext=1",
                (10.0, 20.0, 30.0),
                nominal_rotation,
            )
        ],
    )
    misaligned = _write_geometry(
        tmp_path / "misaligned.json",
        [_surface("vol=17|lay=8|sen=1", (14.0, 22.0, 31.0), misaligned_rotation)],
    )
    aligned = _write_geometry(
        tmp_path / "aligned.json",
        [_surface("vol=17|lay=8|sen=1", (11.0, 20.5, 31.5), aligned_rotation)],
    )

    rows = m.compare_geometry_files(nominal, misaligned, aligned)
    by_dof = {row["dof"]: row for row in rows}

    assert len(rows) == 6
    assert by_dof["tx"]["misaligned_difference"] == pytest.approx(2.0)
    assert by_dof["tx"]["aligned_difference"] == pytest.approx(0.5)
    assert by_dof["tx"]["recovery_percent"] == pytest.approx(75.0)
    assert by_dof["ty"]["misaligned_difference"] == pytest.approx(-4.0)
    assert by_dof["ty"]["aligned_difference"] == pytest.approx(-1.0)
    assert by_dof["ty"]["recovery_percent"] == pytest.approx(75.0)
    assert by_dof["tz"]["recovery_percent"] == pytest.approx(-50.0)
    assert by_dof["rx"]["recovery_percent"] is None
    assert by_dof["rz"]["misaligned_difference"] == pytest.approx(0.2)
    assert by_dof["rz"]["aligned_difference"] == pytest.approx(0.05)
    assert by_dof["rz"]["recovery_percent"] == pytest.approx(75.0)


@pytest.mark.parametrize(
    ("axis", "dof", "initial_angle", "remaining_angle"),
    [("x", "rx", 0.2, 0.05), ("y", "ry", -0.1, -0.025), ("z", "rz", 0.3, 0.075)],
)
def test_rotation_differences_follow_local_rz_ry_rx_convention(
    tmp_path, axis, dof, initial_angle, remaining_angle
):
    m = load_module()
    surface_id = "vol=17|lay=8|sen=1"
    nominal = _write_geometry(tmp_path / "nominal.json", [_surface(surface_id)])
    misaligned = _write_geometry(
        tmp_path / "misaligned.json",
        [_surface(surface_id, rotation=_axis_rotation(axis, initial_angle))],
    )
    aligned = _write_geometry(
        tmp_path / "aligned.json",
        [_surface(surface_id, rotation=_axis_rotation(axis, remaining_angle))],
    )

    rows = m.compare_geometry_files(nominal, misaligned, aligned)
    row = next(item for item in rows if item["dof"] == dof)

    assert row["misaligned_difference"] == pytest.approx(initial_angle)
    assert row["aligned_difference"] == pytest.approx(remaining_angle)
    assert row["recovery_percent"] == pytest.approx(75.0)
    assert row["unit"] == "rad"


def test_gimbal_locked_relative_rotation_is_rejected(tmp_path):
    m = load_module()
    surface_id = "vol=17|lay=8|sen=1"
    nominal = _write_geometry(tmp_path / "nominal.json", [_surface(surface_id)])
    misaligned = _write_geometry(
        tmp_path / "misaligned.json",
        [_surface(surface_id, rotation=_axis_rotation("y", -math.pi / 2.0))],
    )
    aligned = _write_geometry(tmp_path / "aligned.json", [_surface(surface_id)])

    with pytest.raises(ValueError, match="gimbal lock"):
        m.compare_geometry_files(nominal, misaligned, aligned)


def test_summary_excludes_undefined_recovery_values():
    m = load_module()
    rows = [
        {
            "volume_id": 17,
            "layer_id": 8,
            "sensor_id": 1,
            "dof": "tx",
            "unit": "mm",
            "misaligned_difference": -2.0,
            "misaligned_abs_difference": 2.0,
            "aligned_difference": -0.5,
            "aligned_abs_difference": 0.5,
            "recovery_percent": 75.0,
        },
        {
            "volume_id": 17,
            "layer_id": 8,
            "sensor_id": 2,
            "dof": "tx",
            "unit": "mm",
            "misaligned_difference": 0.0,
            "misaligned_abs_difference": 0.0,
            "aligned_difference": 0.1,
            "aligned_abs_difference": 0.1,
            "recovery_percent": None,
        },
    ]

    summary = m.build_summary(rows)
    tx = summary["overall"]["tx"]

    assert summary["surface_count"] == 2
    assert tx["recovery_percent"]["valid_count"] == 1
    assert tx["recovery_percent"]["undefined_count"] == 1
    assert tx["recovery_percent"]["mean"] == pytest.approx(75.0)
    assert tx["aligned_difference"]["rms"] == pytest.approx(math.sqrt(0.13))
    assert summary["by_layer"]["vol=17|lay=8"]["surface_count"] == 2


def _plot_rows():
    rows = []
    values = {
        "tx": ("mm", [-2.0, 2.0], [-0.5, 0.5]),
        "ty": ("mm", [-4.0, 4.0], [-1.0, 1.0]),
        "rz": ("rad", [-0.23456, 0.23456], [-0.012345, 0.012345]),
    }
    for dof, (unit, misaligned, aligned) in values.items():
        for sensor_id, (initial, remaining) in enumerate(
            zip(misaligned, aligned, strict=True), start=1
        ):
            rows.append(
                {
                    "canonical_id": f"vol=17|lay=8|sen={sensor_id}",
                    "volume_id": 17,
                    "layer_id": 8,
                    "sensor_id": sensor_id,
                    "dof": dof,
                    "unit": unit,
                    "misaligned_difference": initial,
                    "misaligned_abs_difference": abs(initial),
                    "aligned_difference": remaining,
                    "aligned_abs_difference": abs(remaining),
                    "recovery_percent": 75.0,
                }
            )
    return rows


def test_histogram_series_use_active_dof_shared_bins_and_rms():
    m = load_module()

    series = m.build_histogram_series(_plot_rows(), active_dof=(0, 5))

    assert [(item.dof, item.unit) for item in series] == [("tx", "mm"), ("rz", "rad")]
    tx = series[0]
    assert len(tx.bin_edges) == 51
    assert tx.misaligned_rms == pytest.approx(2.0)
    assert tx.aligned_rms == pytest.approx(0.5)
    assert np.histogram(tx.misaligned, bins=tx.bin_edges)[0].sum() == 2
    assert np.histogram(tx.aligned, bins=tx.bin_edges)[0].sum() == 2


def test_histogram_writer_creates_one_png_per_layer(tmp_path):
    m = load_module()

    outputs = m.write_difference_histograms(
        _plot_rows(), tmp_path, active_dof=(0, 1, 5)
    )

    assert set(outputs) == {"histogram_vol17_lay8"}
    image_path = outputs["histogram_vol17_lay8"]
    assert image_path == tmp_path / "plots" / "geometry_difference_hist_vol17_lay8.png"
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image_path.stat().st_size > 1000


def test_histogram_legends_use_upper_right_and_two_line_labels(
    monkeypatch, tmp_path
):
    m = load_module()
    import matplotlib.figure
    import matplotlib.legend

    captured_legends = []

    def capture_figure(figure, *_args, **_kwargs):
        captured_legends.extend(
            axis.get_legend() for axis in figure.axes if axis.get_visible()
        )

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_figure)

    m.write_difference_histograms(_plot_rows(), tmp_path, active_dof=(0, 1, 5))

    assert len(captured_legends) == 3
    assert all(
        legend._loc == matplotlib.legend.Legend.codes["upper right"]
        for legend in captured_legends
    )
    assert [
        [text.get_text() for text in legend.get_texts()]
        for legend in captured_legends
    ] == [
        ["Nominal vs Misaligned\nRMS=2 mm", "Nominal vs Aligned\nRMS=0.5 mm"],
        ["Nominal vs Misaligned\nRMS=4 mm", "Nominal vs Aligned\nRMS=1 mm"],
        [
            "Nominal vs Misaligned\nRMS=0.2346 rad",
            "Nominal vs Aligned\nRMS=0.01235 rad",
        ],
    ]


def test_histogram_writer_closes_figure_when_save_fails(monkeypatch, tmp_path):
    m = load_module()
    import matplotlib.figure
    import matplotlib.pyplot as plt

    closed = []
    real_close = plt.close

    def fail_savefig(_self, *_args, **_kwargs):
        raise OSError("synthetic PNG write failure")

    def record_close(figure):
        closed.append(figure)
        real_close(figure)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_savefig)
    monkeypatch.setattr(plt, "close", record_close)

    with pytest.raises(OSError, match="synthetic PNG write failure"):
        m.write_difference_histograms(_plot_rows(), tmp_path, active_dof=(0, 1, 5))

    assert len(closed) == 1


@pytest.mark.parametrize(
    ("nominal_surfaces", "misaligned_surfaces", "aligned_surfaces", "message"),
    [
        (
            [_surface("vol=1|lay=2|sen=3"), _surface("vol=1|lay=2|sen=3|ext=1")],
            [_surface("vol=1|lay=2|sen=3")],
            [_surface("vol=1|lay=2|sen=3")],
            "duplicate",
        ),
        (
            [_surface("vol=1|lay=2|sen=3")],
            [_surface("vol=1|lay=2|sen=3")],
            [_surface("vol=1|lay=2|sen=4")],
            "ID sets differ",
        ),
        (
            [_surface("vol=1|lay=2|sen=3")],
            [_surface("vol=1|lay=2|sen=4")],
            [_surface("vol=1|lay=2|sen=4")],
            "absent from nominal",
        ),
        (
            [_surface("vol=1|lay=2|sen=3")],
            [_surface("vol=1|lay=2|sen=3", rotation=[[1.0, 0.0], [0.0, 1.0]])],
            [_surface("vol=1|lay=2|sen=3")],
            "3x3",
        ),
        (
            [_surface("vol=1|lay=2|sen=3junk")],
            [_surface("vol=1|lay=2|sen=3")],
            [_surface("vol=1|lay=2|sen=3")],
            "vol/lay/sen",
        ),
    ],
)
def test_invalid_geometry_is_rejected(
    tmp_path, nominal_surfaces, misaligned_surfaces, aligned_surfaces, message
):
    m = load_module()
    nominal = _write_geometry(tmp_path / "nominal.json", nominal_surfaces)
    misaligned = _write_geometry(tmp_path / "misaligned.json", misaligned_surfaces)
    aligned = _write_geometry(tmp_path / "aligned.json", aligned_surfaces)

    with pytest.raises(ValueError, match=message):
        m.compare_geometry_files(nominal, misaligned, aligned)


def test_cli_writes_csv_json_and_completed_manifest(tmp_path):
    m = load_module()
    surface_id = "vol=17|lay=8|sen=1"
    nominal = _write_geometry(tmp_path / "nominal.json", [_surface(surface_id)])
    misaligned = _write_geometry(
        tmp_path / "misaligned.json", [_surface(surface_id, (2.0, 0.0, 0.0))]
    )
    aligned = _write_geometry(
        tmp_path / "aligned.json", [_surface(surface_id, (0.5, 0.0, 0.0))]
    )
    output_dir = tmp_path / "report"

    exit_code = m.main(
        [
            "--nominal",
            str(nominal),
            "--misaligned",
            str(misaligned),
            "--aligned",
            str(aligned),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    with (output_dir / "geometry_comparison.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    summary = json.loads((output_dir / "geometry_comparison_summary.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert len(csv_rows) == 6
    assert next(row for row in csv_rows if row["dof"] == "tx")["recovery_percent"] == "75.0"
    assert summary["overall"]["tx"]["aligned_difference"]["rms"] == pytest.approx(0.5)
    assert summary["schema_version"] == 2
    assert summary["conventions"]["coordinate_frame"] == "nominal geometry local frame"
    assert summary["conventions"]["difference_direction"] == (
        "nominal -> misaligned/aligned"
    )
    assert manifest["status"] == "completed"
    assert set(manifest["inputs"]) == {"nominal", "misaligned", "aligned"}
    assert set(manifest["outputs"]) == {
        "comparison_csv",
        "summary_json",
        "histogram_vol17_lay8",
    }
    assert Path(manifest["outputs"]["histogram_vol17_lay8"]).read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    assert len(manifest["inputs"]["nominal"]["sha256"]) == 64


def test_cli_records_failed_manifest_for_malformed_geometry(tmp_path):
    m = load_module()
    surface_id = "vol=17|lay=8|sen=1"
    nominal = _write_geometry(tmp_path / "nominal.json", [_surface(surface_id)])
    misaligned = _write_geometry(
        tmp_path / "misaligned.json",
        [_surface(surface_id, rotation=[[1.0, 0.0], [0.0, 1.0]])],
    )
    aligned = _write_geometry(tmp_path / "aligned.json", [_surface(surface_id)])
    output_dir = tmp_path / "failed-report"

    exit_code = m.main(
        [
            "--nominal",
            str(nominal),
            "--misaligned",
            str(misaligned),
            "--aligned",
            str(aligned),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert exit_code == 1
    assert manifest["status"] == "failed"
    assert "3x3" in manifest["metrics"]["error"]


def test_cli_records_failed_manifest_when_an_input_is_missing(tmp_path):
    m = load_module()
    surface_id = "vol=17|lay=8|sen=1"
    nominal = tmp_path / "missing-nominal.json"
    misaligned = _write_geometry(tmp_path / "misaligned.json", [_surface(surface_id)])
    aligned = _write_geometry(tmp_path / "aligned.json", [_surface(surface_id)])
    output_dir = tmp_path / "missing-input-report"

    exit_code = m.main(
        [
            "--nominal",
            str(nominal),
            "--misaligned",
            str(misaligned),
            "--aligned",
            str(aligned),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert exit_code == 1
    assert manifest["status"] == "failed"
    assert manifest["inputs"]["nominal"]["path"] == str(nominal)
    assert manifest["inputs"]["nominal"]["error"] == "file not found"
    assert len(manifest["inputs"]["misaligned"]["sha256"]) == 64


def test_cli_records_failed_manifest_when_config_is_missing(tmp_path):
    m = load_module()
    surface_id = "vol=17|lay=8|sen=1"
    nominal = _write_geometry(tmp_path / "nominal.json", [_surface(surface_id)])
    misaligned = _write_geometry(tmp_path / "misaligned.json", [_surface(surface_id)])
    aligned = _write_geometry(tmp_path / "aligned.json", [_surface(surface_id)])
    missing_config = tmp_path / "missing-config.toml"
    output_dir = tmp_path / "missing-config-report"

    exit_code = m.main(
        [
            "--config",
            str(missing_config),
            "--nominal",
            str(nominal),
            "--misaligned",
            str(misaligned),
            "--aligned",
            str(aligned),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert exit_code == 1
    assert manifest["status"] == "failed"
    assert manifest["configuration"]["requested_config"] == str(missing_config)
    assert "configuration file not found" in manifest["configuration"]["load_error"]
    assert len(manifest["inputs"]["nominal"]["sha256"]) == 64
