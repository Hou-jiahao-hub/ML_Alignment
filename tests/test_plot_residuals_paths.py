from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_plot_residuals_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_residuals.py"
    spec = importlib.util.spec_from_file_location("plot_residuals", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_residual_plot_cli_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    module = load_plot_residuals_module()
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    args = module.build_argument_parser().parse_args(
        ["--recon-dir", "inputs/recon", "--output-dir", "outputs/residuals"]
    )

    assert args.recon_dir == str(repository_root / "inputs" / "recon")
    assert args.output_dir == str(repository_root / "outputs" / "residuals")


def test_vol_lay_directory_name_accepts_compact_and_underscored_suffixes():
    plot_residuals = load_plot_residuals_module()

    assert (
        plot_residuals.vol_lay_directory_name("trackstates_kf_v17l8_v24l4.root")
        == "Vol17_Lay8__Vol24_Lay4"
    )
    assert (
        plot_residuals.vol_lay_directory_name("trackstates_kf_v24_l4_ml.root")
        == "Vol24_Lay4"
    )


def test_comparison_files_use_target_layer_subdirectories(tmp_path):
    plot_residuals = load_plot_residuals_module()

    files = plot_residuals.comparison_files(
        tmp_path,
        "trackstates_kf_v17l8_v24l4.root",
        aligned_trackstates="trackstates_kf_autograd_ml_v17l8_v24l4_iter0.root",
    )

    assert files == {
        "Nominal": tmp_path
        / "nominal"
        / "Vol17_Lay8__Vol24_Lay4"
        / "trackstates_kf_v17l8_v24l4.root",
        "Aligned": tmp_path
        / "aligned"
        / "Vol17_Lay8__Vol24_Lay4"
        / "trackstates_kf_autograd_ml_v17l8_v24l4_iter0.root",
        "Misaligned": tmp_path
        / "misaligned"
        / "Vol17_Lay8__Vol24_Lay4"
        / "trackstates_kf_v17l8_v24l4.root",
    }


def test_write_residual_png_creates_valid_image_with_three_geometries(tmp_path):
    plot_residuals = load_plot_residuals_module()
    output_png = tmp_path / "trial-004_res_X_Vol17_Lay8_comparison.png"

    written = plot_residuals.write_residual_png(
        {
            "Nominal": np.array([-0.02, 0.00, 0.01]),
            "Aligned": np.array([-0.01, 0.00, 0.01]),
            "Misaligned": np.array([0.03, 0.04, 0.05]),
        },
        output_png,
        var_name="res_X",
        layer_tag="Vol17_Lay8",
        plot_range=(-0.1, 0.1),
        bins=20,
        odd_resolution=0.015,
        output_label="trial-004",
    )

    assert written is True
    assert output_png.read_bytes().startswith(PNG_SIGNATURE)


def test_write_residual_png_skips_empty_and_out_of_range_data(tmp_path):
    plot_residuals = load_plot_residuals_module()
    empty_png = tmp_path / "empty.png"
    written = plot_residuals.write_residual_png(
        {"Nominal": np.array([]), "Aligned": None, "Misaligned": np.array([1.0])},
        empty_png,
        var_name="res_X",
        layer_tag="Vol17_Lay8",
        plot_range=(-0.1, 0.1),
        bins=20,
    )

    assert written is False
    assert not empty_png.exists()


def test_residual_png_legend_uses_two_lines_and_clears_central_peak(
    monkeypatch, tmp_path
):
    plot_residuals = load_plot_residuals_module()
    captured_figures = []
    real_close = plot_residuals.plt.close
    monkeypatch.setattr(
        plot_residuals.plt, "close", lambda figure: captured_figures.append(figure)
    )

    plot_residuals.write_residual_png(
        {
            "Nominal": np.array([-1.0, 1.0]),
            "Aligned": np.array([0.0, 0.0]),
            "Misaligned": np.array([-2.0, 2.0]),
        },
        tmp_path / "legend.png",
        var_name="res_X",
        layer_tag="Vol17_Lay8",
        plot_range=(-3.0, 3.0),
        bins=20,
    )

    figure = captured_figures[0]
    try:
        figure.canvas.draw()
        axis = figure.axes[0]
        legend = axis.get_legend()
        assert [text.get_text() for text in legend.get_texts()] == [
            "Nominal\nMean: 0.0000 mm, Std: 1.0000 mm",
            "Aligned\nMean: 0.0000 mm, Std: 0.0000 mm",
            "Misaligned\nMean: 0.0000 mm, Std: 2.0000 mm",
        ]

        legend_bounds = legend.get_window_extent()
        axis_bounds = axis.get_window_extent()
        central_peak_x = axis.transData.transform((0.0, 0.0))[0]
        assert legend_bounds.x0 > central_peak_x
        assert legend_bounds.x1 <= axis_bounds.x1
        assert legend_bounds.y1 <= axis_bounds.y1
    finally:
        real_close(figure)


def test_histograms_writes_png_when_pyroot_is_available(tmp_path):
    plot_residuals = load_plot_residuals_module()
    if plot_residuals.ROOT is None:
        pytest.skip("PyROOT is unavailable")

    plot_residuals.histograms(
        {"Nominal": np.array([-0.02, 0.00]), "Aligned": np.array([-0.01, 0.01])},
        tmp_path,
        "res_X",
        layer_tag="Vol17_Lay8",
        plot_range=(-0.1, 0.1),
        bins=20,
        output_label="trial-004",
    )

    output_stem = tmp_path / "trial-004_res_X_Vol17_Lay8_comparison"
    assert output_stem.with_suffix(".root").exists()
    assert output_stem.with_suffix(".png").read_bytes().startswith(PNG_SIGNATURE)


def test_histograms_writes_png_without_pyroot(monkeypatch, tmp_path):
    plot_residuals = load_plot_residuals_module()
    monkeypatch.setattr(plot_residuals, "ROOT", None)

    plot_residuals.histograms(
        {"Nominal": np.array([-0.02, 0.00]), "Misaligned": np.array([0.03, 0.04])},
        tmp_path,
        "res_X",
        layer_tag="Vol17_Lay8",
        plot_range=(-0.1, 0.1),
        bins=20,
        output_label="trial-004",
    )

    output_stem = tmp_path / "trial-004_res_X_Vol17_Lay8_comparison"
    assert output_stem.with_suffix(".root").exists()
    assert output_stem.with_suffix(".png").read_bytes().startswith(PNG_SIGNATURE)
