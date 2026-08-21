from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_plot_residuals_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_residuals.py"
    spec = importlib.util.spec_from_file_location("plot_residuals", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
