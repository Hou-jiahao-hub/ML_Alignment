from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def load_plot_chi2ndf_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_chi2ndf.py"
    spec = importlib.util.spec_from_file_location("plot_chi2ndf", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_chi2_plot_cli_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    module = load_plot_chi2ndf_module()
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    args = module.build_argument_parser().parse_args(
        ["--output", "outputs/chi2.png", "--recon-dir", "inputs/recon"]
    )

    assert args.output == str(repository_root / "outputs" / "chi2.png")
    assert args.recon_dir == str(repository_root / "inputs" / "recon")


def test_chi2_ndf_values_drop_non_positive_ndf(capsys):
    plot_chi2ndf = load_plot_chi2ndf_module()

    values = plot_chi2ndf.chi2_ndf_values(
        np.array([2.0, 9.0, 12.0, 8.0]),
        np.array([1.0, 0.0, -3.0, 4.0]),
        label="Nominal",
    )

    assert np.allclose(values, np.array([2.0, 2.0]))
    assert "Nominal: dropped 2/4 tracks with NDF <= 0" in capsys.readouterr().out


def test_default_comparison_files_match_residuals_layout(tmp_path):
    plot_chi2ndf = load_plot_chi2ndf_module()

    files = plot_chi2ndf.comparison_files(
        tmp_path,
        "tracksummary_kf_v17l8_v24l4.root",
        aligned_tracksummary=None,
    )

    assert files == {
        "Nominal": tmp_path / "nominal" / "Vol17_Lay8__Vol24_Lay4" / "tracksummary_kf_v17l8_v24l4.root",
        "Aligned": tmp_path / "aligned" / "Vol17_Lay8__Vol24_Lay4" / "tracksummary_kf_v17l8_v24l4.root",
        "Misaligned": tmp_path / "misaligned" / "Vol17_Lay8__Vol24_Lay4" / "tracksummary_kf_v17l8_v24l4.root",
    }


def test_aligned_tracksummary_override_uses_its_vol_lay_directory(tmp_path):
    plot_chi2ndf = load_plot_chi2ndf_module()

    files = plot_chi2ndf.comparison_files(
        tmp_path,
        "tracksummary_kf_v17l8_v24l4.root",
        aligned_tracksummary="tracksummary_kf_autograd_ml_v17l8_v24l4_iter14.root",
    )

    assert files["Aligned"] == (
        tmp_path
        / "aligned"
        / "Vol17_Lay8__Vol24_Lay4"
        / "tracksummary_kf_autograd_ml_v17l8_v24l4_iter14.root"
    )


def test_cli_help_exposes_only_comparison_mode_options():
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_chi2ndf.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--input" not in result.stdout
    assert "--xmax" not in result.stdout
    assert "--linear-y" not in result.stdout
    assert "--log-y" not in result.stdout
    assert "--recon-dir" in result.stdout
    assert "--tracksummary-name" in result.stdout
    assert "--display-xmax" in result.stdout


def test_comparison_plot_uses_full_range_by_default(monkeypatch, tmp_path):
    plot_chi2ndf = load_plot_chi2ndf_module()

    captured = {}

    class FakeAxis:
        transAxes = object()

        def hist(self, data, **kwargs):
            captured.setdefault("hist_data", []).append(np.asarray(data))
            captured.setdefault("hist_kwargs", []).append(kwargs)

        def set_xlabel(self, value, **_kwargs):
            captured["xlabel"] = value

        def set_ylabel(self, value, **_kwargs):
            captured["ylabel"] = value

        def set_title(self, value, **_kwargs):
            captured["title"] = value

        def set_yscale(self, value):
            captured["yscale"] = value

        def set_xscale(self, value):
            captured["xscale"] = value

        def set_xlim(self, *_args, **_kwargs):
            pass

        def set_ylim(self, *_args, **_kwargs):
            pass

        def margins(self, *_args, **_kwargs):
            pass

        def legend(self, **_kwargs):
            captured["legend"] = True

        def text(self, *_args, **_kwargs):
            captured["stats_text"] = _args[2]
            captured["stats"] = True

    class FakeFig:
        def savefig(self, path, **_kwargs):
            captured["output_path"] = Path(path)

    fake_axis = FakeAxis()
    fake_fig = FakeFig()
    monkeypatch.setattr(plot_chi2ndf.plt, "subplots", lambda **_kwargs: (fake_fig, fake_axis))
    monkeypatch.setattr(plot_chi2ndf.plt, "close", lambda _fig: None)

    data = {
        "Nominal": np.array([1.0, 2.0, 1000.0]),
        "Aligned": np.array([0.5, 1.5, 2500.0]),
    }

    plot_chi2ndf.plot_comparison(data, tmp_path / "chi2ndf.png", bins=20)

    assert captured["output_path"] == tmp_path / "chi2ndf.png"
    assert captured["yscale"] == "log"
    assert captured["xscale"] == "log"
    for values, hist_kwargs in zip(captured["hist_data"], captured["hist_kwargs"]):
        edges = np.asarray(hist_kwargs["bins"], dtype=float)
        assert len(edges) == 21
        # Exact, not approx: logspace endpoints drift by ~1e-13 and would push
        # the extreme values out of the histogram entirely.
        assert edges[0] == 0.5
        assert edges[-1] == 2500.0
        assert "range" not in hist_kwargs
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == len(values)
    assert captured["hist_kwargs"][0]["histtype"] == "step"
    assert captured["hist_kwargs"][0]["fill"] is False


def test_comparison_plot_uses_display_xmax_when_provided(monkeypatch, tmp_path, capsys):
    plot_chi2ndf = load_plot_chi2ndf_module()

    captured = {}

    class FakeAxis:
        transAxes = object()

        def hist(self, data, **kwargs):
            captured.setdefault("hist_data", []).append(np.asarray(data))
            captured.setdefault("hist_kwargs", []).append(kwargs)

        def set_xlabel(self, *_args, **_kwargs):
            pass

        def set_ylabel(self, *_args, **_kwargs):
            pass

        def set_title(self, *_args, **_kwargs):
            pass

        def legend(self, **_kwargs):
            pass

        def text(self, *_args, **_kwargs):
            captured["stats_text"] = _args[2]

        def set_ylim(self, *_args, **_kwargs):
            pass

        def set_xlim(self, *_args, **_kwargs):
            pass

        def set_xscale(self, *_args, **_kwargs):
            pass

        def set_yscale(self, *_args, **_kwargs):
            pass

        def margins(self, *_args, **_kwargs):
            pass

    class FakeFig:
        def savefig(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(plot_chi2ndf.plt, "subplots", lambda **_kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(plot_chi2ndf.plt, "close", lambda _fig: None)

    data = {
        "Nominal": np.array([1.0, 2.0, 1000.0]),
        "Aligned": np.array([0.5, 1.5, 2500.0]),
    }

    plot_chi2ndf.plot_comparison(
        data, tmp_path / "chi2ndf.png", bins=20, display_xmax=50.0
    )

    for values, hist_kwargs in zip(captured["hist_data"], captured["hist_kwargs"]):
        edges = np.asarray(hist_kwargs["bins"], dtype=float)
        assert edges[0] == 0.5
        assert edges[-1] == 50.0
        # Values above display_xmax are excluded on purpose; everything at or
        # below it must still be binned.
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == int(np.sum(values <= 50.0))

    out = capsys.readouterr().out
    assert "display-xmax=50" in out


def test_comparison_plot_reports_nonpositive_entries(monkeypatch, tmp_path, capsys):
    plot_chi2ndf = load_plot_chi2ndf_module()

    captured = {}

    class FakeAxis:
        transAxes = object()

        def hist(self, data, **kwargs):
            captured.setdefault("hist_data", []).append(np.asarray(data))
            captured.setdefault("hist_kwargs", []).append(kwargs)

        def set_xlabel(self, *_args, **_kwargs):
            pass

        def set_ylabel(self, *_args, **_kwargs):
            pass

        def set_title(self, *_args, **_kwargs):
            pass

        def legend(self, **_kwargs):
            pass

        def text(self, *_args, **_kwargs):
            captured["stats_text"] = _args[2]

        def set_ylim(self, *_args, **_kwargs):
            pass

        def set_xlim(self, *_args, **_kwargs):
            pass

        def set_xscale(self, *_args, **_kwargs):
            pass

        def set_yscale(self, *_args, **_kwargs):
            pass

        def margins(self, *_args, **_kwargs):
            pass

    class FakeFig:
        def savefig(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(plot_chi2ndf.plt, "subplots", lambda **_kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(plot_chi2ndf.plt, "close", lambda _fig: None)

    data = {
        "Nominal": np.array([0.0, 1.0, 2.0, 1000.0]),
        "Aligned": np.array([0.5, 1.5, 2500.0]),
    }

    plot_chi2ndf.plot_comparison(data, tmp_path / "chi2ndf.png", bins=20)

    assert "N(<=0)=     1" in captured["stats_text"]
    assert "chi2/NDF <= 0" in capsys.readouterr().out

    for values, hist_kwargs in zip(captured["hist_data"], captured["hist_kwargs"]):
        edges = np.asarray(hist_kwargs["bins"], dtype=float)
        assert edges[0] == 0.5
        assert edges[-1] == 2500.0
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == int(np.sum(values > 0))


@pytest.mark.parametrize(
    "x_min, x_max, bins",
    [
        (0.5, 2500.0, 20),
        (0.5, 2500.0, 100),
        (0.2, 54321.0, 100),
        (0.17424323161443075, 57804.205357142855, 100),
        (1e-3, 1e5, 37),
    ],
)
def test_log_bin_edges_are_exact_at_endpoints(monkeypatch, tmp_path, x_min, x_max, bins):
    """np.logspace endpoints drift by ~1e-13, which can push the extreme values
    out of the histogram. plot_comparison must pin them so nothing is lost."""
    plot_chi2ndf = load_plot_chi2ndf_module()

    captured = {}

    class FakeAxis:
        transAxes = object()

        def hist(self, data, **kwargs):
            captured.setdefault("hist_data", []).append(np.asarray(data))
            captured.setdefault("hist_kwargs", []).append(kwargs)

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    class FakeFig:
        def savefig(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(plot_chi2ndf.plt, "subplots", lambda **_kwargs: (FakeFig(), FakeAxis()))
    monkeypatch.setattr(plot_chi2ndf.plt, "close", lambda _fig: None)

    values = np.array([x_min, np.sqrt(x_min * x_max), x_max])
    plot_chi2ndf.plot_comparison({"Nominal": values}, tmp_path / "chi2ndf.png", bins=bins)

    edges = np.asarray(captured["hist_kwargs"][0]["bins"], dtype=float)
    assert len(edges) == bins + 1
    assert edges[0] == x_min
    assert edges[-1] == x_max
    counts, _ = np.histogram(values, bins=edges)
    assert counts.sum() == len(values)
