from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_plot_hit_number_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_hit_number.py"
    spec = importlib.util.spec_from_file_location("plot_hit_number", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hit_plot_cli_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    module = load_plot_hit_number_module()
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    args = module.build_argument_parser().parse_args(
        ["--output", "outputs/hits.png", "--recon-dir", "inputs/recon"]
    )

    assert args.output == str(repository_root / "outputs" / "hits.png")
    assert args.recon_dir == str(repository_root / "inputs" / "recon")


def test_load_hit_numbers_reads_nmeasurements(monkeypatch, tmp_path):
    plot_hit_number = load_plot_hit_number_module()

    input_path = tmp_path / "tracksummary.root"

    class FakeBranch:
        def array(self, library):
            assert library == "ak"
            return np.array([8, 10, 12], dtype=np.uint32)

    class FakeTree:
        def __contains__(self, key):
            return key == "nMeasurements"

        def __getitem__(self, key):
            assert key == "nMeasurements"
            return FakeBranch()

    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __contains__(self, key):
            return key == "tracksummary"

        def __getitem__(self, key):
            assert key == "tracksummary"
            return FakeTree()

    monkeypatch.setattr(plot_hit_number, "ROOT", None)
    monkeypatch.setattr(plot_hit_number.uproot, "open", lambda path: FakeFile())

    values = plot_hit_number.load_hit_numbers(input_path, label="Nominal")

    assert np.array_equal(values, np.array([8, 10, 12]))


def test_default_comparison_files_match_reconstruction_layout(tmp_path):
    plot_hit_number = load_plot_hit_number_module()

    files = plot_hit_number.comparison_files(
        tmp_path,
        "tracksummary_kf_v17l8_v24l4.root",
        aligned_tracksummary=None,
    )

    assert files == {
        "Nominal": tmp_path / "nominal" / "Vol17_Lay8__Vol24_Lay4" / "tracksummary_kf_v17l8_v24l4.root",
        "Misaligned": tmp_path / "misaligned" / "Vol17_Lay8__Vol24_Lay4" / "tracksummary_kf_v17l8_v24l4.root",
        "Aligned": tmp_path / "aligned" / "Vol17_Lay8__Vol24_Lay4" / "tracksummary_kf_v17l8_v24l4.root",
    }


def test_aligned_tracksummary_override_uses_its_vol_lay_directory(tmp_path):
    plot_hit_number = load_plot_hit_number_module()

    files = plot_hit_number.comparison_files(
        tmp_path,
        "tracksummary_kf_v17l8_v24l4.root",
        aligned_tracksummary="tracksummary_kf_autograd_ml_v17l8_v24l4_iter4.root",
    )

    assert files["Aligned"] == (
        tmp_path
        / "aligned"
        / "Vol17_Lay8__Vol24_Lay4"
        / "tracksummary_kf_autograd_ml_v17l8_v24l4_iter4.root"
    )


def test_plot_comparison_uses_step_histograms_and_reference_labels(monkeypatch, tmp_path):
    plot_hit_number = load_plot_hit_number_module()

    captured = {}

    class FakeAxis:
        def hist(self, data, **kwargs):
            captured.setdefault("hist_data", []).append(np.asarray(data))
            captured.setdefault("hist_kwargs", []).append(kwargs)

        def set_xlabel(self, value, **_kwargs):
            captured["xlabel"] = value

        def set_ylabel(self, value, **_kwargs):
            captured["ylabel"] = value

        def set_xlim(self, *_args):
            captured["xlim"] = _args

        def legend(self, **kwargs):
            captured["legend_kwargs"] = kwargs

        def tick_params(self, **kwargs):
            captured["tick_params"] = kwargs

        def minorticks_on(self):
            captured["minorticks_on"] = True

    class FakeFig:
        def savefig(self, path, **_kwargs):
            captured["output_path"] = Path(path)

    fake_axis = FakeAxis()
    fake_fig = FakeFig()
    monkeypatch.setattr(plot_hit_number.plt, "subplots", lambda **_kwargs: (fake_fig, fake_axis))
    monkeypatch.setattr(plot_hit_number.plt, "close", lambda _fig: None)

    data = {
        "Nominal": np.array([9, 10, 10]),
        "Misaligned": np.array([10, 11]),
        "Aligned": np.array([9, 10, 11]),
    }

    plot_hit_number.plot_comparison(data, tmp_path / "hit_number.png", bins=16, x_range=(6, 22))

    assert captured["output_path"] == tmp_path / "hit_number.png"
    assert captured["xlabel"] == r"$N_{\mathrm{hits}}$"
    assert captured["ylabel"] == "Entries"
    assert captured["xlim"] == (6, 22)
    assert captured["legend_kwargs"]["title"] == "ACTS OpenDataDetector"
    assert [kwargs["label"] for kwargs in captured["hist_kwargs"]] == ["Nominal", "Misaligned", "Aligned"]
    assert all(kwargs["histtype"] == "step" for kwargs in captured["hist_kwargs"])
    assert all(kwargs["fill"] is False for kwargs in captured["hist_kwargs"])


def test_cli_help_exposes_hit_number_options():
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_hit_number.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--tracksummary-name" in result.stdout
    assert "--aligned-tracksummary" in result.stdout
    assert "--range" in result.stdout
