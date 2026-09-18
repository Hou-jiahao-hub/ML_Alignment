"""Phase 1 diagnostics: component-mask awareness, true RMS, empty-component rules."""

from __future__ import annotations

import importlib.util
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch


def test_diagnostics_module_owns_residual_reporting():
    diagnostics = importlib.import_module("alignment_diagnostics")
    before = np.array([[3.0, 4.0]], dtype=np.float64)
    after = np.array([[0.0, 0.0]], dtype=np.float64)
    metrics = diagnostics.print_per_layer_residual_diagnostics(
        before,
        after,
        np.array([17]),
        np.array([8]),
        ((17, 8),),
        return_metrics=True,
    )

    assert metrics[(17, 8)]["rms_before_loc0"] == 3.0
    assert metrics[(17, 8)]["rms_before_loc1"] == 4.0


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("ml_alignment_autograd", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mean_table_rows(output):
    """Data rows of the mean table only; the RMS table shares the row format."""
    begin = output.index("=== Per-layer residual mean")
    end = output.index("=== Per-layer residual RMS")
    return [
        line for line in output[begin:end].splitlines()
        if re.match(r"\s*vol=\d+ lay=\d+", line)
    ]


def rms_table_rows(output):
    begin = output.index("=== Per-layer residual RMS")
    return [
        line for line in output[begin:].splitlines()
        if re.match(r"\s*vol=\d+ lay=\d+", line)
    ]


# --------------------------------------------------------------------------
# T2: RMS must be sqrt(mean(x^2)), not np.std
# --------------------------------------------------------------------------
def test_rms_table_reports_root_mean_square_not_standard_deviation(capsys):
    m = load_module()
    # mean 3.5, std 0.5, true RMS sqrt(12.5) = 3.53553...
    init = np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float64)
    final = np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float64)
    vols = np.array([16, 16])
    lays = np.array([6, 6])

    metrics = m.print_per_layer_residual_diagnostics(
        init, final, vols, lays, ((16, 6),), return_metrics=True
    )

    expected = float(np.sqrt(np.mean(np.array([3.0, 4.0]) ** 2)))
    assert expected == pytest.approx(3.5355339059327378)
    assert metrics[(16, 6)]["rms_after_loc0"] == pytest.approx(expected)
    assert metrics[(16, 6)]["rms_before_loc0"] == pytest.approx(expected)
    # the std value must NOT be what is reported
    assert metrics[(16, 6)]["rms_after_loc0"] != pytest.approx(0.5)

    row = rms_table_rows(capsys.readouterr().out)[0]
    assert "3.5355mm" in row


# --------------------------------------------------------------------------
# T3: printed values are the 4-decimal rounding of the returned floats
# --------------------------------------------------------------------------
def test_printed_table_matches_rounded_return_values(capsys):
    m = load_module()
    rng = np.random.default_rng(7)
    init = rng.normal(size=(9, 2))
    final = rng.normal(size=(9, 2))
    vols = np.full(9, 16)
    lays = np.full(9, 6)

    metrics = m.print_per_layer_residual_diagnostics(
        init, final, vols, lays, ((16, 6),), return_metrics=True
    )
    output = capsys.readouterr().out

    entry = metrics[(16, 6)]
    assert entry["mean_after_loc0"] == pytest.approx(float(np.mean(final[:, 0])))
    assert entry["rms_after_loc0"] == pytest.approx(
        float(np.sqrt(np.mean(final[:, 0] ** 2)))
    )

    printed = [float(v) for v in re.findall(r"([-\d.]+)mm", mean_table_rows(output)[0])]
    assert printed[0] == round(entry["mean_before_loc0"], 4)
    assert printed[1] == round(entry["mean_after_loc0"], 4)
    assert printed[2] == round(entry["mean_before_loc1"], 4)
    assert printed[3] == round(entry["mean_after_loc1"], 4)


# --------------------------------------------------------------------------
# T4: component mask keeps invalid LOC1 zeros out of the statistics
# --------------------------------------------------------------------------
def test_component_mask_excludes_invalid_loc1_from_statistics(capsys):
    m = load_module()
    # Third hit is a 1D strip: its LOC1 residual was forced to 0 upstream.
    init = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    final = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    vols = np.array([16, 16, 16])
    lays = np.array([6, 6, 6])
    component_mask = np.array([[True, True], [True, True], [True, False]])

    metrics = m.print_per_layer_residual_diagnostics(
        init, final, vols, lays, ((16, 6),),
        return_metrics=True, component_mask=component_mask,
    )
    output = capsys.readouterr().out

    entry = metrics[(16, 6)]
    assert entry["n_hits_loc0"] == 3
    assert entry["n_hits_loc1"] == 2
    assert entry["n_hits"] == entry["n_hits_loc0"]  # back-compat alias
    # Without the mask the zero would drag the LOC1 mean down to 4/3.
    assert entry["mean_after_loc1"] == pytest.approx(2.0)
    assert "(n0=3, n1=2)" in output


def test_component_mask_defaults_to_all_valid(capsys):
    m = load_module()
    init = np.zeros((4, 2))
    final = np.zeros((4, 2))
    vols = np.full(4, 16)
    lays = np.full(4, 6)

    metrics = m.print_per_layer_residual_diagnostics(
        init, final, vols, lays, ((16, 6),), return_metrics=True
    )

    assert metrics[(16, 6)]["n_hits_loc0"] == 4
    assert metrics[(16, 6)]["n_hits_loc1"] == 4
    capsys.readouterr()


def test_component_mask_accepts_torch_tensor_input():
    """surface mode passes a torch bool tensor; it must not be mixed with ndarray."""
    m = load_module()
    init = torch.zeros((3, 2), dtype=torch.float32)
    final = torch.ones((3, 2), dtype=torch.float32)

    metrics = m.print_per_layer_residual_diagnostics(
        init, final,
        torch.tensor([16, 16, 16]), torch.tensor([6, 6, 6]),
        ((16, 6),),
        return_metrics=True,
        component_mask=torch.tensor([[True, True], [True, False], [True, False]]),
    )

    assert metrics[(16, 6)]["n_hits_loc0"] == 3
    assert metrics[(16, 6)]["n_hits_loc1"] == 1


# --------------------------------------------------------------------------
# T5: input validation
# --------------------------------------------------------------------------
def test_mismatched_component_mask_shape_raises():
    m = load_module()
    with pytest.raises(ValueError, match="component_mask"):
        m.print_per_layer_residual_diagnostics(
            np.zeros((3, 2)), np.zeros((3, 2)),
            np.full(3, 16), np.full(3, 6), ((16, 6),),
            component_mask=np.ones((2, 2), dtype=bool),
        )


def test_mismatched_residual_shapes_raise():
    m = load_module()
    with pytest.raises(ValueError):
        m.print_per_layer_residual_diagnostics(
            np.zeros((3, 2)), np.zeros((4, 2)),
            np.full(3, 16), np.full(3, 6), ((16, 6),),
        )


def test_mismatched_layer_id_length_raises():
    m = load_module()
    with pytest.raises(ValueError):
        m.print_per_layer_residual_diagnostics(
            np.zeros((3, 2)), np.zeros((3, 2)),
            np.full(2, 16), np.full(2, 6), ((16, 6),),
        )


# --------------------------------------------------------------------------
# T6: empty components -> N/A + nan, never np.mean([])
# --------------------------------------------------------------------------
def test_empty_loc1_component_reports_na_without_warning(capsys):
    m = load_module()
    init = np.array([[1.0, 0.0], [2.0, 0.0]])
    final = np.array([[1.0, 0.0], [2.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = m.print_per_layer_residual_diagnostics(
            init, final, np.full(2, 16), np.full(2, 6), ((16, 6),),
            return_metrics=True,
            component_mask=np.array([[True, False], [True, False]]),
        )
    output = capsys.readouterr().out

    entry = metrics[(16, 6)]
    assert entry["n_hits_loc1"] == 0
    assert entry["n_hits_loc0"] == 2
    for key in ("mean_before_loc1", "mean_after_loc1", "rms_before_loc1", "rms_after_loc1"):
        assert key in entry                      # field structure stays complete
        assert np.isnan(entry[key])
    assert np.isfinite(entry["mean_after_loc0"])
    assert "N/A" in output


def test_empty_loc0_component_keeps_metric_keys(capsys):
    m = load_module()
    init = np.zeros((2, 2))
    final = np.zeros((2, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = m.print_per_layer_residual_diagnostics(
            init, final, np.full(2, 16), np.full(2, 6), ((16, 6),),
            return_metrics=True,
            component_mask=np.zeros((2, 2), dtype=bool),
        )
    output = capsys.readouterr().out

    entry = metrics[(16, 6)]
    assert entry["n_hits_loc0"] == 0
    assert entry["n_hits_loc1"] == 0
    for key in ("mean_before_loc0", "mean_after_loc0", "rms_before_loc0", "rms_after_loc0",
                "mean_before_loc1", "mean_after_loc1", "rms_before_loc1", "rms_after_loc1"):
        assert key in entry
        assert np.isnan(entry[key])
    assert "N/A" in output


# --------------------------------------------------------------------------
# T6 (second half): print_target_residual_stats has no metrics to return,
# so its empty-LOC1 behaviour needs its own test.
# --------------------------------------------------------------------------
def test_target_residual_stats_splits_counts_and_marks_empty_loc1(capsys):
    m = load_module()
    residuals = torch.tensor([[1.0, 2.0], [3.0, 0.0]], dtype=torch.float32)
    vols = torch.tensor([16, 16])
    lays = torch.tensor([6, 6])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m.print_target_residual_stats(
            residuals, vols, lays, ((16, 6),),
            loc1_valid=torch.tensor([True, False]),
        )
    output = capsys.readouterr().out

    assert "N0" in output and "N1" in output
    row = [l for l in output.splitlines() if "vol=16 lay=6" in l][0]
    # LOC1 mean must be 2.0 (the masked-out zero excluded), not 1.0
    assert "2.00000" in row


def test_target_residual_stats_reports_na_when_no_valid_loc1(capsys):
    m = load_module()
    residuals = torch.tensor([[1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m.print_target_residual_stats(
            residuals, torch.tensor([16, 16]), torch.tensor([6, 6]), ((16, 6),),
            loc1_valid=torch.tensor([False, False]),
        )
    output = capsys.readouterr().out

    row = [l for l in output.splitlines() if "vol=16 lay=6" in l][0]
    assert "N/A" in row
    assert "2.00000" in row  # LOC0 mean still reported


def test_target_residual_stats_loc1_valid_defaults_to_all_valid(capsys):
    m = load_module()
    m.print_target_residual_stats(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([16, 16]), torch.tensor([6, 6]), ((16, 6),),
    )
    row = [l for l in capsys.readouterr().out.splitlines() if "vol=16 lay=6" in l][0]
    assert "3.00000" in row  # LOC1 mean over both hits
