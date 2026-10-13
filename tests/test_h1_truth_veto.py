"""Level F (Iteration 41 §B / Iteration 46): the paired Schur-vs-fixed-track H1 truth veto."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import torch


def _module():
    return importlib.import_module("scripts.analysis.run_h1_truth_veto")


# Paired ratio -------------------------------------------------------------------


def test_identical_arms_give_unit_ratio_and_a_bracketing_interval():
    module = _module()
    errors = np.random.default_rng(1).normal(size=400)
    result = module.paired_ratio(errors, errors.copy(), seed=6006, n_bootstrap=2000)
    assert result["nominal_ratio"] == pytest.approx(1.0)
    assert result["ci_lower"] == pytest.approx(1.0) and result["ci_upper"] == pytest.approx(1.0)


def test_a_uniformly_halved_error_has_ratio_one_half_everywhere():
    module = _module()
    errors = np.random.default_rng(2).normal(size=300)
    result = module.paired_ratio(0.5 * errors, errors, seed=6006, n_bootstrap=2000)
    for key in ("nominal_ratio", "ci_lower", "ci_upper"):
        assert result[key] == pytest.approx(0.5)


def test_ci_lower_is_the_lower_quantile_of_the_same_resamples():
    module = _module()
    rng = np.random.default_rng(3)
    baseline = rng.normal(size=200)
    candidate = baseline * rng.uniform(0.5, 1.5, size=200)
    result = module.paired_ratio(candidate, baseline, seed=6006, n_bootstrap=2000)
    draws = np.random.default_rng(6006).integers(0, 200, size=(2000, 200))
    ratio = np.sqrt((candidate[draws] ** 2).sum(1) / (baseline[draws] ** 2).sum(1))
    assert result["ci_upper"] == pytest.approx(float(np.quantile(ratio, 0.975, method="higher")), rel=1e-12)
    assert result["ci_lower"] == pytest.approx(float(np.quantile(ratio, 0.025, method="lower")), rel=1e-12)
    assert result["ci_lower"] < result["nominal_ratio"] < result["ci_upper"]


# Truth attachment ---------------------------------------------------------------


def _toy():
    frame = pd.DataFrame({
        "event_nr": [7, 7, 7, 9, 9],
        "track_nr": [1, 1, 2, 0, 0],
        "volume_id": [17, 24, 17, 17, 24],
        "layer_id": [8, 4, 2, 8, 4],
        "module_id": [3, 5, 1, 4, 6],
        "l_x_hit": [0.1, 0.2, 0.3, 0.4, 0.5],
        "t_x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "t_y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "t_z": [100.0, 200.0, 300.0, 400.0, 500.0],
    })
    keys = {0: (17, 8, 3), 1: (24, 4, 5), 2: (17, 2, 1), 3: (17, 8, 4), 4: (24, 4, 6)}
    # Rows: (9, 0) then (7, 1); hits in the reverse order of the frame.
    event_track_ids = torch.tensor([[9, 0], [7, 1]])
    sensor_idx = torch.tensor([[4, 3], [1, 0]])
    hit_mask = torch.tensor([[True, True], [True, True]])
    measured = torch.tensor([[[0.5, 0.0], [0.4, 0.0]], [[0.2, 0.0], [0.1, 0.0]]], dtype=torch.float64)
    return frame, keys, event_track_ids, sensor_idx, hit_mask, measured


def test_attach_truth_follows_the_hit_keys_not_the_row_order():
    module = _module()
    frame, keys, ids, sensors, mask, measured = _toy()
    truth = module.attach_truth(frame, keys, ids, sensors, mask, measured)
    assert truth[0, 0].tolist() == [5.0, 50.0, 500.0]
    assert truth[1, 1].tolist() == [1.0, 10.0, 100.0]


def test_attach_truth_fails_closed_on_a_measurement_mismatch():
    module = _module()
    frame, keys, ids, sensors, mask, measured = _toy()
    measured[1, 0, 0] = 0.25
    with pytest.raises(ValueError, match="l_x_hit"):
        module.attach_truth(frame, keys, ids, sensors, mask, measured)


def test_attach_truth_fails_closed_on_a_missing_hit():
    module = _module()
    frame, keys, ids, sensors, mask, measured = _toy()
    with pytest.raises(ValueError, match="missing"):
        module.attach_truth(frame.drop(index=0), keys, ids, sensors, mask, measured)


# Verdict --------------------------------------------------------------------------


def _metrics(upper=0.99, lower=0.95, ratio=0.97, cells=None):
    names = [f"{layer}|{dof}" for layer in ("17:8", "24:4") for dof in ("tx", "ty", "rz")]
    geometry = {name: {"nominal_ratio": ratio, "ci_lower": lower, "ci_upper": upper} for name in names}
    for name, values in (cells or {}).items():
        geometry[name] = dict(zip(("nominal_ratio", "ci_lower", "ci_upper"), values))
    tracks = {name: {"nominal_ratio": 1.0, "ci_lower": 0.999, "ci_upper": 1.001} for name in ("target", "ref")}
    return geometry, tracks


def test_verdict_passes_non_inferiority_without_directed_improvement():
    module = _module()
    geometry, tracks = _metrics()
    verdict = module.h1_verdict(geometry, tracks, objective_c=1.0, objective_a=2.0,
                                nominal_reference_independent=True)
    assert verdict["h1_p_passed"]
    assert verdict["h1_d_improved"] == []
    assert not verdict["falsified"]


def test_verdict_vetoes_a_single_inferior_cell():
    module = _module()
    geometry, tracks = _metrics(cells={"24:4|rz": (1.0, 0.99, 1.011)})
    verdict = module.h1_verdict(geometry, tracks, 1.0, 2.0, True)
    assert not verdict["h1_p_passed"]
    assert verdict["h1_p_failed"] == ["geometry:24:4|rz"]


def test_directed_improvement_is_read_only_on_the_three_registered_cells():
    module = _module()
    geometry, tracks = _metrics(cells={"17:8|ty": (0.6, 0.55, 0.65), "17:8|rz": (0.5, 0.45, 0.55)})
    verdict = module.h1_verdict(geometry, tracks, 1.0, 2.0, True)
    assert verdict["h1_d_improved"] == ["17:8|ty"]
    assert verdict["h1_d_claim_scope"] == "post_hoc_single_sample"


def test_falsification_chi2_better_truth_significantly_worse():
    module = _module()
    geometry, tracks = _metrics(cells={"17:8|tx": (1.2, 1.05, 1.3)})
    verdict = module.h1_verdict(geometry, tracks, objective_c=1.0, objective_a=2.0,
                                nominal_reference_independent=True)
    assert verdict["falsified_chi2_better_truth_worse"]
    assert not module.h1_verdict(geometry, tracks, 3.0, 2.0, True)["falsified_chi2_better_truth_worse"]


def test_falsification_improvement_only_under_nominal_reference():
    module = _module()
    geometry, tracks = _metrics(cells={"24:4|ty": (0.7, 0.6, 0.8)})
    assert module.h1_verdict(geometry, tracks, 1.0, 2.0, False)["falsified_nominal_reference_only"]
    assert not module.h1_verdict(geometry, tracks, 1.0, 2.0, True)["falsified_nominal_reference_only"]
