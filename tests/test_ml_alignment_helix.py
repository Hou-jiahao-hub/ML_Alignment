from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_helix.py"
    assert path.exists(), "ml_alignment_helix.py has not been implemented"
    spec = importlib.util.spec_from_file_location("ml_alignment_helix_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_helix_cli_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    module = load_module()
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    args = module.build_parser().parse_args(
        [
            "assess",
            "--nominal-states",
            "inputs/nominal.root",
            "--candidate-states",
            "inputs/candidate.root",
            "--output-summary",
            "outputs/assessment.json",
        ]
    )

    assert args.nominal_states == str(repository_root / "inputs" / "nominal.root")
    assert args.candidate_states == str(repository_root / "inputs" / "candidate.root")
    assert args.output_summary == str(
        repository_root / "outputs" / "assessment.json"
    )


def test_short_strip_loc1_is_kept_with_physical_variance_floor():
    m = load_module()
    variance = torch.tensor([[0.01, 0.04]], dtype=torch.float64)
    volume = torch.tensor([24])
    loc1_valid = torch.tensor([True])

    adjusted, component_mask = m.apply_measurement_variance_policy(
        variance, volume, loc1_valid
    )

    assert component_mask.tolist() == [[True, True]]
    assert torch.allclose(adjusted, torch.tensor([[0.01, 1.44]], dtype=torch.float64))


def test_long_strip_loc1_uses_per_hit_validity_instead_of_volume_only():
    m = load_module()
    variance = torch.tensor([[0.02, 0.5], [0.02, 0.5]], dtype=torch.float64)
    volume = torch.tensor([29, 29])
    loc1_valid = torch.tensor([False, True])

    adjusted, component_mask = m.apply_measurement_variance_policy(
        variance, volume, loc1_valid
    )

    assert component_mask.tolist() == [[True, False], [True, True]]
    assert torch.equal(adjusted, variance)


def test_surface_stop_rule_handles_target_plateau_and_limit():
    m = load_module()

    assert m.surface_stop_reason([4.0, 1.0]) == "chi2_target"
    assert m.surface_stop_reason([2.0, 1.94, 1.89, 1.82]) == "relative_plateau"
    assert m.surface_stop_reason([3.0] * 15) == "relative_plateau"
    assert m.surface_stop_reason([4.0, 3.0], max_iterations=2) == "max_iterations"
    assert m.surface_stop_reason([4.0, 3.0], max_iterations=15) is None


def test_failed_layer_selection_checks_short_strip_loc1():
    m = load_module()
    nominal = {
        "17:8": {"loc0_rmse_mm": 0.01, "loc1_rmse_mm": 0.02},
        "24:4": {"loc0_rmse_mm": 0.04, "loc1_rmse_mm": 1.0},
    }
    candidate = {
        "17:8": {"loc0_rmse_mm": 0.011, "loc1_rmse_mm": 0.021},
        "24:4": {"loc0_rmse_mm": 0.045, "loc1_rmse_mm": 1.25},
    }

    failed, report = m.select_failed_layers(
        nominal, candidate, ((17, 8), (24, 4)), max_ratio=1.2
    )

    assert failed == ((24, 4),)
    assert report["24:4"]["loc1"]["ratio"] == 1.25
    assert report["24:4"]["loc1"]["failed"] is True


def test_helix_point_has_stable_straight_line_limit():
    m = load_module()
    track = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.5, 0.0]], dtype=torch.float64)
    path = torch.tensor([4.0], dtype=torch.float64)

    assert torch.allclose(
        m.helix_point(track, path),
        torch.tensor([[5.0, 2.0, 5.0]], dtype=torch.float64),
        atol=1e-12,
    )


def test_frozen_helix_geometry_loss_only_differentiates_alignment():
    m = load_module()
    tracks = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    centers = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float64)
    rotations = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    measured = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    variance = torch.ones((1, 2), dtype=torch.float64)
    component_mask = torch.tensor([[True, True]])
    alignment = torch.zeros((1, 6), dtype=torch.float64, requires_grad=True)

    loss, diagnostics = m.frozen_helix_geometry_loss(
        tracks,
        centers,
        rotations,
        measured,
        variance,
        component_mask,
        alignment,
    )
    loss.backward()

    assert tracks.requires_grad is False
    assert alignment.grad is not None
    assert torch.isfinite(alignment.grad).all()
    assert diagnostics["invalid_fraction"] == 0.0


def test_frozen_geometry_optimizer_recovers_local_tx_shift():
    m = load_module()
    count = 10
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    geometry = m.WorkingGeometry(
        transforms=[{"ID": "vol=17|lay=8|sen=1"}],
        input_transforms=[{"ID": "vol=17|lay=8|sen=1"}],
        centers=torch.tensor([[5.0, 0.2, 0.0]], dtype=torch.float64),
        rotations=rotation.unsqueeze(0),
        id_to_idx={(17, 8, 1): 0},
        input_indices=[0],
    )
    shape = (count, 1)
    windows = m.TrackWindows(
        measured_local=torch.zeros((*shape, 2), dtype=torch.float64),
        variance=torch.ones((*shape, 2), dtype=torch.float64),
        loc1_valid=torch.ones(shape, dtype=torch.bool),
        smt_local=torch.zeros((*shape, 2), dtype=torch.float64),
        sensor_idx=torch.zeros(shape, dtype=torch.long),
        hit_mask=torch.ones(shape, dtype=torch.bool),
        fit_mask=torch.zeros(shape, dtype=torch.bool),
        holdout_mask=torch.ones(shape, dtype=torch.bool),
        g_smt=torch.zeros((*shape, 3), dtype=torch.float64),
        p_smt=torch.zeros((*shape, 3), dtype=torch.float64),
        volume_id=torch.full(shape, 17, dtype=torch.long),
        layer_id=torch.full(shape, 8, dtype=torch.long),
        event_track_ids=torch.stack((torch.arange(count), torch.ones(count)), dim=1),
        fit_span=torch.full((count,), 200.0, dtype=torch.float64),
    )
    tracks = torch.zeros((count, 6), dtype=torch.float64)

    deltas, diagnostics = m.optimize_frozen_geometry(
        windows,
        tracks,
        geometry,
        ((17, 8),),
        active_dof=(0,),
        iterations=80,
        learning_rate=0.05,
        lambda_l2=0.0,
        min_hits_per_surface=10,
        min_surface_coverage=1.0,
    )

    assert abs(deltas[0, 0] + 0.2) < 0.03
    assert diagnostics["surface_coverage"] == 1.0


def test_measurement_errors_expand_onto_measured_state_rows():
    m = load_module()
    err_x, err_y = m.expand_measurement_errors(
        np.array([0.1, np.nan, 0.2]),
        np.array([0.01, 0.02]),
        np.array([0.03, 0.04]),
    )

    assert np.allclose(err_x[[0, 2]], [0.01, 0.02])
    assert np.allclose(err_y[[0, 2]], [0.03, 0.04])
    assert np.isnan(err_x[1]) and np.isnan(err_y[1])


def test_per_entry_track_ids_broadcast_to_each_state():
    m = load_module()

    result = m.broadcast_entry_values(np.array([7, 9]), np.array([3, 2]))

    assert result.tolist() == [7, 7, 7, 9, 9]


def test_track_windows_exclude_every_requested_target_layer():
    m = load_module()
    n = 10
    volume = torch.tensor([10, 10, 10, 17, 10, 24, 10, 10, 10, 10])
    layer = torch.tensor([1, 2, 3, 8, 4, 4, 5, 6, 7, 8])
    inputs = m.FlatStates(
        sensor_idx=torch.arange(n),
        measured_local=torch.zeros((n, 2), dtype=torch.float64),
        variance=torch.ones((n, 2), dtype=torch.float64),
        loc1_valid=torch.ones(n, dtype=torch.bool),
        smt_local=torch.zeros((n, 2), dtype=torch.float64),
        g_smt=torch.stack((torch.arange(n, dtype=torch.float64) * 50, torch.zeros(n), torch.zeros(n)), dim=1),
        p_smt=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(n, -1).clone(),
        volume_id=volume,
        layer_id=layer,
        module_id=torch.arange(1, n + 1),
        event_id=torch.zeros(n, dtype=torch.long),
        track_id=torch.ones(n, dtype=torch.long),
        path_length=torch.arange(n, dtype=torch.float64) * 50,
    )

    windows = m.build_track_windows(
        inputs,
        ((17, 8), (24, 4)),
        min_fit_hits=6,
        min_fit_span=100.0,
        context_before=4,
        context_after=4,
    )

    assert len(windows.measured_local) == 2
    assert torch.all(windows.holdout_mask.sum(dim=1) == 1)
    target_in_fit = (
        ((windows.volume_id == 17) & (windows.layer_id == 8))
        | ((windows.volume_id == 24) & (windows.layer_id == 4))
    ) & windows.fit_mask
    assert not target_in_fit.any()
