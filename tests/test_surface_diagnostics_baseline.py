"""A4: the surface before/after comparison must use one residual definition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("ml_alignment_autograd_base", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SENTINEL_ROOT_RESIDUAL = [[999.0, 0.0]]


def _install_surface_fixture(m, monkeypatch):
    """Single sensor, identity geometry, one 1-D hit."""
    monkeypatch.setattr(m.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(
        m,
        "load_geometry_from_json",
        lambda _p: (
            [{
                "ID": "vol=16|lay=6|sen=0",
                "Translation": [0.0, 0.0, 0.0],
                "RotationMatrix": torch.eye(3).tolist(),
            }],
            1,
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0),
            {(16, 6, 0): 0},
        ),
    )
    monkeypatch.setattr(
        m,
        "load_variable_information",
        lambda *_a, **_k: m.AlignmentInputs(
            hit_sensor_ids=torch.tensor([0], dtype=torch.long),
            # Deliberately unlike any geometric projection: if this value reaches
            # the diagnostics, the baseline is still the ROOT/KF residual.
            initial_residuals=torch.tensor(SENTINEL_ROOT_RESIDUAL, dtype=torch.float32),
            covariances=torch.ones((1, 2), dtype=torch.float32),
            track_params=torch.zeros((1, 5), dtype=torch.float32),
            track_chi2_ndf=torch.ones((1, 1), dtype=torch.float32),
            hit_global_positions=torch.zeros((1, 3), dtype=torch.float32),
            hit_directions=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            hit_qoverp=torch.ones((1, 1), dtype=torch.float32),
            hit_volume_ids=torch.tensor([16], dtype=torch.long),
            hit_layer_ids=torch.tensor([6], dtype=torch.long),
            hit_module_ids=torch.tensor([0], dtype=torch.long),
            hit_event_ids=torch.tensor([0], dtype=torch.long),
            hit_track_ids=torch.tensor([0], dtype=torch.long),
            measured_local=torch.tensor([[7.0, 123.0]], dtype=torch.float32),
            loc1_valid=torch.tensor([False]),
        ),
    )
    monkeypatch.setattr(m, "validate_against_truth", lambda *_a, **_k: {})
    monkeypatch.setattr(m, "update_geometry_to_json", lambda *_a, **_k: None)


def test_surface_diagnostics_baseline_uses_zero_delta_projection(monkeypatch):
    m = load_module()
    _install_surface_fixture(m, monkeypatch)

    captured = {}

    def recorder(init_res_np, final_res, vols_np, lays_np, target_layers, **kwargs):
        captured["init"] = np.asarray(init_res_np)
        captured["component_mask"] = kwargs.get("component_mask")
        return {(16, 6): {}}

    monkeypatch.setattr(m, "print_per_layer_residual_diagnostics", recorder)

    m.run_alignment_pipeline(
        m.AlignmentRunConfig(epochs=1, make_plot=False, target_layers=((16, 6),))
    )

    assert "init" in captured, "diagnostics were never called"
    # The ROOT/KF residual must not be the baseline any more.
    assert not np.allclose(captured["init"], np.array(SENTINEL_ROOT_RESIDUAL))

    expected = m.compute_autograd_residuals(
        torch.tensor([[7.0, 123.0]], dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.eye(3, dtype=torch.float32).unsqueeze(0),
        torch.zeros((1, 6), dtype=torch.float32),
    )
    component_mask = torch.tensor([[True, False]])
    expected = torch.where(component_mask, expected, torch.zeros_like(expected))
    assert np.allclose(captured["init"], expected.numpy(), atol=1e-6)


def test_surface_diagnostics_receives_component_mask(monkeypatch):
    m = load_module()
    _install_surface_fixture(m, monkeypatch)

    captured = {}

    def recorder(init_res_np, final_res, vols_np, lays_np, target_layers, **kwargs):
        captured["component_mask"] = kwargs.get("component_mask")
        return {(16, 6): {}}

    monkeypatch.setattr(m, "print_per_layer_residual_diagnostics", recorder)

    m.run_alignment_pipeline(
        m.AlignmentRunConfig(epochs=1, make_plot=False, target_layers=((16, 6),))
    )

    mask = captured["component_mask"]
    assert mask is not None, "component_mask was not forwarded to the diagnostics"
    assert np.asarray(mask).tolist() == [[True, False]]


def test_surface_target_metrics_come_from_the_diagnostics_function(monkeypatch):
    """A3: no second, inline metrics computation may survive."""
    m = load_module()
    _install_surface_fixture(m, monkeypatch)

    sentinel = {(16, 6): {"n_hits": 4242, "mean_after_loc0": -1.5}}
    monkeypatch.setattr(
        m, "print_per_layer_residual_diagnostics", lambda *_a, **_k: sentinel
    )

    result = m.run_alignment_pipeline(
        m.AlignmentRunConfig(epochs=1, make_plot=False, target_layers=((16, 6),))
    )

    assert result["target_metrics"] is sentinel
