"""Phase 3: disabling hit weights must not change the training computation.

Multiplying by a uniform weight is mathematically a no-op, but the code path and
the autograd graph do change, so equality is asserted bit-for-bit on the loss,
the gradient and the parameters after one optimizer step.

float32 throughout, matching the surface solver: the legacy path divides by
``mean(w) + 1e-12``, which rounds to exactly 1.0 in float32 but not in float64.
normalize_hit_weights deliberately keeps its float64 semantics unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("ml_alignment_autograd_weights", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


N_SENSORS = 3
N_HITS = 5


def _fixture(m):
    """Deterministic inputs plus the two ingredients of the training step."""
    torch.manual_seed(1234)
    sensor_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    centers = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]], dtype=torch.float32
    )
    rotations = torch.eye(3, dtype=torch.float32).expand(N_SENSORS, 3, 3).contiguous()
    measured = torch.tensor(
        [[0.5, 0.2], [-0.3, 0.4], [0.1, -0.6], [0.7, 0.0], [-0.2, 0.3]],
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [[9.0, 0.1, 0.2], [0.1, 11.0, 0.3], [0.2, 0.3, 13.0],
         [9.5, 0.4, 0.1], [0.3, 11.5, 0.2]],
        dtype=torch.float32,
    )
    directions = torch.tensor(
        [[1.0, 0.05, 0.02], [0.03, 1.0, 0.04], [0.02, 0.01, 1.0],
         [1.0, 0.02, 0.03], [0.04, 1.0, 0.01]],
        dtype=torch.float32,
    )
    covariances = torch.full((N_HITS, 2), 0.01, dtype=torch.float32)
    component_mask = torch.tensor(
        [[True, True], [True, False], [True, True], [True, False], [True, True]]
    )
    volume_ids = torch.tensor([16, 16, 23, 16, 23], dtype=torch.long)
    layer_ids = torch.tensor([6, 6, 4, 6, 4], dtype=torch.long)
    return dict(
        sensor_ids=sensor_ids, centers=centers, rotations=rotations,
        measured=measured, positions=positions, directions=directions,
        covariances=covariances, component_mask=component_mask,
        volume_ids=volume_ids, layer_ids=layer_ids,
    )


def _make_engine(m, reference_state=None):
    engine = m.AlignmentEngine(N_SENSORS)
    with torch.no_grad():
        if reference_state is None:
            engine.alignment_params.copy_(
                torch.tensor(
                    [[0.01, -0.02, 0.0, 0.0, 0.0, 0.003],
                     [-0.03, 0.04, 0.0, 0.0, 0.0, -0.002],
                     [0.02, 0.01, 0.0, 0.0, 0.0, 0.001]],
                    dtype=torch.float32,
                )
            )
        else:
            engine.load_state_dict(reference_state)
    optimizer = torch.optim.Adam(engine.parameters(), lr=m.DEFAULT_ADAM_LR)
    return engine, optimizer


def _residuals(m, engine, data):
    return m.compute_autograd_residuals(
        data["measured"], data["positions"], data["directions"],
        data["sensor_ids"], data["centers"], data["rotations"],
        engine.alignment_params,
    )


def _step(engine, optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    grad = engine.alignment_params.grad.detach().clone()
    optimizer.step()
    return grad, engine.alignment_params.detach().clone()


def test_uniform_weight_loss_equals_manual_raw_chi2():
    """Passes both before and after the refactor: the maths is unchanged."""
    m = load_module()
    data = _fixture(m)

    engine_a, optimizer_a = _make_engine(m)
    engine_b, optimizer_b = _make_engine(m, reference_state=engine_a.state_dict())
    assert torch.equal(engine_a.alignment_params, engine_b.alignment_params)

    # legacy: explicit ones -> normalize_hit_weights -> weighted chi2
    residuals_a = _residuals(m, engine_a, data)
    loss_a = m.physics_informed_residual_loss(
        residuals_a, data["covariances"],
        torch.ones((N_HITS, 1), dtype=torch.float32),
        component_mask=data["component_mask"],
        hit_volume_ids=data["volume_ids"], hit_layer_ids=data["layer_ids"],
    )

    # manual: raw chi2, no weighting involved at all
    residuals_b = _residuals(m, engine_b, data)
    raw = (residuals_b ** 2) / torch.clamp(data["covariances"], min=m.EPS_COVARIANCE)
    loss_b = torch.sum(
        torch.where(data["component_mask"], raw, torch.zeros_like(raw))
    ) / data["component_mask"].sum().clamp(min=1)

    assert torch.equal(loss_a, loss_b)
    grad_a, params_a = _step(engine_a, optimizer_a, loss_a)
    grad_b, params_b = _step(engine_b, optimizer_b, loss_b)
    assert torch.equal(grad_a, grad_b)
    assert torch.equal(params_a, params_b)


def test_residual_loss_supports_unweighted_path():
    """Fails before the refactor (None reaches normalize_hit_weights)."""
    m = load_module()
    data = _fixture(m)

    engine_a, optimizer_a = _make_engine(m)
    engine_b, optimizer_b = _make_engine(m, reference_state=engine_a.state_dict())
    assert torch.equal(engine_a.alignment_params, engine_b.alignment_params)

    residuals_a = _residuals(m, engine_a, data)
    loss_a = m.physics_informed_residual_loss(
        residuals_a, data["covariances"],
        torch.ones((N_HITS, 1), dtype=torch.float32),
        component_mask=data["component_mask"],
        hit_volume_ids=data["volume_ids"], hit_layer_ids=data["layer_ids"],
    )

    residuals_b = _residuals(m, engine_b, data)
    loss_b = m.physics_informed_residual_loss(
        residuals_b, data["covariances"],
        None,
        component_mask=data["component_mask"],
        hit_volume_ids=data["volume_ids"], hit_layer_ids=data["layer_ids"],
    )

    assert torch.equal(loss_a, loss_b)
    grad_a, params_a = _step(engine_a, optimizer_a, loss_a)
    grad_b, params_b = _step(engine_b, optimizer_b, loss_b)
    assert torch.equal(grad_a, grad_b)
    assert torch.equal(params_a, params_b)


def test_unweighted_loss_matches_explicit_expectation():
    """Standalone loss assertion; no gradients involved."""
    m = load_module()
    residuals = torch.tensor([[2.0, 1.0e6], [4.0, 3.0]], dtype=torch.float32)
    covariances = torch.ones_like(residuals)
    component_mask = torch.tensor([[True, False], [True, True]])

    loss = m.physics_informed_residual_loss(
        residuals, covariances, None, component_mask=component_mask
    )

    assert torch.equal(loss, torch.tensor((4.0 + 16.0 + 9.0) / 3.0, dtype=torch.float32))


def test_engine_returns_none_weights_when_disabled():
    m = load_module()
    engine = m.AlignmentEngine(N_SENSORS)
    assert engine.use_hit_weights is False
    data = _fixture(m)

    residuals, weights = engine(
        data["sensor_ids"], torch.zeros((N_HITS, 5)), data["centers"], data["rotations"],
        torch.ones((N_HITS, 1)), data["measured"], data["positions"], data["directions"],
    )

    assert residuals.shape == (N_HITS, 2)
    assert weights is None


def test_normalize_hit_weights_semantics_are_untouched():
    """No all-ones short circuit was introduced: float64 keeps its 1e-12 offset."""
    m = load_module()
    weights = torch.ones((4, 1), dtype=torch.float64)

    normalized = m.normalize_hit_weights(weights)

    assert not torch.equal(normalized, weights)
    assert torch.allclose(normalized, weights)
