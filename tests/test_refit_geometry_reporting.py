"""Geometry export, run summary, measurement refit and box active set."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest
import torch

from mlalign.box_constraints import free_dof_mask
from mlalign.geometry_io import JointGeometry, update_geometry_to_json
from mlalign.helix.measurement_fit import fit_helix_measurements_gauss_newton
from mlalign.helix.trajectory import intersect_plane
from mlalign.helix_init import (
    JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION,
    JOINT_REFIT_PREREGISTRATION_VERSION,
    JOINT_REFIT_SOLVER_GAUSS_NEWTON,
    _measurement_fit_inputs_from_joint,
    _measurement_refit_objective,
    fit_helices_to_measurements,
    measure_joint_refit_stationarity,
)
from mlalign.reporting import write_alignment_summary
from mlalign.track_batches import TrackBatches

from mlalign.helix.gauge import pack_gauge_fixed_track_parameters, reference_gauge_frame
from mlalign.helix.measurement_fit import measurement_standardized_residual


def test_update_geometry_to_json_refreshes_nested_full_matrix(tmp_path):
    output_path = tmp_path / "aligned.json"
    transforms = [
        {
            "ID": "vol=17|lay=8|sen=1",
            "Translation": [1.0, 2.0, 3.0],
            "RotationMatrix": np.eye(3).tolist(),
            "FullMatrix": [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    ]
    deltas = np.array([[0.5, -0.25, 0.0, 0.0, 0.0, np.pi / 2]])

    update_geometry_to_json(transforms, deltas, output_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))[0]
    expected_full_matrix = np.array(
        [
            [0.0, -1.0, 0.0, 1.5],
            [1.0, 0.0, 0.0, 1.75],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(result["FullMatrix"], expected_full_matrix, atol=1e-12)
    assert np.allclose(result["RotationMatrix"], expected_full_matrix[:3, :3], atol=1e-12)
    assert np.allclose(result["Translation"], expected_full_matrix[:3, 3], atol=1e-12)


def test_alignment_summary_keeps_stage_metrics_without_large_tensor_payloads(tmp_path):
    output = tmp_path / "joint_summary.json"
    result = {
        "output_geom": "/tmp/aligned.json",
        "n_tracks": 10,
        "n_hits": 100,
        "initialization_stats": {"retained_tracks": 10, "smt": {"loss": 2.0}},
        "stage_layer_chi2": {"joint": {"17:8": {"LOC0": {"chi2_per_component": 0.5}}}},
        "best_total_loss": 0.5,
        "best_iteration": 12,
        "invalid_fraction": 0.01,
        "predicted_deltas": np.asarray([[0.0, -0.25, 0.0, 0.0, 0.0, 0.0]]),
        "solver": "schur-lm",
        "schur_lm_summary": {"status": "converged", "outer_iterations": 3},
        "final_residuals": torch.zeros((10, 10, 2)),
    }

    write_alignment_summary(result, output)
    summary = json.loads(output.read_text())

    assert summary["initialization_stats"]["retained_tracks"] == 10
    assert summary["stage_layer_chi2"]["joint"]["17:8"]["LOC0"]["chi2_per_component"] == 0.5
    assert summary["solver"] == "schur-lm"
    assert summary["schur_lm_summary"]["status"] == "converged"
    assert summary["correction_max_abs"] == pytest.approx(0.25)
    assert "predicted_deltas" not in summary
    assert "final_residuals" not in summary


def test_alignment_summary_serializes_nonfinite_diagnostics_as_null(tmp_path):
    output = tmp_path / "joint_summary.json"

    write_alignment_summary(
        {
            "truth_metrics": {
                "17:8": {
                    "tx": {"correlation": float("nan")},
                    "ty": {"recovery": float("inf")},
                }
            }
        },
        output,
    )

    summary = json.loads(output.read_text())
    assert summary["truth_metrics"]["17:8"]["tx"]["correlation"] is None
    assert summary["truth_metrics"]["17:8"]["ty"]["recovery"] is None


def build_synthetic_joint_case(n_tracks=4, n_hits=6, noise=0.02):
    """A barrel-like case whose intersections are all valid.

    Planes carry the normal along x with local axes (y, z), so a nearly radial
    helix crosses every one of them.  The last layer is the alignment target and
    is therefore excluded from the refit, which is what makes ``fit_mask`` and
    ``is_target`` complementary here.
    """
    dtype = torch.float64
    generator = torch.Generator().manual_seed(20260826)

    centers = torch.zeros((n_hits, 3), dtype=dtype)
    centers[:, 0] = torch.linspace(50.0, 300.0, n_hits, dtype=dtype)
    basis = torch.stack(
        (
            torch.tensor([0.0, 1.0, 0.0], dtype=dtype),
            torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
            torch.tensor([1.0, 0.0, 0.0], dtype=dtype),
        ),
        dim=1,
    )
    rotations = basis.unsqueeze(0).expand(n_hits, 3, 3).contiguous()

    truth = torch.zeros((n_tracks, 6), dtype=dtype)
    truth[:, 3] = torch.linspace(-0.05, 0.05, n_tracks, dtype=dtype)
    truth[:, 4] = torch.linspace(-0.2, 0.2, n_tracks, dtype=dtype)
    truth[:, 5] = torch.linspace(-2.0e-4, 2.0e-4, n_tracks, dtype=dtype)

    tracks = truth[:, None, :].expand(-1, n_hits, -1)
    hit_centers = centers.unsqueeze(0).expand(n_tracks, -1, -1)
    hit_rotations = rotations.unsqueeze(0).expand(n_tracks, -1, -1, -1)
    _s, points, valid = intersect_plane(
        tracks, hit_centers, hit_rotations[..., :, 2]
    )
    assert bool(valid.all()), "synthetic geometry must intersect every plane"
    local = torch.matmul(
        hit_rotations.transpose(-1, -2), (points - hit_centers).unsqueeze(-1)
    ).squeeze(-1)[..., :2]
    measured = local + noise * torch.randn(
        local.shape, generator=generator, dtype=dtype
    )

    is_target = torch.zeros((n_tracks, n_hits), dtype=torch.bool)
    is_target[:, -1] = True
    batches = TrackBatches(
        measured_local=measured,
        covariances=torch.full_like(measured, noise**2),
        sensor_idx=torch.arange(n_hits).unsqueeze(0).expand(n_tracks, -1).contiguous(),
        hit_mask=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        loc1_mask=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        g_smt=points.clone(),
        p_smt=torch.zeros((n_tracks, n_hits, 3), dtype=dtype),
        volume_id=torch.full((n_tracks, n_hits), 16, dtype=torch.long),
        layer_id=torch.arange(n_hits).unsqueeze(0).expand(n_tracks, -1).contiguous(),
        is_target=is_target,
        event_track_ids=torch.zeros((n_tracks, 2), dtype=torch.long),
        fit_mask=~is_target,
        fit_span=torch.full((n_tracks,), 250.0, dtype=dtype),
    )
    geometry = JointGeometry(
        transforms=[],
        input_transforms=[],
        centers=centers,
        rotations=rotations,
        target_mask=is_target[0].clone(),
        id_to_idx={(16, layer, 0): layer for layer in range(n_hits)},
        input_indices=list(range(n_hits)),
    )
    seed = truth.clone()
    seed[:, :3] += 0.5
    seed[:, 3] += 0.01
    return batches, geometry, truth, seed


def test_refit_stationarity_is_measured_in_the_half_chi2_gauge():
    """The production objective and shared helix residual must use one gauge.

    ``_measurement_refit_objective`` divides by the component count while the
    pre-registered stationarity threshold lives in ``F = 0.5 * chi2``. Undoing
    that normalisation has to land exactly on the shared standardized residual.
    """
    batches, geometry, _truth, seed = build_synthetic_joint_case()

    loss, _invalid, component_mask = _measurement_refit_objective(
        seed, batches, geometry
    )
    autograd_half_chi2 = float(loss) * float(component_mask.sum()) / 2.0

    inputs = _measurement_fit_inputs_from_joint(batches, geometry, seed)
    residual = measurement_standardized_residual(
        pack_gauge_fixed_track_parameters(seed),
        reference_gauge_frame(seed),
        inputs,
    )
    validation_half_chi2 = float(0.5 * residual.square().sum())

    assert validation_half_chi2 > 0.0
    assert abs(autograd_half_chi2 - validation_half_chi2) <= 1e-9 * validation_half_chi2


def test_joint_measurement_inputs_preserve_the_production_objective():
    """Catch an adapter that changes masks, variances, or residual sign."""
    batches, geometry, _truth, seed = build_synthetic_joint_case()

    loss, _invalid, component_mask = _measurement_refit_objective(
        seed, batches, geometry
    )
    inputs = _measurement_fit_inputs_from_joint(batches, geometry, seed)
    frame = reference_gauge_frame(seed)
    theta = pack_gauge_fixed_track_parameters(seed)
    residual = measurement_standardized_residual(theta, frame, inputs)

    expected_half_chi2 = float(loss) * float(component_mask.sum()) / 2.0
    actual_half_chi2 = float(0.5 * residual.square().sum())
    assert actual_half_chi2 > 0.0
    assert actual_half_chi2 == pytest.approx(expected_half_chi2, rel=1.0e-9)


def test_default_joint_refit_does_not_import_validation_module(monkeypatch):
    """The production solver must remain usable when experiments are unavailable."""
    batches, geometry, _truth, seed = build_synthetic_joint_case()
    real_import = __import__

    def reject_validation(name, *args, **kwargs):
        forbidden = (
            "validate_helix_model",
            "broken_helix_model",
            "alignment_experiments",
        )
        if name.endswith(forbidden):
            raise AssertionError(f"production refit imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_validation)
    fitted, stats = fit_helices_to_measurements(batches, geometry, seed)

    assert fitted.shape == seed.shape
    assert stats["solver"] == "gauss-newton"
    assert stats["stationarity_gate"]["non_stationary_row_fraction"] == 0.0


def test_measurement_inputs_from_joint_relabel_without_reinterpreting():
    batches, geometry, _truth, seed = build_synthetic_joint_case()

    relabelled = _measurement_fit_inputs_from_joint(batches, geometry, seed)

    assert torch.equal(relabelled.measured, batches.measured_local)
    assert torch.equal(relabelled.variance, batches.covariances)
    assert torch.equal(relabelled.component_mask[..., 0], batches.fit_mask)
    assert torch.equal(
        relabelled.component_mask[..., 1], batches.fit_mask & batches.loc1_mask
    )
    assert relabelled.centers.shape == (*batches.fit_mask.shape, 3)
    assert relabelled.rotations.shape == (*batches.fit_mask.shape, 3, 3)


def test_measurement_inputs_from_joint_fail_closed_on_target_leakage():
    batches, geometry, _truth, seed = build_synthetic_joint_case()
    batches.fit_mask = torch.ones_like(batches.fit_mask)

    with pytest.raises(ValueError, match="target-layer hits"):
        _measurement_fit_inputs_from_joint(batches, geometry, seed)


def test_refit_stationarity_measurement_leaves_the_fit_untouched():
    batches, geometry, _truth, seed = build_synthetic_joint_case()

    measured_fit, measured_stats = fit_helices_to_measurements(
        batches, geometry, seed, stationarity=True
    )
    plain_fit, plain_stats = fit_helices_to_measurements(
        batches, geometry, seed, stationarity=False
    )

    assert torch.equal(measured_fit, plain_fit)
    assert "stationarity" not in plain_stats
    assert set(measured_stats["stationarity"]) == {"seed", "refit"}
    assert (
        measured_stats["chi2_per_component"] == plain_stats["chi2_per_component"]
    )


def test_refit_stationarity_separates_a_truncated_fit_from_a_solved_one():
    """Both directions, so a decrement that is always zero cannot pass.

    The unrefitted seed must be measured as non-stationary, and the Gauss-Newton
    solution of the same objective must come back stationary.  Without the
    first half this test would still pass if the measurement collapsed to a
    constant.
    """
    batches, geometry, _truth, seed = build_synthetic_joint_case()

    seed_report = measure_joint_refit_stationarity(batches, geometry, seed)
    solved, _stats = fit_helices_to_measurements(batches, geometry, seed)
    solved_report = measure_joint_refit_stationarity(batches, geometry, solved)
    assert seed_report["non_stationary_row_fraction"] == 1.0
    assert (
        solved_report["newton_decrement_p50"]
        < seed_report["newton_decrement_p50"]
    )
    assert solved_report["non_stationary_row_fraction"] == 0.0
    assert (
        solved_report["newton_decrement_max"]
        < solved_report["stationary_threshold"]
    )


def test_gauss_newton_is_the_default_joint_refit_and_meets_the_criterion():
    """Iteration 19: the refit must land on the pre-registered criterion.

    The keys of the removed Adam refit must stay absent rather than carrying a
    placeholder, so a reader of ``initialization_stats`` cannot mistake a
    Gauss-Newton run for one whose checkpoint happened never to improve.
    """
    batches, geometry, _truth, seed = build_synthetic_joint_case()

    _fitted, stats = fit_helices_to_measurements(batches, geometry, seed)

    assert stats["solver"] == JOINT_REFIT_SOLVER_GAUSS_NEWTON
    assert (
        stats["joint_refit_preregistration_version"]
        == JOINT_REFIT_PREREGISTRATION_VERSION
    )
    gate = stats["stationarity_gate"]
    assert gate["n_non_finite"] == 0
    assert (
        gate["non_stationary_row_fraction"]
        <= JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION
    )
    assert stats["convergence"]["converged"] is True
    for adam_only in ("best_iteration", "improved", "first_step"):
        assert adam_only not in stats


def test_joint_refit_rejects_a_truncated_gauss_newton_solution(monkeypatch):
    """Both directions, so a gate that can never fire cannot pass.

    One Gauss-Newton iteration on this case leaves every row non-stationary; the
    full solve leaves none.  Asserting only the second half would also pass for a
    criterion that is never evaluated.
    """
    batches, geometry, _truth, seed = build_synthetic_joint_case()
    full_solver = fit_helix_measurements_gauss_newton

    def truncated_solver(*args, **kwargs):
        return full_solver(*args, **{**kwargs, "max_iters": 1})

    monkeypatch.setattr(
        importlib.import_module("mlalign.helix_init"),
        "fit_helix_measurements_gauss_newton",
        truncated_solver,
    )
    with pytest.raises(ValueError, match="pre-registered stationary point"):
        fit_helices_to_measurements(batches, geometry, seed)

    monkeypatch.undo()
    _fitted, stats = fit_helices_to_measurements(batches, geometry, seed)
    assert stats["stationarity_gate"]["non_stationary_row_fraction"] == 0.0


def test_gauss_newton_refit_stationarity_reuses_the_identical_gate_report(
    monkeypatch,
):
    """The reused refit report must equal an independent recomputation, and the
    diagnostic must then measure only the seed."""
    batches, geometry, _truth, seed = build_synthetic_joint_case()
    measured_rows = []
    original = measure_joint_refit_stationarity

    def counted(batches_, geometry_, parameters, **kwargs):
        measured_rows.append(parameters)
        return original(batches_, geometry_, parameters, **kwargs)

    monkeypatch.setattr(
        importlib.import_module("mlalign.helix_init"),
        "measure_joint_refit_stationarity",
        counted,
    )
    fitted, stats = fit_helices_to_measurements(
        batches, geometry, seed, stationarity=True
    )

    assert len(measured_rows) == 1
    assert torch.equal(measured_rows[0], seed.to(torch.float64))
    assert stats["stationarity"]["refit"] == original(batches, geometry, fitted)
    assert stats["stationarity"]["refit"] == stats["stationarity_gate"]


def test_free_dof_mask_freezes_only_outward_pointing_bounds():
    lower = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0])
    upper = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    #        at upper    at upper    at lower    at lower    interior
    values = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.25])
    #        g<0 outward g>0 inward  g>0 outward g<0 inward  any
    gradient = torch.tensor([-3.0, 3.0, 3.0, -3.0, 7.0])

    free = free_dof_mask(values, gradient, lower, upper)

    assert free.tolist() == [False, True, False, True, True]


def test_free_dof_mask_keeps_a_vanishing_gradient_free():
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    values = torch.tensor([1.0, -1.0])

    free = free_dof_mask(values, torch.zeros(2), lower, upper)

    assert free.tolist() == [True, True]

