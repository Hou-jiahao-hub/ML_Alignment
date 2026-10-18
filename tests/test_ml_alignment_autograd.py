from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_ml_alignment_autograd_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("ml_alignment_autograd", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_production_module_import_excludes_experiment_validation_and_acts():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    code = (
        "import sys, ml_alignment_autograd; "
        "forbidden={'alignment_experiments','broken_helix_model',"
        "'validate_helix_model','acts'}; "
        "loaded={name.rsplit('.',1)[-1] for name in sys.modules}; "
        "assert not (forbidden & loaded), forbidden & loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=scripts,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def load_layout_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "alignment_artifact_layout.py"
    spec = importlib.util.spec_from_file_location("alignment_artifact_layout", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_comparison_config_is_not_exposed():
    ml_alignment = load_ml_alignment_autograd_module()

    assert not hasattr(ml_alignment.AlignmentRunConfig(), "reference_file")


def test_reference_comparison_cli_is_not_exposed():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--reference" not in completed.stdout


def test_jacobian_update_interval_cli_is_not_exposed():
    """The autograd solver never used it; the compatibility flag is gone.

    This is the authoritative gate for the removal -- grep is only an aid.
    """
    script = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--jacobian-update-interval" not in completed.stdout

    rejected = subprocess.run(
        [sys.executable, str(script), "--jacobian-update-interval", "20"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "unrecognized arguments" in rejected.stderr


def test_jacobian_update_interval_config_field_is_not_exposed():
    ml_alignment = load_ml_alignment_autograd_module()

    assert not hasattr(ml_alignment.AlignmentRunConfig(), "jacobian_update_interval")


def test_update_geometry_to_json_refreshes_nested_full_matrix(tmp_path):
    ml_alignment = load_ml_alignment_autograd_module()
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

    ml_alignment.update_geometry_to_json(transforms, deltas, output_path)

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


def test_update_geometry_alignment_gain_scales_translation_and_rotation(tmp_path):
    ml_alignment = load_ml_alignment_autograd_module()
    transforms = [
        {
            "ID": "vol=17|lay=8|sen=1",
            "Translation": [0.0, 0.0, 0.0],
            "RotationMatrix": np.eye(3).tolist(),
        }
    ]
    deltas = np.array([[1.0, -2.0, 0.0, 0.0, 0.0, 0.5]])
    output_path = tmp_path / "polish.json"

    ml_alignment.update_geometry_to_json(
        transforms, deltas, output_path, alignment_gain=0.2
    )

    assert transforms[0]["Translation"] == pytest.approx([0.2, -0.4, 0.0])
    expected = ml_alignment.euler_to_rotation_matrix(0.0, 0.0, 0.1)
    assert np.allclose(transforms[0]["RotationMatrix"], expected)


def test_alignment_summary_keeps_stage_metrics_without_large_tensor_payloads(tmp_path):
    ml_alignment = load_ml_alignment_autograd_module()
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
        "solver": "block-lm",
        "fisher_diagnostics": [
            {"condition_number": 10.0, "regularization": 0.2, "accepted": True},
            {"condition_number": float("inf"), "regularization": 0.5, "accepted": False},
        ],
        "final_residuals": torch.zeros((10, 10, 2)),
    }

    ml_alignment.write_alignment_summary(result, output)
    summary = json.loads(output.read_text())

    assert summary["initialization_stats"]["retained_tracks"] == 10
    assert summary["stage_layer_chi2"]["joint"]["17:8"]["LOC0"]["chi2_per_component"] == 0.5
    assert summary["solver"] == "block-lm"
    assert summary["correction_max_abs"] == pytest.approx(0.25)
    assert summary["fisher_summary"] == {
        "evaluations": 2,
        "accepted_steps": 1,
        "rejected_steps": 1,
        "damping_retries": 0,
        "lstsq_fallbacks": 0,
        "clipped_steps": 0,
        "finite_condition_max": 10.0,
        "regularization_max": 0.5,
    }
    assert "predicted_deltas" not in summary
    assert "final_residuals" not in summary


def test_alignment_summary_serializes_nonfinite_diagnostics_as_null(tmp_path):
    ml_alignment = load_ml_alignment_autograd_module()
    output = tmp_path / "joint_summary.json"

    ml_alignment.write_alignment_summary(
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


def test_autograd_cli_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    ml_alignment = load_ml_alignment_autograd_module()
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    args = ml_alignment.build_argument_parser().parse_args(
        [
            "--input-geom",
            "inputs/current.json",
            "--states",
            "inputs/states.root",
            "--summary",
            "inputs/summary.root",
            "--output-geom",
            "outputs/aligned.json",
            "--nominal-full-geom",
            "inputs/nominal.json",
        ]
    )

    assert args.input_geom == str(repository_root / "inputs" / "current.json")
    assert args.states_root == str(repository_root / "inputs" / "states.root")
    assert args.summary_root == str(repository_root / "inputs" / "summary.root")
    assert args.output_geom == str(repository_root / "outputs" / "aligned.json")
    assert args.nominal_full_geom == str(
        repository_root / "inputs" / "nominal.json"
    )


def test_autograd_residual_gradient_matches_translation_and_gamma_reference():
    ml_alignment = load_ml_alignment_autograd_module()

    centers = torch.zeros((1, 3), dtype=torch.float32)
    rotations = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    hit_sensor_ids = torch.tensor([0], dtype=torch.long)
    hit_global_positions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    hit_directions = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    measured_local = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    alignment_params = torch.zeros((1, 6), dtype=torch.float32, requires_grad=True)

    residuals = ml_alignment.compute_autograd_residuals(
        measured_local,
        hit_global_positions,
        hit_directions,
        hit_sensor_ids,
        centers,
        rotations,
        alignment_params,
    )

    residuals[0, 0].backward(retain_graph=True)
    assert torch.allclose(
        alignment_params.grad[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )

    alignment_params.grad.zero_()
    residuals[0, 1].backward()
    assert torch.allclose(
        alignment_params.grad[0],
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    )


def test_autograd_residual_gradient_includes_tz_path_correction():
    ml_alignment = load_ml_alignment_autograd_module()

    inv_sqrt2 = 2.0 ** -0.5
    centers = torch.zeros((1, 3), dtype=torch.float32)
    rotations = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    hit_sensor_ids = torch.tensor([0], dtype=torch.long)
    hit_global_positions = torch.zeros((1, 3), dtype=torch.float32)
    hit_directions = torch.tensor([[inv_sqrt2, 0.0, inv_sqrt2]], dtype=torch.float32)
    measured_local = torch.zeros((1, 2), dtype=torch.float32)
    alignment_params = torch.zeros((1, 6), dtype=torch.float32, requires_grad=True)

    residuals = ml_alignment.compute_autograd_residuals(
        measured_local,
        hit_global_positions,
        hit_directions,
        hit_sensor_ids,
        centers,
        rotations,
        alignment_params,
    )

    residuals[0, 0].backward()
    assert torch.allclose(
        alignment_params.grad[0],
        torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        atol=1e-6,
    )


def test_autograd_pipeline_does_not_import_align_derivlib(monkeypatch):
    ml_alignment = load_ml_alignment_autograd_module()
    observed_measurements = []
    original_compute_residuals = ml_alignment.compute_autograd_residuals

    def record_measurements(measured_local, *args, **kwargs):
        observed_measurements.append(measured_local.detach().clone())
        return original_compute_residuals(measured_local, *args, **kwargs)

    monkeypatch.setattr(ml_alignment, "compute_autograd_residuals", record_measurements)

    assert not hasattr(ml_alignment, "compute_jacobians_from_alignderivlib")

    monkeypatch.setattr(ml_alignment.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        ml_alignment,
        "load_geometry_from_json",
        lambda _path: (
            [
                {
                    "ID": "vol=17|lay=8|sen=0",
                    "Translation": [0.0, 0.0, 0.0],
                    "RotationMatrix": torch.eye(3).tolist(),
                }
            ],
            1,
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0),
            {(17, 8, 0): 0},
        ),
    )
    monkeypatch.setattr(
        ml_alignment,
        "load_variable_information",
        lambda *_args, **_kwargs: ml_alignment.AlignmentInputs(
            hit_sensor_ids=torch.tensor([0], dtype=torch.long),
            initial_residuals=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            covariances=torch.ones((1, 2), dtype=torch.float32),
            track_params=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
            track_chi2_ndf=torch.ones((1, 1), dtype=torch.float32),
            hit_global_positions=torch.zeros((1, 3), dtype=torch.float32),
            hit_directions=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
            hit_qoverp=torch.ones((1, 1), dtype=torch.float32),
            hit_volume_ids=torch.tensor([17], dtype=torch.long),
            hit_layer_ids=torch.tensor([8], dtype=torch.long),
            hit_module_ids=torch.tensor([0], dtype=torch.long),
            hit_event_ids=torch.tensor([0], dtype=torch.long),
            hit_track_ids=torch.tensor([0], dtype=torch.long),
            measured_local=torch.tensor([[7.0, 123.0]], dtype=torch.float32),
            loc1_valid=torch.tensor([False]),
        ),
    )
    expected_truth_metrics = {"sentinel": 1.0}
    monkeypatch.setattr(
        ml_alignment,
        "validate_against_truth",
        lambda *_args: expected_truth_metrics,
    )
    monkeypatch.setattr(
        ml_alignment, "update_geometry_to_json", lambda *_args, **_kwargs: None
    )

    result = ml_alignment.run_alignment_pipeline(
        ml_alignment.AlignmentRunConfig(
            epochs=1,
            make_plot=False,
            target_layers=((17, 8),),
        )
    )

    assert result["predicted_deltas"].shape == (1, 6)
    assert result["truth_metrics"] == expected_truth_metrics
    assert observed_measurements
    assert torch.equal(observed_measurements[0], torch.tensor([[7.0, 123.0]]))


def test_alignment_run_config_defaults_use_target_subdirectories():
    ml_alignment = load_ml_alignment_autograd_module()

    layout = load_layout_module()
    target_dir = layout.TARGET_LAYER_DIR
    suffix = layout.TARGET_LAYER_SUFFIX

    config = ml_alignment.AlignmentRunConfig()

    assert (
        config.input_geom
        == str(layout.DATA_ROOT / "alignment" / target_dir / f"misalignment_applied_{suffix}.json")
    )
    assert (
        config.states_root
        == str(layout.DATA_ROOT / "reconstruction" / "misaligned" / target_dir / f"trackstates_kf_{suffix}.root")
    )
    assert (
        config.summary_root
        == str(layout.DATA_ROOT / "reconstruction" / "misaligned" / target_dir / f"tracksummary_kf_{suffix}.root")
    )
    assert (
        config.output_geom
        == str(layout.DATA_ROOT / "alignment" / target_dir / f"aligned_transforms_{suffix}_ml.json")
    )


def load_validate_helix_model_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_helix_model.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("validate_helix_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_synthetic_joint_case(m, n_tracks=4, n_hits=6, noise=0.02):
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
    _s, points, valid = m.intersect_plane(
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
    batches = m.TrackBatches(
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
    geometry = m.JointGeometry(
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
    m = load_ml_alignment_autograd_module()
    h = importlib.import_module("helix_model")
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    loss, _invalid, component_mask = m._measurement_refit_objective(
        seed, batches, geometry
    )
    autograd_half_chi2 = float(loss) * float(component_mask.sum()) / 2.0

    inputs = m._measurement_fit_inputs_from_joint(batches, geometry, seed)
    residual = h.measurement_standardized_residual(
        h.pack_gauge_fixed_track_parameters(seed),
        h.reference_gauge_frame(seed),
        inputs,
    )
    validation_half_chi2 = float(0.5 * residual.square().sum())

    assert validation_half_chi2 > 0.0
    assert abs(autograd_half_chi2 - validation_half_chi2) <= 1e-9 * validation_half_chi2


def test_joint_measurement_inputs_preserve_the_production_objective():
    """Catch an adapter that changes masks, variances, or residual sign."""
    m = load_ml_alignment_autograd_module()
    h = importlib.import_module("helix_model")
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    loss, _invalid, component_mask = m._measurement_refit_objective(
        seed, batches, geometry
    )
    inputs = m._measurement_fit_inputs_from_joint(batches, geometry, seed)
    frame = h.reference_gauge_frame(seed)
    theta = h.pack_gauge_fixed_track_parameters(seed)
    residual = h.measurement_standardized_residual(theta, frame, inputs)

    expected_half_chi2 = float(loss) * float(component_mask.sum()) / 2.0
    actual_half_chi2 = float(0.5 * residual.square().sum())
    assert actual_half_chi2 > 0.0
    assert actual_half_chi2 == pytest.approx(expected_half_chi2, rel=1.0e-9)


def test_default_joint_refit_does_not_import_validation_module(monkeypatch):
    """The production solver must remain usable when experiments are unavailable."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)
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
    fitted, stats = m.fit_helices_to_measurements(batches, geometry, seed)

    assert fitted.shape == seed.shape
    assert stats["solver"] == "gauss-newton"
    assert stats["stationarity_gate"]["non_stationary_row_fraction"] == 0.0


def test_measurement_inputs_from_joint_relabel_without_reinterpreting():
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    relabelled = m._measurement_fit_inputs_from_joint(batches, geometry, seed)

    assert torch.equal(relabelled.measured, batches.measured_local)
    assert torch.equal(relabelled.variance, batches.covariances)
    assert torch.equal(relabelled.component_mask[..., 0], batches.fit_mask)
    assert torch.equal(
        relabelled.component_mask[..., 1], batches.fit_mask & batches.loc1_mask
    )
    assert relabelled.centers.shape == (*batches.fit_mask.shape, 3)
    assert relabelled.rotations.shape == (*batches.fit_mask.shape, 3, 3)


def test_measurement_inputs_from_joint_fail_closed_on_target_leakage():
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)
    batches.fit_mask = torch.ones_like(batches.fit_mask)

    with pytest.raises(ValueError, match="target-layer hits"):
        m._measurement_fit_inputs_from_joint(batches, geometry, seed)


def test_refit_stationarity_measurement_leaves_the_fit_untouched():
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    measured_fit, measured_stats = m.fit_helices_to_measurements(
        batches, geometry, seed, stationarity=True
    )
    plain_fit, plain_stats = m.fit_helices_to_measurements(
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
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    seed_report = m.measure_joint_refit_stationarity(batches, geometry, seed)
    solved, _stats = m.fit_helices_to_measurements(batches, geometry, seed)
    solved_report = m.measure_joint_refit_stationarity(batches, geometry, solved)
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
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    _fitted, stats = m.fit_helices_to_measurements(batches, geometry, seed)

    assert stats["solver"] == m.JOINT_REFIT_SOLVER_GAUSS_NEWTON
    assert (
        stats["joint_refit_preregistration_version"]
        == m.JOINT_REFIT_PREREGISTRATION_VERSION
    )
    gate = stats["stationarity_gate"]
    assert gate["n_non_finite"] == 0
    assert (
        gate["non_stationary_row_fraction"]
        <= m.JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION
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
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)
    full_solver = m.fit_helix_measurements_gauss_newton

    def truncated_solver(*args, **kwargs):
        return full_solver(*args, **{**kwargs, "max_iters": 1})

    monkeypatch.setattr(m, "fit_helix_measurements_gauss_newton", truncated_solver)
    with pytest.raises(ValueError, match="pre-registered stationary point"):
        m.fit_helices_to_measurements(batches, geometry, seed)

    monkeypatch.undo()
    _fitted, stats = m.fit_helices_to_measurements(batches, geometry, seed)
    assert stats["stationarity_gate"]["non_stationary_row_fraction"] == 0.0


def test_gauss_newton_refit_stationarity_reuses_the_identical_gate_report(
    monkeypatch,
):
    """The reused refit report must equal an independent recomputation, and the
    diagnostic must then measure only the seed."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)
    measured_rows = []
    original = m.measure_joint_refit_stationarity

    def counted(batches_, geometry_, parameters, **kwargs):
        measured_rows.append(parameters)
        return original(batches_, geometry_, parameters, **kwargs)

    monkeypatch.setattr(m, "measure_joint_refit_stationarity", counted)
    fitted, stats = m.fit_helices_to_measurements(
        batches, geometry, seed, stationarity=True
    )

    assert len(measured_rows) == 1
    assert torch.equal(measured_rows[0], seed.to(torch.float64))
    assert stats["stationarity"]["refit"] == original(batches, geometry, fitted)
    assert stats["stationarity"]["refit"] == stats["stationarity_gate"]


def test_block_lm_free_dof_mask_freezes_only_outward_pointing_bounds():
    m = load_ml_alignment_autograd_module()
    lower = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0])
    upper = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    #        at upper    at upper    at lower    at lower    interior
    values = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.25])
    #        g<0 outward g>0 inward  g>0 outward g<0 inward  any
    gradient = torch.tensor([-3.0, 3.0, 3.0, -3.0, 7.0])

    free = m._block_lm_free_dof_mask(values, gradient, lower, upper)

    assert free.tolist() == [False, True, False, True, True]


def test_block_lm_free_dof_mask_keeps_a_vanishing_gradient_free():
    m = load_ml_alignment_autograd_module()
    lower = torch.tensor([-1.0, -1.0])
    upper = torch.tensor([1.0, 1.0])
    values = torch.tensor([1.0, -1.0])

    free = m._block_lm_free_dof_mask(values, torch.zeros(2), lower, upper)

    assert free.tolist() == [True, True]


def test_block_lm_decrement_matches_the_closed_form_on_a_full_rank_block():
    m = load_ml_alignment_autograd_module()
    fisher = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    gradient = torch.tensor([2.0, 2.0], dtype=torch.float64)

    report = m._block_lm_decrement(fisher, gradient, torch.tensor([True, True]), 1e-4)

    assert report["lambda2"] == pytest.approx(5.0)  # 4/4 + 4/1
    assert report["n_free"] == 2
    assert report["rank_deficient"] is False
    assert report["lambda2_is_lower_bound"] is False
    assert report["null_gradient_max"] == 0.0


def test_block_lm_decrement_drops_floor_supported_directions_and_says_so():
    m = load_ml_alignment_autograd_module()
    fisher = torch.diag(torch.tensor([4.0, 1e-9], dtype=torch.float64))
    free = torch.tensor([True, True])

    quiet = m._block_lm_decrement(fisher, torch.tensor([2.0, 0.0], dtype=torch.float64), free, 1e-4)
    loud = m._block_lm_decrement(fisher, torch.tensor([2.0, 1.0], dtype=torch.float64), free, 1e-4)

    # The weak direction is dropped from both sums, so neither may be read as
    # convergence evidence -- but only the second still has gradient sitting in it.
    assert quiet["lambda2"] == pytest.approx(1.0)
    assert loud["lambda2"] == pytest.approx(1.0)
    assert quiet["rank_deficient"] is True and loud["rank_deficient"] is True
    assert quiet["lambda2_is_lower_bound"] is True
    assert quiet["null_gradient_max"] == pytest.approx(0.0)
    assert loud["null_gradient_max"] == pytest.approx(1.0)


def test_block_lm_decrement_never_exceeds_a_regularized_solve():
    """Regularizing before solving can only shrink the decrement, which is why
    the stationarity check refuses to do it."""
    m = load_ml_alignment_autograd_module()
    fisher = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    gradient = torch.tensor([3.0, -1.0], dtype=torch.float64)
    free = torch.tensor([True, True])

    exact = m._block_lm_decrement(fisher, gradient, free, 0.0)["lambda2"]
    damped = float(
        gradient
        @ torch.linalg.solve(fisher + 1e-3 * torch.eye(2, dtype=torch.float64), gradient)
    )

    assert exact == pytest.approx(float(gradient @ torch.linalg.solve(fisher, gradient)))
    assert damped < exact


def test_block_lm_decrement_reports_a_kkt_point_when_the_free_set_is_empty():
    m = load_ml_alignment_autograd_module()
    fisher = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    gradient = torch.tensor([2.0, 2.0], dtype=torch.float64)

    report = m._block_lm_decrement(fisher, gradient, torch.tensor([False, False]), 1e-4)

    assert report["lambda2"] == 0.0
    assert report["n_free"] == 0
    assert report["rank_deficient"] is False
    assert report["null_gradient_max"] == 0.0


def test_block_lm_fixed_chi2_scores_a_candidate_on_the_original_component_set():
    m = load_ml_alignment_autograd_module()
    covariance = torch.full((3, 2), 0.01, dtype=torch.float64)
    fixed_mask = torch.tensor([[True, True], [True, False], [True, False]])
    current = torch.tensor([[0.1, 0.1], [0.1, 9.0], [3.0, 9.0]], dtype=torch.float64)
    # The candidate improves two components and abandons the hardest one.
    candidate = torch.tensor([[0.05, 0.05], [0.05, 9.0], [3.0, 9.0]], dtype=torch.float64)
    candidate_mask = torch.tensor([[True, True], [True, False], [False, False]])

    before = float(m._block_lm_fixed_chi2(current, covariance, fixed_mask))
    after = float(m._block_lm_fixed_chi2(candidate, covariance, fixed_mask))

    assert before == pytest.approx((0.01 + 0.01 + 0.01 + 9.0) / 0.01)
    assert after == pytest.approx((0.0025 + 0.0025 + 0.0025 + 9.0) / 0.01)
    assert after < before
    # The mean over each side's own mask is the trap: dropping the 3.0 residual
    # makes the candidate look better than it is.
    stale = float(
        (current.square() / covariance)[fixed_mask].mean()
    )
    misleading = float(
        (candidate.square() / covariance)[candidate_mask].mean()
    )
    assert misleading < stale / 100.0


def _block_lm_translation_case(m):
    """The two-plane analytic block-LM problem, shared with test_joint_alignment.

    Normal along x, local axes (y, z), target centre offset by 0.4 mm in y, and
    twelve straight zero-curvature tracks, so the loc0 residual is exactly
    ``0.4 + tx`` with unit derivative.  With sigma = 0.1 over m = 12 rows that
    fixes ``H = m / sigma^2 = 1200`` and ``lambda2 = 1200 e^2`` in closed form.
    """
    module_path = Path(__file__).resolve().parent / "test_joint_alignment.py"
    spec = importlib.util.spec_from_file_location("test_joint_alignment", module_path)
    joint_tests = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = joint_tests
    spec.loader.exec_module(joint_tests)
    return joint_tests._block_lm_translation_problem(m)


def test_block_lm_reports_the_closed_form_curvature_and_convergence():
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=8,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)
    summary = result["block_lm_summary"]

    assert summary["arm"] == "contract-fixed"
    assert summary["claim_scope"] == "convergence-contract-only"
    assert summary["stationary_threshold"] == m.BLOCK_LM_DECREMENT_STATIONARY
    # H = m / sigma^2 = 12 / 0.01, exactly, and the block is full rank.
    assert summary["rank"]["min_eigenvalue_min"] == pytest.approx(1200.0)
    assert summary["rank"]["floor_limited_blocks"] == 0
    assert summary["status_counts"]["converged"] == 1
    assert summary["converged_detail"] == {"interior": 1, "boundary": 0}
    assert summary["decrement_per_component"]["non_stationary_block_fraction"] == 0.0
    # The step cap binds while the 0.4 mm offset is walked down; that is a trust
    # region, not a constraint boundary, so no DOF is frozen.
    assert summary["trust_region"]["clipped_steps"] == 3
    assert summary["boundary"]["frozen_dof_total"] == 0


def test_block_lm_separates_budget_exhaustion_from_convergence():
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=12,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0, 1, 5), seed)
    summary = result["block_lm_summary"]

    # The production arm reports stopped_early = False and calls it a day; the
    # audit says the block never reached a stationary point at all.
    assert result["stopped_early"] is False
    assert result["stop_reason"] == "iteration_budget"
    assert summary["status_counts"]["converged"] == 0
    assert summary["status_counts"]["budget_exhausted"] == 1
    assert summary["decrement_per_component"]["non_stationary_block_fraction"] == 1.0
    assert summary["decrement_per_component"]["max"] > m.BLOCK_LM_DECREMENT_STATIONARY


def test_block_lm_scores_every_candidate_on_the_fixed_component_set():
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=4,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)

    assert result["fisher_diagnostics"]
    for entry in result["fisher_diagnostics"]:
        assert entry["n_components_fixed"] == 24
        assert entry["component_loss"] is False
        # No component is lost here, so the two calibers must agree exactly: the
        # audit adds a reading, it does not redefine the objective.
        assert entry["chi2_fixed_after"] / entry["n_components_fixed"] == pytest.approx(
            entry["loss_after"]
        )
        assert entry["chi2_fixed_before"] / entry["n_components_fixed"] == pytest.approx(
            entry["loss_before"]
        )


def test_block_lm_summary_survives_the_json_round_trip():
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=4,
        lambda_l2=0.0, make_plot=False,
    )
    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)

    written = json.loads(
        json.dumps(m._json_ready(result["block_lm_summary"]), allow_nan=False)
    )

    assert written["arm"] == "contract-fixed"
    assert set(written["status_counts"]) == set(m.BLOCK_LM_TERMINAL_STATUSES)
    # Per-sensor detail stays in memory only.
    assert "blocks" not in written


def _independent_decrement_per_component(m, batches, geometry, tracks, row, sensor, active_dof):
    """Recompute the stationarity statistic without touching the solver's cache.

    Deliberately in physical units, where the solver works in normalized ones:
    ``lambda2 = g^T H^-1 g`` is invariant under the diagonal rescaling
    ``g -> Sg, H -> SHS``, so this is a genuinely separate path that must
    nonetheless land on the same number.
    """
    index = torch.tensor(list(active_dof), dtype=torch.long)
    _residuals, _covariance, mask, _valid = m._sensor_residual_data(
        tracks, row, sensor, batches, geometry
    )

    def standardized(values):
        candidate = row.scatter(0, index, values)
        data, variances, _mask, _ok = m._sensor_residual_data(
            tracks, candidate, sensor, batches, geometry
        )
        return (data / variances.clamp_min(m.EPS_COVARIANCE).sqrt())[mask]

    values = row[index].detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(standardized, values, vectorize=True)
    residual = standardized(values).detach()
    hessian = jacobian.transpose(0, 1) @ jacobian
    gradient = jacobian.transpose(0, 1) @ residual
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    projection = eigenvectors.transpose(0, 1) @ gradient
    lambda2 = float((projection.square() / eigenvalues).sum())
    return lambda2 / int(mask.sum())


def _analytic_multi_dof_case(m):
    """Six planes, exact tracks, no noise, and a geometry offset injected into
    the target plane only -- so the least-squares answer is the offset itself.

    ``build_synthetic_joint_case`` on its own is not analytically solvable: it
    adds measurement noise and hands the solver deliberately wrong frozen
    helices, so the block is fitting geometry to a misspecified model and drifts
    towards its optimum over hundreds of capped steps.  Removing both leaves a
    genuinely full-rank problem with a known answer.
    """
    batches, geometry, truth, _seed = build_synthetic_joint_case(m, noise=0.0)
    batches.covariances = torch.full_like(batches.covariances, 0.01)
    # apply_alignment_torch maps local (tx, ty, tz) to global (tz, tx, ty) here,
    # so a global (y, z) shift is a local (tx, ty) one.
    geometry.centers[-1, 1] += 0.30
    geometry.centers[-1, 2] -= 0.20
    return batches, geometry, truth


@pytest.mark.parametrize("active_dof, iters", [((0,), 8), ((0, 1, 5), 20)])
def test_block_lm_converged_blocks_are_independently_stationary(active_dof, iters):
    """Primary criterion: every block reported converged must survive a
    recomputation of the pre-registered stationarity statistic."""
    m = load_ml_alignment_autograd_module()
    if active_dof == (0,):
        batches, geometry, tracks = _block_lm_translation_case(m)
    else:
        batches, geometry, tracks = _analytic_multi_dof_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=iters,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, active_dof, tracks)
    deltas = torch.as_tensor(result["predicted_deltas"], dtype=torch.float64)

    interior = [
        block for block in result["block_lm_blocks"]
        if block["status"] == m.BLOCK_LM_STATUS_CONVERGED and not block["boundary"]
    ]
    assert interior, "no interior converged block: the assertion below would be vacuous"
    for block in interior:
        sensor = block["sensor_index"]
        recomputed = _independent_decrement_per_component(
            m, batches, geometry, tracks, deltas[sensor], sensor, active_dof
        )
        assert recomputed <= m.BLOCK_LM_DECREMENT_STATIONARY
        assert recomputed == pytest.approx(block["lambda2_per_component"], rel=1e-6)


def test_block_lm_retries_with_more_damping_after_a_rejection(monkeypatch):
    """Primary criterion: a first rejection must raise the damping and try
    again inside the same block visit, not end the solve."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    genuine = m._block_lm_fixed_chi2
    calls = {"n": 0}

    def inflated(residuals, covariance, fixed_mask):
        calls["n"] += 1
        value = genuine(residuals, covariance, fixed_mask)
        return value * 1e6 if calls["n"] == 1 else value

    monkeypatch.setattr(m, "_block_lm_fixed_chi2", inflated)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=8,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)
    summary = result["block_lm_summary"]

    # The retry actually happened -- not merely "it converged in the end".
    assert summary["damping"]["retries_total"] >= 1
    assert summary["damping"]["blocks_with_retry"] == 1
    assert summary["damping"]["max_damping_reached"] >= 10.0 * config.lm_initial_damping
    first, second = result["fisher_diagnostics"][0], result["fisher_diagnostics"][1]
    assert first["accepted"] is False
    assert first["reject_reason"] == "no_decrease"
    assert second["sensor_index"] == first["sensor_index"]
    assert second["iteration"] == first["iteration"]
    assert second["attempt"] == 1
    assert second["damping"] == pytest.approx(10.0 * first["damping"])
    assert second["accepted"] is True
    # And the solve still reaches the analytic answer.
    bound = math.sqrt(24 * m.BLOCK_LM_DECREMENT_STATIONARY * 0.01 / 12)
    assert abs(result["predicted_deltas"][1, 0] + 0.4) <= bound
    assert summary["status_counts"]["converged"] == 1


def test_block_lm_retry_ladder_reaches_the_preregistered_absolute_scale(monkeypatch):
    """Earlier halving must not stop the live solver's retry ladder below 1e5."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    genuine = m._block_lm_fixed_chi2
    calls = {"n": 0}

    def reject_first_eight(residuals, covariance, fixed_mask):
        calls["n"] += 1
        value = genuine(residuals, covariance, fixed_mask)
        return value * 1e6 if calls["n"] <= 8 else value

    monkeypatch.setattr(m, "_block_lm_fixed_chi2", reject_first_eight)
    config = m.AlignmentRunConfig(
        track_mode="free",
        joint_solver="block-lm",
        lm_max_iters=8,
        lm_initial_damping=m.BLOCK_LM_MIN_DAMPING,
        lambda_l2=0.0,
        make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)
    first_visit = [
        entry for entry in result["fisher_diagnostics"]
        if entry["iteration"] == 0
    ]

    assert [entry["damping"] for entry in first_visit] == pytest.approx(
        [1e-12, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
    )
    assert first_visit[-1]["accepted"] is True


@pytest.mark.parametrize("active_dof, iters", [((0,), 8), ((0, 1, 5), 20)])
def test_block_lm_accepted_steps_never_lose_components(active_dof, iters):
    """Secondary criterion: no accepted step may drop a component that was
    valid at the point it stepped from."""
    m = load_ml_alignment_autograd_module()
    if active_dof == (0,):
        batches, geometry, tracks = _block_lm_translation_case(m)
    else:
        batches, geometry, _truth, tracks = build_synthetic_joint_case(m)
    # The misspecified fixture is kept here on purpose: the guard has to hold
    # while the solver is still far from any optimum, not only near one.
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=iters,
        lambda_l2=0.0, make_plot=False,
    )

    guard = m.run_block_lm_alignment(batches, geometry, config, active_dof, tracks)[
        "block_lm_summary"
    ]["component_guard"]

    assert guard["accepted_steps"] > 0
    assert guard["min_components_delta_accepted"] >= 0
    assert guard["min_components_end_minus_start"] >= 0


def test_block_lm_reports_a_structurally_blind_dof_as_rank_deficient():
    """DOF 2 is the local normal: R @ (0, 0, tz) is along x, and both local
    coordinates are independent of the plane's x offset, so that Jacobian column
    is identically zero.  The block must not be reported as converged."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=8,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0, 2), tracks)
    summary = result["block_lm_summary"]

    assert summary["status_counts"]["rank_deficient"] == 1
    assert summary["status_counts"]["converged"] == 0
    assert summary["rank"]["floor_limited_blocks"] == 1
    assert summary["rank"]["null_gradient_max"] == pytest.approx(0.0)
    # block_lm_blocks is ordered by target sensor, and there is exactly one.
    assert result["block_lm_blocks"][0]["sensor_index"] == 1
    assert result["block_lm_blocks"][0]["lambda2_is_lower_bound"] is True
    # Rank deficiency is reported, not treated as a reason to stop: the
    # well-determined direction is still solved.
    assert result["predicted_deltas"][1, 0] == pytest.approx(-0.4, abs=2e-3)


def test_block_lm_lstsq_fallback_is_a_sticky_rank_deficient_terminal_state():
    """A solve fallback is numerical rank evidence even if the last spectrum looks full rank."""
    m = load_ml_alignment_autograd_module()
    record = m._block_lm_new_record(7, m.AlignmentRunConfig())
    record.update(
        {
            "n_components_end": 12,
            "rank_deficient": False,
            "stationary": True,
            "lstsq_fallbacks": 1,
        }
    )

    assert m._block_lm_block_status(record) == m.BLOCK_LM_STATUS_RANK_DEFICIENT


def test_block_lm_refuses_a_candidate_that_buys_its_improvement_by_losing_a_hit(monkeypatch):
    """The trap the fixed comparison set exists to close: an outlier hit whose
    intersection the candidate kills, so the mean over the survivors improves."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    batches.measured_local[3, 1, 0] += 5.0
    genuine = m.intersect_plane
    # Read the unaligned centre off the geometry rather than writing 0.4: the
    # fixture stores it in float32, so the literal does not compare equal to the
    # widened value the solver actually sees.
    nominal_y = float(geometry.centers[1, 1])

    def failing(track_parameters, centers, normals, n_newton=12):
        path, intersections, valid = genuine(track_parameters, centers, normals, n_newton)
        # Only the per-sensor path (1-D validity) and only once the plane has
        # actually moved: the current point keeps the outlier, the candidate
        # loses it.  That asymmetry is exactly what makes the mean over each
        # side's own mask look like an improvement.
        moved = bool((centers[..., 1] - nominal_y).abs().max() > 1e-12)
        if valid.ndim == 1 and moved:
            valid = valid.clone()
            valid[3] = False
        return path, intersections, valid

    monkeypatch.setattr(m, "intersect_plane", failing)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=4,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)
    summary = result["block_lm_summary"]

    assert summary["status_counts"]["intersection_failed"] == 1
    assert summary["status_counts"]["converged"] == 0
    assert summary["component_guard"]["component_loss_rejections"] == (
        1 + m.BLOCK_LM_MAX_DAMPING_RETRIES
    )
    assert summary["component_guard"]["accepted_steps"] == 0
    # Not one step was taken, so the geometry is untouched.
    assert result["predicted_deltas"][1, 0] == 0.0
    # The old caliber would have taken it: scored on its own mask, the
    # candidate that drops the 5.0 mm outlier looks strictly better.
    trap = result["fisher_diagnostics"][0]
    assert trap["component_loss"] is True
    assert trap["components_after"] < trap["components_before"]


def test_block_lm_calls_a_bound_it_cannot_leave_a_converged_kkt_point():
    """With the optimum outside the box, the solver settles on the bound.  The
    projected gradient vanishes there, so this is convergence -- but it is a
    boundary one, and the summary must keep the two apart."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=12,
        lambda_l2=0.0, make_plot=False, joint_clamp_t=0.2,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)
    summary = result["block_lm_summary"]

    assert result["predicted_deltas"][1, 0] == pytest.approx(-0.2)
    assert summary["status_counts"]["converged"] == 1
    assert summary["converged_detail"] == {"interior": 0, "boundary": 1}
    assert summary["boundary"]["free_set_empty_blocks"] == 1
    assert summary["boundary"]["frozen_dof_total"] == 1
    # A boundary block contributes lambda2 = 0 by construction, so it must stay
    # out of the interior decrement quantiles.
    assert summary["decrement_per_component"]["n_blocks"] == 0


def test_block_lm_blocks_are_decoupled_across_target_sensors():
    """The premise behind retiring a converged block for good.

    ``_sensor_residual_data`` selects hits by sensor id and the helices are
    frozen, so one block's residuals cannot depend on another block's alignment
    -- which is why another block moving can never pull a stationary one off its
    optimum.  Perturbing the second target's measurements must leave the first
    target's correction bit-for-bit unchanged.
    """
    m = load_ml_alignment_autograd_module()
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=12,
        lambda_l2=0.0, make_plot=False,
    )
    corrections = []
    for perturbation in (0.0, 7.0):
        batches, geometry, truth = _analytic_multi_dof_case(m)
        # Make the second-to-last plane a target as well, then move only its
        # measurements.
        geometry.target_mask[-2] = True
        batches.is_target[:, -2] = True
        batches.fit_mask[:, -2] = False
        batches.measured_local[:, -2, :] += perturbation
        result = m.run_block_lm_alignment(batches, geometry, config, (0, 1, 5), truth)
        assert result["block_lm_summary"]["blocks_total"] == 2
        corrections.append(result["predicted_deltas"][-1].copy())

    np.testing.assert_array_equal(corrections[0], corrections[1])


def test_block_lm_reports_hits_with_no_valid_intersection_as_intersection_failed(monkeypatch):
    """A target sensor with hits must not disappear into the no-data denominator exclusion."""
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _analytic_multi_dof_case(m)
    failed_sensor = len(geometry.centers) - 2
    geometry.target_mask[failed_sensor] = True
    batches.is_target[:, failed_sensor] = True
    batches.fit_mask[:, failed_sensor] = False
    failed_center = geometry.centers[failed_sensor].double()
    genuine = m.intersect_plane

    def fail_one_sensor(track_parameters, centers, normals, n_newton=12):
        path, intersections, valid = genuine(track_parameters, centers, normals, n_newton)
        if valid.ndim == 1 and torch.allclose(centers[0].double(), failed_center):
            valid = torch.zeros_like(valid)
        return path, intersections, valid

    monkeypatch.setattr(m, "intersect_plane", fail_one_sensor)
    config = m.AlignmentRunConfig(
        track_mode="free",
        joint_solver="block-lm",
        lm_max_iters=20,
        lambda_l2=0.0,
        make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0, 1, 5), tracks)
    summary = result["block_lm_summary"]
    failed = next(
        block for block in result["block_lm_blocks"]
        if block["sensor_index"] == failed_sensor
    )

    assert failed["status"] == m.BLOCK_LM_STATUS_INTERSECTION_FAILED
    assert summary["blocks_without_data"] == 0
    assert summary["status_counts"]["intersection_failed"] == 1
