from __future__ import annotations

import importlib.util
import json
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
    """The two modules must agree on the objective before the decrement means anything.

    ``_measurement_refit_objective`` divides by the component count while the
    pre-registered ``NEWTON_DECREMENT_STATIONARY`` lives in ``F = 0.5 * chi2``.
    Undoing that normalisation has to land exactly on the validation module's
    standardized residual, or the reported decrement is off by the component
    count and the threshold is meaningless.
    """
    m = load_ml_alignment_autograd_module()
    v = load_validate_helix_model_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    loss, _invalid, component_mask = m._measurement_refit_objective(
        seed, batches, geometry
    )
    autograd_half_chi2 = float(loss) * float(component_mask.sum()) / 2.0

    validation_batches = m._validation_batches_from_joint(batches, v)
    inputs = v._measurement_fit_inputs(validation_batches, geometry, seed)
    no_kinks = inputs.node_s.new_zeros((seed.shape[0], 0, 2))
    residual = v._standardized_residual(
        v._pack_parameters(seed, no_kinks),
        0,
        v.reference_gauge_frame(seed),
        inputs.node_s,
        inputs.node_valid,
        inputs.kink_sigma,
        inputs.measured,
        inputs.variance,
        inputs.mask,
        inputs.hit_s,
        inputs.centers,
        inputs.rotations,
    )
    validation_half_chi2 = float(0.5 * residual.square().sum())

    assert validation_half_chi2 > 0.0
    assert abs(autograd_half_chi2 - validation_half_chi2) <= 1e-9 * validation_half_chi2


def test_validation_batches_from_joint_relabels_without_reinterpreting():
    m = load_ml_alignment_autograd_module()
    v = load_validate_helix_model_module()
    batches, _geometry, _truth, _seed = build_synthetic_joint_case(m)

    relabelled = m._validation_batches_from_joint(batches, v)

    assert torch.equal(relabelled.measured_local, batches.measured_local)
    assert torch.equal(relabelled.measurement_variance, batches.covariances)
    assert torch.equal(relabelled.loc1_valid, batches.loc1_mask)
    assert torch.equal(relabelled.fit_mask, batches.fit_mask)
    assert torch.equal(relabelled.sensor_idx, batches.sensor_idx)
    assert torch.equal(relabelled.g_smt, batches.g_smt)
    # The stand-in holdout never overlaps the target layer, and the sentinel mode
    # keeps these rows out of every Gate 4 selection.
    assert not bool((relabelled.holdout_mask & batches.is_target).any())
    assert bool((relabelled.holdout_mask & relabelled.fit_mask).any(dim=1).all())
    assert set(relabelled.prediction_mode) == {m.DIAGNOSTIC_PREDICTION_MODE}
    assert m.DIAGNOSTIC_PREDICTION_MODE not in v.PREDICTION_MODES


def test_validation_batches_from_joint_fails_closed_on_target_leakage():
    m = load_ml_alignment_autograd_module()
    v = load_validate_helix_model_module()
    batches, _geometry, _truth, _seed = build_synthetic_joint_case(m)
    batches.fit_mask = torch.ones_like(batches.fit_mask)

    with pytest.raises(ValueError, match="target-layer hits"):
        m._validation_batches_from_joint(batches, v)


def test_refit_stationarity_measurement_leaves_the_fit_untouched():
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    measured_fit, measured_stats = m.fit_helices_to_measurements(
        batches, geometry, seed, n_iter=20, stationarity=True, solver="adam"
    )
    plain_fit, plain_stats = m.fit_helices_to_measurements(
        batches, geometry, seed, n_iter=20, stationarity=False, solver="adam"
    )

    assert torch.equal(measured_fit, plain_fit)
    assert "stationarity" not in plain_stats
    assert set(measured_stats["stationarity"]) == {"seed", "refit"}
    assert (
        measured_stats["chi2_per_component"] == plain_stats["chi2_per_component"]
    )


def test_refit_stationarity_separates_a_truncated_fit_from_a_solved_one():
    """Both directions, so a decrement that is always zero cannot pass.

    A single Adam step must look worse than twenty, and the Gauss-Newton
    solution of the same objective must come back stationary.  Without the
    second half this test would still pass if the measurement collapsed to a
    constant.
    """
    m = load_ml_alignment_autograd_module()
    v = load_validate_helix_model_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    _one, one_step = m.fit_helices_to_measurements(
        batches, geometry, seed, n_iter=1, stationarity=True, solver="adam"
    )
    _many, many_steps = m.fit_helices_to_measurements(
        batches, geometry, seed, n_iter=20, stationarity=True, solver="adam"
    )
    assert (
        many_steps["stationarity"]["refit"]["newton_decrement_p50"]
        < one_step["stationarity"]["refit"]["newton_decrement_p50"]
    )

    solved, _stats = v.fit_helices_gauss_newton(
        m._validation_batches_from_joint(batches, v), geometry, seed
    )
    solved_report = m.measure_joint_refit_stationarity(batches, geometry, solved)
    assert solved_report["non_stationary_row_fraction"] == 0.0
    assert (
        solved_report["newton_decrement_max"]
        < solved_report["stationary_threshold"]
    )


def test_gauss_newton_is_the_default_joint_refit_and_meets_the_criterion():
    """Iteration 19: the default path must land on the pre-registered criterion.

    The Adam keys must be absent rather than carrying a placeholder, so a reader
    of ``initialization_stats`` cannot mistake a Gauss-Newton run for one whose
    checkpoint happened never to improve.
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
    v = load_validate_helix_model_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    class TruncatedSolver:
        def __getattr__(self, name):
            return getattr(v, name)

        @staticmethod
        def fit_helices_gauss_newton(*args, **kwargs):
            return v.fit_helices_gauss_newton(*args, **{**kwargs, "max_iters": 1})

    monkeypatch.setattr(m, "_load_validation_module", TruncatedSolver)
    with pytest.raises(ValueError, match="pre-registered stationary point"):
        m.fit_helices_to_measurements(batches, geometry, seed)

    monkeypatch.undo()
    _fitted, stats = m.fit_helices_to_measurements(batches, geometry, seed)
    assert stats["stationarity_gate"]["non_stationary_row_fraction"] == 0.0


def test_gauss_newton_refit_improves_on_adam_at_the_same_component_count():
    """The swap changes the optimiser and nothing else.

    ``n_components`` is the first thing checked: the two paths share a component
    mask by construction, so a difference there would be a wiring bug, and a chi2
    comparison across different component counts would be meaningless.
    """
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    _adam_fit, adam_stats = m.fit_helices_to_measurements(
        batches, geometry, seed, n_iter=20, solver="adam"
    )
    _gn_fit, gn_stats = m.fit_helices_to_measurements(
        batches, geometry, seed, solver="gauss-newton"
    )

    assert gn_stats["n_components"] == adam_stats["n_components"]
    assert (
        gn_stats["initial_chi2_per_component"]
        == adam_stats["initial_chi2_per_component"]
    )
    assert gn_stats["chi2_per_component"] <= adam_stats["chi2_per_component"]
    assert (
        gn_stats["stationarity"]["refit"]["non_stationary_row_fraction"]
        < adam_stats["stationarity"]["refit"]["non_stationary_row_fraction"]
    )


def test_unknown_joint_refit_solver_is_rejected():
    m = load_ml_alignment_autograd_module()
    batches, geometry, _truth, seed = build_synthetic_joint_case(m)

    with pytest.raises(ValueError, match="Unknown joint measurement refit solver"):
        m.fit_helices_to_measurements(batches, geometry, seed, solver="lbfgs")


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


def test_block_lm_audit_reports_the_closed_form_curvature_and_convergence():
    m = load_ml_alignment_autograd_module()
    batches, geometry, tracks = _block_lm_translation_case(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=8,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(batches, geometry, config, (0,), tracks)
    summary = result["block_lm_summary"]

    assert summary["arm"] == "audit-only"
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


def test_block_lm_audit_separates_budget_exhaustion_from_convergence():
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


def test_block_lm_audit_records_the_fixed_set_alongside_the_moving_mask():
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


def test_block_lm_audit_summary_survives_the_json_round_trip():
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

    assert written["arm"] == "audit-only"
    assert set(written["status_counts"]) == set(m.BLOCK_LM_TERMINAL_STATUSES)
    # Per-sensor detail stays in memory only.
    assert "blocks" not in written
