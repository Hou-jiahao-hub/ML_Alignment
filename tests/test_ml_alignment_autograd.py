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
