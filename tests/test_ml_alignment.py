from __future__ import annotations

import importlib.util
import importlib.machinery
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def load_ml_alignment_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("ml_alignment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_comparison_config_is_not_exposed():
    ml_alignment = load_ml_alignment_module()

    assert not hasattr(ml_alignment.AlignmentRunConfig(), "reference_file")


def test_reference_comparison_cli_is_not_exposed():
    script = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--reference" not in completed.stdout


def test_add_align_derivlib_to_path_uses_current_python_extension_suffix(
    monkeypatch, tmp_path
):
    ml_alignment = load_ml_alignment_module()

    align_dir = tmp_path / "AlignDerivLib"
    incompatible_dir = align_dir / "build_incompatible"
    compatible_dir = align_dir / "cmake-out"
    incompatible_dir.mkdir(parents=True)
    compatible_dir.mkdir(parents=True)

    incompatible_module = (
        incompatible_dir / "align_deriv_pb.cpython-999-x86_64-linux-gnu.so"
    )
    incompatible_module.touch()

    active_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    compatible_module = compatible_dir / f"align_deriv_pb{active_suffix}"
    compatible_module.touch()

    original_sys_path = list(sys.path)
    monkeypatch.setattr(sys, "path", original_sys_path.copy())

    ml_alignment.add_align_derivlib_to_path(align_dir)

    assert str(compatible_dir) in sys.path
    assert str(incompatible_dir) not in sys.path


def test_per_layer_residual_diagnostics_prints_before_after_mean(capsys):
    ml_alignment = load_ml_alignment_module()

    init_res = np.array(
        [
            [1.0, 4.0],
            [3.0, 6.0],
            [10.0, 20.0],
        ],
        dtype=float,
    )
    final_res = np.array(
        [
            [0.0, 2.0],
            [2.0, 4.0],
            [8.0, 18.0],
        ],
        dtype=float,
    )
    vols = np.array([17, 17, 24])
    lays = np.array([8, 8, 4])

    ml_alignment.print_per_layer_residual_diagnostics(
        init_res, final_res, vols, lays, ((17, 8), (24, 4))
    )

    output = capsys.readouterr().out

    assert "=== Per-layer residual mean (training data) ===" in output
    assert re.search(
        r"vol=17 lay=8 \(n=2\)\s+2\.0000mm\s+1\.0000mm\s+5\.0000mm\s+3\.0000mm",
        output,
    )
    assert "=== Per-layer residual RMS (training data) ===" in output


def test_debug_tensor_summary_reports_shape_dtype_and_invalid_counts(capsys):
    ml_alignment = load_ml_alignment_module()

    values = torch.tensor([[1.0, float("nan")], [float("inf"), 4.0]])

    ml_alignment.debug_tensor_summary("sample_tensor", values, enabled=True)

    output = capsys.readouterr().out

    assert "[debug:dataflow] sample_tensor" in output
    assert "shape=(2, 2)" in output
    assert "dtype=torch.float32" in output
    assert "nan=1" in output
    assert "inf=1" in output
    assert "finite_min=1.000000" in output
    assert "finite_max=4.000000" in output


def test_debug_layer_distribution_is_silent_when_disabled(capsys):
    ml_alignment = load_ml_alignment_module()

    ml_alignment.debug_layer_distribution(
        "layers",
        torch.tensor([17, 17, 24], dtype=torch.long),
        torch.tensor([8, 8, 4], dtype=torch.long),
        enabled=False,
    )

    assert capsys.readouterr().out == ""


def test_debug_layer_distribution_prints_group_counts(capsys):
    ml_alignment = load_ml_alignment_module()

    ml_alignment.debug_layer_distribution(
        "layers",
        torch.tensor([17, 17, 24], dtype=torch.long),
        torch.tensor([8, 8, 4], dtype=torch.long),
        enabled=True,
    )

    output = capsys.readouterr().out

    assert "[debug:dataflow] layers layer distribution" in output
    assert "vol=17 lay=8 count=2" in output
    assert "vol=24 lay=4 count=1" in output


def test_pipeline_computes_jacobians_only_for_target_layer_hits(monkeypatch):
    ml_alignment = load_ml_alignment_module()

    monkeypatch.setattr(ml_alignment.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        ml_alignment,
        "load_geometry_from_json",
        lambda _path: (
            [{"ID": "s0"}, {"ID": "s1"}, {"ID": "s2"}],
            3,
            torch.zeros((3, 3), dtype=torch.float32),
            {(17, 8, 0): 0, (18, 8, 0): 1, (24, 4, 0): 2},
        ),
    )
    monkeypatch.setattr(
        ml_alignment,
        "load_variable_information",
        lambda *_args: ml_alignment.AlignmentInputs(
            hit_sensor_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            initial_residuals=torch.tensor(
                [[1.0, 0.5], [8.0, 4.0], [2.0, 1.0]], dtype=torch.float32
            ),
            covariances=torch.ones((3, 2), dtype=torch.float32),
            track_params=torch.zeros((3, 5), dtype=torch.float32),
            track_chi2_ndf=torch.ones((3, 1), dtype=torch.float32),
            hit_global_positions=torch.zeros((3, 3), dtype=torch.float32),
            hit_directions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            hit_qoverp=torch.ones((3, 1), dtype=torch.float32),
            hit_volume_ids=torch.tensor([17, 18, 24], dtype=torch.long),
            hit_layer_ids=torch.tensor([8, 8, 4], dtype=torch.long),
            hit_module_ids=torch.tensor([0, 0, 0], dtype=torch.long),
            hit_event_ids=torch.tensor([0, 0, 0], dtype=torch.long),
            hit_track_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        ),
    )

    jacobian_hit_counts = []

    def fake_compute_jacobians(hit_sensor_ids, *_args, **_kwargs):
        jacobian_hit_counts.append(len(hit_sensor_ids))
        return torch.ones((len(hit_sensor_ids), 2, 6), dtype=torch.float32)

    monkeypatch.setattr(
        ml_alignment, "compute_jacobians_from_alignderivlib", fake_compute_jacobians
    )
    monkeypatch.setattr(ml_alignment, "validate_against_truth", lambda *_args: None)
    monkeypatch.setattr(ml_alignment, "update_geometry_to_json", lambda *_args: None)

    ml_alignment.run_alignment_pipeline(
        ml_alignment.AlignmentRunConfig(
            epochs=1,
            make_plot=False,
            target_layers=((17, 8), (24, 4)),
        )
    )

    assert jacobian_hit_counts == [2]


def test_pipeline_debug_dataflow_prints_major_dataflow_sections(monkeypatch, capsys):
    ml_alignment = load_ml_alignment_module()

    monkeypatch.setattr(ml_alignment.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        ml_alignment,
        "load_geometry_from_json",
        lambda _path: (
            [
                {
                    "ID": "vol=17|lay=8|sen=0",
                    "Translation": [1.0, 2.0, 3.0],
                    "RotationMatrix": np.eye(3).tolist(),
                },
                {
                    "ID": "vol=24|lay=4|sen=0",
                    "Translation": [4.0, 5.0, 6.0],
                    "RotationMatrix": np.eye(3).tolist(),
                },
            ],
            2,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
            {(17, 8, 0): 0, (24, 4, 0): 1},
        ),
    )
    monkeypatch.setattr(
        ml_alignment,
        "load_variable_information",
        lambda *_args, **_kwargs: ml_alignment.AlignmentInputs(
            hit_sensor_ids=torch.tensor([0, 1], dtype=torch.long),
            initial_residuals=torch.tensor([[1.0, 0.5], [2.0, 1.0]], dtype=torch.float32),
            covariances=torch.ones((2, 2), dtype=torch.float32),
            track_params=torch.zeros((2, 5), dtype=torch.float32),
            track_chi2_ndf=torch.ones((2, 1), dtype=torch.float32),
            hit_global_positions=torch.zeros((2, 3), dtype=torch.float32),
            hit_directions=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
            ),
            hit_qoverp=torch.ones((2, 1), dtype=torch.float32),
            hit_volume_ids=torch.tensor([17, 24], dtype=torch.long),
            hit_layer_ids=torch.tensor([8, 4], dtype=torch.long),
            hit_module_ids=torch.tensor([0, 0], dtype=torch.long),
            hit_event_ids=torch.tensor([0, 0], dtype=torch.long),
            hit_track_ids=torch.tensor([0, 1], dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        ml_alignment,
        "compute_jacobians_from_alignderivlib",
        lambda hit_sensor_ids, *_args, **_kwargs: torch.ones(
            (len(hit_sensor_ids), 2, 6), dtype=torch.float32
        ),
    )
    monkeypatch.setattr(ml_alignment, "validate_against_truth", lambda *_args: None)
    monkeypatch.setattr(ml_alignment, "update_geometry_to_json", lambda *_args: None)

    ml_alignment.run_alignment_pipeline(
        ml_alignment.AlignmentRunConfig(
            epochs=1,
            make_plot=False,
            debug_dataflow=True,
            target_layers=((17, 8), (24, 4)),
        )
    )

    output = capsys.readouterr().out

    assert "[debug:dataflow] geometry.initial_positions" in output
    assert "[debug:dataflow] inputs.initial_residuals" in output
    assert "[debug:dataflow] training layer distribution" in output
    assert "[debug:dataflow] jacobian.initial" in output
    assert "[debug:dataflow] predicted_deltas" in output


def test_physics_loss_normalizes_hit_weights_per_target_layer():
    ml_alignment = load_ml_alignment_module()

    initial_residuals = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [10.0, 10.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    shift_effect = torch.zeros_like(initial_residuals)
    covariances = torch.ones_like(initial_residuals)
    hit_weights = torch.tensor([[1.0], [3.0], [10.0], [30.0]], dtype=torch.float32)
    hit_volume_ids = torch.tensor([17, 17, 24, 24], dtype=torch.long)
    hit_layer_ids = torch.tensor([8, 8, 4, 4], dtype=torch.long)

    loss = ml_alignment.physics_informed_loss(
        initial_residuals,
        shift_effect,
        covariances,
        hit_weights,
        hit_volume_ids=hit_volume_ids,
        hit_layer_ids=hit_layer_ids,
    )

    expected = torch.tensor(50.5, dtype=torch.float32)
    assert torch.isclose(loss, expected)


def test_physics_loss_keeps_global_weight_normalization_without_layer_ids():
    ml_alignment = load_ml_alignment_module()

    initial_residuals = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [10.0, 10.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    shift_effect = torch.zeros_like(initial_residuals)
    covariances = torch.ones_like(initial_residuals)
    hit_weights = torch.tensor([[1.0], [3.0], [10.0], [30.0]], dtype=torch.float32)

    loss = ml_alignment.physics_informed_loss(
        initial_residuals, shift_effect, covariances, hit_weights
    )

    global_mean = hit_weights.mean()
    expected = torch.mean((hit_weights / global_mean) * initial_residuals.square())
    assert torch.isclose(loss, expected)
