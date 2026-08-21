from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_module(module_name: str, filename: str):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def synthetic_zero_alignment_inputs():
    transforms = [
        {
            "ID": "vol=1|lay=1|sen=1",
            "Translation": [0.0, 0.0, 0.0],
            "RotationMatrix": np.eye(3).tolist(),
        },
        {
            "ID": "vol=1|lay=1|sen=2",
            "Translation": [10.0, -4.0, 3.0],
            "RotationMatrix": rotation_matrix(0.17, -0.22, 0.35).tolist(),
        },
    ]
    centers = np.array([item["Translation"] for item in transforms], dtype=np.float64)
    rotations = np.array([item["RotationMatrix"] for item in transforms], dtype=np.float64)

    hit_specs = [
        (0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        (0, [0.0, 0.0, 0.0], [2.0 ** -0.5, 0.0, 2.0 ** -0.5]),
        (0, [0.2, -0.4, 0.0], [0.2, -0.1, 1.0]),
        (1, [2.0, -1.0, 0.0], [0.3, 0.2, 1.0]),
        (1, [-0.5, 0.8, 0.0], [-0.2, 0.4, 1.0]),
    ]

    sensor_ids = []
    global_positions = []
    global_directions = []
    for sensor_idx, local_position, local_direction in hit_specs:
        rotation = rotations[sensor_idx]
        center = centers[sensor_idx]
        sensor_ids.append(sensor_idx)
        global_positions.append(center + rotation @ np.array(local_position, dtype=np.float64))
        global_directions.append(rotation @ np.array(local_direction, dtype=np.float64))

    return (
        transforms,
        torch.tensor(sensor_ids, dtype=torch.long),
        torch.tensor(np.array(global_positions), dtype=torch.float32),
        torch.tensor(np.array(global_directions), dtype=torch.float32),
        torch.ones((len(sensor_ids), 1), dtype=torch.float32),
        torch.tensor(centers, dtype=torch.float64),
        torch.tensor(rotations, dtype=torch.float64),
    )


def autograd_residual_jacobian(
    ml_alignment_autograd,
    hit_sensor_ids,
    hit_global_positions,
    hit_directions,
    sensor_centers,
    sensor_rotations,
):
    num_sensors = int(hit_sensor_ids.max().item()) + 1
    num_hits = len(hit_sensor_ids)
    alignment_params = torch.zeros(
        (num_sensors, 6), dtype=torch.float64, requires_grad=True
    )
    measured_local = torch.zeros((num_hits, 2), dtype=torch.float64)

    residuals = ml_alignment_autograd.compute_autograd_residuals(
        measured_local,
        hit_global_positions.double(),
        hit_directions.double(),
        hit_sensor_ids,
        sensor_centers,
        sensor_rotations,
        alignment_params,
    )

    jacobian = torch.zeros((num_hits, 2, 6), dtype=torch.float64)
    for hit_idx in range(num_hits):
        sensor_idx = int(hit_sensor_ids[hit_idx].item())
        for residual_idx in range(2):
            grad = torch.autograd.grad(
                residuals[hit_idx, residual_idx],
                alignment_params,
                retain_graph=True,
            )[0]
            jacobian[hit_idx, residual_idx] = grad[sensor_idx]
    return jacobian


def test_cpp_and_autograd_residual_jacobians_match_at_zero_alignment():
    ml_alignment = load_module("ml_alignment", "ml_alignment.py")
    ml_alignment_autograd = load_module("ml_alignment_autograd", "ml_alignment_autograd.py")
    (
        transforms,
        hit_sensor_ids,
        hit_global_positions,
        hit_directions,
        hit_qoverp,
        sensor_centers,
        sensor_rotations,
    ) = synthetic_zero_alignment_inputs()

    try:
        cpp_jacobian = ml_alignment.compute_jacobians_from_alignderivlib(
            hit_sensor_ids,
            hit_global_positions,
            hit_directions,
            hit_qoverp,
            transforms,
            current_deltas=None,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"AlignDerivLib pybind extension unavailable: {exc}")

    autograd_jacobian = autograd_residual_jacobian(
        ml_alignment_autograd,
        hit_sensor_ids,
        hit_global_positions,
        hit_directions,
        sensor_centers,
        sensor_rotations,
    )

    diff = autograd_jacobian.float() - cpp_jacobian
    abs_diff = diff.abs()
    rel_diff = abs_diff / cpp_jacobian.abs().clamp_min(1e-8)
    flat_idx = int(abs_diff.argmax().item())
    hit_idx, residual_idx, dof_idx = np.unravel_index(flat_idx, abs_diff.shape)

    print(f"max_abs={abs_diff.max().item():.6g}")
    print(f"mean_abs={abs_diff.mean().item():.6g}")
    print(f"max_rel={rel_diff.max().item():.6g}")

    assert torch.allclose(
        autograd_jacobian.float(),
        cpp_jacobian,
        atol=1e-4,
        rtol=1e-3,
    ), (
        "C++ and autograd residual Jacobians differ at alignment_params=0. "
        f"max_abs={abs_diff.max().item():.6g}, "
        f"mean_abs={abs_diff.mean().item():.6g}, "
        f"max_rel={rel_diff.max().item():.6g}, "
        f"worst=(hit={hit_idx}, residual={residual_idx}, dof={dof_idx}), "
        f"cpp={cpp_jacobian[hit_idx, residual_idx, dof_idx].item():.6g}, "
        f"autograd={autograd_jacobian[hit_idx, residual_idx, dof_idx].item():.6g}"
    )
