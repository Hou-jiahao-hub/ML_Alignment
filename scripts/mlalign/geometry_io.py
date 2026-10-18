"""Geometry JSON input/output and the full working geometry of free mode."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .conventions import parse_target_layers
from .rotations import euler_to_rotation_matrix

FULL_MATRIX_TOLERANCE = 1e-4


@dataclass
class JointGeometry:
    transforms: list[dict]
    input_transforms: list[dict]
    centers: torch.Tensor
    rotations: torch.Tensor
    target_mask: torch.Tensor
    id_to_idx: dict[tuple[int, int, int], int]
    input_indices: list[int]


def _surface_key(item):
    match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", item.get("ID", ""))
    if match is None:
        raise ValueError(f"Surface ID has no vol/lay/sen fields: {item.get('ID', '')!r}")
    return tuple(map(int, match.groups()))


def load_joint_geometry(nominal_path, input_path, target_layers):
    """Build full working geometry and overlay the input target surfaces."""
    with open(nominal_path, "r", encoding="utf-8") as stream:
        nominal = json.load(stream)
    with open(input_path, "r", encoding="utf-8") as stream:
        input_transforms = json.load(stream)
    transforms = json.loads(json.dumps(nominal))
    id_to_idx = {_surface_key(item): index for index, item in enumerate(transforms)}
    target_layers = set(parse_target_layers(target_layers))
    target_mask = torch.zeros(len(transforms), dtype=torch.bool)
    for key, index in id_to_idx.items():
        target_mask[index] = key[:2] in target_layers

    input_indices = []
    for item in input_transforms:
        key = _surface_key(item)
        if key not in id_to_idx:
            raise KeyError(f"Input surface {key} is absent from nominal geometry.")
        if key[:2] not in target_layers:
            raise ValueError(f"Input surface {key} is outside configured target layers.")
        index = id_to_idx[key]
        transforms[index] = json.loads(json.dumps(item))
        input_indices.append(index)

    centers = torch.tensor([item["Translation"] for item in transforms], dtype=torch.float64)
    rotations = torch.tensor([item["RotationMatrix"] for item in transforms], dtype=torch.float64)
    return JointGeometry(
        transforms=transforms,
        input_transforms=json.loads(json.dumps(input_transforms)),
        centers=centers,
        rotations=rotations,
        target_mask=target_mask,
        id_to_idx=id_to_idx,
        input_indices=input_indices,
    )


def export_joint_geometry(geometry, predicted_deltas, output_path):
    """Apply full-geometry corrections while emitting only input target surfaces."""
    input_deltas = np.asarray(predicted_deltas)[geometry.input_indices]
    transforms = json.loads(json.dumps(geometry.input_transforms))
    update_geometry_to_json(transforms, input_deltas, output_path)


def load_geometry_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data or "Translation" not in data[0]:
        raise ValueError(f"The format of {filepath} is incorrect. Please check the JSON structure.")

    transforms_list = data
    num_sensors = len(transforms_list)

    positions = []
    rotations = []
    id_to_idx = {}
    for i, item in enumerate(transforms_list):
        positions.append(item["Translation"])
        rotations.append(item["RotationMatrix"])
        match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", item.get("ID", ""))
        if match:
            vol, lay, sen = map(int, match.groups())
            id_to_idx[(vol, lay, sen)] = i

    return (
        transforms_list,
        num_sensors,
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(rotations, dtype=torch.float32),
        id_to_idx,
    )


def update_geometry_to_json(
    transforms_list, predicted_deltas, output_filepath, alignment_gain=1.0
):
    if not np.isfinite(alignment_gain) or alignment_gain < 0:
        raise ValueError("alignment_gain must be a finite non-negative value.")
    for i, item in enumerate(transforms_list):
        #The delta transform
        delta_t_local = predicted_deltas[i, :3] * alignment_gain
        delta_euler = predicted_deltas[i, 3:] * alignment_gain

        # The original transform
        old_t = np.array(item["Translation"])
        old_R = np.array(item["RotationMatrix"])  # (3,3)
        #The new transform
        new_t = old_t + old_R @ delta_t_local
        item["Translation"] = new_t.tolist()

        alpha, beta, gamma = delta_euler
        R_delta = euler_to_rotation_matrix(alpha, beta, gamma)

        new_R = old_R @ R_delta
        item["RotationMatrix"] = new_R.tolist()

        fm = np.eye(4)
        fm[:3, :3] = new_R
        fm[:3, 3] = new_t

        if "FullMatrix" in item:
            old_fm = item["FullMatrix"]
            if len(old_fm) == 16:
                if abs(old_fm[12] - old_t[0]) < FULL_MATRIX_TOLERANCE:
                    item["FullMatrix"] = fm.flatten('F').tolist()
                else:
                    item["FullMatrix"] = fm.flatten("C").tolist()
            elif np.asarray(old_fm).shape == (4, 4):
                item["FullMatrix"] = fm.tolist()
        else:
            item["FullMatrix"] = fm.tolist()

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f_out:
        json.dump(transforms_list, f_out, indent=4)


def build_target_mask(num_sensors, id_to_idx_map, target_layers):
    target_layers = set(parse_target_layers(target_layers))
    target_mask = torch.zeros(num_sensors, dtype=torch.bool)
    for (vol, lay, _sen), idx in id_to_idx_map.items():
        if (vol, lay) in target_layers:
            target_mask[idx] = True
    return target_mask
