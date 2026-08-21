"""Frozen-helix alignment for surfaces that stall in the surface solver.

The target layers are held out from both the SMT seed fit and the measurement
refit.  Once fitted, every helix is detached permanently and only selected
surface transforms are optimized against target-layer measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
import torch
import torch.optim as optim
import uproot

from alignment_artifact_layout import (
    TARGET_LAYERS,
    alignment_json_path,
    data_path,
    reconstruction_root_path,
    target_layers_spec,
)
from project_config import add_config_argument


PIXEL_VOLUMES = {16, 17, 18}
SHORT_STRIP_VOLUMES = {23, 24, 25}
LONG_STRIP_VOLUMES = {28, 29, 30}
SHORT_STRIP_LOC1_VARIANCE_FLOOR = 1.2**2
EPS_VARIANCE = 1e-8
EPS_NORM = 1e-9
PLANE_PARALLEL_EPS = 1e-6
PLANE_RESIDUAL_TOLERANCE = 1e-3
S_MAX = 1200.0
FULL_MATRIX_TOLERANCE = 1e-4


@dataclass
class FlatStates:
    sensor_idx: torch.Tensor
    measured_local: torch.Tensor
    variance: torch.Tensor
    loc1_valid: torch.Tensor
    smt_local: torch.Tensor
    g_smt: torch.Tensor
    p_smt: torch.Tensor
    volume_id: torch.Tensor
    layer_id: torch.Tensor
    module_id: torch.Tensor
    event_id: torch.Tensor
    track_id: torch.Tensor
    path_length: torch.Tensor


@dataclass
class TrackWindows:
    measured_local: torch.Tensor
    variance: torch.Tensor
    loc1_valid: torch.Tensor
    smt_local: torch.Tensor
    sensor_idx: torch.Tensor
    hit_mask: torch.Tensor
    fit_mask: torch.Tensor
    holdout_mask: torch.Tensor
    g_smt: torch.Tensor
    p_smt: torch.Tensor
    volume_id: torch.Tensor
    layer_id: torch.Tensor
    event_track_ids: torch.Tensor
    fit_span: torch.Tensor


@dataclass
class WorkingGeometry:
    transforms: list[dict]
    input_transforms: list[dict]
    centers: torch.Tensor
    rotations: torch.Tensor
    id_to_idx: dict[tuple[int, int, int], int]
    input_indices: list[int]


def parse_layers(value: str | Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, str):
        layers = tuple((int(v), int(l)) for v, l in value)
    else:
        layers = tuple(
            tuple(map(int, item.split(":")))
            for item in value.split(",")
            if item.strip()
        )
    if not layers or any(len(layer) != 2 for layer in layers):
        raise ValueError("target layers must use volume:layer syntax")
    return layers


def parse_active_dof(value: str | Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(item) for item in (value.split(",") if isinstance(value, str) else value))
    if not values or len(set(values)) != len(values) or any(item < 0 or item > 5 for item in values):
        raise ValueError("active DOF must be unique integers in [0, 5]")
    return values


def apply_measurement_variance_policy(
    variance: torch.Tensor,
    volume_id: torch.Tensor,
    loc1_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return physical measurement variances and per-component validity.

    ShortStrip LOC1 is a real, coarse measurement and receives the documented
    1.2 mm resolution floor.  LongStrip LOC1 follows the ROOT per-hit validity
    instead of being discarded merely because it belongs to a strip volume.
    """
    adjusted = variance.clone()
    short = torch.zeros_like(loc1_valid, dtype=torch.bool)
    for volume in SHORT_STRIP_VOLUMES:
        short |= volume_id == volume
    short_loc1 = short & loc1_valid
    adjusted[..., 1] = torch.where(
        short_loc1,
        torch.clamp(adjusted[..., 1], min=SHORT_STRIP_LOC1_VARIANCE_FLOOR),
        adjusted[..., 1],
    )
    component_mask = torch.stack((torch.ones_like(loc1_valid), loc1_valid), dim=-1)
    return adjusted, component_mask


def surface_stop_reason(
    chi2_history: Iterable[float],
    *,
    chi2_target: float = 1.0,
    relative_tolerance: float = 0.05,
    patience: int = 3,
    max_iterations: int = 15,
) -> str | None:
    history = [float(value) for value in chi2_history]
    if not history:
        return None
    if history[-1] <= chi2_target:
        return "chi2_target"
    if len(history) >= patience + 1:
        changes = [
            abs(new - old) / max(abs(old), 1e-12)
            for old, new in zip(history[-patience - 1 : -1], history[-patience:])
        ]
        if all(change < relative_tolerance for change in changes):
            return "relative_plateau"
    if len(history) >= max_iterations:
        return "max_iterations"
    return None


def _components_for_volume(volume: int) -> tuple[str, ...]:
    if volume in PIXEL_VOLUMES or volume in SHORT_STRIP_VOLUMES:
        return ("loc0", "loc1")
    if volume in LONG_STRIP_VOLUMES:
        return ("loc0", "loc1")  # LOC1 is included only when the metric exists.
    raise ValueError(f"unsupported detector volume {volume}")


def select_failed_layers(
    nominal: dict[str, dict[str, float | None]],
    candidate: dict[str, dict[str, float | None]],
    candidates: str | Iterable[tuple[int, int]],
    *,
    max_ratio: float = 1.2,
) -> tuple[tuple[tuple[int, int], ...], dict]:
    failed_layers: list[tuple[int, int]] = []
    report: dict[str, dict] = {}
    for volume, layer in parse_layers(candidates):
        key = f"{volume}:{layer}"
        if key not in nominal or key not in candidate:
            raise KeyError(f"missing layer metrics for {key}")
        layer_failed = False
        component_report = {}
        for component in _components_for_volume(volume):
            metric = f"{component}_rmse_mm"
            reference = nominal[key].get(metric)
            value = candidate[key].get(metric)
            if reference is None or value is None:
                component_report[component] = {
                    "nominal_rmse_mm": reference,
                    "candidate_rmse_mm": value,
                    "ratio": None,
                    "failed": False,
                    "valid": False,
                }
                continue
            ratio = float(value) / max(float(reference), 1e-12)
            component_failed = ratio > max_ratio
            layer_failed |= component_failed
            component_report[component] = {
                "nominal_rmse_mm": float(reference),
                "candidate_rmse_mm": float(value),
                "ratio": ratio,
                "failed": component_failed,
                "valid": True,
            }
        report[key] = component_report
        if layer_failed:
            failed_layers.append((volume, layer))
    return tuple(failed_layers), report


def _sinc(value: torch.Tensor) -> torch.Tensor:
    return torch.sinc(value / torch.pi)


def helix_point(track: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    x0, y0, z0, psi0, tanl, rho = track.unbind(-1)
    angle = rho * path
    half_sinc = _sinc(angle / 2.0)
    x = x0 + path * torch.cos(psi0 + angle / 2.0) * half_sinc
    y = y0 + path * torch.sin(psi0 + angle / 2.0) * half_sinc
    z = z0 + tanl * path
    return torch.stack((x, y, z), dim=-1)


def helix_tangent(track: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
    _x0, _y0, _z0, psi0, tanl, rho = track.unbind(-1)
    psi = psi0 + rho * path
    return torch.stack((torch.cos(psi), torch.sin(psi), tanl * torch.ones_like(path)), dim=-1)


def intersect_plane(
    track: torch.Tensor,
    center: torch.Tensor,
    normal: torch.Tensor,
    n_newton: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tangent0 = helix_tangent(track, torch.zeros_like(track[..., 0]))
    denominator = torch.sum(tangent0 * normal, dim=-1)
    valid = denominator.abs() >= PLANE_PARALLEL_EPS
    safe = torch.where(
        valid,
        denominator,
        torch.where(denominator >= 0, torch.ones_like(denominator), -torch.ones_like(denominator)),
    )
    path = torch.sum((center - track[..., :3]) * normal, dim=-1) / safe
    valid = valid & torch.isfinite(path) & (path.abs() <= S_MAX)
    path = torch.where(valid, path, torch.zeros_like(path))
    for _ in range(n_newton):
        point = helix_point(track, path)
        tangent = helix_tangent(track, path)
        residual = torch.sum((point - center) * normal, dim=-1)
        derivative = torch.sum(tangent * normal, dim=-1)
        derivative_ok = derivative.abs() >= PLANE_PARALLEL_EPS
        derivative_safe = torch.where(
            derivative_ok,
            derivative,
            torch.where(
                derivative >= 0,
                torch.full_like(derivative, PLANE_PARALLEL_EPS),
                torch.full_like(derivative, -PLANE_PARALLEL_EPS),
            ),
        )
        candidate = path - residual / derivative_safe
        candidate_ok = torch.isfinite(candidate) & (candidate.abs() <= S_MAX) & derivative_ok
        valid = valid & candidate_ok
        path = torch.where(valid, candidate, path)
    point = helix_point(track, path)
    plane_error = torch.sum((point - center) * normal, dim=-1).abs()
    valid = valid & torch.isfinite(point).all(dim=-1) & (plane_error <= PLANE_RESIDUAL_TOLERANCE)
    point = torch.where(valid.unsqueeze(-1), point, helix_point(track, torch.zeros_like(path)))
    path = torch.where(valid, path, torch.zeros_like(path))
    return path, point, valid


def torch_euler_to_rotation_matrix(delta: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = delta.unbind(-1)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    mx = torch.stack((one, zero, zero, zero, cx, -sx, zero, sx, cx), dim=-1).reshape(*rx.shape, 3, 3)
    my = torch.stack((cy, zero, sy, zero, one, zero, -sy, zero, cy), dim=-1).reshape(*rx.shape, 3, 3)
    mz = torch.stack((cz, -sz, zero, sz, cz, zero, zero, zero, one), dim=-1).reshape(*rx.shape, 3, 3)
    return mz @ my @ mx


def apply_alignment(
    centers: torch.Tensor, rotations: torch.Tensor, alignment: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted = centers + torch.matmul(rotations, alignment[..., :3].unsqueeze(-1)).squeeze(-1)
    rotated = rotations @ torch_euler_to_rotation_matrix(alignment[..., 3:])
    return shifted, rotated


def frozen_helix_geometry_loss(
    tracks: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    measured: torch.Tensor,
    variance: torch.Tensor,
    component_mask: torch.Tensor,
    alignment: torch.Tensor,
    *,
    lambda_l2: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    tracks = tracks.detach()
    moved_centers, moved_rotations = apply_alignment(centers, rotations, alignment)
    normals = moved_rotations[..., :, 2]
    _path, intersections, valid = intersect_plane(tracks, moved_centers, normals)
    local = torch.matmul(
        moved_rotations.transpose(-1, -2),
        (intersections - moved_centers).unsqueeze(-1),
    ).squeeze(-1)[..., :2]
    effective = component_mask & valid.unsqueeze(-1)
    if not effective.any():
        raise ValueError("no valid target measurement component remains")
    residual = measured - local
    weighted = residual.square() / variance.clamp_min(EPS_VARIANCE)
    data_loss = torch.where(effective, weighted, torch.zeros_like(weighted)).sum() / effective.sum()
    total = data_loss + float(lambda_l2) * alignment.square().sum()
    diagnostics = {
        "data_loss": float(data_loss.detach().cpu()),
        "invalid_fraction": 1.0 - float(valid.float().mean().detach().cpu()),
        "valid_components": int(effective.sum().detach().cpu()),
    }
    return total, diagnostics


def _surface_key(item: dict) -> tuple[int, int, int]:
    match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", item.get("ID", ""))
    if match is None:
        raise ValueError(f"invalid surface ID {item.get('ID')!r}")
    return tuple(map(int, match.groups()))


def load_working_geometry(nominal_path: str | Path, current_path: str | Path) -> WorkingGeometry:
    nominal = json.loads(Path(nominal_path).read_text(encoding="utf-8"))
    current = json.loads(Path(current_path).read_text(encoding="utf-8"))
    transforms = json.loads(json.dumps(nominal))
    id_to_idx = {_surface_key(item): index for index, item in enumerate(transforms)}
    if len(id_to_idx) != len(transforms):
        raise ValueError("nominal geometry contains duplicate surface IDs")
    input_indices = []
    for item in current:
        key = _surface_key(item)
        if key not in id_to_idx:
            raise KeyError(f"current surface {key} is absent from nominal geometry")
        index = id_to_idx[key]
        transforms[index] = json.loads(json.dumps(item))
        input_indices.append(index)
    return WorkingGeometry(
        transforms=transforms,
        input_transforms=current,
        centers=torch.tensor([item["Translation"] for item in transforms], dtype=torch.float64),
        rotations=torch.tensor([item["RotationMatrix"] for item in transforms], dtype=torch.float64),
        id_to_idx=id_to_idx,
        input_indices=input_indices,
    )


def _tree(root_file: uproot.ReadOnlyDirectory):
    for name in root_file.keys(cycle=False):
        if name == "trackstates" or "trackstates" in name.lower():
            return root_file[name]
    raise KeyError("ROOT file has no trackstates tree")


def expand_measurement_errors(
    local_x: np.ndarray,
    err_x: np.ndarray,
    err_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand compact measurement errors onto flattened track-state rows."""
    local_x = np.asarray(local_x)
    measured = np.isfinite(local_x)
    state_count = len(local_x)
    expanded = []
    for name, values in (("err_x_hit", err_x), ("err_y_hit", err_y)):
        values = np.asarray(values).reshape(-1)
        if len(values) == state_count:
            expanded.append(values.copy())
        elif len(values) == int(measured.sum()):
            result = np.full(state_count, np.nan, dtype=np.float64)
            result[measured] = values
            expanded.append(result)
        else:
            raise ValueError(
                f"{len(values)} flattened {name} values for {int(measured.sum())} measurements"
            )
    return expanded[0], expanded[1]


def broadcast_entry_values(values: np.ndarray, state_counts: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    state_counts = np.asarray(state_counts, dtype=np.int64).reshape(-1)
    if len(values) != len(state_counts):
        raise ValueError(
            f"{len(values)} per-entry values for {len(state_counts)} track-state counts"
        )
    return np.repeat(values, state_counts)


def read_flat_trackstates(path: str | Path, entry_stop: int | None = None) -> dict[str, np.ndarray]:
    per_state = (
        "volume_id", "layer_id", "module_id", "pathLength",
        "l_x_hit", "l_y_hit", "eLOC0_smt", "eLOC1_smt",
        "g_x_smt", "g_y_smt", "g_z_smt", "px_smt", "py_smt", "pz_smt",
    )
    per_entry = ("event_nr", "track_nr")
    with uproot.open(path) as root_file:
        tree = _tree(root_file)
        missing = [
            name for name in per_state + per_entry + ("err_x_hit", "err_y_hit")
            if name not in tree
        ]
        if missing:
            raise KeyError(f"trackstates is missing required branches: {missing}")
        arrays = {name: tree[name].array(entry_stop=entry_stop, library="ak") for name in per_state}
        entry_arrays = {
            name: tree[name].array(entry_stop=entry_stop, library="ak") for name in per_entry
        }
        err_x = tree["err_x_hit"].array(entry_stop=entry_stop, library="ak")
        err_y = tree["err_y_hit"].array(entry_stop=entry_stop, library="ak")
    output = {
        name: np.asarray(ak.to_numpy(ak.flatten(values, axis=None))).reshape(-1)
        for name, values in arrays.items()
    }
    state_count = len(output["l_x_hit"])
    if any(len(values) != state_count for values in output.values()):
        lengths = {name: len(values) for name, values in output.items()}
        raise ValueError(f"per-state ROOT branches have inconsistent flattened lengths: {lengths}")
    state_counts = np.asarray(ak.to_numpy(ak.num(arrays["volume_id"], axis=1)))
    for name, values in entry_arrays.items():
        flat_values = np.asarray(ak.to_numpy(values)).reshape(-1)
        output[name] = broadcast_entry_values(flat_values, state_counts)
    expanded_x, expanded_y = [], []
    for entry in range(len(arrays["l_x_hit"])):
        local_x = np.asarray(ak.to_numpy(arrays["l_x_hit"][entry])).reshape(-1)
        entry_err_x = np.asarray(ak.to_numpy(err_x[entry])).reshape(-1)
        entry_err_y = np.asarray(ak.to_numpy(err_y[entry])).reshape(-1)
        state_err_x, state_err_y = expand_measurement_errors(
            local_x, entry_err_x, entry_err_y
        )
        expanded_x.append(state_err_x)
        expanded_y.append(state_err_y)
    output["err_x_hit"] = np.concatenate(expanded_x)
    output["err_y_hit"] = np.concatenate(expanded_y)
    return output


def flat_states_from_root(
    path: str | Path,
    id_to_idx: dict[tuple[int, int, int], int],
    entry_stop: int | None = None,
) -> FlatStates:
    flat = read_flat_trackstates(path, entry_stop=entry_stop)
    measurement = np.isfinite(flat["l_x_hit"])
    keep = measurement.copy()
    for name in (
        "volume_id", "layer_id", "module_id", "eLOC0_smt", "eLOC1_smt",
        "g_x_smt", "g_y_smt", "g_z_smt", "px_smt", "py_smt", "pz_smt",
        "err_x_hit", "pathLength",
    ):
        keep &= np.isfinite(flat[name])
    keep &= flat["module_id"] > 0
    indices = np.flatnonzero(keep)
    sensor_indices = []
    for i in indices:
        key = (int(flat["volume_id"][i]), int(flat["layer_id"][i]), int(flat["module_id"][i]))
        if key not in id_to_idx:
            raise KeyError(f"measured state references surface absent from geometry: {key}")
        sensor_indices.append(id_to_idx[key])
    loc1_valid = (
        np.isfinite(flat["l_y_hit"][indices])
        & np.isfinite(flat["err_y_hit"][indices])
        & (flat["err_y_hit"][indices] > 0)
    )
    measured = np.column_stack((
        flat["l_x_hit"][indices],
        np.where(loc1_valid, flat["l_y_hit"][indices], 0.0),
    ))
    variance = np.column_stack((
        np.square(flat["err_x_hit"][indices]),
        np.where(loc1_valid, np.square(flat["err_y_hit"][indices]), 1.0),
    ))
    volume = torch.tensor(flat["volume_id"][indices], dtype=torch.long)
    adjusted, _mask = apply_measurement_variance_policy(
        torch.tensor(variance, dtype=torch.float64),
        volume,
        torch.tensor(loc1_valid, dtype=torch.bool),
    )
    return FlatStates(
        sensor_idx=torch.tensor(sensor_indices, dtype=torch.long),
        measured_local=torch.tensor(measured, dtype=torch.float64),
        variance=adjusted,
        loc1_valid=torch.tensor(loc1_valid, dtype=torch.bool),
        smt_local=torch.tensor(
            np.column_stack((flat["eLOC0_smt"][indices], flat["eLOC1_smt"][indices])),
            dtype=torch.float64,
        ),
        g_smt=torch.tensor(
            np.column_stack((flat["g_x_smt"][indices], flat["g_y_smt"][indices], flat["g_z_smt"][indices])),
            dtype=torch.float64,
        ),
        p_smt=torch.tensor(
            np.column_stack((flat["px_smt"][indices], flat["py_smt"][indices], flat["pz_smt"][indices])),
            dtype=torch.float64,
        ),
        volume_id=volume,
        layer_id=torch.tensor(flat["layer_id"][indices], dtype=torch.long),
        module_id=torch.tensor(flat["module_id"][indices], dtype=torch.long),
        event_id=torch.tensor(flat["event_nr"][indices], dtype=torch.long),
        track_id=torch.tensor(flat["track_nr"][indices], dtype=torch.long),
        path_length=torch.tensor(flat["pathLength"][indices], dtype=torch.float64),
    )


def build_track_windows(
    inputs: FlatStates,
    target_layers: str | Iterable[tuple[int, int]],
    *,
    min_fit_hits: int = 6,
    min_fit_span: float = 100.0,
    context_before: int = 4,
    context_after: int = 4,
    max_tracks: int = 0,
) -> TrackWindows:
    targets = set(parse_layers(target_layers))
    event = inputs.event_id.numpy()
    track = inputs.track_id.numpy()
    path = inputs.path_length.numpy()
    order = np.lexsort((path, track, event))
    boundaries = np.flatnonzero(
        (event[order][1:] != event[order][:-1]) | (track[order][1:] != track[order][:-1])
    ) + 1
    selected = []
    for group in np.split(order, boundaries):
        group = group[np.argsort(path[group], kind="stable")]
        fit_all = [i for i in group if (int(inputs.volume_id[i]), int(inputs.layer_id[i])) not in targets]
        holdouts = [i for i in group if (int(inputs.volume_id[i]), int(inputs.layer_id[i])) in targets]
        for holdout in holdouts:
            before = [i for i in fit_all if path[i] < path[holdout]][-context_before:]
            after = [i for i in fit_all if path[i] > path[holdout]][:context_after]
            fit = before + after
            if len(fit) < min_fit_hits:
                continue
            span = float(np.max(path[fit]) - np.min(path[fit]))
            if span < min_fit_span:
                continue
            indices = sorted(fit + [holdout], key=lambda i: path[i])
            selected.append(((int(event[holdout]), int(track[holdout])), indices, holdout, span))
    if max_tracks > 0:
        selected = selected[:max_tracks]
    if not selected:
        raise ValueError("no tracks satisfy the held-out helix selection")
    width = max(len(item[1]) for item in selected)
    n = len(selected)
    measured = torch.zeros((n, width, 2), dtype=torch.float64)
    variance = torch.ones((n, width, 2), dtype=torch.float64)
    loc1 = torch.zeros((n, width), dtype=torch.bool)
    smt_local = torch.zeros((n, width, 2), dtype=torch.float64)
    sensor = torch.zeros((n, width), dtype=torch.long)
    hit_mask = torch.zeros((n, width), dtype=torch.bool)
    fit_mask = torch.zeros((n, width), dtype=torch.bool)
    holdout_mask = torch.zeros((n, width), dtype=torch.bool)
    g_smt = torch.zeros((n, width, 3), dtype=torch.float64)
    p_smt = torch.zeros((n, width, 3), dtype=torch.float64)
    volume = torch.zeros((n, width), dtype=torch.long)
    layer = torch.zeros((n, width), dtype=torch.long)
    event_track = torch.zeros((n, 2), dtype=torch.long)
    spans = torch.zeros(n, dtype=torch.float64)
    for row, (key, indices, holdout, span) in enumerate(selected):
        idx = torch.tensor(indices, dtype=torch.long)
        count = len(indices)
        measured[row, :count] = inputs.measured_local[idx]
        variance[row, :count] = inputs.variance[idx]
        loc1[row, :count] = inputs.loc1_valid[idx]
        smt_local[row, :count] = inputs.smt_local[idx]
        sensor[row, :count] = inputs.sensor_idx[idx]
        hit_mask[row, :count] = True
        g_smt[row, :count] = inputs.g_smt[idx]
        p_smt[row, :count] = inputs.p_smt[idx]
        volume[row, :count] = inputs.volume_id[idx]
        layer[row, :count] = inputs.layer_id[idx]
        holdout_column = indices.index(holdout)
        holdout_mask[row, holdout_column] = True
        fit_mask[row, :count] = True
        fit_mask[row, holdout_column] = False
        event_track[row] = torch.tensor(key)
        spans[row] = span
    return TrackWindows(
        measured, variance, loc1, smt_local, sensor, hit_mask, fit_mask,
        holdout_mask, g_smt, p_smt, volume, layer, event_track, spans
    )


def _initial_helices(windows: TrackWindows, device, dtype) -> torch.Tensor:
    points = windows.g_smt.to(device=device, dtype=dtype)
    directions = windows.p_smt.to(device=device, dtype=dtype)
    mask = windows.fit_mask.to(device)
    first_idx = mask.long().argmax(dim=1)
    rows = torch.arange(len(points), device=device)
    first = points[rows, first_idx]
    direction = directions[rows, first_idx]
    transverse = torch.linalg.norm(direction[:, :2], dim=1).clamp_min(1e-6)
    return torch.cat((
        first,
        torch.atan2(direction[:, 1], direction[:, 0]).unsqueeze(1),
        (direction[:, 2] / transverse).unsqueeze(1),
        torch.zeros((len(points), 1), device=device, dtype=dtype),
    ), dim=1)


def _closest_paths(track: torch.Tensor, points: torch.Tensor, n_newton: int = 8):
    delta = points - track[..., :3]
    path = delta[..., 0] * torch.cos(track[..., 3]) + delta[..., 1] * torch.sin(track[..., 3])
    valid = torch.isfinite(path)
    path = torch.where(valid, path, torch.zeros_like(path))
    for _ in range(n_newton):
        position = helix_point(track, path)
        tangent = helix_tangent(track, path)
        displacement = position - points
        rho = track[..., 5]
        psi = track[..., 3] + rho * path
        second = torch.stack((-rho * torch.sin(psi), rho * torch.cos(psi), torch.zeros_like(rho)), dim=-1)
        first_d = torch.sum(displacement * tangent, dim=-1)
        second_d = torch.sum(tangent.square() + displacement * second, dim=-1)
        safe = torch.where(second_d.abs() > 1e-9, second_d, torch.full_like(second_d, 1e-9))
        candidate = path - first_d / safe
        ok = torch.isfinite(candidate) & (candidate.abs() <= S_MAX)
        valid = valid & ok
        path = torch.where(valid, candidate, path)
    return path, valid


def fit_helices(
    windows: TrackWindows,
    geometry: WorkingGeometry,
    *,
    fit_iterations: int = 500,
    batch_size: int = 2048,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SMT seed then measurement refit, always excluding holdout states."""
    device = torch.device(device)
    fitted_chunks = []
    initial_chunks = []
    n_tracks = len(windows.measured_local)
    for start in range(0, n_tracks, max(1, batch_size)):
        stop = min(start + max(1, batch_size), n_tracks)
        sl = slice(start, stop)
        initial = _initial_helices(_slice_windows(windows, sl), device, dtype)
        initial_chunks.append(initial.detach().cpu())
        points = windows.g_smt[sl].to(device=device, dtype=dtype)
        fit_mask = windows.fit_mask[sl].to(device)
        width = points.shape[1]
        linear = initial[:, :5].clone().requires_grad_(True)
        rho_scaled = (initial[:, 5:] / 1e-4).clone().requires_grad_(True)
        seed_opt = optim.Adam(({"params": [linear], "lr": 0.5}, {"params": [rho_scaled], "lr": 0.5}))
        for _ in range(fit_iterations):
            seed_opt.zero_grad()
            tracks = torch.cat((linear, rho_scaled * 1e-4), dim=1)
            expanded = tracks[:, None, :].expand(-1, width, -1)
            with torch.no_grad():
                paths, valid = _closest_paths(expanded.detach(), points)
            residual = helix_point(expanded, paths) - points
            effective = fit_mask & valid
            loss = torch.where(effective, residual.square().sum(dim=-1), torch.zeros_like(paths)).sum()
            loss = loss / effective.sum().clamp_min(1)
            loss.backward()
            seed_opt.step()
        measured = windows.measured_local[sl].to(device=device, dtype=dtype)
        variance = windows.variance[sl].to(device=device, dtype=dtype)
        loc1 = windows.loc1_valid[sl].to(device)
        sensor = windows.sensor_idx[sl].to(device)
        centers = geometry.centers.to(device=device, dtype=dtype)[sensor]
        rotations = geometry.rotations.to(device=device, dtype=dtype)[sensor]
        linear = linear.detach().requires_grad_(True)
        rho_scaled = rho_scaled.detach().requires_grad_(True)
        fit_opt = optim.Adam(({"params": [linear], "lr": 0.05}, {"params": [rho_scaled], "lr": 0.05}))
        for _ in range(fit_iterations):
            fit_opt.zero_grad()
            tracks = torch.cat((linear, rho_scaled * 1e-4), dim=1)
            expanded = tracks[:, None, :].expand(-1, width, -1)
            _paths, intersections, valid = intersect_plane(expanded, centers, rotations[..., :, 2])
            local = torch.matmul(
                rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
            ).squeeze(-1)[..., :2]
            hit_valid = fit_mask & valid
            component = torch.stack((hit_valid, hit_valid & loc1), dim=-1)
            residual = measured - local
            weighted = residual.square() / variance.clamp_min(EPS_VARIANCE)
            loss = torch.where(component, weighted, torch.zeros_like(weighted)).sum() / component.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite measurement helix loss")
            loss.backward()
            fit_opt.step()
        fitted_chunks.append(torch.cat((linear, rho_scaled * 1e-4), dim=1).detach().cpu())
    return torch.cat(initial_chunks), torch.cat(fitted_chunks)


def _slice_windows(windows: TrackWindows, selection) -> TrackWindows:
    return TrackWindows(*(getattr(windows, field)[selection] for field in windows.__dataclass_fields__))


def _holdout_arrays(windows: TrackWindows, tracks: torch.Tensor, geometry: WorkingGeometry):
    columns = windows.holdout_mask.long().argmax(dim=1)
    rows = torch.arange(len(columns))
    sensor = windows.sensor_idx[rows, columns]
    volume = windows.volume_id[rows, columns]
    layer = windows.layer_id[rows, columns]
    measured = windows.measured_local[rows, columns]
    variance = windows.variance[rows, columns]
    loc1 = windows.loc1_valid[rows, columns]
    centers = geometry.centers[sensor]
    rotations = geometry.rotations[sensor]
    return sensor, volume, layer, measured, variance, loc1, centers, rotations


def helix_preflight_metrics(windows: TrackWindows, tracks: torch.Tensor, geometry: WorkingGeometry) -> dict:
    _sensor, _volume, _layer, _measured, _variance, loc1, centers, rotations = _holdout_arrays(
        windows, tracks, geometry
    )
    _path, points, valid = intersect_plane(tracks, centers, rotations[..., :, 2])
    local = torch.matmul(rotations.transpose(-1, -2), (points - centers).unsqueeze(-1)).squeeze(-1)[..., :2]
    columns = windows.holdout_mask.long().argmax(dim=1)
    rows = torch.arange(len(columns))
    difference = local - windows.smt_local[rows, columns]
    component = torch.stack((valid, valid & loc1), dim=-1)
    valid_difference = difference[component]
    rmse = float(torch.sqrt(valid_difference.square().mean()).item()) if valid_difference.numel() else None
    return {
        "local_component_rmse_mm": rmse,
        "intersection_failure_rate": 1.0 - float(valid.float().mean()),
        "valid_count": int(valid.sum()),
    }


def optimize_frozen_geometry(
    windows: TrackWindows,
    fitted_tracks: torch.Tensor,
    geometry: WorkingGeometry,
    target_layers,
    *,
    active_dof=(0, 1, 5),
    iterations=300,
    learning_rate=1e-2,
    lambda_l2=1e-5,
    min_hits_per_surface=10,
    min_surface_coverage=0.90,
    max_invalid_fraction=0.05,
    device="cpu",
    dtype=torch.float64,
) -> tuple[np.ndarray, dict]:
    targets = set(parse_layers(target_layers))
    active_dof = parse_active_dof(active_dof)
    sensor, volume, layer, measured, variance, loc1, centers, rotations = _holdout_arrays(
        windows, fitted_tracks, geometry
    )
    target_row = torch.tensor([(int(v), int(l)) in targets for v, l in zip(volume, layer)])
    sensor, volume, layer = sensor[target_row], volume[target_row], layer[target_row]
    measured, variance, loc1 = measured[target_row], variance[target_row], loc1[target_row]
    centers, rotations = centers[target_row], rotations[target_row]
    tracks = fitted_tracks[target_row].detach()
    expected = [idx for key, idx in geometry.id_to_idx.items() if key[:2] in targets]
    counts = torch.bincount(sensor, minlength=len(geometry.centers))
    eligible = torch.zeros(len(geometry.centers), dtype=torch.bool)
    eligible[expected] = counts[expected] >= min_hits_per_surface
    coverage = float(eligible[expected].float().mean()) if expected else 0.0
    if coverage < min_surface_coverage:
        raise ValueError(
            f"eligible target-surface coverage {coverage:.2%} is below {min_surface_coverage:.2%}"
        )
    use = eligible[sensor]
    sensor, measured, variance, loc1 = sensor[use], measured[use], variance[use], loc1[use]
    centers, rotations, tracks = centers[use], rotations[use], tracks[use]
    device = torch.device(device)
    tracks = tracks.to(device=device, dtype=dtype)
    centers = centers.to(device=device, dtype=dtype)
    rotations = rotations.to(device=device, dtype=dtype)
    measured = measured.to(device=device, dtype=dtype)
    variance = variance.to(device=device, dtype=dtype)
    component = torch.stack((torch.ones_like(loc1), loc1), dim=-1).to(device)
    sensor = sensor.to(device)
    parameter = torch.nn.Parameter(torch.zeros((len(geometry.centers), 6), device=device, dtype=dtype))
    dof_mask = torch.zeros(6, device=device, dtype=dtype)
    dof_mask[list(active_dof)] = 1.0
    surface_mask = eligible.to(device=device, dtype=dtype).unsqueeze(-1)
    optimizer = optim.Adam([parameter], lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, iterations), eta_min=1e-5)
    best_loss = math.inf
    best = None
    history = []
    invalid_history = []
    for _iteration in range(iterations):
        optimizer.zero_grad()
        effective_parameter = parameter * dof_mask * surface_mask
        data_loss, diagnostics = frozen_helix_geometry_loss(
            tracks, centers, rotations, measured, variance, component,
            effective_parameter[sensor], lambda_l2=0.0,
        )
        loss = data_loss + float(lambda_l2) * effective_parameter.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite frozen-helix geometry loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 10.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            parameter.mul_(dof_mask * surface_mask)
            parameter[:, :3].clamp_(-1.5, 1.5)
            parameter[:, 3:].clamp_(-0.05, 0.05)
        value = float(loss.detach().cpu())
        history.append(value)
        invalid_history.append(diagnostics["invalid_fraction"])
        if diagnostics["invalid_fraction"] <= max_invalid_fraction and value < best_loss:
            best_loss = value
            best = parameter.detach().clone()
    if best is None:
        raise ValueError("no geometry checkpoint satisfies the intersection-failure gate")
    return best.cpu().numpy(), {
        "history": history,
        "best_loss": best_loss,
        "invalid_fraction": invalid_history[-1],
        "eligible_surface_count": int(eligible.sum()),
        "target_surface_count": len(expected),
        "surface_coverage": coverage,
        "target_hit_count": len(tracks),
    }


def _numpy_euler(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    my = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    mz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return mz @ my @ mx


def _rotation_to_euler(rotation: np.ndarray) -> np.ndarray:
    ry = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    rx = np.arctan2(rotation[2, 1], rotation[2, 2])
    rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array((rx, ry, rz), dtype=float)


def geometry_truth_diagnostics(
    geometry: WorkingGeometry,
    deltas: np.ndarray,
    nominal_path: str | Path,
    layers,
) -> dict:
    """Simulation-only before/after distance to nominal in local coordinates."""
    nominal_items = json.loads(Path(nominal_path).read_text(encoding="utf-8"))
    nominal = {_surface_key(item): item for item in nominal_items}
    output = {}
    names = ("tx_mm", "ty_mm", "tz_mm", "rx_rad", "ry_rad", "rz_rad")
    for volume, layer in parse_layers(layers):
        before, after = [], []
        for key, index in geometry.id_to_idx.items():
            if key[:2] != (volume, layer):
                continue
            nominal_item = nominal[key]
            nominal_center = np.asarray(nominal_item["Translation"], dtype=float)
            nominal_rotation = np.asarray(nominal_item["RotationMatrix"], dtype=float)
            current_center = geometry.centers[index].numpy()
            current_rotation = geometry.rotations[index].numpy()
            delta = np.asarray(deltas[index], dtype=float)
            aligned_center = current_center + current_rotation @ delta[:3]
            aligned_rotation = current_rotation @ _numpy_euler(*delta[3:])
            before.append(np.concatenate((
                nominal_rotation.T @ (current_center - nominal_center),
                _rotation_to_euler(nominal_rotation.T @ current_rotation),
            )))
            after.append(np.concatenate((
                nominal_rotation.T @ (aligned_center - nominal_center),
                _rotation_to_euler(nominal_rotation.T @ aligned_rotation),
            )))
        before_array = np.asarray(before)
        after_array = np.asarray(after)
        output[f"{volume}:{layer}"] = {
            "surface_count": len(before),
            "before_rms": {
                name: float(np.sqrt(np.mean(before_array[:, i] ** 2)))
                for i, name in enumerate(names)
            },
            "after_rms": {
                name: float(np.sqrt(np.mean(after_array[:, i] ** 2)))
                for i, name in enumerate(names)
            },
        }
    return output


def export_geometry(geometry: WorkingGeometry, deltas: np.ndarray, output_path: str | Path):
    output = json.loads(json.dumps(geometry.input_transforms))
    for item, index in zip(output, geometry.input_indices):
        delta = deltas[index]
        old_center = np.asarray(item["Translation"], dtype=float)
        old_rotation = np.asarray(item["RotationMatrix"], dtype=float)
        center = old_center + old_rotation @ delta[:3]
        rotation = old_rotation @ _numpy_euler(*delta[3:])
        item["Translation"] = center.tolist()
        item["RotationMatrix"] = rotation.tolist()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = center
        if "FullMatrix" in item and len(item["FullMatrix"]) == 16:
            column_major = abs(item["FullMatrix"][12] - old_center[0]) < FULL_MATRIX_TOLERANCE
            item["FullMatrix"] = matrix.flatten("F" if column_major else "C").tolist()
        else:
            item["FullMatrix"] = matrix.flatten("C").tolist()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


def residual_metrics(path: str | Path, layers) -> dict[str, dict[str, float | None]]:
    names = ("volume_id", "layer_id", "l_x_hit", "l_y_hit", "eLOC0_smt", "eLOC1_smt")
    with uproot.open(path) as root_file:
        tree = _tree(root_file)
        arrays = tree.arrays(names, library="ak")
    flat = {
        name: np.asarray(ak.to_numpy(ak.flatten(arrays[name], axis=None))).reshape(-1)
        for name in names
    }
    output = {}
    for volume, layer in parse_layers(layers):
        selection = (
            (flat["volume_id"].astype(int) == volume)
            & (flat["layer_id"].astype(int) == layer)
            & np.isfinite(flat["l_x_hit"])
            & np.isfinite(flat["eLOC0_smt"])
        )
        loc0 = flat["l_x_hit"][selection] - flat["eLOC0_smt"][selection]
        loc1_selection = selection & np.isfinite(flat["l_y_hit"]) & np.isfinite(flat["eLOC1_smt"])
        loc1 = flat["l_y_hit"][loc1_selection] - flat["eLOC1_smt"][loc1_selection]
        output[f"{volume}:{layer}"] = {
            "loc0_count": int(len(loc0)),
            "loc0_rmse_mm": float(np.sqrt(np.mean(np.square(loc0)))) if len(loc0) else None,
            "loc1_count": int(len(loc1)),
            "loc1_rmse_mm": float(np.sqrt(np.mean(np.square(loc1)))) if len(loc1) else None,
        }
    return output


def run_assess(args) -> dict:
    layers = parse_layers(args.target_layers)
    nominal = residual_metrics(args.nominal_states, layers)
    candidate = residual_metrics(args.candidate_states, layers)
    failed, report = select_failed_layers(nominal, candidate, layers, max_ratio=args.max_ratio)
    summary = {
        "nominal": nominal,
        "candidate": candidate,
        "max_ratio": args.max_ratio,
        "failed_layers": [f"{v}:{l}" for v, l in failed],
        "comparison": report,
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print("FAILED_LAYERS=" + ",".join(summary["failed_layers"]))
    return summary


def run_align(args) -> dict:
    layers = parse_layers(args.target_layers)
    geometry = load_working_geometry(args.nominal_geometry, args.input_geometry)
    entry_stop = args.max_tracks * 4 if args.max_tracks > 0 else None
    states = flat_states_from_root(args.states, geometry.id_to_idx, entry_stop=entry_stop)
    windows = build_track_windows(
        states, layers, min_fit_hits=args.min_fit_hits, min_fit_span=args.min_fit_span,
        context_before=args.context_before, context_after=args.context_after,
        max_tracks=args.max_tracks,
    )
    use_cuda = args.device in {"auto", "cuda"} and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float32 if use_cuda else torch.float64
    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)
    initial, fitted = fit_helices(
        windows, geometry, fit_iterations=args.fit_iterations,
        batch_size=args.fit_batch_size, device=device, dtype=dtype,
    )
    initial_metrics = helix_preflight_metrics(windows, initial, geometry)
    fitted_metrics = helix_preflight_metrics(windows, fitted, geometry)
    if fitted_metrics["intersection_failure_rate"] > args.max_invalid_fraction:
        raise ValueError("fitted helix intersection failure rate exceeds configured limit")
    if (
        fitted_metrics["local_component_rmse_mm"] is None
        or initial_metrics["local_component_rmse_mm"] is None
        or fitted_metrics["local_component_rmse_mm"] >= initial_metrics["local_component_rmse_mm"]
    ):
        raise ValueError("held-out helix prediction did not improve over SMT initialization")
    deltas, alignment_metrics = optimize_frozen_geometry(
        windows, fitted, geometry, layers,
        active_dof=parse_active_dof(args.active_dof), iterations=args.geometry_iterations,
        learning_rate=args.learning_rate, lambda_l2=args.lambda_l2,
        min_hits_per_surface=args.min_hits_per_surface,
        min_surface_coverage=args.min_surface_coverage,
        max_invalid_fraction=args.max_invalid_fraction,
        device=device, dtype=dtype,
    )
    truth_metrics = geometry_truth_diagnostics(
        geometry, deltas, args.nominal_geometry, layers
    )
    export_geometry(geometry, deltas, args.output_geometry)
    summary = {
        "config": {key: value for key, value in vars(args).items() if key != "handler"},
        "target_layers": [f"{v}:{l}" for v, l in layers],
        "device": device,
        "track_window_count": len(windows.measured_local),
        "fit_span_mm": {
            "min": float(windows.fit_span.min()),
            "median": float(windows.fit_span.median()),
            "max": float(windows.fit_span.max()),
        },
        "helix_preflight": {"initial": initial_metrics, "fitted": fitted_metrics},
        "alignment": alignment_metrics,
        "truth_diagnostics": truth_metrics,
        "output_geometry": str(args.output_geometry),
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Held-out frozen-helix surface alignment")
    add_config_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assess = subparsers.add_parser("assess", help="compare ACTS residual RMS to nominal")
    assess.add_argument("--nominal-states", required=True)
    assess.add_argument("--candidate-states", required=True)
    assess.add_argument("--target-layers", default=target_layers_spec())
    assess.add_argument("--max-ratio", type=float, default=1.2)
    assess.add_argument("--output-summary", required=True)
    assess.set_defaults(handler=run_assess)

    align = subparsers.add_parser("align", help="fit held-out helices and optimize geometry")
    align.add_argument("--input-geometry", required=True)
    align.add_argument("--states", required=True)
    align.add_argument("--nominal-geometry", default=str(data_path("odd_transforms.json")))
    align.add_argument("--output-geometry", required=True)
    align.add_argument("--output-summary", required=True)
    align.add_argument("--target-layers", default=target_layers_spec())
    align.add_argument("--active-dof", default="0,1,5")
    align.add_argument("--min-fit-hits", type=int, default=6)
    align.add_argument("--min-fit-span", type=float, default=100.0)
    align.add_argument("--context-before", type=int, default=4)
    align.add_argument("--context-after", type=int, default=4)
    align.add_argument("--fit-iterations", type=int, default=500)
    align.add_argument("--fit-batch-size", type=int, default=2048)
    align.add_argument("--geometry-iterations", type=int, default=300)
    align.add_argument("--learning-rate", type=float, default=1e-2)
    align.add_argument("--lambda-l2", type=float, default=1e-5)
    align.add_argument("--min-hits-per-surface", type=int, default=10)
    align.add_argument("--min-surface-coverage", type=float, default=0.90)
    align.add_argument("--max-invalid-fraction", type=float, default=0.05)
    align.add_argument("--max-tracks", type=int, default=0)
    align.add_argument("--rng-seed", type=int, default=42)
    align.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    align.set_defaults(handler=run_align)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
