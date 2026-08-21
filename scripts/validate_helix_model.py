"""Validate the standalone helix model against nominal ACTS smoothed states.

The helix parameter order and geometry conventions intentionally mirror
``ml_alignment_autograd.py``: ``[x0, y0, z0, psi0, tanl, rho]`` and local
coordinates obtained with ``rotation.T @ (global - center)``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import uproot

from alignment_artifact_layout import (
    TARGET_LAYER_DIR,
    TARGET_LAYERS,
    data_path,
    reconstruction_root_path,
    target_layers_spec,
)
from project_config import absolute_project_path, add_config_argument


S_MAX = 1200.0
PLANE_PARALLEL_EPS = 1e-6
PLANE_RESIDUAL_TOLERANCE = 1e-3
DEFAULT_MIN_FIT_HITS = 8
DEFAULT_MIN_FIT_SPAN = 300.0
DEFAULT_MAX_TRACKS = 1000
DEFAULT_FIT_ITERS = 200
DEFAULT_GEOMETRY_TOLERANCE = 1e-3
DEFAULT_MAX_INTERSECTION_FAILURE_RATE = 0.05
DEFAULT_MAX_PIXEL_LOCAL_RMSE = 0.05
DEFAULT_MAX_STRIP_LOC0_RMSE = 0.10
DEFAULT_MAX_STRIP_LOC1_RMSE = 1.50
DEFAULT_MAX_DIRECTION_ANGLE_RMS = 0.005
DEFAULT_CONTEXT_HITS_BEFORE = 4
DEFAULT_CONTEXT_HITS_AFTER = 4
EPS_VARIANCE = 1e-8


@dataclass
class HelixValidationConfig:
    states: str = str(
        reconstruction_root_path(
            "reconstruction_output", "nominal", "trackstates_kf"
        )
    )
    geometry: str = str(data_path("odd_transforms.json"))
    nominal_full_geometry: str = str(data_path("odd_transforms.json"))
    holdout_layers: tuple[tuple[int, int], ...] | str = TARGET_LAYERS
    min_fit_hits: int = DEFAULT_MIN_FIT_HITS
    min_fit_span: float = DEFAULT_MIN_FIT_SPAN
    max_tracks: int = DEFAULT_MAX_TRACKS
    fit_iters: int = DEFAULT_FIT_ITERS
    gpu: bool = False
    rng_seed: int = 42
    output_dir: str = str(data_path("helix_validation_output") / TARGET_LAYER_DIR)
    geometry_tolerance: float = DEFAULT_GEOMETRY_TOLERANCE
    max_intersection_failure_rate: float = DEFAULT_MAX_INTERSECTION_FAILURE_RATE
    max_pixel_local_rmse: float = DEFAULT_MAX_PIXEL_LOCAL_RMSE
    max_strip_loc0_rmse: float = DEFAULT_MAX_STRIP_LOC0_RMSE
    max_strip_loc1_rmse: float = DEFAULT_MAX_STRIP_LOC1_RMSE
    max_direction_angle_rms: float = DEFAULT_MAX_DIRECTION_ANGLE_RMS
    context_hits_before: int = DEFAULT_CONTEXT_HITS_BEFORE
    context_hits_after: int = DEFAULT_CONTEXT_HITS_AFTER


@dataclass
class HelixInputs:
    hit_sensor_ids: torch.Tensor
    measured_local: torch.Tensor
    measurement_variance: torch.Tensor
    loc1_valid: torch.Tensor
    smt_local: torch.Tensor
    hit_global_positions: torch.Tensor
    hit_directions: torch.Tensor
    hit_volume_ids: torch.Tensor
    hit_layer_ids: torch.Tensor
    hit_module_ids: torch.Tensor
    hit_event_ids: torch.Tensor
    hit_track_ids: torch.Tensor
    hit_path_length: torch.Tensor | None = None


@dataclass
class TrackBatches:
    measured_local: torch.Tensor
    measurement_variance: torch.Tensor
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
    module_id: torch.Tensor
    event_track_ids: torch.Tensor
    fit_span: torch.Tensor


@dataclass
class NominalGeometry:
    transforms: list[dict]
    centers: torch.Tensor
    rotations: torch.Tensor
    id_to_idx: dict[tuple[int, int, int], int]


@dataclass
class HelixEvaluation:
    path_length: torch.Tensor
    global_position: torch.Tensor
    local_position: torch.Tensor
    global_difference: torch.Tensor
    local_difference: torch.Tensor
    tangent_angle: torch.Tensor
    intersection_valid: torch.Tensor


def parse_layers(value) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, str):
        parsed = tuple((int(volume), int(layer)) for volume, layer in value)
    else:
        parsed = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Layer {item!r} must have format volume:layer."
                )
            parsed.append((int(parts[0]), int(parts[1])))
        parsed = tuple(parsed)
    if not parsed:
        raise ValueError("At least one holdout layer is required.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Holdout layers must not contain duplicates.")
    return tuple(parsed)


def _surface_key(item: dict) -> tuple[int, int, int]:
    match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", item.get("ID", ""))
    if match is None:
        raise ValueError(f"Surface ID has no vol/lay/sen fields: {item.get('ID', '')!r}")
    return tuple(map(int, match.groups()))


def load_nominal_geometry(path: str | Path) -> NominalGeometry:
    with open(path, "r", encoding="utf-8") as stream:
        transforms = json.load(stream)
    id_to_idx = {_surface_key(item): index for index, item in enumerate(transforms)}
    if len(id_to_idx) != len(transforms):
        raise ValueError("Nominal geometry contains duplicate surface IDs.")
    centers = torch.tensor(
        [item["Translation"] for item in transforms], dtype=torch.float64
    )
    rotations = torch.tensor(
        [item["RotationMatrix"] for item in transforms], dtype=torch.float64
    )
    if rotations.shape != (len(transforms), 3, 3):
        raise ValueError("Nominal geometry rotation matrices must all be 3x3.")
    return NominalGeometry(
        transforms=transforms,
        centers=centers,
        rotations=rotations,
        id_to_idx=id_to_idx,
    )


def load_working_geometry(
    current_path: str | Path, nominal_full_path: str | Path
) -> NominalGeometry:
    """Overlay a partial current geometry on the full nominal geometry."""
    current_path = Path(current_path)
    nominal_full_path = Path(nominal_full_path)
    if current_path.resolve() == nominal_full_path.resolve():
        return load_nominal_geometry(current_path)
    with open(nominal_full_path, "r", encoding="utf-8") as stream:
        nominal = json.load(stream)
    with open(current_path, "r", encoding="utf-8") as stream:
        current = json.load(stream)
    nominal_index = {_surface_key(item): index for index, item in enumerate(nominal)}
    if len(nominal_index) != len(nominal):
        raise ValueError("Nominal geometry contains duplicate surface IDs.")
    for item in current:
        key = _surface_key(item)
        if key not in nominal_index:
            raise ValueError(f"Current surface {key} is absent from nominal geometry.")
        nominal[nominal_index[key]] = item
    id_to_idx = {_surface_key(item): index for index, item in enumerate(nominal)}
    centers = torch.tensor(
        [item["Translation"] for item in nominal], dtype=torch.float64
    )
    rotations = torch.tensor(
        [item["RotationMatrix"] for item in nominal], dtype=torch.float64
    )
    return NominalGeometry(
        transforms=nominal,
        centers=centers,
        rotations=rotations,
        id_to_idx=id_to_idx,
    )


def _flatten_jagged(array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(array, axis=None))


def _float_array(values) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def expand_measurement_errors(
    l_x_jagged, err_x_jagged, err_y_jagged
) -> tuple[np.ndarray, np.ndarray]:
    """Expand compact per-measurement errors onto per-state rows."""
    expanded_x = []
    expanded_y = []
    for track_index, (local_x, err_x, err_y) in enumerate(
        zip(ak.to_list(l_x_jagged), ak.to_list(err_x_jagged), ak.to_list(err_y_jagged))
    ):
        local_x = _float_array(local_x)
        err_x = _float_array(err_x)
        err_y = _float_array(err_y)
        measurement_mask = np.isfinite(local_x)
        n_states = len(local_x)
        n_measurements = int(measurement_mask.sum())

        def expand(values: np.ndarray, name: str) -> np.ndarray:
            if len(values) == n_states:
                return values
            if len(values) != n_measurements:
                raise ValueError(
                    f"Track entry {track_index} has {n_states} states and "
                    f"{n_measurements} measurements but {len(values)} {name} values."
                )
            result = np.full(n_states, np.nan, dtype=np.float64)
            result[measurement_mask] = values
            return result

        expanded_x.append(expand(err_x, "err_x_hit"))
        expanded_y.append(expand(err_y, "err_y_hit"))
    if not expanded_x:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return np.concatenate(expanded_x), np.concatenate(expanded_y)


def _helix_inputs_from_flat(
    flat: dict[str, np.ndarray],
    id_to_idx: dict[tuple[int, int, int], int],
) -> HelixInputs:
    """Validate flattened measured states and map every one to nominal geometry."""
    measurement = np.isfinite(flat["l_x_hit"])
    if not measurement.any():
        raise ValueError("No measured states with finite l_x_hit are available.")
    modules = flat["module_id"].astype(np.int64)
    invalid_modules = measurement & (modules <= 0)
    if invalid_modules.any():
        raise ValueError(
            f"{int(invalid_modules.sum())} measured states have non-positive module IDs."
        )
    keys = list(
        zip(
            flat["volume_id"].astype(np.int64),
            flat["layer_id"].astype(np.int64),
            modules,
        )
    )
    sensor_ids = np.array([id_to_idx.get(key, -1) for key in keys], dtype=np.int64)
    unmapped = measurement & (sensor_ids < 0)
    if unmapped.any():
        examples = sorted({keys[index] for index in np.flatnonzero(unmapped)})[:5]
        raise ValueError(
            f"{int(unmapped.sum())} measured states reference surfaces absent from "
            f"nominal geometry; examples: {examples}."
        )
    required_finite = (
        "eLOC0_smt",
        "eLOC1_smt",
        "g_x_smt",
        "g_y_smt",
        "g_z_smt",
        "px_smt",
        "py_smt",
        "pz_smt",
    )
    invalid_values = np.zeros(len(measurement), dtype=bool)
    for name in required_finite:
        invalid_values |= measurement & ~np.isfinite(flat[name])
    if invalid_values.any():
        raise ValueError(
            f"{int(invalid_values.sum())} measured states contain non-finite SMT "
            "positions, directions, or local coordinates."
        )
    if "err_x_hit" not in flat or "err_y_hit" not in flat:
        raise ValueError("Measurement errors err_x_hit/err_y_hit are required.")
    invalid_err_x = measurement & (
        ~np.isfinite(flat["err_x_hit"]) | (flat["err_x_hit"] <= 0)
    )
    if invalid_err_x.any():
        raise ValueError(
            f"{int(invalid_err_x.sum())} measured states have invalid err_x_hit."
        )
    loc1_valid_all = (
        measurement
        & np.isfinite(flat["l_y_hit"])
        & np.isfinite(flat["err_y_hit"])
        & (flat["err_y_hit"] > 0)
    )
    keep = measurement

    def tensor(names, dtype=torch.float64):
        values = np.column_stack([flat[name][keep] for name in names])
        return torch.tensor(values, dtype=dtype)

    measured_y = np.where(loc1_valid_all[keep], flat["l_y_hit"][keep], 0.0)
    variance_y = np.where(
        loc1_valid_all[keep], np.square(flat["err_y_hit"][keep]), 1.0
    )
    return HelixInputs(
        hit_sensor_ids=torch.tensor(sensor_ids[keep], dtype=torch.long),
        measured_local=torch.tensor(
            np.column_stack((flat["l_x_hit"][keep], measured_y)),
            dtype=torch.float64,
        ),
        measurement_variance=torch.tensor(
            np.column_stack((np.square(flat["err_x_hit"][keep]), variance_y)),
            dtype=torch.float64,
        ).clamp_min(EPS_VARIANCE),
        loc1_valid=torch.tensor(loc1_valid_all[keep], dtype=torch.bool),
        smt_local=tensor(("eLOC0_smt", "eLOC1_smt")),
        hit_global_positions=tensor(("g_x_smt", "g_y_smt", "g_z_smt")),
        hit_directions=tensor(("px_smt", "py_smt", "pz_smt")),
        hit_volume_ids=torch.tensor(flat["volume_id"][keep], dtype=torch.long),
        hit_layer_ids=torch.tensor(flat["layer_id"][keep], dtype=torch.long),
        hit_module_ids=torch.tensor(modules[keep], dtype=torch.long),
        hit_event_ids=torch.tensor(flat["event_nr"][keep], dtype=torch.long),
        hit_track_ids=torch.tensor(flat["track_nr"][keep], dtype=torch.long),
        hit_path_length=(
            torch.tensor(flat["pathLength"][keep], dtype=torch.float64)
            if "pathLength" in flat
            else None
        ),
    )


def _load_nominal_states_uproot(
    path: str | Path,
    id_to_idx: dict[tuple[int, int, int], int],
    max_track_entries: int | None = None,
) -> HelixInputs:
    """Load measured nominal states needed by the held-out helix validation."""
    required = [
        "volume_id",
        "layer_id",
        "module_id",
        "pathLength",
        "l_x_hit",
        "l_y_hit",
        "eLOC0_smt",
        "eLOC1_smt",
        "g_x_smt",
        "g_y_smt",
        "g_z_smt",
        "px_smt",
        "py_smt",
        "pz_smt",
    ]
    with uproot.open(path) as root_file:
        if "trackstates" not in root_file:
            raise ValueError(f"ROOT file {path} has no 'trackstates' tree.")
        tree = root_file["trackstates"]
        missing = [
            name
            for name in required + ["event_nr", "track_nr"]
            if name not in tree
        ]
        if missing:
            raise ValueError(f"ROOT trackstates is missing required branches: {missing}")
        arrays = tree.arrays(required, entry_stop=max_track_entries, library="ak")
        for branch in ("err_x_hit", "err_y_hit"):
            if branch not in tree:
                raise ValueError(f"ROOT trackstates is missing required branch: {branch}")
        err_x_jagged = tree["err_x_hit"].array(
            entry_stop=max_track_entries, library="ak"
        )
        err_y_jagged = tree["err_y_hit"].array(
            entry_stop=max_track_entries, library="ak"
        )
        counts = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
        event_values = ak.to_numpy(
            tree["event_nr"].array(entry_stop=max_track_entries, library="ak")
        )
        track_values = ak.to_numpy(
            tree["track_nr"].array(entry_stop=max_track_entries, library="ak")
        )
    if len(event_values) != len(counts) or len(track_values) != len(counts):
        raise ValueError("event_nr/track_nr entries do not align with jagged state entries.")
    flat = {name: _flatten_jagged(arrays[name]) for name in required}
    flat["err_x_hit"], flat["err_y_hit"] = expand_measurement_errors(
        arrays["l_x_hit"], err_x_jagged, err_y_jagged
    )
    flat["event_nr"] = np.repeat(event_values, counts)
    flat["track_nr"] = np.repeat(track_values, counts)
    return _helix_inputs_from_flat(flat, id_to_idx)


def _load_nominal_states_pyroot(
    path: str | Path,
    id_to_idx: dict[tuple[int, int, int], int],
    max_track_entries: int | None = None,
) -> HelixInputs:
    """Read only requested track entries without decompressing whole uproot baskets."""
    import ROOT

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError(f"Could not open ROOT file: {path}")
    try:
        tree = root_file.Get("trackstates")
        if tree is None:
            raise ValueError(f"ROOT file {path} has no 'trackstates' tree.")
        per_state = (
            "volume_id",
            "layer_id",
            "module_id",
            "pathLength",
            "l_x_hit",
            "l_y_hit",
            "eLOC0_smt",
            "eLOC1_smt",
            "g_x_smt",
            "g_y_smt",
            "g_z_smt",
            "px_smt",
            "py_smt",
            "pz_smt",
        )
        required = per_state + ("err_x_hit", "err_y_hit", "event_nr", "track_nr")
        missing = [name for name in required if tree.GetBranch(name) is None]
        if missing:
            raise ValueError(f"ROOT trackstates is missing required branches: {missing}")
        flat_lists = {name: [] for name in per_state}
        flat_lists.update({"err_x_hit": [], "err_y_hit": [], "event_nr": [], "track_nr": []})
        entry_count = int(tree.GetEntries())
        if max_track_entries is not None:
            entry_count = min(entry_count, int(max_track_entries))
        for entry in range(entry_count):
            tree.GetEntry(entry)
            state_values = {name: list(getattr(tree, name)) for name in per_state}
            n_states = len(state_values["volume_id"])
            for name, values in state_values.items():
                if len(values) != n_states:
                    raise ValueError(
                        f"Track entry {entry} has {n_states} states but "
                        f"{len(values)} {name} values."
                    )
                flat_lists[name].extend(values)
            local_x = _float_array(state_values["l_x_hit"])
            measurement_mask = np.isfinite(local_x)
            n_measurements = int(measurement_mask.sum())
            for name in ("err_x_hit", "err_y_hit"):
                values = _float_array(list(getattr(tree, name)))
                if len(values) == n_states:
                    expanded = values
                elif len(values) == n_measurements:
                    expanded = np.full(n_states, np.nan, dtype=np.float64)
                    expanded[measurement_mask] = values
                else:
                    raise ValueError(
                        f"Track entry {entry} has {n_states} states and "
                        f"{n_measurements} measurements but {len(values)} {name} values."
                    )
                flat_lists[name].extend(expanded)
            flat_lists["event_nr"].extend([int(tree.event_nr)] * n_states)
            flat_lists["track_nr"].extend([int(tree.track_nr)] * n_states)
        flat = {name: np.asarray(values) for name, values in flat_lists.items()}
        return _helix_inputs_from_flat(flat, id_to_idx)
    finally:
        root_file.Close()


def load_nominal_states(
    path: str | Path,
    id_to_idx: dict[tuple[int, int, int], int],
    max_track_entries: int | None = None,
) -> HelixInputs:
    """Load measured nominal states, preferring entry-wise PyROOT I/O."""
    try:
        import ROOT  # noqa: F401
    except ImportError:
        return _load_nominal_states_uproot(path, id_to_idx, max_track_entries)
    return _load_nominal_states_pyroot(path, id_to_idx, max_track_entries)


def _sinc(value: torch.Tensor) -> torch.Tensor:
    """Return sin(value) / value with a differentiable zero limit."""
    return torch.sinc(value / torch.pi)


def helix_point(track: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Evaluate signed-curvature helices at transverse arc length ``s``."""
    x0, y0, z0, psi0, tanl, rho = track.unbind(-1)
    angle = rho * s
    half_sinc = _sinc(angle / 2.0)
    x = x0 + s * torch.cos(psi0 + angle / 2.0) * half_sinc
    y = y0 + s * torch.sin(psi0 + angle / 2.0) * half_sinc
    z = z0 + tanl * s
    return torch.stack((x, y, z), dim=-1)


def helix_tangent(track: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Differentiate :func:`helix_point` with respect to transverse arc length."""
    _x0, _y0, _z0, psi0, tanl, rho = track.unbind(-1)
    psi = psi0 + rho * s
    return torch.stack(
        (torch.cos(psi), torch.sin(psi), tanl * torch.ones_like(s)), dim=-1
    )


def accept_newton_candidate(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    valid: torch.Tensor,
    candidate_ok: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_valid = valid & candidate_ok
    return torch.where(updated_valid, candidate, previous), updated_valid


def closest_helix_points(
    track: torch.Tensor, points: torch.Tensor, n_newton: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return closest points on each helix to reference global points."""
    delta = points - track[..., :3]
    s = delta[..., 0] * torch.cos(track[..., 3]) + delta[..., 1] * torch.sin(
        track[..., 3]
    )
    valid = torch.isfinite(s)
    s = torch.where(valid, s, torch.zeros_like(s))
    for _ in range(n_newton):
        position = helix_point(track, s)
        tangent = helix_tangent(track, s)
        displacement = position - points
        rho = track[..., 5]
        psi = track[..., 3] + rho * s
        second = torch.stack(
            (
                -rho * torch.sin(psi),
                rho * torch.cos(psi),
                torch.zeros_like(rho),
            ),
            dim=-1,
        )
        first_derivative = torch.sum(displacement * tangent, dim=-1)
        second_derivative = torch.sum(
            tangent * tangent + displacement * second, dim=-1
        )
        safe = torch.where(
            second_derivative.abs() > 1e-9,
            second_derivative,
            torch.where(
                second_derivative >= 0,
                torch.full_like(second_derivative, 1e-9),
                torch.full_like(second_derivative, -1e-9),
            ),
        )
        candidate = s - first_derivative / safe
        candidate_ok = torch.isfinite(candidate) & (candidate.abs() <= S_MAX)
        s, valid = accept_newton_candidate(s, candidate, valid, candidate_ok)
    fitted = helix_point(track, s)
    valid = valid & torch.isfinite(fitted).all(dim=-1)
    fitted = torch.where(valid.unsqueeze(-1), fitted, track[..., :3])
    return fitted, s, valid


def intersect_plane(
    track: torch.Tensor,
    center: torch.Tensor,
    normal: torch.Tensor,
    n_newton: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect helices with detector planes using the joint solver convention."""
    tangent0 = helix_tangent(track, torch.zeros_like(track[..., 0]))
    denominator0 = torch.sum(tangent0 * normal, dim=-1)
    straight_valid = denominator0.abs() >= PLANE_PARALLEL_EPS
    safe_denominator = torch.where(
        straight_valid,
        denominator0,
        torch.where(
            denominator0 >= 0,
            torch.ones_like(denominator0),
            -torch.ones_like(denominator0),
        ),
    )
    s = torch.sum((center - track[..., :3]) * normal, dim=-1) / safe_denominator
    tangent_at_initial = helix_tangent(track, s)
    initially_valid = (
        straight_valid
        & torch.isfinite(s)
        & (
            torch.sum(tangent_at_initial * normal, dim=-1).abs()
            >= PLANE_PARALLEL_EPS
        )
    )
    s = torch.where(initially_valid, s, torch.zeros_like(s))
    iteration_valid = initially_valid
    for _ in range(n_newton):
        point = helix_point(track, s)
        tangent = helix_tangent(track, s)
        residual = torch.sum((point - center) * normal, dim=-1)
        derivative = torch.sum(tangent * normal, dim=-1)
        derivative_safe = torch.where(
            derivative.abs() >= PLANE_PARALLEL_EPS,
            derivative,
            torch.where(
                derivative >= 0,
                torch.full_like(derivative, PLANE_PARALLEL_EPS),
                torch.full_like(derivative, -PLANE_PARALLEL_EPS),
            ),
        )
        candidate = s - residual / derivative_safe
        candidate_ok = (
            torch.isfinite(candidate)
            & (candidate.abs() <= S_MAX)
            & (derivative.abs() >= PLANE_PARALLEL_EPS)
        )
        s, iteration_valid = accept_newton_candidate(
            s, candidate, iteration_valid, candidate_ok
        )
    point = helix_point(track, s)
    final_residual = torch.sum((point - center) * normal, dim=-1).abs()
    finite = torch.isfinite(s) & torch.isfinite(point).all(dim=-1)
    valid = (
        iteration_valid
        & finite
        & (s.abs() <= S_MAX)
        & (final_residual <= PLANE_RESIDUAL_TOLERANCE)
    )
    point = torch.where(
        valid.unsqueeze(-1), point, helix_point(track, torch.zeros_like(s))
    )
    s = torch.where(valid, s, torch.zeros_like(s))
    return s, point, valid


def validate_geometry_consistency(
    inputs: HelixInputs, geometry: NominalGeometry, tolerance: float = 1e-3
) -> dict[str, float]:
    """Verify that nominal global and local SMT states use the supplied geometry."""
    centers = geometry.centers[inputs.hit_sensor_ids].to(
        dtype=inputs.hit_global_positions.dtype
    )
    rotations = geometry.rotations[inputs.hit_sensor_ids].to(
        dtype=inputs.hit_global_positions.dtype
    )
    delta = inputs.hit_global_positions - centers
    local3 = torch.matmul(rotations.transpose(-1, -2), delta.unsqueeze(-1)).squeeze(-1)
    local_difference = local3[:, :2] - inputs.smt_local
    finite_local = torch.isfinite(local_difference)
    if not finite_local.any():
        raise ValueError("No finite SMT local coordinates are available for geometry validation.")
    max_local_error = float(local_difference[finite_local].abs().max())
    plane_distance = torch.sum(delta * rotations[:, :, 2], dim=-1).abs()
    finite_plane = torch.isfinite(plane_distance)
    if not finite_plane.any():
        raise ValueError("No finite SMT plane distances are available for geometry validation.")
    max_plane_distance = float(plane_distance[finite_plane].max())
    if max_local_error > tolerance or max_plane_distance > tolerance:
        raise ValueError(
            "Nominal geometry is inconsistent with SMT states: "
            f"max local error={max_local_error:.6g} mm, "
            f"max plane distance={max_plane_distance:.6g} mm, "
            f"tolerance={tolerance:.6g} mm."
        )
    return {
        "max_local_error_mm": max_local_error,
        "max_plane_distance_mm": max_plane_distance,
        "tolerance_mm": float(tolerance),
    }


def evaluate_helix_on_surfaces(
    track_parameters: torch.Tensor,
    batches: TrackBatches,
    geometry: NominalGeometry,
) -> HelixEvaluation:
    """Propagate frozen helices to the exact nominal surface of every SMT state."""
    device = track_parameters.device
    dtype = track_parameters.dtype
    n_hits = batches.g_smt.shape[1]
    tracks = track_parameters[:, None, :].expand(-1, n_hits, -1)
    sensor_idx = batches.sensor_idx.to(device=device)
    centers = geometry.centers.to(device=device, dtype=dtype)[sensor_idx]
    rotations = geometry.rotations.to(device=device, dtype=dtype)[sensor_idx]
    normals = rotations[..., :, 2]
    path_length, intersections, valid = intersect_plane(
        tracks, centers, normals, n_newton=12
    )
    local3 = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)
    tangent = helix_tangent(tracks, path_length)
    smt_direction = batches.p_smt.to(device=device, dtype=dtype)
    tangent_unit = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp(
        min=1e-12
    )
    smt_unit = smt_direction / torch.linalg.norm(
        smt_direction, dim=-1, keepdim=True
    ).clamp(min=1e-12)
    cosine = torch.sum(tangent_unit * smt_unit, dim=-1).clamp(-1.0, 1.0)
    tangent_angle = torch.acos(cosine)
    global_difference = intersections - batches.g_smt.to(device=device, dtype=dtype)
    local_difference = local3[..., :2] - batches.smt_local.to(device=device, dtype=dtype)
    valid = valid & batches.hit_mask.to(device=device)
    return HelixEvaluation(
        path_length=path_length,
        global_position=intersections,
        local_position=local3[..., :2],
        global_difference=global_difference,
        local_difference=local_difference,
        tangent_angle=tangent_angle,
        intersection_valid=valid,
    )


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "p95": float("nan")}
    return {
        "rmse": float(torch.sqrt(torch.mean(values.square())).cpu()),
        "mae": float(torch.mean(values.abs()).cpu()),
        "p95": float(torch.quantile(values.abs(), 0.95).cpu()),
    }


def summarize_evaluation(
    evaluation: HelixEvaluation, selection_mask: torch.Tensor
) -> dict[str, float | int]:
    """Summarize global, local, direction, and intersection diagnostics."""
    mask = selection_mask.to(evaluation.intersection_valid.device)
    requested = mask
    valid = requested & evaluation.intersection_valid
    count = int(requested.sum().item())
    valid_count = int(valid.sum().item())
    if count == 0:
        raise ValueError("The requested evaluation selection is empty.")
    global_distance = torch.linalg.norm(evaluation.global_difference[valid], dim=-1)
    global_stats = _distribution(global_distance)
    local_values = evaluation.local_difference[valid]
    local_values = local_values[torch.isfinite(local_values)]
    local_stats = _distribution(local_values)
    local0_stats = _distribution(evaluation.local_difference[..., 0][valid])
    local1_stats = _distribution(evaluation.local_difference[..., 1][valid])
    angle_stats = _distribution(evaluation.tangent_angle[valid])
    return {
        "count": count,
        "valid_count": valid_count,
        "global_rmse_mm": global_stats["rmse"],
        "global_mae_mm": global_stats["mae"],
        "global_p95_mm": global_stats["p95"],
        "local_rmse_mm": local_stats["rmse"],
        "local_mae_mm": local_stats["mae"],
        "local_p95_mm": local_stats["p95"],
        "local0_rmse_mm": local0_stats["rmse"],
        "local0_mae_mm": local0_stats["mae"],
        "local0_p95_mm": local0_stats["p95"],
        "local1_rmse_mm": local1_stats["rmse"],
        "local1_mae_mm": local1_stats["mae"],
        "local1_p95_mm": local1_stats["p95"],
        "direction_angle_rms_rad": angle_stats["rmse"],
        "direction_angle_mae_rad": angle_stats["mae"],
        "direction_angle_p95_rad": angle_stats["p95"],
        "intersection_failure_rate": 1.0 - valid_count / count,
    }


def _initial_parameters_from_points(
    points: torch.Tensor, directions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    first_index = mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(len(points), device=points.device)
    first = points[rows, first_index]
    direction = directions[rows, first_index]
    transverse = torch.linalg.norm(direction[:, :2], dim=1).clamp(min=1e-6)
    psi = torch.atan2(direction[:, 1], direction[:, 0])
    tanl = direction[:, 2] / transverse
    rho = torch.zeros((len(points), 1), device=points.device, dtype=points.dtype)
    return torch.cat((first, psi[:, None], tanl[:, None], rho), dim=1)


def initialize_helices(
    batches: TrackBatches, device: str | torch.device = "cpu", dtype=torch.float64
) -> torch.Tensor:
    points = batches.g_smt.to(device=device, dtype=dtype)
    directions = batches.p_smt.to(device=device, dtype=dtype)
    fit_mask = batches.fit_mask.to(device=device)
    return _initial_parameters_from_points(points, directions, fit_mask)


def seed_helices(
    batches: TrackBatches,
    n_iter: int = 200,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Fit helices only to non-holdout SMT points."""
    points = batches.g_smt.to(device=device, dtype=dtype)
    mask = batches.fit_mask.to(device=device)
    n_tracks, n_hits, _ = points.shape
    initial = initialize_helices(batches, device=device, dtype=dtype)
    linear = initial[:, :5].detach().clone().requires_grad_(True)
    rho_scaled = (initial[:, 5:6] / 1e-4).detach().clone().requires_grad_(True)
    optimizer = optim.Adam(
        [
            {"params": [linear], "lr": 0.5},
            {"params": [rho_scaled], "lr": 0.5},
        ]
    )
    for _ in range(n_iter):
        optimizer.zero_grad()
        track_parameters = torch.cat((linear, rho_scaled * 1e-4), dim=1)
        expanded = track_parameters[:, None, :].expand(-1, n_hits, -1)
        with torch.no_grad():
            _fit, path_length, valid = closest_helix_points(
                expanded.detach(), points
            )
        predicted = helix_point(expanded, path_length)
        effective = mask & valid
        squared_distance = torch.sum((predicted - points) ** 2, dim=-1)
        loss = torch.sum(
            torch.where(effective, squared_distance, torch.zeros_like(squared_distance))
        ) / effective.sum().clamp(min=1)
        loss.backward()
        optimizer.step()
    return torch.cat((linear, rho_scaled * 1e-4), dim=1).detach()


def fit_helices_to_measurements(
    batches: TrackBatches,
    geometry: NominalGeometry,
    initial_parameters: torch.Tensor,
    n_iter: int = 200,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Refit helices using only non-holdout measurement components."""
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    initial = initial_parameters.to(device=device, dtype=dtype)
    linear = initial[:, :5].detach().clone().requires_grad_(True)
    rho_scaled = (initial[:, 5:6] / 1e-4).detach().clone().requires_grad_(True)
    optimizer = optim.Adam(
        [
            {"params": [linear], "lr": 0.05},
            {"params": [rho_scaled], "lr": 0.05},
        ]
    )
    sensor_idx = batches.sensor_idx.to(device=device)
    centers = geometry.centers.to(device=device, dtype=dtype)[sensor_idx]
    rotations = geometry.rotations.to(device=device, dtype=dtype)[sensor_idx]
    normals = rotations[..., :, 2]
    measured = batches.measured_local.to(device=device, dtype=dtype)
    variance = batches.measurement_variance.to(device=device, dtype=dtype)
    fit_mask = batches.fit_mask.to(device=device)
    loc1_valid = batches.loc1_valid.to(device=device)
    n_hits = measured.shape[1]

    for _ in range(n_iter):
        optimizer.zero_grad()
        tracks = torch.cat((linear, rho_scaled * 1e-4), dim=1)
        expanded = tracks[:, None, :].expand(-1, n_hits, -1)
        _path, intersections, valid = intersect_plane(
            expanded, centers, normals, n_newton=12
        )
        local3 = torch.matmul(
            rotations.transpose(-1, -2),
            (intersections - centers).unsqueeze(-1),
        ).squeeze(-1)
        residual = measured - local3[..., :2]
        valid_hit = fit_mask & valid
        component_mask = torch.stack(
            (valid_hit, valid_hit & loc1_valid), dim=-1
        )
        if not component_mask.any():
            raise ValueError("No valid measurement components remain for helix fitting.")
        weighted = residual.square() / variance.clamp_min(EPS_VARIANCE)
        loss = torch.sum(
            torch.where(component_mask, weighted, torch.zeros_like(weighted))
        ) / component_mask.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite measurement helix loss.")
        loss.backward()
        optimizer.step()

    return torch.cat((linear, rho_scaled * 1e-4), dim=1).detach()


def build_track_batches(
    inputs: HelixInputs,
    holdout_layers,
    min_fit_hits: int = 8,
    min_fit_span: float = 300.0,
    max_tracks: int = 0,
    context_hits_before: int = 0,
    context_hits_after: int = 0,
) -> TrackBatches:
    """Build one local fit window for every held-out measurement state."""
    if context_hits_before < 0 or context_hits_after < 0:
        raise ValueError("Context hit counts must be non-negative.")
    holdout_layers = {tuple(map(int, layer)) for layer in holdout_layers}
    positions = inputs.hit_global_positions.detach().cpu().numpy()
    event_ids = inputs.hit_event_ids.detach().cpu().numpy()
    track_ids = inputs.hit_track_ids.detach().cpu().numpy()
    volume_ids = inputs.hit_volume_ids.detach().cpu().numpy()
    layer_ids = inputs.hit_layer_ids.detach().cpu().numpy()
    module_ids = inputs.hit_module_ids.detach().cpu().numpy()
    sensor_ids = inputs.hit_sensor_ids.detach().cpu().numpy()
    radius = np.hypot(positions[:, 0], positions[:, 1])
    supplied_path = (
        inputs.hit_path_length.detach().cpu().numpy()
        if inputs.hit_path_length is not None
        else None
    )
    order = np.lexsort(
        (
            sensor_ids,
            module_ids,
            layer_ids,
            volume_ids,
            np.abs(positions[:, 2]),
            radius,
            track_ids,
            event_ids,
        )
    )
    ordered_events = event_ids[order]
    ordered_tracks = track_ids[order]
    boundaries = np.flatnonzero(
        (ordered_events[1:] != ordered_events[:-1])
        | (ordered_tracks[1:] != ordered_tracks[:-1])
    ) + 1
    selected = []
    for indices in np.split(order, boundaries):
        if supplied_path is not None:
            indices = indices[np.argsort(supplied_path[indices], kind="stable")]
            path_values = supplied_path
        else:
            first = int(indices[0])
            direction = inputs.hit_directions[first].detach().cpu().numpy()
            transverse = max(float(np.linalg.norm(direction[:2])), 1e-6)
            unit_xy = direction[:2] / transverse
            proxy = (positions[indices, :2] - positions[first, :2]) @ unit_xy
            indices = indices[np.argsort(proxy, kind="stable")]
            path_values = np.full(len(positions), np.nan, dtype=np.float64)
            path_values[indices] = np.sort(proxy)
        all_fit_indices = [
            int(i)
            for i in indices
            if (int(volume_ids[i]), int(layer_ids[i])) not in holdout_layers
        ]
        holdout_indices = [
            int(i)
            for i in indices
            if (int(volume_ids[i]), int(layer_ids[i])) in holdout_layers
        ]
        if len(all_fit_indices) < min_fit_hits or not holdout_indices:
            continue
        key = (int(event_ids[indices[0]]), int(track_ids[indices[0]]))
        for holdout_index in holdout_indices:
            holdout_path = path_values[holdout_index]
            before = [
                index for index in all_fit_indices if path_values[index] < holdout_path
            ]
            after = [
                index for index in all_fit_indices if path_values[index] > holdout_path
            ]
            if context_hits_before > 0:
                before = before[-context_hits_before:]
            if context_hits_after > 0:
                after = after[:context_hits_after]
            fit_indices = before + after
            if len(fit_indices) < min_fit_hits:
                continue
            fit_paths = path_values[fit_indices]
            fit_span = float(np.max(fit_paths) - np.min(fit_paths))
            if fit_span < min_fit_span:
                continue
            row_indices = sorted(
                fit_indices + [holdout_index], key=lambda index: path_values[index]
            )
            selected.append((key, row_indices, fit_span))
    if max_tracks > 0:
        selected = selected[:max_tracks]
    if not selected:
        raise ValueError("No tracks satisfy the helix validation selection.")

    n_tracks = len(selected)
    max_hits = max(len(indices) for _key, indices, _span in selected)
    dtype = inputs.hit_global_positions.dtype
    smt_local = torch.full((n_tracks, max_hits, 2), torch.nan, dtype=inputs.smt_local.dtype)
    measured_local = torch.zeros(
        (n_tracks, max_hits, 2), dtype=inputs.measured_local.dtype
    )
    measurement_variance = torch.ones(
        (n_tracks, max_hits, 2), dtype=inputs.measurement_variance.dtype
    )
    loc1_valid = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    sensor_idx = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    hit_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    fit_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    holdout_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    g_smt = torch.zeros((n_tracks, max_hits, 3), dtype=dtype)
    p_smt = torch.zeros((n_tracks, max_hits, 3), dtype=inputs.hit_directions.dtype)
    volume_id = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    layer_id = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    module_id = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    event_track_ids = torch.zeros((n_tracks, 2), dtype=torch.long)
    fit_span = torch.zeros(n_tracks, dtype=dtype)
    for row, (key, indices, span) in enumerate(selected):
        count = len(indices)
        index = torch.tensor(indices, dtype=torch.long)
        sensor_idx[row, :count] = inputs.hit_sensor_ids[index]
        measured_local[row, :count] = inputs.measured_local[index]
        measurement_variance[row, :count] = inputs.measurement_variance[index]
        loc1_valid[row, :count] = inputs.loc1_valid[index]
        smt_local[row, :count] = inputs.smt_local[index]
        g_smt[row, :count] = inputs.hit_global_positions[index]
        p_smt[row, :count] = inputs.hit_directions[index]
        volume_id[row, :count] = inputs.hit_volume_ids[index]
        layer_id[row, :count] = inputs.hit_layer_ids[index]
        module_id[row, :count] = inputs.hit_module_ids[index]
        hit_mask[row, :count] = True
        is_holdout = torch.tensor(
            [
                (int(volume_ids[i]), int(layer_ids[i])) in holdout_layers
                for i in indices
            ],
            dtype=torch.bool,
        )
        holdout_mask[row, :count] = is_holdout
        fit_mask[row, :count] = ~is_holdout
        event_track_ids[row] = torch.tensor(key)
        fit_span[row] = span
    return TrackBatches(
        measured_local=measured_local,
        measurement_variance=measurement_variance,
        loc1_valid=loc1_valid,
        smt_local=smt_local,
        sensor_idx=sensor_idx,
        hit_mask=hit_mask,
        fit_mask=fit_mask,
        holdout_mask=holdout_mask,
        g_smt=g_smt,
        p_smt=p_smt,
        volume_id=volume_id,
        layer_id=layer_id,
        module_id=module_id,
        event_track_ids=event_track_ids,
        fit_span=fit_span,
    )


def _comparison_metrics(
    initial: HelixEvaluation,
    fitted: HelixEvaluation,
    selection_mask: torch.Tensor,
) -> dict:
    initial_valid = initial.intersection_valid.cpu()
    fitted_valid = fitted.intersection_valid.cpu()
    requested = selection_mask.cpu()
    common = requested & initial_valid & fitted_valid
    requested_count = int(requested.sum())
    if requested_count == 0:
        raise ValueError("The requested comparison selection is empty.")
    empty_metrics = {
        "count": 0,
        "valid_count": 0,
        "global_rmse_mm": None,
        "global_mae_mm": None,
        "global_p95_mm": None,
        "local_rmse_mm": None,
        "local_mae_mm": None,
        "local_p95_mm": None,
        "local0_rmse_mm": None,
        "local0_mae_mm": None,
        "local0_p95_mm": None,
        "local1_rmse_mm": None,
        "local1_mae_mm": None,
        "local1_p95_mm": None,
        "direction_angle_rms_rad": None,
        "direction_angle_mae_rad": None,
        "direction_angle_p95_rad": None,
        "intersection_failure_rate": None,
    }
    if common.any():
        initial_metrics = summarize_evaluation(initial, common)
        fitted_metrics = summarize_evaluation(fitted, common)
    else:
        initial_metrics = dict(empty_metrics)
        fitted_metrics = dict(empty_metrics)
    return {
        "requested_count": requested_count,
        "common_valid_count": int(common.sum()),
        "initial_intersection_failure_rate": 1.0
        - float((requested & initial_valid).sum()) / requested_count,
        "fitted_intersection_failure_rate": 1.0
        - float((requested & fitted_valid).sum()) / requested_count,
        "initial": initial_metrics,
        "fitted": fitted_metrics,
    }


def _validation_pass(
    holdout: dict,
    holdout_by_layer: dict[str, dict],
    max_failure_rate: float,
    layer_local_limits: dict[str, tuple[float, float]] | None = None,
    max_direction_angle_rms: float | None = None,
) -> tuple[bool, list[str]]:
    reasons = []
    scopes = [("holdout overall", holdout)] + [
        (f"holdout layer {layer}", metrics)
        for layer, metrics in holdout_by_layer.items()
    ]
    comparisons = (
        ("global RMSE", "global_rmse_mm"),
        ("local RMSE", "local_rmse_mm"),
        ("direction-angle RMS", "direction_angle_rms_rad"),
    )
    for scope, metrics in scopes:
        if metrics["common_valid_count"] < 1:
            reasons.append(f"{scope} has no common valid intersection")
        else:
            initial = metrics["initial"]
            fitted = metrics["fitted"]
            for label, key in comparisons:
                before = initial[key]
                after = fitted[key]
                if (
                    before is None
                    or after is None
                    or not np.isfinite(before)
                    or not np.isfinite(after)
                    or not after < before
                ):
                    reasons.append(
                        f"{scope} {label} did not improve ({before} -> {after})"
                    )
        failure_rate = float(metrics["fitted_intersection_failure_rate"])
        if failure_rate > max_failure_rate:
            reasons.append(
                f"{scope} intersection failure rate {failure_rate:.2%} exceeds "
                f"{max_failure_rate:.2%}"
            )
    layer_local_limits = layer_local_limits or {}
    for layer, metrics in holdout_by_layer.items():
        if metrics["common_valid_count"] < 1:
            continue
        fitted = metrics["fitted"]
        limits = layer_local_limits.get(layer)
        if limits is not None:
            for component, limit in zip(("LOC0", "LOC1"), limits):
                value = fitted.get(f"local{component[-1]}_rmse_mm")
                if value is not None and np.isfinite(value) and value > limit:
                    reasons.append(
                        f"holdout layer {layer} {component} RMSE {value:.6g} mm "
                        f"exceeds {limit:.6g} mm"
                    )
        if max_direction_angle_rms is not None:
            value = fitted.get("direction_angle_rms_rad")
            if value is not None and np.isfinite(value) and value > max_direction_angle_rms:
                reasons.append(
                    f"holdout layer {layer} direction-angle RMS {value:.6g} rad "
                    f"exceeds {max_direction_angle_rms:.6g} rad"
                )
    return not reasons, reasons


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def build_comparison_frame(
    batches: TrackBatches,
    initial_parameters: torch.Tensor,
    fitted_parameters: torch.Tensor,
    initial: HelixEvaluation,
    fitted: HelixEvaluation,
) -> pd.DataFrame:
    mask = _as_numpy(batches.hit_mask).astype(bool)
    n_tracks, n_hits = mask.shape
    track_rows = np.repeat(np.arange(n_tracks), n_hits).reshape(n_tracks, n_hits)[mask]
    event_track = _as_numpy(batches.event_track_ids)
    initial_params = _as_numpy(initial_parameters)[track_rows]
    fitted_params = _as_numpy(fitted_parameters)[track_rows]

    def selected(tensor):
        return _as_numpy(tensor)[mask]

    smt_global = selected(batches.g_smt)
    smt_local = selected(batches.smt_local)
    initial_global = selected(initial.global_position)
    fitted_global = selected(fitted.global_position)
    initial_local = selected(initial.local_position)
    fitted_local = selected(fitted.local_position)
    initial_global_diff = selected(initial.global_difference)
    fitted_global_diff = selected(fitted.global_difference)
    initial_local_diff = selected(initial.local_difference)
    fitted_local_diff = selected(fitted.local_difference)
    frame = pd.DataFrame(
        {
            "split": np.where(selected(batches.holdout_mask), "holdout", "fit"),
            "event_nr": event_track[track_rows, 0],
            "track_nr": event_track[track_rows, 1],
            "volume_id": selected(batches.volume_id),
            "layer_id": selected(batches.layer_id),
            "module_id": selected(batches.module_id),
            "sensor_idx": selected(batches.sensor_idx),
            "g_x_smt": smt_global[:, 0],
            "g_y_smt": smt_global[:, 1],
            "g_z_smt": smt_global[:, 2],
            "eLOC0_smt": smt_local[:, 0],
            "eLOC1_smt": smt_local[:, 1],
            "initial_g_x": initial_global[:, 0],
            "initial_g_y": initial_global[:, 1],
            "initial_g_z": initial_global[:, 2],
            "fitted_g_x": fitted_global[:, 0],
            "fitted_g_y": fitted_global[:, 1],
            "fitted_g_z": fitted_global[:, 2],
            "initial_dg_x": initial_global_diff[:, 0],
            "initial_dg_y": initial_global_diff[:, 1],
            "initial_dg_z": initial_global_diff[:, 2],
            "fitted_dg_x": fitted_global_diff[:, 0],
            "fitted_dg_y": fitted_global_diff[:, 1],
            "fitted_dg_z": fitted_global_diff[:, 2],
            "initial_LOC0": initial_local[:, 0],
            "initial_LOC1": initial_local[:, 1],
            "fitted_LOC0": fitted_local[:, 0],
            "fitted_LOC1": fitted_local[:, 1],
            "initial_dLOC0": initial_local_diff[:, 0],
            "initial_dLOC1": initial_local_diff[:, 1],
            "fitted_dLOC0": fitted_local_diff[:, 0],
            "fitted_dLOC1": fitted_local_diff[:, 1],
            "initial_direction_angle_rad": selected(initial.tangent_angle),
            "fitted_direction_angle_rad": selected(fitted.tangent_angle),
            "initial_intersection_valid": selected(initial.intersection_valid),
            "fitted_intersection_valid": selected(fitted.intersection_valid),
        }
    )
    parameter_names = ("x0", "y0", "z0", "psi0", "tanl", "rho")
    for index, name in enumerate(parameter_names):
        frame[f"initial_{name}"] = initial_params[:, index]
        frame[f"fitted_{name}"] = fitted_params[:, index]
    return frame


def write_comparison_csv(path: str | Path, frame: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_helix_validation(config: HelixValidationConfig | None = None) -> dict:
    config = config or HelixValidationConfig()
    holdout_layers = parse_layers(config.holdout_layers)
    if config.min_fit_hits < 2:
        raise ValueError("min_fit_hits must be at least 2.")
    if config.min_fit_span < 0:
        raise ValueError("min_fit_span must be non-negative.")
    if config.fit_iters < 1:
        raise ValueError("fit_iters must be at least 1.")
    if config.max_tracks < 0:
        raise ValueError("max_tracks must be non-negative.")
    if config.context_hits_before < 0 or config.context_hits_after < 0:
        raise ValueError("context hit counts must be non-negative.")
    if config.geometry_tolerance < 0:
        raise ValueError("geometry_tolerance must be non-negative.")
    if not 0.0 <= config.max_intersection_failure_rate <= 1.0:
        raise ValueError("max_intersection_failure_rate must be between 0 and 1.")
    for value, name in (
        (config.max_pixel_local_rmse, "max_pixel_local_rmse"),
        (config.max_strip_loc0_rmse, "max_strip_loc0_rmse"),
        (config.max_strip_loc1_rmse, "max_strip_loc1_rmse"),
        (config.max_direction_angle_rms, "max_direction_angle_rms"),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    for input_path, label in (
        (config.states, "nominal trackstates"),
        (config.geometry, "current geometry"),
        (config.nominal_full_geometry, "nominal full geometry"),
    ):
        if not Path(input_path).exists():
            raise FileNotFoundError(f"{label} does not exist: {input_path}")

    use_cuda = bool(config.gpu and torch.cuda.is_available())
    if config.gpu and not use_cuda:
        print("[warn] --gpu requested but CUDA is unavailable; using CPU float64.")
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    torch.manual_seed(config.rng_seed)
    np.random.seed(config.rng_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.rng_seed)

    geometry = load_working_geometry(
        config.geometry, config.nominal_full_geometry
    )
    entry_stop = config.max_tracks * 4 if config.max_tracks > 0 else None
    entry_description = "all" if entry_stop is None else str(entry_stop)
    print(
        f"Loading nominal SMT states from {config.states} "
        f"(ROOT entry_stop={entry_description})..."
    )
    inputs = load_nominal_states(
        config.states, geometry.id_to_idx, max_track_entries=entry_stop
    )
    geometry_report = validate_geometry_consistency(
        inputs, geometry, tolerance=config.geometry_tolerance
    )
    batches = build_track_batches(
        inputs,
        holdout_layers,
        min_fit_hits=config.min_fit_hits,
        min_fit_span=config.min_fit_span,
        max_tracks=config.max_tracks,
        context_hits_before=config.context_hits_before,
        context_hits_after=config.context_hits_after,
    )
    initial_parameters = initialize_helices(batches, device=device, dtype=dtype)
    seeded_parameters = seed_helices(
        batches, n_iter=config.fit_iters, device=device, dtype=dtype
    )
    fitted_parameters = fit_helices_to_measurements(
        batches,
        geometry,
        seeded_parameters,
        n_iter=config.fit_iters,
        device=device,
        dtype=dtype,
    )
    initial = evaluate_helix_on_surfaces(
        initial_parameters, batches, geometry
    )
    fitted = evaluate_helix_on_surfaces(fitted_parameters, batches, geometry)
    fit_metrics = _comparison_metrics(initial, fitted, batches.fit_mask)
    holdout_metrics = _comparison_metrics(initial, fitted, batches.holdout_mask)
    holdout_by_layer = {}
    for volume, layer in holdout_layers:
        mask = (
            batches.holdout_mask
            & (batches.volume_id == volume)
            & (batches.layer_id == layer)
        )
        if not mask.any():
            raise ValueError(f"Holdout layer {volume}:{layer} has no selected SMT states.")
        holdout_by_layer[f"{volume}:{layer}"] = _comparison_metrics(
            initial, fitted, mask
        )
    layer_local_limits = {}
    for volume, layer in holdout_layers:
        if volume in {16, 17, 18}:
            limits = (
                config.max_pixel_local_rmse,
                config.max_pixel_local_rmse,
            )
        else:
            limits = (
                config.max_strip_loc0_rmse,
                config.max_strip_loc1_rmse,
            )
        layer_local_limits[f"{volume}:{layer}"] = limits
    passed, failure_reasons = _validation_pass(
        holdout_metrics,
        holdout_by_layer,
        config.max_intersection_failure_rate,
        layer_local_limits=layer_local_limits,
        max_direction_angle_rms=config.max_direction_angle_rms,
    )
    summary = {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "seed_source": "smt",
        "fit_source": "measurement",
        "validation_reference": "smt",
        "comparison_baseline": "unrefined_smt_initialization",
        "config": {
            "states": str(config.states),
            "geometry": str(config.geometry),
            "nominal_full_geometry": str(config.nominal_full_geometry),
            "holdout_layers": [f"{volume}:{layer}" for volume, layer in holdout_layers],
            "min_fit_hits": config.min_fit_hits,
            "min_fit_span_mm": config.min_fit_span,
            "max_tracks": config.max_tracks,
            "root_entry_stop": entry_stop,
            "fit_iters": config.fit_iters,
            "context_hits_before": config.context_hits_before,
            "context_hits_after": config.context_hits_after,
            "device": str(device),
            "rng_seed": config.rng_seed,
            "max_intersection_failure_rate": config.max_intersection_failure_rate,
            "max_pixel_local_rmse_mm": config.max_pixel_local_rmse,
            "max_strip_loc0_rmse_mm": config.max_strip_loc0_rmse,
            "max_strip_loc1_rmse_mm": config.max_strip_loc1_rmse,
            "max_direction_angle_rms_rad": config.max_direction_angle_rms,
        },
        "geometry_consistency": geometry_report,
        "n_tracks": len(batches.event_track_ids),
        "n_fit_hits": int(batches.fit_mask.sum()),
        "n_holdout_hits": int(batches.holdout_mask.sum()),
        "fit_span_mm": {
            "min": float(batches.fit_span.min()),
            "median": float(batches.fit_span.median()),
            "max": float(batches.fit_span.max()),
        },
        "fit_surfaces": fit_metrics,
        "holdout_overall": holdout_metrics,
        "holdout_by_layer": holdout_by_layer,
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_frame = build_comparison_frame(
        batches,
        initial_parameters,
        fitted_parameters,
        initial,
        fitted,
    )
    write_comparison_csv(
        output_dir / "helix_smt_comparison.csv",
        comparison_frame,
    )
    with open(output_dir / "helix_validation_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    from helix_validation_plots import write_residual_plots

    write_residual_plots(
        comparison_frame,
        output_dir / "plots",
        holdout_layers,
    )
    print_validation_summary(summary, output_dir)
    return summary


def print_validation_summary(summary: dict, output_dir: Path) -> None:
    print("\n=== Nominal helix validation ===")
    print(
        f"Tracks={summary['n_tracks']} fit_hits={summary['n_fit_hits']} "
        f"holdout_hits={summary['n_holdout_hits']}"
    )
    print(
        f"{'Selection':<16} {'Global RMSE':>14} {'Local RMSE':>14} "
        f"{'Angle RMS':>14} {'Fail rate':>12}"
    )
    rows = [("holdout overall", summary["holdout_overall"])] + list(
        summary["holdout_by_layer"].items()
    )
    for name, comparison in rows:
        fitted = comparison["fitted"]
        global_text = (
            "n/a"
            if fitted["global_rmse_mm"] is None
            else f"{fitted['global_rmse_mm']:.6f}mm"
        )
        local_text = (
            "n/a"
            if fitted["local_rmse_mm"] is None
            else f"{fitted['local_rmse_mm']:.6f}mm"
        )
        angle_text = (
            "n/a"
            if fitted["direction_angle_rms_rad"] is None
            else f"{fitted['direction_angle_rms_rad']:.6g}rad"
        )
        print(
            f"{name:<16} {global_text:>14} "
            f"{local_text:>14} "
            f"{angle_text:>14} "
            f"{comparison['fitted_intersection_failure_rate']:>11.2%}"
        )
    state = "PASS" if summary["passed"] else "FAIL"
    print(f"Result: {state}")
    for reason in summary["failure_reasons"]:
        print(f"  - {reason}")
    print(f"Outputs: {output_dir}")
    print(f"Residual plots: {output_dir / 'plots'}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit helices to nominal SMT states on non-holdout layers and validate "
            "the frozen helices on complete held-out detector layers."
        )
    )
    add_config_argument(parser)
    defaults = HelixValidationConfig()
    parser.add_argument("--states", type=absolute_project_path, default=defaults.states)
    parser.add_argument("--geometry", type=absolute_project_path, default=defaults.geometry)
    parser.add_argument(
        "--nominal-full-geometry", type=absolute_project_path, default=defaults.nominal_full_geometry
    )
    parser.add_argument("--holdout-layers", default=target_layers_spec())
    parser.add_argument("--min-fit-hits", type=int, default=defaults.min_fit_hits)
    parser.add_argument("--min-fit-span", type=float, default=defaults.min_fit_span)
    parser.add_argument("--max-tracks", type=int, default=defaults.max_tracks)
    parser.add_argument("--fit-iters", type=int, default=defaults.fit_iters)
    parser.add_argument(
        "--context-hits-before", type=int, default=defaults.context_hits_before
    )
    parser.add_argument(
        "--context-hits-after", type=int, default=defaults.context_hits_after
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=defaults.rng_seed)
    parser.add_argument("--output-dir", type=absolute_project_path, default=defaults.output_dir)
    parser.add_argument(
        "--geometry-tolerance", type=float, default=defaults.geometry_tolerance
    )
    parser.add_argument(
        "--max-intersection-failure-rate",
        type=float,
        default=defaults.max_intersection_failure_rate,
    )
    parser.add_argument(
        "--max-pixel-local-rmse",
        type=float,
        default=defaults.max_pixel_local_rmse,
    )
    parser.add_argument(
        "--max-strip-loc0-rmse",
        type=float,
        default=defaults.max_strip_loc0_rmse,
    )
    parser.add_argument(
        "--max-strip-loc1-rmse",
        type=float,
        default=defaults.max_strip_loc1_rmse,
    )
    parser.add_argument(
        "--max-direction-angle-rms",
        type=float,
        default=defaults.max_direction_angle_rms,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> HelixValidationConfig:
    return HelixValidationConfig(
        states=args.states,
        geometry=args.geometry,
        nominal_full_geometry=args.nominal_full_geometry,
        holdout_layers=args.holdout_layers,
        min_fit_hits=args.min_fit_hits,
        min_fit_span=args.min_fit_span,
        max_tracks=args.max_tracks,
        fit_iters=args.fit_iters,
        context_hits_before=args.context_hits_before,
        context_hits_after=args.context_hits_after,
        gpu=args.gpu,
        rng_seed=args.rng_seed,
        output_dir=args.output_dir,
        geometry_tolerance=args.geometry_tolerance,
        max_intersection_failure_rate=args.max_intersection_failure_rate,
        max_pixel_local_rmse=args.max_pixel_local_rmse,
        max_strip_loc0_rmse=args.max_strip_loc0_rmse,
        max_strip_loc1_rmse=args.max_strip_loc1_rmse,
        max_direction_angle_rms=args.max_direction_angle_rms,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        summary = run_helix_validation(config_from_args(args))
    except Exception as exc:
        print(f"Helix validation error: {exc}", file=sys.stderr)
        return 2
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
