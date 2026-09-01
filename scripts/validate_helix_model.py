"""Validate helix interpolation/extrapolation against nominal measurements.

The helix parameter order and geometry conventions intentionally mirror
``ml_alignment_autograd.py``: ``[x0, y0, z0, psi0, tanl, rho]`` and local
coordinates obtained with ``rotation.T @ (global - center)``.

SMT positions seed the helix fit, while held-out measurements are used only as
the final prediction reference: ``predicted_local - measured_local``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from experiment_manifest import (
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from project_config import (
    absolute_project_path,
    add_config_argument,
    load_project_config,
)
try:
    from .helix_model import (
        S_MAX,
        accept_newton_candidate,
        assemble_track_parameters,
        broken_helix_track_parameters,
        assign_hit_segments,
        closest_helix_points,
        fit_helix_points_gauss_newton,
        helix_point,
        helix_tangent,
        highland_kink_sigma,
        intersect_plane,
        node_arc_lengths,
        require_stationary_helix_point_fit,
        split_track_parameters,
    )
except ImportError:  # Direct script execution.
    from helix_model import (
        S_MAX,
        accept_newton_candidate,
        assemble_track_parameters,
        broken_helix_track_parameters,
        assign_hit_segments,
        closest_helix_points,
        fit_helix_points_gauss_newton,
        helix_point,
        helix_tangent,
        highland_kink_sigma,
        intersect_plane,
        node_arc_lengths,
        require_stationary_helix_point_fit,
        split_track_parameters,
    )


DEFAULT_MIN_FIT_HITS = 8
DEFAULT_MIN_FIT_SPAN = 300.0
DEFAULT_MIN_EXTRAPOLATION_FIT_HITS = 4
DEFAULT_MIN_EXTRAPOLATION_FIT_SPAN = 100.0
#: A hit count is only a proxy for what actually determines the fit: a strip hit
#: contributes LOC0 alone, a pixel hit contributes LOC0 and LOC1.  Iteration 22
#: measured the real distribution over the 47809 extrapolation rows -- three rows
#: carry 4 residual components against 5 gauge-fixed reference parameters and 299
#: carry exactly 5 -- so the floor belongs on the residual count.  Raising
#: ``DEFAULT_MIN_EXTRAPOLATION_FIT_HITS`` instead would empty a whole group
#: (17:8 forward drops from 5034 rows to 42 at a floor of 6) to remove three bad
#: ones.  5 parameters plus one degree of redundancy.
DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS = 6
DEFAULT_MAX_TRACKS = 1000
DEFAULT_FIT_ITERS = 200
#: ``fit_iters`` bounds the SMT Gauss-Newton seed and the reproducibility-only
#: Adam measurement-refit arm.  The measurement Gauss-Newton budget is
#: ``broken_fit_iters``.
FIT_SOLVERS = ("gauss-newton", "adam")
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
    min_extrapolation_fit_hits: int = DEFAULT_MIN_EXTRAPOLATION_FIT_HITS
    min_extrapolation_fit_span: float = DEFAULT_MIN_EXTRAPOLATION_FIT_SPAN
    min_extrapolation_fit_residuals: int = DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS
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
    model: str = "both"
    material_artifact: str | None = None
    material_artifact_energy_loss: str | None = None
    on_missing_material: str = "fail"
    broken_fit_iters: int = 60
    fit_solver: str = "gauss-newton"


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
    prediction_mode: tuple[str, ...]
    context_hits_before: torch.Tensor
    context_hits_after: torch.Tensor
    nearest_context_distance: torch.Tensor
    farthest_context_distance: torch.Tensor


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


PREDICTION_MODES = (
    "interpolation",
    "forward_extrapolation",
    "backward_extrapolation",
)


def measurement_residuals(
    predicted_local: torch.Tensor,
    measured_local: torch.Tensor,
    measurement_variance: torch.Tensor,
    loc1_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prediction-minus-measurement residuals and sigma normalization."""
    residual = predicted_local - measured_local
    component_valid = torch.stack(
        (torch.ones_like(loc1_valid, dtype=torch.bool), loc1_valid), dim=-1
    )
    residual = torch.where(
        component_valid, residual, torch.full_like(residual, torch.nan)
    )
    normalized = residual / torch.sqrt(measurement_variance.clamp_min(EPS_VARIANCE))
    return residual, normalized


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


def _signed_distribution(
    values: torch.Tensor, normalized: torch.Tensor
) -> dict[str, float | int | None]:
    finite = torch.isfinite(values) & torch.isfinite(normalized)
    values = values[finite]
    normalized = normalized[finite]
    if values.numel() == 0:
        return {
            "count": 0,
            "bias": None,
            "std": None,
            "rmse": None,
            "mae": None,
            "p95": None,
            "normalized_mean": None,
            "normalized_rms": None,
        }
    return {
        "count": int(values.numel()),
        "bias": float(values.mean().cpu()),
        "std": float(values.std(correction=0).cpu()),
        "rmse": float(torch.sqrt(values.square().mean()).cpu()),
        "mae": float(values.abs().mean().cpu()),
        "p95": float(torch.quantile(values.abs(), 0.95).cpu()),
        "normalized_mean": float(normalized.mean().cpu()),
        "normalized_rms": float(torch.sqrt(normalized.square().mean()).cpu()),
    }


def summarize_measurement_evaluation(
    evaluation: HelixEvaluation,
    batches: TrackBatches,
    selection_mask: torch.Tensor,
) -> dict[str, float | int | None]:
    """Summarize prediction-minus-measurement accuracy for selected hits."""
    device = evaluation.local_position.device
    requested = selection_mask.to(device=device)
    requested_count = int(requested.sum().item())
    if requested_count == 0:
        raise ValueError("The requested measurement evaluation selection is empty.")
    valid = requested & evaluation.intersection_valid
    residual, normalized = measurement_residuals(
        evaluation.local_position,
        batches.measured_local.to(device=device, dtype=evaluation.local_position.dtype),
        batches.measurement_variance.to(
            device=device, dtype=evaluation.local_position.dtype
        ),
        batches.loc1_valid.to(device=device),
    )
    loc0 = _signed_distribution(residual[..., 0][valid], normalized[..., 0][valid])
    loc1 = _signed_distribution(residual[..., 1][valid], normalized[..., 1][valid])
    direction = _distribution(evaluation.tangent_angle[valid])
    result: dict[str, float | int | None] = {
        "count": requested_count,
        "valid_count": int(valid.sum().item()),
        "intersection_failure_rate": 1.0
        - int(valid.sum().item()) / requested_count,
        "direction_angle_rms_rad": direction["rmse"],
        "direction_angle_mae_rad": direction["mae"],
        "direction_angle_p95_rad": direction["p95"],
    }
    for component, stats in (("local0", loc0), ("local1", loc1)):
        result[f"{component}_count"] = stats["count"]
        result[f"{component}_bias_mm"] = stats["bias"]
        result[f"{component}_std_mm"] = stats["std"]
        result[f"{component}_rmse_mm"] = stats["rmse"]
        result[f"{component}_mae_mm"] = stats["mae"]
        result[f"{component}_p95_mm"] = stats["p95"]
        result[f"{component}_normalized_mean"] = stats["normalized_mean"]
        result[f"{component}_normalized_rms"] = stats["normalized_rms"]
    return result


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
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Fit helices only to non-holdout SMT points with damped Gauss-Newton."""
    points = batches.g_smt.to(device=device, dtype=dtype)
    mask = batches.fit_mask.to(device=device)
    initial = initialize_helices(batches, device=device, dtype=dtype)
    fitted, diagnostics = fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=n_iter
    )
    return (fitted, diagnostics) if return_diagnostics else fitted


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
    min_extrapolation_fit_hits: int | None = None,
    min_extrapolation_fit_span: float | None = None,
    min_extrapolation_fit_residuals: int = DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS,
) -> TrackBatches:
    """Build one local fit window for every held-out measurement state."""
    if context_hits_before < 0 or context_hits_after < 0:
        raise ValueError("Context hit counts must be non-negative.")
    extrapolation_hits = (
        min_fit_hits
        if min_extrapolation_fit_hits is None
        else min_extrapolation_fit_hits
    )
    extrapolation_span = (
        min_fit_span
        if min_extrapolation_fit_span is None
        else min_extrapolation_fit_span
    )
    if extrapolation_hits < 2:
        raise ValueError("min_extrapolation_fit_hits must be at least 2.")
    if extrapolation_span < 0:
        raise ValueError("min_extrapolation_fit_span must be non-negative.")
    if min_extrapolation_fit_residuals < 1:
        raise ValueError("min_extrapolation_fit_residuals must be at least 1.")
    holdout_layers = {tuple(map(int, layer)) for layer in holdout_layers}
    positions = inputs.hit_global_positions.detach().cpu().numpy()
    event_ids = inputs.hit_event_ids.detach().cpu().numpy()
    track_ids = inputs.hit_track_ids.detach().cpu().numpy()
    volume_ids = inputs.hit_volume_ids.detach().cpu().numpy()
    layer_ids = inputs.hit_layer_ids.detach().cpu().numpy()
    module_ids = inputs.hit_module_ids.detach().cpu().numpy()
    sensor_ids = inputs.hit_sensor_ids.detach().cpu().numpy()
    loc1_valid_hits = inputs.loc1_valid.detach().cpu().numpy()
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
    selected_by_mode = {mode: [] for mode in PREDICTION_MODES}
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
            before_all = [
                index for index in all_fit_indices if path_values[index] < holdout_path
            ]
            after_all = [
                index for index in all_fit_indices if path_values[index] > holdout_path
            ]
            interpolation_before = (
                before_all[-context_hits_before:]
                if context_hits_before > 0
                else before_all
            )
            interpolation_after = (
                after_all[:context_hits_after]
                if context_hits_after > 0
                else after_all
            )
            candidates = (
                (
                    "interpolation",
                    interpolation_before,
                    interpolation_after,
                ),
                (
                    "forward_extrapolation",
                    before_all[-extrapolation_hits:],
                    [],
                ),
                (
                    "backward_extrapolation",
                    [],
                    after_all[:extrapolation_hits],
                ),
            )
            for mode, before, after in candidates:
                if mode == "interpolation" and (not before or not after):
                    continue
                fit_indices = before + after
                required_hits = (
                    min_fit_hits if mode == "interpolation" else extrapolation_hits
                )
                required_span = (
                    min_fit_span if mode == "interpolation" else extrapolation_span
                )
                if len(fit_indices) < required_hits:
                    continue
                if mode != "interpolation":
                    # Every hit gives LOC0; only a valid LOC1 gives a second
                    # residual.  A window below the floor cannot determine the
                    # reference helix, and its prediction would report the
                    # arbitrariness of an underdetermined fit as model error.
                    fit_residuals = len(fit_indices) + int(
                        np.count_nonzero(loc1_valid_hits[fit_indices])
                    )
                    if fit_residuals < min_extrapolation_fit_residuals:
                        continue
                fit_paths = path_values[fit_indices]
                fit_span = float(np.max(fit_paths) - np.min(fit_paths))
                if fit_span < required_span:
                    continue
                row_indices = sorted(
                    fit_indices + [holdout_index], key=lambda index: path_values[index]
                )
                selected_by_mode[mode].append(
                    (
                        key,
                        row_indices,
                        fit_span,
                        mode,
                        len(before),
                        len(after),
                        float(np.min(np.abs(fit_paths - holdout_path))),
                        float(np.max(np.abs(fit_paths - holdout_path))),
                    )
                )
    selected = []
    for mode in PREDICTION_MODES:
        mode_rows = selected_by_mode[mode]
        if max_tracks > 0:
            mode_rows = mode_rows[:max_tracks]
        selected.extend(mode_rows)
    if not selected:
        raise ValueError("No tracks satisfy the helix validation selection.")

    n_tracks = len(selected)
    max_hits = max(len(item[1]) for item in selected)
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
    prediction_mode = []
    selected_before = torch.zeros(n_tracks, dtype=torch.long)
    selected_after = torch.zeros(n_tracks, dtype=torch.long)
    nearest_context_distance = torch.zeros(n_tracks, dtype=dtype)
    farthest_context_distance = torch.zeros(n_tracks, dtype=dtype)
    for row, (
        key,
        indices,
        span,
        mode,
        before_count,
        after_count,
        nearest_distance,
        farthest_distance,
    ) in enumerate(selected):
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
        prediction_mode.append(mode)
        selected_before[row] = before_count
        selected_after[row] = after_count
        nearest_context_distance[row] = nearest_distance
        farthest_context_distance[row] = farthest_distance
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
        prediction_mode=tuple(prediction_mode),
        context_hits_before=selected_before,
        context_hits_after=selected_after,
        nearest_context_distance=nearest_context_distance,
        farthest_context_distance=farthest_context_distance,
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


def _empty_measurement_metrics() -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "count": 0,
        "valid_count": 0,
        "intersection_failure_rate": None,
        "direction_angle_rms_rad": None,
        "direction_angle_mae_rad": None,
        "direction_angle_p95_rad": None,
    }
    for component in ("local0", "local1"):
        metrics[f"{component}_count"] = 0
        for suffix in (
            "bias_mm",
            "std_mm",
            "rmse_mm",
            "mae_mm",
            "p95_mm",
            "normalized_mean",
            "normalized_rms",
        ):
            metrics[f"{component}_{suffix}"] = None
    return metrics


def _measurement_comparison_metrics(
    initial: HelixEvaluation,
    fitted: HelixEvaluation,
    batches: TrackBatches,
    selection_mask: torch.Tensor,
) -> dict:
    requested = selection_mask.cpu()
    requested_count = int(requested.sum().item())
    if requested_count == 0:
        return {
            "requested_count": 0,
            "common_valid_count": 0,
            "initial_intersection_failure_rate": None,
            "fitted_intersection_failure_rate": None,
            "initial": _empty_measurement_metrics(),
            "fitted": _empty_measurement_metrics(),
        }
    initial_valid = initial.intersection_valid.cpu()
    fitted_valid = fitted.intersection_valid.cpu()
    common = requested & initial_valid & fitted_valid
    if common.any():
        initial_metrics = summarize_measurement_evaluation(initial, batches, common)
        fitted_metrics = summarize_measurement_evaluation(fitted, batches, common)
    else:
        initial_metrics = _empty_measurement_metrics()
        fitted_metrics = _empty_measurement_metrics()
    return {
        "requested_count": requested_count,
        "common_valid_count": int(common.sum().item()),
        "initial_intersection_failure_rate": 1.0
        - float((requested & initial_valid).sum().item()) / requested_count,
        "fitted_intersection_failure_rate": 1.0
        - float((requested & fitted_valid).sum().item()) / requested_count,
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


def _measurement_validation_pass(
    holdout_by_layer_and_mode: dict[str, dict[str, dict]],
    max_failure_rate: float,
    layer_local_limits: dict[str, tuple[float, float]],
    max_direction_angle_rms: float,
) -> tuple[bool, list[str]]:
    """Apply absolute measurement-accuracy gates to every layer and mode."""
    reasons: list[str] = []
    for layer, by_mode in holdout_by_layer_and_mode.items():
        for mode in PREDICTION_MODES:
            scope = f"holdout layer {layer} {mode}"
            metrics = by_mode.get(mode)
            if metrics is None or metrics.get("requested_count", 0) < 1:
                reasons.append(f"{scope} has no selected measurement")
                continue
            if metrics.get("common_valid_count", 0) < 1:
                reasons.append(f"{scope} has no common valid intersection")
                continue
            failure_rate = float(metrics["fitted_intersection_failure_rate"])
            if failure_rate > max_failure_rate:
                reasons.append(
                    f"{scope} intersection failure rate {failure_rate:.2%} exceeds "
                    f"{max_failure_rate:.2%}"
                )
            fitted = metrics["fitted"]
            limits = layer_local_limits[layer]
            for index, component in enumerate(("LOC0", "LOC1")):
                count = int(fitted.get(f"local{index}_count", 0))
                value = fitted.get(f"local{index}_rmse_mm")
                if count < 1 or value is None:
                    continue
                if not np.isfinite(value) or value > limits[index]:
                    reasons.append(
                        f"{scope} {component} RMSE {value:.6g} mm exceeds "
                        f"{limits[index]:.6g} mm"
                    )
            angle = fitted.get("direction_angle_rms_rad")
            if (
                angle is None
                or not np.isfinite(angle)
                or angle > max_direction_angle_rms
            ):
                reasons.append(
                    f"{scope} direction-angle RMS {angle} rad exceeds "
                    f"{max_direction_angle_rms:.6g} rad"
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
    measured_local = selected(batches.measured_local)
    measurement_variance = selected(batches.measurement_variance)
    loc1_valid = selected(batches.loc1_valid).astype(bool)
    initial_measurement_residual, initial_measurement_normalized = measurement_residuals(
        initial.local_position,
        batches.measured_local.to(
            device=initial.local_position.device, dtype=initial.local_position.dtype
        ),
        batches.measurement_variance.to(
            device=initial.local_position.device, dtype=initial.local_position.dtype
        ),
        batches.loc1_valid.to(device=initial.local_position.device),
    )
    fitted_measurement_residual, fitted_measurement_normalized = measurement_residuals(
        fitted.local_position,
        batches.measured_local.to(
            device=fitted.local_position.device, dtype=fitted.local_position.dtype
        ),
        batches.measurement_variance.to(
            device=fitted.local_position.device, dtype=fitted.local_position.dtype
        ),
        batches.loc1_valid.to(device=fitted.local_position.device),
    )
    row_modes = np.asarray(batches.prediction_mode, dtype=object)
    context_before = _as_numpy(batches.context_hits_before)[track_rows]
    context_after = _as_numpy(batches.context_hits_after)[track_rows]
    nearest_context_distance = _as_numpy(batches.nearest_context_distance)[track_rows]
    farthest_context_distance = _as_numpy(batches.farthest_context_distance)[track_rows]
    frame = pd.DataFrame(
        {
            "split": np.where(selected(batches.holdout_mask), "holdout", "fit"),
            "prediction_mode": row_modes[track_rows],
            "context_hits_before": context_before,
            "context_hits_after": context_after,
            "nearest_context_distance_mm": nearest_context_distance,
            "farthest_context_distance_mm": farthest_context_distance,
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
            "measurement_LOC0": measured_local[:, 0],
            "measurement_LOC1": np.where(
                loc1_valid, measured_local[:, 1], np.nan
            ),
            "measurement_sigma_LOC0": np.sqrt(measurement_variance[:, 0]),
            "measurement_sigma_LOC1": np.where(
                loc1_valid, np.sqrt(measurement_variance[:, 1]), np.nan
            ),
            "measurement_LOC1_valid": loc1_valid,
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
            "initial_measurement_residual_LOC0": selected(
                initial_measurement_residual[..., 0]
            ),
            "initial_measurement_residual_LOC1": selected(
                initial_measurement_residual[..., 1]
            ),
            "fitted_measurement_residual_LOC0": selected(
                fitted_measurement_residual[..., 0]
            ),
            "fitted_measurement_residual_LOC1": selected(
                fitted_measurement_residual[..., 1]
            ),
            "initial_measurement_normalized_LOC0": selected(
                initial_measurement_normalized[..., 0]
            ),
            "initial_measurement_normalized_LOC1": selected(
                initial_measurement_normalized[..., 1]
            ),
            "fitted_measurement_normalized_LOC0": selected(
                fitted_measurement_normalized[..., 0]
            ),
            "fitted_measurement_normalized_LOC1": selected(
                fitted_measurement_normalized[..., 1]
            ),
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
    if config.min_extrapolation_fit_hits < 2:
        raise ValueError("min_extrapolation_fit_hits must be at least 2.")
    if config.min_extrapolation_fit_span < 0:
        raise ValueError("min_extrapolation_fit_span must be non-negative.")
    if config.min_extrapolation_fit_residuals < 1:
        raise ValueError("min_extrapolation_fit_residuals must be at least 1.")
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
    if config.model not in {"single", "broken", "both"}:
        raise ValueError("model must be one of single, broken, both.")
    if config.on_missing_material not in {"fail", "drop"}:
        raise ValueError("on_missing_material must be fail or drop.")
    if config.broken_fit_iters < 1:
        raise ValueError("broken_fit_iters must be at least 1.")
    if config.fit_solver not in FIT_SOLVERS:
        raise ValueError(
            f"fit_solver must be one of {', '.join(sorted(FIT_SOLVERS))}."
        )
    if config.model == "broken" and not config.material_artifact:
        raise ValueError("--model broken requires --material-artifact.")
    for input_path, label in (
        (config.states, "nominal trackstates"),
        (config.geometry, "current geometry"),
        (config.nominal_full_geometry, "nominal full geometry"),
    ):
        if not Path(input_path).exists():
            raise FileNotFoundError(f"{label} does not exist: {input_path}")
    # Gate 4 runs only when a material artifact is supplied; ``--model single``
    # keeps the legacy single-helix-only run even if one is.
    run_gate4 = bool(config.material_artifact) and config.model != "single"
    material = (
        load_material_artifact(config.material_artifact) if run_gate4 else None
    )
    energy_loss_material = (
        load_material_artifact(config.material_artifact_energy_loss)
        if run_gate4 and config.material_artifact_energy_loss
        else None
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_experiment_manifest(
        output_dir,
        run_id=f"helix-validation-{output_dir.name}",
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        config=load_project_config(),
        inputs={
            "nominal_trackstates": config.states,
            "current_geometry": config.geometry,
            "nominal_full_geometry": config.nominal_full_geometry,
            **(
                {"material_artifact": config.material_artifact}
                if config.material_artifact
                else {}
            ),
            **(
                {
                    "material_artifact_energy_loss":
                        config.material_artifact_energy_loss
                }
                if config.material_artifact_energy_loss
                else {}
            ),
        },
        outputs={
            "summary": output_dir / "helix_validation_summary.json",
            "measurement_comparison": output_dir
            / "helix_measurement_comparison.csv",
            "smt_comparison": output_dir / "helix_smt_comparison.csv",
            "plots": output_dir / "plots",
        },
        metrics={
            "effective_validation_config": {
                "states": str(config.states),
                "geometry": str(config.geometry),
                "nominal_full_geometry": str(config.nominal_full_geometry),
                "holdout_layers": [
                    f"{volume}:{layer}" for volume, layer in holdout_layers
                ],
                "min_fit_hits": config.min_fit_hits,
                "min_fit_span_mm": config.min_fit_span,
                "min_extrapolation_fit_hits": config.min_extrapolation_fit_hits,
                "min_extrapolation_fit_span_mm": config.min_extrapolation_fit_span,
                "min_extrapolation_fit_residuals": config.min_extrapolation_fit_residuals,
                "max_tracks_per_mode": config.max_tracks,
                "fit_iters": config.fit_iters,
                "context_hits_before": config.context_hits_before,
                "context_hits_after": config.context_hits_after,
                "rng_seed": config.rng_seed,
            }
        },
        status="running",
    )

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
        min_extrapolation_fit_hits=config.min_extrapolation_fit_hits,
        min_extrapolation_fit_span=config.min_extrapolation_fit_span,
        min_extrapolation_fit_residuals=config.min_extrapolation_fit_residuals,
    )
    initial_parameters = initialize_helices(batches, device=device, dtype=dtype)
    seeded_parameters, seed_diagnostics = seed_helices(
        batches,
        n_iter=config.fit_iters,
        device=device,
        dtype=dtype,
        return_diagnostics=True,
    )
    require_stationary_helix_point_fit(seed_diagnostics)
    if config.fit_solver == "gauss-newton":
        fitted_parameters, fit_diagnostics = fit_helices_gauss_newton(
            batches,
            geometry,
            seeded_parameters,
            max_iters=config.broken_fit_iters,
            device=device,
            dtype=dtype,
        )
    else:
        fitted_parameters = fit_helices_to_measurements(
            batches,
            geometry,
            seeded_parameters,
            n_iter=config.fit_iters,
            device=device,
            dtype=dtype,
        )
        fit_diagnostics = {
            "solver": "adam",
            "iterations": config.fit_iters,
            "stationarity": stationarity_report(
                measurement_newton_decrement(
                    batches, geometry, fitted_parameters,
                    device=device, dtype=dtype,
                )
            ),
        }
    initial = evaluate_helix_on_surfaces(
        initial_parameters, batches, geometry
    )
    fitted = evaluate_helix_on_surfaces(fitted_parameters, batches, geometry)
    fit_metrics = _measurement_comparison_metrics(
        initial, fitted, batches, batches.fit_mask
    )
    holdout_metrics = _measurement_comparison_metrics(
        initial, fitted, batches, batches.holdout_mask
    )
    smt_diagnostics = {
        "fit_surfaces": _comparison_metrics(initial, fitted, batches.fit_mask),
        "holdout_overall": _comparison_metrics(
            initial, fitted, batches.holdout_mask
        ),
    }
    mode_rows = {
        mode: torch.tensor(
            [value == mode for value in batches.prediction_mode], dtype=torch.bool
        )[:, None].expand_as(batches.holdout_mask)
        for mode in PREDICTION_MODES
    }
    holdout_by_mode = {
        mode: _measurement_comparison_metrics(
            initial, fitted, batches, batches.holdout_mask & mode_rows[mode]
        )
        for mode in PREDICTION_MODES
    }
    holdout_by_layer = {}
    holdout_by_layer_and_mode = {}
    for volume, layer in holdout_layers:
        mask = (
            batches.holdout_mask
            & (batches.volume_id == volume)
            & (batches.layer_id == layer)
        )
        layer_key = f"{volume}:{layer}"
        holdout_by_layer[layer_key] = _measurement_comparison_metrics(
            initial, fitted, batches, mask
        )
        holdout_by_layer_and_mode[layer_key] = {
            mode: _measurement_comparison_metrics(
                initial, fitted, batches, mask & mode_rows[mode]
            )
            for mode in PREDICTION_MODES
        }
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
    passed, failure_reasons = _measurement_validation_pass(
        holdout_by_layer_and_mode,
        config.max_intersection_failure_rate,
        layer_local_limits=layer_local_limits,
        max_direction_angle_rms=config.max_direction_angle_rms,
    )
    # A refit that never reached its optimum makes every downstream residual a
    # statement about the optimiser, not about the model.  The same threshold
    # already gates the Gate 4 arms.
    stalled = fit_diagnostics["stationarity"]["non_stationary_row_fraction"]
    if stalled > MAX_NON_STATIONARY_ROW_FRACTION:
        passed = False
        failure_reasons = list(failure_reasons) + [
            f"measurement refit ({fit_diagnostics['solver']}): {stalled:.4f} of "
            f"rows are non-stationary (Newton decrement > "
            f"{NEWTON_DECREMENT_STATIONARY}), above the pre-registered "
            f"{MAX_NON_STATIONARY_ROW_FRACTION}"
        ]
    summary = {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "seed_source": "smt",
        "fit_source": "measurement",
        "validation_reference": "measurement",
        "comparison_baseline": "unrefined_smt_initialization",
        "config": {
            "states": str(config.states),
            "geometry": str(config.geometry),
            "nominal_full_geometry": str(config.nominal_full_geometry),
            "holdout_layers": [f"{volume}:{layer}" for volume, layer in holdout_layers],
            "min_fit_hits": config.min_fit_hits,
            "min_fit_span_mm": config.min_fit_span,
            "min_extrapolation_fit_hits": config.min_extrapolation_fit_hits,
            "min_extrapolation_fit_span_mm": config.min_extrapolation_fit_span,
            "min_extrapolation_fit_residuals": config.min_extrapolation_fit_residuals,
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
            "model": config.model,
            "material_artifact": config.material_artifact,
            "material_artifact_energy_loss": config.material_artifact_energy_loss,
            "on_missing_material": config.on_missing_material,
            "broken_fit_iters": config.broken_fit_iters,
            "fit_solver": config.fit_solver,
        },
        "measurement_fit": json_safe(fit_diagnostics),
        "smt_seed": json_safe(seed_diagnostics),
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
        "holdout_by_mode": holdout_by_mode,
        "holdout_by_layer": holdout_by_layer,
        "holdout_by_layer_and_mode": holdout_by_layer_and_mode,
        "smt_diagnostics": smt_diagnostics,
    }
    if run_gate4:
        broken_gate = run_paired_holdout(
            batches,
            geometry,
            fitted_parameters,
            material,
            holdout_layers,
            output_dir,
            device=device,
            dtype=dtype,
            model=config.model,
            on_missing_material=config.on_missing_material,
            max_iters=config.broken_fit_iters,
            energy_loss_material=energy_loss_material,
        )
        summary["broken_helix_gate"] = json_safe(broken_gate)
        if broken_gate["status"] != "passed":
            summary["passed"] = False
            summary["failure_reasons"] = list(summary["failure_reasons"]) + [
                f"broken-helix gate {broken_gate['status']}: {reason}"
                for reason in broken_gate["reasons"]
            ]
    else:
        summary["broken_helix_gate"] = {
            "status": "skipped",
            "reason": (
                "no --material-artifact supplied"
                if not config.material_artifact
                else "--model single"
            ),
        }
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
    write_comparison_csv(
        output_dir / "helix_measurement_comparison.csv",
        comparison_frame.loc[comparison_frame["split"] == "holdout"],
    )
    with open(output_dir / "helix_validation_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    from helix_validation_plots import write_residual_plots

    write_residual_plots(
        comparison_frame,
        output_dir / "plots",
        holdout_layers,
    )
    finalize_experiment_manifest(
        manifest_path,
        status="completed",
        outputs={
            "summary": output_dir / "helix_validation_summary.json",
            "measurement_comparison": output_dir
            / "helix_measurement_comparison.csv",
            "smt_comparison": output_dir / "helix_smt_comparison.csv",
            "plots": output_dir / "plots",
            **(
                {"holdout_keys": output_dir / "holdout_keys.csv"}
                if run_gate4
                else {}
            ),
        },
        metrics={"validation_summary": summary},
    )
    print_validation_summary(summary, output_dir)
    return summary


def print_validation_summary(summary: dict, output_dir: Path) -> None:
    print("\n=== Nominal helix measurement validation ===")
    print(
        f"Tracks={summary['n_tracks']} fit_hits={summary['n_fit_hits']} "
        f"holdout_hits={summary['n_holdout_hits']}"
    )
    print(
        f"{'Selection':<38} {'LOC0 RMSE':>14} {'LOC1 RMSE':>14} "
        f"{'Angle RMS':>14} {'Fail rate':>12}"
    )
    rows = [("holdout overall", summary["holdout_overall"])]
    rows.extend(
        (f"{layer} {mode}", metrics)
        for layer, by_mode in summary["holdout_by_layer_and_mode"].items()
        for mode, metrics in by_mode.items()
    )
    for name, comparison in rows:
        fitted = comparison["fitted"]
        loc0_text = (
            "n/a"
            if fitted["local0_rmse_mm"] is None
            else f"{fitted['local0_rmse_mm']:.6f}mm"
        )
        loc1_text = (
            "n/a"
            if fitted["local1_rmse_mm"] is None
            else f"{fitted['local1_rmse_mm']:.6f}mm"
        )
        angle_text = (
            "n/a"
            if fitted["direction_angle_rms_rad"] is None
            else f"{fitted['direction_angle_rms_rad']:.6g}rad"
        )
        print(
            f"{name:<38} {loc0_text:>14} "
            f"{loc1_text:>14} "
            f"{angle_text:>14} "
            f"{comparison['fitted_intersection_failure_rate'] or 0.0:>11.2%}"
        )
    gate = summary.get("broken_helix_gate", {})
    if gate.get("status") not in (None, "skipped"):
        print("\n=== Gate 4: paired single vs broken helix ===")
        print(
            f"rows={gate['n_rows_used']}/{gate['n_rows_total']} "
            f"keys_sha256={gate['holdout_keys_sha256'][:16]}... "
            f"nodes_in_window={gate['nodes_in_window']}"
        )
        header = (
            f"{'Group':<40} {'N':>6} {'RMSE single':>13} {'RMSE broken':>13} "
            f"{'dRMSE 95% CI':>24} {'dNLL 95% CI':>24}"
        )
        print(header)
        for layer_key, by_mode in gate["groups"].items():
            for mode, by_component in by_mode.items():
                for name, entry in by_component.items():
                    if entry["n"] == 0:
                        continue
                    rmse = entry["relative_rmse_change"]
                    nll = entry["delta_mean_nll"]

                    def cell(value, spec):
                        return (
                            "n/a".rjust(int(re.sub(r"[^0-9]", "", spec.split(".")[0]) or 0))
                            if value is None or not math.isfinite(value)
                            else format(value, spec)
                        )

                    print(
                        f"{layer_key + ' ' + mode + ' ' + name:<40} {entry['n']:>6} "
                        f"{cell(entry['single']['rmse_mm'], '>13.6f')} "
                        f"{cell(entry['broken']['rmse_mm'], '>13.6f')} "
                        f"[{cell(rmse['low'], '>+9.3%')},{cell(rmse['high'], '>+9.3%')}] "
                        f"[{cell(nll['low'], '>+10.4g')},{cell(nll['high'], '>+10.4g')}]"
                    )
        print(f"Broken-helix gate: {gate['status'].upper()}")
        print(
            "Energy-loss re-verification: "
            f"{gate['energy_loss_reverification']['status']}"
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
    parser.add_argument(
        "--min-extrapolation-fit-hits",
        type=int,
        default=defaults.min_extrapolation_fit_hits,
    )
    parser.add_argument(
        "--min-extrapolation-fit-span",
        type=float,
        default=defaults.min_extrapolation_fit_span,
    )
    parser.add_argument(
        "--min-extrapolation-fit-residuals",
        type=int,
        default=defaults.min_extrapolation_fit_residuals,
    )
    parser.add_argument("--max-tracks", type=int, default=defaults.max_tracks)
    parser.add_argument("--fit-iters", type=int, default=defaults.fit_iters)
    parser.add_argument(
        "--fit-solver",
        choices=FIT_SOLVERS,
        default=defaults.fit_solver,
        help=(
            "Solver for the measurement refit. gauss-newton is the default; "
            "adam reproduces the pre-Iteration-14 behaviour and is not expected "
            "to reach the optimum."
        ),
    )
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
    parser.add_argument(
        "--model",
        choices=("single", "broken", "both"),
        default=defaults.model,
        help=(
            "Which track model to evaluate against the material artifact. Both "
            "arms always run in one process on one shared row set."
        ),
    )
    parser.add_argument(
        "--material-artifact",
        type=absolute_project_path,
        default=defaults.material_artifact,
        help="Schema-1.1 material NPZ from scripts/material_artifact.py.",
    )
    parser.add_argument(
        "--material-artifact-energy-loss",
        type=absolute_project_path,
        default=defaults.material_artifact_energy_loss,
        help=(
            "Optional second artifact exported with energyLoss=True; supplying it "
            "runs the Gate 0 energy-loss re-verification that is still open."
        ),
    )
    parser.add_argument(
        "--on-missing-material",
        choices=("fail", "drop"),
        default=defaults.on_missing_material,
    )
    parser.add_argument(
        "--broken-fit-iters", type=int, default=defaults.broken_fit_iters
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
        min_extrapolation_fit_hits=args.min_extrapolation_fit_hits,
        min_extrapolation_fit_residuals=args.min_extrapolation_fit_residuals,
        min_extrapolation_fit_span=args.min_extrapolation_fit_span,
        max_tracks=args.max_tracks,
        fit_iters=args.fit_iters,
        fit_solver=args.fit_solver,
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
        model=args.model,
        material_artifact=args.material_artifact,
        material_artifact_energy_loss=args.material_artifact_energy_loss,
        on_missing_material=args.on_missing_material,
        broken_fit_iters=args.broken_fit_iters,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = config_from_args(args)
    try:
        summary = run_helix_validation(config)
    except Exception as exc:
        manifest_path = Path(config.output_dir) / "manifest.json"
        if manifest_path.exists():
            finalize_experiment_manifest(
                manifest_path,
                status="failed",
                metrics={"error": f"{type(exc).__name__}: {exc}"},
            )
        print(f"Helix validation error: {exc}", file=sys.stderr)
        return 2
    return 0 if summary["passed"] else 1




# ---------------------------------------------------------------------------
# Posterior and predictive covariance
# ---------------------------------------------------------------------------

#: Objective convention.  Everything below assumes ``F = 0.5 * chi2`` so that the
#: Gauss-Newton Hessian IS the posterior precision with no extra factor.  Using
#: the chi2 Hessian instead would report exactly half the true variance, which is
#: why ``tests/test_validate_helix_model.py`` cross-checks the scale against a
#: Monte-Carlo sampling covariance rather than against this same formula.
HALF_CHI2_OBJECTIVE = True

#: Pre-registration v2 (Iteration 17).  v1 used ``PRODUCTION_DAMPING = 1e-10``
#: with the whole of ``DAMPING_SCAN`` gating the verdict.  Both were wrong once
#: the structurally inactive directions were understood:
#:
#: * A padded kink slot is an exact null direction of H and of J.  Damping it
#:   relative to the mean diagonal forced the factorisation towards 1e13
#:   condition numbers, which is why ``damping = 0`` used to fail outright.
#:   ``predictive_covariance`` now fills those directions instead, so the exact
#:   posterior is solvable and the production value can be zero: no bias, and
#:   the two arms are equivalent by construction rather than by tuning.
#: * A non-zero *relative* damping can never be equivalent across the arms --
#:   their spectra differ because the kinks are genuinely soft.  At 1e-10 the
#:   broken arm's reported variance moved by 18.5% (p99) against 0.045% for the
#:   single arm.  No choice of relative damping removes that asymmetry.
PRODUCTION_DAMPING = 0.0
#: Recorded in full as a diagnostic: what the answer would be under increasing
#: regularisation.  It does not gate, because a large damping is a large bias by
#: definition and "the conclusion changes if you bias it" is trivially true.
DAMPING_SCAN = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
#: A row whose active subspace is rank deficient has no posterior to report.  On
#: the full 20000-event set exactly one row of 70352 hits this: 4 hits against 5
#: gauge-fixed reference parameters, an underdetermined fit whose prediction is
#: meaningless.  Rows above this fraction make the verdict inconclusive rather
#: than quietly resting on a reduced row set.
MAX_RANK_DEFICIENT_ROW_FRACTION = 1.0e-3
#: What does gate: the verdict must not move across dampings whose bias is
#: numerically negligible.  Measured on real rows, 1e-12 shifts the broken arm's
#: predictive variance by 0.34% (p99) and the single arm's by 4.5e-6.
DAMPING_STABILITY_SCAN = (0.0, 1e-12)


def gauss_newton_hessian(
    jacobian: torch.Tensor,
    variance: torch.Tensor,
    prior_precision: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``J^T V^-1 J + Q^-1``, the posterior precision of ``F = 0.5 chi2``.

    ``jacobian`` is ``[n_residual, n_parameter]`` and ``variance`` holds the
    per-component measurement variances, matching the repository's diagonal
    measurement-covariance contract.  ``prior_precision`` carries the scattering
    prior ``blkdiag(Q_j^-1)`` padded to the full parameter vector.
    """
    weighted = jacobian / variance.unsqueeze(-1)
    hessian = jacobian.transpose(-1, -2) @ weighted
    if prior_precision is not None:
        hessian = hessian + prior_precision
    return 0.5 * (hessian + hessian.transpose(-1, -2))


def predictive_covariance(
    hessian: torch.Tensor,
    prediction_jacobian: torch.Tensor,
    damping: float = PRODUCTION_DAMPING,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Propagate the posterior onto a prediction: ``J C J^T`` with ``C = H^-1``.

    The inverse is never formed.  A Cholesky factorisation plus a triangular
    solve is both cheaper and better conditioned, and it fails loudly on an
    indefinite Hessian instead of silently returning garbage.

    Damping is applied relative to the mean diagonal so it means the same thing
    across the very different scales of the reference parameters and the kinks.
    """
    symmetric = 0.5 * (hessian + hessian.transpose(-1, -2))
    n_parameters = symmetric.shape[-1]
    # A padded kink slot enters no residual, so its row and column of J -- and
    # therefore of H -- are exactly zero.  Those directions are structural, not
    # a sick fit: the broken arm pads ``node_s`` to a fixed width, the single arm
    # has nothing to pad.  Damping them relative to the mean diagonal would hand
    # each one a variance of 1/(damping*scale); J annihilates that analytically,
    # but it drags the factorisation to ~1e13 condition numbers, and the rounding
    # error that follows is what made the two arms respond differently to the
    # same pre-registered damping grid.
    active = symmetric.abs().amax(dim=-1) > 0
    n_active = active.sum(dim=-1)
    if int(n_active.min()) == 0:
        raise torch.linalg.LinAlgError(
            "Every parameter direction is structurally inactive; the Hessian "
            "carries no information at all."
        )
    # The prediction must not depend on a direction the fit cannot constrain.
    # Verified rather than assumed: if it ever does, the covariance below would
    # silently inherit whatever variance the filler assigns.
    leak = (prediction_jacobian * (~active).unsqueeze(-2)).abs().amax()
    tolerance = prediction_jacobian.abs().amax() * 1.0e-12
    if float(leak) > float(tolerance):
        raise ValueError(
            "The prediction Jacobian is non-zero on a structurally inactive "
            f"parameter direction (|J| = {float(leak):.3e} > {float(tolerance):.3e}); "
            "the predictive covariance would be an artefact of the regulariser."
        )
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    # ``eigvalsh`` returns ascending order, so the structural zeros come first
    # and the solvable subspace starts at ``n_parameters - n_active``.  Reporting
    # the conditioning of that subspace is the point: an exactly-zero eigenvalue
    # from padding says nothing about whether the fit is well determined.
    first_active = (n_parameters - n_active).clamp_min(0)
    smallest_active = eigenvalues.gather(-1, first_active.unsqueeze(-1)).squeeze(-1)
    minimum = float(smallest_active.min())
    maximum = float(eigenvalues[..., -1].max())
    diagonal = symmetric.diagonal(dim1=-2, dim2=-1)
    # Averaged over the solvable subspace only.  Including the structural zeros
    # would make the same physical fit report a different scale purely because it
    # was padded to a different width.
    scale = (diagonal * active).sum(dim=-1) / n_active.to(diagonal.dtype)
    # Active directions keep the pre-registered relative damping.  Inactive ones
    # are filled to the same scale, which leaves the factorisation as well
    # conditioned as the solvable subspace and, because J is zero there, leaves
    # the covariance analytically untouched.
    # A rank-deficient active subspace is a different animal from a rounding
    # problem, and the ladder below must not paper over it: a direction the fit
    # genuinely cannot resolve has no posterior width, and any number reported
    # for it would come from the regulariser alone.  The standard numerical-rank
    # test separates the two.
    # A rank-deficient active subspace has no posterior width to report, and no
    # damping can honestly supply one -- it would return the regulariser's own
    # number wearing a physics label.  The standard numerical-rank test finds
    # them: on the full 20000-event set exactly one row of 70352 qualifies, a
    # single-arm fit with 4 hits against 5 gauge-fixed parameters.
    # Per row, against that row's own largest eigenvalue.  Using the batch-wide
    # maximum would scale every row's floor by the stiffest track in the chunk
    # and condemn healthy rows wholesale.
    rank_floor = (
        eigenvalues[..., -1] * torch.finfo(symmetric.dtype).eps * n_parameters
    )
    degenerate = smallest_active <= rank_floor
    if symmetric.ndim == 2:
        # One matrix and nothing to fall back on: fail loudly, as the docstring
        # promises, rather than handing back a silent NaN.
        if bool(degenerate):
            raise torch.linalg.LinAlgError(
                "The active subspace is rank deficient (smallest active "
                f"eigenvalue {float(smallest_active):.3e} <= {float(rank_floor):.3e}); "
                "the posterior is undefined."
            )
    level = torch.where(active, damping * scale.unsqueeze(-1), scale.unsqueeze(-1))
    regularized = symmetric + torch.diag_embed(level)
    if bool(degenerate.any()):
        # Keep the batch factorisable; the rows below are masked to NaN anyway,
        # and the callers' finite checks drop them from BOTH arms, so the paired
        # row set stays shared.
        identity = torch.eye(
            n_parameters, device=symmetric.device, dtype=symmetric.dtype
        )
        regularized = torch.where(
            degenerate.reshape(degenerate.shape + (1, 1)), identity, regularized
        )
    factor = torch.linalg.cholesky(regularized)
    solved = torch.cholesky_solve(prediction_jacobian.transpose(-1, -2), factor)
    covariance = prediction_jacobian @ solved
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    if bool(degenerate.any()):
        covariance = torch.where(
            degenerate.reshape(degenerate.shape + (1, 1)),
            torch.full_like(covariance, float("nan")),
            covariance,
        )
    diagnostics = {
        "cholesky_ok": True,
        "damping": float(damping),
        "symmetry_residual": float(
            (hessian - hessian.transpose(-1, -2)).norm()
            / hessian.norm().clamp_min(torch.finfo(hessian.dtype).tiny)
        ),
        "batch_size": int(hessian.shape[0]) if hessian.ndim == 3 else 1,
        "min_eigenvalue": minimum,
        "max_eigenvalue": maximum,
        "condition_number": float("inf") if minimum <= 0 else maximum / minimum,
        "n_inactive_directions": int((n_parameters - n_active).max()),
        "n_rank_deficient_rows": int(degenerate.sum()),
    }
    return covariance, diagnostics


# ---------------------------------------------------------------------------
# Broken-helix MAP fit
# ---------------------------------------------------------------------------

GAUSS_NEWTON_MAX_ITERS = 60
#: Relative, not absolute.  The reference helix carries a near-degenerate
#: "slide the origin along the trajectory" direction: freezing ``node_s`` stops
#: it being an exact gauge, but it still sits ~12 orders below the stiffest
#: mode.  An absolute objective-decrease test therefore never fires -- the
#: solver keeps crawling along that direction long after the fit is converged
#: in every quantity anyone reads out.
GAUSS_NEWTON_TOLERANCE = 1e-10
GAUSS_NEWTON_INITIAL_DAMPING = 1e-8
GAUSS_NEWTON_MIN_DAMPING = 1e-12


def reference_gauge_frame(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix the reference helix's gauge: an origin and two transverse axes.

    Sliding ``(x0, y0, z0)`` along the trajectory while rotating ``psi0`` by
    ``rho * ds`` traces the identical curve, so the six reference parameters
    are one redundant.  Freezing ``node_s`` breaks that degeneracy only through
    the kinks moving with the origin, which is an O(kink) effect -- in practice
    ~12 orders below the stiffest mode, leaving the normal equations singular
    for every practical purpose.

    Constraining the origin to the plane through it perpendicular to the
    initial direction removes the redundancy outright, exactly as a perigee or
    reference-surface convention does.  The frame belongs to the linearisation
    point and is held fixed for the whole fit.
    """
    direction = helix_tangent(reference, torch.zeros_like(reference[:, 0]))
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    helper = torch.zeros_like(direction)
    helper[:, 2] = 1.0
    # Fall back to x when the track runs along z and the cross product vanishes.
    degenerate = direction[:, 2].abs() > 1.0 - 1e-6
    helper[degenerate] = torch.tensor(
        [1.0, 0.0, 0.0], dtype=helper.dtype, device=helper.device
    )
    axis0 = torch.cross(helper, direction, dim=-1)
    axis0 = axis0 / axis0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis1 = torch.cross(direction, axis0, dim=-1)
    return reference[:, :3].clone(), axis0, axis1


def _pack_parameters(reference: torch.Tensor, kinks: torch.Tensor) -> torch.Tensor:
    """Stack the gauge-fixed reference and every kink into one vector per track.

    The two transverse origin offsets start at zero by construction, so the
    frame returned by :func:`reference_gauge_frame` must accompany this vector
    everywhere it is unpacked.  Curvature is carried in the optimiser's scaled
    units so one step size means the same thing for millimetres, radians and
    inverse millimetres.
    """
    _linear, rho_scaled = split_track_parameters(reference)
    offsets = torch.zeros((reference.shape[0], 2), dtype=reference.dtype, device=reference.device)
    return torch.cat(
        (offsets, reference[:, 3:5], rho_scaled, kinks.flatten(start_dim=1)), dim=1
    )


def _unpack_parameters(
    theta: torch.Tensor, n_nodes: int, frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    origin, axis0, axis1 = frame
    position = origin + theta[:, 0:1] * axis0 + theta[:, 1:2] * axis1
    reference = assemble_track_parameters(
        torch.cat((position, theta[:, 2:4]), dim=1), theta[:, 4:5]
    )
    kinks = theta[:, 5:].reshape(theta.shape[0], n_nodes, 2)
    return reference, kinks


def broken_helix_local_prediction(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict local (LOC0, LOC1) on every surface through the broken helix.

    ``hit_s`` is only used to pick each hit's segment; the arc length that
    actually reaches the surface comes from the plane intersection, so a kink
    upstream moves the prediction exactly as it should.
    """
    per_hit = broken_helix_track_parameters(
        reference, kinks, node_s, node_valid, hit_s
    )
    _path, intersections, valid = intersect_plane(
        per_hit, centers, rotations[..., :, 2], n_newton=12
    )
    local3 = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)
    return local3[..., :2], valid


def _standardized_residual(
    theta: torch.Tensor,
    n_nodes: int,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    kink_sigma: torch.Tensor,
    measured: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Return ``[T, 2H + 2N]``: whitened measurements stacked on whitened kinks.

    Writing the scattering prior as extra standardized residuals makes the
    Gauss-Newton normal equations exactly ``J^T J = H`` of ``F = 0.5 chi2``,
    with no separate prior term to get the factor wrong on.
    """
    reference, kinks = _unpack_parameters(theta, n_nodes, frame)
    local, valid = broken_helix_local_prediction(
        reference, kinks, node_s, node_valid, hit_s, centers, rotations
    )
    residual = (measured - local) / variance.clamp_min(EPS_VARIANCE).sqrt()
    active = mask & valid.unsqueeze(-1)
    residual = torch.where(active, residual, torch.zeros_like(residual))
    prior = kinks / kink_sigma.clamp_min(EPS_VARIANCE).unsqueeze(-1)
    prior = torch.where(
        node_valid.unsqueeze(-1), prior, torch.zeros_like(prior)
    )
    return torch.cat((residual.flatten(1), prior.flatten(1)), dim=1)


def _single_track_residual(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    kink_sigma: torch.Tensor,
    measured: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Single-track wrapper so ``vmap`` can differentiate one row at a time."""
    return _standardized_residual(
        theta.unsqueeze(0),
        node_s.shape[0],
        (origin.unsqueeze(0), axis0.unsqueeze(0), axis1.unsqueeze(0)),
        node_s.unsqueeze(0),
        node_valid.unsqueeze(0),
        kink_sigma.unsqueeze(0),
        measured.unsqueeze(0),
        variance.unsqueeze(0),
        mask.unsqueeze(0),
        hit_s.unsqueeze(0),
        centers.unsqueeze(0),
        rotations.unsqueeze(0),
    ).squeeze(0)


_BATCHED_RESIDUAL_JACOBIAN = torch.func.vmap(
    torch.func.jacrev(_single_track_residual, argnums=0)
)


def fit_broken_helix(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    kink_sigma: torch.Tensor,
    measured: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | bool]]:
    """MAP-fit the reference helix and its kinks by damped Gauss-Newton.

    Gauss-Newton rather than Adam: the closure that validates the scattering
    prior needs the actual MAP, and a first-order optimiser stopped after a
    fixed budget leaves a shrinkage bias that is indistinguishable from a
    mis-scaled prior.  Each track carries its own damping and a step is kept
    only when its objective decreases, so a bad linearisation costs one
    iteration instead of the fit.
    """
    n_nodes = node_s.shape[1]
    frame = reference_gauge_frame(reference)
    arguments = (
        node_s,
        node_valid,
        kink_sigma,
        measured,
        variance,
        mask,
        hit_s,
        centers,
        rotations,
    )
    theta = _pack_parameters(reference, kinks).clone()

    def residual_of(values: torch.Tensor) -> torch.Tensor:
        return _standardized_residual(values, n_nodes, frame, *arguments)

    def objective(values: torch.Tensor) -> torch.Tensor:
        return 0.5 * residual_of(values).square().sum(dim=1)

    current = objective(theta)
    initial = float(current.mean())
    identity = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)
    level = torch.full((theta.shape[0], 1, 1), float(damping), dtype=theta.dtype)
    converged = False
    iterations = 0
    gradient = torch.zeros_like(theta)
    for iterations in range(1, max_iters + 1):
        jacobian = _BATCHED_RESIDUAL_JACOBIAN(theta, *frame, *arguments)
        residual = residual_of(theta)
        hessian = jacobian.transpose(1, 2) @ jacobian
        gradient = (jacobian.transpose(1, 2) @ residual.unsqueeze(-1)).squeeze(-1)
        scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
        regularized = hessian + level * scale[:, None, None] * identity
        step = torch.linalg.solve(regularized, -gradient.unsqueeze(-1)).squeeze(-1)
        candidate = theta + step
        candidate_objective = objective(candidate)
        improved = candidate_objective < current
        theta = torch.where(improved.unsqueeze(-1), candidate, theta)
        level = torch.where(
            improved[:, None, None],
            (level * 0.1).clamp_min(GAUSS_NEWTON_MIN_DAMPING),
            level * 10.0,
        )
        decrease = (current - candidate_objective).clamp_min(0.0) / current.abs().clamp_min(1.0)
        current = torch.where(improved, candidate_objective, current)
        change = float(decrease.max())
        if change < tolerance:
            converged = True
            break
    final_reference, final_kinks = _unpack_parameters(theta, n_nodes, frame)
    return (
        final_reference.detach(),
        final_kinks.detach(),
        {
            "converged": bool(converged),
            "iterations": int(iterations),
            "initial_objective": initial,
            "final_objective": float(current.mean()),
            "max_relative_decrease": float(change),
            "max_gradient_norm": float(gradient.norm(dim=1).max()),
        },
    )


def _broken_predictive_pieces(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    kink_sigma: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    holdout_s: torch.Tensor,
    holdout_centers: torch.Tensor,
    holdout_rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Posterior precision and prediction Jacobian, before any damping.

    Split out so a damping scan reuses one pair of expensive ``vmap`` Jacobian
    passes instead of recomputing them per damping value.
    """
    n_nodes = node_s.shape[1]
    frame = reference_gauge_frame(reference)
    theta = _pack_parameters(reference, kinks)
    arguments = (
        node_s,
        node_valid,
        kink_sigma,
        torch.zeros_like(variance),
        variance,
        mask,
        hit_s,
        centers,
        rotations,
    )
    jacobian = _BATCHED_RESIDUAL_JACOBIAN(theta, *frame, *arguments)
    hessian = jacobian.transpose(1, 2) @ jacobian

    def predict(values: torch.Tensor) -> torch.Tensor:
        predicted_reference, predicted_kinks = _unpack_parameters(values, n_nodes, frame)
        local, _valid = broken_helix_local_prediction(
            predicted_reference,
            predicted_kinks,
            node_s,
            node_valid,
            holdout_s,
            holdout_centers,
            holdout_rotations,
        )
        return local.flatten(1)

    def predict_single(theta_row, origin_row, axis0_row, axis1_row, node_s_row,
                       node_valid_row, holdout_row, center_row, rotation_row):
        predicted_reference, predicted_kinks = _unpack_parameters(
            theta_row.unsqueeze(0),
            n_nodes,
            (origin_row.unsqueeze(0), axis0_row.unsqueeze(0), axis1_row.unsqueeze(0)),
        )
        local, _valid = broken_helix_local_prediction(
            predicted_reference,
            predicted_kinks,
            node_s_row.unsqueeze(0),
            node_valid_row.unsqueeze(0),
            holdout_row.unsqueeze(0),
            center_row.unsqueeze(0),
            rotation_row.unsqueeze(0),
        )
        return local.squeeze(0).flatten()

    prediction_jacobian = torch.func.vmap(
        torch.func.jacrev(predict_single, argnums=0)
    )(theta, *frame, node_s, node_valid, holdout_s, holdout_centers, holdout_rotations)

    predicted = predict(theta).reshape(holdout_s.shape[0], -1, 2)
    return predicted.detach(), hessian, prediction_jacobian


def broken_helix_predictive_distribution(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    kink_sigma: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    holdout_s: torch.Tensor,
    holdout_centers: torch.Tensor,
    holdout_rotations: torch.Tensor,
    damping: float = PRODUCTION_DAMPING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict held-out surfaces and propagate the posterior onto them.

    The returned variance covers only the fitted parameters.  Callers add the
    held-out surface's own measurement variance before forming a pull; the
    point of the exercise is that leaving the parameter term out is exactly the
    error that made the single helix look mis-calibrated.
    """
    predicted, hessian, prediction_jacobian = _broken_predictive_pieces(
        reference,
        kinks,
        node_s,
        node_valid,
        kink_sigma,
        variance,
        mask,
        hit_s,
        centers,
        rotations,
        holdout_s,
        holdout_centers,
        holdout_rotations,
    )
    covariance, _diagnostics = predictive_covariance(
        hessian, prediction_jacobian, damping=damping
    )
    return predicted, covariance.diagonal(dim1=1, dim2=2).reshape(
        holdout_s.shape[0], -1, 2
    ).detach()


# ---------------------------------------------------------------------------
# Gate 4: paired single/broken held-out comparison
# ---------------------------------------------------------------------------

#: All of the following were pre-registered at Gate 1 and must not be tuned
#: after looking at a result.  See ``docs/history/ml_optimization_log.md``.
#: Bumped whenever any constant in this block changes meaning or value, so a
#: summary written under one set of rules can never be mistaken for another.
#: v1: Gate 1 (Iteration 13).  v2: Iteration 17, damping (see PRODUCTION_DAMPING).
#: v3: Iteration 22, the extrapolation residual floor (see
#: DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS).  That constant lives outside this
#: block, but it decides which rows reach Gate 4 at all, so a v2 summary and a v3
#: summary are not comparable and must not be read as one series.
PREREGISTRATION_VERSION = 3
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260825
MIN_GROUP_COMPONENTS = 200
INTERPOLATION_RMSE_IMPROVEMENT = 0.20
EXTRAPOLATION_RMSE_REGRESSION = 0.05
PREDICTIVE_CHI2_BAND = (0.7, 1.5)
MAX_ABS_PULL_MEAN = 0.2
MAX_MISSING_MATERIAL_FRACTION = 0.02
#: Gate 0 left the energy-loss item open; these close it or reopen the choice.
ENERGY_LOSS_MAX_SEGMENT_FLIP_RATE = 0.0
ENERGY_LOSS_MAX_PREDICTION_SIGMA_P99 = 0.10
#: Registered before the diagnostic was ever evaluated: if more than this share of
#: rows is still non-stationary, the run is a numerical failure and its verdict
#: says nothing about the physics either way.
MAX_NON_STATIONARY_ROW_FRACTION = 0.01

COMPONENT_NAMES = ("LOC0", "LOC1")
MODEL_ARMS = ("single", "broken")

#: A group with no statistics cannot vacuum-pass the gate: every entry here must
#: reach ``MIN_GROUP_COMPONENTS`` or the whole verdict is ``inconclusive``.
REQUIRED_MODES = PREDICTION_MODES
REQUIRED_COMPONENT = "LOC0"

MATERIAL_ARTIFACT_FIELDS = (
    "event_nr",
    "track_nr",
    "q_over_p_gev_inv",
    "node_offsets",
    "position_mm",
    "x_over_x0",
)


def json_safe(value):
    """Replace non-finite floats with ``None`` so the summary stays valid JSON.

    ``inf`` shows up honestly -- an undamped condition number, a bootstrap CI
    with no finite resamples -- and the summary is written with
    ``allow_nan=False`` on purpose, so it has to be represented rather than
    silently dropped.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@dataclass
class MaterialNodes:
    """Per-track material budget, addressed by ``(event_nr, track_nr)``."""

    lookup: dict[tuple[int, int], tuple[int, int, int]]
    position_mm: np.ndarray
    x_over_x0: np.ndarray
    q_over_p_gev_inv: np.ndarray
    source: str


def load_material_artifact(path: str | Path) -> MaterialNodes:
    """Read a schema-1.1 material NPZ and index it by track key."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"material artifact does not exist: {path}")
    with np.load(path) as data:
        missing = [name for name in MATERIAL_ARTIFACT_FIELDS if name not in data.files]
        if missing:
            raise KeyError(
                f"material artifact {path} is missing fields: {sorted(missing)}"
            )
        event_nr = np.asarray(data["event_nr"], dtype=np.int64)
        track_nr = np.asarray(data["track_nr"], dtype=np.int64)
        offsets = np.asarray(data["node_offsets"], dtype=np.int64)
        position_mm = np.asarray(data["position_mm"], dtype=np.float64)
        x_over_x0 = np.asarray(data["x_over_x0"], dtype=np.float64)
        q_over_p = np.asarray(data["q_over_p_gev_inv"], dtype=np.float64)
    if offsets.shape[0] != event_nr.shape[0] + 1:
        raise ValueError("node_offsets length does not match the track count.")
    lookup: dict[tuple[int, int], tuple[int, int, int]] = {}
    for row, (event, track) in enumerate(zip(event_nr.tolist(), track_nr.tolist())):
        key = (int(event), int(track))
        if key in lookup:
            raise ValueError(f"duplicate track key in material artifact: {key}")
        lookup[key] = (int(offsets[row]), int(offsets[row + 1]), row)
    return MaterialNodes(
        lookup=lookup,
        position_mm=position_mm,
        x_over_x0=x_over_x0,
        q_over_p_gev_inv=q_over_p,
        source=str(path),
    )


def material_row_coverage(
    batches: TrackBatches, material: MaterialNodes
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split batch rows into those with material and those without."""
    present: list[int] = []
    missing: list[tuple[int, int]] = []
    for row in range(batches.event_track_ids.shape[0]):
        key = (
            int(batches.event_track_ids[row, 0]),
            int(batches.event_track_ids[row, 1]),
        )
        if key in material.lookup:
            present.append(row)
        else:
            missing.append(key)
    return present, missing


def filter_track_batches(batches: TrackBatches, rows: list[int]) -> TrackBatches:
    """Keep the given batch rows.  Both arms must be handed the same result."""
    if not rows:
        raise ValueError("No batch rows survive material matching.")
    index = torch.tensor(rows, dtype=torch.long)
    return TrackBatches(
        measured_local=batches.measured_local[index],
        measurement_variance=batches.measurement_variance[index],
        loc1_valid=batches.loc1_valid[index],
        smt_local=batches.smt_local[index],
        sensor_idx=batches.sensor_idx[index],
        hit_mask=batches.hit_mask[index],
        fit_mask=batches.fit_mask[index],
        holdout_mask=batches.holdout_mask[index],
        g_smt=batches.g_smt[index],
        p_smt=batches.p_smt[index],
        volume_id=batches.volume_id[index],
        layer_id=batches.layer_id[index],
        module_id=batches.module_id[index],
        event_track_ids=batches.event_track_ids[index],
        fit_span=batches.fit_span[index],
        prediction_mode=tuple(batches.prediction_mode[row] for row in rows),
        context_hits_before=batches.context_hits_before[index],
        context_hits_after=batches.context_hits_after[index],
        nearest_context_distance=batches.nearest_context_distance[index],
        farthest_context_distance=batches.farthest_context_distance[index],
    )


def gather_row_nodes(
    batches: TrackBatches,
    material: MaterialNodes,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each row's material nodes to a common width.

    The same physical track appears in several rows -- one per held-out hit and
    prediction mode -- and its nodes are copied into every one of them.  That is
    deliberate: each row is an independent local fit with its own window.
    """
    n_rows = batches.event_track_ids.shape[0]
    spans = []
    for row in range(n_rows):
        key = (
            int(batches.event_track_ids[row, 0]),
            int(batches.event_track_ids[row, 1]),
        )
        spans.append(material.lookup[key])
    max_nodes = max((hi - lo) for lo, hi, _ in spans)
    position = torch.zeros((n_rows, max_nodes, 3), dtype=dtype, device=device)
    x_over_x0 = torch.zeros((n_rows, max_nodes), dtype=dtype, device=device)
    present = torch.zeros((n_rows, max_nodes), dtype=torch.bool, device=device)
    q_over_p = torch.zeros(n_rows, dtype=dtype, device=device)
    for row, (lo, hi, track_row) in enumerate(spans):
        count = hi - lo
        position[row, :count] = torch.as_tensor(
            material.position_mm[lo:hi], dtype=dtype, device=device
        )
        x_over_x0[row, :count] = torch.as_tensor(
            material.x_over_x0[lo:hi], dtype=dtype, device=device
        )
        present[row, :count] = True
        q_over_p[row] = float(material.q_over_p_gev_inv[track_row])
    return position, x_over_x0, present, q_over_p


@dataclass
class BrokenHelixInputs:
    """Everything the broken-helix solver needs for one arm, already frozen."""

    node_s: torch.Tensor
    node_valid: torch.Tensor
    kink_sigma: torch.Tensor
    hit_s: torch.Tensor
    centers: torch.Tensor
    rotations: torch.Tensor
    measured: torch.Tensor
    variance: torch.Tensor
    mask: torch.Tensor
    holdout_s: torch.Tensor
    holdout_centers: torch.Tensor
    holdout_rotations: torch.Tensor
    holdout_measured: torch.Tensor
    holdout_variance: torch.Tensor
    holdout_loc1_valid: torch.Tensor
    n_nodes_in_window: torch.Tensor


def prepare_broken_inputs(
    batches: TrackBatches,
    geometry: NominalGeometry,
    reference: torch.Tensor,
    node_position: torch.Tensor | None,
    node_x_over_x0: torch.Tensor | None,
    node_present: torch.Tensor | None,
    q_over_p: torch.Tensor | None,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> BrokenHelixInputs:
    """Freeze the linearisation: recompute every arc length from ``reference``.

    ``node_s`` and ``hit_s`` come from the same frozen helix, which is the only
    way the window clip and the segment assignment can be mutually consistent --
    the exporter's own path length is a different gauge and is never used here.
    Passing ``node_position=None`` builds the nodeless (single-helix) arm, which
    then runs through exactly the same solver.
    """
    reference = reference.to(device=device, dtype=dtype)
    n_rows, n_hits = batches.hit_mask.shape
    sensor_idx = batches.sensor_idx.to(device=device)
    centers = geometry.centers.to(device=device, dtype=dtype)[sensor_idx]
    rotations = geometry.rotations.to(device=device, dtype=dtype)[sensor_idx]
    measured = batches.measured_local.to(device=device, dtype=dtype)
    variance = batches.measurement_variance.to(device=device, dtype=dtype)
    fit_mask = batches.fit_mask.to(device=device)
    holdout_mask = batches.holdout_mask.to(device=device)
    hit_mask = batches.hit_mask.to(device=device)
    loc1_valid = batches.loc1_valid.to(device=device)

    points = batches.g_smt.to(device=device, dtype=dtype)
    _fitted, hit_s, _ok = closest_helix_points(
        reference[:, None, :].expand(-1, n_hits, -1), points
    )
    hit_s = hit_s.detach()

    # Fit window: only material the trajectory actually crosses between the
    # first and last hit of THIS row may bend it.  Without the clip an
    # extrapolating row would pick up kinks it never traverses.
    large = torch.full_like(hit_s, float("inf"))
    window_low = torch.where(hit_mask, hit_s, large).min(dim=1).values
    window_high = torch.where(hit_mask, hit_s, -large).max(dim=1).values

    if node_position is None:
        node_s = hit_s.new_zeros((n_rows, 0))
        node_valid = torch.zeros((n_rows, 0), dtype=torch.bool, device=device)
        kink_sigma = hit_s.new_zeros((n_rows, 0))
    else:
        node_s, node_ok = node_arc_lengths(reference, node_position, node_present)
        inside = (node_s >= window_low[:, None]) & (node_s <= window_high[:, None])
        node_valid = node_ok & inside
        kink_sigma = highland_kink_sigma(
            torch.where(node_valid, node_x_over_x0, torch.zeros_like(node_x_over_x0)),
            q_over_p,
        )
        # The segment recursion walks nodes in index order, so valid nodes must
        # be sorted by arc length; masked ones are parked at the end.
        order = torch.argsort(
            torch.where(node_valid, node_s, torch.full_like(node_s, float("inf"))),
            dim=1,
        )
        node_s = torch.gather(node_s, 1, order)
        node_valid = torch.gather(node_valid, 1, order)
        kink_sigma = torch.gather(kink_sigma, 1, order)
        keep = int(node_valid.sum(dim=1).max())
        node_s = node_s[:, :keep].contiguous()
        node_valid = node_valid[:, :keep].contiguous()
        kink_sigma = kink_sigma[:, :keep].contiguous()

    component_mask = torch.stack((fit_mask, fit_mask & loc1_valid), dim=-1)
    holdout_index = holdout_mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(n_rows, device=device)
    return BrokenHelixInputs(
        node_s=node_s,
        node_valid=node_valid,
        kink_sigma=kink_sigma,
        hit_s=hit_s,
        centers=centers,
        rotations=rotations,
        measured=measured,
        variance=variance,
        mask=component_mask,
        holdout_s=hit_s[rows, holdout_index][:, None],
        holdout_centers=centers[rows, holdout_index][:, None],
        holdout_rotations=rotations[rows, holdout_index][:, None],
        holdout_measured=measured[rows, holdout_index][:, None],
        holdout_variance=variance[rows, holdout_index][:, None],
        holdout_loc1_valid=loc1_valid[rows, holdout_index],
        n_nodes_in_window=node_valid.sum(dim=1),
    )


#: One ``vmap``-of-``jacrev`` pass materialises a [rows, residuals, parameters]
#: Jacobian plus its autograd intermediates, so a full-detector run is chunked.
#: Rows are independent, which makes the split exact rather than approximate.
BROKEN_ARM_CHUNK_ROWS = 2048


def _slice_inputs(inputs: BrokenHelixInputs, start: int, stop: int) -> BrokenHelixInputs:
    return BrokenHelixInputs(
        **{
            name: getattr(inputs, name)[start:stop]
            for name in inputs.__dataclass_fields__
        }
    )


def run_broken_arm(
    inputs: BrokenHelixInputs,
    reference: torch.Tensor,
    dampings: tuple[float, ...] = DAMPING_SCAN,
    production_damping: float = PRODUCTION_DAMPING,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    chunk_rows: int = BROKEN_ARM_CHUNK_ROWS,
) -> dict:
    """MAP-fit one arm and evaluate it on the held-out surface, in row chunks."""
    n_rows = reference.shape[0]
    if chunk_rows <= 0 or n_rows <= chunk_rows:
        return _run_broken_arm_chunk(
            inputs, reference, dampings, production_damping, max_iters
        )
    chunks = [
        _run_broken_arm_chunk(
            _slice_inputs(inputs, start, start + chunk_rows),
            reference[start : start + chunk_rows],
            dampings,
            production_damping,
            max_iters,
        )
        for start in range(0, n_rows, chunk_rows)
    ]
    merged = {
        name: torch.cat([chunk[name] for chunk in chunks], dim=0)
        for name in (
            "reference",
            "kinks",
            "predicted",
            "predictive_variance",
            "intersection_valid",
            "segment_of_hit",
            "newton_decrement",
        )
    }
    merged["predictive_variance_by_damping"] = {
        damping: (
            None
            if any(
                chunk["predictive_variance_by_damping"][damping] is None
                for chunk in chunks
            )
            else torch.cat(
                [chunk["predictive_variance_by_damping"][damping] for chunk in chunks],
                dim=0,
            )
        )
        for damping in chunks[0]["predictive_variance_by_damping"]
    }
    merged["convergence"] = _merge_convergence(
        [chunk["convergence"] for chunk in chunks],
        [float(chunk["predicted"].shape[0]) for chunk in chunks],
    )
    merged["covariance_diagnostics"] = {
        key: _merge_covariance_diagnostics(
            [chunk["covariance_diagnostics"][key] for chunk in chunks]
        )
        for key in chunks[0]["covariance_diagnostics"]
    }
    return merged


def _merge_covariance_diagnostics(entries: list[dict]) -> dict:
    """Worst case across chunks, so one bad track cannot average itself away."""
    failures = [entry for entry in entries if "error" in entry]
    if failures:
        return {"error": failures[0]["error"], "n_failed_chunks": len(failures)}
    return {
        "cholesky_ok": all(entry["cholesky_ok"] for entry in entries),
        "damping": entries[0]["damping"],
        "symmetry_residual": max(entry["symmetry_residual"] for entry in entries),
        "batch_size": sum(entry["batch_size"] for entry in entries),
        "min_eigenvalue": min(entry["min_eigenvalue"] for entry in entries),
        "max_eigenvalue": max(entry["max_eigenvalue"] for entry in entries),
        "condition_number": max(entry["condition_number"] for entry in entries),
        "n_inactive_directions": max(
            entry["n_inactive_directions"] for entry in entries
        ),
        # Summed, not maxed: these rows are dropped from the comparison and the
        # gate needs the total, not the worst chunk.
        "n_rank_deficient_rows": sum(
            entry["n_rank_deficient_rows"] for entry in entries
        ),
    }


#: A row counts as stationary when the Newton decrement -- the objective drop the
#: quadratic model still promises -- is below this.  ``max_gradient_norm`` alone
#: cannot decide it: the gradient is in whitened units, so a row with a 0.01 mm
#: sigma carries a numerically large gradient while sitting at its optimum.
NEWTON_DECREMENT_STATIONARY = 1.0e-6


def newton_decrement(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    kink_sigma: torch.Tensor,
    measured: torch.Tensor,
    variance: torch.Tensor,
    mask: torch.Tensor,
    hit_s: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    damping: float = GAUSS_NEWTON_MIN_DAMPING,
) -> torch.Tensor:
    """Per-row ``g^T H^-1 g``: the objective decrease still on the table.

    This is the scale-free convergence statement.  Reporting it per row is what
    lets a negative physics result be separated from a handful of rows whose
    linearisation stalled.
    """
    n_nodes = node_s.shape[1]
    frame = reference_gauge_frame(reference)
    arguments = (
        node_s, node_valid, kink_sigma, measured, variance, mask,
        hit_s, centers, rotations,
    )
    theta = _pack_parameters(reference, kinks)
    jacobian = _BATCHED_RESIDUAL_JACOBIAN(theta, *frame, *arguments)
    residual = _standardized_residual(theta, n_nodes, frame, *arguments)
    hessian = jacobian.transpose(1, 2) @ jacobian
    gradient = (jacobian.transpose(1, 2) @ residual.unsqueeze(-1)).squeeze(-1)
    scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    identity = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)
    regularized = hessian + damping * scale[:, None, None] * identity
    step = torch.linalg.solve(regularized, gradient.unsqueeze(-1)).squeeze(-1)
    return (gradient * step).sum(dim=1).detach()


def stationarity_report(decrement: torch.Tensor) -> dict[str, float | int]:
    """Summarise per-row Newton decrements into the pre-registered gate inputs."""
    finite = decrement[torch.isfinite(decrement)]
    quantile = (
        (lambda q: float(finite.quantile(q)))
        if finite.numel()
        else (lambda q: float("nan"))
    )
    return {
        "stationary_threshold": NEWTON_DECREMENT_STATIONARY,
        "non_stationary_row_fraction": (
            float((decrement > NEWTON_DECREMENT_STATIONARY).to(torch.float64).mean())
            if decrement.numel()
            else 0.0
        ),
        "n_non_finite": int(decrement.numel() - finite.numel()),
        "newton_decrement_p50": quantile(0.50),
        "newton_decrement_p90": quantile(0.90),
        "newton_decrement_p99": quantile(0.99),
        "newton_decrement_max": float(finite.max()) if finite.numel() else float("nan"),
    }


def _merge_convergence(stats: list[dict], weights: list[float]) -> dict:
    """Row-count weighted objectives, worst case for every failure statistic."""
    total = float(sum(weights)) or 1.0
    share = [weight / total for weight in weights]
    return {
        "converged": all(entry["converged"] for entry in stats),
        "iterations": max(entry["iterations"] for entry in stats),
        "initial_objective": float(
            sum(w * entry["initial_objective"] for w, entry in zip(share, stats))
        ),
        "final_objective": float(
            sum(w * entry["final_objective"] for w, entry in zip(share, stats))
        ),
        "max_relative_decrease": max(entry["max_relative_decrease"] for entry in stats),
        "max_gradient_norm": max(entry["max_gradient_norm"] for entry in stats),
        "n_chunks": len(stats),
    }


def _measurement_fit_inputs(
    batches: TrackBatches,
    geometry: NominalGeometry,
    parameters: torch.Tensor,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> BrokenHelixInputs:
    """The nodeless (single-helix) inputs: one frozen linearisation per row."""
    return prepare_broken_inputs(
        batches, geometry, parameters, None, None, None, None,
        device=device, dtype=dtype,
    )


def fit_helices_gauss_newton(
    batches: TrackBatches,
    geometry: NominalGeometry,
    initial_parameters: torch.Tensor,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    chunk_rows: int = BROKEN_ARM_CHUNK_ROWS,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> tuple[torch.Tensor, dict]:
    """Refit helices to the non-holdout measurements by damped Gauss-Newton.

    This is ``fit_helices_to_measurements`` minus the optimiser.  The objective,
    the component mask and the held-out surface are identical; only the way the
    minimum is reached changes.  Iteration 14 measured the difference on real
    data: Adam stopped on budget at 17:8 interpolation LOC0 RMSE 0.076 mm where
    the same objective, solved properly, reaches 0.022 mm.  What Iteration 10
    read as unmodelled multiple scattering was largely this gap.

    Runs through ``fit_broken_helix`` with zero nodes rather than a second
    solver, so the main path and the Gate 4 single arm cannot drift apart.
    """
    reference = initial_parameters.to(device=device, dtype=dtype)
    inputs = _measurement_fit_inputs(
        batches, geometry, reference, device=device, dtype=dtype
    )
    n_rows = reference.shape[0]
    step = n_rows if chunk_rows <= 0 else chunk_rows
    fitted: list[torch.Tensor] = []
    decrements: list[torch.Tensor] = []
    stats: list[dict] = []
    weights: list[float] = []
    for start in range(0, n_rows, max(step, 1)):
        stop = min(start + max(step, 1), n_rows)
        piece = _slice_inputs(inputs, start, stop)
        kinks = piece.node_s.new_zeros((stop - start, 0, 2))
        block, block_kinks, block_stats = fit_broken_helix(
            reference[start:stop],
            kinks,
            piece.node_s,
            piece.node_valid,
            piece.kink_sigma,
            piece.measured,
            piece.variance,
            piece.mask,
            piece.hit_s,
            piece.centers,
            piece.rotations,
            max_iters=max_iters,
        )
        fitted.append(block)
        decrements.append(
            newton_decrement(
                block,
                block_kinks,
                piece.node_s,
                piece.node_valid,
                piece.kink_sigma,
                piece.measured,
                piece.variance,
                piece.mask,
                piece.hit_s,
                piece.centers,
                piece.rotations,
            )
        )
        stats.append(block_stats)
        weights.append(float(stop - start))
    parameters = torch.cat(fitted, dim=0)
    decrement = torch.cat(decrements, dim=0)
    return parameters, {
        "solver": "gauss-newton",
        "convergence": _merge_convergence(stats, weights),
        "stationarity": stationarity_report(decrement),
    }


def measurement_newton_decrement(
    batches: TrackBatches,
    geometry: NominalGeometry,
    parameters: torch.Tensor,
    chunk_rows: int = BROKEN_ARM_CHUNK_ROWS,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Per-row ``g^T H^-1 g`` of the measurement fit, whatever produced it.

    Applying the same scale-free stationarity test to the Adam output is how
    "the solver stopped early" becomes a measured statement rather than an
    inference from a large whitened gradient norm.  Chunked for the same reason
    the arms are: one ``jacrev`` pass materialises a full per-row Jacobian.
    """
    reference = parameters.to(device=device, dtype=dtype)
    inputs = _measurement_fit_inputs(
        batches, geometry, reference, device=device, dtype=dtype
    )
    n_rows = reference.shape[0]
    step = n_rows if chunk_rows <= 0 else chunk_rows
    pieces = []
    for start in range(0, n_rows, max(step, 1)):
        stop = min(start + max(step, 1), n_rows)
        piece = _slice_inputs(inputs, start, stop)
        pieces.append(
            newton_decrement(
                reference[start:stop],
                piece.node_s.new_zeros((stop - start, 0, 2)),
                piece.node_s,
                piece.node_valid,
                piece.kink_sigma,
                piece.measured,
                piece.variance,
                piece.mask,
                piece.hit_s,
                piece.centers,
                piece.rotations,
            )
        )
    return torch.cat(pieces, dim=0)


def _run_broken_arm_chunk(
    inputs: BrokenHelixInputs,
    reference: torch.Tensor,
    dampings: tuple[float, ...] = DAMPING_SCAN,
    production_damping: float = PRODUCTION_DAMPING,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
) -> dict:
    """MAP-fit one arm and evaluate it on the held-out surface."""
    n_rows = reference.shape[0]
    n_nodes = inputs.node_s.shape[1]
    kinks = inputs.node_s.new_zeros((n_rows, n_nodes, 2))
    fitted_reference, fitted_kinks, stats = fit_broken_helix(
        reference,
        kinks,
        inputs.node_s,
        inputs.node_valid,
        inputs.kink_sigma,
        inputs.measured,
        inputs.variance,
        inputs.mask,
        inputs.hit_s,
        inputs.centers,
        inputs.rotations,
        max_iters=max_iters,
    )
    decrement = newton_decrement(
        fitted_reference,
        fitted_kinks,
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
    predicted, hessian, prediction_jacobian = _broken_predictive_pieces(
        fitted_reference,
        fitted_kinks,
        inputs.node_s,
        inputs.node_valid,
        inputs.kink_sigma,
        inputs.variance,
        inputs.mask,
        inputs.hit_s,
        inputs.centers,
        inputs.rotations,
        inputs.holdout_s,
        inputs.holdout_centers,
        inputs.holdout_rotations,
    )
    _local, valid = broken_helix_local_prediction(
        fitted_reference,
        fitted_kinks,
        inputs.node_s,
        inputs.node_valid,
        inputs.holdout_s,
        inputs.holdout_centers,
        inputs.holdout_rotations,
    )
    variance_by_damping = {}
    diagnostics_by_damping = {}
    for damping in dict.fromkeys((*dampings, production_damping)):
        try:
            covariance, diagnostics = predictive_covariance(
                hessian, prediction_jacobian, damping=damping
            )
        except torch.linalg.LinAlgError as error:
            variance_by_damping[damping] = None
            diagnostics_by_damping[damping] = {"error": str(error)}
            continue
        variance_by_damping[damping] = (
            covariance.diagonal(dim1=1, dim2=2).reshape(n_rows, -1, 2).detach()
        )
        diagnostics_by_damping[damping] = diagnostics
    production = variance_by_damping.get(production_damping)
    if production is None:
        raise FloatingPointError(
            "Predictive covariance is not solvable at the pre-registered "
            f"production damping {production_damping}."
        )
    return {
        "reference": fitted_reference,
        "kinks": fitted_kinks,
        "convergence": stats,
        "predicted": predicted[:, 0, :],
        "predictive_variance": production[:, 0, :],
        "predictive_variance_by_damping": {
            key: None if value is None else value[:, 0, :]
            for key, value in variance_by_damping.items()
        },
        "covariance_diagnostics": {
            str(key): value for key, value in diagnostics_by_damping.items()
        },
        "newton_decrement": decrement,
        "intersection_valid": valid[:, 0],
        "segment_of_hit": assign_hit_segments(
            inputs.node_s, inputs.holdout_s, inputs.node_valid
        )[:, 0],
    }


def _component_arrays(
    arm: dict, inputs: BrokenHelixInputs, component: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual, total sigma^2 and validity for one local component."""
    residual = (arm["predicted"][:, component] - inputs.holdout_measured[:, 0, component])
    total_variance = (
        inputs.holdout_variance[:, 0, component].clamp_min(EPS_VARIANCE)
        + arm["predictive_variance"][:, component].clamp_min(0.0)
    )
    valid = arm["intersection_valid"].clone()
    if component == 1:
        valid = valid & inputs.holdout_loc1_valid
    valid = valid & torch.isfinite(residual) & torch.isfinite(total_variance)
    return (
        residual.detach().cpu().numpy(),
        total_variance.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
    )


def _diagonal_nll(residual: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """Component-wise diagonal Gaussian NLL, the pre-registered gate statistic."""
    return 0.5 * (np.log(2.0 * math.pi * total_variance) + residual**2 / total_variance)


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def paired_bootstrap_ci(
    single: np.ndarray,
    broken: np.ndarray,
    statistic,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Percentile CI of a paired statistic; rows are resampled together."""
    n_rows = single.shape[0]
    if n_rows == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan")}
    draws = rng.integers(0, n_rows, size=(resamples, n_rows))
    samples = np.array(
        [statistic(single[draw], broken[draw]) for draw in draws], dtype=np.float64
    )
    finite = samples[np.isfinite(samples)]
    low, high = (
        (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
        if finite.size
        else (float("nan"), float("nan"))
    )
    return {
        "point": float(statistic(single, broken)),
        "low": low,
        "high": high,
        "n_finite_resamples": int(finite.size),
    }


def _relative_rmse_change(single: np.ndarray, broken: np.ndarray) -> float:
    base = _rmse(single)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return (_rmse(broken) - base) / base


def _mean_difference(single: np.ndarray, broken: np.ndarray) -> float:
    return float(np.mean(broken) - np.mean(single))


def compare_arms(
    batches: TrackBatches,
    inputs: BrokenHelixInputs,
    arms: dict[str, dict],
    holdout_layers,
    rng: np.random.Generator,
) -> dict:
    """Per (layer, mode, component) paired comparison with bootstrap CIs."""
    modes = np.array(batches.prediction_mode)
    holdout_index = batches.holdout_mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(batches.holdout_mask.shape[0])
    holdout_volume = batches.volume_id[rows, holdout_index].numpy()
    holdout_layer = batches.layer_id[rows, holdout_index].numpy()

    arrays = {
        arm: {
            component: _component_arrays(arms[arm], inputs, component)
            for component in (0, 1)
        }
        for arm in MODEL_ARMS
    }
    groups: dict[str, dict] = {}
    for volume, layer in holdout_layers:
        layer_key = f"{volume}:{layer}"
        layer_rows = (holdout_volume == volume) & (holdout_layer == layer)
        groups[layer_key] = {}
        for mode in PREDICTION_MODES:
            mode_rows = layer_rows & (modes == mode)
            groups[layer_key][mode] = {}
            for component, name in enumerate(COMPONENT_NAMES):
                valid = mode_rows.copy()
                for arm in MODEL_ARMS:
                    valid &= arrays[arm][component][2]
                selection = np.flatnonzero(valid)
                entry = {
                    "n": int(selection.size),
                    "n_rows_in_group": int(np.count_nonzero(mode_rows)),
                }
                if selection.size == 0:
                    groups[layer_key][mode][name] = entry
                    continue
                residuals = {}
                nll = {}
                for arm in MODEL_ARMS:
                    residual, total_variance, _ok = arrays[arm][component]
                    residuals[arm] = residual[selection]
                    nll[arm] = _diagonal_nll(
                        residual[selection], total_variance[selection]
                    )
                    pull = residual[selection] / np.sqrt(total_variance[selection])
                    entry[arm] = {
                        "rmse_mm": _rmse(residuals[arm]),
                        "bias_mm": float(np.mean(residuals[arm])),
                        "mean_nll": float(np.mean(nll[arm])),
                        "pull_mean": float(np.mean(pull)),
                        "pull_std": float(np.std(pull, ddof=1)) if pull.size > 1 else float("nan"),
                        "predictive_chi2_per_component": float(np.mean(pull**2)),
                        "intersection_failure_rate": float(
                            1.0
                            - np.count_nonzero(arrays[arm][component][2][mode_rows])
                            / max(int(np.count_nonzero(mode_rows)), 1)
                        ),
                    }
                entry["relative_rmse_change"] = paired_bootstrap_ci(
                    residuals["single"], residuals["broken"], _relative_rmse_change, rng
                )
                entry["delta_mean_nll"] = paired_bootstrap_ci(
                    nll["single"], nll["broken"], _mean_difference, rng
                )
                groups[layer_key][mode][name] = entry
    return groups


def _damping_sensitivity(
    batches: TrackBatches,
    inputs: BrokenHelixInputs,
    arms: dict[str, dict],
    holdout_layers,
) -> dict:
    """Does any gated conclusion flip across the pre-registered damping grid?"""
    report: dict[str, dict] = {}
    modes = np.array(batches.prediction_mode)
    holdout_index = batches.holdout_mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(batches.holdout_mask.shape[0])
    holdout_volume = batches.volume_id[rows, holdout_index].numpy()
    holdout_layer = batches.layer_id[rows, holdout_index].numpy()
    for damping in DAMPING_SCAN:
        entry: dict[str, dict] = {}
        usable = True
        for arm in MODEL_ARMS:
            if arms[arm]["predictive_variance_by_damping"].get(damping) is None:
                usable = False
        if not usable:
            report[str(damping)] = {"solvable": False}
            continue
        for volume, layer in holdout_layers:
            layer_key = f"{volume}:{layer}"
            entry[layer_key] = {}
            layer_rows = (holdout_volume == volume) & (holdout_layer == layer)
            for mode in PREDICTION_MODES:
                mode_rows = layer_rows & (modes == mode)
                per_component = {}
                for component, name in enumerate(COMPONENT_NAMES):
                    values = {}
                    valid = mode_rows.copy()
                    for arm in MODEL_ARMS:
                        residual = (
                            arms[arm]["predicted"][:, component]
                            - inputs.holdout_measured[:, 0, component]
                        )
                        total = (
                            inputs.holdout_variance[:, 0, component].clamp_min(EPS_VARIANCE)
                            + arms[arm]["predictive_variance_by_damping"][damping][
                                :, component
                            ].clamp_min(0.0)
                        )
                        ok = arms[arm]["intersection_valid"].clone()
                        if component == 1:
                            ok = ok & inputs.holdout_loc1_valid
                        ok = ok & torch.isfinite(residual) & torch.isfinite(total)
                        values[arm] = (
                            residual.detach().cpu().numpy(),
                            total.detach().cpu().numpy(),
                        )
                        valid &= ok.detach().cpu().numpy()
                    selection = np.flatnonzero(valid)
                    if selection.size == 0:
                        per_component[name] = None
                        continue
                    nll = {
                        arm: float(
                            np.mean(
                                _diagonal_nll(
                                    values[arm][0][selection], values[arm][1][selection]
                                )
                            )
                        )
                        for arm in MODEL_ARMS
                    }
                    per_component[name] = {
                        "delta_mean_nll": nll["broken"] - nll["single"],
                        "nll_improves": bool(nll["broken"] < nll["single"]),
                    }
                entry[layer_key][mode] = per_component
        report[str(damping)] = {"solvable": True, "groups": entry}
    verdicts: dict[str, set] = {}
    for damping, payload in report.items():
        if not payload.get("solvable"):
            continue
        for layer_key, by_mode in payload["groups"].items():
            for mode, by_component in by_mode.items():
                for name, values in by_component.items():
                    if values is None:
                        continue
                    verdicts.setdefault(f"{layer_key}|{mode}|{name}", set()).add(
                        values["nll_improves"]
                    )
    flipped = sorted(key for key, values in verdicts.items() if len(values) > 1)
    stability: dict[str, set] = {}
    for damping in DAMPING_STABILITY_SCAN:
        payload = report.get(str(damping))
        if payload is None or not payload.get("solvable"):
            continue
        for layer_key, by_mode in payload["groups"].items():
            for mode, by_component in by_mode.items():
                for name, values in by_component.items():
                    if values is None:
                        continue
                    stability.setdefault(f"{layer_key}|{mode}|{name}", set()).add(
                        values["nll_improves"]
                    )
    unstable = sorted(key for key, values in stability.items() if len(values) > 1)
    return {
        "by_damping": report,
        "flipped_groups": flipped,
        "stability_scan": list(DAMPING_STABILITY_SCAN),
        "unstable_groups": unstable,
        "stable": not unstable,
    }


def write_holdout_keys(
    path: Path, batches: TrackBatches, inputs: BrokenHelixInputs
) -> str:
    """Write the row identity list shared by both arms and return its SHA256."""
    holdout_index = batches.holdout_mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(batches.holdout_mask.shape[0])
    sensor = batches.sensor_idx[rows, holdout_index].tolist()
    holdout_s = inputs.holdout_s[:, 0].detach().cpu().numpy()
    records = [
        (
            int(batches.event_track_ids[row, 0]),
            int(batches.event_track_ids[row, 1]),
            batches.prediction_mode[row],
            int(sensor[row]),
            float(holdout_s[row]),
        )
        for row in range(batches.holdout_mask.shape[0])
    ]
    records.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["event_nr", "track_nr", "mode", "holdout_sensor_idx", "holdout_path_s_mm"]
        )
        for record in records:
            writer.writerow([record[0], record[1], record[2], record[3], f"{record[4]:.9e}"])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_broken_helix_gate(groups: dict, holdout_layers) -> tuple[str, list[str]]:
    """Apply the pre-registered, mode-conditioned acceptance table."""
    reasons: list[str] = []
    inconclusive: list[str] = []
    for volume, layer in holdout_layers:
        layer_key = f"{volume}:{layer}"
        for mode in REQUIRED_MODES:
            entry = groups.get(layer_key, {}).get(mode, {}).get(REQUIRED_COMPONENT)
            if entry is None or entry["n"] < MIN_GROUP_COMPONENTS:
                count = 0 if entry is None else entry["n"]
                inconclusive.append(
                    f"required group {layer_key}|{mode}|{REQUIRED_COMPONENT} has "
                    f"N={count} < {MIN_GROUP_COMPONENTS}"
                )
    for layer_key, by_mode in groups.items():
        for mode, by_component in by_mode.items():
            for name, entry in by_component.items():
                if entry["n"] < MIN_GROUP_COMPONENTS:
                    continue
                label = f"{layer_key}|{mode}|{name}"
                broken = entry["broken"]
                if broken["intersection_failure_rate"] > DEFAULT_MAX_INTERSECTION_FAILURE_RATE:
                    reasons.append(
                        f"{label}: intersection failure rate "
                        f"{broken['intersection_failure_rate']:.3f} > "
                        f"{DEFAULT_MAX_INTERSECTION_FAILURE_RATE}"
                    )
                chi2 = broken["predictive_chi2_per_component"]
                if not PREDICTIVE_CHI2_BAND[0] <= chi2 <= PREDICTIVE_CHI2_BAND[1]:
                    reasons.append(
                        f"{label}: predictive chi2/component {chi2:.3f} outside "
                        f"{PREDICTIVE_CHI2_BAND}"
                    )
                if abs(broken["pull_mean"]) > MAX_ABS_PULL_MEAN:
                    reasons.append(
                        f"{label}: |pull mean| {abs(broken['pull_mean']):.3f} > "
                        f"{MAX_ABS_PULL_MEAN}"
                    )
                nll_high = entry["delta_mean_nll"]["high"]
                if not (np.isfinite(nll_high) and nll_high < 0.0):
                    reasons.append(
                        f"{label}: held-out NLL CI upper bound {nll_high:.4g} is not < 0"
                    )
                rmse_high = entry["relative_rmse_change"]["high"]
                if mode == "interpolation" and name == "LOC0":
                    limit = -INTERPOLATION_RMSE_IMPROVEMENT
                    if not (np.isfinite(rmse_high) and rmse_high < limit):
                        reasons.append(
                            f"{label}: relative RMSE CI upper bound {rmse_high:.4g} is "
                            f"not < {limit}"
                        )
                elif mode != "interpolation":
                    limit = EXTRAPOLATION_RMSE_REGRESSION
                    if not (np.isfinite(rmse_high) and rmse_high < limit):
                        reasons.append(
                            f"{label}: relative RMSE CI upper bound {rmse_high:.4g} is "
                            f"not < {limit}"
                        )
    if inconclusive:
        return "inconclusive", inconclusive + reasons
    return ("passed" if not reasons else "failed"), reasons


def energy_loss_reverification(
    batches: TrackBatches,
    geometry: NominalGeometry,
    reference: torch.Tensor,
    baseline: dict,
    baseline_inputs: BrokenHelixInputs,
    material: MaterialNodes,
    device: str | torch.device,
    dtype,
    max_iters: int,
) -> dict:
    """Close (or reopen) the energy-loss item Gate 0 deliberately left open.

    Gate 0 only bounded the effect by an order-of-magnitude argument.  This runs
    the whole broken arm twice -- once on each node set -- and measures the two
    quantities that were pre-registered there.
    """
    position, x_over_x0, present, q_over_p = gather_row_nodes(
        batches, material, device=device, dtype=dtype
    )
    inputs = prepare_broken_inputs(
        batches,
        geometry,
        reference,
        position,
        x_over_x0,
        present,
        q_over_p,
        device=device,
        dtype=dtype,
    )
    arm = run_broken_arm(inputs, reference, max_iters=max_iters)
    flips = (arm["segment_of_hit"] != baseline["segment_of_hit"]).to(torch.float64)
    flip_rate = float(flips.mean())
    difference = (arm["predicted"] - baseline["predicted"]).abs()
    sigma = baseline_inputs.holdout_variance[:, 0, :].clamp_min(EPS_VARIANCE).sqrt()
    ratio = (difference / sigma).detach().cpu().numpy()
    valid = (
        baseline["intersection_valid"] & arm["intersection_valid"]
    ).detach().cpu().numpy()
    component_ratio = {}
    worst = 0.0
    for component, name in enumerate(COMPONENT_NAMES):
        ok = valid.copy()
        if component == 1:
            ok &= baseline_inputs.holdout_loc1_valid.detach().cpu().numpy()
        values = ratio[ok, component]
        values = values[np.isfinite(values)]
        p99 = float(np.percentile(values, 99)) if values.size else float("nan")
        component_ratio[name] = {
            "n": int(values.size),
            "p99_ratio_to_sigma": p99,
            "max_ratio_to_sigma": float(values.max()) if values.size else float("nan"),
        }
        if np.isfinite(p99):
            worst = max(worst, p99)
    passed = (
        flip_rate <= ENERGY_LOSS_MAX_SEGMENT_FLIP_RATE
        and worst <= ENERGY_LOSS_MAX_PREDICTION_SIGMA_P99
    )
    return {
        "status": "closed" if passed else "open",
        "artifact": material.source,
        "segment_flip_rate": flip_rate,
        "segment_flip_limit": ENERGY_LOSS_MAX_SEGMENT_FLIP_RATE,
        "prediction_shift": component_ratio,
        "prediction_shift_limit_p99": ENERGY_LOSS_MAX_PREDICTION_SIGMA_P99,
        "convergence": arm["convergence"],
    }


def run_paired_holdout(
    batches: TrackBatches,
    geometry: NominalGeometry,
    reference: torch.Tensor,
    material: MaterialNodes,
    holdout_layers,
    output_dir: Path,
    device: str | torch.device = "cpu",
    dtype=torch.float64,
    model: str = "both",
    on_missing_material: str = "fail",
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    energy_loss_material: MaterialNodes | None = None,
) -> dict:
    """Gate 4: run both arms on one shared row set and apply the gate."""
    if model not in {"single", "broken", "both"}:
        raise ValueError("model must be one of single, broken, both.")
    if on_missing_material not in {"fail", "drop"}:
        raise ValueError("on_missing_material must be fail or drop.")
    present, missing = material_row_coverage(batches, material)
    total_rows = batches.event_track_ids.shape[0]
    missing_fraction = len(missing) / max(total_rows, 1)
    if missing:
        if on_missing_material == "fail":
            raise ValueError(
                f"{len(missing)} of {total_rows} batch rows have no material nodes; "
                "pass --on-missing-material drop to accept a reduced row set."
            )
        if missing_fraction > MAX_MISSING_MATERIAL_FRACTION:
            raise ValueError(
                f"missing material fraction {missing_fraction:.4f} exceeds the "
                f"pre-registered limit {MAX_MISSING_MATERIAL_FRACTION}"
            )
    # Both arms are handed this one object: identical rows by construction.
    shared = filter_track_batches(batches, present) if missing else batches
    reference = reference[torch.tensor(present, dtype=torch.long)] if missing else reference
    reference = reference.to(device=device, dtype=dtype)

    position, x_over_x0, node_present, q_over_p = gather_row_nodes(
        shared, material, device=device, dtype=dtype
    )
    broken_inputs = prepare_broken_inputs(
        shared, geometry, reference, position, x_over_x0, node_present, q_over_p,
        device=device, dtype=dtype,
    )
    single_inputs = prepare_broken_inputs(
        shared, geometry, reference, None, None, None, None, device=device, dtype=dtype
    )
    arms = {
        "single": run_broken_arm(single_inputs, reference, max_iters=max_iters),
        "broken": run_broken_arm(broken_inputs, reference, max_iters=max_iters),
    }
    keys_path = output_dir / "holdout_keys.csv"
    keys_digest = write_holdout_keys(keys_path, shared, broken_inputs)
    # Cheap but not redundant: the assertion is what turns "same object" from a
    # code-reading claim into a checked one, and it costs one temporary file.
    mirror = output_dir / "holdout_keys_single_arm.csv"
    if write_holdout_keys(mirror, shared, single_inputs) != keys_digest:
        raise ValueError("The two arms do not share an identical held-out row set.")
    mirror.unlink()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = compare_arms(shared, broken_inputs, arms, holdout_layers, rng)
    status, reasons = evaluate_broken_helix_gate(groups, holdout_layers)
    sensitivity = _damping_sensitivity(shared, broken_inputs, arms, holdout_layers)
    if not sensitivity["stable"]:
        status = "failed" if status == "passed" else status
        reasons.append(
            "damping sensitivity: NLL verdict flips across the negligible-bias "
            f"dampings {list(DAMPING_STABILITY_SCAN)} for "
            f"{sensitivity['unstable_groups']}"
        )
    n_rows_used = int(shared.event_track_ids.shape[0])
    for arm in MODEL_ARMS:
        production = arms[arm]["covariance_diagnostics"].get(str(PRODUCTION_DAMPING))
        deficient = int((production or {}).get("n_rank_deficient_rows", 0))
        fraction = deficient / max(n_rows_used, 1)
        if deficient and fraction > MAX_RANK_DEFICIENT_ROW_FRACTION:
            status = "inconclusive"
            reasons.append(
                f"{arm} arm: {deficient} of {n_rows_used} rows ({fraction:.2%}) are "
                "rank deficient on their active subspace and were dropped from the "
                f"comparison, above the pre-registered "
                f"{MAX_RANK_DEFICIENT_ROW_FRACTION:.2%}"
            )
    stationarity = {}
    for arm in MODEL_ARMS:
        decrement = arms[arm]["newton_decrement"]
        stationarity[arm] = stationarity_report(decrement)
        fraction = stationarity[arm]["non_stationary_row_fraction"]
        if fraction > MAX_NON_STATIONARY_ROW_FRACTION:
            status = "inconclusive"
            reasons.append(
                f"{arm} arm: {fraction:.4f} of rows are non-stationary "
                f"(Newton decrement > {NEWTON_DECREMENT_STATIONARY}), above the "
                f"pre-registered {MAX_NON_STATIONARY_ROW_FRACTION}; this is a "
                "numerical failure, not a physics result"
            )
    for arm in MODEL_ARMS:
        if not arms[arm]["convergence"]["converged"]:
            status = "failed" if status == "passed" else status
            reasons.append(
                f"{arm} arm did not converge (numerical failure, not a physics "
                f"result): {arms[arm]['convergence']}"
            )
    energy_loss = {"status": "open", "reason": "no energyLoss=True artifact supplied"}
    if energy_loss_material is not None:
        energy_loss = energy_loss_reverification(
            shared,
            geometry,
            reference,
            arms["broken"],
            broken_inputs,
            energy_loss_material,
            device,
            dtype,
            max_iters,
        )
    return {
        "status": status,
        "reasons": reasons,
        "model": model,
        "material_artifact": material.source,
        "n_rows_total": total_rows,
        "n_rows_used": n_rows_used,
        "n_rows_missing_material": len(missing),
        "missing_material_fraction": missing_fraction,
        "on_missing_material": on_missing_material,
        "holdout_keys_sha256": keys_digest,
        "holdout_keys_csv": str(keys_path),
        "nodes_in_window": {
            "min": int(broken_inputs.n_nodes_in_window.min()),
            "median": int(broken_inputs.n_nodes_in_window.median()),
            "max": int(broken_inputs.n_nodes_in_window.max()),
        },
        "convergence": {arm: arms[arm]["convergence"] for arm in MODEL_ARMS},
        "stationarity": stationarity,
        "covariance_diagnostics": {
            arm: arms[arm]["covariance_diagnostics"] for arm in MODEL_ARMS
        },
        "groups": groups,
        "damping_sensitivity": sensitivity,
        "energy_loss_reverification": energy_loss,
        "pre_registered": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "min_group_components": MIN_GROUP_COMPONENTS,
            "interpolation_rmse_improvement": INTERPOLATION_RMSE_IMPROVEMENT,
            "extrapolation_rmse_regression": EXTRAPOLATION_RMSE_REGRESSION,
            "predictive_chi2_band": list(PREDICTIVE_CHI2_BAND),
            "max_abs_pull_mean": MAX_ABS_PULL_MEAN,
            "production_damping": PRODUCTION_DAMPING,
            "newton_decrement_stationary": NEWTON_DECREMENT_STATIONARY,
            "max_non_stationary_row_fraction": MAX_NON_STATIONARY_ROW_FRACTION,
            "damping_scan": list(DAMPING_SCAN),
            "damping_stability_scan": list(DAMPING_STABILITY_SCAN),
            "preregistration_version": PREREGISTRATION_VERSION,
            "required_groups": [
                f"{volume}:{layer}|{mode}|{REQUIRED_COMPONENT}"
                for volume, layer in holdout_layers
                for mode in REQUIRED_MODES
            ],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
