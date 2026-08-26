from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import json
import math
import os
import numpy as np
import uproot
import pandas as pd
import awkward as ak
import re
from dataclasses import dataclass, field
from pathlib import Path

from alignment_artifact_layout import (
    ACTIVE_DOF as CONFIG_ACTIVE_DOF,
    TARGET_LAYERS,
    alignment_json_path,
    data_path,
    reconstruction_root_path,
    target_layers_spec,
)
from project_config import absolute_project_path, add_config_argument
from event_split import DEFAULT_SPLIT_SEED, PARTITIONS, event_partition_mask
try:
    from .helix_model import (
        CURVATURE_SCALE,
        S_MAX,
        accept_newton_candidate,
        assemble_track_parameters as assemble_joint_track_parameters,
        closest_helix_points,
        helix_point,
        helix_tangent,
        intersect_plane,
        split_track_parameters as split_joint_track_parameters,
    )
except ImportError:  # Direct script execution.
    from helix_model import (
        CURVATURE_SCALE,
        S_MAX,
        accept_newton_candidate,
        assemble_track_parameters as assemble_joint_track_parameters,
        closest_helix_points,
        helix_point,
        helix_tangent,
        intersect_plane,
        split_track_parameters as split_joint_track_parameters,
    )

EPS_COVARIANCE = 1e-8
EPS_WEIGHT = 1e-12
EPS_DIRECTION_NORM = 1e-9
FULL_MATRIX_TOLERANCE = 1e-4
STRIP_VOLUMES = {23, 24, 25}
STRIP_Y_FLOOR = 1.44  # (1.2 mm)^2
DEFAULT_TARGET_LAYERS = TARGET_LAYERS
# Shared by both track modes.  tz is excluded (never injected and poorly
# constrained by a planar residual); rz must stay in -- it is the only
# rotation the current misalignment actually produces.
DEFAULT_ACTIVE_DOF = CONFIG_ACTIVE_DOF
DEFAULT_EPOCHS = 200
DEFAULT_LAMBDA_L2 = 1e-5
DEFAULT_ADAM_LR = 1e-2
DEFAULT_ADAM_ETA_MIN = 1e-5
ADAM_CLAMP_T = 1.5
ADAM_CLAMP_R = 0.05
JOINT_ITERS = 1000
JOINT_LR_TRANS = 1e-2
JOINT_LR_TRACK = 3e-2
JOINT_CLAMP_T = 1.5
JOINT_CLAMP_R = 0.05
JOINT_CHECKPOINT_INTERVAL = 10
JOINT_PATIENCE = 200
JOINT_SEED_MODE = "non-target-measurement"
JOINT_SEED_MIN_HITS = 8
JOINT_SEED_MIN_SPAN = 300.0
JOINT_SMT_SEED_ITERS = 200
JOINT_MEASUREMENT_REFIT_ITERS = 200
JOINT_SOLVER = "block-lm"

#: Solver for the joint initialization's measurement refit.  Iteration 18 measured
#: the Adam path on real data: Newton decrement p90 = 4.17e5 against a threshold of
#: 1e-6, with 100% of rows non-stationary after the full 200-iteration budget.  The
#: same objective solved by Gauss-Newton comes back stationary, and the two
#: objectives were shown to agree bit-for-bit, so the optimiser is the only thing
#: that changes here.  Adam stays reachable so runs made before Iteration 19 remain
#: reproducible and so the two can be compared under one command.
JOINT_REFIT_SOLVER_GAUSS_NEWTON = "gauss-newton"
JOINT_REFIT_SOLVER_ADAM = "adam"
JOINT_REFIT_SOLVERS = (JOINT_REFIT_SOLVER_GAUSS_NEWTON, JOINT_REFIT_SOLVER_ADAM)
JOINT_REFIT_SOLVER = JOINT_REFIT_SOLVER_GAUSS_NEWTON

#: Pre-registration for the joint refit, kept separate from the Gate 4 block's
#: ``PREREGISTRATION_VERSION``.  That constant governs the verdict rules in
#: ``validate_helix_model``; none of them change here, and Gate 4 does not consume
#: joint output, so bumping it would make every Gate 4 summary claim a rules change
#: that never happened.  Bump this one whenever a constant below changes meaning.
#: v1: Iteration 19, the Adam -> Gauss-Newton swap.
JOINT_REFIT_PREREGISTRATION_VERSION = 1

#: Acceptance criterion for the Gauss-Newton refit, fixed before the swap was run.
#: The per-row threshold itself is not redefined here: it is the pre-registered
#: ``NEWTON_DECREMENT_STATIONARY = 1e-6`` in the ``F = 0.5 * chi2`` gauge.  This is
#: only how many rows may miss it, and the value is carried over by analogy from
#: ``MAX_RANK_DEFICIENT_ROW_FRACTION`` -- a handful of pathological rows does not
#: invalidate a fit, above that the fit is not usable.  It was set from that
#: precedent, not from looking at what the solver produced.
JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION = 1.0e-3
LM_MAX_ITERS = 20
LM_INITIAL_DAMPING = 1e-3
FISHER_EIGEN_FLOOR = 1e-4
LM_MAX_STEP_T = 0.1
LM_MAX_STEP_R = 1e-3
MEASUREMENT_REFIT_LR_POSITION = 1e-3
MEASUREMENT_REFIT_LR_DIRECTION = 1e-5
MEASUREMENT_REFIT_LR_RHO_SCALED = 1e-3
# EXPERIMENTAL: hit-quality weighting is temporarily disabled; no caller sets
# use_hit_weights=True.  Kept so the experiment can be restarted.
HIT_WEIGHT_FLOOR = 0.3
HIT_QUALITY_INPUT_SIZE = 9
HIT_QUALITY_HIDDEN_ONE = 32
HIT_QUALITY_HIDDEN_TWO = 16
HIT_QUALITY_OUTPUT_SIZE = 1

@dataclass
class AlignmentInputs:
    hit_sensor_ids: torch.Tensor
    initial_residuals: torch.Tensor
    covariances: torch.Tensor
    # EXPERIMENTAL: consumed only by the disabled hit-quality network.
    track_params: torch.Tensor
    track_chi2_ndf: torch.Tensor
    hit_global_positions: torch.Tensor
    hit_directions: torch.Tensor
    hit_qoverp: torch.Tensor
    hit_volume_ids: torch.Tensor
    hit_layer_ids: torch.Tensor
    hit_module_ids: torch.Tensor
    hit_event_ids: torch.Tensor
    hit_track_ids: torch.Tensor
    measured_local: torch.Tensor | None = None
    loc1_valid: torch.Tensor | None = None
    # Measurement variance BEFORE the STRIP_Y_FLOOR clip is applied; kept as an
    # observable diagnostic and pinned by the ROOT loader contract test.
    raw_covariances: torch.Tensor | None = None
    merged_df: pd.DataFrame | None = field(default=None, repr=False)


def filter_alignment_inputs_by_event_partition(
    inputs: AlignmentInputs,
    partition: str,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> AlignmentInputs:
    """Return a hit-level view restricted to one deterministic event partition."""
    mask_numpy = event_partition_mask(
        inputs.hit_event_ids.detach().cpu().numpy(), partition, split_seed
    )
    mask = torch.as_tensor(mask_numpy, device=inputs.hit_event_ids.device, dtype=torch.bool)
    if not mask.any():
        raise ValueError(f"Event partition '{partition}' contains no alignment hits")

    def selected(value):
        return value[mask.to(value.device)] if value is not None else None

    merged = None
    if inputs.merged_df is not None:
        merged = inputs.merged_df.loc[mask_numpy].reset_index(drop=True)
    return AlignmentInputs(
        hit_sensor_ids=selected(inputs.hit_sensor_ids),
        initial_residuals=selected(inputs.initial_residuals),
        covariances=selected(inputs.covariances),
        track_params=selected(inputs.track_params),
        track_chi2_ndf=selected(inputs.track_chi2_ndf),
        hit_global_positions=selected(inputs.hit_global_positions),
        hit_directions=selected(inputs.hit_directions),
        hit_qoverp=selected(inputs.hit_qoverp),
        hit_volume_ids=selected(inputs.hit_volume_ids),
        hit_layer_ids=selected(inputs.hit_layer_ids),
        hit_module_ids=selected(inputs.hit_module_ids),
        hit_event_ids=selected(inputs.hit_event_ids),
        hit_track_ids=selected(inputs.hit_track_ids),
        measured_local=selected(inputs.measured_local),
        loc1_valid=selected(inputs.loc1_valid),
        raw_covariances=selected(inputs.raw_covariances),
        merged_df=merged,
    )


@dataclass
class TrackBatches:
    measured_local: torch.Tensor
    covariances: torch.Tensor
    sensor_idx: torch.Tensor
    hit_mask: torch.Tensor
    loc1_mask: torch.Tensor
    g_smt: torch.Tensor
    p_smt: torch.Tensor
    volume_id: torch.Tensor
    layer_id: torch.Tensor
    is_target: torch.Tensor
    event_track_ids: torch.Tensor
    fit_mask: torch.Tensor | None = None
    fit_span: torch.Tensor | None = None
    selection_stats: dict[str, int] = field(default_factory=dict)


@dataclass
class JointGeometry:
    transforms: list[dict]
    input_transforms: list[dict]
    centers: torch.Tensor
    rotations: torch.Tensor
    target_mask: torch.Tensor
    id_to_idx: dict[tuple[int, int, int], int]
    input_indices: list[int]


@dataclass
class AlignmentRunConfig:
    input_geom: str = str(alignment_json_path("alignment_output", "misalignment_applied"))
    states_root: str = str(
        reconstruction_root_path(
            "reconstruction_output",
            "misaligned",
            "trackstates_kf",
        )
    )
    summary_root: str = str(
        reconstruction_root_path(
            "reconstruction_output",
            "misaligned",
            "tracksummary_kf",
        )
    )
    output_geom: str = str(
        alignment_json_path(
            "alignment_output",
            "aligned_transforms",
            stem_prefix="ml",
        )
    )
    nominal_file: str = str(data_path("odd_transforms.json"))
    iter_label: str | None = None
    make_plot: bool = True
    active_dof: tuple[int, ...] | str | None = None
    target_layers: tuple[tuple[int, int], ...] | str = DEFAULT_TARGET_LAYERS
    fix_reference: bool = True
    epochs: int = DEFAULT_EPOCHS
    lambda_l2: float = DEFAULT_LAMBDA_L2
    track_mode: str = "surface"
    joint_iters: int = JOINT_ITERS
    min_hits: int = 8
    max_tracks: int = 0
    track_batch_size: int = 0
    rng_seed: int = 42
    gpu: bool = False
    rot_lr_scale: float = 15.0
    joint_lr_trans: float = JOINT_LR_TRANS
    joint_lr_track: float = JOINT_LR_TRACK
    joint_clamp_t: float = JOINT_CLAMP_T
    joint_clamp_r: float = JOINT_CLAMP_R
    nominal_full_geom: str = str(data_path("odd_transforms.json"))
    freeze_tracks: bool = False
    joint_checkpoint_interval: int = JOINT_CHECKPOINT_INTERVAL
    joint_patience: int = JOINT_PATIENCE
    joint_min_delta: float = 0.0
    joint_max_invalid_fraction: float = 0.05
    joint_seed_mode: str = JOINT_SEED_MODE
    joint_seed_min_hits: int = JOINT_SEED_MIN_HITS
    joint_seed_min_span: float = JOINT_SEED_MIN_SPAN
    joint_smt_seed_iters: int = JOINT_SMT_SEED_ITERS
    joint_measurement_refit_iters: int = JOINT_MEASUREMENT_REFIT_ITERS
    alignment_gain: float = 1.0
    joint_solver: str = JOINT_SOLVER
    lm_max_iters: int = LM_MAX_ITERS
    lm_initial_damping: float = LM_INITIAL_DAMPING
    fisher_eigen_floor: float = FISHER_EIGEN_FLOOR
    lm_max_step_t: float = LM_MAX_STEP_T
    lm_max_step_r: float = LM_MAX_STEP_R
    measurement_refit_lr_position: float = MEASUREMENT_REFIT_LR_POSITION
    measurement_refit_lr_direction: float = MEASUREMENT_REFIT_LR_DIRECTION
    measurement_refit_lr_rho_scaled: float = MEASUREMENT_REFIT_LR_RHO_SCALED
    joint_refit_solver: str = JOINT_REFIT_SOLVER
    joint_refit_stationarity: bool = True
    event_partition: str | None = None
    event_split_seed: int = DEFAULT_SPLIT_SEED


class JointBestCheckpoint:
    """Track and restore the lowest evaluated joint objective."""

    def __init__(self, parameters, patience=0, min_delta=0.0):
        self.parameters = tuple(parameters)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.best_iteration = -1
        self._state = None

    def observe(self, loss, iteration):
        loss = float(loss)
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_iteration = int(iteration)
            self._state = tuple(parameter.detach().clone() for parameter in self.parameters)
        return (
            self.patience > 0
            and self._state is not None
            and int(iteration) - self.best_iteration >= self.patience
        )

    def restore(self):
        if self._state is None:
            raise RuntimeError("No joint checkpoint has been observed.")
        with torch.no_grad():
            for parameter, value in zip(self.parameters, self._state):
                parameter.copy_(value)


def clamp_parameter_and_reset_adam(parameter, lower, upper, optimizer):
    """Clamp a parameter and clear Adam moments for entries that hit a bound."""
    with torch.no_grad():
        saturated = (parameter < lower) | (parameter > upper)
        parameter.clamp_(min=lower, max=upper)
        state = optimizer.state.get(parameter, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None:
                moment.masked_fill_(saturated, 0.0)
    return saturated


def joint_checkpoint_score(total_loss, invalid_fraction, max_invalid_fraction):
    """Reject checkpoints that reduce loss by dropping too many intersections."""
    if float(invalid_fraction) > float(max_invalid_fraction):
        return float("inf")
    return float(total_loss)


def summarize_joint_layer_chi2(
    residuals,
    covariance,
    component_mask,
    volume_id,
    layer_id,
    target_layers,
):
    """Return per-layer, per-component chi2 for padded joint tensors."""
    residuals = residuals.detach().cpu()
    covariance = covariance.detach().cpu()
    component_mask = component_mask.detach().cpu()
    volume_id = volume_id.detach().cpu()
    layer_id = layer_id.detach().cpu()
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    output = {}
    for volume, layer in parse_target_layers(target_layers):
        layer_mask = (volume_id == volume) & (layer_id == layer)
        components = {}
        for index, name in enumerate(("LOC0", "LOC1")):
            mask = layer_mask & component_mask[..., index]
            count = int(mask.sum())
            components[name] = {
                "chi2_per_component": (
                    float(weighted[..., index][mask].mean()) if count else None
                ),
                "n_components": count,
            }
        output[f"{volume}:{layer}"] = components
    return output


def build_track_batches(
    inputs,
    target_layers,
    min_hits=JOINT_SEED_MIN_HITS,
    min_span=JOINT_SEED_MIN_SPAN,
    max_tracks=0,
):
    """Group flat hit tensors into deterministic padded per-track tensors."""
    if inputs.measured_local is None or inputs.loc1_valid is None:
        raise ValueError("Joint mode requires measured_local and loc1_valid.")
    target_layers = set(parse_target_layers(target_layers))
    n_hits = len(inputs.hit_sensor_ids)
    positions = inputs.hit_global_positions.detach().cpu().numpy()
    event_ids = inputs.hit_event_ids.detach().cpu().numpy()
    track_ids = inputs.hit_track_ids.detach().cpu().numpy()
    volume_ids = inputs.hit_volume_ids.detach().cpu().numpy()
    layer_ids = inputs.hit_layer_ids.detach().cpu().numpy()
    module_ids = inputs.hit_module_ids.detach().cpu().numpy()
    sensor_ids = inputs.hit_sensor_ids.detach().cpu().numpy()
    radius = np.hypot(positions[:, 0], positions[:, 1])
    order = np.lexsort((
        sensor_ids, module_ids, layer_ids, volume_ids,
        np.abs(positions[:, 2]), radius, track_ids, event_ids,
    ))
    ordered_events = event_ids[order]
    ordered_tracks = track_ids[order]
    boundaries = np.flatnonzero(
        (ordered_events[1:] != ordered_events[:-1])
        | (ordered_tracks[1:] != ordered_tracks[:-1])
    ) + 1
    groups = []
    for indices in np.split(order, boundaries):
        key = (int(event_ids[indices[0]]), int(track_ids[indices[0]]))
        groups.append((key, indices.tolist()))

    selected = []
    for key, indices in groups:
        target_flags = [
            (int(inputs.hit_volume_ids[i]), int(inputs.hit_layer_ids[i]))
            in target_layers
            for i in indices
        ]
        fit_indices = [i for i, is_target in zip(indices, target_flags) if not is_target]
        fit_positions = positions[fit_indices]
        fit_span = (
            float(np.linalg.norm(np.diff(fit_positions, axis=0), axis=1).sum())
            if len(fit_positions) > 1
            else 0.0
        )
        if any(target_flags) and len(fit_indices) >= min_hits and fit_span >= min_span:
            selected.append((key, indices, fit_span))
    if max_tracks and max_tracks > 0:
        selected = selected[:max_tracks]
    if not selected:
        raise ValueError("No tracks satisfy the joint-alignment selection.")

    n_tracks = len(selected)
    max_hits = max(len(indices) for _, indices, _fit_span in selected)
    measured = torch.zeros((n_tracks, max_hits, 2), dtype=inputs.measured_local.dtype)
    covariance = torch.ones((n_tracks, max_hits, 2), dtype=inputs.covariances.dtype)
    sensor_idx = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    hit_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    loc1_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    g_smt = torch.zeros((n_tracks, max_hits, 3), dtype=inputs.hit_global_positions.dtype)
    p_smt = torch.zeros((n_tracks, max_hits, 3), dtype=inputs.hit_directions.dtype)
    volume_id = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    layer_id = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    is_target = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    fit_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    fit_span = torch.zeros(n_tracks, dtype=inputs.hit_global_positions.dtype)
    event_track_ids = torch.zeros((n_tracks, 2), dtype=torch.long)
    for track_index, (key, indices, track_fit_span) in enumerate(selected):
        count = len(indices)
        index = torch.tensor(indices, dtype=torch.long)
        measured[track_index, :count] = inputs.measured_local[index]
        covariance[track_index, :count] = inputs.covariances[index]
        sensor_idx[track_index, :count] = inputs.hit_sensor_ids[index]
        hit_mask[track_index, :count] = True
        loc1_mask[track_index, :count] = inputs.loc1_valid[index]
        g_smt[track_index, :count] = inputs.hit_global_positions[index]
        p_smt[track_index, :count] = inputs.hit_directions[index]
        volume_id[track_index, :count] = inputs.hit_volume_ids[index]
        layer_id[track_index, :count] = inputs.hit_layer_ids[index]
        event_track_ids[track_index] = torch.tensor(key)
        fit_span[track_index] = track_fit_span
        for hit_index, flat_index in enumerate(indices):
            is_target[track_index, hit_index] = (
                int(inputs.hit_volume_ids[flat_index]), int(inputs.hit_layer_ids[flat_index])
            ) in target_layers
        fit_mask[track_index, :count] = ~is_target[track_index, :count]
    return TrackBatches(
        measured_local=measured,
        covariances=covariance,
        sensor_idx=sensor_idx,
        hit_mask=hit_mask,
        loc1_mask=loc1_mask,
        g_smt=g_smt,
        p_smt=p_smt,
        volume_id=volume_id,
        layer_id=layer_id,
        is_target=is_target,
        event_track_ids=event_track_ids,
        fit_mask=fit_mask,
        fit_span=fit_span,
        selection_stats={
            "input_tracks": len(groups),
            "retained_tracks": len(selected),
            "excluded_tracks": len(groups) - len(selected),
        },
    )


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


def _non_target_fit_mask(batches):
    if batches.fit_mask is None:
        raise ValueError(
            "Joint initialization requires a non-target fit_mask; "
            "falling back to all hits is forbidden."
        )
    if (batches.fit_mask & batches.is_target).any():
        raise ValueError("Joint initialization fit_mask contains target-layer hits.")
    return batches.fit_mask


def seed_helices(batches, n_iter=200, device="cpu", dtype=torch.float64):
    """Fit per-track helix warm starts to ACTS smoothed global points."""
    points = batches.g_smt.to(device=device, dtype=dtype)
    mask = _non_target_fit_mask(batches).to(device=device)
    n_tracks, n_hits, _ = points.shape
    first_index = mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(n_tracks, device=device)
    first = points[rows, first_index]
    direction = batches.p_smt.to(device=device, dtype=dtype)[rows, first_index]
    transverse = torch.linalg.norm(direction[:, :2], dim=1).clamp(min=1e-6)
    psi = torch.atan2(direction[:, 1], direction[:, 0])
    tanl = direction[:, 2] / transverse
    linear = torch.cat((first, psi[:, None], tanl[:, None]), dim=1).detach().clone().requires_grad_(True)
    rho_scaled = torch.zeros((n_tracks, 1), device=device, dtype=dtype, requires_grad=True)
    optimizer = optim.Adam([
        {"params": [linear], "lr": 0.5},
        {"params": [rho_scaled], "lr": 0.5},
    ])
    for _ in range(n_iter):
        optimizer.zero_grad()
        track_parameters = assemble_joint_track_parameters(linear, rho_scaled)
        expanded = track_parameters[:, None, :].expand(-1, n_hits, -1)
        with torch.no_grad():
            _fit, path_length, valid = closest_helix_points(expanded.detach(), points)
        predicted = helix_point(expanded, path_length)
        effective = mask & valid
        squared_distance = torch.sum((predicted - points) ** 2, dim=-1)
        loss = torch.sum(torch.where(effective, squared_distance, torch.zeros_like(squared_distance)))
        loss = loss / effective.sum().clamp(min=1)
        loss.backward()
        optimizer.step()
    return assemble_joint_track_parameters(linear, rho_scaled).detach()


def smt_seed_statistics(batches, track_parameters):
    """Summarize the non-target SMT point fit used only for initialization."""
    points = batches.g_smt.to(track_parameters.device, track_parameters.dtype)
    fit_mask = _non_target_fit_mask(batches).to(track_parameters.device)
    expanded = track_parameters[:, None, :].expand(-1, points.shape[1], -1)
    fitted, _path, valid = closest_helix_points(expanded, points)
    effective = fit_mask & valid
    if not effective.any():
        raise ValueError("No valid non-target SMT points remain after helix seeding.")
    squared_distance = torch.sum((fitted - points).square(), dim=-1)
    loss = torch.sum(torch.where(effective, squared_distance, torch.zeros_like(squared_distance)))
    loss = loss / effective.sum()
    invalid_fraction = 1.0 - float(valid[fit_mask].float().mean().cpu())
    return {
        "loss": float(loss.cpu()),
        "n_points": int(effective.sum().cpu()),
        "invalid_fraction": invalid_fraction,
    }


def _measurement_refit_objective(
    track_parameters,
    batches,
    geometry,
):
    """Evaluate non-target measurement chi2 on fixed working geometry."""
    measured = batches.measured_local.to(track_parameters.device, track_parameters.dtype)
    covariance = batches.covariances.to(track_parameters.device, track_parameters.dtype)
    sensor_idx = batches.sensor_idx.to(track_parameters.device)
    fit_mask = _non_target_fit_mask(batches).to(track_parameters.device)
    loc1_mask = batches.loc1_mask.to(track_parameters.device)
    n_hits = measured.shape[1]
    tracks = track_parameters[:, None, :].expand(-1, n_hits, -1)
    centers = geometry.centers.to(track_parameters.device, track_parameters.dtype)[sensor_idx]
    rotations = geometry.rotations.to(track_parameters.device, track_parameters.dtype)[sensor_idx]
    _path, intersections, intersection_valid = intersect_plane(
        tracks, centers, rotations[..., :, 2]
    )
    local = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)[..., :2]
    residuals = measured - local
    valid_hits = fit_mask & intersection_valid
    component_mask = torch.stack((valid_hits, valid_hits & loc1_mask), dim=-1)
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    loss = torch.sum(torch.where(component_mask, weighted, torch.zeros_like(weighted)))
    loss = loss / component_mask.sum().clamp(min=1)
    invalid_fraction = 1.0 - float(
        intersection_valid[fit_mask].float().mean().detach().cpu()
    )
    return loss, invalid_fraction, component_mask


#: Prediction mode for batches built only to be measured, never fitted.  Gate 4
#: selects rows by ``prediction_mode`` equality against ``PREDICTION_MODES``, so a
#: value outside that tuple selects nothing rather than letting a placeholder row
#: reach a physics result.
DIAGNOSTIC_PREDICTION_MODE = "joint-refit-diagnostic"


def _load_validation_module():
    """Import ``validate_helix_model`` lazily.

    It is a 4000-line entry point; importing it at module scope would make every
    autograd run pay for the Gate 4 machinery.  Nothing runs at its import time
    beyond constants and one ``vmap`` construction, and it does not import this
    module, so there is no cycle.
    """
    try:
        from . import validate_helix_model
    except ImportError:  # Direct script execution.
        import validate_helix_model
    return validate_helix_model


def _validation_batches_from_joint(batches, validation):
    """Re-label joint batches as the validation module's ``TrackBatches``.

    Both modules carry the same measurements under different field names
    (``covariances``/``loc1_mask`` here, ``measurement_variance``/``loc1_valid``
    there).  Building the real dataclass instead of duck-typing means a field
    added on either side fails loudly here rather than silently reading a
    placeholder.

    ``holdout_mask`` has no meaning during joint initialization -- nothing is held
    out.  ``newton_decrement`` reads none of the ``holdout_*`` inputs, so the first
    fitted hit stands in, and ``prediction_mode`` carries the diagnostic sentinel
    so this object cannot contribute rows to a Gate 4 verdict.
    """
    fit_mask = _non_target_fit_mask(batches)
    n_rows = fit_mask.shape[0]
    if not bool(fit_mask.any(dim=1).all()):
        raise ValueError("Every joint track needs at least one non-target hit.")
    holdout_mask = torch.zeros_like(fit_mask)
    holdout_mask[
        torch.arange(n_rows, device=fit_mask.device),
        fit_mask.to(torch.int64).argmax(dim=1),
    ] = True
    real_dtype = batches.measured_local.dtype
    zeros_long = torch.zeros(n_rows, dtype=torch.long)
    zeros_real = torch.zeros(n_rows, dtype=real_dtype)
    return validation.TrackBatches(
        measured_local=batches.measured_local,
        measurement_variance=batches.covariances,
        loc1_valid=batches.loc1_mask,
        smt_local=torch.zeros_like(batches.measured_local),
        sensor_idx=batches.sensor_idx,
        hit_mask=batches.hit_mask,
        fit_mask=fit_mask,
        holdout_mask=holdout_mask,
        g_smt=batches.g_smt,
        p_smt=batches.p_smt,
        volume_id=batches.volume_id,
        layer_id=batches.layer_id,
        module_id=torch.zeros_like(batches.volume_id),
        event_track_ids=batches.event_track_ids,
        fit_span=batches.fit_span if batches.fit_span is not None else zeros_real,
        prediction_mode=(DIAGNOSTIC_PREDICTION_MODE,) * n_rows,
        context_hits_before=zeros_long,
        context_hits_after=zeros_long,
        nearest_context_distance=zeros_real,
        farthest_context_distance=zeros_real,
    )


def measure_joint_refit_stationarity(
    batches,
    geometry,
    parameters,
    device="cpu",
    dtype=torch.float64,
):
    """Per-row Newton decrement of the joint measurement refit.

    Turns "the optimiser may have stopped on budget" into a number.  Measurement
    only: it reads the parameters and never changes them.

    Deliberately routed through the validation module's ``newton_decrement`` so
    this and the Gate 4 single arm cannot drift apart, and so the value lands in
    the pre-registered ``F = 0.5 * chi2`` gauge that ``NEWTON_DECREMENT_STATIONARY``
    was set in.  ``_measurement_refit_objective`` divides by the component count,
    which would scale every decrement by that factor -- order 1e-5 on a real run --
    and make the threshold meaningless.

    Runs in float64 on the CPU by default: a decrement compared against 1e-6 is
    below float32 resolution.
    """
    validation = _load_validation_module()
    measured = _validation_batches_from_joint(batches, validation)
    decrement = validation.measurement_newton_decrement(
        measured, geometry, parameters, device=device, dtype=dtype
    )
    return validation.stationarity_report(decrement)


def _refit_helices_adam(
    batches,
    geometry,
    initial,
    n_iter,
    max_invalid_fraction,
    lr_position,
    lr_direction,
    lr_rho_scaled,
    initial_loss,
):
    """The first-order refit, kept reachable so pre-Iteration-19 runs reproduce.

    Iteration 18 measured what it actually delivers: after the full 200-iteration
    budget the Newton decrement p90 is 4.17e5 against a 1e-6 threshold, so this is
    a truncated descent, not a minimum.  Nothing in it changed with the swap.
    """
    linear_initial, rho_initial = split_joint_track_parameters(initial)
    position = nn.Parameter(linear_initial[:, :3].clone())
    direction = nn.Parameter(linear_initial[:, 3:].clone())
    rho_scaled = nn.Parameter(rho_initial.clone())
    optimizer = optim.Adam(
        (
            {"params": [position], "lr": lr_position},
            {"params": [direction], "lr": lr_direction},
            {"params": [rho_scaled], "lr": lr_rho_scaled},
        )
    )
    checkpoint = JointBestCheckpoint((position, direction, rho_scaled))
    checkpoint.observe(initial_loss, iteration=-1)
    first_step = None
    for iteration in range(n_iter):
        optimizer.zero_grad()
        linear = torch.cat((position, direction), dim=1)
        parameters = assemble_joint_track_parameters(linear, rho_scaled)
        loss, invalid_fraction, component_mask = _measurement_refit_objective(
            parameters, batches, geometry
        )
        if not component_mask.any():
            raise ValueError("No valid non-target measurement components remain for helix refit.")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite non-target measurement helix loss.")
        loss.backward()
        before = (
            position.detach().clone(),
            direction.detach().clone(),
            rho_scaled.detach().clone(),
        )
        gradient_norms = {
            "position": float(position.grad.norm().detach().cpu()),
            "direction": float(direction.grad.norm().detach().cpu()),
            "rho_scaled": float(rho_scaled.grad.norm().detach().cpu()),
        }
        optimizer.step()
        with torch.no_grad():
            evaluated = assemble_joint_track_parameters(
                torch.cat((position, direction), dim=1), rho_scaled
            )
            evaluated_loss, evaluated_invalid, _evaluated_mask = _measurement_refit_objective(
                evaluated, batches, geometry
            )
        evaluated_value = joint_checkpoint_score(
            float(evaluated_loss.cpu()), evaluated_invalid, max_invalid_fraction
        )
        if iteration == 0:
            updates = (
                position.detach() - before[0],
                direction.detach() - before[1],
                rho_scaled.detach() - before[2],
            )
            names = ("position", "direction", "rho_scaled")
            first_step = {
                "loss_before": float(loss.detach().cpu()),
                "loss_after": float(evaluated_loss.cpu()),
                "invalid_fraction_after": evaluated_invalid,
                "gradient_norms": gradient_norms,
                "update_rms": {
                    name: float(torch.sqrt(torch.mean(update.square())).cpu())
                    for name, update in zip(names, updates, strict=True)
                },
                "update_max": {
                    name: float(update.abs().max().cpu())
                    for name, update in zip(names, updates, strict=True)
                },
            }
        checkpoint.observe(evaluated_value, iteration=iteration)
    checkpoint.restore()
    fitted = assemble_joint_track_parameters(
        torch.cat((position, direction), dim=1), rho_scaled
    ).detach()
    return fitted, {
        "best_iteration": checkpoint.best_iteration,
        "first_step": first_step,
    }


def _refit_helices_gauss_newton(batches, geometry, initial, device):
    """Solve the same objective, and refuse to return a solution that is not one.

    Routed through the validation module's solver rather than a second
    implementation, for the same reason the measurement was: the joint path and
    the Gate 4 single arm cannot drift apart if there is only one solver.  The
    adapter and the input preparation are the ones Iteration 18 already validated,
    so the objective here is the one that was shown to agree bit-for-bit.

    Always float64.  The pre-registered stationarity threshold is 1e-6, which is
    below float32 resolution, and the joint pipeline runs float32 whenever it runs
    on the GPU.  The caller casts the result back to its own dtype.
    """
    validation = _load_validation_module()
    measured = _validation_batches_from_joint(batches, validation)
    fitted, solver_stats = validation.fit_helices_gauss_newton(
        measured, geometry, initial, device=device, dtype=torch.float64
    )
    report = solver_stats["stationarity"]
    if report["n_non_finite"]:
        raise FloatingPointError(
            "Gauss-Newton joint refit produced non-finite Newton decrements on "
            f"{report['n_non_finite']} rows."
        )
    fraction = float(report["non_stationary_row_fraction"])
    if fraction > JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION:
        raise ValueError(
            "Gauss-Newton joint refit did not reach the pre-registered stationary "
            f"point: {fraction:.3%} of rows exceed a Newton decrement of "
            f"{report['stationary_threshold']:.3g} (limit "
            f"{JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION:.3%}, p90 "
            f"{report['newton_decrement_p90']:.6g})."
        )
    return fitted, {
        "convergence": solver_stats["convergence"],
        "stationarity_gate": report,
    }


def fit_helices_to_measurements(
    batches,
    geometry,
    initial_parameters,
    n_iter=JOINT_MEASUREMENT_REFIT_ITERS,
    device="cpu",
    dtype=torch.float64,
    max_invalid_fraction=0.05,
    lr_position=MEASUREMENT_REFIT_LR_POSITION,
    lr_direction=MEASUREMENT_REFIT_LR_DIRECTION,
    lr_rho_scaled=MEASUREMENT_REFIT_LR_RHO_SCALED,
    stationarity=True,
    solver=JOINT_REFIT_SOLVER,
):
    """Refit helices using only non-target local measurements and fixed surfaces.

    ``solver`` selects how the minimum is reached; the objective, the component
    mask and the seed are the same either way.  ``n_iter`` and the three ``lr_``
    arguments only reach the Adam path.

    ``stationarity`` adds a read-only Newton-decrement measurement of the seed and
    of the refit output.  It does not touch the returned parameters: the seed
    figure is what separates "the optimiser converged" from "the seed was already
    close".  It is a diagnostic, and switching it off does not switch off the
    pre-registered acceptance criterion the Gauss-Newton path is held to -- that
    one reads the decrement the solver computes for itself.
    """
    if solver not in JOINT_REFIT_SOLVERS:
        raise ValueError(
            f"Unknown joint measurement refit solver {solver!r}; "
            f"expected one of {JOINT_REFIT_SOLVERS}."
        )
    if n_iter < 1:
        raise ValueError("joint measurement refit iterations must be at least 1.")
    initial = initial_parameters.to(device=device, dtype=dtype)
    with torch.no_grad():
        initial_loss_tensor, initial_invalid, _initial_mask = _measurement_refit_objective(
            initial, batches, geometry
        )
    initial_loss = float(initial_loss_tensor.cpu())
    if initial_invalid > max_invalid_fraction:
        raise ValueError(
            "Initial non-target measurement refit exceeds invalid-intersection limit: "
            f"{initial_invalid:.2%} > {max_invalid_fraction:.2%}."
        )
    if solver == JOINT_REFIT_SOLVER_ADAM:
        fitted, solver_stats = _refit_helices_adam(
            batches,
            geometry,
            initial,
            n_iter,
            max_invalid_fraction,
            lr_position,
            lr_direction,
            lr_rho_scaled,
            initial_loss,
        )
    else:
        fitted, solver_stats = _refit_helices_gauss_newton(
            batches, geometry, initial, device
        )
        fitted = fitted.to(device=initial.device, dtype=dtype)
    with torch.no_grad():
        evaluated_loss, final_invalid, component_mask = _measurement_refit_objective(
            fitted, batches, geometry
        )
        final_loss = float(evaluated_loss.cpu())
    if final_invalid > max_invalid_fraction:
        raise ValueError(
            "Non-target measurement refit intersection failure rate exceeds limit: "
            f"{final_invalid:.2%} > {max_invalid_fraction:.2%}."
        )
    stats = {
        "solver": solver,
        "joint_refit_preregistration_version": JOINT_REFIT_PREREGISTRATION_VERSION,
        "chi2_per_component": final_loss,
        "initial_chi2_per_component": initial_loss,
        "n_components": int(component_mask.sum().cpu()),
        "invalid_fraction": final_invalid,
        "parameter_updates": {
            "position_rms": float(torch.sqrt(torch.mean((fitted[:, :3] - initial[:, :3]).square())).cpu()),
            "direction_rms": float(torch.sqrt(torch.mean((fitted[:, 3:5] - initial[:, 3:5]).square())).cpu()),
            "rho_rms": float(torch.sqrt(torch.mean((fitted[:, 5] - initial[:, 5]).square())).cpu()),
        },
    }
    stats.update(solver_stats)
    if solver == JOINT_REFIT_SOLVER_ADAM:
        stats["improved"] = bool(
            stats["best_iteration"] >= 0 and final_loss < initial_loss
        )
    if stationarity:
        stats["stationarity"] = {
            "seed": measure_joint_refit_stationarity(batches, geometry, initial),
            "refit": measure_joint_refit_stationarity(batches, geometry, fitted),
        }
    return fitted, stats


def _joint_forward(track_parameters, alignment, batches, geometry, track_indices=None):
    if track_indices is None:
        track_indices = torch.arange(len(track_parameters), device=track_parameters.device)
    measured = batches.measured_local.to(track_parameters.device, track_parameters.dtype)[track_indices]
    covariance = batches.covariances.to(track_parameters.device, track_parameters.dtype)[track_indices]
    sensor_idx = batches.sensor_idx.to(track_parameters.device)[track_indices]
    hit_mask = batches.hit_mask.to(track_parameters.device)[track_indices]
    loc1_mask = batches.loc1_mask.to(track_parameters.device)[track_indices]
    n_hits = measured.shape[1]
    tracks = track_parameters[track_indices, None, :].expand(-1, n_hits, -1)
    centers0 = geometry.centers.to(track_parameters.device, track_parameters.dtype)
    rotations0 = geometry.rotations.to(track_parameters.device, track_parameters.dtype)
    centers, rotations = apply_alignment_torch(centers0, rotations0, alignment)
    hit_centers = centers[sensor_idx]
    hit_rotations = rotations[sensor_idx]
    normals = hit_rotations[..., :, 2]
    _path, intersections, intersection_valid = intersect_plane(
        tracks, hit_centers, normals, n_newton=12
    )
    local3 = torch.matmul(
        hit_rotations.transpose(-1, -2), (intersections - hit_centers).unsqueeze(-1)
    ).squeeze(-1)
    residuals = measured - local3[..., :2]
    valid = hit_mask & intersection_valid
    component_mask = torch.stack((valid, valid & loc1_mask), dim=-1)
    residuals = torch.where(component_mask, residuals, torch.zeros_like(residuals))
    return residuals, covariance, component_mask, intersection_valid


def regularize_fisher_matrix(fisher, damping, eigen_floor_ratio):
    """Regularize a normalized Fisher matrix and report weak-mode diagnostics."""
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("Fisher matrix must be square")
    if damping < 0 or eigen_floor_ratio < 0:
        raise ValueError("LM damping and Fisher eigen floor must be non-negative")
    symmetric = 0.5 * (fisher + fisher.transpose(-1, -2))
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues[0].detach().cpu())
    maximum = float(eigenvalues[-1].detach().cpu())
    floor = max(maximum, 0.0) * float(eigen_floor_ratio)
    regularization = max(0.0, floor - minimum) + float(damping) * max(maximum, 1.0)
    regularized = symmetric + torch.eye(
        len(symmetric), device=symmetric.device, dtype=symmetric.dtype
    ) * regularization
    condition = float("inf") if minimum <= 0 else maximum / minimum
    return regularized, {
        "min_eigenvalue": minimum,
        "max_eigenvalue": maximum,
        "condition_number": condition,
        "eigen_floor": floor,
        "regularization": regularization,
    }


def _sensor_residual_data(
    track_parameters,
    alignment_row,
    sensor_index,
    batches,
    geometry,
):
    device, dtype = track_parameters.device, track_parameters.dtype
    hit_selection = (
        batches.hit_mask.to(device)
        & batches.is_target.to(device)
        & (batches.sensor_idx.to(device) == int(sensor_index))
    )
    track_indices, hit_indices = torch.nonzero(hit_selection, as_tuple=True)
    if len(track_indices) == 0:
        return None
    tracks = track_parameters[track_indices]
    center0 = geometry.centers.to(device=device, dtype=dtype)[sensor_index]
    rotation0 = geometry.rotations.to(device=device, dtype=dtype)[sensor_index]
    center, rotation = apply_alignment_torch(center0, rotation0, alignment_row)
    centers = center.expand(len(tracks), -1)
    rotations = rotation.expand(len(tracks), -1, -1)
    _path, intersections, intersection_valid = intersect_plane(
        tracks, centers, rotations[..., :, 2], n_newton=12
    )
    local = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)[..., :2]
    measured = batches.measured_local.to(device=device, dtype=dtype)[track_indices, hit_indices]
    covariance = batches.covariances.to(device=device, dtype=dtype)[track_indices, hit_indices]
    loc1 = batches.loc1_mask.to(device)[track_indices, hit_indices]
    residuals = measured - local
    component_mask = torch.stack((intersection_valid, intersection_valid & loc1), dim=-1)
    return residuals, covariance, component_mask, intersection_valid


def _target_alignment_evaluation(track_parameters, alignment, batches, geometry):
    residuals, covariance, component_mask, intersection_valid = _joint_forward(
        track_parameters, alignment, batches, geometry
    )
    target_components = component_mask & batches.is_target.to(component_mask.device).unsqueeze(-1)
    if not target_components.any():
        raise ValueError("No valid target measurement components remain for block LM")
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    loss = torch.sum(torch.where(target_components, weighted, torch.zeros_like(weighted)))
    loss = loss / target_components.sum()
    valid_hits = batches.hit_mask.to(intersection_valid.device)
    invalid_fraction = 1.0 - float(intersection_valid[valid_hits].float().mean().detach().cpu())
    return loss, invalid_fraction, residuals, component_mask


def run_block_lm_alignment(
    batches,
    geometry,
    config,
    active_dof=DEFAULT_ACTIVE_DOF,
    initial_track_params=None,
):
    """Fit target geometry with fixed non-target-refitted helices using block LM."""
    use_cuda = bool(config.gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    if initial_track_params is None:
        initial_track_params = seed_helices(batches, device=device, dtype=dtype)
    tracks = initial_track_params.to(device=device, dtype=dtype).detach()
    active = tuple(int(index) for index in active_dof)
    if not active:
        raise ValueError("Block LM requires at least one active alignment DOF")
    if any(index < 0 or index >= 6 for index in active):
        raise ValueError("Block LM active DOF indices must be in [0, 5]")
    if config.lm_max_iters < 1:
        raise ValueError("lm_max_iters must be at least 1")
    n_surfaces = len(geometry.centers)
    alignment = torch.zeros((n_surfaces, 6), device=device, dtype=dtype)
    active_index = torch.tensor(active, device=device, dtype=torch.long)
    physical_scales = torch.tensor(
        [1.0, 1.0, 1.0, 0.01, 0.01, 0.01], device=device, dtype=dtype
    )[active_index]
    initial_loss, initial_invalid, initial_residuals, initial_mask = (
        _target_alignment_evaluation(tracks, alignment, batches, geometry)
    )
    if initial_invalid > config.joint_max_invalid_fraction:
        raise ValueError(
            "Initial block-LM helices exceed the invalid-intersection limit: "
            f"{initial_invalid:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    best_loss = float(initial_loss.detach().cpu())
    best_iteration = -1
    history = []
    checkpoint_history = [(-1, best_loss)]
    invalid_history = [initial_invalid]
    fisher_diagnostics = []
    damping = float(config.lm_initial_damping)
    stopped_early = False
    target_sensors = torch.nonzero(geometry.target_mask, as_tuple=False).flatten().tolist()
    for iteration in range(config.lm_max_iters):
        accepted_any = False
        largest_step = 0.0
        for sensor_index in target_sensors:
            current = alignment[sensor_index].detach()
            current_data = _sensor_residual_data(
                tracks, current, sensor_index, batches, geometry
            )
            if current_data is None:
                continue
            residuals, covariance, component_mask, _valid = current_data
            fixed_mask = component_mask.detach()
            if not fixed_mask.any():
                continue

            def standardized(active_values):
                row = current.scatter(0, active_index, active_values)
                values, variances, _mask, _intersection_valid = _sensor_residual_data(
                    tracks, row, sensor_index, batches, geometry
                )
                return (values / variances.clamp_min(EPS_COVARIANCE).sqrt())[fixed_mask]

            active_values = current[active_index].detach().requires_grad_(True)
            standardized_residual = standardized(active_values)
            jacobian = torch.autograd.functional.jacobian(
                standardized, active_values, vectorize=True
            ).detach()
            residual_vector = standardized_residual.detach()
            normalized_jacobian = jacobian * physical_scales
            fisher = normalized_jacobian.transpose(0, 1) @ normalized_jacobian
            gradient = normalized_jacobian.transpose(0, 1) @ residual_vector
            regularized, diagnostics = regularize_fisher_matrix(
                fisher,
                damping=damping,
                eigen_floor_ratio=config.fisher_eigen_floor,
            )
            try:
                normalized_step = -torch.linalg.solve(regularized, gradient)
            except RuntimeError:
                normalized_step = -torch.linalg.lstsq(regularized, gradient[:, None]).solution[:, 0]
            physical_step = normalized_step * physical_scales
            per_dof_limit = torch.tensor(
                [config.lm_max_step_t if index < 3 else config.lm_max_step_r for index in active],
                device=device,
                dtype=dtype,
            )
            physical_step = torch.maximum(
                torch.minimum(physical_step, per_dof_limit), -per_dof_limit
            )
            candidate = current.clone()
            candidate[active_index] += physical_step
            candidate[:3].clamp_(-config.joint_clamp_t, config.joint_clamp_t)
            candidate[3:].clamp_(-config.joint_clamp_r, config.joint_clamp_r)
            candidate_data = _sensor_residual_data(
                tracks, candidate, sensor_index, batches, geometry
            )
            candidate_residuals, candidate_covariance, candidate_mask, candidate_valid = candidate_data
            current_weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
            candidate_weighted = candidate_residuals.square() / candidate_covariance.clamp_min(EPS_COVARIANCE)
            current_sensor_loss = current_weighted[component_mask].mean()
            candidate_sensor_loss = candidate_weighted[candidate_mask].mean()
            sensor_invalid = 1.0 - float(candidate_valid.float().mean().detach().cpu())
            accepted = bool(
                torch.isfinite(candidate_sensor_loss)
                and sensor_invalid <= config.joint_max_invalid_fraction
                and candidate_sensor_loss < current_sensor_loss
            )
            if accepted:
                alignment[sensor_index] = candidate
                accepted_any = True
                largest_step = max(largest_step, float(physical_step.abs().max().detach().cpu()))
            diagnostics.update({
                "iteration": iteration,
                "sensor_index": int(sensor_index),
                "accepted": accepted,
                "step_rms": float(torch.sqrt(torch.mean(physical_step.square())).detach().cpu()),
                "step_max": float(physical_step.abs().max().detach().cpu()),
                "gradient_norm": float(gradient.norm().detach().cpu()),
                "loss_before": float(current_sensor_loss.detach().cpu()),
                "loss_after": float(candidate_sensor_loss.detach().cpu()),
            })
            fisher_diagnostics.append(diagnostics)
        evaluated_loss, invalid_fraction, _residuals, _mask = _target_alignment_evaluation(
            tracks, alignment, batches, geometry
        )
        evaluated_value = float(evaluated_loss.detach().cpu())
        history.append(evaluated_value)
        checkpoint_history.append((iteration, evaluated_value))
        invalid_history.append(invalid_fraction)
        if evaluated_value < best_loss:
            best_loss = evaluated_value
            best_iteration = iteration
        damping = max(damping * 0.5, 1e-12) if accepted_any else damping * 10.0
        if not accepted_any or largest_step < 1e-10:
            stopped_early = True
            break
    final_loss, final_invalid, final_residuals, final_mask = _target_alignment_evaluation(
        tracks, alignment, batches, geometry
    )
    if final_invalid > config.joint_max_invalid_fraction:
        raise ValueError(
            "Block-LM result exceeds the invalid-intersection limit: "
            f"{final_invalid:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    return {
        "solver": "block-lm",
        "predicted_deltas": alignment.detach().cpu().double().numpy(),
        "track_params": tracks.detach().cpu().double().numpy(),
        "final_residuals": final_residuals.detach().cpu(),
        "final_component_mask": final_mask.detach().cpu(),
        "initial_residuals": initial_residuals.detach().cpu(),
        "initial_component_mask": initial_mask.detach().cpu(),
        "history": history,
        "checkpoint_history": checkpoint_history,
        "best_total_loss": best_loss,
        "last_total_loss": float(final_loss.detach().cpu()),
        "best_iteration": best_iteration,
        "stopped_early": stopped_early,
        "initial_total_loss": float(initial_loss.detach().cpu()),
        "initial_invalid_fraction": initial_invalid,
        "invalid_fraction": final_invalid,
        "invalid_history": invalid_history,
        "fisher_diagnostics": fisher_diagnostics,
    }


def run_joint_alignment(
    batches,
    geometry,
    config,
    active_dof=DEFAULT_ACTIVE_DOF,
    initial_track_params=None,
):
    """Jointly optimize free per-track helices and target-surface alignment."""
    use_cuda = bool(config.gpu and torch.cuda.is_available())
    if config.gpu and not use_cuda:
        print("  [warn] --gpu requested but CUDA is unavailable; using CPU float64.")
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    torch.manual_seed(config.rng_seed)
    np.random.seed(config.rng_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.rng_seed)

    if initial_track_params is None:
        initial_track_params = seed_helices(batches, device=device, dtype=dtype)
    initial_track_params = initial_track_params.to(device=device, dtype=dtype)
    initial_linear, initial_rho_scaled = split_joint_track_parameters(initial_track_params)
    track_linear = nn.Parameter(initial_linear.clone())
    track_rho_scaled = nn.Parameter(initial_rho_scaled.clone())
    n_surfaces = len(geometry.centers)
    translation = nn.Parameter(torch.zeros((n_surfaces, 3), device=device, dtype=dtype))
    rotation = nn.Parameter(torch.zeros((n_surfaces, 3), device=device, dtype=dtype))
    dof_mask = torch.zeros(6, device=device, dtype=dtype)
    dof_mask[list(active_dof)] = 1.0
    target_mask = geometry.target_mask.to(device=device, dtype=dtype).unsqueeze(-1)
    parameter_groups = [
        {"params": [translation], "lr": config.joint_lr_trans},
        {"params": [rotation], "lr": config.joint_lr_trans / config.rot_lr_scale},
    ]
    if not config.freeze_tracks:
        parameter_groups[0:0] = [
            {"params": [track_linear], "lr": config.joint_lr_track},
            {"params": [track_rho_scaled], "lr": config.joint_lr_track},
        ]
    optimizer = optim.Adam(parameter_groups)
    checkpoint = JointBestCheckpoint(
        (track_linear, track_rho_scaled, translation, rotation),
        patience=config.joint_patience,
        min_delta=config.joint_min_delta,
    )
    history = []
    checkpoint_history = []
    invalid_history = []
    n_tracks = len(track_linear)
    batch_size = (
        min(config.track_batch_size, n_tracks)
        if config.track_batch_size > 0
        else n_tracks
    )
    with torch.no_grad():
        initial_tracks = assemble_joint_track_parameters(track_linear, track_rho_scaled)
        initial_alignment = torch.zeros((n_surfaces, 6), device=device, dtype=dtype)
        initial_residuals, _initial_covariance, initial_component_mask, _initial_valid = (
            _joint_forward(initial_tracks, initial_alignment, batches, geometry)
        )
        initial_weighted = initial_residuals.square() / torch.clamp(
            _initial_covariance, min=EPS_COVARIANCE
        )
        initial_total_loss = float((torch.sum(torch.where(
            initial_component_mask, initial_weighted, torch.zeros_like(initial_weighted)
        )) / initial_component_mask.sum().clamp(min=1)).cpu())
        initial_invalid_fraction = 1.0 - float(
            _initial_valid[batches.hit_mask.to(device)].float().mean().item()
        )
    initial_checkpoint_loss = joint_checkpoint_score(
        initial_total_loss,
        initial_invalid_fraction,
        config.joint_max_invalid_fraction,
    )
    if not np.isfinite(initial_checkpoint_loss):
        raise ValueError(
            "Initial helix seed has too many invalid intersections for a safe checkpoint: "
            f"{initial_invalid_fraction:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    checkpoint.observe(initial_checkpoint_loss, iteration=-1)
    checkpoint_history.append((-1, initial_total_loss))

    if config.joint_iters < 1:
        raise ValueError("joint_iters must be at least 1.")
    checkpoint_interval = max(1, int(config.joint_checkpoint_interval))
    last_total_loss = float("inf")
    stopped_early = False
    for iteration in range(config.joint_iters):
        start = (iteration * batch_size) % n_tracks
        track_indices = torch.arange(start, min(start + batch_size, n_tracks), device=device)
        if len(track_indices) < batch_size:
            track_indices = torch.cat((
                track_indices,
                torch.arange(0, batch_size - len(track_indices), device=device),
            ))
        optimizer.zero_grad()
        track_parameters = assemble_joint_track_parameters(track_linear, track_rho_scaled)
        alignment = torch.cat((translation, rotation), dim=1)
        alignment = alignment * target_mask * dof_mask
        residuals, covariance, component_mask, intersection_valid = _joint_forward(
            track_parameters, alignment, batches, geometry, track_indices
        )
        weighted = residuals.square() / torch.clamp(covariance, min=EPS_COVARIANCE)
        data_loss = torch.sum(torch.where(component_mask, weighted, torch.zeros_like(weighted)))
        data_loss = data_loss / component_mask.sum().clamp(min=1)
        l2 = config.lambda_l2 * alignment.square().sum()
        loss = data_loss + l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite joint loss at iteration {iteration}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([track_linear, track_rho_scaled, translation, rotation], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            translation.mul_(target_mask * dof_mask[:3])
            rotation.mul_(target_mask * dof_mask[3:])
        clamp_parameter_and_reset_adam(
            translation, -config.joint_clamp_t, config.joint_clamp_t, optimizer
        )
        clamp_parameter_and_reset_adam(
            rotation, -config.joint_clamp_r, config.joint_clamp_r, optimizer
        )
        invalid_fraction = 1.0 - float(
            intersection_valid[batches.hit_mask.to(device)[track_indices]].float().mean().item()
        )
        history.append(float(data_loss.detach().cpu()))
        invalid_history.append(invalid_fraction)
        evaluate_checkpoint = (
            iteration == 0
            or (iteration + 1) % checkpoint_interval == 0
            or iteration == config.joint_iters - 1
        )
        if evaluate_checkpoint:
            with torch.no_grad():
                evaluated_tracks = assemble_joint_track_parameters(track_linear, track_rho_scaled)
                evaluated_alignment = torch.cat((translation, rotation), dim=1)
                evaluated_alignment = evaluated_alignment * target_mask * dof_mask
                evaluated_residuals, evaluated_covariance, evaluated_mask, evaluated_valid = _joint_forward(
                    evaluated_tracks, evaluated_alignment, batches, geometry
                )
                evaluated_weighted = evaluated_residuals.square() / torch.clamp(
                    evaluated_covariance, min=EPS_COVARIANCE
                )
                evaluated_data_loss = torch.sum(torch.where(
                    evaluated_mask, evaluated_weighted, torch.zeros_like(evaluated_weighted)
                )) / evaluated_mask.sum().clamp(min=1)
                evaluated_total_loss = (
                    evaluated_data_loss + config.lambda_l2 * evaluated_alignment.square().sum()
                )
                last_total_loss = float(evaluated_total_loss.cpu())
                evaluated_invalid_fraction = 1.0 - float(
                    evaluated_valid[batches.hit_mask.to(device)].float().mean().item()
                )
            checkpoint_loss = joint_checkpoint_score(
                last_total_loss,
                evaluated_invalid_fraction,
                config.joint_max_invalid_fraction,
            )
            checkpoint_history.append((iteration, last_total_loss))
            if checkpoint.observe(checkpoint_loss, iteration):
                stopped_early = True
                print(
                    f"Joint early stop at iteration {iteration}; "
                    f"best={checkpoint.best_loss:.4f} at iteration {checkpoint.best_iteration}."
                )
                break
        if iteration % 100 == 0 or iteration == config.joint_iters - 1:
            warning = " WARNING: >5%" if invalid_fraction > 0.05 else ""
            print(
                f"Joint [{iteration:04d}/{config.joint_iters}] chi2/component={history[-1]:.4f} "
                f"invalid={invalid_fraction:.2%}{warning}"
            )

    checkpoint.restore()
    with torch.no_grad():
        track_parameters = assemble_joint_track_parameters(track_linear, track_rho_scaled)
        predicted = torch.cat((translation, rotation), dim=1)
        predicted = predicted * target_mask * dof_mask
        final_residuals, _covariance, final_component_mask, _valid = _joint_forward(
            track_parameters, predicted, batches, geometry
        )
        final_invalid_fraction = 1.0 - float(
            _valid[batches.hit_mask.to(device)].float().mean().cpu()
        )
    if final_invalid_fraction > config.joint_max_invalid_fraction:
        raise ValueError(
            "Best joint checkpoint exceeds the invalid-intersection limit: "
            f"{final_invalid_fraction:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    return {
        "predicted_deltas": predicted.detach().cpu().double().numpy(),
        "track_params": track_parameters.detach().cpu().double().numpy(),
        "final_residuals": final_residuals.detach().cpu(),
        "final_component_mask": final_component_mask.detach().cpu(),
        "initial_residuals": initial_residuals.detach().cpu(),
        "initial_component_mask": initial_component_mask.detach().cpu(),
        "history": history,
        "checkpoint_history": checkpoint_history,
        "best_total_loss": checkpoint.best_loss,
        "last_total_loss": last_total_loss,
        "best_iteration": checkpoint.best_iteration,
        "stopped_early": stopped_early,
        "initial_total_loss": initial_total_loss,
        "initial_invalid_fraction": initial_invalid_fraction,
        "invalid_fraction": final_invalid_fraction,
    }

class AlignmentEngine(nn.Module):
    def __init__(self, num_sensors, use_hit_weights=False):
        super().__init__()
        self.num_sensors = num_sensors
        # EXPERIMENTAL: temporarily disabled; no caller sets use_hit_weights=True.
        # The network and its feature chain are kept so the experiment can be
        # restarted, but the disabled path must stay free of its cost.
        self.use_hit_weights = use_hit_weights

        self.alignment_params = nn.Parameter(torch.zeros(num_sensors, 6))

        if self.use_hit_weights:
            self.hit_quality_net = nn.Sequential(
                nn.Linear(HIT_QUALITY_INPUT_SIZE, HIT_QUALITY_HIDDEN_ONE),
                nn.GELU(),
                nn.Linear(HIT_QUALITY_HIDDEN_ONE, HIT_QUALITY_HIDDEN_TWO),
                nn.GELU(),
                nn.Linear(HIT_QUALITY_HIDDEN_TWO, HIT_QUALITY_OUTPUT_SIZE),
            )

    def forward(
        self,
        hit_sensor_ids,
        track_params,
        sensor_center_positions,
        sensor_rotations,
        track_chi2_ndf,
        measured_local,
        hit_global_positions,
        hit_directions,
    ):
        """Return ``(residuals, hit_weights)`` as ``tuple[Tensor, Tensor | None]``.

        ``hit_weights`` is ``None`` when the experimental weighting is disabled,
        which selects the unweighted loss path instead of multiplying by ones.
        """
        residuals = compute_autograd_residuals(
            measured_local,
            hit_global_positions,
            hit_directions,
            hit_sensor_ids,
            sensor_center_positions,
            sensor_rotations,
            self.alignment_params,
        )
        if not self.use_hit_weights:
            return residuals, None

        # EXPERIMENTAL: only reachable with use_hit_weights=True.
        hit_sensor_center_positions = sensor_center_positions[hit_sensor_ids]
        advanced_features = torch.cat([track_params, hit_sensor_center_positions, track_chi2_ndf], dim=-1)
        raw_logits = self.hit_quality_net(advanced_features)
        hit_weights = HIT_WEIGHT_FLOOR + (1.0 - HIT_WEIGHT_FLOOR) * torch.sigmoid(raw_logits)
        return residuals, hit_weights

def normalize_hit_weights(hit_weights, hit_volume_ids=None, hit_layer_ids=None):
    if hit_volume_ids is None and hit_layer_ids is None:
        return hit_weights / (torch.mean(hit_weights) + EPS_WEIGHT)
    if hit_volume_ids is None or hit_layer_ids is None:
        raise ValueError("hit_volume_ids and hit_layer_ids must be provided together.")
    if len(hit_weights) != len(hit_volume_ids) or len(hit_weights) != len(hit_layer_ids):
        raise ValueError("Hit weight and layer identifier tensors must have matching lengths.")

    volume_ids = hit_volume_ids.to(device=hit_weights.device)
    layer_ids = hit_layer_ids.to(device=hit_weights.device)
    layer_keys = torch.stack((volume_ids, layer_ids), dim=1)
    unique_layers = torch.unique(layer_keys, dim=0)

    normalized = torch.zeros_like(hit_weights)
    for vol, lay in unique_layers:
        layer_mask = (volume_ids == vol) & (layer_ids == lay)
        layer_scale = torch.mean(hit_weights[layer_mask]) + EPS_WEIGHT
        layer_weight = layer_mask.to(hit_weights.dtype).unsqueeze(-1)
        normalized = normalized + layer_weight * (hit_weights / layer_scale)
    return normalized


def physics_informed_residual_loss(
    residuals,
    covariances,
    hit_weights,
    component_mask=None,
    hit_volume_ids=None,
    hit_layer_ids=None,
):
    """``hit_weights=None`` selects the unweighted path (no normalisation)."""
    raw_chi2 = (residuals ** 2) / torch.clamp(covariances, min=EPS_COVARIANCE)

    if hit_weights is None:
        weighted_chi2 = raw_chi2
    else:
        w_normalized = normalize_hit_weights(hit_weights, hit_volume_ids, hit_layer_ids)
        weighted_chi2 = w_normalized * raw_chi2
    if component_mask is None:
        return torch.mean(weighted_chi2)
    return torch.sum(torch.where(component_mask, weighted_chi2, torch.zeros_like(weighted_chi2))) / (
        component_mask.sum().clamp(min=1)
    )

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

def parse_active_dof(active_dof):
    """Parse active alignment DOF from CLI text or an iterable of integers."""
    names = {
        "tx": 0, "ty": 1, "tz": 2,
        "rx": 3, "alpha": 3,
        "ry": 4, "beta": 4,
        "rz": 5, "gamma": 5,
    }
    if active_dof is None or active_dof == "":
        return DEFAULT_ACTIVE_DOF
    if isinstance(active_dof, str):
        raw_items = [item.strip().lower() for item in active_dof.split(",") if item.strip()]
    else:
        raw_items = list(active_dof)

    parsed = []
    for item in raw_items:
        if isinstance(item, str) and item in names:
            value = names[item]
        else:
            value = int(item)
        if value < 0 or value > 5:
            raise ValueError(f"Active DOF {value} is outside the valid range [0, 5].")
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("At least one active DOF must be selected.")
    return tuple(parsed)


def parse_target_layers(target_layers):
    """Parse target layer spec like '17:8,24:4' into ((17, 8), (24, 4))."""
    if target_layers is None or target_layers == "":
        return DEFAULT_TARGET_LAYERS
    if not isinstance(target_layers, str):
        return tuple((int(vol), int(lay)) for vol, lay in target_layers)

    parsed = []
    for item in target_layers.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Target layer '{item}' must have format volume:layer.")
        parsed.append((int(parts[0]), int(parts[1])))
    if not parsed:
        raise ValueError("At least one target layer must be selected.")
    return tuple(parsed)


def _read_root_columns(path, tree_name, columns, entry_stop=None):
    """Read ROOT columns with PyROOT first; uproot remains a fixture fallback."""
    try:
        import ROOT
    except ImportError:
        ROOT = None
    if ROOT is not None:
        frame = ROOT.RDataFrame(tree_name, str(path))
        requested = list(columns)
        available = {str(name) for name in frame.GetColumnNames()}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                f"ROOT tree {tree_name!r} is missing columns: {', '.join(missing)}; "
                f"requested columns: {', '.join(requested)}"
            )
        if entry_stop is not None:
            frame = frame.Range(int(entry_stop))
        try:
            return frame.AsNumpy(requested), "pyroot"
        except Exception as error:
            raise ValueError(
                f"PyROOT could not read {tree_name} columns from {path}: {error}"
            ) from error
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"ROOT tree {tree_name!r} is missing from {path}")
        tree = root_file[tree_name]
        missing = sorted(set(columns) - set(tree.keys()))
        if missing:
            raise ValueError(
                f"ROOT tree {tree_name!r} is missing columns: {', '.join(missing)}"
            )
        return tree.arrays(
            list(columns), entry_stop=entry_stop, library="ak", how=dict
        ), "uproot"


def _root_column_entries(values):
    if isinstance(values, ak.Array):
        return ak.to_list(values)
    return list(values)


def _flatten_root_entries(values):
    entries = _root_column_entries(values)
    if not entries:
        return np.asarray([], dtype=float)
    return np.concatenate([np.asarray(entry) for entry in entries])


def load_variable_information(
    tracksummary_filepath,
    trackstates_filepath,
    id_to_idx_map,
    max_track_entries=None,
    load_summary=True,
):
    # 1. Read tracksummary
    sum_df = None
    if load_summary:
        sum_keys = ["event_nr", "track_nr", "chi2Sum", "NDF"]
        sum_arrays, summary_backend = _read_root_columns(
            tracksummary_filepath,
            "tracksummary",
            sum_keys,
            entry_stop=max_track_entries,
        )
        track_entries = _root_column_entries(sum_arrays["track_nr"])
        summary_is_jagged = bool(
            track_entries and np.asarray(track_entries[0]).ndim > 0
        )
        if summary_is_jagged:
            summary_counts = np.asarray(
                [len(np.asarray(entry)) for entry in track_entries], dtype=np.int64
            )
            event_values = np.asarray(sum_arrays["event_nr"])
            summary_data = {
                "event_nr": np.repeat(event_values, summary_counts),
                "track_nr": _flatten_root_entries(sum_arrays["track_nr"]),
                "chi2Sum": _flatten_root_entries(sum_arrays["chi2Sum"]),
                "NDF": _flatten_root_entries(sum_arrays["NDF"]),
            }
        else:
            summary_data = {
                name: np.asarray(sum_arrays[name]) for name in sum_keys
            }
        sum_df = pd.DataFrame(summary_data).reset_index(drop=True)
        sum_df["event_nr"] = sum_df["event_nr"].astype(np.int64)
        sum_df["track_nr"] = sum_df["track_nr"].astype(np.int64)
        sum_df = sum_df.rename(columns={"chi2Sum":"chi2", "NDF": "ndf"})
        sum_df["chi2_ndf"] = sum_df["chi2"] / sum_df["ndf"]
        print(f"Tracks loaded: {len(sum_df)} trajectories ({summary_backend})")

    # 2. Load per-state columns only.
    # Background: the ROOT trackstates tree has two array-length classes per event:
    #   - per-state  (N_states / event): volume_id, l_x_hit, eLOC0_smt, g_x_smt, ...
    #   - per-measurement (N_meas / event, N_meas <= N_states): dim_hit, err_x_hit, err_y_hit
    #   - per-event  (scalar): event_nr, track_nr
    # Loading all classes together with ak.to_dataframe() truncates to N_meas rows and
    # fails to broadcast event_nr. Load per-state columns here, then expand true
    # measurement errors onto the measured states.
    per_state_keys = [
        "volume_id", "layer_id", "module_id",
        "l_x_hit", "l_y_hit",             # NaN for hole/material states
        "eLOC0_smt", "eLOC1_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
        "px_smt", "py_smt", "pz_smt",
        "ePHI_smt", "eTHETA_smt", "eQOP_smt",
    ]
    state_keys = per_state_keys + [
        "err_x_hit", "err_y_hit", "event_nr", "track_nr"
    ]
    state_columns, states_backend = _read_root_columns(
        trackstates_filepath,
        "trackstates",
        state_keys,
        entry_stop=max_track_entries,
    )
    states_df = pd.DataFrame(
        {name: _flatten_root_entries(state_columns[name]) for name in per_state_keys}
    )

    # Load true measurement errors separately: err_x_hit / err_y_hit are per-measurement
    # (N_meas ≤ N_states per track).  Expand them to per-state length by inserting NaN
    # at hole states, keyed by the per-state l_x_hit NaN pattern.
    # Both branches are required: the variance columns below consume them
    # unconditionally, so there is no posterior-variance fallback to degrade to.
    # A length mismatch used to leave the whole entry as NaN, after which the
    # affected hits were dropped silently by the dropna() further down.
    try:
        lx_jagged = _root_column_entries(state_columns["l_x_hit"])
        errx_jagged = _root_column_entries(state_columns["err_x_hit"])
        erry_jagged = _root_column_entries(state_columns["err_y_hit"])
        errx_flat = []
        erry_flat = []
        for i in range(len(lx_jagged)):
            lx_tr = np.asarray(lx_jagged[i])
            is_meas = ~np.isnan(lx_tr)
            n_st = len(lx_tr)
            n_meas = int(is_meas.sum())
            ex = np.full(n_st, np.nan)
            ey = np.full(n_st, np.nan)
            em_x = np.asarray(errx_jagged[i])
            em_y = np.asarray(erry_jagged[i])
            for name, values in (("err_x_hit", em_x), ("err_y_hit", em_y)):
                if len(values) != n_meas:
                    raise ValueError(
                        f"{name} length mismatch at entry {i}: "
                        f"got {len(values)}, expected {n_meas}"
                    )
            ex[is_meas] = em_x
            ey[is_meas] = em_y
            errx_flat.extend(ex.tolist())
            erry_flat.extend(ey.tolist())
        states_df["err_x_hit"] = errx_flat
        states_df["err_y_hit"] = erry_flat
        print(
            "  Loaded true measurement errors (err_x_hit / err_y_hit) "
            f"via per-state expansion ({states_backend})."
        )
    except (KeyError, ValueError) as exc:
        # Interpolate the cause so the entry/length detail stays visible at the
        # top level; `raise ... from exc` alone would hide it behind __cause__.
        raise ValueError(
            f"Required err_x_hit/err_y_hit measurement errors could not be loaded: {exc}"
        ) from exc

    # 3. Broadcast per-event event_nr / track_nr to per-state level.
    volume_entries = _root_column_entries(state_columns["volume_id"])
    counts = np.asarray([len(entry) for entry in volume_entries], dtype=np.int64)
    ev_flat = np.asarray(state_columns["event_nr"])
    tr_flat = np.asarray(state_columns["track_nr"])
    states_df["event_nr"] = np.repeat(ev_flat, counts)
    states_df["track_nr"] = np.repeat(tr_flat, counts)
    states_df["event_nr"] = states_df["event_nr"].astype(np.int64)
    states_df["track_nr"] = states_df["track_nr"].astype(np.int64)

    # 4. Inner merge: map track-level chi2/ndf onto each hit state.
    if sum_df is None:
        merged_df = states_df
        merged_df["chi2_ndf"] = 0.0
    else:
        merged_df = pd.merge(states_df, sum_df, on=["event_nr", "track_nr"], how="inner")

    # 5. Remove hole/material states: l_x_hit is NaN for states without a measurement.
    merged_df = merged_df[merged_df["l_x_hit"].notna()].copy()

    # 6. Map sensor ID.
    sensor_index = pd.Series(id_to_idx_map)
    sensor_index.index = pd.MultiIndex.from_tuples(
        sensor_index.index, names=["volume_id", "layer_id", "module_id"]
    )
    hit_index = pd.MultiIndex.from_frame(
        merged_df[["volume_id", "layer_id", "module_id"]]
    )
    merged_df["sensor_idx"] = hit_index.map(sensor_index)

    # 7. ACTS-like alignment residuals.
    # AlignmentAlgorithm builds residuals from the measurement and the smoothed state.
    merged_df["res_eLOC0"] = merged_df["l_x_hit"] - merged_df["eLOC0_smt"]
    merged_df["res_eLOC1"] = merged_df["l_y_hit"] - merged_df["eLOC1_smt"]

    meas_var0 = (merged_df["err_x_hit"] ** 2).clip(lower=EPS_WEIGHT)
    meas_var1 = (merged_df["err_y_hit"] ** 2).clip(lower=EPS_WEIGHT)
    merged_df["loc1_valid"] = merged_df["err_y_hit"].notna()
    merged_df["measured_loc1"] = merged_df["l_y_hit"].fillna(0.0)
    merged_df["raw_var_eLOC0"] = meas_var0
    merged_df["raw_var_eLOC1"] = meas_var1.fillna(1.0)
    merged_df["var_eLOC0"] = meas_var0
    merged_df["var_eLOC1"] = meas_var1
    print("  Using ACTS-like residuals: l_x/y_hit - eLOC0/1_smt .")

    # 8. Handle 1D strip hits: l_y_hit is NaN for 1D measurements.
    # Set LOC1 residual = 0 and weight = 0 so it contributes nothing to the normal equation.
    is_1d = merged_df["err_y_hit"].isna()
    merged_df.loc[is_1d, "res_eLOC1"] = 0.0
    merged_df.loc[is_1d, "var_eLOC1"] = 1.0
    n_1d = int(is_1d.sum())
    n_2d = int((~is_1d).sum())
    print(f"  Hit breakdown: {n_2d} 2D measurement, {n_1d} 1D measurement")

    # Physics floor for strip detector Y direction.
    # Vol 23/24/25 (ShortStrip barrel/endcap): intrinsic Y resolution ~1.2mm.
    # the optimizer to over-correct Y.  Clip var_eLOC1 to at least (1.2mm)^2.
    is_strip_vol = merged_df["volume_id"].isin(STRIP_VOLUMES)
    before_floor = merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].mean() if is_strip_vol.any() else 0.0
    strip_2d = is_strip_vol & ~is_1d
    merged_df.loc[strip_2d, "var_eLOC1"] = merged_df.loc[strip_2d, "var_eLOC1"].clip(lower=STRIP_Y_FLOOR)
    if is_strip_vol.any():
        after_floor = merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].mean()
        print(f"  Strip Y variance floor applied: mean var_LOC1 {before_floor:.4f} -> {after_floor:.4f} mm^2")

    # 9. Drop remaining invalids (sensor not in alignment set, NaN smoothed params).
    required_cols = [
        "sensor_idx", "res_eLOC0", "res_eLOC1",
        "var_eLOC0", "var_eLOC1",  "eLOC0_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
    ]
    merged_df = merged_df.dropna(subset=required_cols)
    merged_df = merged_df[merged_df["module_id"] > 0].copy()
    merged_df["r_xy"] = np.hypot(merged_df["g_x_smt"], merged_df["g_y_smt"])
    merged_df["abs_g_z"] = merged_df["g_z_smt"].abs()
    merged_df = merged_df.sort_values(
        [
            "event_nr", "track_nr", "r_xy", "abs_g_z",
            "volume_id", "layer_id", "module_id",
            "g_x_smt", "g_y_smt", "g_z_smt",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    num_hits = len(merged_df)
    print(f"Feature alignment complete! A total of {num_hits} valid tensor data points have been constructed")

    # 10. Build PyTorch Tensors.
    hit_sensor_ids = torch.tensor(merged_df["sensor_idx"].values, dtype=torch.long)

    # Alignment residuals: r = m_k - H * x_smt
    initial_residuals = torch.tensor(
        merged_df[["res_eLOC0", "res_eLOC1"]].values, dtype=torch.float32)

    # Measurement variances: V = err_x/y_hit^2 (diagonal LOC0/LOC1 components)
    covariances = torch.tensor(
        merged_df[["var_eLOC0", "var_eLOC1"]].values, dtype=torch.float32)
    raw_covariances = torch.tensor(
        merged_df[["raw_var_eLOC0", "raw_var_eLOC1"]].values, dtype=torch.float32)
    measured_local = torch.tensor(
        merged_df[["l_x_hit", "measured_loc1"]].values, dtype=torch.float32)
    loc1_valid = torch.tensor(merged_df["loc1_valid"].values, dtype=torch.bool)

    # Smoothed track parameters as neural-net input features.
    # Jacobian is also evaluated at the smoothed position/direction (consistent
    # with AlignmentEngine.hpp:229-233 using freeSmoothed).
    track_params = torch.tensor(
        merged_df[["eLOC0_smt", "eLOC1_smt", "eTHETA_smt", "ePHI_smt", "eQOP_smt"]].values,
        dtype=torch.float32)

    track_chi2_ndf = torch.tensor(merged_df[["chi2_ndf"]].values, dtype=torch.float32)

    hit_global_positions = torch.tensor(
        merged_df[["g_x_smt", "g_y_smt", "g_z_smt"]].values, dtype=torch.float32)
    hit_directions = torch.tensor(
        merged_df[["px_smt", "py_smt", "pz_smt"]].values, dtype=torch.float32)
    hit_qoverp = torch.tensor(merged_df[["eQOP_smt"]].values, dtype=torch.float32)

    hit_volume_ids = torch.tensor(merged_df["volume_id"].values, dtype=torch.long)
    hit_layer_ids  = torch.tensor(merged_df["layer_id"].values,  dtype=torch.long)
    hit_module_ids = torch.tensor(merged_df["module_id"].values, dtype=torch.long)
    hit_event_ids = torch.tensor(merged_df["event_nr"].values, dtype=torch.long)
    hit_track_ids = torch.tensor(merged_df["track_nr"].values, dtype=torch.long)

    inputs = AlignmentInputs(
        hit_sensor_ids=hit_sensor_ids,
        initial_residuals=initial_residuals,
        covariances=covariances,
        track_params=track_params,
        track_chi2_ndf=track_chi2_ndf,
        hit_global_positions=hit_global_positions,
        hit_directions=hit_directions,
        hit_qoverp=hit_qoverp,
        hit_volume_ids=hit_volume_ids,
        hit_layer_ids=hit_layer_ids,
        hit_module_ids=hit_module_ids,
        hit_event_ids=hit_event_ids,
        hit_track_ids=hit_track_ids,
        measured_local=measured_local,
        loc1_valid=loc1_valid,
        raw_covariances=raw_covariances,
        merged_df=merged_df,
    )
    return inputs

def euler_to_rotation_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(rotation):
    """Decompose a rotation using the project's ``Rz @ Ry @ Rx`` convention."""
    rotation = np.asarray(rotation, dtype=float)
    ry = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    rx = np.arctan2(rotation[2, 1], rotation[2, 2])
    rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    return float(rx), float(ry), float(rz)


def torch_euler_to_rotation_matrix(delta_rot):
    alpha, beta, gamma = delta_rot.unbind(-1)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    ones = torch.ones_like(alpha)
    zeros = torch.zeros_like(alpha)

    Rx = torch.stack(
        [ones, zeros, zeros, zeros, ca, -sa, zeros, sa, ca], dim=-1
    ).reshape(*alpha.shape, 3, 3)
    Ry = torch.stack(
        [cb, zeros, sb, zeros, ones, zeros, -sb, zeros, cb], dim=-1
    ).reshape(*alpha.shape, 3, 3)
    Rz = torch.stack(
        [cg, -sg, zeros, sg, cg, zeros, zeros, zeros, ones], dim=-1
    ).reshape(*alpha.shape, 3, 3)
    return Rz @ Ry @ Rx


def apply_alignment_torch(center0, rotation0, alignment_params):
    delta_t_local = alignment_params[..., :3]
    delta_rot = alignment_params[..., 3:]
    center = center0 + torch.matmul(rotation0, delta_t_local.unsqueeze(-1)).squeeze(-1)
    rotation = rotation0 @ torch_euler_to_rotation_matrix(delta_rot)
    return center, rotation


def project_track_to_aligned_surface(
    hit_global_positions,
    hit_directions,
    center,
    rotation,
):
    direction_norm = torch.linalg.norm(hit_directions, dim=-1, keepdim=True)
    direction = hit_directions / torch.clamp(direction_norm, min=EPS_DIRECTION_NORM)
    normal = rotation[..., :, 2]
    numerator = torch.sum((center - hit_global_positions) * normal, dim=-1)
    denominator = torch.sum(direction * normal, dim=-1)
    eps = torch.full_like(denominator, EPS_DIRECTION_NORM)
    denominator_safe = torch.where(
        denominator.abs() > EPS_DIRECTION_NORM,
        denominator,
        torch.where(denominator >= 0, eps, -eps),
    )
    path_length = numerator / denominator_safe
    return hit_global_positions + path_length.unsqueeze(-1) * direction


def compute_autograd_residuals(
    measured_local,
    hit_global_positions,
    hit_directions,
    hit_sensor_ids,
    sensor_center_positions,
    sensor_rotations,
    alignment_params,
):
    current_params = alignment_params[hit_sensor_ids]
    center0 = sensor_center_positions[hit_sensor_ids].to(
        device=current_params.device, dtype=current_params.dtype
    )
    rotation0 = sensor_rotations[hit_sensor_ids].to(
        device=current_params.device, dtype=current_params.dtype
    )
    hit_global_positions = hit_global_positions.to(
        device=current_params.device, dtype=current_params.dtype
    )
    hit_directions = hit_directions.to(device=current_params.device, dtype=current_params.dtype)
    measured_local = measured_local.to(device=current_params.device, dtype=current_params.dtype)

    center, rotation = apply_alignment_torch(center0, rotation0, current_params)
    intersection = project_track_to_aligned_surface(
        hit_global_positions, hit_directions, center, rotation
    )
    local3 = torch.matmul(
        rotation.transpose(-1, -2), (intersection - center).unsqueeze(-1)
    ).squeeze(-1)
    predicted_local = local3[..., :2]
    return measured_local - predicted_local


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


def combine_component_masks(hit_mask, initial_component_mask, final_component_mask):
    """Intersect before/after component validity and report per-side exclusions.

    Comparing "before" residuals selected by the *final* mask is unfair: a hit
    whose initial intersection failed had its residual zeroed, so it would enter
    the before statistics as an artificial 0.0.

    LOC0 is the gate because ``_joint_forward`` builds the component mask as
    ``(valid, valid & loc1_mask)`` — a valid LOC1 always implies a valid LOC0.
    The exclusion counts are taken on LOC0 and intersected with ``hit_mask`` so
    padding never contributes and neither side is counted twice.
    """
    both = initial_component_mask & final_component_mask
    hit_valid = hit_mask & both[..., 0]
    initial_only = hit_mask & initial_component_mask[..., 0] & ~final_component_mask[..., 0]
    final_only = hit_mask & ~initial_component_mask[..., 0] & final_component_mask[..., 0]
    return hit_valid, both[hit_valid], int(initial_only.sum()), int(final_only.sum())


def _as_numpy(values):
    """Detach torch tensors without forcing a dtype promotion.

    Callers mix torch tensors and ndarrays; converting only some of them would
    leave `ndarray & Tensor` mixed operations whose semantics depend on the
    conversion rules.  No `dtype=` is passed: numpy's `float` is float64 and the
    surface residuals are float32, so promoting here would change the reported
    statistics.
    """
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


def _stat_pair(values):
    """(mean, RMS) of a 1-D array, or (nan, nan) when it is empty.

    RMS is sqrt(mean(x^2)) — not np.std.  A detector misalignment shows up as a
    non-zero residual *mean*, which np.std subtracts away.
    """
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.sqrt(np.mean(values ** 2)))


def _fmt_mm(value):
    return "       N/A" if not np.isfinite(value) else f"{value:>9.4f}mm"


def print_target_residual_stats(
    initial_residuals, hit_volume_ids, hit_layer_ids, target_layers, loc1_valid=None
):
    res_np = _as_numpy(initial_residuals)
    vols_np = _as_numpy(hit_volume_ids)
    lays_np = _as_numpy(hit_layer_ids)
    if loc1_valid is None:
        loc1_np = np.ones(len(res_np), dtype=bool)
    else:
        loc1_np = _as_numpy(loc1_valid).astype(bool)

    print("\n=== Target-layer ACTS KF-smoothed residuals before solve ===")
    print(f"  {'Layer':<20}  {'N0':>8}  {'N1':>8}  {'mean LOC0':>12}  {'RMS LOC0':>10}"
          f"  {'mean LOC1':>12}  {'RMS LOC1':>10}")
    for vol, lay in parse_target_layers(target_layers):
        mask = (vols_np == vol) & (lays_np == lay)
        if mask.sum() == 0:
            print(f"  vol={vol} lay={lay:<6}  no valid hits")
            continue
        mask1 = mask & loc1_np
        mean0, rms0 = _stat_pair(res_np[mask, 0])
        mean1, rms1 = _stat_pair(res_np[mask1, 1])
        label = f"vol={vol} lay={lay}"
        cell = lambda v, w: f"{'N/A':>{w}}" if not np.isfinite(v) else f"{v:>{w}.5f}"
        print(f"  {label:<20}  {int(mask.sum()):>8}  {int(mask1.sum()):>8}  "
              f"{cell(mean0, 12)}  {cell(rms0, 10)}  {cell(mean1, 12)}  {cell(rms1, 10)}")


def print_per_layer_residual_diagnostics(
    init_res_np, final_res, vols_np, lays_np, target_layers,
    return_metrics=False,
    component_mask=None,
):
    """Per-layer before/after residual tables.

    ``component_mask`` is an ``(N, 2)`` boolean array marking which residual
    components are real measurements.  Without it the zeros written into LOC1
    for 1-D strip hits are averaged in and dilute the Y statistics.
    New parameters are appended so existing positional calls keep working.
    """
    init_res_np = _as_numpy(init_res_np)
    final_res = _as_numpy(final_res)
    vols_np = _as_numpy(vols_np)
    lays_np = _as_numpy(lays_np)

    if init_res_np.shape != final_res.shape:
        raise ValueError(
            f"init_res_np shape {init_res_np.shape} != final_res shape {final_res.shape}"
        )
    if init_res_np.ndim != 2 or init_res_np.shape[1] != 2:
        raise ValueError(f"residuals must have shape (N, 2), got {init_res_np.shape}")
    for name, array in (("init_res_np", init_res_np), ("final_res", final_res)):
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    n_hits = init_res_np.shape[0]
    if len(vols_np) != n_hits or len(lays_np) != n_hits:
        raise ValueError(
            f"vols_np ({len(vols_np)}) and lays_np ({len(lays_np)}) must both have "
            f"length {n_hits}"
        )

    if component_mask is None:
        component_mask = np.ones(init_res_np.shape, dtype=bool)
    else:
        component_mask = _as_numpy(component_mask).astype(bool)
    if component_mask.shape != init_res_np.shape:
        raise ValueError(
            f"component_mask shape {component_mask.shape} != residual shape "
            f"{init_res_np.shape}"
        )

    header = (f"  {'Layer':<20}  {'X_before':>10}  {'X_after':>10}"
              f"  {'Y_before':>10}  {'Y_after':>10}")
    metrics = {}
    layer_masks = {}
    for vol, lay in target_layers:
        layer = (vols_np == vol) & (lays_np == lay)
        layer_masks[(vol, lay)] = (layer & component_mask[:, 0], layer & component_mask[:, 1])

    print("\n=== Per-layer residual mean (training data) ===")
    print(header)
    for vol, lay in target_layers:
        mask0, mask1 = layer_masks[(vol, lay)]
        n0, n1 = int(mask0.sum()), int(mask1.sum())
        x_bef, rms_x_bef = _stat_pair(init_res_np[mask0, 0])
        x_aft, rms_x_aft = _stat_pair(final_res[mask0, 0])
        y_bef, rms_y_bef = _stat_pair(init_res_np[mask1, 1])
        y_aft, rms_y_aft = _stat_pair(final_res[mask1, 1])
        label = f"vol={vol} lay={lay} (n0={n0}, n1={n1})"
        print(f"  {label:<28}  {_fmt_mm(x_bef)}  {_fmt_mm(x_aft)}"
              f"  {_fmt_mm(y_bef)}  {_fmt_mm(y_aft)}")
        metrics[(vol, lay)] = {
            "n_hits": n0,          # retained alias for existing callers
            "n_hits_loc0": n0,
            "n_hits_loc1": n1,
            "mean_before_loc0": x_bef, "mean_after_loc0": x_aft,
            "mean_before_loc1": y_bef, "mean_after_loc1": y_aft,
            "rms_before_loc0": rms_x_bef, "rms_after_loc0": rms_x_aft,
            "rms_before_loc1": rms_y_bef, "rms_after_loc1": rms_y_aft,
        }

    print("\n=== Per-layer residual RMS (training data) ===")
    print(header)
    for vol, lay in target_layers:
        entry = metrics[(vol, lay)]
        label = f"vol={vol} lay={lay} (n0={entry['n_hits_loc0']}, n1={entry['n_hits_loc1']})"
        print(f"  {label:<28}  {_fmt_mm(entry['rms_before_loc0'])}"
              f"  {_fmt_mm(entry['rms_after_loc0'])}"
              f"  {_fmt_mm(entry['rms_before_loc1'])}"
              f"  {_fmt_mm(entry['rms_after_loc1'])}")
    print()
    return metrics if return_metrics else None

def validate_against_truth(transforms_list, predicted_deltas, nominal_json,
                           target_layers):
    """Compare ML correction against the TRUE misalignment per sensor (Iteration 4).

    The real ground truth: applying δα to the misaligned surface must recover the
    nominal one.  In ml convention (new_c = c_mis + R_mis·δt, new_R = R_mis·Rδ):
        δt_true  = R_misᵀ (c_nom - c_mis)        (local tx, ty, tz)
        Rδ_true  = R_misᵀ R_nom                  → rz = atan2(Rδ[1,0], Rδ[0,0])
    Reports per-DOF correlation + recovery fraction over each target layer; this is
    the metric that actually decides whether the local signal is recoverable.
    """
    if not os.path.exists(nominal_json):
        print(f"  [truth] nominal {nominal_json} not found — skipping.")
        return {}
    nom = {}
    with open(nominal_json, "r", encoding="utf-8") as f:
        nominal_data = json.load(f)
    for it in nominal_data:
        m = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
        if m:
            nom[tuple(map(int, m.groups()))] = (np.array(it["Translation"], float),
                                                np.array(it["RotationMatrix"], float))
    print("\n=== ML vs TRUE misalignment (local frame, per sensor) ===")
    metrics = {}
    for vol, lay in target_layers:
        true_v, pred_v = [], []
        for i, it in enumerate(transforms_list):
            m = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
            if not m:
                continue
            key = tuple(map(int, m.groups()))
            if key[0] != vol or key[1] != lay or key not in nom:
                continue
            c_mis = np.array(it["Translation"], float); R_mis = np.array(it["RotationMatrix"], float)
            c_nom, R_nom = nom[key]
            dt_true = R_mis.T @ (c_nom - c_mis)
            Rd = R_mis.T @ R_nom
            rx_true, ry_true, rz_true = rotation_matrix_to_euler(Rd)
            true_v.append([*dt_true, rx_true, ry_true, rz_true])
            p = predicted_deltas[i]
            pred_v.append(p[:6])
        if not true_v:
            continue
        true_v = np.array(true_v); pred_v = np.array(pred_v)
        names = ["tx", "ty", "tz", "rx", "ry", "rz"]
        print(f"  vol={vol} lay={lay} (n={len(true_v)})")
        layer_metrics = {}
        for j, nm in enumerate(names):
            t, p = true_v[:, j], pred_v[:, j]
            corr = np.corrcoef(t, p)[0, 1] if np.std(t) > 0 and np.std(p) > 0 else float("nan")
            recov = (p @ t) / (t @ t) if (t @ t) > 0 else float("nan")  # projection / recovery fraction
            unit = "mm" if j < 3 else "rad"
            layer_metrics[nm] = {
                "corr": float(corr),
                "recovery": float(recov),
                "true_rms": float(np.sqrt((t ** 2).mean())),
                "pred_rms": float(np.sqrt((p ** 2).mean())),
                "unit": unit,
            }
            print(f"    {nm}: corr={corr:+.3f}  recovery={recov:+.3f}  "
                  f"|true|rms={np.sqrt((t**2).mean()):.4f}{unit}  |pred|rms={np.sqrt((p**2).mean()):.4f}{unit}")
        metrics[(vol, lay)] = layer_metrics
    return metrics


def run_free_alignment_pipeline(config, target_layers, active_dof):
    if not os.path.exists(config.nominal_full_geom):
        raise FileNotFoundError(
            f"Full nominal geometry '{config.nominal_full_geom}' is required for free mode."
        )
    geometry = load_joint_geometry(
        config.nominal_full_geom, config.input_geom, target_layers
    )
    print(
        f"Joint working geometry: {len(geometry.transforms)} surfaces, "
        f"{int(geometry.target_mask.sum())} trainable targets, "
        f"{len(geometry.input_indices)} overlaid input surfaces."
    )
    inputs = load_variable_information(
        config.summary_root,
        config.states_root,
        geometry.id_to_idx,
        max_track_entries=(config.max_tracks * 4 if config.max_tracks > 0 else None),
        load_summary=False,
    )
    if config.event_partition is not None:
        inputs = filter_alignment_inputs_by_event_partition(
            inputs, config.event_partition, config.event_split_seed
        )
    print_target_residual_stats(
        inputs.initial_residuals, inputs.hit_volume_ids, inputs.hit_layer_ids, target_layers,
        loc1_valid=inputs.loc1_valid,
    )
    batches = build_track_batches(
        inputs,
        target_layers,
        min_hits=config.joint_seed_min_hits,
        min_span=config.joint_seed_min_span,
        max_tracks=config.max_tracks,
    )
    print(
        f"Joint track selection: {len(batches.event_track_ids)} tracks, "
        f"{int(batches.hit_mask.sum())} hits, padded width {batches.hit_mask.shape[1]}."
    )
    seeded_parameters = seed_helices(
        batches,
        n_iter=config.joint_smt_seed_iters,
        device="cuda" if config.gpu and torch.cuda.is_available() else "cpu",
        dtype=torch.float32 if config.gpu and torch.cuda.is_available() else torch.float64,
    )
    smt_stats = smt_seed_statistics(batches, seeded_parameters)
    if smt_stats["invalid_fraction"] > config.joint_max_invalid_fraction:
        raise ValueError(
            "SMT helix seed intersection failure rate exceeds limit: "
            f"{smt_stats['invalid_fraction']:.2%} > "
            f"{config.joint_max_invalid_fraction:.2%}."
        )
    refitted_parameters, measurement_stats = fit_helices_to_measurements(
        batches,
        geometry,
        seeded_parameters,
        n_iter=config.joint_measurement_refit_iters,
        device=seeded_parameters.device,
        dtype=seeded_parameters.dtype,
        max_invalid_fraction=config.joint_max_invalid_fraction,
        lr_position=config.measurement_refit_lr_position,
        lr_direction=config.measurement_refit_lr_direction,
        lr_rho_scaled=config.measurement_refit_lr_rho_scaled,
        stationarity=config.joint_refit_stationarity,
        solver=config.joint_refit_solver,
    )
    print(
        "Joint initialization: "
        f"tracks={batches.selection_stats.get('input_tracks', len(batches.event_track_ids))} "
        f"retained={batches.selection_stats.get('retained_tracks', len(batches.event_track_ids))} "
        f"excluded={batches.selection_stats.get('excluded_tracks', 0)} "
        f"SMT loss={smt_stats['loss']:.6g} "
        f"refit solver={measurement_stats['solver']} "
        f"measurement chi2/component={measurement_stats['chi2_per_component']:.6g} "
        f"n_components={measurement_stats['n_components']}."
    )
    if "stationarity" in measurement_stats:
        seed_report = measurement_stats["stationarity"]["seed"]
        refit_report = measurement_stats["stationarity"]["refit"]
        print(
            "Joint refit stationarity (F = 0.5 chi2 gauge, threshold "
            f"{refit_report['stationary_threshold']:.3g}): "
            f"non-stationary rows {seed_report['non_stationary_row_fraction']:.2%} "
            f"-> {refit_report['non_stationary_row_fraction']:.2%}, "
            f"decrement p90 {seed_report['newton_decrement_p90']:.3g} "
            f"-> {refit_report['newton_decrement_p90']:.3g}, "
            f"non-finite {refit_report['n_non_finite']}."
        )
    with torch.no_grad():
        seed_alignment = torch.zeros(
            (len(geometry.centers), 6),
            device=seeded_parameters.device,
            dtype=seeded_parameters.dtype,
        )
        seed_residuals, seed_covariance, seed_component_mask, _seed_valid = (
            _joint_forward(seeded_parameters, seed_alignment, batches, geometry)
        )
    seed_layer_chi2 = summarize_joint_layer_chi2(
        seed_residuals,
        seed_covariance,
        seed_component_mask,
        batches.volume_id,
        batches.layer_id,
        target_layers,
    )
    solver = (
        run_block_lm_alignment
        if config.joint_solver == "block-lm"
        else run_joint_alignment
    )
    result = solver(
        batches,
        geometry,
        config,
        active_dof=active_dof,
        initial_track_params=refitted_parameters,
    )
    predicted_deltas = result["predicted_deltas"]
    truth_metrics = validate_against_truth(
        geometry.transforms,
        predicted_deltas,
        config.nominal_file,
        target_layers,
    )
    Path(config.output_geom).parent.mkdir(parents=True, exist_ok=True)
    export_joint_geometry(geometry, predicted_deltas, config.output_geom)
    print(f"Export successful: {config.output_geom}")

    if config.make_plot and result["history"]:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(result["history"]) + 1), result["history"], color="firebrick")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Joint chi2 / valid component")
        plt.title("Free track + geometry joint optimization")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plot_path = Path(config.output_geom).parent / f"loss_curve_{Path(config.output_geom).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training convergence plot saved: '{plot_path}'")

    initial_residuals = result["initial_residuals"].numpy()
    final_residuals = result["final_residuals"].numpy()
    hit_valid, component_mask, initial_only, final_only = combine_component_masks(
        batches.hit_mask,
        result["initial_component_mask"],
        result["final_component_mask"],
    )
    print(f"Diagnostics hit selection: initial_only={initial_only} "
          f"final_only={final_only} (excluded from the before/after comparison)")
    valid = hit_valid.numpy()
    target_metrics = print_per_layer_residual_diagnostics(
        initial_residuals[valid], final_residuals[valid],
        batches.volume_id.numpy()[valid], batches.layer_id.numpy()[valid],
        target_layers, return_metrics=True, component_mask=component_mask,
    )
    stage_layer_chi2 = {
        "smt_seed": seed_layer_chi2,
        "measurement_refit": summarize_joint_layer_chi2(
            result["initial_residuals"],
            batches.covariances,
            result["initial_component_mask"],
            batches.volume_id,
            batches.layer_id,
            target_layers,
        ),
        "joint": summarize_joint_layer_chi2(
            result["final_residuals"],
            batches.covariances,
            result["final_component_mask"],
            batches.volume_id,
            batches.layer_id,
            target_layers,
        ),
    }

    result.update({
        "output_geom": config.output_geom,
        "truth_metrics": truth_metrics,
        "n_tracks": len(batches.event_track_ids),
        "n_hits": int(batches.hit_mask.sum()),
        "target_metrics": target_metrics,
        "initialization_stats": {
            **batches.selection_stats,
            "fit_hits": int(batches.fit_mask.sum()),
            "min_fit_span": float(batches.fit_span.min()),
            "smt": smt_stats,
            "measurement_refit": measurement_stats,
        },
        "stage_layer_chi2": stage_layer_chi2,
    })
    return result


def resolve_active_dof(config):
    """Both track modes share one default; parse_active_dof handles None and ''."""
    return parse_active_dof(config.active_dof)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_alignment_summary(result, output_path):
    """Write compact solver observability without duplicating hit-sized arrays."""
    retained_keys = (
        "output_geom",
        "n_tracks",
        "n_hits",
        "initialization_stats",
        "stage_layer_chi2",
        "target_metrics",
        "truth_metrics",
        "best_total_loss",
        "last_total_loss",
        "best_iteration",
        "stopped_early",
        "initial_total_loss",
        "initial_invalid_fraction",
        "invalid_fraction",
        "solver",
    )
    summary = {key: _json_ready(result[key]) for key in retained_keys if key in result}
    if "predicted_deltas" in result:
        corrections = np.asarray(result["predicted_deltas"], dtype=float)
        summary["correction_max_abs"] = (
            float(np.max(np.abs(corrections))) if corrections.size else 0.0
        )
    fisher = result.get("fisher_diagnostics")
    if fisher is not None:
        finite_conditions = [
            float(item["condition_number"])
            for item in fisher
            if math.isfinite(float(item.get("condition_number", float("inf"))))
        ]
        regularizations = [float(item.get("regularization", 0.0)) for item in fisher]
        summary["fisher_summary"] = {
            "evaluations": len(fisher),
            "accepted_steps": sum(bool(item.get("accepted")) for item in fisher),
            "finite_condition_max": max(finite_conditions) if finite_conditions else None,
            "regularization_max": max(regularizations) if regularizations else None,
        }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination


def run_alignment_pipeline(config=None):
    if config is None:
        config = AlignmentRunConfig()

    if config.track_mode not in {"surface", "free"}:
        raise ValueError("track_mode must be 'surface' or 'free'.")
    ACTIVE_DOF = resolve_active_dof(config)
    DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]
    print(f"Active alignment DOF: {[DOF_NAMES[d] for d in ACTIVE_DOF]}")
    TARGET_LAYERS = parse_target_layers(config.target_layers)
    IDEAL_JSON_FILE = config.input_geom
    ALIGNED_JSON_FILE = config.output_geom
    SUMMARY_ROOT_FILE = config.summary_root
    STATES_ROOT_FILE = config.states_root

    if config.track_mode == "free":
        for path, label in (
            (config.input_geom, "input geometry"),
            (config.summary_root, "track summary"),
            (config.states_root, "track states"),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {label}: {path}")
        return run_free_alignment_pipeline(config, TARGET_LAYERS, ACTIVE_DOF)

    print("======================================================")
    print(" Activate the ML-Driven detector alignment engine ")
    if config.iter_label is not None:
        print(f" Iteration {config.iter_label}")
    print("======================================================")

    if not os.path.exists(IDEAL_JSON_FILE):
        raise FileNotFoundError(f"ACTS geometry file '{IDEAL_JSON_FILE}' not found.")

    print(f"Reading detector misaligned geometry: {IDEAL_JSON_FILE}")
    (
        transforms_list,
        NUM_SENSORS,
        initial_positions,
        initial_rotations,
        id_to_idx_map,
    ) = load_geometry_from_json(IDEAL_JSON_FILE)
    print(f"Geometry file loaded successfully! {NUM_SENSORS} detector modules resolved.")

    if not os.path.exists(SUMMARY_ROOT_FILE) or not os.path.exists(STATES_ROOT_FILE):
        missing = [path for path in (SUMMARY_ROOT_FILE, STATES_ROOT_FILE) if not os.path.exists(path)]
        raise FileNotFoundError(f"ROOT file not found: {', '.join(missing)}")

    inputs = load_variable_information(
        SUMMARY_ROOT_FILE,
        STATES_ROOT_FILE,
        id_to_idx_map,
    )
    if config.event_partition is not None:
        inputs = filter_alignment_inputs_by_event_partition(
            inputs, config.event_partition, config.event_split_seed
        )
    if inputs.measured_local is None or inputs.loc1_valid is None:
        raise ValueError("Surface mode requires measured_local and loc1_valid.")

    print_target_residual_stats(
        inputs.initial_residuals, inputs.hit_volume_ids, inputs.hit_layer_ids, TARGET_LAYERS,
        loc1_valid=inputs.loc1_valid,
    )

    training_hit_mask = torch.zeros(len(inputs.hit_sensor_ids), dtype=torch.bool)
    for vol, lay in TARGET_LAYERS:
        training_hit_mask |= (inputs.hit_volume_ids == vol) & (inputs.hit_layer_ids == lay)
    training_hit_count = int(training_hit_mask.sum())
    print(f"Training hit mask: {training_hit_count} target-layer hits "
          f"out of {len(training_hit_mask)} total")
    if training_hit_count == 0:
        raise ValueError(f"No hits found for target layers {TARGET_LAYERS}.")

    train_hit_sensor_ids = inputs.hit_sensor_ids[training_hit_mask]
    train_track_params = inputs.track_params[training_hit_mask]
    train_track_chi2_ndf = inputs.track_chi2_ndf[training_hit_mask]
    train_measured_local = inputs.measured_local[training_hit_mask]
    train_loc1_valid = inputs.loc1_valid[training_hit_mask]
    train_component_mask = torch.stack(
        (torch.ones_like(train_loc1_valid), train_loc1_valid), dim=1
    )
    train_covariances = inputs.covariances[training_hit_mask]
    train_hit_global_positions = inputs.hit_global_positions[training_hit_mask]
    train_hit_directions = inputs.hit_directions[training_hit_mask]
    train_hit_volume_ids = inputs.hit_volume_ids[training_hit_mask]
    train_hit_layer_ids = inputs.hit_layer_ids[training_hit_mask]

    print("Using PyTorch autograd residual projection; no C++ alignment Jacobian is imported.")

    history_epochs = []
    history_chi2_loss = []

    # Build the fixed-reference mask: True = target surface (calibrated),
    # False = reference surface (held fixed at δ=0).
    target_mask = None
    if config.fix_reference:
        target_mask = build_target_mask(NUM_SENSORS, id_to_idx_map, TARGET_LAYERS)
        print(f"Fixed-reference mode: {int(target_mask.sum())} target surfaces "
              f"({TARGET_LAYERS}) calibrated, {NUM_SENSORS - int(target_mask.sum())} held as reference.")

    engine = AlignmentEngine(NUM_SENSORS)
    dof_mask = torch.zeros(6, dtype=torch.bool)
    for d in ACTIVE_DOF:
        dof_mask[d] = True
    optimizer = optim.Adam(engine.parameters(), lr=DEFAULT_ADAM_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=DEFAULT_ADAM_ETA_MIN
    )
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()

        cur_res, hit_weights = engine(
            train_hit_sensor_ids,
            train_track_params,
            initial_positions,
            initial_rotations,
            train_track_chi2_ndf,
            train_measured_local,
            train_hit_global_positions,
            train_hit_directions,
        )

        chi2_loss = physics_informed_residual_loss(
            cur_res, train_covariances,
            hit_weights,
            component_mask=train_component_mask,
            hit_volume_ids=train_hit_volume_ids,
            hit_layer_ids=train_hit_layer_ids,
        )

        l2_reg = config.lambda_l2 * torch.sum(engine.alignment_params ** 2)
        loss = chi2_loss + l2_reg
        loss.backward()
        if engine.alignment_params.grad is not None:
            engine.alignment_params.grad[:, ~dof_mask] = 0.0
            if target_mask is not None:
                engine.alignment_params.grad[~target_mask] = 0.0
        torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        history_epochs.append(epoch)
        history_chi2_loss.append(chi2_loss.item())

        with torch.no_grad():
            engine.alignment_params[:, ~dof_mask] = 0.0
            if target_mask is not None:
                engine.alignment_params[~target_mask] = 0.0
        alignment_lower = engine.alignment_params.new_tensor(
            [-ADAM_CLAMP_T] * 3 + [-ADAM_CLAMP_R] * 3
        )
        alignment_upper = -alignment_lower
        clamp_parameter_and_reset_adam(
            engine.alignment_params, alignment_lower, alignment_upper, optimizer
        )

        with torch.no_grad():
            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                cur_res = compute_autograd_residuals(
                    train_measured_local,
                    train_hit_global_positions,
                    train_hit_directions,
                    train_hit_sensor_ids,
                    initial_positions,
                    initial_rotations,
                    engine.alignment_params,
                )
                cur_chi2_per_hit = (cur_res ** 2) / torch.clamp(
                    train_covariances, min=EPS_COVARIANCE
                )
                cur_chi2_per_hit = torch.where(
                    train_component_mask,
                    cur_chi2_per_hit,
                    torch.zeros_like(cur_chi2_per_hit),
                )
                layer_info = []
                for vl, ll in TARGET_LAYERS:
                    vmask = (train_hit_volume_ids == vl) & (train_hit_layer_ids == ll)
                    if vmask.any():
                        lchi2 = cur_chi2_per_hit[vmask].mean().item()
                        layer_info.append(f"v{vl}l{ll}={lchi2:.2f}")
                # EXPERIMENTAL: weight diagnostics only when the network is on.
                # Keeping them conditional preserves observability for a future
                # restart while costing the disabled path nothing -- and
                # hit_weights is None here, so detach()/normalize() would crash.
                weight_info = []
                if engine.use_hit_weights:
                    tw = hit_weights.detach()
                    twn = normalize_hit_weights(
                        hit_weights, train_hit_volume_ids, train_hit_layer_ids
                    ).detach()
                    weight_info.append(
                        f"W: mean={tw.mean():.3f} std={tw.std(unbiased=False):.3f} min={tw.min():.3f}"
                    )
                    for vl, ll in TARGET_LAYERS:
                        vmask = (train_hit_volume_ids == vl) & (train_hit_layer_ids == ll)
                        if vmask.any():
                            weight_info.append(
                                f"Wv{vl}l{ll}: raw={tw[vmask].mean():.3f} "
                                f"norm={twn[vmask].mean():.3f}"
                            )
                # The Epoch line itself is never conditional: the invariant
                # extractor and the acceptance criteria both depend on it.
                print(f"Epoch [{epoch:03d}/{config.epochs}] | LR: {current_lr:.6f} | "
                      f"Chi2: {chi2_loss.item():.4f} | L2: {l2_reg.item():.6f} | "
                      f"Total: {loss.item():.4f} | "
                      + " | ".join(layer_info + weight_info))

    predicted_deltas = engine.alignment_params.detach().cpu().numpy()

    predicted_delta_tensor = torch.tensor(predicted_deltas, dtype=torch.float32)

    # ── Per-layer residual diagnostics ──────────────────────────────────────
    # Before and after must share one residual definition.  The ROOT/KF residual
    # (l_x/y_hit - eLOC0/1_smt) and the geometric projection only agree
    # approximately at delta=0, so using the former as the baseline would fold
    # that discrepancy into the reported improvement.  Evaluate the same
    # projection at delta=0 instead.
    def _masked_residuals(deltas):
        residuals = compute_autograd_residuals(
            train_measured_local,
            train_hit_global_positions,
            train_hit_directions,
            train_hit_sensor_ids,
            initial_positions,
            initial_rotations,
            deltas,
        )
        return torch.where(
            train_component_mask, residuals, torch.zeros_like(residuals)
        ).numpy()

    with torch.no_grad():
        init_res_np = _masked_residuals(torch.zeros_like(predicted_delta_tensor))
        final_res = _masked_residuals(predicted_delta_tensor)
    vols_np  = train_hit_volume_ids.numpy()
    lays_np  = train_hit_layer_ids.numpy()

    target_metrics = print_per_layer_residual_diagnostics(
        init_res_np, final_res, vols_np, lays_np, TARGET_LAYERS,
        return_metrics=True, component_mask=train_component_mask,
    )

    # The decisive check vs TRUE misalignment.
    truth_metrics = validate_against_truth(
        transforms_list, predicted_deltas, config.nominal_file, TARGET_LAYERS
    )

    Path(ALIGNED_JSON_FILE).parent.mkdir(parents=True, exist_ok=True)
    update_geometry_to_json(
        transforms_list,
        predicted_deltas,
        ALIGNED_JSON_FILE,
        alignment_gain=config.alignment_gain,
    )

    print(f"Export successful! The aligned, complete geometry file has been saved to: {ALIGNED_JSON_FILE}")

    # Draw and save loss line chart.
    if config.make_plot and history_epochs:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(history_epochs, history_chi2_loss, label='Chi2 loss', color='firebrick', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss(Log Scale)', fontsize=12)
        plt.title("ML Alignment Engine Training Curve", fontsize=14, fontweight='bold')
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plot_filepath = str(Path(ALIGNED_JSON_FILE).parent / f"loss_curve_{Path(ALIGNED_JSON_FILE).stem}.png")
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"Training convergence plot saved: '{plot_filepath}'")

    return {
        "output_geom": ALIGNED_JSON_FILE,
        "target_metrics": target_metrics,
        "truth_metrics": truth_metrics,
        "predicted_deltas": predicted_deltas,
    }


def build_argument_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="ML alignment solver (Adam neural-network optimizer). "
                    "Reads current geometry + reconstruction residuals → writes updated geometry."
    )
    add_config_argument(p)
    p.add_argument("--input-geom", type=absolute_project_path, default=str(alignment_json_path("alignment_output", "misalignment_applied")),
                   help="Current geometry G_k (correction base; delta is composed onto it).")
    p.add_argument("--states", dest="states_root", type=absolute_project_path,
                   default=str(reconstruction_root_path("reconstruction_output", "misaligned", "trackstates_kf")),
                   help="trackstates ROOT from reconstruction performed with G_k.")
    p.add_argument("--summary", dest="summary_root", type=absolute_project_path,
                   default=str(reconstruction_root_path("reconstruction_output", "misaligned", "tracksummary_kf")),
                   help="tracksummary ROOT from reconstruction performed with G_k.")
    p.add_argument("--output-geom", type=absolute_project_path, default=str(alignment_json_path("alignment_output", "aligned_transforms", stem_prefix="ml")),
                   help="Updated geometry G_{k+1} = G_k compose delta.")
    p.add_argument("--nominal", dest="nominal_file", type=absolute_project_path, default=str(data_path("odd_transforms.json")),
                   help="Nominal/truth geometry for the convergence metric (|true|rms).")
    p.add_argument("--iter", dest="iter_label", default=None,
                   help="Iteration label for logging.")
    p.add_argument("--active-dof", default=None,
                   help="Comma-separated active DOF. Default: 0,1,3,4,5 for both track modes.")
    p.add_argument("--target-layers", default=target_layers_spec(),
                   help=f"Comma-separated target layers as volume:layer. Default: {target_layers_spec()}.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"Number of Adam training epochs. Default: {DEFAULT_EPOCHS}.")
    p.add_argument("--lambda-l2", type=float, default=DEFAULT_LAMBDA_L2,
                   help=f"L2 regularisation weight. Default: {DEFAULT_LAMBDA_L2}.")
    p.add_argument("--track-mode", choices=("surface", "free"), default="surface",
                   help="Track treatment: fixed smoothed surface solve or free joint helices.")
    p.add_argument("--joint-iters", type=int, default=JOINT_ITERS)
    p.add_argument(
        "--joint-solver",
        choices=("block-lm", "adam"),
        default=JOINT_SOLVER,
        help="Free-mode geometry solver; block-lm keeps non-target-refitted tracks fixed.",
    )
    p.add_argument("--lm-max-iters", type=int, default=LM_MAX_ITERS)
    p.add_argument("--lm-initial-damping", type=float, default=LM_INITIAL_DAMPING)
    p.add_argument("--fisher-eigen-floor", type=float, default=FISHER_EIGEN_FLOOR)
    p.add_argument("--lm-max-step-t", type=float, default=LM_MAX_STEP_T)
    p.add_argument("--lm-max-step-r", type=float, default=LM_MAX_STEP_R)
    p.add_argument("--joint-checkpoint-interval", type=int, default=JOINT_CHECKPOINT_INTERVAL,
                   help="Full-dataset objective evaluation interval for best checkpointing.")
    p.add_argument("--joint-patience", type=int, default=JOINT_PATIENCE,
                   help="Stop after this many iterations without full-objective improvement; 0 disables.")
    p.add_argument("--joint-min-delta", type=float, default=0.0,
                   help="Minimum full-objective improvement required to reset patience.")
    p.add_argument(
        "--joint-seed-mode",
        choices=("non-target-measurement",),
        default=JOINT_SEED_MODE,
        help="Free-helix initialization architecture (fixed to non-target measurement).",
    )
    p.add_argument("--joint-seed-min-hits", type=int, default=JOINT_SEED_MIN_HITS)
    p.add_argument("--joint-seed-min-span", type=float, default=JOINT_SEED_MIN_SPAN)
    p.add_argument("--joint-smt-seed-iters", type=int, default=JOINT_SMT_SEED_ITERS)
    p.add_argument(
        "--joint-measurement-refit-iters",
        type=int,
        default=JOINT_MEASUREMENT_REFIT_ITERS,
    )
    p.add_argument(
        "--measurement-refit-lr-position",
        type=float,
        default=MEASUREMENT_REFIT_LR_POSITION,
    )
    p.add_argument(
        "--measurement-refit-lr-direction",
        type=float,
        default=MEASUREMENT_REFIT_LR_DIRECTION,
    )
    p.add_argument(
        "--measurement-refit-lr-rho-scaled",
        type=float,
        default=MEASUREMENT_REFIT_LR_RHO_SCALED,
    )
    p.add_argument(
        "--joint-refit-solver",
        choices=JOINT_REFIT_SOLVERS,
        default=JOINT_REFIT_SOLVER,
        help="Solver for the joint initialization measurement refit.",
    )
    p.add_argument(
        "--no-joint-refit-stationarity",
        action="store_true",
        help="Skip the Newton-decrement measurement of the measurement refit.",
    )
    p.add_argument("--min-hits", type=int, default=8)
    p.add_argument("--max-tracks", type=int, default=0, help="0 keeps all selected tracks.")
    p.add_argument("--track-batch-size", type=int, default=0, help="0 uses full-batch.")
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--event-partition", choices=PARTITIONS, default=None)
    p.add_argument("--event-split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--gpu", action="store_true", help="Use CUDA float32; CPU uses float64.")
    p.add_argument("--rot-lr-scale", type=float, default=15.0)
    p.add_argument("--joint-lr-trans", type=float, default=JOINT_LR_TRANS)
    p.add_argument("--joint-lr-track", type=float, default=JOINT_LR_TRACK)
    p.add_argument("--joint-clamp-t", type=float, default=JOINT_CLAMP_T)
    p.add_argument("--joint-clamp-r", type=float, default=JOINT_CLAMP_R)
    p.add_argument("--nominal-full-geom", type=absolute_project_path, default=str(data_path("odd_transforms.json")),
                   help="Full nominal geometry used to anchor non-target hits in free mode.")
    p.add_argument("--freeze-tracks", action="store_true",
                   help="Freeze seeded helices and optimize geometry only in free mode.")
    p.add_argument(
        "--alignment-gain",
        type=float,
        default=1.0,
        help="Scale geometry corrections when writing output (surface polish uses 0.2).",
    )
    p.add_argument(
        "--output-summary",
        type=absolute_project_path,
        default=None,
        help="Optional compact JSON containing solver stages and metrics.",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip the loss-curve PNG.")
    return p


def alignment_config_from_args(a):
    return AlignmentRunConfig(
        input_geom=a.input_geom,
        states_root=a.states_root,
        summary_root=a.summary_root,
        output_geom=a.output_geom,
        nominal_file=a.nominal_file,
        iter_label=a.iter_label,
        make_plot=not a.no_plot,
        active_dof=a.active_dof,
        target_layers=a.target_layers,
        epochs=a.epochs,
        lambda_l2=a.lambda_l2,
        track_mode=a.track_mode,
        joint_iters=a.joint_iters,
        joint_solver=a.joint_solver,
        lm_max_iters=a.lm_max_iters,
        lm_initial_damping=a.lm_initial_damping,
        fisher_eigen_floor=a.fisher_eigen_floor,
        lm_max_step_t=a.lm_max_step_t,
        lm_max_step_r=a.lm_max_step_r,
        joint_checkpoint_interval=a.joint_checkpoint_interval,
        joint_patience=a.joint_patience,
        joint_min_delta=a.joint_min_delta,
        min_hits=a.min_hits,
        max_tracks=a.max_tracks,
        track_batch_size=a.track_batch_size,
        rng_seed=a.rng_seed,
        event_partition=a.event_partition,
        event_split_seed=a.event_split_seed,
        gpu=a.gpu,
        rot_lr_scale=a.rot_lr_scale,
        joint_lr_trans=a.joint_lr_trans,
        joint_lr_track=a.joint_lr_track,
        joint_clamp_t=a.joint_clamp_t,
        joint_clamp_r=a.joint_clamp_r,
        nominal_full_geom=a.nominal_full_geom,
        freeze_tracks=a.freeze_tracks,
        joint_seed_mode=a.joint_seed_mode,
        joint_seed_min_hits=a.joint_seed_min_hits,
        joint_seed_min_span=a.joint_seed_min_span,
        joint_smt_seed_iters=a.joint_smt_seed_iters,
        joint_measurement_refit_iters=a.joint_measurement_refit_iters,
        measurement_refit_lr_position=a.measurement_refit_lr_position,
        measurement_refit_lr_direction=a.measurement_refit_lr_direction,
        measurement_refit_lr_rho_scaled=a.measurement_refit_lr_rho_scaled,
        joint_refit_solver=a.joint_refit_solver,
        joint_refit_stationarity=not a.no_joint_refit_stationarity,
        alignment_gain=a.alignment_gain,
    )


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    alignment_result = run_alignment_pipeline(alignment_config_from_args(args))
    if args.output_summary is not None:
        write_alignment_summary(alignment_result, args.output_summary)
