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
try:
    from .alignment_diagnostics import (
        print_per_layer_residual_diagnostics,
        print_target_residual_stats,
        validate_against_truth,
    )
except ImportError:  # Direct script execution.
    from alignment_diagnostics import (
        print_per_layer_residual_diagnostics,
        print_target_residual_stats,
        validate_against_truth,
    )
try:
    from .helix_model import (
        CURVATURE_SCALE,
        MeasurementFitInputs,
        S_MAX,
        accept_newton_candidate,
        assemble_track_parameters as assemble_joint_track_parameters,
        closest_helix_points,
        fit_helix_points_gauss_newton,
        fit_helix_measurements_gauss_newton,
        helix_point,
        helix_tangent,
        intersect_plane,
        measurement_newton_decrement,
        measurement_stationarity_report,
        require_stationary_helix_point_fit,
        split_track_parameters as split_joint_track_parameters,
    )
except ImportError:  # Direct script execution.
    from helix_model import (
        CURVATURE_SCALE,
        MeasurementFitInputs,
        S_MAX,
        accept_newton_candidate,
        assemble_track_parameters as assemble_joint_track_parameters,
        closest_helix_points,
        fit_helix_points_gauss_newton,
        fit_helix_measurements_gauss_newton,
        helix_point,
        helix_tangent,
        intersect_plane,
        measurement_newton_decrement,
        measurement_stationarity_report,
        require_stationary_helix_point_fit,
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
#: Block-LM budget and per-iteration step caps.  A sensor can only travel
#: ``cap * accepted_steps``, so these three constants jointly decide whether a
#: misalignment is reachable at all.  The former values (20 iterations,
#: 1e-3 rad) were calibrated for a geometry that two surface warm starts had
#: already pre-corrected: against the raw injected 0.030 rad they need at least
#: 30 accepted steps, which 20 iterations cannot supply.  Measured on the
#: 20260901 joint-only runs, that stranded 14 of 2268 sensors at 1-6 capped
#: steps regardless of the iteration budget.  At 5e-3 rad the same run
#: reproduced the warm-start pipeline's geometry and its ACTS acceptance to
#: within 1e-7, and converged in 13 iterations.
LM_MAX_ITERS = 60
LM_INITIAL_DAMPING = 1e-3
FISHER_EIGEN_FLOOR = 1e-4
LM_MAX_STEP_T = 0.1
LM_MAX_STEP_R = 5e-3

#: Pre-registration for the block-LM convergence contract, kept separate from
#: ``JOINT_REFIT_PREREGISTRATION_VERSION`` above and from the Gate 4 block's
#: ``PREREGISTRATION_VERSION``: those govern the measurement refit and the
#: validation verdicts, neither of which changes here.
#: v1: the L1 audit -- per-sensor damping, retry after rejection, fixed
#: comparison set, and a stationarity/KKT stopping test.
#: v2: review hardening -- undefined truth comparisons fail closed, sensors
#: with hits but no valid intersections remain visible, the retry ladder keeps
#: its absolute scale after accepted-step damping reductions, and any least-
#: squares fallback is sticky rank-deficiency evidence.
BLOCK_LM_PREREGISTRATION_VERSION = 2

#: Stationarity threshold for one sensor block, normalized by component count:
#: a block is stationary when ``lambda2 / n_components <=`` this value, where
#: ``lambda2 = g_free^T H_free^-1 g_free`` is the Newton decrement restricted to
#: the DOFs the box constraints leave free.  ``standardized()`` returns
#: ``r / sigma``, so ``r^T r = chi2``, ``g = J^T r = grad F`` and ``H = J^T J``:
#: the gauge is ``F = 0.5 * chi2``, the same one ``NEWTON_DECREMENT_STATIONARY``
#: (validate_helix_model.py) was frozen in, and the value is carried over from
#: it.  The caliber differs -- that constant is per fitted track row, this one is
#: per measurement component -- so the two are frozen independently and neither
#: may be adjusted to make the other pass.
#: One deliberate divergence: ``newton_decrement`` regularizes before solving,
#: which can only shrink the decrement.  The check here adds no regularization
#: at all, because a decrement biased low is exactly how a non-stationary block
#: gets reported as converged.
#: This threshold is evidence of convergence, not evidence of accuracy.  A block
#: that meets it has no descent left to offer under its own quadratic model; it
#: says nothing about how close the geometry is to truth.
BLOCK_LM_DECREMENT_STATIONARY = 1.0e-6

#: How many times a rejected candidate may be retried at ten-fold damping within
#: one outer iteration.  From ``LM_INITIAL_DAMPING = 1e-3`` eight rungs reach
#: 1e5, where the step is ~1e-8 of the Gauss-Newton step and its direction has
#: collapsed onto steepest descent -- for a smooth objective that descends unless
#: the gradient already vanishes, which the stationarity test catches first.
#: Still no descent after eight rungs therefore falsifies smoothness (the
#: intersection solver's validity branches and its ``|s| <= 1200 mm`` cut) and is
#: worth reporting rather than spending more budget on.  Retries reuse the
#: Jacobian, so a full ladder costs a fraction of one extra block visit.
BLOCK_LM_MAX_DAMPING_RETRIES = 8
BLOCK_LM_MAX_DAMPING = 1.0e12
BLOCK_LM_MIN_DAMPING = 1.0e-12
#: Tolerance for "sitting on a box constraint".  ``clamp_`` makes the equality
#: exact, so this only absorbs the round trip through the candidate row.
BLOCK_LM_BOUND_ATOL = 1.0e-12

MEASUREMENT_REFIT_LR_POSITION = 1e-3
MEASUREMENT_REFIT_LR_DIRECTION = 1e-5
MEASUREMENT_REFIT_LR_RHO_SCALED = 1e-3
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
    # Variance before STRIP_Y_FLOOR; diagnostic only, never read by a solver.
    raw_covariances: torch.Tensor | None = None


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
    raw_covariance = (
        None
        if inputs.raw_covariances is None
        else torch.ones((n_tracks, max_hits, 2), dtype=inputs.raw_covariances.dtype)
    )
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
        if raw_covariance is not None:
            raw_covariance[track_index, :count] = inputs.raw_covariances[index]
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
        raw_covariances=raw_covariance,
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


def seed_helices(
    batches,
    n_iter=200,
    device="cpu",
    dtype=torch.float64,
    return_diagnostics=False,
):
    """Fit per-track helix warm starts to ACTS smoothed points with LM."""
    points = batches.g_smt.to(device=device, dtype=dtype)
    mask = _non_target_fit_mask(batches).to(device=device)
    n_tracks, _n_hits, _ = points.shape
    first_index = mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(n_tracks, device=device)
    first = points[rows, first_index]
    direction = batches.p_smt.to(device=device, dtype=dtype)[rows, first_index]
    transverse = torch.linalg.norm(direction[:, :2], dim=1).clamp(min=1e-6)
    psi = torch.atan2(direction[:, 1], direction[:, 0])
    tanl = direction[:, 2] / transverse
    initial = torch.cat(
        (
            first,
            psi[:, None],
            tanl[:, None],
            torch.zeros((n_tracks, 1), device=device, dtype=dtype),
        ),
        dim=1,
    )
    fitted, diagnostics = fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=n_iter
    )
    return (fitted, diagnostics) if return_diagnostics else fitted


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


def _measurement_fit_inputs_from_joint(batches, geometry, parameters):
    """Build the ACTS-independent single-helix measurement-fit tensors."""
    fit_mask = _non_target_fit_mask(batches)
    if not bool(fit_mask.any(dim=1).all()):
        raise ValueError("Every joint track needs at least one non-target hit.")
    device = parameters.device
    dtype = parameters.dtype
    sensor_idx = batches.sensor_idx.to(device=device)
    centers = geometry.centers.to(device=device, dtype=dtype)[sensor_idx]
    rotations = geometry.rotations.to(device=device, dtype=dtype)[sensor_idx]
    fit_mask = fit_mask.to(device=device)
    loc1_mask = batches.loc1_mask.to(device=device)
    component_mask = torch.stack((fit_mask, fit_mask & loc1_mask), dim=-1)
    return MeasurementFitInputs(
        measured=batches.measured_local.to(device=device, dtype=dtype),
        variance=batches.covariances.to(device=device, dtype=dtype),
        component_mask=component_mask,
        centers=centers,
        rotations=rotations,
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

    The shared helix model evaluates the unnormalised ``F = 0.5 * chi2`` gauge.
    ``_measurement_refit_objective`` divides by the component count, which would
    scale every decrement and make the pre-registered threshold meaningless.

    Runs in float64 on the CPU by default: a decrement compared against 1e-6 is
    below float32 resolution.
    """
    measured = _measurement_fit_inputs_from_joint(batches, geometry, parameters)
    decrement = measurement_newton_decrement(
        parameters, measured, device=device, dtype=dtype
    )
    return measurement_stationarity_report(decrement)


def _load_alignment_experiments():
    try:
        from . import alignment_experiments
    except ImportError:  # Direct script execution.
        import alignment_experiments
    return alignment_experiments


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
    """Load the historical first-order refit only when explicitly requested."""
    import sys

    experiments = _load_alignment_experiments()
    return experiments.refit_helices_adam(
        sys.modules[__name__],
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


def _refit_helices_gauss_newton(batches, geometry, initial, device):
    """Solve the same objective, and refuse to return a solution that is not one.

    The production solver lives in ``helix_model`` and has no dependency on the
    holdout-validation or broken-helix experiment entry points.

    Always float64.  The pre-registered stationarity threshold is 1e-6, which is
    below float32 resolution, and the joint pipeline runs float32 whenever it runs
    on the GPU.  The caller casts the result back to its own dtype.
    """
    measured = _measurement_fit_inputs_from_joint(batches, geometry, initial)
    fitted, solver_stats = fit_helix_measurements_gauss_newton(
        initial,
        measured,
        device=device,
        dtype=torch.float64,
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
    one reads the decrement the solver computes for itself.  On the float64 CPU
    Gauss-Newton path the refit figure is that same gate report, reused.
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
        # The Gauss-Newton gate already evaluated the same float64 CPU decrement
        # on these exact parameters; recomputing it costs a full Jacobian pass.
        # Any cast to float32 or another device changes the parameters, so only
        # then is the refit report measured again.
        reuse_gate = (
            solver == JOINT_REFIT_SOLVER_GAUSS_NEWTON
            and fitted.dtype == torch.float64
            and fitted.device.type == "cpu"
        )
        stats["stationarity"] = {
            "seed": measure_joint_refit_stationarity(batches, geometry, initial),
            "refit": (
                dict(stats["stationarity_gate"])
                if reuse_gate
                else measure_joint_refit_stationarity(batches, geometry, fitted)
            ),
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


def _block_lm_fixed_chi2(residuals, covariance, fixed_mask):
    """Weighted residual sum of squares on a *fixed* comparison set.

    The acceptance test has to hold the component set still.  Scoring a
    candidate on its own mask lets a step that kills the hardest intersections
    look like an improvement, because the mean is taken over whatever survived.
    """
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    return torch.sum(torch.where(fixed_mask, weighted, torch.zeros_like(weighted)))


def _block_lm_retry_damping(damping, initial_damping, retry_number):
    """Raise damping on a retry without losing the pre-registered absolute ladder.

    Accepted steps may have halved a sensor's damping as far as
    ``BLOCK_LM_MIN_DAMPING``.  The retry number therefore anchors the ladder at
    the production initial damping instead of merely multiplying that tiny
    carried value eight times.
    """
    anchor = max(float(initial_damping), LM_INITIAL_DAMPING)
    return min(
        max(float(damping) * 10.0, anchor * (10.0 ** int(retry_number))),
        BLOCK_LM_MAX_DAMPING,
    )


def _block_lm_free_dof_mask(values, gradient, lower, upper, atol=BLOCK_LM_BOUND_ATOL):
    """Active-set mask for the box constraints: which DOFs may still move.

    From the projected gradient ``P = x - clip(x - g, l, u)`` with a descent
    direction of ``-g``: at the upper bound ``g < 0`` wants to push further up
    and is infeasible, at the lower bound ``g > 0`` wants to push further down.
    Those DOFs are frozen; everything else, an interior point or a bound whose
    gradient points back inside, stays free.  ``g == 0`` never freezes -- it
    contributes nothing to the decrement and is better left visible to the rank
    check.

    ``values``/``lower``/``upper`` are physical, ``gradient`` is normalized.
    Mixing them is safe because only the sign of the gradient is read, and
    ``physical_scales`` is strictly positive.
    """
    at_upper = values >= upper - atol
    at_lower = values <= lower + atol
    return ~((at_upper & (gradient < 0.0)) | (at_lower & (gradient > 0.0)))


def _block_lm_decrement(fisher, gradient, free_mask, eigen_floor_ratio):
    """Newton decrement ``g^T H^-1 g`` on the free set, plus rank diagnostics.

    Solved through the eigendecomposition rather than by regularizing and
    calling ``solve``: any floor added to ``H`` can only shrink the decrement
    (``(H + eI)^-1 <= H^-1``), which would turn "not converged" into "converged"
    -- the failure this contract exists to remove.

    When a direction is weaker than the Fisher floor its curvature is supplied
    by the floor, not by the data, so it is dropped from the sum.  The remaining
    value is then a *lower bound* on the true decrement and cannot be used to
    claim convergence; ``lambda2_is_lower_bound`` says so, and
    ``null_gradient_max`` reports how much gradient is left sitting in the
    dropped directions.
    """
    free_fisher = fisher[free_mask][:, free_mask]
    free_gradient = gradient[free_mask]
    n_free = int(free_gradient.numel())
    if n_free == 0:
        # Every DOF sits on a bound with its gradient pointing outwards: the
        # projected gradient is zero, so this is a KKT point of the box problem.
        return {
            "lambda2": 0.0,
            "n_free": 0,
            "rank_deficient": False,
            "lambda2_is_lower_bound": False,
            "min_eigenvalue_free": float("nan"),
            "max_eigenvalue_free": float("nan"),
            "eigen_floor_free": float("nan"),
            "null_gradient_max": 0.0,
        }
    symmetric = 0.5 * (free_fisher + free_fisher.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    minimum = float(eigenvalues[0].detach().cpu())
    maximum = float(eigenvalues[-1].detach().cpu())
    floor = max(maximum, 0.0) * float(eigen_floor_ratio)
    weak = eigenvalues <= floor
    projection = eigenvectors.transpose(-1, -2) @ free_gradient
    strong = ~weak
    lambda2 = (
        float((projection[strong].square() / eigenvalues[strong]).sum().detach().cpu())
        if bool(strong.any())
        else 0.0
    )
    return {
        "lambda2": lambda2,
        "n_free": n_free,
        "rank_deficient": bool(weak.any()) or minimum <= 0.0,
        "lambda2_is_lower_bound": bool(weak.any()),
        "min_eigenvalue_free": minimum,
        "max_eigenvalue_free": maximum,
        "eigen_floor_free": floor,
        "null_gradient_max": (
            float(projection[weak].abs().max().detach().cpu()) if bool(weak.any()) else 0.0
        ),
    }


#: The four terminal states a sensor block can end a block-LM run in, plus the
#: internal ``no_data`` case for sensors that carry no target hit at all.  A run
#: reports counts and fractions over these; per-sensor detail stays in memory.
BLOCK_LM_STATUS_CONVERGED = "converged"
BLOCK_LM_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
BLOCK_LM_STATUS_RANK_DEFICIENT = "rank_deficient"
BLOCK_LM_STATUS_INTERSECTION_FAILED = "intersection_failed"
BLOCK_LM_STATUS_NO_DATA = "no_data"
BLOCK_LM_TERMINAL_STATUSES = (
    BLOCK_LM_STATUS_CONVERGED,
    BLOCK_LM_STATUS_BUDGET_EXHAUSTED,
    BLOCK_LM_STATUS_RANK_DEFICIENT,
    BLOCK_LM_STATUS_INTERSECTION_FAILED,
)


def _block_lm_new_record(sensor_index, config):
    """Blank audit record for one sensor block."""
    return {
        "sensor_index": int(sensor_index),
        "evaluations": 0,
        "accepted_steps": 0,
        "accepted_component_deltas": [],
        "retries": 0,
        "clipped_steps": 0,
        "lstsq_fallbacks": 0,
        "component_loss_rejections": 0,
        "invalid_fraction_rejections": 0,
        "had_component_loss": False,
        "had_invalid_rejection": False,
        "final_damping": float(config.lm_initial_damping),
        "max_damping_seen": float(config.lm_initial_damping),
        "min_damping_seen": float(config.lm_initial_damping),
        "exhausted_reason": "outer_iters",
        "retired_at_iteration": None,
        "n_components_start": None,
        "n_components_end": None,
        "lambda2": None,
        "lambda2_per_component": None,
        "lambda2_is_lower_bound": False,
        "n_free": None,
        "frozen_dof": 0,
        "boundary": False,
        "rank_deficient": False,
        "stationary": False,
        "min_eigenvalue_free": None,
        "null_gradient_max": None,
        "chi2_fixed": None,
    }


def _block_lm_linearized_block(
    tracks, current, sensor_index, batches, geometry, active_index, physical_scales, fixed_mask
):
    """One Gauss-Newton linearisation of a sensor block on a fixed component set.

    ``standardized`` returns ``r / sigma`` restricted to ``fixed_mask``, so
    ``r^T r`` is the chi2 of that set, ``gradient`` is grad F and ``fisher`` is
    the Gauss-Newton Hessian, both in the ``F = 0.5 * chi2`` gauge and scaled to
    dimensionless units by ``physical_scales``.

    Damping does not enter here, which is what makes retrying a rejected step
    cheap: the whole ladder reuses one Jacobian.
    """

    def standardized(active_values):
        row = current.scatter(0, active_index, active_values)
        values, variances, _mask, _intersection_valid = _sensor_residual_data(
            tracks, row, sensor_index, batches, geometry
        )
        return (values / variances.clamp_min(EPS_COVARIANCE).sqrt())[fixed_mask]

    active_values = current[active_index].detach().requires_grad_(True)
    residual_vector = standardized(active_values).detach()
    jacobian = torch.autograd.functional.jacobian(
        standardized, active_values, vectorize=True
    ).detach()
    normalized_jacobian = jacobian * physical_scales
    fisher = normalized_jacobian.transpose(0, 1) @ normalized_jacobian
    gradient = normalized_jacobian.transpose(0, 1) @ residual_vector
    return residual_vector, fisher, gradient


def _block_lm_finalize_record(record):
    """Stamp the terminal state onto a block record once the run has stopped."""
    record["status"] = _block_lm_block_status(record)
    return record


def _block_lm_block_status(record):
    """Terminal state of one block, in the pre-registered priority order.

    ``intersection_failed`` outranks ``rank_deficient`` because a candidate that
    cannot keep its own intersections says nothing about curvature, and
    ``rank_deficient`` outranks ``converged`` because a decrement computed with
    weak directions dropped is a lower bound -- passing a threshold with a
    statistic biased low is precisely the false convergence this contract
    removes.
    """
    if record.get("n_components_end") is None:
        return BLOCK_LM_STATUS_NO_DATA
    if record.get("had_component_loss") or record.get("had_invalid_rejection"):
        return BLOCK_LM_STATUS_INTERSECTION_FAILED
    if record.get("rank_deficient") or record.get("lstsq_fallbacks", 0) > 0:
        return BLOCK_LM_STATUS_RANK_DEFICIENT
    if record.get("stationary"):
        return BLOCK_LM_STATUS_CONVERGED
    return BLOCK_LM_STATUS_BUDGET_EXHAUSTED


def _block_lm_run_summary(blocks, arm, stop_reason, config):
    """Run-level roll-up of the block terminal states.

    Deliberately scalar: with a couple of thousand target sensors, per-sensor
    rows would grow the summary by an order of magnitude while saying nothing
    the counts and worst cases do not.  ``block_lm_blocks`` keeps the detail in
    memory for tests and ad-hoc analysis.
    """
    with_data = [item for item in blocks if item["status"] != BLOCK_LM_STATUS_NO_DATA]
    denominator = len(with_data)
    counts = {name: 0 for name in BLOCK_LM_TERMINAL_STATUSES}
    for item in with_data:
        counts[item["status"]] += 1
    # Boundary blocks stop at a KKT point with lambda2 identically zero; letting
    # them into the quantiles would drag the distribution down for free.
    interior = [
        item["lambda2_per_component"]
        for item in with_data
        if not item["boundary"] and item["lambda2_per_component"] is not None
    ]
    finite = sorted(value for value in interior if math.isfinite(value))

    def quantile(fraction):
        if not finite:
            return float("nan")
        position = fraction * (len(finite) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        weight = position - low
        return finite[low] * (1.0 - weight) + finite[high] * weight

    accepted_deltas = [
        value for item in with_data for value in item["accepted_component_deltas"]
    ]
    span_deltas = [item["n_components_end"] - item["n_components_start"] for item in with_data]
    damping_high = [item["max_damping_seen"] for item in with_data]
    damping_low = [item["min_damping_seen"] for item in with_data]
    eigenvalues = [
        item["min_eigenvalue_free"]
        for item in with_data
        if item["min_eigenvalue_free"] is not None
        and math.isfinite(item["min_eigenvalue_free"])
    ]
    null_gradients = [
        item["null_gradient_max"]
        for item in with_data
        if item["null_gradient_max"] is not None
        and math.isfinite(item["null_gradient_max"])
    ]
    clipped = sum(item["clipped_steps"] for item in with_data)
    evaluations = sum(item["evaluations"] for item in with_data)
    return {
        "claim_scope": "convergence-contract-only",
        "arm": arm,
        "preregistration_version": BLOCK_LM_PREREGISTRATION_VERSION,
        "stationary_threshold": BLOCK_LM_DECREMENT_STATIONARY,
        "max_damping_retries": BLOCK_LM_MAX_DAMPING_RETRIES,
        "stop_reason": stop_reason,
        "blocks_total": len(blocks),
        "blocks_without_data": len(blocks) - denominator,
        "status_counts": counts,
        "status_fractions": {
            name: (counts[name] / denominator if denominator else 0.0)
            for name in BLOCK_LM_TERMINAL_STATUSES
        },
        "converged_detail": {
            "interior": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_CONVERGED and not item["boundary"]
            ),
            "boundary": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_CONVERGED and item["boundary"]
            ),
        },
        "budget_exhausted_detail": {
            "outer_iters": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_BUDGET_EXHAUSTED
                and item["exhausted_reason"] == "outer_iters"
            ),
            "damping_retries": sum(
                1
                for item in with_data
                if item["status"] == BLOCK_LM_STATUS_BUDGET_EXHAUSTED
                and item["exhausted_reason"] == "damping_retries"
            ),
        },
        "decrement_per_component": {
            "interior_only": True,
            "n_blocks": len(interior),
            "n_non_finite": len(interior) - len(finite),
            "non_stationary_block_fraction": (
                sum(1 for value in interior if not (value <= BLOCK_LM_DECREMENT_STATIONARY))
                / len(interior)
                if interior
                else 0.0
            ),
            "p50": quantile(0.50),
            "p90": quantile(0.90),
            "p99": quantile(0.99),
            "max": finite[-1] if finite else float("nan"),
        },
        "component_guard": {
            "accepted_steps": len(accepted_deltas),
            "min_components_delta_accepted": min(accepted_deltas) if accepted_deltas else 0,
            "min_components_end_minus_start": min(span_deltas) if span_deltas else 0,
            "component_loss_rejections": sum(
                item["component_loss_rejections"] for item in with_data
            ),
            "invalid_fraction_rejections": sum(
                item["invalid_fraction_rejections"] for item in with_data
            ),
        },
        "damping": {
            "blocks_with_retry": sum(1 for item in with_data if item["retries"] > 0),
            "retries_total": sum(item["retries"] for item in with_data),
            "max_damping_reached": (
                max(damping_high) if damping_high else float(config.lm_initial_damping)
            ),
            "min_damping_reached": (
                min(damping_low) if damping_low else float(config.lm_initial_damping)
            ),
        },
        "trust_region": {
            "clipped_steps": clipped,
            "clipped_step_fraction": (clipped / evaluations if evaluations else 0.0),
        },
        "boundary": {
            "blocks_with_active_bound": sum(1 for item in with_data if item["boundary"]),
            "free_set_empty_blocks": sum(1 for item in with_data if item["n_free"] == 0),
            "frozen_dof_total": sum(item["frozen_dof"] for item in with_data),
        },
        "rank": {
            "lstsq_fallbacks": sum(item["lstsq_fallbacks"] for item in with_data),
            "min_eigenvalue_min": min(eigenvalues) if eigenvalues else float("nan"),
            "floor_limited_blocks": sum(
                1 for item in with_data if item["lambda2_is_lower_bound"]
            ),
            "null_gradient_max": max(null_gradients) if null_gradients else float("nan"),
        },
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
    """Fit target geometry with fixed non-target-refitted helices using block LM.

    Each target sensor is its own block: ``_sensor_residual_data`` selects hits
    by sensor id alone and the helices are frozen, so a block's residuals depend
    only on its own alignment row.  The blocks are therefore exactly decoupled,
    which is why one that reaches a stationary point can be retired for good.

    A block stops for one of four reasons, reported rather than collapsed into a
    flag: it is stationary (a free-set Newton decrement per component under
    ``BLOCK_LM_DECREMENT_STATIONARY``, or a KKT point of the box constraints),
    the iteration or damping budget ran out, the Fisher matrix is rank deficient
    so the decrement is only a lower bound, or no damping keeps the candidate's
    intersections alive.

    A candidate is accepted only if it lowers the chi2 **on the component set
    the current point has**, and only if it keeps every one of those components.
    Scoring each side on its own mask is what lets a step that kills the hardest
    intersections buy a lower mean with less information.  One consequence worth
    knowing: since accepted candidates may add components, the fixed set grows,
    and ``history`` compares objectives with different numbers of terms in them.
    It is not guaranteed monotone, and a rise in it is not by itself a bug.

    This fixes the convergence contract.  It is not a physics change: if the
    geometry it produces matches the baseline, the claim is that the contract
    holds, not that accuracy improved.
    """
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
    stopped_early = False
    target_sensors = torch.nonzero(geometry.target_mask, as_tuple=False).flatten().tolist()
    per_dof_limit = torch.tensor(
        [config.lm_max_step_t if index < 3 else config.lm_max_step_r for index in active],
        device=device,
        dtype=dtype,
    )
    bound_limit = torch.tensor(
        [config.joint_clamp_t if index < 3 else config.joint_clamp_r for index in active],
        device=device,
        dtype=dtype,
    )
    audit = {index: _block_lm_new_record(index, config) for index in target_sensors}
    # One damping per sensor, carried across outer iterations.  A shared scalar
    # lets a sensor that accepted a step halve the damping of a sensor that did
    # not, which is the opposite of what a trust region is for.
    damping_by_sensor = {index: float(config.lm_initial_damping) for index in target_sensors}
    active_sensors = list(target_sensors)
    stop_reason = "iteration_budget"
    for iteration in range(config.lm_max_iters):
        if not active_sensors:
            stop_reason = "all_blocks_terminal"
            break
        still_active = []
        for sensor_index in active_sensors:
            record = audit[sensor_index]
            current = alignment[sensor_index].detach()
            current_data = _sensor_residual_data(
                tracks, current, sensor_index, batches, geometry
            )
            if current_data is None:
                continue
            residuals, covariance, component_mask, current_valid = current_data
            # The comparison set for this visit.  Because an accepted candidate
            # must keep every component the current point has, this can only
            # grow -- which is also why ``history`` is not guaranteed monotone:
            # a larger set is a different objective, with more terms in it.
            fixed_mask = component_mask.detach()
            if not fixed_mask.any():
                # The sensor has target hits (otherwise current_data is None),
                # but none currently intersect.  This is an intersection
                # failure, not absence of data, and must remain in the status
                # denominator.
                record["n_components_start"] = 0
                record["n_components_end"] = 0
                record["had_invalid_rejection"] = True
                record["retired_at_iteration"] = iteration
                continue
            n_components = int(fixed_mask.sum())
            if record["n_components_start"] is None:
                record["n_components_start"] = n_components
            record["n_components_end"] = n_components
            current_invalid = 1.0 - float(current_valid.float().mean().detach().cpu())

            residual_vector, fisher, gradient = _block_lm_linearized_block(
                tracks, current, sensor_index, batches, geometry,
                active_index, physical_scales, fixed_mask,
            )
            chi2_before = float(residual_vector.square().sum().detach().cpu())

            free_mask = _block_lm_free_dof_mask(
                current[active_index].detach(), gradient, -bound_limit, bound_limit
            )
            report = _block_lm_decrement(
                fisher, gradient, free_mask, config.fisher_eigen_floor
            )
            n_free = report["n_free"]
            per_component = (
                report["lambda2"] / n_components if n_components else float("nan")
            )
            # An empty free set means every active DOF sits on a bound with its
            # gradient pointing outwards: the projected gradient vanishes, so
            # this is a KKT point of the box-constrained problem.
            stationary = n_free == 0 or per_component <= BLOCK_LM_DECREMENT_STATIONARY
            record.update({
                "lambda2": report["lambda2"],
                "lambda2_per_component": per_component,
                "lambda2_is_lower_bound": report["lambda2_is_lower_bound"],
                "n_free": n_free,
                "frozen_dof": len(active) - n_free,
                "boundary": n_free < len(active),
                "rank_deficient": report["rank_deficient"],
                "stationary": stationary,
                "min_eigenvalue_free": report["min_eigenvalue_free"],
                "null_gradient_max": report["null_gradient_max"],
            })
            if report["rank_deficient"]:
                # Reported, not retired.  The decrement is a lower bound here,
                # so it cannot certify convergence -- but stopping these blocks
                # outright would change the geometry wholesale and make "the
                # contract was fixed" indistinguishable from "the answer moved".
                pass
            elif stationary:
                record["retired_at_iteration"] = iteration
                continue

            accepted = False
            ladder_lost = False
            ladder_invalid = False
            for attempt in range(1 + BLOCK_LM_MAX_DAMPING_RETRIES):
                damping = damping_by_sensor[sensor_index]
                regularized, diagnostics = regularize_fisher_matrix(
                    fisher,
                    damping=damping,
                    eigen_floor_ratio=config.fisher_eigen_floor,
                )
                lstsq_fallback = False
                try:
                    normalized_step = -torch.linalg.solve(regularized, gradient)
                except RuntimeError:
                    lstsq_fallback = True
                    normalized_step = -torch.linalg.lstsq(
                        regularized, gradient[:, None]
                    ).solution[:, 0]
                physical_step = normalized_step * physical_scales
                # The step cap is a trust region, not a feasible-set boundary:
                # it never enters the KKT test, but a capped step is no longer
                # the LM step, so it is counted separately.
                step_clipped = bool((physical_step.abs() > per_dof_limit).any())
                physical_step = torch.maximum(
                    torch.minimum(physical_step, per_dof_limit), -per_dof_limit
                )
                candidate = current.clone()
                candidate[active_index] += physical_step
                candidate[:3].clamp_(-config.joint_clamp_t, config.joint_clamp_t)
                candidate[3:].clamp_(-config.joint_clamp_r, config.joint_clamp_r)
                (
                    candidate_residuals,
                    candidate_covariance,
                    candidate_mask,
                    candidate_valid,
                ) = _sensor_residual_data(
                    tracks, candidate, sensor_index, batches, geometry
                )
                # Both sides scored on the same set.  Scoring the candidate on
                # its own mask is how a step that kills the hardest
                # intersections buys a lower mean with less information.
                chi2_after = _block_lm_fixed_chi2(
                    candidate_residuals, candidate_covariance, fixed_mask
                )
                lost = bool((fixed_mask & ~candidate_mask).any())
                sensor_invalid = 1.0 - float(
                    candidate_valid.float().mean().detach().cpu()
                )
                # Anchored on where this sensor already is: the run-level limit
                # is checked globally at entry, so a sensor that starts out
                # degraded would otherwise be frozen for reasons that have
                # nothing to do with the step under test.
                invalid_ok = sensor_invalid <= max(
                    current_invalid, config.joint_max_invalid_fraction
                )
                ok = bool(
                    (not lost)
                    and invalid_ok
                    and torch.isfinite(chi2_after)
                    and float(chi2_after) < chi2_before
                )
                reject_reason = None
                if not ok:
                    reject_reason = (
                        "component_loss" if lost
                        else "invalid_fraction" if not invalid_ok
                        else "no_decrease"
                    )
                    ladder_lost = ladder_lost or lost
                    ladder_invalid = ladder_invalid or not invalid_ok
                    if lost:
                        record["component_loss_rejections"] += 1
                    if not invalid_ok:
                        record["invalid_fraction_rejections"] += 1
                record["evaluations"] += 1
                record["clipped_steps"] += int(step_clipped)
                record["lstsq_fallbacks"] += int(lstsq_fallback)
                record["final_damping"] = float(damping)
                record["max_damping_seen"] = max(record["max_damping_seen"], float(damping))
                record["min_damping_seen"] = min(record["min_damping_seen"], float(damping))
                diagnostics.update({
                    "iteration": iteration,
                    "sensor_index": int(sensor_index),
                    "accepted": ok,
                    "attempt": attempt,
                    "damping": float(damping),
                    "step_rms": float(
                        torch.sqrt(torch.mean(physical_step.square())).detach().cpu()
                    ),
                    "step_max": float(physical_step.abs().max().detach().cpu()),
                    "step_clipped": step_clipped,
                    "lstsq_fallback": lstsq_fallback,
                    "gradient_norm": float(gradient.norm().detach().cpu()),
                    "lambda2": report["lambda2"],
                    "lambda2_per_component": per_component,
                    "lambda2_is_lower_bound": report["lambda2_is_lower_bound"],
                    "n_free": n_free,
                    "rank_deficient": report["rank_deficient"],
                    "null_gradient_max": report["null_gradient_max"],
                    "chi2_fixed_before": chi2_before,
                    "chi2_fixed_after": float(chi2_after.detach().cpu()),
                    "n_components_fixed": n_components,
                    "components_before": n_components,
                    "components_after": int(candidate_mask.sum()),
                    "component_loss": lost,
                    "reject_reason": reject_reason,
                    "sensor_invalid_fraction": sensor_invalid,
                    "loss_before": chi2_before / n_components,
                    "loss_after": float(chi2_after.detach().cpu()) / n_components,
                })
                fisher_diagnostics.append(diagnostics)
                if ok:
                    alignment[sensor_index] = candidate
                    record["accepted_steps"] += 1
                    record["accepted_component_deltas"].append(
                        int(candidate_mask.sum()) - n_components
                    )
                    damping_by_sensor[sensor_index] = max(
                        damping * 0.5, BLOCK_LM_MIN_DAMPING
                    )
                    accepted = True
                    break
                if damping >= BLOCK_LM_MAX_DAMPING:
                    break
                # Written back, so the raised damping is used by the next solve
                # whether that is this ladder's retry or the next iteration.
                record["retries"] += 1
                damping_by_sensor[sensor_index] = _block_lm_retry_damping(
                    damping,
                    initial_damping=config.lm_initial_damping,
                    retry_number=attempt + 1,
                )
            if accepted:
                still_active.append(sensor_index)
            else:
                # No feasible descent step at this point under any damping.
                record["had_component_loss"] = ladder_lost
                record["had_invalid_rejection"] = ladder_invalid
                if not (ladder_lost or ladder_invalid):
                    record["exhausted_reason"] = "damping_retries"
                record["retired_at_iteration"] = iteration
        active_sensors = still_active
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
    stopped_early = stop_reason == "all_blocks_terminal"
    blocks = [_block_lm_finalize_record(audit[index]) for index in target_sensors]
    block_lm_summary = _block_lm_run_summary(blocks, "contract-fixed", stop_reason, config)
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
        "stop_reason": stop_reason,
        "initial_total_loss": float(initial_loss.detach().cpu()),
        "initial_invalid_fraction": initial_invalid,
        "invalid_fraction": final_invalid,
        "invalid_history": invalid_history,
        "fisher_diagnostics": fisher_diagnostics,
        "block_lm_summary": block_lm_summary,
        "block_lm_blocks": blocks,
    }


def select_joint_solver(config):
    """The free-mode geometry solver named by ``config.joint_solver``."""
    if config.joint_solver == "block-lm":
        return run_block_lm_alignment
    if config.joint_solver == "schur-lm":
        from joint_alignment_schur import run_schur_lm_alignment

        return run_schur_lm_alignment
    return run_joint_alignment


def run_joint_alignment(
    batches,
    geometry,
    config,
    active_dof=DEFAULT_ACTIVE_DOF,
    initial_track_params=None,
):
    """Lazily run the historical joint Adam experiment."""
    import sys

    return _load_alignment_experiments().run_joint_alignment(
        sys.modules[__name__],
        batches,
        geometry,
        config,
        active_dof,
        initial_track_params,
    )


class AlignmentEngine(nn.Module):
    """Production surface-alignment parameters with unweighted residuals."""

    def __new__(cls, num_sensors, use_hit_weights=False):
        if use_hit_weights:
            import sys

            experiments = _load_alignment_experiments()
            return experiments.HitQualityAlignmentEngine(
                sys.modules[__name__], num_sensors
            )
        return super().__new__(cls)

    def __init__(self, num_sensors, use_hit_weights=False):
        super().__init__()
        self.num_sensors = num_sensors
        self.use_hit_weights = False
        self.alignment_params = nn.Parameter(torch.zeros(num_sensors, 6))

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
        del track_params, track_chi2_ndf
        residuals = compute_autograd_residuals(
            measured_local,
            hit_global_positions,
            hit_directions,
            hit_sensor_ids,
            sensor_center_positions,
            sensor_rotations,
            self.alignment_params,
        )
        return residuals, None


def normalize_hit_weights(hit_weights, hit_volume_ids=None, hit_layer_ids=None):
    """Compatibility entry point for the historical weighting experiment."""
    return _load_alignment_experiments().normalize_hit_weights(
        hit_weights, hit_volume_ids, hit_layer_ids
    )


def physics_informed_residual_loss(
    residuals,
    covariances,
    hit_weights,
    component_mask=None,
    hit_volume_ids=None,
    hit_layer_ids=None,
):
    """Compute production chi2, loading experimental weighting only on request."""
    if hit_weights is not None:
        return _load_alignment_experiments().weighted_residual_loss(
            residuals,
            covariances,
            hit_weights,
            EPS_COVARIANCE,
            component_mask=component_mask,
            hit_volume_ids=hit_volume_ids,
            hit_layer_ids=hit_layer_ids,
        )
    raw_chi2 = residuals.square() / torch.clamp(
        covariances, min=EPS_COVARIANCE
    )
    if component_mask is None:
        return torch.mean(raw_chi2)
    return torch.sum(
        torch.where(component_mask, raw_chi2, torch.zeros_like(raw_chi2))
    ) / component_mask.sum().clamp(min=1)


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
    seeded_parameters, seed_solver_stats = seed_helices(
        batches,
        n_iter=config.joint_smt_seed_iters,
        device="cuda" if config.gpu and torch.cuda.is_available() else "cpu",
        dtype=torch.float32 if config.gpu and torch.cuda.is_available() else torch.float64,
        return_diagnostics=True,
    )
    smt_stats = {
        **smt_seed_statistics(batches, seeded_parameters),
        "solver": seed_solver_stats["solver"],
        "convergence": {
            key: value for key, value in seed_solver_stats.items() if key != "solver"
        },
    }
    require_stationary_helix_point_fit(seed_solver_stats)
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
    solver = select_joint_solver(config)
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
        "stop_reason",
        "block_lm_summary",
        "schur_lm_summary",
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
        accepted_steps = sum(bool(item.get("accepted")) for item in fisher)
        summary["fisher_summary"] = {
            "evaluations": len(fisher),
            "accepted_steps": accepted_steps,
            "rejected_steps": len(fisher) - accepted_steps,
            "damping_retries": sum(int(item.get("attempt", 0) or 0) for item in fisher),
            "lstsq_fallbacks": sum(bool(item.get("lstsq_fallback")) for item in fisher),
            "clipped_steps": sum(bool(item.get("step_clipped")) for item in fisher),
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
        choices=("block-lm", "schur-lm", "adam"),
        default=JOINT_SOLVER,
        help=(
            "Free-mode geometry solver; block-lm keeps non-target-refitted tracks fixed, "
            "schur-lm fits geometry and tracks jointly (Schur-reduced LM, CPU float64, "
            "no per-DOF step caps)."
        ),
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


def __getattr__(name):
    """Lazily preserve imports of historical experiment-only helpers."""
    if name in {
        "HIT_QUALITY_HIDDEN_ONE",
        "HIT_QUALITY_HIDDEN_TWO",
        "HIT_QUALITY_INPUT_SIZE",
        "HIT_QUALITY_OUTPUT_SIZE",
        "HIT_WEIGHT_FLOOR",
        "JointBestCheckpoint",
        "joint_checkpoint_score",
    }:
        value = getattr(_load_alignment_experiments(), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    alignment_result = run_alignment_pipeline(alignment_config_from_args(args))
    if args.output_summary is not None:
        write_alignment_summary(alignment_result, args.output_summary)
