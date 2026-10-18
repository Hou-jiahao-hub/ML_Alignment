"""Per-track padded hit tensors and the non-target fit mask."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .conventions import parse_target_layers

JOINT_SEED_MIN_HITS = 8
JOINT_SEED_MIN_SPAN = 300.0


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


def _non_target_fit_mask(batches):
    if batches.fit_mask is None:
        raise ValueError(
            "Joint initialization requires a non-target fit_mask; "
            "falling back to all hits is forbidden."
        )
    if (batches.fit_mask & batches.is_target).any():
        raise ValueError("Joint initialization fit_mask contains target-layer hits.")
    return batches.fit_mask
