"""Helix-to-surface local residuals on the free-mode working geometry."""

from __future__ import annotations

import torch

from .conventions import EPS_COVARIANCE, parse_target_layers
from .helix.trajectory import intersect_plane
from .rotations import apply_alignment_torch


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


def _joint_forward(track_parameters, alignment, batches, geometry, track_indices=None):
    if track_indices is None:
        track_indices = torch.arange(len(track_parameters))
    measured = batches.measured_local.to(track_parameters.dtype)[track_indices]
    covariance = batches.covariances.to(track_parameters.dtype)[track_indices]
    sensor_idx = batches.sensor_idx[track_indices]
    hit_mask = batches.hit_mask[track_indices]
    loc1_mask = batches.loc1_mask[track_indices]
    n_hits = measured.shape[1]
    tracks = track_parameters[track_indices, None, :].expand(-1, n_hits, -1)
    centers0 = geometry.centers.to(track_parameters.dtype)
    rotations0 = geometry.rotations.to(track_parameters.dtype)
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


def _sensor_residual_data(
    track_parameters,
    alignment_row,
    sensor_index,
    batches,
    geometry,
):
    dtype = track_parameters.dtype
    hit_selection = (
        batches.hit_mask
        & batches.is_target
        & (batches.sensor_idx == int(sensor_index))
    )
    track_indices, hit_indices = torch.nonzero(hit_selection, as_tuple=True)
    if len(track_indices) == 0:
        return None
    tracks = track_parameters[track_indices]
    center0 = geometry.centers.to(dtype=dtype)[sensor_index]
    rotation0 = geometry.rotations.to(dtype=dtype)[sensor_index]
    center, rotation = apply_alignment_torch(center0, rotation0, alignment_row)
    centers = center.expand(len(tracks), -1)
    rotations = rotation.expand(len(tracks), -1, -1)
    _path, intersections, intersection_valid = intersect_plane(
        tracks, centers, rotations[..., :, 2], n_newton=12
    )
    local = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)[..., :2]
    measured = batches.measured_local.to(dtype=dtype)[track_indices, hit_indices]
    covariance = batches.covariances.to(dtype=dtype)[track_indices, hit_indices]
    loc1 = batches.loc1_mask[track_indices, hit_indices]
    residuals = measured - local
    component_mask = torch.stack((intersection_valid, intersection_valid & loc1), dim=-1)
    return residuals, covariance, component_mask, intersection_valid


def _target_alignment_evaluation(track_parameters, alignment, batches, geometry):
    residuals, covariance, component_mask, intersection_valid = _joint_forward(
        track_parameters, alignment, batches, geometry
    )
    target_components = component_mask & batches.is_target.unsqueeze(-1)
    if not target_components.any():
        raise ValueError("No valid target measurement components remain for block LM")
    weighted = residuals.square() / covariance.clamp_min(EPS_COVARIANCE)
    loss = torch.sum(torch.where(target_components, weighted, torch.zeros_like(weighted)))
    loss = loss / target_components.sum()
    valid_hits = batches.hit_mask
    invalid_fraction = 1.0 - float(intersection_valid[valid_hits].float().mean().detach().cpu())
    return loss, invalid_fraction, residuals, component_mask


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
