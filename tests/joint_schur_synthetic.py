"""Synthetic helix / plane problems shared by the ``test_joint_schur_*`` suites.

Level B of Iteration 34: real helix and intersection code on known planes.
Reference planes are normal to x; target planes may be yawed about z so that
near-parallel incidence and rotation observability can be exercised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ml_alignment_autograd as M
from helix_model import MeasurementFitInputs, measurement_local_prediction

DTYPE = torch.float64
# Local frame (u, v, n) = (y, z, x): planes normal to x, measuring y and z.
ROTATION = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)


def yawed_rotation(yaw_deg: float) -> torch.Tensor:
    angle = math.radians(yaw_deg)
    yaw = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    return yaw @ ROTATION


@dataclass
class SyntheticProblem:
    batches: object
    geometry: object
    truth: torch.Tensor
    true_alignment: torch.Tensor
    n_ref: int


def make_tracks(n_tracks: int, seed: int, tanl: float | None = None, curvature: float = 2.0e-5):
    generator = torch.Generator().manual_seed(seed)
    truth = torch.zeros((n_tracks, 6), dtype=DTYPE)
    truth[:, 1] = 20.0 * torch.randn(n_tracks, generator=generator, dtype=DTYPE)
    truth[:, 2] = 20.0 * torch.randn(n_tracks, generator=generator, dtype=DTYPE)
    truth[:, 3] = 0.03 * torch.randn(n_tracks, generator=generator, dtype=DTYPE)
    if tanl is None:
        truth[:, 4] = 0.05 * torch.randn(n_tracks, generator=generator, dtype=DTYPE)
    else:
        truth[:, 4] = tanl
    truth[:, 5] = curvature * (1.0 + 0.5 * torch.randn(n_tracks, generator=generator, dtype=DTYPE))
    return truth


def make_problem(
    truth: torch.Tensor,
    targets=((450.0, 0.0), (550.0, 0.0)),
    topology=None,
    n_ref: int = 7,
    ref_span=(100.0, 700.0),
    ref_variance=(0.01, 0.04),
    target_variance=(0.01, 0.04),
    target_raw_variance=None,
    loc1_targets=(),
    injection=None,
    noise_seed: int | None = None,
) -> SyntheticProblem:
    """Build batches whose measurements come from ``truth`` on the injected geometry.

    ``targets`` are (x, yaw in degrees); ``topology[t]`` lists the targets track
    ``t`` crosses (default: all).  ``loc1_targets`` are 1D strips (LOC1 masked).
    ``injection`` [n_targets, 6] misaligns the true target surfaces; the returned
    geometry is the nominal model.
    """
    n_tracks = truth.shape[0]
    ref_x = torch.linspace(ref_span[0], ref_span[1], n_ref, dtype=DTYPE)
    n_targets = len(targets)
    n_surfaces = n_ref + n_targets
    centers = torch.zeros((n_surfaces, 3), dtype=DTYPE)
    centers[:n_ref, 0] = ref_x
    rotations = ROTATION.expand(n_surfaces, 3, 3).clone()
    for index, (x, yaw) in enumerate(targets):
        centers[n_ref + index, 0] = x
        rotations[n_ref + index] = yawed_rotation(yaw)
    target_mask = torch.zeros(n_surfaces, dtype=torch.bool)
    target_mask[n_ref:] = True
    geometry = M.JointGeometry(
        transforms=[],
        input_transforms=[],
        centers=centers,
        rotations=rotations,
        target_mask=target_mask,
        id_to_idx={},
        input_indices=[],
    )
    true_alignment = torch.zeros((n_surfaces, 6), dtype=DTYPE)
    if injection is not None:
        true_alignment[n_ref:] = torch.as_tensor(injection, dtype=DTYPE)
    true_centers, true_rotations = M.apply_alignment_torch(centers, rotations, true_alignment)

    if topology is None:
        topology = [list(range(n_targets))] * n_tracks
    rows = []
    for track in range(n_tracks):
        surfaces = list(range(n_ref)) + [n_ref + t for t in topology[track]]
        rows.append(sorted(surfaces, key=lambda s: float(centers[s, 0])))
    n_hits = max(len(row) for row in rows)
    sensor_idx = torch.zeros((n_tracks, n_hits), dtype=torch.long)
    hit_mask = torch.zeros((n_tracks, n_hits), dtype=torch.bool)
    for track, row in enumerate(rows):
        sensor_idx[track, : len(row)] = torch.tensor(row)
        hit_mask[track, : len(row)] = True
    is_target = hit_mask & target_mask[sensor_idx]
    loc1 = hit_mask.clone()
    for target in loc1_targets:
        loc1 &= sensor_idx != n_ref + target
    variance = torch.ones((n_tracks, n_hits, 2), dtype=DTYPE)
    variance[hit_mask & ~is_target] = torch.tensor(ref_variance, dtype=DTYPE)
    variance[is_target] = torch.tensor(target_variance, dtype=DTYPE)
    raw_variance = variance.clone()
    if target_raw_variance is not None:
        raw_variance[is_target] = torch.tensor(target_raw_variance, dtype=DTYPE)

    exact, valid = measurement_local_prediction(
        truth,
        MeasurementFitInputs(
            measured=torch.zeros((n_tracks, n_hits, 2), dtype=DTYPE),
            variance=variance,
            component_mask=torch.stack((hit_mask, hit_mask & loc1), -1),
            centers=true_centers[sensor_idx],
            rotations=true_rotations[sensor_idx],
        ),
    )
    if not bool(valid[hit_mask].all()):
        raise ValueError("A synthetic track misses one of its planes.")
    measured = torch.where(hit_mask.unsqueeze(-1), exact, torch.zeros_like(exact))
    if noise_seed is not None:
        generator = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(measured.shape, generator=generator, dtype=DTYPE)
        measured = measured + noise * raw_variance.sqrt() * hit_mask.unsqueeze(-1)
    batches = M.TrackBatches(
        measured_local=measured,
        covariances=variance,
        sensor_idx=sensor_idx,
        hit_mask=hit_mask,
        loc1_mask=loc1,
        g_smt=torch.zeros((n_tracks, n_hits, 3), dtype=DTYPE),
        p_smt=torch.zeros((n_tracks, n_hits, 3), dtype=DTYPE),
        volume_id=torch.where(is_target, 17, 10),
        layer_id=torch.where(is_target, 8, 2),
        is_target=is_target,
        event_track_ids=torch.stack(
            (torch.zeros(n_tracks, dtype=torch.long), torch.arange(n_tracks)), 1
        ),
        fit_mask=hit_mask & ~is_target,
        raw_covariances=raw_variance,
    )
    return SyntheticProblem(batches, geometry, truth, true_alignment, n_ref)


def ref_only_fit(problem: SyntheticProblem, start: torch.Tensor | None = None) -> torch.Tensor:
    """Production ref-only Gauss-Newton refit (target hits never enter)."""
    start = problem.truth if start is None else start
    fitted, _stats = M.fit_helices_to_measurements(
        problem.batches, problem.geometry, start, dtype=DTYPE, stationarity=False
    )
    return fitted.detach().to(DTYPE)


def random_linearization(topology, n_blocks: int, n_ref_hits: int = 6, seed: int = 0, k: int = 3):
    """Level A: a pure-algebra ``TrackLinearization`` with random A, B and r.

    ``topology[t]`` lists the geometry blocks track ``t`` crosses (one hit each).
    """
    import joint_alignment_schur as J

    generator = torch.Generator().manual_seed(seed)
    n_tracks = len(topology)
    n_hits = n_ref_hits + max(len(row) for row in topology)
    hit_mask = torch.zeros((n_tracks, n_hits), dtype=torch.bool)
    is_target = torch.zeros_like(hit_mask)
    block_index = torch.full((n_tracks, n_hits), -1, dtype=torch.long)
    for track, row in enumerate(topology):
        hit_mask[track, : n_ref_hits + len(row)] = True
        for slot, block in enumerate(row):
            is_target[track, n_ref_hits + slot] = True
            block_index[track, n_ref_hits + slot] = block
    mask = hit_mask.unsqueeze(-1).expand(-1, -1, 2)

    def draw(*shape):
        return torch.randn(shape, generator=generator, dtype=DTYPE)

    residual = draw(n_tracks, n_hits, 2) * mask
    geometry_jacobian = draw(n_tracks, n_hits, 2, k) * is_target[..., None, None]
    track_jacobian = draw(n_tracks, n_hits, 2, 5) * hit_mask[..., None, None]
    return J.TrackLinearization(
        parameters=torch.zeros((n_tracks, 6), dtype=DTYPE),
        residual=residual,
        geometry_jacobian=geometry_jacobian,
        track_jacobian=track_jacobian,
        component_mask=mask.clone(),
        is_target=is_target,
        is_ref=hit_mask & ~is_target,
        block_index=block_index,
        target_sensors=torch.arange(n_blocks),
        active_dof=tuple(range(k)),
    )


def shared_topology_problem(noise_seed=3, injection=None, n_tracks=10, seed=1):
    """Level B: three 2D targets; 0 and 1 share tracks, 2 is crossed alone."""
    truth = make_tracks(n_tracks, seed=seed)
    topology = [[0, 1] if t % 2 == 0 else ([2] if t % 4 == 1 else [0, 1]) for t in range(n_tracks)]
    topology[-1] = [2]
    topology[-2] = [2]
    return make_problem(
        truth,
        targets=((450.0, 0.0), (550.0, 15.0), (850.0, 0.0)),
        topology=topology,
        injection=injection,
        noise_seed=noise_seed,
    )
