"""P3: Jacobian closure of the joint linearisation (Iteration 34 §B, 35)."""

from __future__ import annotations

import torch

from mlalign import schur_lm as J
import joint_schur_synthetic as syn
from mlalign.helix.gauge import pack_gauge_fixed_track_parameters, reference_gauge_frame
from mlalign.helix.measurement_fit import (
    MEASUREMENT_VARIANCE_FLOOR,
    MeasurementFitInputs,
    measurement_residual_jacobian,
    measurement_standardized_residual,
)
from mlalign.helix.trajectory import helix_tangent, intersect_plane

from mlalign.conventions import EPS_COVARIANCE
from mlalign.rotations import apply_alignment_torch

# Iteration 35: Richardson extrapolation of central differences at h and h/2.
FD_STEP = 1.0e-3
RTOL = 1.0e-6
ATOL_COLUMN = 1.0e-8
PRODUCTION_RTOL = 1.0e-10
NEAR_PARALLEL_COS = 0.3
ACTIVE = (0, 1, 5)
# 650 mm is yawed 75 deg (|cos| ~ 0.26, near-parallel); 850 mm is an
# extrapolated 1D strip.
TARGETS = ((450.0, 0.0), (550.0, 15.0), (650.0, 75.0), (850.0, 0.0))
TOPOLOGY = [[0, 1], [0, 1, 2], [1, 2], [2, 3], [0, 3], [0, 1, 2, 3], [1], [3], [0, 2], [1, 3]]
STRIP_TARGET = 3


def _problem():
    truth = syn.make_tracks(len(TOPOLOGY), seed=5)
    problem = syn.make_problem(
        truth,
        targets=TARGETS,
        topology=TOPOLOGY,
        loc1_targets=(STRIP_TARGET,),
        injection=[[0.02, -0.01, 0.0, 0.0, 0.0, 2.0e-4]] * len(TARGETS),
        noise_seed=7,
    )
    # A generic point: off the ref-only optimum, with a nonzero alignment on
    # every target DOF (including the inactive ones).
    generator = torch.Generator().manual_seed(9)
    parameters = syn.ref_only_fit(problem)
    parameters = parameters + torch.tensor([0.0, 0.05, 0.05, 1e-3, 1e-3, 1e-6], dtype=torch.float64) * torch.randn(
        parameters.shape, generator=generator, dtype=torch.float64
    )
    alignment = torch.zeros((len(problem.geometry.centers), 6), dtype=torch.float64)
    alignment[problem.n_ref :] = torch.tensor(
        [0.03, -0.02, 0.01, 1e-3, -2e-3, 3e-3], dtype=torch.float64
    ) * torch.randn((len(TARGETS), 6), generator=generator, dtype=torch.float64)
    return problem, parameters, alignment


def _residual(problem, parameters, theta, alignment, component_mask):
    """Production whitened residual on the frozen set, an independent path."""
    centers, rotations = apply_alignment_torch(
        problem.geometry.centers, problem.geometry.rotations, alignment
    )
    sensor = problem.batches.sensor_idx
    inputs = MeasurementFitInputs(
        measured=problem.batches.measured_local,
        variance=problem.batches.covariances,
        component_mask=component_mask,
        centers=centers[sensor],
        rotations=rotations[sensor],
    )
    frame = reference_gauge_frame(parameters)
    return measurement_standardized_residual(theta, frame, inputs).reshape(component_mask.shape)


def _richardson(evaluate):
    """``(4 D(h/2) - D(h)) / 3`` of the central difference ``D``."""

    def central(h):
        return (evaluate(h) - evaluate(-h)) / (2 * h)

    return (4 * central(FD_STEP / 2) - central(FD_STEP)) / 3


def _closes(ad, fd):
    """Element-wise Iteration 34 §B criterion; ``ad``/``fd`` are one column."""
    tolerance = RTOL * ad.abs() + ATOL_COLUMN * ad.abs().max()
    return (ad - fd).abs() <= tolerance


def _cos_incidence(problem, parameters, alignment):
    centers, rotations = apply_alignment_torch(
        problem.geometry.centers, problem.geometry.rotations, alignment
    )
    sensor = problem.batches.sensor_idx
    tracks = parameters[:, None, :].expand(-1, sensor.shape[1], -1)
    normals = rotations[sensor][..., :, 2]
    s, _point, _valid = intersect_plane(tracks, centers[sensor], normals)
    tangent = helix_tangent(tracks, s)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)
    return (tangent * normals).sum(-1).abs()


def test_autograd_jacobians_match_central_differences_in_every_stratum():
    problem, parameters, alignment = _problem()
    lin = J.linearize_tracks(parameters, problem.batches, problem.geometry, alignment, ACTIVE)
    mask = lin.component_mask
    theta = pack_gauge_fixed_track_parameters(parameters)

    checks = []  # (ad column, fd column) over all [T, H, 2] entries
    for j in range(5):
        unit = torch.zeros_like(theta)
        unit[:, j] = 1.0
        fd = _richardson(
            lambda h: _residual(problem, parameters, theta + h * unit, alignment, mask)
        )
        checks.append((lin.track_jacobian[..., j], fd))
    for block, sensor in enumerate(lin.target_sensors.tolist()):
        on_block = (lin.block_index == block).unsqueeze(-1)
        for column, dof in enumerate(ACTIVE):
            unit = torch.zeros_like(alignment)
            unit[sensor, dof] = J.GEOMETRY_SCALES[dof]
            fd = _richardson(
                lambda h: _residual(problem, parameters, theta, alignment + h * unit, mask)
            )
            ad = torch.where(on_block, lin.geometry_jacobian[..., column], 0.0)
            checks.append((ad, fd))

    cos = _cos_incidence(problem, parameters, alignment)
    near_parallel = (cos < NEAR_PARALLEL_COS) & lin.is_target
    strip = lin.is_target & ~problem.batches.loc1_mask
    strata = {
        "near_parallel": near_parallel.unsqueeze(-1).expand_as(mask) & mask,
        "strip_1d": strip.unsqueeze(-1).expand_as(mask) & mask,
        "loc0": mask & torch.tensor([True, False]),
        "loc1": mask & torch.tensor([False, True]),
        "ref": lin.is_ref.unsqueeze(-1) & mask,
        "target": lin.is_target.unsqueeze(-1) & mask,
    }
    for name, selector in strata.items():
        assert int(selector.sum()) > 0, f"empty stratum {name}"
    for ad, fd in checks:
        passed = _closes(ad, fd)
        for name, selector in strata.items():
            assert bool(passed[selector].all()), (
                name,
                float(((ad - fd).abs() / ad.abs().clamp_min(1e-300))[selector & ~passed].max()),
            )
        # Outside the frozen set every derivative is exactly zero.
        assert bool((ad[~mask] == 0).all()) and bool((fd[~mask] == 0).all())


def test_structural_zeros_are_exact():
    problem, parameters, alignment = _problem()
    lin = J.linearize_tracks(parameters, problem.batches, problem.geometry, alignment, ACTIVE)

    assert bool((lin.geometry_jacobian[lin.is_ref] == 0).all())
    strip = lin.is_target & ~problem.batches.loc1_mask
    assert int(strip.sum()) > 0
    for tensor in (lin.residual, lin.geometry_jacobian, lin.track_jacobian):
        assert bool((tensor[strip][:, 1] == 0).all())
    padding = ~problem.batches.hit_mask
    assert int(padding.sum()) > 0
    for tensor in (lin.residual, lin.geometry_jacobian, lin.track_jacobian):
        assert bool((tensor[padding] == 0).all())
    assert bool((lin.block_index[~lin.is_target] == -1).all())


def test_reference_track_jacobian_matches_production_measurement_fit():
    problem, parameters, alignment = _problem()
    lin = J.linearize_tracks(parameters, problem.batches, problem.geometry, alignment, ACTIVE)
    centers, rotations = apply_alignment_torch(
        problem.geometry.centers, problem.geometry.rotations, alignment
    )
    sensor = problem.batches.sensor_idx
    ref_mask = lin.component_mask & lin.is_ref.unsqueeze(-1)
    inputs = MeasurementFitInputs(
        measured=problem.batches.measured_local,
        variance=problem.batches.covariances,
        component_mask=ref_mask,
        centers=centers[sensor],
        rotations=rotations[sensor],
    )
    production = measurement_residual_jacobian(
        pack_gauge_fixed_track_parameters(parameters), reference_gauge_frame(parameters), inputs
    )
    ours = (lin.track_jacobian * lin.is_ref[..., None, None]).flatten(1, 2)

    assert torch.linalg.norm(ours - production) <= PRODUCTION_RTOL * torch.linalg.norm(production)


def test_both_whitening_paths_share_one_variance_floor():
    assert EPS_COVARIANCE == MEASUREMENT_VARIANCE_FLOOR == 1.0e-8
