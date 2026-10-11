"""P2: track-block prerequisites (Iteration 34 §C): SPD blocks, gauge-frame
lifecycle and Monte-Carlo pull calibration of ``Gamma_ref^-1``."""

from __future__ import annotations

import itertools

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn
from helix_model import (
    MeasurementFitInputs,
    measurement_residual_jacobian,
    measurement_standardized_residual,
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    unpack_gauge_fixed_track_parameters,
)

SYMMETRY_RTOL = 1.0e-12
PSD_RTOL = 1.0e-12
LIFECYCLE_RTOL = 1.0e-12
STALE_FRAME_MIN_RELATIVE = 1.0e-6
N_REPLICAS = 4000
PULL_SEED = 20261011
PULL_TOLERANCE = 0.05
GN_DECREMENT = 1.0e-12
# (n_ref, tan lambda, target x): the ref planes span 100-800 mm.
PULL_CELLS = list(itertools.product((8, 16), (0.2, 1.5), (450.0, 950.0)))


def _blocks():
    problem = syn.shared_topology_problem(injection=[[0.01, 0.01, 0, 0, 0, 1e-4]] * 3)
    lin = J.linearize_tracks(syn.ref_only_fit(problem), problem.batches, problem.geometry)
    return J.normal_blocks(lin)


def test_track_blocks_are_symmetric_positive_definite_and_ordered():
    blocks = _blocks()
    diagnostics = J.track_block_diagnostics(blocks)
    for name in ("gamma_ref", "gamma"):
        summary = diagnostics[name]
        assert bool(summary["finite"].all())
        assert float(summary["asymmetry"].max()) <= SYMMETRY_RTOL
        assert bool(summary["cholesky_ok"].all())
        assert bool((summary["rank"] == 5).all())
    gap = torch.linalg.eigvalsh(blocks.gamma - blocks.gamma_ref)[:, 0]
    largest = torch.linalg.eigvalsh(blocks.gamma)[:, -1]
    assert bool((gap >= -PSD_RTOL * largest).all())


def test_committed_step_is_physical_and_the_next_frame_is_rebuilt_from_it():
    truth = syn.make_tracks(6, seed=2)
    generator = torch.Generator().manual_seed(4)
    delta = torch.tensor([0.5, -0.5, 1e-2, 1e-2, 0.1], dtype=torch.float64) * torch.randn(
        (6, 5), generator=generator, dtype=torch.float64
    )
    p1 = J.commit_track_step(truth, delta)
    frame1 = reference_gauge_frame(p1)
    roundtrip = unpack_gauge_fixed_track_parameters(pack_gauge_fixed_track_parameters(p1), frame1)
    assert torch.linalg.norm(roundtrip - p1) <= LIFECYCLE_RTOL * torch.linalg.norm(p1)

    second = torch.tensor([0.3, 0.2, -1e-2, 5e-3, -0.1], dtype=torch.float64).expand(6, 5)
    correct = J.commit_track_step(p1, second)
    stale = unpack_gauge_fixed_track_parameters(
        pack_gauge_fixed_track_parameters(truth) + delta + second, reference_gauge_frame(truth)
    )
    relative = torch.linalg.norm(stale - correct, dim=1) / torch.linalg.norm(correct, dim=1)
    assert bool((relative >= STALE_FRAME_MIN_RELATIVE).all())


def test_linearisation_accepts_only_physical_helices():
    problem = syn.shared_topology_problem()
    gauge_vector = pack_gauge_fixed_track_parameters(problem.truth)
    with pytest.raises(ValueError, match="shape"):
        J.linearize_tracks(gauge_vector, problem.batches, problem.geometry)


def _inputs(problem, component_mask):
    sensor = problem.batches.sensor_idx
    return MeasurementFitInputs(
        measured=problem.batches.measured_local,
        variance=problem.batches.covariances,
        component_mask=component_mask,
        centers=problem.geometry.centers[sensor],
        rotations=problem.geometry.rotations[sensor],
    )


def _pulls(n_ref, tanl, target_x, seed, target_raw_variance=None):
    """Ref-only GN in the truth's own gauge frame, then pulls in that frame."""
    single = syn.make_tracks(1, seed=seed, tanl=tanl)
    truth = single.expand(N_REPLICAS, 6).clone()
    problem = syn.make_problem(
        truth,
        targets=((target_x, 0.0),),
        n_ref=n_ref,
        ref_span=(100.0, 800.0),
        target_raw_variance=target_raw_variance,
        noise_seed=seed,
    )
    mask = torch.stack((problem.batches.hit_mask, problem.batches.hit_mask & problem.batches.loc1_mask), -1)
    ref_inputs = _inputs(problem, mask & ~problem.batches.is_target.unsqueeze(-1))
    target_inputs = _inputs(problem, mask & problem.batches.is_target.unsqueeze(-1))
    frame = reference_gauge_frame(truth)
    theta_true = pack_gauge_fixed_track_parameters(truth)
    theta = theta_true.clone()
    for _ in range(20):
        r = measurement_standardized_residual(theta, frame, ref_inputs)
        jac = measurement_residual_jacobian(theta, frame, ref_inputs)
        hessian = jac.transpose(1, 2) @ jac
        gradient = (jac.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)
        step = -torch.linalg.solve(hessian, gradient)
        theta = theta + step
        if float((-(gradient * step).sum(1)).max()) <= GN_DECREMENT:
            break
    jac = measurement_residual_jacobian(theta, frame, ref_inputs)
    covariance = torch.linalg.inv(jac.transpose(1, 2) @ jac)
    track_pulls = (theta - theta_true) / covariance.diagonal(dim1=1, dim2=2).sqrt()

    target_jac = measurement_residual_jacobian(theta, frame, target_inputs)
    target_r = measurement_standardized_residual(theta, frame, target_inputs)
    propagated = torch.einsum("tri,tij,trj->tr", target_jac, covariance, target_jac)
    active = target_inputs.component_mask.flatten(1)
    target_pulls = target_r / (1.0 + propagated).sqrt()
    loc = torch.arange(active.shape[1]) % 2
    by_component = [target_pulls[active & (loc == c)] for c in (0, 1)]
    return track_pulls, by_component


@pytest.mark.slow
@pytest.mark.parametrize("cell", range(len(PULL_CELLS)))
def test_reference_track_covariance_is_calibrated(cell):
    n_ref, tanl, target_x = PULL_CELLS[cell]
    track_pulls, target_pulls = _pulls(n_ref, tanl, target_x, PULL_SEED + cell)
    regime = "interpolation" if target_x < 800.0 else "extrapolation"
    columns = [track_pulls[:, i] for i in range(5)] + target_pulls
    names = ["o0", "o1", "psi0", "tanl", "rho"] + ["target LOC0", "target LOC1"]
    for name, pulls in zip(names, columns):
        assert len(pulls) == N_REPLICAS
        mean, std = float(pulls.mean()), float(pulls.std())
        print(f"n_ref={n_ref} tanl={tanl} {regime} {name}: mean {mean:+.4f} std {std:.4f}")
        assert abs(mean) <= PULL_TOLERANCE, (name, mean)
        assert abs(std - 1.0) <= PULL_TOLERANCE, (name, std)


@pytest.mark.slow
def test_floor_active_target_pull_is_reported_not_judged():
    # The production floor inflates the modelled variance above the true one;
    # Iteration 34 §C reports this cell and does not judge it.
    _track, target_pulls = _pulls(8, 0.2, 450.0, PULL_SEED + 100, target_raw_variance=(0.01, 0.01))
    std = float(target_pulls[1].std())
    print(f"floor-active target LOC1 pull std {std:.4f} (modelled variance 4x the true one)")
    assert torch.isfinite(target_pulls[1]).all()
