"""P4: dense joint vs Schur oracle and algebraic properties (Iteration 34 §D)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mlalign import schur_lm as J
import joint_schur_synthetic as syn

ORACLE_RTOL = 1.0e-8
ALGEBRA_RTOL = 1.0e-10
MAX_LEVEL_A_CONDITION = 1.0e6
# Blocks 0/1 share tracks, 2/3 share tracks, 4 is crossed alone.
LEVEL_A_TOPOLOGY = [[0, 1], [0, 1], [1, 0], [2, 3], [3, 2], [2, 3], [4], [4], [0], [1], [2], [3], [4]]
N_LEVEL_A_BLOCKS = 5


def _level_a(seed=0):
    return syn.random_linearization(LEVEL_A_TOPOLOGY, N_LEVEL_A_BLOCKS, seed=seed)


def _level_b():
    problem = syn.shared_topology_problem(injection=[[0.01, 0.01, 0, 0, 0, 1e-4]] * 3)
    return J.linearize_tracks(syn.ref_only_fit(problem), problem.batches, problem.geometry)


def _joint_hessian_condition(blocks):
    """Condition number of the undamped dense joint Hessian, built independently."""
    n, k, t = blocks.n_blocks, blocks.block_size, blocks.n_tracks
    columns = []
    for unit in torch.eye(n * k + 5 * t, dtype=torch.float64):
        du, dtheta = unit[: n * k].reshape(n, k), unit[n * k :].reshape(t, 5)
        information, _ = blocks.geometry_information()
        geo = (information @ du.unsqueeze(-1)).squeeze(-1)
        geo = geo.index_add(0, blocks.pair_block, (blocks.G @ dtheta[blocks.pair_track].unsqueeze(-1)).squeeze(-1))
        trk = (blocks.gamma @ dtheta.unsqueeze(-1)).squeeze(-1)
        trk = trk.index_add(
            0, blocks.pair_track, (blocks.G.transpose(-1, -2) @ du[blocks.pair_block].unsqueeze(-1)).squeeze(-1)
        )
        columns.append(torch.cat((geo.flatten(), trk.flatten())))
    return float(torch.linalg.cond(torch.stack(columns, 1)))


def _schur_step(blocks, lam):
    system = J.schur_reduce(blocks, lam)
    du = J.solve_geometry_dense(system)
    return du, system.back_substitute(du)


def _relative(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))


@pytest.mark.parametrize("level", ["A", "B"])
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_schur_step_and_back_substitution_match_the_dense_joint_solve(level, lam):
    lin = _level_a() if level == "A" else _level_b()
    blocks = J.normal_blocks(lin)
    if level == "A":
        condition = _joint_hessian_condition(blocks)
        print(f"Level A joint Hessian condition {condition:.3e}")
        assert condition <= MAX_LEVEL_A_CONDITION

    dense_du, dense_dtheta = J.solve_joint_dense(blocks, lam)
    schur_du, schur_dtheta = _schur_step(blocks, lam)

    assert _relative(schur_du, dense_du) < ORACLE_RTOL
    assert _relative(schur_dtheta, dense_dtheta) < ORACLE_RTOL


@pytest.mark.parametrize("level", ["A", "B"])
def test_reduced_system_is_symmetric_globally_and_per_track(level):
    lin = _level_a() if level == "A" else _level_b()
    S = J.schur_reduce(J.normal_blocks(lin)).to_dense()
    assert torch.linalg.norm(S - S.T) <= ALGEBRA_RTOL * torch.linalg.norm(S)
    for track in range(lin.residual.shape[0]):
        single = J.schur_reduce(J.normal_blocks(_select_tracks(lin, [track]))).to_dense()
        assert torch.linalg.norm(single - single.T) <= ALGEBRA_RTOL * torch.linalg.norm(single)


def _select_tracks(lin, tracks):
    index = torch.as_tensor(tracks)
    return dataclasses.replace(
        lin,
        parameters=lin.parameters[index],
        residual=lin.residual[index],
        geometry_jacobian=lin.geometry_jacobian[index],
        track_jacobian=lin.track_jacobian[index],
        component_mask=lin.component_mask[index],
        is_target=lin.is_target[index],
        is_ref=lin.is_ref[index],
        block_index=lin.block_index[index],
    )


def test_zero_coupling_reduces_to_the_fixed_track_system():
    lin = _level_a()
    target = lin.is_target[..., None, None]
    uncoupled = dataclasses.replace(lin, track_jacobian=torch.where(target, 0.0, lin.track_jacobian))
    blocks = J.normal_blocks(uncoupled)
    assert bool((blocks.G == 0).all())
    system = J.schur_reduce(blocks)
    information, gradient = blocks.geometry_information()

    expected = torch.block_diag(*information)
    assert torch.linalg.norm(system.to_dense() - expected) <= ALGEBRA_RTOL * torch.linalg.norm(expected)
    assert torch.linalg.norm(system.rhs - gradient) <= ALGEBRA_RTOL * torch.linalg.norm(gradient)


def test_infinitely_precise_reference_tracks_approach_fixed_track():
    lin = _level_a()
    gaps = []
    for kappa in (1.0, 1.0e4, 1.0e8):
        scale = torch.where(lin.is_ref, kappa**0.5, 1.0)[..., None, None]
        blocks = J.normal_blocks(dataclasses.replace(lin, track_jacobian=lin.track_jacobian * scale))
        fixed = torch.block_diag(*blocks.geometry_information()[0])
        S = J.schur_reduce(blocks).to_dense()
        gaps.append(float(torch.linalg.norm(S - fixed) / torch.linalg.norm(fixed)))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] <= 1.0e-6


def test_multi_track_accumulation_equals_the_per_track_sum():
    lin = _level_a()
    total = J.schur_reduce(J.normal_blocks(lin))
    S = torch.zeros_like(total.to_dense())
    rhs = torch.zeros_like(total.rhs)
    for track in range(lin.residual.shape[0]):
        single = J.schur_reduce(J.normal_blocks(_select_tracks(lin, [track])))
        S += single.to_dense()
        rhs += single.rhs
    assert torch.linalg.norm(total.to_dense() - S) <= ALGEBRA_RTOL * torch.linalg.norm(S)
    assert torch.linalg.norm(total.rhs - rhs) <= ALGEBRA_RTOL * torch.linalg.norm(rhs)


@pytest.mark.parametrize("level", ["A", "B"])
def test_only_co_touched_sensor_pairs_get_off_diagonal_blocks(level):
    lin = _level_a() if level == "A" else _level_b()
    system = J.schur_reduce(J.normal_blocks(lin))
    expected = {(i, i) for i in range(len(lin.target_sensors))}
    for track in range(lin.residual.shape[0]):
        touched = set(lin.block_index[track][lin.is_target[track]].tolist())
        expected |= {(i, j) for i in touched for j in touched}

    assert system.block_keys() == expected
    # Both kinds must be present: coupled pairs and pairs that share no track.
    assert any(i != j for i, j in expected)
    assert len(expected) < system.n_blocks**2
    dense = system.to_dense().reshape(system.n_blocks, 3, system.n_blocks, 3)
    for i in range(system.n_blocks):
        for j in range(system.n_blocks):
            block = dense[i, :, j, :]
            if (i, j) in expected:
                assert float(block.abs().max()) > 0.0
            else:
                assert bool((block == 0).all())
