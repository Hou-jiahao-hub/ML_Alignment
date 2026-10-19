"""P11: the diagonal-only Schur ablation (Iteration 43, Level A)."""

from __future__ import annotations

import math

import torch

from mlalign import schur_lm as J
import joint_schur_synthetic as syn
from test_joint_schur_solver import WIDE, LinearModel, _dense_gauss_newton, _start

DTYPE = torch.float64
SHARED_TOPOLOGY = [[0, 1], [0, 1], [1, 0], [2, 3], [3, 2], [2, 3], [4], [4], [0], [1], [2], [3], [4]]
#: Every track crosses one target; each target is crossed by three tracks (full rank).
DISJOINT_TOPOLOGY = [[block] for block in range(5) for _repeat in range(3)]


def _system(topology, n_blocks, seed=0):
    lin = syn.random_linearization(topology, n_blocks, seed=seed)
    return lin, J.schur_reduce(J.normal_blocks(lin), 0.0)


def _diagonal_loop(lin, values=None, tracks=None, **kwargs):
    start_values, start_tracks = _start(lin)
    return J.run_diagonal_only_loop(
        LinearModel(lin),
        start_values if values is None else values,
        start_tracks if tracks is None else tracks,
        WIDE,
        **kwargs,
    )


# P11-1 -------------------------------------------------------------------------


def test_diagonal_only_drops_exactly_the_off_diagonal_blocks():
    _lin, system = _system(SHARED_TOPOLOGY, 5)
    diagonal = J.diagonal_only(system)
    off = system.rows != system.cols
    assert bool(off.any())
    assert bool((diagonal.rows == diagonal.cols).all())
    assert torch.equal(diagonal.rows, system.rows[~off])
    assert torch.equal(diagonal.values, system.values[~off])
    assert torch.equal(diagonal.rhs, system.rhs)
    assert torch.equal(diagonal.track_step, system.track_step)
    assert torch.equal(diagonal.track_gain, system.track_gain)


# P11-2 -------------------------------------------------------------------------


def test_without_shared_tracks_diagonal_only_equals_full_schur():
    lin, system = _system(DISJOINT_TOPOLOGY, 5)
    diagonal = J.diagonal_only(system)
    assert torch.equal(diagonal.rows, system.rows) and torch.equal(diagonal.cols, system.cols)
    assert torch.equal(diagonal.values, system.values)
    assert not bool(J.diagonal_block_rank_deficient(system).any())

    values, tracks = _start(lin)
    full = J.schur_lm_loop(LinearModel(lin), values, tracks, WIDE)
    result, deficient = _diagonal_loop(lin)
    assert not bool(deficient.any())
    assert result.status == full.status == J.SCHUR_LM_STATUS_CONVERGED
    assert torch.allclose(result.values, full.values, rtol=1e-12, atol=1e-14)
    assert torch.allclose(result.tracks, full.tracks, rtol=1e-12, atol=1e-14)


# P11-3 -------------------------------------------------------------------------


def test_with_shared_tracks_one_diagonal_step_differs_from_the_full_step():
    _lin, system = _system(SHARED_TOPOLOGY, 5)
    full = J.solve_geometry_dense(system)
    diagonal = J.solve_geometry_diagonal(system)
    assert float((diagonal - full).norm() / full.norm()) > 1e-3


def test_diagonal_only_shares_the_full_schur_fixed_point():
    lin = syn.random_linearization(SHARED_TOPOLOGY, 5, seed=0)
    du, dtheta = _dense_gauss_newton(lin)
    result, deficient = _diagonal_loop(lin, values=du, tracks=dtheta)
    assert not bool(deficient.any())
    assert result.status == J.SCHUR_LM_STATUS_CONVERGED
    assert len(result.trials) == 0
    assert result.stationarity.lambda2_geo <= 1e-20


# P11-4 -------------------------------------------------------------------------


def test_a_block_crossed_by_one_track_is_rank_deficient_and_reported_undefined():
    topology = DISJOINT_TOPOLOGY + [[5]]
    lin, system = _system(topology, 6)
    flags = J.diagonal_block_rank_deficient(system)
    assert flags.tolist() == [False] * 5 + [True]

    result, deficient = _diagonal_loop(lin)
    assert deficient.tolist() == [False] * 5 + [True]
    assert result.status == J.SCHUR_LM_STATUS_RANK_DEFICIENT
    assert all(math.isnan(float(value)) for value in result.values[5])
    assert bool(torch.isfinite(result.values[:5]).all())

    values, tracks = _start(lin)
    full = J.schur_lm_loop(LinearModel(lin), values, tracks, WIDE)
    assert full.status == J.SCHUR_LM_STATUS_RANK_DEFICIENT
