"""P4: Woodbury oracle for the undamped Schur reduction (Iteration 34 §D).

``S = sum_t A_t^T (I + B_t Gamma_ref^-1 B_t^T)^-1 A_t`` over target rows, in
whitened units where ``V = I``.  The right-hand side is assembled here from
the raw rows of the linearisation with dense solves, independently of the
module's per-track Cholesky path.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn

WOODBURY_RTOL = 1.0e-8


def _level_a():
    topology = [[0, 1], [0, 1], [1, 0], [2, 3], [3, 2], [2, 3], [4], [4], [0], [1], [2], [3], [4]]
    return syn.random_linearization(topology, 5, seed=4)


def _level_b():
    problem = syn.shared_topology_problem(injection=[[0.01, 0.01, 0, 0, 0, 1e-4]] * 3)
    return J.linearize_tracks(syn.ref_only_fit(problem), problem.batches, problem.geometry)


def _rows(lin, track, selector):
    """Dense row blocks of one track restricted to ``selector`` hits."""
    hits = torch.nonzero(selector[track], as_tuple=True)[0]
    n, k = len(lin.target_sensors), lin.geometry_jacobian.shape[-1]
    A = torch.zeros((2 * len(hits), n * k), dtype=torch.float64)
    for row, hit in enumerate(hits.tolist()):
        block = int(lin.block_index[track, hit])
        if block >= 0:
            A[2 * row : 2 * row + 2, block * k : (block + 1) * k] = lin.geometry_jacobian[track, hit]
    B = lin.track_jacobian[track, hits].reshape(-1, 5)
    r = lin.residual[track, hits].reshape(-1)
    return A, B, r


def _ref_stationary(lin):
    """Project every track's ref residual off its ref Jacobian, so beta_ref = 0."""
    residual = lin.residual.clone()
    for track in range(residual.shape[0]):
        hits = torch.nonzero(lin.is_ref[track], as_tuple=True)[0]
        B = lin.track_jacobian[track, hits].reshape(-1, 5)
        r = residual[track, hits].reshape(-1)
        r = r - B @ torch.linalg.solve(B.T @ B, B.T @ r)
        residual[track, hits] = r.reshape(-1, 2)
    return dataclasses.replace(lin, residual=residual)


def _woodbury(lin):
    n, k = len(lin.target_sensors), lin.geometry_jacobian.shape[-1]
    S = torch.zeros((n * k, n * k), dtype=torch.float64)
    b = torch.zeros(n * k, dtype=torch.float64)
    for track in range(lin.residual.shape[0]):
        _A_ref, B_ref, _r_ref = _rows(lin, track, lin.is_ref)
        A, B, r = _rows(lin, track, lin.is_target)
        gamma_ref = B_ref.T @ B_ref
        R = torch.eye(len(r), dtype=torch.float64) + B @ torch.linalg.solve(gamma_ref, B.T)
        S += A.T @ torch.linalg.solve(R, A)
        b += A.T @ torch.linalg.solve(R, r)
    return S, b


@pytest.mark.parametrize("level", ["A", "B"])
def test_undamped_schur_information_equals_the_woodbury_form(level):
    lin = _ref_stationary(_level_a() if level == "A" else _level_b())
    blocks = J.normal_blocks(lin)
    assert float(blocks.beta_ref.abs().max()) <= 1.0e-10 * float(blocks.beta.abs().max())
    system = J.schur_reduce(blocks, lam=0.0)
    S_woodbury, b_woodbury = _woodbury(lin)

    S = system.to_dense()
    assert torch.linalg.norm(S - S_woodbury) / torch.linalg.norm(S) < WOODBURY_RTOL
    b = system.rhs.flatten()
    assert torch.linalg.norm(b - b_woodbury) / torch.linalg.norm(b) < WOODBURY_RTOL
