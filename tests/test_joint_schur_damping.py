"""P4: large-damping safety of the joint LM step (Iteration 34 §D).

Damping the whole joint step must stop both geometry and tracks.  Damping only
the geometry block instead lets a target-inclusive track-only update absorb
the misalignment; that rejected formulation lives here, as a negative
control, and nowhere in the production module.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn

LAMBDAS = (1.0e2, 1.0e4, 1.0e6, 1.0e8)
MAX_DECADE_RATIO = 0.02
MAX_RESIDUAL_STEP = 1.0e-6
ABSORPTION_RTOL = 1.0e-4


def _blocks():
    problem = syn.shared_topology_problem(injection=[[0.01, 0.01, 0, 0, 0, 1e-4]] * 3)
    lin = J.linearize_tracks(syn.ref_only_fit(problem), problem.batches, problem.geometry)
    return J.normal_blocks(lin)


def _joint_step(blocks, lam):
    system = J.schur_reduce(blocks, lam)
    du = J.solve_geometry_dense(system)
    return du, system.back_substitute(du)


def _geometry_only_damped_step(blocks, lam):
    """Negative control: ``C + lam D_a`` damped, ``Gamma`` left undamped."""
    d_a, d_theta = J.marquardt_scales(blocks)
    return J.solve_joint_dense(blocks, lam, (d_a, torch.zeros_like(d_theta)))


def test_joint_damping_stops_geometry_and_tracks():
    blocks = _blocks()
    du0, dtheta0 = _joint_step(blocks, 0.0)
    norms = {lam: tuple(map(torch.linalg.norm, _joint_step(blocks, lam))) for lam in LAMBDAS}
    for name, index in (("du", 0), ("dtheta", 1)):
        print(name, [f"{float(norms[lam][index]):.3e}" for lam in LAMBDAS])

    for low, high in zip(LAMBDAS[1:], LAMBDAS[2:]):
        for index in (0, 1):
            assert norms[high][index] <= MAX_DECADE_RATIO * norms[low][index]
    assert norms[1.0e8][0] <= MAX_RESIDUAL_STEP * torch.linalg.norm(du0)
    assert norms[1.0e8][1] <= MAX_RESIDUAL_STEP * torch.linalg.norm(dtheta0)


def test_geometry_only_damping_reaches_the_track_absorption_limit():
    blocks = _blocks()
    du0, _ = _geometry_only_damped_step(blocks, 0.0)
    du, dtheta = _geometry_only_damped_step(blocks, 1.0e8)
    absorption = -torch.linalg.solve(blocks.gamma, blocks.beta.unsqueeze(-1)).squeeze(-1)

    assert torch.linalg.norm(du) <= MAX_RESIDUAL_STEP * torch.linalg.norm(du0)
    assert torch.linalg.norm(dtheta - absorption) <= ABSORPTION_RTOL * torch.linalg.norm(absorption)
    # The absorbed motion is real: the injection is visible to the tracks.
    assert float(torch.linalg.norm(absorption)) > 0.0


def test_production_module_has_no_geometry_only_damping():
    assert "geometry_only" not in inspect.getsource(J)


def test_damping_a_direction_without_curvature_is_refused():
    blocks = _blocks()
    d_a, d_theta = J.marquardt_scales(blocks)
    d_a = d_a.clone()
    d_a[0, 0] = 0.0
    with pytest.raises(ValueError, match="remove inactive"):
        J.schur_reduce(blocks, 1.0, (d_a, d_theta))
    J.schur_reduce(blocks, 0.0, (d_a, d_theta))
