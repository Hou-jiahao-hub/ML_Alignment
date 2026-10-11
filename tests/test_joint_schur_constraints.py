"""P6: box constraints and KKT of ``schur-lm`` (Iteration 37 §A.3, §D R7-R9, R11).

Bounds act on geometry only.  Bound-active DOFs are removed from the joint
system, and the whole joint step is scaled by one ``alpha`` to stay feasible.
Clamping coupled components one by one after the solve is not a GN/LM step,
and legacy ``lm_max_step_t/r`` clipping does not exist on this path.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn
from ml_alignment_autograd import JOINT_CLAMP_R, JOINT_CLAMP_T, LM_MAX_STEP_T

DTYPE = torch.float64
ORACLE_ATOL = 1.0e-10
ACTIVE = (0, 1, 5)


def test_bounds_are_the_block_lm_feasible_region():
    bounds = J.geometry_bounds(ACTIVE)
    assert bounds.tolist() == [JOINT_CLAMP_T, JOINT_CLAMP_T, JOINT_CLAMP_R]


def test_free_mask_freezes_only_bounds_whose_gradient_points_outwards():
    bounds = torch.tensor([1.5, 1.5, 0.05], dtype=DTYPE)
    values = torch.tensor([[1.5, 1.5, 0.0], [-1.5, -1.5, -0.05]], dtype=DTYPE)
    gradient = torch.tensor([[-1.0, 1.0, 5.0], [1.0, -1.0, 2.0]], dtype=DTYPE)

    free = J.free_geometry_mask(values, gradient, bounds)

    assert free.tolist() == [[False, True, True], [False, True, False]]


# R8 -------------------------------------------------------------------------


def test_step_to_bound_scales_the_whole_step_and_lands_on_the_blocking_bound():
    bounds = torch.tensor([1.5, 1.5, 0.05], dtype=DTYPE)
    values = torch.tensor([[1.0, -1.4, 0.0]], dtype=DTYPE)
    delta_u = torch.tensor([[1.0, -0.4, 3.0]], dtype=DTYPE)  # rz in units of 10 mrad

    alpha, blocking = J.step_to_bound(values, delta_u, bounds, ACTIVE)
    moved = J.apply_geometry_step(values, delta_u, alpha, bounds, ACTIVE)

    assert alpha == pytest.approx(0.25, rel=1e-14)
    assert blocking.tolist() == [[False, True, False]]
    assert float(moved[0, 1]) == -1.5
    assert moved[0, 0] == pytest.approx(1.25, rel=1e-14)
    assert moved[0, 2] == pytest.approx(0.0075, rel=1e-14)


def test_a_dof_on_its_bound_moving_outwards_blocks_with_zero_alpha():
    bounds = torch.tensor([1.5, 1.5, 0.05], dtype=DTYPE)
    alpha, blocking = J.step_to_bound(
        torch.tensor([[1.5, 0.0, 0.0]], dtype=DTYPE),
        torch.tensor([[0.2, 0.1, 0.0]], dtype=DTYPE),
        bounds,
        ACTIVE,
    )
    assert alpha == 0.0
    assert blocking.tolist() == [[True, False, False]]


def test_a_large_feasible_step_is_never_clipped():
    """The legacy per-DOF ``lm_max_step_t`` would cut this 0.5 mm step to 0.1 mm."""
    bounds = J.geometry_bounds(ACTIVE)
    values = torch.zeros((1, 3), dtype=DTYPE)
    delta_u = torch.tensor([[0.5, 0.0, 0.0]], dtype=DTYPE)
    assert 0.5 > LM_MAX_STEP_T

    alpha, blocking = J.step_to_bound(values, delta_u, bounds, ACTIVE)

    assert alpha == 1.0 and not bool(blocking.any())
    assert J.apply_geometry_step(values, delta_u, alpha, bounds, ACTIVE)[0, 0] == 0.5


def test_feasible_step_is_the_unrestricted_lm_step_when_no_bound_is_near():
    lin = syn.random_linearization([[0, 1], [0], [1], [0, 1]], 2, seed=5)
    blocks = J.normal_blocks(lin)
    system = J.schur_reduce(blocks, 1.0e-3)
    bounds = torch.full((3,), 1.0e3, dtype=DTYPE)
    values = torch.zeros((2, 3), dtype=DTYPE)
    free = torch.ones((2, 3), dtype=torch.bool)

    step = J.feasible_joint_step(system, values, free, bounds, lin.active_dof)

    du = J.solve_geometry_dense(system)
    assert step.alpha == 1.0 and step.n_resolves == 0
    assert torch.equal(step.delta_u, du)
    assert torch.equal(step.delta_theta, system.back_substitute(du))


def test_coupled_step_through_a_bound_freezes_that_dof_and_resolves():
    """``b_0 > 0`` at the upper bound leaves DOF 0 free, but the coupled step pushes it out."""
    S = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=DTYPE)
    b = torch.tensor([0.1, 1.0], dtype=DTYPE)
    system = J.SchurSystem(
        rows=torch.tensor([0]),
        cols=torch.tensor([0]),
        values=S[None],
        rhs=b[None],
        lam=0.0,
        track_step=torch.zeros((1, 5), dtype=DTYPE),
        track_gain=torch.zeros((1, 5, 2), dtype=DTYPE),
        pair_track=torch.tensor([0]),
        pair_block=torch.tensor([0]),
    )
    bounds = torch.tensor([1.5, 1.5], dtype=DTYPE)
    values = torch.tensor([[1.5, 0.0]], dtype=DTYPE)
    free = J.free_geometry_mask(values, b[None], bounds)
    assert free.tolist() == [[True, True]]
    assert float(J.solve_geometry_dense(system)[0, 0]) > 0.0

    step = J.feasible_joint_step(system, values, free, bounds, (0, 1))

    assert step.free.tolist() == [[False, True]]
    assert step.n_resolves == 1
    assert step.delta_u.tolist() == [[0.0, -1.0]]
    assert step.alpha == 1.0


def test_restricting_the_reduced_system_equals_deleting_dofs_from_the_joint_system():
    lin = syn.random_linearization([[0, 1], [0, 1], [1], [0]], 2, seed=6)
    blocks = J.normal_blocks(lin)
    free = torch.tensor([[True, False, True], [False, True, True]])
    _jac, _res, hessian, gradient = syn.dense_joint_system(lin)
    keep = torch.cat((free.flatten(), torch.ones(len(gradient) - free.numel(), dtype=torch.bool)))
    expected = torch.zeros_like(gradient)
    expected[keep] = -torch.linalg.solve(hessian[keep][:, keep], gradient[keep])

    system = J.restrict_to_free(J.schur_reduce(blocks, 0.0), free)
    du = J.solve_geometry_dense(system)
    dtheta = system.back_substitute(du)

    assert torch.allclose(torch.cat((du.flatten(), dtheta.flatten())), expected, rtol=1e-10, atol=1e-12)
    assert bool((du[~free] == 0.0).all())


# R7 -------------------------------------------------------------------------


def _kkt_oracle(lin, bounds):
    """Enumerate every active set of the box QP; independent of the Schur path."""
    _jac, _res, hessian, gradient = syn.dense_joint_system(lin)
    n_geo = len(lin.target_sensors) * lin.geometry_jacobian.shape[-1]
    upper = bounds.repeat(len(lin.target_sensors))
    best = None
    for assignment in itertools.product((0, -1, 1), repeat=n_geo):
        side = torch.tensor(assignment, dtype=DTYPE)
        fixed = torch.cat((side != 0, torch.zeros(len(gradient) - n_geo, dtype=torch.bool)))
        x = torch.zeros_like(gradient)
        x[:n_geo] = side * upper
        rhs = -(gradient[~fixed] + hessian[~fixed][:, fixed] @ x[fixed])
        x[~fixed] = torch.linalg.solve(hessian[~fixed][:, ~fixed], rhs)
        if bool((x[:n_geo].abs() > upper + 1e-12).any()):
            continue
        grad = hessian @ x + gradient
        if bool((grad[:n_geo][side > 0] > 1e-10).any()) or bool((grad[:n_geo][side < 0] < -1e-10).any()):
            continue
        value = float(0.5 * x @ hessian @ x + gradient @ x)
        if best is None or value < best[0]:
            best = (value, x, side)
    return best


def _active_set_solve(lin, bounds, lam=0.0, max_iters=30):
    """Test-side driver of the contract functions on a linear problem (no LM loop yet)."""
    n, k = len(lin.target_sensors), lin.geometry_jacobian.shape[-1]
    u = torch.zeros((n, k), dtype=DTYPE)
    theta = torch.zeros((lin.residual.shape[0], 5), dtype=DTYPE)
    for _ in range(max_iters):
        current = J.normal_blocks(dataclasses.replace(lin, residual=syn.shifted_linear_residual(lin, u, theta)))
        undamped = J.schur_reduce(current, 0.0)
        free = J.free_geometry_mask(u, undamped.rhs, bounds)
        stationarity = J.global_stationarity(current, undamped, free)
        if stationarity.stationary and stationarity.lambda2_geo + stationarity.lambda2_trk_max < 1e-20:
            return u, theta, free, stationarity
        step = J.feasible_joint_step(J.schur_reduce(current, lam), u, free, bounds, lin.active_dof)
        u = J.apply_geometry_step(u, step.delta_u, step.alpha, bounds, lin.active_dof)
        theta = theta + step.alpha * step.delta_theta
    raise AssertionError("active-set driver did not reach a KKT point")


@pytest.mark.parametrize("seed", [7, 8, 9])
@pytest.mark.parametrize("shrink", [(0.5, 10.0, 10.0), (0.5, 0.5, 10.0), (0.3, 0.3, 0.3)])
def test_active_set_reaches_the_enumerated_kkt_point(seed, shrink):
    lin = syn.random_linearization([[0], [0], [0], [0]], 1, seed=seed)
    _jac, _res, hessian, gradient = syn.dense_joint_system(lin)
    unconstrained = torch.linalg.solve(hessian, gradient)[:3].abs()
    bounds = unconstrained * torch.tensor(shrink, dtype=DTYPE)

    value, x, side = _kkt_oracle(lin, bounds)
    u, theta, free, stationarity = _active_set_solve(lin, bounds)

    assert bool((side != 0).any())
    assert torch.allclose(u.flatten(), x[:3], rtol=0.0, atol=ORACLE_ATOL)
    assert torch.allclose(theta.flatten(), x[3:], rtol=0.0, atol=ORACLE_ATOL * 10)
    assert J.global_terminal_status(stationary=stationarity.stationary) == J.SCHUR_LM_STATUS_CONVERGED
    assert stationarity.n_free == int(free.sum())


def test_damped_active_set_reaches_the_same_kkt_point():
    lin = syn.random_linearization([[0], [0], [0], [0]], 1, seed=7)
    _jac, _res, hessian, gradient = syn.dense_joint_system(lin)
    bounds = torch.linalg.solve(hessian, gradient)[:3].abs() * torch.tensor([0.5, 0.5, 10.0], dtype=DTYPE)

    _value, x, _side = _kkt_oracle(lin, bounds)
    u, _theta, _free, _stationarity = _active_set_solve(lin, bounds, lam=1.0e-3, max_iters=60)

    assert torch.allclose(u.flatten(), x[:3], rtol=0.0, atol=ORACLE_ATOL)


def test_every_dof_frozen_is_a_kkt_point_with_zero_geometry_decrement():
    lin = syn.random_linearization([[0], [0], [0], [0]], 1, seed=7)
    blocks = J.normal_blocks(lin)
    free = torch.zeros((1, 3), dtype=torch.bool)
    result = J.global_stationarity(blocks, J.schur_reduce(blocks, 0.0), free)
    assert result.n_free == 0 and result.lambda2_geo == 0.0
    assert not result.rank_deficient


# R9 -------------------------------------------------------------------------


def test_constraint_stall_needs_consecutive_tiny_alphas():
    count = 0
    for alpha in (1e-4, 1e-4):
        count = J.update_stall_count(count, alpha)
    assert count == J.SCHUR_LM_STALL_ITERS - 1
    count = J.update_stall_count(count, J.SCHUR_LM_ALPHA_STALL)
    assert count == 0
    for alpha in (1e-4, 5e-4, 9.99e-4):
        count = J.update_stall_count(count, alpha)
    assert count == J.SCHUR_LM_STALL_ITERS
    assert J.global_terminal_status(constraint_stalled=count >= J.SCHUR_LM_STALL_ITERS) == (
        J.SCHUR_LM_STATUS_CONSTRAINT_STALLED
    )


# R11 ------------------------------------------------------------------------


def test_schur_module_has_no_legacy_step_clipping():
    source = inspect.getsource(J)
    assert "LM_MAX_STEP" not in source
    assert "lm_max_step" not in source
