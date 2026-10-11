"""P6: global convergence contract of ``schur-lm`` (Iteration 37 §D, R1-R6, R10).

The 2026-09-08 block-LM contract (per-sensor damping, per-block terminal
states, permanent retirement of converged sensors) rests on exact decoupling
of the target sensors.  Shared tracks break that, so ``schur-lm`` gets one
global model, gain ratio, damping rule, decrement and terminal state.  The
block-LM contract itself is untouched and still applies to ``block-lm``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn

ALGEBRA_RTOL = 1.0e-10
LEVEL_A_TOPOLOGY = [[0, 1], [0, 1], [1, 0], [2, 3], [3, 2], [2, 3], [4], [4], [0], [1], [2], [3], [4]]
N_LEVEL_A_BLOCKS = 5


def _level_a(seed=0):
    return syn.random_linearization(LEVEL_A_TOPOLOGY, N_LEVEL_A_BLOCKS, seed=seed)


def _dense_joint(lin):
    jacobian, residual, hessian, gradient = syn.dense_joint_system(lin)
    return hessian, gradient, jacobian, residual


def _lm_step(blocks, lam, free=None):
    system = J.schur_reduce(blocks, lam)
    if free is not None:
        system = J.restrict_to_free(system, free)
    du = J.solve_geometry_dense(system)
    return du, system.back_substitute(du)


def _linear_objective(lin, du, dtheta):
    """``F = 0.5 |r + A du + B dtheta|^2`` of a Level A (linear) problem, from raw arrays."""
    return 0.5 * float(syn.shifted_linear_residual(lin, du, dtheta).square().sum())


# R1 -------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 1.0e-3, 1.0])
def test_block_model_terms_match_the_dense_joint_model(lam):
    lin = _level_a()
    blocks = J.normal_blocks(lin)
    du, dtheta = _lm_step(blocks, lam)
    hessian, gradient, _jac, _res = _dense_joint(lin)
    p = torch.cat((du.flatten(), dtheta.flatten()))

    model = J.joint_model_reduction(blocks, du, dtheta)

    assert model.gradient_step == pytest.approx(float(gradient @ p), rel=ALGEBRA_RTOL)
    assert model.curvature == pytest.approx(float(p @ hessian @ p), rel=ALGEBRA_RTOL)
    assert model.predicted == pytest.approx(
        float(-gradient @ p - 0.5 * p @ hessian @ p), rel=ALGEBRA_RTOL
    )


def test_lm_step_has_strictly_positive_predicted_reduction():
    blocks = J.normal_blocks(_level_a())
    for lam in (0.0, 1.0e-3, 1.0, 1.0e3):
        du, dtheta = _lm_step(blocks, lam)
        for alpha in (1.0, 0.3, 1.0e-3):
            assert J.joint_model_reduction(blocks, alpha * du, alpha * dtheta).predicted > 0.0


# R2 -------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 1.0e-3, 1.0, 1.0e3])
@pytest.mark.parametrize("alpha", [1.0, 0.3])
def test_gain_ratio_is_one_for_linear_residuals(lam, alpha):
    lin = _level_a()
    blocks = J.normal_blocks(lin)
    du, dtheta = _lm_step(blocks, lam)
    zero_u, zero_theta = torch.zeros_like(du), torch.zeros_like(dtheta)
    actual = _linear_objective(lin, zero_u, zero_theta) - _linear_objective(
        lin, alpha * du, alpha * dtheta
    )

    rho = J.gain_ratio(actual, J.joint_model_reduction(blocks, alpha * du, alpha * dtheta).predicted)

    assert abs(rho - 1.0) <= ALGEBRA_RTOL


def test_gain_ratio_on_the_unexecuted_step_is_wrong():
    """Negative control: model reduction of the full step, objective of the scaled one."""
    lin = _level_a()
    blocks = J.normal_blocks(lin)
    du, dtheta = _lm_step(blocks, 0.0)
    alpha = 0.3
    actual = _linear_objective(lin, torch.zeros_like(du), torch.zeros_like(dtheta)) - _linear_objective(
        lin, alpha * du, alpha * dtheta
    )

    wrong = J.gain_ratio(actual, J.joint_model_reduction(blocks, du, dtheta).predicted)

    assert abs(wrong - 1.0) > 0.1


@pytest.mark.parametrize("predicted", [0.0, -1.0, math.nan, math.inf])
def test_non_positive_or_non_finite_predicted_reduction_is_a_numerical_failure(predicted):
    with pytest.raises(J.ModelReductionError):
        J.gain_ratio(1.0, predicted)


def test_non_finite_actual_reduction_gives_an_invalid_ratio():
    assert math.isnan(J.gain_ratio(math.nan, 1.0))
    assert not J.update_damping(1.0e-3, J.gain_ratio(math.inf, 1.0)).accepted


# R3 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rho", "valid", "accepted", "factor", "branch"),
    [
        (0.5, False, False, 10.0, "reject"),
        (math.nan, True, False, 10.0, "reject"),
        (-3.0, True, False, 10.0, "reject"),
        (0.5e-4, True, False, 10.0, "reject"),
        (1.0e-4, True, True, 10.0, "accept_increase"),
        (0.2, True, True, 10.0, "accept_increase"),
        (0.25, True, True, 1.0, "accept_keep"),
        (0.5, True, True, 1.0, "accept_keep"),
        (0.75, True, True, 0.1, "accept_decrease"),
        (1.0, True, True, 0.1, "accept_decrease"),
        (1.7, True, True, 0.1, "accept_decrease"),
    ],
)
def test_damping_rule_has_four_deterministic_branches(rho, valid, accepted, factor, branch):
    decision = J.update_damping(1.0, rho, valid=valid)

    assert decision.accepted is accepted
    assert decision.branch == branch
    assert decision.lam == pytest.approx(factor)
    assert decision.saturated is False


def test_damping_saturates_at_its_bounds():
    assert J.update_damping(J.SCHUR_LM_LAMBDA_MAX / 2, 0.0).lam == J.SCHUR_LM_LAMBDA_MAX
    assert J.update_damping(J.SCHUR_LM_LAMBDA_MIN * 2, 1.0).lam == J.SCHUR_LM_LAMBDA_MIN
    ceiling = J.update_damping(J.SCHUR_LM_LAMBDA_MAX, 0.0)
    assert not ceiling.accepted and ceiling.saturated
    assert ceiling.lam == J.SCHUR_LM_LAMBDA_MAX
    # Accepting at the ceiling is not saturation: the step was usable.
    assert not J.update_damping(J.SCHUR_LM_LAMBDA_MAX, 0.1).saturated


def test_damping_rule_constants_are_the_preregistered_ones():
    assert J.SCHUR_LM_PREREGISTRATION_VERSION == 1
    assert J.SCHUR_LM_LAMBDA_INIT == 1.0e-3
    assert (J.SCHUR_LM_LAMBDA_INCREASE, J.SCHUR_LM_LAMBDA_DECREASE) == (10.0, 10.0)
    assert (
        J.SCHUR_LM_RHO_ACCEPT_MIN,
        J.SCHUR_LM_RHO_INCREASE_THRESHOLD,
        J.SCHUR_LM_RHO_DECREASE_THRESHOLD,
    ) == (1.0e-4, 0.25, 0.75)
    assert (J.SCHUR_LM_LAMBDA_MIN, J.SCHUR_LM_LAMBDA_MAX) == (1.0e-12, 1.0e12)
    assert (J.SCHUR_LM_MAX_RETRIES, J.SCHUR_LM_MAX_ITERS) == (8, 60)
    assert (J.SCHUR_LM_EPSILON_GEO, J.SCHUR_LM_EPSILON_TRK, J.SCHUR_LM_RANK_TOL) == (1.0e-2, 1.0e-2, 1.0e-4)
    assert (J.SCHUR_LM_ALPHA_STALL, J.SCHUR_LM_STALL_ITERS) == (1.0e-3, 3)
    assert J.SCHUR_LM_LAMBDA_INIT_SCAN == (1.0e-5, 1.0e-3, 1.0e-1)


# R4 -------------------------------------------------------------------------


@pytest.mark.parametrize("frozen", [(), ((0, 1), (3, 0), (3, 2))])
def test_global_decrement_splits_into_geometry_and_track_parts(frozen):
    lin = _level_a(seed=4)
    blocks = J.normal_blocks(lin)
    free = torch.ones((N_LEVEL_A_BLOCKS, 3), dtype=torch.bool)
    for block, dof in frozen:
        free[block, dof] = False
    hessian, gradient, _jac, _res = _dense_joint(lin)
    keep = torch.cat((free.flatten(), torch.ones(len(gradient) - free.numel(), dtype=torch.bool)))
    expected = float(gradient[keep] @ torch.linalg.solve(hessian[keep][:, keep], gradient[keep]))

    result = J.global_stationarity(blocks, J.schur_reduce(blocks, 0.0), free)

    assert result.lambda2_geo + float(result.lambda2_trk.sum()) == pytest.approx(expected, rel=ALGEBRA_RTOL)
    assert result.n_free == int(free.sum())
    assert not result.rank_deficient and not result.lambda2_is_lower_bound
    assert result.null_gradient_max == 0.0


def test_global_decrement_requires_the_undamped_system():
    blocks = J.normal_blocks(_level_a())
    with pytest.raises(ValueError):
        J.global_stationarity(blocks, J.schur_reduce(blocks, 1.0))


def test_stationarity_needs_both_geometry_and_every_track():
    blocks = J.normal_blocks(_level_a())
    result = J.global_stationarity(blocks, J.schur_reduce(blocks, 0.0))
    assert not result.stationary

    # Zero residual: every gradient vanishes.
    lin = dataclasses.replace(_level_a(), residual=torch.zeros_like(_level_a().residual))
    zero = J.normal_blocks(lin)
    assert J.global_stationarity(zero, J.schur_reduce(zero, 0.0)).stationary

    # Only the tracks are off their minimum: geometry alone must not pass.
    lin = _level_a()
    blocks = J.normal_blocks(lin)
    du, dtheta = _lm_step(blocks, 0.0)
    shifted = syn.shifted_linear_residual(lin, du, dtheta)
    at_minimum = J.normal_blocks(dataclasses.replace(lin, residual=shifted))
    result = J.global_stationarity(at_minimum, J.schur_reduce(at_minimum, 0.0))
    assert result.lambda2_geo < 1.0e-20 and result.lambda2_trk_max < 1.0e-20
    assert result.stationary
    dtheta_off = torch.zeros_like(dtheta)
    dtheta_off[0, 0] = 1.0
    off = shifted + (lin.track_jacobian @ dtheta_off[:, None, :, None]).squeeze(-1) * lin.component_mask
    off_blocks = J.normal_blocks(dataclasses.replace(lin, residual=off))
    result = J.global_stationarity(off_blocks, J.schur_reduce(off_blocks, 0.0))
    assert result.lambda2_trk_max > J.SCHUR_LM_EPSILON_TRK
    assert not result.stationary


# R5 -------------------------------------------------------------------------


def test_weak_geometry_direction_is_rank_deficient_and_a_lower_bound():
    lin = _level_a(seed=2)
    jacobian = lin.geometry_jacobian.clone()
    on_block = (lin.block_index == 4).unsqueeze(-1)
    jacobian[..., 2] = torch.where(on_block, jacobian[..., 0], jacobian[..., 2])
    weak = J.normal_blocks(dataclasses.replace(lin, geometry_jacobian=jacobian))

    result = J.global_stationarity(weak, J.schur_reduce(weak, 0.0))

    assert result.rank_deficient
    assert result.lambda2_is_lower_bound
    assert result.min_scaled_eigenvalue <= J.SCHUR_LM_RANK_TOL
    assert result.null_gradient_max > 0.0

    full = J.normal_blocks(lin)
    healthy = J.global_stationarity(full, J.schur_reduce(full, 0.0))
    assert not healthy.rank_deficient and healthy.min_scaled_eigenvalue > J.SCHUR_LM_RANK_TOL


def test_rank_deficient_wins_over_a_stationary_lower_bound():
    assert (
        J.global_terminal_status(rank_deficient=True, stationary=True)
        == J.SCHUR_LM_STATUS_RANK_DEFICIENT
    )


# R6 -------------------------------------------------------------------------


def _joint_gauss_newton(problem, iterations=8):
    tracks = syn.ref_only_fit(problem)
    alignment = torch.zeros_like(problem.true_alignment)
    embedding = None
    for _ in range(iterations):
        lin = J.linearize_tracks(tracks, problem.batches, problem.geometry, alignment)
        embedding = J.geometry_embedding(lin.active_dof)
        blocks = J.normal_blocks(lin)
        system = J.schur_reduce(blocks, 0.0)
        du = J.solve_geometry_dense(system)
        alignment[lin.target_sensors] += du @ embedding.T
        tracks = J.commit_track_step(tracks, system.back_substitute(du))
    return tracks, alignment, lin


def _per_sensor_decrement(system, blocks):
    """Descriptive per-sensor ``b_i^T S_ii^-1 b_i``; never a convergence verdict."""
    diagonal = system.values[system.rows == system.cols]
    return torch.einsum("ni,ni->n", system.rhs, torch.linalg.solve(diagonal, system.rhs))


def test_moving_one_sensor_takes_its_track_sharing_neighbour_off_stationarity():
    """Replaces the block-LM decoupling regression for ``schur-lm``.

    Sensors 0 and 1 share tracks, sensor 2 is crossed alone.  At the joint
    stationary point, moving sensor 1 must leave sensor 2 untouched but push
    sensor 0 off stationarity, so no sensor can retire once it has converged.
    The fixed-track gradient ``c_0`` does not move: that decoupling is exactly
    what Schur removes.
    """
    problem = syn.shared_topology_problem(injection=[[0.05, 0.05, 0, 0, 0, 2e-4]] * 3)
    tracks, alignment, lin = _joint_gauss_newton(problem)
    blocks = J.normal_blocks(lin)
    before = _per_sensor_decrement(J.schur_reduce(blocks, 0.0), blocks)
    _, c_before = blocks.geometry_information()
    assert float(before.max()) < 1.0e-12

    moved = alignment.clone()
    sensor_1 = int(lin.target_sensors[1])
    moved[sensor_1, 0] += 0.1
    lin_moved = J.linearize_tracks(tracks, problem.batches, problem.geometry, moved)
    blocks_moved = J.normal_blocks(lin_moved)
    after = _per_sensor_decrement(J.schur_reduce(blocks_moved, 0.0), blocks_moved)
    _, c_after = blocks_moved.geometry_information()

    assert float(after[0]) > J.SCHUR_LM_EPSILON_GEO
    assert float(after[2]) < 1.0e-12
    assert torch.allclose(c_after[0], c_before[0], rtol=0.0, atol=1.0e-10)


# R10 ------------------------------------------------------------------------

_PRIORITY = [
    ("numerical_failure", J.SCHUR_LM_STATUS_NUMERICAL_FAILURE),
    ("intersection_failed", J.SCHUR_LM_STATUS_INTERSECTION_FAILED),
    ("rank_deficient", J.SCHUR_LM_STATUS_RANK_DEFICIENT),
    ("constraint_stalled", J.SCHUR_LM_STATUS_CONSTRAINT_STALLED),
    ("stationary", J.SCHUR_LM_STATUS_CONVERGED),
]


def _flags(name):
    if name == "intersection_failed":
        return {"retries_exhausted": True, "component_loss_seen": True}
    return {name: True}


@pytest.mark.parametrize("index", range(len(_PRIORITY)))
def test_terminal_status_priority(index):
    name, status = _PRIORITY[index]
    flags = _flags(name)
    for lower, _ in _PRIORITY[index + 1 :]:
        flags.update(_flags(lower))
    assert J.global_terminal_status(**flags) == status


def test_terminal_status_falls_back_to_budget_exhausted():
    assert J.global_terminal_status() == J.SCHUR_LM_STATUS_BUDGET_EXHAUSTED
    # Retries ran out without any component loss: a damping ceiling, not an intersection failure.
    assert J.global_terminal_status(retries_exhausted=True) == J.SCHUR_LM_STATUS_BUDGET_EXHAUSTED
    assert J.global_terminal_status(component_loss_seen=True) == J.SCHUR_LM_STATUS_BUDGET_EXHAUSTED
    assert set(J.SCHUR_LM_TERMINAL_STATUSES) == {status for _, status in _PRIORITY} | {
        J.SCHUR_LM_STATUS_BUDGET_EXHAUSTED
    }
