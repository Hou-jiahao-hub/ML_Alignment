"""P8: the ``schur-lm`` joint LM loop (Iteration 38 §C, L1-L7).

The loop is exercised on two models through one code path: a Level A linear
problem, where the residual ``r + A du + B dtheta`` is exact so every gain
ratio must be one, and the Level B helix/plane problem with controlled
geometry injections.  Terminal states follow Iteration 37 §C.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn
import ml_alignment_autograd as M
import test_joint_schur_constraints as constraints

DTYPE = torch.float64
ALGEBRA_RTOL = 1.0e-10
LEVEL_A_TOPOLOGY = [[0, 1], [0, 1], [1, 0], [2, 3], [3, 2], [2, 3], [4], [4], [0], [1], [2], [3], [4]]
WIDE = torch.full((3,), 1.0e6, dtype=DTYPE)


class LinearModel:
    """Level A: ``values`` are ``u`` (unit scales on DOF 0-2), ``tracks`` are ``theta``."""

    def __init__(self, lin):
        self.lin = lin
        self.active_dof = lin.active_dof

    def linearize(self, values, tracks):
        return dataclasses.replace(self.lin, residual=syn.shifted_linear_residual(self.lin, values, tracks))

    def evaluate(self, values, tracks, frozen):
        residual = syn.shifted_linear_residual(self.lin, values, tracks)
        return J.evaluate_objective(residual, self.lin.component_mask, frozen, self.lin.is_ref, self.lin.is_target)

    def advance_tracks(self, tracks, delta_theta):
        return tracks + delta_theta


class AlwaysLosesComponents(LinearModel):
    def evaluate(self, values, tracks, frozen):
        return dataclasses.replace(super().evaluate(values, tracks, frozen), lost=True)


def _start(lin):
    n, k = len(lin.target_sensors), lin.geometry_jacobian.shape[-1]
    return torch.zeros((n, k), dtype=DTYPE), torch.zeros((lin.residual.shape[0], 5), dtype=DTYPE)


def _loop(lin, model=None, bounds=WIDE, **kwargs):
    values, tracks = _start(lin)
    return J.schur_lm_loop(model or LinearModel(lin), values, tracks, bounds, **kwargs)


def _dense_gauss_newton(lin):
    _jac, _res, hessian, gradient = syn.dense_joint_system(lin)
    step = -torch.linalg.solve(hessian, gradient)
    n, k = len(lin.target_sensors), lin.geometry_jacobian.shape[-1]
    return step[: n * k].reshape(n, k), step[n * k :].reshape(-1, 5)


def _config():
    return M.AlignmentRunConfig(track_mode="free", joint_solver="schur-lm")


# L1 -------------------------------------------------------------------------


def test_linear_problem_has_unit_gain_ratio_on_every_accepted_step():
    result = _loop(syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0))
    accepted = [row for row in result.trials if row["accepted"]]

    assert result.status == J.SCHUR_LM_STATUS_CONVERGED
    assert accepted
    assert all(abs(row["rho"] - 1.0) <= ALGEBRA_RTOL for row in accepted)
    assert result.stationarity.lambda2_geo <= J.SCHUR_LM_EPSILON_GEO


def test_linear_problem_lands_on_the_dense_gauss_newton_solution():
    """With ``lambda_init = lambda_min`` the first LM step is the GN step to 1e-12.

    The default ``lambda_init`` legitimately stops once less than 0.1 sigma is
    left (Iteration 37 §B), which is not a 1e-10 comparison; this isolates
    the algebra of the loop from the stopping rule.
    """
    lin = syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0)
    du, dtheta = _dense_gauss_newton(lin)
    result = _loop(lin, lambda_init=J.SCHUR_LM_LAMBDA_MIN)

    assert result.status == J.SCHUR_LM_STATUS_CONVERGED
    assert float((result.values - du).norm() / du.norm()) <= ALGEBRA_RTOL
    assert float((result.tracks - dtheta).norm() / dtheta.norm()) <= ALGEBRA_RTOL


def test_trial_log_carries_the_required_fields():
    result = _loop(syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0))
    required = {
        "iteration", "attempt", "lam", "branch", "accepted", "reject_reason", "alpha", "n_resolves",
        "n_free", "n_structurally_absent", "objective_before", "objective_after", "chi2_ref_before",
        "chi2_ref_after", "chi2_tgt_before", "chi2_tgt_after", "n_components_ref", "n_components_tgt",
        "predicted", "actual", "rho", "geo_step_norm", "track_step_norm", "target_geometry_part",
        "target_track_part",
    }
    assert required <= set(result.trials[0])
    assert {"lambda2_geo", "lambda2_trk_max", "rank_deficient", "min_scaled_eigenvalue",
            "largest_component_dof"} <= set(result.iterations[0])


# L3 -------------------------------------------------------------------------


def test_huge_damping_moves_neither_geometry_nor_tracks():
    lin = syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0)
    du, dtheta = _dense_gauss_newton(lin)
    result = _loop(lin, lambda_init=1.0e10, max_iters=1)
    first = result.trials[0]

    assert first["geo_step_norm"] <= 1.0e-8 * float(du.norm())
    assert first["track_step_norm"] <= 1.0e-8 * float(dtheta.norm())


# L4 -------------------------------------------------------------------------


def test_singular_track_block_is_a_numerical_failure():
    lin = syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0)
    track_jacobian = lin.track_jacobian.clone()
    track_jacobian[0] = 0.0
    result = _loop(dataclasses.replace(lin, track_jacobian=track_jacobian))

    assert result.status == J.SCHUR_LM_STATUS_NUMERICAL_FAILURE
    assert result.reason == "track_block_factorization"


def test_losing_components_on_every_trial_is_an_intersection_failure():
    lin = syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0)
    result = _loop(lin, model=AlwaysLosesComponents(lin))

    assert result.status == J.SCHUR_LM_STATUS_INTERSECTION_FAILED
    assert len(result.trials) == 1 + J.SCHUR_LM_MAX_RETRIES
    assert all(row["reject_reason"] == "component_loss" for row in result.trials)
    assert not any(row["accepted"] for row in result.trials)
    values, tracks = _start(lin)
    assert torch.equal(result.values, values) and torch.equal(result.tracks, tracks)


def test_outer_budget_is_budget_exhausted():
    result = _loop(syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=0), lambda_init=1.0, max_iters=1)

    assert not result.stationarity.stationary
    assert result.status == J.SCHUR_LM_STATUS_BUDGET_EXHAUSTED
    assert result.reason == "outer_iters"


def test_optimum_on_a_bound_converges_with_that_dof_on_the_bound():
    lin = syn.random_linearization([[0], [0], [0], [0]], 1, seed=7)
    _jac, _res, hessian, gradient = syn.dense_joint_system(lin)
    bounds = torch.linalg.solve(hessian, gradient)[:3].abs() * torch.tensor([0.5, 0.5, 10.0], dtype=DTYPE)
    _value, _x, side = constraints._kkt_oracle(lin, bounds)

    result = _loop(lin, bounds=bounds)

    assert result.status == J.SCHUR_LM_STATUS_CONVERGED
    on_bound = side != 0
    assert bool(on_bound.any())
    assert torch.allclose(result.values.flatten()[on_bound], side[on_bound] * bounds[on_bound], rtol=0.0, atol=1e-12)
    assert any(row["alpha"] < 1.0 for row in result.trials if row["accepted"])


def test_weak_geometry_direction_is_rank_deficient_and_never_converged():
    lin = syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=2)
    jacobian = lin.geometry_jacobian.clone()
    on_block = (lin.block_index == 4).unsqueeze(-1)
    jacobian[..., 2] = torch.where(on_block, jacobian[..., 0], jacobian[..., 2])
    result = _loop(dataclasses.replace(lin, geometry_jacobian=jacobian))

    assert result.status == J.SCHUR_LM_STATUS_RANK_DEFICIENT
    assert result.stationarity.rank_deficient and result.stationarity.stationary


# L2 / L5 / L6 (Level B, real helix and intersection code) ---------------------

INJECTION = [[0.05, 0.03, 0, 0, 0, 2e-4], [-0.04, 0.05, 0, 0, 0, -1e-4], [0.03, -0.02, 0, 0, 0, 1.5e-4]]
ACTIVE = (0, 1, 5)


def _level_b_run(sign=1.0, noise_seed=None, **kwargs):
    injection = sign * torch.tensor(INJECTION, dtype=DTYPE)
    problem = syn.shared_topology_problem(injection=injection, noise_seed=noise_seed)
    tracks = syn.ref_only_fit(problem)
    result = J.run_schur_lm_alignment(
        problem.batches, problem.geometry, _config(), initial_track_params=tracks, **kwargs
    )
    return problem, result


def _estimated(problem, result):
    deltas = torch.as_tensor(result["predicted_deltas"], dtype=DTYPE)
    return deltas[problem.n_ref :][:, list(ACTIVE)], problem.true_alignment[problem.n_ref :][:, list(ACTIVE)]


@pytest.mark.slow
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_noiseless_injection_is_recovered(sign):
    problem, result = _level_b_run(sign)
    estimated, injected = _estimated(problem, result)

    assert result["schur_lm_summary"]["status"] == J.SCHUR_LM_STATUS_CONVERGED
    assert float((estimated - injected).norm()) <= 0.01 * float(injected.norm())


@pytest.mark.slow
def test_positive_and_negative_injections_give_opposite_corrections():
    problem_p, result_p = _level_b_run(1.0)
    problem_n, result_n = _level_b_run(-1.0)
    plus, _ = _estimated(problem_p, result_p)
    minus, _ = _estimated(problem_n, result_n)

    assert bool((torch.sign(plus) == -torch.sign(minus)).all())
    assert float((plus + minus).norm()) <= 0.01 * float(plus.norm())


@pytest.mark.slow
def test_noisy_injection_converges_with_monotone_objective_and_no_component_loss():
    problem, result = _level_b_run(1.0, noise_seed=3)
    accepted = [row for row in result["schur_lm_trials"] if row["accepted"]]

    assert result["schur_lm_summary"]["status"] == J.SCHUR_LM_STATUS_CONVERGED
    assert accepted
    assert all(row["objective_after"] < row["objective_before"] for row in accepted)
    assert all(row["reject_reason"] != "component_loss" for row in accepted)
    # With 10 tracks the statistical error exceeds a 0.05 mm injection, so the
    # check is against the exact joint optimum: lambda2 <= 1e-2 leaves at most
    # 0.1 sigma of Newton step, and no single DOF can be further off than that.
    estimated, _injected = _estimated(problem, result)
    optimum, sigma = _dense_joint_optimum(problem)
    assert float(((estimated - optimum).abs() / sigma).max()) <= J.SCHUR_LM_EPSILON_GEO ** 0.5


def _dense_joint_optimum(problem, iterations=15):
    """Undamped dense joint GN to its fixed point, and the marginal sigma of each DOF there."""
    tracks = syn.ref_only_fit(problem)
    alignment = torch.zeros_like(problem.true_alignment)
    for _ in range(iterations):
        lin = J.linearize_tracks(tracks, problem.batches, problem.geometry, alignment, ACTIVE)
        blocks = J.normal_blocks(lin)
        system = J.schur_reduce(blocks, 0.0)
        du = J.solve_geometry_dense(system)
        alignment[lin.target_sensors] += du @ J.geometry_embedding(ACTIVE).T
        tracks = J.commit_track_step(tracks, system.back_substitute(du))
    scales = torch.tensor([J.GEOMETRY_SCALES[dof] for dof in ACTIVE], dtype=DTYPE)
    sigma = torch.linalg.inv(system.to_dense()).diagonal().reshape(-1, len(ACTIVE)).sqrt() * scales
    return alignment[problem.n_ref :][:, list(ACTIVE)], sigma


def test_objective_at_the_linearisation_point_is_half_the_residual_norm():
    problem = syn.shared_topology_problem(injection=INJECTION, noise_seed=3)
    tracks = syn.ref_only_fit(problem)
    model = J.HelixSchurModel(problem.batches, problem.geometry, ACTIVE)
    values = torch.zeros((len(model.target_sensors), 3), dtype=DTYPE)
    lin = model.linearize(values, tracks)

    evaluation = model.evaluate(values, tracks, lin.component_mask)

    assert evaluation.objective == pytest.approx(0.5 * float(lin.residual.square().sum()), rel=1e-12)
    assert not evaluation.lost and evaluation.finite
    assert evaluation.n_components_ref + evaluation.n_components_tgt == int(lin.component_mask.sum())


@pytest.mark.slow
def test_lambda_init_scan_reaches_the_same_geometry():
    """Iteration 37 §B grid; equivalence 2 sqrt(eps_geo) sigma per DOF (Iteration 39).

    Each converged run is within sqrt(eps_geo) of the optimum in the S metric,
    hence within sqrt(eps_geo) sigma_i per DOF, and two runs within twice that.
    """
    runs = {lam: _level_b_run(1.0, noise_seed=3, lambda_init=lam) for lam in J.SCHUR_LM_LAMBDA_INIT_SCAN}
    estimates = {}
    for lam, (problem, result) in runs.items():
        assert result["schur_lm_summary"]["status"] == J.SCHUR_LM_STATUS_CONVERGED, lam
        estimates[lam] = _estimated(problem, result)[0]
    _optimum, sigma = _dense_joint_optimum(problem)
    reference = estimates[J.SCHUR_LM_LAMBDA_INIT]
    tolerance = 2.0 * J.SCHUR_LM_EPSILON_GEO ** 0.5
    for lam, estimate in estimates.items():
        difference = (estimate - reference).abs()
        in_sigma = float((difference / sigma).max())
        print(f"\nL5 lambda_init={lam:g}: max |d tx,ty| = {float(difference[:, :2].max()):.3g} mm, "
              f"max |d rz| = {float(difference[:, 2].max()):.3g} rad, max |d|/sigma = {in_sigma:.3g}")
        assert in_sigma <= tolerance, lam


def test_loc0_only_target_has_a_structurally_absent_ty():
    """Without rz, ty on a LOC0-only target has exactly zero information at every point."""
    truth = syn.make_tracks(10, seed=1)
    problem = syn.make_problem(
        truth,
        targets=((450.0, 0.0), (550.0, 0.0)),
        loc1_targets=(0,),
        injection=[[0.05, 0.04, 0, 0, 0, 0], [-0.03, 0.02, 0, 0, 0, 0]],
    )
    result = J.run_schur_lm_alignment(
        problem.batches, problem.geometry, _config(), active_dof=(0, 1),
        initial_track_params=syn.ref_only_fit(problem),
    )
    deltas = torch.as_tensor(result["predicted_deltas"], dtype=DTYPE)[problem.n_ref :]
    summary = result["schur_lm_summary"]

    assert summary["status"] == J.SCHUR_LM_STATUS_CONVERGED
    assert summary["structurally_absent"] == [[problem.n_ref, 1]]
    assert float(deltas[0, 1]) == 0.0
    assert float(deltas[0, 0]) == pytest.approx(0.05, rel=1e-2)
    assert deltas[1, :2].tolist() == pytest.approx([-0.03, 0.02], rel=1e-2)


# L7 -------------------------------------------------------------------------


def test_cli_accepts_schur_lm_and_dispatches_to_it():
    args = M.build_argument_parser().parse_args(["--track-mode", "free", "--joint-solver", "schur-lm"])
    config = M.alignment_config_from_args(args)

    assert config.joint_solver == "schur-lm"
    assert M.select_joint_solver(config) is J.run_schur_lm_alignment


def test_block_lm_stays_the_default_solver():
    config = M.alignment_config_from_args(M.build_argument_parser().parse_args(["--track-mode", "free"]))
    assert config.joint_solver == "block-lm"
    assert M.select_joint_solver(config) is M.run_block_lm_alignment


def test_schur_lm_summary_is_retained_in_the_run_summary():
    source = inspect.getsource(M)
    assert '"schur_lm_summary"' in source


def test_schur_module_still_has_no_legacy_step_clipping():
    assert "LM_MAX_STEP" not in inspect.getsource(J)
    assert "lm_max_step" not in inspect.getsource(J)
