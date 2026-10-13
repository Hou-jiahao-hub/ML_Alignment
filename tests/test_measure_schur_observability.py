"""Schur P9 (Iteration 42): Schur-system observability and reference transfer."""

from __future__ import annotations

import dataclasses
import importlib
import math

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn
import ml_alignment_autograd as M

DTYPE = torch.float64
ACTIVE = (0, 1, 5)
SCALES = torch.tensor([1.0, 1.0, 0.01], dtype=DTYPE)
UNIT = torch.ones(3, dtype=DTYPE)


def _module():
    return importlib.import_module("scripts.analysis.measure_schur_observability")


def _observability():
    return importlib.import_module("scripts.analysis.measure_alignment_observability")


def _level_a(topology=((0, 1), (0, 1), (1, 2), (2,), (0,), (0, 1, 2)), n_blocks=3, seed=4):
    return syn.random_linearization([list(row) for row in topology], n_blocks, seed=seed)


def _without_track_coupling(lin):
    """``B = 0`` on target rows: ``G = 0`` and ``Gamma_tgt = 0``, so ``S = C``."""
    track_jacobian = lin.track_jacobian * (~lin.is_target)[..., None, None]
    return dataclasses.replace(lin, track_jacobian=track_jacobian)


# S1: dense S, diagonal blocks, retention ----------------------------------------


def test_dense_schur_matches_the_oracle_assembly():
    module = _module()
    system = J.schur_reduce(J.normal_blocks(_level_a()), 0.0)
    assert torch.allclose(module.dense_schur(system), system.to_dense(), rtol=0.0, atol=1e-12)


def test_without_track_coupling_schur_equals_fixed_track():
    module = _module()
    blocks = J.normal_blocks(_without_track_coupling(_level_a()))
    system = J.schur_reduce(blocks, 0.0)
    dense = module.dense_schur(system)
    information, _gradient = blocks.geometry_information()
    s_ii = module.diagonal_blocks(dense, 3)
    assert torch.allclose(s_ii, information, rtol=1e-12, atol=1e-12)
    assert torch.allclose(module.information_retention(s_ii, information), torch.ones(3, dtype=DTYPE))
    sigma_c = _observability().statistical_sigma(information, UNIT)
    assert torch.allclose(module.marginal_sigma(dense, UNIT), sigma_c, rtol=1e-10)


def test_marginal_sigma_is_the_scaled_diagonal_of_the_global_inverse():
    module = _module()
    dense = J.schur_reduce(J.normal_blocks(_level_a()), 0.0).to_dense()
    expected = torch.linalg.inv(dense).diagonal().sqrt().reshape(3, 3) * SCALES
    assert torch.allclose(module.marginal_sigma(dense, SCALES), expected, rtol=1e-12)


# S2: the information-loss check ----------------------------------------------


def test_loewner_check_passes_on_a_true_schur_reduction():
    module = _module()
    blocks = J.normal_blocks(_level_a())
    dense = module.dense_schur(J.schur_reduce(blocks, 0.0))
    information, _gradient = blocks.geometry_information()
    check = module.loewner_check(information, module.diagonal_blocks(dense, 3), dense, UNIT)
    assert check["passed"]
    assert check["min_retention_gap"] >= -1e-9
    assert check["n_sigma_order_violations"] == 0
    retention = module.information_retention(module.diagonal_blocks(dense, 3), information)
    assert bool((retention < 1.0).all())


def test_loewner_check_fails_when_schur_claims_more_information():
    module = _module()
    blocks = J.normal_blocks(_level_a())
    information, _gradient = blocks.geometry_information()
    inflated = 1.5 * information
    dense = torch.block_diag(*inflated)
    check = module.loewner_check(information, inflated, dense, UNIT)
    assert not check["passed"]
    assert check["min_retention_gap"] < 0.0


# S3: coupling and the global spectrum ------------------------------------------


def test_coupling_strength_is_the_frobenius_norm_over_the_trace_geometric_mean():
    module = _module()
    system = J.schur_reduce(J.normal_blocks(_level_a()), 0.0)
    dense = system.to_dense().reshape(3, 3, 3, 3).permute(0, 2, 1, 3)
    rows, cols, rho = module.coupling_strength(system)
    assert len(rows) == 6
    assert bool((rows != cols).all())
    for i, j, value in zip(rows.tolist(), cols.tolist(), rho.tolist()):
        expected = torch.linalg.matrix_norm(dense[i, j]) / torch.sqrt(
            dense[i, i].trace() * dense[j, j].trace()
        )
        assert value == pytest.approx(float(expected), rel=1e-12)


def test_sensors_without_a_shared_track_get_no_coupling():
    module = _module()
    system = J.schur_reduce(J.normal_blocks(_level_a(topology=((0,), (1,), (0,), (1,)), n_blocks=2)), 0.0)
    rows, _cols, rho = module.coupling_strength(system)
    assert len(rows) == 0 and len(rho) == 0


def test_global_spectrum_of_a_block_diagonal_system_is_the_union_of_local_spectra():
    module = _module()
    observability = _observability()
    blocks = J.normal_blocks(_level_a(topology=((0,), (1,), (0,), (1,), (0,), (1,)), n_blocks=2))
    dense = module.dense_schur(J.schur_reduce(blocks, 0.0))
    spectrum = module.global_jacobi_spectrum(dense, 3, n_lowest=4)
    local = observability.jacobi_spectrum(module.diagonal_blocks(dense, 3)).eigenvalues
    expected = torch.sort(local.flatten()).values
    assert torch.allclose(spectrum["eigenvalues"], expected, rtol=1e-10)
    participation = spectrum["participation"]
    assert participation.shape == (4, 2)
    assert torch.allclose(participation.sum(dim=1), torch.ones(4, dtype=DTYPE), atol=1e-12)
    # Each local mode lives on its own sensor.
    assert torch.allclose(participation.max(dim=1).values, torch.ones(4, dtype=DTYPE), atol=1e-10)
    assert spectrum["n_weak"] == int((expected <= J.SCHUR_LM_RANK_TOL).sum())


# S4: reference transfer, algebra ---------------------------------------------


def test_schur_reference_transfer_equals_the_dense_joint_response():
    module = _module()
    blocks = J.normal_blocks(_level_a())
    dense = module.dense_schur(J.schur_reduce(blocks, 0.0))
    generator = torch.Generator().manual_seed(9)
    dbeta = torch.randn(blocks.beta_ref.shape, generator=generator, dtype=DTYPE)
    moved = dataclasses.replace(blocks, beta_ref=blocks.beta_ref + dbeta)
    before, _ = J.solve_joint_dense(blocks, 0.0)
    after, _ = J.solve_joint_dense(moved, 0.0)
    transfer = module.schur_reference_transfer(blocks, torch.linalg.inv(dense), dbeta, SCALES)
    assert torch.allclose(transfer, (after - before) * SCALES, rtol=1e-9, atol=1e-12)


def test_fixed_track_reference_transfer_follows_the_ref_only_refit():
    module = _module()
    blocks = J.normal_blocks(_level_a())
    generator = torch.Generator().manual_seed(10)
    dbeta = torch.randn(blocks.beta_ref.shape, generator=generator, dtype=DTYPE)
    dtheta = -torch.linalg.solve(blocks.gamma_ref, dbeta.unsqueeze(-1)).squeeze(-1)
    information, _gradient = blocks.geometry_information()
    dc = torch.zeros((3, 3), dtype=DTYPE).index_add_(
        0, blocks.pair_block, (blocks.G @ dtheta[blocks.pair_track].unsqueeze(-1)).squeeze(-1)
    )
    expected = -torch.linalg.solve(information, dc.unsqueeze(-1)).squeeze(-1) * SCALES
    transfer = module.fixed_track_reference_transfer(blocks, dbeta, SCALES)
    assert torch.allclose(transfer, expected, rtol=1e-10, atol=1e-14)


# S5: reference transfer on the helix model ---------------------------------------


def _helix_problem():
    truth = syn.make_tracks(40, seed=11)
    return syn.make_problem(truth, targets=((450.0, 0.0), (550.0, 15.0)))


def _reference_rows(problem):
    rows = torch.zeros(len(problem.geometry.centers), dtype=torch.bool)
    rows[: problem.n_ref] = True
    return rows


def test_reference_dbeta_reproduces_the_m1_fixed_track_oracle():
    module = _module()
    observability = _observability()
    problem = _helix_problem()
    tracks = syn.ref_only_fit(problem)
    rows = _reference_rows(problem)
    delta = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    lin = J.linearize_tracks(tracks, problem.batches, problem.geometry, None, ACTIVE)
    blocks = J.normal_blocks(lin)
    dbeta = module.reference_dbeta(lin, tracks, problem.batches, problem.geometry, None, rows, delta)
    transfer = module.fixed_track_reference_transfer(blocks, dbeta, SCALES)
    oracle = observability.linear_transfer_oracle(
        tracks, problem.batches, problem.geometry, rows, delta, ACTIVE, SCALES
    )
    assert torch.allclose(transfer, oracle, rtol=1e-10, atol=1e-14)


def test_schur_reference_transfer_predicts_the_full_schur_lm_response():
    module = _module()
    observability = _observability()
    problem = _helix_problem()
    tracks = syn.ref_only_fit(problem)
    rows = _reference_rows(problem)
    amplitude = 0.01
    delta = torch.tensor([0.0, amplitude, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    lin = J.linearize_tracks(tracks, problem.batches, problem.geometry, None, ACTIVE)
    blocks = J.normal_blocks(lin)
    dense = module.dense_schur(J.schur_reduce(blocks, 0.0))
    dbeta = module.reference_dbeta(lin, tracks, problem.batches, problem.geometry, None, rows, delta)
    linear = module.schur_reference_transfer(blocks, torch.linalg.inv(dense), dbeta, SCALES)

    responses = []
    for sign in (1.0, -1.0):
        geometry = observability.inject_rows(problem.geometry, rows, sign * delta)
        result = J.run_schur_lm_alignment(
            problem.batches, geometry, M.AlignmentRunConfig(), ACTIVE, initial_track_params=tracks
        )
        assert result["schur_lm_summary"]["status"] == J.SCHUR_LM_STATUS_CONVERGED
        responses.append(
            torch.as_tensor(result["predicted_deltas"])[problem.geometry.target_mask][:, list(ACTIVE)]
        )
    nonlinear = 0.5 * (responses[0] - responses[1])
    assert torch.allclose(nonlinear, linear, rtol=2e-2, atol=2e-2 * amplitude)
    assert float(linear.abs().max()) > 0.1 * amplitude


# S6: diagnostic helpers ----------------------------------------------------------


def test_level_c_subset_is_deterministic_and_event_based():
    module = _module()
    events = torch.tensor([0, 32, 1, 64, 33, 0, 31])
    index = module.level_c_rows(events)
    assert index.tolist() == [0, 1, 3, 5]


def test_lambda_equivalence_uses_point_two_sigma_per_dof():
    module = _module()
    reference = torch.zeros((2, 3), dtype=DTYPE)
    sigma = torch.tensor([[1.0, 2.0, 0.1], [1.0, 1.0, 1.0]], dtype=DTYPE)
    inside = torch.tensor([[0.19, -0.39, 0.019], [0.0, 0.0, 0.0]], dtype=DTYPE)
    outside = inside.clone()
    outside[1, 2] = 0.21
    assert module.lambda_equivalence(inside, reference, sigma)["equivalent"]
    result = module.lambda_equivalence(outside, reference, sigma)
    assert not result["equivalent"]
    assert result["max_over_sigma"] == pytest.approx(0.21)


# S7: P10 amplitude scan (Iteration 44) --------------------------------------------


def test_injection_recovery_is_one_for_an_exact_undo():
    module = _module()
    control = torch.tensor([[0.1, -0.2, 0.003], [0.0, 0.05, -0.001]], dtype=DTYPE)
    injection = torch.tensor([0.25, 0.25, 0.01], dtype=DTYPE)
    recovered = control - injection
    masks = {"a": torch.tensor([True, False]), "b": torch.tensor([False, True])}
    result = module.injection_recovery(recovered, control, injection, masks)
    for layer in masks:
        assert result[layer] == pytest.approx([1.0, 1.0, 1.0])


def test_trial_summary_reads_first_step_and_rejections():
    module = _module()
    trials = [
        {"iteration": 0, "attempt": 0, "accepted": False, "reject_reason": "insufficient_decrease", "rho": -0.5,
         "alpha": 1.0, "n_resolves": 0, "n_free": 10, "predicted": 2.0, "actual": -1.0,
         "target_geometry_part": 3.0, "target_track_part": 1.0},
        {"iteration": 0, "attempt": 1, "accepted": True, "reject_reason": None, "rho": 0.9,
         "alpha": 0.5, "n_resolves": 1, "n_free": 8, "predicted": 1.0, "actual": 0.9,
         "target_geometry_part": 2.0, "target_track_part": 0.5},
    ]
    summary = module.trial_summary(trials)
    assert summary["first_rho"] == pytest.approx(-0.5)
    assert summary["first_accepted_rho"] == pytest.approx(0.9)
    assert summary["rejections"] == {"insufficient_decrease": 1}
    assert summary["min_alpha"] == pytest.approx(0.5)
    assert summary["min_n_free"] == 8
    assert summary["max_resolves"] == 1


def test_transfer_ratio_handles_a_zero_fixed_track_transfer():
    module = _module()
    assert module.transfer_ratio(2.0, 4.0) == pytest.approx(0.5)
    assert module.transfer_ratio(1.0e-6, 0.0) == math.inf
    assert math.isnan(module.transfer_ratio(0.0, 0.0))
