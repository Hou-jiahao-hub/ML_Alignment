"""Roadmap M1 (Iteration 40): fixed-track observability on synthetic Level B problems."""

from __future__ import annotations

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


def _module():
    return importlib.import_module("scripts.analysis.measure_alignment_observability")


def _problem(noise_seed=None, n_tracks=40, loc1_targets=()):
    truth = syn.make_tracks(n_tracks, seed=11)
    return syn.make_problem(
        truth,
        targets=((450.0, 0.0), (550.0, 15.0)),
        loc1_targets=loc1_targets,
        noise_seed=noise_seed,
    )


def _config(**overrides):
    return M.AlignmentRunConfig(lm_max_iters=60, **overrides)


def _random_spd(n, k, seed):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn((n, k, k), generator=generator, dtype=DTYPE)
    scale = torch.exp(torch.randn((n, k), generator=generator, dtype=DTYPE))
    matrix = base @ base.transpose(-1, -2) + 0.1 * torch.eye(k, dtype=DTYPE)
    return matrix * scale[:, :, None] * scale[:, None, :]


# O1: spectrum ---------------------------------------------------------------


def test_jacobi_spectrum_matches_an_independent_eigendecomposition():
    module = _module()
    information = _random_spd(6, 3, seed=1)
    spectrum = module.jacobi_spectrum(information)
    for sensor in range(6):
        d = information[sensor].diagonal().rsqrt()
        expected, vectors = torch.linalg.eigh(information[sensor] * d[:, None] * d[None, :])
        assert torch.allclose(spectrum.eigenvalues[sensor], expected, rtol=1e-12, atol=0.0)
        overlap = (spectrum.eigenvectors[sensor].T @ vectors).abs()
        assert torch.allclose(overlap, torch.eye(3, dtype=DTYPE), atol=1e-10)
    assert not spectrum.structurally_absent.any()
    assert spectrum.observable.all()


def test_structurally_absent_dof_is_excluded_and_padded_last():
    module = _module()
    information = _random_spd(2, 3, seed=2)
    information[1, 1, :] = 0.0
    information[1, :, 1] = 0.0
    spectrum = module.jacobi_spectrum(information)
    assert spectrum.structurally_absent[1].tolist() == [False, True, False]
    assert math.isnan(float(spectrum.eigenvalues[1, 2]))
    assert not bool(spectrum.observable[1, 2])
    assert float(spectrum.eigenvectors[1, 1].abs().max()) == 0.0
    sub = information[1][[0, 2]][:, [0, 2]]
    d = sub.diagonal().rsqrt()
    expected = torch.linalg.eigvalsh(sub * d[:, None] * d[None, :])
    assert torch.allclose(spectrum.eigenvalues[1, :2], expected, rtol=1e-12, atol=0.0)


def test_loc0_only_target_ty_is_absent_at_rz_zero_and_weak_at_rz_nonzero():
    module = _module()
    problem = _problem(loc1_targets=(0,))
    tracks = syn.ref_only_fit(problem)
    information, _gradient, sensors = module.target_information(
        tracks, problem.batches, problem.geometry, None, ACTIVE
    )
    spectrum = module.jacobi_spectrum(information)
    first = int(torch.nonzero(sensors == problem.n_ref)[0])
    assert spectrum.structurally_absent[first].tolist() == [False, True, False]

    alignment = torch.zeros((len(problem.geometry.centers), 6), dtype=DTYPE)
    alignment[problem.n_ref, 5] = 0.02
    information, _gradient, _sensors = module.target_information(
        tracks, problem.batches, problem.geometry, alignment, ACTIVE
    )
    spectrum = module.jacobi_spectrum(information)
    assert not spectrum.structurally_absent[first].any()
    assert int((~spectrum.observable[first]).sum()) == 1


def test_target_only_linearisation_equals_the_full_one_on_target_rows():
    module = _module()
    problem = _problem(noise_seed=3)
    tracks = syn.ref_only_fit(problem)
    information, gradient, sensors = module.target_information(
        tracks, problem.batches, problem.geometry, None, ACTIVE
    )
    full = J.normal_blocks(J.linearize_tracks(tracks, problem.batches, problem.geometry, None, ACTIVE))
    expected_information, expected_gradient = full.geometry_information()
    assert torch.equal(sensors, J.linearize_tracks(
        tracks, problem.batches, problem.geometry, None, ACTIVE
    ).target_sensors)
    assert torch.allclose(information, expected_information, rtol=1e-12, atol=1e-14)
    assert torch.allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-14)


def test_statistical_sigma_is_the_physical_marginal_of_the_inverse():
    module = _module()
    information = _random_spd(3, 3, seed=4)
    sigma = module.statistical_sigma(information, SCALES)
    for sensor in range(3):
        expected = torch.linalg.inv(information[sensor]).diagonal().sqrt() * SCALES
        assert torch.allclose(sigma[sensor], expected, rtol=1e-12)
    information[2, 0, :] = 0.0
    information[2, :, 0] = 0.0
    assert torch.isnan(module.statistical_sigma(information, SCALES)[2]).all()


def test_block_lm_weak_flag_uses_the_relative_unit_scaled_floor():
    module = _module()
    information = torch.diag_embed(torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 5.0e-5]], dtype=DTYPE))
    assert module.block_lm_weak(information).tolist() == [False, True]


# O2: in-memory injection ----------------------------------------------------


def test_injection_moves_only_the_selected_rows_in_the_solver_convention():
    module = _module()
    problem = _problem()
    rows = problem.geometry.target_mask.clone()
    delta = torch.tensor([0.5, -0.2, 0.0, 0.0, 0.0, 0.02], dtype=DTYPE)
    injected = module.inject_rows(problem.geometry, rows, delta)
    expected_c, expected_r = M.apply_alignment_torch(
        problem.geometry.centers[rows], problem.geometry.rotations[rows], delta.expand(int(rows.sum()), 6)
    )
    assert torch.allclose(injected.centers[rows], expected_c, rtol=0.0, atol=1e-12)
    assert torch.allclose(injected.rotations[rows], expected_r, rtol=0.0, atol=1e-12)
    assert torch.equal(injected.centers[~rows], problem.geometry.centers[~rows])
    assert torch.equal(injected.rotations[~rows], problem.geometry.rotations[~rows])
    assert torch.equal(problem.geometry.centers, _problem().geometry.centers)


def test_injection_accepts_the_surface_indices_reference_layers_returns():
    module = _module()
    problem = _problem()
    rows = module.reference_layers(problem.batches)[(10, 2)]
    mask = torch.zeros(len(problem.geometry.centers), dtype=torch.bool)
    mask[rows] = True
    delta = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 1.0e-4], dtype=DTYPE)
    by_index = module.inject_rows(problem.geometry, rows, delta)
    by_mask = module.inject_rows(problem.geometry, mask, delta)
    assert torch.equal(by_index.centers, by_mask.centers)
    assert torch.equal(by_index.rotations, by_mask.rotations)


def test_reference_layers_group_non_target_surfaces_by_volume_and_layer():
    module = _module()
    problem = _problem()
    layers = module.reference_layers(problem.batches)
    assert list(layers) == [(10, 2)]
    assert layers[(10, 2)].tolist() == list(range(problem.n_ref))


# O3: response matrix and mode slopes -----------------------------------------


def test_mode_slopes_identity_response_gives_unit_slopes():
    module = _module()
    information = _random_spd(4, 3, seed=5)
    spectrum = module.jacobi_spectrum(information)
    response = torch.eye(3, dtype=DTYPE).expand(4, 3, 3)
    slopes = module.mode_slopes(response, information, spectrum, SCALES)
    assert torch.allclose(slopes, torch.ones_like(slopes), rtol=0.0, atol=1e-12)


def test_mode_slopes_isolate_a_half_recovered_direction():
    module = _module()
    information = torch.diag_embed(torch.tensor([[4.0, 1.0, 9.0]], dtype=DTYPE))
    spectrum = module.jacobi_spectrum(information)
    response = torch.diag_embed(torch.tensor([[1.0, 0.5, 1.0]], dtype=DTYPE))
    slopes = module.mode_slopes(response, information, spectrum, SCALES)
    assert sorted(slopes[0].tolist()) == pytest.approx([0.5, 1.0, 1.0], abs=1e-12)


def test_response_matrix_from_plus_minus_runs():
    module = _module()
    amplitudes = torch.tensor([0.5, 0.5, 0.02], dtype=DTYPE)
    truth = torch.tensor([[1.0, 0.1, 0.0], [0.0, 0.9, 0.0], [0.2, 0.0, 1.0]], dtype=DTYPE)
    plus = {d: (-amplitudes[d] * truth[:, d]).expand(2, 3) for d in range(3)}
    minus = {d: (amplitudes[d] * truth[:, d]).expand(2, 3) for d in range(3)}
    response = module.response_matrix(plus, minus, amplitudes)
    assert torch.allclose(response, truth.expand(2, 3, 3), atol=1e-14)


@pytest.mark.slow
def test_block_lm_recovers_plus_minus_target_injections_with_unit_response():
    module = _module()
    problem = _problem(noise_seed=3)
    tracks = syn.ref_only_fit(problem)
    amplitudes = torch.tensor([0.5, 0.5, 0.02], dtype=DTYPE)
    plus, minus = {}, {}
    for column, dof in enumerate(ACTIVE):
        for sign, store in ((1.0, plus), (-1.0, minus)):
            delta = torch.zeros(6, dtype=DTYPE)
            delta[dof] = sign * amplitudes[column]
            geometry = module.inject_rows(problem.geometry, problem.geometry.target_mask, delta)
            result = M.run_block_lm_alignment(
                problem.batches, geometry, _config(), ACTIVE, initial_track_params=tracks
            )
            store[column] = torch.as_tensor(result["predicted_deltas"])[problem.geometry.target_mask][:, list(ACTIVE)]
    response = module.response_matrix(plus, minus, amplitudes)
    information, _gradient, _sensors = module.target_information(
        tracks, problem.batches, problem.geometry, None, ACTIVE
    )
    spectrum = module.jacobi_spectrum(information)
    slopes = module.mode_slopes(response, information, spectrum, SCALES)
    assert torch.allclose(slopes, torch.ones_like(slopes), atol=1e-3)


# O4: reference-layer linear response ------------------------------------------


@pytest.mark.slow
def test_reference_linear_transfer_matches_independent_oracle_and_full_block_lm():
    module = _module()
    problem = _problem()
    tracks = syn.ref_only_fit(problem)
    information, gradient, _sensors = module.target_information(
        tracks, problem.batches, problem.geometry, None, ACTIVE
    )
    rows = torch.zeros(len(problem.geometry.centers), dtype=torch.bool)
    rows[: problem.n_ref] = True
    amplitude = 0.01
    delta = torch.tensor([amplitude, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=DTYPE)

    transfers, nonlinear = [], []
    for sign in (1.0, -1.0):
        geometry = module.inject_rows(problem.geometry, rows, sign * delta)
        moved = module.refit_tracks(problem.batches, geometry, tracks)
        _info, moved_gradient, _s = module.target_information(
            moved, problem.batches, problem.geometry, None, ACTIVE
        )
        transfers.append(module.linear_transfer(information, gradient, moved_gradient, SCALES))
        result = M.run_block_lm_alignment(
            problem.batches, geometry, _config(), ACTIVE, initial_track_params=moved
        )
        nonlinear.append(
            torch.as_tensor(result["predicted_deltas"])[problem.geometry.target_mask][:, list(ACTIVE)]
        )
    linear = 0.5 * (transfers[0] - transfers[1])
    full = 0.5 * (nonlinear[0] - nonlinear[1])

    oracle = module.linear_transfer_oracle(
        tracks, problem.batches, problem.geometry, rows, delta, ACTIVE, SCALES
    )
    assert torch.allclose(linear, oracle, rtol=1e-3, atol=1e-3 * amplitude)
    assert torch.allclose(full, linear, rtol=2e-2, atol=2e-2 * amplitude)
    # The unrotated target sees a coherent reference shift along its own u axis one for one.
    assert float(linear[0, 0]) == pytest.approx(amplitude, rel=1e-3)


# O5: event partition -----------------------------------------------------------


def test_event_partition_is_deterministic_disjoint_and_complete():
    module = _module()
    events = torch.tensor([7, 3, 3, 8, 0, 1, 5, 12, 7], dtype=torch.long)
    parts = module.event_partition(events, 4)
    assert len(parts) == 4
    joined = torch.sort(torch.cat(parts)).values
    assert joined.tolist() == list(range(len(events)))
    for index, part in enumerate(parts):
        assert bool((events[part] % 4 == index).all())
    again = module.event_partition(events, 4)
    assert all(torch.equal(a, b) for a, b in zip(parts, again))


# O6: bootstrap and truth -------------------------------------------------------


def test_bootstrap_mean_ci_is_reproducible_and_degenerate_for_constants():
    module = _module()
    assert module.bootstrap_mean_ci(torch.full((50,), 1.0, dtype=DTYPE)) == (1.0, 1.0, 1.0)
    generator = torch.Generator().manual_seed(9)
    values = 1.0 + 0.1 * torch.randn(2000, generator=generator, dtype=DTYPE)
    first = module.bootstrap_mean_ci(values)
    assert first == module.bootstrap_mean_ci(values)
    assert first[1] < 1.0 < first[2]
    assert first[1] < first[0] < first[2]


def test_paired_rms_ratio_ci_is_one_for_identical_arms():
    module = _module()
    errors = torch.linspace(-1.0, 1.0, 101, dtype=DTYPE)
    assert module.paired_rms_ratio_ci(errors, errors) == (1.0, 1.0, 1.0)
    ratio, low, high = module.paired_rms_ratio_ci(2.0 * errors, errors)
    assert ratio == pytest.approx(2.0)
    assert low == pytest.approx(2.0) and high == pytest.approx(2.0)


def test_true_corrections_follow_the_validate_against_truth_formula():
    module = _module()
    generator = torch.Generator().manual_seed(13)
    nominal_c = torch.randn((4, 3), generator=generator, dtype=DTYPE)
    nominal_r = torch.linalg.qr(torch.randn((4, 3, 3), generator=generator, dtype=DTYPE)).Q
    nominal_r = nominal_r * torch.linalg.det(nominal_r)[:, None, None]
    delta = 0.01 * torch.randn((4, 6), generator=generator, dtype=DTYPE)
    misaligned_c, misaligned_r = M.apply_alignment_torch(nominal_c, nominal_r, delta)
    truth = module.true_corrections(misaligned_c, misaligned_r, nominal_c, nominal_r)
    for row in range(4):
        r_mis = misaligned_r[row]
        expected_t = r_mis.T @ (nominal_c[row] - misaligned_c[row])
        relative = r_mis.T @ nominal_r[row]
        expected_rz = math.atan2(float(relative[1, 0]), float(relative[0, 0]))
        assert torch.allclose(truth[row, :3], expected_t, atol=1e-14)
        assert float(truth[row, 5]) == pytest.approx(expected_rz, abs=1e-14)
    corrected_c, corrected_r = M.apply_alignment_torch(misaligned_c, misaligned_r, truth)
    assert torch.allclose(corrected_c, nominal_c, atol=1e-12)
    assert torch.allclose(corrected_r, nominal_r, atol=1e-12)
