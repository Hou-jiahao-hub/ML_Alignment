import math

import pytest
import torch

from mlalign.helix.gauge import (
    assemble_track_parameters,
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    split_track_parameters,
)
from mlalign.helix.measurement_fit import (
    MeasurementFitInputs,
    _single_measurement_residual,
    fit_helix_measurements_gauss_newton,
    measurement_newton_decrement,
    measurement_residual_jacobian,
    measurement_standardized_residual,
    measurement_stationarity_report,
)
from mlalign.helix.point_fit import (
    _BATCHED_POINT_FIT_JACOBIAN,
    _single_point_fit_residual,
    fit_helix_points_gauss_newton,
    require_stationary_helix_point_fit,
)
from mlalign.helix.trajectory import (
    closest_helix_points,
    helix_point,
    helix_tangent,
    intersect_plane,
)


def _measurement_fit_case(n_tracks=2):
    dtype = torch.float64
    truth = torch.tensor(
        [
            [2.0, -1.0, 0.5, 0.2, 0.10, 2.0e-4],
            [-3.0, 4.0, -1.5, -0.4, -0.08, -1.5e-4],
        ],
        dtype=dtype,
    )[:n_tracks]
    path = torch.linspace(-250.0, 350.0, 12, dtype=dtype)
    expanded = truth[:, None, :].expand(-1, len(path), -1)
    centers = helix_point(expanded, path.expand(len(truth), -1))
    tangents = helix_tangent(expanded, path.expand(len(truth), -1))
    normals = torch.nn.functional.normalize(tangents, dim=-1)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand_as(normals)
    local0 = torch.nn.functional.normalize(torch.cross(up, normals, dim=-1), dim=-1)
    local1 = torch.cross(normals, local0, dim=-1)
    rotations = torch.stack((local0, local1, normals), dim=-1)
    measured = torch.zeros((len(truth), len(path), 2), dtype=dtype)
    measured[..., 0] = 0.01 * torch.sin(path / 31.0)
    variance = torch.full_like(measured, 4.0e-4)
    component_mask = torch.ones_like(measured, dtype=torch.bool)
    component_mask[:, -2:, 1] = False
    inputs = MeasurementFitInputs(
        measured=measured,
        variance=variance,
        component_mask=component_mask,
        centers=centers,
        rotations=rotations,
    )
    seed = truth + torch.tensor(
        [[0.5, -0.3, 0.2, 0.02, -0.01, 0.3e-4]], dtype=dtype
    )
    return inputs, truth, seed


def test_shared_helix_point_and_tangent_match_hand_checked_circle():
    track = torch.tensor(
        [[1.0, -2.0, 3.0, 0.0, 0.5, 0.01]], dtype=torch.float64
    )
    path = torch.tensor([100.0], dtype=torch.float64)

    point = helix_point(track, path)
    tangent = helix_tangent(track, path)

    assert point[0, 0].item() == pytest.approx(1.0 + 100.0 * math.sin(1.0))
    assert point[0, 1].item() == pytest.approx(-2.0 + 100.0 * (1.0 - math.cos(1.0)))
    assert point[0, 2].item() == pytest.approx(53.0)
    assert tangent[0].tolist() == pytest.approx(
        [math.cos(1.0), math.sin(1.0), 0.5]
    )


def test_measurement_residual_jacobian_matches_finite_difference():
    """Catch a Jacobian that differentiates a different residual convention."""
    inputs, _truth, seed = _measurement_fit_case(n_tracks=1)
    frame = reference_gauge_frame(seed)
    theta = pack_gauge_fixed_track_parameters(seed)

    residual = measurement_standardized_residual(theta, frame, inputs)
    jacobian = measurement_residual_jacobian(theta, frame, inputs)
    eps = 1.0e-6
    plus = theta.clone()
    minus = theta.clone()
    plus[:, 0] += eps
    minus[:, 0] -= eps
    finite_difference = (
        measurement_standardized_residual(plus, frame, inputs)
        - measurement_standardized_residual(minus, frame, inputs)
    ) / (2.0 * eps)

    assert residual.shape == (1, inputs.measured.shape[1] * 2)
    torch.testing.assert_close(
        jacobian[:, :, 0], finite_difference, atol=2.0e-6, rtol=2.0e-5
    )


def test_forward_mode_jacobians_match_reverse_mode_reference():
    """Catch forward mode differentiating through the detached closest-point
    path, or otherwise departing from the reverse-mode Jacobian it replaced."""
    inputs, truth, seed = _measurement_fit_case()
    frame = reference_gauge_frame(seed)
    theta = pack_gauge_fixed_track_parameters(seed)
    measurement_args = (
        theta, *frame, inputs.measured, inputs.variance,
        inputs.component_mask, inputs.centers, inputs.rotations,
    )
    reference = torch.func.vmap(
        torch.func.jacrev(_single_measurement_residual, argnums=0)
    )(*measurement_args)
    torch.testing.assert_close(
        measurement_residual_jacobian(theta, frame, inputs),
        reference, atol=1.0e-9, rtol=1.0e-9,
    )

    path = torch.linspace(-30.0, 80.0, 12, dtype=torch.float64)
    points = helix_point(
        truth[:, None, :].expand(-1, len(path), -1), path.expand(len(truth), -1)
    )
    points[..., 2] += 0.01 * torch.sin(path / 7.0)
    mask = torch.ones(points.shape[:2], dtype=torch.bool)
    point_args = (theta, *frame, points, mask)
    point_reference = torch.func.vmap(
        torch.func.jacrev(_single_point_fit_residual, argnums=0)
    )(*point_args)
    torch.testing.assert_close(
        _BATCHED_POINT_FIT_JACOBIAN(*point_args),
        point_reference, atol=1.0e-9, rtol=1.0e-9,
    )


def test_measurement_gauss_newton_is_chunk_exact_and_stationary():
    """Catch budget exhaustion being reported as a solved measurement fit."""
    inputs, _truth, seed = _measurement_fit_case()

    whole, whole_stats = fit_helix_measurements_gauss_newton(
        seed, inputs, chunk_rows=0
    )
    chunked, chunked_stats = fit_helix_measurements_gauss_newton(
        seed, inputs, chunk_rows=1
    )
    decrement = measurement_newton_decrement(whole, inputs, chunk_rows=1)
    report = measurement_stationarity_report(decrement)

    assert torch.equal(whole, chunked)
    assert whole_stats["convergence"]["converged"] is True
    assert chunked_stats["convergence"]["n_chunks"] == 2
    assert report["n_non_finite"] == 0
    assert report["non_stationary_row_fraction"] == 0.0
    assert (
        report["newton_decrement_max"]
        < report["stationary_threshold"]
    )

    _truncated, truncated_stats = fit_helix_measurements_gauss_newton(
        seed, inputs, max_iters=1
    )
    assert truncated_stats["convergence"]["converged"] is False


def test_measurement_fit_ignores_fully_masked_padding_hits():
    inputs, _truth, seed = _measurement_fit_case(n_tracks=1)
    base = MeasurementFitInputs(
        measured=inputs.measured[:, :-1],
        variance=inputs.variance[:, :-1],
        component_mask=inputs.component_mask[:, :-1],
        centers=inputs.centers[:, :-1],
        rotations=inputs.rotations[:, :-1],
    )
    padded_mask = inputs.component_mask.clone()
    padded_mask[:, -1] = False
    padded_centers = inputs.centers.clone()
    padded_centers[:, -1] = torch.tensor(
        [5000.0, 5000.0, 5000.0], dtype=torch.float64
    )
    padded = MeasurementFitInputs(
        measured=inputs.measured,
        variance=inputs.variance,
        component_mask=padded_mask,
        centers=padded_centers,
        rotations=inputs.rotations,
    )

    expected, _ = fit_helix_measurements_gauss_newton(seed, base)
    actual, _ = fit_helix_measurements_gauss_newton(seed, padded)

    torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=0.0)


def test_measurement_fit_rejects_non_finite_residuals():
    inputs, _truth, seed = _measurement_fit_case(n_tracks=1)
    measured = inputs.measured.clone()
    measured[0, 0, 0] = float("nan")
    invalid = MeasurementFitInputs(
        measured=measured,
        variance=inputs.variance,
        component_mask=inputs.component_mask,
        centers=inputs.centers,
        rotations=inputs.rotations,
    )

    with pytest.raises(FloatingPointError, match="Non-finite initial"):
        fit_helix_measurements_gauss_newton(seed, invalid)


def test_measurement_fit_rejects_a_row_without_valid_intersections():
    """An all-masked-by-geometry row must not masquerade as a zero residual."""
    dtype = torch.float64
    seed = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype
    )
    rotations = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
        dtype=dtype,
    ).reshape(1, 1, 3, 3)
    inputs = MeasurementFitInputs(
        measured=torch.zeros((1, 1, 2), dtype=dtype),
        variance=torch.ones((1, 1, 2), dtype=dtype),
        component_mask=torch.ones((1, 1, 2), dtype=torch.bool),
        centers=torch.tensor([[[0.0, 5.0, 0.0]]], dtype=dtype),
        rotations=rotations,
    )

    with pytest.raises(ValueError, match="no valid measurement components"):
        fit_helix_measurements_gauss_newton(seed, inputs)


def test_measurement_fit_rejects_an_underconstrained_row():
    """LM damping must not turn a rank-one row into a reported solution."""
    dtype = torch.float64
    seed = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype
    )
    inputs = MeasurementFitInputs(
        measured=torch.zeros((1, 1, 2), dtype=dtype),
        variance=torch.ones((1, 1, 2), dtype=dtype),
        component_mask=torch.tensor([[[True, False]]]),
        centers=torch.tensor([[[10.0, 0.0, 0.0]]], dtype=dtype),
        rotations=torch.tensor(
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=dtype,
        ).reshape(1, 1, 3, 3),
    )

    with pytest.raises(ValueError, match="underconstrained"):
        fit_helix_measurements_gauss_newton(seed, inputs)
    with pytest.raises(ValueError, match="underconstrained"):
        measurement_newton_decrement(seed, inputs)


def test_shared_closest_point_and_plane_intersection_report_invalid_paths():
    track = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    point = torch.tensor([[25.0, 3.0, 0.0]], dtype=torch.float64)
    fitted, path, closest_valid = closest_helix_points(track, point)
    assert closest_valid.tolist() == [True]
    assert path.tolist() == pytest.approx([25.0])
    torch.testing.assert_close(
        fitted, torch.tensor([[25.0, 0.0, 0.0]], dtype=torch.float64)
    )

    _path, fallback, intersection_valid = intersect_plane(
        track,
        center=torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64),
        normal=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
    )
    assert intersection_valid.tolist() == [False]
    torch.testing.assert_close(fallback, torch.zeros_like(fallback))


def test_curvature_optimizer_scaling_round_trips_physical_parameters():
    tracks = torch.tensor(
        [[1.0, 2.0, 3.0, 0.4, -0.2, 2.5e-4]], dtype=torch.float64
    )
    linear, scaled = split_track_parameters(tracks)
    torch.testing.assert_close(
        scaled, torch.tensor([[2.5]], dtype=torch.float64)
    )
    assert torch.equal(assemble_track_parameters(linear, scaled), tracks)


def test_point_seed_gauss_newton_reaches_the_noiseless_helix_stationary_point():
    """Catch a seed solver that merely exhausts a first-order iteration budget."""
    truth = torch.tensor(
        [
            [2.0, -1.0, 0.5, 0.4, 0.2, 2.0e-3],
            [-3.0, 4.0, -1.5, -0.6, -0.1, -1.5e-3],
        ],
        dtype=torch.float64,
    )
    path = torch.linspace(-30.0, 80.0, 12, dtype=torch.float64)
    points = helix_point(
        truth[:, None, :].expand(-1, len(path), -1),
        path.expand(len(truth), -1),
    )
    # Leave a small deterministic irreducible component so dec/F remains a
    # meaningful stationarity ratio instead of becoming 0/0 at exact closure.
    points[..., 2] += 0.01 * torch.sin(path / 7.0)
    mask = torch.ones(points.shape[:2], dtype=torch.bool)
    initial = truth + torch.tensor(
        [
            [0.8, -0.5, 0.3, 0.08, -0.05, 4.0e-4],
            [-0.6, 0.7, -0.4, -0.06, 0.04, -3.0e-4],
        ],
        dtype=torch.float64,
    )

    fitted, diagnostics = fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=30
    )
    chunked, chunked_diagnostics = fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=30, chunk_rows=1
    )
    expanded = fitted[:, None, :].expand(-1, points.shape[1], -1)
    closest, _path, valid = closest_helix_points(expanded, points)
    rms = torch.sqrt((closest[mask & valid] - points[mask & valid]).square().mean())

    assert diagnostics["solver"] == "gauss-newton"
    assert diagnostics["final_objective"] < diagnostics["initial_objective"] * 1.0e-4
    assert diagnostics["relative_newton_decrement_max"] < 1.0e-6
    assert rms.item() < 0.01
    torch.testing.assert_close(chunked, fitted, rtol=0.0, atol=3.0e-8)
    assert chunked_diagnostics["final_objective"] == pytest.approx(
        diagnostics["final_objective"], rel=0.0, abs=1.0e-12
    )


def test_point_seed_ignores_closest_point_invalid_inputs():
    """An out-of-range closest path is not an SMT residual and cannot bias the fit."""
    path = torch.linspace(0.0, 90.0, 10, dtype=torch.float64)
    points = torch.stack((path, torch.zeros_like(path), torch.zeros_like(path)), dim=1)
    points = torch.cat(
        (points, torch.tensor([[2000.0, 100.0, 0.0]], dtype=torch.float64)), dim=0
    ).unsqueeze(0)
    initial = torch.zeros((1, 6), dtype=torch.float64)
    mask = torch.ones((1, len(points[0])), dtype=torch.bool)

    fitted, diagnostics = fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=10
    )

    torch.testing.assert_close(fitted, initial, rtol=0.0, atol=0.0)
    assert diagnostics["converged"] is True
    assert diagnostics["non_stationary_row_fraction"] == 0.0


def test_point_seed_reports_budget_exhaustion_per_row():
    truth = torch.tensor([[2.0, -1.0, 0.5, 0.4, 0.2, 2.0e-3]], dtype=torch.float64)
    path = torch.linspace(-30.0, 80.0, 12, dtype=torch.float64)
    points = helix_point(truth[:, None, :].expand(-1, len(path), -1), path[None])
    initial = truth + torch.tensor(
        [[0.8, -0.5, 0.3, 0.08, -0.05, 4.0e-4]], dtype=torch.float64
    )

    _fitted, diagnostics = fit_helix_points_gauss_newton(
        initial, points, torch.ones(points.shape[:2], dtype=torch.bool), max_iters=1
    )

    assert diagnostics["converged"] is False
    assert diagnostics["non_stationary_row_fraction"] == 1.0
    assert diagnostics["relative_newton_decrement_max"] > 1.0e-6
    with pytest.raises(ValueError, match="non-stationary"):
        require_stationary_helix_point_fit(diagnostics)
    non_finite = {
        **diagnostics,
        "n_non_finite": 1,
        "non_stationary_row_fraction": 1.0 / 2048.0,
    }
    with pytest.raises(ValueError, match="non-finite"):
        require_stationary_helix_point_fit(non_finite)


def test_point_seed_converges_mixed_exact_and_non_exact_rows_in_float32():
    """Already-stationary rows must not block other rows in a shared GPU-like chunk."""
    dtype = torch.float32
    path = torch.linspace(-30.0, 80.0, 12, dtype=dtype)
    truth = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.1, 0.0], [2.0, -1.0, 0.5, 0.4, 0.2, 2e-3]],
        dtype=dtype,
    )
    points = helix_point(
        truth[:, None, :].expand(-1, len(path), -1), path.expand(2, -1)
    )
    points[1, :, 2] += 0.01 * torch.sin(path / 7.0)
    initial = truth.clone()
    initial[1] += torch.tensor(
        [0.8, -0.5, 0.3, 0.08, -0.05, 4e-4], dtype=dtype
    )

    fitted, diagnostics = fit_helix_points_gauss_newton(
        initial,
        points,
        torch.ones(points.shape[:2], dtype=torch.bool),
        max_iters=30,
        chunk_rows=0,
    )

    torch.testing.assert_close(fitted[0], truth[0], rtol=0.0, atol=0.0)
    assert diagnostics["converged"] is True
    assert diagnostics["iterations"] < 30
    assert diagnostics["non_stationary_row_fraction"] == 0.0
