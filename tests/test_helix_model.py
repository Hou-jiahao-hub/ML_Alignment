import importlib
import math
import sys

import pytest
import torch


def _measurement_fit_case(model, n_tracks=2):
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
    centers = model.helix_point(expanded, path.expand(len(truth), -1))
    tangents = model.helix_tangent(expanded, path.expand(len(truth), -1))
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
    inputs = model.MeasurementFitInputs(
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
    model = importlib.import_module("scripts.helix_model")
    track = torch.tensor(
        [[1.0, -2.0, 3.0, 0.0, 0.5, 0.01]], dtype=torch.float64
    )
    path = torch.tensor([100.0], dtype=torch.float64)

    point = model.helix_point(track, path)
    tangent = model.helix_tangent(track, path)

    assert point[0, 0].item() == pytest.approx(1.0 + 100.0 * math.sin(1.0))
    assert point[0, 1].item() == pytest.approx(-2.0 + 100.0 * (1.0 - math.cos(1.0)))
    assert point[0, 2].item() == pytest.approx(53.0)
    assert tangent[0].tolist() == pytest.approx(
        [math.cos(1.0), math.sin(1.0), 0.5]
    )


def test_measurement_residual_jacobian_matches_finite_difference():
    """Catch a Jacobian that differentiates a different residual convention."""
    model = importlib.import_module("scripts.helix_model")
    inputs, _truth, seed = _measurement_fit_case(model, n_tracks=1)
    frame = model.reference_gauge_frame(seed)
    theta = model.pack_gauge_fixed_track_parameters(seed)

    residual = model.measurement_standardized_residual(theta, frame, inputs)
    jacobian = model.measurement_residual_jacobian(theta, frame, inputs)
    eps = 1.0e-6
    plus = theta.clone()
    minus = theta.clone()
    plus[:, 0] += eps
    minus[:, 0] -= eps
    finite_difference = (
        model.measurement_standardized_residual(plus, frame, inputs)
        - model.measurement_standardized_residual(minus, frame, inputs)
    ) / (2.0 * eps)

    assert residual.shape == (1, inputs.measured.shape[1] * 2)
    torch.testing.assert_close(
        jacobian[:, :, 0], finite_difference, atol=2.0e-6, rtol=2.0e-5
    )


def test_measurement_gauss_newton_is_chunk_exact_and_stationary():
    """Catch budget exhaustion being reported as a solved measurement fit."""
    model = importlib.import_module("scripts.helix_model")
    inputs, _truth, seed = _measurement_fit_case(model)

    whole, whole_stats = model.fit_helix_measurements_gauss_newton(
        seed, inputs, chunk_rows=0
    )
    chunked, chunked_stats = model.fit_helix_measurements_gauss_newton(
        seed, inputs, chunk_rows=1
    )
    decrement = model.measurement_newton_decrement(whole, inputs, chunk_rows=1)
    report = model.measurement_stationarity_report(decrement)

    assert torch.equal(whole, chunked)
    assert whole_stats["convergence"]["converged"] is True
    assert chunked_stats["convergence"]["n_chunks"] == 2
    assert report["n_non_finite"] == 0
    assert report["non_stationary_row_fraction"] == 0.0
    assert (
        report["newton_decrement_max"]
        < report["stationary_threshold"]
    )

    _truncated, truncated_stats = model.fit_helix_measurements_gauss_newton(
        seed, inputs, max_iters=1
    )
    assert truncated_stats["convergence"]["converged"] is False


def test_measurement_fit_ignores_fully_masked_padding_hits():
    model = importlib.import_module("scripts.helix_model")
    inputs, _truth, seed = _measurement_fit_case(model, n_tracks=1)
    base = model.MeasurementFitInputs(
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
    padded = model.MeasurementFitInputs(
        measured=inputs.measured,
        variance=inputs.variance,
        component_mask=padded_mask,
        centers=padded_centers,
        rotations=inputs.rotations,
    )

    expected, _ = model.fit_helix_measurements_gauss_newton(seed, base)
    actual, _ = model.fit_helix_measurements_gauss_newton(seed, padded)

    torch.testing.assert_close(actual, expected, atol=1.0e-10, rtol=0.0)


def test_measurement_fit_rejects_non_finite_residuals():
    model = importlib.import_module("scripts.helix_model")
    inputs, _truth, seed = _measurement_fit_case(model, n_tracks=1)
    measured = inputs.measured.clone()
    measured[0, 0, 0] = float("nan")
    invalid = model.MeasurementFitInputs(
        measured=measured,
        variance=inputs.variance,
        component_mask=inputs.component_mask,
        centers=inputs.centers,
        rotations=inputs.rotations,
    )

    with pytest.raises(FloatingPointError, match="Non-finite initial"):
        model.fit_helix_measurements_gauss_newton(seed, invalid)


def test_shared_closest_point_and_plane_intersection_report_invalid_paths():
    model = importlib.import_module("scripts.helix_model")
    track = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    point = torch.tensor([[25.0, 3.0, 0.0]], dtype=torch.float64)
    fitted, path, closest_valid = model.closest_helix_points(track, point)
    assert closest_valid.tolist() == [True]
    assert path.tolist() == pytest.approx([25.0])
    torch.testing.assert_close(
        fitted, torch.tensor([[25.0, 0.0, 0.0]], dtype=torch.float64)
    )

    _path, fallback, intersection_valid = model.intersect_plane(
        track,
        center=torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64),
        normal=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
    )
    assert intersection_valid.tolist() == [False]
    torch.testing.assert_close(fallback, torch.zeros_like(fallback))


def test_broken_helix_math_is_a_lazy_compatibility_import():
    """Production helix imports must not eagerly load material experiments."""
    sys.modules.pop("scripts.broken_helix_model", None)
    model = importlib.reload(importlib.import_module("scripts.helix_model"))
    assert "scripts.broken_helix_model" not in sys.modules

    broken = importlib.import_module("scripts.broken_helix_model")
    x_over_x0 = torch.tensor([[0.01]], dtype=torch.float64)
    q_over_p = torch.tensor([0.2], dtype=torch.float64)
    expected = broken.highland_kink_sigma(x_over_x0, q_over_p)
    actual = model.highland_kink_sigma(x_over_x0, q_over_p)

    torch.testing.assert_close(actual, expected)


def test_curvature_optimizer_scaling_round_trips_physical_parameters():
    model = importlib.import_module("scripts.helix_model")
    tracks = torch.tensor(
        [[1.0, 2.0, 3.0, 0.4, -0.2, 2.5e-4]], dtype=torch.float64
    )
    linear, scaled = model.split_track_parameters(tracks)
    torch.testing.assert_close(
        scaled, torch.tensor([[2.5]], dtype=torch.float64)
    )
    assert torch.equal(model.assemble_track_parameters(linear, scaled), tracks)


def test_point_seed_gauss_newton_reaches_the_noiseless_helix_stationary_point():
    """Catch a seed solver that merely exhausts a first-order iteration budget."""
    model = importlib.import_module("scripts.helix_model")
    truth = torch.tensor(
        [
            [2.0, -1.0, 0.5, 0.4, 0.2, 2.0e-3],
            [-3.0, 4.0, -1.5, -0.6, -0.1, -1.5e-3],
        ],
        dtype=torch.float64,
    )
    path = torch.linspace(-30.0, 80.0, 12, dtype=torch.float64)
    points = model.helix_point(
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

    fitted, diagnostics = model.fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=30
    )
    chunked, chunked_diagnostics = model.fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=30, chunk_rows=1
    )
    expanded = fitted[:, None, :].expand(-1, points.shape[1], -1)
    closest, _path, valid = model.closest_helix_points(expanded, points)
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
    model = importlib.import_module("scripts.helix_model")
    path = torch.linspace(0.0, 90.0, 10, dtype=torch.float64)
    points = torch.stack((path, torch.zeros_like(path), torch.zeros_like(path)), dim=1)
    points = torch.cat(
        (points, torch.tensor([[2000.0, 100.0, 0.0]], dtype=torch.float64)), dim=0
    ).unsqueeze(0)
    initial = torch.zeros((1, 6), dtype=torch.float64)
    mask = torch.ones((1, len(points[0])), dtype=torch.bool)

    fitted, diagnostics = model.fit_helix_points_gauss_newton(
        initial, points, mask, max_iters=10
    )

    torch.testing.assert_close(fitted, initial, rtol=0.0, atol=0.0)
    assert diagnostics["converged"] is True
    assert diagnostics["non_stationary_row_fraction"] == 0.0


def test_point_seed_reports_budget_exhaustion_per_row():
    model = importlib.import_module("scripts.helix_model")
    truth = torch.tensor([[2.0, -1.0, 0.5, 0.4, 0.2, 2.0e-3]], dtype=torch.float64)
    path = torch.linspace(-30.0, 80.0, 12, dtype=torch.float64)
    points = model.helix_point(truth[:, None, :].expand(-1, len(path), -1), path[None])
    initial = truth + torch.tensor(
        [[0.8, -0.5, 0.3, 0.08, -0.05, 4.0e-4]], dtype=torch.float64
    )

    _fitted, diagnostics = model.fit_helix_points_gauss_newton(
        initial, points, torch.ones(points.shape[:2], dtype=torch.bool), max_iters=1
    )

    assert diagnostics["converged"] is False
    assert diagnostics["non_stationary_row_fraction"] == 1.0
    assert diagnostics["relative_newton_decrement_max"] > 1.0e-6
    with pytest.raises(ValueError, match="non-stationary"):
        model.require_stationary_helix_point_fit(diagnostics)
    non_finite = {
        **diagnostics,
        "n_non_finite": 1,
        "non_stationary_row_fraction": 1.0 / 2048.0,
    }
    with pytest.raises(ValueError, match="non-finite"):
        model.require_stationary_helix_point_fit(non_finite)


def test_point_seed_converges_mixed_exact_and_non_exact_rows_in_float32():
    """Already-stationary rows must not block other rows in a shared GPU-like chunk."""
    model = importlib.import_module("scripts.helix_model")
    dtype = torch.float32
    path = torch.linspace(-30.0, 80.0, 12, dtype=dtype)
    truth = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.1, 0.0], [2.0, -1.0, 0.5, 0.4, 0.2, 2e-3]],
        dtype=dtype,
    )
    points = model.helix_point(
        truth[:, None, :].expand(-1, len(path), -1), path.expand(2, -1)
    )
    points[1, :, 2] += 0.01 * torch.sin(path / 7.0)
    initial = truth.clone()
    initial[1] += torch.tensor(
        [0.8, -0.5, 0.3, 0.08, -0.05, 4e-4], dtype=dtype
    )

    fitted, diagnostics = model.fit_helix_points_gauss_newton(
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


def test_both_consumers_reexport_the_shared_math_implementation():
    shared = importlib.import_module("scripts.helix_model")
    alignment = importlib.import_module("scripts.ml_alignment_autograd")
    validation = importlib.import_module("scripts.validate_helix_model")

    for name in (
        "helix_point",
        "helix_tangent",
        "closest_helix_points",
        "intersect_plane",
    ):
        assert getattr(alignment, name) is getattr(shared, name)
        assert getattr(validation, name) is getattr(shared, name)


# ---------------------------------------------------------------------------
# Highland multiple-scattering prior (work package 2, Gate 2)
# ---------------------------------------------------------------------------


def _acts_theta0(x_over_x0, q_over_p, mass, abs_charge=1.0):
    """Reference implementation transcribed from ACTS Interactions.cpp:484-491."""
    q2_over_beta2 = abs_charge**2 + (mass * q_over_p) ** 2
    t = math.sqrt(x_over_x0 * q2_over_beta2)
    momentum_inv = abs(q_over_p / abs_charge)
    return 0.0136 * momentum_inv * t * (1.0 + 0.038 * 2 * math.log(t))


def test_highland_kink_sigma_matches_acts_rpp2018_form():
    model = importlib.import_module("scripts.helix_model")
    x_over_x0 = torch.tensor([[0.01, 0.05]], dtype=torch.float64)
    q_over_p = torch.tensor([0.2], dtype=torch.float64)

    sigma = model.highland_kink_sigma(x_over_x0, q_over_p)

    expected = [
        _acts_theta0(0.01, 0.2, model.MUON_MASS_GEV),
        _acts_theta0(0.05, 0.2, model.MUON_MASS_GEV),
    ]
    assert sigma[0].tolist() == pytest.approx(expected, rel=1e-12)


def test_highland_kink_sigma_uses_exact_zero_limit_for_vacuum_nodes():
    """x/X0 == 0 must return exactly 0 with finite gradients, not log(clamp(...))."""
    model = importlib.import_module("scripts.helix_model")
    x_over_x0 = torch.tensor(
        [[0.0, 0.02, 0.0]], dtype=torch.float64, requires_grad=True
    )
    q_over_p = torch.tensor([0.5], dtype=torch.float64)

    sigma = model.highland_kink_sigma(x_over_x0, q_over_p)
    assert torch.isfinite(sigma).all()
    assert sigma[0, 0].item() == 0.0
    assert sigma[0, 2].item() == 0.0
    assert sigma[0, 1].item() > 0.0

    sigma.sum().backward()
    assert torch.isfinite(x_over_x0.grad).all()


def test_highland_kink_sigma_scales_with_thickness_and_momentum():
    model = importlib.import_module("scripts.helix_model")
    # Thickness x4 -> sigma approximately x2 (sqrt scaling, log term is a slow drift).
    thin = model.highland_kink_sigma(
        torch.tensor([[0.005]], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
    )
    thick = model.highland_kink_sigma(
        torch.tensor([[0.020]], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
    )
    assert (thick / thin).item() == pytest.approx(2.0, rel=0.10)

    # Momentum x2 (q/p halved) -> sigma halved for an ultra-relativistic muon.
    slow = model.highland_kink_sigma(
        torch.tensor([[0.01]], dtype=torch.float64),
        torch.tensor([0.2], dtype=torch.float64),
    )
    fast = model.highland_kink_sigma(
        torch.tensor([[0.01]], dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
    )
    assert (fast / slow).item() == pytest.approx(0.5, rel=1e-3)


def test_highland_kink_sigma_split_matches_analytic_ratio_not_conservation():
    """Splitting a slab changes the summed variance by a known analytic factor.

    ACTS/PDG evaluate the log term with the per-slab x/X0 (Interactions.cpp:516),
    so sum(theta0^2) is NOT invariant under splitting.  Asserting invariance would
    encode wrong physics; assert the closed-form ratio instead.
    """
    model = importlib.import_module("scripts.helix_model")
    x_over_x0, q_over_p, pieces = 0.02, 0.15, 5
    mass = model.MUON_MASS_GEV

    whole = model.highland_kink_sigma(
        torch.tensor([[x_over_x0]], dtype=torch.float64),
        torch.tensor([q_over_p], dtype=torch.float64),
    )
    split = model.highland_kink_sigma(
        torch.full((1, pieces), x_over_x0 / pieces, dtype=torch.float64),
        torch.tensor([q_over_p], dtype=torch.float64),
    )

    u = x_over_x0 * (1.0 + (mass * q_over_p) ** 2)
    analytic_ratio = (
        (1.0 + 0.038 * (math.log(u) - math.log(pieces))) / (1.0 + 0.038 * math.log(u))
    ) ** 2
    assert (split.square().sum() / whole.square().sum()).item() == pytest.approx(
        analytic_ratio, rel=1e-12
    )
    assert analytic_ratio != pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Broken helix: segments, kinks and the frozen-reference coordinate contract
# ---------------------------------------------------------------------------


def _reference(dtype=torch.float64):
    return torch.tensor([[10.0, -5.0, 2.0, 0.35, 0.60, 3.0e-4]], dtype=dtype)


def test_broken_helix_reduces_to_the_single_helix_without_material():
    """Zero-material limit: point, tangent AND gradient must be identical."""
    model = importlib.import_module("scripts.helix_model")
    reference = _reference().requires_grad_(True)
    node_s = torch.tensor([[100.0, 300.0]], dtype=torch.float64)
    node_valid = torch.zeros((1, 2), dtype=torch.bool)
    kinks = torch.zeros((1, 2, 2), dtype=torch.float64)
    hit_s = torch.tensor([[50.0, 200.0, 400.0]], dtype=torch.float64)

    per_hit = model.broken_helix_track_parameters(
        reference, kinks, node_s, node_valid, hit_s
    )
    expanded = reference[:, None, :].expand(-1, hit_s.shape[1], -1)
    torch.testing.assert_close(per_hit, expanded)

    local_s = model.broken_helix_local_path(node_s, node_valid, hit_s)
    torch.testing.assert_close(local_s, hit_s)

    broken_point = model.helix_point(per_hit, local_s)
    single_point = model.helix_point(expanded, hit_s)
    torch.testing.assert_close(broken_point, single_point)
    torch.testing.assert_close(
        model.helix_tangent(per_hit, local_s), model.helix_tangent(expanded, hit_s)
    )

    broken_grad = torch.autograd.grad(broken_point.sum(), reference, retain_graph=True)[0]
    single_grad = torch.autograd.grad(single_point.sum(), reference)[0]
    torch.testing.assert_close(broken_grad, single_grad)


def test_zero_kinks_are_a_no_op_even_when_nodes_are_active():
    model = importlib.import_module("scripts.helix_model")
    reference = _reference()
    node_s = torch.tensor([[100.0, 300.0]], dtype=torch.float64)
    node_valid = torch.ones((1, 2), dtype=torch.bool)
    kinks = torch.zeros((1, 2, 2), dtype=torch.float64)
    hit_s = torch.tensor([[50.0, 200.0, 400.0]], dtype=torch.float64)

    per_hit = model.broken_helix_track_parameters(
        reference, kinks, node_s, node_valid, hit_s
    )
    local_s = model.broken_helix_local_path(node_s, node_valid, hit_s)
    torch.testing.assert_close(
        model.helix_point(per_hit, local_s),
        model.helix_point(reference[:, None, :].expand(-1, 3, -1), hit_s),
    )


def test_both_kink_components_deflect_by_the_same_physical_angle():
    """sigmaPhi = theta0 / sin(theta) makes the 2D prior isotropic in 3D."""
    model = importlib.import_module("scripts.helix_model")
    reference = _reference()
    node_s = torch.tensor([[100.0]], dtype=torch.float64)
    node_valid = torch.ones((1, 1), dtype=torch.bool)
    delta = 1.0e-5

    angles = []
    for component in (0, 1):
        kinks = torch.zeros((1, 1, 2), dtype=torch.float64)
        kinks[0, 0, component] = delta
        segments = model.broken_helix_segments(reference, kinks, node_s, node_valid)
        zero = torch.zeros((1, 1), dtype=torch.float64)
        incoming = model.helix_tangent(segments[:, 0:1, :], node_s)
        outgoing = model.helix_tangent(segments[:, 1:2, :], zero)
        incoming = incoming / incoming.norm(dim=-1, keepdim=True)
        outgoing = outgoing / outgoing.norm(dim=-1, keepdim=True)
        angles.append(
            float(torch.acos((incoming * outgoing).sum(-1).clamp(-1.0, 1.0)))
        )
    assert angles[0] == pytest.approx(delta, rel=1e-5)
    assert angles[1] == pytest.approx(delta, rel=1e-5)


def test_segments_stay_continuous_in_position_across_every_kink():
    model = importlib.import_module("scripts.helix_model")
    reference = _reference()
    node_s = torch.tensor([[80.0, 250.0, 500.0]], dtype=torch.float64)
    node_valid = torch.ones((1, 3), dtype=torch.bool)
    kinks = torch.tensor([[[2e-4, -1e-4], [-3e-4, 2e-4], [1e-4, 1e-4]]], dtype=torch.float64)

    segments = model.broken_helix_segments(reference, kinks, node_s, node_valid)
    origin_s = torch.cat(
        (torch.zeros((1, 1), dtype=torch.float64), node_s), dim=1
    )
    for index in range(3):
        arrival = model.helix_point(
            segments[:, index, :], node_s[:, index] - origin_s[:, index]
        )
        departure = segments[:, index + 1, :3]
        torch.testing.assert_close(arrival, departure)


def test_padding_nodes_transfer_the_segment_unchanged():
    model = importlib.import_module("scripts.helix_model")
    reference = _reference()
    node_s = torch.tensor([[80.0, 0.0, 250.0]], dtype=torch.float64)
    node_valid = torch.tensor([[True, False, True]])
    kinks = torch.tensor([[[2e-4, -1e-4], [9.9, 9.9], [-3e-4, 2e-4]]], dtype=torch.float64)

    segments = model.broken_helix_segments(reference, kinks, node_s, node_valid)
    # The masked node must neither move the origin nor apply its garbage kink.
    torch.testing.assert_close(segments[:, 1, :], segments[:, 2, :])

    compact = model.broken_helix_segments(
        reference,
        kinks[:, [0, 2], :],
        node_s[:, [0, 2]],
        torch.ones((1, 2), dtype=torch.bool),
    )
    torch.testing.assert_close(segments[:, 3, :], compact[:, 2, :])


def test_hit_segment_assignment_is_deterministic_and_detached():
    model = importlib.import_module("scripts.helix_model")
    node_s = torch.tensor([[100.0, 300.0, 0.0]], dtype=torch.float64)
    node_valid = torch.tensor([[True, True, False]])
    hit_s = torch.tensor([[-50.0, 100.0, 150.0, 300.0, 900.0]], dtype=torch.float64)

    first = model.assign_hit_segments(node_s, hit_s, node_valid)
    second = model.assign_hit_segments(node_s, hit_s, node_valid)
    assert first.dtype == torch.int64
    assert not first.requires_grad
    assert first.tolist() == second.tolist()
    # A hit exactly on a node belongs to the segment after that node.
    assert first.tolist() == [[0, 1, 1, 2, 2]]


def test_reference_reparameterisation_leaves_prediction_invariant():
    """Regression for the frozen-reference coordinate contract.

    Two validation rows fit their own reference helix, so the same physical
    track gets different ``s`` origins.  Node arc lengths are therefore not
    transferable, but once each row recomputes them against its own frozen
    reference the segment assignment and the predicted global points must agree.
    """
    model = importlib.import_module("scripts.helix_model")
    reference = _reference()
    shift = 137.0

    # Same trajectory, origin moved forward along it: a pure re-parameterisation.
    x0, y0, z0, psi0, tanl, rho = reference[0].tolist()
    moved_origin = model.helix_point(
        reference, torch.tensor([shift], dtype=torch.float64)
    )
    shifted = torch.tensor(
        [[*moved_origin[0].tolist(), psi0 + rho * shift, tanl, rho]],
        dtype=torch.float64,
    )
    del x0, y0, z0, psi0

    node_positions = model.helix_point(
        reference[:, None, :].expand(-1, 3, -1),
        torch.tensor([[80.0, 250.0, 500.0]], dtype=torch.float64),
    )
    hit_positions = model.helix_point(
        reference[:, None, :].expand(-1, 4, -1),
        torch.tensor([[40.0, 150.0, 300.0, 700.0]], dtype=torch.float64),
    )
    node_valid = torch.ones((1, 3), dtype=torch.bool)
    kinks = torch.tensor([[[2e-4, -1e-4], [-3e-4, 2e-4], [1e-4, 1e-4]]], dtype=torch.float64)

    predictions = []
    assignments = []
    for frozen in (reference, shifted):
        node_s, node_ok = model.node_arc_lengths(frozen, node_positions, node_valid)
        hit_s, _hit_ok = model.node_arc_lengths(
            frozen, hit_positions, torch.ones((1, 4), dtype=torch.bool)
        )
        assert bool(node_ok.all())
        assignments.append(model.assign_hit_segments(node_s, hit_s, node_valid).tolist())
        per_hit = model.broken_helix_track_parameters(
            frozen, kinks, node_s, node_valid, hit_s
        )
        local_s = model.broken_helix_local_path(node_s, node_valid, hit_s)
        predictions.append(model.helix_point(per_hit, local_s))

    # The two rows disagree on s by the shift ...
    assert assignments[0] == assignments[1]
    # ... but must agree on every predicted global position.
    torch.testing.assert_close(predictions[0], predictions[1], atol=1e-9, rtol=0)


def test_broken_helix_accepts_a_nodeless_track():
    """N == 0 is the single-helix arm of the paired holdout, not an edge case.

    Both arms must go through this same code so the comparison cannot be
    confounded by two different optimisers.
    """
    module = importlib.import_module("scripts.helix_model")
    reference = torch.tensor([[1.0, 2.0, 0.0, 0.3, 0.5, 1.0e-4]], dtype=torch.float64)
    node_s = torch.zeros((1, 0), dtype=torch.float64)
    node_valid = torch.zeros((1, 0), dtype=torch.bool)
    kinks = torch.zeros((1, 0, 2), dtype=torch.float64)
    hit_s = torch.tensor([[50.0, 200.0, 400.0]], dtype=torch.float64)

    origins = module.segment_origin_path(node_s, node_valid)
    assert origins.shape == (1, 1)
    assert bool((origins == 0).all())
    assert module.assign_hit_segments(node_s, hit_s, node_valid).tolist() == [[0, 0, 0]]

    per_hit = module.broken_helix_track_parameters(
        reference, kinks, node_s, node_valid, hit_s
    )
    expected = reference[:, None, :].expand(-1, 3, -1)
    torch.testing.assert_close(per_hit, expected)
