"""ACTS-independent differentiable six-parameter helix mathematics.

The physical parameter order is ``(x0, y0, z0, psi0, tan_lambda, rho)``.
Path length is transverse arc length in millimetres and ``rho`` is signed
curvature in inverse millimetres.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


CURVATURE_SCALE = 1.0e-4
S_MAX = 1200.0
PLANE_PARALLEL_EPS = 1.0e-6
PLANE_RESIDUAL_TOLERANCE = 1.0e-3
CLOSEST_POINT_CURVATURE_EPS = 1.0e-9
GAUSS_NEWTON_MAX_ITERS = 60
GAUSS_NEWTON_TOLERANCE = 1.0e-10
GAUSS_NEWTON_INITIAL_DAMPING = 1.0e-8
GAUSS_NEWTON_MIN_DAMPING = 1.0e-12


def _sinc(value: torch.Tensor) -> torch.Tensor:
    """Return ``sin(value) / value`` with a differentiable zero limit."""
    return torch.sinc(value / torch.pi)


def helix_point(track: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Evaluate signed-curvature helices at transverse arc length ``s``."""
    x0, y0, z0, psi0, tanl, rho = track.unbind(-1)
    angle = rho * s
    half_sinc = _sinc(angle / 2.0)
    x = x0 + s * torch.cos(psi0 + angle / 2.0) * half_sinc
    y = y0 + s * torch.sin(psi0 + angle / 2.0) * half_sinc
    z = z0 + tanl * s
    return torch.stack((x, y, z), dim=-1)


def helix_tangent(track: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Differentiate :func:`helix_point` with respect to transverse arc length."""
    _x0, _y0, _z0, psi0, tanl, rho = track.unbind(-1)
    psi = psi0 + rho * s
    return torch.stack(
        (torch.cos(psi), torch.sin(psi), tanl * torch.ones_like(s)), dim=-1
    )


def accept_newton_candidate(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    valid: torch.Tensor,
    candidate_ok: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept feasible Newton steps without reviving an invalid path."""
    updated_valid = valid & candidate_ok
    return torch.where(updated_valid, candidate, previous), updated_valid


def closest_helix_points(
    track: torch.Tensor,
    points: torch.Tensor,
    n_newton: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return closest points on each helix to reference global points."""
    delta = points - track[..., :3]
    s = delta[..., 0] * torch.cos(track[..., 3]) + delta[..., 1] * torch.sin(
        track[..., 3]
    )
    valid = torch.isfinite(s)
    s = torch.where(valid, s, torch.zeros_like(s))
    for _ in range(n_newton):
        position = helix_point(track, s)
        tangent = helix_tangent(track, s)
        displacement = position - points
        rho = track[..., 5]
        psi = track[..., 3] + rho * s
        second = torch.stack(
            (-rho * torch.sin(psi), rho * torch.cos(psi), torch.zeros_like(rho)),
            dim=-1,
        )
        first_derivative = torch.sum(displacement * tangent, dim=-1)
        second_derivative = torch.sum(
            tangent * tangent + displacement * second, dim=-1
        )
        safe = torch.where(
            second_derivative.abs() > CLOSEST_POINT_CURVATURE_EPS,
            second_derivative,
            torch.where(
                second_derivative >= 0,
                torch.full_like(second_derivative, CLOSEST_POINT_CURVATURE_EPS),
                torch.full_like(second_derivative, -CLOSEST_POINT_CURVATURE_EPS),
            ),
        )
        candidate = s - first_derivative / safe
        candidate_ok = torch.isfinite(candidate) & (candidate.abs() <= S_MAX)
        s, valid = accept_newton_candidate(s, candidate, valid, candidate_ok)
    fitted = helix_point(track, s)
    valid = valid & torch.isfinite(fitted).all(dim=-1)
    fitted = torch.where(valid.unsqueeze(-1), fitted, track[..., :3])
    return fitted, s, valid


def intersect_plane(
    track: torch.Tensor,
    center: torch.Tensor,
    normal: torch.Tensor,
    n_newton: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect helices with planes and return ``(s, point, valid_mask)``."""
    tangent0 = helix_tangent(track, torch.zeros_like(track[..., 0]))
    denominator0 = torch.sum(tangent0 * normal, dim=-1)
    straight_valid = denominator0.abs() >= PLANE_PARALLEL_EPS
    safe_denominator = torch.where(
        straight_valid,
        denominator0,
        torch.where(
            denominator0 >= 0,
            torch.ones_like(denominator0),
            -torch.ones_like(denominator0),
        ),
    )
    s = torch.sum((center - track[..., :3]) * normal, dim=-1) / safe_denominator
    tangent_at_initial = helix_tangent(track, s)
    initially_valid = (
        straight_valid
        & torch.isfinite(s)
        & (
            torch.sum(tangent_at_initial * normal, dim=-1).abs()
            >= PLANE_PARALLEL_EPS
        )
    )
    s = torch.where(initially_valid, s, torch.zeros_like(s))
    iteration_valid = initially_valid
    for _ in range(n_newton):
        point = helix_point(track, s)
        tangent = helix_tangent(track, s)
        residual = torch.sum((point - center) * normal, dim=-1)
        derivative = torch.sum(tangent * normal, dim=-1)
        derivative_safe = torch.where(
            derivative.abs() >= PLANE_PARALLEL_EPS,
            derivative,
            torch.where(
                derivative >= 0,
                torch.full_like(derivative, PLANE_PARALLEL_EPS),
                torch.full_like(derivative, -PLANE_PARALLEL_EPS),
            ),
        )
        candidate = s - residual / derivative_safe
        candidate_ok = (
            torch.isfinite(candidate)
            & (candidate.abs() <= S_MAX)
            & (derivative.abs() >= PLANE_PARALLEL_EPS)
        )
        s, iteration_valid = accept_newton_candidate(
            s, candidate, iteration_valid, candidate_ok
        )
    point = helix_point(track, s)
    final_residual = torch.sum((point - center) * normal, dim=-1).abs()
    finite = torch.isfinite(s) & torch.isfinite(point).all(dim=-1)
    valid = (
        iteration_valid
        & finite
        & (s.abs() <= S_MAX)
        & (final_residual <= PLANE_RESIDUAL_TOLERANCE)
    )
    point = torch.where(
        valid.unsqueeze(-1), point, helix_point(track, torch.zeros_like(s))
    )
    s = torch.where(valid, s, torch.zeros_like(s))
    return s, point, valid


def split_track_parameters(
    track_parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split physical helices into linear parameters and scaled curvature."""
    return track_parameters[..., :5], track_parameters[..., 5:6] / CURVATURE_SCALE


def assemble_track_parameters(
    linear: torch.Tensor, rho_scaled: torch.Tensor
) -> torch.Tensor:
    """Assemble physical helix parameters from optimizer representation."""
    return torch.cat((linear, rho_scaled * CURVATURE_SCALE), dim=-1)


def reference_gauge_frame(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix the redundant helix origin to a transverse reference plane."""
    direction = helix_tangent(reference, torch.zeros_like(reference[:, 0]))
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    helper = torch.zeros_like(direction)
    helper[:, 2] = 1.0
    degenerate = direction[:, 2].abs() > 1.0 - 1.0e-6
    helper[degenerate] = torch.tensor(
        [1.0, 0.0, 0.0], dtype=helper.dtype, device=helper.device
    )
    axis0 = torch.cross(helper, direction, dim=-1)
    axis0 = axis0 / axis0.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    axis1 = torch.cross(direction, axis0, dim=-1)
    return reference[:, :3].clone(), axis0, axis1


def pack_gauge_fixed_track_parameters(reference: torch.Tensor) -> torch.Tensor:
    """Pack a helix into the five observable parameters of a fixed gauge."""
    _linear, rho_scaled = split_track_parameters(reference)
    offsets = torch.zeros(
        (reference.shape[0], 2), dtype=reference.dtype, device=reference.device
    )
    return torch.cat((offsets, reference[:, 3:5], rho_scaled), dim=1)


def unpack_gauge_fixed_track_parameters(
    theta: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Recover physical helix parameters from a fixed-gauge representation."""
    origin, axis0, axis1 = frame
    position = origin + theta[:, 0:1] * axis0 + theta[:, 1:2] * axis1
    return assemble_track_parameters(
        torch.cat((position, theta[:, 2:4]), dim=1), theta[:, 4:5]
    )


def damped_gauss_newton(
    theta: torch.Tensor,
    residual_of: Callable[[torch.Tensor], torch.Tensor],
    jacobian_of: Callable[[torch.Tensor], torch.Tensor],
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Minimise independent row residuals with the measurement-refit LM rule."""
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1.")
    current = 0.5 * residual_of(theta).square().sum(dim=1)
    if not torch.isfinite(current).all():
        raise FloatingPointError("Non-finite initial Gauss-Newton objective.")
    initial = float(current.mean())
    identity = torch.eye(theta.shape[1], dtype=theta.dtype, device=theta.device)
    level = torch.full(
        (theta.shape[0], 1, 1),
        float(damping),
        dtype=theta.dtype,
        device=theta.device,
    )
    converged = False
    iterations = 0
    change = float("inf")
    for iterations in range(1, max_iters + 1):
        residual = residual_of(theta)
        jacobian = jacobian_of(theta)
        hessian = jacobian.transpose(1, 2) @ jacobian
        gradient = (jacobian.transpose(1, 2) @ residual.unsqueeze(-1)).squeeze(-1)
        scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
        regularized = hessian + level * scale[:, None, None] * identity
        step = torch.linalg.solve(regularized, -gradient.unsqueeze(-1)).squeeze(-1)
        candidate = theta + step
        candidate_objective = 0.5 * residual_of(candidate).square().sum(dim=1)
        improved = torch.isfinite(candidate_objective) & (
            candidate_objective < current
        )
        theta = torch.where(improved.unsqueeze(-1), candidate, theta)
        level = torch.where(
            improved[:, None, None],
            (level * 0.1).clamp_min(GAUSS_NEWTON_MIN_DAMPING),
            level * 10.0,
        )
        decrease = (current - candidate_objective).clamp_min(
            0.0
        ) / current.abs().clamp_min(1.0)
        current = torch.where(improved, candidate_objective, current)
        change = float(decrease.max())
        # A rejected step has zero decrease too, but is not convergence: its
        # damping must grow and the row must get another candidate.
        if bool(improved.all()) and change < tolerance:
            converged = True
            break

    residual = residual_of(theta)
    jacobian = jacobian_of(theta)
    hessian = jacobian.transpose(1, 2) @ jacobian
    gradient = (jacobian.transpose(1, 2) @ residual.unsqueeze(-1)).squeeze(-1)
    scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.0e-30)
    stationary_hessian = (
        hessian + GAUSS_NEWTON_MIN_DAMPING * scale[:, None, None] * identity
    )
    decrement = (
        gradient.unsqueeze(1)
        @ torch.linalg.solve(stationary_hessian, gradient.unsqueeze(-1))
    ).flatten()
    relative_decrement = decrement / current.clamp_min(1.0e-30)
    return theta, {
        "converged": bool(converged),
        "iterations": int(iterations),
        "initial_objective": initial,
        "final_objective": float(current.mean()),
        "max_relative_decrease": float(change),
        "max_gradient_norm": float(gradient.norm(dim=1).max()),
        "relative_newton_decrement_max": float(relative_decrement.max()),
    }


def _single_point_fit_residual(
    theta: torch.Tensor,
    origin: torch.Tensor,
    axis0: torch.Tensor,
    axis1: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    reference = unpack_gauge_fixed_track_parameters(
        theta.unsqueeze(0),
        (origin.unsqueeze(0), axis0.unsqueeze(0), axis1.unsqueeze(0)),
    ).squeeze(0)
    expanded = reference.expand(points.shape[0], -1)
    with torch.no_grad():
        _fitted, path, _valid = closest_helix_points(expanded.detach(), points)
    predicted = helix_point(expanded, path)
    residual = torch.where(
        mask.unsqueeze(-1), predicted - points, torch.zeros_like(points)
    )
    return residual.flatten()


_BATCHED_POINT_FIT_JACOBIAN = torch.func.vmap(
    torch.func.jacrev(_single_point_fit_residual, argnums=0)
)


def _fit_helix_points_gauss_newton_block(
    initial: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Fit per-track helix seeds to global points by damped Gauss-Newton."""
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial must have shape [tracks, 6].")
    if points.shape != (initial.shape[0], mask.shape[1], 3):
        raise ValueError("points must have shape [tracks, hits, 3].")
    if mask.shape != points.shape[:2]:
        raise ValueError("mask must have shape [tracks, hits].")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every helix seed needs at least one fitted point.")

    frame = reference_gauge_frame(initial)
    theta = pack_gauge_fixed_track_parameters(initial)

    def residual_of(values: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(_single_point_fit_residual)(values, *frame, points, mask)

    def jacobian_of(values: torch.Tensor) -> torch.Tensor:
        return _BATCHED_POINT_FIT_JACOBIAN(values, *frame, points, mask)

    fitted_theta, stats = damped_gauss_newton(
        theta,
        residual_of,
        jacobian_of,
        max_iters=max_iters,
        damping=damping,
        tolerance=tolerance,
    )
    fitted = unpack_gauge_fixed_track_parameters(fitted_theta, frame).detach()
    return fitted, {"solver": "gauss-newton", **stats}


def fit_helix_points_gauss_newton(
    initial: torch.Tensor,
    points: torch.Tensor,
    mask: torch.Tensor,
    max_iters: int = GAUSS_NEWTON_MAX_ITERS,
    damping: float = GAUSS_NEWTON_INITIAL_DAMPING,
    tolerance: float = GAUSS_NEWTON_TOLERANCE,
    chunk_rows: int = 2048,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Fit point seeds with the measurement-refit LM strategy in row chunks."""
    if initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial must have shape [tracks, 6].")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [tracks, hits].")
    if mask.shape != points.shape[:2]:
        raise ValueError("mask must have shape [tracks, hits].")
    if points.ndim != 3 or points.shape != (initial.shape[0], mask.shape[1], 3):
        raise ValueError("points must have shape [tracks, hits, 3].")
    if initial.shape[0] == 0:
        raise ValueError("At least one helix seed is required.")

    step = initial.shape[0] if chunk_rows <= 0 else chunk_rows
    fitted_blocks: list[torch.Tensor] = []
    stats_blocks: list[dict[str, float | int | bool | str]] = []
    weights: list[int] = []
    for start in range(0, initial.shape[0], max(step, 1)):
        stop = min(start + max(step, 1), initial.shape[0])
        fitted, stats = _fit_helix_points_gauss_newton_block(
            initial[start:stop],
            points[start:stop],
            mask[start:stop],
            max_iters=max_iters,
            damping=damping,
            tolerance=tolerance,
        )
        fitted_blocks.append(fitted)
        stats_blocks.append(stats)
        weights.append(stop - start)

    total = float(sum(weights))
    diagnostics: dict[str, float | int | bool | str] = {
        "solver": "gauss-newton",
        "converged": all(bool(stats["converged"]) for stats in stats_blocks),
        "iterations": max(int(stats["iterations"]) for stats in stats_blocks),
        "initial_objective": sum(
            float(stats["initial_objective"]) * weight
            for stats, weight in zip(stats_blocks, weights)
        ) / total,
        "final_objective": sum(
            float(stats["final_objective"]) * weight
            for stats, weight in zip(stats_blocks, weights)
        ) / total,
        "max_relative_decrease": max(
            float(stats["max_relative_decrease"]) for stats in stats_blocks
        ),
        "max_gradient_norm": max(
            float(stats["max_gradient_norm"]) for stats in stats_blocks
        ),
        "relative_newton_decrement_max": max(
            float(stats["relative_newton_decrement_max"]) for stats in stats_blocks
        ),
        "n_chunks": len(stats_blocks),
    }
    return torch.cat(fitted_blocks, dim=0), diagnostics


# ---------------------------------------------------------------------------
# Highland multiple-scattering prior
# ---------------------------------------------------------------------------

MUON_MASS_GEV = 0.1056583755
HIGHLAND_ENERGY_GEV = 0.0136
HIGHLAND_LOG_COEFFICIENT = 0.038


def highland_kink_sigma(
    x_over_x0: torch.Tensor,
    q_over_p_gev_inv: torch.Tensor,
    mass_gev: float = MUON_MASS_GEV,
    abs_charge: float = 1.0,
) -> torch.Tensor:
    """Return the Highland projected-angle width for each material node.

    Transcribes RPP2018 eq. 33.15 exactly as ACTS does in
    ``Core/src/Material/Interactions.cpp:484-491``, including the ``q^2/beta^2``
    factor inside the logarithm.  ``x_over_x0`` is ``[..., N]`` and
    ``q_over_p_gev_inv`` is ``[...]``; the signed ``q/p`` is constant along a
    track, matching the material artifact contract.

    Vacuum nodes (``x/X0 == 0``) return exactly zero through a masked limit.
    Clamping the logarithm instead would fabricate a discontinuous non-zero
    floor at the clamp threshold.
    """
    q_over_p = q_over_p_gev_inv.unsqueeze(-1).to(x_over_x0.dtype)
    q2_over_beta2 = abs_charge**2 + (mass_gev * q_over_p) ** 2
    positive = x_over_x0 > 0
    # Evaluate the log on a safe surrogate so that vacuum entries never produce
    # non-finite intermediates in either the forward or the backward pass.
    safe = torch.where(positive, x_over_x0, torch.ones_like(x_over_x0))
    t = torch.sqrt(safe * q2_over_beta2)
    momentum_inv = torch.abs(q_over_p / abs_charge)
    sigma = (
        HIGHLAND_ENERGY_GEV
        * momentum_inv
        * t
        * (1.0 + HIGHLAND_LOG_COEFFICIENT * 2.0 * torch.log(t))
    )
    return torch.where(positive, sigma, torch.zeros_like(sigma))


# ---------------------------------------------------------------------------
# Broken helix: a reference trajectory plus two-dimensional material kinks
# ---------------------------------------------------------------------------

KINK_THETA_EPS = 1.0e-9


def node_arc_lengths(
    reference: torch.Tensor,
    node_positions: torch.Tensor,
    node_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project material nodes onto a frozen reference helix.

    The helix ``s`` origin is fixed by ``(x0, y0, z0)``, so arc lengths are NOT
    a track-level invariant: two fits of the same trajectory over different
    windows produce different ``s``.  Consumers must therefore recompute node
    and hit arc lengths against whichever reference they froze, rather than
    reusing an arc length that came out of the exporter.

    Returned arc lengths are detached: they define the linearisation point and
    must not drift while the kinks and the reference are being optimised.
    """
    n_nodes = node_positions.shape[-2]
    expanded = reference[:, None, :].expand(-1, n_nodes, -1)
    _fitted, path, ok = closest_helix_points(expanded, node_positions)
    path = torch.where(node_valid, path, torch.zeros_like(path))
    return path.detach(), ok & node_valid


def segment_origin_path(node_s: torch.Tensor, node_valid: torch.Tensor) -> torch.Tensor:
    """Reference arc length at which every segment starts, shape ``[T, N+1]``.

    Segment 0 starts at the reference origin.  A masked node transfers the
    previous origin unchanged so padding never shifts the local coordinate.
    """
    # ``new_zeros`` rather than ``node_s[:, 0]`` so a nodeless track (N == 0) is
    # a legal input: that is exactly how the single-helix arm of the paired
    # holdout runs, and it must go through this same solver to stay comparable.
    origins = [node_s.new_zeros(node_s.shape[0])]
    current = origins[0]
    for index in range(node_s.shape[1]):
        current = torch.where(node_valid[:, index], node_s[:, index], current)
        origins.append(current)
    return torch.stack(origins, dim=1)


def assign_hit_segments(
    node_s: torch.Tensor, hit_s: torch.Tensor, node_valid: torch.Tensor
) -> torch.Tensor:
    """Return the segment index of every hit, as a detached integer tensor.

    A hit sitting exactly on a node belongs to the segment after it.  Keeping
    this integer and detached is what stops the autograd graph from developing
    a discontinuity when a kink nudges a hit across a node.
    """
    active = node_valid.unsqueeze(1) & (
        node_s.unsqueeze(1) <= hit_s.unsqueeze(-1)
    )
    return active.sum(dim=-1).to(torch.int64).detach()


def broken_helix_segments(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
) -> torch.Tensor:
    """Build the ``[T, N+1, 6]`` per-segment helix parameters.

    Segments stay continuous in position; only the direction changes at a node.
    The two kink components follow the ACTS convention
    (``PointwiseMaterialInteraction.cpp:69-74``): ``sigmaPhi = theta0 / sin(theta)``
    and ``sigmaTheta = theta0``, which makes a diagonal prior isotropic in the
    plane transverse to the incident direction.

    Signed curvature is shared by every segment: the first version keeps
    curvature constant and does not model energy loss.

    Arc lengths between nodes are taken from the reference trajectory, which is
    the standard broken-lines linearisation rather than an exact re-integration.
    """
    segments = [reference]
    current = reference
    origin = node_s.new_zeros(node_s.shape[0])
    for index in range(node_s.shape[1]):
        valid = node_valid[:, index]
        delta = torch.where(
            valid, node_s[:, index] - origin, torch.zeros_like(origin)
        )
        arrival = helix_point(current, delta)
        psi = current[..., 3] + current[..., 5] * delta
        tanl = current[..., 4]
        theta = torch.atan2(torch.ones_like(tanl), tanl)
        sin_theta = torch.sin(theta).clamp_min(KINK_THETA_EPS)
        psi_kinked = psi + kinks[:, index, 0] / sin_theta
        theta_kinked = (theta + kinks[:, index, 1]).clamp(
            KINK_THETA_EPS, torch.pi - KINK_THETA_EPS
        )
        tanl_kinked = torch.cos(theta_kinked) / torch.sin(theta_kinked)
        candidate = torch.stack(
            (
                arrival[..., 0],
                arrival[..., 1],
                arrival[..., 2],
                psi_kinked,
                tanl_kinked,
                current[..., 5],
            ),
            dim=-1,
        )
        current = torch.where(valid.unsqueeze(-1), candidate, current)
        origin = torch.where(valid, node_s[:, index], origin)
        segments.append(current)
    return torch.stack(segments, dim=1)


def broken_helix_local_path(
    node_s: torch.Tensor, node_valid: torch.Tensor, hit_s: torch.Tensor
) -> torch.Tensor:
    """Arc length of every hit measured from the origin of its own segment."""
    origins = segment_origin_path(node_s, node_valid)
    segment_of_hit = assign_hit_segments(node_s, hit_s, node_valid)
    return hit_s - torch.gather(origins, 1, segment_of_hit)


def broken_helix_track_parameters(
    reference: torch.Tensor,
    kinks: torch.Tensor,
    node_s: torch.Tensor,
    node_valid: torch.Tensor,
    hit_s: torch.Tensor,
) -> torch.Tensor:
    """Return per-hit helix parameters, shape ``[T, H, 6]``.

    This is deliberately the shape every existing consumer already builds with
    ``track_parameters[:, None, :].expand(-1, n_hits, -1)``, so
    :func:`intersect_plane` and the local projection downstream need no change.
    Pair it with :func:`broken_helix_local_path` for the matching arc lengths.
    """
    segments = broken_helix_segments(reference, kinks, node_s, node_valid)
    segment_of_hit = assign_hit_segments(node_s, hit_s, node_valid)
    index = segment_of_hit.unsqueeze(-1).expand(-1, -1, segments.shape[-1])
    return torch.gather(segments, 1, index)
