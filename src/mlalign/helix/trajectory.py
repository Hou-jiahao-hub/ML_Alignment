"""Helix evaluation, closest points and plane intersections."""

from __future__ import annotations

import torch

S_MAX = 1200.0
PLANE_PARALLEL_EPS = 1.0e-6
PLANE_RESIDUAL_TOLERANCE = 1.0e-3
CLOSEST_POINT_CURVATURE_EPS = 1.0e-9


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
