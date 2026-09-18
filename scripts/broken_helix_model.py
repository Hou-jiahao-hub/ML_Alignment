"""Experimental material-aware broken-helix mathematics.

This module depends only on the ACTS-independent single-helix core.  Production
alignment imports :mod:`helix_model` without loading these material models.
"""

from __future__ import annotations

import torch

try:
    from .helix_model import closest_helix_points, helix_point, helix_tangent
except ImportError:  # Direct script execution.
    from helix_model import closest_helix_points, helix_point, helix_tangent


MUON_MASS_GEV = 0.1056583755
HIGHLAND_ENERGY_GEV = 0.0136
HIGHLAND_LOG_COEFFICIENT = 0.038
KINK_THETA_EPS = 1.0e-9


def highland_kink_sigma(
    x_over_x0: torch.Tensor,
    q_over_p_gev_inv: torch.Tensor,
    mass_gev: float = MUON_MASS_GEV,
    abs_charge: float = 1.0,
) -> torch.Tensor:
    """Return the Highland projected-angle width for each material node."""
    q_over_p = q_over_p_gev_inv.unsqueeze(-1).to(x_over_x0.dtype)
    q2_over_beta2 = abs_charge**2 + (mass_gev * q_over_p) ** 2
    positive = x_over_x0 > 0
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


def node_arc_lengths(
    reference: torch.Tensor,
    node_positions: torch.Tensor,
    node_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project material nodes onto a frozen reference helix."""
    n_nodes = node_positions.shape[-2]
    expanded = reference[:, None, :].expand(-1, n_nodes, -1)
    _fitted, path, ok = closest_helix_points(expanded, node_positions)
    path = torch.where(node_valid, path, torch.zeros_like(path))
    return path.detach(), ok & node_valid


def segment_origin_path(
    node_s: torch.Tensor, node_valid: torch.Tensor
) -> torch.Tensor:
    """Return the reference arc length at which every segment starts."""
    origins = [node_s.new_zeros(node_s.shape[0])]
    current = origins[0]
    for index in range(node_s.shape[1]):
        current = torch.where(node_valid[:, index], node_s[:, index], current)
        origins.append(current)
    return torch.stack(origins, dim=1)


def assign_hit_segments(
    node_s: torch.Tensor, hit_s: torch.Tensor, node_valid: torch.Tensor
) -> torch.Tensor:
    """Return the detached segment index of every hit."""
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
    """Build continuous per-segment helices with direction kinks."""
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
    """Return each hit's arc length relative to its segment origin."""
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
    """Return the broken-helix parameters selected for each hit."""
    segments = broken_helix_segments(reference, kinks, node_s, node_valid)
    segment_of_hit = assign_hit_segments(node_s, hit_s, node_valid)
    index = segment_of_hit.unsqueeze(-1).expand(-1, -1, segments.shape[-1])
    return torch.gather(segments, 1, index)


__all__ = [
    "HIGHLAND_ENERGY_GEV",
    "HIGHLAND_LOG_COEFFICIENT",
    "KINK_THETA_EPS",
    "MUON_MASS_GEV",
    "assign_hit_segments",
    "broken_helix_local_path",
    "broken_helix_segments",
    "broken_helix_track_parameters",
    "highland_kink_sigma",
    "node_arc_lengths",
    "segment_origin_path",
]
