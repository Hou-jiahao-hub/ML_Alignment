"""Optimizer representations of a helix: scaled curvature and the fixed gauge."""

from __future__ import annotations

import torch

from .trajectory import helix_tangent

CURVATURE_SCALE = 1.0e-4


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
