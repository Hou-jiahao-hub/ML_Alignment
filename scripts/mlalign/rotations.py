"""Local six-DOF alignment corrections in the project's ``Rz @ Ry @ Rx`` convention."""

from __future__ import annotations

import numpy as np
import torch


def euler_to_rotation_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(rotation):
    """Decompose a rotation using the project's ``Rz @ Ry @ Rx`` convention."""
    rotation = np.asarray(rotation, dtype=float)
    ry = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    rx = np.arctan2(rotation[2, 1], rotation[2, 2])
    rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    return float(rx), float(ry), float(rz)


def torch_euler_to_rotation_matrix(delta_rot):
    alpha, beta, gamma = delta_rot.unbind(-1)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    cb, sb = torch.cos(beta), torch.sin(beta)
    cg, sg = torch.cos(gamma), torch.sin(gamma)
    ones = torch.ones_like(alpha)
    zeros = torch.zeros_like(alpha)

    Rx = torch.stack(
        [ones, zeros, zeros, zeros, ca, -sa, zeros, sa, ca], dim=-1
    ).reshape(*alpha.shape, 3, 3)
    Ry = torch.stack(
        [cb, zeros, sb, zeros, ones, zeros, -sb, zeros, cb], dim=-1
    ).reshape(*alpha.shape, 3, 3)
    Rz = torch.stack(
        [cg, -sg, zeros, sg, cg, zeros, zeros, zeros, ones], dim=-1
    ).reshape(*alpha.shape, 3, 3)
    return Rz @ Ry @ Rx


def apply_alignment_torch(center0, rotation0, alignment_params):
    delta_t_local = alignment_params[..., :3]
    delta_rot = alignment_params[..., 3:]
    center = center0 + torch.matmul(rotation0, delta_t_local.unsqueeze(-1)).squeeze(-1)
    rotation = rotation0 @ torch_euler_to_rotation_matrix(delta_rot)
    return center, rotation
