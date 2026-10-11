"""P4: geometry-vs-track attribution of a pure geometry injection (Iteration 34 §D).

Noise-free data from a misaligned truth, tracks at the ref-only fit (which,
without noise, is the truth): the target residual is pure misalignment.  The
first-order target residual reduction of the joint step splits into
``f_geo = -<r, A du> / |r|^2`` and ``f_trk = -<r, B dtheta> / |r|^2``.
"""

from __future__ import annotations

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn

INJECTION = (0.01, 0.01, 0.0, 0.0, 0.0, 1.0e-4)
ACTIVE = (0, 1, 5)
MIN_F_GEO = 0.99
MAX_ABS_F_TRK = 0.01
RECOVERY_RTOL = 1.0e-3


def _setup(sign):
    injection = [[sign * value for value in INJECTION]] * 3
    problem = syn.shared_topology_problem(noise_seed=None, injection=injection, n_tracks=12)
    lin = J.linearize_tracks(syn.ref_only_fit(problem), problem.batches, problem.geometry)
    return problem, lin, J.normal_blocks(lin)


def _shares(lin, du, dtheta):
    target = lin.is_target
    r = lin.residual[target]
    A = lin.geometry_jacobian[target]
    B = lin.track_jacobian[target]
    a_step = (A @ du[lin.block_index[target]].unsqueeze(-1)).squeeze(-1)
    tracks = torch.nonzero(target, as_tuple=True)[0]
    b_step = (B @ dtheta[tracks].unsqueeze(-1)).squeeze(-1)
    norm = (r * r).sum()
    return float(-(r * a_step).sum() / norm), float(-(r * b_step).sum() / norm)


def _joint_step(blocks):
    system = J.schur_reduce(blocks, 0.0)
    du = J.solve_geometry_dense(system)
    return du, system.back_substitute(du)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_geometry_carries_the_injected_signal(sign):
    problem, lin, blocks = _setup(sign)
    assert float(lin.residual[lin.is_ref].abs().max()) < 1.0e-8
    du, dtheta = _joint_step(blocks)
    f_geo, f_trk = _shares(lin, du, dtheta)
    print(f"sign {sign:+.0f}: f_geo {f_geo:.6f}  f_trk {f_trk:.2e}")

    assert f_geo >= MIN_F_GEO
    assert abs(f_trk) <= MAX_ABS_F_TRK
    scales = torch.tensor([J.GEOMETRY_SCALES[d] for d in ACTIVE], dtype=torch.float64)
    injected = problem.true_alignment[lin.target_sensors][:, list(ACTIVE)]
    recovered = du * scales
    assert torch.linalg.norm(recovered - injected) <= RECOVERY_RTOL * torch.linalg.norm(injected)
    # Stricter than the pre-registration: the same bound in dimensionless u,
    # where the rotation is not swamped by the translations.
    assert torch.linalg.norm(du - injected / scales) <= RECOVERY_RTOL * torch.linalg.norm(injected / scales)


def test_positive_and_negative_injections_recover_opposite_corrections():
    du_plus, _ = _joint_step(_setup(1.0)[2])
    du_minus, _ = _joint_step(_setup(-1.0)[2])
    assert torch.linalg.norm(du_plus + du_minus) <= RECOVERY_RTOL * torch.linalg.norm(du_plus)


def test_geometry_only_damping_moves_the_signal_into_the_tracks():
    _problem, lin, blocks = _setup(1.0)
    _f_geo, joint_f_trk = _shares(lin, *_joint_step(blocks))
    d_a, d_theta = J.marquardt_scales(blocks)
    control = J.solve_joint_dense(blocks, 1.0e8, (d_a, torch.zeros_like(d_theta)))
    control_f_geo, control_f_trk = _shares(lin, *control)
    print(f"joint f_trk {joint_f_trk:.2e}; geometry-only control f_geo {control_f_geo:.2e} f_trk {control_f_trk:.3f}")
    assert control_f_trk > joint_f_trk
