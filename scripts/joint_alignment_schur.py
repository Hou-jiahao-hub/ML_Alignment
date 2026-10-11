"""Joint track-geometry Gauss-Newton blocks and their Schur reduction.

plan.md §3.3-§3.7 / Phases P2-P4, pre-registered in ``ml_optimization_log.md``
Iteration 34.  Every residual is whitened, ``r = (m - y) / sqrt(V)``, so that
``F = 0.5 r^T r`` and a track's joint normal equations read

    [ C + lam D_a      G           ] [du    ]     [c   ]
    [ G^T              Gamma + lam D_theta ] [dtheta] = - [beta]

with ``Gamma = Gamma_ref + Gamma_tgt``: target residuals stay in the
objective, so their ``B^T B`` stays in the track block.

Geometry increments are the dimensionless ``u`` with ``a = GEOMETRY_SCALES * u``
(block-LM's ``physical_scales``) on the active DOF of target sensors only;
non-target geometry is fixed, so reference rows have no geometry derivative.
Track increments live in the production five-parameter gauge frame, which is
rebuilt from the physical helix at every linearisation and never reused.

Only what the algebraic oracles need exists here: no gain ratio, no damping
adaptation, no sparse production accumulation.  The ``*_dense`` helpers are
for tiny oracle problems and must not be used at production scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from helix_model import (
    MeasurementFitInputs,
    _measurement_standardized_residual_tensors,
    measurement_local_prediction,
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    unpack_gauge_fixed_track_parameters,
)
from ml_alignment_autograd import DEFAULT_ACTIVE_DOF, apply_alignment_torch

#: Units of the dimensionless geometry increment, identical to block-LM's
#: ``physical_scales``: mm for translations, 10 mrad for rotations.
GEOMETRY_SCALES = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01)
N_TRACK_PARAMETERS = 5
DTYPE = torch.float64


class TrackBlockFactorizationError(torch.linalg.LinAlgError):
    """A damped track block ``Gamma + lam D_theta`` is not positive definite."""

    def __init__(self, tracks: torch.Tensor):
        self.tracks = tracks
        super().__init__(
            f"Cholesky failed on {len(tracks)} track blocks "
            f"(first: {tracks[:10].tolist()})."
        )


def geometry_embedding(active_dof: Sequence[int]) -> torch.Tensor:
    """Map the dimensionless active increment ``u`` onto a 6-DOF alignment row."""
    embedding = torch.zeros((6, len(active_dof)), dtype=DTYPE)
    for column, dof in enumerate(active_dof):
        embedding[dof, column] = GEOMETRY_SCALES[dof]
    return embedding


def _hit_residual(
    theta,
    u,
    origin,
    axis0,
    axis1,
    measured,
    variance,
    component_mask,
    center0,
    rotation0,
    base_alignment,
    embedding,
):
    center, rotation = apply_alignment_torch(
        center0, rotation0, base_alignment + embedding @ u
    )
    return _measurement_standardized_residual_tensors(
        theta[None],
        (origin[None], axis0[None], axis1[None]),
        measured[None, None],
        variance[None, None],
        component_mask[None, None],
        center[None, None],
        rotation[None, None],
    )[0]


_HIT_IN_DIMS = (0,) * 11 + (None,)
_BATCHED_HIT_RESIDUAL = torch.func.vmap(_hit_residual, in_dims=_HIT_IN_DIMS)
_BATCHED_HIT_JACOBIAN = torch.func.vmap(
    torch.func.jacrev(_hit_residual, argnums=(0, 1)), in_dims=_HIT_IN_DIMS
)


@dataclass(frozen=True)
class TrackLinearization:
    """Whitened residuals and Jacobians of every ref and target component.

    Shapes: ``residual`` [T, H, 2], ``geometry_jacobian`` (A) [T, H, 2, k],
    ``track_jacobian`` (B) [T, H, 2, 5].  ``component_mask`` is the frozen
    valid set; every quantity is exactly zero outside it.  ``block_index``
    maps target hits to their row in ``target_sensors`` and is -1 elsewhere.
    """

    parameters: torch.Tensor
    residual: torch.Tensor
    geometry_jacobian: torch.Tensor
    track_jacobian: torch.Tensor
    component_mask: torch.Tensor
    is_target: torch.Tensor
    is_ref: torch.Tensor
    block_index: torch.Tensor
    target_sensors: torch.Tensor
    active_dof: tuple[int, ...]


def linearize_tracks(
    parameters: torch.Tensor,
    batches,
    geometry,
    alignment: torch.Tensor | None = None,
    active_dof: Sequence[int] = DEFAULT_ACTIVE_DOF,
    chunk_hits: int = 65_536,
) -> TrackLinearization:
    """Linearise all ref and target components at ``(alignment, parameters)``.

    ``parameters`` are physical helices [T, 6]; the gauge frame is built from
    them here.  Target rows are assembled directly, so the production
    ``_non_target_fit_mask`` guard is neither used nor relaxed.
    """
    active_dof = tuple(int(dof) for dof in active_dof)
    parameters = parameters.detach().to(device="cpu", dtype=DTYPE)
    hit_mask = batches.hit_mask.to("cpu")
    n_tracks, n_hits = hit_mask.shape
    if parameters.shape != (n_tracks, 6):
        raise ValueError("parameters must have shape [tracks, 6].")
    centers0 = geometry.centers.to(DTYPE)
    rotations0 = geometry.rotations.to(DTYPE)
    if alignment is None:
        alignment = torch.zeros((len(centers0), 6), dtype=DTYPE)
    alignment = alignment.detach().to(DTYPE)
    sensor_idx = batches.sensor_idx.to("cpu")
    is_target = batches.is_target.to("cpu") & hit_mask
    if not bool(geometry.target_mask[sensor_idx[is_target]].all()):
        raise ValueError("A target hit sits on a surface that is not a target surface.")
    target_sensors = torch.unique(sensor_idx[is_target])
    lookup = torch.full((len(centers0),), -1, dtype=torch.long)
    lookup[target_sensors] = torch.arange(len(target_sensors))
    block_index = torch.where(is_target, lookup[sensor_idx], -1)

    centers, rotations = apply_alignment_torch(centers0, rotations0, alignment)
    measured = batches.measured_local.to(DTYPE)
    variance = batches.covariances.to(DTYPE)
    component_mask = torch.stack((hit_mask, hit_mask & batches.loc1_mask.to("cpu")), -1)
    _local, valid = measurement_local_prediction(
        parameters,
        MeasurementFitInputs(
            measured=measured,
            variance=variance,
            component_mask=component_mask,
            centers=centers[sensor_idx],
            rotations=rotations[sensor_idx],
        ),
    )
    component_mask = component_mask & valid.unsqueeze(-1)

    origin, axis0, axis1 = reference_gauge_frame(parameters)
    theta = pack_gauge_fixed_track_parameters(parameters)
    embedding = geometry_embedding(active_dof)
    track_index, hit_index = torch.nonzero(hit_mask, as_tuple=True)
    k = len(active_dof)
    residual = torch.zeros((n_tracks, n_hits, 2), dtype=DTYPE)
    geometry_jacobian = torch.zeros((n_tracks, n_hits, 2, k), dtype=DTYPE)
    track_jacobian = torch.zeros((n_tracks, n_hits, 2, N_TRACK_PARAMETERS), dtype=DTYPE)
    for start in range(0, len(track_index), chunk_hits):
        t = track_index[start : start + chunk_hits]
        h = hit_index[start : start + chunk_hits]
        s = sensor_idx[t, h]
        arguments = (
            theta[t],
            torch.zeros((len(t), k), dtype=DTYPE),
            origin[t],
            axis0[t],
            axis1[t],
            measured[t, h],
            variance[t, h],
            component_mask[t, h],
            centers0[s],
            rotations0[s],
            alignment[s],
            embedding,
        )
        residual[t, h] = _BATCHED_HIT_RESIDUAL(*arguments)
        d_theta, d_u = _BATCHED_HIT_JACOBIAN(*arguments)
        track_jacobian[t, h] = d_theta
        # Reference geometry is fixed: its rows carry no geometry derivative.
        geometry_jacobian[t, h] = d_u * is_target[t, h, None, None]
    return TrackLinearization(
        parameters=parameters,
        residual=residual,
        geometry_jacobian=geometry_jacobian,
        track_jacobian=track_jacobian,
        component_mask=component_mask,
        is_target=is_target,
        is_ref=hit_mask & ~is_target,
        block_index=block_index,
        target_sensors=target_sensors,
        active_dof=active_dof,
    )


def commit_track_step(parameters: torch.Tensor, delta_theta: torch.Tensor) -> torch.Tensor:
    """Apply a gauge-frame increment and return the physical helices.

    The frame is rebuilt from ``parameters``, the state the increment was
    solved at; the result is physical, so the next linearisation builds a new
    frame and no 5-vector ever crosses an iteration.
    """
    frame = reference_gauge_frame(parameters)
    return unpack_gauge_fixed_track_parameters(
        pack_gauge_fixed_track_parameters(parameters) + delta_theta, frame
    )


@dataclass(frozen=True)
class JointNormalBlocks:
    """Per-track ``Gamma_ref``, ``Gamma_tgt``, ``beta`` and per (track, sensor) pair
    ``C``, ``G``, ``c``.  Pairs exist only where a track crosses a target sensor."""

    gamma_ref: torch.Tensor
    gamma_tgt: torch.Tensor
    beta_ref: torch.Tensor
    beta_tgt: torch.Tensor
    pair_track: torch.Tensor
    pair_block: torch.Tensor
    C: torch.Tensor
    G: torch.Tensor
    c: torch.Tensor
    n_blocks: int

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_ref + self.gamma_tgt

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_ref + self.beta_tgt

    @property
    def n_tracks(self) -> int:
        return self.gamma_ref.shape[0]

    @property
    def block_size(self) -> int:
        return self.C.shape[-1]

    def geometry_information(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed-track ``C`` and ``c`` summed per sensor: [n, k, k], [n, k]."""
        k = self.block_size
        information = torch.zeros((self.n_blocks, k, k), dtype=DTYPE)
        gradient = torch.zeros((self.n_blocks, k), dtype=DTYPE)
        information.index_add_(0, self.pair_block, self.C)
        gradient.index_add_(0, self.pair_block, self.c)
        return information, gradient


def normal_blocks(lin: TrackLinearization) -> JointNormalBlocks:
    B, A, r = lin.track_jacobian, lin.geometry_jacobian, lin.residual
    ref = lin.is_ref.to(DTYPE)
    tgt = lin.is_target.to(DTYPE)
    gamma_ref = torch.einsum("thci,thcj,th->tij", B, B, ref)
    gamma_tgt = torch.einsum("thci,thcj,th->tij", B, B, tgt)
    beta_ref = torch.einsum("thci,thc,th->ti", B, r, ref)
    beta_tgt = torch.einsum("thci,thc,th->ti", B, r, tgt)

    n_blocks = len(lin.target_sensors)
    track_index, hit_index = torch.nonzero(lin.is_target, as_tuple=True)
    keys = track_index * n_blocks + lin.block_index[track_index, hit_index]
    pair_keys, inverse = torch.unique(keys, return_inverse=True)
    n_pairs, k = len(pair_keys), A.shape[-1]
    A_hit = A[track_index, hit_index]
    B_hit = B[track_index, hit_index]
    r_hit = r[track_index, hit_index]
    C = torch.zeros((n_pairs, k, k), dtype=DTYPE).index_add_(
        0, inverse, A_hit.transpose(-1, -2) @ A_hit
    )
    G = torch.zeros((n_pairs, k, N_TRACK_PARAMETERS), dtype=DTYPE).index_add_(
        0, inverse, A_hit.transpose(-1, -2) @ B_hit
    )
    c = torch.zeros((n_pairs, k), dtype=DTYPE).index_add_(
        0, inverse, (A_hit.transpose(-1, -2) @ r_hit.unsqueeze(-1)).squeeze(-1)
    )
    return JointNormalBlocks(
        gamma_ref=gamma_ref,
        gamma_tgt=gamma_tgt,
        beta_ref=beta_ref,
        beta_tgt=beta_tgt,
        pair_track=pair_keys // n_blocks,
        pair_block=pair_keys % n_blocks,
        C=C,
        G=G,
        c=c,
        n_blocks=n_blocks,
    )


def marquardt_scales(blocks: JointNormalBlocks) -> tuple[torch.Tensor, torch.Tensor]:
    """``D_a = diag(C)`` per sensor and ``D_theta = diag(Gamma)`` per track, no floor."""
    information, _gradient = blocks.geometry_information()
    return (
        information.diagonal(dim1=-2, dim2=-1).clone(),
        blocks.gamma.diagonal(dim1=-2, dim2=-1).clone(),
    )


def _same_track_pairs(pair_track: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """All ordered (p, q) pair indices that share a track, diagonal included."""
    order = torch.argsort(pair_track, stable=True)
    counts = torch.bincount(pair_track[order])
    starts = torch.cumsum(counts, 0) - counts
    tracks = torch.nonzero(counts, as_tuple=True)[0]
    left, right = [], []
    for m in range(int(counts.max()) if len(counts) else 0):
        for n in range(int(counts.max())):
            owners = tracks[counts[tracks] > max(m, n)]
            left.append(order[starts[owners] + m])
            right.append(order[starts[owners] + n])
    if not left:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    return torch.cat(left), torch.cat(right)


@dataclass(frozen=True)
class SchurSystem:
    """Reduced geometry system ``S du = -b`` in coordinate block form.

    ``rows``/``cols``/``values`` hold only blocks that exist: the per-sensor
    diagonal and sensor pairs co-touched by at least one track.
    """

    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    rhs: torch.Tensor
    lam: float
    track_step: torch.Tensor
    track_gain: torch.Tensor
    pair_track: torch.Tensor
    pair_block: torch.Tensor

    @property
    def n_blocks(self) -> int:
        return self.rhs.shape[0]

    def block_keys(self) -> set[tuple[int, int]]:
        return set(zip(self.rows.tolist(), self.cols.tolist()))

    def to_dense(self) -> torch.Tensor:
        """Dense ``S``; oracle-sized problems only."""
        n, k = self.rhs.shape
        dense = torch.zeros((n, k, n, k), dtype=DTYPE)
        dense[self.rows, :, self.cols, :] = self.values
        return dense.reshape(n * k, n * k)

    def back_substitute(self, delta_u: torch.Tensor) -> torch.Tensor:
        """``dtheta = -(Gamma + lam D_theta)^-1 (beta + G^T du)`` per track."""
        coupled = (self.track_gain @ delta_u[self.pair_block].unsqueeze(-1)).squeeze(-1)
        return -self.track_step - torch.zeros_like(self.track_step).index_add_(
            0, self.pair_track, coupled
        )


def schur_reduce(
    blocks: JointNormalBlocks,
    lam: float = 0.0,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> SchurSystem:
    """Eliminate every track from the jointly damped system.

    ``lam = 0`` is the undamped Gauss-Newton reduction the Woodbury oracle
    checks.  For ``lam > 0`` both blocks are damped, ``C + lam D_a`` and
    ``Gamma + lam D_theta``, so ``lam -> inf`` moves neither geometry nor tracks.
    """
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    if scales is None:
        scales = marquardt_scales(blocks)
    d_a, d_theta = scales
    if lam > 0.0 and not (bool((d_a > 0).all()) and bool((d_theta > 0).all())):
        raise ValueError(
            "A damped direction has zero Hessian diagonal; remove inactive "
            "directions instead of damping them."
        )
    damped_gamma = blocks.gamma + lam * torch.diag_embed(d_theta)
    factor, info = torch.linalg.cholesky_ex(damped_gamma)
    if bool((info != 0).any()):
        raise TrackBlockFactorizationError(torch.nonzero(info, as_tuple=True)[0])
    track_step = torch.cholesky_solve(blocks.beta.unsqueeze(-1), factor).squeeze(-1)
    track_gain = torch.cholesky_solve(
        blocks.G.transpose(-1, -2), factor[blocks.pair_track]
    )

    information, gradient = blocks.geometry_information()
    information = information + lam * torch.diag_embed(d_a)
    rhs = gradient - torch.zeros_like(gradient).index_add_(
        0,
        blocks.pair_block,
        (blocks.G @ track_step[blocks.pair_track].unsqueeze(-1)).squeeze(-1),
    )
    left, right = _same_track_pairs(blocks.pair_track)
    n = blocks.n_blocks
    rows = torch.cat((torch.arange(n), blocks.pair_block[left]))
    cols = torch.cat((torch.arange(n), blocks.pair_block[right]))
    values = torch.cat((information, -blocks.G[left] @ track_gain[right]))
    keys, inverse = torch.unique(rows * n + cols, return_inverse=True)
    coalesced = torch.zeros((len(keys), *values.shape[1:]), dtype=DTYPE)
    coalesced.index_add_(0, inverse, values)
    return SchurSystem(
        rows=keys // n,
        cols=keys % n,
        values=coalesced,
        rhs=rhs,
        lam=lam,
        track_step=track_step,
        track_gain=track_gain,
        pair_track=blocks.pair_track,
        pair_block=blocks.pair_block,
    )


def solve_geometry_dense(system: SchurSystem) -> torch.Tensor:
    """``du = -S^-1 b`` by a dense solve; oracle-sized problems only."""
    n, k = system.rhs.shape
    return -torch.linalg.solve(system.to_dense(), system.rhs.reshape(n * k)).reshape(n, k)


def solve_joint_dense(
    blocks: JointNormalBlocks,
    lam: float = 0.0,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the full damped joint system without elimination; oracle only."""
    if scales is None:
        scales = marquardt_scales(blocks)
    d_a, d_theta = scales
    n, k, n_tracks = blocks.n_blocks, blocks.block_size, blocks.n_tracks
    size = n * k + n_tracks * N_TRACK_PARAMETERS
    hessian = torch.zeros((size, size), dtype=DTYPE)
    gradient = torch.zeros(size, dtype=DTYPE)
    information, geometry_gradient = blocks.geometry_information()
    information = information + lam * torch.diag_embed(d_a)
    for block in range(n):
        rows = slice(block * k, (block + 1) * k)
        hessian[rows, rows] = information[block]
        gradient[rows] = geometry_gradient[block]
    offset = n * k
    damped_gamma = blocks.gamma + lam * torch.diag_embed(d_theta)
    for track in range(n_tracks):
        rows = slice(offset + track * 5, offset + (track + 1) * 5)
        hessian[rows, rows] = damped_gamma[track]
        gradient[rows] = blocks.beta[track]
    for pair in range(len(blocks.pair_track)):
        geo = slice(int(blocks.pair_block[pair]) * k, (int(blocks.pair_block[pair]) + 1) * k)
        trk = int(blocks.pair_track[pair])
        trk = slice(offset + trk * 5, offset + (trk + 1) * 5)
        hessian[geo, trk] += blocks.G[pair]
        hessian[trk, geo] += blocks.G[pair].T
    step = -torch.linalg.solve(hessian, gradient)
    return step[:offset].reshape(n, k), step[offset:].reshape(n_tracks, 5)


def _symmetric_block_summary(matrices: torch.Tensor) -> dict[str, torch.Tensor]:
    norm = torch.linalg.matrix_norm(matrices).clamp_min(1.0e-300)
    asymmetry = torch.linalg.matrix_norm(matrices - matrices.transpose(-1, -2)) / norm
    finite = torch.isfinite(matrices).all(dim=(-1, -2))
    safe = torch.where(finite[:, None, None], matrices, torch.eye(matrices.shape[-1], dtype=DTYPE))
    _factor, info = torch.linalg.cholesky_ex(safe)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (safe + safe.transpose(-1, -2)))
    smallest, largest = eigenvalues[:, 0], eigenvalues[:, -1]
    condition = torch.where(
        smallest > 0, largest / smallest, torch.full_like(smallest, float("inf"))
    )
    return {
        "finite": finite,
        "asymmetry": asymmetry,
        "cholesky_ok": (info == 0) & finite,
        "eigenvalues": eigenvalues,
        "rank": torch.linalg.matrix_rank(safe),
        "condition": condition,
    }


def track_block_diagnostics(blocks: JointNormalBlocks) -> dict[str, dict[str, torch.Tensor]]:
    """Finiteness, symmetry, Cholesky, spectrum, rank and condition of ``Gamma_ref`` and ``Gamma``."""
    return {
        "gamma_ref": _symmetric_block_summary(blocks.gamma_ref),
        "gamma": _symmetric_block_summary(blocks.gamma),
    }
