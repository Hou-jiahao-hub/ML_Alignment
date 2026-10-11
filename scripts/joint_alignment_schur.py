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

The global ``schur-lm`` convergence and constraint contract (plan.md §P6,
Iteration 37) lives here as pure functions: the undamped joint model and gain
ratio, the damping rule, the box active set with a uniformly scaled feasible
step, the global decrement with its rank test, and the terminal state.  The
2026-09-08 block-LM contract is not reused: shared tracks couple the target
sensors, so there are no per-sensor damping, terminal states or retirement.
Legacy per-DOF post-solve step clipping does not exist on this path.

Production scale (plan.md §P7, Iteration 38): the reduced system is
assembled as CSC from its 3x3 coordinate blocks and solved with SuperLU
(``solve_geometry_sparse``); the decrement and rank test run per connected
component of the sensor coupling graph (``global_stationarity_sparse``).
There is no LM loop yet.  The ``*_dense`` helpers, ``global_stationarity``
included, remain tiny-problem oracles.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from helix_model import (
    MeasurementFitInputs,
    _measurement_standardized_residual_tensors,
    measurement_local_prediction,
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    unpack_gauge_fixed_track_parameters,
)
from ml_alignment_autograd import (
    BLOCK_LM_BOUND_ATOL,
    DEFAULT_ACTIVE_DOF,
    JOINT_CLAMP_R,
    JOINT_CLAMP_T,
    _block_lm_free_dof_mask,
    apply_alignment_torch,
)

#: Units of the dimensionless geometry increment, identical to block-LM's
#: ``physical_scales``: mm for translations, 10 mrad for rotations.
GEOMETRY_SCALES = (1.0, 1.0, 1.0, 0.01, 0.01, 0.01)
N_TRACK_PARAMETERS = 5
DTYPE = torch.float64

#: Pre-registration of the ``schur-lm`` contract (``ml_optimization_log.md``
#: Iteration 37 §B).  Independent of ``BLOCK_LM_PREREGISTRATION_VERSION``: the
#: block-LM constants govern ``block-lm`` only and none is inherited here.
SCHUR_LM_PREREGISTRATION_VERSION = 1
#: Set explicitly, not taken from ``LM_INITIAL_DAMPING``.  With Marquardt
#: scaling lambda is a relative curvature fraction; 1e-3 makes the first step
#: nearly Gauss-Newton.  Its influence is checked by the scan below (P8).
SCHUR_LM_LAMBDA_INIT = 1.0e-3
SCHUR_LM_LAMBDA_INIT_SCAN = (1.0e-5, 1.0e-3, 1.0e-1)
SCHUR_LM_LAMBDA_INCREASE = 10.0
SCHUR_LM_LAMBDA_DECREASE = 10.0
#: Moré (1978): accept any real decrease that is not rounding noise; tighten
#: below 0.25, relax above 0.75.
SCHUR_LM_RHO_ACCEPT_MIN = 1.0e-4
SCHUR_LM_RHO_INCREASE_THRESHOLD = 0.25
SCHUR_LM_RHO_DECREASE_THRESHOLD = 0.75
#: At 1e12 the step is ~1e-12 of the Gauss-Newton step, below the decrease a
#: float64 objective can resolve.
SCHUR_LM_LAMBDA_MIN = 1.0e-12
SCHUR_LM_LAMBDA_MAX = 1.0e12
#: Rejections per outer iteration.  Eight decades leave a ~1e-8 step along
#: steepest descent, which must descend on a smooth objective; failing that
#: falsifies smoothness (intersection validity) and is reported, not retried.
SCHUR_LM_MAX_RETRIES = 8
#: Outer budget; block-LM converged in 13 iterations in Iteration 30.
SCHUR_LM_MAX_ITERS = 60
#: The decrement is the squared length of the remaining Newton step in the
#: Gauss-Newton information metric, i.e. in statistical sigma.  1e-2 means the
#: geometry, and every single track, has less than 0.1 sigma left to move.
#: This is a global quantity; the per-component 1e-6 of block-LM is not reused.
SCHUR_LM_EPSILON_GEO = 1.0e-2
SCHUR_LM_EPSILON_TRK = 1.0e-2
#: Eigenvalue floor of the Jacobi-scaled reduced information.  Below it a
#: combination of DOFs is at least 100x less constrained than its diagonal
#: information suggests.
SCHUR_LM_RANK_TOL = 1.0e-4
#: Consecutive accepted steps with ``alpha`` below ``SCHUR_LM_ALPHA_STALL``
#: before the run is ``constraint_stalled``.
SCHUR_LM_ALPHA_STALL = 1.0e-3
SCHUR_LM_STALL_ITERS = 3

SCHUR_LM_STATUS_CONVERGED = "converged"
SCHUR_LM_STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
SCHUR_LM_STATUS_RANK_DEFICIENT = "rank_deficient"
SCHUR_LM_STATUS_INTERSECTION_FAILED = "intersection_failed"
SCHUR_LM_STATUS_NUMERICAL_FAILURE = "numerical_failure"
SCHUR_LM_STATUS_CONSTRAINT_STALLED = "constraint_stalled"
SCHUR_LM_TERMINAL_STATUSES = (
    SCHUR_LM_STATUS_CONVERGED,
    SCHUR_LM_STATUS_BUDGET_EXHAUSTED,
    SCHUR_LM_STATUS_RANK_DEFICIENT,
    SCHUR_LM_STATUS_INTERSECTION_FAILED,
    SCHUR_LM_STATUS_NUMERICAL_FAILURE,
    SCHUR_LM_STATUS_CONSTRAINT_STALLED,
)


class ModelReductionError(ArithmeticError):
    """The joint model predicts no decrease: a numerical failure, not a rejection."""


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


@dataclass(frozen=True)
class _HitInputs:
    """Per-hit arguments of ``_hit_residual`` at one ``(alignment, parameters)`` point."""

    component_mask: torch.Tensor
    is_target: torch.Tensor
    block_index: torch.Tensor
    target_sensors: torch.Tensor
    track_index: torch.Tensor
    hit_index: torch.Tensor
    arguments: tuple
    k: int

    def chunk(self, start: int, stop: int):
        t = self.track_index[start:stop]
        h = self.hit_index[start:stop]
        (theta, origin, axis0, axis1, measured, variance, mask, centers0, rotations0, alignment,
         sensor_idx, embedding) = self.arguments
        s = sensor_idx[t, h]
        return t, h, (
            theta[t],
            torch.zeros((len(t), self.k), dtype=DTYPE),
            origin[t],
            axis0[t],
            axis1[t],
            measured[t, h],
            variance[t, h],
            mask[t, h],
            centers0[s],
            rotations0[s],
            alignment[s],
            embedding,
        )


def _hit_inputs(parameters, batches, geometry, alignment, active_dof) -> _HitInputs:
    hit_mask = batches.hit_mask.to("cpu")
    n_tracks, _n_hits = hit_mask.shape
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
    track_index, hit_index = torch.nonzero(hit_mask, as_tuple=True)
    return _HitInputs(
        component_mask=component_mask,
        is_target=is_target,
        block_index=block_index,
        target_sensors=target_sensors,
        track_index=track_index,
        hit_index=hit_index,
        arguments=(
            theta, origin, axis0, axis1, measured, variance, component_mask, centers0, rotations0,
            alignment, sensor_idx, geometry_embedding(active_dof),
        ),
        k=len(active_dof),
    )


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
    inputs = _hit_inputs(parameters, batches, geometry, alignment, active_dof)
    n_tracks, n_hits = inputs.is_target.shape
    k = inputs.k
    residual = torch.zeros((n_tracks, n_hits, 2), dtype=DTYPE)
    geometry_jacobian = torch.zeros((n_tracks, n_hits, 2, k), dtype=DTYPE)
    track_jacobian = torch.zeros((n_tracks, n_hits, 2, N_TRACK_PARAMETERS), dtype=DTYPE)
    for start in range(0, len(inputs.track_index), chunk_hits):
        t, h, arguments = inputs.chunk(start, start + chunk_hits)
        residual[t, h] = _BATCHED_HIT_RESIDUAL(*arguments)
        d_theta, d_u = _BATCHED_HIT_JACOBIAN(*arguments)
        track_jacobian[t, h] = d_theta
        # Reference geometry is fixed: its rows carry no geometry derivative.
        geometry_jacobian[t, h] = d_u * inputs.is_target[t, h, None, None]
    hit_mask = batches.hit_mask.to("cpu")
    return TrackLinearization(
        parameters=parameters,
        residual=residual,
        geometry_jacobian=geometry_jacobian,
        track_jacobian=track_jacobian,
        component_mask=inputs.component_mask,
        is_target=inputs.is_target,
        is_ref=hit_mask & ~inputs.is_target,
        block_index=inputs.block_index,
        target_sensors=inputs.target_sensors,
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


def to_sparse(system: SchurSystem) -> csc_matrix:
    """``S`` as CSC.  Block keys are sorted, so the assembly is deterministic."""
    n, k = system.rhs.shape
    offsets = torch.arange(k)
    rows = (system.rows[:, None, None] * k + offsets[None, :, None]).expand(-1, k, k)
    cols = (system.cols[:, None, None] * k + offsets[None, None, :]).expand(-1, k, k)
    return csc_matrix(
        (
            system.values.reshape(-1).numpy(),
            (rows.reshape(-1).numpy(), cols.reshape(-1).numpy()),
        ),
        shape=(n * k, n * k),
    )


def solve_geometry_sparse(system: SchurSystem) -> torch.Tensor:
    """``du = -S^-1 b`` by SuperLU with symmetric ordering and diagonal pivots.

    A singular factor raises ``RuntimeError``; the loop reports it as a
    numerical failure.
    """
    n, k = system.rhs.shape
    factor = splu(
        to_sparse(system),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    solution = factor.solve(system.rhs.reshape(n * k).numpy())
    return -torch.from_numpy(solution).reshape(n, k)


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


@dataclass(frozen=True)
class ModelReduction:
    """Undamped joint Gauss-Newton model at a step ``p``: ``g^T p``, ``p^T H p``
    and the predicted decrease ``-g^T p - p^T H p / 2``."""

    gradient_step: float
    curvature: float
    predicted: float


def joint_model_reduction(
    blocks: JointNormalBlocks, delta_u: torch.Tensor, delta_theta: torch.Tensor
) -> ModelReduction:
    """Predicted decrease of ``F`` at the step actually executed.

    The model is the undamped one: lambda only chooses the step, so the gain
    ratio compares like with like and equals one on linear residuals.  Target
    hits depend on one sensor each, so ``C`` has no cross-sensor term.
    """
    du = delta_u[blocks.pair_block]
    dtheta = delta_theta[blocks.pair_track]
    gradient_step = float(
        torch.einsum("pi,pi->", blocks.c, du) + torch.einsum("ti,ti->", blocks.beta, delta_theta)
    )
    curvature = float(
        torch.einsum("pi,pij,pj->", du, blocks.C, du)
        + 2.0 * torch.einsum("pi,pij,pj->", du, blocks.G, dtheta)
        + torch.einsum("ti,tij,tj->", delta_theta, blocks.gamma, delta_theta)
    )
    return ModelReduction(gradient_step, curvature, -gradient_step - 0.5 * curvature)


def gain_ratio(actual: float, predicted: float) -> float:
    """``rho = actual / predicted``; NaN when the trial objective is not finite.

    An LM step scaled by ``alpha`` in (0, 1] always has a positive predicted
    decrease, so a non-positive or non-finite one is a numerical failure.
    """
    predicted = float(predicted)
    if not math.isfinite(predicted) or predicted <= 0.0:
        raise ModelReductionError(f"Predicted joint decrease is {predicted!r}.")
    actual = float(actual)
    return actual / predicted if math.isfinite(actual) else math.nan


@dataclass(frozen=True)
class DampingDecision:
    accepted: bool
    lam: float
    branch: str
    #: Rejected while lambda was already at its ceiling: the outer iteration
    #: has failed and the run must stop.
    saturated: bool


def update_damping(lam: float, rho: float, valid: bool = True) -> DampingDecision:
    """The four-branch deterministic damping rule of Iteration 37 §B."""
    lam = float(lam)
    raised = min(lam * SCHUR_LM_LAMBDA_INCREASE, SCHUR_LM_LAMBDA_MAX)
    if not valid or not math.isfinite(rho) or rho < SCHUR_LM_RHO_ACCEPT_MIN:
        return DampingDecision(False, raised, "reject", lam >= SCHUR_LM_LAMBDA_MAX)
    if rho < SCHUR_LM_RHO_INCREASE_THRESHOLD:
        return DampingDecision(True, raised, "accept_increase", False)
    if rho < SCHUR_LM_RHO_DECREASE_THRESHOLD:
        return DampingDecision(True, lam, "accept_keep", False)
    lowered = max(lam / SCHUR_LM_LAMBDA_DECREASE, SCHUR_LM_LAMBDA_MIN)
    return DampingDecision(True, lowered, "accept_decrease", False)


def geometry_bounds(
    active_dof: Sequence[int],
    clamp_t: float = JOINT_CLAMP_T,
    clamp_r: float = JOINT_CLAMP_R,
) -> torch.Tensor:
    """Physical half-widths of the box ``|a| <= bound``, block-LM's feasible region."""
    return torch.tensor([clamp_t if dof < 3 else clamp_r for dof in active_dof], dtype=DTYPE)


def free_geometry_mask(
    values: torch.Tensor, gradient: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    """DOFs that may still move: not on a bound whose gradient points outwards.

    ``values`` are the physical active alignment [n, k]; ``gradient`` is the
    undamped reduced gradient ``b = c - G Gamma^-1 beta`` (only its sign is read).
    """
    return _block_lm_free_dof_mask(values, gradient, -bounds, bounds)


def restrict_to_free(system: SchurSystem, free: torch.Tensor) -> SchurSystem:
    """Delete frozen DOFs from the reduced system.

    Track elimination does not depend on which geometry DOFs are frozen, so
    deleting rows and columns of the joint system is ``S[F, F]``, ``b[F]``.
    Frozen DOFs keep an identity row and a zero right-hand side, which pins
    them at ``du = 0`` without changing the free block.
    """
    free = free.to(DTYPE)
    mask = free[system.rows].unsqueeze(-1) * free[system.cols].unsqueeze(-2)
    pinned = torch.diag_embed(1.0 - free)[system.rows] * (system.rows == system.cols)[:, None, None]
    return dataclasses.replace(
        system, values=system.values * mask + pinned, rhs=system.rhs * free
    )


def _physical_step(delta_u: torch.Tensor, active_dof: Sequence[int]) -> torch.Tensor:
    scales = torch.tensor([GEOMETRY_SCALES[dof] for dof in active_dof], dtype=DTYPE)
    return delta_u * scales


def step_to_bound(
    values: torch.Tensor,
    delta_u: torch.Tensor,
    bounds: torch.Tensor,
    active_dof: Sequence[int],
) -> tuple[float, torch.Tensor]:
    """Largest ``alpha`` in [0, 1] keeping ``values + alpha * step`` in the box.

    Returns ``alpha`` and the DOFs that block it.  ``alpha = 0`` means a DOF on
    its bound is pushed outwards by the coupled step.
    """
    step = _physical_step(delta_u, active_dof)
    room = torch.where(step > 0.0, bounds - values, -bounds - values)
    with torch.no_grad():
        limits = torch.where(step != 0.0, room / step, torch.full_like(step, math.inf))
    limits = limits.clamp_min(0.0)
    alpha = min(1.0, float(limits.min())) if limits.numel() else 1.0
    blocking = (limits <= alpha) & (step != 0.0) if alpha < 1.0 else torch.zeros_like(step, dtype=torch.bool)
    return alpha, blocking


def apply_geometry_step(
    values: torch.Tensor,
    delta_u: torch.Tensor,
    alpha: float,
    bounds: torch.Tensor,
    active_dof: Sequence[int],
) -> torch.Tensor:
    """``values + alpha * step`` with the blocking DOFs snapped onto their bound.

    The snap only removes rounding of ``alpha * step``; anything beyond
    ``BLOCK_LM_BOUND_ATOL`` outside the box is an error, never a clamp.
    """
    moved = values + alpha * _physical_step(delta_u, active_dof)
    excess = moved.abs() - bounds
    if bool((excess > BLOCK_LM_BOUND_ATOL).any()):
        raise ValueError("Geometry step leaves the box; scale it with step_to_bound first.")
    return torch.where(excess > 0.0, torch.sign(moved) * bounds, moved)


@dataclass(frozen=True)
class FeasibleStep:
    delta_u: torch.Tensor
    delta_theta: torch.Tensor
    alpha: float
    free: torch.Tensor
    n_resolves: int


def feasible_joint_step(
    system: SchurSystem,
    values: torch.Tensor,
    free: torch.Tensor,
    bounds: torch.Tensor,
    active_dof: Sequence[int],
    solve=None,
) -> FeasibleStep:
    """Solve on the free set and scale the whole joint step to stay feasible.

    When the coupled step drives a bound DOF outwards (``alpha = 0``), that DOF
    joins the frozen set and the same linearisation is solved again.  The
    returned ``delta_u``/``delta_theta`` are unscaled; the executed step is
    ``alpha`` times both, and the model decrease must be taken there.
    """
    solve = solve_geometry_dense if solve is None else solve
    free = free.clone()
    n_resolves = 0
    while True:
        restricted = restrict_to_free(system, free)
        delta_u = solve(restricted) if bool(free.any()) else torch.zeros_like(system.rhs)
        alpha, blocking = step_to_bound(values, delta_u, bounds, active_dof)
        if alpha > 0.0 or not bool(free.any()):
            return FeasibleStep(delta_u, restricted.back_substitute(delta_u), alpha, free, n_resolves)
        free &= ~blocking
        n_resolves += 1


def update_stall_count(count: int, alpha: float) -> int:
    """Consecutive accepted steps whose ``alpha`` fell below ``SCHUR_LM_ALPHA_STALL``."""
    return count + 1 if alpha < SCHUR_LM_ALPHA_STALL else 0


@dataclass(frozen=True)
class GlobalStationarity:
    lambda2_geo: float
    lambda2_trk: torch.Tensor
    lambda2_trk_max: float
    n_free: int
    rank_deficient: bool
    lambda2_is_lower_bound: bool
    min_scaled_eigenvalue: float
    null_gradient_max: float
    #: Free DOF of the largest coupled component the eigendecomposition saw.
    largest_component_dof: int = 0

    @property
    def stationary(self) -> bool:
        """Both thresholds met.  Not ``converged`` on its own: see ``rank_deficient``."""
        return (
            self.lambda2_geo <= SCHUR_LM_EPSILON_GEO
            and self.lambda2_trk_max <= SCHUR_LM_EPSILON_TRK
        )


@dataclass(frozen=True)
class _ScaledDecrement:
    lambda2_geo: float
    weak: bool
    min_scaled_eigenvalue: float
    null_gradient_max: float


def _scaled_decrement(
    information: torch.Tensor, gradient: torch.Tensor, rank_tol: float
) -> _ScaledDecrement | None:
    """``b^T S^-1 b`` over the strong eigendirections of the Jacobi-scaled ``S``.

    ``None`` when ``diag(S)`` has a non-positive entry.
    """
    diagonal = information.diagonal()
    if not bool((diagonal > 0.0).all()):
        return None
    scale = diagonal.rsqrt()
    scaled = information * scale[:, None] * scale[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (scaled + scaled.T))
    projection = eigenvectors.T @ (gradient * scale)
    weak = eigenvalues <= rank_tol
    strong = ~weak
    return _ScaledDecrement(
        lambda2_geo=float((projection[strong].square() / eigenvalues[strong]).sum()),
        weak=bool(weak.any()),
        min_scaled_eigenvalue=float(eigenvalues[0]),
        null_gradient_max=float(projection[weak].abs().max()) if bool(weak.any()) else 0.0,
    )


def _track_decrement(blocks: JointNormalBlocks, system: SchurSystem, free):
    if system.lam != 0.0:
        raise ValueError("The decrement is defined on the undamped system (lam = 0).")
    lambda2_trk = torch.einsum("ti,ti->t", blocks.beta, system.track_step)
    trk_max = float(lambda2_trk.max()) if lambda2_trk.numel() else 0.0
    if free is None:
        free = torch.ones_like(system.rhs, dtype=torch.bool)
    return lambda2_trk, trk_max, free


def global_stationarity(
    blocks: JointNormalBlocks,
    system: SchurSystem,
    free: torch.Tensor | None = None,
    rank_tol: float = SCHUR_LM_RANK_TOL,
) -> GlobalStationarity:
    """Joint Newton decrement ``g^T H^-1 g = b_F^T S_F^-1 b_F + sum_t beta_t^T Gamma_t^-1 beta_t``.

    Needs the undamped system.  The geometry part is summed in the eigenbasis
    of the Jacobi-scaled ``S_F`` without any regularisation: directions at or
    below ``rank_tol`` are dropped, which makes the value a lower bound and
    the run ``rank_deficient``.  Dense eigh: oracle-sized problems only.
    """
    lambda2_trk, trk_max, free = _track_decrement(blocks, system, free)
    keep = free.flatten()
    n_free = int(keep.sum())
    if n_free == 0:
        return GlobalStationarity(0.0, lambda2_trk, trk_max, 0, False, False, math.nan, 0.0)
    result = _scaled_decrement(
        system.to_dense()[keep][:, keep], system.rhs.flatten()[keep], rank_tol
    )
    if result is None:
        return GlobalStationarity(math.nan, lambda2_trk, trk_max, n_free, True, True, math.nan, math.nan)
    return GlobalStationarity(
        lambda2_geo=result.lambda2_geo,
        lambda2_trk=lambda2_trk,
        lambda2_trk_max=trk_max,
        n_free=n_free,
        rank_deficient=result.weak,
        lambda2_is_lower_bound=result.weak,
        min_scaled_eigenvalue=result.min_scaled_eigenvalue,
        null_gradient_max=result.null_gradient_max,
        largest_component_dof=n_free,
    )


def coupling_components(system: SchurSystem) -> tuple[int, torch.Tensor]:
    """Connected components of the sensor graph whose edges are off-diagonal blocks."""
    n = system.n_blocks
    off = system.rows != system.cols
    graph = csr_matrix(
        (
            torch.ones(int(off.sum()), dtype=DTYPE).numpy(),
            (system.rows[off].numpy(), system.cols[off].numpy()),
        ),
        shape=(n, n),
    )
    n_components, labels = connected_components(graph, directed=False)
    return int(n_components), torch.from_numpy(labels).to(torch.long)


def global_stationarity_sparse(
    blocks: JointNormalBlocks,
    system: SchurSystem,
    free: torch.Tensor | None = None,
    rank_tol: float = SCHUR_LM_RANK_TOL,
) -> GlobalStationarity:
    """``global_stationarity`` for production: one dense eigh per coupled component.

    ``S`` is block diagonal over the connected components of the coupling
    graph, so the spectrum and the decrement are their union and sum; only a
    component's own block is ever densified (Iteration 38 §A.2).
    """
    lambda2_trk, trk_max, free = _track_decrement(blocks, system, free)
    n_free = int(free.sum())
    if n_free == 0:
        return GlobalStationarity(0.0, lambda2_trk, trk_max, 0, False, False, math.nan, 0.0)
    k = system.rhs.shape[1]
    n_components, labels = coupling_components(system)
    matrix = to_sparse(system).tocsr()
    gradient = system.rhs.flatten()
    dof_free = free.flatten()
    dof_label = labels.repeat_interleave(k)
    lambda2_geo, weak, largest = 0.0, False, 0
    min_eigenvalue, null_max = math.inf, 0.0
    for component in range(n_components):
        dofs = torch.nonzero((dof_label == component) & dof_free, as_tuple=True)[0]
        if len(dofs) == 0:
            continue
        index = dofs.numpy()
        information = torch.from_numpy(matrix[index][:, index].toarray())
        result = _scaled_decrement(information, gradient[dofs], rank_tol)
        if result is None:
            return GlobalStationarity(
                math.nan, lambda2_trk, trk_max, n_free, True, True, math.nan, math.nan, len(dofs)
            )
        lambda2_geo += result.lambda2_geo
        weak = weak or result.weak
        min_eigenvalue = min(min_eigenvalue, result.min_scaled_eigenvalue)
        null_max = max(null_max, result.null_gradient_max)
        largest = max(largest, len(dofs))
    return GlobalStationarity(
        lambda2_geo=lambda2_geo,
        lambda2_trk=lambda2_trk,
        lambda2_trk_max=trk_max,
        n_free=n_free,
        rank_deficient=weak,
        lambda2_is_lower_bound=weak,
        min_scaled_eigenvalue=min_eigenvalue,
        null_gradient_max=null_max,
        largest_component_dof=largest,
    )


def global_terminal_status(
    *,
    numerical_failure: bool = False,
    retries_exhausted: bool = False,
    component_loss_seen: bool = False,
    rank_deficient: bool = False,
    constraint_stalled: bool = False,
    stationary: bool = False,
) -> str:
    """One run-level state, highest priority first (Iteration 37 §C).

    Rank deficiency outranks stationarity: the decrement is then a lower bound,
    and a biased-low statistic must not certify convergence.  Retries that run
    out without any component loss are a damping ceiling, i.e. budget.
    """
    if numerical_failure:
        return SCHUR_LM_STATUS_NUMERICAL_FAILURE
    if retries_exhausted and component_loss_seen:
        return SCHUR_LM_STATUS_INTERSECTION_FAILED
    if rank_deficient:
        return SCHUR_LM_STATUS_RANK_DEFICIENT
    if constraint_stalled:
        return SCHUR_LM_STATUS_CONSTRAINT_STALLED
    if stationary:
        return SCHUR_LM_STATUS_CONVERGED
    return SCHUR_LM_STATUS_BUDGET_EXHAUSTED

