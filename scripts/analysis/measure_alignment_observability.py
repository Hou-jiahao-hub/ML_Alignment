"""Roadmap M1: observability of the fixed-track block-LM solve and its reference transfer.

Pre-registered in ``ml_optimization_log.md`` Iteration 40 (Schur plan P0.3).
Measurement only: the production ``block-lm`` solver is called unchanged.

* Spectrum: the unregularised per-sensor Fisher ``C`` (the same blocks
  ``joint_alignment_schur`` reduces) in Jacobi scaling, ``mu_k > tau`` being a
  declared-observable mode.
* Target injection: whole-layer +/- shifts give each sensor's response matrix
  ``K`` (ideal ``I``) and per-mode recovery slopes, on the full sample and on
  four event-disjoint subsamples.
* Fisher floor: the floor moved tenfold either way; the only place truth is read.
* Reference layers: every non-target layer shifted coherently, refit, and the
  target correction it induces, ``-C^-1 dc``; the two largest arms are re-run
  through the full solver.

The ref-only refit reads only non-target surfaces and the seed reads no
geometry, so one cached refit serves every target-injection arm.  Injections
edit ``geometry.centers/rotations`` in memory; ``geometry.transforms`` goes
stale, which is harmless because nothing here exports geometry.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import joint_alignment_schur as J  # noqa: E402
import ml_alignment_autograd as M  # noqa: E402
from experiment_manifest import (  # noqa: E402
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from helix_model import require_stationary_helix_point_fit  # noqa: E402
from measure_prediction_error_budget import identity_gate  # noqa: E402
from measure_track_block_conditioning import select_rows  # noqa: E402
from project_config import add_config_argument, load_project_config  # noqa: E402

PREREGISTRATION = "ml_optimization_log.md Iteration 40"
DTYPE = torch.float64
ACTIVE_DOF = (0, 1, 5)
DOF_NAMES = ("tx", "ty", "rz")
#: Physical units of the solver's dimensionless increment for ``ACTIVE_DOF``.
SCALES = torch.tensor([J.GEOMETRY_SCALES[dof] for dof in ACTIVE_DOF], dtype=DTYPE)

#: Iteration 40 §B: Jacobi-scaled eigenvalue above which a mode is declared observable.
RANK_TOL = J.SCHUR_LM_RANK_TOL
#: Iteration 40 §C: whole-layer target injection amplitudes (tx, ty mm; rz rad).
TARGET_AMPLITUDES = torch.tensor([0.5, 0.5, 0.02], dtype=DTYPE)
N_SUBSAMPLES = 4
SLOPE_BAND = (0.9, 1.1)
#: Iteration 40 §D.
FLOOR_ARMS = (1.0e-3, 1.0e-5)
FLOOR_RATIO_MAX = 1.01
#: Iteration 40 §E: reference-layer amplitudes (tx, ty mm; rz rad).
REFERENCE_AMPLITUDES = torch.tensor([0.010, 0.010, 1.0e-4], dtype=DTYPE)
N_NONLINEAR_ARMS = 2
NONLINEAR_SIGMA_TOL = 0.1
NONLINEAR_FRACTION_MIN = 0.95
#: Iteration 40 §A: control vs the P0.1 baseline output geometry.
EQUIVALENCE_T_MM = 1.0e-6
EQUIVALENCE_RZ_RAD = 1.0e-8
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261011
LM_MAX_ITERS = 60

SCOPE = "diagnostic_only; must not inform solver tuning"


class ExperimentInterrupted(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Spectrum:
    """Per-sensor Jacobi-scaled spectrum of ``C``.

    ``eigenvalues`` [n, k] ascending over the present DOF, NaN-padded at the
    end for structurally absent ones; ``eigenvectors`` [n, k, k] hold them as
    columns in the full DOF space (zero on absent DOF and padded columns).
    """

    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    structurally_absent: torch.Tensor
    observable: torch.Tensor


def jacobi_spectrum(information: torch.Tensor, rank_tol: float = RANK_TOL) -> Spectrum:
    n, k = information.shape[:2]
    absent = information.diagonal(dim1=-2, dim2=-1) <= 0.0
    values = torch.full((n, k), math.nan, dtype=DTYPE)
    vectors = torch.zeros((n, k, k), dtype=DTYPE)
    for sensor in range(n):
        present = torch.nonzero(~absent[sensor], as_tuple=True)[0]
        m = len(present)
        if m == 0:
            continue
        sub = information[sensor][present][:, present]
        scale = sub.diagonal().rsqrt()
        scaled = sub * scale[:, None] * scale[None, :]
        mu, w = torch.linalg.eigh(0.5 * (scaled + scaled.T))
        values[sensor, :m] = mu
        vectors[sensor, present[:, None], torch.arange(m)[None, :]] = w
    observable = torch.nan_to_num(values, nan=-math.inf) > rank_tol
    return Spectrum(values, vectors, absent, observable)


def statistical_sigma(information: torch.Tensor, scales: torch.Tensor = SCALES) -> torch.Tensor:
    """``sqrt(diag(C^-1))`` in physical units; NaN unless every mode is observable."""
    full_rank = jacobi_spectrum(information).observable.all(dim=-1)
    safe = torch.where(
        full_rank[:, None, None], information, torch.eye(information.shape[-1], dtype=DTYPE)
    )
    sigma = torch.linalg.inv(safe).diagonal(dim1=-2, dim2=-1).sqrt() * scales
    return torch.where(full_rank[:, None], sigma, torch.full_like(sigma, math.nan))


def block_lm_weak(information: torch.Tensor, ratio: float = M.FISHER_EIGEN_FLOOR) -> torch.Tensor:
    """block-LM's own weak test: ``eig_min <= eig_max * ratio`` on the unit-scaled Fisher."""
    eigenvalues = torch.linalg.eigvalsh(0.5 * (information + information.transpose(-1, -2)))
    return eigenvalues[:, 0] <= eigenvalues[:, -1] * ratio


def _target_only(batches):
    values = dict(vars(batches))
    values["hit_mask"] = batches.hit_mask & batches.is_target
    return M.TrackBatches(**values)


def target_information(tracks, batches, geometry, alignment, active_dof=ACTIVE_DOF):
    """Fixed-track ``C`` and ``c`` per target sensor, and the sensor surface indices.

    Only target rows enter ``C`` and ``c`` and every hit is linearised on its
    own, so the reference hits are dropped before linearising.
    """
    lin = J.linearize_tracks(tracks, _target_only(batches), geometry, alignment, active_dof)
    information, gradient = J.normal_blocks(lin).geometry_information()
    return information, gradient, lin.target_sensors


def component_counts(batches, sensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Target LOC0 / LOC1 components per sensor, before intersection validity."""
    target = batches.is_target & batches.hit_mask
    n = int(batches.sensor_idx.max()) + 1 if batches.sensor_idx.numel() else 0
    n = max(n, int(sensors.max()) + 1 if len(sensors) else 0)
    loc0 = torch.bincount(batches.sensor_idx[target], minlength=n)
    loc1 = torch.bincount(batches.sensor_idx[target & batches.loc1_mask], minlength=n)
    return loc0[sensors], loc1[sensors]


# --------------------------------------------------------------------------- #
# Injection and response
# --------------------------------------------------------------------------- #


def inject_rows(geometry, rows: torch.Tensor, delta: torch.Tensor):
    """A copy of ``geometry`` with ``rows`` (mask or indices) moved by the local 6-vector ``delta``."""
    if rows.dtype != torch.bool:
        mask = torch.zeros(len(geometry.centers), dtype=torch.bool)
        mask[rows] = True
        rows = mask
    centers = geometry.centers.clone()
    rotations = geometry.rotations.clone()
    count = int(rows.sum())
    moved_c, moved_r = M.apply_alignment_torch(
        centers[rows], rotations[rows], delta.to(DTYPE).expand(count, 6)
    )
    centers[rows] = moved_c
    rotations[rows] = moved_r
    return dataclasses.replace(geometry, centers=centers, rotations=rotations)


def reference_layers(batches) -> dict[tuple[int, int], torch.Tensor]:
    """Non-target ``(volume, layer)`` -> sorted surface indices carrying reference hits."""
    ref = batches.hit_mask & ~batches.is_target
    keys = torch.stack((batches.volume_id[ref], batches.layer_id[ref]), -1)
    sensors = batches.sensor_idx[ref]
    layers = {}
    for key in sorted({tuple(int(v) for v in row) for row in keys.tolist()}):
        match = (keys[:, 0] == key[0]) & (keys[:, 1] == key[1])
        layers[key] = torch.unique(sensors[match])
    return layers


def response_matrix(plus: dict, minus: dict, amplitudes: torch.Tensor) -> torch.Tensor:
    """``K[:, :, d] = -(a(+delta_d) - a(-delta_d)) / (2 delta_d)``, physical units."""
    columns = [
        -(torch.as_tensor(plus[d]) - torch.as_tensor(minus[d])) / (2.0 * amplitudes[d])
        for d in range(len(amplitudes))
    ]
    return torch.stack(columns, dim=-1)


def mode_slopes(response, information, spectrum: Spectrum, scales=SCALES) -> torch.Tensor:
    """``s_k = w_k^T D^1/2 K_u D^-1/2 w_k`` with ``K_u = S^-1 K S`` on the present DOF."""
    n, k = spectrum.eigenvalues.shape
    slopes = torch.full((n, k), math.nan, dtype=DTYPE)
    for sensor in range(n):
        present = torch.nonzero(~spectrum.structurally_absent[sensor], as_tuple=True)[0]
        m = len(present)
        if m == 0:
            continue
        s = scales[present]
        k_u = response[sensor][present][:, present] / s[:, None] * s[None, :]
        root = information[sensor][present, present].sqrt()
        k_z = root[:, None] * k_u / root[None, :]
        w = spectrum.eigenvectors[sensor][present][:, :m]
        slopes[sensor, :m] = torch.einsum("ik,ij,jk->k", w, k_z, w)
    return slopes


def linear_transfer(information, gradient, moved_gradient, scales=SCALES) -> torch.Tensor:
    """First-order correction ``-S C^-1 (c' - c)``; NaN rows where ``C`` is not full rank."""
    full_rank = jacobi_spectrum(information).observable.all(dim=-1)
    safe = torch.where(
        full_rank[:, None, None], information, torch.eye(information.shape[-1], dtype=DTYPE)
    )
    step = -torch.linalg.solve(safe, (moved_gradient - gradient).unsqueeze(-1)).squeeze(-1)
    transfer = step * scales
    return torch.where(full_rank[:, None], transfer, torch.full_like(transfer, math.nan))


def refit_tracks(batches, geometry, tracks, return_stats: bool = False):
    """Production ref-only Gauss-Newton refit, warm-started at ``tracks``."""
    fitted, stats = M.fit_helices_to_measurements(
        batches, geometry, tracks, dtype=DTYPE, stationarity=False
    )
    fitted = fitted.detach().to(DTYPE)
    return (fitted, stats) if return_stats else fitted


def linear_transfer_oracle(tracks, batches, geometry, rows, delta, active_dof, scales=SCALES):
    """Independent first-order oracle for a reference shift (tests only).

    Reference residuals change by a central difference ``dr``; the ref-only
    optimum moves by ``dtheta = -Gamma_ref^-1 B_ref^T dr``; the target gradient
    by ``G dtheta``; the geometry by ``-C^-1 dc``.  No refit is run.
    """
    lin = J.linearize_tracks(tracks, batches, geometry, None, active_dof)
    blocks = J.normal_blocks(lin)
    plus, _valid = J.whitened_residuals(tracks, batches, inject_rows(geometry, rows, delta), None, active_dof)
    minus, _valid = J.whitened_residuals(tracks, batches, inject_rows(geometry, rows, -delta), None, active_dof)
    ref = (lin.component_mask & lin.is_ref.unsqueeze(-1)).to(DTYPE)
    dr = 0.5 * (plus - minus) * ref
    dbeta = torch.einsum("thci,thc->ti", lin.track_jacobian, dr)
    dtheta = -torch.linalg.solve(blocks.gamma_ref, dbeta.unsqueeze(-1)).squeeze(-1)
    information, gradient = blocks.geometry_information()
    dc = torch.zeros_like(gradient).index_add_(
        0, blocks.pair_block, (blocks.G @ dtheta[blocks.pair_track].unsqueeze(-1)).squeeze(-1)
    )
    return linear_transfer(information, gradient, gradient + dc, scales)


# --------------------------------------------------------------------------- #
# Statistics and truth
# --------------------------------------------------------------------------- #


def event_partition(event_ids: torch.Tensor, n: int = N_SUBSAMPLES) -> list[torch.Tensor]:
    """Track rows split by ``event_id % n``: deterministic and event-disjoint."""
    return [torch.nonzero(event_ids % n == part, as_tuple=True)[0] for part in range(n)]


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, n, (n_boot, n), generator=generator)


def _percentiles(samples: torch.Tensor, level: float) -> tuple[float, float]:
    tail = 0.5 * (1.0 - level)
    bounds = torch.quantile(samples, torch.tensor([tail, 1.0 - tail], dtype=DTYPE))
    return float(bounds[0]), float(bounds[1])


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, level=0.95):
    """Mean and percentile bootstrap CI over the entries of ``values``."""
    values = torch.as_tensor(values, dtype=DTYPE).flatten()
    means = values[_bootstrap_indices(len(values), n_boot, seed)].mean(dim=1)
    low, high = _percentiles(means, level)
    return float(values.mean()), low, high


def paired_rms_ratio_ci(errors_a, errors_b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, level=0.95):
    """``RMS(a) / RMS(b)`` with a paired bootstrap over the shared entries."""
    a = torch.as_tensor(errors_a, dtype=DTYPE).flatten()
    b = torch.as_tensor(errors_b, dtype=DTYPE).flatten()
    index = _bootstrap_indices(len(a), n_boot, seed)
    ratios = a[index].square().mean(1).sqrt() / b[index].square().mean(1).sqrt()
    low, high = _percentiles(ratios, level)
    return float(a.square().mean().sqrt() / b.square().mean().sqrt()), low, high


def true_corrections(misaligned_c, misaligned_r, nominal_c, nominal_r) -> torch.Tensor:
    """The 6-DOF local correction taking each misaligned surface back to nominal.

    Same formula as ``alignment_diagnostics.validate_against_truth``:
    ``dt = R_mis^T (c_nom - c_mis)`` and the Euler angles of ``R_mis^T R_nom``.
    """
    dt = (misaligned_r.transpose(-1, -2) @ (nominal_c - misaligned_c).unsqueeze(-1)).squeeze(-1)
    relative = misaligned_r.transpose(-1, -2) @ nominal_r
    ry = -torch.asin(relative[:, 2, 0].clamp(-1.0, 1.0))
    rx = torch.atan2(relative[:, 2, 1], relative[:, 2, 2])
    rz = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
    return torch.cat((dt, torch.stack((rx, ry, rz), -1)), -1)


# --------------------------------------------------------------------------- #
# Real-data driver
# --------------------------------------------------------------------------- #


def _repo_path(*parts) -> str:
    return str(SCRIPT_DIR.parents[1].joinpath(*parts))


DEFAULTS = {
    "input_geom": _repo_path("data/alignment/Vol17_Lay8__Vol24_Lay4/misalignment_applied_v17l8_v24l4.json"),
    "nominal_full_geom": _repo_path("data/geometry/odd_transforms.json"),
    "states": _repo_path(
        "data/reconstruction/misaligned/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root"
    ),
    "summary": _repo_path(
        "data/reconstruction/misaligned/Vol17_Lay8__Vol24_Lay4/tracksummary_kf_v17l8_v24l4.root"
    ),
    "baseline_manifest": _repo_path(
        "data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/schur-p0-fixedtrack-20260922/manifest.json"
    ),
    "baseline_aligned_geom": _repo_path(
        "data/alignment/Vol17_Lay8__Vol24_Lay4/"
        "aligned_transforms_schur-p0-fixedtrack-20260922_joint_free.json"
    ),
    "output_root": _repo_path("data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4"),
}

STAGES = (
    "prepare", "control", "spectrum", "target", "floor",
    "reference-linear", "reference-nonlinear", "report",
)


def _save_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return path


def _manifest_status(directory: Path) -> str | None:
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("status")


class Context:
    """Lazily loaded state shared by the stages of one run."""

    def __init__(self, args):
        self.args = args
        self.root = Path(args.output_root) / args.run_id
        self.config = load_project_config(args.config)
        self.target_layers = M.parse_target_layers(args.target_layers)
        self._geometry = None
        self._cache = None

    @property
    def cache_dir(self) -> Path:
        return self.root / "prepare" / "cache"

    @property
    def geometry(self):
        if self._geometry is None:
            self._geometry = M.load_joint_geometry(
                self.args.nominal_full_geom, self.args.input_geom, self.target_layers
            )
        return self._geometry

    @property
    def target_surfaces(self) -> torch.Tensor:
        return torch.nonzero(self.geometry.target_mask, as_tuple=True)[0]

    def surface_keys(self) -> dict[int, tuple[int, int, int]]:
        return {index: key for key, index in self.geometry.id_to_idx.items()}

    def cache(self) -> dict:
        if self._cache is None:
            batches = torch.load(self.cache_dir / "batches.pt", weights_only=False)
            tracks = torch.load(self.cache_dir / "tracks.pt", weights_only=False)
            self._cache = {"batches": M.TrackBatches(**batches), "tracks": tracks}
        return self._cache

    def solver_config(self, floor: float = M.FISHER_EIGEN_FLOOR):
        return M.AlignmentRunConfig(
            lm_max_iters=LM_MAX_ITERS, joint_solver="block-lm", fisher_eigen_floor=floor
        )

    def control_alignment(self) -> torch.Tensor:
        arm = np.load(self.root / "control" / "arms" / "full_control.npz")
        alignment = torch.zeros((len(self.geometry.centers), 6), dtype=DTYPE)
        alignment[torch.as_tensor(arm["surfaces"])] = torch.as_tensor(arm["deltas"])
        return alignment


def run_block_lm(ctx: Context, batches, tracks, geometry, path: Path, floor=M.FISHER_EIGEN_FLOOR):
    """One block-LM arm, saved as ``path`` (.npz) with a JSON summary; skipped if present."""
    if path.is_file():
        return dict(np.load(path, allow_pickle=False))
    started = time.perf_counter()
    result = M.run_block_lm_alignment(
        batches, geometry, ctx.solver_config(floor), ACTIVE_DOF, initial_track_params=tracks
    )
    elapsed = time.perf_counter() - started
    surfaces = ctx.target_surfaces
    deltas = torch.as_tensor(result["predicted_deltas"])[surfaces]
    status = {int(block["sensor_index"]): block["status"] for block in result["block_lm_blocks"]}
    statuses = np.array([status.get(int(s), "absent") for s in surfaces.tolist()])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, surfaces=surfaces.numpy(), deltas=deltas.numpy(), status=statuses)
    _save_json(path.with_suffix(".json"), {
        "seconds": elapsed,
        "fisher_eigen_floor": floor,
        "n_tracks": int(tracks.shape[0]),
        "stop_reason": result["stop_reason"],
        "best_iteration": result["best_iteration"],
        "block_lm_summary": result["block_lm_summary"],
    })
    print(f"  [arm] {path.name}: {elapsed:.0f} s, stop={result['stop_reason']}", flush=True)
    return dict(np.load(path, allow_pickle=False))


def _subsamples(ctx: Context):
    batches, tracks = ctx.cache()["batches"], ctx.cache()["tracks"]
    yield "full", batches, tracks, None
    if ctx.args.subsamples == 0:
        return
    for part, index in enumerate(event_partition(batches.event_track_ids[:, 0], ctx.args.subsamples)):
        yield f"sub{part}", select_rows(batches, index), tracks[index], index


def stage_prepare(ctx: Context, metrics: dict) -> dict:
    args = ctx.args
    timings = {}
    started = time.perf_counter()
    inputs = M.load_variable_information(
        args.summary,
        args.states,
        ctx.geometry.id_to_idx,
        max_track_entries=(args.max_tracks * 4 if args.max_tracks > 0 else None),
        load_summary=False,
    )
    batches = M.build_track_batches(
        inputs, ctx.target_layers, min_hits=args.min_hits, min_span=args.min_span,
        max_tracks=args.max_tracks,
    )
    del inputs
    timings["load_and_batch_s"] = time.perf_counter() - started
    started = time.perf_counter()
    seeded, seed_stats = M.seed_helices(
        batches, n_iter=args.smt_seed_iters, dtype=DTYPE, return_diagnostics=True
    )
    require_stationary_helix_point_fit(seed_stats)
    timings["seed_s"] = time.perf_counter() - started
    started = time.perf_counter()
    refitted, refit_stats = refit_tracks(batches, ctx.geometry, seeded, return_stats=True)
    timings["refit_s"] = time.perf_counter() - started
    baseline = json.loads(Path(args.baseline_manifest).read_text(encoding="utf-8"))
    own = json.loads((ctx.root / "prepare" / "manifest.json").read_text(encoding="utf-8"))
    gate = identity_gate(
        batches.selection_stats["retained_tracks"],
        refit_stats["n_components"],
        full_row_set=args.max_tracks == 0,
        manifest_inputs=own["inputs"],
        baseline_inputs=baseline["inputs"],
    )
    ctx.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(vars(batches)), ctx.cache_dir / "batches.pt")
    torch.save(refitted, ctx.cache_dir / "tracks.pt")
    metrics.update({
        "identity_gate": gate,
        "selection": batches.selection_stats,
        "refit": {key: refit_stats[key] for key in ("chi2_per_component", "n_components", "invalid_fraction")},
        "seed_non_stationary_row_fraction": float(seed_stats["non_stationary_row_fraction"]),
        "timings": timings,
    })
    print(f"  identity gate: {gate}; timings {timings}", flush=True)
    return {"batches": ctx.cache_dir / "batches.pt", "tracks": ctx.cache_dir / "tracks.pt"}


def _equivalence_to_baseline(ctx: Context, arm) -> dict:
    baseline = M.load_joint_geometry(
        ctx.args.nominal_full_geom, ctx.args.baseline_aligned_geom, ctx.target_layers
    )
    surfaces = torch.as_tensor(arm["surfaces"])
    deltas = torch.as_tensor(arm["deltas"])
    centers, rotations = M.apply_alignment_torch(
        ctx.geometry.centers[surfaces], ctx.geometry.rotations[surfaces], deltas
    )
    ref_c, ref_r = baseline.centers[surfaces], baseline.rotations[surfaces]
    local = (ref_r.transpose(-1, -2) @ (centers - ref_c).unsqueeze(-1)).squeeze(-1)
    relative = ref_r.transpose(-1, -2) @ rotations
    rz = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
    return {"max_translation_mm": float(local.abs().max()), "max_rz_rad": float(rz.abs().max())}


def stage_control(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    path = ctx.root / "control" / "arms" / "full_control.npz"
    arm = run_block_lm(ctx, cache["batches"], cache["tracks"], ctx.geometry, path)
    statuses = arm["status"]
    metrics["status_counts"] = {str(k): int(v) for k, v in zip(*np.unique(statuses, return_counts=True))}
    equivalence = _equivalence_to_baseline(ctx, arm)
    metrics["equivalence_to_p0"] = equivalence
    if ctx.args.max_tracks == 0:
        passed = (
            equivalence["max_translation_mm"] <= EQUIVALENCE_T_MM
            and equivalence["max_rz_rad"] <= EQUIVALENCE_RZ_RAD
        )
        metrics["equivalence_gate"] = "passed" if passed else "failed"
        if not passed:
            raise ValueError(f"Control differs from the P0.1 baseline: {equivalence}")
    else:
        metrics["equivalence_gate"] = "not_applicable_subset"
    return {"control": path}


def _spectrum_record(ctx, batches, tracks, alignment):
    information, gradient, sensors = target_information(tracks, batches, ctx.geometry, alignment)
    return information, gradient, sensors


def _to_target_order(ctx: Context, sensors: torch.Tensor, values: torch.Tensor, fill=math.nan):
    """Scatter per-sensor rows onto the order of ``ctx.target_surfaces``."""
    lookup = {int(s): i for i, s in enumerate(ctx.target_surfaces.tolist())}
    shape = (len(ctx.target_surfaces), *values.shape[1:])
    out = torch.full(shape, fill, dtype=values.dtype) if values.is_floating_point() else torch.zeros(
        shape, dtype=values.dtype
    )
    out[torch.tensor([lookup[int(s)] for s in sensors.tolist()], dtype=torch.long)] = values
    return out


def stage_spectrum(ctx: Context, metrics: dict) -> dict:
    alignment = ctx.control_alignment()
    outputs = {}
    for name, batches, tracks, _index in _subsamples(ctx):
        points = {"star": alignment} if name != "full" else {"zero": None, "star": alignment}
        record = {}
        for point, value in points.items():
            information, gradient, sensors = _spectrum_record(ctx, batches, tracks, value)
            spectrum = jacobi_spectrum(information)
            record[f"{point}_information"] = _to_target_order(ctx, sensors, information)
            record[f"{point}_gradient"] = _to_target_order(ctx, sensors, gradient)
            record[f"{point}_eigenvalues"] = _to_target_order(ctx, sensors, spectrum.eigenvalues)
            record[f"{point}_eigenvectors"] = _to_target_order(ctx, sensors, spectrum.eigenvectors)
            record[f"{point}_absent"] = _to_target_order(ctx, sensors, spectrum.structurally_absent)
            record[f"{point}_observable"] = _to_target_order(ctx, sensors, spectrum.observable)
            record[f"{point}_has_data"] = _to_target_order(
                ctx, sensors, torch.ones(len(sensors), dtype=torch.bool)
            )
            if point == "star":
                record["sigma"] = _to_target_order(ctx, sensors, statistical_sigma(information))
                record["block_lm_weak"] = _to_target_order(ctx, sensors, block_lm_weak(information))
                loc0, loc1 = component_counts(batches, sensors)
                record["n_loc0"] = _to_target_order(ctx, sensors, loc0)
                record["n_loc1"] = _to_target_order(ctx, sensors, loc1)
        path = ctx.root / "spectrum" / f"spectrum_{name}.npz"
        np.savez(path, surfaces=ctx.target_surfaces.numpy(), **{k: v.numpy() for k, v in record.items()})
        outputs[f"spectrum_{name}"] = path
        if name == "full":
            has = record["star_has_data"]
            loc0_only = has & (record["n_loc1"] == 0) & (record["n_loc0"] > 0)
            metrics["full"] = {
                "n_sensors_with_data": int(has.sum()),
                "n_loc0_only": int(loc0_only.sum()),
                "n_structurally_absent_dof": int(record["star_absent"][has].sum()),
                "n_weak_modes": int((~record["star_observable"][has] & ~torch.isnan(
                    record["star_eigenvalues"][has])).sum()),
                "n_block_lm_weak": int(record["block_lm_weak"][has].sum()),
            }
            _write_spectrum_csv(ctx, record, ctx.root / "spectrum" / "per_sensor.csv")
            outputs["per_sensor"] = ctx.root / "spectrum" / "per_sensor.csv"
    return outputs


def _write_spectrum_csv(ctx: Context, record, path: Path):
    import pandas as pd

    keys = ctx.surface_keys()
    rows = []
    for i, surface in enumerate(ctx.target_surfaces.tolist()):
        volume, layer, sensor = keys.get(surface, (-1, -1, -1))
        row = {"surface": surface, "volume": volume, "layer": layer, "sensor": sensor,
               "has_data": bool(record["star_has_data"][i]),
               "n_loc0": int(record["n_loc0"][i]), "n_loc1": int(record["n_loc1"][i]),
               "block_lm_weak": bool(record["block_lm_weak"][i])}
        for k in range(3):
            row[f"mu{k}_star"] = float(record["star_eigenvalues"][i, k])
            row[f"mu{k}_zero"] = float(record["zero_eigenvalues"][i, k])
            row[f"observable{k}"] = bool(record["star_observable"][i, k])
        for d, name in enumerate(DOF_NAMES):
            row[f"absent_{name}"] = bool(record["star_absent"][i, d])
            row[f"sigma_{name}"] = float(record["sigma"][i, d])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _target_arm_path(ctx, sample, dof, sign):
    return ctx.root / "target" / "arms" / f"{sample}_{DOF_NAMES[dof]}_{'plus' if sign > 0 else 'minus'}.npz"


def stage_target(ctx: Context, metrics: dict) -> dict:
    timings = {}
    for name, batches, tracks, _index in _subsamples(ctx):
        for column, dof in enumerate(ACTIVE_DOF):
            for sign in (1.0, -1.0):
                delta = torch.zeros(6, dtype=DTYPE)
                delta[dof] = sign * TARGET_AMPLITUDES[column]
                geometry = inject_rows(ctx.geometry, ctx.geometry.target_mask, delta)
                path = _target_arm_path(ctx, name, column, sign)
                run_block_lm(ctx, batches, tracks, geometry, path)
                timings[path.name] = json.loads(path.with_suffix(".json").read_text())["seconds"]
    metrics["seconds"] = timings
    return {"arms": ctx.root / "target" / "arms"}


def stage_floor(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    for floor in FLOOR_ARMS:
        path = ctx.root / "floor" / "arms" / f"full_floor_{floor:.0e}.npz"
        run_block_lm(ctx, cache["batches"], cache["tracks"], ctx.geometry, path, floor=floor)
    return {"arms": ctx.root / "floor" / "arms"}


def _shift_reference(ctx, rows, column, sign):
    delta = torch.zeros(6, dtype=DTYPE)
    delta[ACTIVE_DOF[column]] = sign * REFERENCE_AMPLITUDES[column]
    return inject_rows(ctx.geometry, rows, delta)


def _refit_affected(ctx, batches, tracks, geometry, rows):
    """Refit only tracks with a reference hit on ``rows``; the others' objective is unchanged."""
    surfaces = torch.zeros(len(geometry.centers), dtype=torch.bool)
    surfaces[rows] = True
    fit = batches.fit_mask & batches.hit_mask
    affected = torch.nonzero((surfaces[batches.sensor_idx] & fit).any(dim=1), as_tuple=True)[0]
    moved = tracks.clone()
    fitted, stats = refit_tracks(select_rows(batches, affected), geometry, tracks[affected], True)
    moved[affected] = fitted
    return moved, stats, len(affected)


def stage_reference_linear(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    batches, tracks = cache["batches"], cache["tracks"]
    alignment = ctx.control_alignment()
    information, gradient, sensors = target_information(tracks, batches, ctx.geometry, alignment)
    layers = reference_layers(batches)
    metrics["layers"] = [list(key) for key in layers]
    arms_dir = ctx.root / "reference-linear" / "arms"
    for key, rows in layers.items():
        for column in range(len(ACTIVE_DOF)):
            for sign in (1.0, -1.0):
                path = arms_dir / f"{key[0]}_{key[1]}_{DOF_NAMES[column]}_{'plus' if sign > 0 else 'minus'}.npz"
                if path.is_file():
                    continue
                started = time.perf_counter()
                geometry = _shift_reference(ctx, rows, column, sign)
                moved, stats, n_affected = _refit_affected(ctx, batches, tracks, geometry, rows)
                _info, moved_gradient, moved_sensors = target_information(
                    moved, batches, ctx.geometry, alignment
                )
                if not torch.equal(moved_sensors, sensors):
                    raise ValueError("Target sensor set changed under a reference shift.")
                transfer = linear_transfer(information, gradient, moved_gradient)
                arms_dir.mkdir(parents=True, exist_ok=True)
                np.savez(path, surfaces=ctx.target_surfaces.numpy(),
                         transfer=_to_target_order(ctx, sensors, transfer).numpy())
                _save_json(path.with_suffix(".json"), {
                    "seconds": time.perf_counter() - started,
                    "n_affected_tracks": n_affected,
                    "refit": {k: stats[k] for k in ("chi2_per_component", "n_components", "invalid_fraction")},
                })
                print(f"  [ref] {path.name}: {n_affected} tracks", flush=True)
    return {"arms": arms_dir}


def _reference_bias(ctx, key, column):
    arms = ctx.root / "reference-linear" / "arms"
    plus = np.load(arms / f"{key[0]}_{key[1]}_{DOF_NAMES[column]}_plus.npz")["transfer"]
    minus = np.load(arms / f"{key[0]}_{key[1]}_{DOF_NAMES[column]}_minus.npz")["transfer"]
    return 0.5 * (plus - minus)


def _target_layer_masks(ctx: Context) -> dict[tuple[int, int], np.ndarray]:
    keys = ctx.surface_keys()
    layers = np.array([keys[s][:2] for s in ctx.target_surfaces.tolist()])
    return {
        tuple(layer): (layers[:, 0] == layer[0]) & (layers[:, 1] == layer[1])
        for layer in ctx.target_layers
    }


def reference_table(ctx: Context) -> list[dict]:
    sigma = np.load(ctx.root / "spectrum" / "spectrum_full.npz")["sigma"]
    layers = json.loads((ctx.root / "reference-linear" / "manifest.json").read_text())["metrics"]["layers"]
    table = []
    for key in map(tuple, layers):
        for column in range(len(ACTIVE_DOF)):
            bias = _reference_bias(ctx, key, column)
            for target, mask in _target_layer_masks(ctx).items():
                for tdof in range(len(ACTIVE_DOF)):
                    values, sig = bias[mask, tdof], sigma[mask, tdof]
                    finite = np.isfinite(values) & np.isfinite(sig)
                    rms = float(np.sqrt(np.mean(values[finite] ** 2)))
                    median = float(np.median(sig[finite]))
                    table.append({
                        "reference_layer": list(key), "reference_dof": DOF_NAMES[column],
                        "target_layer": list(target), "target_dof": DOF_NAMES[tdof],
                        "n_sensors": int(finite.sum()), "rms_bias": rms, "median_sigma": median,
                        "ratio": rms / median, "reference_sensitive": rms > median,
                    })
    return table


def stage_reference_nonlinear(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    batches, tracks = cache["batches"], cache["tracks"]
    table = reference_table(ctx)
    scores = {}
    for row in table:
        arm = (tuple(row["reference_layer"]), row["reference_dof"])
        scores[arm] = max(scores.get(arm, -math.inf), row["ratio"])
    chosen = sorted(scores, key=lambda arm: (-scores[arm], arm))[:N_NONLINEAR_ARMS]
    metrics["chosen_arms"] = [[list(key), dof, scores[(key, dof)]] for key, dof in chosen]
    layers = reference_layers(batches)
    for key, dof in chosen:
        column = DOF_NAMES.index(dof)
        for sign in (1.0, -1.0):
            path = ctx.root / "reference-nonlinear" / "arms" / (
                f"{key[0]}_{key[1]}_{dof}_{'plus' if sign > 0 else 'minus'}.npz"
            )
            if path.is_file():
                continue
            geometry = _shift_reference(ctx, layers[key], column, sign)
            moved, _stats, _n = _refit_affected(ctx, batches, tracks, geometry, layers[key])
            run_block_lm(ctx, batches, moved, geometry, path)
    return {"arms": ctx.root / "reference-nonlinear" / "arms"}


def _converged(*arms) -> np.ndarray:
    return np.logical_and.reduce([arm["status"] == M.BLOCK_LM_STATUS_CONVERGED for arm in arms])


def _cell_ci(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0, "mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "inside": False}
    mean, low, high = bootstrap_mean_ci(torch.as_tensor(values))
    return {"n": int(len(values)), "mean": mean, "ci_low": low, "ci_high": high,
            "inside": SLOPE_BAND[0] <= low and high <= SLOPE_BAND[1],
            "fraction_outside_band": float(np.mean((values < SLOPE_BAND[0]) | (values > SLOPE_BAND[1])))}


def _primary(ctx: Context) -> dict:
    samples = ["full"] + [f"sub{part}" for part in range(ctx.args.subsamples)]
    masks = _target_layer_masks(ctx)
    cells: dict[str, dict] = {}
    weak_slopes = {}
    for sample in samples:
        spectrum_file = np.load(ctx.root / "spectrum" / f"spectrum_{sample}.npz")
        information = torch.as_tensor(spectrum_file["star_information"])
        has = torch.as_tensor(spectrum_file["star_has_data"])
        spectrum = Spectrum(
            torch.as_tensor(spectrum_file["star_eigenvalues"]),
            torch.as_tensor(spectrum_file["star_eigenvectors"]),
            torch.as_tensor(spectrum_file["star_absent"]),
            torch.as_tensor(spectrum_file["star_observable"]),
        )
        plus = {c: np.load(_target_arm_path(ctx, sample, c, 1.0)) for c in range(3)}
        minus = {c: np.load(_target_arm_path(ctx, sample, c, -1.0)) for c in range(3)}
        converged = _converged(*plus.values(), *minus.values())
        response = response_matrix(
            {c: plus[c]["deltas"][:, list(ACTIVE_DOF)] for c in range(3)},
            {c: minus[c]["deltas"][:, list(ACTIVE_DOF)] for c in range(3)},
            TARGET_AMPLITUDES,
        )
        safe_information = torch.where(has[:, None, None], information, torch.eye(3, dtype=DTYPE))
        slopes = mode_slopes(response, safe_information, spectrum).numpy()
        observable = spectrum.observable.numpy() & has.numpy()[:, None]
        for layer, mask in masks.items():
            for k in range(3):
                keep = mask & converged & observable[:, k] & np.isfinite(slopes[:, k])
                name = f"{layer[0]}:{layer[1]}|mode{k}"
                cells.setdefault(name, {})[sample] = _cell_ci(slopes[keep, k])
                weak = mask & converged & ~observable[:, k] & has.numpy() & np.isfinite(slopes[:, k])
                weak_slopes.setdefault(name, {})[sample] = {
                    "n": int(weak.sum()),
                    "median": float(np.median(slopes[weak, k])) if weak.any() else math.nan,
                }
    verdict = {
        name: ("observable_confirmed" if all(entry["inside"] for entry in per.values())
               else "unobservable_or_prior_dominated")
        for name, per in cells.items()
    }
    return {"cells": cells, "verdict": verdict, "weak_mode_slopes": weak_slopes,
            "passed": all(value == "observable_confirmed" for value in verdict.values())}


def _floor_secondary(ctx: Context) -> dict:
    nominal = json.loads(Path(ctx.args.nominal_full_geom).read_text(encoding="utf-8"))
    surfaces = ctx.target_surfaces
    nominal_c = torch.tensor([nominal[s]["Translation"] for s in surfaces.tolist()], dtype=DTYPE)
    nominal_r = torch.tensor([nominal[s]["RotationMatrix"] for s in surfaces.tolist()], dtype=DTYPE)
    truth = true_corrections(
        ctx.geometry.centers[surfaces], ctx.geometry.rotations[surfaces], nominal_c, nominal_r
    )[:, list(ACTIVE_DOF)].numpy()
    control = np.load(ctx.root / "control" / "arms" / "full_control.npz")
    sigma = np.load(ctx.root / "spectrum" / "spectrum_full.npz")["sigma"]
    result = {}
    for floor in FLOOR_ARMS:
        arm = np.load(ctx.root / "floor" / "arms" / f"full_floor_{floor:.0e}.npz")
        data = (arm["status"] != M.BLOCK_LM_STATUS_NO_DATA) & (control["status"] != M.BLOCK_LM_STATUS_NO_DATA)
        data &= (arm["status"] != "absent") & (control["status"] != "absent")
        a = arm["deltas"][:, list(ACTIVE_DOF)]
        b = control["deltas"][:, list(ACTIVE_DOF)]
        cells = {}
        for layer, mask in _target_layer_masks(ctx).items():
            for d, name in enumerate(DOF_NAMES):
                keep = mask & data
                ratio, low, high = paired_rms_ratio_ci(a[keep, d] - truth[keep, d], b[keep, d] - truth[keep, d])
                shift = np.abs(a[keep, d] - b[keep, d]) / sigma[keep, d]
                shift = shift[np.isfinite(shift)]
                cells[f"{layer[0]}:{layer[1]}|{name}"] = {
                    "n": int(keep.sum()), "ratio": ratio, "ci_low": low, "ci_high": high,
                    "passed": high <= FLOOR_RATIO_MAX,
                    "shift_over_sigma_p50": float(np.median(shift)) if len(shift) else math.nan,
                    "shift_over_sigma_max": float(shift.max()) if len(shift) else math.nan,
                }
        result[f"{floor:.0e}"] = cells
    return {"scope": SCOPE, "cells": result,
            "passed": all(cell["passed"] for cells in result.values() for cell in cells.values())}


def _reference_secondary(ctx: Context) -> dict:
    table = reference_table(ctx)
    sigma = np.load(ctx.root / "spectrum" / "spectrum_full.npz")["sigma"]
    chosen = json.loads((ctx.root / "reference-nonlinear" / "manifest.json").read_text())["metrics"]["chosen_arms"]
    checks = []
    for key, dof, _score in chosen:
        arms = ctx.root / "reference-nonlinear" / "arms"
        plus = np.load(arms / f"{key[0]}_{key[1]}_{dof}_plus.npz")
        minus = np.load(arms / f"{key[0]}_{key[1]}_{dof}_minus.npz")
        nonlinear = 0.5 * (plus["deltas"] - minus["deltas"])[:, list(ACTIVE_DOF)]
        linear = _reference_bias(ctx, tuple(key), DOF_NAMES.index(dof))
        keep = _converged(plus, minus) & np.isfinite(linear).all(1) & np.isfinite(sigma).all(1)
        within = (np.abs(nonlinear - linear)[keep] <= NONLINEAR_SIGMA_TOL * sigma[keep]).all(1)
        fraction = float(within.mean()) if len(within) else math.nan
        checks.append({"reference_layer": key, "reference_dof": dof, "n_sensors": int(keep.sum()),
                       "fraction_within": fraction,
                       "linearity": "verified" if fraction >= NONLINEAR_FRACTION_MIN else "linearity_unverified"})
    sensitive = [row for row in table if row["reference_sensitive"]]
    return {"table": table, "nonlinear_checks": checks, "n_reference_sensitive": len(sensitive),
            "reference_sensitive": sensitive}


def stage_report(ctx: Context, metrics: dict) -> dict:
    prepare = json.loads((ctx.root / "prepare" / "manifest.json").read_text())["metrics"]
    control = json.loads((ctx.root / "control" / "manifest.json").read_text())["metrics"]
    spectrum = json.loads((ctx.root / "spectrum" / "manifest.json").read_text())["metrics"]
    control_arm = np.load(ctx.root / "control" / "arms" / "full_control.npz")
    keys = ctx.surface_keys()
    not_converged = [
        {"surface": keys.get(int(s), int(s)), "status": str(status)}
        for s, status in zip(control_arm["surfaces"], control_arm["status"])
        if status != M.BLOCK_LM_STATUS_CONVERGED
    ]
    result = {
        "preregistration": PREREGISTRATION,
        "identity_gate": prepare["identity_gate"],
        "equivalence_gate": control["equivalence_gate"],
        "equivalence_to_p0": control["equivalence_to_p0"],
        "control_not_converged": not_converged,
        "spectrum": spectrum["full"],
        "primary_target_injection": _primary(ctx),
        "secondary_floor": _floor_secondary(ctx) if ctx.args.read_truth else "skipped (--no-truth)",
        "secondary_reference": _reference_secondary(ctx),
    }
    path = _save_json(ctx.root / "report" / "m1_result.json", result)
    metrics.update({
        "primary_passed": result["primary_target_injection"]["passed"],
        "floor_passed": (result["secondary_floor"]["passed"]
                         if isinstance(result["secondary_floor"], dict) else None),
        "n_reference_sensitive": result["secondary_reference"]["n_reference_sensitive"],
    })
    return {"result": path}


STAGE_FUNCTIONS = {
    "prepare": stage_prepare,
    "control": stage_control,
    "spectrum": stage_spectrum,
    "target": stage_target,
    "floor": stage_floor,
    "reference-linear": stage_reference_linear,
    "reference-nonlinear": stage_reference_nonlinear,
    "report": stage_report,
}


def _stage_inputs(ctx: Context, stage: str) -> dict:
    args = ctx.args
    inputs = {"geometry": args.input_geom, "nominal_geometry": args.nominal_full_geom}
    if stage == "prepare":
        inputs.update(states=args.states, summary=args.summary, baseline_manifest=args.baseline_manifest)
    else:
        inputs.update(cache_batches=ctx.cache_dir / "batches.pt", cache_tracks=ctx.cache_dir / "tracks.pt")
    if stage == "control":
        inputs["baseline_aligned_geometry"] = args.baseline_aligned_geom
    return inputs


def run_stage(ctx: Context, stage: str, command: list[str]) -> dict:
    directory = ctx.root / stage
    if _manifest_status(directory) == "completed":
        print(f"[skip] {stage}: already completed", flush=True)
        return json.loads((directory / "manifest.json").read_text())["metrics"]
    metrics = {"stage": stage, "preregistration": PREREGISTRATION, "scope": SCOPE}
    manifest = write_experiment_manifest(
        directory,
        run_id=f"{ctx.args.run_id}/{stage}",
        command=command,
        config=ctx.config,
        inputs=_stage_inputs(ctx, stage),
        metrics=metrics,
        capture_source_snapshot=ctx.args.capture_source_snapshot,
    )
    started = time.perf_counter()
    print(f"[run ] {stage}", flush=True)
    try:
        outputs = STAGE_FUNCTIONS[stage](ctx, metrics) or {}
        metrics["duration_s"] = time.perf_counter() - started
        finalize_experiment_manifest(manifest, status="completed", metrics=metrics, outputs=outputs)
        return metrics
    except BaseException as error:
        metrics.update(duration_s=time.perf_counter() - started,
                       error_type=type(error).__name__, error_message=str(error))
        finalize_experiment_manifest(manifest, status="failed", metrics=metrics)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_argument(parser)
    parser.add_argument("stage", choices=(*STAGES, "run-all"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=DEFAULTS["output_root"])
    for name in ("input_geom", "nominal_full_geom", "states", "summary", "baseline_manifest",
                 "baseline_aligned_geom"):
        parser.add_argument(f"--{name.replace('_', '-')}", default=DEFAULTS[name])
    parser.add_argument("--target-layers", default="17:8,24:4")
    parser.add_argument("--min-hits", type=int, default=M.JOINT_SEED_MIN_HITS)
    parser.add_argument("--min-span", type=float, default=M.JOINT_SEED_MIN_SPAN)
    parser.add_argument("--smt-seed-iters", type=int, default=2000)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--subsamples", type=int, default=N_SUBSAMPLES,
                        help="Event-disjoint replicas; only a --max-tracks smoke may change it.")
    parser.add_argument("--no-truth", dest="read_truth", action="store_false",
                        help="Skip the Fisher-floor truth comparison in the report (smoke runs).")
    parser.add_argument("--capture-source-snapshot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tracks == 0 and args.subsamples != N_SUBSAMPLES:
        raise SystemExit(f"--subsamples is fixed at {N_SUBSAMPLES} on the full row set (Iteration 40).")
    ctx = Context(args)
    command = [sys.executable, str(Path(__file__).resolve()),
               *(sys.argv[1:] if argv is None else argv)]

    def interrupt(signum, _frame):
        raise ExperimentInterrupted(f"received signal {signum}")

    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for signum in old_handlers:
        signal.signal(signum, interrupt)
    try:
        stages = STAGES if args.stage == "run-all" else (args.stage,)
        for stage in stages:
            run_stage(ctx, stage, command)
        return 0
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
