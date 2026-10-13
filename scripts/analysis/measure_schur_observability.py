"""Schur P9 on real data: ``schur-lm`` Level C/D and the reduced-system observability.

Pre-registered in ``ml_optimization_log.md`` Iteration 42 (criteria for H1 in
Iteration 41).  No truth is read here; ``schur-lm`` is called directly, never
through the pipeline's ``validate_against_truth``.

* Level C: one ``schur-lm`` run on a deterministic event subset (engineering gate).
* Level D: the full-sample control from ``a = 0``, the ``lambda_init`` scan, the
  production marginal sigma ``sqrt(diag(S^-1))`` and, if both gates pass, the
  exported Schur candidate.
* Spectrum: ``S_ii`` against the fixed-track ``C_ii`` of M1 at the same points,
  retention ``tr(S_ii)/tr(C_ii)``, marginal sigmas, the global Jacobi spectrum
  and the normalised coupling ``rho^S_ij``, with a hard information-loss check.
* Target injection: M1's design with ``schur-lm`` and ``S_ii`` modes.
* Reference transfer: the first-order geometry response to a reference-layer
  shift, ``-S^-1 db`` with ``db = -G Gamma^-1 dbeta``, next to the fixed-track
  ``-C^-1 G dtheta_ref`` from the same ``dbeta``.

The stage driver, cache and statistics are M1's (``measure_alignment_observability``).
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import joint_alignment_schur as J  # noqa: E402
import measure_alignment_observability as O  # noqa: E402
import ml_alignment_autograd as M  # noqa: E402
from experiment_manifest import (  # noqa: E402
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from measure_track_block_conditioning import select_rows  # noqa: E402
from project_config import add_config_argument  # noqa: E402

PREREGISTRATION = "ml_optimization_log.md Iteration 42"
DTYPE = torch.float64
ACTIVE_DOF = O.ACTIVE_DOF
DOF_NAMES = O.DOF_NAMES
SCALES = O.SCALES
RANK_TOL = J.SCHUR_LM_RANK_TOL

#: Iteration 42 §B: Level C subset ``event_id % LEVEL_C_MODULUS == 0``.
LEVEL_C_MODULUS = 32
#: Iteration 42 §C / Iteration 39: lambda_init arms and the 0.2 sigma tolerance.
LAMBDA_SCAN = (1.0e-5, 1.0e-1)
LAMBDA_TOLERANCE_SIGMA = 2.0 * math.sqrt(J.SCHUR_LM_EPSILON_GEO)
#: Iteration 42 §D: relative slack of the information-loss check.
LOEWNER_RTOL = 1.0e-9
N_LOWEST_MODES = 10
N_TOP_PAIRS = 20
#: Iteration 42 §F / Iteration 41 §B.
ORACLE_SIGMA_TOL = O.NONLINEAR_SIGMA_TOL
ORACLE_FRACTION_MIN = O.NONLINEAR_FRACTION_MIN
TRANSFER_DECREASE_RATIO = 0.90
TRANSFER_INCREASE_RATIO = 1.10

M1_RUN_ID = "m1-fixed-track-observability-20261011"
SCOPE = "diagnostic_only; no truth read; must not inform solver tuning"


# --------------------------------------------------------------------------- #
# Reduced-system quantities
# --------------------------------------------------------------------------- #


def dense_schur(system: J.SchurSystem) -> torch.Tensor:
    """``S`` as a dense matrix via the production CSC assembly."""
    return torch.from_numpy(J.to_sparse(system).toarray())


def diagonal_blocks(dense: torch.Tensor, k: int) -> torch.Tensor:
    """The per-sensor ``k x k`` diagonal blocks ``S_ii`` [n, k, k]."""
    n = dense.shape[0] // k
    blocks = dense.reshape(n, k, n, k)
    index = torch.arange(n)
    return blocks[index, :, index, :]


def information_retention(s_ii: torch.Tensor, c_ii: torch.Tensor) -> torch.Tensor:
    """``R_i = tr(S_ii) / tr(C_ii)`` in u units; a diagnostic, not a probability."""
    return s_ii.diagonal(dim1=-2, dim2=-1).sum(-1) / c_ii.diagonal(dim1=-2, dim2=-1).sum(-1)


def marginal_sigma(dense: torch.Tensor, scales: torch.Tensor = SCALES) -> torch.Tensor:
    """``sqrt(diag(S^-1))`` in physical units [n, k]; the cross-sensor coupling is included."""
    k = len(scales)
    variance = torch.linalg.inv(dense).diagonal()
    return variance.sqrt().reshape(-1, k) * scales


def loewner_check(c_ii, s_ii, dense, scales=SCALES, rtol: float = LOEWNER_RTOL) -> dict:
    """Iteration 42 §D: eliminating tracks can only remove information.

    ``C_ii - S_ii`` must be PSD per sensor, and the marginal sigmas must be
    ordered ``sigma_S >= sigma_S,local >= sigma_C`` per DOF.
    """
    norm = torch.linalg.matrix_norm(c_ii)
    gap = torch.linalg.eigvalsh(0.5 * ((c_ii - s_ii) + (c_ii - s_ii).transpose(-1, -2)))[:, 0]
    relative_gap = gap / norm
    sigma_c = O.statistical_sigma(c_ii, scales)
    sigma_local = O.statistical_sigma(s_ii, scales)
    sigma_global = marginal_sigma(dense, scales)
    finite = torch.isfinite(sigma_c) & torch.isfinite(sigma_local) & torch.isfinite(sigma_global)
    order = (sigma_global >= sigma_local * (1.0 - rtol)) & (sigma_local >= sigma_c * (1.0 - rtol))
    violations = int((finite & ~order).sum())
    return {
        "passed": bool((relative_gap >= -rtol).all()) and violations == 0,
        "min_retention_gap": float(relative_gap.min()),
        "n_sigma_order_violations": violations,
        "n_sigma_compared": int(finite.sum()),
    }


def coupling_strength(system: J.SchurSystem):
    """``rho^S_ij = ||S_ij||_F / sqrt(tr S_ii tr S_jj)`` for every off-diagonal block."""
    diagonal = system.rows == system.cols
    trace = torch.zeros(system.n_blocks, dtype=DTYPE)
    trace[system.rows[diagonal]] = system.values[diagonal].diagonal(dim1=-2, dim2=-1).sum(-1)
    off = ~diagonal
    rows, cols = system.rows[off], system.cols[off]
    rho = torch.linalg.matrix_norm(system.values[off]) / torch.sqrt(trace[rows] * trace[cols])
    return rows, cols, rho


def global_jacobi_spectrum(dense: torch.Tensor, k: int, n_lowest: int = N_LOWEST_MODES,
                           rank_tol: float = RANK_TOL) -> dict:
    """Eigenvalues of ``D^-1/2 S D^-1/2`` and the per-sensor weight of the lowest modes."""
    scale = dense.diagonal().rsqrt()
    scaled = dense * scale[:, None] * scale[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (scaled + scaled.T))
    lowest = eigenvectors[:, :n_lowest].T
    participation = lowest.square().reshape(len(lowest), -1, k).sum(-1)
    return {
        "eigenvalues": eigenvalues,
        "n_weak": int((eigenvalues <= rank_tol).sum()),
        "lowest_vectors": lowest,
        "participation": participation,
    }


# --------------------------------------------------------------------------- #
# Reference transfer
# --------------------------------------------------------------------------- #


def reference_dbeta(lin, tracks, batches, geometry, alignment, rows, delta, active_dof=ACTIVE_DOF):
    """``dbeta_t = sum B_ref^T dr`` for a central-difference reference shift ``+-delta``.

    Only tracks with a reference component on ``rows`` can change; the others
    are skipped.  A component enters only if it is valid at both shifts and in
    ``lin``'s frozen set.
    """
    if rows.dtype != torch.bool:
        mask = torch.zeros(len(geometry.centers), dtype=torch.bool)
        mask[rows] = True
        rows = mask
    on_rows = rows[batches.sensor_idx] & lin.is_ref
    affected = torch.nonzero(on_rows.any(dim=1), as_tuple=True)[0]
    dbeta = torch.zeros((lin.residual.shape[0], J.N_TRACK_PARAMETERS), dtype=DTYPE)
    if len(affected) == 0:
        return dbeta
    sub = select_rows(batches, affected)
    shifted = []
    for sign in (1.0, -1.0):
        moved = O.inject_rows(geometry, rows, sign * delta)
        shifted.append(J.whitened_residuals(tracks[affected], sub, moved, alignment, active_dof))
    (plus, valid_plus), (minus, valid_minus) = shifted
    keep = (lin.component_mask[affected] & lin.is_ref[affected].unsqueeze(-1)
            & valid_plus & valid_minus).to(DTYPE)
    dr = 0.5 * (plus - minus) * keep
    dbeta[affected] = torch.einsum("thci,thc->ti", lin.track_jacobian[affected], dr)
    return dbeta


def _pair_sum(blocks: J.JointNormalBlocks, track_vectors: torch.Tensor) -> torch.Tensor:
    """``sum_t G_ti v_t`` per sensor for per-track 5-vectors ``v``."""
    k = blocks.block_size
    return torch.zeros((blocks.n_blocks, k), dtype=DTYPE).index_add_(
        0, blocks.pair_block, (blocks.G @ track_vectors[blocks.pair_track].unsqueeze(-1)).squeeze(-1)
    )


def schur_reference_transfer(blocks, inverse, dbeta, scales=SCALES) -> torch.Tensor:
    """Joint first-order response ``-S^-1 db``, ``db = -sum_t G_ti Gamma_t^-1 dbeta_t``, physical."""
    solved = torch.linalg.solve(blocks.gamma, dbeta.unsqueeze(-1)).squeeze(-1)
    db = -_pair_sum(blocks, solved)
    step = -(inverse @ db.reshape(-1)).reshape(db.shape)
    return step * scales


def fixed_track_reference_transfer(blocks, dbeta, scales=SCALES) -> torch.Tensor:
    """Fixed-track response: ref-only refit ``dtheta = -Gamma_ref^-1 dbeta``, then ``-C^-1 G dtheta``."""
    dtheta = -torch.linalg.solve(blocks.gamma_ref, dbeta.unsqueeze(-1)).squeeze(-1)
    information, gradient = blocks.geometry_information()
    return O.linear_transfer(information, gradient, gradient + _pair_sum(blocks, dtheta), scales)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def level_c_rows(event_ids: torch.Tensor, modulus: int = LEVEL_C_MODULUS) -> torch.Tensor:
    return torch.nonzero(event_ids % modulus == 0, as_tuple=True)[0]


def lambda_equivalence(values, reference, sigma, tolerance: float = LAMBDA_TOLERANCE_SIGMA) -> dict:
    """Iteration 39: ``|a(lambda_init) - a(ref)| <= 0.2 sigma`` on every DOF."""
    ratio = (torch.as_tensor(values) - torch.as_tensor(reference)).abs() / torch.as_tensor(sigma)
    finite = torch.isfinite(ratio)
    worst = float(ratio[finite].max()) if bool(finite.any()) else math.nan
    return {
        "equivalent": bool(finite.all()) and worst <= tolerance,
        "max_over_sigma": worst,
        "n_non_finite": int((~finite).sum()),
        "tolerance_sigma": tolerance,
    }


def injection_recovery(values, control, injection, masks) -> dict:
    """Iteration 44 §B: ``mean_i (a_i(s) - a_i(control)) / (-s delta)`` per layer and DOF."""
    values, control = torch.as_tensor(values), torch.as_tensor(control)
    ratio = (values - control) / (-torch.as_tensor(injection))
    return {name: ratio[torch.as_tensor(mask)].mean(dim=0).tolist() for name, mask in masks.items()}


def trial_summary(trials: list) -> dict:
    """Iteration 44 §B: first-step gain ratio, rejections and constraint activity of one run."""
    accepted = [row for row in trials if row["accepted"]]
    rejections: dict[str, int] = {}
    for row in trials:
        if not row["accepted"]:
            rejections[row["reject_reason"]] = rejections.get(row["reject_reason"], 0) + 1
    first = trials[0] if trials else None
    return {
        "n_trials": len(trials),
        "rho": [row["rho"] for row in trials],
        "first_rho": first["rho"] if first else math.nan,
        "first_accepted_rho": accepted[0]["rho"] if accepted else math.nan,
        "first_predicted": first["predicted"] if first else math.nan,
        "first_actual": first["actual"] if first else math.nan,
        "first_attribution": (
            [first["target_geometry_part"], first["target_track_part"]] if first else None
        ),
        "rejections": rejections,
        "min_alpha": min((row["alpha"] for row in accepted), default=math.nan),
        "max_resolves": max((row["n_resolves"] for row in trials), default=0),
        "min_n_free": min((row["n_free"] for row in trials), default=0),
    }


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1.0e6


# --------------------------------------------------------------------------- #
# Real-data driver
# --------------------------------------------------------------------------- #


STAGES = ("level-c", "control", "lambda-scan", "sigma", "spectrum", "target", "reference", "amplitude", "report")
#: Iteration 44 §A: P10 scale factors on ``(0.5 mm, 0.5 mm, 0.02 rad)``, both signs.
AMPLITUDE_SCALES = (0.5, 1.0, 2.0)
AMPLITUDE_UNIT = torch.tensor([0.5, 0.5, 0.02], dtype=DTYPE)
LINEAR_RHO_BAND = (0.75, 1.25)
RECOVERY_TOLERANCE = 0.1


class Context(O.Context):
    """M1's context, reading M1's cache and writing to this run's directory."""

    def __init__(self, args):
        super().__init__(args)
        self.m1_root = Path(args.output_root) / args.m1_run_id

    @property
    def cache_dir(self) -> Path:
        return self.m1_root / "prepare" / "cache"

    def control_alignment(self) -> torch.Tensor:
        """M1's fixed-track converged point ``a*``."""
        arm = np.load(self.m1_root / "control" / "arms" / "full_control.npz")
        return self._alignment(arm["surfaces"], arm["deltas"])

    def schur_point(self):
        """``a_S*`` and the tracks of this run's control."""
        arm = np.load(self.root / "control" / "arms" / "full_control.npz")
        return self._alignment(arm["surfaces"], arm["deltas"]), torch.as_tensor(arm["tracks"])

    def _alignment(self, surfaces, deltas) -> torch.Tensor:
        alignment = torch.zeros((len(self.geometry.centers), 6), dtype=DTYPE)
        alignment[torch.as_tensor(surfaces)] = torch.as_tensor(deltas)
        return alignment


def run_schur(ctx: Context, batches, tracks, geometry, path: Path, lambda_init=J.SCHUR_LM_LAMBDA_INIT):
    """One ``schur-lm`` arm saved as ``path`` (.npz) with a JSON summary; skipped if present."""
    if path.is_file():
        return dict(np.load(path, allow_pickle=False))
    started = time.perf_counter()
    result = J.run_schur_lm_alignment(
        batches, geometry, M.AlignmentRunConfig(), ACTIVE_DOF,
        initial_track_params=tracks, lambda_init=lambda_init,
    )
    elapsed = time.perf_counter() - started
    summary = result["schur_lm_summary"]
    surfaces = ctx.target_surfaces
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        surfaces=surfaces.numpy(),
        deltas=np.asarray(result["predicted_deltas"])[surfaces.numpy()],
        tracks=np.asarray(result["track_params"]),
        status=np.array(summary["status"]),
        reason=np.array(summary["reason"]),
    )
    trials = result["schur_lm_trials"]
    O._save_json(path.with_suffix(".json"), {
        "seconds": elapsed,
        "seconds_per_outer_iteration": elapsed / max(summary["outer_iterations"], 1),
        "peak_rss_gb": _peak_rss_gb(),
        "lambda_init": lambda_init,
        "n_tracks": int(tracks.shape[0]),
        "schur_lm_summary": summary,
        "iterations": result["schur_lm_iterations"],
        "trials": trials,
        "n_components_per_iteration": sorted({
            (row["iteration"], row["n_components_ref"], row["n_components_tgt"]) for row in trials
        }),
        "invalid_fraction": result["invalid_fraction"],
    })
    print(f"  [arm] {path.name}: {elapsed:.0f} s, {summary['status']} ({summary['reason']}), "
          f"{summary['outer_iterations']} outer", flush=True)
    return dict(np.load(path, allow_pickle=False))


def _usable(arm) -> bool:
    return str(arm["reason"]) == "stationary"


def stage_level_c(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    index = level_c_rows(cache["batches"].event_track_ids[:, 0])
    batches = select_rows(cache["batches"], index)
    path = ctx.root / "level-c" / "arms" / "level_c.npz"
    arm = run_schur(ctx, batches, cache["tracks"][index], ctx.geometry, path)
    summary = json.loads(path.with_suffix(".json").read_text())
    sensors = torch.unique(batches.sensor_idx[batches.is_target & batches.hit_mask])
    loc0, loc1 = O.component_counts(batches, sensors)
    finite = bool(np.isfinite(arm["deltas"]).all() and np.isfinite(arm["tracks"]).all())
    status = str(arm["status"])
    metrics.update({
        "n_tracks": int(len(index)),
        "status": status,
        "reason": str(arm["reason"]),
        "finite": finite,
        "n_loc0_only_targets": int(((loc1 == 0) & (loc0 > 0)).sum()),
        "seconds": summary["seconds"],
        "seconds_per_outer_iteration": summary["seconds_per_outer_iteration"],
        "peak_rss_gb": summary["peak_rss_gb"],
        "component_loss_rejections": summary["schur_lm_summary"]["component_loss_rejections"],
        "gate": "passed" if finite and status != J.SCHUR_LM_STATUS_NUMERICAL_FAILURE else "failed",
    })
    if metrics["gate"] != "passed":
        raise ValueError(f"Level C gate failed: status={status}, finite={finite}")
    return {"arm": path}


def stage_control(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    path = ctx.root / "control" / "arms" / "full_control.npz"
    arm = run_schur(ctx, cache["batches"], cache["tracks"], ctx.geometry, path)
    summary = json.loads(path.with_suffix(".json").read_text())
    metrics.update({
        "status": str(arm["status"]),
        "reason": str(arm["reason"]),
        "seconds": summary["seconds"],
        "peak_rss_gb": summary["peak_rss_gb"],
        "schur_lm_summary": summary["schur_lm_summary"],
        "gate_d1": "passed" if str(arm["status"]) == J.SCHUR_LM_STATUS_CONVERGED else "failed",
    })
    if metrics["gate_d1"] != "passed":
        raise ValueError(f"Gate D1 failed: control status {arm['status']} ({arm['reason']})")
    return {"control": path}


def stage_lambda_scan(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    for value in LAMBDA_SCAN:
        path = ctx.root / "lambda-scan" / "arms" / f"full_lambda_{value:.0e}.npz"
        arm = run_schur(ctx, cache["batches"], cache["tracks"], ctx.geometry, path, lambda_init=value)
        metrics[f"{value:.0e}"] = {"status": str(arm["status"]), "reason": str(arm["reason"])}
    return {"arms": ctx.root / "lambda-scan" / "arms"}


def _system_at(ctx: Context, batches, tracks, alignment):
    lin = J.linearize_tracks(tracks, batches, ctx.geometry, alignment, ACTIVE_DOF)
    blocks = J.normal_blocks(lin)
    return lin, blocks, J.schur_reduce(blocks, 0.0)


def stage_sigma(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    alignment, tracks = ctx.schur_point()
    _lin, blocks, system = _system_at(ctx, cache["batches"], tracks, alignment)
    sigma = O._to_target_order(ctx, _lin.target_sensors, marginal_sigma(dense_schur(system)))
    control = np.load(ctx.root / "control" / "arms" / "full_control.npz")
    reference = control["deltas"][:, list(ACTIVE_DOF)]
    has = torch.isfinite(sigma).all(dim=1).numpy()
    scan = {}
    for value in LAMBDA_SCAN:
        arm = np.load(ctx.root / "lambda-scan" / "arms" / f"full_lambda_{value:.0e}.npz")
        result = lambda_equivalence(arm["deltas"][has][:, list(ACTIVE_DOF)], reference[has], sigma[has])
        result["status"] = str(arm["status"])
        result["equivalent"] = result["equivalent"] and result["status"] == J.SCHUR_LM_STATUS_CONVERGED
        scan[f"{value:.0e}"] = result
    passed = all(entry["equivalent"] for entry in scan.values())
    path = ctx.root / "sigma" / "sigma_schur_star.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, surfaces=ctx.target_surfaces.numpy(), sigma=sigma.numpy())
    metrics.update({
        "n_sensors_with_sigma": int(has.sum()),
        "sigma_median": {
            f"{layer[0]}:{layer[1]}|{name}": float(np.nanmedian(sigma.numpy()[mask, d]))
            for layer, mask in O._target_layer_masks(ctx).items()
            for d, name in enumerate(DOF_NAMES)
        },
        "lambda_scan": scan,
        "gate_d2": "lambda_init_insensitive" if passed else "solver_sensitive",
    })
    outputs = {"sigma": path}
    if passed:
        deltas = torch.zeros((len(ctx.geometry.centers), 6), dtype=DTYPE)
        deltas[ctx.target_surfaces] = torch.as_tensor(control["deltas"])
        candidate = Path(ctx.args.candidate_dir) / f"aligned_transforms_{ctx.args.run_id}_schur_lm.json"
        M.export_joint_geometry(ctx.geometry, deltas.numpy(), str(candidate))
        metrics["candidate"] = str(candidate)
        outputs["candidate"] = candidate
    else:
        raise ValueError(f"Gate D2 failed: {scan}")
    return outputs


def _spectrum_point(ctx: Context, batches, tracks, alignment) -> dict:
    lin, blocks, system = _system_at(ctx, batches, tracks, alignment)
    k = len(ACTIVE_DOF)
    dense = dense_schur(system)
    information, _gradient = blocks.geometry_information()
    s_ii = diagonal_blocks(dense, k)
    spectrum = O.jacobi_spectrum(s_ii)
    global_spectrum = global_jacobi_spectrum(dense, k)
    rows, cols, rho = coupling_strength(system)
    order = lambda values: O._to_target_order(ctx, lin.target_sensors, values)  # noqa: E731
    record = {
        "information_c": order(information),
        "information_s": order(s_ii),
        "retention": order(information_retention(s_ii, information)),
        "sigma_c": order(O.statistical_sigma(information)),
        "sigma_s_local": order(O.statistical_sigma(s_ii)),
        "sigma_s": order(marginal_sigma(dense)),
        "eigenvalues": order(spectrum.eigenvalues),
        "eigenvectors": order(spectrum.eigenvectors),
        "observable": order(spectrum.observable),
        "has_data": order(torch.ones(len(lin.target_sensors), dtype=torch.bool)),
        "global_eigenvalues": global_spectrum["eigenvalues"],
        "global_lowest_participation": global_spectrum["participation"],
        "coupling_rows": lin.target_sensors[rows],
        "coupling_cols": lin.target_sensors[cols],
        "coupling_rho": rho,
    }
    check = loewner_check(information, s_ii, dense)
    return {"record": record, "loewner": check, "n_global_weak": global_spectrum["n_weak"]}


def _quantiles(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"n": 0}
    return {"n": int(len(finite)), "min": float(finite.min()), "p05": float(np.quantile(finite, 0.05)),
            "p50": float(np.median(finite)), "p95": float(np.quantile(finite, 0.95)),
            "max": float(finite.max())}


def stage_spectrum(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    batches, ref_tracks = cache["batches"], cache["tracks"]
    schur_alignment, schur_tracks = ctx.schur_point()
    points = {
        "zero": (ref_tracks, None),
        "star": (ref_tracks, ctx.control_alignment()),
        "schur": (schur_tracks, schur_alignment),
    }
    outputs, summaries = {}, {}
    for name, (tracks, alignment) in points.items():
        result = _spectrum_point(ctx, batches, tracks, alignment)
        record = result["record"]
        path = ctx.root / "spectrum" / f"spectrum_{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, surfaces=ctx.target_surfaces.numpy(), **{k: v.numpy() for k, v in record.items()})
        outputs[name] = path
        per_cell = {}
        for layer, mask in O._target_layer_masks(ctx).items():
            key = f"{layer[0]}:{layer[1]}"
            retention = record["retention"].numpy()[mask]
            per_cell[key] = {"retention": _quantiles(retention)}
            for d, dof in enumerate(DOF_NAMES):
                ratio = (record["sigma_s"][:, d] / record["sigma_c"][:, d]).numpy()[mask]
                per_cell[key][dof] = {
                    "sigma_c_median": float(np.nanmedian(record["sigma_c"].numpy()[mask, d])),
                    "sigma_s_median": float(np.nanmedian(record["sigma_s"].numpy()[mask, d])),
                    "sigma_ratio": _quantiles(ratio),
                }
            local_weak = ~record["observable"].numpy()[mask] & record["has_data"].numpy()[mask, None]
            per_cell[key]["n_local_weak_modes"] = int(local_weak.sum())
        rho = record["coupling_rho"].numpy()
        top = np.argsort(-rho)[:N_TOP_PAIRS]
        keys = ctx.surface_keys()
        summaries[name] = {
            "loewner": result["loewner"],
            "n_global_weak": result["n_global_weak"],
            "global_lowest_eigenvalues": record["global_eigenvalues"][:N_LOWEST_MODES].tolist(),
            "cells": per_cell,
            "coupling_rho": _quantiles(rho),
            "n_coupling_blocks": int(len(rho)),
            "top_pairs": [
                {"i": keys.get(int(record["coupling_rows"][p]), int(record["coupling_rows"][p])),
                 "j": keys.get(int(record["coupling_cols"][p]), int(record["coupling_cols"][p])),
                 "rho": float(rho[p])}
                for p in top
            ],
        }
    metrics["points"] = summaries
    metrics["gate_d3"] = "passed" if all(s["loewner"]["passed"] for s in summaries.values()) else "failed"
    if metrics["gate_d3"] != "passed":
        raise ValueError(f"Gate D3 failed: {[s['loewner'] for s in summaries.values()]}")
    return outputs


def _target_arm_path(ctx, sample, column, sign):
    return ctx.root / "target" / "arms" / f"{sample}_{DOF_NAMES[column]}_{'plus' if sign > 0 else 'minus'}.npz"


def _sample_spectrum(ctx, sample, batches, tracks) -> Path:
    """``S_ii`` and its Jacobi spectrum at ``a*`` on one sample's tracks."""
    path = ctx.root / "target" / "spectrum" / f"spectrum_{sample}.npz"
    if path.is_file():
        return path
    lin, _blocks, system = _system_at(ctx, batches, tracks, ctx.control_alignment())
    s_ii = diagonal_blocks(dense_schur(system), len(ACTIVE_DOF))
    spectrum = O.jacobi_spectrum(s_ii)
    order = lambda values: O._to_target_order(ctx, lin.target_sensors, values)  # noqa: E731
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, surfaces=ctx.target_surfaces.numpy(), information=order(s_ii).numpy(),
        eigenvalues=order(spectrum.eigenvalues).numpy(), eigenvectors=order(spectrum.eigenvectors).numpy(),
        absent=order(spectrum.structurally_absent).numpy(), observable=order(spectrum.observable).numpy(),
        has_data=order(torch.ones(len(lin.target_sensors), dtype=torch.bool)).numpy(),
    )
    return path


def stage_target(ctx: Context, metrics: dict) -> dict:
    timings = {}
    for name, batches, tracks, _index in O._subsamples(ctx):
        _sample_spectrum(ctx, name, batches, tracks)
        for column, dof in enumerate(ACTIVE_DOF):
            for sign in (1.0, -1.0):
                delta = torch.zeros(6, dtype=DTYPE)
                delta[dof] = sign * O.TARGET_AMPLITUDES[column]
                geometry = O.inject_rows(ctx.geometry, ctx.geometry.target_mask, delta)
                path = _target_arm_path(ctx, name, column, sign)
                run_schur(ctx, batches, tracks, geometry, path)
                timings[path.name] = json.loads(path.with_suffix(".json").read_text())["seconds"]
    metrics["seconds"] = timings
    return {"arms": ctx.root / "target" / "arms"}


def stage_amplitude(ctx: Context, metrics: dict) -> dict:
    """P10 (Iteration 44): coherent all-DOF target injections of growing size on top of the misalignment."""
    cache = ctx.cache()
    control = np.load(ctx.root / "control" / "arms" / "full_control.npz")["deltas"][:, list(ACTIVE_DOF)]
    masks = {f"{layer[0]}:{layer[1]}": mask for layer, mask in O._target_layer_masks(ctx).items()}
    runs, linear = {}, []
    stop = []
    for scale in AMPLITUDE_SCALES:
        for sign in (1.0, -1.0):
            injection = sign * scale * AMPLITUDE_UNIT
            delta = torch.zeros(6, dtype=DTYPE)
            delta[list(ACTIVE_DOF)] = injection
            geometry = O.inject_rows(ctx.geometry, ctx.geometry.target_mask, delta)
            name = f"s{scale:g}_{'plus' if sign > 0 else 'minus'}"
            path = ctx.root / "amplitude" / "arms" / f"{name}.npz"
            arm = run_schur(ctx, cache["batches"], cache["tracks"], geometry, path)
            log = json.loads(path.with_suffix(".json").read_text())
            summary = trial_summary(log["trials"])
            summary.update(status=str(arm["status"]), reason=str(arm["reason"]),
                           outer_iterations=log["schur_lm_summary"]["outer_iterations"],
                           seconds=log["seconds"])
            if str(arm["status"]) == J.SCHUR_LM_STATUS_CONVERGED:
                recovery = injection_recovery(arm["deltas"][:, list(ACTIVE_DOF)], control, injection, masks)
                summary["recovery"] = recovery
                if any(abs(value - 1.0) > RECOVERY_TOLERANCE for per in recovery.values() for value in per):
                    stop.append(f"{name}: recovery {recovery}")
            if str(arm["status"]) in (J.SCHUR_LM_STATUS_NUMERICAL_FAILURE, J.SCHUR_LM_STATUS_INTERSECTION_FAILED):
                stop.append(f"{name}: {arm['status']}")
            if LINEAR_RHO_BAND[0] <= summary["first_rho"] <= LINEAR_RHO_BAND[1]:
                linear.append(scale)
            runs[name] = summary
    metrics["runs"] = runs
    metrics["gn_trusted_max_scale"] = max(
        (scale for scale in AMPLITUDE_SCALES if linear.count(scale) == 2), default=None
    )
    metrics["stop_conditions"] = stop
    if stop:
        raise ValueError(f"P10 stop condition: {stop}")
    return {"arms": ctx.root / "amplitude" / "arms"}


def _reference_bias_m1(ctx, key, column):
    arms = ctx.m1_root / "reference-linear" / "arms"
    plus = np.load(arms / f"{key[0]}_{key[1]}_{DOF_NAMES[column]}_plus.npz")["transfer"]
    minus = np.load(arms / f"{key[0]}_{key[1]}_{DOF_NAMES[column]}_minus.npz")["transfer"]
    return 0.5 * (plus - minus)


def stage_reference(ctx: Context, metrics: dict) -> dict:
    cache = ctx.cache()
    batches, tracks = cache["batches"], cache["tracks"]
    alignment = ctx.control_alignment()
    lin, blocks, system = _system_at(ctx, batches, tracks, alignment)
    inverse = torch.linalg.inv(dense_schur(system))
    sigma_c = np.load(ctx.m1_root / "spectrum" / "spectrum_full.npz")["sigma"]
    layers = O.reference_layers(batches)
    arms_dir = ctx.root / "reference" / "arms"
    oracle_checks = []
    for key, rows in layers.items():
        for column in range(len(ACTIVE_DOF)):
            path = arms_dir / f"{key[0]}_{key[1]}_{DOF_NAMES[column]}.npz"
            if not path.is_file():
                delta = torch.zeros(6, dtype=DTYPE)
                delta[ACTIVE_DOF[column]] = O.REFERENCE_AMPLITUDES[column]
                dbeta = reference_dbeta(lin, tracks, batches, ctx.geometry, alignment, rows, delta)
                schur = O._to_target_order(ctx, lin.target_sensors, schur_reference_transfer(blocks, inverse, dbeta))
                fixed = O._to_target_order(ctx, lin.target_sensors, fixed_track_reference_transfer(blocks, dbeta))
                arms_dir.mkdir(parents=True, exist_ok=True)
                np.savez(path, surfaces=ctx.target_surfaces.numpy(), schur=schur.numpy(), fixed=fixed.numpy())
            arm = np.load(path)
            m1 = _reference_bias_m1(ctx, key, column)
            keep = np.isfinite(m1).all(1) & np.isfinite(arm["fixed"]).all(1) & np.isfinite(sigma_c).all(1)
            within = (np.abs(arm["fixed"] - m1)[keep] <= ORACLE_SIGMA_TOL * sigma_c[keep]).all(1)
            fraction = float(within.mean()) if len(within) else math.nan
            oracle_checks.append({"reference_layer": list(key), "reference_dof": DOF_NAMES[column],
                                  "n_sensors": int(keep.sum()), "fraction_within": fraction,
                                  "passed": fraction >= ORACLE_FRACTION_MIN})
    metrics["layers"] = [list(key) for key in layers]
    metrics["oracle_checks"] = oracle_checks
    metrics["gate_f1"] = "passed" if all(row["passed"] for row in oracle_checks) else "failed"
    return {"arms": arms_dir}


def transfer_ratio(schur_rms: float, fixed_rms: float) -> float:
    """Schur / fixed-track RMS transfer; a reference layer with no shared track has fixed = 0 exactly."""
    if fixed_rms > 0.0:
        return schur_rms / fixed_rms
    return math.inf if schur_rms > 0.0 else math.nan


def reference_comparison(ctx: Context) -> dict:
    """Iteration 41 §B H1-R from the transfer arms, the M1 table and ``sigma_S`` at ``a*``."""
    m1 = json.loads((ctx.m1_root / "report" / "m1_result.json").read_text())["secondary_reference"]["table"]
    sigma_s = np.load(ctx.root / "spectrum" / "spectrum_star.npz")["sigma_s"]
    masks = O._target_layer_masks(ctx)
    rows = []
    for entry in m1:
        key = tuple(entry["reference_layer"])
        arm = np.load(ctx.root / "reference" / "arms" / f"{key[0]}_{key[1]}_{entry['reference_dof']}.npz")
        mask = masks[tuple(entry["target_layer"])]
        tdof = DOF_NAMES.index(entry["target_dof"])
        values, sig = arm["schur"][mask, tdof], sigma_s[mask, tdof]
        finite = np.isfinite(values) & np.isfinite(sig)
        rms = float(np.sqrt(np.mean(values[finite] ** 2)))
        median = float(np.median(sig[finite]))
        rows.append({
            **{name: entry[name] for name in ("reference_layer", "reference_dof", "target_layer", "target_dof")},
            "rms_bias_fixed": entry["rms_bias"], "median_sigma_c": entry["median_sigma"],
            "fixed_sensitive": entry["reference_sensitive"],
            "rms_bias_schur": rms, "median_sigma_s": median, "n_sensors_schur": int(finite.sum()),
            "transfer_ratio": transfer_ratio(rms, entry["rms_bias"]),
            "schur_sensitive": rms > median,
        })
    sensitive = [row for row in rows if row["fixed_sensitive"]]
    for row in sensitive:
        row["decreased"] = row["transfer_ratio"] <= TRANSFER_DECREASE_RATIO
    condition_i = all(row["transfer_ratio"] <= TRANSFER_INCREASE_RATIO for row in sensitive)
    new_sensitive = [row for row in rows if not row["fixed_sensitive"] and row["schur_sensitive"]]
    return {
        "table": rows,
        "fixed_sensitive": sensitive,
        "n_decreased": sum(row["decreased"] for row in sensitive),
        "condition_i_no_increase": condition_i,
        "condition_ii_no_new_sensitive": len(new_sensitive) == 0,
        "new_sensitive": new_sensitive,
        "nominal_reference_independent": condition_i and len(new_sensitive) == 0,
    }


def _primary(ctx: Context) -> dict:
    samples = ["full"] + [f"sub{part}" for part in range(ctx.args.subsamples)]
    masks = O._target_layer_masks(ctx)
    cells: dict[str, dict] = {}
    for sample in samples:
        spectrum_file = np.load(ctx.root / "target" / "spectrum" / f"spectrum_{sample}.npz")
        has = torch.as_tensor(spectrum_file["has_data"])
        information = torch.as_tensor(spectrum_file["information"])
        spectrum = O.Spectrum(
            torch.as_tensor(spectrum_file["eigenvalues"]), torch.as_tensor(spectrum_file["eigenvectors"]),
            torch.as_tensor(spectrum_file["absent"]), torch.as_tensor(spectrum_file["observable"]),
        )
        plus = {c: np.load(_target_arm_path(ctx, sample, c, 1.0)) for c in range(3)}
        minus = {c: np.load(_target_arm_path(ctx, sample, c, -1.0)) for c in range(3)}
        usable = all(_usable(arm) for arm in (*plus.values(), *minus.values()))
        response = O.response_matrix(
            {c: plus[c]["deltas"][:, list(ACTIVE_DOF)] for c in range(3)},
            {c: minus[c]["deltas"][:, list(ACTIVE_DOF)] for c in range(3)},
            O.TARGET_AMPLITUDES,
        )
        safe = torch.where(has[:, None, None], information, torch.eye(3, dtype=DTYPE))
        slopes = O.mode_slopes(response, safe, spectrum).numpy()
        observable = spectrum.observable.numpy() & has.numpy()[:, None]
        for layer, mask in masks.items():
            for k in range(3):
                name = f"{layer[0]}:{layer[1]}|mode{k}"
                if not usable:
                    cells.setdefault(name, {})[sample] = {"n": 0, "inside": False, "not_evaluable": True}
                    continue
                keep = mask & observable[:, k] & np.isfinite(slopes[:, k])
                cells.setdefault(name, {})[sample] = O._cell_ci(slopes[keep, k])
    verdict = {
        name: ("observable_confirmed" if all(entry["inside"] for entry in per.values())
               else "unobservable_or_prior_dominated")
        for name, per in cells.items()
    }
    m1 = json.loads((ctx.m1_root / "report" / "m1_result.json").read_text())["primary_target_injection"]
    return {"cells": cells, "verdict": verdict, "m1_verdict": m1["verdict"],
            "m1_full": {name: per["full"] for name, per in m1["cells"].items()},
            "passed": all(value == "observable_confirmed" for value in verdict.values())}


def stage_report(ctx: Context, metrics: dict) -> dict:
    def stage_metrics(name):
        return json.loads((ctx.root / name / "manifest.json").read_text())["metrics"]

    reference = stage_metrics("reference")
    result = {
        "preregistration": PREREGISTRATION,
        "level_c": stage_metrics("level-c"),
        "control": stage_metrics("control"),
        "sigma": stage_metrics("sigma"),
        "spectrum": stage_metrics("spectrum")["points"],
        "gate_d3": stage_metrics("spectrum")["gate_d3"],
        "primary_target_injection": _primary(ctx),
        "reference_gate_f1": reference["gate_f1"],
        "reference_oracle_checks": reference["oracle_checks"],
        "h1_r": reference_comparison(ctx) if reference["gate_f1"] == "passed" else "not_evaluable",
    }
    path = O._save_json(ctx.root / "report" / "p9_result.json", result)
    metrics.update({
        "primary_passed": result["primary_target_injection"]["passed"],
        "gate_f1": result["reference_gate_f1"],
        "nominal_reference_independent": (
            result["h1_r"]["nominal_reference_independent"] if isinstance(result["h1_r"], dict) else None
        ),
    })
    return {"result": path}


STAGE_FUNCTIONS = {
    "level-c": stage_level_c,
    "control": stage_control,
    "lambda-scan": stage_lambda_scan,
    "sigma": stage_sigma,
    "spectrum": stage_spectrum,
    "target": stage_target,
    "reference": stage_reference,
    "amplitude": stage_amplitude,
    "report": stage_report,
}


def _stage_inputs(ctx: Context) -> dict:
    return {
        "geometry": ctx.args.input_geom,
        "nominal_geometry": ctx.args.nominal_full_geom,
        "cache_batches": ctx.cache_dir / "batches.pt",
        "cache_tracks": ctx.cache_dir / "tracks.pt",
        "m1_control": ctx.m1_root / "control" / "arms" / "full_control.npz",
    }


def run_stage(ctx: Context, stage: str, command: list[str]) -> dict:
    directory = ctx.root / stage
    if O._manifest_status(directory) == "completed":
        print(f"[skip] {stage}: already completed", flush=True)
        return json.loads((directory / "manifest.json").read_text())["metrics"]
    metrics = {"stage": stage, "preregistration": PREREGISTRATION, "scope": SCOPE}
    manifest = write_experiment_manifest(
        directory,
        run_id=f"{ctx.args.run_id}/{stage}",
        command=command,
        config=ctx.config,
        inputs=_stage_inputs(ctx),
        metrics=metrics,
        capture_source_snapshot=ctx.args.capture_source_snapshot,
    )
    started = time.perf_counter()
    print(f"[run ] {stage}", flush=True)
    try:
        outputs = STAGE_FUNCTIONS[stage](ctx, metrics) or {}
        metrics["duration_s"] = time.perf_counter() - started
        metrics["peak_rss_gb"] = metrics.get("peak_rss_gb", _peak_rss_gb())
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
    parser.add_argument("--m1-run-id", default=M1_RUN_ID)
    parser.add_argument("--output-root", default=O.DEFAULTS["output_root"])
    parser.add_argument("--candidate-dir", default=O._repo_path("data/alignment/Vol17_Lay8__Vol24_Lay4"))
    for name in ("input_geom", "nominal_full_geom"):
        parser.add_argument(f"--{name.replace('_', '-')}", default=O.DEFAULTS[name])
    parser.add_argument("--target-layers", default="17:8,24:4")
    parser.add_argument("--subsamples", type=int, default=O.N_SUBSAMPLES)
    parser.add_argument("--capture-source-snapshot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subsamples != O.N_SUBSAMPLES:
        raise SystemExit(f"--subsamples is fixed at {O.N_SUBSAMPLES} (Iteration 42 §E).")
    ctx = Context(args)
    command = [sys.executable, str(Path(__file__).resolve()),
               *(sys.argv[1:] if argv is None else argv)]

    def interrupt(signum, _frame):
        raise O.ExperimentInterrupted(f"received signal {signum}")

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
