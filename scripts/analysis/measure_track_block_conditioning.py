"""Schur P2/P3 on real tracks: track-block conditioning and Jacobian closure.

plan.md Phases P2/P3, pre-registered in ``ml_optimization_log.md`` Iteration 34
§C (finite differences amended by Iteration 35).  On a production-selected
subset (first ``--max-tracks`` retained tracks) and at the start point of a
future Schur solve (input geometry, zero alignment, ref-only refit), report
``Gamma_ref`` / ``Gamma`` conditioning, the ref-only Newton decrement, and
check autograd ``A``/``B`` against Richardson finite differences on a fixed
random sample of tracks.

Diagnostic only: nothing here feeds a solver, and truth is never read.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
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
from helix_model import (  # noqa: E402
    MEASUREMENT_NEWTON_DECREMENT_STATIONARY,
    MeasurementFitInputs,
    helix_tangent,
    intersect_plane,
    measurement_newton_decrement,
    measurement_standardized_residual,
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    require_stationary_helix_point_fit,
)
from measure_prediction_error_budget import BASELINE_INPUT_KEYS  # noqa: E402
from project_config import add_config_argument, load_project_config  # noqa: E402

PREREGISTRATION = "ml_optimization_log.md Iteration 34 §C (FD per Iteration 35)"
FD_STEP = 1.0e-3
FD_RTOL = 1.0e-6
FD_ATOL_COLUMN = 1.0e-8
NEAR_PARALLEL_COS = 0.3
SIGNIFICANT_FRACTION = 1.0e-3
CLOSURE_SAMPLE_SEED = 20261011
QUANTILES = (0.5, 0.9, 0.99)


class ExperimentInterrupted(RuntimeError):
    pass


def select_rows(batches, index: torch.Tensor):
    """A ``TrackBatches`` restricted to ``index`` rows."""
    values = {}
    for name, value in vars(batches).items():
        values[name] = value[index] if isinstance(value, torch.Tensor) else value
    return M.TrackBatches(**values)


def _summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    result = {f"p{int(q * 100)}": float(np.quantile(finite, q)) for q in QUANTILES}
    result["max"] = float(finite.max())
    result["n_non_finite"] = int((~np.isfinite(values)).sum())
    return result


def _residual(batches, geometry, parameters, theta, alignment, component_mask):
    centers, rotations = M.apply_alignment_torch(
        geometry.centers.to(torch.float64), geometry.rotations.to(torch.float64), alignment
    )
    sensor = batches.sensor_idx
    inputs = MeasurementFitInputs(
        measured=batches.measured_local.to(torch.float64),
        variance=batches.covariances.to(torch.float64),
        component_mask=component_mask,
        centers=centers[sensor],
        rotations=rotations[sensor],
    )
    return measurement_standardized_residual(
        theta, reference_gauge_frame(parameters), inputs
    ).reshape(component_mask.shape)


def _richardson(evaluate):
    def central(h):
        return (evaluate(h) - evaluate(-h)) / (2 * h)

    return (4 * central(FD_STEP / 2) - central(FD_STEP)) / 3


def jacobian_closure(batches, geometry, parameters) -> dict[str, Any]:
    """Iteration 34 §B criterion on every element of ``A`` and ``B``, by stratum."""
    lin = J.linearize_tracks(parameters, batches, geometry)
    alignment = torch.zeros((len(geometry.centers), 6), dtype=torch.float64)
    mask = lin.component_mask
    theta = pack_gauge_fixed_track_parameters(parameters)

    sensor = batches.sensor_idx
    centers = geometry.centers.to(torch.float64)[sensor]
    normals = geometry.rotations.to(torch.float64)[sensor][..., :, 2]
    tracks = parameters[:, None, :].expand(-1, sensor.shape[1], -1)
    s, _point, _valid = intersect_plane(tracks, centers, normals)
    tangent = helix_tangent(tracks, s)
    cos = ((tangent / tangent.norm(dim=-1, keepdim=True)) * normals).sum(-1).abs()
    hit_strata = {
        "ref": lin.is_ref,
        "target": lin.is_target,
        "target_near_parallel": lin.is_target & (cos < NEAR_PARALLEL_COS),
        "target_strip_1d": lin.is_target & ~batches.loc1_mask,
    }
    strata = {}
    for name, hits in hit_strata.items():
        for component, label in ((0, "LOC0"), (1, "LOC1")):
            selector = torch.zeros_like(mask)
            selector[..., component] = hits
            strata[f"{name}_{label}"] = selector & mask

    counts = {
        name: {"n": 0, "n_failed": 0, "max_tolerance_ratio": 0.0, "max_rel_significant": 0.0}
        for name in strata
    }

    def record(column, ad, fd):
        scale = ad.abs().max()
        tolerance = FD_RTOL * ad.abs() + FD_ATOL_COLUMN * scale
        difference = (ad - fd).abs()
        passed = difference <= tolerance
        # Reporting only: how much of the tolerance is used, and the relative
        # error on elements that are not numerically zero within their column.
        ratio = torch.where(
            tolerance > 0, difference / tolerance.clamp_min(1e-300),
            torch.where(difference == 0, 0.0, float("inf")),
        )
        significant = ad.abs() >= SIGNIFICANT_FRACTION * scale
        relative = difference / ad.abs().clamp_min(1e-300)
        for name, selector in strata.items():
            counts[name]["n"] += int(selector.sum())
            counts[name]["n_failed"] += int((~passed & selector).sum())
            if bool(selector.any()):
                counts[name]["max_tolerance_ratio"] = max(
                    counts[name]["max_tolerance_ratio"], float(ratio[selector].max())
                )
            chosen = selector & significant & (ad != 0)
            if bool(chosen.any()):
                counts[name]["max_rel_significant"] = max(
                    counts[name]["max_rel_significant"], float(relative[chosen].max())
                )
        structural = bool((ad[~mask] == 0).all()) and bool((fd[~mask] == 0).all())
        if not structural:
            raise AssertionError(f"Non-zero derivative outside the frozen set in {column}.")

    for j in range(5):
        unit = torch.zeros_like(theta)
        unit[:, j] = 1.0
        fd = _richardson(
            lambda h: _residual(batches, geometry, parameters, theta + h * unit, alignment, mask)
        )
        record(f"theta{j}", lin.track_jacobian[..., j], fd)
    for block, surface in enumerate(lin.target_sensors.tolist()):
        on_block = (lin.block_index == block).unsqueeze(-1)
        for column, dof in enumerate(lin.active_dof):
            unit = torch.zeros_like(alignment)
            unit[surface, dof] = J.GEOMETRY_SCALES[dof]
            fd = _richardson(
                lambda h: _residual(batches, geometry, parameters, theta, alignment + h * unit, mask)
            )
            ad = torch.where(on_block, lin.geometry_jacobian[..., column], 0.0)
            record(f"block{block}_dof{dof}", ad, fd)
    total_failed = sum(item["n_failed"] for item in counts.values())
    return {
        "n_tracks": int(parameters.shape[0]),
        "n_target_blocks": len(lin.target_sensors),
        "strata": counts,
        "n_failed": int(total_failed),
        "passed": total_failed == 0,
    }


def track_blocks(batches, geometry, parameters) -> tuple[pd.DataFrame, dict[str, Any]]:
    lin = J.linearize_tracks(parameters, batches, geometry)
    blocks = J.normal_blocks(lin)
    diagnostics = J.track_block_diagnostics(blocks)
    decrement = measurement_newton_decrement(
        parameters, M._measurement_fit_inputs_from_joint(batches, geometry, parameters)
    )
    trace_ratio = (
        blocks.gamma_tgt.diagonal(dim1=-2, dim2=-1).sum(-1)
        / blocks.gamma_ref.diagonal(dim1=-2, dim2=-1).sum(-1)
    )
    table = pd.DataFrame(
        {
            "event_id": batches.event_track_ids[:, 0].numpy(),
            "track_id": batches.event_track_ids[:, 1].numpy(),
            "n_ref_components": (lin.component_mask & lin.is_ref.unsqueeze(-1)).sum((1, 2)).numpy(),
            "n_target_components": (lin.component_mask & lin.is_target.unsqueeze(-1)).sum((1, 2)).numpy(),
            "ref_newton_decrement": decrement.numpy(),
            "trace_gamma_tgt_over_ref": trace_ratio.numpy(),
            **{
                f"{name}_{field}": diagnostics[name][field].to(torch.float64).numpy()
                for name in ("gamma_ref", "gamma")
                for field in ("condition", "rank", "asymmetry")
            },
            **{
                f"{name}_cholesky_ok": diagnostics[name]["cholesky_ok"].numpy()
                for name in ("gamma_ref", "gamma")
            },
        }
    )
    summary = {
        "n_tracks": len(table),
        "n_ref_components": int(table.n_ref_components.sum()),
        "n_target_components": int(table.n_target_components.sum()),
        "ref_newton_decrement": _summary(table.ref_newton_decrement.to_numpy()),
        "n_ref_non_stationary": int(
            (table.ref_newton_decrement > MEASUREMENT_NEWTON_DECREMENT_STATIONARY).sum()
        ),
        "trace_gamma_tgt_over_ref": _summary(table.trace_gamma_tgt_over_ref.to_numpy()),
    }
    for name in ("gamma_ref", "gamma"):
        summary[name] = {
            "condition": _summary(table[f"{name}_condition"].to_numpy()),
            "rank_counts": {
                str(int(rank)): int(count)
                for rank, count in table[f"{name}_rank"].value_counts().sort_index().items()
            },
            "max_asymmetry": float(table[f"{name}_asymmetry"].max()),
            "n_cholesky_failed": int((~table[f"{name}_cholesky_ok"]).sum()),
            "n_non_finite": int((~diagnostics[name]["finite"]).sum()),
        }
    return table, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_argument(parser)
    parser.add_argument("--input-geom", required=True)
    parser.add_argument("--nominal-full-geom", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-layers", default="17:8,24:4")
    parser.add_argument("--min-hits", type=int, default=8)
    parser.add_argument("--min-span", type=float, default=300.0)
    parser.add_argument("--smt-seed-iters", type=int, default=2000)
    parser.add_argument("--max-tracks", type=int, default=2000)
    parser.add_argument("--closure-tracks", type=int, default=200)
    parser.add_argument("--capture-source-snapshot", action="store_true")
    return parser


def check_baseline_inputs(own: dict, baseline: dict) -> None:
    for key in BASELINE_INPUT_KEYS:
        if own["inputs"][key]["sha256"] != baseline["inputs"][key]["sha256"]:
            raise ValueError(f"Input {key!r} differs from the P0.1 baseline.")


def run(args) -> dict[str, Any]:
    if args.max_tracks <= 0:
        raise ValueError("This diagnostic is pre-registered on a subset; --max-tracks must be > 0.")
    output_dir = Path(args.output_dir)
    baseline = json.loads(Path(args.baseline_manifest).read_text(encoding="utf-8"))
    check_baseline_inputs(json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")), baseline)
    target_layers = M.parse_target_layers(args.target_layers)
    geometry = M.load_joint_geometry(args.nominal_full_geom, args.input_geom, target_layers)
    inputs = M.load_variable_information(
        args.summary,
        args.states,
        geometry.id_to_idx,
        max_track_entries=args.max_tracks * 4,
        load_summary=False,
    )
    batches = M.build_track_batches(
        inputs, target_layers, min_hits=args.min_hits, min_span=args.min_span, max_tracks=args.max_tracks
    )
    del inputs
    seeded, seed_stats = M.seed_helices(
        batches, n_iter=args.smt_seed_iters, dtype=torch.float64, return_diagnostics=True
    )
    require_stationary_helix_point_fit(seed_stats)
    refitted, refit_stats = M.fit_helices_to_measurements(
        batches, geometry, seeded, dtype=torch.float64, stationarity=False
    )
    parameters = refitted.detach().to(device="cpu", dtype=torch.float64)

    table, blocks_summary = track_blocks(batches, geometry, parameters)
    generator = torch.Generator().manual_seed(CLOSURE_SAMPLE_SEED)
    sample = torch.sort(
        torch.randperm(len(parameters), generator=generator)[: args.closure_tracks]
    ).values
    closure = jacobian_closure(select_rows(batches, sample), geometry, parameters[sample])

    gate = {
        "n_non_finite": blocks_summary["gamma_ref"]["n_non_finite"] + blocks_summary["gamma"]["n_non_finite"],
        "n_cholesky_failed": blocks_summary["gamma_ref"]["n_cholesky_failed"]
        + blocks_summary["gamma"]["n_cholesky_failed"],
        "jacobian_closure_passed": closure["passed"],
    }
    gate["passed"] = (
        gate["n_non_finite"] == 0 and gate["n_cholesky_failed"] == 0 and gate["jacobian_closure_passed"]
    )
    result = {
        "preregistration": PREREGISTRATION,
        "scope": "diagnostic_only; must not inform solver tuning",
        "selection": batches.selection_stats,
        "seed_non_stationary_row_fraction": float(seed_stats["non_stationary_row_fraction"]),
        "refit": {key: refit_stats[key] for key in ("chi2_per_component", "n_components", "invalid_fraction")},
        "refit_non_stationary_row_fraction": float(
            refit_stats["stationarity_gate"]["non_stationary_row_fraction"]
        ),
        "linearisation_point": "input geometry, zero alignment, ref-only refit",
        "track_blocks": blocks_summary,
        "jacobian_closure": {**closure, "sample_seed": CLOSURE_SAMPLE_SEED},
        "gate": gate,
    }
    table.to_csv(output_dir / "tracks.csv.gz", index=False)
    (output_dir / "track_block_conditioning.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    command_args = list(sys.argv[1:] if argv is None else argv)

    def interrupt(signum, _frame):
        raise ExperimentInterrupted(f"received signal {signum}")

    old_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    for signum in old_handlers:
        signal.signal(signum, interrupt)
    try:
        write_experiment_manifest(
            output_dir,
            run_id=args.run_id,
            command=[sys.executable, str(Path(__file__).resolve()), *command_args],
            config=load_project_config(args.config),
            inputs={
                "geometry": args.input_geom,
                "nominal_geometry": args.nominal_full_geom,
                "states": args.states,
                "summary": args.summary,
                "baseline_manifest": args.baseline_manifest,
            },
            outputs={
                "track_block_conditioning": output_dir / "track_block_conditioning.json",
                "tracks": output_dir / "tracks.csv.gz",
            },
            metrics={"scope": "diagnostic_only; must not inform solver tuning"},
            capture_source_snapshot=args.capture_source_snapshot,
        )
        result = run(args)
        finalize_experiment_manifest(manifest_path, status="completed", metrics={"gate": result["gate"]})
        print(json.dumps(result["gate"], indent=2))
        return 0
    except BaseException as error:
        if manifest_path.is_file():
            finalize_experiment_manifest(
                manifest_path,
                status="failed",
                metrics={"error_type": type(error).__name__, "error_message": str(error)},
            )
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
