"""Schur P1: split target prediction error into measurement, ref-track and model terms.

plan.md §4 Q1 / Phase P1, pre-registered in ``ml_optimization_log.md``
Iteration 33.  Each target measurement component k is predicted by the
ref-only (non-target) production refit, and its residual variance is split as

    E[r_k^2] = V_k + P_k + U,    P_k = (B Gamma_ref^-1 B^T)_kk

where ``Gamma_ref`` is the refit's own Gauss-Newton information in the
5-parameter gauge frame and ``B`` the target-row Jacobian.  ``U`` is whatever
neither the measurement variance nor the propagated track uncertainty explains.

Diagnostic only: nothing here feeds a solver, and truth is never read.  The
target-row Jacobian is built by constructing ``MeasurementFitInputs`` directly;
the production ``_non_target_fit_mask`` guard is left untouched.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import ml_alignment_autograd as M  # noqa: E402
from experiment_manifest import (  # noqa: E402
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from helix_model import (  # noqa: E402
    MeasurementFitInputs,
    measurement_local_prediction,
    measurement_residual_jacobian,
    pack_gauge_fixed_track_parameters,
    reference_gauge_frame,
    require_stationary_helix_point_fit,
)
from project_config import add_config_argument, load_project_config  # noqa: E402
from validate_helix_model import (  # noqa: E402
    MAX_RANK_DEFICIENT_ROW_FRACTION,
    PRODUCTION_DAMPING,
    predictive_covariance,
)

PREREGISTRATION = "ml_optimization_log.md Iteration 33"
#: Decision thresholds, fixed before any P1 data were read (Iteration 33 §D).
STAT_IMPORTANT_RATIO = 0.10
MODEL_DOMINANCE_MIN_FRACTION_OF_V = 0.10
#: Identity of the P0.1 baseline row set (``schur-p0-fixedtrack-20260922``).
EXPECTED_RETAINED_TRACKS = 61_482
EXPECTED_REFIT_COMPONENTS = 1_154_761
#: Inputs whose bytes must equal the P0.1 baseline manifest.
BASELINE_INPUT_KEYS = ("geometry", "states", "nominal_geometry")

REGIMES = ("interpolation", "forward_extrapolation", "backward_extrapolation")
COMPONENTS = ("LOC0", "LOC1")


class ExperimentInterrupted(RuntimeError):
    pass


def prediction_regime(fit_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Code each target hit by where the fit hits sit in the batch hit order.

    ``build_track_batches`` orders hits by transverse radius then |z|.  Returns
    0/1/2 for ``REGIMES`` and -1 for non-target or padded slots.
    """
    fit = fit_mask.to(torch.int64)
    before = (torch.cumsum(fit, dim=1) - fit) > 0
    after = (torch.flip(torch.cumsum(torch.flip(fit, (1,)), dim=1), (1,)) - fit) > 0
    regime = torch.full(fit_mask.shape, -1, dtype=torch.int64)
    regime[target_mask & before & after] = 0
    regime[target_mask & before & ~after] = 1
    regime[target_mask & ~before & after] = 2
    if bool((target_mask & ~before & ~after).any()):
        raise ValueError("A target hit has no fit hit on either side.")
    return regime


def _fit_inputs(batches, geometry, rows: slice) -> MeasurementFitInputs:
    sensor_idx = batches.sensor_idx[rows]
    hit_mask = batches.hit_mask[rows]
    component_mask = torch.stack((hit_mask, hit_mask & batches.loc1_mask[rows]), dim=-1)
    return MeasurementFitInputs(
        measured=batches.measured_local[rows].to(torch.float64),
        variance=batches.covariances[rows].to(torch.float64),
        component_mask=component_mask,
        centers=geometry.centers.to(torch.float64)[sensor_idx],
        rotations=geometry.rotations.to(torch.float64)[sensor_idx],
    )


def component_budget(
    parameters: torch.Tensor,
    batches,
    budget_geometry,
    comparison_geometry,
    chunk_rows: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per target component: V, V_raw, P and residuals on both target geometries.

    ``budget_geometry`` supplies the Jacobians and the primary residual;
    ``comparison_geometry`` only an extra residual column.  Non-target surfaces
    must be identical in both, which is what lets one ref-only refit serve both.
    """
    if batches.raw_covariances is None:
        raise ValueError("P1 needs TrackBatches.raw_covariances for the floor split.")
    non_target = ~budget_geometry.target_mask
    if not (
        torch.equal(budget_geometry.centers[non_target], comparison_geometry.centers[non_target])
        and torch.equal(
            budget_geometry.rotations[non_target], comparison_geometry.rotations[non_target]
        )
    ):
        raise ValueError("The two geometries differ on non-target surfaces.")
    fit_mask = M._non_target_fit_mask(batches)
    regime_all = prediction_regime(fit_mask, batches.is_target & batches.hit_mask)
    parameters = parameters.detach().to(device="cpu", dtype=torch.float64)

    frames = []
    n_rank_deficient = 0
    n_component_dropped_invalid = 0
    condition_numbers = []
    for start in range(0, parameters.shape[0], chunk_rows):
        rows = slice(start, min(start + chunk_rows, parameters.shape[0]))
        params = parameters[rows]
        inputs = _fit_inputs(batches, budget_geometry, rows)
        comparison_inputs = _fit_inputs(batches, comparison_geometry, rows)
        frame = reference_gauge_frame(params)
        theta = pack_gauge_fixed_track_parameters(params)
        # Whitened: rows are (m - y) / sqrt(V), so J^T J is the GN information.
        jacobian = measurement_residual_jacobian(theta, frame, inputs)
        local, valid = measurement_local_prediction(params, inputs)
        comparison_local, comparison_valid = measurement_local_prediction(
            params, comparison_inputs
        )
        components = inputs.component_mask
        ref_rows = components & (fit_mask[rows] & valid).unsqueeze(-1)
        target = batches.is_target[rows] & batches.hit_mask[rows]
        target_any = components & target.unsqueeze(-1)
        target_rows = target_any & (valid & comparison_valid).unsqueeze(-1)
        n_component_dropped_invalid += int((target_any & ~target_rows).sum())

        ref_jacobian = jacobian * ref_rows.flatten(1).unsqueeze(-1)
        gamma_ref = ref_jacobian.transpose(-1, -2) @ ref_jacobian
        target_jacobian = jacobian * target_rows.flatten(1).unsqueeze(-1)
        covariance, diagnostics = predictive_covariance(
            gamma_ref, target_jacobian, damping=PRODUCTION_DAMPING
        )
        n_rank_deficient += diagnostics["n_rank_deficient_rows"]
        eigenvalues = torch.linalg.eigvalsh(gamma_ref)
        smallest = eigenvalues[:, 0]
        condition_numbers.append(
            torch.where(
                smallest > 0, eigenvalues[:, -1] / smallest, torch.full_like(smallest, np.inf)
            ).numpy()
        )
        ratio = covariance.diagonal(dim1=-2, dim2=-1).reshape(components.shape)

        variance = inputs.variance
        raw_variance = batches.raw_covariances[rows].to(torch.float64)
        track_index, hit_index, component = torch.nonzero(target_rows, as_tuple=True)
        selector = (track_index, hit_index, component)
        sensor = batches.sensor_idx[rows][track_index, hit_index]
        frames.append(
            pd.DataFrame(
                {
                    "track": (track_index + start).numpy(),
                    "event_id": batches.event_track_ids[rows][track_index, 0].numpy(),
                    "track_id": batches.event_track_ids[rows][track_index, 1].numpy(),
                    "sensor": sensor.numpy(),
                    "layer": [
                        f"{int(v)}:{int(l)}"
                        for v, l in zip(
                            batches.volume_id[rows][track_index, hit_index],
                            batches.layer_id[rows][track_index, hit_index],
                        )
                    ],
                    "component": np.asarray(COMPONENTS)[component.numpy()],
                    "regime": np.asarray(REGIMES)[
                        regime_all[rows][track_index, hit_index].numpy()
                    ],
                    "floor_active": (raw_variance[selector] < variance[selector]).numpy(),
                    "V": variance[selector].numpy(),
                    "V_raw": raw_variance[selector].numpy(),
                    "P": (ratio[selector] * variance[selector]).numpy(),
                    "r_budget": (inputs.measured - local)[selector].numpy(),
                    "r_comparison": (comparison_inputs.measured - comparison_local)[
                        selector
                    ].numpy(),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    # predictive_covariance returns NaN for rank-deficient rows; they have no
    # posterior and are dropped (and counted), never filled.
    undefined = ~np.isfinite(table["P"].to_numpy())
    table = table[~undefined].reset_index(drop=True)
    if not np.isfinite(table[["V", "V_raw", "r_budget", "r_comparison"]].to_numpy()).all():
        raise FloatingPointError("Non-finite variance or residual in the P1 table.")
    condition = np.concatenate(condition_numbers)
    diagnostics = {
        "n_tracks": int(parameters.shape[0]),
        "n_rank_deficient_rows": int(n_rank_deficient),
        "rank_deficient_row_fraction": n_rank_deficient / max(int(parameters.shape[0]), 1),
        "n_target_components_dropped_rank_deficient": int(undefined.sum()),
        "n_target_components_dropped_invalid_intersection": n_component_dropped_invalid,
        "gamma_ref_condition_p50": float(np.quantile(condition, 0.5)),
        "gamma_ref_condition_p99": float(np.quantile(condition, 0.99)),
        "gamma_ref_condition_max": float(condition.max()),
    }
    return table, diagnostics


def _summarize(group: pd.DataFrame) -> dict[str, Any]:
    v_mean = float(group["V"].mean())
    p_mean = float(group["P"].mean())
    m_budget = float(np.square(group["r_budget"]).mean())
    return {
        "n_components": int(len(group)),
        "n_tracks": int(group["track"].nunique()),
        "V_mean": v_mean,
        "V_raw_mean": float(group["V_raw"].mean()),
        "P_mean": p_mean,
        "R_stat": float(group["P"].sum() / group["V"].sum()),
        "R_stat_raw": float(group["P"].sum() / group["V_raw"].sum()),
        "P_over_V_median": float((group["P"] / group["V"]).median()),
        "M_budget": m_budget,
        "M_comparison": float(np.square(group["r_comparison"]).mean()),
        "U": m_budget - v_mean - p_mean,
        "pull2_with_P": float((np.square(group["r_budget"]) / (group["V"] + group["P"])).mean()),
        "pull2_without_P": float((np.square(group["r_budget"]) / group["V"]).mean()),
        "components_per_sensor_median": float(group.groupby("sensor").size().median()),
    }


def aggregate(table: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Return (primary cells per layer×component, strata per cell×regime×floor)."""
    cells = [
        {"layer": layer, "component": component, **_summarize(group)}
        for (layer, component), group in table.groupby(["layer", "component"], sort=True)
    ]
    strata = [
        {
            "layer": layer,
            "component": component,
            "regime": regime,
            "floor_active": bool(floor),
            **_summarize(group),
        }
        for (layer, component, regime, floor), group in table.groupby(
            ["layer", "component", "regime", "floor_active"], sort=True
        )
    ]
    return cells, strata


def classify(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the Iteration 33 §D decision rule to the primary cells."""
    labelled = []
    for cell in cells:
        statistical = cell["R_stat"] >= STAT_IMPORTANT_RATIO
        model = (
            cell["U"] > cell["P_mean"]
            and cell["U"] > MODEL_DOMINANCE_MIN_FRACTION_OF_V * cell["V_mean"]
        )
        labelled.append(
            {
                "layer": cell["layer"],
                "component": cell["component"],
                "statistical_important": bool(statistical),
                "model_dominated": bool(model),
            }
        )
    if any(c["statistical_important"] and not c["model_dominated"] for c in labelled):
        outcome = "outcome_1_statistical_important"
    elif any(c["model_dominated"] for c in labelled):
        outcome = "outcome_2_model_error_dominates"
    else:
        outcome = "outcome_3_both_small"
    return {"cells": labelled, "outcome": outcome}


def identity_gate(
    retained_tracks: int,
    refit_components: int,
    full_row_set: bool,
    manifest_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    baseline_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Fail closed unless the row set is the P0.1 baseline's; subsets are flagged."""
    if manifest_inputs is not None and baseline_inputs is not None:
        for key in BASELINE_INPUT_KEYS:
            if manifest_inputs[key]["sha256"] != baseline_inputs[key]["sha256"]:
                raise ValueError(f"Input {key!r} differs from the P0.1 baseline.")
    if not full_row_set:
        return "not_applicable_subset"
    if retained_tracks != EXPECTED_RETAINED_TRACKS:
        raise ValueError(
            f"Retained tracks {retained_tracks} != baseline {EXPECTED_RETAINED_TRACKS}."
        )
    if refit_components != EXPECTED_REFIT_COMPONENTS:
        raise ValueError(
            f"Refit components {refit_components} != baseline {EXPECTED_REFIT_COMPONENTS}."
        )
    return "passed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_argument(parser)
    parser.add_argument("--input-geom", required=True)
    parser.add_argument("--aligned-geom", required=True)
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
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=1000)
    parser.add_argument("--capture-source-snapshot", action="store_true")
    return parser


def run(args) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    target_layers = M.parse_target_layers(args.target_layers)
    input_geometry = M.load_joint_geometry(args.nominal_full_geom, args.input_geom, target_layers)
    aligned_geometry = M.load_joint_geometry(
        args.nominal_full_geom, args.aligned_geom, target_layers
    )
    inputs = M.load_variable_information(
        args.summary,
        args.states,
        input_geometry.id_to_idx,
        max_track_entries=(args.max_tracks * 4 if args.max_tracks > 0 else None),
        load_summary=False,
    )
    batches = M.build_track_batches(
        inputs,
        target_layers,
        min_hits=args.min_hits,
        min_span=args.min_span,
        max_tracks=args.max_tracks,
    )
    del inputs
    seeded, seed_stats = M.seed_helices(
        batches, n_iter=args.smt_seed_iters, dtype=torch.float64, return_diagnostics=True
    )
    require_stationary_helix_point_fit(seed_stats)
    # The refit reads only non-target measurements and surfaces, so the input
    # and the aligned geometry give the same tracks.
    refitted, refit_stats = M.fit_helices_to_measurements(
        batches, input_geometry, seeded, dtype=torch.float64, stationarity=False
    )
    baseline = json.loads(Path(args.baseline_manifest).read_text(encoding="utf-8"))
    own = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    gate = identity_gate(
        batches.selection_stats["retained_tracks"],
        refit_stats["n_components"],
        full_row_set=args.max_tracks == 0,
        manifest_inputs=own["inputs"],
        baseline_inputs=baseline["inputs"],
    )
    table, diagnostics = component_budget(
        refitted, batches, aligned_geometry, input_geometry, chunk_rows=args.chunk_rows
    )
    cells, strata = aggregate(table)
    verdict = classify(cells)
    if diagnostics["rank_deficient_row_fraction"] > MAX_RANK_DEFICIENT_ROW_FRACTION:
        verdict["outcome"] = "inconclusive_rank_deficient"
    if gate != "passed":
        verdict["outcome"] = f"inconclusive_{gate}"
    result = {
        "preregistration": PREREGISTRATION,
        "scope": "diagnostic_only; must not inform solver tuning",
        "identity_gate": gate,
        "selection": batches.selection_stats,
        "seed_non_stationary_row_fraction": float(seed_stats["non_stationary_row_fraction"]),
        "refit": {
            key: refit_stats[key]
            for key in ("chi2_per_component", "n_components", "invalid_fraction")
        },
        "refit_non_stationary_row_fraction": float(
            refit_stats["stationarity_gate"]["non_stationary_row_fraction"]
        ),
        "budget_geometry": "aligned",
        "comparison_geometry": "input",
        "diagnostics": diagnostics,
        "thresholds": {
            "stat_important_ratio": STAT_IMPORTANT_RATIO,
            "model_dominance_min_fraction_of_V": MODEL_DOMINANCE_MIN_FRACTION_OF_V,
        },
        "cells": cells,
        "verdict": verdict,
    }
    table.to_csv(output_dir / "components.csv.gz", index=False)
    pd.DataFrame(strata).to_csv(output_dir / "strata.csv", index=False)
    (output_dir / "error_budget.json").write_text(
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
    outputs = {
        "error_budget": output_dir / "error_budget.json",
        "strata": output_dir / "strata.csv",
        "components": output_dir / "components.csv.gz",
    }
    try:
        write_experiment_manifest(
            output_dir,
            run_id=args.run_id,
            command=[sys.executable, str(Path(__file__).resolve()), *command_args],
            config=load_project_config(args.config),
            inputs={
                "geometry": args.input_geom,
                "aligned_geometry": args.aligned_geom,
                "nominal_geometry": args.nominal_full_geom,
                "states": args.states,
                "summary": args.summary,
                "baseline_manifest": args.baseline_manifest,
            },
            outputs=outputs,
            metrics={"scope": "diagnostic_only; must not inform solver tuning"},
            capture_source_snapshot=args.capture_source_snapshot,
        )
        result = run(args)
        finalize_experiment_manifest(
            manifest_path,
            status="completed",
            metrics={"identity_gate": result["identity_gate"], "verdict": result["verdict"]},
        )
        print(json.dumps(result["verdict"], indent=2))
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
