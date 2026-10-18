"""Free-mode orchestration shared by stages B and C: initialize, solve, report."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from alignment_diagnostics import (
    print_per_layer_residual_diagnostics,
    print_target_residual_stats,
    validate_against_truth,
)
from .geometry_io import export_joint_geometry, load_joint_geometry
from .helix.point_fit import require_stationary_helix_point_fit
from .helix_init import (
    fit_helices_to_measurements,
    seed_helices,
    smt_seed_statistics,
)
from .joint_residuals import (
    _joint_forward,
    combine_component_masks,
    summarize_joint_layer_chi2,
)
from .root_inputs import load_variable_information
from .stage_ntr_helix import run_block_lm_alignment
from .track_batches import build_track_batches

JOINT_SOLVER = "block-lm"
JOINT_SOLVERS = ("block-lm", "schur-lm")


def select_joint_solver(config):
    """The free-mode geometry solver named by ``config.joint_solver``."""
    if config.joint_solver == "block-lm":
        return run_block_lm_alignment
    if config.joint_solver == "schur-lm":
        from joint_alignment_schur import run_schur_lm_alignment

        return run_schur_lm_alignment
    raise ValueError(
        f"Unknown free-mode geometry solver {config.joint_solver!r}; "
        f"expected one of {JOINT_SOLVERS}."
    )


def run_free_alignment_pipeline(config, target_layers, active_dof):
    if not os.path.exists(config.nominal_full_geom):
        raise FileNotFoundError(
            f"Full nominal geometry '{config.nominal_full_geom}' is required for free mode."
        )
    geometry = load_joint_geometry(
        config.nominal_full_geom, config.input_geom, target_layers
    )
    print(
        f"Joint working geometry: {len(geometry.transforms)} surfaces, "
        f"{int(geometry.target_mask.sum())} trainable targets, "
        f"{len(geometry.input_indices)} overlaid input surfaces."
    )
    inputs = load_variable_information(
        config.summary_root,
        config.states_root,
        geometry.id_to_idx,
        max_track_entries=(config.max_tracks * 4 if config.max_tracks > 0 else None),
        load_summary=False,
    )
    print_target_residual_stats(
        inputs.initial_residuals, inputs.hit_volume_ids, inputs.hit_layer_ids, target_layers,
        loc1_valid=inputs.loc1_valid,
    )
    batches = build_track_batches(
        inputs,
        target_layers,
        min_hits=config.joint_seed_min_hits,
        min_span=config.joint_seed_min_span,
        max_tracks=config.max_tracks,
    )
    print(
        f"Joint track selection: {len(batches.event_track_ids)} tracks, "
        f"{int(batches.hit_mask.sum())} hits, padded width {batches.hit_mask.shape[1]}."
    )
    seeded_parameters, seed_solver_stats = seed_helices(
        batches,
        n_iter=config.joint_smt_seed_iters,
        device="cuda" if config.gpu and torch.cuda.is_available() else "cpu",
        dtype=torch.float32 if config.gpu and torch.cuda.is_available() else torch.float64,
        return_diagnostics=True,
    )
    smt_stats = {
        **smt_seed_statistics(batches, seeded_parameters),
        "solver": seed_solver_stats["solver"],
        "convergence": {
            key: value for key, value in seed_solver_stats.items() if key != "solver"
        },
    }
    require_stationary_helix_point_fit(seed_solver_stats)
    if smt_stats["invalid_fraction"] > config.joint_max_invalid_fraction:
        raise ValueError(
            "SMT helix seed intersection failure rate exceeds limit: "
            f"{smt_stats['invalid_fraction']:.2%} > "
            f"{config.joint_max_invalid_fraction:.2%}."
        )
    refitted_parameters, measurement_stats = fit_helices_to_measurements(
        batches,
        geometry,
        seeded_parameters,
        device=seeded_parameters.device,
        dtype=seeded_parameters.dtype,
        max_invalid_fraction=config.joint_max_invalid_fraction,
        stationarity=config.joint_refit_stationarity,
    )
    print(
        "Joint initialization: "
        f"tracks={batches.selection_stats.get('input_tracks', len(batches.event_track_ids))} "
        f"retained={batches.selection_stats.get('retained_tracks', len(batches.event_track_ids))} "
        f"excluded={batches.selection_stats.get('excluded_tracks', 0)} "
        f"SMT loss={smt_stats['loss']:.6g} "
        f"refit solver={measurement_stats['solver']} "
        f"measurement chi2/component={measurement_stats['chi2_per_component']:.6g} "
        f"n_components={measurement_stats['n_components']}."
    )
    if "stationarity" in measurement_stats:
        seed_report = measurement_stats["stationarity"]["seed"]
        refit_report = measurement_stats["stationarity"]["refit"]
        print(
            "Joint refit stationarity (F = 0.5 chi2 gauge, threshold "
            f"{refit_report['stationary_threshold']:.3g}): "
            f"non-stationary rows {seed_report['non_stationary_row_fraction']:.2%} "
            f"-> {refit_report['non_stationary_row_fraction']:.2%}, "
            f"decrement p90 {seed_report['newton_decrement_p90']:.3g} "
            f"-> {refit_report['newton_decrement_p90']:.3g}, "
            f"non-finite {refit_report['n_non_finite']}."
        )
    with torch.no_grad():
        seed_alignment = torch.zeros(
            (len(geometry.centers), 6),
            device=seeded_parameters.device,
            dtype=seeded_parameters.dtype,
        )
        seed_residuals, seed_covariance, seed_component_mask, _seed_valid = (
            _joint_forward(seeded_parameters, seed_alignment, batches, geometry)
        )
    seed_layer_chi2 = summarize_joint_layer_chi2(
        seed_residuals,
        seed_covariance,
        seed_component_mask,
        batches.volume_id,
        batches.layer_id,
        target_layers,
    )
    solver = select_joint_solver(config)
    result = solver(
        batches,
        geometry,
        config,
        active_dof=active_dof,
        initial_track_params=refitted_parameters,
    )
    predicted_deltas = result["predicted_deltas"]
    truth_metrics = validate_against_truth(
        geometry.transforms,
        predicted_deltas,
        config.nominal_file,
        target_layers,
    )
    Path(config.output_geom).parent.mkdir(parents=True, exist_ok=True)
    export_joint_geometry(geometry, predicted_deltas, config.output_geom)
    print(f"Export successful: {config.output_geom}")

    if config.make_plot and result["history"]:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(result["history"]) + 1), result["history"], color="firebrick")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Joint chi2 / valid component")
        plt.title("Free track + geometry joint optimization")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plot_path = Path(config.output_geom).parent / f"loss_curve_{Path(config.output_geom).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Training convergence plot saved: '{plot_path}'")

    initial_residuals = result["initial_residuals"].numpy()
    final_residuals = result["final_residuals"].numpy()
    hit_valid, component_mask, initial_only, final_only = combine_component_masks(
        batches.hit_mask,
        result["initial_component_mask"],
        result["final_component_mask"],
    )
    print(f"Diagnostics hit selection: initial_only={initial_only} "
          f"final_only={final_only} (excluded from the before/after comparison)")
    valid = hit_valid.numpy()
    target_metrics = print_per_layer_residual_diagnostics(
        initial_residuals[valid], final_residuals[valid],
        batches.volume_id.numpy()[valid], batches.layer_id.numpy()[valid],
        target_layers, return_metrics=True, component_mask=component_mask,
    )
    stage_layer_chi2 = {
        "smt_seed": seed_layer_chi2,
        "measurement_refit": summarize_joint_layer_chi2(
            result["initial_residuals"],
            batches.covariances,
            result["initial_component_mask"],
            batches.volume_id,
            batches.layer_id,
            target_layers,
        ),
        "joint": summarize_joint_layer_chi2(
            result["final_residuals"],
            batches.covariances,
            result["final_component_mask"],
            batches.volume_id,
            batches.layer_id,
            target_layers,
        ),
    }

    result.update({
        "output_geom": config.output_geom,
        "truth_metrics": truth_metrics,
        "n_tracks": len(batches.event_track_ids),
        "n_hits": int(batches.hit_mask.sum()),
        "target_metrics": target_metrics,
        "initialization_stats": {
            **batches.selection_stats,
            "fit_hits": int(batches.fit_mask.sum()),
            "min_fit_span": float(batches.fit_span.min()),
            "smt": smt_stats,
            "measurement_refit": measurement_stats,
        },
        "stage_layer_chi2": stage_layer_chi2,
    })
    return result
