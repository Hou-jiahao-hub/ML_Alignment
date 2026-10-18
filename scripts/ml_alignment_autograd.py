"""Command-line entry point of the ML alignment solvers, and their import facade.

The implementation lives in ``mlalign``, split along the three key optimisations:

* ``mlalign.stage_surface`` -- surface mode, frozen KF-smoothed tracks;
* ``mlalign.stage_ntr_helix`` -- free mode, non-target-refitted helices and
  fixed-track block-LM, orchestrated by ``mlalign.free_helix_pipeline``;
* ``joint_alignment_schur`` -- free mode, joint Schur-reduced LM.

Every name this module exported before the split is re-exported below so
existing imports keep working.  Monkeypatching one of them here no longer
reaches the code that calls it: patch the defining ``mlalign`` module instead.
"""

from __future__ import annotations

import os

import uproot  # noqa: F401

from alignment_artifact_layout import (  # noqa: F401
    alignment_json_path,
    ACTIVE_DOF as CONFIG_ACTIVE_DOF,
    data_path,
    reconstruction_root_path,
    TARGET_LAYERS,
    target_layers_spec,
)
from alignment_diagnostics import (  # noqa: F401
    print_per_layer_residual_diagnostics,
    print_target_residual_stats,
    validate_against_truth,
)
from project_config import (  # noqa: F401
    absolute_project_path,
    add_config_argument,
)
try:  # Imported as part of the ``scripts`` package.
    from .mlalign.block_lm_contract import (  # noqa: F401
        _block_lm_block_status,
        _block_lm_finalize_record,
        _block_lm_new_record,
        _block_lm_run_summary,
        BLOCK_LM_DECREMENT_STATIONARY,
        BLOCK_LM_MAX_DAMPING_RETRIES,
        BLOCK_LM_PREREGISTRATION_VERSION,
        BLOCK_LM_STATUS_BUDGET_EXHAUSTED,
        BLOCK_LM_STATUS_CONVERGED,
        BLOCK_LM_STATUS_INTERSECTION_FAILED,
        BLOCK_LM_STATUS_NO_DATA,
        BLOCK_LM_STATUS_RANK_DEFICIENT,
        BLOCK_LM_TERMINAL_STATUSES,
    )
    from .mlalign.box_constraints import (  # noqa: F401
        _block_lm_free_dof_mask,
        BLOCK_LM_BOUND_ATOL,
        JOINT_CLAMP_R,
        JOINT_CLAMP_T,
    )
    from .mlalign.conventions import (  # noqa: F401
        DEFAULT_ACTIVE_DOF,
        DEFAULT_TARGET_LAYERS,
        EPS_COVARIANCE,
        parse_active_dof,
        parse_target_layers,
        resolve_active_dof,
    )
    from .mlalign.free_helix_pipeline import (  # noqa: F401
        JOINT_SOLVER,
        JOINT_SOLVERS,
        run_free_alignment_pipeline,
        select_joint_solver,
    )
    from .mlalign.geometry_io import (  # noqa: F401
        _surface_key,
        build_target_mask,
        export_joint_geometry,
        FULL_MATRIX_TOLERANCE,
        JointGeometry,
        load_geometry_from_json,
        load_joint_geometry,
        update_geometry_to_json,
    )
    from .mlalign.helix.gauge import (  # noqa: F401
        assemble_track_parameters as assemble_joint_track_parameters,
        CURVATURE_SCALE,
        split_track_parameters as split_joint_track_parameters,
    )
    from .mlalign.helix.measurement_fit import (  # noqa: F401
        fit_helix_measurements_gauss_newton,
        measurement_newton_decrement,
        measurement_stationarity_report,
        MeasurementFitInputs,
    )
    from .mlalign.helix.point_fit import (  # noqa: F401
        fit_helix_points_gauss_newton,
        require_stationary_helix_point_fit,
    )
    from .mlalign.helix.trajectory import (  # noqa: F401
        accept_newton_candidate,
        closest_helix_points,
        helix_point,
        helix_tangent,
        intersect_plane,
        S_MAX,
    )
    from .mlalign.helix_init import (  # noqa: F401
        _measurement_fit_inputs_from_joint,
        _measurement_refit_objective,
        _refit_helices_gauss_newton,
        fit_helices_to_measurements,
        JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION,
        JOINT_REFIT_PREREGISTRATION_VERSION,
        JOINT_REFIT_SOLVER_GAUSS_NEWTON,
        JOINT_SMT_SEED_ITERS,
        measure_joint_refit_stationarity,
        seed_helices,
        smt_seed_statistics,
    )
    from .mlalign.joint_residuals import (  # noqa: F401
        _joint_forward,
        _sensor_residual_data,
        _target_alignment_evaluation,
        combine_component_masks,
        summarize_joint_layer_chi2,
    )
    from .mlalign.reporting import (  # noqa: F401
        _json_ready,
        write_alignment_summary,
    )
    from .mlalign.root_inputs import (  # noqa: F401
        _flatten_root_entries,
        _read_root_columns,
        _root_column_entries,
        AlignmentInputs,
        EPS_WEIGHT,
        load_variable_information,
        STRIP_VOLUMES,
        STRIP_Y_FLOOR,
    )
    from .mlalign.rotations import (  # noqa: F401
        apply_alignment_torch,
        euler_to_rotation_matrix,
        rotation_matrix_to_euler,
        torch_euler_to_rotation_matrix,
    )
    from .mlalign.run_config import (  # noqa: F401
        AlignmentRunConfig,
        JOINT_SEED_MODE,
    )
    from .mlalign.stage_ntr_helix import (  # noqa: F401
        _block_lm_decrement,
        _block_lm_fixed_chi2,
        _block_lm_linearized_block,
        _block_lm_retry_damping,
        BLOCK_LM_MAX_DAMPING,
        BLOCK_LM_MIN_DAMPING,
        FISHER_EIGEN_FLOOR,
        LM_INITIAL_DAMPING,
        LM_MAX_ITERS,
        LM_MAX_STEP_R,
        LM_MAX_STEP_T,
        regularize_fisher_matrix,
        run_block_lm_alignment,
    )
    from .mlalign.stage_surface import (  # noqa: F401
        ADAM_CLAMP_R,
        ADAM_CLAMP_T,
        AlignmentEngine,
        clamp_parameter_and_reset_adam,
        compute_autograd_residuals,
        DEFAULT_ADAM_ETA_MIN,
        DEFAULT_ADAM_LR,
        DEFAULT_EPOCHS,
        DEFAULT_LAMBDA_L2,
        EPS_DIRECTION_NORM,
        physics_informed_residual_loss,
        project_track_to_aligned_surface,
        run_surface_alignment,
    )
    from .mlalign.track_batches import (  # noqa: F401
        _non_target_fit_mask,
        build_track_batches,
        JOINT_SEED_MIN_HITS,
        JOINT_SEED_MIN_SPAN,
        TrackBatches,
    )
except ImportError:  # Direct script execution.
    from mlalign.block_lm_contract import (  # noqa: F401
        _block_lm_block_status,
        _block_lm_finalize_record,
        _block_lm_new_record,
        _block_lm_run_summary,
        BLOCK_LM_DECREMENT_STATIONARY,
        BLOCK_LM_MAX_DAMPING_RETRIES,
        BLOCK_LM_PREREGISTRATION_VERSION,
        BLOCK_LM_STATUS_BUDGET_EXHAUSTED,
        BLOCK_LM_STATUS_CONVERGED,
        BLOCK_LM_STATUS_INTERSECTION_FAILED,
        BLOCK_LM_STATUS_NO_DATA,
        BLOCK_LM_STATUS_RANK_DEFICIENT,
        BLOCK_LM_TERMINAL_STATUSES,
    )
    from mlalign.box_constraints import (  # noqa: F401
        _block_lm_free_dof_mask,
        BLOCK_LM_BOUND_ATOL,
        JOINT_CLAMP_R,
        JOINT_CLAMP_T,
    )
    from mlalign.conventions import (  # noqa: F401
        DEFAULT_ACTIVE_DOF,
        DEFAULT_TARGET_LAYERS,
        EPS_COVARIANCE,
        parse_active_dof,
        parse_target_layers,
        resolve_active_dof,
    )
    from mlalign.free_helix_pipeline import (  # noqa: F401
        JOINT_SOLVER,
        JOINT_SOLVERS,
        run_free_alignment_pipeline,
        select_joint_solver,
    )
    from mlalign.geometry_io import (  # noqa: F401
        _surface_key,
        build_target_mask,
        export_joint_geometry,
        FULL_MATRIX_TOLERANCE,
        JointGeometry,
        load_geometry_from_json,
        load_joint_geometry,
        update_geometry_to_json,
    )
    from mlalign.helix.gauge import (  # noqa: F401
        assemble_track_parameters as assemble_joint_track_parameters,
        CURVATURE_SCALE,
        split_track_parameters as split_joint_track_parameters,
    )
    from mlalign.helix.measurement_fit import (  # noqa: F401
        fit_helix_measurements_gauss_newton,
        measurement_newton_decrement,
        measurement_stationarity_report,
        MeasurementFitInputs,
    )
    from mlalign.helix.point_fit import (  # noqa: F401
        fit_helix_points_gauss_newton,
        require_stationary_helix_point_fit,
    )
    from mlalign.helix.trajectory import (  # noqa: F401
        accept_newton_candidate,
        closest_helix_points,
        helix_point,
        helix_tangent,
        intersect_plane,
        S_MAX,
    )
    from mlalign.helix_init import (  # noqa: F401
        _measurement_fit_inputs_from_joint,
        _measurement_refit_objective,
        _refit_helices_gauss_newton,
        fit_helices_to_measurements,
        JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION,
        JOINT_REFIT_PREREGISTRATION_VERSION,
        JOINT_REFIT_SOLVER_GAUSS_NEWTON,
        JOINT_SMT_SEED_ITERS,
        measure_joint_refit_stationarity,
        seed_helices,
        smt_seed_statistics,
    )
    from mlalign.joint_residuals import (  # noqa: F401
        _joint_forward,
        _sensor_residual_data,
        _target_alignment_evaluation,
        combine_component_masks,
        summarize_joint_layer_chi2,
    )
    from mlalign.reporting import (  # noqa: F401
        _json_ready,
        write_alignment_summary,
    )
    from mlalign.root_inputs import (  # noqa: F401
        _flatten_root_entries,
        _read_root_columns,
        _root_column_entries,
        AlignmentInputs,
        EPS_WEIGHT,
        load_variable_information,
        STRIP_VOLUMES,
        STRIP_Y_FLOOR,
    )
    from mlalign.rotations import (  # noqa: F401
        apply_alignment_torch,
        euler_to_rotation_matrix,
        rotation_matrix_to_euler,
        torch_euler_to_rotation_matrix,
    )
    from mlalign.run_config import (  # noqa: F401
        AlignmentRunConfig,
        JOINT_SEED_MODE,
    )
    from mlalign.stage_ntr_helix import (  # noqa: F401
        _block_lm_decrement,
        _block_lm_fixed_chi2,
        _block_lm_linearized_block,
        _block_lm_retry_damping,
        BLOCK_LM_MAX_DAMPING,
        BLOCK_LM_MIN_DAMPING,
        FISHER_EIGEN_FLOOR,
        LM_INITIAL_DAMPING,
        LM_MAX_ITERS,
        LM_MAX_STEP_R,
        LM_MAX_STEP_T,
        regularize_fisher_matrix,
        run_block_lm_alignment,
    )
    from mlalign.stage_surface import (  # noqa: F401
        ADAM_CLAMP_R,
        ADAM_CLAMP_T,
        AlignmentEngine,
        clamp_parameter_and_reset_adam,
        compute_autograd_residuals,
        DEFAULT_ADAM_ETA_MIN,
        DEFAULT_ADAM_LR,
        DEFAULT_EPOCHS,
        DEFAULT_LAMBDA_L2,
        EPS_DIRECTION_NORM,
        physics_informed_residual_loss,
        project_track_to_aligned_surface,
        run_surface_alignment,
    )
    from mlalign.track_batches import (  # noqa: F401
        _non_target_fit_mask,
        build_track_batches,
        JOINT_SEED_MIN_HITS,
        JOINT_SEED_MIN_SPAN,
        TrackBatches,
    )


def run_alignment_pipeline(config=None):
    if config is None:
        config = AlignmentRunConfig()

    if config.track_mode not in {"surface", "free"}:
        raise ValueError("track_mode must be 'surface' or 'free'.")
    ACTIVE_DOF = resolve_active_dof(config)
    DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]
    print(f"Active alignment DOF: {[DOF_NAMES[d] for d in ACTIVE_DOF]}")
    TARGET_LAYERS = parse_target_layers(config.target_layers)

    if config.track_mode == "free":
        for path, label in (
            (config.input_geom, "input geometry"),
            (config.summary_root, "track summary"),
            (config.states_root, "track states"),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {label}: {path}")
        return run_free_alignment_pipeline(config, TARGET_LAYERS, ACTIVE_DOF)
    return run_surface_alignment(config, TARGET_LAYERS, ACTIVE_DOF)


def build_argument_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="ML alignment solver (Adam neural-network optimizer). "
                    "Reads current geometry + reconstruction residuals → writes updated geometry."
    )
    add_config_argument(p)
    p.add_argument("--input-geom", type=absolute_project_path, default=str(alignment_json_path("alignment_output", "misalignment_applied")),
                   help="Current geometry G_k (correction base; delta is composed onto it).")
    p.add_argument("--states", dest="states_root", type=absolute_project_path,
                   default=str(reconstruction_root_path("reconstruction_output", "misaligned", "trackstates_kf")),
                   help="trackstates ROOT from reconstruction performed with G_k.")
    p.add_argument("--summary", dest="summary_root", type=absolute_project_path,
                   default=str(reconstruction_root_path("reconstruction_output", "misaligned", "tracksummary_kf")),
                   help="tracksummary ROOT from reconstruction performed with G_k.")
    p.add_argument("--output-geom", type=absolute_project_path, default=str(alignment_json_path("alignment_output", "aligned_transforms", stem_prefix="ml")),
                   help="Updated geometry G_{k+1} = G_k compose delta.")
    p.add_argument("--nominal", dest="nominal_file", type=absolute_project_path, default=str(data_path("odd_transforms.json")),
                   help="Nominal/truth geometry for the convergence metric (|true|rms).")
    p.add_argument("--iter", dest="iter_label", default=None,
                   help="Iteration label for logging.")
    p.add_argument("--active-dof", default=None,
                   help="Comma-separated active DOF. Default: 0,1,3,4,5 for both track modes.")
    p.add_argument("--target-layers", default=target_layers_spec(),
                   help=f"Comma-separated target layers as volume:layer. Default: {target_layers_spec()}.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"Number of Adam training epochs. Default: {DEFAULT_EPOCHS}.")
    p.add_argument("--lambda-l2", type=float, default=DEFAULT_LAMBDA_L2,
                   help=f"L2 regularisation weight. Default: {DEFAULT_LAMBDA_L2}.")
    p.add_argument("--track-mode", choices=("surface", "free"), default="surface",
                   help="Track treatment: fixed smoothed surface solve or free joint helices.")
    p.add_argument(
        "--joint-solver",
        choices=JOINT_SOLVERS,
        default=JOINT_SOLVER,
        help=(
            "Free-mode geometry solver; block-lm keeps non-target-refitted tracks fixed, "
            "schur-lm fits geometry and tracks jointly (Schur-reduced LM, CPU float64, "
            "no per-DOF step caps)."
        ),
    )
    p.add_argument("--lm-max-iters", type=int, default=LM_MAX_ITERS)
    p.add_argument("--lm-initial-damping", type=float, default=LM_INITIAL_DAMPING)
    p.add_argument("--fisher-eigen-floor", type=float, default=FISHER_EIGEN_FLOOR)
    p.add_argument("--lm-max-step-t", type=float, default=LM_MAX_STEP_T)
    p.add_argument("--lm-max-step-r", type=float, default=LM_MAX_STEP_R)
    p.add_argument(
        "--joint-seed-mode",
        choices=("non-target-measurement",),
        default=JOINT_SEED_MODE,
        help="Free-helix initialization architecture (fixed to non-target measurement).",
    )
    p.add_argument("--joint-seed-min-hits", type=int, default=JOINT_SEED_MIN_HITS)
    p.add_argument("--joint-seed-min-span", type=float, default=JOINT_SEED_MIN_SPAN)
    p.add_argument("--joint-smt-seed-iters", type=int, default=JOINT_SMT_SEED_ITERS)
    p.add_argument(
        "--no-joint-refit-stationarity",
        action="store_true",
        help="Skip the Newton-decrement measurement of the measurement refit.",
    )
    p.add_argument("--min-hits", type=int, default=8)
    p.add_argument("--max-tracks", type=int, default=0, help="0 keeps all selected tracks.")
    p.add_argument("--gpu", action="store_true", help="Use CUDA float32; CPU uses float64.")
    p.add_argument("--joint-clamp-t", type=float, default=JOINT_CLAMP_T)
    p.add_argument("--joint-clamp-r", type=float, default=JOINT_CLAMP_R)
    p.add_argument("--nominal-full-geom", type=absolute_project_path, default=str(data_path("odd_transforms.json")),
                   help="Full nominal geometry used to anchor non-target hits in free mode.")
    p.add_argument(
        "--alignment-gain",
        type=float,
        default=1.0,
        help="Scale geometry corrections when writing output (surface polish uses 0.2).",
    )
    p.add_argument(
        "--output-summary",
        type=absolute_project_path,
        default=None,
        help="Optional compact JSON containing solver stages and metrics.",
    )
    p.add_argument("--no-plot", action="store_true", help="Skip the loss-curve PNG.")
    return p


def alignment_config_from_args(a):
    return AlignmentRunConfig(
        input_geom=a.input_geom,
        states_root=a.states_root,
        summary_root=a.summary_root,
        output_geom=a.output_geom,
        nominal_file=a.nominal_file,
        iter_label=a.iter_label,
        make_plot=not a.no_plot,
        active_dof=a.active_dof,
        target_layers=a.target_layers,
        epochs=a.epochs,
        lambda_l2=a.lambda_l2,
        track_mode=a.track_mode,
        joint_solver=a.joint_solver,
        lm_max_iters=a.lm_max_iters,
        lm_initial_damping=a.lm_initial_damping,
        fisher_eigen_floor=a.fisher_eigen_floor,
        lm_max_step_t=a.lm_max_step_t,
        lm_max_step_r=a.lm_max_step_r,
        min_hits=a.min_hits,
        max_tracks=a.max_tracks,
        gpu=a.gpu,
        joint_clamp_t=a.joint_clamp_t,
        joint_clamp_r=a.joint_clamp_r,
        nominal_full_geom=a.nominal_full_geom,
        joint_seed_mode=a.joint_seed_mode,
        joint_seed_min_hits=a.joint_seed_min_hits,
        joint_seed_min_span=a.joint_seed_min_span,
        joint_smt_seed_iters=a.joint_smt_seed_iters,
        joint_refit_stationarity=not a.no_joint_refit_stationarity,
        alignment_gain=a.alignment_gain,
    )


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    alignment_result = run_alignment_pipeline(alignment_config_from_args(args))
    if args.output_summary is not None:
        write_alignment_summary(alignment_result, args.output_summary)
