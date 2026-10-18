#!/usr/bin/env bash
# =============================================================================
# Iterative ML alignment pipeline.
#
# Each iteration:
#   1. (conda pytorch) solve one ML alignment step with the hardcoded solver below
#   2. (LCG/ACTS)      reconstruct with the newly written geometry
#   3. (conda pytorch) write residual comparison ROOT files for target layers
#
# The conda and LCG environments run in separate subshells. This prevents LCG
# variables such as PYTHONHOME/PYTHONPATH from leaking into conda Python.
# =============================================================================
# Do not enable `set -u`: setup_acts.sh sources LCG scripts that may reference
# unset shell variables.
set -eo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$JOB_DIR"

CONFIG_PATH=""
PRE_ARGS=("$@")
for ((idx = 0; idx < ${#PRE_ARGS[@]}; idx++)); do
    if [[ "${PRE_ARGS[$idx]}" == "--config" ]]; then
        ((idx + 1 < ${#PRE_ARGS[@]})) || { echo "ERROR: --config requires a path" >&2; exit 1; }
        CONFIG_PATH="${PRE_ARGS[$((idx + 1))]}"
    elif [[ "${PRE_ARGS[$idx]}" == --config=* ]]; then
        CONFIG_PATH="${PRE_ARGS[$idx]#--config=}"
    fi
done
if [[ -n "$CONFIG_PATH" ]]; then
    eval "$(python alignment_artifact_layout.py --config "$CONFIG_PATH" --shell)"
else
    eval "$(python alignment_artifact_layout.py --shell)"
fi

ITERS=3
EVENTS=100000
START_ITER=0
TARGET_LAYERS="$TARGET_LAYERS_SPEC"
ACTIVE_DOF="$ACTIVE_DOF_SPEC"
EPOCHS=300
JACOBIAN_UPDATE_INTERVAL=20
LAMBDA_L2=1e-5
SUFFIX="$TARGET_LAYER_SUFFIX"

# Hardcoded solver selection for derivative-comparison runs.
#   cpp      -> ml_alignment.py
#   autograd -> ml_alignment_autograd.py
# Edit RUN_TAG with the solver so complete iteration outputs do not overwrite each other.
ALIGNMENT_SOLVER="autograd"
RUN_TAG="autograd_free"

# Track handling for the autograd solver.
#   surface -> keep the ACTS KF-smoothed tracks frozen, uses EPOCHS
#   free    -> refit non-target helices, then solve the geometry with block-LM defaults
# surface leaves the strip v direction unrecoverable because the KF absorbs that
# misalignment into the track parameters; free breaks that degeneracy.
TRACK_MODE="free"

ALIGNMENT_DIR="$ALIGNMENT_ROOT"
SIMULATION_DIR="$SIMULATION_ROOT"
RECONSTRUCTION_DIR="$RECONSTRUCTION_ROOT"
LOG_DIR="$ITERATIVE_LOG_ROOT"

INPUT_GEOM=""
INPUT_STATES=""
INPUT_SUMMARY=""

ACTS_SETUP="$ML_ALIGNMENT_ACTS_SETUP"
CONDA_SH="$ML_ALIGNMENT_CONDA_SETUP"
CONDA_ENV="pytorch"
DRY_RUN=0
PLOT_RESIDUALS=1

usage() {
    # Unquoted heredoc so target-layer values come from alignment_artifact_layout.
    # Every literal backslash below is doubled: in an unquoted heredoc a trailing
    # `\` would be consumed as a line continuation and join the example lines.
    cat <<EOF
Usage:
  ./run_ml_iterative.sh [options]

Options:
  --iters N              Number of iterations to run (default: 3)
  --events N             Number of events for reconstruction (default: 100000)
  --start-iter N         First output iteration number (default: 0)
  --input-geom PATH      Iteration-0 geometry input
  --states PATH          Iteration-0 trackstates ROOT input
  --summary PATH         Iteration-0 tracksummary ROOT input
  --target-layers SPEC   Target layers (must remain $TARGET_LAYERS_SPEC)
  --active-dof SPEC      Active DOF list passed to the hardcoded ML solver (default: 0,1,5)
  --epochs N             Number of Adam training epochs (default: 300)
  --jacobian-update-interval N
                          Recompute Jacobians every N epochs for cpp solver only (default: 20)
  --lambda-l2 VALUE      L2 regularisation weight passed to the ML solver (default: 1e-5)
  --suffix NAME          File suffix base (must remain $TARGET_LAYER_SUFFIX)
  --alignment-dir PATH   Alignment output directory (default: alignment_output)
  --simulation-dir PATH  Simulation input directory (default: simulation_output)
  --reconstruction-dir PATH
                          Reconstruction output directory (default: reconstruction_output)
  --log-dir PATH         Log directory (default: iterative_ml_logs)
  --acts-setup PATH      ACTS setup script (default: ~/alignTest/setup_acts.sh)
  --conda-sh PATH        conda shell hook (default: ~/miniconda3/etc/profile.d/conda.sh)
  --conda-env NAME       conda environment name (default: pytorch)
  --no-plot              Skip plot_residuals.py after reconstruction
  --dry-run              Print planned commands without running them
  -h, --help             Show this help

Default iteration-0 inputs:
  --input-geom alignment_output/$TARGET_LAYER_DIR/misalignment_applied_$TARGET_LAYER_SUFFIX.json
  --states reconstruction_output/misaligned/$TARGET_LAYER_DIR/trackstates_kf_$TARGET_LAYER_SUFFIX.root
  --summary reconstruction_output/misaligned/$TARGET_LAYER_DIR/tracksummary_kf_$TARGET_LAYER_SUFFIX.root

Resume example:
  ./run_ml_iterative.sh \\
    --iters 3 \\
    --start-iter 4 \\
    --input-geom alignment_output/$TARGET_LAYER_DIR/aligned_transforms_autograd_ml_${TARGET_LAYER_SUFFIX}_iter3.json \\
    --states reconstruction_output/aligned/$TARGET_LAYER_DIR/trackstates_kf_autograd_ml_${TARGET_LAYER_SUFFIX}_iter3.root \\
    --summary reconstruction_output/aligned/$TARGET_LAYER_DIR/tracksummary_kf_autograd_ml_${TARGET_LAYER_SUFFIX}_iter3.root

Hardcoded derivative solver selection:
  Edit ALIGNMENT_SOLVER and RUN_TAG near the top of this script.
  ALIGNMENT_SOLVER="cpp"      RUN_TAG="cpp"      -> ml_alignment.py
  ALIGNMENT_SOLVER="autograd" RUN_TAG="autograd" -> ml_alignment_autograd.py
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) shift 2 ;;
        --config=*) shift ;;
        --iters) ITERS="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --start-iter) START_ITER="$2"; shift 2 ;;
        --input-geom) INPUT_GEOM="$2"; shift 2 ;;
        --states) INPUT_STATES="$2"; shift 2 ;;
        --summary) INPUT_SUMMARY="$2"; shift 2 ;;
        --target-layers) TARGET_LAYERS="$2"; shift 2 ;;
        --active-dof) ACTIVE_DOF="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --jacobian-update-interval) JACOBIAN_UPDATE_INTERVAL="$2"; shift 2 ;;
        --lambda-l2) LAMBDA_L2="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        --alignment-dir) ALIGNMENT_DIR="$2"; shift 2 ;;
        --simulation-dir) SIMULATION_DIR="$2"; shift 2 ;;
        --reconstruction-dir) RECONSTRUCTION_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --acts-setup) ACTS_SETUP="$2"; shift 2 ;;
        --conda-sh) CONDA_SH="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --no-plot) PLOT_RESIDUALS=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ "$ITERS" =~ ^[0-9]+$ ]] || die "--iters must be a non-negative integer"
[[ "$EVENTS" =~ ^[0-9]+$ ]] || die "--events must be a non-negative integer"
[[ "$START_ITER" =~ ^[0-9]+$ ]] || die "--start-iter must be a non-negative integer"
[[ "$EPOCHS" =~ ^[0-9]+$ ]] || die "--epochs must be a non-negative integer"
[[ "$JACOBIAN_UPDATE_INTERVAL" =~ ^[0-9]+$ ]] || die "--jacobian-update-interval must be a non-negative integer"

case "$TRACK_MODE" in
    surface|free) ;;
    *) die "unknown hardcoded TRACK_MODE: $TRACK_MODE (expected surface or free)" ;;
esac

case "$ALIGNMENT_SOLVER" in
    cpp)
        ALIGNMENT_SCRIPT="ml_alignment.py"
        [[ "$TRACK_MODE" == "surface" ]] || \
            die "ALIGNMENT_SOLVER=cpp does not support TRACK_MODE=$TRACK_MODE; ml_alignment.py has no --track-mode"
        ;;
    autograd)
        ALIGNMENT_SCRIPT="ml_alignment_autograd.py"
        ;;
    *)
        die "unknown hardcoded ALIGNMENT_SOLVER: $ALIGNMENT_SOLVER"
        ;;
esac

[[ -n "$RUN_TAG" ]] || die "hardcoded RUN_TAG must not be empty"

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

absolute_project_path() {
    case "$1" in
        /*) realpath -m "$1" ;;
        *) realpath -m "$ML_ALIGNMENT_REPOSITORY_ROOT/$1" ;;
    esac
}

require_file() {
    local path="$1"
    [[ -e "$path" ]] || die "missing required file: $path"
}

with_target_dir() {
    local path="$1"
    if [[ "$(basename "$path")" == "$TARGET_LAYER_DIR" ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$path" "$TARGET_LAYER_DIR"
    fi
}

ALIGNMENT_DIR="$(absolute_project_path "$ALIGNMENT_DIR")"
SIMULATION_DIR="$(absolute_project_path "$SIMULATION_DIR")"
RECONSTRUCTION_DIR="$(absolute_project_path "$RECONSTRUCTION_DIR")"
LOG_DIR="$(absolute_project_path "$LOG_DIR")"
ACTS_SETUP="$(absolute_project_path "$ACTS_SETUP")"
CONDA_SH="$(absolute_project_path "$CONDA_SH")"
[[ -z "$INPUT_GEOM" ]] || INPUT_GEOM="$(absolute_project_path "$INPUT_GEOM")"
[[ -z "$INPUT_STATES" ]] || INPUT_STATES="$(absolute_project_path "$INPUT_STATES")"
[[ -z "$INPUT_SUMMARY" ]] || INPUT_SUMMARY="$(absolute_project_path "$INPUT_SUMMARY")"

run_conda() {
    local log_file="$1"
    shift

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[conda:$CONDA_ENV] log: $log_file"
        print_command "$@"
        return 0
    fi

    (
        unset PYTHONHOME PYTHONPATH ROOTSYS
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
        "$@"
    ) > "$log_file" 2>&1
}

run_acts() {
    local log_file="$1"
    shift

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[acts] log: $log_file"
        print_command "$@"
        return 0
    fi

    (
        unset PYTHONHOME PYTHONPATH ROOTSYS CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
        source "$ACTS_SETUP"
        "$@"
    ) > "$log_file" 2>&1
}

completed_iters=0
pipeline_start=$SECONDS

on_exit() {
    local elapsed=$(( SECONDS - pipeline_start ))
    echo
    echo "--- Pipeline summary ---"
    echo "  Completed iterations: $completed_iters / $ITERS"
    echo "  Current geometry    : $current_geom"
    echo "  Elapsed time        : $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
}
trap on_exit EXIT

ALIGNMENT_TARGET_DIR="$(with_target_dir "$ALIGNMENT_DIR")"
MISALIGNED_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/misaligned")"
ALIGNED_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/aligned")"
LOG_DIR="$(with_target_dir "$LOG_DIR")"
RESIDUALS_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/residuals")"
if [[ -z "$INPUT_GEOM" ]]; then
    INPUT_GEOM="$ALIGNMENT_TARGET_DIR/$MISALIGNMENT_JSON"
fi
if [[ -z "$INPUT_STATES" ]]; then
    INPUT_STATES="$MISALIGNED_RECON_DIR/trackstates_kf_${SUFFIX}.root"
fi
if [[ -z "$INPUT_SUMMARY" ]]; then
    INPUT_SUMMARY="$MISALIGNED_RECON_DIR/tracksummary_kf_${SUFFIX}.root"
fi

[[ "$SUFFIX" == "$TARGET_LAYER_SUFFIX" ]] || die "--suffix must remain $TARGET_LAYER_SUFFIX"
[[ "$TARGET_LAYERS" == "$TARGET_LAYERS_SPEC" ]] || die "--target-layers must remain $TARGET_LAYERS_SPEC"

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_file "$ACTS_SETUP"
    require_file "$CONDA_SH"
    require_file "$GEOMETRY_ROOT/odd_transforms.json"
    require_file "$INPUT_GEOM"
    require_file "$INPUT_STATES"
    require_file "$INPUT_SUMMARY"
    require_file "$ALIGNMENT_SCRIPT"
    mkdir -p "$LOG_DIR" "$ALIGNMENT_TARGET_DIR" "$ALIGNED_RECON_DIR" "$MISALIGNED_RECON_DIR" "$RESIDUALS_DIR"
else
    echo "Dry run: commands will be printed but not executed."
fi

echo "=========================================================="
echo " Iterative ML alignment"
echo "=========================================================="
echo "  iters        : $ITERS"
echo "  events       : $EVENTS"
echo "  start iter   : $START_ITER"
echo "  target layers: $TARGET_LAYERS"
echo "  active dof   : $ACTIVE_DOF"
if [[ "$ALIGNMENT_SOLVER" != "cpp" && "$TRACK_MODE" == "free" ]]; then
    echo "  track mode   : $TRACK_MODE (block-lm defaults)"
else
    echo "  track mode   : $TRACK_MODE (epochs: $EPOCHS)"
fi
echo "  alignment solver: $ALIGNMENT_SOLVER"
echo "  alignment script: $ALIGNMENT_SCRIPT"
echo "  run tag      : $RUN_TAG"
# The Jacobian interval only exists for the cpp solver; the autograd solver has
# no such setting, so nothing about it is reported for that run.
if [[ "$ALIGNMENT_SOLVER" == "cpp" ]]; then
    echo "  jacobian int.: $JACOBIAN_UPDATE_INTERVAL"
fi
echo "  lambda l2    : $LAMBDA_L2"
echo "  input geom   : $INPUT_GEOM"
echo "  input states : $INPUT_STATES"
echo "  input summary: $INPUT_SUMMARY"
echo

current_geom="$INPUT_GEOM"
current_states="$INPUT_STATES"
current_summary="$INPUT_SUMMARY"

for ((offset=0; offset<ITERS; offset++)); do
    iter=$((START_ITER + offset))
    iter_label="${RUN_TAG}_ml_${SUFFIX}_iter${iter}"
    output_geom="$ALIGNMENT_TARGET_DIR/aligned_transforms_${iter_label}.json"
    aligned_file_arg="$output_geom"
    aligned_trackstates="trackstates_kf_${iter_label}.root"
    aligned_summary="tracksummary_kf_${iter_label}.root"
    next_states="$ALIGNED_RECON_DIR/$aligned_trackstates"
    next_summary="$ALIGNED_RECON_DIR/$aligned_summary"

    echo "########## ITERATION $iter ##########"

    ml_cmd=(
        python "$ALIGNMENT_SCRIPT"
        --input-geom "$current_geom"
        --states "$current_states"
        --summary "$current_summary"
        --output-geom "$output_geom"
        --active-dof "$ACTIVE_DOF"
        --target-layers "$TARGET_LAYERS"
        --lambda-l2 "$LAMBDA_L2"
        --iter "$iter"
        --no-plot
    )
    if [[ "$ALIGNMENT_SOLVER" == "cpp" ]]; then
        ml_cmd+=(--epochs "$EPOCHS")
        ml_cmd+=(--jacobian-update-interval "$JACOBIAN_UPDATE_INTERVAL")
    else
        ml_cmd+=(--track-mode "$TRACK_MODE")
        if [[ "$TRACK_MODE" == "surface" ]]; then
            ml_cmd+=(--epochs "$EPOCHS")
        fi
    fi
    echo "[$iter] ML solve"
    run_conda "$LOG_DIR/${iter_label}.log" "${ml_cmd[@]}" || die "ML solve failed; see $LOG_DIR/${iter_label}.log"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        require_file "$output_geom"
        grep -E "Per-layer residual mean|vol=" "$LOG_DIR/${iter_label}.log" | tail -8 || true
    fi

    recon_cmd=(
        python alignment_odd.py reconstruction
        --mode aligned
        --num-events "$EVENTS"
        --input-dir "$SIMULATION_DIR"
        --output-dir "$RECONSTRUCTION_DIR"
        --aligned-file "$aligned_file_arg"
        --output-suffix "$iter_label"
    )
    echo "[$iter] ACTS reconstruction"
    run_acts "$LOG_DIR/recon_${iter_label}.log" "${recon_cmd[@]}" || die "reconstruction failed; see $LOG_DIR/recon_${iter_label}.log"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        require_file "$next_states"
        require_file "$next_summary"
    fi

    if [[ "$PLOT_RESIDUALS" -eq 1 ]]; then
        PLOT_OUT_DIR="$RESIDUALS_DIR/$iter_label"

        # Plot commands are generated from TARGET_LAYERS_SPEC so the plotted
        # layers can never drift away from the layers actually being solved.
        # Local IFS only: a global IFS change would leak into the rest of the run.
        IFS=',' read -r -a target_pairs <<< "$TARGET_LAYERS_SPEC"
        for pair in "${target_pairs[@]}"; do
            IFS=':' read -r vol lay <<< "$pair"
            plot_label="${iter_label}_v${vol}l${lay}"

            case "$vol" in
                16|17|18)
                    # Pixel: both projections share the same fine range.
                    plot_specs=("X Y|-1.0 1.0|")
                    ;;
                23|24|25)
                    # ShortStrip: intrinsic Y resolution is ~1.2 mm, so Y needs a
                    # much wider range and is plotted separately from X.
                    plot_specs=("X|-1.0 1.0|_X" "Y|-6.0 6.0|_Y")
                    ;;
                *)
                    die "unsupported plotting volume: $vol (only Pixel 16/17/18 and ShortStrip 23/24/25 supported)"
                    ;;
            esac

            for spec in "${plot_specs[@]}"; do
                IFS='|' read -r residuals range log_tag <<< "$spec"
                plot_cmd=(
                    python plot_residuals.py
                    --recon-dir "$RECONSTRUCTION_DIR"
                    --output-dir "$PLOT_OUT_DIR"
                    --aligned-trackstates "$aligned_trackstates"
                    --label "$plot_label"
                    --volume_id "$vol"
                    --layer_id "$lay"
                    --range $range
                    --residuals $residuals
                )
                plot_log="$LOG_DIR/plot_${plot_label}${log_tag}.log"
                echo "[$iter] Residual plots for vol=$vol lay=$lay (${residuals// /})"
                run_conda "$plot_log" "${plot_cmd[@]}" || die "plot_residuals failed; see $plot_log"
            done
        done
    fi

    current_geom="$output_geom"
    current_states="$next_states"
    current_summary="$next_summary"
    completed_iters=$((completed_iters + 1))

    iter_elapsed=$(( SECONDS - pipeline_start ))
    echo "[$iter] Done (total elapsed: $(( iter_elapsed / 60 ))m $(( iter_elapsed % 60 ))s)"
    echo
done

echo "=========================================================="
echo " Iterative pipeline finished"
echo "=========================================================="
echo "  final geometry: $current_geom"
echo "  final states  : $current_states"
echo "  final summary : $current_summary"
echo "  logs          : $LOG_DIR"
