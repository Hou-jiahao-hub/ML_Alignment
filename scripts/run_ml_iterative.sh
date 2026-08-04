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

eval "$(python alignment_artifact_layout.py --shell)"

ITERS=3
EVENTS=100000
START_ITER=0
TARGET_LAYERS="$TARGET_LAYERS_SPEC"
ACTIVE_DOF="0,1,3,4,5"
EPOCHS=300
JACOBIAN_UPDATE_INTERVAL=20
LAMBDA_L2=1e-5
SUFFIX="$TARGET_LAYER_SUFFIX"

# Hardcoded solver selection for derivative-comparison runs.
#   cpp      -> ml_alignment.py
#   autograd -> ml_alignment_autograd.py
# Edit RUN_TAG with the solver so complete iteration outputs do not overwrite each other.
ALIGNMENT_SOLVER="autograd"
RUN_TAG="autograd"

ALIGNMENT_DIR="alignment_output"
SIMULATION_DIR="simulation_output"
RECONSTRUCTION_DIR="reconstruction_output"
LOG_DIR="iterative_ml_logs"

INPUT_GEOM=""
INPUT_STATES=""
INPUT_SUMMARY=""

ACTS_SETUP="$HOME/alignTest/setup_acts.sh"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="pytorch"
DRY_RUN=0
PLOT_RESIDUALS=1

usage() {
    cat <<'EOF'
Usage:
  ./run_ml_iterative.sh [options]

Options:
  --iters N              Number of iterations to run (default: 3)
  --events N             Number of events for reconstruction (default: 100000)
  --start-iter N         First output iteration number (default: 0)
  --input-geom PATH      Iteration-0 geometry input
  --states PATH          Iteration-0 trackstates ROOT input
  --summary PATH         Iteration-0 tracksummary ROOT input
  --target-layers SPEC   Target layers, e.g. 17:8,24:4 (default: 17:8,24:4)
  --active-dof SPEC      Active DOF list passed to the hardcoded ML solver (default: 0,1,3,4,5)
  --epochs N             Number of Adam training epochs (default: 300)
  --jacobian-update-interval N
                          Recompute Jacobians every N epochs for cpp solver only (default: 20)
  --lambda-l2 VALUE      L2 regularisation weight passed to the ML solver (default: 1e-5)
  --suffix NAME          File suffix base (must remain v17l8_v24l4)
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
  --input-geom alignment_output/Vol17_Lay8__Vol24_Lay4/misalignment_applied_v17l8_v24l4.json
  --states reconstruction_output/misaligned/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root
  --summary reconstruction_output/misaligned/Vol17_Lay8__Vol24_Lay4/tracksummary_kf_v17l8_v24l4.root

Resume example:
  ./run_ml_iterative.sh \
    --iters 3 \
    --start-iter 4 \
    --input-geom alignment_output/Vol17_Lay8__Vol24_Lay4/aligned_transforms_autograd_ml_v17l8_v24l4_iter3.json \
    --states reconstruction_output/aligned/Vol17_Lay8__Vol24_Lay4/trackstates_kf_autograd_ml_v17l8_v24l4_iter3.root \
    --summary reconstruction_output/aligned/Vol17_Lay8__Vol24_Lay4/tracksummary_kf_autograd_ml_v17l8_v24l4_iter3.root

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

case "$ALIGNMENT_SOLVER" in
    cpp)
        ALIGNMENT_SCRIPT="ml_alignment.py"
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

relative_to_alignment_output() {
    realpath -m --relative-to="$ALIGNMENT_DIR" "$1"
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
    require_file "odd_transforms.json"
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
echo "  epochs       : $EPOCHS"
echo "  alignment solver: $ALIGNMENT_SOLVER"
echo "  alignment script: $ALIGNMENT_SCRIPT"
echo "  run tag      : $RUN_TAG"
if [[ "$ALIGNMENT_SOLVER" == "cpp" ]]; then
    echo "  jacobian int.: $JACOBIAN_UPDATE_INTERVAL"
else
    echo "  jacobian int.: ignored for autograd solver"
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
    aligned_file_arg="$(relative_to_alignment_output "$output_geom")"
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
        --epochs "$EPOCHS"
        --lambda-l2 "$LAMBDA_L2"
        --iter "$iter"
        --no-plot
    )
    if [[ "$ALIGNMENT_SOLVER" == "cpp" ]]; then
        ml_cmd+=(--jacobian-update-interval "$JACOBIAN_UPDATE_INTERVAL")
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

        plot_v17_cmd=(
            python plot_residuals.py
            --recon-dir "$RECONSTRUCTION_DIR"
            --output-dir "$PLOT_OUT_DIR"
            --aligned-trackstates "$aligned_trackstates"
            --label "${iter_label}_v17l8"
            --volume_id 17
            --layer_id 8
            --range -1.0 1.0
            --residuals X Y
        )
        echo "[$iter] Residual plots for vol=17 lay=8"
        run_conda "$LOG_DIR/plot_${iter_label}_v17l8.log" "${plot_v17_cmd[@]}" || die "plot_residuals failed; see $LOG_DIR/plot_${iter_label}_v17l8.log"

        plot_v24x_cmd=(
            python plot_residuals.py
            --recon-dir "$RECONSTRUCTION_DIR"
            --output-dir "$PLOT_OUT_DIR"
            --aligned-trackstates "$aligned_trackstates"
            --label "${iter_label}_v24l4"
            --volume_id 24
            --layer_id 4
            --range -1.0 1.0
            --residuals X
        )
        echo "[$iter] Residual plots for vol=24 lay=4 (X)"
        run_conda "$LOG_DIR/plot_${iter_label}_v24l4_X.log" "${plot_v24x_cmd[@]}" || die "plot_residuals failed; see $LOG_DIR/plot_${iter_label}_v24l4_X.log"

        plot_v24y_cmd=(
            python plot_residuals.py
            --recon-dir "$RECONSTRUCTION_DIR"
            --output-dir "$PLOT_OUT_DIR"
            --aligned-trackstates "$aligned_trackstates"
            --label "${iter_label}_v24l4"
            --volume_id 24
            --layer_id 4
            --range -6.0 6.0
            --residuals Y
        )
        echo "[$iter] Residual plots for vol=24 lay=4 (Y)"
        run_conda "$LOG_DIR/plot_${iter_label}_v24l4_Y.log" "${plot_v24y_cmd[@]}" || die "plot_residuals failed; see $LOG_DIR/plot_${iter_label}_v24l4_Y.log"
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
