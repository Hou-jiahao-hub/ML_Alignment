#!/usr/bin/env bash
# Joint free helix -> ACTS acceptance -> guarded polish.
# The two surface warm starts were removed on 2026-09-01: block-LM reaches the
# same geometry from the raw misaligned input once LM_MAX_STEP_R spans the
# injected rotation (see data/experiments/.../joint-only-20260901-c-step5e3).
set -eo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$JOB_DIR"
ORIGINAL_ARGS=("$@")
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
    eval "$(python3 alignment_artifact_layout.py --config "$CONFIG_PATH" --shell)"
else
    eval "$(python3 alignment_artifact_layout.py --shell)"
fi
CONFIG_ARGS=()
[[ -z "$CONFIG_PATH" ]] || CONFIG_ARGS=(--config "$CONFIG_PATH")

RUN_TAG=""
EVENTS=100000
EXPECTED_TARGET_LAYERS="17:8,24:4"
TARGET_LAYERS="$EXPECTED_TARGET_LAYERS"
ACTIVE_DOF="$ACTIVE_DOF_SPEC"
SURFACE_EPOCHS=200
JOINT_ITERS=1000
JOINT_SMT_SEED_ITERS=2000
JOINT_MEASUREMENT_REFIT_ITERS=200
LM_MAX_ITERS=60
MAX_TRACKS=0
SKIP_BLIND_VALIDATION=0
NO_EVENT_SPLIT=0
LAMBDA_L2=1e-5
ALIGNMENT_DIR="$ALIGNMENT_ROOT"
SIMULATION_DIR="$SIMULATION_ROOT"
RECONSTRUCTION_DIR="$RECONSTRUCTION_ROOT"
LOG_DIR="$SURFACE_HELIX_LOG_ROOT"
INPUT_GEOM=""
INPUT_STATES=""
INPUT_SUMMARY=""
NOMINAL_STATES=""
HELIX_CANDIDATES=""
BASELINE_RATIO=""
ACTS_SETUP="$ML_ALIGNMENT_ACTS_SETUP"
CONDA_SH="$ML_ALIGNMENT_CONDA_SETUP"
CONDA_ENV="pytorch"
DRY_RUN=0
EVENT_SPLIT_SEED=6006

usage() {
    cat <<EOF
Usage: ./run_surface_helix_experiment.sh --run-tag NAME [options]

Required:
  --run-tag NAME             Unique namespace for all new artifacts

Options:
  --events N                 ACTS reconstruction events (default: 100000)
  --surface-epochs N         Epochs in each guarded polish surface solve
  --joint-iters N            Joint free-helix iterations (default: 1000)
  --joint-smt-seed-iters N   Non-target SMT seed iterations (default: 2000).
                             A cap, not a cost: the damped Gauss-Newton seed
                             converges at ~367 iterations and breaks out, so
                             200 only truncated it below the 0.1% stationarity
                             gate.  See ml_optimization_log.md Iteration 26.
  --joint-measurement-refit-iters N
                              Non-target measurement refit iterations (default: 200)
  --lm-max-iters N            Block-LM geometry iterations (default: 60)
  --max-tracks N             Limit free-helix tracks; 0 keeps all (default: 0)
  --input-geom PATH          Initial misaligned geometry
  --states PATH              Initial misaligned trackstates
  --summary PATH             Initial misaligned tracksummary
  --alignment-dir PATH       Alignment output base
  --simulation-dir PATH      Existing simulation input
  --reconstruction-dir PATH  Reconstruction output base
  --log-dir PATH             Experiment output base
  --acts-setup PATH          ACTS environment setup script
  --conda-sh PATH            Conda shell hook
  --conda-env NAME           Conda environment name
  --nominal-states PATH      Nominal states used only to freeze acceptance thresholds
  --event-split-seed N       Stable 80/10/10 event split seed (default: 6006)
  --baseline-ratio VALUE     Deprecated; ignored
  --helix-candidates SPEC    Deprecated; ignored
  --skip-blind-validation    Stop after calibration; leave the one-shot blind
                             validation split untouched
  --no-event-split           Fit and score on every event instead of the hash
                             80/10/10 partitions.  Requires
                             --skip-blind-validation: with no split the
                             validation acceptance would re-score the very
                             events that selected the geometry.  Acceptance
                             thresholds stay hash-calibrated on nominal states,
                             which no stage of this run fits.
  --dry-run                  Print both conditional polish candidates
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) shift 2 ;;
        --config=*) shift ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --surface-epochs) SURFACE_EPOCHS="$2"; shift 2 ;;
        --joint-iters) JOINT_ITERS="$2"; shift 2 ;;
        --joint-smt-seed-iters) JOINT_SMT_SEED_ITERS="$2"; shift 2 ;;
        --joint-measurement-refit-iters) JOINT_MEASUREMENT_REFIT_ITERS="$2"; shift 2 ;;
        --lm-max-iters) LM_MAX_ITERS="$2"; shift 2 ;;
        --max-tracks) MAX_TRACKS="$2"; shift 2 ;;
        --input-geom) INPUT_GEOM="$2"; shift 2 ;;
        --states) INPUT_STATES="$2"; shift 2 ;;
        --summary) INPUT_SUMMARY="$2"; shift 2 ;;
        --nominal-states) NOMINAL_STATES="$2"; shift 2 ;;
        --baseline-ratio) BASELINE_RATIO="$2"; shift 2 ;;
        --helix-candidates) HELIX_CANDIDATES="$2"; shift 2 ;;
        --alignment-dir) ALIGNMENT_DIR="$2"; shift 2 ;;
        --simulation-dir) SIMULATION_DIR="$2"; shift 2 ;;
        --reconstruction-dir) RECONSTRUCTION_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --acts-setup) ACTS_SETUP="$2"; shift 2 ;;
        --conda-sh) CONDA_SH="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --event-split-seed) EVENT_SPLIT_SEED="$2"; shift 2 ;;
        --skip-blind-validation) SKIP_BLIND_VALIDATION=1; shift ;;
        --no-event-split) NO_EVENT_SPLIT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$RUN_TAG" ]] || die "--run-tag is required"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "--run-tag contains unsafe characters"
[[ "$EVENTS" =~ ^[1-9][0-9]*$ ]] || die "--events must be positive"
[[ "$SURFACE_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "--surface-epochs must be positive"
[[ "$JOINT_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--joint-iters must be positive"
[[ "$JOINT_SMT_SEED_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--joint-smt-seed-iters must be positive"
[[ "$JOINT_MEASUREMENT_REFIT_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--joint-measurement-refit-iters must be positive"
[[ "$LM_MAX_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--lm-max-iters must be positive"
[[ "$MAX_TRACKS" =~ ^[0-9]+$ ]] || die "--max-tracks must be non-negative"
[[ "$EVENT_SPLIT_SEED" =~ ^[0-9]+$ ]] || die "--event-split-seed must be non-negative"
[[ "$TARGET_LAYERS_SPEC" == "$EXPECTED_TARGET_LAYERS" ]] || \
    die "configured target layers must remain $EXPECTED_TARGET_LAYERS"
[[ "$ACTIVE_DOF" == "0,1,5" ]] || die "this workflow requires active DOF 0,1,5"
if [[ "$NO_EVENT_SPLIT" -eq 1 && "$SKIP_BLIND_VALIDATION" -eq 0 ]]; then
    die "--no-event-split requires --skip-blind-validation: without a split the validation acceptance re-scores the events that selected the geometry"
fi

[[ -z "$NOMINAL_STATES" ]] || echo "note: --nominal-states supplies only the frozen acceptance baseline" >&2
[[ -z "$BASELINE_RATIO" ]] || echo "deprecated: --baseline-ratio no longer controls the workflow" >&2
[[ -z "$HELIX_CANDIDATES" ]] || echo "deprecated: --helix-candidates no longer selects layers" >&2

with_target_dir() {
    if [[ "$(basename "$1")" == "$TARGET_LAYER_DIR" ]]; then printf '%s\n' "$1"
    else printf '%s/%s\n' "$1" "$TARGET_LAYER_DIR"; fi
}

absolute_project_path() {
    case "$1" in
        /*) realpath -m "$1" ;;
        *) realpath -m "$ML_ALIGNMENT_REPOSITORY_ROOT/$1" ;;
    esac
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
[[ -z "$NOMINAL_STATES" ]] || NOMINAL_STATES="$(absolute_project_path "$NOMINAL_STATES")"

ALIGNMENT_TARGET_DIR="$(with_target_dir "$ALIGNMENT_DIR")"
MISALIGNED_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/misaligned")"
ALIGNED_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/aligned")"
RUN_DIR="$(with_target_dir "$LOG_DIR")/$RUN_TAG"
[[ -n "$INPUT_GEOM" ]] || INPUT_GEOM="$ALIGNMENT_TARGET_DIR/$MISALIGNMENT_JSON"
[[ -n "$INPUT_STATES" ]] || INPUT_STATES="$MISALIGNED_RECON_DIR/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"
[[ -n "$INPUT_SUMMARY" ]] || INPUT_SUMMARY="$MISALIGNED_RECON_DIR/tracksummary_kf_${TARGET_LAYER_SUFFIX}.root"
[[ -n "$NOMINAL_STATES" ]] || NOMINAL_STATES="$(with_target_dir "$RECONSTRUCTION_DIR/nominal")/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

# Empty under --no-event-split: both solvers then see every event, because
# ml_alignment_autograd.py only filters when --event-partition is supplied.
SOLVER_PARTITION_ARGS=(--event-partition train --event-split-seed "$EVENT_SPLIT_SEED")
[[ "$NO_EVENT_SPLIT" -eq 0 ]] || SOLVER_PARTITION_ARGS=()

run_conda() {
    local log_file="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[conda:$CONDA_ENV]"
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
    local log_file="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[acts]"
        print_command "$@"
        return 0
    fi
    (
        unset PYTHONHOME PYTHONPATH ROOTSYS CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
        source "$ACTS_SETUP"
        "$@"
    ) > "$log_file" 2>&1
}

require_file() { [[ -f "$1" ]] || die "missing required file: $1"; }

geometry_path() { printf '%s/aligned_transforms_%s.json\n' "$ALIGNMENT_TARGET_DIR" "$1"; }
states_path() { printf '%s/trackstates_kf_%s.root\n' "$ALIGNED_RECON_DIR" "$1"; }
summary_path() { printf '%s/tracksummary_kf_%s.root\n' "$ALIGNED_RECON_DIR" "$1"; }
acceptance_path() { printf '%s/acceptance_%s.json\n' "$RUN_DIR" "$1"; }
BASELINE_PROFILE="$RUN_DIR/acceptance_baseline.json"

echo "maximum surface polish    : 2"
echo "joint target layers       : $TARGET_LAYERS"
echo "active DOF                : $ACTIVE_DOF"
echo "run tag                   : $RUN_TAG"

MANIFEST_PATH="$RUN_DIR/manifest.json"
MANIFEST_FINALIZED=0
CURRENT_STAGE="preflight"
finalize_failed_manifest() {
    local exit_code="$?"
    if [[ "$DRY_RUN" -eq 0 && "$MANIFEST_FINALIZED" -eq 0 && -f "$MANIFEST_PATH" ]]; then
        local failure_metrics
        local failure_outputs=()
        failure_metrics="$(python3 -c 'import json,os,sys; joint_path,acceptance_path=sys.argv[1:3]; metrics={"exit_code":int(sys.argv[3]),"stop_reason":"unexpected_exit","stage":sys.argv[4]}; metrics["joint_free"]=json.load(open(joint_path)) if os.path.isfile(joint_path) else None; metrics["latest_acceptance"]=json.load(open(acceptance_path)) if acceptance_path and os.path.isfile(acceptance_path) else None; print(json.dumps(metrics,allow_nan=False))' \
            "$RUN_DIR/joint_free_summary.json" "${current_acceptance:-}" "$exit_code" "$CURRENT_STAGE" 2>/dev/null || \
            printf '{"exit_code":%s,"stop_reason":"unexpected_exit","stage":"%s"}' "$exit_code" "$CURRENT_STAGE")"
        for failed_label in \
            "${RUN_TAG}_joint_free" "${RUN_TAG}_surface_polish1" \
            "${RUN_TAG}_surface_polish2"; do
            [[ ! -f "$(geometry_path "$failed_label")" ]] || \
                failure_outputs+=(--output "${failed_label}_geometry=$(geometry_path "$failed_label")")
            [[ ! -f "$(states_path "$failed_label")" ]] || \
                failure_outputs+=(--output "${failed_label}_states=$(states_path "$failed_label")")
            [[ ! -f "$(summary_path "$failed_label")" ]] || \
                failure_outputs+=(--output "${failed_label}_summary=$(summary_path "$failed_label")")
            [[ ! -f "$(acceptance_path "$failed_label")" ]] || \
                failure_outputs+=(--output "${failed_label}_acceptance=$(acceptance_path "$failed_label")")
        done
        python3 experiment_manifest.py finalize --manifest "$MANIFEST_PATH" \
            --status failed "${failure_outputs[@]}" --metrics-json "$failure_metrics" \
            >/dev/null 2>&1 || true
    fi
}
trap finalize_failed_manifest EXIT
trap 'exit 130' INT TERM HUP

if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ ! -e "$RUN_DIR" ]] || die "run tag already exists: $RUN_DIR"
    for label in \
        "${RUN_TAG}_joint_free" "${RUN_TAG}_surface_polish1" \
        "${RUN_TAG}_surface_polish2"; do
        [[ ! -e "$(geometry_path "$label")" ]] || die "geometry already exists: $(geometry_path "$label")"
        [[ ! -e "$(states_path "$label")" ]] || die "reconstruction already exists: $(states_path "$label")"
    done
    mkdir -p "$RUN_DIR" "$ALIGNMENT_TARGET_DIR" "$ALIGNED_RECON_DIR"
    command_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$0" "${ORIGINAL_ARGS[@]}")"
    python3 experiment_manifest.py start --run-dir "$RUN_DIR" --run-id "$RUN_TAG" \
        --allow-missing-inputs \
        --command-json "$command_json" --input "geometry=$INPUT_GEOM" \
        --input "states=$INPUT_STATES" --input "summary=$INPUT_SUMMARY" \
        --input "nominal_states=$NOMINAL_STATES" \
        --input "simulation_dataset=$SIMULATION_DIR" \
        --input "nominal_geometry=$GEOMETRY_ROOT/odd_transforms.json" \
        --input "acts_setup=$ACTS_SETUP" --input "conda_setup=$CONDA_SH" \
        --input "odd_digi_config=$ML_ALIGNMENT_ACTS_DIGI_CONFIG" \
        --input "odd_material_map=$ML_ALIGNMENT_ODD_MATERIAL_MAP" \
        "${CONFIG_ARGS[@]}" >/dev/null
    require_file "$INPUT_GEOM"
    require_file "$INPUT_STATES"
    require_file "$INPUT_SUMMARY"
    require_file "$NOMINAL_STATES"
    require_file "$GEOMETRY_ROOT/odd_transforms.json"
    [[ -d "$SIMULATION_DIR" ]] || die "missing simulation directory: $SIMULATION_DIR"
    require_file "$ACTS_SETUP"
    require_file "$CONDA_SH"
    require_file "$ML_ALIGNMENT_ACTS_DIGI_CONFIG"
    require_file "$ML_ALIGNMENT_ODD_MATERIAL_MAP"
fi

CURRENT_STAGE="acceptance_baseline"
run_conda "$RUN_DIR/acceptance_baseline.log" \
    python calibrate_acceptance.py "${CONFIG_ARGS[@]}" --states "$NOMINAL_STATES" \
    --output "$BASELINE_PROFILE" --target-layers "$TARGET_LAYERS" \
    --split-seed "$EVENT_SPLIT_SEED" --bootstrap-seed "$EVENT_SPLIT_SEED"
if [[ "$DRY_RUN" -eq 0 ]]; then require_file "$BASELINE_PROFILE"; fi

reconstruct() {
    local label="$1" geometry="$2"
    run_acts "$RUN_DIR/reconstruct_${label}.log" \
        python alignment_odd.py reconstruction --mode aligned --num-events "$EVENTS" \
        --input-dir "$SIMULATION_DIR" --output-dir "$RECONSTRUCTION_DIR" \
        --aligned-file "$geometry" --output-suffix "$label"
}

run_acceptance() {
    local label="$1" states="$2" output="$3" partition="$4"
    local partition_args=(--event-partition "$partition" --split-seed "$EVENT_SPLIT_SEED")
    [[ "$NO_EVENT_SPLIT" -eq 0 ]] || partition_args=()
    if [[ "$DRY_RUN" -eq 1 ]]; then
        run_conda "$RUN_DIR/acceptance_${label}.log" \
            python reconstruction_acceptance.py "${CONFIG_ARGS[@]}" --states "$states" --output "$output" \
            --target-layers "$TARGET_LAYERS" --threshold-profile "$BASELINE_PROFILE" \
            "${partition_args[@]}"
        return 0
    fi
    set +e
    run_conda "$RUN_DIR/acceptance_${label}.log" \
        python reconstruction_acceptance.py "${CONFIG_ARGS[@]}" --states "$states" --output "$output" \
        --target-layers "$TARGET_LAYERS" --threshold-profile "$BASELINE_PROFILE" \
        "${partition_args[@]}"
    local status="$?"
    set -e
    require_file "$output"
    return "$status"
}

current_geom="$INPUT_GEOM"
current_states="$INPUT_STATES"
current_summary="$INPUT_SUMMARY"

# The joint solver runs straight on the supplied misaligned geometry.  Its
# guard still needs an arm to improve on, so the input geometry is scored once
# on the calibration split.  Its reconstruction already exists -- it is what
# --states/--summary point at -- so this costs one acceptance call, not a run.
baseline_label="${RUN_TAG}_baseline_input"
baseline_acceptance="$(acceptance_path "$baseline_label")"
CURRENT_STAGE="baseline_calibration_acceptance"
run_acceptance "$baseline_label" "$current_states" "$baseline_acceptance" calibration || true
baseline_geom="$current_geom"
baseline_states="$current_states"
baseline_summary="$current_summary"

joint_label="${RUN_TAG}_joint_free"
CURRENT_STAGE="joint_free"
joint_geom="$(geometry_path "$joint_label")"
joint_solver_summary="$RUN_DIR/joint_free_summary.json"
run_conda "$RUN_DIR/${joint_label}.log" \
    python ml_alignment_autograd.py "${CONFIG_ARGS[@]}" --track-mode free \
    --input-geom "$current_geom" --states "$current_states" --summary "$current_summary" \
    --output-geom "$joint_geom" --target-layers "$TARGET_LAYERS" --active-dof "$ACTIVE_DOF" \
    --joint-seed-mode non-target-measurement --joint-seed-min-hits 8 \
    --joint-seed-min-span 300 --joint-smt-seed-iters "$JOINT_SMT_SEED_ITERS" \
    --joint-measurement-refit-iters "$JOINT_MEASUREMENT_REFIT_ITERS" \
    --joint-solver block-lm --lm-max-iters "$LM_MAX_ITERS" \
    "${SOLVER_PARTITION_ARGS[@]}" \
    --joint-iters "$JOINT_ITERS" --max-tracks "$MAX_TRACKS" \
    --lambda-l2 "$LAMBDA_L2" --nominal-full-geom "$GEOMETRY_ROOT/odd_transforms.json" \
    --output-summary "$joint_solver_summary" --no-plot
reconstruct "$joint_label" "$joint_geom"
joint_states="$(states_path "$joint_label")"
joint_summary="$(summary_path "$joint_label")"
joint_acceptance="$(acceptance_path "$joint_label")"
CURRENT_STAGE="joint_acceptance"
run_acceptance "$joint_label" "$joint_states" "$joint_acceptance" calibration || true

current_geom="$joint_geom"
current_states="$joint_states"
current_summary="$joint_summary"
current_acceptance="$joint_acceptance"
stop_reason="joint_acceptance_passed"
polish_attempts=0
joint_adopted=1

if [[ "$DRY_RUN" -eq 0 ]]; then
    joint_improved="$(python3 -c 'import json,sys; from reconstruction_acceptance import acceptance_score,polish_improves; joint=json.load(open(sys.argv[1])); baseline=json.load(open(sys.argv[2])); print("1" if polish_improves(acceptance_score(joint),acceptance_score(baseline)) else "0")' "$joint_acceptance" "$baseline_acceptance")"
    joint_effective="$(python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); print("1" if value.get("best_iteration",-1)>=0 and value.get("correction_max_abs",0)>0 else "0")' "$joint_solver_summary")"
    if [[ "$joint_improved" == "0" || "$joint_effective" == "0" ]]; then
        current_geom="$baseline_geom"
        current_states="$baseline_states"
        current_summary="$baseline_summary"
        current_acceptance="$baseline_acceptance"
        stop_reason="joint_not_improved"
        joint_adopted=0
    fi
fi

for polish_round in 1 2; do
    if [[ "$DRY_RUN" -eq 0 ]]; then
        passed="$(python3 -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["passed"] else "0")' "$current_acceptance")"
        [[ "$passed" == "0" ]] || break
        failed_layers="$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["failed_layers"]))' "$current_acceptance")"
        [[ -n "$failed_layers" ]] || die "acceptance failed without failed_layers"
    else
        failed_layers="$TARGET_LAYERS"
    fi
    label="${RUN_TAG}_surface_polish${polish_round}"
    CURRENT_STAGE="surface_polish${polish_round}"
    candidate_geom="$(geometry_path "$label")"
    run_conda "$RUN_DIR/${label}.log" \
        python ml_alignment_autograd.py "${CONFIG_ARGS[@]}" --track-mode surface \
        --input-geom "$current_geom" --states "$current_states" --summary "$current_summary" \
        --output-geom "$candidate_geom" --target-layers "$failed_layers" \
        --active-dof "$ACTIVE_DOF" --alignment-gain 0.2 --epochs "$SURFACE_EPOCHS" \
        "${SOLVER_PARTITION_ARGS[@]}" \
        --lambda-l2 "$LAMBDA_L2" --iter "polish${polish_round}" --no-plot
    reconstruct "$label" "$candidate_geom"
    candidate_states="$(states_path "$label")"
    candidate_summary="$(summary_path "$label")"
    candidate_acceptance="$(acceptance_path "$label")"
    run_acceptance "$label" "$candidate_states" "$candidate_acceptance" calibration || true
    polish_attempts="$polish_round"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        improved="$(python3 -c 'import json,sys; from reconstruction_acceptance import acceptance_score,polish_improves; c=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); print("1" if polish_improves(acceptance_score(c),acceptance_score(p)) else "0")' "$candidate_acceptance" "$current_acceptance")"
        if [[ "$improved" == "0" ]]; then
            stop_reason="polish_not_improved"
            break
        fi
    fi
    current_geom="$candidate_geom"
    current_states="$candidate_states"
    current_summary="$candidate_summary"
    current_acceptance="$candidate_acceptance"
    stop_reason="polish_limit_reached"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        passed="$(python3 -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["passed"] else "0")' "$current_acceptance")"
        if [[ "$passed" == "1" ]]; then
            stop_reason="polish_acceptance_passed"
            break
        fi
    fi
done

calibration_acceptance="$current_acceptance"
final_validation_acceptance="$(acceptance_path "${RUN_TAG}_validation")"
if [[ "$SKIP_BLIND_VALIDATION" -eq 1 ]]; then
    echo "note: --skip-blind-validation is set; the validation split stays untouched" >&2
else
    CURRENT_STAGE="blind_validation"
    run_acceptance "${RUN_TAG}_validation" "$current_states" "$final_validation_acceptance" validation || true
    current_acceptance="$final_validation_acceptance"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_file "$current_geom"
    require_file "$current_states"
    require_file "$current_summary"
    require_file "$current_acceptance"
fi

CURRENT_STAGE="final_diagnostics"
run_conda "$RUN_DIR/residual_comparison.log" \
    python plot_residuals.py "${CONFIG_ARGS[@]}" --recon-dir "$RECONSTRUCTION_DIR" \
    --aligned-trackstates "$current_states" --label "$RUN_TAG" --output-dir "$RUN_DIR"
run_conda "$RUN_DIR/chi2_comparison.log" \
    python plot_chi2ndf.py "${CONFIG_ARGS[@]}" --recon-dir "$RECONSTRUCTION_DIR" \
    --aligned-tracksummary "$current_summary" --output "$RUN_DIR/track_chi2_ndf_comparison.png"
run_conda "$RUN_DIR/geometry_comparison.log" \
    python compare_geometries.py "${CONFIG_ARGS[@]}" --nominal "$GEOMETRY_ROOT/odd_transforms.json" \
    --misaligned "$INPUT_GEOM" --aligned "$current_geom" --output-dir "$RUN_DIR/geometry_comparison"

if [[ "$DRY_RUN" -eq 0 ]]; then
    CURRENT_STAGE="finalize"
    summary_args=(
        --run-tag "$RUN_TAG"
        --joint-summary "$joint_solver_summary"
        --joint-adopted "$joint_adopted"
        --calibration-acceptance "$calibration_acceptance"
        --acceptance-baseline "$BASELINE_PROFILE"
        --final-geometry "$current_geom"
        --final-trackstates "$current_states"
        --final-tracksummary "$current_summary"
        --polish-attempts "$polish_attempts"
        --selection-stop-reason "$stop_reason"
        --event-split-seed "$EVENT_SPLIT_SEED"
        --output "$RUN_DIR/experiment_summary.json"
    )
    if [[ "$NO_EVENT_SPLIT" -eq 1 ]]; then
        summary_args+=(--no-event-split)
    fi
    if [[ "$SKIP_BLIND_VALIDATION" -eq 1 ]]; then
        summary_args+=(--skip-blind-validation)
    else
        summary_args+=(--final-acceptance "$current_acceptance")
    fi
    python3 write_experiment_summary.py "${summary_args[@]}"
    manifest_outputs=(
        --output "baseline_input_acceptance=$baseline_acceptance"
        --output "joint_geometry=$joint_geom"
        --output "joint_states=$joint_states"
        --output "joint_summary=$joint_summary"
        --output "joint_acceptance=$joint_acceptance"
        --output "joint_solver_summary=$joint_solver_summary"
        --output "acceptance_baseline=$BASELINE_PROFILE"
        --output "calibration_acceptance=$calibration_acceptance"
        --output "final_geometry=$current_geom"
        --output "final_states=$current_states"
        --output "final_summary=$current_summary"
        --output "acceptance=$current_acceptance"
        --output "experiment_summary=$RUN_DIR/experiment_summary.json"
        --output "track_chi2_comparison=$RUN_DIR/track_chi2_ndf_comparison.png"
        --output "geometry_comparison=$RUN_DIR/geometry_comparison"
    )
    if [[ "$SKIP_BLIND_VALIDATION" -eq 0 ]]; then
        manifest_outputs+=(--output "validation_acceptance=$final_validation_acceptance")
    fi
    for ((polish_index=1; polish_index<=polish_attempts; polish_index++)); do
        polish_label="${RUN_TAG}_surface_polish${polish_index}"
        manifest_outputs+=(
            --output "polish${polish_index}_geometry=$(geometry_path "$polish_label")"
            --output "polish${polish_index}_states=$(states_path "$polish_label")"
            --output "polish${polish_index}_summary=$(summary_path "$polish_label")"
            --output "polish${polish_index}_acceptance=$(acceptance_path "$polish_label")"
        )
    done
    python3 experiment_manifest.py finalize --manifest "$MANIFEST_PATH" --status completed \
        "${manifest_outputs[@]}" --metrics-file "$RUN_DIR/experiment_summary.json" >/dev/null
    MANIFEST_FINALIZED=1
fi

echo "Experiment artifacts: $RUN_DIR"
