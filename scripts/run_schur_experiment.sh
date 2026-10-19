#!/usr/bin/env bash
# Schur-LM joint alignment -> ACTS re-reconstruction -> acceptance guard.
#
# The input geometry is scored once up front; the Schur-LM geometry is kept
# only if its acceptance score improves on it.  Every stage reads the full
# nominal event set, so acceptance is a diagnostic_only same-sample guard.
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
JOINT_SMT_SEED_ITERS=2000
LM_MAX_ITERS=60
MAX_TRACKS=0
ALIGNMENT_DIR="$ALIGNMENT_ROOT"
SIMULATION_DIR="$SIMULATION_ROOT"
RECONSTRUCTION_DIR="$RECONSTRUCTION_ROOT"
LOG_DIR="$SCHUR_LM_LOG_ROOT"
INPUT_GEOM=""
INPUT_STATES=""
INPUT_SUMMARY=""
NOMINAL_STATES=""
ACTS_SETUP="$ML_ALIGNMENT_ACTS_SETUP"
CONDA_SH="$ML_ALIGNMENT_CONDA_SETUP"
CONDA_ENV="pytorch"
DRY_RUN=0
BOOTSTRAP_SEED=6006

usage() {
    cat <<EOF
Usage: ./run_schur_experiment.sh --run-tag NAME [options]

Required:
  --run-tag NAME             Unique namespace for all new artifacts

Options:
  --events N                 ACTS reconstruction events (default: 100000)
  --joint-smt-seed-iters N   Non-target SMT seed iterations (default: 2000).
                             A cap, not a cost: the damped Gauss-Newton seed
                             converges at ~367 iterations and breaks out, so
                             200 only truncated it below the 0.1% stationarity
                             gate.  See ml_optimization_log.md Iteration 26.
  --lm-max-iters N           Schur-LM outer iterations (default: 60, the
                             pre-registered budget)
  --max-tracks N             Limit tracks; 0 keeps all (default: 0)
  --input-geom PATH          Initial misaligned geometry
  --states PATH              Initial misaligned trackstates
  --summary PATH             Initial misaligned tracksummary (plots only)
  --alignment-dir PATH       Alignment output base
  --simulation-dir PATH      Existing simulation input
  --reconstruction-dir PATH  Reconstruction output base
  --log-dir PATH             Experiment output base
  --acts-setup PATH          ACTS environment setup script
  --conda-sh PATH            Conda shell hook
  --conda-env NAME           Conda environment name (needs mlalign installed)
  --nominal-states PATH      Nominal states used only to freeze acceptance thresholds
  --bootstrap-seed N         Nominal-states bootstrap seed for the frozen
                             acceptance threshold (default: 6006)
  --dry-run                  Print the commands without running them
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) shift 2 ;;
        --config=*) shift ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --joint-smt-seed-iters) JOINT_SMT_SEED_ITERS="$2"; shift 2 ;;
        --lm-max-iters) LM_MAX_ITERS="$2"; shift 2 ;;
        --max-tracks) MAX_TRACKS="$2"; shift 2 ;;
        --input-geom) INPUT_GEOM="$2"; shift 2 ;;
        --states) INPUT_STATES="$2"; shift 2 ;;
        --summary) INPUT_SUMMARY="$2"; shift 2 ;;
        --nominal-states) NOMINAL_STATES="$2"; shift 2 ;;
        --alignment-dir) ALIGNMENT_DIR="$2"; shift 2 ;;
        --simulation-dir) SIMULATION_DIR="$2"; shift 2 ;;
        --reconstruction-dir) RECONSTRUCTION_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --acts-setup) ACTS_SETUP="$2"; shift 2 ;;
        --conda-sh) CONDA_SH="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --bootstrap-seed) BOOTSTRAP_SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$RUN_TAG" ]] || die "--run-tag is required"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "--run-tag contains unsafe characters"
[[ "$EVENTS" =~ ^[1-9][0-9]*$ ]] || die "--events must be positive"
[[ "$JOINT_SMT_SEED_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--joint-smt-seed-iters must be positive"
[[ "$LM_MAX_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--lm-max-iters must be positive"
[[ "$MAX_TRACKS" =~ ^[0-9]+$ ]] || die "--max-tracks must be non-negative"
[[ "$BOOTSTRAP_SEED" =~ ^[0-9]+$ ]] || die "--bootstrap-seed must be non-negative"
[[ "$TARGET_LAYERS_SPEC" == "$EXPECTED_TARGET_LAYERS" ]] || \
    die "configured target layers must remain $EXPECTED_TARGET_LAYERS"
[[ "$ACTIVE_DOF" == "0,1,5" ]] || die "this workflow requires active DOF 0,1,5"

[[ -z "$NOMINAL_STATES" ]] || echo "note: --nominal-states supplies only the frozen acceptance baseline" >&2

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
NOMINAL_GEOMETRY="$GEOMETRY_ROOT/odd_transforms.json"
[[ -n "$INPUT_GEOM" ]] || INPUT_GEOM="$ALIGNMENT_TARGET_DIR/$MISALIGNMENT_JSON"
[[ -n "$INPUT_STATES" ]] || INPUT_STATES="$MISALIGNED_RECON_DIR/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"
[[ -n "$INPUT_SUMMARY" ]] || INPUT_SUMMARY="$MISALIGNED_RECON_DIR/tracksummary_kf_${TARGET_LAYER_SUFFIX}.root"
[[ -n "$NOMINAL_STATES" ]] || NOMINAL_STATES="$(with_target_dir "$RECONSTRUCTION_DIR/nominal")/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

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

joint_label="${RUN_TAG}_schur_lm"
joint_geom="$(geometry_path "$joint_label")"
joint_states="$(states_path "$joint_label")"
joint_summary="$(summary_path "$joint_label")"
joint_acceptance="$(acceptance_path "$joint_label")"
joint_solver_summary="$RUN_DIR/schur_lm_summary.json"

echo "solver                    : schur-lm"
echo "target layers             : $TARGET_LAYERS"
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
        failure_metrics="$(python3 -c 'import json,os,sys; joint_path,acceptance_path=sys.argv[1:3]; metrics={"exit_code":int(sys.argv[3]),"stop_reason":"unexpected_exit","stage":sys.argv[4]}; metrics["schur_lm"]=json.load(open(joint_path)) if os.path.isfile(joint_path) else None; metrics["latest_acceptance"]=json.load(open(acceptance_path)) if acceptance_path and os.path.isfile(acceptance_path) else None; print(json.dumps(metrics,allow_nan=False))' \
            "$joint_solver_summary" "${current_acceptance:-}" "$exit_code" "$CURRENT_STAGE" 2>/dev/null || \
            printf '{"exit_code":%s,"stop_reason":"unexpected_exit","stage":"%s"}' "$exit_code" "$CURRENT_STAGE")"
        [[ ! -f "$joint_geom" ]] || failure_outputs+=(--output "schur_lm_geometry=$joint_geom")
        [[ ! -f "$joint_states" ]] || failure_outputs+=(--output "schur_lm_states=$joint_states")
        [[ ! -f "$joint_summary" ]] || failure_outputs+=(--output "schur_lm_tracksummary=$joint_summary")
        [[ ! -f "$joint_acceptance" ]] || failure_outputs+=(--output "schur_lm_acceptance=$joint_acceptance")
        python3 experiment_manifest.py finalize --manifest "$MANIFEST_PATH" \
            --status failed "${failure_outputs[@]}" --metrics-json "$failure_metrics" \
            >/dev/null 2>&1 || true
    fi
}
trap finalize_failed_manifest EXIT
trap 'exit 130' INT TERM HUP

if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ ! -e "$RUN_DIR" ]] || die "run tag already exists: $RUN_DIR"
    [[ ! -e "$joint_geom" ]] || die "geometry already exists: $joint_geom"
    [[ ! -e "$joint_states" ]] || die "reconstruction already exists: $joint_states"
    mkdir -p "$RUN_DIR" "$ALIGNMENT_TARGET_DIR" "$ALIGNED_RECON_DIR"
    command_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$0" "${ORIGINAL_ARGS[@]}")"
    python3 experiment_manifest.py start --run-dir "$RUN_DIR" --run-id "$RUN_TAG" \
        --allow-missing-inputs \
        --command-json "$command_json" --input "geometry=$INPUT_GEOM" \
        --input "states=$INPUT_STATES" --input "summary=$INPUT_SUMMARY" \
        --input "nominal_states=$NOMINAL_STATES" \
        --input "simulation_dataset=$SIMULATION_DIR" \
        --input "nominal_geometry=$NOMINAL_GEOMETRY" \
        --input "acts_setup=$ACTS_SETUP" --input "conda_setup=$CONDA_SH" \
        --input "odd_digi_config=$ML_ALIGNMENT_ACTS_DIGI_CONFIG" \
        --input "odd_material_map=$ML_ALIGNMENT_ODD_MATERIAL_MAP" \
        --acts-setup "$ACTS_SETUP" --conda-env "$CONDA_ENV" --conda-setup "$CONDA_SH" \
        --capture-source-snapshot \
        "${CONFIG_ARGS[@]}" >/dev/null
    require_file "$INPUT_GEOM"
    require_file "$INPUT_STATES"
    require_file "$INPUT_SUMMARY"
    require_file "$NOMINAL_STATES"
    require_file "$NOMINAL_GEOMETRY"
    [[ -d "$SIMULATION_DIR" ]] || die "missing simulation directory: $SIMULATION_DIR"
    require_file "$ACTS_SETUP"
    require_file "$CONDA_SH"
    require_file "$ML_ALIGNMENT_ACTS_DIGI_CONFIG"
    require_file "$ML_ALIGNMENT_ODD_MATERIAL_MAP"
fi
run_conda "$RUN_DIR/preflight.log" python -c "import mlalign" || \
    die "conda env $CONDA_ENV cannot import mlalign; run 'pip install -e .' there"

CURRENT_STAGE="acceptance_baseline"
run_conda "$RUN_DIR/acceptance_baseline.log" \
    python calibrate_acceptance.py "${CONFIG_ARGS[@]}" --states "$NOMINAL_STATES" \
    --output "$BASELINE_PROFILE" --target-layers "$TARGET_LAYERS" \
    --bootstrap-seed "$BOOTSTRAP_SEED"
if [[ "$DRY_RUN" -eq 0 ]]; then
    require_file "$BASELINE_PROFILE"
    # Computed once, right after the profile is frozen, and threaded into
    # every acceptance call below: a profile edited after this point (by any
    # means, not only the fields _verify_profile_identity already checks)
    # must fail closed rather than silently score against a changed baseline.
    PROFILE_SHA256="$(sha256sum "$BASELINE_PROFILE" | cut -d' ' -f1)"
else
    PROFILE_SHA256="dry-run-placeholder"
fi

reconstruct() {
    local label="$1" geometry="$2"
    run_acts "$RUN_DIR/reconstruct_${label}.log" \
        python alignment_odd.py reconstruction --mode aligned --num-events "$EVENTS" \
        --input-dir "$SIMULATION_DIR" --output-dir "$RECONSTRUCTION_DIR" \
        --aligned-file "$geometry" --output-suffix "$label"
}

run_acceptance() {
    local label="$1" states="$2" output="$3"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        run_conda "$RUN_DIR/acceptance_${label}.log" \
            python reconstruction_acceptance.py "${CONFIG_ARGS[@]}" --states "$states" --output "$output" \
            --target-layers "$TARGET_LAYERS" --threshold-profile "$BASELINE_PROFILE" \
            --threshold-profile-sha256 "$PROFILE_SHA256"
        return 0
    fi
    set +e
    run_conda "$RUN_DIR/acceptance_${label}.log" \
        python reconstruction_acceptance.py "${CONFIG_ARGS[@]}" --states "$states" --output "$output" \
        --target-layers "$TARGET_LAYERS" --threshold-profile "$BASELINE_PROFILE" \
        --threshold-profile-sha256 "$PROFILE_SHA256"
    local status="$?"
    set -e
    require_file "$output"
    return "$status"
}

# The guard needs an arm to improve on, so the input geometry is scored once.
# Its reconstruction already exists -- it is what --states/--summary point at --
# so this costs one acceptance call, not a run.
baseline_label="${RUN_TAG}_baseline_input"
baseline_acceptance="$(acceptance_path "$baseline_label")"
CURRENT_STAGE="baseline_acceptance"
run_acceptance "$baseline_label" "$INPUT_STATES" "$baseline_acceptance" || true

CURRENT_STAGE="schur_lm"
run_conda "$RUN_DIR/${joint_label}.log" \
    mlalign-align --input-geom "$INPUT_GEOM" --states "$INPUT_STATES" \
    --nominal-full-geom "$NOMINAL_GEOMETRY" --output-geom "$joint_geom" \
    --target-layers "$TARGET_LAYERS" --active-dof "$ACTIVE_DOF" \
    --joint-smt-seed-iters "$JOINT_SMT_SEED_ITERS" --lm-max-iters "$LM_MAX_ITERS" \
    --max-tracks "$MAX_TRACKS" --output-summary "$joint_solver_summary"
reconstruct "$joint_label" "$joint_geom"
CURRENT_STAGE="schur_lm_acceptance"
run_acceptance "$joint_label" "$joint_states" "$joint_acceptance" || true

current_geom="$joint_geom"
current_states="$joint_states"
current_summary="$joint_summary"
current_acceptance="$joint_acceptance"
joint_adopted=1
stop_reason="schur_lm_adopted"

if [[ "$DRY_RUN" -eq 0 ]]; then
    joint_improved="$(python3 -c 'import json,sys; from reconstruction_acceptance import acceptance_score,polish_improves; joint=json.load(open(sys.argv[1])); baseline=json.load(open(sys.argv[2])); print("1" if polish_improves(acceptance_score(joint),acceptance_score(baseline)) else "0")' "$joint_acceptance" "$baseline_acceptance")"
    joint_effective="$(python3 -c 'import json,sys; from write_experiment_summary import joint_solver_effective; print("1" if joint_solver_effective(json.load(open(sys.argv[1]))) else "0")' "$joint_solver_summary")"
    if [[ "$joint_improved" == "0" || "$joint_effective" == "0" ]]; then
        current_geom="$INPUT_GEOM"
        current_states="$INPUT_STATES"
        current_summary="$INPUT_SUMMARY"
        current_acceptance="$baseline_acceptance"
        joint_adopted=0
        stop_reason="schur_lm_not_improved"
    fi
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
    python compare_geometries.py "${CONFIG_ARGS[@]}" --nominal "$NOMINAL_GEOMETRY" \
    --misaligned "$INPUT_GEOM" --aligned "$current_geom" --output-dir "$RUN_DIR/geometry_comparison"

if [[ "$DRY_RUN" -eq 0 ]]; then
    CURRENT_STAGE="finalize"
    python3 write_experiment_summary.py \
        --run-tag "$RUN_TAG" \
        --joint-summary "$joint_solver_summary" \
        --joint-adopted "$joint_adopted" \
        --final-acceptance "$current_acceptance" \
        --acceptance-baseline "$BASELINE_PROFILE" \
        --final-geometry "$current_geom" \
        --final-trackstates "$current_states" \
        --final-tracksummary "$current_summary" \
        --selection-stop-reason "$stop_reason" \
        --output "$RUN_DIR/experiment_summary.json"
    python3 experiment_manifest.py finalize --manifest "$MANIFEST_PATH" --status completed \
        --output "baseline_input_acceptance=$baseline_acceptance" \
        --output "schur_lm_geometry=$joint_geom" \
        --output "schur_lm_states=$joint_states" \
        --output "schur_lm_tracksummary=$joint_summary" \
        --output "schur_lm_acceptance=$joint_acceptance" \
        --output "schur_lm_summary=$joint_solver_summary" \
        --output "acceptance_baseline=$BASELINE_PROFILE" \
        --output "final_geometry=$current_geom" \
        --output "final_states=$current_states" \
        --output "final_summary=$current_summary" \
        --output "acceptance=$current_acceptance" \
        --output "experiment_summary=$RUN_DIR/experiment_summary.json" \
        --output "track_chi2_comparison=$RUN_DIR/track_chi2_ndf_comparison.png" \
        --output "geometry_comparison=$RUN_DIR/geometry_comparison" \
        --metrics-file "$RUN_DIR/experiment_summary.json" >/dev/null
    MANIFEST_FINALIZED=1
fi

echo "Experiment artifacts: $RUN_DIR"
