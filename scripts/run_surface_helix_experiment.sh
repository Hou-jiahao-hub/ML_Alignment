#!/usr/bin/env bash
# Independent surface -> frozen-helix -> ACTS validation experiment.
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
    eval "$(python3 alignment_artifact_layout.py --config "$CONFIG_PATH" --shell)"
else
    eval "$(python3 alignment_artifact_layout.py --shell)"
fi

RUN_TAG=""
EVENTS=100000
TARGET_LAYERS="$TARGET_LAYERS_SPEC"
HELIX_CANDIDATES="$TARGET_LAYERS_SPEC"
ACTIVE_DOF="$ACTIVE_DOF_SPEC"
SURFACE_EPOCHS=300
SURFACE_REL_TOL=0.05
SURFACE_PATIENCE=3
SURFACE_CHI2_TARGET=1.0
MAX_SURFACE_ITERS=15
BASELINE_RATIO=1.2
LAMBDA_L2=1e-5
ALIGNMENT_DIR="$ALIGNMENT_ROOT"
SIMULATION_DIR="$SIMULATION_ROOT"
RECONSTRUCTION_DIR="$RECONSTRUCTION_ROOT"
LOG_DIR="$SURFACE_HELIX_LOG_ROOT"
INPUT_GEOM=""
INPUT_STATES=""
INPUT_SUMMARY=""
NOMINAL_STATES=""
ACTS_SETUP="$ML_ALIGNMENT_ACTS_ROOT/setup_acts.sh"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="pytorch"
DRY_RUN=0

usage() {
    cat <<EOF
Usage: ./run_surface_helix_experiment.sh --run-tag NAME [options]

Required:
  --run-tag NAME             Unique namespace for all new artifacts

Options:
  --events N                 ACTS reconstruction events (default: 100000)
  --helix-candidates SPEC    Explicit candidate layers (default: $TARGET_LAYERS_SPEC)
  --surface-rel-tol VALUE    Relative chi2 plateau tolerance (default: 0.05)
  --surface-patience N       Consecutive small changes required (default: 3)
  --surface-chi2-target V    Absolute chi2/component target (default: 1.0)
  --max-surface-iters N      Hard outer-loop limit (default: 15)
  --baseline-ratio VALUE     Candidate/nominal RMS layer gate (default: 1.2)
  --input-geom PATH          Initial misaligned geometry
  --states PATH              Initial misaligned trackstates
  --summary PATH             Initial misaligned tracksummary
  --nominal-states PATH      Nominal trackstates baseline
  --alignment-dir PATH       Alignment output base
  --simulation-dir PATH      Existing simulation input
  --reconstruction-dir PATH  Reconstruction output base
  --log-dir PATH             Experiment log base
  --acts-setup PATH          ACTS environment setup script
  --conda-sh PATH            Conda shell hook
  --conda-env NAME           Conda environment name
  --dry-run                  Print a representative complete workflow
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) shift 2 ;;
        --config=*) shift ;;
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --helix-candidates) HELIX_CANDIDATES="$2"; shift 2 ;;
        --surface-rel-tol) SURFACE_REL_TOL="$2"; shift 2 ;;
        --surface-patience) SURFACE_PATIENCE="$2"; shift 2 ;;
        --surface-chi2-target) SURFACE_CHI2_TARGET="$2"; shift 2 ;;
        --max-surface-iters) MAX_SURFACE_ITERS="$2"; shift 2 ;;
        --baseline-ratio) BASELINE_RATIO="$2"; shift 2 ;;
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
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$RUN_TAG" ]] || die "--run-tag is required"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "--run-tag contains unsafe characters"
[[ "$EVENTS" =~ ^[0-9]+$ ]] || die "--events must be a non-negative integer"
[[ "$SURFACE_PATIENCE" =~ ^[0-9]+$ ]] || die "--surface-patience must be an integer"
[[ "$MAX_SURFACE_ITERS" =~ ^[1-9][0-9]*$ ]] || die "--max-surface-iters must be positive"
[[ "$TARGET_LAYERS" == "$TARGET_LAYERS_SPEC" ]] || die "target layer configuration drift"

with_target_dir() {
    if [[ "$(basename "$1")" == "$TARGET_LAYER_DIR" ]]; then printf '%s\n' "$1";
    else printf '%s/%s\n' "$1" "$TARGET_LAYER_DIR"; fi
}

ALIGNMENT_TARGET_DIR="$(with_target_dir "$ALIGNMENT_DIR")"
NOMINAL_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/nominal")"
MISALIGNED_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/misaligned")"
ALIGNED_RECON_DIR="$(with_target_dir "$RECONSTRUCTION_DIR/aligned")"
RUN_LOG_DIR="$(with_target_dir "$LOG_DIR")/$RUN_TAG"

[[ -n "$INPUT_GEOM" ]] || INPUT_GEOM="$ALIGNMENT_TARGET_DIR/$MISALIGNMENT_JSON"
[[ -n "$INPUT_STATES" ]] || INPUT_STATES="$MISALIGNED_RECON_DIR/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"
[[ -n "$INPUT_SUMMARY" ]] || INPUT_SUMMARY="$MISALIGNED_RECON_DIR/tracksummary_kf_${TARGET_LAYER_SUFFIX}.root"
[[ -n "$NOMINAL_STATES" ]] || NOMINAL_STATES="$NOMINAL_RECON_DIR/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

run_conda() {
    local log_file="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then print_command "$@"; return 0; fi
    (
        unset PYTHONHOME PYTHONPATH ROOTSYS
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
        "$@"
    ) > "$log_file" 2>&1
}

run_acts() {
    local log_file="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then print_command "$@"; return 0; fi
    (
        unset PYTHONHOME PYTHONPATH ROOTSYS CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
        source "$ACTS_SETUP"
        "$@"
    ) > "$log_file" 2>&1
}

relative_to_alignment_output() { realpath -m --relative-to="$ALIGNMENT_DIR" "$1"; }
require_file() { [[ -f "$1" ]] || die "missing required file: $1"; }

echo "surface relative tolerance : $SURFACE_REL_TOL"
echo "surface patience           : $SURFACE_PATIENCE"
echo "surface chi2 target        : $SURFACE_CHI2_TARGET"
echo "maximum surface iterations : $MAX_SURFACE_ITERS"
echo "helix candidates           : $HELIX_CANDIDATES"
echo "active DOF                 : $ACTIVE_DOF"
echo "run tag                    : $RUN_TAG"

if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ ! -e "$RUN_LOG_DIR" ]] || die "run tag already exists: $RUN_LOG_DIR"
    for ((index=0; index<MAX_SURFACE_ITERS; index++)); do
        [[ ! -e "$ALIGNMENT_TARGET_DIR/aligned_transforms_${RUN_TAG}_surface_iter${index}.json" ]] \
            || die "surface geometry already exists for run tag $RUN_TAG"
        [[ ! -e "$ALIGNED_RECON_DIR/trackstates_kf_${RUN_TAG}_surface_iter${index}.root" ]] \
            || die "surface reconstruction already exists for run tag $RUN_TAG"
    done
    [[ ! -e "$ALIGNMENT_TARGET_DIR/aligned_transforms_${RUN_TAG}_helix.json" ]] \
        || die "helix geometry already exists for run tag $RUN_TAG"
    [[ ! -e "$ALIGNED_RECON_DIR/trackstates_kf_${RUN_TAG}_helix.root" ]] \
        || die "helix reconstruction already exists for run tag $RUN_TAG"
    require_file "$INPUT_GEOM"
    require_file "$INPUT_STATES"
    require_file "$INPUT_SUMMARY"
    require_file "$NOMINAL_STATES"
    require_file "$ACTS_SETUP"
    require_file "$CONDA_SH"
    mkdir -p "$RUN_LOG_DIR" "$ALIGNMENT_TARGET_DIR" "$ALIGNED_RECON_DIR"
fi

current_geom="$INPUT_GEOM"
current_states="$INPUT_STATES"
current_summary="$INPUT_SUMMARY"
chi2_history=()
stop_reason=""

for ((iteration=0; iteration<MAX_SURFACE_ITERS; iteration++)); do
    label="${RUN_TAG}_surface_iter${iteration}"
    output_geom="$ALIGNMENT_TARGET_DIR/aligned_transforms_${label}.json"
    next_states="$ALIGNED_RECON_DIR/trackstates_kf_${label}.root"
    next_summary="$ALIGNED_RECON_DIR/tracksummary_kf_${label}.root"
    surface_log="$RUN_LOG_DIR/${label}.log"
    recon_log="$RUN_LOG_DIR/recon_${label}.log"

    surface_cmd=(
        python ml_alignment_autograd.py
        --input-geom "$current_geom" --states "$current_states" --summary "$current_summary"
        --output-geom "$output_geom" --target-layers "$TARGET_LAYERS"
        --active-dof "$ACTIVE_DOF" --track-mode surface --epochs "$SURFACE_EPOCHS"
        --lambda-l2 "$LAMBDA_L2" --iter "$iteration" --no-plot
    )
    run_conda "$surface_log" "${surface_cmd[@]}"

    aligned_arg="$(relative_to_alignment_output "$output_geom")"
    recon_cmd=(
        python alignment_odd.py reconstruction --mode aligned --num-events "$EVENTS"
        --input-dir "$SIMULATION_DIR" --output-dir "$RECONSTRUCTION_DIR"
        --aligned-file "$aligned_arg" --output-suffix "$label"
    )
    run_acts "$recon_log" "${recon_cmd[@]}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        current_geom="$output_geom"; current_states="$next_states"; current_summary="$next_summary"
        break
    fi
    require_file "$output_geom"; require_file "$next_states"; require_file "$next_summary"
    chi2="$(sed -n 's/.*| Chi2: \([^ ]*\).*/\1/p' "$surface_log" | tail -1)"
    [[ "$chi2" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || die "cannot parse final Chi2 from $surface_log"
    chi2_history+=("$chi2")
    history_csv="$(IFS=,; echo "${chi2_history[*]}")"
    stop_reason="$(python3 -c 'import sys; from ml_alignment_helix import surface_stop_reason; h=[float(x) for x in sys.argv[1].split(",") if x]; print(surface_stop_reason(h, chi2_target=float(sys.argv[2]), relative_tolerance=float(sys.argv[3]), patience=int(sys.argv[4]), max_iterations=int(sys.argv[5])) or "")' "$history_csv" "$SURFACE_CHI2_TARGET" "$SURFACE_REL_TOL" "$SURFACE_PATIENCE" "$MAX_SURFACE_ITERS")"
    current_geom="$output_geom"; current_states="$next_states"; current_summary="$next_summary"
    [[ -z "$stop_reason" ]] || break
done

surface_assessment="$RUN_LOG_DIR/surface_assessment.json"
assess_cmd=(
    python ml_alignment_helix.py assess --nominal-states "$NOMINAL_STATES"
    --candidate-states "$current_states" --target-layers "$HELIX_CANDIDATES"
    --max-ratio "$BASELINE_RATIO" --output-summary "$surface_assessment"
)
run_conda "$RUN_LOG_DIR/surface_assessment.log" "${assess_cmd[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    failed_layers="$HELIX_CANDIDATES"
else
    require_file "$surface_assessment"
    failed_layers="$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["failed_layers"]))' "$surface_assessment")"
fi

if [[ -z "$failed_layers" ]]; then
    echo "No candidate layer exceeds the nominal-relative RMS gate; helix stage skipped."
    if [[ "$DRY_RUN" -eq 0 ]]; then
        history_csv="$(IFS=,; echo "${chi2_history[*]}")"
        python3 -c 'import json,sys; json.dump({"surface_chi2_history":[float(x) for x in sys.argv[1].split(",") if x],"surface_stop_reason":sys.argv[2],"helix_layers":[],"helix_skipped":True,"passed":True},open(sys.argv[3],"w"),indent=2)' "$history_csv" "$stop_reason" "$RUN_LOG_DIR/experiment_summary.json"
    fi
    exit 0
fi

helix_label="${RUN_TAG}_helix"
helix_geometry="$ALIGNMENT_TARGET_DIR/aligned_transforms_${helix_label}.json"
helix_summary="$RUN_LOG_DIR/helix_alignment_summary.json"
helix_cmd=(
    python ml_alignment_helix.py align --input-geometry "$current_geom" --states "$current_states"
    --nominal-geometry "$GEOMETRY_ROOT/odd_transforms.json" --output-geometry "$helix_geometry"
    --output-summary "$helix_summary" --target-layers "$failed_layers"
    --active-dof "$ACTIVE_DOF"
)
run_conda "$RUN_LOG_DIR/helix_alignment.log" "${helix_cmd[@]}"

helix_states="$ALIGNED_RECON_DIR/trackstates_kf_${helix_label}.root"
helix_recon_cmd=(
    python alignment_odd.py reconstruction --mode aligned --num-events "$EVENTS"
    --input-dir "$SIMULATION_DIR" --output-dir "$RECONSTRUCTION_DIR"
    --aligned-file "$(relative_to_alignment_output "$helix_geometry")" --output-suffix "$helix_label"
)
run_acts "$RUN_LOG_DIR/recon_${helix_label}.log" "${helix_recon_cmd[@]}"

final_assessment="$RUN_LOG_DIR/helix_assessment.json"
final_assess_cmd=(
    python ml_alignment_helix.py assess --nominal-states "$NOMINAL_STATES"
    --candidate-states "$helix_states" --target-layers "$failed_layers"
    --max-ratio "$BASELINE_RATIO" --output-summary "$final_assessment"
)
run_conda "$RUN_LOG_DIR/helix_assessment.log" "${final_assess_cmd[@]}"

if [[ "$DRY_RUN" -eq 0 ]]; then
    require_file "$helix_geometry"; require_file "$helix_states"; require_file "$final_assessment"
    history_csv="$(IFS=,; echo "${chi2_history[*]}")"
    python3 -c 'import json,sys; surface=json.load(open(sys.argv[1])); final=json.load(open(sys.argv[2])); layers=sys.argv[3].split(","); ratios={}; improved=[]; degraded=[]
for layer in layers:
  ratios[layer]={}
  for comp in ("loc0","loc1"):
    s=surface["candidate"][layer].get(comp+"_rmse_mm"); f=final["candidate"][layer].get(comp+"_rmse_mm")
    if s is None or f is None: continue
    ratio=f/max(s,1e-12); ratios[layer][comp]=ratio
    improved.append(1-ratio); degraded.append(ratio)
passed=bool(improved) and max(improved)>=0.10 and all(x<=1.05 for x in degraded)
out={"surface_chi2_history":[float(x) for x in sys.argv[4].split(",") if x],"surface_stop_reason":sys.argv[5],"helix_layers":layers,"helix_vs_surface_ratio":ratios,"passed":passed}; json.dump(out,open(sys.argv[6],"w"),indent=2)' "$surface_assessment" "$final_assessment" "$failed_layers" "$history_csv" "$stop_reason" "$RUN_LOG_DIR/experiment_summary.json"
fi

echo "Experiment artifacts: $RUN_LOG_DIR"
