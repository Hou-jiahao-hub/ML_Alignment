#!/usr/bin/env bash
# Level E for one frozen candidate: ACTS reconstruction -> production acceptance.
# No solver, no polish, no truth (ml_optimization_log.md Iteration 45).  The
# threshold profile is an existing frozen one, so arms are scored alike.
set -eo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$JOB_DIR"
ORIGINAL_ARGS=("$@")
eval "$(python3 alignment_artifact_layout.py --shell)"

RUN_TAG=""
EVENTS=100000
TARGET_LAYERS="17:8,24:4"
CANDIDATE_GEOM=""
CANDIDATE_SHA256=""
PROFILE=""
PROFILE_SHA256=""
BASELINE_SCORE=""
SIMULATION_DIR="$SIMULATION_ROOT"
RECONSTRUCTION_DIR="$RECONSTRUCTION_ROOT"
LOG_DIR="$SURFACE_HELIX_LOG_ROOT"
ACTS_SETUP="$ML_ALIGNMENT_ACTS_SETUP"
CONDA_SH="$ML_ALIGNMENT_CONDA_SETUP"
CONDA_ENV="pytorch"
DRY_RUN=0

die() { echo "ERROR: $*" >&2; exit 1; }
usage() {
    cat <<EOF
Usage: ./run_candidate_acceptance.sh --run-tag NAME --candidate-geom PATH --candidate-sha256 HEX
         --threshold-profile PATH --threshold-profile-sha256 HEX --expected-baseline-score JSON [options]
  --events N            ACTS reconstruction events (default: 100000)
  --dry-run             Print the commands only
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-tag) RUN_TAG="$2"; shift 2 ;;
        --candidate-geom) CANDIDATE_GEOM="$2"; shift 2 ;;
        --candidate-sha256) CANDIDATE_SHA256="$2"; shift 2 ;;
        --threshold-profile) PROFILE="$2"; shift 2 ;;
        --threshold-profile-sha256) PROFILE_SHA256="$2"; shift 2 ;;
        --expected-baseline-score) BASELINE_SCORE="$2"; shift 2 ;;
        --events) EVENTS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ -n "$RUN_TAG" && -n "$CANDIDATE_GEOM" && -n "$CANDIDATE_SHA256" && -n "$PROFILE" && \
   -n "$PROFILE_SHA256" && -n "$BASELINE_SCORE" ]] || { usage; die "missing required option"; }
[[ "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "--run-tag contains unsafe characters"
[[ "$EVENTS" =~ ^[1-9][0-9]*$ ]] || die "--events must be positive"

absolute_project_path() {
    case "$1" in
        /*) realpath -m "$1" ;;
        *) realpath -m "$ML_ALIGNMENT_REPOSITORY_ROOT/$1" ;;
    esac
}
CANDIDATE_GEOM="$(absolute_project_path "$CANDIDATE_GEOM")"
PROFILE="$(absolute_project_path "$PROFILE")"
with_target_dir() { printf '%s/%s\n' "$1" "$TARGET_LAYER_DIR"; }
MISALIGNED_STATES="$(with_target_dir "$RECONSTRUCTION_DIR/misaligned")/trackstates_kf_${TARGET_LAYER_SUFFIX}.root"
LABEL="${RUN_TAG}_schur_lm"
STATES="$(with_target_dir "$RECONSTRUCTION_DIR/aligned")/trackstates_kf_${LABEL}.root"
RUN_DIR="$(with_target_dir "$LOG_DIR")/$RUN_TAG"
BASELINE_ACCEPTANCE="$RUN_DIR/acceptance_baseline_input.json"
CANDIDATE_ACCEPTANCE="$RUN_DIR/acceptance_${LABEL}.json"
VERDICT="$RUN_DIR/level_e_verdict.json"

print_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }
run_conda() {
    local log_file="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "[conda:$CONDA_ENV]"; print_command "$@"; return 0; fi
    ( unset PYTHONHOME PYTHONPATH ROOTSYS; source "$CONDA_SH"; conda activate "$CONDA_ENV"; "$@" ) > "$log_file" 2>&1
}
run_acts() {
    local log_file="$1"; shift
    if [[ "$DRY_RUN" -eq 1 ]]; then echo "[acts]"; print_command "$@"; return 0; fi
    ( unset PYTHONHOME PYTHONPATH ROOTSYS CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
      source "$ACTS_SETUP"; "$@" ) > "$log_file" 2>&1
}
acceptance() {
    run_conda "$RUN_DIR/acceptance_$1.log" python reconstruction_acceptance.py --states "$2" --output "$3" \
        --target-layers "$TARGET_LAYERS" --threshold-profile "$PROFILE" --threshold-profile-sha256 "$PROFILE_SHA256" || true
    [[ "$DRY_RUN" -eq 1 || -f "$3" ]] || die "acceptance produced no output: $3"
}

MANIFEST_PATH="$RUN_DIR/manifest.json"
MANIFEST_FINALIZED=0
finalize_failed_manifest() {
    local exit_code="$?"
    if [[ "$DRY_RUN" -eq 0 && "$MANIFEST_FINALIZED" -eq 0 && -f "$MANIFEST_PATH" ]]; then
        python3 experiment_manifest.py finalize --manifest "$MANIFEST_PATH" --status failed \
            --metrics-json "{\"exit_code\": $exit_code, \"stage\": \"${STAGE:-preflight}\"}" >/dev/null 2>&1 || true
    fi
}
trap finalize_failed_manifest EXIT
trap 'exit 130' INT TERM HUP

if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ ! -e "$RUN_DIR" ]] || die "run tag already exists: $RUN_DIR"
    [[ ! -e "$STATES" ]] || die "reconstruction already exists: $STATES"
    [[ "$(sha256sum "$CANDIDATE_GEOM" | cut -d' ' -f1)" == "$CANDIDATE_SHA256" ]] || die "candidate SHA256 mismatch"
    [[ "$(sha256sum "$PROFILE" | cut -d' ' -f1)" == "$PROFILE_SHA256" ]] || die "threshold profile SHA256 mismatch"
    mkdir -p "$RUN_DIR"
    command_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$0" "${ORIGINAL_ARGS[@]}")"
    python3 experiment_manifest.py start --run-dir "$RUN_DIR" --run-id "$RUN_TAG" --command-json "$command_json" \
        --input "candidate_geometry=$CANDIDATE_GEOM" --input "threshold_profile=$PROFILE" \
        --input "misaligned_states=$MISALIGNED_STATES" --input "simulation_dataset=$SIMULATION_DIR" \
        --input "acts_setup=$ACTS_SETUP" --input "conda_setup=$CONDA_SH" \
        --acts-setup "$ACTS_SETUP" --conda-env "$CONDA_ENV" --conda-setup "$CONDA_SH" \
        --capture-source-snapshot >/dev/null
fi

STAGE="baseline_acceptance"
acceptance baseline_input "$MISALIGNED_STATES" "$BASELINE_ACCEPTANCE"
STAGE="reconstruction"
run_acts "$RUN_DIR/reconstruct_${LABEL}.log" \
    python alignment_odd.py reconstruction --mode aligned --num-events "$EVENTS" \
    --input-dir "$SIMULATION_DIR" --output-dir "$RECONSTRUCTION_DIR" \
    --aligned-file "$CANDIDATE_GEOM" --output-suffix "$LABEL"
[[ "$DRY_RUN" -eq 1 || -f "$STATES" ]] || die "reconstruction produced no states: $STATES"
STAGE="candidate_acceptance"
acceptance "$LABEL" "$STATES" "$CANDIDATE_ACCEPTANCE"
[[ "$DRY_RUN" -eq 0 ]] || exit 0

STAGE="verdict"
run_conda "$RUN_DIR/verdict.log" python -c '
import json, sys
from reconstruction_acceptance import acceptance_score, polish_improves
candidate, baseline = (json.load(open(path)) for path in sys.argv[1:3])
expected = json.loads(sys.argv[3])
base_score, cand_score = list(acceptance_score(baseline)), list(acceptance_score(candidate))
checks = {
    "baseline_identity": base_score == expected,
    "E1_metadata": candidate.get("schema_version") == "3.1" and candidate.get("profile_type") == "diagnostic_only"
        and candidate.get("verdict_eligible") is False and candidate.get("profile_identity_verified") is True,
    "E2_all_components_passed": bool(candidate.get("passed")) and not candidate.get("failed_components"),
    "E3_improves_on_baseline": polish_improves(cand_score, base_score),
}
json.dump({"checks": checks, "baseline_score": base_score, "candidate_score": cand_score,
           "failed_components": candidate.get("failed_components"),
           "passed": all(checks.values())}, open(sys.argv[4], "w"), indent=2)
' "$CANDIDATE_ACCEPTANCE" "$BASELINE_ACCEPTANCE" "$BASELINE_SCORE" "$VERDICT"
[[ -f "$VERDICT" ]] || die "verdict was not written"

python3 experiment_manifest.py finalize --manifest "$MANIFEST_PATH" --status completed \
    --metrics-file "$VERDICT" --output "verdict=$VERDICT" --output "candidate_acceptance=$CANDIDATE_ACCEPTANCE" \
    --output "baseline_acceptance=$BASELINE_ACCEPTANCE" --output "states=$STATES" >/dev/null
MANIFEST_FINALIZED=1
cat "$VERDICT"
