#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TEST_DIR/.." && pwd)"
JOB_DIR="$REPO_DIR/scripts"
SCRIPT="$JOB_DIR/run_surface_helix_experiment.sh"
eval "$(python3 "$JOB_DIR/alignment_artifact_layout.py" --shell)"

[[ -x "$SCRIPT" ]] || {
    echo "run_surface_helix_experiment.sh is missing or not executable" >&2
    exit 1
}

output="$(mktemp)"
trap 'rm -f "$output"' EXIT

bash "$SCRIPT" \
    --run-tag tdd_fresh \
    --dry-run \
    --input-geom inputs/current.json \
    --states inputs/states.root \
    --summary inputs/summary.root \
    --nominal-states inputs/nominal.root \
    --baseline-ratio 1.5 \
    --helix-candidates 17:8 \
    > "$output" 2>&1

grep -Fq -- "--nominal-states supplies only the frozen acceptance baseline" "$output"
grep -Fq -- "deprecated: --baseline-ratio no longer controls the workflow" "$output"
grep -Fq -- "deprecated: --helix-candidates no longer selects layers" "$output"
grep -Fq -- "[conda:pytorch]" "$output"
grep -Fq -- "[acts]" "$output"

[[ "$(grep -Fc -- "--track-mode surface" "$output")" -eq 4 ]]
[[ "$(grep -Fc -- "--track-mode free" "$output")" -eq 1 ]]
[[ "$(grep -Fc -- "python calibrate_acceptance.py" "$output")" -eq 1 ]]
[[ "$(grep -Fc -- "python reconstruction_acceptance.py" "$output")" -eq 5 ]]
[[ "$(grep -Fc -- "--event-partition train" "$output")" -eq 5 ]]
[[ "$(grep -Fc -- "--event-partition calibration" "$output")" -eq 4 ]]
[[ "$(grep -Fc -- "--event-partition validation" "$output")" -eq 1 ]]
[[ "$(grep -Fc -- "--threshold-profile" "$output")" -eq 5 ]]
grep -Fq -- "tdd_fresh_surface_warm1" "$output"
grep -Fq -- "tdd_fresh_surface_warm2" "$output"
grep -Fq -- "tdd_fresh_joint_free" "$output"
grep -Fq -- "tdd_fresh_surface_polish1" "$output"
grep -Fq -- "tdd_fresh_surface_polish2" "$output"
grep -Fq -- "--target-layers 17:8\,24:4" "$output"
grep -Fq -- "--active-dof 0\,1\,5" "$output"
grep -Fq -- "--joint-seed-mode non-target-measurement" "$output"
grep -Fq -- "--joint-solver block-lm" "$output"
grep -Fq -- "--lm-max-iters 20" "$output"
grep -Fq -- "--alignment-gain 0.2" "$output"
grep -Fq -- 'EXPECTED_TARGET_LAYERS="17:8,24:4"' "$SCRIPT"
grep -Fq -- '[[ "$TARGET_LAYERS_SPEC" == "$EXPECTED_TARGET_LAYERS" ]]' "$SCRIPT"
grep -Fq -- '--input "simulation_dataset=$SIMULATION_DIR"' "$SCRIPT"
grep -Fq -- '--input "nominal_geometry=$GEOMETRY_ROOT/odd_transforms.json"' "$SCRIPT"
grep -Fq -- '--allow-missing-inputs' "$SCRIPT"

if grep -Fq -- "ml_alignment_helix.py" "$output"; then
    echo "new workflow still calls nominal assessment or frozen-helix align" >&2
    exit 1
fi

warm1="$(grep -n -m1 -- "tdd_fresh_surface_warm1" "$output" | cut -d: -f1)"
warm2="$(grep -n -m1 -- "tdd_fresh_surface_warm2" "$output" | cut -d: -f1)"
warm2_accept="$(grep -n -m1 -- "acceptance_tdd_fresh_surface_warm2.json" "$output" | cut -d: -f1)"
joint="$(grep -n -m1 -- "tdd_fresh_joint_free" "$output" | cut -d: -f1)"
accept="$(grep -n -m1 -- "acceptance_tdd_fresh_joint_free.json" "$output" | cut -d: -f1)"
polish="$(grep -n -m1 -- "tdd_fresh_surface_polish1" "$output" | cut -d: -f1)"
validation="$(grep -n -m1 -- "acceptance_tdd_fresh_validation.json" "$output" | cut -d: -f1)"
(( warm1 < warm2 && warm2 < warm2_accept && warm2_accept < joint && joint < accept && accept < polish && polish < validation ))

if grep -Fq -- "--threshold 1.0" "$output"; then
    echo "new workflow still uses the uncalibrated absolute acceptance threshold" >&2
    exit 1
fi

if bash "$SCRIPT" --dry-run > /dev/null 2>&1; then
    echo "driver accepted a missing --run-tag" >&2
    exit 1
fi

# --skip-blind-validation must leave the one-shot validation split untouched.
skip_output="$(mktemp)"
trap 'rm -f "$output" "$skip_output"' EXIT

bash "$SCRIPT" \
    --run-tag tdd_skip \
    --dry-run \
    --skip-blind-validation \
    --input-geom inputs/current.json \
    --states inputs/states.root \
    --summary inputs/summary.root \
    --nominal-states inputs/nominal.root \
    > "$skip_output" 2>&1

grep -Fq -- "the validation split stays untouched" "$skip_output"
[[ "$(grep -Fc -- "--event-partition validation" "$skip_output")" -eq 0 ]]
[[ "$(grep -Fc -- "acceptance_tdd_skip_validation.json" "$skip_output")" -eq 0 ]]
# every calibration-side acceptance still runs
[[ "$(grep -Fc -- "--event-partition calibration" "$skip_output")" -eq 4 ]]
[[ "$(grep -Fc -- "python reconstruction_acceptance.py" "$skip_output")" -eq 4 ]]
