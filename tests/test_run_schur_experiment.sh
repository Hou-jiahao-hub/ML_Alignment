#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TEST_DIR/.." && pwd)"
JOB_DIR="$REPO_DIR/scripts"
SCRIPT="$JOB_DIR/run_schur_experiment.sh"
eval "$(python3 "$JOB_DIR/alignment_artifact_layout.py" --shell)"

[[ -x "$SCRIPT" ]] || {
    echo "run_schur_experiment.sh is missing or not executable" >&2
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
    > "$output" 2>&1

grep -Fq -- "--nominal-states supplies only the frozen acceptance baseline" "$output"
grep -Fq -- "[conda:pytorch]" "$output"
grep -Fq -- "[acts]" "$output"

# One Schur-LM solve through the package CLI; no surface or block-LM stage.
[[ "$(grep -Fc -- "mlalign-align" "$output")" -eq 1 ]]
[[ "$(grep -Fc -- "--track-mode" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "--joint-solver" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "ml_alignment_autograd.py" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "--alignment-gain" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "--gpu" "$output")" -eq 0 ]]
grep -Fq -- 'import\ mlalign' "$output"
[[ "$(grep -Fc -- "python calibrate_acceptance.py" "$output")" -eq 1 ]]
# baseline_input and schur_lm, both on every event against one frozen profile.
[[ "$(grep -Fc -- "python reconstruction_acceptance.py" "$output")" -eq 2 ]]
[[ "$(grep -Fc -- "--threshold-profile " "$output")" -eq 2 ]]
[[ "$(grep -Fc -- "python alignment_odd.py reconstruction" "$output")" -eq 1 ]]
grep -Fq -- "acceptance_tdd_fresh_baseline_input.json" "$output"
grep -Fq -- "aligned_transforms_tdd_fresh_schur_lm.json" "$output"
grep -Fq -- "acceptance_tdd_fresh_schur_lm.json" "$output"
grep -Fq -- "--target-layers 17:8\,24:4" "$output"
grep -Fq -- "--active-dof 0\,1\,5" "$output"
grep -Fq -- "--lm-max-iters 60" "$output"
# 200 truncated the damped Gauss-Newton seed below the 0.1% stationarity gate
# on real data; it converges at ~367 and breaks out (Iteration 26).
grep -Fq -- "--joint-smt-seed-iters 2000" "$output"
grep -Fq -- "--nominal-full-geom $GEOMETRY_ROOT/odd_transforms.json" "$output"
grep -Fq -- "$SCHUR_LM_LOG_ROOT/$TARGET_LAYER_DIR/tdd_fresh" "$output"
grep -Fq -- 'EXPECTED_TARGET_LAYERS="17:8,24:4"' "$SCRIPT"
grep -Fq -- '[[ "$TARGET_LAYERS_SPEC" == "$EXPECTED_TARGET_LAYERS" ]]' "$SCRIPT"
grep -Fq -- '--input "simulation_dataset=$SIMULATION_DIR"' "$SCRIPT"
grep -Fq -- '--input "nominal_geometry=$NOMINAL_GEOMETRY"' "$SCRIPT"
grep -Fq -- '--allow-missing-inputs' "$SCRIPT"

preflight="$(grep -n -m1 -F -- 'import\ mlalign' "$output" | cut -d: -f1)"
baseline_accept="$(grep -n -m1 -- "acceptance_tdd_fresh_baseline_input.json" "$output" | cut -d: -f1)"
joint="$(grep -n -m1 -- "mlalign-align" "$output" | cut -d: -f1)"
reconstruct="$(grep -n -m1 -- "alignment_odd.py reconstruction" "$output" | cut -d: -f1)"
accept="$(grep -n -m1 -- "acceptance_tdd_fresh_schur_lm.json" "$output" | cut -d: -f1)"
plots="$(grep -n -m1 -- "plot_residuals.py" "$output" | cut -d: -f1)"
(( preflight < baseline_accept && baseline_accept < joint && joint < reconstruct \
   && reconstruct < accept && accept < plots ))

if grep -Fq -- "--threshold 1.0" "$output"; then
    echo "workflow uses the uncalibrated absolute acceptance threshold" >&2
    exit 1
fi

if bash "$SCRIPT" --dry-run > /dev/null 2>&1; then
    echo "driver accepted a missing --run-tag" >&2
    exit 1
fi

for retired in --surface-epochs --helix-candidates --baseline-ratio; do
    if bash "$SCRIPT" --run-tag x --dry-run "$retired" 1 > /dev/null 2>&1; then
        echo "driver accepted retired option $retired" >&2
        exit 1
    fi
done
