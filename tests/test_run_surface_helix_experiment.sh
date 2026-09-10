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

# Only the two guarded polish candidates use the surface solver now; the two
# surface warm starts were removed once joint-only was shown to reproduce their
# geometry and acceptance end to end.
[[ "$(grep -Fc -- "--track-mode surface" "$output")" -eq 2 ]]
[[ "$(grep -Fc -- "--track-mode free" "$output")" -eq 1 ]]
[[ "$(grep -Fc -- "python calibrate_acceptance.py" "$output")" -eq 1 ]]
# Fits and scores every event: baseline_input, joint_free, and the two
# surface-polish rounds -- there is no separate held-out validation call.
[[ "$(grep -Fc -- "python reconstruction_acceptance.py" "$output")" -eq 4 ]]
[[ "$(grep -Fc -- "--event-partition" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "--event-split-seed" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "--split-seed" "$output")" -eq 0 ]]
[[ "$(grep -Fc -- "--threshold-profile" "$output")" -eq 4 ]]
if grep -Fq -- "tdd_fresh_surface_warm" "$output"; then
    echo "workflow still runs a surface warm start" >&2
    exit 1
fi
grep -Fq -- "acceptance_tdd_fresh_baseline_input.json" "$output"
grep -Fq -- "tdd_fresh_joint_free" "$output"
grep -Fq -- "tdd_fresh_surface_polish1" "$output"
grep -Fq -- "tdd_fresh_surface_polish2" "$output"
grep -Fq -- "--target-layers 17:8\,24:4" "$output"
grep -Fq -- "--active-dof 0\,1\,5" "$output"
grep -Fq -- "--joint-seed-mode non-target-measurement" "$output"
grep -Fq -- "--joint-solver block-lm" "$output"
grep -Fq -- "--lm-max-iters 60" "$output"
# 200 truncated the damped Gauss-Newton seed below the 0.1% stationarity gate
# on real data; it converges at ~367 and breaks out (Iteration 26).
grep -Fq -- "--joint-smt-seed-iters 2000" "$output"
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

baseline_accept="$(grep -n -m1 -- "acceptance_tdd_fresh_baseline_input.json" "$output" | cut -d: -f1)"
joint="$(grep -n -m1 -- "tdd_fresh_joint_free" "$output" | cut -d: -f1)"
accept="$(grep -n -m1 -- "acceptance_tdd_fresh_joint_free.json" "$output" | cut -d: -f1)"
polish="$(grep -n -m1 -- "tdd_fresh_surface_polish1" "$output" | cut -d: -f1)"
(( baseline_accept < joint && joint < accept && accept < polish ))

if grep -Fq -- "--threshold 1.0" "$output"; then
    echo "new workflow still uses the uncalibrated absolute acceptance threshold" >&2
    exit 1
fi

if bash "$SCRIPT" --dry-run > /dev/null 2>&1; then
    echo "driver accepted a missing --run-tag" >&2
    exit 1
fi
