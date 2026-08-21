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
    > "$output"

grep -Fq -- "surface relative tolerance : 0.05" "$output"
grep -Fq -- "surface patience           : 3" "$output"
grep -Fq -- "surface chi2 target        : 1.0" "$output"
grep -Fq -- "maximum surface iterations : 15" "$output"
grep -Fq -- "helix candidates           : 17:8,24:4" "$output"
grep -Fq -- "python ml_alignment_autograd.py" "$output"
grep -Fq -- "--track-mode surface" "$output"
grep -Fq -- "--active-dof 0\,1\,5" "$output"
grep -Fq -- "python ml_alignment_helix.py assess" "$output"
grep -Fq -- "python ml_alignment_helix.py align" "$output"
grep -Fq -- "python alignment_odd.py reconstruction" "$output"
grep -Fq -- "tdd_fresh_surface_iter0" "$output"
grep -Fq -- "tdd_fresh_helix" "$output"
grep -Fq -- "--input-geom $REPO_DIR/inputs/current.json" "$output"
grep -Fq -- "--states $REPO_DIR/inputs/states.root" "$output"
grep -Fq -- "--summary $REPO_DIR/inputs/summary.root" "$output"
grep -Fq -- "--nominal-states $REPO_DIR/inputs/nominal.root" "$output"
grep -Fq -- "--aligned-file $ALIGNMENT_ROOT/$TARGET_LAYER_DIR/aligned_transforms_tdd_fresh_surface_iter0.json" "$output"
grep -Fq -- "--aligned-file $ALIGNMENT_ROOT/$TARGET_LAYER_DIR/aligned_transforms_tdd_fresh_helix.json" "$output"

if bash "$SCRIPT" --dry-run > /dev/null 2>&1; then
    echo "driver accepted a missing --run-tag" >&2
    exit 1
fi
