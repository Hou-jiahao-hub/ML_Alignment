#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TEST_DIR/.." && pwd)"
JOB_DIR="$REPO_DIR/scripts"
SCRIPT="$JOB_DIR/run_ml_iterative.sh"

# Target layers come from the single source of truth; never hardcode them here.
# `set -u` would abort on undefined variables, so this must run before first use.
eval "$(python "$JOB_DIR/alignment_artifact_layout.py" --shell)"

DIR="$TARGET_LAYER_DIR"
SUF="$TARGET_LAYER_SUFFIX"

# Solver knobs are hardcoded near the top of run_ml_iterative.sh and are meant to
# be edited when switching solvers.  Read them back instead of duplicating the
# values here, so flipping a knob does not turn this path-consistency gate red.
read_const() {
    local name="$1"
    sed -n "s/^${name}=\"\\(.*\\)\"\$/\\1/p" "$SCRIPT" | head -1
}
RUN_TAG="$(read_const RUN_TAG)"
TRACK_MODE="$(read_const TRACK_MODE)"
ACTIVE_DOF="$ACTIVE_DOF_SPEC"
[[ -n "$RUN_TAG" && -n "$TRACK_MODE" && -n "$ACTIVE_DOF" ]] \
    || { echo "failed to read solver constants from $SCRIPT" >&2; exit 1; }

# print_command escapes commas, so build the expected --active-dof spelling.
ACTIVE_DOF_ESC="${ACTIVE_DOF//,/\\,}"

# Label embedded in every artifact name for this run.
TAG="${RUN_TAG}_ml"

tmp_default="$(mktemp)"
tmp_resume="$(mktemp)"
trap 'rm -f "$tmp_default" "$tmp_resume"' EXIT

# `set -e` does NOT abort on a command whose status is inverted with `!`, so a
# bare `! grep ...` silently never fails.  Use this helper for every negative
# assertion instead.
refute_grep() {
    local pattern="$1" file="$2"
    if grep -Fq -- "$pattern" "$file"; then
        echo "unexpected match for '$pattern' in $file" >&2
        exit 1
    fi
}

bash "$SCRIPT" --dry-run --iters 1 > "$tmp_default"

grep -Fq -- "alignment solver: autograd" "$tmp_default"
grep -Fq -- "alignment script: ml_alignment_autograd.py" "$tmp_default"
grep -Fq -- "run tag      : $RUN_TAG" "$tmp_default"
grep -Fq -- "python ml_alignment_autograd.py" "$tmp_default"
grep -Fq -- "--input-geom $ALIGNMENT_ROOT/$DIR/misalignment_applied_$SUF.json" "$tmp_default"
grep -Fq -- "--states $RECONSTRUCTION_ROOT/misaligned/$DIR/trackstates_kf_$SUF.root" "$tmp_default"
grep -Fq -- "--summary $RECONSTRUCTION_ROOT/misaligned/$DIR/tracksummary_kf_$SUF.root" "$tmp_default"
grep -Fq -- "--output-geom $ALIGNMENT_ROOT/$DIR/aligned_transforms_${TAG}_${SUF}_iter0.json" "$tmp_default"
grep -Fq -- "--active-dof $ACTIVE_DOF_ESC" "$tmp_default"
grep -Fq -- "--lambda-l2 1e-5" "$tmp_default"

# The autograd solver must always be told which track mode to use, and the
# iteration-count flag must match that mode: surface consumes --epochs, free
# consumes --joint-iters.  Mixing them silently falls back to the wrong solver
# path, which is exactly the bug this gate exists to catch.
grep -Fq -- "--track-mode $TRACK_MODE" "$tmp_default"
if [[ "$TRACK_MODE" == "free" ]]; then
    grep -Fq -- "--joint-iters" "$tmp_default"
    refute_grep "--epochs" "$tmp_default"
else
    grep -Fq -- "--epochs 300" "$tmp_default"
    refute_grep "--joint-iters" "$tmp_default"
fi

refute_grep "--alignment-solver" "$tmp_default"
refute_grep "--solver" "$tmp_default"
refute_grep "--gain" "$tmp_default"
grep -Fq -- "--aligned-file $ALIGNMENT_ROOT/$DIR/aligned_transforms_${TAG}_${SUF}_iter0.json" "$tmp_default"
grep -Fq -- "--output-suffix ${TAG}_${SUF}_iter0" "$tmp_default"
grep -Fq -- "--aligned-trackstates trackstates_kf_${TAG}_${SUF}_iter0.root" "$tmp_default"

# The autograd solver must not receive the Jacobian-only compatibility flag,
# and the run header must not advertise a Jacobian setting for it either.
refute_grep "--jacobian-update-interval" "$tmp_default"
refute_grep "jacobian int." "$tmp_default"

# Plot commands must be generated from TARGET_LAYERS_SPEC, not hardcoded.
# This is the authoritative gate for the solve-layer == plot-layer invariant.
IFS=',' read -r -a target_pairs <<< "$TARGET_LAYERS_SPEC"
for pair in "${target_pairs[@]}"; do
    IFS=':' read -r vol lay <<< "$pair"
    grep -Fq -- "--label ${TAG}_${SUF}_iter0_v${vol}l${lay}" "$tmp_default"
    grep -Fq -- "--volume_id $vol --layer_id $lay" "$tmp_default"
done

# No plot command may target a layer outside the configured spec.
while read -r vol lay; do
    found=0
    for pair in "${target_pairs[@]}"; do
        [[ "$pair" == "$vol:$lay" ]] && found=1
    done
    [[ "$found" -eq 1 ]] || { echo "plot targets unconfigured layer $vol:$lay" >&2; exit 1; }
done < <(grep -o -- "--volume_id [0-9]* --layer_id [0-9]*" "$tmp_default" \
         | awk '{print $2, $4}' | sort -u)

# Unsupported volumes must fail loudly rather than being silently skipped.
# A dynamic test cannot reach that branch: the script enforces
# --target-layers == TARGET_LAYERS_SPEC, so a static assertion is used instead.
grep -Fq -- "unsupported plotting volume" "$SCRIPT"

bash "$SCRIPT" \
  --dry-run \
  --iters 2 \
  --start-iter 4 \
  --input-geom "alignment_output/$DIR/aligned_transforms_${TAG}_${SUF}_iter3.json" \
  --states "reconstruction_output/aligned/$DIR/trackstates_kf_${TAG}_${SUF}_iter3.root" \
  --summary "reconstruction_output/aligned/$DIR/tracksummary_kf_${TAG}_${SUF}_iter3.root" \
  > "$tmp_resume"

grep -Fq -- "--input-geom $REPO_DIR/alignment_output/$DIR/aligned_transforms_${TAG}_${SUF}_iter3.json" "$tmp_resume"
grep -Fq -- "--output-geom $ALIGNMENT_ROOT/$DIR/aligned_transforms_${TAG}_${SUF}_iter4.json" "$tmp_resume"
grep -Fq -- "--active-dof $ACTIVE_DOF_ESC" "$tmp_resume"
grep -Fq -- "--aligned-file $ALIGNMENT_ROOT/$DIR/aligned_transforms_${TAG}_${SUF}_iter4.json" "$tmp_resume"
grep -Fq -- "--states $RECONSTRUCTION_ROOT/aligned/$DIR/trackstates_kf_${TAG}_${SUF}_iter4.root" "$tmp_resume"
grep -Fq -- "--summary $RECONSTRUCTION_ROOT/aligned/$DIR/tracksummary_kf_${TAG}_${SUF}_iter4.root" "$tmp_resume"
grep -Fq -- "--output-geom $ALIGNMENT_ROOT/$DIR/aligned_transforms_${TAG}_${SUF}_iter5.json" "$tmp_resume"
grep -Fq -- "--output-suffix ${TAG}_${SUF}_iter5" "$tmp_resume"
