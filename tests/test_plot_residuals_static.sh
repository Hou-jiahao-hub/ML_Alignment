#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$TEST_DIR/.." && pwd)"
JOB_DIR="$REPO_DIR/scripts"
SCRIPT="$JOB_DIR/plot_residuals.py"

# `set -e` does NOT abort on a command whose status is inverted with `!`, so a
# bare `! grep ...` silently never fails.  Use this helper for negative checks.
refute_grep() {
    local pattern="$1"
    if grep -Fq -- "$pattern" "$SCRIPT"; then
        echo "unexpected match for '$pattern' in $SCRIPT" >&2
        exit 1
    fi
}

# Target layers and artifact names must come from alignment_artifact_layout,
# the single source of truth.  Switching layers there must not require editing
# this script, so any local literal is a regression.
grep -Fq "from alignment_artifact_layout import TARGET_LAYERS, artifact_filename" "$SCRIPT"
grep -Fq 'DEFAULT_TRACKSTATES = artifact_filename("trackstates_kf", ".root")' "$SCRIPT"
refute_grep "TARGET_LAYERS = ("
refute_grep "trackstates_kf_v"

grep -Fq 'layer_tag = f"Vol{vol_id}_Lay{lay_id}"' "$SCRIPT"
grep -Fq "for vol_id, lay_id in TARGET_LAYERS:" "$SCRIPT"
grep -Fq "sub_cfg = get_subsystem_config(vol_id)" "$SCRIPT"
refute_grep "for subsystem_name, sub_cfg in SUBSYSTEMS.items()"
