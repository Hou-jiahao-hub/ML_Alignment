#!/usr/bin/env bash
# Source this file in a fresh shell. Do not activate conda in the same shell.

ml_alignment_setup_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d /cvmfs/sft.cern.ch/lcg ]]; then
    echo "ERROR: /cvmfs/sft.cern.ch/lcg is not mounted." >&2
    echo "Mount CVMFS with your system administrator's approved procedure, then retry." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -n "${ML_ALIGNMENT_CONFIG:-}" ]]; then
    eval "$(python3 "$ml_alignment_setup_dir/scripts/alignment_artifact_layout.py" \
        --config "$ML_ALIGNMENT_CONFIG" --shell)"
else
    eval "$(python3 "$ml_alignment_setup_dir/scripts/alignment_artifact_layout.py" --shell)"
fi

echo "Loading LCG_109..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh

export ACTS_HOME="${ACTS_HOME:-$ML_ALIGNMENT_ACTS_ROOT}"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$ACTS_HOME/install/lib64:$ACTS_HOME/install/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ACTS_HOME/build/python:${PYTHONPATH:-}"
export ODD_PATH="$ACTS_HOME/thirdparty/OpenDataDetector"

echo "ACTS environment ready: $ACTS_HOME"
