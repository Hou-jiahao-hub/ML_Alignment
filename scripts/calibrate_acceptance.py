"""Create a frozen reconstruction-acceptance profile from nominal states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .alignment_artifact_layout import target_layers_spec
    from .event_split import DEFAULT_SPLIT_SEED
    from .project_config import absolute_project_path, add_config_argument
    from .reconstruction_acceptance import (
        build_acceptance_baseline,
        load_reconstruction_frame,
        parse_layers,
    )
except ImportError:
    from alignment_artifact_layout import target_layers_spec
    from event_split import DEFAULT_SPLIT_SEED
    from project_config import absolute_project_path, add_config_argument
    from reconstruction_acceptance import (
        build_acceptance_baseline,
        load_reconstruction_frame,
        parse_layers,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--states", type=absolute_project_path, required=True)
    parser.add_argument("--output", type=absolute_project_path, required=True)
    parser.add_argument("--target-layers", default=target_layers_spec())
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=6006)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    states = Path(args.states)
    frame = load_reconstruction_frame(states, require_event_nr=True)
    result = build_acceptance_baseline(
        frame,
        parse_layers(args.target_layers),
        split_seed=args.split_seed,
        bootstrap_seed=args.bootstrap_seed,
        n_bootstrap=args.bootstrap_replicates,
    )
    result["source"] = {
        "states": str(states),
        "sha256": _sha256(states) if states.is_file() else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
