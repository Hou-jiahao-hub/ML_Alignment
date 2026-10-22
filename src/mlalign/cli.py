"""``mlalign-align``: Schur-LM alignment of the target layers from ACTS trackstates."""

from __future__ import annotations

import argparse
import sys

from .config import (
    add_input_arguments,
    add_output_arguments,
    add_seed_arguments,
    add_selection_arguments,
    add_solver_arguments,
    alignment_config_from_args,
)
from .pipeline import run_alignment
from .reporting import write_alignment_summary
from .timing import RunTimings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlalign-align",
        description="Joint track-geometry Schur-LM alignment: reads the current "
                    "geometry and its ACTS trackstates, writes the aligned geometry.",
    )
    add_input_arguments(parser)
    add_selection_arguments(parser)
    add_seed_arguments(parser)
    add_solver_arguments(parser)
    add_output_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = alignment_config_from_args(args)
    timings = RunTimings()
    result = {}
    try:
        result = run_alignment(config, timings=timings)
    except BaseException as exc:
        result.update({
            "timings": timings.snapshot(),
            "execution_status": (
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            ),
            "execution_error": f"{type(exc).__name__}: {exc}",
        })
        if args.output_summary is not None:
            try:
                write_alignment_summary(result, args.output_summary)
            except Exception as save_error:
                print(f"Could not save timing summary: {save_error}", file=sys.stderr)
        raise
    result.update({"execution_status": "completed", "execution_error": None})
    if args.output_summary is not None:
        write_alignment_summary(result, args.output_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
