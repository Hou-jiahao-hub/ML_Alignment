"""``mlalign-align``: Schur-LM alignment of the target layers from ACTS trackstates."""

from __future__ import annotations

import argparse

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
    result = run_alignment(alignment_config_from_args(args))
    if args.output_summary is not None:
        write_alignment_summary(result, args.output_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
