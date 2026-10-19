"""Default settings, run configuration and CLI helpers of the Schur-LM alignment.

Every default is a module-level constant so that a copy of the package carries
its own configuration: nothing here reads a TOML file or a repository path.
The values are the ones the mainline orchestrator ran with on the ODD
(``Vol17_Lay8__Vol24_Lay4``, tx/ty/rz).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

#: The whole pipeline runs on the CPU in float64.  The refit stationarity
#: threshold (1e-6) and the Schur-LM decrements are below float32 resolution.
DTYPE = torch.float64

DEFAULT_TARGET_LAYERS = ((17, 8), (24, 4))
#: tx, ty, rz.  tz is never injected and poorly constrained by a planar
#: residual; rz is the only rotation the misalignment produces.
DEFAULT_ACTIVE_DOF = (0, 1, 5)

#: 0 keeps every selected track.
DEFAULT_MAX_TRACKS = 0
DEFAULT_SEED_MIN_HITS = 8
DEFAULT_SEED_MIN_SPAN = 300.0
#: The orchestrator value.  The former CLI default of 200 leaves the SMT point
#: seed non-stationary on real data.
DEFAULT_SMT_SEED_ITERS = 2000
DEFAULT_REFIT_STATIONARITY = True
DEFAULT_MAX_INVALID_FRACTION = 0.05

#: Box bounds on the geometry corrections: mm for translations, rad for rotations.
DEFAULT_CLAMP_T = 1.5
DEFAULT_CLAMP_R = 0.05
#: Outer Schur-LM budget; equal to the pre-registered ``SCHUR_LM_MAX_ITERS``.
#: Changing it departs from the pre-registered convergence contract.
DEFAULT_MAX_ITERS = 60

DEFAULT_MAKE_PLOT = False


@dataclass
class AlignmentConfig:
    """One Schur-LM run: input geometry + ACTS trackstates -> aligned geometry."""

    input_geom: str
    states_root: str
    output_geom: str
    nominal_full_geom: str
    #: Truth geometry for the diagnostic comparison; defaults to ``nominal_full_geom``.
    nominal_file: str | None = None
    target_layers: tuple[tuple[int, int], ...] | str = DEFAULT_TARGET_LAYERS
    active_dof: tuple[int, ...] | str | None = DEFAULT_ACTIVE_DOF
    max_tracks: int = DEFAULT_MAX_TRACKS
    joint_seed_min_hits: int = DEFAULT_SEED_MIN_HITS
    joint_seed_min_span: float = DEFAULT_SEED_MIN_SPAN
    joint_smt_seed_iters: int = DEFAULT_SMT_SEED_ITERS
    joint_refit_stationarity: bool = DEFAULT_REFIT_STATIONARITY
    joint_max_invalid_fraction: float = DEFAULT_MAX_INVALID_FRACTION
    joint_clamp_t: float = DEFAULT_CLAMP_T
    joint_clamp_r: float = DEFAULT_CLAMP_R
    lm_max_iters: int = DEFAULT_MAX_ITERS
    make_plot: bool = DEFAULT_MAKE_PLOT

    @property
    def truth_geom(self) -> str:
        return self.nominal_file if self.nominal_file is not None else self.nominal_full_geom


def _layers_spec(layers) -> str:
    return ",".join(f"{volume}:{layer}" for volume, layer in layers)


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-geom", required=True,
                        help="Geometry JSON of the target surfaces to correct.")
    parser.add_argument("--states", dest="states_root", required=True,
                        help="ACTS trackstates ROOT reconstructed with --input-geom.")
    parser.add_argument("--nominal-full-geom", required=True,
                        help="Full nominal geometry JSON; anchors the non-target hits.")
    parser.add_argument("--nominal", dest="nominal_file", default=None,
                        help="Truth geometry for the diagnostic comparison "
                             "(default: --nominal-full-geom).")


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-layers", default=_layers_spec(DEFAULT_TARGET_LAYERS),
                        help="Comma-separated volume:layer list. "
                             f"Default: {_layers_spec(DEFAULT_TARGET_LAYERS)}.")
    parser.add_argument("--active-dof", default=",".join(map(str, DEFAULT_ACTIVE_DOF)),
                        help="Comma-separated DOF indices or names (tx,ty,tz,rx,ry,rz). "
                             f"Default: {','.join(map(str, DEFAULT_ACTIVE_DOF))}.")
    parser.add_argument("--max-tracks", type=int, default=DEFAULT_MAX_TRACKS,
                        help="0 keeps every selected track.")


def add_seed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--joint-seed-min-hits", type=int, default=DEFAULT_SEED_MIN_HITS)
    parser.add_argument("--joint-seed-min-span", type=float, default=DEFAULT_SEED_MIN_SPAN)
    parser.add_argument("--joint-smt-seed-iters", type=int, default=DEFAULT_SMT_SEED_ITERS)
    parser.add_argument("--no-joint-refit-stationarity", action="store_true",
                        help="Skip the Newton-decrement measurement of the measurement refit.")


def add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--joint-max-invalid-fraction", type=float,
                        default=DEFAULT_MAX_INVALID_FRACTION)
    parser.add_argument("--joint-clamp-t", type=float, default=DEFAULT_CLAMP_T)
    parser.add_argument("--joint-clamp-r", type=float, default=DEFAULT_CLAMP_R)
    parser.add_argument("--lm-max-iters", type=int, default=DEFAULT_MAX_ITERS,
                        help="Outer Schur-LM budget (pre-registered value: "
                             f"{DEFAULT_MAX_ITERS}).")


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-geom", required=True,
                        help="Aligned geometry JSON (input target surfaces only).")
    parser.add_argument("--output-summary", default=None,
                        help="Optional compact JSON of solver stages and metrics.")
    parser.add_argument("--plot", action="store_true",
                        help="Write the convergence curve next to --output-geom.")


def alignment_config_from_args(args: argparse.Namespace) -> AlignmentConfig:
    return AlignmentConfig(
        input_geom=args.input_geom,
        states_root=args.states_root,
        output_geom=args.output_geom,
        nominal_full_geom=args.nominal_full_geom,
        nominal_file=args.nominal_file,
        target_layers=args.target_layers,
        active_dof=args.active_dof,
        max_tracks=args.max_tracks,
        joint_seed_min_hits=args.joint_seed_min_hits,
        joint_seed_min_span=args.joint_seed_min_span,
        joint_smt_seed_iters=args.joint_smt_seed_iters,
        joint_refit_stationarity=not args.no_joint_refit_stationarity,
        joint_max_invalid_fraction=args.joint_max_invalid_fraction,
        joint_clamp_t=args.joint_clamp_t,
        joint_clamp_r=args.joint_clamp_r,
        lm_max_iters=args.lm_max_iters,
        make_plot=args.plot,
    )
