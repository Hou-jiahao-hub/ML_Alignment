"""Package defaults equal the validated mainline run settings."""

from __future__ import annotations

import pytest

from mlalign import config
from mlalign.box_constraints import JOINT_CLAMP_R, JOINT_CLAMP_T
from mlalign.cli import build_argument_parser
from mlalign.config import alignment_config_from_args
from mlalign.conventions import parse_active_dof, parse_target_layers
from mlalign.schur_lm import DTYPE, SCHUR_LM_MAX_ITERS
from mlalign.track_batches import JOINT_SEED_MIN_HITS, JOINT_SEED_MIN_SPAN

REQUIRED = [
    "--input-geom", "in.json",
    "--states", "states.root",
    "--nominal-full-geom", "nominal.json",
    "--output-geom", "out.json",
]


def test_defaults_match_the_mainline_orchestrator():
    assert config.DEFAULT_TARGET_LAYERS == ((17, 8), (24, 4))
    assert config.DEFAULT_ACTIVE_DOF == (0, 1, 5)
    assert config.DEFAULT_SMT_SEED_ITERS == 2000
    assert config.DEFAULT_SEED_MIN_HITS == 8
    assert config.DEFAULT_SEED_MIN_SPAN == 300.0
    assert config.DEFAULT_MAX_INVALID_FRACTION == 0.05
    assert (config.DEFAULT_CLAMP_T, config.DEFAULT_CLAMP_R) == (1.5, 0.05)
    assert config.DEFAULT_MAKE_PLOT is False


def test_max_iters_default_is_the_preregistered_schur_budget():
    assert config.DEFAULT_MAX_ITERS == SCHUR_LM_MAX_ITERS


def test_solver_modules_read_their_defaults_from_config():
    assert (JOINT_CLAMP_T, JOINT_CLAMP_R) == (config.DEFAULT_CLAMP_T, config.DEFAULT_CLAMP_R)
    assert (JOINT_SEED_MIN_HITS, JOINT_SEED_MIN_SPAN) == (
        config.DEFAULT_SEED_MIN_HITS,
        config.DEFAULT_SEED_MIN_SPAN,
    )
    assert config.DTYPE is DTYPE


def test_package_defaults_agree_with_the_workflow_toml():
    """The workflow layer names its data directories from configs/default.toml."""
    from project_config import DEFAULT_CONFIG_PATH, load_project_config

    workflow = load_project_config(DEFAULT_CONFIG_PATH)
    assert workflow.target_layers == config.DEFAULT_TARGET_LAYERS
    assert workflow.active_dof == config.DEFAULT_ACTIVE_DOF


def test_cli_without_options_reproduces_the_dataclass_defaults():
    from_cli = alignment_config_from_args(build_argument_parser().parse_args(REQUIRED))
    direct = config.AlignmentConfig(
        input_geom="in.json",
        states_root="states.root",
        output_geom="out.json",
        nominal_full_geom="nominal.json",
    )

    assert parse_target_layers(from_cli.target_layers) == parse_target_layers(direct.target_layers)
    assert parse_active_dof(from_cli.active_dof) == parse_active_dof(direct.active_dof)
    for field in (
        "max_tracks", "joint_seed_min_hits", "joint_seed_min_span", "joint_smt_seed_iters",
        "joint_refit_stationarity", "joint_max_invalid_fraction", "joint_clamp_t",
        "joint_clamp_r", "lm_max_iters", "make_plot", "nominal_file",
    ):
        assert getattr(from_cli, field) == getattr(direct, field), field


def test_truth_geometry_defaults_to_the_full_nominal_geometry():
    cfg = config.AlignmentConfig(
        input_geom="in.json", states_root="s.root", output_geom="o.json", nominal_full_geom="n.json"
    )
    assert cfg.truth_geom == "n.json"
    cfg.nominal_file = "truth.json"
    assert cfg.truth_geom == "truth.json"


@pytest.mark.parametrize("flag", ["--gpu", "--track-mode=free", "--joint-solver=block-lm", "--lambda-l2=1"])
def test_retired_options_are_rejected(flag):
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([*REQUIRED, flag])


def test_empty_active_dof_falls_back_to_the_default():
    assert parse_active_dof("") == config.DEFAULT_ACTIVE_DOF
    assert parse_active_dof(None) == config.DEFAULT_ACTIVE_DOF
