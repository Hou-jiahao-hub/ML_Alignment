"""Run configuration shared by the surface and free pipelines and the CLI."""

from __future__ import annotations

from dataclasses import dataclass

from alignment_artifact_layout import (
    alignment_json_path,
    data_path,
    reconstruction_root_path,
)
from .box_constraints import JOINT_CLAMP_R, JOINT_CLAMP_T
from .conventions import DEFAULT_TARGET_LAYERS
from .free_helix_pipeline import JOINT_SOLVER
from .helix_init import JOINT_SMT_SEED_ITERS
from .stage_ntr_helix import (
    FISHER_EIGEN_FLOOR,
    LM_INITIAL_DAMPING,
    LM_MAX_ITERS,
    LM_MAX_STEP_R,
    LM_MAX_STEP_T,
)
from .stage_surface import DEFAULT_EPOCHS, DEFAULT_LAMBDA_L2
from .track_batches import JOINT_SEED_MIN_HITS, JOINT_SEED_MIN_SPAN

JOINT_SEED_MODE = "non-target-measurement"


@dataclass
class AlignmentRunConfig:
    input_geom: str = str(alignment_json_path("alignment_output", "misalignment_applied"))
    states_root: str = str(
        reconstruction_root_path(
            "reconstruction_output",
            "misaligned",
            "trackstates_kf",
        )
    )
    summary_root: str = str(
        reconstruction_root_path(
            "reconstruction_output",
            "misaligned",
            "tracksummary_kf",
        )
    )
    output_geom: str = str(
        alignment_json_path(
            "alignment_output",
            "aligned_transforms",
            stem_prefix="ml",
        )
    )
    nominal_file: str = str(data_path("odd_transforms.json"))
    iter_label: str | None = None
    make_plot: bool = True
    active_dof: tuple[int, ...] | str | None = None
    target_layers: tuple[tuple[int, int], ...] | str = DEFAULT_TARGET_LAYERS
    fix_reference: bool = True
    epochs: int = DEFAULT_EPOCHS
    lambda_l2: float = DEFAULT_LAMBDA_L2
    track_mode: str = "surface"
    min_hits: int = 8
    max_tracks: int = 0
    gpu: bool = False
    joint_clamp_t: float = JOINT_CLAMP_T
    joint_clamp_r: float = JOINT_CLAMP_R
    nominal_full_geom: str = str(data_path("odd_transforms.json"))
    joint_max_invalid_fraction: float = 0.05
    joint_seed_mode: str = JOINT_SEED_MODE
    joint_seed_min_hits: int = JOINT_SEED_MIN_HITS
    joint_seed_min_span: float = JOINT_SEED_MIN_SPAN
    joint_smt_seed_iters: int = JOINT_SMT_SEED_ITERS
    alignment_gain: float = 1.0
    joint_solver: str = JOINT_SOLVER
    lm_max_iters: int = LM_MAX_ITERS
    lm_initial_damping: float = LM_INITIAL_DAMPING
    fisher_eigen_floor: float = FISHER_EIGEN_FLOOR
    lm_max_step_t: float = LM_MAX_STEP_T
    lm_max_step_r: float = LM_MAX_STEP_R
    joint_refit_stationarity: bool = True
