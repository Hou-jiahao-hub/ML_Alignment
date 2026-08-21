from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from project_config import activate_cli_config, load_project_config


activate_cli_config()
PROJECT_CONFIG = load_project_config()
TARGET_LAYERS = PROJECT_CONFIG.target_layers
ACTIVE_DOF = PROJECT_CONFIG.active_dof
TARGET_LAYER_SUFFIX = PROJECT_CONFIG.target_layer_suffix
TARGET_LAYER_DIR = PROJECT_CONFIG.target_layer_dir
DATA_ROOT = PROJECT_CONFIG.data_root
ACTS_ROOT = PROJECT_CONFIG.acts_root

_LEGACY_DATA_ROOTS = {
    "alignment_output": DATA_ROOT / "alignment",
    "reconstruction_output": DATA_ROOT / "reconstruction",
    "simulation_output": DATA_ROOT / "simulation",
    "iterative_ml_logs": DATA_ROOT / "experiments" / "iterative_ml",
    "surface_helix_logs": DATA_ROOT / "experiments" / "surface_helix",
    "helix_validation_output": DATA_ROOT / "experiments" / "helix_validation",
    "ablation": DATA_ROOT / "experiments" / "ablation",
    "archive": DATA_ROOT / "archive",
}


def data_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path == Path("odd_transforms.json"):
        return DATA_ROOT / "geometry" / path.name
    if path.parts and path.parts[0] in _LEGACY_DATA_ROOTS:
        return _LEGACY_DATA_ROOTS[path.parts[0]].joinpath(*path.parts[1:])
    return PROJECT_CONFIG.repository_root / path


def target_layers_spec(target_layers: tuple[tuple[int, int], ...] = TARGET_LAYERS) -> str:
    return ",".join(f"{vol}:{lay}" for vol, lay in target_layers)


def target_volume_map(
    target_layers: tuple[tuple[int, int], ...] = TARGET_LAYERS,
) -> dict[int, dict[int, int]]:
    return {vol: {lay: -1} for vol, lay in target_layers}


def target_subdir(base_dir: str | Path) -> Path:
    base_path = data_path(base_dir)
    if base_path.name == TARGET_LAYER_DIR:
        return base_path
    return base_path / TARGET_LAYER_DIR


def artifact_filename(
    stem: str,
    ext: str,
    *,
    stem_prefix: str | None = None,
    iteration: int | None = None,
) -> str:
    ext = ext if ext.startswith(".") else f".{ext}"
    parts = [stem] if stem else []
    if iteration is None:
        parts.append(TARGET_LAYER_SUFFIX)
        if stem_prefix:
            parts.append(stem_prefix)
    else:
        if stem_prefix:
            parts.append(stem_prefix)
        parts.append(TARGET_LAYER_SUFFIX)
        parts.append(f"iter{iteration}")
    return "_".join(parts) + ext


def alignment_json_path(
    base_dir: str | Path,
    stem: str,
    *,
    stem_prefix: str | None = None,
    iteration: int | None = None,
) -> Path:
    return target_subdir(base_dir) / artifact_filename(
        stem,
        ".json",
        stem_prefix=stem_prefix,
        iteration=iteration,
    )


def reconstruction_root_path(
    base_dir: str | Path,
    mode: str,
    stem: str,
    *,
    stem_prefix: str | None = None,
    iteration: int | None = None,
) -> Path:
    return target_subdir(Path(base_dir) / mode) / artifact_filename(
        stem,
        ".root",
        stem_prefix=stem_prefix,
        iteration=iteration,
    )


def iteration_label(run_tag: str, iteration: int) -> str:
    prefix = f"{run_tag}_ml" if run_tag else "ml"
    return artifact_filename(
        "",
        "",
        stem_prefix=prefix,
        iteration=iteration,
    ).strip("_")


def shell_exports() -> str:
    values = {
        "ML_ALIGNMENT_DATA_ROOT": str(DATA_ROOT),
        "ML_ALIGNMENT_ACTS_ROOT": str(ACTS_ROOT),
        "ALIGNMENT_ROOT": str(DATA_ROOT / "alignment"),
        "RECONSTRUCTION_ROOT": str(DATA_ROOT / "reconstruction"),
        "SIMULATION_ROOT": str(DATA_ROOT / "simulation"),
        "GEOMETRY_ROOT": str(DATA_ROOT / "geometry"),
        "ITERATIVE_LOG_ROOT": str(DATA_ROOT / "experiments" / "iterative_ml"),
        "SURFACE_HELIX_LOG_ROOT": str(DATA_ROOT / "experiments" / "surface_helix"),
        "TARGET_LAYER_DIR": TARGET_LAYER_DIR,
        "TARGET_LAYER_SUFFIX": TARGET_LAYER_SUFFIX,
        "TARGET_LAYERS_SPEC": target_layers_spec(),
        "ACTIVE_DOF_SPEC": ",".join(str(value) for value in ACTIVE_DOF),
        "MISALIGNMENT_JSON": artifact_filename("misalignment_applied", ".json"),
        "ALIGNED_REFERENCE_JSON": artifact_filename("aligned_transforms", ".json"),
        "DEFAULT_ML_OUTPUT_JSON": artifact_filename(
            "aligned_transforms",
            ".json",
            stem_prefix="ml",
        ),
    }
    return "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in values.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared target-layer output layout for alignment artifacts."
    )
    parser.add_argument(
        "--config",
        help="Path to a TOML project configuration override.",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print shell variable assignments for the shared layout constants.",
    )
    args = parser.parse_args()

    if args.shell:
        print(shell_exports())
    else:
        print(f"TARGET_LAYERS={target_layers_spec()}")
        print(f"TARGET_LAYER_DIR={TARGET_LAYER_DIR}")
        print(f"TARGET_LAYER_SUFFIX={TARGET_LAYER_SUFFIX}")


if __name__ == "__main__":
    main()
