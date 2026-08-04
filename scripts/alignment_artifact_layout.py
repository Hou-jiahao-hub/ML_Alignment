from __future__ import annotations

import argparse
import shlex
from pathlib import Path

TARGET_LAYERS: tuple[tuple[int, int], ...] = ((17, 8), (24, 4))
TARGET_LAYER_SUFFIX = "v17l8_v24l4"
TARGET_LAYER_DIR = "Vol17_Lay8__Vol24_Lay4"


def target_layers_spec(target_layers: tuple[tuple[int, int], ...] = TARGET_LAYERS) -> str:
    return ",".join(f"{vol}:{lay}" for vol, lay in target_layers)


def target_volume_map(
    target_layers: tuple[tuple[int, int], ...] = TARGET_LAYERS,
) -> dict[int, dict[int, int]]:
    return {vol: {lay: -1} for vol, lay in target_layers}


def target_subdir(base_dir: str | Path) -> Path:
    base_path = Path(base_dir)
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
        "TARGET_LAYER_DIR": TARGET_LAYER_DIR,
        "TARGET_LAYER_SUFFIX": TARGET_LAYER_SUFFIX,
        "TARGET_LAYERS_SPEC": target_layers_spec(),
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
