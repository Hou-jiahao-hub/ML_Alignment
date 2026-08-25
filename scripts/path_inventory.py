from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project_config import (
    ProjectConfig,
    activate_cli_config,
    add_config_argument,
    load_project_config,
)


activate_cli_config()


def build_path_inventory(config: ProjectConfig | None = None) -> dict[str, Any]:
    config = load_project_config() if config is None else config
    data = config.data_root
    target_dir = config.target_layer_dir
    suffix = config.target_layer_suffix

    alignment_target = data / "alignment" / target_dir
    reconstruction = data / "reconstruction"
    nominal_reconstruction = reconstruction / "nominal" / target_dir
    misaligned_reconstruction = reconstruction / "misaligned" / target_dir
    aligned_reconstruction = reconstruction / "aligned" / target_dir
    residuals = reconstruction / "residuals" / target_dir
    iterative_logs = data / "experiments" / "iterative_ml" / target_dir
    surface_helix_logs = data / "experiments" / "surface_helix" / target_dir
    helix_validation = data / "experiments" / "helix_validation" / target_dir

    return {
        "schema_version": 1,
        "target_layers": [
            f"{volume}:{layer}" for volume, layer in config.target_layers
        ],
        "roots": {
            "repository": str(config.repository_root),
            "data": str(data),
            "acts": str(config.acts_root),
            "align_derivlib": str(config.align_derivlib_root),
            "odd": str(config.odd_root),
        },
        "external_inputs": {
            "acts_setup": str(config.acts_setup),
            "conda_setup": str(config.conda_setup),
            "odd_material_map": str(config.odd_material_map),
            "odd_digi_config": str(config.odd_digi_config),
            "align_derivlib_build_dir": str(config.align_derivlib_build_dir),
        },
        "data_inputs": {
            "nominal_geometry": str(data / "geometry" / "odd_transforms.json"),
            "simulation_particles": str(data / "simulation" / "particles.root"),
            "misaligned_geometry": str(
                alignment_target / f"misalignment_applied_{suffix}.json"
            ),
            "nominal_trackstates": str(
                nominal_reconstruction / f"trackstates_kf_{suffix}.root"
            ),
            "nominal_tracksummary": str(
                nominal_reconstruction / f"tracksummary_kf_{suffix}.root"
            ),
            "misaligned_trackstates": str(
                misaligned_reconstruction / f"trackstates_kf_{suffix}.root"
            ),
            "misaligned_tracksummary": str(
                misaligned_reconstruction / f"tracksummary_kf_{suffix}.root"
            ),
        },
        "output_directories": {
            "simulation": str(data / "simulation"),
            "alignment_target": str(alignment_target),
            "reconstruction_nominal": str(nominal_reconstruction),
            "reconstruction_misaligned": str(misaligned_reconstruction),
            "reconstruction_aligned": str(aligned_reconstruction),
            "residuals": str(residuals),
            "iterative_logs": str(iterative_logs),
            "surface_helix_logs": str(surface_helix_logs),
            "helix_validation": str(helix_validation),
        },
        "default_outputs": {
            "aligned_geometry": str(
                alignment_target / f"aligned_transforms_{suffix}_ml.json"
            ),
            "loss_curve": str(
                alignment_target / f"loss_curve_aligned_transforms_{suffix}_ml.png"
            ),
            "chi2_plot": str(
                reconstruction / "chi2_ndf" / target_dir / f"chi2ndf_{suffix}.png"
            ),
            "hit_number_plot": str(
                reconstruction
                / "hit_number"
                / target_dir
                / f"hit_number_{suffix}.png"
            ),
            "helix_validation_summary": str(
                helix_validation / "helix_validation_summary.json"
            ),
            "helix_validation_csv": str(
                helix_validation / "helix_smt_comparison.csv"
            ),
            "helix_measurement_validation_csv": str(
                helix_validation / "helix_measurement_comparison.csv"
            ),
        },
        "output_templates": {
            "iterative_geometry": str(
                alignment_target
                / f"aligned_transforms_{{run_tag}}_ml_{suffix}_iter{{iteration}}.json"
            ),
            "iterative_trackstates": str(
                aligned_reconstruction
                / f"trackstates_kf_{{run_tag}}_ml_{suffix}_iter{{iteration}}.root"
            ),
            "iterative_tracksummary": str(
                aligned_reconstruction
                / f"tracksummary_kf_{{run_tag}}_ml_{suffix}_iter{{iteration}}.root"
            ),
            "surface_geometry": str(
                alignment_target
                / "aligned_transforms_{run_tag}_surface_iter{iteration}.json"
            ),
            "helix_geometry": str(
                alignment_target / "aligned_transforms_{run_tag}_helix.json"
            ),
            "experiment_summary": str(
                surface_helix_logs / "{run_tag}" / "experiment_summary.json"
            ),
            "residual_root": str(
                residuals / "{label}_res_{axis}_{layer}_comparison.root"
            ),
        },
    }


def _text_lines(inventory: dict[str, Any]):
    for section, values in inventory.items():
        if isinstance(values, dict):
            for name, value in values.items():
                yield f"{section}.{name}={value}"
        else:
            yield f"{section}={values}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the effective ML_Alignment input/output path contract."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--json", action="store_true", help="Emit the schema-versioned JSON form."
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    inventory = build_path_inventory()
    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print("\n".join(_text_lines(inventory)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
