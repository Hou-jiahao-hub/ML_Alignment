from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _path_values(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from _path_values(nested)
    elif isinstance(value, str):
        yield value


def test_inventory_reports_absolute_inputs_outputs_and_runtime_dependencies():
    from path_inventory import build_path_inventory

    repository_root = Path(__file__).resolve().parents[1]
    inventory = build_path_inventory()

    assert inventory["schema_version"] == 1
    assert inventory["roots"] == {
        "repository": str(repository_root),
        "data": str(repository_root / "data"),
        "acts": str(repository_root.parent / "alignTest"),
        "align_derivlib": str(repository_root / "AlignDerivLib"),
        "odd": str(repository_root.parent / "alignTest" / "thirdparty" / "OpenDataDetector"),
    }
    assert inventory["external_inputs"]["align_derivlib_build_dir"] == str(
        repository_root / "AlignDerivLib" / "build_pybind11"
    )
    assert inventory["data_inputs"]["nominal_geometry"] == str(
        repository_root / "data" / "geometry" / "odd_transforms.json"
    )
    assert inventory["data_inputs"]["misaligned_trackstates"] == str(
        repository_root
        / "data"
        / "reconstruction"
        / "misaligned"
        / "Vol17_Lay8__Vol24_Lay4"
        / "trackstates_kf_v17l8_v24l4.root"
    )
    assert inventory["default_outputs"]["helix_measurement_validation_csv"] == str(
        repository_root
        / "data"
        / "experiments"
        / "helix_validation"
        / "Vol17_Lay8__Vol24_Lay4"
        / "helix_measurement_comparison.csv"
    )

    for section in (
        "roots",
        "external_inputs",
        "data_inputs",
        "output_directories",
        "default_outputs",
        "output_templates",
    ):
        assert all(value.startswith("/") for value in _path_values(inventory[section]))


def test_inventory_cli_is_independent_of_current_working_directory(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "path_inventory.py"
    command = [sys.executable, str(script), "--json"]

    from_repository = subprocess.run(
        command, cwd=repository_root, check=True, capture_output=True, text=True
    )
    from_external_directory = subprocess.run(
        command, cwd=tmp_path, check=True, capture_output=True, text=True
    )

    assert json.loads(from_repository.stdout) == json.loads(
        from_external_directory.stdout
    )


def test_inventory_exposes_the_material_artifact_contract():
    """Material export writes below data/ and must be discoverable."""
    import importlib

    inventory = importlib.import_module("scripts.path_inventory").build_path_inventory()
    directories = inventory["output_directories"]
    templates = inventory["output_templates"]

    assert "/data/experiments/material/" in directories["material_export"]
    # ROOT names come from --output-suffix; only the experiment directory is
    # keyed by --run-tag.  Tie the template to the producer so it cannot drift.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from alignment_artifact_layout import artifact_filename

    for name in ("material_root", "propsummary_root"):
        assert "{output_suffix}" in templates[name], name
        assert "/data/reconstruction/nominal/" in templates[name], name
        stem = name.replace("_root", "") + "_kf"
        assert templates[name].format(output_suffix="mat0").endswith(
            f"{stem}_mat0.root"
        )
        # An empty suffix must fall back to the repository-wide naming rule.
        assert artifact_filename(stem, ".root").startswith(f"{stem}_")

    for name in (
        "material_artifact",
        "material_export_manifest",
        "material_convert_manifest",
        "material_artifact_metadata",
    ):
        assert "{run_tag}" in templates[name], name

    # Export and conversion must never share a manifest path.
    assert (
        templates["material_export_manifest"]
        != templates["material_convert_manifest"]
    )
    assert templates["material_artifact"].endswith("material_nodes.npz")


def test_inventory_exposes_the_paired_holdout_key_list():
    """Gate 4's row-identity list is the paired contract; it must be locatable."""
    import importlib

    inventory = importlib.import_module("scripts.path_inventory").build_path_inventory()
    keys = inventory["default_outputs"]["helix_holdout_keys"]
    assert keys.endswith("/holdout_keys.csv")
    assert "/data/experiments/helix_validation/" in keys
    # It lives beside the summary whose SHA256 refers to it.
    assert (
        Path(keys).parent
        == Path(inventory["default_outputs"]["helix_validation_summary"]).parent
    )
