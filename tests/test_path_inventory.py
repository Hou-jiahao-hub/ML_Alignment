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
