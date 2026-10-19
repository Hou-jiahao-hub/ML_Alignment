from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def load_layout_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "alignment_artifact_layout.py"
    spec = importlib.util.spec_from_file_location(
        "alignment_artifact_layout", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_layer_constants_match_requested_layout():
    layout = load_layout_module()

    # Canary: intentionally literal.  Its whole value is to fail when the target
    # layers are switched, forcing every dependent artefact to be updated too.
    # Deriving these from the module itself would make the test a tautology.
    assert layout.TARGET_LAYERS == ((17, 8), (24, 4))
    assert layout.TARGET_LAYER_SUFFIX == "v17l8_v24l4"
    assert layout.TARGET_LAYER_DIR == "Vol17_Lay8__Vol24_Lay4"


def test_artifact_filename_adds_suffix_and_iteration_token():
    layout = load_layout_module()
    suffix = layout.TARGET_LAYER_SUFFIX

    assert layout.artifact_filename("misalignment_applied", ".json") == (
        f"misalignment_applied_{suffix}.json"
    )
    assert layout.artifact_filename(
        "aligned_transforms",
        ".json",
        stem_prefix="autograd_ml",
        iteration=2,
    ) == f"aligned_transforms_autograd_ml_{suffix}_iter2.json"


def test_output_directories_append_target_layer_subdirectory():
    layout = load_layout_module()
    target_dir = layout.TARGET_LAYER_DIR

    assert layout.target_subdir(Path("alignment_output")) == (
        layout.DATA_ROOT / "alignment" / target_dir
    )
    assert layout.target_subdir(Path("reconstruction_output") / "misaligned") == (
        layout.DATA_ROOT / "reconstruction" / "misaligned" / target_dir
    )


def test_legacy_artifact_paths_resolve_inside_configured_data_root():
    layout = load_layout_module()

    assert layout.data_path("simulation_output") == layout.DATA_ROOT / "simulation"
    assert layout.data_path("odd_transforms.json") == (
        layout.DATA_ROOT / "geometry" / "odd_transforms.json"
    )
    assert layout.data_path("archive/logs/run.log") == (
        layout.DATA_ROOT / "archive" / "logs" / "run.log"
    )
    absolute = Path("/tmp/external.root")
    assert layout.data_path(absolute) == absolute


def test_shell_exports_include_absolute_runtime_dependency_paths():
    layout = load_layout_module()
    exports = dict(
        line.split("=", maxsplit=1) for line in layout.shell_exports().splitlines()
    )

    expected = {
        "ML_ALIGNMENT_REPOSITORY_ROOT": layout.PROJECT_CONFIG.repository_root,
        "ML_ALIGNMENT_ACTS_SETUP": layout.PROJECT_CONFIG.acts_setup,
        "ML_ALIGNMENT_ODD_ROOT": layout.PROJECT_CONFIG.odd_root,
        "ML_ALIGNMENT_ODD_MATERIAL_MAP": layout.PROJECT_CONFIG.odd_material_map,
        "ML_ALIGNMENT_ACTS_DIGI_CONFIG": layout.PROJECT_CONFIG.odd_digi_config,
        "ML_ALIGNMENT_CONDA_SETUP": layout.PROJECT_CONFIG.conda_setup,
    }
    assert {key: Path(exports[key]) for key in expected} == expected
    assert Path(exports["SCHUR_LM_LOG_ROOT"]) == layout.DATA_ROOT / "experiments" / "schur_lm"
    assert "SURFACE_HELIX_LOG_ROOT" not in exports


def test_layout_uses_effective_project_configuration(monkeypatch, tmp_path):
    config_path = tmp_path / "layers.toml"
    config_path.write_text(
        '[alignment]\ntarget_layers = ["7:8", "9:10"]\nactive_dof = [0, 5]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ML_ALIGNMENT_CONFIG", str(config_path))

    layout = load_layout_module()

    assert layout.TARGET_LAYERS == ((7, 8), (9, 10))
    assert layout.ACTIVE_DOF == (0, 5)
    assert layout.TARGET_LAYER_SUFFIX == "v7l8_v9l10"
    assert layout.TARGET_LAYER_DIR == "Vol7_Lay8__Vol9_Lay10"
    sys.modules.pop("alignment_artifact_layout", None)


def test_layout_cli_accepts_config_before_loading_defaults(tmp_path):
    config_path = tmp_path / "layers.toml"
    config_path.write_text(
        '[alignment]\ntarget_layers = ["3:4"]\n', encoding="utf-8"
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "alignment_artifact_layout.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(config_path), "--shell"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TARGET_LAYER_SUFFIX=v3l4" in completed.stdout
    assert "TARGET_LAYER_DIR=Vol3_Lay4" in completed.stdout
