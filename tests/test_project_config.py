from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


def test_default_config_derives_consistent_artifact_names(monkeypatch):
    monkeypatch.delenv("ML_ALIGNMENT_CONFIG", raising=False)

    from project_config import load_project_config

    config = load_project_config()

    assert config.target_layers == ((17, 8), (24, 4))
    assert config.active_dof == (0, 1, 5)
    assert config.target_layer_suffix == "v17l8_v24l4"
    assert config.target_layer_dir == "Vol17_Lay8__Vol24_Lay4"
    assert config.data_root == config.repository_root / "data"
    assert config.acts_root == config.repository_root.parent / "alignTest"


def test_effective_active_dof_is_shared_by_both_solver_modules(monkeypatch):
    monkeypatch.delenv("ML_ALIGNMENT_CONFIG", raising=False)
    for module_name in ("alignment_artifact_layout", "ml_alignment", "ml_alignment_autograd"):
        sys.modules.pop(module_name, None)

    import ml_alignment
    import ml_alignment_autograd

    assert ml_alignment.DEFAULT_ACTIVE_DOF == (0, 1, 5)
    assert ml_alignment_autograd.DEFAULT_ACTIVE_DOF == (0, 1, 5)


def test_explicit_config_overrides_environment_and_resolves_from_repository(
    monkeypatch, tmp_path
):
    from project_config import load_project_config

    environment_config = tmp_path / "environment.toml"
    environment_config.write_text(
        '[alignment]\ntarget_layers = ["1:2"]\n', encoding="utf-8"
    )
    explicit_config = tmp_path / "explicit.toml"
    explicit_config.write_text(
        """
[project]
data_root = "local-data"

[acts]
root = "/opt/acts"

[alignment]
target_layers = ["7:8", "9:10"]
active_dof = [0, 5]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ML_ALIGNMENT_CONFIG", str(environment_config))

    config = load_project_config(explicit_config)

    assert config.target_layers == ((7, 8), (9, 10))
    assert config.active_dof == (0, 5)
    assert config.data_root == config.repository_root / "local-data"
    assert config.acts_root == Path("/opt/acts")


def test_cli_config_activation_supports_split_and_equals_forms(monkeypatch, tmp_path):
    import os

    from project_config import activate_cli_config

    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    monkeypatch.delenv("ML_ALIGNMENT_CONFIG", raising=False)

    assert activate_cli_config(["tool.py", "--config", str(first)]) == first.resolve()
    assert os.environ["ML_ALIGNMENT_CONFIG"] == str(first.resolve())

    assert activate_cli_config(["tool.py", f"--config={second}"]) == second.resolve()
    assert os.environ["ML_ALIGNMENT_CONFIG"] == str(second.resolve())
    os.environ.pop("ML_ALIGNMENT_CONFIG", None)


@pytest.mark.parametrize(
    "script_name",
    [
        "ml_alignment.py",
        "ml_alignment_autograd.py",
        "ml_alignment_helix.py",
        "validate_helix_model.py",
        "plot_residuals.py",
        "plot_chi2ndf.py",
        "plot_hit_number.py",
    ],
)
def test_pure_python_clis_expose_project_config_override(script_name):
    script = Path(__file__).resolve().parents[1] / "scripts" / script_name

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in completed.stdout


@pytest.mark.parametrize(
    ("toml_text", "message"),
    [
        ('[alignment]\ntarget_layers = ["17"]\n', "volume:layer"),
        ('[alignment]\ntarget_layers = ["17:8", "17:8"]\n', "duplicate"),
        ("[alignment]\nactive_dof = [0, 6]\n", "0 through 5"),
    ],
)
def test_invalid_alignment_configuration_fails_closed(tmp_path, toml_text, message):
    from project_config import load_project_config

    path = tmp_path / "invalid.toml"
    path.write_text(toml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_project_config(path)
