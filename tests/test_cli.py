"""The installed package is self-contained: no repository paths, TOML or ACTS."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def _run(code, tmp_path, env=None):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_package_is_imported_from_src(tmp_path):
    completed = _run("import mlalign; print(mlalign.__file__)", tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()).parent == REPOSITORY / "src" / "mlalign"


def test_package_import_loads_no_workflow_or_acts_module(tmp_path):
    code = (
        "import sys, mlalign, mlalign.cli, mlalign.schur_lm; "
        "forbidden = {'acts', 'tomli', 'tomllib', 'project_config', 'alignment_artifact_layout', "
        "'experiment_manifest', 'matplotlib'}; "
        "loaded = {name.split('.')[0] for name in sys.modules}; "
        "assert not (forbidden & loaded), forbidden & loaded"
    )
    completed = _run(code, tmp_path)
    assert completed.returncode == 0, completed.stderr


def test_package_imports_without_the_conda_library_path(tmp_path):
    """``mlalign`` loads uproot before torch, so the system libstdc++ is never picked."""
    env = {key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"}
    completed = _run("import mlalign.schur_lm", tmp_path, env=env)
    assert completed.returncode == 0, completed.stderr


def test_module_entry_point_prints_help(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "mlalign", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--joint-smt-seed-iters" in completed.stdout
    assert "--gpu" not in completed.stdout


def test_relative_paths_resolve_against_the_working_directory(tmp_path):
    code = (
        "from mlalign.cli import build_argument_parser; "
        "from mlalign.config import alignment_config_from_args; "
        "a = build_argument_parser().parse_args(['--input-geom', 'g.json', '--states', 's.root', "
        "'--nominal-full-geom', 'n.json', '--output-geom', 'o.json']); "
        "print(alignment_config_from_args(a).input_geom)"
    )
    completed = _run(code, tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "g.json"
