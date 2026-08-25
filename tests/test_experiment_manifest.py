from __future__ import annotations

import hashlib
import json
from dataclasses import replace


def test_manifest_records_reproducible_inputs_and_effective_configuration(tmp_path):
    from experiment_manifest import write_experiment_manifest
    from project_config import load_project_config

    input_path = tmp_path / "states.root"
    input_path.write_bytes(b"fixture-data")
    run_dir = tmp_path / "runs" / "smoke-001"

    manifest_path = write_experiment_manifest(
        run_dir,
        run_id="smoke-001",
        command=["python", "scripts/ml_alignment.py", "--epochs", "2"],
        config=load_project_config(),
        inputs={"states": input_path},
        outputs={"geometry": run_dir / "aligned.json"},
        metrics={"loss": 1.25},
        status="completed",
        created_at="2026-08-20T12:00:00Z",
        git_commit="abc123",
        environment={"python": "3.10.0", "platform": "test"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == run_dir / "manifest.json"
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "smoke-001"
    assert manifest["git_commit"] == "abc123"
    assert manifest["configuration"]["target_layers"] == ["17:8", "24:4"]
    assert manifest["inputs"]["states"] == {
        "path": str(input_path.resolve()),
        "bytes": 12,
        "sha256": hashlib.sha256(b"fixture-data").hexdigest(),
    }
    assert manifest["outputs"]["geometry"] == str(
        (run_dir / "aligned.json").resolve()
    )
    assert manifest["metrics"] == {"loss": 1.25}
    assert manifest["status"] == "completed"


def test_manifest_rejects_unknown_status_and_missing_input(tmp_path):
    from experiment_manifest import write_experiment_manifest
    from project_config import load_project_config

    config = load_project_config()

    try:
        write_experiment_manifest(
            tmp_path / "run",
            run_id="bad",
            command=["false"],
            config=config,
            inputs={"states": tmp_path / "missing.root"},
            status="unknown",
        )
    except ValueError as error:
        assert "status" in str(error)
    else:
        raise AssertionError("unknown manifest status was accepted")

    try:
        write_experiment_manifest(
            tmp_path / "run",
            run_id="missing",
            command=["python"],
            config=config,
            inputs={"states": tmp_path / "missing.root"},
            status="failed",
        )
    except FileNotFoundError as error:
        assert "missing.root" in str(error)
    else:
        raise AssertionError("missing manifest input was accepted")


def test_manifest_relative_paths_use_configured_repository_root(monkeypatch, tmp_path):
    from experiment_manifest import write_experiment_manifest
    from project_config import load_project_config

    repository_root = tmp_path / "repository"
    input_path = repository_root / "inputs" / "states.root"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"states")
    config = replace(load_project_config(), repository_root=repository_root)
    external_cwd = tmp_path / "caller"
    external_cwd.mkdir()
    monkeypatch.chdir(external_cwd)

    manifest_path = write_experiment_manifest(
        "runs/test",
        run_id="test",
        command=["python", "solver.py"],
        config=config,
        inputs={"states": "inputs/states.root"},
        outputs={"geometry": "outputs/aligned.json"},
        git_commit="abc123",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == repository_root / "runs" / "test" / "manifest.json"
    assert manifest["inputs"]["states"]["path"] == str(input_path)
    assert manifest["outputs"]["geometry"] == str(
        repository_root / "outputs" / "aligned.json"
    )


def test_finalize_manifest_updates_status_without_rehashing_inputs(tmp_path, monkeypatch):
    import experiment_manifest
    from project_config import load_project_config

    input_path = tmp_path / "states.root"
    input_path.write_bytes(b"original")
    manifest_path = experiment_manifest.write_experiment_manifest(
        tmp_path / "run",
        run_id="helix-validation",
        command=["python", "scripts/validate_helix_model.py"],
        config=load_project_config(),
        inputs={"states": input_path},
        status="running",
        git_commit="abc123",
    )
    original_hash = json.loads(manifest_path.read_text())["inputs"]["states"]["sha256"]
    input_path.write_bytes(b"changed-after-start")
    monkeypatch.setattr(
        experiment_manifest,
        "_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("input was rehashed")),
    )

    experiment_manifest.finalize_experiment_manifest(
        manifest_path,
        status="completed",
        outputs={"summary": tmp_path / "run" / "summary.json"},
        metrics={"passed": False, "local0_rmse_mm": 0.2},
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["inputs"]["states"]["sha256"] == original_hash
    assert manifest["metrics"]["passed"] is False
    assert manifest["outputs"]["summary"].endswith("summary.json")


def test_manifest_can_record_missing_preflight_input_and_hash_directory(tmp_path):
    from experiment_manifest import write_experiment_manifest
    from project_config import load_project_config

    dataset = tmp_path / "simulation"
    dataset.mkdir()
    (dataset / "event1.csv").write_bytes(b"one")
    (dataset / "event2.csv").write_bytes(b"two")
    missing = tmp_path / "missing.root"

    manifest_path = write_experiment_manifest(
        tmp_path / "run",
        run_id="preflight",
        command=["bash", "run.sh"],
        config=load_project_config(),
        inputs={"simulation": dataset, "states": missing},
        allow_missing_inputs=True,
    )

    inputs = json.loads(manifest_path.read_text())["inputs"]
    assert inputs["states"] == {
        "path": str(missing.resolve()),
        "exists": False,
        "bytes": None,
        "sha256": None,
    }
    assert inputs["simulation"]["kind"] == "directory"
    assert inputs["simulation"]["files"] == 2
    assert inputs["simulation"]["bytes"] == 6
    assert len(inputs["simulation"]["sha256"]) == 64


def test_preflight_manifest_exists_before_slow_directory_hash(
    tmp_path, monkeypatch
):
    import experiment_manifest
    from project_config import load_project_config

    dataset = tmp_path / "simulation"
    dataset.mkdir()
    run_dir = tmp_path / "run"

    def inspect_preflight(path):
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "running"
        assert manifest["inputs"]["simulation"]["hash_status"] == "pending"
        return {
            "path": str(path),
            "kind": "directory",
            "files": 0,
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }

    monkeypatch.setattr(experiment_manifest, "_directory_record", inspect_preflight)
    experiment_manifest.write_experiment_manifest(
        run_dir,
        run_id="preflight-visible",
        command=["bash", "run.sh"],
        config=load_project_config(),
        inputs={"simulation": dataset},
        allow_missing_inputs=True,
    )


def test_finalize_manifest_merges_existing_stage_observability(tmp_path):
    from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from project_config import load_project_config

    source = tmp_path / "input.root"
    source.write_bytes(b"input")
    manifest_path = write_experiment_manifest(
        tmp_path / "run",
        run_id="merge",
        command=["bash", "run.sh"],
        config=load_project_config(),
        inputs={"states": source},
        outputs={"warm1": tmp_path / "warm1.json"},
        metrics={"warm_rounds_completed": 1},
    )

    finalize_experiment_manifest(
        manifest_path,
        status="failed",
        outputs={"joint": tmp_path / "joint.json"},
        metrics={"stop_reason": "joint_failed"},
    )

    manifest = json.loads(manifest_path.read_text())
    assert sorted(manifest["outputs"]) == ["joint", "warm1"]
    assert manifest["metrics"] == {
        "warm_rounds_completed": 1,
        "stop_reason": "joint_failed",
    }


def test_manifest_cli_start_and_finalize_support_driver_exit_trap(tmp_path):
    import experiment_manifest

    input_path = tmp_path / "input.root"
    input_path.write_bytes(b"input")
    run_dir = tmp_path / "trial-006"
    assert experiment_manifest.main(
        [
            "start",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "trial-006",
            "--command-json",
            '["bash", "run.sh"]',
            "--input",
            f"states={input_path}",
        ]
    ) == 0
    manifest_path = run_dir / "manifest.json"
    assert json.loads(manifest_path.read_text())["status"] == "running"

    assert experiment_manifest.main(
        [
            "finalize",
            "--manifest",
            str(manifest_path),
            "--status",
            "failed",
            "--metrics-json",
            '{"stop_reason": "interrupted"}',
        ]
    ) == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["stop_reason"] == "interrupted"
