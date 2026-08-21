from __future__ import annotations

import hashlib
import json


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
