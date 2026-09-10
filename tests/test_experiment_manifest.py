from __future__ import annotations

import hashlib
import json
import subprocess
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
    assert manifest["schema_version"] == 2
    assert manifest["run_id"] == "smoke-001"
    assert manifest["git_commit"] == "abc123"
    assert manifest["git"]["commit"] == "abc123"
    assert isinstance(manifest["git"]["dirty"], bool)
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


def test_manifest_refuses_to_clobber_another_stage_in_the_same_directory(tmp_path):
    """Two stages sharing a run directory silently destroyed each other's record.

    The exporter and the converter both wrote ``manifest.json`` into
    ``<target>/<run_tag>/``; whichever ran last won, and the ACTS-side
    ``exporter_config`` plus input hashes were lost without any error.
    """
    import pytest
    from project_config import load_project_config

    import experiment_manifest as module

    config = load_project_config()
    source = tmp_path / "input.txt"
    source.write_text("payload", encoding="utf-8")

    first = module.write_experiment_manifest(
        tmp_path / "run",
        run_id="shared-tag",
        command=["/usr/bin/python", "/repo/scripts/alignment_odd.py", "material"],
        config=config,
        inputs={"simulation_dir": source},
        status="completed",
    )
    assert json.loads(first.read_text())["metrics"] == {}

    with pytest.raises(FileExistsError, match="different stage"):
        module.write_experiment_manifest(
            tmp_path / "run",
            run_id="shared-tag",
            command=["/usr/bin/python", "/repo/scripts/material_artifact.py"],
            config=config,
            inputs={"material_root": source},
            status="running",
        )
    # The original record survives untouched.
    kept = json.loads(first.read_text())
    assert kept["command"][1].endswith("alignment_odd.py")
    assert "simulation_dir" in kept["inputs"]


def test_manifest_allows_rerunning_the_same_stage(tmp_path):
    from project_config import load_project_config

    import experiment_manifest as module

    config = load_project_config()
    source = tmp_path / "input.txt"
    source.write_text("payload", encoding="utf-8")
    command = ["/usr/bin/python", "/repo/scripts/material_artifact.py"]

    module.write_experiment_manifest(
        tmp_path / "run",
        run_id="tag",
        command=command,
        config=config,
        inputs={"material_root": source},
        status="failed",
    )
    again = module.write_experiment_manifest(
        tmp_path / "run",
        run_id="tag",
        command=command,
        config=config,
        inputs={"material_root": source},
        status="running",
    )
    assert json.loads(again.read_text())["status"] == "running"


def _init_git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


def test_manifest_records_clean_git_state(tmp_path):
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "tracked.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)

    config = replace(load_project_config(), repository_root=repo)
    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="clean",
        command=["python"],
        config=config,
        inputs={},
        status="running",
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["git"]["dirty"] is False
    assert manifest["git"]["tracked_diff_sha256"] is None
    assert manifest["git"]["untracked_files"] == []
    assert manifest["git"]["probe_failed"] is False


def test_manifest_records_dirty_git_state(tmp_path):
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    tracked = repo / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    tracked.write_text("changed\n", encoding="utf-8")
    (repo / "scratch.txt").write_text("new-file\n", encoding="utf-8")

    config = replace(load_project_config(), repository_root=repo)
    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="dirty",
        command=["python"],
        config=config,
        inputs={},
        status="running",
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["git"]["dirty"] is True
    assert manifest["git"]["probe_failed"] is False
    assert manifest["git"]["tracked_diff_sha256"] is not None
    assert manifest["git"]["untracked_files"] == [
        {"path": "scratch.txt", "sha256": hashlib.sha256(b"new-file\n").hexdigest()}
    ]


def test_manifest_git_state_probe_failure_is_explicit(tmp_path):
    from project_config import load_project_config

    import experiment_manifest as module

    non_git = tmp_path / "not-a-repo"
    non_git.mkdir()
    config = replace(load_project_config(), repository_root=non_git)

    manifest_path = module.write_experiment_manifest(
        non_git / "run",
        run_id="probe-fail",
        command=["python"],
        config=config,
        inputs={},
        status="running",
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["git"]["probe_failed"] is True
    assert manifest["git"]["dirty"] is None
    assert manifest["git"]["untracked_files"] is None


def _init_source_repo(repo: Path) -> None:
    """A small repo with a tracked file, an untracked file, and an ignored one.

    Mirrors the real layout's ``data/*`` ignore rule: the source snapshot must
    include the first two and must never pick up the third.
    """
    _init_git_repo(repo)
    (repo / "tracked.py").write_text("print('tracked')\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "tracked.py", ".gitignore"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    (repo / "untracked.py").write_text("print('untracked')\n", encoding="utf-8")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "scratch.dat").write_text("must not be archived\n", encoding="utf-8")


def test_source_snapshot_captures_tracked_and_untracked_files_and_restores_bit_identical(
    tmp_path,
):
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_source_repo(repo)
    config = replace(load_project_config(), repository_root=repo)

    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="snapshot",
        command=["python"],
        config=config,
        inputs={},
        status="running",
        capture_source_snapshot=True,
    )
    manifest = json.loads(manifest_path.read_text())
    snapshot = manifest["git"]["source_snapshot"]
    assert snapshot["format_version"]
    assert snapshot["file_count"] == 3  # tracked.py, .gitignore, untracked.py
    assert len(snapshot["archive_sha256"]) == 64

    destination = tmp_path / "restored"
    restored = module.restore_source_snapshot(manifest_path, destination)
    assert restored == destination
    assert (destination / "tracked.py").read_text(encoding="utf-8") == "print('tracked')\n"
    assert (destination / "untracked.py").read_text(encoding="utf-8") == "print('untracked')\n"
    assert not (destination / "ignored").exists()
    assert not (destination / "run").exists()
    restored_files = sorted(
        str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()
    )
    assert restored_files == [".gitignore", "tracked.py", "untracked.py"]


def test_write_experiment_manifest_defaults_to_skipping_source_snapshot(tmp_path):
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_source_repo(repo)
    config = replace(load_project_config(), repository_root=repo)

    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="no-snapshot",
        command=["python"],
        config=config,
        inputs={},
        status="running",
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git"]["source_snapshot"] is None
    assert not (repo / "run" / "provenance").exists()


def test_write_experiment_manifest_raises_and_leaves_running_manifest_when_source_archiving_fails(
    tmp_path,
):
    import pytest
    from project_config import load_project_config

    import experiment_manifest as module

    non_git = tmp_path / "not-a-repo"
    non_git.mkdir()
    config = replace(load_project_config(), repository_root=non_git)
    run_dir = non_git / "run"

    with pytest.raises(module.SourceSnapshotError):
        module.write_experiment_manifest(
            run_dir,
            run_id="archiving-fails",
            command=["python"],
            config=config,
            inputs={},
            allow_missing_inputs=True,
            capture_source_snapshot=True,
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "running"


def test_restore_source_fails_closed_on_archive_hash_mismatch(tmp_path):
    import pytest
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_source_repo(repo)
    config = replace(load_project_config(), repository_root=repo)
    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="tamper",
        command=["python"],
        config=config,
        inputs={},
        capture_source_snapshot=True,
    )
    manifest = json.loads(manifest_path.read_text())
    archive_path = manifest_path.parent / manifest["git"]["source_snapshot"]["archive_path"]
    archive_path.write_bytes(archive_path.read_bytes() + b"\x00")

    with pytest.raises(module.SourceSnapshotError, match="hash mismatch"):
        module.restore_source_snapshot(manifest_path, tmp_path / "restored")


def test_restore_source_fails_closed_on_missing_archive_file(tmp_path):
    import pytest
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_source_repo(repo)
    config = replace(load_project_config(), repository_root=repo)
    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="delete-archive",
        command=["python"],
        config=config,
        inputs={},
        capture_source_snapshot=True,
    )
    manifest = json.loads(manifest_path.read_text())
    archive_path = manifest_path.parent / manifest["git"]["source_snapshot"]["archive_path"]
    archive_path.unlink()

    with pytest.raises(module.SourceSnapshotError, match="missing"):
        module.restore_source_snapshot(manifest_path, tmp_path / "restored")


def test_restore_source_fails_closed_when_manifest_has_no_source_snapshot(tmp_path):
    import pytest
    from project_config import load_project_config

    import experiment_manifest as module

    manifest_path = module.write_experiment_manifest(
        tmp_path / "run",
        run_id="no-snapshot-recorded",
        command=["python"],
        config=load_project_config(),
        inputs={},
    )

    with pytest.raises(module.SourceSnapshotError, match="no recorded source_snapshot"):
        module.restore_source_snapshot(manifest_path, tmp_path / "restored")


def test_restore_source_refuses_nonempty_destination(tmp_path):
    import pytest
    from project_config import load_project_config

    import experiment_manifest as module

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_source_repo(repo)
    config = replace(load_project_config(), repository_root=repo)
    manifest_path = module.write_experiment_manifest(
        repo / "run",
        run_id="nonempty-destination",
        command=["python"],
        config=config,
        inputs={},
        capture_source_snapshot=True,
    )
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "leftover.txt").write_text("stale\n", encoding="utf-8")

    with pytest.raises(module.SourceSnapshotError, match="not empty"):
        module.restore_source_snapshot(manifest_path, destination)


def test_manifest_cli_start_records_source_snapshot_when_flag_set(tmp_path):
    import experiment_manifest

    run_dir = tmp_path / "run"
    assert experiment_manifest.main(
        [
            "start",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "cli-snapshot",
            "--command-json",
            '["bash", "run.sh"]',
            "--capture-source-snapshot",
        ]
    ) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    snapshot = manifest["git"]["source_snapshot"]
    assert snapshot["file_count"] > 0
    assert (run_dir / snapshot["archive_path"]).is_file()

    restored = tmp_path / "restored"
    assert experiment_manifest.main(
        [
            "restore-source",
            "--manifest",
            str(run_dir / "manifest.json"),
            "--destination",
            str(restored),
        ]
    ) == 0
    assert (restored / "scripts" / "experiment_manifest.py").is_file()


def test_manifest_cli_start_records_acts_and_conda_identity(tmp_path):
    import experiment_manifest

    input_path = tmp_path / "input.root"
    input_path.write_bytes(b"input")
    acts_setup = tmp_path / "setup_acts.sh"
    acts_setup.write_text("#!/usr/bin/env bash\necho acts\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    assert experiment_manifest.main(
        [
            "start",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "env-identity",
            "--command-json",
            '["bash", "run.sh"]',
            "--input",
            f"states={input_path}",
            "--acts-setup",
            str(acts_setup),
            "--conda-env",
            "pytorch",
            "--conda-setup",
            str(acts_setup),
        ]
    ) == 0

    manifest = json.loads((run_dir / "manifest.json").read_text())
    environment = manifest["environment"]
    assert environment["conda_env_name"] == "pytorch"
    assert environment["acts_setup_path"] == str(acts_setup)
    assert environment["acts_setup_sha256"] == hashlib.sha256(acts_setup.read_bytes()).hexdigest()
    assert environment["conda_setup_sha256"] == hashlib.sha256(acts_setup.read_bytes()).hexdigest()
