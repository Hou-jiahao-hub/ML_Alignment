from __future__ import annotations

import hashlib
import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from project_config import ProjectConfig, load_project_config, resolve_project_path


VALID_STATUSES = {"running", "completed", "failed"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_record(path: Path) -> dict[str, Any]:
    """Hash directory contents and relative names in deterministic order."""
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        total_bytes += size
        file_count += 1
    return {
        "path": str(path),
        "kind": "directory",
        "files": file_count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _git_commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty_state(repository_root: Path) -> dict[str, Any]:
    """Capture enough git state to tell a dirty run apart from its commit.

    ``trial-010``'s manifest recorded only ``git_commit`` while the working
    tree in fact had uncommitted changes; the commit alone could not replay
    that run.  ``dirty=None`` (not ``False``) signals the probe itself failed
    (e.g. ``repository_root`` is not a git repository), so a probe failure
    can never be silently read as "clean".
    """
    diff = subprocess.run(
        ["git", "-C", str(repository_root), "diff", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "--others", "--exclude-standard"],
        check=False,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return {
            "dirty": None,
            "tracked_diff": None,
            "tracked_diff_sha256": None,
            "untracked_files": None,
            "probe_failed": True,
        }
    tracked_diff = diff.stdout or None
    untracked_files = []
    for relative in sorted(line for line in untracked.stdout.splitlines() if line):
        candidate = repository_root / relative
        if candidate.is_file():
            untracked_files.append({"path": relative, "sha256": _sha256(candidate)})
    return {
        "dirty": bool(tracked_diff) or bool(untracked_files),
        "tracked_diff": tracked_diff,
        "tracked_diff_sha256": (
            hashlib.sha256(tracked_diff.encode("utf-8")).hexdigest() if tracked_diff else None
        ),
        "untracked_files": untracked_files,
        "probe_failed": False,
    }


class SourceSnapshotError(RuntimeError):
    """The run's source tree could not be archived, or restored, reliably.

    Unlike ``_git_dirty_state`` (diagnostic, fails open with ``probe_failed``),
    this is the actual replay mechanism for ``trial-010``'s "dirty tree can't
    be replayed from its commit alone" gap, so a failure here must abort the
    run rather than silently record an unusable snapshot.
    """


SOURCE_SNAPSHOT_FORMAT_VERSION = "tar_gz_v1"
#: Reserved member name for the archive's own per-file hash listing. Never a
#: real repository path (git paths cannot start with ``.``+this prefix in
#: practice here), so it cannot collide with archived source files.
_SNAPSHOT_MANIFEST_MEMBER = ".ml_alignment_source_manifest.json"


def _collect_source_files(repository_root: Path, exclude_dir: Path | None) -> list[str]:
    """Every git-tracked path plus non-ignored untracked file, minus ``exclude_dir``.

    Excluding ``exclude_dir`` (the run directory currently being written) is
    required even though run directories normally live under the ignored
    ``data/`` tree: a caller-supplied run directory elsewhere in the
    repository must not fold this run's own manifest/archive into itself.
    """
    tracked = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    untracked = subprocess.run(
        ["git", "-C", str(repository_root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=False,
        capture_output=True,
    )
    if tracked.returncode != 0 or untracked.returncode != 0:
        raise SourceSnapshotError(
            f"git ls-files failed in {repository_root}; cannot build a reliable source snapshot"
        )
    names: set[str] = set()
    for blob in (tracked.stdout, untracked.stdout):
        for raw in blob.split(b"\0"):
            if raw:
                names.add(raw.decode("utf-8"))
    exclude_prefix: str | None = None
    if exclude_dir is not None:
        try:
            exclude_prefix = exclude_dir.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            exclude_prefix = None
    if exclude_prefix:
        names = {
            name
            for name in names
            if name != exclude_prefix and not name.startswith(exclude_prefix + "/")
        }
    return sorted(names)


def _write_source_snapshot(repository_root: Path, destination_dir: Path) -> dict[str, Any]:
    """Archive the exact source bytes this run used, independent of git commit/dirty state."""
    repository_root = repository_root.resolve()
    relative_paths = _collect_source_files(repository_root, destination_dir)
    provenance_dir = destination_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    archive_path = provenance_dir / "source_snapshot.tar.gz"
    file_hashes: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, "w:gz") as archive:
            for relative in relative_paths:
                source_path = repository_root / relative
                if not source_path.is_file():
                    continue
                file_hashes[relative] = _sha256(source_path)
                archive.add(source_path, arcname=relative, recursive=False)
            manifest_bytes = json.dumps(file_hashes, sort_keys=True).encode("utf-8")
            manifest_info = tarfile.TarInfo(name=_SNAPSHOT_MANIFEST_MEMBER)
            manifest_info.size = len(manifest_bytes)
            archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
    except OSError as error:
        raise SourceSnapshotError(f"failed to archive source tree: {error}") from error
    return {
        "format_version": SOURCE_SNAPSHOT_FORMAT_VERSION,
        "archive_path": archive_path.relative_to(destination_dir).as_posix(),
        "archive_sha256": _sha256(archive_path),
        "file_count": len(file_hashes),
        "status": "written",
    }


def _extract_safely(archive: tarfile.TarFile, destination: Path) -> None:
    """``TarFile.extractall`` without the 3.12+ ``filter`` kwarg this repo's Python lacks."""
    destination_resolved = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise SourceSnapshotError(f"refusing to extract unsafe path: {member.name}")
    archive.extractall(destination)


def restore_source_snapshot(manifest_path: str | Path, destination: str | Path) -> Path:
    """Restore a run's exact source tree into an empty ``destination`` directory."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = (manifest.get("git") or {}).get("source_snapshot")
    if not snapshot:
        raise SourceSnapshotError(
            f"manifest {manifest_path} has no recorded source_snapshot to restore"
        )
    run_dir = manifest_path.parent
    archive_path = run_dir / snapshot["archive_path"]
    if not archive_path.is_file():
        raise SourceSnapshotError(f"source snapshot archive is missing: {archive_path}")
    actual_archive_hash = _sha256(archive_path)
    if actual_archive_hash != snapshot["archive_sha256"]:
        raise SourceSnapshotError(
            f"source snapshot archive hash mismatch for {archive_path}: "
            f"expected {snapshot['archive_sha256']}, got {actual_archive_hash}"
        )
    destination = Path(destination)
    if destination.exists():
        if any(destination.iterdir()):
            raise SourceSnapshotError(f"restore destination is not empty: {destination}")
    else:
        destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _extract_safely(archive, destination)
    manifest_member = destination / _SNAPSHOT_MANIFEST_MEMBER
    if not manifest_member.is_file():
        raise SourceSnapshotError("restored archive is missing its internal file manifest")
    recorded_hashes = json.loads(manifest_member.read_text(encoding="utf-8"))
    manifest_member.unlink()
    if len(recorded_hashes) != snapshot["file_count"]:
        raise SourceSnapshotError(
            f"restored file count {len(recorded_hashes)} does not match recorded "
            f"file_count {snapshot['file_count']}"
        )
    for relative, expected_hash in recorded_hashes.items():
        restored_path = destination / relative
        if not restored_path.is_file():
            raise SourceSnapshotError(f"restored tree is missing file: {relative}")
        if _sha256(restored_path) != expected_hash:
            raise SourceSnapshotError(f"restored file does not match recorded hash: {relative}")
    return destination


def _default_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _build_environment_identity(
    *,
    acts_setup: str | None = None,
    conda_env: str | None = None,
    conda_setup: str | None = None,
) -> dict[str, Any]:
    """Record the ACTS/conda identity a pipeline actually used.

    ``experiment_manifest.py`` runs as a bare ``python3`` call outside the
    ``run_conda``/``run_acts`` wrappers in ``run_schur_experiment.sh``,
    so ``_default_environment()`` alone reflects whatever interpreter is on
    ``PATH`` at that point, not the environments the pipeline stages actually
    ran in.  Callers that know those paths (the shell driver does) should
    pass them explicitly.
    """
    environment: dict[str, Any] = dict(_default_environment())
    environment["conda_env_name"] = conda_env
    for label, value in (("acts_setup", acts_setup), ("conda_setup", conda_setup)):
        path = Path(value) if value else None
        environment[f"{label}_path"] = str(path) if path else None
        environment[f"{label}_sha256"] = _sha256(path) if path and path.is_file() else None
    return environment


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=".manifest-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)
    return destination


def _stage_name(command: Sequence[str]) -> str | None:
    """Identify the producing script so two stages cannot share a run directory."""
    for entry in command[1:]:
        text = str(entry)
        if text.endswith(".py"):
            return Path(text).name
    return None


def _guard_existing_manifest(destination: Path, command: Sequence[str]) -> None:
    """Refuse to overwrite a manifest that a different stage wrote.

    A pipeline stage that reuses another stage's run directory used to replace
    its ``manifest.json`` outright, destroying that stage's input hashes and
    configuration with no error.  Re-running the same stage stays allowed.
    """
    if not destination.exists():
        return
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    previous = _stage_name(existing.get("command", []))
    current = _stage_name(command)
    if previous and current and previous != current:
        raise FileExistsError(
            f"refusing to overwrite {destination}: it belongs to a different stage "
            f"('{previous}', not '{current}'). Give each stage its own run directory."
        )


def write_experiment_manifest(
    run_dir: str | Path,
    *,
    run_id: str,
    command: Sequence[str],
    config: ProjectConfig,
    inputs: Mapping[str, str | Path],
    outputs: Mapping[str, str | Path] | None = None,
    metrics: Mapping[str, Any] | None = None,
    status: str = "running",
    created_at: str | None = None,
    git_commit: str | None = None,
    environment: Mapping[str, str] | None = None,
    allow_missing_inputs: bool = False,
    capture_source_snapshot: bool = False,
) -> Path:
    if status not in VALID_STATUSES:
        raise ValueError(f"manifest status must be one of {sorted(VALID_STATUSES)}")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    destination_dir = resolve_project_path(run_dir, base=config.repository_root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "manifest.json"
    _guard_existing_manifest(destination, command)
    manifest_created_at = created_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    manifest_git_commit = git_commit or _git_commit(config.repository_root)
    manifest_environment = dict(environment or _default_environment())
    output_records = {
        name: str(resolve_project_path(value, base=config.repository_root))
        for name, value in (outputs or {}).items()
    }
    if allow_missing_inputs:
        _atomic_write_json(
            destination,
            {
                "schema_version": 2,
                "run_id": run_id,
                "created_at": manifest_created_at,
                "status": "running",
                "git_commit": manifest_git_commit,
                "git": {
                    "commit": manifest_git_commit,
                    "dirty": None,
                    "tracked_diff_sha256": None,
                    "untracked_files": None,
                    "probe_failed": None,
                    "hash_status": "pending",
                    "source_snapshot": {"status": "pending"} if capture_source_snapshot else None,
                },
                "command": list(command),
                "configuration": config.as_dict(),
                "environment": manifest_environment,
                "inputs": {
                    name: {
                        "path": str(resolve_project_path(value, base=config.repository_root)),
                        "exists": resolve_project_path(
                            value, base=config.repository_root
                        ).exists(),
                        "bytes": None,
                        "sha256": None,
                        "hash_status": "pending",
                    }
                    for name, value in inputs.items()
                },
                "outputs": output_records,
                "metrics": dict(metrics or {}),
            },
        )

    # Computed after the preflight manifest is visible on disk, alongside the
    # other slow, potentially-failing work below (matches
    # test_preflight_manifest_exists_before_slow_directory_hash's contract: a
    # crash here must still leave a "running" manifest behind for the caller
    # to finalize as failed, not abort before anything is ever written).
    dirty_state = _git_dirty_state(config.repository_root)
    source_snapshot = (
        _write_source_snapshot(config.repository_root, destination_dir)
        if capture_source_snapshot
        else None
    )
    manifest_git = {
        "commit": manifest_git_commit,
        "dirty": dirty_state["dirty"],
        "tracked_diff_sha256": dirty_state["tracked_diff_sha256"],
        "untracked_files": dirty_state["untracked_files"],
        "probe_failed": dirty_state["probe_failed"],
        "source_snapshot": source_snapshot,
    }

    input_records: dict[str, dict[str, Any]] = {}
    for name, value in inputs.items():
        input_path = resolve_project_path(value, base=config.repository_root)
        if not input_path.exists():
            if allow_missing_inputs:
                input_records[name] = {
                    "path": str(input_path),
                    "exists": False,
                    "bytes": None,
                    "sha256": None,
                }
                continue
            raise FileNotFoundError(f"manifest input not found: {input_path}")
        if input_path.is_dir():
            input_records[name] = _directory_record(input_path)
        elif input_path.is_file():
            input_records[name] = {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": _sha256(input_path),
            }
        else:
            raise FileNotFoundError(f"manifest input is not a file or directory: {input_path}")

    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "created_at": manifest_created_at,
        "status": status,
        "git_commit": manifest_git_commit,
        "git": manifest_git,
        "command": list(command),
        "configuration": config.as_dict(),
        "environment": manifest_environment,
        "inputs": input_records,
        "outputs": output_records,
        "metrics": dict(metrics or {}),
    }

    return _atomic_write_json(destination, manifest)


def finalize_experiment_manifest(
    manifest_path: str | Path,
    *,
    status: str,
    outputs: Mapping[str, str | Path] | None = None,
    metrics: Mapping[str, Any] | None = None,
    completed_at: str | None = None,
) -> Path:
    """Finalize an existing manifest without recomputing immutable input hashes."""
    if status not in {"completed", "failed"}:
        raise ValueError("final manifest status must be 'completed' or 'failed'")
    destination = Path(manifest_path).resolve()
    with destination.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    repository_root = Path(manifest["configuration"]["repository_root"])
    manifest["status"] = status
    manifest["completed_at"] = completed_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    updated_outputs = dict(manifest.get("outputs", {}))
    updated_outputs.update({
        name: str(resolve_project_path(value, base=repository_root))
        for name, value in (outputs or {}).items()
    })
    manifest["outputs"] = updated_outputs
    updated_metrics = dict(manifest.get("metrics", {}))
    updated_metrics.update(metrics or {})
    manifest["metrics"] = updated_metrics
    return _atomic_write_json(destination, manifest)


def update_running_experiment_manifest(
    manifest_path: str | Path,
    *,
    metrics: Mapping[str, Any],
) -> Path:
    """Atomically merge live metrics into a manifest that is still running.

    Long-running callers use this for diagnostic progress without recomputing
    immutable input hashes or changing the terminal lifecycle fields.
    """
    destination = Path(manifest_path).resolve()
    with destination.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "running":
        raise ValueError(f"cannot update non-running manifest: {destination}")
    updated_metrics = dict(manifest.get("metrics", {}))
    updated_metrics.update(metrics)
    manifest["metrics"] = updated_metrics
    return _atomic_write_json(destination, manifest)


def _key_value_pairs(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError(f"Expected non-empty NAME=PATH, got {value!r}")
        result[name] = path
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or finalize experiment manifests.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--run-dir", required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--command-json", required=True)
    start.add_argument("--input", action="append", default=[])
    start.add_argument("--allow-missing-inputs", action="store_true")
    start.add_argument("--config", default=None)
    start.add_argument("--acts-setup", default=None)
    start.add_argument("--conda-env", default=None)
    start.add_argument("--conda-setup", default=None)
    start.add_argument("--capture-source-snapshot", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--status", choices=("completed", "failed"), required=True)
    finalize.add_argument("--output", action="append", default=[])
    finalize.add_argument("--metrics-json", default=None)
    finalize.add_argument("--metrics-file", default=None)

    restore = subparsers.add_parser("restore-source")
    restore.add_argument("--manifest", required=True)
    restore.add_argument("--destination", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.action == "start":
        command = json.loads(args.command_json)
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("--command-json must encode a list of strings")
        manifest = write_experiment_manifest(
            args.run_dir,
            run_id=args.run_id,
            command=command,
            config=load_project_config(args.config),
            inputs=_key_value_pairs(args.input),
            allow_missing_inputs=args.allow_missing_inputs,
            status="running",
            environment=_build_environment_identity(
                acts_setup=args.acts_setup,
                conda_env=args.conda_env,
                conda_setup=args.conda_setup,
            ),
            capture_source_snapshot=args.capture_source_snapshot,
        )
    elif args.action == "finalize":
        if args.metrics_json is not None and args.metrics_file is not None:
            raise ValueError("Use only one of --metrics-json and --metrics-file")
        if args.metrics_file is not None:
            metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
        else:
            metrics = json.loads(args.metrics_json or "{}")
        manifest = finalize_experiment_manifest(
            args.manifest,
            status=args.status,
            outputs=_key_value_pairs(args.output),
            metrics=metrics,
        )
    else:
        restored = restore_source_snapshot(args.manifest, args.destination)
        print(restored)
        return 0
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
