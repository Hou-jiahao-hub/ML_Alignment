from __future__ import annotations

import hashlib
import argparse
import json
import platform
import subprocess
import sys
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


def _default_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


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
) -> Path:
    if status not in VALID_STATUSES:
        raise ValueError(f"manifest status must be one of {sorted(VALID_STATUSES)}")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    destination_dir = resolve_project_path(run_dir, base=config.repository_root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "manifest.json"
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
                "schema_version": 1,
                "run_id": run_id,
                "created_at": manifest_created_at,
                "status": "running",
                "git_commit": manifest_git_commit,
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
        "schema_version": 1,
        "run_id": run_id,
        "created_at": manifest_created_at,
        "status": status,
        "git_commit": manifest_git_commit,
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

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", required=True)
    finalize.add_argument("--status", choices=("completed", "failed"), required=True)
    finalize.add_argument("--output", action="append", default=[])
    finalize.add_argument("--metrics-json", default=None)
    finalize.add_argument("--metrics-file", default=None)
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
        )
    else:
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
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
