from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from project_config import ProjectConfig, resolve_project_path


VALID_STATUSES = {"running", "completed", "failed"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> Path:
    if status not in VALID_STATUSES:
        raise ValueError(f"manifest status must be one of {sorted(VALID_STATUSES)}")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    input_records: dict[str, dict[str, Any]] = {}
    for name, value in inputs.items():
        input_path = resolve_project_path(value, base=config.repository_root)
        if not input_path.is_file():
            raise FileNotFoundError(f"manifest input not found: {input_path}")
        input_records[name] = {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": _sha256(input_path),
        }

    output_records = {
        name: str(resolve_project_path(value, base=config.repository_root))
        for name, value in (outputs or {}).items()
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "git_commit": git_commit or _git_commit(config.repository_root),
        "command": list(command),
        "configuration": config.as_dict(),
        "environment": dict(environment or _default_environment()),
        "inputs": input_records,
        "outputs": output_records,
        "metrics": dict(metrics or {}),
    }

    destination_dir = resolve_project_path(run_dir, base=config.repository_root)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "manifest.json"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination_dir,
        prefix=".manifest-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)
    return destination
