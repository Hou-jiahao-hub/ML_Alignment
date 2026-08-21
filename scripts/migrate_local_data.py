from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from project_config import absolute_project_path


MIGRATION_MAP = (
    ("odd_transforms.json", "geometry/odd_transforms.json"),
    ("simulation_output", "simulation"),
    ("alignment_output", "alignment"),
    ("reconstruction_output", "reconstruction"),
    ("iterative_ml_logs", "experiments/iterative_ml"),
    ("surface_helix_logs", "experiments/surface_helix"),
    ("helix_validation_output", "experiments/helix_validation"),
    ("ablation", "experiments/ablation"),
    ("archive", "archive"),
    ("alignLog", "archive/logs/alignLog"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _migration_files(
    source_root: Path, destination_root: Path
) -> Iterable[tuple[Path, Path, str]]:
    for source_name, destination_name in MIGRATION_MAP:
        source = source_root / source_name
        destination = destination_root / destination_name
        if source.is_file():
            yield source, destination, destination_name
        elif source.is_dir():
            for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
                relative = source_file.relative_to(source)
                destination_file = destination / relative
                yield (
                    source_file,
                    destination_file,
                    destination_file.relative_to(destination_root).as_posix(),
                )


def _validated_roots(source: str | Path, destination: str | Path) -> tuple[Path, Path]:
    source_root = Path(source).expanduser().resolve()
    destination_root = Path(destination).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"migration source directory not found: {source_root}")
    if source_root == destination_root or source_root in destination_root.parents:
        raise ValueError("migration destination must be outside the source tree")
    return source_root, destination_root


def _inventory(source_root: Path, destination_root: Path) -> list[dict]:
    entries = []
    for source, destination, relative in _migration_files(source_root, destination_root):
        if not destination.is_file():
            raise ValueError(f"missing destination file: {destination}")
        source_size = source.stat().st_size
        destination_size = destination.stat().st_size
        if source_size != destination_size:
            raise ValueError(f"size mismatch for {relative}")
        source_hash = _sha256(source)
        destination_hash = _sha256(destination)
        if source_hash != destination_hash:
            raise ValueError(f"checksum mismatch for {relative}")
        entries.append(
            {
                "path": relative,
                "bytes": source_size,
                "sha256": source_hash,
            }
        )
    return entries


def verify_migration(source: str | Path, destination: str | Path) -> dict:
    source_root, destination_root = _validated_roots(source, destination)
    entries = _inventory(source_root, destination_root)
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "verified": True,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }


def _write_inventory(destination_root: Path, inventory: dict) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / "migration-inventory.json"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination_root,
        prefix=".migration-inventory-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(destination)
    return destination


def migrate_local_data(
    source: str | Path, destination: str | Path, *, dry_run: bool = False
) -> dict:
    source_root, destination_root = _validated_roots(source, destination)
    files = list(_migration_files(source_root, destination_root))
    summary = {
        "schema_version": 1,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "verified": False,
        "file_count": len(files),
        "total_bytes": sum(source_file.stat().st_size for source_file, _, _ in files),
        "entries": [],
    }
    if dry_run:
        return summary

    for index, (source_file, destination_file, relative) in enumerate(files, start=1):
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if destination_file.exists():
            if not destination_file.is_file():
                raise FileExistsError(f"destination is not a file: {destination_file}")
            same_size = source_file.stat().st_size == destination_file.stat().st_size
            if not same_size or _sha256(source_file) != _sha256(destination_file):
                raise FileExistsError(
                    f"destination differs and will not be overwritten: {relative}"
                )
        else:
            shutil.copy2(source_file, destination_file)
        if index % 100 == 0 or index == len(files):
            print(f"copied/verified {index}/{len(files)} files")

    inventory = verify_migration(source_root, destination_root)
    _write_inventory(destination_root, inventory)
    return inventory


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy detector-alignment artifacts without deleting or overwriting sources."
    )
    parser.add_argument(
        "--source", type=absolute_project_path, required=True,
        help="Former ACTS job directory",
    )
    parser.add_argument(
        "--destination", type=absolute_project_path, required=True,
        help="ML_Alignment data directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inventory only; write nothing")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify an existing migration"
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    result = (
        verify_migration(args.source, args.destination)
        if args.verify_only
        else migrate_local_data(args.source, args.destination, dry_run=args.dry_run)
    )
    print(
        f"files={result['file_count']} bytes={result['total_bytes']} "
        f"verified={result['verified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
