from __future__ import annotations

import json
from pathlib import Path

import pytest


def _source_tree(root: Path) -> None:
    (root / "simulation_output").mkdir(parents=True)
    (root / "reconstruction_output" / "nominal").mkdir(parents=True)
    (root / "archive" / "logs").mkdir(parents=True)
    (root / "odd_transforms.json").write_bytes(b"geometry")
    (root / "simulation_output" / "particles.root").write_bytes(b"simulation")
    (root / "reconstruction_output" / "nominal" / "tracks.root").write_bytes(
        b"tracks"
    )
    (root / "archive" / "logs" / "old.log").write_bytes(b"history")


def test_migration_copies_canonical_layout_and_writes_verified_inventory(tmp_path):
    from migrate_local_data import migrate_local_data

    source = tmp_path / "job"
    destination = tmp_path / "repository" / "data"
    _source_tree(source)

    result = migrate_local_data(source, destination)

    assert (destination / "geometry" / "odd_transforms.json").read_bytes() == b"geometry"
    assert (destination / "simulation" / "particles.root").read_bytes() == b"simulation"
    assert (
        destination / "reconstruction" / "nominal" / "tracks.root"
    ).read_bytes() == b"tracks"
    assert (destination / "archive" / "logs" / "old.log").read_bytes() == b"history"
    assert result["verified"] is True
    assert result["file_count"] == 4
    manifest = json.loads(
        (destination / "migration-inventory.json").read_text(encoding="utf-8")
    )
    assert manifest["verified"] is True
    assert manifest["total_bytes"] == sum(len(value) for value in (b"geometry", b"simulation", b"tracks", b"history"))


def test_migration_refuses_to_overwrite_different_existing_file(tmp_path):
    from migrate_local_data import migrate_local_data

    source = tmp_path / "job"
    destination = tmp_path / "data"
    _source_tree(source)
    target = destination / "geometry" / "odd_transforms.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="differs"):
        migrate_local_data(source, destination)

    assert target.read_bytes() == b"do-not-overwrite"


def test_verify_only_detects_destination_corruption(tmp_path):
    from migrate_local_data import migrate_local_data, verify_migration

    source = tmp_path / "job"
    destination = tmp_path / "data"
    _source_tree(source)
    migrate_local_data(source, destination)
    (destination / "simulation" / "particles.root").write_bytes(b"corruption")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_migration(source, destination)


def test_dry_run_does_not_create_destination(tmp_path):
    from migrate_local_data import migrate_local_data

    source = tmp_path / "job"
    destination = tmp_path / "data"
    _source_tree(source)

    result = migrate_local_data(source, destination, dry_run=True)

    assert result["file_count"] == 4
    assert not destination.exists()
