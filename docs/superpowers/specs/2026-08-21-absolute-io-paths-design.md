# Absolute Input/Output Path Design

## Goal

Make every ML_Alignment I/O and runtime dependency path independent of the
process working directory and of the scripts' former ACTS nesting.

## Architecture

`project_config.py` is the only path-resolution boundary. Versioned TOML may
contain portable relative paths, but `ProjectConfig`, argparse consumers, and
shell exports expose normalized absolute paths. Relative values are anchored to
the repository root. Machine-specific roots remain in ignored local config.

`alignment_artifact_layout.py` continues to own target-layer naming and data
layout. A separate read-only inventory command exposes roots, external inputs,
stable artifacts, output directories, and dynamic output templates as JSON.

## Compatibility

Absolute CLI paths retain their meaning. Relative CLI paths intentionally stop
depending on the caller's working directory and always resolve from the
repository root. Shell drivers pass absolute aligned-geometry paths to the ACTS
adapter rather than relying on its former alignment-directory special case.

## Acceptance criteria

- No production entrypoint derives ACTS or AlignDerivLib through obsolete
  parent-directory traversal.
- Calling an entrypoint from the repository or another directory produces the
  same effective paths.
- Path inventory JSON is deterministic, schema-versioned, and absolute.
- Python, shell, C++, and available ACTS smoke tests pass without moving data.
