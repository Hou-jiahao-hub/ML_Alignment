# Absolute Input/Output Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize all ML_Alignment I/O and runtime dependency paths before use.

**Architecture:** Centralize absolute path resolution in `project_config.py`, keep artifact naming in `alignment_artifact_layout.py`, migrate every Python and shell entrypoint to those interfaces, and publish a schema-versioned path inventory.

**Tech Stack:** Python 3.9+, argparse, TOML, Bash, pytest, ACTS/LCG, CMake.

**Spec:** `docs/superpowers/specs/2026-08-21-absolute-io-paths-design.md`

## Global constraints

- Relative CLI and config paths resolve from the repository root.
- Tracked source must not hard-code machine-specific `/home` paths.
- Existing data and experiment outputs must not be moved or overwritten.
- ACTS and conda environments remain isolated.

### Task 1: Central path contract

- [x] Add failing tests for CWD-independent resolution and dependency roots.
- [x] Add public path resolvers, typed config fields, and derived absolute paths.
- [x] Export the complete absolute runtime contract to shell drivers.

### Task 2: Entrypoint migration

- [x] Add failing parser and shell dry-run tests for relative input/output paths.
- [x] Normalize solver, helix, validation, plotting, and migration CLI paths.
- [x] Replace obsolete ACTS and AlignDerivLib directory traversal.
- [x] Pass absolute aligned geometry paths between experiment stages.

### Task 3: Inventory and documentation

- [x] Add failing tests for deterministic absolute inventory output.
- [x] Implement `path_inventory.py --json` and its text form.
- [x] Document producers, consumers, stable paths, and dynamic templates.

### Task 4: Verification

- [x] Run default Python and shell CI.
- [x] Run C++ tests and syntax checks.
- [x] Run ACTS environment smoke checks with the current machine override.
- [x] Review the final diff and commit the isolated feature branch.
