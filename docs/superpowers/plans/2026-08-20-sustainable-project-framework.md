# Sustainable Project Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `ML_Alignment` as a Codex-ready, independently versioned detector-alignment repository while retaining ACTS as an external runtime.

**Architecture:** Synchronize the proven research workflow first, then add a typed TOML configuration boundary, experiment manifests, explicit dependency/test groups, and local data governance. Preserve current script entrypoints in this phase and defer large-module packaging.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pandas, Awkward, Uproot, SciPy, Matplotlib, pytest, Bash, CMake, pybind11, ACTS/LCG.

**Spec:** `docs/superpowers/specs/2026-08-20-sustainable-project-framework-design.md`

## Global Constraints

- `/home/houjiahao/ML_Alignment` is the only future code source of truth.
- ACTS and conda/ML environments must never be active in the same shell.
- Large artifacts are local-only under `data/` and never committed.
- Existing data is copied and verified; source deletion is outside this plan.
- Existing CLI behavior remains compatible in the baseline phase.

---

### Task 1: Synchronize the current implementation

**Files:** current scripts, tests, and project documentation.

- [ ] Copy the current source-of-truth scripts without generated artifacts.
- [ ] Copy all Python and shell regression tests.
- [ ] Merge current mathematical and optimization documentation.
- [ ] Collect the synchronized test suite from the repository root.

### Task 2: Add project configuration and manifest contracts

**Files:** `configs/default.toml`, `scripts/project_config.py`, and focused tests.

**Interfaces:**
- Produces: `load_project_config(path: str | Path | None = None) -> ProjectConfig`.
- Produces: deterministic target-layer suffix and directory derivation.
- Produces: `write_experiment_manifest(...) -> Path`.

- [ ] Write failing tests for defaults, overrides, derivation, and validation.
- [ ] Implement the minimal typed TOML loader and compatibility exports.
- [ ] Write failing tests for deterministic experiment manifests.
- [ ] Implement manifest creation and atomic JSON writing.

### Task 3: Add repository engineering infrastructure

**Files:** `pyproject.toml`, `Makefile`, CI workflow, `AGENTS.md`, README, and architecture documentation.

- [ ] Declare base, ML, plotting, test, and optional binding dependencies.
- [ ] Separate pure-Python, binding, shell, and ACTS integration test commands.
- [ ] Add CI for pure-Python tests and shell dry-runs.
- [ ] Document architecture, environments, data layout, and Codex rules.

### Task 4: Migrate local artifacts safely

**Files:** ignored `data/` tree and migration inventories.

- [ ] Create the documented local data hierarchy.
- [ ] Copy geometry, simulation, alignment, reconstruction, experiments, and archives.
- [ ] Compare file counts, byte totals, and checksums between source and destination.
- [ ] Configure local defaults to use the migrated data root without deleting sources.

### Task 5: Verify and hand off

- [ ] Run the complete pure-Python test suite.
- [ ] Run shell dry-run tests and configuration contract tests.
- [ ] Run C++ tests and report optional binding status.
- [ ] Run available non-destructive ACTS/ML smoke checks.
- [ ] Review Git diff and repository status before branch handoff.
