# ML_Alignment agent guide

## Mission

This repository is the single source of truth for detector-alignment algorithm
development. ACTS is an external runtime used to generate inputs and validate
geometry; core solvers must remain usable without importing ACTS.

## Read before changing algorithms

1. `docs/architecture.md`
2. `docs/ml_alignment_principles.md`
3. `docs/history/ml_optimization_log.md`

The optimization history records failed approaches and observability limits.
Do not repeat an experiment without explaining what changed in its assumptions.

## Code comprehension

Before accessing or using any variable, read its definition and the relevant
surrounding implementation. Never infer a variable's actual purpose from its
name alone.

## Environments

- ACTS simulation/reconstruction: source `setup_acts.sh` in a fresh shell.
- ML, plots, and unit tests: use the conda `pytorch` environment.
- Never source LCG and then run conda Python in the same shell. LCG sets
  `PYTHONHOME`, which breaks the conda interpreter.

## Configuration and data

- Versioned defaults: `configs/default.toml`.
- Machine override: `configs/local.toml` or `ML_ALIGNMENT_CONFIG`.
- CLI overrides use `--config PATH`.
- Relative configuration and CLI paths resolve from the repository root; use
  `python scripts/path_inventory.py --json` to inspect effective absolute paths.
- Target directory and suffix are derived from `alignment.target_layers`; never
  hard-code them in another module.
- Large artifacts belong below `data/` and must not be committed.
- Never delete or overwrite source ROOT/JSON artifacts while migrating data.
  Copy, inventory, verify, switch configuration, then request explicit approval
  before removing the source.

## Experiment records

Every algorithm experiment, including failed or aborted runs, must write a
schema-versioned `manifest.json` recording the effective configuration,
command, input hashes, metrics available at termination, outputs, status, and
Git commit. An experiment is not complete until its manifest has been written.

## Dependency boundaries

- Only ACTS adapter scripts (`alignment_odd.py`, `generate_odd_transform.py`) may
  import `acts`.
- `ml_alignment*.py`, validation, plotting, configuration, and manifest code
  must remain importable without ACTS.
- Keep existing script CLIs compatible during the baseline phase. Large-module
  package refactoring is a separate project.

## Verification

- Quick Python suite: `make test`
- Shell path/dry-run contracts: `make test-shell`
- Both: `make ci`
- Environment-sensitive ROOT I/O: `make test-root-io`
- Header-only C++ core: `make test-cpp`
- Optional C++/pybind11 build: `make test-bindings`

For algorithm changes, add a failing regression test first. Record experiment
configuration, input hashes, metrics, and Git commit in `manifest.json`.
