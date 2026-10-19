# ML_Alignment agent guide

## Mission

This repository is the single source of truth for the Schur-LM
detector-alignment algorithm. The algorithm is the installable package
`src/mlalign/`; `scripts/` is the workflow layer around it. ACTS is an external
runtime used to generate inputs and validate geometry; `mlalign` must remain
usable without importing ACTS.

## Read before changing algorithms

1. `docs/architecture.md`
2. `docs/history/ml_optimization_log.md`
3. The module docstring of `src/mlalign/schur_lm.py` (pre-registered contract)

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
- Relative configuration and workflow CLI paths resolve from the repository
  root; `python scripts/alignment_artifact_layout.py --shell` prints the
  effective absolute paths. `mlalign-align` resolves relative paths against the
  working directory and reads no TOML: its defaults are the constants in
  `src/mlalign/config.py`, which must agree with `configs/default.toml`.
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
- `mlalign`, acceptance, plotting, configuration, and manifest code must
  remain importable without ACTS.
- `mlalign` must not import anything from `scripts/` (no TOML, no repository
  paths). New solver code goes into the `mlalign` module that owns it (see the
  layering in `docs/architecture.md`).
- Each module has one import name. `mlalign/__init__.py` re-exports the public
  API only; monkeypatch a name in its defining module, never via a re-export.
- The pipeline runs on the CPU in float64; there is no GPU path.

## Verification

- Quick Python suite: `make test`
- Shell path/dry-run contracts: `make test-shell`
- Both: `make ci`
- Environment-sensitive ROOT I/O: `make test-root-io`
- Tests import the installed package: run `python -m pip install -e .` in the
  conda environment once.

For algorithm changes, add a failing regression test first. Record experiment
configuration, input hashes, metrics, and Git commit in `manifest.json`.
