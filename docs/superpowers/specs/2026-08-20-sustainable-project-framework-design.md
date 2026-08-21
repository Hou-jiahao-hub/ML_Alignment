# Sustainable Project Framework Design

## Goal

Make `/home/houjiahao/ML_Alignment` the single source of truth for continued
detector-alignment development while retaining ACTS as an external data
generation and reconstruction-validation runtime.

## Architecture

The repository has two dependency zones. ACTS adapter scripts generate
simulation, geometry, and reconstruction artifacts. The Python alignment core
loads those artifacts and performs analytic-Jacobian, autograd, joint, and
frozen-helix alignment without importing ACTS. Existing script CLIs remain
compatible during this baseline phase; packaging the large algorithm modules is
deferred.

Configuration is loaded from a versioned TOML default and an optional local
override. Target-layer directory names and filename suffixes are derived from
the layer list, so they cannot drift independently. Explicit CLI values override
configuration, followed by `ML_ALIGNMENT_CONFIG`, local configuration, and the
versioned default.

## Data and experiment model

All local artifacts live below `/home/houjiahao/ML_Alignment/data`, which is
ignored by Git. Geometry, simulation, alignment, reconstruction, experiment,
and archive data use separate subdirectories. Every reproducible experiment
writes a small JSON manifest containing the repository commit, effective
configuration, command arguments, input hashes, environment metadata, outputs,
metrics, and status.

Migration is copy-first. Source and destination inventories are compared before
the new data root is enabled. The source data is retained until separately
approved for deletion.

## Engineering contract

- `AGENTS.md` is the Codex entrypoint and names required reading, environment
  isolation rules, safe data practices, and verification commands.
- Pure-Python unit and shell dry-run tests run without ACTS.
- C++ and Python-binding tests form an explicit optional group.
- ACTS end-to-end checks require the LCG environment and run locally or on a
  dedicated runner.
- Current CLI paths remain supported while project configuration is introduced.

## Acceptance criteria

- The current calibration scripts, validation tools, tests, and documentation
  are tracked in `ML_Alignment` with no hand-maintained duplicate source tree.
- Python tests and shell dry-run tests are discoverable from the repository root.
- Effective target layers, suffixes, and directories always agree.
- Core solver modules do not import ACTS.
- Migrated data inventories match, and new defaults resolve below `data/`.
