# ML_Alignment

Detector-alignment research and validation built around ACTS-produced tracking
data. ACTS remains the simulation and reconstruction runtime; analytic,
autograd, joint, and frozen-helix solvers are developed in this repository.

## Repository map

```text
ML_Alignment/
├── AlignDerivLib/     C++/pybind11 analytic alignment Jacobians
├── configs/           versioned defaults and local override template
├── data/              ignored local geometry, ROOT, experiments, and archives
├── docs/              architecture, mathematics, and experiment history
├── scripts/           ACTS adapters, solvers, validation, plots, and drivers
├── tests/             Python and shell regression tests
├── AGENTS.md          Codex development contract
├── Makefile           verification entrypoints
└── pyproject.toml     Python dependencies and pytest configuration
```

The detailed component and data flow is in [docs/architecture.md](docs/architecture.md).
The complete input/output contract is in [docs/io-paths.md](docs/io-paths.md).
Research notes and the model decision for material-aware track fitting are in
[docs/research/material_aware_tracks/](docs/research/material_aware_tracks/README.md).

## Environment setup

Two environments are required and must not be combined in one shell.

### ML and tests

Use the existing conda PyTorch environment, or install the Python dependencies:

```bash
python -m pip install -e '.[test]'
make ci
```

PyROOT is a system/conda dependency and is not installed by `pyproject.toml`.

### ACTS simulation and reconstruction

For installing and mounting `sft.cern.ch` on a standalone workstation, follow
the [local CVMFS workstation guide](docs/cvmfs-local-workstation-guide.md).

Copy the machine template if the ACTS checkout is not the default sibling path:

```bash
cp configs/local.toml.example configs/local.toml
source setup_acts.sh
```

`setup_acts.sh` uses the configured/existing `/home/houjiahao/alignTest` checkout.
LCG modifies `PYTHONHOME`; leave that shell before activating conda.

## Configuration

Defaults live in `configs/default.toml`:

```toml
[project]
data_root = "data"
align_derivlib_root = "AlignDerivLib"

[acts]
root = "../alignTest"

[runtime]
conda_setup = "$HOME/miniconda3/etc/profile.d/conda.sh"

[alignment]
target_layers = ["17:8", "24:4"]
active_dof = [0, 1, 5]
```

Use an ignored `configs/local.toml`, `ML_ALIGNMENT_CONFIG`, or `--config PATH`
for an override. Target suffixes and output directory names are derived from
`target_layers`; do not edit them separately.

Configuration and CLI paths are converted to absolute paths relative to the
repository root before use. Inspect the effective contract from any directory:

```bash
python scripts/path_inventory.py --json
```

## Workflow

1. In the ACTS shell, generate or reconstruct inputs with
   `scripts/alignment_odd.py`.
2. In the ML shell, run one of `ml_alignment.py`,
   `ml_alignment_autograd.py`, or `ml_alignment_helix.py`.
3. In the ACTS shell, reconstruct with the aligned geometry.
4. In the ML shell, validate and plot residual, chi2/NDF, and hit diagnostics.

The iterative driver preserves environment isolation:

```bash
scripts/run_ml_iterative.sh --dry-run --iters 1
```

The surface-to-frozen-helix experiment is available through:

```bash
scripts/run_surface_helix_experiment.sh --run-tag trial-001 --dry-run
```

## Data and reproducibility

Bulk artifacts live under `data/` and are never committed. Each retained run
should use `scripts/experiment_manifest.py` to record its effective config,
command, Git commit, input hashes, outputs, metrics, and status.

Historical per-surface methods have known observability limits because the
Kalman fit absorbs part of detector misalignment into track parameters. Read
`docs/history/ml_optimization_log.md` before changing solver objectives.

## Verification

```bash
make test          # Python suite, excluding environment-sensitive ROOT I/O
make test-shell    # path and dry-run contracts
make ci            # default CI gate
make test-root-io  # explicit ROOT/fsspec integration check
make test-cpp      # header-only C++ Jacobian/surface checks
make test-bindings # C++ and pybind11 checks
```
