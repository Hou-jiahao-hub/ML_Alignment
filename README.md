# ML_Alignment

Joint track-geometry alignment of tracking-detector surfaces with a
Schur-reduced Levenberg-Marquardt solver (`schur-lm`). The algorithm is the
installable, ACTS-independent package `mlalign`; ACTS remains the simulation
and reconstruction runtime of the workflow around it.

## Repository map

```text
ML_Alignment/
├── src/mlalign/       the Schur-LM alignment package (pip-installable, no ACTS)
├── scripts/           workflow layer: ACTS adapter, acceptance, plots, orchestration
├── configs/           workflow defaults (TOML) and local override template
├── data/              ignored local geometry, ROOT, experiments, and archives
├── docs/              architecture, mathematics, and experiment history
├── tests/             Python and shell regression tests
├── AGENTS.md          development contract
├── Makefile           verification entrypoints
└── pyproject.toml     package metadata, console script and pytest configuration
```

Component and data flow: [docs/architecture.md](docs/architecture.md).
Input/output contract: [docs/io-paths.md](docs/io-paths.md).

## The `mlalign` package

```bash
python -m pip install -e '.[test]'      # inside the conda pytorch environment
```

Command line (`mlalign-align`, or `python -m mlalign`):

```bash
mlalign-align \
    --input-geom misalignment_applied_v17l8_v24l4.json \
    --states trackstates_kf_v17l8_v24l4.root \
    --nominal-full-geom odd_transforms.json \
    --output-geom aligned.json --output-summary summary.json
```

Python:

```python
import mlalign

config = mlalign.AlignmentConfig(
    input_geom="misalignment_applied.json",
    states_root="trackstates_kf.root",
    output_geom="aligned.json",
    nominal_full_geom="odd_transforms.json",
)
result = mlalign.run_alignment(config)
mlalign.write_alignment_summary(result, "summary.json")
```

Pipeline: load the full nominal geometry with the input target surfaces
overlaid and the ACTS `trackstates` -> select tracks (>= 8 hits, >= 300 mm
span) -> SMT point seed -> non-target measurement Gauss-Newton refit ->
joint Schur-LM over target geometry and every track -> export the corrected
input surfaces. CPU, float64. Relative paths resolve against the working
directory.

Every default is a constant in `src/mlalign/config.py` and equals the
validated mainline run: target layers `17:8,24:4`, active DOF `tx,ty,rz`
(`0,1,5`), 2000 SMT seed iterations, box bounds 1.5 mm / 0.05 rad, and the
pre-registered Schur-LM budget of 60 outer iterations.

## Workflow (scripts/)

Two environments are required and must not be combined in one shell.

- ACTS: `source setup_acts.sh` in a fresh shell (needs the `sft.cern.ch` CVMFS
  mount, see the [local CVMFS workstation guide](docs/cvmfs-local-workstation-guide.md)).
  LCG modifies `PYTHONHOME`; leave that shell before activating conda.
- ML, acceptance and plots: the conda `pytorch` environment with `mlalign`
  installed. PyROOT is a system/conda dependency.

1. ACTS shell: `python scripts/alignment_odd.py full` simulates, injects the
   misalignment, runs the ACTS-Alignment baseline and reconstructs the nominal
   and misaligned samples.
2. Either shell: `scripts/run_schur_experiment.sh --run-tag NAME` runs the
   frozen acceptance profile, Schur-LM, ACTS re-reconstruction, the acceptance
   guard, plots and geometry comparison, and writes `manifest.json`.

```bash
scripts/run_schur_experiment.sh --run-tag trial-001 --dry-run
```

The workflow reads `configs/default.toml` (data root, ACTS checkout, conda
hook, target layers, active DOF); override it with an ignored
`configs/local.toml`, `ML_ALIGNMENT_CONFIG` or `--config PATH`. Its
`[alignment]` values must agree with the package defaults; `tests/test_config.py`
checks that they do.

## Data and reproducibility

Bulk artifacts live under `data/` and are never committed. Each retained run
records its configuration, command, Git commit, input hashes, outputs, metrics
and status in `manifest.json` (`scripts/experiment_manifest.py`).

Read `docs/history/ml_optimization_log.md` before changing solver objectives.
Code retired on 2026-10-19 (surface and block-LM solvers, AlignDerivLib, Gate 4
holdout validation, material and pT=100 studies, gate analyses) is reproducible
from the tag `pre-schur-package`.

## Verification

```bash
make test          # Python suite, excluding environment-sensitive ROOT I/O
make test-shell    # orchestration dry-run and path contracts
make ci            # default CI gate
make test-root-io  # explicit ROOT/fsspec integration check
```
