# ML_Alignment architecture

## System boundary

`ML_Alignment` owns the Schur-LM alignment algorithm, the workflow around it,
diagnostics, and their contracts. `/home/houjiahao/alignTest` is an external
ACTS checkout; it is not a second source tree for this project.

```text
ACTS adapter (scripts/)               mlalign package (src/, ACTS-independent)
simulation -> ROOT/geometry JSON -> loaders -> Schur-LM -> aligned geometry
      ^                                                        |
      +----------- ACTS reconstruction + acceptance <----------+
```

The package never imports from `scripts/`: it reads no TOML and knows no
repository path, so it can be installed and used on another detector setup.
Its defaults are module constants in `src/mlalign/config.py`.

## Components

### Package `src/mlalign/`

| Module | Responsibility |
|---|---|
| `config.py` | `DEFAULT_*` constants, `DTYPE`, `AlignmentConfig`, `add_*_arguments`, `alignment_config_from_args` |
| `cli.py`, `__main__.py` | `mlalign-align` / `python -m mlalign` |
| `pipeline.py` | `run_alignment(config)`: load, select, seed, refit, Schur-LM, truth diagnostic, export, per-layer report |
| `schur_lm.py` | Joint track-geometry Schur-reduced LM and its pre-registered convergence contract |
| `helix_init.py` | Non-target SMT point seed and measurement Gauss–Newton refit with its stationarity gate |
| `helix/` | Six-parameter helix (`trajectory`, `gauge`), row-wise LM (`gauss_newton`), point seed (`point_fit`), single-helix measurement fit (`measurement_fit`) |
| `geometry_io.py` | Full nominal geometry with the input target surfaces overlaid; export of the corrected input surfaces |
| `root_inputs.py` | ACTS `trackstates` ROOT -> per-hit tensors |
| `track_batches.py` | Per-track padded tensors and the non-target fit mask |
| `joint_residuals.py` | Helix-to-surface residuals, per-layer chi2, component-mask merging |
| `box_constraints.py`, `conventions.py`, `rotations.py` | Correction bounds and free-DOF active set, DOF/target-layer parsing, Euler convention |
| `diagnostics.py`, `reporting.py` | Residual tables and truth-geometry comparison; compact JSON summary |
| `timing.py` | Per-run wall and process CPU timing, stage status, and thread configuration |

### Workflow `scripts/`

| Component | Responsibility | ACTS dependency |
|---|---|---|
| `alignment_odd.py` | Simulation, misalignment injection, ACTS-Alignment baseline, reconstruction (`full`, `simulation`, `alignment`, `reconstruction`) | Required |
| `generate_odd_transform.py` | Export nominal ODD surface geometry | Required |
| `run_schur_experiment.sh` | End-to-end orchestration across the ACTS and conda environments | Calls the adapter |
| `calibrate_acceptance.py`, `reconstruction_acceptance.py`, `event_identity.py` | Frozen bootstrap acceptance profile, reconstructed chi2 acceptance, event identity | None |
| `compare_geometries.py`, `plot_residuals.py`, `plot_chi2ndf.py` | Geometry truth comparison (including ACTS vs Schur), residual and chi2/NDF plots | None (ROOT/uproot) |
| `project_config.py`, `alignment_artifact_layout.py` | TOML configuration, `data/` layout, shell exports | None |
| `experiment_manifest.py`, `write_experiment_summary.py` | Reproducibility manifest, source snapshot, experiment summary | None |

## Pipeline

```text
run_alignment(config)
  load_joint_geometry          nominal full geometry + input target surfaces
  load_variable_information    trackstates -> AlignmentInputs (tracksummary is not read)
  build_track_batches          >= 8 non-target hits, >= 300 mm span, a target hit
  seed_helices                 non-target SMT points, damped GN, <= 2000 iterations
  fit_helices_to_measurements  non-target measurements, damped GN, stationarity gate
  run_schur_lm_alignment       target geometry and every track jointly, <= 60 outer iterations
  validate_against_truth       diagnostic only
  export_joint_geometry        corrected input surfaces -> aligned JSON
```

`run_alignment()` returns schema-versioned `timings` with wall seconds,
process CPU seconds, stage status and thread configuration. The CLI writes
these only into its existing `--output-summary` JSON, including partial timings
on Python exceptions or keyboard interruption. `pipeline_total` covers the
alignment pipeline, excluding imports, summary serialization and external ACTS
processes. Process CPU time includes all threads in the current process.

Geometry corrections compose locally: `center = center0 + R0·delta_t`,
`R = R0·R(delta_rot)`; mm and rad. Everything runs on the CPU in float64:
the refit stationarity threshold (1e-6) and the Schur-LM decrements are below
float32 resolution.

### Schur-LM

Residuals are whitened; geometry increments are dimensionless,
`a = GEOMETRY_SCALES * u` on the active DOF of target sensors, and track
increments live in the five-parameter gauge frame. Each track's joint normal
equations are reduced onto the geometry (Schur complement), the reduced system
is assembled as CSC and solved with SuperLU, and tracks are back-substituted.
The contract -- Marquardt damping with gain-ratio updates, the box active set
with a uniformly scaled feasible step, the global decrement and rank test per
connected component of the sensor coupling graph, and the terminal states --
is pre-registered (`ml_optimization_log.md` Iterations 34-38) and lives as
`SCHUR_LM_*` constants in `schur_lm.py`. `--lm-max-iters` sets the outer
budget; its default equals `SCHUR_LM_MAX_ITERS`.

### Measurement refit and seed

The non-target measurement refit is minimised by damped Gauss–Newton on one
six-parameter helix per track. It replaced a first-order Adam refit whose rows
stopped short of a stationary point (Newton decrement median ~12 against a
1e-6 threshold). The run records the refit stationarity in the summary and
fails when more than `JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION` of the rows
miss the threshold.

`seed_helices` is a separate geometric prefit to the SMT points with the same
damped Gauss–Newton strategy; its objective stays unweighted Euclidean
distance. Closest-point invalid entries are excluded, accepted candidates may
not invalidate an initially valid point, and a non-stationary row fraction
above 0.1% fails closed. With 200 iterations the seed was truncated below that
gate on real data; it converges at ~367 and breaks out, hence the 2000 cap.

### Module layering

`mlalign` imports in one direction only:

```text
config
  -> conventions, rotations, helix/{trajectory,gauge,gauss_newton}
  -> helix/{measurement_fit,point_fit}, geometry_io, root_inputs, track_batches, box_constraints
  -> joint_residuals, helix_init
  -> schur_lm (imports helix_init and joint_residuals lazily)
  -> diagnostics, reporting, pipeline -> cli
```

Each module has one import name. `mlalign/__init__.py` re-exports the public
API (`AlignmentConfig`, `run_alignment`, `write_alignment_summary`) and imports
uproot/awkward before torch, which keeps torch from loading the system
libstdc++. A test that monkeypatches a dependency must patch the module that
calls it.

## Orchestration

`run_schur_experiment.sh` fits and scores on the full nominal event set in one
pass; there is no train/calibration/validation split. It freezes a versioned
acceptance profile from the nominal states, scores the supplied misaligned
geometry once to give the guard an arm to improve on, runs `mlalign-align`,
re-reconstructs with ACTS, scores again, and keeps the Schur-LM geometry only
if its acceptance score improves and the solver produced a non-zero update.
Selection and final acceptance read the same events, so every acceptance
result is `diagnostic_only`: an internal same-sample guard, not evidence of
generalization.

The profile's event identity hashes the event IDs actually present in the
nominal `trackstates`; it cannot represent generated events absent from that
file. The pinned profile SHA256 protects the exact profile bytes, while the
experiment manifest's input SHA256 records the ROOT inputs themselves.

## Configuration flow

The package takes its configuration from `AlignmentConfig` or the CLI; relative
paths resolve against the working directory.

The workflow merges, with the last source winning:

1. `configs/default.toml`
2. ignored `configs/local.toml`, when present
3. `ML_ALIGNMENT_CONFIG`
4. explicit `--config PATH`

Workflow paths resolve from the repository root. A target layer such as `17:8`
derives `v17l8` for filenames and `Vol17_Lay8` for directories; consumers must
not maintain separate copies of those values. `python
scripts/alignment_artifact_layout.py --shell` prints the effective paths. The
workflow's `[alignment]` target layers and active DOF must equal the package
defaults (`tests/test_config.py`).

## Data layout

```text
data/
  geometry/                 nominal and working geometry JSON
  simulation/               generated events and measurements
  alignment/                misaligned and solved geometry
  reconstruction/           nominal, misaligned, and aligned ROOT outputs
  experiments/
    schur_lm/               run_schur_experiment.sh runs (<target>/<run-tag>/)
    ...                     earlier namespaces (surface_helix, pt100_alignment, ...)
  archive/                  historical logs, figures, and retired artifacts
```

The directory is local-only. Git stores code, small test fixtures, experiment
manifests, and metric summaries -- not bulk ROOT files or generated plots.

## Input/output contracts

- Geometry JSON is a list of surfaces with `ID` (`vol=|lay=|sen=`),
  `Translation` and `RotationMatrix`, optionally `FullMatrix`.
- Trackstate ROOT branches are validated by loader tests; measurement-error
  arrays are mapped onto the measured states.
- Alignment output is a geometry JSON of the input target surfaces with the
  local six-DOF corrections applied.
- Every retained experiment writes a schema-versioned `manifest.json`.

## Retired components

The surface (Adam) and block-LM solvers, the AlignDerivLib analytic-Jacobian
solver, frozen-helix holdout alignment, Gate 4 holdout validation, broken-helix
and material studies, helix-recovery and helix-truth diagnostics, and the gate
analysis scripts were removed on 2026-10-19. Their contracts are described in
`docs/history/` and the optimization log; tag `pre-schur-package` reproduces
them.

## Verification tiers

The default suite excludes the environment-sensitive ROOT I/O test that can
stall in incompatible fsspec/uproot combinations. Shell dry-run tests verify
cross-environment command construction without launching ACTS. ACTS end-to-end
validation remains an explicit tier because it needs the CVMFS/LCG runtime.
