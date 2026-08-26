# ML_Alignment architecture

## System boundary

`ML_Alignment` owns alignment algorithms, experiment orchestration, diagnostics,
and their contracts. `/home/houjiahao/alignTest` is an external ACTS checkout;
it is not a second source tree for this project.

```text
ACTS adapter                         ACTS-independent core
simulation -> ROOT/geometry JSON -> loaders -> solver -> aligned geometry
      ^                                                   |
      +----------- ACTS reconstruction validation <-------+
```

## Components

| Component | Responsibility | ACTS dependency |
|---|---|---|
| `alignment_odd.py` | Simulation, misalignment injection, reconstruction, material export | Required |
| `generate_odd_transform.py` | Export nominal ODD surface geometry | Required |
| `alignment_artifact_layout.py` | Compatibility API for paths and target-layer naming | None |
| `project_config.py` | Typed TOML configuration and validation | None |
| `ml_alignment.py` | Analytic-Jacobian surface solver | AlignDerivLib only |
| `ml_alignment_autograd.py` | Autograd surface and joint track/geometry solvers | None |
| `helix_model.py` | Shared six-parameter helix, Newton intersections, curvature scaling, broken-helix segments and the Highland kink prior | None |
| `ml_alignment_helix.py` | Frozen-helix holdout alignment | None |
| `material_artifact.py` | Per-track material-node artifact from ACTS ROOT output | None |
| `validate_helix_model.py` | Holdout validation, acceptance gates, paired single/broken-helix comparison | None |
| `reconstruction_acceptance.py` | Reconstructed measurement/smoothed-state chi2 acceptance | None |
| `plot_*.py` | Residual, chi2/NDF, and hit diagnostics | ROOT/uproot |
| `experiment_manifest.py` | Reproducibility metadata and input hashes | None |
| `path_inventory.py` | Effective absolute input/output path contract | None |

The production surface/helix experiment uses a deterministic event split
(80% train, 10% calibration, 10% blind validation). It runs two surface warm
starts, a non-target helix refit, a fixed-track block-LM target-geometry solve,
and at most two gain-0.2 surface polish candidates. Candidate selection reads
only the calibration partition; the selected geometry is evaluated once on the
validation partition. Nominal states are used once to create a versioned frozen
acceptance profile, while nominal geometry/truth comparisons remain diagnostic.

## Configuration flow

Configuration is merged in this order, with the last source winning:

1. `configs/default.toml`
2. ignored `configs/local.toml`, when present
3. `ML_ALIGNMENT_CONFIG`
4. explicit `--config PATH`

Relative paths resolve from the repository root. A target layer such as `17:8`
derives `v17l8` for filenames and `Vol17_Lay8` for directories. Consumers must
not maintain separate copies of those values.

Before I/O, both default and explicit CLI paths are normalized to absolute
paths. Runtime dependency roots for ACTS, ODD, AlignDerivLib, and Conda use the
same configuration boundary. `scripts/path_inventory.py --json` is the
machine-readable source for the effective input/output contract.

## Data layout

```text
data/
  geometry/                 nominal and working geometry JSON
  simulation/               generated events and measurements
  alignment/                misaligned and solved geometry
  reconstruction/          nominal, misaligned, and aligned ROOT outputs
  experiments/
    iterative_ml/
    surface_helix/
    helix_validation/
    material/
    ablation/
  archive/                  historical logs, figures, and retired artifacts
```

The directory is local-only. Git stores code, small test fixtures, experiment
manifests, and metric summaries—not bulk ROOT files or generated plots.

## Material artifact contract

`alignment_odd.py material` runs the standard reconstruction chain and adds a
material-recording propagation over the same selected tracks, writing
`material_kf_*.root` and `propsummary_kf_*.root` next to the usual states and
summary. `material_artifact.py` joins the three tables and emits a schema-1.1
NPZ with a manifest.

The join needs all three files: `RootMaterialTrackWriter` drops the track index
and iterates an unordered map, and the direction it stores carries no momentum,
so `propsummary_kf` is the bridge that supplies `(event_nr, track_nr)` next to
the same start bound parameters.

The artifact's `path_length_mm` is a cumulative transverse chord length used
only for ordering and completeness checks. It is not a track-level invariant:
a helix `s` origin belongs to whichever fit produced it. Consumers must
recompute node and hit arc lengths from `position_mm` against their own frozen
reference before clipping windows or assigning segments.

## Measurement refit solver

`validate_helix_model.py --fit-solver {gauss-newton,adam}` chooses how the
non-holdout measurement refit is minimised. Both solvers minimise the *same*
half-chi2 over the same component mask; only the descent differs.

`gauss-newton` is the default and runs through `fit_broken_helix` with zero
material nodes, i.e. the identical solver the Gate 4 single arm uses. `adam` is
the pre-Iteration-14 first-order refit, kept only so that finding stays
reproducible: on real data every one of its rows stops short of a stationary
point (Newton decrement median ~12 against a 1e-6 threshold), which inflated the
held-out residual by a factor of 2.7-3.9 on the LOC0 groups and was previously
mis-read as unmodelled multiple scattering.

The run records `measurement_fit.stationarity` in the summary and fails when the
non-stationary row fraction exceeds `MAX_NON_STATIONARY_ROW_FRACTION` -- the same
threshold that gates the Gate 4 arms, so a refit that never reached its optimum
cannot be reported as a statement about the model.

`seed_helices` is a separate geometric prefit to the SMT points and still uses
Adam; it only supplies the Gauss-Newton starting point.

## Paired holdout contract (Gate 4)

`validate_helix_model.py --material-artifact ...` evaluates two track models on
**one** row set built by a single `build_track_batches` call: `single` (no
material nodes) and `broken` (kinks at every node inside the row's fit window).
Both arms run the same damped Gauss-Newton MAP solver -- the single arm is the
broken arm with `N == 0` nodes -- so no optimiser difference can be mistaken for
a model difference.

Every row recomputes `node_s` and `hit_s` from its own frozen reference, then
clips nodes to `[min, max]` of that row's hit arc lengths and sorts them; the
segment assignment is detached integer data. The row identity list is written to
`holdout_keys.csv` and its SHA256 recorded in the summary, and the two arms'
lists are compared byte for byte before any metric is formed.

Acceptance is per `(layer, mode, component)`: a group needs
`N >= MIN_GROUP_COMPONENTS` to be gated, and any *required* group below that
makes the whole verdict `inconclusive` rather than `passed`. Improvement means
the paired-bootstrap 95% CI bound clears the threshold, never the point
estimate. All thresholds, the bootstrap seed and the production damping are
module-level constants pre-registered at Gate 1.

## Input/output contracts

- Geometry JSON contains surface IDs, translations, and rotation matrices.
- Trackstate and tracksummary ROOT branches are validated by loader tests.
- Alignment output is a geometry JSON with corrections applied in the local
  six-degree-of-freedom convention.
- Every retained experiment writes schema-versioned `manifest.json` metadata.

## Verification tiers

The default suite excludes the environment-sensitive ROOT I/O test that can
stall in incompatible fsspec/uproot combinations. Shell dry-run tests verify
cross-environment command construction without launching ACTS. C++ bindings and
ACTS end-to-end validation remain explicit tiers because they need system
dependencies unavailable on generic CI runners.
