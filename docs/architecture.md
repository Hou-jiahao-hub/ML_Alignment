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
| `ml_alignment_autograd.py` | Production autograd surface solver, helix initialization, measurement refit and block-LM geometry solver | None |
| `helix_model.py` | Production six-parameter helix, intersections, point seed and independent single-helix measurement Gauss–Newton fit | None |
| `broken_helix_model.py` | Experimental material nodes, Highland prior and broken-helix segments | None |
| `alignment_experiments.py` | Historical Adam refits, joint Adam alignment and learned hit weighting | None |
| `alignment_diagnostics.py` | Residual reporting and optional truth-geometry comparison | None |
| `ml_alignment_helix.py` | Frozen-helix holdout alignment | None |
| `material_artifact.py` | Per-track material-node artifact from ACTS ROOT output | None |
| `validate_helix_model.py` | Holdout validation, acceptance gates, paired single/broken-helix comparison | None |
| `compare_helix_recovery.py` | Paired nominal/misaligned helix seed-to-refit recovery diagnostic | None |
| `reconstruction_acceptance.py` | Reconstructed measurement/smoothed-state chi2 acceptance | None |
| `plot_*.py` | Residual, chi2/NDF, and hit diagnostics | ROOT/uproot |
| `experiment_manifest.py` | Reproducibility metadata and input hashes | None |
| `path_inventory.py` | Effective absolute input/output path contract | None |

The production surface/helix experiment fits and scores on the full nominal
event set in one pass; there is no train/calibration/validation split (removed
2026-09-10 -- the split had stopped doing any real work once the mainline
moved to full-event runs). It scores the supplied misaligned geometry once up
front to give the guard an arm to improve on, then runs a non-target helix
refit, a fixed-track block-LM target-geometry solve, and at most two gain-0.2
surface polish candidates.

The two surface warm starts it used to run first were removed on 2026-09-01.
Block-LM reaches the same geometry from the raw misaligned input once
`LM_MAX_STEP_R` spans the injected rotation: the joint-only run reproduced the
warm-start pipeline's per-sensor geometry to 2.4e-7 mm / 8.8e-9 rad and its
acceptance to 1e-7, in 13 LM iterations instead of 15 and without the two
extra ACTS reconstructions. See
`data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/joint-only-20260901-*`. Candidate
selection and the final acceptance check both read the same full event set,
so every acceptance result -- including the final one -- is `diagnostic_only`;
it is an internal same-sample guard, not evidence of generalization or an
independent validation verdict. Nominal states are used once to create a
versioned frozen acceptance profile whose bootstrap characterizes this full
sample's own event-level sampling uncertainty, while nominal geometry/truth
comparisons remain diagnostic.

The profile's event identity hashes the event IDs actually present in the
nominal `trackstates`; it cannot represent generated events absent from that
file. The pinned profile SHA256 protects the exact profile bytes, while the
experiment manifest's input SHA256 records the ROOT inputs themselves.

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
    helix_recovery/
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

`gauss-newton` is the default. Its production implementation lives in
`helix_model.py` and directly fits one six-parameter helix to local measurements;
it does not import the holdout validator or enter the broken-helix zero-node
branch. A regression test compares its residual, solution and Newton decrement
against that historical zero-node path. `adam` is the pre-Iteration-14
first-order refit, kept in `alignment_experiments.py` so the finding remains
reproducible: on real data every one of its rows stops short of a stationary
point (Newton decrement median ~12 against a 1e-6 threshold), which inflated the
held-out residual by a factor of 2.7-3.9 on the LOC0 groups and was previously
mis-read as unmodelled multiple scattering.

The run records `measurement_fit.stationarity` in the summary and fails when the
non-stationary row fraction exceeds `MAX_NON_STATIONARY_ROW_FRACTION` -- the same
threshold that gates the Gate 4 arms, so a refit that never reached its optimum
cannot be reported as a statement about the model.

`seed_helices` is a separate geometric prefit to the SMT points. It now uses
the same per-track damped Gauss--Newton/LM strategy as the measurement refit:
gauge-fixed parameters, scaled curvature, descent-only step acceptance and
adaptive damping. The point objective remains unweighted Euclidean distance,
so only the solver strategy is shared with the measurement refit. Closest-point
invalid entries remain excluded, accepted candidates may not invalidate an
initially valid point, and a non-stationary row fraction above 0.1% fails closed.
Any non-finite Newton decrement fails immediately regardless of that allowance.

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
module-level constants carrying a `PREREGISTRATION_VERSION`; any change to their
value or meaning must bump it.

### Predictive covariance and damping

`predictive_covariance` returns `J H^-1 J^T` by Cholesky, never an explicit
inverse. The broken arm pads `node_s` to a fixed width, and a padded slot enters
no residual and no prediction, so its row and column of both `H` and `J` are
*exactly* zero. Those directions are structural, not evidence of a sick fit, and
they are filled to the active mean diagonal rather than damped: `J` annihilates
them analytically, so the covariance is unchanged, while the factorisation stops
being asked to invert a ~1e13 condition number. A prediction that *does* depend
on an inactive direction is refused, because the variance reported for it would
be an artefact of the regulariser. Reported conditioning describes the solvable
subspace only.

With the structural nulls handled, the exact posterior is solvable and
`PRODUCTION_DAMPING` is `0.0` (pre-registration v2). This matters for the paired
comparison: a non-zero *relative* damping can never be equivalent across the two
arms, because their spectra genuinely differ -- the kinks are soft directions the
single helix does not have. Measured on real rows at the v1 value of `1e-10`, the
broken arm's reported variance moved 18.5% (p99) against 0.045% for the single
arm. `DAMPING_SCAN` is still recorded in full as a diagnostic, but only
`DAMPING_STABILITY_SCAN` -- dampings whose bias is numerically negligible --
decides the sensitivity gate; a verdict that moves only under a large,
deliberate bias is not a flip.

A row whose *active* subspace is rank deficient is a third case, and neither the
filler nor the damping may rescue it: there is no posterior width to report, so
any number would be the regulariser's. Those rows are returned as NaN and the
callers' finite checks drop them from **both** arms, keeping the paired row set
shared; the count is gated by `MAX_RANK_DEFICIENT_ROW_FRACTION` so a silently
shrinking row set cannot pass as a result. On the full 20000-event set exactly
one row of 70352 qualifies -- a single-arm fit with 4 hits against 5 gauge-fixed
parameters. Passing a single such matrix, rather than a batch, still raises.

## Paired helix-recovery diagnostic

`compare_helix_recovery.py` tests whether the shared standalone helix machinery
actually removes a branch-dependent geometry effect. It loads nominal and
misaligned `trackstates` together with uproot and assigns every measured hit its
ordinal among measured states in that track. Predicted, filtered, and hole-state
positions therefore cannot shift an otherwise identical measurement identity.
Before deterministic event or track limits, the loader requires identical
event/track/state identities, measurements, measurement uncertainties, target
membership, and LOC availability. It never intersects mismatched files. SMT
local/global positions, directions, and sensor indices remain branch-local; both
fitted branches are evaluated on nominal surfaces for the final paired comparison.

Each retained target hit becomes one frozen probe row. The protected exposure
fits only non-target measurements, while the exposed exposure additionally fits
that row's probe. Both nominal and misaligned branches run the same LM seed and
Gauss--Newton measurement refit, forming up to four independently reported
arms; a single-exposure request does not execute the other exposure.
Event-cluster bootstraps and numerical-stationarity checks produce separate
protected/exposed verdicts; a scientific `failed` or `inconclusive` verdict is a
completed experiment rather than a runtime failure.

Every run starts `manifest.json` before input hashing or ROOT loading and writes
`paired_track_keys.csv`, flattened `per_track_stage_metrics.csv`, a strict-JSON
`helix_recovery_summary.json`, and three diagnostic PNGs per requested exposure.
The manifest hashes both ROOT and both geometry inputs and is finalized on both
normal, exceptional, and interrupted exits while retaining every metric already
computed. Runnable smoke and partitioned examples are in
[`docs/helix-recovery.md`](helix-recovery.md).

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
