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
| `alignment_odd.py` | Simulation, misalignment injection, reconstruction | Required |
| `generate_odd_transform.py` | Export nominal ODD surface geometry | Required |
| `alignment_artifact_layout.py` | Compatibility API for paths and target-layer naming | None |
| `project_config.py` | Typed TOML configuration and validation | None |
| `ml_alignment.py` | Analytic-Jacobian surface solver | AlignDerivLib only |
| `ml_alignment_autograd.py` | Autograd surface and joint track/geometry solvers | None |
| `ml_alignment_helix.py` | Frozen-helix holdout alignment | None |
| `validate_helix_model.py` | Holdout validation and acceptance gates | None |
| `plot_*.py` | Residual, chi2/NDF, and hit diagnostics | ROOT/uproot |
| `experiment_manifest.py` | Reproducibility metadata and input hashes | None |
| `path_inventory.py` | Effective absolute input/output path contract | None |

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
    ablation/
  archive/                  historical logs, figures, and retired artifacts
```

The directory is local-only. Git stores code, small test fixtures, experiment
manifests, and metric summaries—not bulk ROOT files or generated plots.

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
