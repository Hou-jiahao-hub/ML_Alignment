# ML_Alignment

Machine-learning based detector alignment for the ACTS tracking framework, using the Open Data Detector (ODD).

## Overview

This project builds an ML alignment pipeline that is independent of ACTS at inference time but uses ACTS-produced track data as training input.

### Data Flow

1. **Extraction** — Run `scripts/alignment_odd.py` (ACTS CKF/alignment) to produce per-hit residuals, track parameters, and GeometryIdentifiers.
2. **Feature engineering** — Call `AlignDerivLib` Python bindings to compute Jacobians (∂res/∂α) as ML input features.
3. **ML model (PyTorch)** — Maps per-surface feature vectors → 6-DOF corrections (δt_x, δt_y, δt_z, δr_x, δr_y, δr_z).
4. **Feedback loop** — Predicted corrections are written as a JSON alignment map and re-ingested via `AlignmentDecorator` in ACTS for validation.

## Repository Structure

```
ML_Alignment/
├── AlignDerivLib/          # Header-only C++ library: alignment residual Jacobians
│   ├── include/AlignDeriv/ # Public headers
│   ├── bindings/           # Python bindings (pybind11)
│   ├── tests/              # Unit tests
│   └── CMakeLists.txt
├── scripts/
│   ├── alignment_artifact_layout.py  # Single source of truth for target layers,
│   │                                 # file-name suffixes and output directories
│   ├── alignment_odd.py              # ACTS simulation / misalignment / reconstruction CLI
│   ├── ml_alignment.py               # Solver A: analytic Jacobians from AlignDerivLib
│   ├── ml_alignment_autograd.py      # Solver B: pure PyTorch autograd, no ACTS dependency
│   ├── run_ml_iterative.sh           # Multi-round driver: solve → reconstruct → plot
│   ├── plot_residuals.py             # Residual comparison plots
│   ├── plot_chi2ndf.py               # Track chi2/ndf comparison plots
│   ├── plot_hit_number.py            # Hit multiplicity comparison plots
│   ├── generate_odd_transform.py     # ODD surface transform extractor
│   └── AlignmentDerivBridge.hpp      # C++ bridge header
├── docs/
│   ├── alignment_math_theory.md  # Mathematical derivation
│   ├── ml_alignment_revise.md    # Development notes
│   └── fix.md                    # Known issues & fixes
├── setup_acts.sh           # Environment setup script
└── CLAUDE.md               # Project guidelines for Claude Code
```

## Environments

Two separate environments are required, and they **must never be active at the same time**:

| Stage | Environment | Activation |
|---|---|---|
| ACTS simulation / alignment / reconstruction | LCG_109 | `source setup_acts.sh` |
| ML training, residual plotting, tests | conda `pytorch` (with ROOT 6.40) | `conda activate pytorch` |

> Sourcing the LCG view sets `PYTHONHOME`, which breaks the conda interpreter with
> `ModuleNotFoundError: No module named 'encodings'`. Run the two stages in separate
> shells. `scripts/run_ml_iterative.sh` isolates them internally via subshells
> (see its `run_conda` / `run_acts` helpers).

## Setup

```bash
# 1. Activate the LCG_109 environment
source setup_acts.sh

# 2. Build the AlignDerivLib Python bindings (needed by ml_alignment.py only)
cd AlignDerivLib
cmake -B build_pybind11 \
  -DALIGNDERIV_BUILD_PYTHON_PYBIND11=ON \
  -DALIGNDERIV_BUILD_PYTHON=OFF \
  -DALIGNDERIV_BUILD_TESTS=OFF
cmake --build build_pybind11
```

### Generating `odd_transforms.json`

The nominal (un-misaligned) geometry of all 18824 sensitive surfaces is **not tracked
in this repository** (~19 MB). Generate it once inside the LCG environment:

```bash
python scripts/generate_odd_transform.py
```

It requires `acts.examples.odd.getOpenDataDetector` and the ODD material maps, so it
cannot be produced on a machine without ACTS. It is only needed for the optional
`--nominal` truth-comparison diagnostics.

## Configuring the target layers

Target layers, file-name suffixes and output sub-directories are defined **once** in
`scripts/alignment_artifact_layout.py`; every other script imports them from there:

```python
TARGET_LAYERS = ((17, 8), (24, 4))
TARGET_LAYER_SUFFIX = "v17l8_v24l4"
TARGET_LAYER_DIR = "Vol17_Lay8__Vol24_Lay4"
```

Changing these three constants is sufficient to switch target layers — no other file
needs editing. Shell scripts obtain the same values via
`eval "$(python alignment_artifact_layout.py --shell)"`.

## Running

### 1. Full ACTS workflow (LCG environment)

```bash
python scripts/alignment_odd.py full --num-events 100000
```

Runs simulation → misalignment injection → reconstruction (nominal / misaligned / aligned).

### 2. ML alignment (conda environment)

Two interchangeable solvers optimise the same per-surface 6-DOF correction
`[δt_x, δt_y, δt_z, δr_x, δr_y, δr_z]` in the ACTS local surface frame
(`local0 = u`, `local1 = v`), but differ in how the residual derivatives are obtained:

| | `ml_alignment.py` | `ml_alignment_autograd.py` |
|---|---|---|
| Jacobian source | analytic, from `AlignDerivLib` | pure PyTorch autograd |
| External dependency | `AlignDerivLib/build_pybind11` on `sys.path` | none |
| Residual model | first-order linearisation, periodically re-linearised | full non-linear surface transform + track intersection each step |

```bash
python scripts/ml_alignment.py \
  --input-geom  alignment_output/<DIR>/misalignment_applied_<SUFFIX>.json \
  --states      reconstruction_output/misaligned/<DIR>/trackstates_kf_<SUFFIX>.root \
  --summary     reconstruction_output/misaligned/<DIR>/tracksummary_kf_<SUFFIX>.root \
  --output-geom alignment_output/<DIR>/aligned_transforms_ml_<SUFFIX>.json \
  --target-layers 17:8,24:4 --active-dof 0,1,3,4,5 --epochs 300 --lambda-l2 1e-5
```

The injected misalignment consists of `tx`/`ty` translations plus `rx`/`ry` tilts, and
the tilts dominate the residuals — `--active-dof` **must** include indices 3 and 4.

### 3. Validation (LCG environment)

```bash
python scripts/alignment_odd.py reconstruction --mode aligned \
  --aligned-file <DIR>/aligned_transforms_ml_<SUFFIX>.json \
  --output-suffix ml --num-events 100000
```

### 4. Plots (conda environment)

```bash
python scripts/plot_residuals.py --range -1.0 1.0 --volume_id 17 --layer_id 8 \
  --recon-dir reconstruction_output --output-dir <OUT_DIR>
```

Volumes 23/24/25 (ShortStrip) have an intrinsic Y resolution of ~1.2 mm; use `--range -6.0 6.0`
for their Y residuals and `-1.0 1.0` for X.

### 5. Iterative refinement

A single ML pass is usually insufficient. `run_ml_iterative.sh` loops
solve → reconstruct → plot, feeding each round's geometry into the next:

```bash
./scripts/run_ml_iterative.sh --iters 3 --events 100000
```

The solver choice is hard-coded at the top of the script (`ALIGNMENT_SOLVER`:
`cpp` → `ml_alignment.py`, `autograd` → `ml_alignment_autograd.py`), not a CLI flag.

## Known limitation

Both solvers perform an **independent per-surface least-squares fit**. The Kalman filter
absorbs part of the misalignment into the track parameters themselves, so "minimising
single-surface residuals" and "converging to the true geometry" are not the same
objective — the sign of the predicted correction relative to truth can flip between
iterations, and repeated per-surface solving does not converge to the truth.

A simultaneous track + alignment χ² minimisation (Millepede-style) resolves this and
reaches `tx` correlation above 0.99. Consult `docs/` before extending either solver.

## Dependencies

- ACTS framework — data extraction and final validation only; the ML solvers do not link against it
- Eigen3
- pybind11 ≥ 2.12 — versions below 2.12 are ABI-incompatible with numpy 2.x and will segfault
- PyTorch
- numpy ≥ 2.0
- ROOT ≥ 6.40 with PyROOT — required by the plotting scripts
  (`conda install -c conda-forge "root=6.40.00=py310he617d7b_0"`)
