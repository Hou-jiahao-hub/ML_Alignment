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
│   ├── alignment_odd.py          # ACTS CKF + alignment sequence
│   ├── ml_alignment.py           # PyTorch ML training & inference
│   ├── plot_residuals.py         # Residual diagnostic plots
│   ├── generate_odd_transform.py # ODD surface transform extractor
│   └── AlignmentDerivBridge.hpp  # C++ bridge header
├── docs/
│   ├── alignment_math_theory.md  # Mathematical derivation
│   ├── ml_alignment_revise.md    # Development notes
│   └── fix.md                    # Known issues & fixes
├── setup_acts.sh           # Environment setup script
└── CLAUDE.md               # Project guidelines for Claude Code
```

## Setup

```bash
# 1. Activate LCG_109 environment
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh

# 2. (Or) activate conda PyTorch environment
source setup_acts.sh

# 3. Build AlignDerivLib Python bindings
cd AlignDerivLib
cmake -B build_pybind11 \
  -DALIGNDERIV_BUILD_PYTHON_PYBIND11=ON \
  -DALIGNDERIV_BUILD_PYTHON=OFF \
  -DALIGNDERIV_BUILD_TESTS=OFF
cmake --build build_pybind11
```

## Running

```bash
# Run ACTS alignment sequence
python scripts/alignment_odd.py

# Train ML alignment model
python scripts/ml_alignment.py

# Plot residuals
python scripts/plot_residuals.py
```

## Dependencies

- ACTS framework (for data extraction step only)
- Eigen3
- pybind11 ≥ 2.12
- PyTorch
- numpy ≥ 2.0
