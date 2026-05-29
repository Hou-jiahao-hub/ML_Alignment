# ACTS Alignment Project Guidelines

## Environment & Compilation
- **Environment**: All compilations MUST use the LCG_109 view. 
- **Activation**: Always assume `source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh` has been run.
- **Variables**: `ACTS_HOME` is `~/alignTest`. `ODD_PATH` is already set; use `os.environ.get('ODD_PATH')` in Python scripts.
- **Build System**: Use modern CMake and C++17/20.

## AlignDerivLib — Standalone Alignment Jacobian Library
- **Location**: `~/alignTest/AlignDerivLib/`, header-only, built to `AlignDerivLib/build_pybind11/`.
- **Purpose**: Computes the alignment residual Jacobian chain (∂residual/∂α) independent of ACTS. This library is the mathematical backbone that provides the training signal for ML alignment.
- **Python bindings**: Use **pybind11 ≥ 2.12**. pybind11 < 2.12 is ABI-incompatible with numpy 2.x and will segfault; ≥ 2.12 is fully compatible. Verified: pybind11 3.0.4 + numpy 2.2.6 in the `pytorch` conda environment (Python 3.10.20).
  - Binding source: `bindings/PyBindings_pybind11.cpp`, output module name: `align_deriv_pb`
  - CMake options: `-DALIGNDERIV_BUILD_PYTHON_PYBIND11=ON -DALIGNDERIV_BUILD_PYTHON=OFF -DALIGNDERIV_BUILD_TESTS=OFF`
  - Also set: `-DEigen3_DIR=<conda_prefix>/share/eigen3/cmake` and `-Dpybind11_DIR=<conda_prefix>/lib/python3.10/site-packages/pybind11/share/cmake/pybind11`
  - Runtime import: `sys.path.insert(0, "<build_pybind11_dir>"); import align_deriv_pb as ad`

## ML-Based Alignment Architecture
This project builds an ML alignment pipeline that is **independent of ACTS at inference time** but uses ACTS-produced track data as training input.

### Data Flow
1. **Extraction**: Run the ACTS CKF/alignment sequence (`alignment_odd.py`) to produce per-hit residuals, track parameters, and GeometryIdentifiers. Export as CSV/JSON.
2. **Feature engineering**: Call `AlignDerivLib` Python bindings to compute Jacobians (∂res/∂α) as ML input features, alongside raw residuals, hit positions, and track angles.
3. **ML model (PyTorch)**: Maps per-surface feature vectors → 6-DOF corrections (δt_x, δt_y, δt_z, δr_x, δr_y, δr_z) in the ACTS local surface frame.
4. **Feedback loop**: Predicted corrections are written as a JSON alignment map and re-ingested via `AlignmentDecorator` in ACTS for validation comparison.

### Model & Training Guidelines
- All ML code uses **PyTorch**. Custom loss functions must be pure `torch.Tensor` ops (no numpy inside `forward()`) to remain GPU-compatible.
- Primary training signal: minimize `Σ_hits (res - J·δα)²` where J is from AlignDerivLib. Always include an L2 regularisation term on δα to prevent runaway corrections.
- Corrections must be expressed in the **ACTS local surface frame**: local0 = u, local1 = v; rotation angles follow the Acts::Surface right-hand convention.
- Batch all hits belonging to one surface together; do not update surfaces one hit at a time.

### Validation Protocol
- Compare ML corrections against the ACTS Kalman-filter alignment baseline (`acts.examples.alignment`) as the reference.
- Primary metrics: mean unbiased residual per surface and per-track χ²/ndf. Both must improve after applying corrections before the result is considered valid.
- Always run `alignment_odd.py` before and after applying ML corrections as a benchmark.

## Code Style & Physics Context
- **C++**: Focus on memory safety and matrix computation efficiency. Default to using the `Eigen` library for any global-local parameter matrix reductions.
- **Python/ML**: When writing alignment calibration models, use PyTorch. Ensure custom loss functions (e.g., for track residuals) are optimized for GPU execution.
- **Physics Algorithms**: Maintain strict mathematical rigor when dealing with Kalman Filter or Broken Lines fit algorithms.
- **Coordinate conventions**: Always follow ACTS local surface coordinates (local0 = u, local1 = v) and the `Acts::Surface::globalToLocal` transform conventions when building feature vectors or interpreting alignment corrections.
- **Standalone vs. ACTS code**: `AlignDerivLib` and any ML training code must compile and run without linking to ACTS. Only the data extraction step and final validation may depend on the ACTS Python bindings.
