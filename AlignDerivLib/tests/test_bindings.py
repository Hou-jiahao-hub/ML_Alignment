"""
test_bindings.py — smoke test for the align_deriv Python extension.

Run from the build directory:
    python3 tests/test_bindings.py
Or with ALIGN_DERIV_BUILD_DIR set to a custom build path:
    ALIGN_DERIV_BUILD_DIR=/path/to/build python3 tests/test_bindings.py
"""

import sys
import os
import math

import numpy as np

# Locate the built .so alongside this script or via env var
build_dir = os.environ.get(
    "ALIGN_DERIV_BUILD_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build"),
)
sys.path.insert(0, build_dir)

import align_deriv as ad  # noqa: E402  (import after sys.path adjustment)

# ── 1. Module and type availability ─────────────────────────────────────────
assert hasattr(ad, "PlaneSurface"),          "PlaneSurface not exported"
assert hasattr(ad, "FreeTrackParameters"),   "FreeTrackParameters not exported"
assert hasattr(ad, "BoundTrackParameters"),  "BoundTrackParameters not exported"
print("[1] Module exports OK")

# ── 2. Construct a PlaneSurface ──────────────────────────────────────────────
center   = np.zeros(3, dtype=np.float64)
rotation = np.eye(3,  dtype=np.float64)
surface  = ad.PlaneSurface(center, rotation)

assert np.allclose(surface.center(),   center)
assert np.allclose(surface.rotation(), rotation)
print("[2] PlaneSurface construction OK")

# ── 3. Construct FreeTrackParameters ────────────────────────────────────────
pos = np.array([1.0, 0.0, 0.0], dtype=np.float64)
direc = np.array([0.0, 0.0, 1.0], dtype=np.float64)
ftp  = ad.FreeTrackParameters(pos, 0.0, direc, 1.0)

assert math.isclose(ftp.x(),    1.0)
assert math.isclose(ftp.dirZ(), 1.0)
assert np.allclose(ftp.direction(), direc)
print("[3] FreeTrackParameters construction OK")

# ── 4. alignmentToResidualDerivative: shape and type ────────────────────────
intersection = np.array([1.0, 0.0, 0.0], dtype=np.float64)
J = surface.alignmentToResidualDerivative(ftp, intersection, 0.0)

assert isinstance(J, np.ndarray),     f"Expected numpy.ndarray, got {type(J)}"
assert J.shape == (2, 6),             f"Expected shape (2, 6), got {J.shape}"
assert J.dtype == np.float64,         f"Expected float64, got {J.dtype}"
print(f"[4] alignmentToResidualDerivative shape={J.shape} dtype={J.dtype}  OK")
print(f"    J =\n{J}")

# ── 5. Analytical value check (matches C++ test 6) ───────────────────────────
#   Expected:  [[-1, 0, 0, 0, 0,  0],
#               [ 0,-1, 0, 0, 0, -1]]
expected = np.array([[-1, 0, 0, 0, 0,  0],
                     [ 0,-1, 0, 0, 0, -1]], dtype=np.float64)
assert np.allclose(J, expected, atol=1e-10), \
    f"Value mismatch:\n{J}\n!= expected:\n{expected}"
print("[5] Analytical value check OK")

# ── 6. alignmentToBoundDerivative: shape ────────────────────────────────────
K = surface.alignmentToBoundDerivative(ftp, intersection, 0.0)
assert K.shape == (6, 6), f"Expected shape (6, 6), got {K.shape}"
assert np.allclose(K[:2, :], J), "Top-2 rows of 6×6 must match 2×6 residual derivative"
print(f"[6] alignmentToBoundDerivative shape={K.shape}  OK")

# ── 7. rotationToLocalAxesDerivative: tuple of three 3×3 arrays ─────────────
dRdAlpha, dRdBeta, dRdGamma = surface.rotationToLocalAxesDerivative()
for name, M in [("dRdAlpha", dRdAlpha), ("dRdBeta", dRdBeta), ("dRdGamma", dRdGamma)]:
    assert M.shape == (3, 3), f"{name}: expected (3,3), got {M.shape}"
print("[7] rotationToLocalAxesDerivative OK")

# ── 8. BoundTrackParameters round-trip ──────────────────────────────────────
btp = ad.BoundTrackParameters(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
assert math.isclose(btp.loc0(),   0.1)
assert math.isclose(btp.loc1(),   0.2)
assert math.isclose(btp.qOverP(), 0.5)
lp = btp.localPosition()
assert lp.shape == (2,) and math.isclose(lp[0], 0.1) and math.isclose(lp[1], 0.2)
print("[8] BoundTrackParameters OK")

print("\ntest_bindings PASSED")
