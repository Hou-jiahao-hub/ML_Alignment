#pragma once

#include <Eigen/Core>

// ============================================================================
//  AlignDeriv/Types.hpp
//
//  Canonical type aliases for the alignment residual Jacobian engine.
//
//  All dimensions follow the ACTS / alignment-math-theory convention:
//    α  ∈ ℝ⁶  — alignment parameters  (tx, ty, tz, αx, αy, αz)
//    ξ  ∈ ℝ⁶  — bound track params    (loc0, loc1, φ, θ, q/p, t)
//    τ  ∈ ℝ⁸  — free  track params    (x, y, z, t, dx, dy, dz, q/p)
//
//  Final result:  dResidual/dα  ∈ ℝ^{2×6}
// ============================================================================

namespace AlignDeriv {

// ── Scalar ──────────────────────────────────────────────────────────────────
using Scalar = double;

// ── Fixed-size column vectors ────────────────────────────────────────────────
/// 2D bound local measurement (loc0, loc1) / residual
using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

/// 3D global position / momentum direction / surface centre
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

/// 6D bound track params ξ  OR  alignment params α
using Vector6 = Eigen::Matrix<Scalar, 6, 1>;

/// 8D free track params τ = (x, y, z, t, dx, dy, dz, q/p)
using Vector8 = Eigen::Matrix<Scalar, 8, 1>;

// ── Explicit matrix-naming aliases (match paper notation) ───────────────────

/// pathDerivative: ∂τ/∂s  — 8×1 column vector
using Matrix8x1 = Eigen::Matrix<Scalar, 8, 1>;

/// alignToPath:    ∂s/∂α  — 1×6 row vector
using Matrix1x6 = Eigen::Matrix<Scalar, 1, 6>;

/// localCartesianToBoundLocalDerivative: ∂(loc0,loc1)/∂(lx,ly,lz) — 2×3
using Matrix2x3 = Eigen::Matrix<Scalar, 2, 3>;

/// alignToLoc3D:   ∂r_local3D/∂α  — 3×6
using Matrix3x6 = Eigen::Matrix<Scalar, 3, 6>;

/// static contribution to bound-param derivative:  2×3 × 3×6  — 2×6
using Matrix2x6 = Eigen::Matrix<Scalar, 2, 6>;

/// jacToLocal:     ∂ξ/∂τ  (free → bound Jacobian) — 6×8
using Matrix6x8 = Eigen::Matrix<Scalar, 6, 8>;

/// sphericalToFreeDir: ∂d̂/∂(φ,θ)  — 3×2
using Matrix3x2 = Eigen::Matrix<Scalar, 3, 2>;

// ── Square matrices ──────────────────────────────────────────────────────────

/// Layer composite rotation matrix R (global axes as columns) — 3×3
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;

/// Intermediate dynamic correction: (∂ξ/∂s)·(∂s/∂α) — 6×6
using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;

// ── Semantic aliases matching the paper's variable names ────────────────────
using AlignmentVector  = Vector6;   ///< α ∈ ℝ⁶
using BoundParamVector = Vector6;   ///< ξ ∈ ℝ⁶
using FreeParamVector  = Vector8;   ///< τ ∈ ℝ⁸
using GlobalPosition   = Vector3;   ///< r ∈ ℝ³
using GlobalDirection  = Vector3;   ///< d̂ ∈ ℝ³  (unit vector)
using SurfaceCentre    = Vector3;   ///< c ∈ ℝ³

/// Final output: dResidual/dα — 2×6
using ResidualJacobian = Matrix2x6;

} // namespace AlignDeriv
