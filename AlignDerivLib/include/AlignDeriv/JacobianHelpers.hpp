#pragma once

#include "Types.hpp"
#include "Parameters.hpp"
#include <cmath>
#include <algorithm>  // std::clamp

// ============================================================================
//  AlignDeriv/JacobianHelpers.hpp
//
//  Detector-geometry-independent Jacobian building blocks:
//
//  1.  sphericalToFreeDirectionJacobian  —  ∂d̂/∂(φ, θ)         (3×2)
//  2.  freeToSphericalDirectionJacobian  —  ∂(φ, θ)/∂d̂          (2×3)
//  3.  calculatePathDerivative           —  ∂τ/∂s for a straight track  (8×1)
//
//  All functions are inline and header-only; no compiled units required.
// ============================================================================

namespace AlignDeriv {

// ────────────────────────────────────────────────────────────────────────────
//  ∂d̂/∂(φ, θ)  —  3×2 spherical-to-Cartesian Jacobian
//
//  Convention: d̂ = (sinθ cosφ,  sinθ sinφ,  cosθ)
//
//  col 0  (∂d̂/∂φ) = (−sinθ sinφ,   sinθ cosφ,   0)
//  col 1  (∂d̂/∂θ) = ( cosθ cosφ,   cosθ sinφ,  −sinθ)
// ────────────────────────────────────────────────────────────────────────────
inline Matrix3x2
sphericalToFreeDirectionJacobian(Scalar phi, Scalar theta) {
    const Scalar sinPhi   = std::sin(phi);
    const Scalar cosPhi   = std::cos(phi);
    const Scalar sinTheta = std::sin(theta);
    const Scalar cosTheta = std::cos(theta);

    Matrix3x2 J;
    // ∂d̂/∂φ
    J(0, 0) = -sinTheta * sinPhi;
    J(1, 0) =  sinTheta * cosPhi;
    J(2, 0) =  Scalar(0);
    // ∂d̂/∂θ
    J(0, 1) =  cosTheta * cosPhi;
    J(1, 1) =  cosTheta * sinPhi;
    J(2, 1) = -sinTheta;

    return J;
}

// ────────────────────────────────────────────────────────────────────────────
//  ∂(φ, θ)/∂d̂  —  2×3 Cartesian-to-spherical Jacobian
//
//  Derived from:
//    φ = atan2(dy, dx)
//    θ = acos(dz)          (d̂ is a unit vector)
//
//  row 0 (∂φ/∂d̂) = (−sinφ/sinθ,  cosφ/sinθ,  0)
//  row 1 (∂θ/∂d̂) = ( cosθ cosφ,  cosθ sinφ, −sinθ)
//
//  Singularity guard: when |sinθ| < kEps (θ ≈ 0 or π, i.e. direction nearly
//  parallel to z-axis) the φ-row is zeroed to avoid division by zero.
// ────────────────────────────────────────────────────────────────────────────
inline Matrix2x3
freeToSphericalDirectionJacobian(const GlobalDirection& dir) {
    const Scalar phi   = std::atan2(dir.y(), dir.x());
    const Scalar theta = std::acos(std::clamp(dir.z(), Scalar(-1), Scalar(1)));

    const Scalar sinPhi   = std::sin(phi);
    const Scalar cosPhi   = std::cos(phi);
    const Scalar sinTheta = std::sin(theta);
    const Scalar cosTheta = std::cos(theta);

    constexpr Scalar kEps = Scalar(1e-9);

    Matrix2x3 J;
    J.setZero();

    if (std::abs(sinTheta) > kEps) {
        // row 0: ∂φ/∂d̂
        J(0, 0) = -sinPhi / sinTheta;
        J(0, 1) =  cosPhi / sinTheta;
        J(0, 2) =  Scalar(0);
    }
    // row 1: ∂θ/∂d̂  (well-defined everywhere on the unit sphere)
    J(1, 0) =  cosTheta * cosPhi;
    J(1, 1) =  cosTheta * sinPhi;
    J(1, 2) = -sinTheta;

    return J;
}

// ────────────────────────────────────────────────────────────────────────────
//  ∂τ/∂s  —  8×1 path derivative for straight-line propagation
//
//  From the straight-line track model τ(s) = τ₀ + s · (∂τ/∂s):
//
//    ∂x/∂s = dx,  ∂y/∂s = dy,  ∂z/∂s = dz
//    ∂t/∂s = 0    (engineering simplification per ACTS alignment engine)
//    ∂(d̂)/∂s = 0  (direction constant on a straight track)
//    ∂(q/p)/∂s = 0
//
//  Result layout matches FreeTrackParameters: [x,y,z,t,dx,dy,dz,q/p]
// ────────────────────────────────────────────────────────────────────────────
inline Matrix8x1
calculatePathDerivative(const FreeTrackParameters& freeParams) {
    Matrix8x1 deriv = Matrix8x1::Zero();
    deriv.template head<3>() = freeParams.direction();
    return deriv;
}

}  // namespace AlignDeriv
