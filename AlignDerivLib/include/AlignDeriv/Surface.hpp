#pragma once

#include "Types.hpp"
#include "JacobianHelpers.hpp"   // freeToSphericalDirectionJacobian, calculatePathDerivative
#include "Parameters.hpp"        // FreeTrackParameters, BoundTrackParameters

// ============================================================================
//  AlignDeriv/Surface.hpp
//
//  Abstract base class for all detector surface geometries used in the
//  alignment Jacobian chain.
//
//  Coordinate convention (matches ACTS):
//    rotation_ columns = local x, y, z axes expressed in global frame
//    global = center_ + rotation_ * local
//    local  = rotation_.transpose() * (global − center_)
//
//  Alignment rotation parameters α = (alpha, beta, gamma) represent
//  infinitesimal rotations about local x, y, z respectively.  At zero
//  perturbation the SO(3) Lie algebra gives:
//
//    ∂R/∂alpha|₀ = R₀ · Lx,   Lx = [[0, 0, 0],[0, 0,−1],[0, 1, 0]]
//    ∂R/∂beta |₀ = R₀ · Ly,   Ly = [[0, 0, 1],[0, 0, 0],[−1,0, 0]]
//    ∂R/∂gamma|₀ = R₀ · Lz,   Lz = [[0,−1, 0],[1, 0, 0],[0, 0, 0]]
// ============================================================================

namespace AlignDeriv {

/// Three 3×3 partial derivatives of R w.r.t. alignment rotation parameters
struct RotationDerivatives {
    Matrix3 dRdAlpha;   ///< ∂R/∂alpha  (infinitesimal rotation about local x)
    Matrix3 dRdBeta;    ///< ∂R/∂beta   (infinitesimal rotation about local y)
    Matrix3 dRdGamma;   ///< ∂R/∂gamma  (infinitesimal rotation about local z)
};

// ─────────────────────────────────────────────────────────────────────────────
class Surface {
public:
    Surface(const Vector3& center, const Matrix3& rotation)
        : center_(center), rotation_(rotation) {}

    virtual ~Surface() = default;

    const Vector3& center()   const { return center_; }
    const Matrix3& rotation() const { return rotation_; }

    // ── Coordinate helpers ────────────────────────────────────────────────────

    Vector3 toLocal(const Vector3& globalPos) const {
        return rotation_.transpose() * (globalPos - center_);
    }

    Vector3 toGlobal(const Vector3& localPos) const {
        return center_ + rotation_ * localPos;
    }

    // ── Pure virtual interface ────────────────────────────────────────────────

    /// ∂(loc0, loc1)/∂(lx, ly, lz)  — 2×3 Jacobian from local Cartesian to
    /// surface-bound coordinates, evaluated at globalPosition.
    virtual Matrix2x3 localCartesianToBoundLocalDerivative(
        const Vector3& globalPosition) const = 0;

    // ── Lie algebra rotation derivative ───────────────────────────────────────

    /// Returns ∂R/∂(alpha, beta, gamma) at zero perturbation.
    ///
    /// For R(α,β,γ) = R₀·Rx(α)·Ry(β)·Rz(γ), differentiated at α=β=γ=0:
    ///   ∂R/∂alpha = R₀ · Lx,  ∂R/∂beta = R₀ · Ly,  ∂R/∂gamma = R₀ · Lz
    RotationDerivatives rotationToLocalAxesDerivative() const {
        Matrix3 Lx = Matrix3::Zero();
        Lx(1, 2) = Scalar(-1);
        Lx(2, 1) = Scalar( 1);

        Matrix3 Ly = Matrix3::Zero();
        Ly(0, 2) = Scalar( 1);
        Ly(2, 0) = Scalar(-1);

        Matrix3 Lz = Matrix3::Zero();
        Lz(0, 1) = Scalar(-1);
        Lz(1, 0) = Scalar( 1);

        return {rotation_ * Lx, rotation_ * Ly, rotation_ * Lz};
    }

    // ── Full alignment derivative chain ───────────────────────────────────────

    /// ∂ξ/∂α — 6×6 matrix mapping alignment perturbations to bound-param shifts.
    ///
    /// Assembly (plane-surface constraint: n·(r−c) = 0, n = rotation_[:,2]):
    ///   1. alignToLoc3D      (3×6) = [−I₃ | dRᵢᵀ·r_rel]
    ///   2. alignToBoundNoCorr(6×6) — J_proj·alignToLoc3D in top-2 rows
    ///   3. alignToPath       (1×6) — ∂s/∂α from differentiating the constraint
    ///   4. jacToLocal        (6×8) — instantaneous free→bound Jacobian at surface
    ///   5. Total = alignToBoundNoCorr + jacToLocal · (∂τ/∂s) · alignToPath
    Matrix6 alignmentToBoundDerivative(
        const FreeTrackParameters& freeTrack,
        const Vector3&             intersection,
        Scalar /*pathLength*/) const
    {
        const auto [dRdAlpha, dRdBeta, dRdGamma] = rotationToLocalAxesDerivative();
        const Vector3 r_rel = intersection - center_;

        // ── 1. alignToLoc3D (3×6) ────────────────────────────────────────────
        Matrix3x6 alignToLoc3D = Matrix3x6::Zero();
        alignToLoc3D.leftCols<3>() = -Matrix3::Identity();
        alignToLoc3D.col(3) = dRdAlpha.transpose() * r_rel;
        alignToLoc3D.col(4) = dRdBeta.transpose()  * r_rel;
        alignToLoc3D.col(5) = dRdGamma.transpose() * r_rel;

        // ── 2. alignToBoundWithoutCorrection (6×6) ────────────────────────────
        const Matrix2x3 J_proj = localCartesianToBoundLocalDerivative(intersection);
        Matrix6 alignToBoundNoCorr = Matrix6::Zero();
        alignToBoundNoCorr.topRows<2>() = J_proj * alignToLoc3D;

        // ── 3. alignToPath (1×6): ∂s/∂α from plane constraint n·(r−c)=0 ─────
        const Vector3 normal    = rotation_.col(2);   // local z-axis in global frame
        const Vector3 direction = freeTrack.direction();
        const Scalar  nDotD     = normal.dot(direction);

        const Vector3 ez(Scalar(0), Scalar(0), Scalar(1));
        Matrix1x6 alignToPath = Matrix1x6::Zero();
        alignToPath(0, 2) =  Scalar(1) / nDotD;                         // ∂s/∂tz
        alignToPath(0, 3) = -(dRdAlpha * ez).dot(r_rel) / nDotD;        // ∂s/∂α
        alignToPath(0, 4) = -(dRdBeta  * ez).dot(r_rel) / nDotD;        // ∂s/∂β
        alignToPath(0, 5) = -(dRdGamma * ez).dot(r_rel) / nDotD;        // ∂s/∂γ

        // ── 4. jacToLocal (6×8): free → bound Jacobian at the surface ─────────
        const Matrix2x3 J_loc_pos = J_proj * rotation_.transpose();
        const Matrix2x3 J_sph     = freeToSphericalDirectionJacobian(direction);

        Matrix6x8 jacToLocal = Matrix6x8::Zero();
        jacToLocal.block<2, 3>(0, FreeTrackParameters::kX)    = J_loc_pos;
        jacToLocal.block<2, 3>(2, FreeTrackParameters::kDirX) = J_sph;
        jacToLocal(4, FreeTrackParameters::kQOverP) = Scalar(1);
        jacToLocal(5, FreeTrackParameters::kTime)   = Scalar(1);

        // ── 5. Total = static term + path-length correction ───────────────────
        const Matrix8x1 pathDeriv = calculatePathDerivative(freeTrack);
        return alignToBoundNoCorr + jacToLocal * pathDeriv * alignToPath;
    }

    /// ∂residual/∂α — top-2 rows of alignmentToBoundDerivative (2×6).
    /// Directly usable as the alignment-to-measurement Jacobian in a track fit.
    Matrix2x6 alignmentToResidualDerivative(
        const FreeTrackParameters& freeTrack,
        const Vector3&             intersection,
        Scalar                     pathLength) const
    {
        return alignmentToBoundDerivative(freeTrack, intersection, pathLength)
                   .topRows<2>();
    }

protected:
    Vector3 center_;
    Matrix3 rotation_;
};

}  // namespace AlignDeriv
