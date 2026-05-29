#pragma once

#include "Types.hpp"

// ============================================================================
//  AlignDeriv/Parameters.hpp
//
//  POD-like parameter structs for the two track-state representations used
//  in alignment Jacobian computation.
//
//  Both structs carry a fixed-size Eigen vector plus strongly-typed index
//  constants and named accessors so that call sites look like physics, not
//  raw array indexing.
// ============================================================================

namespace AlignDeriv {

// ────────────────────────────────────────────────────────────────────────────
//  Free-space track parameters  τ ∈ ℝ⁸
//
//  Layout:  [ x,  y,  z,  t,  dx,  dy,  dz,  q/p ]
//            0    1    2    3    4    5    6    7
//
//  The direction triple (dx, dy, dz) is a unit vector d̂.
// ────────────────────────────────────────────────────────────────────────────
struct FreeTrackParameters {
    // ── Index constants ─────────────────────────────────────────────────────
    static constexpr int kX      = 0;
    static constexpr int kY      = 1;
    static constexpr int kZ      = 2;
    static constexpr int kTime   = 3;
    static constexpr int kDirX   = 4;
    static constexpr int kDirY   = 5;
    static constexpr int kDirZ   = 6;
    static constexpr int kQOverP = 7;

    // ── Storage ─────────────────────────────────────────────────────────────
    FreeParamVector params = FreeParamVector::Zero();

    // ── Constructors ─────────────────────────────────────────────────────────
    FreeTrackParameters() = default;

    explicit FreeTrackParameters(const FreeParamVector& v) : params(v) {}

    FreeTrackParameters(const GlobalPosition& pos,
                        Scalar                time,
                        const GlobalDirection& dir,
                        Scalar                qOverP) {
        params.template head<3>()    = pos;
        params[kTime]                = time;
        params.template segment<3>(kDirX) = dir;
        params[kQOverP]              = qOverP;
    }

    // ── Scalar accessors ────────────────────────────────────────────────────
    Scalar& x()      { return params[kX]; }
    Scalar& y()      { return params[kY]; }
    Scalar& z()      { return params[kZ]; }
    Scalar& time()   { return params[kTime]; }
    Scalar& dirX()   { return params[kDirX]; }
    Scalar& dirY()   { return params[kDirY]; }
    Scalar& dirZ()   { return params[kDirZ]; }
    Scalar& qOverP() { return params[kQOverP]; }

    Scalar x()      const { return params[kX]; }
    Scalar y()      const { return params[kY]; }
    Scalar z()      const { return params[kZ]; }
    Scalar time()   const { return params[kTime]; }
    Scalar dirX()   const { return params[kDirX]; }
    Scalar dirY()   const { return params[kDirY]; }
    Scalar dirZ()   const { return params[kDirZ]; }
    Scalar qOverP() const { return params[kQOverP]; }

    // ── Block accessors ─────────────────────────────────────────────────────

    /// Global position r = (x, y, z)
    Eigen::Block<FreeParamVector, 3, 1> position() {
        return params.template head<3>();
    }
    GlobalPosition position() const {
        return params.template head<3>();
    }

    /// Momentum direction unit vector d̂ = (dx, dy, dz)
    Eigen::Block<FreeParamVector, 3, 1> direction() {
        return params.template segment<3>(kDirX);
    }
    GlobalDirection direction() const {
        return params.template segment<3>(kDirX);
    }
};

// ────────────────────────────────────────────────────────────────────────────
//  Bound (surface-local) track parameters  ξ ∈ ℝ⁶
//
//  Layout:  [ loc0,  loc1,  φ,  θ,  q/p,  t ]
//              0      1     2    3    4     5
//
//  loc0/loc1 are 2D coordinates in the local surface frame.
//  φ (phi)   is the azimuthal angle of the momentum in global frame.
//  θ (theta) is the polar angle    of the momentum in global frame.
// ────────────────────────────────────────────────────────────────────────────
struct BoundTrackParameters {
    // ── Index constants ─────────────────────────────────────────────────────
    static constexpr int kLoc0   = 0;
    static constexpr int kLoc1   = 1;
    static constexpr int kPhi    = 2;
    static constexpr int kTheta  = 3;
    static constexpr int kQOverP = 4;
    static constexpr int kTime   = 5;

    // ── Storage ─────────────────────────────────────────────────────────────
    BoundParamVector params = BoundParamVector::Zero();

    // ── Constructors ─────────────────────────────────────────────────────────
    BoundTrackParameters() = default;

    explicit BoundTrackParameters(const BoundParamVector& v) : params(v) {}

    BoundTrackParameters(Scalar loc0, Scalar loc1,
                         Scalar phi,  Scalar theta,
                         Scalar qOverP, Scalar time) {
        params[kLoc0]   = loc0;
        params[kLoc1]   = loc1;
        params[kPhi]    = phi;
        params[kTheta]  = theta;
        params[kQOverP] = qOverP;
        params[kTime]   = time;
    }

    // ── Scalar accessors ────────────────────────────────────────────────────
    Scalar& loc0()   { return params[kLoc0]; }
    Scalar& loc1()   { return params[kLoc1]; }
    Scalar& phi()    { return params[kPhi]; }
    Scalar& theta()  { return params[kTheta]; }
    Scalar& qOverP() { return params[kQOverP]; }
    Scalar& time()   { return params[kTime]; }

    Scalar loc0()   const { return params[kLoc0]; }
    Scalar loc1()   const { return params[kLoc1]; }
    Scalar phi()    const { return params[kPhi]; }
    Scalar theta()  const { return params[kTheta]; }
    Scalar qOverP() const { return params[kQOverP]; }
    Scalar time()   const { return params[kTime]; }

    // ── Block accessors ─────────────────────────────────────────────────────

    /// 2D local position (loc0, loc1) used as the measurement/residual base
    Vector2 localPosition() const {
        return params.template head<2>();
    }
};

} // namespace AlignDeriv
