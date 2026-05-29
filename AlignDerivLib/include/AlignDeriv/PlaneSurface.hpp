#pragma once

#include "Surface.hpp"

// ============================================================================
//  AlignDeriv/PlaneSurface.hpp
//
//  Planar detector surface.
//
//  Bound local coordinates:
//    loc0 = lx   (local x coordinate in the surface plane)
//    loc1 = ly   (local y coordinate in the surface plane)
//    lz is the surface-normal offset — not a bound coordinate.
//
//  Because the mapping (lx, ly, lz) → (loc0, loc1) is purely linear, the
//  Jacobian is constant everywhere on (and off) the surface:
//
//    ∂(loc0, loc1)/∂(lx, ly, lz) = [[1, 0, 0],
//                                    [0, 1, 0]]
// ============================================================================

namespace AlignDeriv {

class PlaneSurface : public Surface {
public:
    using Surface::Surface;

    /// Constant 2×3 projection — independent of position
    Matrix2x3 localCartesianToBoundLocalDerivative(
        const Vector3& /*globalPosition*/) const override {
        Matrix2x3 J = Matrix2x3::Zero();
        J(0, 0) = Scalar(1);   // ∂loc0/∂lx
        J(1, 1) = Scalar(1);   // ∂loc1/∂ly
        return J;
    }
};

}  // namespace AlignDeriv
