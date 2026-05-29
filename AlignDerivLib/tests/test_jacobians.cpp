#include <AlignDeriv/Types.hpp>
#include <AlignDeriv/Parameters.hpp>
#include <AlignDeriv/JacobianHelpers.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace AlignDeriv;

static bool near(Scalar a, Scalar b, Scalar tol = 1e-10) {
    return std::abs(a - b) < tol;
}

int main() {
    constexpr Scalar pi = 3.14159265358979323846;

    // ── 1. Compile-time dimension assertions ─────────────────────────────────
    {
        auto J = sphericalToFreeDirectionJacobian(0.0, pi / 2);
        static_assert(decltype(J)::RowsAtCompileTime == 3, "");
        static_assert(decltype(J)::ColsAtCompileTime == 2, "");
    }
    {
        auto J = freeToSphericalDirectionJacobian(GlobalDirection(1, 0, 0));
        static_assert(decltype(J)::RowsAtCompileTime == 2, "");
        static_assert(decltype(J)::ColsAtCompileTime == 3, "");
    }
    {
        FreeTrackParameters ftp(GlobalPosition(0, 0, 0), 0.0,
                                GlobalDirection(0, 0, 1), 1.0);
        auto pd = calculatePathDerivative(ftp);
        static_assert(decltype(pd)::RowsAtCompileTime == 8, "");
        static_assert(decltype(pd)::ColsAtCompileTime == 1, "");
    }

    // ── 2. sphericalToFreeDirectionJacobian: known values ────────────────────
    // φ=0, θ=π/2  →  d̂ = (1, 0, 0)
    //   ∂d̂/∂φ = (0, 1, 0)
    //   ∂d̂/∂θ = (0, 0, -1)
    {
        auto J = sphericalToFreeDirectionJacobian(0.0, pi / 2);
        assert(near(J(0, 0),  0.0));   // ∂dx/∂φ
        assert(near(J(1, 0),  1.0));   // ∂dy/∂φ
        assert(near(J(2, 0),  0.0));   // ∂dz/∂φ
        assert(near(J(0, 1),  0.0));   // ∂dx/∂θ
        assert(near(J(1, 1),  0.0));   // ∂dy/∂θ
        assert(near(J(2, 1), -1.0));   // ∂dz/∂θ
    }
    // φ=π/2, θ=π/2  →  d̂ = (0, 1, 0)
    //   ∂d̂/∂φ = (-1, 0, 0)
    //   ∂d̂/∂θ = (0, 0, -1)
    {
        auto J = sphericalToFreeDirectionJacobian(pi / 2, pi / 2);
        assert(near(J(0, 0), -1.0));   // ∂dx/∂φ
        assert(near(J(1, 0),  0.0));   // ∂dy/∂φ
        assert(near(J(2, 0),  0.0));   // ∂dz/∂φ
        assert(near(J(0, 1),  0.0));   // ∂dx/∂θ
        assert(near(J(1, 1),  0.0));   // ∂dy/∂θ
        assert(near(J(2, 1), -1.0));   // ∂dz/∂θ
    }

    // ── 3. Round-trip: (freeToSpherical) × (sphericalToFree) = I₂ ───────────
    // Valid for any non-polar direction.
    auto check_roundtrip = [&](Scalar phi, Scalar theta) {
        const GlobalDirection d(std::sin(theta) * std::cos(phi),
                                std::sin(theta) * std::sin(phi),
                                std::cos(theta));
        auto J1 = sphericalToFreeDirectionJacobian(phi, theta);  // 3×2
        auto J2 = freeToSphericalDirectionJacobian(d);            // 2×3
        auto R  = J2 * J1;  // 2×2
        assert(near(R(0, 0), 1.0, 1e-9));
        assert(near(R(0, 1), 0.0, 1e-9));
        assert(near(R(1, 0), 0.0, 1e-9));
        assert(near(R(1, 1), 1.0, 1e-9));
    };

    check_roundtrip(0.0,       pi / 2);     // along +x
    check_roundtrip(pi / 2,    pi / 2);     // along +y
    check_roundtrip(pi / 4,    pi / 3);     // arbitrary diagonal
    check_roundtrip(3 * pi / 4, 2 * pi / 3);

    // ── 4. freeToSphericalDirectionJacobian: pole singularity guard ──────────
    // d̂ = (0, 0, 1)  →  θ = 0,  φ = atan2(0,0) = 0
    //   φ-row zeroed  (sinθ = 0, no valid azimuthal derivative)
    //   θ-row:  ∂θ/∂d̂ = (cosθ cosφ, cosθ sinφ, -sinθ) = (1, 0, 0)
    {
        auto J = freeToSphericalDirectionJacobian(GlobalDirection(0, 0, 1));
        // φ-row must be zero
        assert(near(J(0, 0), 0.0));
        assert(near(J(0, 1), 0.0));
        assert(near(J(0, 2), 0.0));
        // θ-row
        assert(near(J(1, 0), 1.0));
        assert(near(J(1, 1), 0.0));
        assert(near(J(1, 2), 0.0));
    }
    // d̂ = (0, 0, -1)  →  θ = π,  sinθ = 0
    //   θ-row: ∂θ/∂d̂ = (cosπ cosφ, cosπ sinφ, -sinπ) = (-1, 0, 0)
    {
        auto J = freeToSphericalDirectionJacobian(GlobalDirection(0, 0, -1));
        assert(near(J(0, 0), 0.0));
        assert(near(J(0, 1), 0.0));
        assert(near(J(0, 2), 0.0));
        assert(near(J(1, 0), -1.0));
        assert(near(J(1, 1),  0.0));
        assert(near(J(1, 2),  0.0));
    }

    // ── 5. calculatePathDerivative ────────────────────────────────────────────
    // Track along z-axis: d̂ = (0, 0, 1)
    //   pathDerivative[0:3] = (0, 0, 1)
    //   pathDerivative[3:8] = 0
    {
        FreeTrackParameters ftp(GlobalPosition(1, 2, 3), 0.5,
                                GlobalDirection(0, 0, 1), 2.0);
        auto pd = calculatePathDerivative(ftp);
        assert(near(pd[FreeTrackParameters::kX],     0.0));
        assert(near(pd[FreeTrackParameters::kY],     0.0));
        assert(near(pd[FreeTrackParameters::kZ],     1.0));
        assert(near(pd[FreeTrackParameters::kTime],  0.0));
        assert(near(pd[FreeTrackParameters::kDirX],  0.0));
        assert(near(pd[FreeTrackParameters::kDirY],  0.0));
        assert(near(pd[FreeTrackParameters::kDirZ],  0.0));
        assert(near(pd[FreeTrackParameters::kQOverP],0.0));
    }
    // Track along arbitrary direction
    {
        const Scalar phi   = pi / 4;
        const Scalar theta = pi / 3;
        GlobalDirection d(std::sin(theta) * std::cos(phi),
                          std::sin(theta) * std::sin(phi),
                          std::cos(theta));
        FreeTrackParameters ftp(GlobalPosition(0, 0, 0), 0.0, d, 1.0);
        auto pd = calculatePathDerivative(ftp);
        assert(near(pd[0], d.x()));
        assert(near(pd[1], d.y()));
        assert(near(pd[2], d.z()));
        assert(pd.tail<5>().isZero());
    }

    std::cout << "JacobianHelpers test PASSED\n";
    return 0;
}
