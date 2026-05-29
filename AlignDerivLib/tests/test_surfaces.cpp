#include <AlignDeriv/Surface.hpp>
#include <AlignDeriv/PlaneSurface.hpp>
#include <AlignDeriv/Parameters.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace AlignDeriv;

static bool near(Scalar a, Scalar b, Scalar tol = 1e-10) {
    return std::abs(a - b) < tol;
}

int main() {
    constexpr Scalar pi = 3.14159265358979323846;

    // ── 1. PlaneSurface: compile-time layout ─────────────────────────────────
    {
        PlaneSurface s(Vector3::Zero(), Matrix3::Identity());
        auto J = s.localCartesianToBoundLocalDerivative(Vector3(1, 2, 3));
        static_assert(decltype(J)::RowsAtCompileTime == 2, "");
        static_assert(decltype(J)::ColsAtCompileTime == 3, "");
    }

    // ── 2. PlaneSurface: constant Jacobian [[1,0,0],[0,1,0]] ─────────────────
    // Must hold for any global position and any surface placement.
    {
        // Identity placement
        PlaneSurface s(Vector3(0, 0, 0), Matrix3::Identity());

        for (const Vector3& gp : {Vector3(0, 0, 0),
                                   Vector3(1, 2, 3),
                                   Vector3(-5, 7, 100)}) {
            auto J = s.localCartesianToBoundLocalDerivative(gp);
            assert(near(J(0, 0), 1.0));  assert(near(J(0, 1), 0.0));  assert(near(J(0, 2), 0.0));
            assert(near(J(1, 0), 0.0));  assert(near(J(1, 1), 1.0));  assert(near(J(1, 2), 0.0));
        }
    }
    {
        // 90° rotation about z: local x → global y, local y → global -x
        Matrix3 R;
        R <<  0, -1, 0,
              1,  0, 0,
              0,  0, 1;
        PlaneSurface s(Vector3(10, 20, 30), R);

        auto J = s.localCartesianToBoundLocalDerivative(Vector3(0, 0, 0));
        assert(near(J(0, 0), 1.0));  assert(near(J(0, 1), 0.0));  assert(near(J(0, 2), 0.0));
        assert(near(J(1, 0), 0.0));  assert(near(J(1, 1), 1.0));  assert(near(J(1, 2), 0.0));
    }

    // ── 3. toLocal / toGlobal round-trip ─────────────────────────────────────
    {
        // Arbitrary rotation (45° about z)
        Matrix3 R;
        R << std::cos(pi/4), -std::sin(pi/4), 0,
             std::sin(pi/4),  std::cos(pi/4), 0,
             0,               0,              1;
        PlaneSurface s(Vector3(1, 2, 3), R);

        Vector3 global(4, 5, 6);
        Vector3 roundtrip = s.toGlobal(s.toLocal(global));
        assert((global - roundtrip).norm() < 1e-12);
    }

    // ── 4. rotationToLocalAxesDerivative: identity rotation ──────────────────
    // With R = I the derivatives must equal the raw Lie generators Lx, Ly, Lz.
    {
        PlaneSurface s(Vector3::Zero(), Matrix3::Identity());
        auto [dRdAlpha, dRdBeta, dRdGamma] = s.rotationToLocalAxesDerivative();

        // Expected: Lx = [[0,0,0],[0,0,-1],[0,1,0]]
        assert(near(dRdAlpha(0,0),  0.0));  assert(near(dRdAlpha(0,1),  0.0));  assert(near(dRdAlpha(0,2),  0.0));
        assert(near(dRdAlpha(1,0),  0.0));  assert(near(dRdAlpha(1,1),  0.0));  assert(near(dRdAlpha(1,2), -1.0));
        assert(near(dRdAlpha(2,0),  0.0));  assert(near(dRdAlpha(2,1),  1.0));  assert(near(dRdAlpha(2,2),  0.0));

        // Expected: Ly = [[0,0,1],[0,0,0],[-1,0,0]]
        assert(near(dRdBeta(0,0),  0.0));   assert(near(dRdBeta(0,1),  0.0));   assert(near(dRdBeta(0,2),  1.0));
        assert(near(dRdBeta(1,0),  0.0));   assert(near(dRdBeta(1,1),  0.0));   assert(near(dRdBeta(1,2),  0.0));
        assert(near(dRdBeta(2,0), -1.0));   assert(near(dRdBeta(2,1),  0.0));   assert(near(dRdBeta(2,2),  0.0));

        // Expected: Lz = [[0,-1,0],[1,0,0],[0,0,0]]
        assert(near(dRdGamma(0,0),  0.0));  assert(near(dRdGamma(0,1), -1.0));  assert(near(dRdGamma(0,2),  0.0));
        assert(near(dRdGamma(1,0),  1.0));  assert(near(dRdGamma(1,1),  0.0));  assert(near(dRdGamma(1,2),  0.0));
        assert(near(dRdGamma(2,0),  0.0));  assert(near(dRdGamma(2,1),  0.0));  assert(near(dRdGamma(2,2),  0.0));
    }

    // ── 5. rotationToLocalAxesDerivative: numerical finite-difference check ───
    // Verify ∂R/∂alpha ≈ (Rx(δ)·R₀ − R₀) / δ for small δ.
    {
        PlaneSurface s(Vector3::Zero(), Matrix3::Identity());
        const Scalar delta = 1e-7;

        auto [dRdAlpha, dRdBeta, dRdGamma] = s.rotationToLocalAxesDerivative();

        // Rx(δ) — rotation by δ about x-axis
        Matrix3 Rx = Matrix3::Identity();
        Rx(1, 1) =  std::cos(delta);  Rx(1, 2) = -std::sin(delta);
        Rx(2, 1) =  std::sin(delta);  Rx(2, 2) =  std::cos(delta);

        Matrix3 Ry = Matrix3::Identity();
        Ry(0, 0) =  std::cos(delta);  Ry(0, 2) =  std::sin(delta);
        Ry(2, 0) = -std::sin(delta);  Ry(2, 2) =  std::cos(delta);

        Matrix3 Rz = Matrix3::Identity();
        Rz(0, 0) =  std::cos(delta);  Rz(0, 1) = -std::sin(delta);
        Rz(1, 0) =  std::sin(delta);  Rz(1, 1) =  std::cos(delta);

        const Matrix3 R0 = Matrix3::Identity();
        Matrix3 fdAlpha = (Rx * R0 - R0) / delta;
        Matrix3 fdBeta  = (Ry * R0 - R0) / delta;
        Matrix3 fdGamma = (Rz * R0 - R0) / delta;

        assert((dRdAlpha - fdAlpha).norm() < 1e-6);
        assert((dRdBeta  - fdBeta ).norm() < 1e-6);
        assert((dRdGamma - fdGamma).norm() < 1e-6);
    }

    // ── 6. alignmentToResidualDerivative: perpendicular track ────────────────
    // Identity surface, intersection at (1,0,0), track along +z: d̂=(0,0,1).
    // n = (0,0,1), n·d̂ = 1, r_rel = (1,0,0).
    //
    // Expected 2×6 result (analytically derived):
    //   [[-1, 0, 0, 0,  0,  0],
    //    [ 0,-1, 0, 0,  0, -1]]
    //
    // The (1,5) entry of -1 arises because rotating about local z (gamma) by
    // δγ tilts the y-axis toward the intersection offset x=1, shifting loc1.
    {
        PlaneSurface s(Vector3::Zero(), Matrix3::Identity());
        FreeTrackParameters ftp(Vector3(1, 0, 0), 0.0, Vector3(0, 0, 1), 1.0);
        Vector3 intersection(1, 0, 0);

        auto J = s.alignmentToResidualDerivative(ftp, intersection, 0.0);
        static_assert(decltype(J)::RowsAtCompileTime == 2, "");
        static_assert(decltype(J)::ColsAtCompileTime == 6, "");

        // Row 0 (loc0)
        assert(near(J(0, 0), -1.0));  assert(near(J(0, 1),  0.0));
        assert(near(J(0, 2),  0.0));  assert(near(J(0, 3),  0.0));
        assert(near(J(0, 4),  0.0));  assert(near(J(0, 5),  0.0));
        // Row 1 (loc1)
        assert(near(J(1, 0),  0.0));  assert(near(J(1, 1), -1.0));
        assert(near(J(1, 2),  0.0));  assert(near(J(1, 3),  0.0));
        assert(near(J(1, 4),  0.0));  assert(near(J(1, 5), -1.0));
    }

    // ── 7. alignmentToResidualDerivative: path-correction term ───────────────
    // Identity surface, intersection at center (r_rel=0), track at 45°:
    // d̂ = (1/√2, 0, 1/√2).  n·d̂ = 1/√2.
    //
    // With r_rel=0 all rotation columns of alignToLoc3D vanish, and the
    // path-correction term produces ∂loc0/∂tz = +1 (track travels further
    // in x when surface shifts along z).
    //
    // Expected 2×6:
    //   [[-1, 0, +1, 0, 0, 0],
    //    [ 0,-1,  0, 0, 0, 0]]
    {
        const Scalar inv_sqrt2 = Scalar(1) / std::sqrt(Scalar(2));
        PlaneSurface s(Vector3::Zero(), Matrix3::Identity());
        FreeTrackParameters ftp(Vector3::Zero(), 0.0,
                                Vector3(inv_sqrt2, 0, inv_sqrt2), 1.0);
        Vector3 intersection = Vector3::Zero();

        auto J = s.alignmentToResidualDerivative(ftp, intersection, 0.0);

        assert(near(J(0, 0), -1.0, 1e-9));  assert(near(J(0, 1), 0.0, 1e-9));
        assert(near(J(0, 2),  1.0, 1e-9));  assert(near(J(0, 3), 0.0, 1e-9));
        assert(near(J(0, 4),  0.0, 1e-9));  assert(near(J(0, 5), 0.0, 1e-9));

        assert(near(J(1, 0),  0.0, 1e-9));  assert(near(J(1, 1), -1.0, 1e-9));
        assert(near(J(1, 2),  0.0, 1e-9));  assert(near(J(1, 3),  0.0, 1e-9));
        assert(near(J(1, 4),  0.0, 1e-9));  assert(near(J(1, 5),  0.0, 1e-9));
    }

    std::cout << "Surface test PASSED\n";
    return 0;
}
