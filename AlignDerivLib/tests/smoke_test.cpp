#include <AlignDeriv/Types.hpp>
#include <AlignDeriv/Parameters.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace AlignDeriv;

int main() {
    // ── Verify compile-time dimensions via Eigen traits ──────────────────────
    static_assert(Vector3::RowsAtCompileTime    == 3);
    static_assert(Vector6::RowsAtCompileTime    == 6);
    static_assert(Vector8::RowsAtCompileTime    == 8);
    static_assert(Matrix2x3::RowsAtCompileTime  == 2 && Matrix2x3::ColsAtCompileTime  == 3);
    static_assert(Matrix3x6::RowsAtCompileTime  == 3 && Matrix3x6::ColsAtCompileTime  == 6);
    static_assert(Matrix2x6::RowsAtCompileTime  == 2 && Matrix2x6::ColsAtCompileTime  == 6);
    static_assert(Matrix6x8::RowsAtCompileTime  == 6 && Matrix6x8::ColsAtCompileTime  == 8);
    static_assert(Matrix8x1::RowsAtCompileTime  == 8 && Matrix8x1::ColsAtCompileTime  == 1);
    static_assert(Matrix1x6::RowsAtCompileTime  == 1 && Matrix1x6::ColsAtCompileTime  == 6);
    static_assert(ResidualJacobian::RowsAtCompileTime == 2 &&
                  ResidualJacobian::ColsAtCompileTime == 6);

    // ── FreeTrackParameters ──────────────────────────────────────────────────
    FreeTrackParameters ftp(
        GlobalPosition(1.0, 2.0, 3.0),
        /*time=*/  0.0,
        GlobalDirection(0.0, 0.0, 1.0),  // along z
        /*q/p=*/   1.0
    );
    assert(ftp.x()      == 1.0);
    assert(ftp.y()      == 2.0);
    assert(ftp.z()      == 3.0);
    assert(ftp.dirZ()   == 1.0);
    assert(ftp.params[FreeTrackParameters::kQOverP] == 1.0);
    assert(ftp.direction().norm() == 1.0);

    // mutate through accessor
    ftp.x() = 10.0;
    assert(ftp.params[0] == 10.0);

    // ── BoundTrackParameters ─────────────────────────────────────────────────
    constexpr Scalar pi = 3.14159265358979323846;
    BoundTrackParameters btp(
        /*loc0=*/  0.5,
        /*loc1=*/ -0.3,
        /*phi=*/   0.0,
        /*theta=*/ pi / 2.0,
        /*q/p=*/   0.5,
        /*time=*/  0.0
    );
    assert(btp.loc0()   == 0.5);
    assert(btp.loc1()   == -0.3);
    assert(btp.theta()  == pi / 2.0);
    assert(btp.localPosition()[0] == 0.5);
    assert(btp.localPosition()[1] == -0.3);

    // ── Zero initialisation guarantees ───────────────────────────────────────
    FreeTrackParameters ftp_zero;
    assert(ftp_zero.params.isZero());
    BoundTrackParameters btp_zero;
    assert(btp_zero.params.isZero());

    std::cout << "AlignDeriv smoke test PASSED\n";
    return 0;
}
