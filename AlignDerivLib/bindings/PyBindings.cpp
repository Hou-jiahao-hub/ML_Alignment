#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>   // zero-copy Eigen ↔ NumPy (numpy 2.x compatible)
#include <nanobind/stl/string.h>
#include <nanobind/trampoline.h>    // NB_TRAMPOLINE / NB_OVERRIDE_PURE

#include <AlignDeriv/Parameters.hpp>
#include <AlignDeriv/Surface.hpp>
#include <AlignDeriv/PlaneSurface.hpp>

namespace nb = nanobind;
using namespace AlignDeriv;

// Trampoline: allows Python subclasses of Surface to override the virtual method.
struct PySurface : public Surface {
    NB_TRAMPOLINE(Surface, 1);  // 1 pure-virtual method

    Matrix2x3 localCartesianToBoundLocalDerivative(
        const Vector3& globalPosition) const override {
        NB_OVERRIDE_PURE(localCartesianToBoundLocalDerivative, globalPosition);
    }
};

NB_MODULE(align_deriv, m) {
    m.doc() = "AlignDeriv: alignment residual Jacobian engine (∂residual/∂α)";

    // ── FreeTrackParameters ──────────────────────────────────────────────────
    nb::class_<FreeTrackParameters>(m, "FreeTrackParameters",
        "Free-space track parameters τ ∈ ℝ⁸: (x,y,z,t,dx,dy,dz,q/p)")
        .def(nb::init<>())
        .def(nb::init<const FreeParamVector&>(), nb::arg("params"))
        .def(nb::init<const GlobalPosition&, Scalar, const GlobalDirection&, Scalar>(),
             nb::arg("position"), nb::arg("time"),
             nb::arg("direction"), nb::arg("qOverP"))
        .def_rw("params", &FreeTrackParameters::params)
        .def("position",  [](const FreeTrackParameters& f) -> GlobalPosition  { return f.position(); })
        .def("direction", [](const FreeTrackParameters& f) -> GlobalDirection { return f.direction(); })
        .def("x",      [](const FreeTrackParameters& f) { return f.x(); })
        .def("y",      [](const FreeTrackParameters& f) { return f.y(); })
        .def("z",      [](const FreeTrackParameters& f) { return f.z(); })
        .def("time",   [](const FreeTrackParameters& f) { return f.time(); })
        .def("dirX",   [](const FreeTrackParameters& f) { return f.dirX(); })
        .def("dirY",   [](const FreeTrackParameters& f) { return f.dirY(); })
        .def("dirZ",   [](const FreeTrackParameters& f) { return f.dirZ(); })
        .def("qOverP", [](const FreeTrackParameters& f) { return f.qOverP(); })
        .def("__repr__", [](const FreeTrackParameters& f) {
            return std::string("<FreeTrackParameters pos=(") +
                   std::to_string(f.x()) + "," +
                   std::to_string(f.y()) + "," +
                   std::to_string(f.z()) + ") dir=(" +
                   std::to_string(f.dirX()) + "," +
                   std::to_string(f.dirY()) + "," +
                   std::to_string(f.dirZ()) + ")>";
        });

    // ── BoundTrackParameters ─────────────────────────────────────────────────
    nb::class_<BoundTrackParameters>(m, "BoundTrackParameters",
        "Surface-bound track parameters ξ ∈ ℝ⁶: (loc0,loc1,φ,θ,q/p,t)")
        .def(nb::init<>())
        .def(nb::init<const BoundParamVector&>(), nb::arg("params"))
        .def(nb::init<Scalar, Scalar, Scalar, Scalar, Scalar, Scalar>(),
             nb::arg("loc0"), nb::arg("loc1"),
             nb::arg("phi"),  nb::arg("theta"),
             nb::arg("qOverP"), nb::arg("time"))
        .def_rw("params", &BoundTrackParameters::params)
        .def("localPosition", &BoundTrackParameters::localPosition)
        .def("loc0",   [](const BoundTrackParameters& b) { return b.loc0(); })
        .def("loc1",   [](const BoundTrackParameters& b) { return b.loc1(); })
        .def("phi",    [](const BoundTrackParameters& b) { return b.phi(); })
        .def("theta",  [](const BoundTrackParameters& b) { return b.theta(); })
        .def("qOverP", [](const BoundTrackParameters& b) { return b.qOverP(); })
        .def("time",   [](const BoundTrackParameters& b) { return b.time(); });

    // ── Surface (abstract base) ──────────────────────────────────────────────
    nb::class_<Surface, PySurface>(m, "Surface",
        "Abstract detector surface base class")
        .def(nb::init<const Vector3&, const Matrix3&>(),
             nb::arg("center"), nb::arg("rotation"))
        .def("center",   [](const Surface& s) -> Vector3 { return s.center(); })
        .def("rotation", [](const Surface& s) -> Matrix3 { return s.rotation(); })
        .def("toLocal",  &Surface::toLocal,  nb::arg("global_pos"))
        .def("toGlobal", &Surface::toGlobal, nb::arg("local_pos"))
        .def("localCartesianToBoundLocalDerivative",
             &Surface::localCartesianToBoundLocalDerivative,
             nb::arg("global_position"))
        .def("rotationToLocalAxesDerivative",
             [](const Surface& s) {
                 auto [dRdAlpha, dRdBeta, dRdGamma] = s.rotationToLocalAxesDerivative();
                 return nb::make_tuple(dRdAlpha, dRdBeta, dRdGamma);
             },
             "Returns (dR/dAlpha, dR/dBeta, dR/dGamma) as a tuple of 3×3 numpy arrays")
        .def("alignmentToBoundDerivative",
             &Surface::alignmentToBoundDerivative,
             nb::arg("free_track"), nb::arg("intersection"), nb::arg("path_length"),
             "6×6 matrix ∂ξ/∂α — full bound-parameter derivative w.r.t. alignment params")
        .def("alignmentToResidualDerivative",
             &Surface::alignmentToResidualDerivative,
             nb::arg("free_track"), nb::arg("intersection"), nb::arg("path_length"),
             "2×6 matrix ∂residual/∂α — top-2 rows of alignmentToBoundDerivative");

    // ── PlaneSurface ─────────────────────────────────────────────────────────
    nb::class_<PlaneSurface, Surface>(m, "PlaneSurface",
        "Planar detector surface with constant 2×3 local-coordinate Jacobian")
        .def(nb::init<const Vector3&, const Matrix3&>(),
             nb::arg("center"), nb::arg("rotation"),
             "center: 3D global position of surface centre\n"
             "rotation: 3×3 matrix, columns = local x, y, z axes in global frame");
}
