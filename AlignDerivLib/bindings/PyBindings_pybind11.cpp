#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   // Eigen ↔ numpy (pybind11 >= 2.12 supports numpy 2.x)
#include <pybind11/stl.h>

#include <AlignDeriv/Parameters.hpp>
#include <AlignDeriv/Surface.hpp>
#include <AlignDeriv/PlaneSurface.hpp>

namespace py = pybind11;
using namespace AlignDeriv;

// Trampoline: allows Python subclasses of Surface to override the pure virtual method.
class PySurface : public Surface {
public:
    using Surface::Surface;

    Matrix2x3 localCartesianToBoundLocalDerivative(
        const Vector3& globalPosition) const override {
        PYBIND11_OVERRIDE_PURE(
            Matrix2x3,
            Surface,
            localCartesianToBoundLocalDerivative,
            globalPosition
        );
    }
};

PYBIND11_MODULE(align_deriv_pb, m) {
    m.doc() = "AlignDeriv: alignment residual Jacobian engine (pybind11 build, numpy 2.x compatible via pybind11 >= 2.12)";

    // ── FreeTrackParameters ──────────────────────────────────────────────────
    py::class_<FreeTrackParameters>(m, "FreeTrackParameters",
        "Free-space track parameters τ ∈ ℝ⁸: (x,y,z,t,dx,dy,dz,q/p)")
        .def(py::init<>())
        .def(py::init<const FreeParamVector&>(), py::arg("params"))
        .def(py::init<const GlobalPosition&, Scalar, const GlobalDirection&, Scalar>(),
             py::arg("position"), py::arg("time"),
             py::arg("direction"), py::arg("qOverP"))
        .def_readwrite("params", &FreeTrackParameters::params)
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
    py::class_<BoundTrackParameters>(m, "BoundTrackParameters",
        "Surface-bound track parameters ξ ∈ ℝ⁶: (loc0,loc1,φ,θ,q/p,t)")
        .def(py::init<>())
        .def(py::init<const BoundParamVector&>(), py::arg("params"))
        .def(py::init<Scalar, Scalar, Scalar, Scalar, Scalar, Scalar>(),
             py::arg("loc0"), py::arg("loc1"),
             py::arg("phi"),  py::arg("theta"),
             py::arg("qOverP"), py::arg("time"))
        .def_readwrite("params", &BoundTrackParameters::params)
        .def("localPosition", &BoundTrackParameters::localPosition)
        .def("loc0",   [](const BoundTrackParameters& b) { return b.loc0(); })
        .def("loc1",   [](const BoundTrackParameters& b) { return b.loc1(); })
        .def("phi",    [](const BoundTrackParameters& b) { return b.phi(); })
        .def("theta",  [](const BoundTrackParameters& b) { return b.theta(); })
        .def("qOverP", [](const BoundTrackParameters& b) { return b.qOverP(); })
        .def("time",   [](const BoundTrackParameters& b) { return b.time(); });

    // ── Surface (abstract base) ──────────────────────────────────────────────
    py::class_<Surface, PySurface>(m, "Surface",
        "Abstract detector surface base class")
        .def(py::init<const Vector3&, const Matrix3&>(),
             py::arg("center"), py::arg("rotation"))
        .def("center",   [](const Surface& s) -> Vector3 { return s.center(); })
        .def("rotation", [](const Surface& s) -> Matrix3 { return s.rotation(); })
        .def("toLocal",  &Surface::toLocal,  py::arg("global_pos"))
        .def("toGlobal", &Surface::toGlobal, py::arg("local_pos"))
        .def("localCartesianToBoundLocalDerivative",
             &Surface::localCartesianToBoundLocalDerivative,
             py::arg("global_position"))
        .def("rotationToLocalAxesDerivative",
             [](const Surface& s) {
                 auto [dRdAlpha, dRdBeta, dRdGamma] = s.rotationToLocalAxesDerivative();
                 return py::make_tuple(dRdAlpha, dRdBeta, dRdGamma);
             },
             "Returns (dR/dAlpha, dR/dBeta, dR/dGamma) as a tuple of 3×3 numpy arrays")
        .def("alignmentToBoundDerivative",
             &Surface::alignmentToBoundDerivative,
             py::arg("free_track"), py::arg("intersection"), py::arg("path_length"),
             "6×6 matrix ∂ξ/∂α — full bound-parameter derivative w.r.t. alignment params")
        .def("alignmentToResidualDerivative",
             &Surface::alignmentToResidualDerivative,
             py::arg("free_track"), py::arg("intersection"), py::arg("path_length"),
             "2×6 matrix ∂residual/∂α — top-2 rows of alignmentToBoundDerivative");

    // ── PlaneSurface ─────────────────────────────────────────────────────────
    py::class_<PlaneSurface, Surface>(m, "PlaneSurface",
        "Planar detector surface with constant 2×3 local-coordinate Jacobian")
        .def(py::init<const Vector3&, const Matrix3&>(),
             py::arg("center"), py::arg("rotation"),
             "center: 3D global position of surface centre\n"
             "rotation: 3×3 matrix, columns = local x, y, z axes in global frame");
}
