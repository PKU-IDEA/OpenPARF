/**
 * @file   geometry.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "pybind/geometry.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(Point2DInt);
PYBIND11_MAKE_OPAQUE(VectorPoint2DInt);

PYBIND11_MAKE_OPAQUE(Point2DUint);
PYBIND11_MAKE_OPAQUE(VectorPoint2DUint);

PYBIND11_MAKE_OPAQUE(Point2DFloat);
PYBIND11_MAKE_OPAQUE(VectorPoint2DFloat);

PYBIND11_MAKE_OPAQUE(Point2DDouble);
PYBIND11_MAKE_OPAQUE(VectorPoint2DDouble);

PYBIND11_MAKE_OPAQUE(Point3DInt);
PYBIND11_MAKE_OPAQUE(VectorPoint3DInt);

PYBIND11_MAKE_OPAQUE(Point3DUint);
PYBIND11_MAKE_OPAQUE(VectorPoint3DUint);

PYBIND11_MAKE_OPAQUE(Point3DFloat);
PYBIND11_MAKE_OPAQUE(VectorPoint3DFloat);

PYBIND11_MAKE_OPAQUE(Point3DDouble);
PYBIND11_MAKE_OPAQUE(VectorPoint3DDouble);

PYBIND11_MAKE_OPAQUE(BoxInt);
PYBIND11_MAKE_OPAQUE(VectorBoxInt);

PYBIND11_MAKE_OPAQUE(BoxUint);
PYBIND11_MAKE_OPAQUE(VectorBoxUint);

PYBIND11_MAKE_OPAQUE(BoxFloat);
PYBIND11_MAKE_OPAQUE(VectorBoxFloat);

PYBIND11_MAKE_OPAQUE(BoxDouble);
PYBIND11_MAKE_OPAQUE(VectorBoxDouble);

PYBIND11_MAKE_OPAQUE(Size2DInt);
PYBIND11_MAKE_OPAQUE(VectorSize2DInt);

PYBIND11_MAKE_OPAQUE(Size2DUint);
PYBIND11_MAKE_OPAQUE(VectorSize2DUint);

PYBIND11_MAKE_OPAQUE(Size2DFloat);
PYBIND11_MAKE_OPAQUE(VectorSize2DFloat);

PYBIND11_MAKE_OPAQUE(Size2DDouble);
PYBIND11_MAKE_OPAQUE(VectorSize2DDouble);

PYBIND11_MAKE_OPAQUE(Size3DInt);
PYBIND11_MAKE_OPAQUE(VectorSize3DInt);

PYBIND11_MAKE_OPAQUE(Size3DUint);
PYBIND11_MAKE_OPAQUE(VectorSize3DUint);

PYBIND11_MAKE_OPAQUE(Size3DFloat);
PYBIND11_MAKE_OPAQUE(VectorSize3DFloat);

PYBIND11_MAKE_OPAQUE(Size3DDouble);
PYBIND11_MAKE_OPAQUE(VectorSize3DDouble);

void bind_geometry(py::module &m) {
  m.doc() = R"pbdoc(
        Pybind11 plugin
        -----------------------
        .. currentmodule:: geometry
        .. autosummary::
           :toctree: _generate
           Point
           Box
    )pbdoc";

  py::class_<Point2DInt>(m, "Point2DInt")
      .def(py::init<>())
      .def(py::init<int32_t const &, int32_t const &>())
      .def("x", &Point2DInt::x)
      .def("y", &Point2DInt::y)
      .def("setX", (void (Point2DInt::*)(int32_t const &)) & Point2DInt::setX)
      .def("setY", (void (Point2DInt::*)(int32_t const &)) & Point2DInt::setY)
      .def("set", (void (Point2DInt::*)(int32_t const &, int32_t const &)) &
                      Point2DInt::set)
      .def("__copy__", [](Point2DInt const &rhs) { return Point2DInt(rhs); })
      .def("__repr__",
           [](Point2DInt const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point2DInt const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Point2DUint>(m, "Point2DUint")
      .def(py::init<>())
      .def(py::init<uint32_t const &, uint32_t const &>())
      .def("x", &Point2DUint::x)
      .def("y", &Point2DUint::y)
      .def("setX",
           (void (Point2DUint::*)(uint32_t const &)) & Point2DUint::setX)
      .def("setY",
           (void (Point2DUint::*)(uint32_t const &)) & Point2DUint::setY)
      .def("set", (void (Point2DUint::*)(uint32_t const &, uint32_t const &)) &
                      Point2DUint::set)
      .def("__copy__", [](Point2DUint const &rhs) { return Point2DUint(rhs); })
      .def("__repr__",
           [](Point2DUint const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point2DUint const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Point2DFloat>(m, "Point2DFloat")
      .def(py::init<>())
      .def(py::init<float const &, float const &>())
      .def("x", &Point2DFloat::x)
      .def("y", &Point2DFloat::y)
      .def("setX", (void (Point2DFloat::*)(float const &)) & Point2DFloat::setX)
      .def("setY", (void (Point2DFloat::*)(float const &)) & Point2DFloat::setY)
      .def("set", (void (Point2DFloat::*)(float const &, float const &)) &
                      Point2DFloat::set)
      .def("__copy__",
           [](Point2DFloat const &rhs) { return Point2DFloat(rhs); })
      .def("__repr__",
           [](Point2DFloat const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point2DFloat const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Point2DDouble>(m, "Point2DDouble")
      .def(py::init<>())
      .def(py::init<double const &, double const &>())
      .def("x", &Point2DDouble::x)
      .def("y", &Point2DDouble::y)
      .def("setX",
           (void (Point2DDouble::*)(double const &)) & Point2DDouble::setX)
      .def("setY",
           (void (Point2DDouble::*)(double const &)) & Point2DDouble::setY)
      .def("set", (void (Point2DDouble::*)(double const &, double const &)) &
                      Point2DDouble::set)
      .def("__copy__",
           [](Point2DDouble const &rhs) { return Point2DDouble(rhs); })
      .def("__repr__",
           [](Point2DDouble const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point2DDouble const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Point3DInt>(m, "Point3DInt")
      .def(py::init<>())
      .def(py::init<int32_t const &, int32_t const &, int32_t const &>())
      .def("x", &Point3DInt::x)
      .def("y", &Point3DInt::y)
      .def("z", &Point3DInt::z)
      .def("setX", (void (Point3DInt::*)(int32_t const &)) & Point3DInt::setX)
      .def("setY", (void (Point3DInt::*)(int32_t const &)) & Point3DInt::setY)
      .def("setZ", (void (Point3DInt::*)(int32_t const &)) & Point3DInt::setZ)
      .def("set", (void (Point3DInt::*)(int32_t const &, int32_t const &,
                                        int32_t const &)) &
                      Point3DInt::set)
      .def("__copy__", [](Point3DInt const &rhs) { return Point3DInt(rhs); })
      .def("__repr__",
           [](Point3DInt const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point3DInt const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<Point3DUint>(m, "Point3DUint")
      .def(py::init<>())
      .def(py::init<uint32_t const &, uint32_t const &, uint32_t const &>())
      .def("x", &Point3DUint::x)
      .def("y", &Point3DUint::y)
      .def("z", &Point3DUint::z)
      .def("setX",
           (void (Point3DUint::*)(uint32_t const &)) & Point3DUint::setX)
      .def("setY",
           (void (Point3DUint::*)(uint32_t const &)) & Point3DUint::setY)
      .def("setZ",
           (void (Point3DUint::*)(uint32_t const &)) & Point3DUint::setZ)
      .def("set", (void (Point3DUint::*)(uint32_t const &, uint32_t const &,
                                         uint32_t const &)) &
                      Point3DUint::set)
      .def("__copy__", [](Point3DUint const &rhs) { return Point3DUint(rhs); })
      .def("__repr__",
           [](Point3DUint const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point3DUint const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<Point3DFloat>(m, "Point3DFloat")
      .def(py::init<>())
      .def(py::init<float const &, float const &, float const &>())
      .def("x", &Point3DFloat::x)
      .def("y", &Point3DFloat::y)
      .def("z", &Point3DFloat::z)
      .def("setX", (void (Point3DFloat::*)(float const &)) & Point3DFloat::setX)
      .def("setY", (void (Point3DFloat::*)(float const &)) & Point3DFloat::setY)
      .def("setZ", (void (Point3DFloat::*)(float const &)) & Point3DFloat::setZ)
      .def("set", (void (Point3DFloat::*)(float const &, float const &,
                                          float const &)) &
                      Point3DFloat::set)
      .def("__copy__",
           [](Point3DFloat const &rhs) { return Point3DFloat(rhs); })
      .def("__repr__",
           [](Point3DFloat const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point3DFloat const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<Point3DDouble>(m, "Point3DDouble")
      .def(py::init<>())
      .def(py::init<double const &, double const &, double const &>())
      .def("x", &Point3DDouble::x)
      .def("y", &Point3DDouble::y)
      .def("z", &Point3DDouble::z)
      .def("setX",
           (void (Point3DDouble::*)(double const &)) & Point3DDouble::setX)
      .def("setY",
           (void (Point3DDouble::*)(double const &)) & Point3DDouble::setY)
      .def("setZ",
           (void (Point3DDouble::*)(double const &)) & Point3DDouble::setZ)
      .def("set", (void (Point3DDouble::*)(double const &, double const &,
                                           double const &)) &
                      Point3DDouble::set)
      .def("__copy__",
           [](Point3DDouble const &rhs) { return Point3DDouble(rhs); })
      .def("__repr__",
           [](Point3DDouble const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Point3DDouble const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<BoxInt>(m, "BoxInt")
      .def(py::init<>())
      .def(py::init<int32_t, int32_t, int32_t, int32_t>())
      .def("xl", &BoxInt::xl)
      .def("yl", &BoxInt::yl)
      .def("xh", &BoxInt::xh)
      .def("yh", &BoxInt::yh)
      .def("width", &BoxInt::width)
      .def("height", &BoxInt::height)
      .def("area", &BoxInt::area)
      .def("set",
           (void (BoxInt::*)(int32_t, int32_t, int32_t, int32_t)) & BoxInt::set)
      .def("__copy__", [](BoxInt const &rhs) { return BoxInt(rhs); })
      .def("__repr__",
           [](BoxInt const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](BoxInt const &rhs) {
        return py::list(py::make_tuple(rhs.xl(), rhs.yl(), rhs.xh(), rhs.yh()));
      });

  py::class_<BoxUint>(m, "BoxUint")
      .def(py::init<>())
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>())
      .def("xl", &BoxUint::xl)
      .def("yl", &BoxUint::yl)
      .def("xh", &BoxUint::xh)
      .def("yh", &BoxUint::yh)
      .def("width", &BoxUint::width)
      .def("height", &BoxUint::height)
      .def("area", &BoxUint::area)
      .def("set", (void (BoxUint::*)(uint32_t, uint32_t, uint32_t, uint32_t)) &
                      BoxUint::set)
      .def("__copy__", [](BoxUint const &rhs) { return BoxUint(rhs); })
      .def("__repr__",
           [](BoxUint const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](BoxUint const &rhs) {
        return py::list(py::make_tuple(rhs.xl(), rhs.yl(), rhs.xh(), rhs.yh()));
      });

  py::class_<BoxFloat>(m, "BoxFloat")
      .def(py::init<>())
      .def(py::init<float, float, float, float>())
      .def("xl", &BoxFloat::xl)
      .def("yl", &BoxFloat::yl)
      .def("xh", &BoxFloat::xh)
      .def("yh", &BoxFloat::yh)
      .def("width", &BoxFloat::width)
      .def("height", &BoxFloat::height)
      .def("area", &BoxFloat::area)
      .def("set",
           (void (BoxFloat::*)(float, float, float, float)) & BoxFloat::set)
      .def("__copy__", [](BoxFloat const &rhs) { return BoxFloat(rhs); })
      .def("__repr__",
           [](BoxFloat const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](BoxFloat const &rhs) {
        return py::list(py::make_tuple(rhs.xl(), rhs.yl(), rhs.xh(), rhs.yh()));
      });

  py::class_<BoxDouble>(m, "BoxDouble")
      .def(py::init<>())
      .def(py::init<double, double, double, double>())
      .def("xl", &BoxDouble::xl)
      .def("yl", &BoxDouble::yl)
      .def("xh", &BoxDouble::xh)
      .def("yh", &BoxDouble::yh)
      .def("width", &BoxDouble::width)
      .def("height", &BoxDouble::height)
      .def("area", &BoxDouble::area)
      .def("set", (void (BoxDouble::*)(double, double, double, double)) &
                      BoxDouble::set)
      .def("__copy__", [](BoxDouble const &rhs) { return BoxDouble(rhs); })
      .def("__repr__",
           [](BoxDouble const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](BoxDouble const &rhs) {
        return py::list(py::make_tuple(rhs.xl(), rhs.yl(), rhs.xh(), rhs.yh()));
      });

  py::class_<Size2DInt>(m, "Size2DInt")
      .def(py::init<>())
      .def(py::init<int32_t const &, int32_t const &>())
      .def("x", &Size2DInt::x)
      .def("width", &Size2DInt::width)
      .def("y", &Size2DInt::y)
      .def("height", &Size2DInt::height)
      .def("setX", (void (Size2DInt::*)(int32_t const &)) & Size2DInt::setX)
      .def("setWidth",
           (void (Size2DInt::*)(int32_t const &)) & Size2DInt::setWidth)
      .def("setY", (void (Size2DInt::*)(int32_t const &)) & Size2DInt::setY)
      .def("setHeight",
           (void (Size2DInt::*)(int32_t const &)) & Size2DInt::setHeight)
      .def("set", (void (Size2DInt::*)(int32_t const &, int32_t const &)) &
                      Size2DInt::set)
      .def("product", &Size2DInt::product)
      .def("__copy__", [](Size2DInt const &rhs) { return Size2DInt(rhs); })
      .def("__repr__",
           [](Size2DInt const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size2DInt const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Size2DUint>(m, "Size2DUint")
      .def(py::init<>())
      .def(py::init<uint32_t const &, uint32_t const &>())
      .def("x", &Size2DUint::x)
      .def("width", &Size2DUint::width)
      .def("y", &Size2DUint::y)
      .def("height", &Size2DUint::height)
      .def("setX", (void (Size2DUint::*)(uint32_t const &)) & Size2DUint::setX)
      .def("setWidth",
           (void (Size2DUint::*)(uint32_t const &)) & Size2DUint::setWidth)
      .def("setY", (void (Size2DUint::*)(uint32_t const &)) & Size2DUint::setY)
      .def("setHeight",
           (void (Size2DUint::*)(uint32_t const &)) & Size2DUint::setHeight)
      .def("set", (void (Size2DUint::*)(uint32_t const &, uint32_t const &)) &
                      Size2DUint::set)
      .def("product", &Size2DUint::product)
      .def("__copy__", [](Size2DUint const &rhs) { return Size2DUint(rhs); })
      .def("__repr__",
           [](Size2DUint const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size2DUint const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Size2DFloat>(m, "Size2DFloat")
      .def(py::init<>())
      .def(py::init<float const &, float const &>())
      .def("x", &Size2DFloat::x)
      .def("width", &Size2DFloat::width)
      .def("y", &Size2DFloat::y)
      .def("height", &Size2DFloat::height)
      .def("setX", (void (Size2DFloat::*)(float const &)) & Size2DFloat::setX)
      .def("setWidth",
           (void (Size2DFloat::*)(float const &)) & Size2DFloat::setWidth)
      .def("setY", (void (Size2DFloat::*)(float const &)) & Size2DFloat::setY)
      .def("setHeight",
           (void (Size2DFloat::*)(float const &)) & Size2DFloat::setHeight)
      .def("set", (void (Size2DFloat::*)(float const &, float const &)) &
                      Size2DFloat::set)
      .def("product", &Size2DFloat::product)
      .def("__copy__", [](Size2DFloat const &rhs) { return Size2DFloat(rhs); })
      .def("__repr__",
           [](Size2DFloat const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size2DFloat const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Size2DDouble>(m, "Size2DDouble")
      .def(py::init<>())
      .def(py::init<double const &, double const &>())
      .def("x", &Size2DDouble::x)
      .def("width", &Size2DDouble::width)
      .def("y", &Size2DDouble::y)
      .def("height", &Size2DDouble::height)
      .def("setX",
           (void (Size2DDouble::*)(double const &)) & Size2DDouble::setX)
      .def("setWidth",
           (void (Size2DDouble::*)(double const &)) & Size2DDouble::setWidth)
      .def("setY",
           (void (Size2DDouble::*)(double const &)) & Size2DDouble::setY)
      .def("setHeight",
           (void (Size2DDouble::*)(double const &)) & Size2DDouble::setHeight)
      .def("set", (void (Size2DDouble::*)(double const &, double const &)) &
                      Size2DDouble::set)
      .def("product", &Size2DDouble::product)
      .def("__copy__",
           [](Size2DDouble const &rhs) { return Size2DDouble(rhs); })
      .def("__repr__",
           [](Size2DDouble const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size2DDouble const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y()));
      });

  py::class_<Size3DInt>(m, "Size3DInt")
      .def(py::init<>())
      .def(py::init<int32_t const &, int32_t const &, int32_t const &>())
      .def("x", &Size3DInt::x)
      .def("y", &Size3DInt::y)
      .def("z", &Size3DInt::z)
      .def("setX", (void (Size3DInt::*)(int32_t const &)) & Size3DInt::setX)
      .def("setY", (void (Size3DInt::*)(int32_t const &)) & Size3DInt::setY)
      .def("setZ", (void (Size3DInt::*)(int32_t const &)) & Size3DInt::setZ)
      .def("set", (void (Size3DInt::*)(int32_t const &, int32_t const &,
                                       int32_t const &)) &
                      Size3DInt::set)
      .def("product", &Size3DInt::product)
      .def("__copy__", [](Size3DInt const &rhs) { return Size3DInt(rhs); })
      .def("__repr__",
           [](Size3DInt const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size3DInt const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<Size3DUint>(m, "Size3DUint")
      .def(py::init<>())
      .def(py::init<uint32_t const &, uint32_t const &, uint32_t const &>())
      .def("x", &Size3DUint::x)
      .def("y", &Size3DUint::y)
      .def("z", &Size3DUint::z)
      .def("setX", (void (Size3DUint::*)(uint32_t const &)) & Size3DUint::setX)
      .def("setY", (void (Size3DUint::*)(uint32_t const &)) & Size3DUint::setY)
      .def("setZ", (void (Size3DUint::*)(uint32_t const &)) & Size3DUint::setZ)
      .def("set", (void (Size3DUint::*)(uint32_t const &, uint32_t const &,
                                        uint32_t const &)) &
                      Size3DUint::set)
      .def("product", &Size3DUint::product)
      .def("__copy__", [](Size3DUint const &rhs) { return Size3DUint(rhs); })
      .def("__repr__",
           [](Size3DUint const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size3DUint const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<Size3DFloat>(m, "Size3DFloat")
      .def(py::init<>())
      .def(py::init<float const &, float const &, float const &>())
      .def("x", &Size3DFloat::x)
      .def("y", &Size3DFloat::y)
      .def("z", &Size3DFloat::z)
      .def("setX", (void (Size3DFloat::*)(float const &)) & Size3DFloat::setX)
      .def("setY", (void (Size3DFloat::*)(float const &)) & Size3DFloat::setY)
      .def("setZ", (void (Size3DFloat::*)(float const &)) & Size3DFloat::setZ)
      .def("set", (void (Size3DFloat::*)(float const &, float const &,
                                         float const &)) &
                      Size3DFloat::set)
      .def("product", &Size3DFloat::product)
      .def("__copy__", [](Size3DFloat const &rhs) { return Size3DFloat(rhs); })
      .def("__repr__",
           [](Size3DFloat const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size3DFloat const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

  py::class_<Size3DDouble>(m, "Size3DDouble")
      .def(py::init<>())
      .def(py::init<double const &, double const &, double const &>())
      .def("x", &Size3DDouble::x)
      .def("y", &Size3DDouble::y)
      .def("z", &Size3DDouble::z)
      .def("setX",
           (void (Size3DDouble::*)(double const &)) & Size3DDouble::setX)
      .def("setY",
           (void (Size3DDouble::*)(double const &)) & Size3DDouble::setY)
      .def("setZ",
           (void (Size3DDouble::*)(double const &)) & Size3DDouble::setZ)
      .def("set", (void (Size3DDouble::*)(double const &, double const &,
                                          double const &)) &
                      Size3DDouble::set)
      .def("product", &Size3DDouble::product)
      .def("__copy__",
           [](Size3DDouble const &rhs) { return Size3DDouble(rhs); })
      .def("__repr__",
           [](Size3DDouble const &rhs) {
             std::stringstream os;
             os << rhs;
             return os.str();
           })
      .def("tolist", [](Size3DDouble const &rhs) {
        return py::list(py::make_tuple(rhs.x(), rhs.y(), rhs.z()));
      });

#define BIND_VECTOR_POINTSIZE_2D_TOLIST(T)                       \
  [](T const &rhs) {                                             \
    py::list ret(rhs.size());                                    \
    for (std::size_t i = 0; i < rhs.size(); ++i) {               \
      ret[i] = py::list(py::make_tuple(rhs[i].x(), rhs[i].y())); \
    }                                                            \
    return ret;                                                  \
  }

#define BIND_VECTOR_POINTSIZE_3D_TOLIST(T)                                   \
  [](T const &rhs) {                                                         \
    py::list ret(rhs.size());                                                \
    for (std::size_t i = 0; i < rhs.size(); ++i) {                           \
      ret[i] = py::list(py::make_tuple(rhs[i].x(), rhs[i].y(), rhs[i].z())); \
    }                                                                        \
    return ret;                                                              \
  }

#define BIND_VECTOR_BOX_TOLIST(T)                                              \
  [](T const &rhs) {                                                           \
    py::list ret(rhs.size());                                                  \
    for (std::size_t i = 0; i < rhs.size(); ++i) {                             \
      ret[i] = py::list(                                                       \
          py::make_tuple(rhs[i].xl(), rhs[i].yl(), rhs[i].xh(), rhs[i].yh())); \
    }                                                                          \
    return ret;                                                                \
  }

  py::bind_vector<VectorPoint2DInt>(m, "VectorPoint2DInt")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorPoint2DInt));
  py::bind_vector<VectorPoint2DUint>(m, "VectorPoint2DUint")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorPoint2DUint));

  py::bind_vector<VectorPoint2DFloat>(m, "VectorPoint2DFloat")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorPoint2DFloat));

  py::bind_vector<VectorPoint2DDouble>(m, "VectorPoint2DDouble")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorPoint2DDouble));

  py::bind_vector<VectorPoint3DInt>(m, "VectorPoint3DInt")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorPoint3DInt));

  py::bind_vector<VectorPoint3DUint>(m, "VectorPoint3DUint")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorPoint3DUint));

  py::bind_vector<VectorPoint3DFloat>(m, "VectorPoint3DFloat")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorPoint3DFloat));

  py::bind_vector<VectorPoint3DDouble>(m, "VectorPoint3DDouble")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorPoint3DDouble));

  py::bind_vector<VectorBoxInt>(m, "VectorBoxInt")
      .def("tolist", BIND_VECTOR_BOX_TOLIST(VectorBoxInt));
  py::bind_vector<VectorBoxUint>(m, "VectorBoxUint")
      .def("tolist", BIND_VECTOR_BOX_TOLIST(VectorBoxUint));
  py::bind_vector<VectorBoxFloat>(m, "VectorBoxFloat")
      .def("tolist", BIND_VECTOR_BOX_TOLIST(VectorBoxFloat));
  py::bind_vector<VectorBoxDouble>(m, "VectorBoxDouble")
      .def("tolist", BIND_VECTOR_BOX_TOLIST(VectorBoxDouble));

  py::bind_vector<VectorSize2DInt>(m, "VectorSize2DInt")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorSize2DInt));

  py::bind_vector<VectorSize2DUint>(m, "VectorSize2DUint")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorSize2DUint));

  py::bind_vector<VectorSize2DFloat>(m, "VectorSize2DFloat")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorSize2DFloat));

  py::bind_vector<VectorSize2DDouble>(m, "VectorSize2DDouble")
      .def("tolist", BIND_VECTOR_POINTSIZE_2D_TOLIST(VectorSize2DDouble));

  py::bind_vector<VectorSize3DInt>(m, "VectorSize3DInt")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorSize3DInt));

  py::bind_vector<VectorSize3DUint>(m, "VectorSize3DUint")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorSize3DUint));

  py::bind_vector<VectorSize3DFloat>(m, "VectorSize3DFloat")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorSize3DFloat));

  py::bind_vector<VectorSize3DDouble>(m, "VectorSize3DDouble")
      .def("tolist", BIND_VECTOR_POINTSIZE_3D_TOLIST(VectorSize3DDouble));

#undef BIND_VECTOR_POINTSIZE_2D_TOLIST
#undef BIND_VECTOR_POINTSIZE_3D_TOLIST
#undef BIND_VECTOR_BOX_TOLIST

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "develop";
#endif
}
