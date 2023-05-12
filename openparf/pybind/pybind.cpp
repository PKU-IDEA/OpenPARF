/**
 * @file   pybind.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "pybind/util.h"

namespace py = pybind11;

void bind_util(py::module &);
void bind_geometry(py::module &);
void bind_container(py::module &);
void bind_database(py::module &);
void bind_blend2d(py::module &);

PYBIND11_MODULE(OPENPARF_NAMESPACE, m) {
  m.doc() = R"pbdoc(
        Pybind11 plugin
        -----------------------
        .. currentmodule:: openparf
        .. autosummary::
           :toctree: _generate
           util
           geometry
           container
           database
    )pbdoc";

  bind_util(m);

  auto m_geometry = m.def_submodule("geometry");
  bind_geometry(m_geometry);

  auto m_container = m.def_submodule("container");
  bind_container(m_container);

  auto m_database = m.def_submodule("database");
  bind_database(m_database);

  bind_blend2d(m);
}
