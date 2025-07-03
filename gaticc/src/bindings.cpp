#include "gati.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

#define CATCH_BLOCK \
  catch (const std::exception &e) {\
    std::string ss = e.what();\
    throw std::runtime_error(ss);\
  }\

#define WRAP_EXCEPT3(fn, a1, a2, a3) \
  try {\
    return fn(a1, a2, a3);\
  } \
  CATCH_BLOCK \

#define WRAP_EXCEPT4(fn, a1, a2, a3, a4) \
  try {\
    return fn(a1, a2, a3, a4);\
  } \
  CATCH_BLOCK \


/* These 'safe' functions catch any exception and throw them as RuntimeError */

auto safe_compile = [](const string& onnx_path, const string &gml_path, const vss& rest) {
  WRAP_EXCEPT3(compile, onnx_path, gml_path, rest);
};

auto safe_sim = [](const string& onnx_path, py::array arr, const vss& rest) {
  WRAP_EXCEPT3(sim, onnx_path, arr, rest);
};

auto safe_run = [](const std::string& onnx_path, const std::string& gml_path, py::array arr, const vss& rest) {
  WRAP_EXCEPT4(run, onnx_path, gml_path, arr, rest);
};

PYBIND11_MODULE(_gati, m) {
  m.def("compile", safe_compile, "onnx_path"_a, "gml_path"_a, "rest"_a = py::list());
  m.def("info", &info, "onnx_path"_a, "rest"_a = py::list());
  m.def("version", []() {gbl_args.print_version();});
  m.def("help", []() {gbl_args.print_usage();});
  m.def("sim", safe_sim, "onnx_path"_a, "inp"_a, "rest"_a = py::list());
  m.def("run", safe_run, "onnx_path"_a, "gml_path"_a, "inp"_a, "rest"_a = py::list());
}
