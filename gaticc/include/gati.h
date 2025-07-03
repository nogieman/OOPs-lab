#pragma once

#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using std::string;
using vss = std::vector<std::pair<string, string>>;
namespace py = pybind11;

__attribute__((visibility("default"))) int compile(const string& onnx_path, const string &gml_path, const vss& );
__attribute__((visibility("default"))) int info(const string& onnx_path, const vss& rest);
__attribute__((visibility("default"))) py::list sim(const std::string& onnx_path, py::array arr, const vss& rest);
__attribute__((visibility("default"))) py::list run(const string& onnx_path, const string& gml_path, py::array arr, const vss& rest);
