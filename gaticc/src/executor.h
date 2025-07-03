#pragma once


#include "onnx_parser.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#include <vector>

class DispatchTable {
  bool dump_all;
  bool dump_none;
  std::vector<std::string> tbl;

public:
  DispatchTable();
  /* all nodes with no out-edges directly quality for dispatch */
  DispatchTable(Op::Graph graph);
  /* True if l's outputs need to be dumped */
  bool should_dispatch(const Op::LayerBase *l);
  void print();
};

/* Executor iterates over layers one by one, executing each one of them
 *
 * I am aware that dynamic_cast of a base into child is a code smell. I am
 * letting this one in.
 */

class Executor {
  /* A pool of heterogenously typed vectors corresponding to
   * `VirtualAddress` registers
   */
  TensorPool tensor_pool;
  DispatchTable dispatch_table;
  /* inputT: input type of the entire model
   * outputT: output type of the entire model
   */
  template <typename inputT>
  TensorPool run_aux(const Op::Parser &parser, Tensor<inputT>* arr);
  void print_extra_info(const Op::LayerBase *l);

public:
  Executor();
  TensorPool run(const std::string& onnx_path, py::array arr);
};

template <typename inputT>
TensorPool Executor::run_aux(const Op::Parser &parser, Tensor<inputT> *arr) {
  Tensor<inputT> *full_batch = arr;

  /* TODO: add checks here if inputs are batched and matches expected dims */
  if (full_batch->dims_size() <= 1) {
    log_fatal("Expects input images to be greater than 1 dimensional (N,...) "
              "got a {} dimensional image\n",
              full_batch->dims_size());
  }
  TensorPool ret;
  std::vector<Op::LayerBase *> order = parser.get_execution_order();
  Timer<std::chrono::seconds> tt;
  tt.start();
  for (int i = 0; i < full_batch->dims_at(0); ++i) {
    tensor_pool.free();

    Tensor<inputT> *slice{get_slice(full_batch, std::vector<int>{i})};
    if (order.at(0)->input_dims[0] != slice->get_dims()) {
      log_fatal("Expected input dims {}, got input of dimensions {}\n",
                order.at(0)->input_dims[0], slice->get_dims());
    }
    tensor_pool.set<Tensor<inputT> *>(0, slice);
    for (Op::LayerBase *l : order) {
      print_extra_info(l);
      l->dispatch = dispatch_table.should_dispatch(l);
      l->run(tensor_pool);

      if (parser.has_graph_output(l) || l->dispatch) {
        for (auto type : l->output_type) {
          /* TODO: use unique_ptr */
          if (type == onnx::TensorProto_DataType_INT8) {
            Tensor<int8_t> *out =
                tensor_pool.get<Tensor<int8_t> *>(l->outputs.at(0));
            Tensor<int8_t> *out_copy = new TensorCreate(out, l->name);
            ret.push_back<Tensor<int8_t> *>(out_copy);
          } else if (type == onnx::TensorProto_DataType_FLOAT) {
            Tensor<float> *out =
                tensor_pool.get<Tensor<float> *>(l->outputs.at(0));
            Tensor<float> *out_copy = new TensorCreate(out, l->name);
            ret.push_back<Tensor<float> *>(out_copy);
          } else if (type == onnx::TensorProto_DataType_INT32) {
            Tensor<int> *out =
                tensor_pool.get<Tensor<int> *>(l->outputs.at(0));
            Tensor<int> *out_copy = new TensorCreate(out, l->name);
            ret.push_back<Tensor<int> *>(out_copy);
          } else {
            log_fatal("Output type of layer {} ({}) is not supported\n", l->name,
                      Op::get_tensorproto_dtype_name(type));
          }
        }
      }
    }
  }
  tt.stop();
  if (get_verbose()) {
    tt.report("Total time taken by the model: ");
  }
  return ret;
}
