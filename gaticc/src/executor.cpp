#include "executor.h"
#include "sim.h"
#include "utils.h"
#include "onnx_parser.h"
#include "optimization.h"

DispatchTable::DispatchTable() {
  dump_all = false;
  dump_none = false;
  if (gbl_args.has_option("dispatch")) {
    std::string arg = gbl_args["dispatch"].as<std::string>();
    if (strcmp(arg.c_str(), "all") == 0) {
      dump_all = true;
    } else if (strcmp(arg.c_str(), "none") == 0) {
      dump_none = true;
    } else {
      tbl = parse_csv_string<std::string>(arg);
    }
  }
}

static void check_dispatch_table_validity(const std::vector<std::string> &tbl,
                                          const Op::Graph &graph) {
  std::vector<std::string> graph_nodes;
  auto vitr = boost::vertices(graph);
  for (auto itr = vitr.first; itr != vitr.second; ++itr) {
    graph_nodes.push_back(graph[*itr]->name);
  }
  for (const auto &i : tbl) {
    auto itr = std::find(graph_nodes.begin(), graph_nodes.end(), i);
    if (itr == graph_nodes.end()) {
      log_fatal("Could not find layer {} in modified execution graph: either "
                "its not possible to dump this layer's contents or this layer "
                "does not exist in the graph (check netron graph for correct "
                "names)\n",
                i);
    }
  }
}

DispatchTable::DispatchTable(Op::Graph graph) {
  /* TODO: DRY in the constructor above */
  dump_all = false;
  dump_none = false;
  if (gbl_args.has_option("dispatch")) {
    std::string arg = gbl_args["dispatch"].as<std::string>();
    if (strcmp(arg.c_str(), "all") == 0) {
      dump_all = true;
    } else if (strcmp(arg.c_str(), "none") == 0) {
      dump_none = true;
    } else {
      tbl = parse_csv_string<std::string>(arg);
      check_dispatch_table_validity(tbl, graph);
    }
  } else {
    auto vitr = boost::vertices(graph);
    for (auto itr = vitr.first; itr != vitr.second; ++itr) {
      if (boost::out_degree(*itr, graph) == 0) {
        tbl.push_back(graph[*itr]->name);
      }
    }
  }
}

bool DispatchTable::should_dispatch(const Op::LayerBase *l) {
  if (dump_all) {
    return true;
  } else if (dump_none) {
    return false;
  } else {
    auto start = tbl.begin();
    auto stop = tbl.end();
    auto itr = std::find(start, stop, l->name);
    return (itr != stop) ? true : false;
  }
}

void DispatchTable::print() {
  std::cout << "Dispatch Table: \n";
  for (auto i : tbl) {
    std::cout << i << '\n';
  }
}

void Executor::print_extra_info(const Op::LayerBase *l) {
  if (get_verbose()) {
    std::cout << "Running " << l->op_type() << ' ' << l->name << ' '
              << Op::get_tensorproto_dtype_name(l->input_type[0]) << ' '
              << Op::get_tensorproto_dtype_name(l->output_type[0]) << '\n';
  }
}

Executor::Executor() {
  dispatch_table = DispatchTable();
}

TensorPool Executor::run(const std::string& onnx_path, py::array arr) {
  Op::Parser parser(onnx_path);
  split_large_kernel(parser.get_graph());

  TPDT input_type = parser.get_model_input_type();

  int total_regs = parser.get_total_registers() + 1;
  tensor_pool.resize(total_regs);

  if (input_type == onnx::TensorProto_DataType_FLOAT) {
    Tensor<float> *input = new TensorCreate<float>(arr, "data");
    return run_aux<float>(parser, input);
  } else if (input_type == onnx::TensorProto_DataType_INT8) {
    Tensor<int8_t> *input = new TensorCreate<int8_t>(arr, "data");
    return run_aux<int8_t>(parser, input);
  } else {
    log_fatal("Unsupported input type {}\n", Op::get_tensorproto_dtype_name(input_type));
    return TensorPool();
  }
}

/* Should be used only for layers with single input and output (like Conv, Relu etc.).
 * For layers with multiple io (like Add), prefer doing custom instantiations
 */
template <typename inputT, typename outputT>
static std::pair<Tensor<inputT>*, Tensor<outputT>*> get_tensorpool_io(TensorPool &pool, const Op::LayerBase *l) {
  if (pool.has_value(l->outputs.at(0))) {
    pool.free(l->outputs.at(0));
  }
  Tensor<inputT> *input = pool.get<Tensor<inputT> *>(l->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(l->output_dims.at(0), l->output_names.at(0));
  pool.set<Tensor<outputT> *>(l->outputs.at(0), output);
  return std::pair(input, output); 
}

template <typename T>
static void check_dispatch(const Op::LayerBase *l, const Tensor<T> *output) {
  if (l->dispatch) {
    pickle_tensor(output, l->name + ".tensor");
    if (get_verbose()) {
      output->print();
    }
  }
}

template <typename T>
static void run_noop(Op::LayerBase *l, TensorPool &tensor_pool) {
  Tensor<T> *input;
  Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  *output = std::move(*input);
  check_dispatch(l, output);
}

void Op::Layer::NoOp::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_INT32 &&
      output_type[0] == onnx::TensorProto_DataType_INT32) {
    run_noop<int32_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
             output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_noop<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             output_type[0] == onnx::TensorProto_DataType_INT8) {
    run_noop<int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8 &&
             output_type[0] == onnx::TensorProto_DataType_UINT8) {
    run_noop<uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

/* helper function for Op::Layer::Conv::run() */
template <typename inputT, typename weightT, typename outputT>
static void run_conv(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Conv *cc = dynamic_cast<Op::Layer::Conv *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  ConvEngine<inputT, weightT, outputT> cc_engine(cc);
  cc_engine.run(input, output);
  tt.stop();
  check_dispatch(l, output);
  if (get_verbose()) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::Conv::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_conv<float, float, float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

/* helper function for Op::Layer::Conv::run() */
template <typename T>
static void run_relu(Op::LayerBase *l, TensorPool &tensor_pool) {
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  Relu<T> relu;
  relu.exec(input, output);
  check_dispatch(l, output);
}

void Op::Layer::Relu::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_relu<float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_clip(Op::LayerBase *l, TensorPool &tensor_pool) {
  auto *cc = dynamic_cast<Op::Layer::Clip *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  Relu<T> relu(cc->m_max);
  relu.exec(input, output);
  check_dispatch(l, output);
}

void Op::Layer::Clip::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_clip<float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_maxpool(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Maxpool *cc = dynamic_cast<Op::Layer::Maxpool *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  maxpool<T>(input, output, cc->m_cp);
  check_dispatch(l, output);
}

void Op::Layer::Maxpool::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_maxpool<float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_flatten(Op::LayerBase *l, TensorPool &tensor_pool) {
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  flatten<T>(input, output);
  check_dispatch(l, output);
}

void Op::Layer::Flatten::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);
  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_flatten<float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename outputT>
static void run_gemm(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Gemm *cc = dynamic_cast<Op::Layer::Gemm *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  VA<inputT, inputT, inputT, outputT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, output);
  tt.stop();
  check_dispatch(l, output);
  if (l->dispatch) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::Gemm::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_gemm<float, float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_dropout(Op::LayerBase *l, TensorPool &tensor_pool) {
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  *output = *input;
  check_dispatch(l, output);
}

void Op::Layer::Dropout::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);
  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_dropout<float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_reshape(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Reshape *cc = dynamic_cast<Op::Layer::Reshape *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);

  int negative_ones =
      std::count(cc->new_shape.begin(), cc->new_shape.end(), -1);
  if (negative_ones > 1) {
    log_fatal("didn't expect more than one -1 in shape for node {}\n", l->name);
  }
  reshape<T>(input, output, cc->new_shape);
  check_dispatch(l, output);
}

void Op::Layer::Reshape::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_reshape<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8) {
    run_reshape<int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT32) {
    run_reshape<int>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8) {
    run_reshape<uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_transpose(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Transpose *cc = dynamic_cast<Op::Layer::Transpose *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  transpose<T>(input, output, cc->perm);
  check_dispatch(l, output);
}

void Op::Layer::Transpose::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_transpose<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8) {
    run_transpose<int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT32) {
    run_transpose<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

/* TODO: refactor to share this with gemm */
template <typename inputT, typename outputT>
static void run_matmul(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::MatMul *cc = dynamic_cast<Op::Layer::MatMul *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  VA<inputT, inputT, inputT, outputT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, output);
  tt.stop();
  check_dispatch(l, output);
  if (l->dispatch) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::MatMul::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_matmul<float, float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename outputT>
static void run_eltwise(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::Eltwise *cc = dynamic_cast<Op::Layer::Eltwise *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input1 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  std::vector<int> ofmap_dims{1, input1->dims_iterator(-1)};
  Tensor<outputT> *output = new TensorCreate<outputT>(ofmap_dims);
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);
  Tensor<inputT> *input2;
  if (cc->inputs.size() > 1) {
    // both inputs are non-initializers (i.e. available only at runtime)
    input2 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(1));
    tensor_eltwise(output, input1, input2, cc->operator_type);
  } else {
    // one of the inputs is an initializer (available statically)
    input2 = new TensorExtant<inputT>(cc->constant_data);
    tensor_eltwise(output, input1, input2, cc->operator_type);
    delete input2;
  }
  check_dispatch(l, output);
}

void Op::Layer::Eltwise::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_eltwise<float, float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             output_type[0] == onnx::TensorProto_DataType_INT32) {
    run_eltwise<int8_t, int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename outputT>
static void run_quantize_linear(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QuantizeLinear *cc = dynamic_cast<Op::Layer::QuantizeLinear *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  std::vector<float> scales{cc->scale};
  std::vector<int> zero_point;
  if (std::holds_alternative<uint8_t>(cc->zero_point)) {
    zero_point.push_back((int)std::get<uint8_t>(cc->zero_point));
  } else if (std::holds_alternative<int8_t>(cc->zero_point)) {
    zero_point.push_back((int)std::get<int8_t>(cc->zero_point));
  } else {
    log_fatal("cant deduce zero point type for layer {}\n", l->name);
  }
  Timer<std::chrono::microseconds> tt;
  tt.start();
  quantize<inputT, outputT>(input, output, scales, zero_point);
  tt.stop();
  if (get_verbose()) {
    log_info("Quantize linear time: {} us\n", tt.difference().count());
  }
  check_dispatch(l, output);
}

void Op::Layer::QuantizeLinear::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_UINT8) {
    run_quantize_linear<float, uint8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
             output_type[0] == onnx::TensorProto_DataType_INT8) {
    run_quantize_linear<float, int8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename weightT, typename intrT, typename outputT>
static void run_qconv(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearConv *cc = dynamic_cast<Op::Layer::QLinearConv *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);

  std::unique_ptr<Tensor<intrT>> intr_output{
      new TensorCreate<intrT>(cc->output_dims[0])};
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  ConvEngine<inputT, weightT, intrT> cc_engine(cc);
  cc_engine.run(input, intr_output.get());

  if (l->output_type[0] == onnx::TensorProto_DataType_INT32) {
    auto it_out = output->begin();
    for (auto it_in = intr_output->begin(); it_in != intr_output->end();
         ++it_in, ++it_out) {
      *it_out = static_cast<outputT>(*it_in);
    }
  } else {
    std::vector<float> scales =
        compute_output_scale(cc->x_scale, cc->w_scale, cc->y_scale);
    using variantT = std::variant<int8_t, uint8_t>;
    std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
    quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);
  }
  tt.stop();
  check_dispatch(l, output);
  if (l->dispatch) {
    pickle_tensor(intr_output.get(), l->name + "_32bit_acc" + ".tensor");
  }
  if (get_verbose()) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::QLinearConv::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_UINT8 &&
             weight_type == onnx::TensorProto_DataType_UINT8) {
    run_qconv<uint8_t, uint8_t, int, uint8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             output_type[0] == onnx::TensorProto_DataType_INT32) {
    run_qconv<int8_t, int8_t, int, int32_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_INT8) {
    run_qconv<int8_t, int8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8 &&
             weight_type == onnx::TensorProto_DataType_INT8) {
    run_qconv<uint8_t, int8_t, int, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: %s, %s",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename outputT>
static void run_dequantize_linear(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::DequantizeLinear *cc =
      dynamic_cast<Op::Layer::DequantizeLinear *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  std::vector<int> zero_point{cc->zero_point};
  std::vector<float> scales;
  if (std::holds_alternative<float>(cc->scale)) {
    scales.push_back((float)std::get<float>(cc->scale));
  } else if (std::holds_alternative<double>(cc->scale)) {
    log_info("converting scale from double to float for layer {}", l->name);
    scales.push_back((float)std::get<double>(cc->scale));
  } else {
    log_fatal("cant deduce zero point type for layer {}", l->name);
  }
  dequantize<inputT, outputT>(input, output, scales, zero_point);
  check_dispatch(l, output);
}

void Op::Layer::DequantizeLinear::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_UINT8 &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_dequantize_linear<uint8_t, float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_dequantize_linear<int8_t, float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename weightT, typename intrT, typename outputT>
static void run_qmatmul(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearMatMul *cc = dynamic_cast<Op::Layer::QLinearMatMul *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  std::unique_ptr<Tensor<intrT>> intr_output{
      new TensorCreate<intrT>(cc->output_dims[0])};

  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);

  VA<inputT, weightT, weightT, intrT> va(*cc);
  /* TODO: get architecture size from gbl_args */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  va.run(input, intr_output.get());
  std::vector<float> scales =
      compute_output_scale(cc->a_scale, cc->b_scale, cc->y_scale);
  quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);
  tt.stop();
  check_dispatch(l, output);
  if (l->dispatch) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::QLinearMatMul::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_INT8) {
    run_qmatmul<int8_t, int8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_UINT8) {
    run_qmatmul<int8_t, uint8_t, int, int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8 &&
             weight_type == onnx::TensorProto_DataType_INT8) {
    run_qmatmul<uint8_t, int8_t, int, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename intrT, typename outputT>
static void run_qeltwise(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearEltwise *cc = dynamic_cast<Op::Layer::QLinearEltwise *>(l);
  if (tensor_pool.has_value(cc->outputs.at(0))) {
    tensor_pool.free(cc->outputs.at(0));
  }
  Tensor<inputT> *input1 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(0));
  Tensor<outputT> *output = new TensorCreate<outputT>(cc->output_dims[0], cc->output_names.at(0));
  tensor_pool.set<Tensor<outputT> *>(cc->outputs.at(0), output);
  std::unique_ptr<Tensor<intrT>> intr_output{new TensorCreate<intrT>(cc->output_dims.at(0))};
  Tensor<inputT> *input2;

  if constexpr (std::is_same<inputT, int32_t>()) {
    input2 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(1));
    tensor_qeltwise(intr_output.get(), input1, input2, 1, 1, 0, 0,
                    cc->operator_type);
    if constexpr (std::is_same<outputT, int8_t>()) {
      std::vector<float> x_scale {cc->a_scale};
      std::vector<float> w_scale {cc->b_scale};
      std::vector<float> scales = compute_output_scale(x_scale, w_scale, cc->o_scale);
      using variantT = std::variant<int8_t, uint8_t>;
      std::vector<int> zero_points = variant2vec<variantT, int>(cc->zero_point);
      quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);
    } else {
      auto it_out = output->begin();
      for (auto it_in = intr_output->begin(); it_in != intr_output->end();
           ++it_in, ++it_out) {
        *it_out = static_cast<outputT>(*it_in);
      }
    }
  } else {
    if (cc->inputs.size() > 1) {
      // both inputs are non-initializers (i.e. available only at runtime)
      input2 = tensor_pool.get<Tensor<inputT> *>(cc->inputs.at(1));
      if (input1->name() == cc->input_names.at(0 /* QLE_A */) &&
          input2->name() == cc->input_names.at(3 /* QLE_B */)) {
        tensor_qeltwise(intr_output.get(), input1, input2, cc->a_scale,
                        cc->b_scale, cc->a_zp, cc->b_zp, cc->operator_type);
      } else {
        tensor_qeltwise(intr_output.get(), input2, input1, cc->a_scale,
                        cc->b_scale, cc->a_zp, cc->b_zp, cc->operator_type);
      }
    } else {
      // one of the inputs is an initializer (available statically)
      input2 = new TensorExtant<inputT>(cc->constant_data);
      tensor_qeltwise(intr_output.get(), input1, input2, cc->a_scale,
                      cc->b_scale, cc->a_zp, cc->b_zp, cc->operator_type);
      delete input2;
    }
    using variantT = std::variant<int8_t, uint8_t>;
    std::vector<int> zero_points = variant2vec<variantT, int>(cc->zero_point);
    quantize<intrT, outputT>(intr_output.get(), output, cc->o_scale,
                             zero_points);
  }
  check_dispatch(l, output);
}

void Op::Layer::QLinearEltwise::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_qeltwise<float, float, float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8) {
    run_qeltwise<int8_t, fp_t, int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8) {
    run_qeltwise<uint8_t, fp_t, uint8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT32 &&
             output_type[0] == onnx::TensorProto_DataType_INT32) {
    run_qeltwise<int32_t, int32_t, int32_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT32 &&
             output_type[0] == onnx::TensorProto_DataType_INT8) {
    run_qeltwise<int32_t, int32_t, int8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename weightT, typename intrT, typename outputT>
static void run_qgemm(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QGemm *cc = dynamic_cast<Op::Layer::QGemm *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  std::unique_ptr<Tensor<intrT>> intr_output{
      new TensorCreate<intrT>(cc->output_dims[0])};
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  VA<inputT, weightT, int32_t, intrT> va(*cc);
  va.run(input, intr_output.get());
  std::vector<float> scales =
      compute_output_scale(cc->a_scale, cc->b_scale, cc->y_scale);
  quantize<intrT, outputT>(intr_output.get(), output, scales, zero_points);

  tt.stop();
  check_dispatch(l, output);
  if (get_verbose()) {
    tt.report("Time taken: ");
  }
}

void Op::Layer::QGemm::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  if (input_type[0] == onnx::TensorProto_DataType_INT8 &&
             weight_type == onnx::TensorProto_DataType_INT8 &&
             bias_type == onnx::TensorProto_DataType_INT32) {
    run_qgemm<int8_t, int8_t, int32_t, int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8 &&
             weight_type == onnx::TensorProto_DataType_UINT8 &&
             bias_type == onnx::TensorProto_DataType_INT32) {
    run_qgemm<uint8_t, uint8_t, int32_t, uint8_t>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename inputT, typename outputT>
static void run_logsoftmax(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::LogSoftmax *cc = dynamic_cast<Op::Layer::LogSoftmax *>(l);
  Tensor<inputT> *input; Tensor<outputT> *output;
  std::tie(input, output) = get_tensorpool_io<inputT, outputT>(tensor_pool, l);
  logsoftmax(output, input, cc->axis);
  check_dispatch(l, output);
}

void Op::Layer::LogSoftmax::run(TensorPool &tensor_pool) {
  if (input_type[0] == onnx::TensorProto_DataType_UNDEFINED) {
    log_fatal("input_type[0] for layer {} UNDEFINED", this->name);
  }
  if (output_type[0] == onnx::TensorProto_DataType_UNDEFINED) {
    log_fatal("output_type[0] for layer {} UNDEFINED", this->name);
  }

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT &&
      output_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_logsoftmax<float, float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_DOUBLE &&
             output_type[0] == onnx::TensorProto_DataType_DOUBLE) {
    run_logsoftmax<double, double>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_qlinearaveragepool(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::QLinearAveragePool *cc =
      dynamic_cast<Op::Layer::QLinearAveragePool *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  average_pool<T>(input, output, cc->m_cp);
  check_dispatch(l, output);
}

void Op::Layer::QLinearAveragePool::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_INT8) {
    run_qlinearaveragepool<int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_UINT8) {
    run_qlinearaveragepool<uint8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_qlinearaveragepool<float>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_batchnorm(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::BatchNorm *cc = dynamic_cast<Op::Layer::BatchNorm *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  std::unique_ptr<Tensor<T>> scale{new TensorExtant<T>(cc->scale)};
  std::unique_ptr<Tensor<T>> bias{new TensorExtant<T>(cc->B)};
  std::unique_ptr<Tensor<T>> mean{new TensorExtant<T>(cc->mean)};
  std::unique_ptr<Tensor<T>> var{new TensorExtant<T>(cc->var)};
  batchnorm<T>(input, output, cc->epsilon, cc->momentum, scale.get(),
               bias.get(), mean.get(), var.get());
  check_dispatch(l, output);
}

void Op::Layer::BatchNorm::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_batchnorm<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_DOUBLE) {
    run_batchnorm<double>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}

template <typename T>
static void run_abs(Op::LayerBase *l, TensorPool &tensor_pool) {
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  xabs<T>(input, output);
  check_dispatch(l, output);
}

void Op::Layer::Abs::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_abs<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_DOUBLE) {
    run_abs<double>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8) {
    run_abs<int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT32) {
    run_abs<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type : {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]));
  }
}

template <typename T>
static void run_reduce_mean(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::ReduceMean *cc = dynamic_cast<Op::Layer::ReduceMean *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  reduce_mean<T>(input, output, cc->m_axis, cc->m_keepdims);
  check_dispatch(l, output);
}

void Op::Layer::ReduceMean::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_reduce_mean<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_DOUBLE) {
    run_reduce_mean<double>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT8) {
    run_reduce_mean<int8_t>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_INT32) {
    run_reduce_mean<int>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type : {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]));
  }
}

template <typename T>
static void run_averagepool(Op::LayerBase *l, TensorPool &tensor_pool) {
  Op::Layer::AveragePool *cc = dynamic_cast<Op::Layer::AveragePool *>(l);
  Tensor<T> *input; Tensor<T> *output;
  std::tie(input, output) = get_tensorpool_io<T, T>(tensor_pool, l);
  average_pool<T>(input, output, cc->m_cp);
  check_dispatch(l, output);
}

void Op::Layer::AveragePool::run(TensorPool &tensor_pool) {
  assert(input_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(output_type[0] != onnx::TensorProto_DataType_UNDEFINED);
  assert(input_type[0] == output_type[0]);

  if (input_type[0] == onnx::TensorProto_DataType_FLOAT) {
    run_averagepool<float>(this, tensor_pool);
  } else if (input_type[0] == onnx::TensorProto_DataType_DOUBLE) {
    run_averagepool<double>(this, tensor_pool);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type[0]),
              Op::get_tensorproto_dtype_name(output_type[0]));
  }
}
