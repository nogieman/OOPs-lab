#include "onnx_parser.h"
#include "utils.h"
#include <queue>

#define CONV_WEIGHT_TENSOR_DIMS 4
#define GEMM_WEIGHT_TENSOR_DIMS 2
#define BIAS_TENSOR_DIMS 1

/* An onnx file contains most importantly a list of:
 *
 * [nodes]        // deescription of the network
 * [initializers] // weights/kernels for layers that require it
 * [value_infos]  // contains shape information
 *
 * this has to be read from the onnx file and populate
 * the Op::Model graph. The graph polymorphically
 * contains nodes that correspond to layers found
 * in ML models. Vertices in the graph are layers,
 * edges are connections b/w the layers. Weighted
 * layers contain pointers to TensorProto objects
 * in the onnx file.
 */

const char *Op::LayerBase::op_type() const { return "(null)"; }
std::string Op::LayerBase::params() const { return "(null)"; }
void Op::LayerBase::set_initializer_params(int n, const onnx::TensorProto &t) {
  ignore_unused(n);
  ignore_unused(t);
}
void Op::LayerBase::set_value_info_params(const onnx::ValueInfoProto &t) {
  ignore_unused(t);
}
void Op::LayerBase::run(TensorPool &tensor_pool) {
  ignore_unused(tensor_pool);
  log_fatal("No overrides present for this layer {}: {}\n", this->op_type(),
            name);
}
void Op::LayerBase::set_attributes(const onnx::NodeProto &node) {
  ignore_unused(node);
  return;
}

void Op::LayerBase::set_constant_params(int n, const onnx::NodeProto &) {
  return;
}

void Op::LayerBase::infer_shape(const IVec2D &input_dims) {
  ignore_unused(input_dims);
  log_fatal("Shape Inference Un-implemented for this layer {}: {}\n",
            this->op_type(), this->name);
}

void Op::LayerBase::infer_type(const std::vector<TPDT> &input_types) {
  ignore_unused(input_types);
  log_fatal("Type inference un-implemented for this layer {}: {}\n",
            this->op_type(), this->name);
}

int Op::LayerBase::get_inst(InstBlob &insts, AddressGen &gen,
                            InitializerTable &tbl) {
  ignore_unused(insts);
  ignore_unused(gen);
  ignore_unused(tbl);
  log_fatal("Instruction generation un-implemented for this layer {}: {}\n",
            this->op_type(), this->name);
}

void Op::LayerBase::get_opcodes(std::vector<int> &op_codes) {
  ignore_unused(op_codes);
  log_fatal("Opcode generation un-implemented for this layer {}: {}\n",
            this->op_type(), this->name);
}

uint32_t Op::LayerBase::get_weight_size() {
  log_fatal("Weight size un-implemented for this layer {}: {}\n",
            this->op_type(), this->name);
}

void Op::LayerBase::align_weights(BinBlob &, InitializerTable &) {
  return;
}

std::pair<int,int> Op::LayerBase::get_iterations() const {
  log_warn("get_iterations() not implemented for this layer {}\n", this->name);
  return std::pair(0,0);
}

std::vector<float> Op::LayerBase::get_output_scale(void) {
  return std::vector<float>{0.f};
}

void Op::LayerBase::set_output_scale(const std::vector<float>& ) {
}

/* Get a array of ints from attr and store into array */
static void parse_onnx_ints(const onnx::AttributeProto &attr, int *attr_array) {
  assert(attr.type() == onnx::AttributeProto::INTS &&
         "expected attributes of type INTS");
  auto ints = attr.ints();
  for (int i = 0; i < ints.size(); ++i) {
    attr_array[i] = ints.at(i);
  }
}

static void parse_onnx_ints(const onnx::AttributeProto &attr,
                            std::vector<int> &attr_array) {
  assert(attr.type() == onnx::AttributeProto::INTS &&
         "expected attributes of type INTS");
  auto ints = attr.ints();
  for (int i = 0; i < ints.size(); ++i) {
    attr_array.push_back(ints.at(i));
  }
}

const char *Op::Layer::NoOp::op_type() const { return m_optype; }
Op::Layer::NoOp::NoOp() {}

Op::Layer::Conv::Conv() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT] = 1;
  m_cp.stride[TENSOR_2D_WIDTH] = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH] = 1;
  weights = nullptr;
  bias = nullptr;
}

const char *Op::Layer::Conv::op_type() const { return m_optype; }
std::string Op::Layer::Conv::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "(IW,IH: " << this->input_dims[TENSOR_4D_WIDTH] << ","
     << this->input_dims[TENSOR_4D_HEIGHT] << ") "
     << "(KN,IC,KH,KW: " << m_cp.kn << ","
     << this->input_dims[TENSOR_4D_CHANNELS] << "," << m_cp.k[TENSOR_2D_WIDTH]
     << "," << m_cp.k[TENSOR_2D_HEIGHT] << ") "
     << "(S,P,D: " << m_cp.stride[TENSOR_2D_WIDTH] << "," << m_cp.pad[I_LEFT]
     << "," << m_cp.dilation[TENSOR_2D_WIDTH] << ") ";
  ret = ss.str();
  return ret;
}

void Op::Layer::Conv::set_initializer_params(int n,
                                             const onnx::TensorProto &t) {
  ignore_unused(n);
  /* TODO: use n instead of relying on DIMS */
  if (t.dims_size() == CONV_WEIGHT_TENSOR_DIMS) {
    m_cp.kn = t.dims()[0];
    m_cp.k[0] = t.dims()[2];
    m_cp.k[1] = t.dims()[3];
    weights = &t;
  } else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;
  }
}

void Op::Layer::Conv::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "dilations") {
      assert(itr->ints().size() == 2 && "expected dilations to be 2 integers");
      parse_onnx_ints(*itr, m_cp.dilation);
    }
  }
}

void Op::Layer::Conv::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  assert(input_dims[0].size() == 4); // NCHW
  this->output_dims.resize(1);
  this->output_dims[0].resize(4);
  this->output_dims[0][0] = input_dims[0][0];
  this->output_dims[0][1] = this->m_cp.kn;
  this->output_dims[0][2] = sa_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[0][3] = sa_odims_cols(this->m_cp, input_dims[0]);
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Conv::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

const char *Op::Layer::Relu::op_type() const { return m_optype; }

void Op::Layer::Relu::infer_shape(const IVec2D &input_dims) {
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = input_dims;
}

void Op::Layer::Relu::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

Op::Layer::Clip::Clip() {
  /* defaults */
  m_min = INT_MIN;
  m_max = INT_MAX;
}
const char *Op::Layer::Clip::op_type() const { return m_optype; }
std::string Op::Layer::Clip::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "Clip: (" << m_min << ", " << m_max << ")";
  ret = ss.str();
  return ret;
}

void Op::Layer::Clip::set_attributes(const onnx::NodeProto &node) {
  /* TODO: this */
  if (node.op_type() == "Constant") {
    for (auto i : node.input()) {
      std::cout << "input " << i << '\n';
    }
    for (auto i : node.output()) {
      std::cout << "output " << i << '\n';
    }
    std::cout << '\n';
  }
}

enum CLIP_INITIALIZERS { CLIP_MIN = 1, CLIP_MAX = 2 };

void Op::Layer::Clip::set_constant_params(int n, const onnx::NodeProto &node) {
  float val = 0.0f;
  for (const auto& a : node.attribute()) {
    if (a.name() == "value") {
      const onnx::TensorProto &t = a.t();
      if (t.float_data_size() > 0) {
        val = t.float_data(0);
      } else if (!t.raw_data().empty()) {
        std::memcpy(&val, t.raw_data().data(), sizeof(float));
      }
    }
  }
  m_min = (n == CLIP_MIN) ? val : m_min; 
  m_max = (n == CLIP_MAX) ? val : m_max;
}

void Op::Layer::Clip::infer_shape(const IVec2D &input_dims) {
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = input_dims;
}
void Op::Layer::Clip::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}


void Op::Layer::Clip::set_initializer_params(int n,
                                             const onnx::TensorProto &t) {
  float val = 0.0f;
  if (t.float_data_size() > 0) {
    val = t.float_data(0);
  } else if (!t.raw_data().empty()) {
    std::memcpy(&val, t.raw_data().data(), sizeof(float));
  }

  switch (n) {
  case CLIP_MIN:
    this->m_min = val;
    break;
  case CLIP_MAX:
    this->m_max = val;
    break;
  }
}

Op::Layer::Gemm::Gemm() {
  m_cp = {};
  m_cp.alpha = 1.0;
  m_cp.beta = 1.0;
  m_cp.transA = 0;
  m_cp.transB = 0;
}
const char *Op::Layer::Gemm::op_type() const { return m_optype; }
std::string Op::Layer::Gemm::params() const {
  static std::string ret;
  std::stringstream ss;
  ss << "IH,IW,WR,WC: " << this->input_dims[0][TENSOR_2D_HEIGHT] << ","
     << this->input_dims[0][TENSOR_2D_WIDTH] << "," << m_cp.wr << "," << m_cp.wc
     << " alpha,beta,transA,transB: " << m_cp.alpha << "," << m_cp.beta << ","
     << m_cp.transA << "," << m_cp.transB;
  ret = ss.str();
  return ret;
}

void Op::Layer::Gemm::set_initializer_params(int n,
                                             const onnx::TensorProto &t) {
  ignore_unused(n);
  /* use n instead of this */
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  } else if (t.dims_size() == BIAS_TENSOR_DIMS) {
    bias = &t;
  }
}

void Op::Layer::Gemm::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "alpha") {
      if (itr->has_f()) {
        m_cp.alpha = itr->f();
      }
    } else if (itr->name() == "beta") {
      if (itr->has_f()) {
        m_cp.beta = itr->f();
      }
    } else if (itr->name() == "transA") {
      if (itr->has_i()) {
        m_cp.transA = itr->i();
      }
    } else if (itr->name() == "transB") {
      if (itr->has_i()) {
        m_cp.transB = itr->i();
      }
    }
  }
}

void Op::Layer::Gemm::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  assert(input_dims[0].size() == 2);
  this->input_dims = input_dims;
  this->output_dims.resize(1);
  this->output_dims[0].resize(2);
  this->output_dims[0].at(0) = input_dims[0].at(0);
  if (m_cp.transB) {
    assert(input_dims[0].at(1) == this->m_cp.wc &&
           "Gemm, matrix dimensions do not match");
    this->output_dims[0].at(1) = this->m_cp.wr;
  } else {
    assert(input_dims[0].at(1) == this->m_cp.wr &&
           "Gemm, matrix dimensions do not match");
    this->output_dims[0].at(1) = this->m_cp.wc;
  }
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Gemm::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

static std::stringstream get_pool_params(const std::vector<int> &idims,
                                         const Op::PoolParams &m_cp) {
  std::stringstream ss;
  ss << "(IC,IW,IH: " << idims[TENSOR_4D_CHANNELS] << ","
     << idims[TENSOR_4D_WIDTH] << "," << idims[TENSOR_4D_HEIGHT] << ") "
     << "(KS: " << m_cp.k[TENSOR_2D_HEIGHT] << "," << m_cp.k[TENSOR_2D_WIDTH]
     << ") "
     << "(pad: " << m_cp.pad[I_LEFT] << "," << m_cp.pad[I_UP] << ","
     << m_cp.pad[I_RIGHT] << "," << m_cp.pad[I_DOWN] << ") "
     << "(stride: " << m_cp.stride[TENSOR_2D_HEIGHT] << ","
     << m_cp.stride[TENSOR_2D_WIDTH] << ") "
     << "(dilation: " << m_cp.dilation[TENSOR_2D_WIDTH] << ","
     << m_cp.dilation[TENSOR_2D_HEIGHT] << ")";
  return ss;
}

Op::Layer::Maxpool::Maxpool() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT] = 1;
  m_cp.stride[TENSOR_2D_WIDTH] = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH] = 1;
}

const char *Op::Layer::Maxpool::op_type() const { return m_optype; }
std::string Op::Layer::Maxpool::params() const {
  return get_pool_params(this->input_dims[0], this->m_cp).str();
}

void Op::Layer::Maxpool::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "dilations") {
      assert(itr->ints().size() == 2 && "expected dilations to be 2 integers");
      parse_onnx_ints(*itr, m_cp.dilation);
    } else if (itr->name() == "ceil_mode") {
      if (itr->has_i()) {
        int ceil_mode = static_cast<int>(itr->i());
        if (ceil_mode != 0) {
          log_fatal("Unsupported ceil_mode {} in layer {}\n", ceil_mode,
                    node.name());
        }
      }
    } else if (itr->name() == "storage_order") {
      if (itr->has_i()) {
        int order = static_cast<int>(itr->i());
        if (order != 0) {
          log_fatal("Unsupported storage_order {} in layer {}\n", order,
                    node.name());
        }
      }
    }
  }
}

void Op::Layer::Maxpool::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  assert(input_dims[0].size() == 4);
  this->output_dims.resize(1);
  this->output_dims[0].resize(4);
  this->output_dims[0][0] = input_dims[0][0];
  this->output_dims[0][1] = input_dims[0][1];
  this->output_dims[0][2] = mp_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[0][3] = mp_odims_cols(this->m_cp, input_dims[0]);
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Maxpool::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

const char *Op::Layer::Flatten::op_type() const { return m_optype; }

void Op::Layer::Flatten::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  int total_elements = prod(input_dims[0].begin(), input_dims[0].end(), 1);
  this->output_dims.resize(1);
  this->output_dims[0].resize(2);
  this->output_dims[0].at(0) = 1;
  this->output_dims[0].at(1) = total_elements;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Flatten::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

Op::Layer::Dropout::Dropout() { drop = 0.f; }
const char *Op::Layer::Dropout::op_type() const { return m_optype; }
std::string Op::Layer::Dropout::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "Drop: " << drop;
  ret = ss.str();
  return ret;
}

void Op::Layer::Dropout::set_initializer_params(int,
                                                const onnx::TensorProto &t) {
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    this->drop = t.float_data()[0];
  }
}

void Op::Layer::Dropout::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Dropout::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

Op::Layer::Eltwise::Eltwise(int op) : operator_type(op) {
  constant_data = nullptr;
}

const char *Op::Layer::Eltwise::op_type() const { return m_optype; }

void Op::Layer::Eltwise::set_initializer_params(int, const onnx::TensorProto &t) {
  constant_data = &t;
}

void Op::Layer::Eltwise::infer_shape(const IVec2D &input_dims) {
  /* TODO: allow support for broadcasts */
  assert(input_dims.size() >= 1);
  auto og = input_dims[0];
  /* all inputs should be equal to the first input in size */
  // auto compare_fn = [&og](const std::vector<int> &v) {
  //   assert(v == og); };
  // std::for_each(input_dims.begin(), input_dims.end(), compare_fn);
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Eltwise::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}


const char *Op::Layer::BatchNorm::op_type() const { return m_optype; }

std::string Op::Layer::BatchNorm::params() const {
  std::stringstream ss;
  ss << "momentum: " << momentum << " epsilon: " << epsilon;
  return ss.str();
}

void Op::Layer::BatchNorm::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::BatchNorm::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

enum BN_INITIALIZERS {
  BN_X = 0,
  BN_SCALE = 1,
  BN_BIAS = 2,
  BN_INPUT_MEAN = 3,
  BN_INPUT_VAR = 4,
};

void Op::Layer::BatchNorm::set_initializer_params(int n,
                                                  const onnx::TensorProto &t) {
  switch (n) {
  case BN_X:
    break;
  case BN_SCALE:
    this->scale = &t;
    break;
  case BN_BIAS:
    this->B = &t;
    break;
  case BN_INPUT_MEAN:
    this->mean = &t;
    break;
  case BN_INPUT_VAR:
    this->var = &t;
    break;
  default:
    log_fatal("unknown initializer for layer {}\n", this->name);
    break;
  }
}

void Op::Layer::BatchNorm::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "epsilon") {
      if (itr->has_f()) {
        epsilon = itr->f();
      }
    } else if (itr->name() == "momentum") {
      if (itr->has_f()) {
        momentum = itr->f();
      }
    } else if (itr->name() == "training_mode") {
      if (itr->has_i()) {
        if (itr->i() == 1) {
          log_fatal("In node {}, training_mode = 1 is not supported\n",
                    node.name());
        }
      }
    }
  }
}

const char *Op::Layer::ReorderOutput::op_type() const { return m_optype; }

const char *Op::Layer::Reshape::op_type() const { return m_optype; }

std::string Op::Layer::Reshape::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "(shape: ";
  for (int64_t i : new_shape) {
    ss << i << ", ";
  }
  ss << ")";
  ret = ss.str();
  return ret;
}

void Op::Layer::Reshape::set_initializer_params(int,
                                                const onnx::TensorProto &t) {
  if (t.dims_size() != 1) {
    log_fatal(
        "New shape expected to be a linear vector, got vector of size {} for\n"
        " tensor {}",
        t.dims_size(), t.name());
  }

  if (t.int64_data_size() > 0) {
    for (int64_t val : t.int32_data()) {
      new_shape.push_back(val);
    }
  } else if (t.has_raw_data()) {
    /* oddly enough, protobuf uses std::string to hold bytes, hence
     * the need for reinterpret_cast
     */
    const int64_t *raw_ptr =
        reinterpret_cast<const int64_t *>(t.raw_data().c_str());
    for (int i = 0; i < t.dims(0); ++i) {
      new_shape.push_back(raw_ptr[i]);
    }
  } else {
    log_fatal("Do not know how to interpret TensorProto for {}\n", t.name());
  }
}

/* Deduces and removes -1/0 from old_shape to return
 * a correct new_shape.
 * See https://onnx.ai/onnx/operators/onnx__Reshape.html#reshape
 *
 * TODO: handle 0s in shape (does not do it presently)
 */
std::vector<int> deduce_new_shape(std::vector<int> old_shape,
                                      int input_total_size) {
  auto itr = std::find(old_shape.begin(), old_shape.end(), -1);
  if (itr != old_shape.end()) {
    int remaining_size = std::abs(prod(old_shape.begin(), old_shape.end(), 1));
    assert(input_total_size % remaining_size == 0 &&
           "unable to deduce new shape");
    int remaining_dim = input_total_size / remaining_size;
    *itr = remaining_dim;
  }
  return old_shape;
}

void Op::Layer::Reshape::infer_shape(const IVec2D &input_dims) {
  this->input_dims = input_dims;
  if (new_shape.size() > 0) {
    this->output_dims.push_back(deduce_new_shape(new_shape, prod(input_dims.at(0))));
  } else {
    this->output_dims = input_dims;
  }
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Reshape::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  if (this->input_type.size() == 0) {
    this->input_type.resize(input_types.size());
  }
  for (size_t i = 0; i < input_types.size(); ++i) {
    if (input_types[i] != onnx::TensorProto_DataType_UNDEFINED) {
      this->input_type[i] = input_types[i];
    }
  }
  if (input_types.at(0) != onnx::TensorProto_DataType_UNDEFINED) {
    this->output_type.push_back(input_types.at(0));
  }
}

Op::Layer::DequantizeLinear::DequantizeLinear()
    : scale{0.0}, zero_point{0}, axis{0}, block_size{0} {}

const char *Op::Layer::DequantizeLinear::op_type() const { return m_optype; }

std::string Op::Layer::DequantizeLinear::params() const {
  std::string ret;
  std::stringstream ss;
  if (std::holds_alternative<float>(scale)) {
    ss << "Scale: " << std::get<float>(scale) << ", Zero Point: " << zero_point;
  } else if (std::holds_alternative<double>(scale)) {
    ss << "Scale: " << std::get<double>(scale)
       << ", Zero Point: " << zero_point;
  } else {
    log_fatal("cannot format zero point of unknown type for layer {}\n",
              this->name);
  }
  ret = ss.str();
  return ret;
}

void Op::Layer::DequantizeLinear::set_initializer_params(
    int, const onnx::TensorProto &t) {
  /* TODO: use n */
  if (t.data_type() == onnx::TensorProto_DataType_FLOAT) {
    /* its a scale value */
    scale = (float)t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_DOUBLE) {
    scale = (double)t.float_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
    zero_point = t.int32_data(0);
  } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
    zero_point = t.int32_data(0);
  } else {
    log_fatal("Could not find an initializer of expected types\n");
  }
}

void Op::Layer::DequantizeLinear::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = {}\n", itr->i());
        }
        axis = itr->i();
      }
    } else if (itr->name() == "block_size") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = {}\n", itr->i());
        }
        block_size = itr->i();
      }
    }
  }
}

void Op::Layer::DequantizeLinear::infer_type(
    const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  if (std::holds_alternative<float>(this->scale)) {
    this->output_type.push_back(onnx::TensorProto_DataType_FLOAT);
  } else if (std::holds_alternative<double>(this->scale)) {
    this->output_type.push_back(onnx::TensorProto_DataType_DOUBLE);
  } else {
    log_fatal("could not deduce output type for layer {}\n", this->name);
  }
}

void Op::Layer::DequantizeLinear::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

std::vector<float> Op::Layer::DequantizeLinear::get_output_scale(void) {
  if (std::holds_alternative<float>(this->scale)) {
    return std::vector<float>{std::get<float>(this->scale)};
  } else if (std::holds_alternative<double>(this->scale)) {
    return std::vector<float>{static_cast<float>(std::get<double>(this->scale))};
  } else {
    log_fatal("scale variant of {} holds an unhandled type of data\n", this->name);
  }
}

void Op::Layer::DequantizeLinear::set_output_scale(const std::vector<float>& v) {
  assert(v.size() > 0 && "Input vector (v) to DequantizeLinear::set_output_scale expected to contain atleast one value");
  this->scale = v.at(0);
}

const char *Op::Layer::QuantizeLinear::op_type() const { return m_optype; }

std::string Op::Layer::QuantizeLinear::params() const {
  std::string ret;
  std::stringstream ss;
  if (std::holds_alternative<int8_t>(zero_point)) {
    ss << "Scale: " << scale
       << ", Zero Point: " << (int)std::get<int8_t>(zero_point);
  } else if (std::holds_alternative<uint8_t>(zero_point)) {
    ss << "Scale: " << scale
       << ", Zero Point: " << (int)std::get<uint8_t>(zero_point);
  } else {
    log_fatal("cannot format zero point of unknown type for layer {}\n",
              this->name);
  }
  ret = ss.str();
  return ret;
}

enum QL_INITIALIZERS {
  QL_X = 0,
  QL_YSCALE = 1,
  QL_YZP = 2,
};

void Op::Layer::QuantizeLinear::set_initializer_params(
    int n, const onnx::TensorProto &t) {
  switch (n) {
  case QL_X:
    break;
  case QL_YSCALE:
    if (t.float_data_size() > 0) {
      scale = t.float_data(0);
    } else if (t.double_data_size() > 0) {
      scale = static_cast<float>(t.double_data(0));
    } else {
      log_fatal("cant deduce the type of QL_YSCALE for tensor {}\n", t.name());
    }
    break;
  case QL_YZP:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      zero_point = (uint8_t)t.int32_data(0);
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      zero_point = (int8_t)t.int32_data(0);
    } else {
      log_fatal("cant deduce QL_YZP for tensor {}\n", t.name());
    }
    break;
  default:
    log_fatal(
        "nth ({}): Too many or too little initializers to QuantizeLinear\n", n);
  };
}

Op::Layer::QuantizeLinear::QuantizeLinear()
    : scale{1.0}, axis{0}, block_size{0}, output_dtype{0}, saturate{1} {}

void Op::Layer::QuantizeLinear::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::QuantizeLinear::infer_type(
    const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  if (std::holds_alternative<int8_t>(this->zero_point)) {
    this->output_type.push_back(onnx::TensorProto_DataType_INT8);
  } else if (std::holds_alternative<uint8_t>(this->zero_point)) {
    this->output_type.push_back(onnx::TensorProto_DataType_UINT8);
  } else {
    log_fatal("could not deduce output type for layer {}\n", this->name);
  }
}

void Op::Layer::QuantizeLinear::set_attributes(const onnx::NodeProto &node) {
  auto attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = {}\n", itr->i());
        }
        axis = itr->i();
      }
    } else if (itr->name() == "block_size") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = {}\n", itr->i());
        }
        block_size = itr->i();
      }
    } else if (itr->name() == "output_dtype") {
      if (itr->has_i()) {
        if (itr->i() != 0) {
          log_fatal("axes != 0 are unsupported, axis = {}\n", itr->i());
        }
        output_dtype = itr->i();
      }
    } else if (itr->name() == "saturate") {
      if (itr->has_i()) {
        if (itr->i() != 1) {
          log_fatal("axes != 0 are unsupported, axis = {}\n", itr->i());
        }
        saturate = itr->i();
      }
    }
  }
}

std::vector<float> Op::Layer::QuantizeLinear::get_output_scale(void) {
  return std::vector<float>{this->scale};
}

void Op::Layer::QuantizeLinear::set_output_scale(const std::vector<float>& v) {
  assert(v.size() > 0 && "Input vector (v) to DequantizeLinear::set_output_scale expected to contain atleast one value");
  this->scale = v.at(0);
}

Op::Layer::QLinearConv::QLinearConv() {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT] = 1;
  m_cp.stride[TENSOR_2D_WIDTH] = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH] = 1;
  m_cp.ki = 0;
  weights = nullptr;
  bias = nullptr;
}

const char *Op::Layer::QLinearConv::op_type() const { return m_optype; }
std::string Op::Layer::QLinearConv::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "(IC,IH,IW: " << this->input_dims[0][TENSOR_4D_CHANNELS] << ","
     << this->input_dims[0][TENSOR_4D_HEIGHT] << "," << this->input_dims[0][TENSOR_2D_WIDTH] << ") "
     << "(KN,KC,KH,KW: " << m_cp.kn << ","
     << weights->dims()[TENSOR_4D_CHANNELS] << ","
     << m_cp.k[TENSOR_2D_WIDTH] << "," << m_cp.k[TENSOR_2D_HEIGHT] << ") "
     << "(S,P,D: " << m_cp.stride[TENSOR_2D_WIDTH] << "," << m_cp.pad[I_LEFT]
     << "," << m_cp.dilation[TENSOR_2D_WIDTH] << ") ";

  /* store scales */
  ss << "x_scale: ";
  for (size_t i = 0; i < x_scale.size(); ++i) {
    if (i > 2) {
      ss << "...";
      break;
    }
    ss << x_scale[i] << ' ';
  }
  ss << "x_zero_point: ";
  for (size_t i = 0; i < x_zero_point.size(); ++i) {
    if (i > 2) {
      ss << "...";
      break;
    }
    if (std::holds_alternative<int8_t>(x_zero_point[i])) {
      ss << (int)std::get<int8_t>(x_zero_point[i]) << ' ';
    } else if (std::holds_alternative<uint8_t>(x_zero_point[i])) {
      ss << (int)std::get<uint8_t>(x_zero_point[i]) << ' ';
    } else {
      log_fatal("cant get type for x_zero_point\n");
    }
  }
  ss << '\n';
  ss << "Pipeline Odims: ";
  for (const auto &i : pipelined_output_dims) {
    std::cout << "[ ";
    for (int j : i) {
      ss << j << ' ';
    }
    std::cout << "], ";
  }
  ret = ss.str();
  return ret;
}

std::vector<float> Op::Layer::QLinearConv::get_output_scale(void) {
  return y_scale;
}
void Op::Layer::QLinearConv::set_output_scale(const std::vector<float>& v) {
  assert(v.size() == y_scale.size());
  y_scale = v;
}

enum QLC_INITIALIZERS {
  QLC_X_SCALE = 1,
  QLC_X_ZERO_POINT = 2,
  QLC_W = 3,
  QLC_W_SCALE = 4,
  QLC_W_ZERO_POINT = 5,
  QLC_Y_SCALE = 6,
  QLC_Y_ZERO_POINT = 7,
  QLC_B = 8
};

void Op::Layer::QLinearConv::set_initializer_params(
    int n, const onnx::TensorProto &t) {
  switch (n) {
  case QLC_X_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      x_scale.push_back(i);
    }
    break;
  case QLC_X_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      x_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      x_zero_point.push_back((int8_t)t.int32_data(0));
    }
    break;
  case QLC_W:
    m_cp.kn = t.dims()[0];
    m_cp.k[0] = t.dims()[2];
    m_cp.k[1] = t.dims()[3];
    weights = &t;
    break;
  case QLC_W_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      w_scale.push_back(i);
    }
    break;
  case QLC_W_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      w_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      w_zero_point.push_back((int8_t)t.int32_data(0));
    }
    break;
  case QLC_Y_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      y_scale.push_back(i);
    }
    break;
  case QLC_Y_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      y_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      y_zero_point.push_back((int8_t)t.int32_data(0));
    }
    break;
  case QLC_B:
    bias = &t;
    break;
  default:
    log_fatal("unknown initializer for layer {}\n", this->name);
    break;
  }
}

void Op::Layer::QLinearConv::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "dilations") {
      assert(itr->ints().size() == 2 && "expected dilations to be 2 integers");
      parse_onnx_ints(*itr, m_cp.dilation);
    }
  }
}

void Op::Layer::QLinearConv::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  assert(input_dims[0].size() == 4); // NCHW
  this->output_dims.resize(1);
  this->output_dims[0].resize(4);
  this->output_dims[0][0] = input_dims[0][0];
  this->output_dims[0][1] = this->m_cp.kn;
  this->output_dims[0][2] = sa_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[0][3] = sa_odims_cols(this->m_cp, input_dims[0]);
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::QLinearConv::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  /* TODO: get output type from y_zero_point */
  this->output_type = input_types;
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

Op::Layer::QLinearMatMul::QLinearMatMul() { m_cp = {}; }
const char *Op::Layer::QLinearMatMul::op_type() const { return m_optype; }
std::string Op::Layer::QLinearMatMul::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "IH,IW,WR,WC: " << this->input_dims[0][TENSOR_2D_HEIGHT] << ","
     << this->input_dims[0][TENSOR_2D_WIDTH] << "," << m_cp.wr << "," << m_cp.wc
     << " scale,zp: " << y_scale[0];
  ret = ss.str();
  return ret;
}

enum QLMM_INITIALIZERS {
  QLMM_A_SCALE = 1,
  QLMM_A_ZERO_POINT = 2,
  QLMM_B = 3,
  QLMM_B_SCALE = 4,
  QLMM_B_ZERO_POINT = 5,
  QLMM_Y_SCALE = 6,
  QLMM_Y_ZERO_POINT = 7
};

void Op::Layer::QLinearMatMul::set_initializer_params(
    int n, const onnx::TensorProto &t) {
  switch (n) {
  case QLMM_A_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      a_scale.push_back(i);
    }
    break;
  case QLMM_A_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      a_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      a_zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  case QLMM_B:
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
    break;
  case QLMM_B_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      b_scale.push_back(i);
    }
    break;
  case QLMM_B_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      b_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      b_zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  case QLMM_Y_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      y_scale.push_back(i);
    }
    break;
  case QLMM_Y_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      y_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      y_zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  default:
    log_fatal("unknown inputs number {} for tensor {}\n", n, t.name());
    break;
  }
}

void Op::Layer::QLinearMatMul::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);

  assert(input_dims[0].size() == 2);
  this->input_dims = input_dims;
  assert(input_dims[0].at(1) == this->m_cp.wr &&
         "QLinearMatMul, matrix dimensions do not match");
  this->output_dims.resize(1);
  this->output_dims[0].resize(2);
  this->output_dims[0].at(0) = input_dims[0].at(0);
  this->output_dims[0].at(1) = this->m_cp.wc;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::QLinearMatMul::infer_type(
    const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
}

Op::Layer::QLinearEltwise::QLinearEltwise(int op) : operator_type(op) {
  constant_data = nullptr;
}

const char *Op::Layer::QLinearEltwise::op_type() const { return m_optype; }

enum QLE_INITIALIZERS {
  QLE_A = 0,
  QLE_SCALE = 1,
  QLE_ZERO_POINT = 2,
  QLE_B = 3,
  QLE_B_SCALE = 4,
  QLE_B_ZERO_POINT = 5,
  QLE_C_SCALE = 6,
  QLE_C_ZERO_POINT = 7
};

std::vector<float> Op::Layer::QLinearEltwise::get_output_scale(void) {
  return o_scale;
}
void Op::Layer::QLinearEltwise::set_output_scale(const std::vector<float>& v) {
  assert(v.size() == o_scale.size());
  o_scale = v;
}

void Op::Layer::QLinearEltwise::set_initializer_params(int n,
                                                   const onnx::TensorProto &t) {
  switch (n) {
  case QLE_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      a_scale = i;
    }
    break;
  case QLE_ZERO_POINT:
    assert(t.int32_data_size() > 0);
    a_zp = (int)t.int32_data(0);
    break;
  case QLE_B:
    constant_data = &t;
    break;
  case QLE_B_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      b_scale = i;
    }
    break;
  case QLE_B_ZERO_POINT:
    assert(t.int32_data_size() > 0);
    b_zp = (int)t.int32_data(0);
    break;
  case QLE_C_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      o_scale.push_back(i);
    }
    break;
  case QLE_C_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  default:
    log_fatal("unknown inputs number {} for tensor {}\n", n, t.name());
    break;
  }
}

void Op::Layer::QLinearEltwise::infer_shape(const IVec2D &input_dims) {
  if (this->output_dims.size() != 0) {
    return;
  }
  if (this->constant_data != nullptr) {
    std::vector<int> weight_dims = get_tensorproto_shape(*this->constant_data);
    if (!is_broadcastable(input_dims[0], weight_dims)) {
      log_fatal(
          "input_dims and weight_dims can't be broadcasted for layer {}\n",
          this->name);
    }
  }
  if (input_dims[0].size() != 4) {
    log_fatal("cant infer shape for layer {}, dims.size() = {} != 4\n",
              this->name, input_dims.size());
  }
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::QLinearEltwise::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  if (this->input_type.size() != 0) {
    return;
  }
  this->input_type = input_types;
  this->output_type = input_types;
}

const char *Op::Layer::Transpose::op_type() const { return m_optype; }

std::string Op::Layer::Transpose::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "(perm: ";
  for (int64_t i : perm) {
    ss << i << ", ";
  }
  ss << ")";
  ret = ss.str();
  return ret;
}

void Op::Layer::Transpose::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "perm") {
      if (itr->ints_size() < 1) {
        log_fatal("expected node {} to contain perm info\n", node.name());
      }
      perm.resize(itr->ints_size(), 0);
      parse_onnx_ints(*itr, perm.data());
    } else {
      log_info("Parser un-implemented for attribute {} at node {}\n",
               itr->name(), node.name());
    }
  }
}

void Op::Layer::Transpose::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  for (const auto& i : input_dims) {
    this->output_dims.push_back(permute(i, this->perm));
  }
  this->pipelined_output_dims = this->output_dims;
}
void Op::Layer::Transpose::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

Op::Layer::MatMul::MatMul() { m_cp = {}; }

const char *Op::Layer::MatMul::op_type() const { return m_optype; }

std::string Op::Layer::MatMul::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "IH,IW,WR,WC: " << this->input_dims[0][TENSOR_2D_HEIGHT] << ","
     << this->input_dims[0][TENSOR_2D_WIDTH] << "," << m_cp.wr << "," << m_cp.wc;
  ret = ss.str();
  return ret;
}

void Op::Layer::MatMul::set_initializer_params(int,
                                               const onnx::TensorProto &t) {
  /* TODO: use n */
  if (t.dims_size() == GEMM_WEIGHT_TENSOR_DIMS) {
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
  }
}

Op::Layer::QGemm::QGemm() {
  m_cp = {};
  m_cp.alpha = 1.0;
  m_cp.beta = 1.0;
  m_cp.transA = 0;
  m_cp.transB = 0;
}
const char *Op::Layer::QGemm::op_type() const { return m_optype; }
std::string Op::Layer::QGemm::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "IH,IW,WR,WC: " << this->input_dims[0][TENSOR_2D_HEIGHT] << ' '
     << this->input_dims[0][TENSOR_2D_WIDTH] << ' ' << m_cp.wr << ' ' << m_cp.wc
     << ' ';

  ss << "alpha,beta,transA,transB: " << m_cp.alpha << ' ' << m_cp.beta << ' '
     << m_cp.transA << ' ' << m_cp.transB << '\n';

  ss << "Former Dims ";
  for (int i : former_layer_dims) {
    ss << i << ' ';
  }
  ret = ss.str();
  return ret;
}

std::vector<float> Op::Layer::QGemm::get_output_scale(void) {
  return y_scale;
}
void Op::Layer::QGemm::set_output_scale(const std::vector<float>& v) {
  assert(v.size() == y_scale.size());
  y_scale = v;
}

enum QGEMM_INITIALIZERS {
  QLG_A_SCALE = 1,
  QLG_A_ZERO_POINT = 2,
  QLG_B = 3,
  QLG_B_SCALE = 4,
  QLG_B_ZERO_POINT = 5,
  QLG_C = 6,
  QLG_Y_SCALE = 7,
  QLG_Y_ZERO_POINT = 8
};

void Op::Layer::QGemm::set_initializer_params(int n,
                                              const onnx::TensorProto &t) {
  switch (n) {
  case QLG_A_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      a_scale.push_back(i);
    }
    break;
  case QLG_A_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      a_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      a_zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  case QLG_B:
    m_cp.wr = t.dims()[0];
    m_cp.wc = t.dims()[1];
    weights = &t;
    break;
  case QLG_B_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      b_scale.push_back(i);
    }
    break;
  case QLG_B_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      b_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      b_zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  case QLG_C:
    bias = &t;
    break;
  case QLG_Y_SCALE:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    for (auto i : t.float_data()) {
      y_scale.push_back(i);
    }
    break;
  case QLG_Y_ZERO_POINT:
    if (t.data_type() == onnx::TensorProto_DataType_UINT8) {
      y_zero_point.push_back((uint8_t)t.int32_data(0));
    } else if (t.data_type() == onnx::TensorProto_DataType_INT8) {
      y_zero_point.push_back((int8_t)t.int32_data(0));
    } else {
      log_fatal("cant deduce zero point for tensor {}\n", t.name());
    }
    break;
  default:
    log_fatal("unknown inputs number {} for tensor {}\n", n, t.name());
    break;
  }
}

void Op::Layer::QGemm::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "alpha") {
      if (itr->has_f()) {
        m_cp.alpha = itr->f();
      }
    } else if (itr->name() == "beta") {
      if (itr->has_f()) {
        m_cp.beta = itr->f();
      }
    } else if (itr->name() == "transA") {
      if (itr->has_i()) {
        m_cp.transA = itr->i();
      }
    } else if (itr->name() == "transB") {
      if (itr->has_i()) {
        m_cp.transB = itr->i();
      }
    }
  }
}

void Op::Layer::QGemm::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  assert(input_dims[0].size() == 2);
  this->input_dims = input_dims;
  this->output_dims.resize(1);
  this->output_dims[0].resize(2);
  this->output_dims[0].at(0) = input_dims[0].at(0);
  if (m_cp.transB) {
    assert(input_dims[0].at(1) == this->m_cp.wc &&
           "QGemm, matrix dimensions do not match");
    this->output_dims[0].at(1) = this->m_cp.wr;
  } else {
    assert(input_dims[0].at(1) == this->m_cp.wr &&
           "QGemm, matrix dimensions do not match");
    this->output_dims[0].at(1) = this->m_cp.wc;
  }
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::QGemm::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
  this->weight_type = Op::get_type_from_tensor_proto(*this->weights);
  this->bias_type = Op::get_type_from_tensor_proto(*this->bias);
}

Op::Layer::LogSoftmax::LogSoftmax() { axis = -1; }

const char *Op::Layer::LogSoftmax::op_type() const { return m_optype; }

std::string Op::Layer::LogSoftmax::params() const {
  std::string pbuf;
  std::stringstream ss;
  ss << "Axis: " << axis;
  pbuf = ss.str();
  return pbuf;
}

void Op::Layer::LogSoftmax::set_initializer_params(int,
                                                   const onnx::TensorProto &) {
  return;
}

void Op::Layer::LogSoftmax::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        axis = static_cast<int>(itr->i());
      } else {
        log_fatal("cannot find attribute 'axis' in layer {}, is it an integer?",
                  node.name());
      }
    }
  }
}

void Op::Layer::LogSoftmax::infer_shape(const IVec2D &input_dims) {
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::LogSoftmax::infer_type(const std::vector<TPDT> &input_types) {
  this->input_type = input_types;
  this->output_type = input_types;
}

Op::Layer::QLinearAveragePool::QLinearAveragePool(bool gbl) {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT] = 1;
  m_cp.stride[TENSOR_2D_WIDTH] = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH] = 1;
  m_cp.gbl = gbl;
  x_scale = 0;
  y_scale = 0;
}

const char *Op::Layer::QLinearAveragePool::op_type() const { return m_optype; }

std::string Op::Layer::QLinearAveragePool::params() const {
  return get_pool_params(this->input_dims[0], this->m_cp).str();
}

void Op::Layer::QLinearAveragePool::set_attributes(
    const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "ceil_mode") {
      if (itr->has_i()) {
        int ceil_mode = static_cast<int>(itr->i());
        if (ceil_mode != 0) {
          log_fatal("Unsupported ceil_mode {} in layer {}\n", ceil_mode,
                    node.name());
        }
      }
    } else if (itr->name() == "channels_last") {
      if (itr->has_i()) {
        int channels_last = static_cast<int>(itr->i());
        if (channels_last != 0) {
          log_fatal("Unsupported channels_last {} in layer {}\n", channels_last,
                    node.name());
        }
      }
    } else if (itr->name() == "count_include_pad") {
      if (itr->has_i()) {
        int count_include_pad = static_cast<int>(itr->i());
        if (count_include_pad != 1) {
          log_fatal("Unsupported count_include_pad {} in layer {}\n",
                    count_include_pad, node.name());
        }
      }
    } else if (itr->name() == "auto_pad") {
      if (itr->has_s()) {
        auto auto_pad = itr->s();
        auto expected_auto_pad = "NOTSET";
        if (auto_pad != expected_auto_pad) {
          log_fatal("Unsupported auto_pad type {}, expected auto pad to be {} "
                    "in layer {}\n",
                    auto_pad, expected_auto_pad, node.name());
        }
      }
    }
  }
}

void Op::Layer::QLinearAveragePool::infer_type(
    const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

void Op::Layer::QLinearAveragePool::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  assert(input_dims[0].size() == 4);
  if (this->m_cp.gbl) {
    this->m_cp.k[0] = input_dims[0][2];
    this->m_cp.k[1] = input_dims[0][3];
  }
  this->output_dims.resize(1);
  this->output_dims[0].resize(4);
  this->output_dims[0][0] = input_dims[0][0];
  this->output_dims[0][1] = input_dims[0][1];
  this->output_dims[0][2] = mp_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[0][3] = mp_odims_cols(this->m_cp, input_dims[0]);
  this->pipelined_output_dims = this->output_dims;
}

const char *Op::Layer::Abs::op_type() const { return m_optype; }

void Op::Layer::Abs::infer_shape(const IVec2D &input_dims) {
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Abs::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

Op::Layer::ReduceMean::ReduceMean() : m_axis{-1}, m_keepdims{1} {}

const char *Op::Layer::ReduceMean::op_type() const { return m_optype; }

std::string Op::Layer::ReduceMean::params() const {
  std::stringstream ss;
  ss << "axis: " << m_axis << " keepdims: " << m_keepdims;
  return ss.str();
}

void Op::Layer::ReduceMean::infer_shape(const IVec2D &input_dims) {
  this->input_dims = input_dims;
  // this->output_dims = reduced_shape(this->input_dims, m_axis, m_keepdims);
  log_warn("Using a hacky (incorrect) implementation of reduce_mean. Inputs "
           "pass through\n");
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::ReduceMean::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

void Op::Layer::ReduceMean::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axes") {
      std::vector<int> axes;
      parse_onnx_ints(*itr, axes);
      m_axis = axes.at(0);
    } else if (itr->name() == "keepdims") {
      if (itr->has_i()) {
        m_keepdims = static_cast<int>(itr->i());
      }
    }
  }
}

Op::Layer::AveragePool::AveragePool(bool gbl) {
  /* zero initialize */
  m_cp = {};
  /* overwrite with sane defaults */
  m_cp.stride[TENSOR_2D_HEIGHT] = 1;
  m_cp.stride[TENSOR_2D_WIDTH] = 1;
  m_cp.dilation[TENSOR_2D_HEIGHT] = 1;
  m_cp.dilation[TENSOR_2D_WIDTH] = 1;
  m_cp.gbl = gbl;
}

const char *Op::Layer::AveragePool::op_type() const { return m_optype; }

std::string Op::Layer::AveragePool::params() const {
  return get_pool_params(this->input_dims[0], this->m_cp).str();
}

void Op::Layer::AveragePool::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "kernel_shape") {
      assert(itr->ints().size() == 2 &&
             "expected kernel shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.k);
    } else if (itr->name() == "strides") {
      assert(itr->ints().size() == 2 &&
             "expected strides shape to be 2 integers");
      parse_onnx_ints(*itr, m_cp.stride);
    } else if (itr->name() == "pads") {
      assert(itr->ints().size() == 4 && "expected pads shape to be 4 integers");
      parse_onnx_ints(*itr, m_cp.pad);
    } else if (itr->name() == "ceil_mode") {
      if (itr->has_i()) {
        int ceil_mode = static_cast<int>(itr->i());
        if (ceil_mode != 0) {
          log_fatal("Unsupported ceil_mode {} in layer {}\n", ceil_mode,
                    node.name());
        }
      }
    } else if (itr->name() == "count_include_pad") {
      if (itr->has_i()) {
        int count_include_pad = static_cast<int>(itr->i());
        if (count_include_pad != 1) {
          log_fatal("Unsupported count_include_pad {} in layer {}\n",
                    count_include_pad, node.name());
        }
      }
    } else if (itr->name() == "auto_pad") {
      if (itr->has_s()) {
        auto auto_pad = itr->s();
        auto expected_auto_pad = "NOTSET";
        if (auto_pad != expected_auto_pad) {
          log_fatal("Unsupported auto_pad type {}, expected auto pad to be {} "
                    "in layer {}\n",
                    auto_pad, expected_auto_pad, node.name());
        }
      }
    }
  }
}

void Op::Layer::AveragePool::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

void Op::Layer::AveragePool::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  assert(input_dims[0].size() == 4);
  if (this->m_cp.gbl) {
    this->m_cp.k[0] = input_dims[0][2];
    this->m_cp.k[1] = input_dims[0][3];
  }
  this->output_dims[0].resize(4);
  this->output_dims[0][0] = input_dims[0][0];
  this->output_dims[0][1] = input_dims[0][1];
  this->output_dims[0][2] = mp_odims_row(this->m_cp, input_dims[0]);
  this->output_dims[0][3] = mp_odims_cols(this->m_cp, input_dims[0]);
  this->pipelined_output_dims = this->output_dims;
}

const char *Op::Layer::Shape::op_type() const { return m_optype; }

void Op::Layer::Shape::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type.push_back(onnx::TensorProto_DataType_INT64);
}

void Op::Layer::Shape::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims[0].push_back(this->input_dims[0].size());
  this->pipelined_output_dims = this->output_dims;
}

Op::Layer::Gather::Gather() : m_axis{0}, m_indices{nullptr} {}

const char *Op::Layer::Gather::op_type() const { return m_optype; }

std::string Op::Layer::Gather::params() const {
  std::stringstream ss;
  ss << "[dynamic out shape] axis: " << m_axis;
  return ss.str();
}

void Op::Layer::Gather::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type.push_back(onnx::TensorProto_DataType_INT64);
}

void Op::Layer::Gather::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims = input_dims;
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Gather::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        m_axis = static_cast<int>(itr->i());
      }
    }
  }
}

const char *Op::Layer::Unsqueeze::op_type() const { return m_optype; }

void Op::Layer::Unsqueeze::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

void Op::Layer::Unsqueeze::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims[0] = unsqueeze_shape(this->input_dims[0], axis);
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Unsqueeze::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    // odd spelling for 'axis', i know
    if (itr->name() == "axes") {
      parse_onnx_ints(*itr, axis);
    }
  }
}

const char *Op::Layer::Concat::op_type() const { return m_optype; }

void Op::Layer::Concat::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type = input_types;
}

void Op::Layer::Concat::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() >= 1);
  this->input_dims = input_dims;
  this->output_dims[0] = concat_shape(input_dims, m_axis);
  this->pipelined_output_dims = this->output_dims;
}

void Op::Layer::Concat::set_attributes(const onnx::NodeProto &node) {
  const auto &attribute = node.attribute();
  for (auto itr = attribute.begin(); itr != attribute.end(); ++itr) {
    if (itr->name() == "axis") {
      if (itr->has_i()) {
        m_axis = static_cast<int>(itr->i());
      }
    }
  }
}

Op::Layer::NMS::NMS() {}

const char *Op::Layer::NMS::op_type() const { return m_optype; }

std::string Op::Layer::NMS::params() const {
  std::string ret;
  std::stringstream ss;
  ss << "(max_boxes: " << max_output_boxes << "), (iou_thresh: " << std::fixed
     << std::setprecision(2) << iou_threshold
     << "), (score_thresh: " << std::fixed << std::setprecision(2)
     << score_threshold << ")";
  ret = ss.str();
  return ret;
}

void Op::Layer::NMS::set_attributes(const onnx::NodeProto &node) {
  for (const auto &attr : node.attribute()) {
    if (attr.name() == "center_point_box") {
      if (attr.has_i()) {
        center_point_box = attr.i();
      } else {
        log_fatal("Expected 'center_point_box' attribute to be an integer.");
      }
    }
  }
}

void Op::Layer::NMS::infer_shape(const IVec2D &input_dims) {
  assert(input_dims.size() == 2);
  this->input_dims = input_dims;
  this->output_dims = {
      {static_cast<int>(max_output_boxes), 3}}; //(selected_boxes, class, score)
}

void Op::Layer::NMS::infer_type(const std::vector<TPDT> &input_types) {
  assert(input_types.size() >= 1);
  this->input_type = input_types;
  this->output_type.push_back(onnx::TensorProto_DataType_INT64);
}

enum NMS_INITIALIZER {
  MAX_OUT_BOXES = 2,
  IOU_THRESHOLD = 3,
  SCORE_THRESHOLD = 4
};

void Op::Layer::NMS::set_initializer_params(int n, const onnx::TensorProto &t) {
  switch (n) {
  case MAX_OUT_BOXES:
    assert(t.data_type() == onnx::TensorProto_DataType_INT64);
    max_output_boxes = t.int64_data(0);
    break;
  case IOU_THRESHOLD:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    iou_threshold = t.float_data(0);
    break;
  case SCORE_THRESHOLD:
    assert(t.data_type() == onnx::TensorProto_DataType_FLOAT);
    score_threshold = t.float_data(0);
    break;
  default:
    log_fatal("unknown inputs number {} for tensor {}\n", n, t.name());
    break;
  }
}


/* Auxillary Graph Functions */

bool Op::is_root_node(Op::Vertex v, const Op::Graph *g) {
  return boost::in_degree(v, *g) == 0;
}

bool Op::are_equal_nodes(Op::Vertex v1, Op::Vertex v2, const Op::Graph *g) {
  return ((*g)[v1]->name == (*g)[v2]->name);
}

Op::Vertex Op::get_root_node(const Op::Graph *g) {
  auto verts = boost::vertices(*g);
  for (auto it = verts.first; it != verts.second; ++it) {
    if (is_root_node(*it, g)) {
      return *it;
    }
  }
}

/* Op::Model */

void Op::Model::add(Op::LayerBase *layer, const onnx::NodeProto &node) {
  Op::Vertex v = boost::add_vertex(layer, g);

  if (!node.has_name()) {
    log_fatal("Layer of type {} has no name\n", node.op_type());
  }
  layer->name = node.name();

  name_vertex_map.insert({node.name(), v});

  for (int i = 0; i < node.input().size(); ++i) {
    /* find value_info param for `i` */
    auto itr2 = value_info_map.find(node.input(i));
    if (itr2 != value_info_map.end()) {
      layer->set_value_info_params(itr2->second);
    }
    /* find initializer for `i` */
    auto itr3 = initializer_map.find(node.input(i));
    if (itr3 != initializer_map.end()) {
      layer->set_initializer_params(i, itr3->second);
    }
    auto itr4 = constant_pool.find(node.input(i));
    if (itr4 != constant_pool.end()) {
      layer->set_constant_params(i, itr4->second);
    }
  }
  for (auto i : node.output()) {
    output_map.insert({i, v});
  }
  layer->set_attributes(node);
}

bool Op::Model::is_graph_input(const std::string &s) const {
  auto itr = graph_input_map.find(s);
  if (itr != graph_input_map.end()) {
    return true;
  }
  return false;
}

bool Op::Model::is_initializer(const std::string &s) const {
  auto itr = initializer_map.find(s);
  if (itr != initializer_map.end()) {
    return true;
  }
  return false;
}

bool Op::Model::is_constant(const std::string &s) const {
  auto itr = constant_pool.find(s);
  if (itr != constant_pool.end()) {
    return true;
  }
  return false;
}

bool Op::Model::is_graph_output(const std::string &s) const {
  auto itr = graph_output_map.find(s);
  if (itr != graph_output_map.end()) {
    return true;
  }
  return false;
}

void Op::Model::connect(const onnx::NodeProto &node) {
  /* find the Op::Vertex for `node` */
  Op::Vertex current_node;
  auto itr = name_vertex_map.find(node.name());
  if (itr != name_vertex_map.end()) {
    // found vertex for current node
    current_node = itr->second;
  } else {
    log_fatal("Coudn't find node {} in name_vertex_map\n", node.name());
  }
  for (auto i : node.input()) {
    /* for inputs that are not initializers or inputs to the
     * graph, connect them to the current node
     */
    if (!is_initializer(i) && !is_graph_input(i) && !is_constant(i)) {
      auto itr = output_map.find(i);
      if (itr != output_map.end()) {
        /* connect */
        boost::add_edge(itr->second, current_node, g);
      } else {
        log_fatal("Coudn't find node {} in output_map or constant_pool\n", i);
      }
    }
  }
}

void Op::Model::save_initializers(const onnx::TensorProto &t) {
  initializer_map.insert({t.name(), t});
}

void Op::Model::save_input_output_names() {
  auto exec_order = crt_exec_order(g);
  for (Op::LayerBase *l : exec_order) {
    auto itr = name_node_map.find(l->name);
    if (itr == name_node_map.end()) {
      log_fatal("Couldn't find layer {} in name_node_map\n", l->name);
    }
    for (auto i : itr->second.input()) {
      l->input_names.push_back(i);
    }
    for (auto i : itr->second.output()) {
      l->output_names.push_back(i);
    }
  }
}

void Op::Model::save_graph_outputs(const onnx::ValueInfoProto &t) {
  graph_output_map.insert({t.name(), t});
}

void Op::Model::save_graph_inputs(const onnx::ValueInfoProto &t) {
  graph_input_map.insert({t.name(), t});
}

void Op::Model::save_value_info(const onnx::ValueInfoProto &t) {
  value_info_map.insert({t.name(), t});
}

void Op::Model::save_first_layer_input_dims(const onnx::ValueInfoProto &t) {
  auto vertices = boost::vertices(g);
  auto first_node = vertices.first;
  LayerBase *layer = g[*first_node];
  layer->set_value_info_params(t);
}

size_t Op::Model::size(void) { return boost::num_vertices(g); }
size_t Op::Model::size(void) const { return boost::num_vertices(g); }

bool Op::Model::has_graph_output(Op::LayerBase *l) const {
  if (graph_output_map.size() != 1) {
    log_fatal("Graphs with only one outputs are currently supported\n");
  }
  auto graph_out = graph_output_map.begin();
  auto output_name = (graph_out->second).name();
  auto itr = output_map.find(output_name);
  if (itr != output_map.end() && g[itr->second]->name == l->name) {
    return true;
  }
  return false;
}

void Op::print_node(Op::Vertex v, const Op::Graph *g) {
  LayerBase *node = (*g)[v];
  Op::print_node(node);
  std::cout << "Out Degree: " << boost::out_degree(v, (*g)) << " (";
  auto out_edges = boost::out_edges(v, (*g));
  for (auto ei = out_edges.first; ei != out_edges.second; ++ei) {
    Op::Vertex dest_vertex = boost::target(*ei, (*g));
    std::cout << (*g)[dest_vertex]->name << ", ";
  }
  std::cout << ")\n";

  std::cout << "In Degree: " << boost::in_degree(v, (*g)) << " (";
  auto in_edges = boost::in_edges(v, (*g));
  for (auto ei = in_edges.first; ei != in_edges.second; ++ei) {
    Op::Vertex source_vertex = boost::source(*ei, (*g));
    std::cout << (*g)[source_vertex]->name << ", ";
  }
  std::cout << ")\n";
  std::cout << '\n';
}

void Op::print_node(const LayerBase *node) {
  std::cout << "Type: " << node->op_type() << '\n';
  std::cout << "Params: " << node->params() << '\n';
  std::cout << "Name: " << node->name << '\n';
  std::cout << "Input Registers: ";
  for (auto i : node->inputs) {
    std::cout << i << ' ';
  }
  std::cout << '\n';
  std::cout << "Output Registers: ";
  for (auto i : node->outputs) {
    std::cout << i << ' ';
  }
  std::cout << '\n';
  std::cout << "Input Type: ";
  for (const auto &i : node->input_type) {
    std::cout << Op::get_tensorproto_dtype_name(i) << ' ';
  }
  std::cout << '\n';
  std::cout << "Output Type: ";
  for (const auto &i : node->output_type) {
    std::cout << Op::get_tensorproto_dtype_name(i) << ' ';
  }
  std::cout << '\n';
  const char *device = Op::get_device_name(node->device);
  std::cout << "Device " << device << '\n';
  std::cout << "Input dims: ";
  for (const auto &i : node->input_dims) {
    print_vec("", i);
    std::cout << " ";
  }
  std::cout << std::endl;
  std::cout << "Output dims: ";
  for (const auto &i : node->output_dims) {
    print_vec("", i);
    std::cout << " ";
  }
  std::cout << std::endl;
}

const char *Op::get_device_name(int device) {
  switch (device) {
  case DEVICE_UNKNOWN:
    return "DEVICE_UNKNOWN";
    break;
  case DEVICE_CPU:
    return "DEVICE_CPU";
    break;
  case DEVICE_FPGA:
    return "DEVICE_FPGA";
    break;
  default:
    log_fatal("unknown device enum {}, can't get name\n", device);
    break;
  }
}

const char *Op::get_tensorproto_dtype_name(TPDT type) {
  switch (type) {
  case 0:
    return "UNDEFINED";
    break;
  case 1:
    return "FLOAT";
    break;
  case 2:
    return "UINT8";
    break;
  case 3:
    return "INT8";
    break;
  case 4:
    return "UINT16";
    break;
  case 5:
    return "INT16";
    break;
  case 6:
    return "INT32";
    break;
  case 7:
    return "INT64";
    break;
  case 8:
    return "STRING";
    break;
  case 9:
    return "BOOL";
    break;
  case 10:
    return "FLOAT16";
    break;
  case 11:
    return "DOUBLE";
    break;
  case 12:
    return "UINT32";
    break;
  case 13:
    return "UINT64";
    break;
  case 14:
    return "COMPLEX64";
    break;
  case 15:
    return "COMPLEX128";
    break;
  case 16:
    return "BFLOAT16";
    break;
  case 17:
    return "FLOAT8E4M3FN";
    break;
  case 18:
    return "FLOAT8E4M3FNUZ";
    break;
  case 19:
    return "FLOAT8E5M2";
    break;
  case 20:
    return "FLOAT8E5M2FNUZ";
    break;
  default:
    return "UNKNOWN";
    break;
  }
}

std::vector<int> Op::get_tensorproto_shape(const onnx::TensorProto &t) {
  const auto &dims = t.dims();
  std::vector<int> ret_dims;
  for (auto i : dims) {
    ret_dims.push_back(i);
  }
  return ret_dims;
}

long Op::time_estimate(Op::Graph graph) {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(graph);
  long cycles = 0;

  auto sa_arch = get_sa_arch();
  int va_size = get_va_size();

  if (!gbl_args.has_option("timeest")) {
    log_fatal("--timeest expects frequency as an argument, see --help\n");
  }
  int frequency = gbl_args["timeest"].as<int>();

  int qle_time = 0;
  for (auto itr = vb; itr != ve; ++itr) {
    LayerBase *node = graph[*itr];
    if (is_conv_like(node->op_type())) {
      int input_columns = node->output_dims[0][TENSOR_4D_WIDTH] *
                          node->output_dims[0][TENSOR_4D_HEIGHT];
      int kern_itr = 0; int chan_itr = 0;
      std::tie(kern_itr, chan_itr) = node->get_iterations();
      int t = kern_itr * chan_itr * input_columns;
      cycles += t;
      std::cout << "Time: " << (float)t / (frequency * 1e3) << "ms\n";
      Op::print_node(*itr, &graph);
    } else if (is_gemm_like(node->op_type())) {
      auto wr_wc = get_true_rc_weights(node);
      int available_pe_columns = va_size;
      int cols = wr_wc[1];
      if (wr_wc[1] < 32) {
        cols = 32;
      }
      int t = ((float)cols / (float)available_pe_columns) * wr_wc[0];
      cycles += t;
      std::cout << "Time: " << (float)t / (frequency * 1e3) << "ms\n";
      Op::print_node(*itr, &graph);
    } else if (strcmp(node->op_type(), "QLinearEltwise") == 0) {
      int t = node->output_dims[0][TENSOR_4D_WIDTH] * node->output_dims[0][TENSOR_4D_HEIGHT] * ceil_div(node->output_dims[0][TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]) * 2;
      cycles += t;
      qle_time += t;
      std::cout << "Time: " << (float)t / (frequency * 1e3) << "ms\n";
      Op::print_node(*itr, &graph);
    }
  }
  std::cout << "QLE Time: " << (float) qle_time / (frequency * 1e3) << "ms\n";
  std::cout << "Total Estimated time: "
            << (float)cycles / (frequency * 1e3) << "ms\n";
  return cycles;
}

void Op::Model::bare_summary(void) const {
  Op::VertexIterator vb, ve;
  std::tie(vb, ve) = boost::vertices(g);
  for (auto itr = vb; itr != ve; ++itr) {
    Op::print_node(*itr, &g);
  }
}


void Op::Model::update_registers(void) { RegisterAllocator ral(g); }

/* recursively calls `virtual LayerBase::infer_shape` on each node and its child
 * nodes */

/*
 * Consider the following directed graph:
 *           +-------+
 *      +--->|   1   |<-----+
 *      |    |       |      |
 *      |    +-------+      |
 *      |                   |
 *      |                   |
 *  +---v---+               |
 *  |   2   |               |
 *  |       |               |
 *  +---^---+               |
 *      |               +---v---+
 *      |               |   5   |
 *      |               |       |
 *  +---+---+           +-------+
 *  |   3   |               ^
 *  |       |               |
 *  +-------+               |
 *      ^                   |
 *      |                   |
 *      |       +-------+   |
 *      |       |   4   |   |
 *      +------>|       |<--+
 *              +-------+
 *
 * This function performs a breadth-first traversal of the graph nodes.
 * For each node, it calls `infer_shape` using the output shapes of its parent nodes.
 * The `infer_shape` function is only called when all parent nodes have already
 * had their shapes inferred.
 *
 * This ensures the traversal order is: 1, 2, 5, 3, 4.
 *
 * If the output shape of node 1 is X, then:
 * - The input shapes for nodes 2 and 5 become X
 * - Nodes 2 and 5 internally calculate their own output shapes (say Y1 and Y2, respectively)
 * - These shapes are then passed down to their child nodes
 * Consequently:
 * - The input shape for node 3 will be Y1
 * - The input shape for node 4 will be Y2
 */
void Op::Model::deduce_shapes(const IVec2D &input_dims) {
  std::queue<Op::Vertex> S;
  /* all nodes on which shape inference is done */
  std::unordered_set<Op::Vertex> done_set;
  Op::Graph gcopy = g;

  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  /* set first layer's input dims */
  IVec2D tmp = input_dims;
  gcopy[v]->infer_shape(tmp);
  done_set.insert(v);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, gcopy)});
    }

    for (auto [src, dest] : edges_to_remove) {
      /* make sure all parents of 'dest' have underwent infer_shape */
      auto in_edges = boost::in_edges(dest, gcopy);
      bool dest_parents_done = 1;
      for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
        Op::Vertex dsource = boost::source(*itr, gcopy);
        auto present = done_set.find(dsource);
        if (present == done_set.end()) {
          dest_parents_done = 0;
        } 
      }

      if (dest_parents_done) {
        auto in_dims = Op::get_dims_of_in_edges(dest, gcopy);
        gcopy[dest]->infer_shape(in_dims);
        done_set.insert(dest);
        boost::remove_edge(src, dest, gcopy);
        if (boost::in_degree(dest, gcopy) == 0) {
          S.push(dest);
        }
      } else {
        S.push(n);
      }
    }
  }
}

/* Operates almost exactly like deduce_shape but calls infer_type instead of 
 * infer_shape
 */
void Op::Model::deduce_types(const std::vector<TPDT> &input_types) {
  std::queue<Op::Vertex> S;
  std::unordered_set<Op::Vertex> done_set;
  Op::Graph gcopy = g;

  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  /* set first layer's input dims */
  gcopy[v]->infer_type(input_types);
  done_set.insert(v);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, gcopy)});
    }

    for (auto [src, dest] : edges_to_remove) {
      /* make sure all parents of 'dest' have underwent infer_shape */
      auto in_edges = boost::in_edges(dest, gcopy);
      bool dest_parents_done = 1;
      for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
        Op::Vertex dsource = boost::source(*itr, gcopy);
        auto present = done_set.find(dsource);
        if (present == done_set.end()) {
          dest_parents_done = 0;
        } 
      }

      if (dest_parents_done) {
        auto itr2 = name_node_map.find(gcopy[dest]->name);
        if (itr2 == name_node_map.end()) {
          log_fatal("could not find {} in name_node_map\n", gcopy[dest]->name);
        }
        onnx::NodeProto &np = itr2->second;
        auto i_nodes = Op::get_input_nodes(np, g, output_map);
        auto in_types = Op::get_types_of_in_edges(dest, gcopy, i_nodes);
        gcopy[dest]->infer_type(in_types);
        done_set.insert(dest);
        boost::remove_edge(src, dest, gcopy);
        if (boost::in_degree(dest, gcopy) == 0) {
          S.push(dest);
        }
      } else {
        S.push(n);
      }
    }
  }
}

void Op::Model::set_input_type(const onnx::NodeProto &node, Op::LayerBase *l) {
  /* Update LayerBase->input_type for inputs of a node */
  for (const auto &input : node.input()) {
    /* If a node is found in value_info_map, it is likely
     * not an initializer
     */
    if (!is_initializer(input)) {
      auto vitr = value_info_map.find(input);
      if (vitr != value_info_map.end()) {
        l->input_type.push_back(get_type_from_value_info(vitr->second));
        break;
      }
      /* Same, if is found in graph_input/graph_output map */
      auto gi_itr = graph_input_map.find(input);
      if (gi_itr != graph_input_map.end()) {
        l->input_type.push_back(get_type_from_value_info(gi_itr->second));
        break;
      }
      log_fatal(
          "Couldn't find {} for node {} in value_info_map or graph_input_map\n",
          input, node.name());
    }
  }
}

void Op::Model::set_output_type(const onnx::NodeProto &node, Op::LayerBase *l) {
  /* Update types for outputs of a node */
  for (const auto &output : node.output()) {
    if (!is_initializer(output)) {
      auto vitr = value_info_map.find(output);
      if (vitr != value_info_map.end()) {
        l->output_type.push_back(get_type_from_value_info(vitr->second));
        break;
      }
      const auto go_itr = graph_output_map.find(output);
      if (go_itr != graph_output_map.end()) {
        l->output_type.push_back(get_type_from_value_info(go_itr->second));
        break;
      }
      log_fatal("Couldn't find {} for node {} in value_info_map or "
                "graph_output_map\n",
                output, node.name());
    }
  }
}

TPDT Op::get_type_from_value_info(const onnx::ValueInfoProto &v) {
  if (!v.has_type()) {
    /* TODO: bug, last node's types are not being deduced */
    log_info(
        "graph input's valueinfoproto named \"{}\" does not have a data type\n",
        v.name());
    return onnx::TensorProto_DataType_UNDEFINED;
  }
  const onnx::TypeProto &type = v.type();
  if (!type.has_tensor_type()) {
    log_fatal("input to the graph is not a a TensorType\n");
  }
  const onnx::TypeProto_Tensor &tensor = type.tensor_type();
  if (!tensor.has_elem_type()) {
    log_fatal("tensor for graph's input does not have a elem_type\n");
  }
  return static_cast<TPDT>(tensor.elem_type());
}

TPDT Op::get_type_from_tensor_proto(const onnx::TensorProto &v) {
  if (v.has_data_type()) {
    return (TPDT)v.data_type();
  } else {
    log_fatal("could not deduce type for tensor {}\n", v.name());
  }
}

const onnx::TensorShapeProto &
Op::get_tensor_shape_proto(const onnx::ValueInfoProto &t) {
  if (!t.has_type()) {
    log_fatal("valueinfoproto {} does not have a type\n", t.name());
  }
  const onnx::TypeProto &type = t.type();
  if (!type.has_tensor_type()) {
    log_fatal("valuefatalproto {} has a type but does not have a tensor_type\n",
              t.name());
  }
  const onnx::TypeProto_Tensor &tensor = type.tensor_type();
  if (!tensor.has_shape()) {
    log_fatal("valuefatalproto {} does not have a shape\n", t.name());
  }
  const onnx::TensorShapeProto &shape = tensor.shape();
  return shape;
}

std::vector<int> Op::get_dims_from_value_info(const onnx::ValueInfoProto &v) {
  const auto &shape = Op::get_tensor_shape_proto(v);
  std::vector<int> dims;
  for (const auto &i : shape.dim()) {
    if (i.has_dim_value()) {
      dims.push_back(static_cast<int>(i.dim_value()));
    } else if (i.has_dim_param()) {
      /* default batch size */
      log_info("got dim parameter {}, setting it to 1 (default)\n",
               i.dim_param());
      dims.push_back(1);
    } else if (i.has_denotation()) {
      log_info("found denotation, but ignoring (needs support)\n");
    }
  }
  return dims;
}

std::vector<Op::Vertex> get_parents(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> ret;
  auto edges = boost::in_edges(v, g);
  for (auto itr = edges.first; itr != edges.second; ++itr) {
    Op::Vertex src_v = boost::source(*itr, g);
    ret.push_back(src_v);
  }
  return ret;
}

std::vector<Op::Vertex> get_children(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> ret;
  auto edges = boost::out_edges(v, g);
  for (auto itr = edges.first; itr != edges.second; ++itr) {
    Op::Vertex src_v = boost::target(*itr, g);
    ret.push_back(src_v);
  }
  return ret;
}

IVec2D Op::get_dims_of_in_edges(Op::Vertex v, const Op::Graph &g) {
  IVec2D ret;
  auto in_edges = boost::in_edges(v, g);
  for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
    Op::Vertex src_vertex = boost::source(*itr, g);
    for (const auto &out_dim : g[src_vertex]->output_dims) {
      ret.push_back(out_dim);
    }
  }
  return ret;
}

std::vector<std::string>
Op::get_input_nodes(const onnx::NodeProto &np, const Op::Graph &g,
                    const std::map<std::string, Op::Vertex> &output_map) {
  std::vector<std::string> ret;
  for (const auto &i : np.input()) {
    auto itr = output_map.find(i);
    if (itr == output_map.end()) {
      continue;
    }
    Op::Vertex v = itr->second;
    ret.push_back(g[v]->name);
  }
  return ret;
}

std::vector<TPDT>
Op::get_types_of_in_edges(Op::Vertex v, const Op::Graph &g,
                          const std::vector<std::string> &i_nodes) {
  std::vector<TPDT> ret(i_nodes.size());
  std::unordered_map<std::string, int> name_index;
  int index = 0;
  for (const auto &i : i_nodes) {
    name_index.insert({i, index++});
  }
  auto in_edges = boost::in_edges(v, g);
  for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
    Op::Vertex src_vertex = boost::source(*itr, g);
    int idex = name_index[g[src_vertex]->name];
    if (g[src_vertex]->output_type.size() > 0) {
      ret.at(idex) = g[src_vertex]->output_type[0];
    }
  }
  return ret;
}

int Op::tpdt_sizeof(TPDT v) {
  int32_t dtype = (int32_t)v;
  switch (dtype) {
  case 0:
    log_fatal("cannot calculate sizeof for type {}\n", dtype);
    break;
  case 1:
    return sizeof(float);
    break;
  case 2:
    return sizeof(uint8_t);
    break;
  case 3:
    return sizeof(int8_t);
    break;
  case 4:
    return sizeof(uint16_t);
    break;
  case 5:
    return sizeof(int16_t);
    break;
  case 6:
    return sizeof(int32_t);
    break;
  case 7:
    return sizeof(int16_t);
    break;
  case 10:
    /* 10 is FLOAT16, equal in size to uint16_t */
    return sizeof(uint16_t);
    break;
  case 11:
    return sizeof(double);
    break;
  case 12:
    return sizeof(uint32_t);
    break;
  case 13:
    return sizeof(uint64_t);
    break;
  default:
    log_fatal("could not calculate sizeof() for type {}\n", dtype);
    break;
  }
}

int Op::tensorproto_sizeof(const onnx::TensorProto *t) {
  if (!t->has_data_type()) {
    log_fatal("could not deduce type for tensor {}\n", t->name());
  }
  TPDT dtype = (TPDT)t->data_type();
  if (dtype == 0) {
    log_fatal("cannot  calculate sizeof() for tensor: {} of type UNDEFINED\n",
              t->name());
  }
  return tpdt_sizeof(dtype);
}

/* A tensor shape is valid if:
 *  1. it matches expected dims
 *  2. all but 0th dims are have a dim_value()
 */
bool Op::is_valid_tensor_shape(const onnx::TensorShapeProto &shape,
                               int expected_dims) {
  assert(shape.dim_size() == expected_dims &&
         "Value info expected conv dimensions to be 4");

  if (shape.dim_size() == expected_dims) {
    auto dims = shape.dim();
    if (!dims.at(0).has_dim_value()) {
      log_info("ValueInfoProto has params for Batch dimensions (not value):"
               " and the param is: {}\n",
               dims.at(0).dim_param());
    }
    std::for_each(dims.begin() + 1, dims.end(), [](auto &val) {
      ignore_unused(val);
      assert(val.has_dim_value() &&
             "Model could be missing shape information, consider running "
             "it through shape inference");
    });
    return true;
  }
  return false;
}

bool Op::dtype_eq(int32_t t1, TPDT t2) {
  TPDT ptr_dtype = static_cast<TPDT>(t1);
  return ptr_dtype == t2;
}

std::vector<int> Op::get_true_rc_weights(const Op::LayerBase *node) {
  std::vector<int> ret(2, 0);
  if (Op::isa<const Op::Layer::Gemm *>(node)) {
    const Op::Layer::Gemm *g = dynamic_cast<const Op::Layer::Gemm *>(node);
    if (g->m_cp.transB) {
      ret[TENSOR_2D_HEIGHT] = g->m_cp.wc;
      ret[TENSOR_2D_WIDTH] = g->m_cp.wr;
    } else {
      ret[TENSOR_2D_HEIGHT] = g->m_cp.wr;
      ret[TENSOR_2D_WIDTH] = g->m_cp.wc;
    }
  } else if (Op::isa<const Op::Layer::QGemm *>(node)) {
    const Op::Layer::QGemm *g = dynamic_cast<const Op::Layer::QGemm *>(node);
    if (g->m_cp.transB) {
      ret[TENSOR_2D_HEIGHT] = g->m_cp.wc;
      ret[TENSOR_2D_WIDTH] = g->m_cp.wr;
    } else {
      ret[TENSOR_2D_HEIGHT] = g->m_cp.wr;
      ret[TENSOR_2D_WIDTH] = g->m_cp.wc;
    }
  } else {
    log_fatal("dunno what typa gemm this ({}) is, mate\n", node->name);
  }
  return ret;
}

std::vector<int> Op::get_true_rc_inputs(const Op::LayerBase *node) {
  std::vector<int> ret(2, 0);
  if (Op::isa<const Op::Layer::Gemm *>(node)) {
    const Op::Layer::Gemm *g = dynamic_cast<const Op::Layer::Gemm *>(node);
    if (g->m_cp.transA) {
      ret[TENSOR_2D_HEIGHT] = g->input_dims[0][TENSOR_2D_WIDTH];
      ret[TENSOR_2D_WIDTH] = g->input_dims[0][TENSOR_2D_HEIGHT];
    } else {
      ret[TENSOR_2D_HEIGHT] = g->input_dims[0][TENSOR_2D_HEIGHT];
      ret[TENSOR_2D_WIDTH] = g->input_dims[0][TENSOR_2D_WIDTH];
    }
  } else if (Op::isa<const Op::Layer::QGemm *>(node)) {
    const Op::Layer::QGemm *g = dynamic_cast<const Op::Layer::QGemm *>(node);
    if (g->m_cp.transA) {
      ret[TENSOR_2D_HEIGHT] = g->input_dims[0][TENSOR_2D_WIDTH];
      ret[TENSOR_2D_WIDTH] = g->input_dims[0][TENSOR_2D_HEIGHT];
    } else {
      ret[TENSOR_2D_HEIGHT] = g->input_dims[0][TENSOR_2D_HEIGHT];
      ret[TENSOR_2D_WIDTH] = g->input_dims[0][TENSOR_2D_WIDTH];
    }
  } else {
    log_fatal("dunno what typa gemm this ({}) is, mate\n", node->name);
  }
  return ret;
}

std::vector<Op::LayerBase *> Op::Model::get_execution_order(void) const {
  return crt_exec_order(g);
}

std::vector<Op::LayerBase *> crt_exec_order(Op::Graph gcopy) {
  std::vector<Op::LayerBase *> execution_order;
  std::queue<Op::Vertex> S;
  S.push(Op::get_root_node(&gcopy));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    execution_order.push_back(gcopy[n]);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, gcopy)});
    }

    for (auto [src, dest] : edges_to_remove) {
      if (!Op::are_equal_nodes(src, dest, &gcopy)) {
        boost::remove_edge(src, dest, gcopy);
        if (boost::in_degree(dest, gcopy) == 0) {
          S.push(dest);
        }
      }
    }
  }
  return execution_order;
}

void Op::print_opgraph(Op::Graph gcopy) {
  std::queue<Op::Vertex> S;
  auto vitr = boost::vertices(gcopy);
  Op::Vertex v = *(vitr.first);
  S.push(v);

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::print_node(n, &gcopy);
    S.pop();

    auto out_edges = boost::out_edges(n, gcopy);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, gcopy)});
    }

    for (auto [src, dest] : edges_to_remove) {
      boost::remove_edge(src, dest, gcopy);
      if (boost::in_degree(dest, gcopy) == 0) {
        S.push(dest);
      }
    }
  }
}

/* Conv and its derivatives */
static std::vector<std::string> conv_like_tbl{"Conv", "QLinearConv"};

bool Op::is_conv_like(std::string op_type) {
  return std::find(conv_like_tbl.cbegin(), conv_like_tbl.cend(), op_type) !=
         conv_like_tbl.cend();
}

/* Gemm and its derivatives */
static std::vector<std::string> gemm_like_tbl{"QGemm", "Gemm"};

bool Op::is_gemm_like(std::string op_type) {
  return std::find(gemm_like_tbl.cbegin(), gemm_like_tbl.cend(), op_type) !=
         gemm_like_tbl.cend();
}

void Op::Model::summary(void) const { print_opgraph(g); }

Op::Graph Op::Model::get_graph() const { return g; }

Op::Graph &Op::Model::get_graph() { return g; }

Op::Neighbours Op::Model::get_neighbouring_vertices(Op::Vertex v) const {
  return boost::adjacent_vertices(v, g);
}

void Op::Model::add_to_constant_pool(onnx::NodeProto node) {
  for (const auto &i : node.output()) {
    constant_pool.insert({i, node});
  }
}

void Op::Model::add_to_name_node(onnx::NodeProto node) {
  name_node_map.insert({node.name(), node});
}

void Op::Parser::add_operator(onnx::NodeProto &node) {
  auto opt = node.op_type();
  if (opt == "Conv") {
    m_model.add(new Op::Layer::Conv(), node);
  } else if (opt == "Relu") {
    m_model.add(new Op::Layer::Relu(), node);
  } else if (opt == "Gemm") {
    m_model.add(new Op::Layer::Gemm(), node);
  } else if (opt == "MaxPool") {
    m_model.add(new Op::Layer::Maxpool(), node);
  } else if (opt == "Flatten") {
    m_model.add(new Op::Layer::Flatten(), node);
  } else if (opt == "Dropout") {
    m_model.add(new Op::Layer::Dropout(), node);
  } else if (opt == "Constant") {
    // do nothing, constants have already been added
  } else if (opt == "Clip") {
    m_model.add(new Op::Layer::Clip(), node);
  } else if (opt == "Add") {
    m_model.add(new Op::Layer::Eltwise(ELTWISE_ADD), node);
  } else if (opt == "Mul") {
    m_model.add(new Op::Layer::Eltwise(ELTWISE_MULT), node);
  } else if (opt == "Sub") {
    m_model.add(new Op::Layer::Eltwise(ELTWISE_SUB), node);
  } else if (opt == "GlobalAveragePool") {
    m_model.add(new Op::Layer::QLinearAveragePool(true), node);
  } else if (opt == "BatchNormalization") {
    m_model.add(new Op::Layer::BatchNorm(), node);
  } else if (opt == "ReorderOutput") {
    m_model.add(new Op::Layer::ReorderOutput(), node);
  } else if (opt == "Reshape") {
    m_model.add(new Op::Layer::Reshape(), node);
  } else if (opt == "QuantizeLinear") {
    m_model.add(new Op::Layer::QuantizeLinear(), node);
  } else if (opt == "QLinearConv") {
    m_model.add(new Op::Layer::QLinearConv(), node);
  } else if (opt == "DequantizeLinear") {
    m_model.add(new Op::Layer::DequantizeLinear(), node);
  } else if (opt == "QLinearMatMul") {
    m_model.add(new Op::Layer::QLinearMatMul(), node);
  } else if (opt == "QLinearAdd") {
    m_model.add(new Op::Layer::QLinearEltwise(ELTWISE_ADD), node);
  } else if (opt == "QLinearMul") {
    m_model.add(new Op::Layer::QLinearEltwise(ELTWISE_MULT), node);
  } else if (opt == "QLinearMul") {
    m_model.add(new Op::Layer::QLinearEltwise(ELTWISE_SUB), node);
  } else if (opt == "Transpose") {
    m_model.add(new Op::Layer::Transpose(), node);
  } else if (opt == "MatMul") {
    m_model.add(new Op::Layer::MatMul(), node);
  } else if (opt == "QGemm") {
    m_model.add(new Op::Layer::QGemm(), node);
  } else if (opt == "LogSoftmax") {
    m_model.add(new Op::Layer::LogSoftmax(), node);
  } else if (opt == "QLinearAveragePool") {
    m_model.add(new Op::Layer::QLinearAveragePool(), node);
  } else if (opt == "QLinearGlobalAveragePool") {
    m_model.add(new Op::Layer::QLinearAveragePool(true), node);
  } else if (opt == "Abs") {
    m_model.add(new Op::Layer::Abs(), node);
  } else if (opt == "ReduceMean") {
    m_model.add(new Op::Layer::ReduceMean(), node);
  } else if (opt == "AveragePool") {
    m_model.add(new Op::Layer::AveragePool(), node);
  } else if (opt == "Shape") {
    m_model.add(new Op::Layer::Shape(), node);
  } else if (opt == "Gather") {
    m_model.add(new Op::Layer::Gather(), node);
  } else if (opt == "Unsqueeze") {
    m_model.add(new Op::Layer::Unsqueeze(), node);
  } else if (opt == "Concat") {
    m_model.add(new Op::Layer::Concat(), node);
  } else if (opt == "NonMaxSuppression") {
    m_model.add(new Op::Layer::NMS(), node);
  } else {
    log_fatal("Unimplemented Operator: {}\n", opt);
  }
}

/* In onnx, all information relating to a node is not stored
 * in one place. Actual kernels are stored in initializers (TensorProto),
 * i/o shapes are stored in valueinfo, static shapes are stored in
 * attributes. The parser goes over the model in passes, collecting
 * information and storing it in a Op::Model object. Some passes
 * depend on other passes, therefore, order of execution of passes
 * matter.
 */
Op::Parser::Parser(std::string const &filename) {
  log_info2("Starting parser by opening {}\n", filename); 
  loaded_model.open(filename, std::ios::in | std::ios::binary);
  if (loaded_model.fail()) {
    log_fatal("{}: {}\n", filename, strerror(errno));
  }
  model_proto =
      google::protobuf::Arena::CreateMessage<onnx::ModelProto>(&arena);
  model_proto->ParseFromIstream(&loaded_model);
  const onnx::GraphProto &m_graph = model_proto->graph();

  log_info2("Saving graph outputs\n", filename); 
  pass_save_graph_outputs(m_graph);
  log_info2("Saving graph inputs\n", filename); 
  pass_save_graph_inputs(m_graph);
  log_info2("Saving value infos\n", filename); 
  pass_save_value_infos(m_graph);
  log_info2("Saving initializers\n", filename); 
  pass_save_initializers(m_graph);
  log_info2("Saving nodes\n", filename); 
  pass_save_nodes(m_graph);

  /* TODO: remove this, requires i/o part of all *Params structs to
   * be removed from the struct and all its users must use LayerBase
   * io */
  m_model.save_first_layer_input_dims(m_graph.input().at(0));
  
  std::vector<TPDT> input_types;
  for (const auto &i : m_graph.input()) {
    input_types.push_back(get_type_from_value_info(i));
  }
  log_info2("Starting Type Inference\n");
  m_model.deduce_types(input_types);
  /* first layer's input dims */
  google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> m_graph_inputs =
      m_graph.input();
  IVec2D input_dims;
  for (int i = 0; i < m_graph_inputs.size(); i++) {
    std::vector<int> tmp_dims = get_dims_from_value_info(m_graph.input()[i]);
    input_dims.push_back(tmp_dims);
  }

  log_info2("Starting Shape Inference\n");
  m_model.deduce_shapes(input_dims);
  log_info2("Setting devices\n");
  pass_set_device(get_graph());
  log_info2("Updating Registers through register allocator\n");
  m_model.update_registers();
  log_info2("Parsing Finished\n");
  m_model.save_input_output_names();
}

void Op::Parser::summary() const { m_model.bare_summary(); }
void Op::Parser::bare_summary() const { m_model.bare_summary(); }

std::vector<Op::LayerBase *> Op::Parser::get_execution_order(void) const{
  return m_model.get_execution_order();
}

Op::Graph Op::Parser::get_graph() const { return m_model.get_graph(); }

Op::Graph &Op::Parser::get_graph() { return m_model.get_graph(); }

TPDT Op::Parser::get_model_input_type(void) const {
  std::vector<Op::LayerBase *> order = get_execution_order();
  return order.at(0)->input_type[0];
}

TPDT Op::Parser::get_model_output_type(void) const {
  std::vector<Op::LayerBase *> order = get_execution_order();
  return order.at(order.size() - 1)->output_type[0];
}

/* get the maximum register that was ever used in the
 * model
 */
int Op::Parser::get_total_registers(void) const {
  std::vector<Op::LayerBase *> order = get_execution_order();

  int max = 0;
  for (Op::LayerBase *l : order) {
    max = std::max(max, *std::max_element(l->inputs.begin(), l->inputs.end()));
    max =
        std::max(max, *std::max_element(l->outputs.begin(), l->outputs.end()));
  }
  return max;
}

bool Op::Parser::has_graph_output(Op::LayerBase *l) const {
  return m_model.has_graph_output(l);
}

void Op::Parser::pass_save_graph_inputs(const onnx::GraphProto &graph) {
  const auto &graph_inputs = graph.input();
  for (const auto &i : graph_inputs) {
    m_model.save_graph_inputs(i);
  }
}
void Op::Parser::pass_save_graph_outputs(const onnx::GraphProto &graph) {
  const auto &graph_outputs = graph.output();
  for (const auto &i : graph_outputs) {
    m_model.save_graph_outputs(i);
  }
}
void Op::Parser::pass_save_value_infos(const onnx::GraphProto &graph) {
  const auto &value_infos = graph.value_info();
  for (int i = 0; i < value_infos.size(); ++i) {
    m_model.save_value_info(value_infos.at(i));
  }
}

void Op::Parser::pass_save_initializers(const onnx::GraphProto &graph) {
  const auto &initializers = graph.initializer();
  for (int i = 0; i < initializers.size(); ++i) {
    m_model.save_initializers(initializers.at(i));
  }
}
void Op::Parser::pass_save_nodes(const onnx::GraphProto &graph) {
  auto nodes = graph.node();
  /* add constants */
  for (const auto &i : nodes) {
    if (i.op_type() == "Constant") {
      m_model.add_to_constant_pool(i);
    } else {
      m_model.add_to_name_node(i);
    }
  }
  for (auto i : nodes) {
    add_operator(i);
  }
  for (int i = 0; i < nodes.size(); ++i) {
    if (nodes.at(i).op_type() == "Constant") {
      /* Skip Constants, deal with them by adding in the
       * "constant_pool"
       */
      continue;
    }
    m_model.connect(nodes.at(i));
  }
}

Op::Parser::~Parser() { loaded_model.close(); }

Op::RegisterAllocator::RegisterAllocator(Op::Graph g) {
  register_set.resize(default_size, 0);
  clear_regs(g);

  std::queue<Op::Vertex> S;
  S.push(get_root_node(&g));
  Op::Vertex n = S.front();
  Op::LayerBase *node = g[n];

  if (Op::is_root_node(n, &g)) {
    for (int i = 0; i < node->input_dims.size(); i++) {
      node->inputs.push_back(acquire(node->name));
    }
    for (int i = 0; i < node->output_dims.size(); i++) {
      node->outputs.push_back(acquire(node->name));
    }
    if (register_set.at(node->inputs.at(0)) == 1) {
      relinquish(node->inputs.at(0));
    }
  }

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *node = g[n];
    S.pop();

    auto out_edges = boost::out_edges(n, g);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, g)});
    }

    for (auto [src, dest] : edges_to_remove) {
      if (!Op::are_equal_nodes(src, dest, &g)) {
        traverse(&g, src, dest);
        boost::remove_edge(src, dest, g);
        if (boost::in_degree(dest, g) == 0) {
          S.push(dest);
        }
      }
    }
  }
}

Op::VirtualAddress
Op::RegisterAllocator::acquire(const std::string &node_name) {
  // find the first available register
  auto itr = std::find(register_set.begin(), register_set.end(), 0);
  if (itr != register_set.end()) {
    Op::VirtualAddress reg_num = itr - register_set.begin();
    ref(node_name, reg_num);
    return reg_num;
  } else {
    log_fatal("Out of registers!\n");
    return -1; // will never reach here
  }
}

void Op::RegisterAllocator::ref(const std::string &node_name,
                                Op::VirtualAddress a) {
  register_set.at(a) = string_hash(node_name);
}

void Op::RegisterAllocator::relinquish(Op::VirtualAddress a) {
  if (register_set.at(a) != 0) {
    register_set.at(a) = 0;
  }
}

void Op::RegisterAllocator::traverse(Op::Graph *g, Op::Vertex source,
                                     Op::Vertex target) {
  Op::LayerBase *src_node = (*g)[source];
  Op::LayerBase *dst_node = (*g)[target];

  dst_node->inputs.push_back(src_node->outputs.at(0));
  int od = boost::out_degree(source, *g);
  if (od == 1) {
    for (Op::VirtualAddress reg_val : src_node->inputs) {
      if (register_set.at(reg_val) == string_hash(src_node->name)) {
        relinquish(reg_val);
      }
    }
  }
  int id = src_node->inputs.size();
  if (id > 1 && od == 1) {
    /* relinquish all inputs unconditionally */
    for (Op::VirtualAddress reg_val : src_node->inputs) {
      relinquish(reg_val);
    }
  }

  if (dst_node->outputs.size() == 0) {
    dst_node->outputs.push_back(acquire(dst_node->name));
    ref(dst_node->name, dst_node->inputs.at(0));
  }
}

void Op::RegisterAllocator::clear_regs(Op::Graph g) {
  std::queue<Op::Vertex> S;
  S.push(get_root_node(&g));

  while (!S.empty()) {
    Op::Vertex n = S.front();
    Op::LayerBase *node = g[n];
    node->inputs.resize(0);
    node->outputs.resize(0);
    S.pop();

    auto out_edges = boost::out_edges(n, g);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, g)});
    }
    for (auto [src, dest] : edges_to_remove) {
      boost::remove_edge(src, dest, g);
      if (boost::in_degree(dest, g) == 0) {
        S.push(dest);
      }
    }
  }
}
