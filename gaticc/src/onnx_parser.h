#pragma once
#include <bitset>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include "google/protobuf/arena.h"

#include "onnx.pb.h"
#include "tensor.h"
#include "utils.h"
#include "instructions.h"

/* Indices for accessing dimensions. I_BATCH should be read as
 * index for BATCH dimension
 *
 * To access of dimension array (dim[]) of size 4, use
 * dim[I_BATCH]. Instead of implicity assuming indices for
 * dimensions
 */
#define TENSOR_4D_BATCH 0
#define TENSOR_4D_CHANNELS 1
#define TENSOR_4D_HEIGHT 2
#define TENSOR_4D_WIDTH 3

#define TENSOR_2D_HEIGHT 0
#define TENSOR_2D_WIDTH 1

#define TENSOR_2D_ROWS 0
#define TENSOR_2D_COLS 1

#define SA_ARCH_ROW 0
#define SA_ARCH_COLS 1
#define SA_ARCH_N 2

/*  onnx represents padding with 4 co-ordinates, these are
 *  stored in clock-wise manner in a array starting LEFT,
 *  UP, RIGHT, DOWN
 */

#define I_LEFT 1
#define I_RIGHT 3
#define I_UP 0
#define I_DOWN 2

/*For NMS*/
#define I_NMS_INPUT_BOXES 0
#define I_NMS_INPUT_SCORES 1
#define I_INPUT_BOXES_COUNT 1
#define I_CLASSES_COUNT 1

using TPDT = onnx::TensorProto_DataType;
using InstBlob = std::vector<std::bitset<INST_SIZE_BITS>>;
using IVec2D = std::vector<std::vector<int>>;

enum DEVICES { DEVICE_UNKNOWN, DEVICE_CPU, DEVICE_FPGA };

/* forward declaration, definitions in instgen.{cpp,h}, rt.{cpp,h} */
class AddressGen;
class InitializerTable;
class BinBlob;
class TensorPool;
class Rah;

/* Onnx Parser external interface */
namespace Op {

struct ConvParams {
  int kn;        /* total number of kernels */
  int k[2];      /* kernel width/height */
  int pad[4];    /* padding across all four sides */
  int stride[2]; /* stride horizontally/vertically */
  int dilation[2];
  int ki;        /* Acts as an offset indicating the row from which to start reading the input for convolution */
};

struct GemmParams {
  int wr; /* weight rows */
  int wc; /* weight columns */
  float alpha;
  float beta;
  int transA;
  int transB;
};

struct PoolParams {
  int k[2];      /* kernel width/height */
  int pad[4];    /* padding across all four sides */
  int stride[2]; /* stride horizontally/vertically */
  int dilation[2];
  bool gbl;      /* global pooling */
};

using VirtualAddress = int;
using IOAddrPair =
    std::pair<std::vector<Op::VirtualAddress>, std::vector<Op::VirtualAddress>>;
using IOAddrTbl = std::map<std::string, IOAddrPair>;

struct LayerBase {
  std::string name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  virtual const char *op_type() const;
  /* Returns a pretty-formatted string of hyper-parameters that
   * the layer takes. Layers without any parameters may not override
   * this.
   */
  virtual std::string params() const;
  /* Initializers are onnx::TensorProto objects that contains
   * weights of a weighted layer (for eg, conv, gemm, batchnorm).
   * Classes that override this function should be weighted layers
   * that store a pointer to all the TensorProto they care about.
   * Not all layers have a TensorProto.
   * See Op::Layer::Conv for eg.
   */
  virtual void set_initializer_params(int n, const onnx::TensorProto &t);
  /* ValueInfos are onnx::ValueInfoProto objects that contain
   * shape/dimension information among other things of a node.
   * Classes that override this function should be layers that
   * care about these extra values.
   * Not all layers have a ValueInfoProto.
   * See Op::Layer::Conv for eg.
   */
  virtual void set_value_info_params(const onnx::ValueInfoProto &t);

  /* Attributes are static information such as kernel_shape,
   * strides, pads, dilations etc.
   */
  virtual void set_attributes(const onnx::NodeProto &node);

  /* Like set_initializer_params, but input is from a NodeProto */
  virtual void set_constant_params(int n, const onnx::NodeProto &node);

  virtual void run(TensorPool &tensor_pool);

  virtual void infer_shape(const IVec2D &input_dims);

  virtual void infer_type(const std::vector<TPDT> &input_types);

  /* push one or more INST_SIZE_BITS instruction in `insts`, how many
   * are decided by the override. return total dwp_packets required
   * by this instruction
   */
  virtual int get_inst(InstBlob &insts, AddressGen &gen, InitializerTable &tbl);
  /* push one or more opcodes corresponding to instructions
   * that this layer generates.
   * for example, Conv layer generates any where from
   * 3 or 4 instructions: conv,output,start or conv,output,tail,start
   * each override pushes opcodes corresponding to the instructions
   * that shall be generated that layer
   */
  virtual void get_opcodes(std::vector<int> &op_codes);

  /* Return total elements present in weights and biases
   * of each layer, aligned according to underlying
   * implementation engines such as systolic arrays or vector
   * arrays (FC). Used mostly by instruction generation routines
   */
  virtual uint32_t get_weight_size();

  /* aligned shapes are new shapes aligned with DRAM word
   * size and SA/VA sizes that a layer will posess
   * when executing on the fpga. Layers that do not
   * modify shape, will, for now, emit un-aligned dims
   */
  virtual IVec2D aligned_input() const;
  virtual IVec2D aligned_output() const;

  virtual void align_weights(BinBlob &blob, InitializerTable &tbl);

  virtual std::vector<float> get_output_scale(void);
  virtual void set_output_scale(const std::vector<float>& v);

  virtual std::pair<int,int> get_iterations() const;

  virtual void send_input(TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &io_tbl) const;

  std::vector<VirtualAddress> inputs;
  std::vector<VirtualAddress> outputs;

  /* Assertion: A node may have many inputs/outputs but all of the
   * same type
   */
  std::vector<TPDT> input_type;
  std::vector<TPDT> output_type;

  /* Dimensions of the input feature map */
  IVec2D input_dims;
  IVec2D output_dims;
  /* pipelined_output_dims is equal to output_dims in all
   * except when an operator that modifies the shape is
   * present in the pipeline. Operators such as maxpool
   * are an ideal case
   */
  IVec2D pipelined_output_dims;

  /* Device on which this node would be executed */
  int device;

  /* All nodes with a parameter should have a constructor to
   * initialize them. See conv for eg.
   */

  /* 1 if current node's outputs need to be received from the FPGA or
   * dumped by the simulator
   */
  bool dispatch;
};

namespace Layer {

struct NoOp : public LayerBase {
  const char *m_optype = "NoOp";
  const char *op_type() const override;
  NoOp();
  void get_opcodes(std::vector<int> &op_codes) override;
  int get_inst(InstBlob &insts, AddressGen &gen, InitializerTable &tbl) override;
  uint32_t get_weight_size() override;
  void run(TensorPool &tensor_pool) override;
};

struct Conv : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "Conv";
  TPDT weight_type;

  Conv();
  ConvParams m_cp;
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct Relu : public LayerBase {
  const char *m_optype = "Relu";
  const char *op_type() const override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
};

struct Clip : public LayerBase {
  const char *m_optype = "Clip";
  int m_min;
  int m_max;
  Clip();
  const char *op_type() const override;
  std::string params() const override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_constant_params(int n, const onnx::NodeProto &) override;
  int get_inst(InstBlob &insts, AddressGen &gen, InitializerTable &tbl) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  void run(TensorPool &tensor_pool) override;
};

struct Gemm : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "Gemm";
  TPDT weight_type;
  GemmParams m_cp;
  Gemm();
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
};

struct Maxpool : public LayerBase {
  const char *m_optype = "Maxpool";
  PoolParams m_cp;
  Maxpool();
  const char *op_type() const override;
  std::string params() const override;
  void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
};

struct Flatten : public LayerBase {
  const char *m_optype = "Flatten";
  const char *op_type() const override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
};

struct Dropout : public LayerBase {
  const char *m_optype = "Dropout";
  float drop;
  Dropout();
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct Eltwise : public LayerBase {
  const char *m_optype = "Eltwise";
  const onnx::TensorProto *constant_data;
  const int operator_type; 
  Eltwise(int op);
  const char *op_type() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct BatchNorm : public LayerBase {
  const char *m_optype = "BatchNorm";
  const char *op_type() const override;
  /* BatchNorm has static parameters namely epsilon and momentum.
   * These are not used during inference, hence the omission of
   * params() override.
   */
  float epsilon;
  float momentum;

  const onnx::TensorProto *scale;
  const onnx::TensorProto *B;
  const onnx::TensorProto *mean;
  const onnx::TensorProto *var;

  std::string params() const override;
  void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct ReorderOutput : public LayerBase {
  const char *m_optype = "ReorderOutput";
  const char *op_type() const override;
  /* TODO: this layer, what even is this? */
};

/* https://onnx.ai/onnx/operators/onnx__Reshape.html */
struct Reshape : public LayerBase {
  const char *m_optype = "Reshape";
  const char *op_type() const override;
  std::string params() const override;
  std::vector<int> new_shape;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;

  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
};

struct QuantizeLinear : public LayerBase {
  const char *m_optype = "QuantizeLinear";
  const char *op_type() const override;
  std::string params() const override;
  float scale;
  /* TODO: float8e etc types missing */
  std::variant<uint8_t, int8_t, uint16_t, int16_t> zero_point;
  int axis;
  int block_size;
  int output_dtype;
  int saturate;
  QuantizeLinear();
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  std::vector<float> get_output_scale(void) override;
  void set_output_scale(const std::vector<float>& v) override;
};

struct DequantizeLinear : public LayerBase {
  const char *m_optype = "DequantizeLinear";
  const char *op_type() const override;
  std::string params() const override;
  std::variant<float, double> scale;
  int zero_point;
  int axis;
  int block_size;
  DequantizeLinear();
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
  std::vector<float> get_output_scale(void) override;
  void set_output_scale(const std::vector<float>& v) override;
};

struct QLinearMatMul : public LayerBase {
  const onnx::TensorProto *weights;
  const char *m_optype = "QLinearMatMul";
  TPDT weight_type;
  GemmParams m_cp;
  QLinearMatMul();
  std::vector<float> a_scale;
  std::vector<float> b_scale;
  std::vector<float> y_scale;
  std::vector<std::variant<int8_t, uint8_t>> a_zero_point;
  std::vector<std::variant<int8_t, uint8_t>> b_zero_point;
  std::vector<std::variant<int8_t, uint8_t>> y_zero_point;
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
};

struct QLinearEltwise : public LayerBase {
  const onnx::TensorProto *constant_data;
  const int operator_type;
  float a_scale;
  float b_scale;
  int a_zp;
  int b_zp;
  std::vector<float> o_scale;
  std::vector<std::variant<int8_t, uint8_t>> zero_point;
  QLinearEltwise(int op);
  const char *m_optype = "QLinearEltwise";
  const char *op_type() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
  std::vector<float> get_output_scale(void) override;
  void set_output_scale(const std::vector<float>& v) override;
  IVec2D aligned_input() const override;
  IVec2D aligned_output() const override;
};

struct Transpose : public LayerBase {
  const onnx::TensorProto *addend;
  const char *m_optype = "Transpose";
  const char *op_type() const override;
  std::string params() const override;
  std::vector<int> perm;
  void set_attributes(const onnx::NodeProto &node) override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;

  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;

  void send_input(TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &io_tbl) const override;
};

struct MatMul : public LayerBase {
  const onnx::TensorProto *weights;
  const char *m_optype = "MatMul";
  GemmParams m_cp;
  MatMul();
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void run(TensorPool &tensor_pool) override;
};

struct QGemm : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  const char *m_optype = "QGemm";
  TPDT weight_type;
  TPDT bias_type;
  GemmParams m_cp;
  /* Occasionally, a conv follows a gemm, in such a case, the FPGA needs to
   * know this so convolution's output order can be transposed into a linear
   * order that gemm expects. Dims of said former conv layer, is stored
   * in this the vector 'former_layer_dims'.
   */
  std::vector<int> former_layer_dims;
  std::vector<float> a_scale;
  std::vector<float> b_scale;
  std::vector<float> y_scale;
  std::vector<std::variant<int8_t, uint8_t>> a_zero_point;
  std::vector<std::variant<int8_t, uint8_t>> b_zero_point;
  std::vector<std::variant<int8_t, uint8_t>> y_zero_point;
  QGemm();
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
  IVec2D aligned_input() const override;
  IVec2D aligned_output() const override;
  void align_weights(BinBlob &blob, InitializerTable &tbl) override;
  std::vector<float> get_output_scale(void) override;
  void set_output_scale(const std::vector<float>& v) override;
  std::pair<int,int> get_iterations() const override;
};

struct QLinearConv : public LayerBase {
  const onnx::TensorProto *weights;
  const onnx::TensorProto *bias;
  TPDT weight_type;
  const char *m_optype = "QLinearConv";
  QLinearConv();
  ConvParams m_cp;
  std::vector<float> x_scale;
  std::vector<std::variant<int8_t, uint8_t>> x_zero_point;
  std::vector<float> w_scale;
  std::vector<std::variant<int8_t, uint8_t>> w_zero_point;
  std::vector<float> y_scale;
  std::vector<std::variant<int8_t, uint8_t>> y_zero_point;
  const char *op_type() const override;
  std::string params() const override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  IVec2D aligned_input() const override;
  IVec2D aligned_output() const override;
  void align_weights(BinBlob &blob, InitializerTable &tbl) override;
  std::vector<float> get_output_scale(void) override;
  void set_output_scale(const std::vector<float>& v) override;
  std::pair<int,int> get_iterations() const override;
  void send_input(TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &io_tbl) const override;
};

struct LogSoftmax : public LayerBase {
  const char *m_optype = "LogSoftmax";
  const char *op_type() const override;
  std::string params() const override;
  int axis;
  LogSoftmax();
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void run(TensorPool &tensor_pool) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
};

struct QLinearAveragePool : public LayerBase {
  const char *m_optype = "QLinearAveragePool";
  PoolParams m_cp;
  QLinearAveragePool(bool gbl = 0);
  float x_scale;
  float y_scale;
  std::variant<uint8_t, int8_t> x_zero_points;
  std::variant<uint8_t, int8_t> y_zero_points;

  const char *op_type() const override;
  std::string params() const override;
  void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  uint32_t get_weight_size() override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
};

struct Abs : public LayerBase {
  const char *m_optype = "Abs";
  const char *op_type() const override;
  void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct ReduceMean : public LayerBase {
  const char *m_optype = "ReduceMean";
  const char *op_type() const override;
  std::string params() const override;

  int m_axis;
  int m_keepdims;

  ReduceMean();
  void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct AveragePool : public LayerBase {
  const char *m_optype = "AveragePool";
  PoolParams m_cp;
  AveragePool(bool gbl = 0);

  const char *op_type() const override;
  std::string params() const override;
  void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct Shape : public LayerBase {
  const char *m_optype = "Shape";

  const char *op_type() const override;
  // void run(TensorPool &tensor_pool) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

/* https://onnx.ai/onnx/operators/onnx__Gather.html */
struct Gather : public LayerBase {
  const char *m_optype = "Gather";

  Gather();
  int m_axis;
  Tensor<int> *m_indices;

  const char *op_type() const override;
  std::string params() const override;
  // void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

/* https://onnx.ai/onnx/operators/onnx__Unsqueeze.html */
struct Unsqueeze : public LayerBase {
  const char *m_optype = "Unsqueeze";

  std::vector<int> axis;

  const char *op_type() const override;
  // void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

/* https://onnx.ai/onnx/operators/onnx__Concat.html */
struct Concat : public LayerBase {
  const char *m_optype = "Concat";

  int m_axis;

  const char *op_type() const override;
  // void run(TensorPool &tensor_pool) override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const IVec2D &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
};

struct NMS : public LayerBase {
  const char *m_optype = "NonMaxSuppression";
  int64_t max_output_boxes;
  float iou_threshold;
  float score_threshold;
  int64_t center_point_box;

  NMS();
  const char *op_type() const override;
  std::string params() const override;
  void set_attributes(const onnx::NodeProto &node) override;
  void infer_shape(const std::vector<std::vector<int>> &input_dims) override;
  void infer_type(const std::vector<TPDT> &input_types) override;
  void set_initializer_params(int n, const onnx::TensorProto &t) override;
  void get_opcodes(std::vector<int> &op_codes) override;
  int get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) override;
  uint32_t get_weight_size() override;
};

} // namespace Layer

using Graph = boost::adjacency_list<boost::vecS, boost::listS,
                                    boost::bidirectionalS, LayerBase *>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using VertexIterator = Graph::vertex_iterator;
using AdjacencyIterator = Graph::adjacency_iterator;
using Neighbours = std::pair<Op::AdjacencyIterator, Op::AdjacencyIterator>;

/* Auxillary functions (no where else to put them...) */

const char *get_device_name(int device);
bool is_conv_like(std::string op_type);
bool is_gemm_like(std::string op_type);
void print_opgraph(Op::Graph gcopy);
bool is_root_node(Op::Vertex v, const Op::Graph *g);
bool are_equal_nodes(Op::Vertex v1, Op::Vertex v2, const Op::Graph *g);
void print_node(const LayerBase *node);
void print_node(Op::Vertex v, const Op::Graph *g);
Vertex get_root_node(const Op::Graph *g);
const char *get_tensorproto_dtype_name(TPDT type);
std::vector<int> get_tensorproto_shape(const onnx::TensorProto &t);
TPDT get_type_from_value_info(const onnx::ValueInfoProto &v);
TPDT get_type_from_tensor_proto(const onnx::TensorProto &v);
const onnx::TensorShapeProto &
get_tensor_shape_proto(const onnx::ValueInfoProto &t);
bool is_valid_tensor_shape(const onnx::TensorShapeProto &shape,
                           int expected_dims);
std::vector<int> get_dims_from_value_info(const onnx::ValueInfoProto &v);
IVec2D get_dims_of_in_edges(Op::Vertex v, const Op::Graph &g);
std::vector<TPDT>
get_types_of_in_edges(Op::Vertex v, const Op::Graph &g,
                      const std::vector<std::string> &i_nodes);
std::vector<std::string>
get_input_nodes(const onnx::NodeProto &np, const Op::Graph &g,
                const std::map<std::string, Op::Vertex> &output_map);
/* size in bytes */
int tensorproto_sizeof(const onnx::TensorProto *t);
/* size in bytes */
int tpdt_sizeof(TPDT v);
/* compare t1 and t2 */
bool dtype_eq(int32_t t1, TPDT t2);

std::vector<int> get_true_rc_weights(const Op::LayerBase *l);
std::vector<int> get_true_rc_inputs(const Op::LayerBase *l);

/* Return the total cycles required by the entire model */
long time_estimate(Op::Graph graph);

inline int sa_odims_row(Op::ConvParams const &cp,
                        const std::vector<int> &input_dims) {
  // o = ((iw - kw + 2p) / s) + 1
  return ((input_dims[TENSOR_4D_HEIGHT] - cp.k[TENSOR_2D_HEIGHT] +
           cp.pad[I_UP] + cp.pad[I_DOWN]) /
          cp.stride[TENSOR_2D_HEIGHT]) +
         1;
}

inline int sa_odims_cols(Op::ConvParams const &cp,
                         const std::vector<int> &input_dims) {
  return ((input_dims[TENSOR_4D_WIDTH] - cp.k[TENSOR_2D_WIDTH] + cp.pad[I_LEFT] +
           cp.pad[I_RIGHT]) /
          cp.stride[TENSOR_2D_WIDTH]) +
         1;
}

inline int mp_odims_row(Op::PoolParams const &cp,
                        const std::vector<int> &input_dims) {
  // o = ((iw - kw + 2p) / s) + 1
  return std::floor((input_dims[TENSOR_4D_HEIGHT] - cp.k[TENSOR_2D_HEIGHT] +
                     cp.pad[I_UP] + cp.pad[I_DOWN]) /
                    cp.stride[TENSOR_2D_HEIGHT]) +
         1;
}

inline int mp_odims_cols(Op::PoolParams const &cp,
                         const std::vector<int> &input_dims) {
  return std::floor((input_dims[TENSOR_4D_WIDTH] - cp.k[TENSOR_2D_WIDTH] +
                     cp.pad[I_LEFT] + cp.pad[I_RIGHT]) /
                    cp.stride[TENSOR_2D_WIDTH]) +
         1;
}

class Model {
  Op::Graph g;
  /* maps an output from a node its corresponding vertex in 'g' */
  std::map<std::string, Op::Vertex> name_vertex_map;
  std::map<std::string, Op::Vertex> output_map;
  std::map<std::string, const onnx::TensorProto &> initializer_map;
  std::map<std::string, const onnx::ValueInfoProto &> value_info_map;
  std::map<std::string, const onnx::ValueInfoProto &> graph_output_map;
  std::map<std::string, const onnx::ValueInfoProto &> graph_input_map;
  /* All 'Constants' in the onnx model are looked up using this table */
  std::map<std::string, onnx::NodeProto> constant_pool;
  std::map<std::string, onnx::NodeProto> name_node_map;


  bool is_graph_input(const std::string &s) const;
  bool is_graph_output(const std::string &s) const;
  bool is_initializer(const std::string &s) const;
  bool is_constant(const std::string &s) const;

  void set_input_type(const onnx::NodeProto &node, Op::LayerBase *l);
  void set_output_type(const onnx::NodeProto &node, Op::LayerBase *l);

  Op::Neighbours get_neighbouring_vertices(Op::Vertex v) const;

public:
  void update_registers(void);
  void deduce_types(const std::vector<TPDT> &input_types);
  void deduce_shapes(const IVec2D &input_dims);

  void save_graph_inputs(const onnx::ValueInfoProto &t);
  void save_graph_outputs(const onnx::ValueInfoProto &t);
  void save_value_info(const onnx::ValueInfoProto &t);
  void save_initializers(const onnx::TensorProto &t);
  void save_attribute(const onnx::NodeProto &node);
  void save_input_output_names();

  void add(LayerBase *layer, const onnx::NodeProto &node);
  void add_to_constant_pool(onnx::NodeProto node);
  void add_to_name_node(onnx::NodeProto node);
  void connect(const onnx::NodeProto &node);
  void save_first_layer_input_dims(const onnx::ValueInfoProto &t);

  /* return the topologically sorted graph (g)
   * used by LayerExecutors to execute layers
   */
  std::vector<Op::LayerBase *> get_execution_order(void) const;

  /* Print a summary of the network (traversed only through the
   * boost::vertices() of g) */
  void bare_summary(void) const;
  /* Print a summary of the network (traversed like a graph in topological
   * order) */
  void summary(void) const;

  size_t size(void);
  size_t size(void) const;

  /* true if 'l' has an output that is also a graph_output */
  bool has_graph_output(Op::LayerBase *l) const;

  Op::Graph get_graph() const;
  Op::Graph &get_graph();
};

class Parser {
  Model m_model;
  std::ifstream loaded_model;
  google::protobuf::Arena arena;
  onnx::ModelProto *model_proto;

  void add_operator(onnx::NodeProto &node);
  void pass_save_graph_inputs(const onnx::GraphProto &graph);
  void pass_save_graph_outputs(const onnx::GraphProto &graph);
  void pass_save_value_infos(const onnx::GraphProto &graph);
  void pass_save_initializers(const onnx::GraphProto &graph);
  void pass_save_nodes(const onnx::GraphProto &graph);
  void pass_set_device(Op::Graph gcopy);

public:
  Parser(std::string const &filename);
  void summary(void) const;
  void bare_summary(void) const;
  std::vector<Op::LayerBase *> get_execution_order(void) const;
  TPDT get_model_input_type(void) const;
  TPDT get_model_output_type(void) const;
  int get_total_registers(void) const;
  /* true if 'l' has an output that is also a graph_output */
  bool has_graph_output(Op::LayerBase *l) const;
  Op::Graph get_graph() const;
  Op::Graph &get_graph();
  ~Parser();
};

class RegisterAllocator {
  /* default size of the register set */
  const int default_size = 512;
  std::vector<int> register_set;

  void traverse(Op::Graph *g, Op::Vertex source, Op::Vertex target);
  VirtualAddress acquire(const std::string &node_name);
  void relinquish(VirtualAddress a);
  void ref(const std::string &node_name, VirtualAddress a);
  void clear_regs(Op::Graph g);

public:
  RegisterAllocator(Op::Graph g);
};

template <typename T> bool isa(const Op::LayerBase *l) {
  return dynamic_cast<T>(l) ? true : false;
}

Op::LayerBase *get_last_layer(const Op::Parser &parser);
} // namespace Op

std::vector<Op::Vertex> get_parents(Op::Vertex v, Op::Graph &g);
std::vector<Op::Vertex> get_children(Op::Vertex v, Op::Graph &g);

template <typename T> inline bool is_pointwise_conv(const T &dims) {
  if (dims[TENSOR_4D_HEIGHT] == 1 && dims[TENSOR_4D_WIDTH] == 1) {
    return true;
  }
  return false;
}

template <typename T1, typename T2> inline bool is_depthwise_conv(const T1 &dims, const T2 &input_dims) {
  if (dims[TENSOR_4D_CHANNELS] == 1 && input_dims[TENSOR_4D_CHANNELS] > 1) {
    return true;
  }
  return false;
}

template <typename T1, typename T2> inline bool is_regular_conv(const T1 &dims, const T2 &input_dims) {
  if (dims[TENSOR_4D_CHANNELS] == input_dims[TENSOR_4D_CHANNELS] && 
      dims[TENSOR_4D_WIDTH] * dims[TENSOR_4D_HEIGHT] > 1) {
    return true;
  }
  return false;
}

inline bool is_sa_regular_optimal(const std::vector<int>& sa_arch) {
  if (sa_arch[SA_ARCH_COLS] != sa_arch[SA_ARCH_N]) {
    return false;
  }
  return true; 
}

std::vector<Op::LayerBase *> crt_exec_order(Op::Graph gcopy);
std::vector<int> deduce_new_shape(std::vector<int> old_shape, int input_total_size);
