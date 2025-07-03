#pragma once

#include "onnx_parser.h"
#include "tensor.h"
#include <any>
#include <bitset>

#include "instructions.h"

#define check_overflow(value, bits)                                            \
  do {                                                                         \
    if (value >= (static_cast<int64_t>(1) << bits)) {                                                \
      log_fatal("value {} ({}) overflows a {} bit ({}) field\n", value,        \
                #value, bits, #bits);                                          \
    }                                                                          \
  } while (0)

class InitializerTable {
  std::unordered_map<std::string, uint32_t> tbl;
public:
  void push_back(const std::string& name, uint32_t addr);
  uint32_t get(const std::string& name);
};

using IOAddrPair =
    std::pair<std::vector<Op::VirtualAddress>, std::vector<Op::VirtualAddress>>;
using IOAddrTbl = std::map<std::string, IOAddrPair>;


/* Megablock and Miniblock
 *
 * All operators, implemented or not, can be divided into two sects: Megablock
 * and Miniblock
 *
 * Megablocks are a set of miniblocks that execute in a pipeline.  Input to a
 * megablock comes from dram and output from a megablock is written back to
 * dram. As miniblocks are arranged in a pipeline, input comes from a previous
 * miniblock. A megablock opener is the first miniblock of a pipeline. Only one
 * megablock can execute at a time. All miniblocks execute at the same time.
 *
 * Currently, (TODO: this should be updated later), there are two megablocks:
 * conv and fc and many miniblocks: relu, maxpool, bias, quantizer,
 * outputpipeline etc. When a convolution is happening, these miniblocks form a
 * megablock: conv, bias, quantizer, relu, maxpool, output When a FC is
 * happening, these miniblocks form a megablock: fc, bias, quantizer, relu,
 * output
 *
 * Some miniblocks can be skipped, for example, maxpool is skipped if a maxpool
 * op does not follow convolution.
 */

bool is_megablock(const Op::LayerBase *l);
bool is_miniblock(const Op::LayerBase *l);

/* This layer modifies the dimensions of its input */
bool changes_dimension_count(const Op::LayerBase *l);

/* InstGen generates according to the ISA
 *
 * It does this in multiple different passes passing over the execution order
 * as returned by parser. Instructions in the isa are compact, for example, the
 * tail instruction has information related to relu, quantization, batchnorm,
 * bias etc. On the other hand, onnx represents these as separate layers or as
 * a part of a layer corresponding to an entirely different instruction (for
 * example, bias info can be found in conv nodes). To deal with this, InstGen
 * generates the final instructions in a emit-merge strategy. Each node in onnx
 * emits all the instructions it is capable of in a InstBlob, later a pass over
 * InstBlob merges like instructions into one by ORing them together.
 *
 * Example: If an onnx graph contains CONV -> RELU -> MAXPOOL -> FC -> RELU,
 *
 * In the emit phase, these instructions will be generated (in order):
 *
 *  CONV, OutputBlock (from conv node), Tail (from bias), Tail (from relu),
 *  Tail (from maxpool) FC, OutputBlock (from fc node), Tail (from fc bias),
 *  Tail (from relu)
 *
 * In the merge phase, like instructions will be combined thusly to result in
 * these instructions:
 *
 *  CONV, OutputBlock (from conv node), Tail (bias, relu, maxpool),
 *  FC, OutputBlock (from fc node), Tail (fc bias, relu)
 *
 */

/* TODO: explain DWP */

class InstGen {
  InstBlob ret_inst;
  InitializerTable init_tbl;
  /* Records the io addresses for the chopped onnx graph based
   * on which instructions were generated
   */
  IOAddrTbl io_addr_tbl;
  /* Total bytes to be allocated including instructions, weights, io
   * data, and partial sum data
   */
  int total_model_size_cpu;
  int total_model_size_fpga;
  int total_dwp_packets;

  void insert_io_addr_tbl(Op::LayerBase *l);

public:
  InstGen(const Op::Parser &parser);
  InitializerTable get_tbl();
  InstBlob get_blob();
  IOAddrTbl get_io_addr_tbl();
  int model_size_cpu();
  int model_size_fpga();
  int dwp_packets();
};

/*
 * AddressGen generates addresses to be substituted in config instructions.
 * It does this by separating the address space (ideally all of the available
 * ram) in 4 distinct regions as shown below.
 *
 * +----------+---------------------+--------------------+--------------------+
 * |          |                     |                    |                    |
 * | Config   |  Weights & Biases   |    Input/Output    |    Accumulants     |
 * |          |                     |                    |                    |
 * +----------+---------------------+--------------------+--------------------+
 * 0                                                                         MAX
 *
 * Config starts at address 0 and its size is known a priori. Same for weights
 * and biases. Input/Output are final activations of layers i.e. intermidiate
 * values of the model and are stored in I/O region. Accumulants are
 * intermidiate values of a layer (as opposed to a model), they tend to be
 * greater in width than I/O (where I/O would be 8bit, Accumulants would be
 * 32bits), are stored in the final segement. Data in config region is allocated
 * all  at once, it fits all the instructions. Data is w/b region is allocated
 * on a FCFS basis. As a result, weights/biases for first layer to be executed
 * will come first in the ram. Data is I/O is allocated based on VirtualAddress
 * registers assigned to each LayerBase by RegisterAllocator. Data is
 * Accumulants is allocated in the same fashion as I/O but with a fixed offset
 * and data width.
 */

class AddressGen {
  /* pointer to the current address from which ram
   * addresses can be assigned
   */
  uint32_t current_address;
  /* Size (in words) occupied by inst region */
  int inst_region_size;
  int io_region_register_size;
  int weight_region_size;
  int max_io_reg;
  std::vector<Op::LayerBase *> m_exec_order;

  uint32_t ram_size_max;

  void addr_incr(uint32_t size);

  int get_total_instructions(const std::vector<Op::LayerBase *> &order);
  int get_io_region_register_size(const std::vector<Op::LayerBase *> &order);
  int get_weight_size(const std::vector<Op::LayerBase *> &order);
  int get_max_io_reg(const std::vector<Op::LayerBase *> &order);

public:
  AddressGen(Op::Graph graph);
  /* get a address in weights/bias region */
  uint32_t alloc(uint32_t size);
  /* get a address in io region */
  uint32_t io_addr_from_register(Op::VirtualAddress reg);
  /* get a address in accumulant region */
  uint32_t ps_addr_from_register(Op::VirtualAddress reg);
  int io_reg_size() const;
  int get_model_size_cpu() const;
  int get_model_size_fpga() const;
  std::vector<Op::LayerBase *> get_exec_order() const;
};

void pretty_print(const InstBlob &blob);
void pretty_print_html(const InstBlob &blob);


template <typename T1, typename T2>
std::vector<int> aligned_conv_weight_dims(const T1 &wdims, const T2 &idims) {
  assert(wdims.size() == 4);
  auto w = wdims;
  auto sa_arch = get_sa_arch();
  if (is_pointwise_conv(w)) {
    w[TENSOR_4D_BATCH] = ceil_mod(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
    w[TENSOR_4D_CHANNELS] = ceil_mod(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_ROW]);
  } else if (is_depthwise_conv(w, idims)) {
    w[TENSOR_4D_BATCH] = ceil_mod(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
    w[TENSOR_4D_CHANNELS] = ceil_mod(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_COLS]);
  } else {
    if (is_sa_regular_optimal(sa_arch)) {
      w[TENSOR_4D_BATCH] = ceil_mod(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_COLS]);
      w[TENSOR_4D_CHANNELS] = ceil_mod(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);
    } else {
      w[TENSOR_4D_BATCH] = ceil_mod(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
      w[TENSOR_4D_CHANNELS] = ceil_mod(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_COLS]);
    }
  }
  std::vector<int> ret(wdims.size());
  std::copy(w.begin(), w.end(), ret.begin());
  return ret;
}

inline int aligned_conv_weight(const Op::LayerBase *l) {
  auto sa_arch = get_sa_arch();
  int chan_itr = 0; int kern_itr = 0;
  std::tie(kern_itr, chan_itr) = l->get_iterations();
  int ret = kern_itr * chan_itr * prod(sa_arch);
  return ret;
}

template <typename T> int aligned_conv_bias(const T &dims) {
  assert(dims.size() == 1);
  auto sa_arch = get_sa_arch();
  int ret = ceil_mod(dims[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
  return ret;
}

/* out_mod here is the factor by which to pad the outputs of the
 * set of  systolic arrays. Consider an architecture with 9,4,4 arrangement.
 * In this case, the SA set will process 4 channels at a time. So, if the
 * output of a layer were to be (28,28), in total there'd be (28,28)x4
 * output elements emitted by the SA set. Since, we are generating
 * 28x28x4 at a time on-chip, this number should be aligned with WORD_SIZE
 *
 * In this case,
 *  Total output elements = (28x28x4) / 32
 *                        = (28x28) / 8
 */
inline int get_conv_out_mod() {
  auto sa_arch = get_sa_arch();
  return WORD_SIZE / sa_arch[SA_ARCH_N];
}

inline int get_conv_in_mod() {
  auto sa_arch = get_sa_arch();
  return WORD_SIZE / sa_arch[SA_ARCH_N];
}

/* accumulant_mod is calculated in a similar fashion. since, accumulators
 * are 32 bits i.e. 4 times the size of outputs (which are 8bits), we can
 * fit less of accumulatans in one DRAM dispatch. As a results, the output
 * mod is smaller.
 */
inline int get_conv_acc_mod() {
  auto sa_arch = get_sa_arch();
  int accumulant_mod = ((WORD_SIZE / sa_arch[SA_ARCH_N]) / (ACC_SIZE / 8));
  return accumulant_mod < 1 ? 1 : accumulant_mod;
}

template <typename T1, typename T2> IVec2D aligned_conv_input_dims(const T1 &dims, const T2 &wdims) {
  assert(!dims.empty() && dims[0].size() == 4);
  auto sa_arch = get_sa_arch();
  std::vector<int> i = dims[0];
  if (is_pointwise_conv(wdims)) {
    i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_ROW]);
  } else {
    i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);
  }
  IVec2D ret;
  ret.push_back(i);
  return ret;
}

template <typename T1, typename T2> int aligned_conv_input(const T1 &dims, const T2 &wdims) {
  auto iVec = aligned_conv_input_dims(dims, wdims);
  assert(!iVec.empty() && iVec[0].size() == 4);
  auto &i = iVec[0];
  int ret =
      ceil_mod(i[TENSOR_4D_WIDTH] * i[TENSOR_4D_HEIGHT], get_conv_in_mod()) *
      i[TENSOR_4D_CHANNELS];
  return ret;
}

template <typename T> IVec2D aligned_conv_output_dims(const T &dims) {
  assert(!dims.empty() && dims[0].size() == 4);
  auto sa_arch = get_sa_arch();
  std::vector<int> i = dims[0];
  i[TENSOR_4D_CHANNELS] = ceil_mod(i[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);
  IVec2D ret;
  ret.push_back(i);
  return ret;
}

template <typename T> int aligned_conv_output(const T &dims) {
  auto iVec = aligned_conv_output_dims(dims);
  assert(!iVec.empty() && iVec[0].size() == 4);
  auto &i = iVec[0];
  int ret =
      ceil_mod(i[TENSOR_4D_WIDTH] * i[TENSOR_4D_HEIGHT], get_conv_out_mod()) *
      i[TENSOR_4D_CHANNELS];
  return ret;
}

template <typename T> int aligned_conv_acc(const T &dims) {
  auto sa_arch = get_sa_arch();
  int ret =
      dims[TENSOR_4D_HEIGHT] * dims[TENSOR_4D_WIDTH] * sa_arch[1] * ACC_SIZE;
  ret = ceil_mod(ret, get_conv_acc_mod());
  return ret;
}

template <typename T> std::vector<int> aligned_fc_weight_dims(const T &dims) {
  assert(dims.size() == 2);
  auto va_size = get_va_size();
  auto w = dims;
  /* FIXME: introduce deduction transpose here */
  assert(WORD_SIZE == va_size && "not neccessary but needs fixing");
  w[0] = ceil_mod(w[0], WORD_SIZE);
  w[1] = ceil_mod(w[1], va_size);
  std::vector<int> ret{w[0], w[1]};
  return ret;
}

template <typename T> int aligned_fc_weight(const T &dims) {
  auto w = aligned_fc_weight_dims(dims);
  int ret = prod(w.begin(), w.end(), 1);
  ret = ceil_mod(ret, WORD_SIZE);
  return ret;
}

template <typename T> int aligned_fc_bias(const T &dims) {
  assert(dims.size() == 1);
  auto sa_arch = get_sa_arch();
  auto va_size = get_va_size();
  /* total bias is equal to the number of columns in the FC matrix,
   * so align first to va_size. since, bias addition is handled by
   * bias add blocks connected to the SA, there would be sa_cols
   * number of bias adds i.e. at a time, sa_cols number of bias
   * would be required. for example, a 9x6x6 architecture, there
   * biases will next be alinged to 6. now, since 6 alignement lead
   * to data being un-aligned to AXI_ADDR_WIDTH, also align it to
   * AXI_ADDR_WIDTH
   *
   * In total, there'll be 3 alignments: first wrt va_size, then wrt
   * sa_cols, then wrt AXI_ADDR_WIDTH
   */
  int ret = ceil_mod(dims[0], va_size);
  ret = ceil_mod(ret, sa_arch[SA_ARCH_COLS]);
  return ret;
}

template <typename T> IVec2D aligned_fc_io_dims(const T &dims) {
  assert(dims[0].size() == 2);
  assert(dims[0][0] == 1);
  int va_size = get_va_size();
  int ret = ceil_mod(dims[0][1], va_size);
  return IVec2D{{1, ret}};
}

template <typename T> int aligned_fc_io(const T &dims) {
  auto ret = aligned_fc_io_dims(dims);
  return ret[0][1];
}

template <typename T> std::vector<int> aligned_channels(const T &dims) {
  if (dims.size() != 4) {
    log_fatal("need dims to be 4-dimensional, got {}\n", dims.size());
  }
  auto sa_arch = get_sa_arch();
  std::vector<int> ret{
      dims.at(TENSOR_4D_BATCH),
      ceil_mod(dims.at(TENSOR_4D_CHANNELS), sa_arch[SA_ARCH_N]),
      dims.at(TENSOR_4D_HEIGHT), dims.at(TENSOR_4D_WIDTH)};
  return ret;
}

/* get nth byte (0 being LSB), of a */
template <typename T> inline char get_byte(T a, int n) {
  assert(n < sizeof(T) && n >= 0);
  char c = (a >> (n * 8)) & 0xff;
  return c;
}

template <std::size_t sz>
inline char get_byte(const std::bitset<sz> &a, int n) {
  assert(n < (sz / 8) && n >= 0);
  std::bitset<sz> c = (a >> (n * 8)) & std::bitset<sz>{0xff};
  return (char)c.to_ulong();
}

/* true if any of dims exceeds limits */
template <typename T>
inline bool is_out_of_bounds(const T &dims, const T &limit) {
  assert(dims.size() == limit.size() && "dims should be the same"
                                        " size as limits");
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] >= limit[i]) {
      return true;
    }
  }
  return false;
}

class BinBlob {

  template <typename T> void generic_append(T a);

public:

  char *m_data;
  /* total capacity */
  size_t m_size;
  /* byte wise index into data (current ptr) */
  size_t m_ptr;

  BinBlob(size_t size);
  ~BinBlob();
  void append(int a);
  void append(uint8_t a);
  void append(int8_t a);
  void append(uint32_t a);
  void append(float a);
  void append_dwp_header(uint32_t size, uint32_t addr);

  void append(const InstBlob &instblob, uint32_t addr);
  void append_zeroth_inst(uint32_t start_addr, uint32_t end_addr);

  size_t size() const;
  void print() const;
  void pretty_print() const;
  void write(const std::string &filename) const;

  char *get_data();
  const char *get_cdata() const;
  template <typename T> void append(const std::vector<T> &vec);
  template <typename T> void append(T i) = delete;
};

template <typename T> void BinBlob::generic_append(T a) {
  /* reverse iteration for big endian */
  for (int i = sizeof(T) - 1; i >= 0; --i) {
    char c = get_byte(a, i);
    m_data[m_ptr++] = c;
  }
}

template <typename T> void BinBlob::append(const std::vector<T> &vec) {
  assert(vec.size() > 0);
  assert(vec.size() * sizeof(vec[0]) <= (m_size - m_ptr));
  for (T i : vec) {
    generic_append(i);
  }
}

/* Prepares and optionally serializes gml model into
 * gml files
 */
class GmlGen {
  /* origin address */
  uint32_t m_org;

public:
  GmlGen(uint32_t org);
  BinBlob generate_gml(Op::Parser &parser);
};

class GmlCheck {
  void check_alignment(int addr) const;

public:
  GmlCheck(const InstBlob &instblob, const BinBlob &binblob);
  GmlCheck();
  void check_citr_kitr(const InstBlob &instblob) const;
  void check_addresses(const InstBlob &instblob) const;
  void check_weight_address_continuity(const InstBlob &instblob) const;
  int check_conv_weight_continuity(
      const std::bitset<INST_SIZE_BITS> &inst) const;
  int check_bias_continuity(const std::bitset<INST_SIZE_BITS> &inst) const;
  int check_fc_weight_continuity(const std::bitset<INST_SIZE_BITS> &inst) const;
  void check_fc_flatten(const InstBlob &instblob) const;
  void check_dwp(const BinBlob &binblob) const;
};

namespace Pass {

std::vector<Op::LayerBase *> remove_dqxq(Op::Graph graph);
Op::Graph reassign_registers(Op::Graph graph);
void absorb(Op::Graph &graph);

void adjust_scale_shift(Op::Graph graph);

void extract_conv_true_odims(Op::Graph graph);

void mark_cfg(const std::vector<Op::LayerBase *> &order);

InstBlob insert_start_inst(const InstBlob &insts);

Op::Graph create_megablock_graph(Op::Graph graph);

}; // namespace Pass

#define inst_get(bs, param)                                                    \
  (bitset_range_get<param##_COUNT>(bs, param##_LOW, param##_HIGH))

#define inst_set(bs, value, param) \
  (bitset_range_set(bs, std::bitset<param##_COUNT>{value}, param##_LOW, param##_HIGH))

int extract_opcode(const std::bitset<INST_SIZE_BITS> &inst);
/* true is opcode is a megablock */
bool is_megablock_op_code(int i);

/* all input/output tensors (this excludes weights+instructions packet)
 * have a DWP_HEADER as a start and a DWP_HEADER as end packet
 */
inline size_t io_tensor_packet_size(size_t tensor_size) {
  return tensor_size + (DWP_HEADER_BYTES * 2);
}

template <typename T>
void check_dwp_header(const T *data, size_t size, uint32_t expected_ds,
                      uint32_t expected_addr) {
  ignore_unused(size);
  assert(size >= DWP_HEADER_BYTES);
  uint32_t sop = bytes2int(data);
  uint32_t ds = bytes2int(data + 4);
  uint32_t hash = bytes2int(data + 8);

  if (sop != DWP_SOP) {
    log_fatal("expected DWP_SOP {}, got 0x{} from FPGA\n", DWP_SOP, sop);
  }
  if (ds != expected_ds) {
    log_fatal("expected_ds {}, got {}\n", expected_ds, ds);
  }
  if (hash != expected_addr) {
    log_fatal("expected_addr {}, got {}\n", expected_addr, hash);
  }
}

bool is_op_type(const Op::LayerBase *l, const char *op_type);
