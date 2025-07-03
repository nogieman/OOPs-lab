#include "pch.h"

#include "executor.h"
#include "instgen.h"
#include "onnx_parser.h"
#include "optimization.h"
#include "sim.h"
#include "utils.h"
#include <queue>
#include <stack>

static std::set<std::string> miniblock_tbl{
  "QLinearConv",        "Relu", "Maxpool", "QGemm",     "Flatten",
  "QLinearAveragePool", "Conv", "Gemm",    "QLinearEltwise", "QLinearGlobalAveragePool", "NoOp", "Transpose"};

static std::set<std::string> megablock_tbl{
    "QLinearConv", "QGemm",      "Conv",
    "Gemm",        "QLinearEltwise", "NonMaxSuppression", "NoOp", "Transpose"};

static std::set<int> megablock_opcode_tbl{OP_CONV, OP_FC, OP_EltWise, OP_NMS, OP_TRANSPOSE};

bool is_miniblock(const Op::LayerBase *l) {
  auto itr = miniblock_tbl.find(std::string(l->op_type()));
  if (itr != miniblock_tbl.end()) {
    return true;
  }
  return false;
}

bool is_megablock(const Op::LayerBase *l) {
  auto itr = megablock_tbl.find(std::string(l->op_type()));
  if (itr != megablock_tbl.end()) {
    return true;
  }
  return false;
}

bool is_megablock_op_code(int i) {
  auto itr = megablock_opcode_tbl.find(i);
  if (itr != megablock_opcode_tbl.end()) {
    return true;
  }
  return false;
}

bool changes_dimension_count(const Op::LayerBase *l) {
  int c1 = l->input_dims[0].size();
  int c2 = l->output_dims[0].size();
  if (c1 != c2) {
    return true;
  }
  return false;
}

bool is_op_type(const Op::LayerBase *l, const char *op_type) {
  return std::strcmp(l->op_type(), op_type) == 0;
}

template <typename T> using CmpFunc = std::function<bool(T, T)>;

template <typename T> using CmpApplyFunc = std::function<T(T, T)>;

template <typename T>
static std::vector<T> collapse_identical_adjacent(const std::vector<T> &v,
                                                  CmpFunc<T> cmp,
                                                  CmpApplyFunc<T> cmp_apply) {
  std::stack<T> s;
  for (auto itr = v.rbegin(); itr != v.rend(); ++itr) {
    s.push(*itr);
  }
  std::vector<T> ret;
  ret.push_back(s.top());
  s.pop();
  while (s.empty() != true) {
    if (cmp(ret.at(ret.size() - 1), s.top())) {
      ret.push_back(s.top());
    } else {
      ret.at(ret.size() - 1) = cmp_apply(ret.at(ret.size() - 1), s.top());
    }
    s.pop();
  }
  return ret;
}

template <typename T> using FlagFunc = std::function<bool(T)>;

/* Insert `v` where `func` returns true */
template <typename T>
static std::vector<T> insert_inst(const std::vector<T> &v, FlagFunc<T> func,
                                  T val) {
  std::vector<T> ret;
  for (size_t i = 0; i < v.size(); ++i) {
    if (func(v.at(i)) && i != 0) {
      ret.push_back(val);
    }
    ret.push_back(v.at(i));
  }
  return ret;
}

static void connect_parents_to_children(const std::vector<Op::Vertex> &parents,
                                        const std::vector<Op::Vertex> &children,
                                        Op::Graph &g) {
  for (Op::Vertex i : parents) {
    for (Op::Vertex j : children) {
      boost::add_edge(i, j, g);
    }
  }
}

/* remove a vertex but connect its parents to its children */
static void safe_remove_vertex(Op::Vertex v, Op::Graph &g) {
  std::vector<Op::Vertex> src_vertices = get_parents(v, g);
  std::vector<Op::Vertex> dest_vertices = get_children(v, g);
  connect_parents_to_children(src_vertices, dest_vertices, g);
  boost::clear_vertex(v, g);
  boost::remove_vertex(v, g);
}

/* Take a subset of layers of the form 'dequantize -> x -> x -> ... -> *
 * quantize' from a model and remove dequantize and quantize from the top and
 * bottom x here are any layers that do not modify the data, or said another
 * way, have the same types for input/output. for example, relu, maxpool,
 * flatten
 */
std::vector<Op::LayerBase *> Pass::remove_dqxq(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  int cnt = 0;
  bool in_zone = false;

  for (next = vi; vi != vi_end; vi = next, cnt++) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (std::strcmp(l->op_type(), "DequantizeLinear") == 0 &&
        l->device == DEVICE_UNKNOWN) {
      in_zone = true;
      safe_remove_vertex(*vi, graph);
      continue;
    }
    if (in_zone) {
      if (std::strcmp(l->op_type(), "QuantizeLinear") == 0 &&
          l->device == DEVICE_UNKNOWN) {
        in_zone = false;
        safe_remove_vertex(*vi, graph);
        continue;
      }
      if (l->input_type[0] != l->output_type[0]) {
        log_fatal("could not remove layer {}\n", l->name);
      }
    }
  }
  return crt_exec_order(graph);
}

/* creates a graph with only megablocks connected to each other */
Op::Graph Pass::create_megablock_graph(Op::Graph graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);
  for (next = vi; vi != vi_end; vi = next) {
    next++;
    Op::LayerBase *l = graph[*vi];
    if (!is_megablock(l)) {
      safe_remove_vertex(*vi, graph);
    }
  }
  return graph;
}

Op::LayerBase *Op::get_last_layer(const Op::Parser &parser) {
  auto graph = parser.get_graph();
  auto mega_block = Pass::create_megablock_graph(graph);
  auto verts = boost::vertices(mega_block);
  for (auto it = verts.first; it != verts.second; ++it) {
    if (boost::out_degree(*it, mega_block) == 0) {
      return mega_block[*it];
    }
  }
}

/* addresses are only used by megablocks (i.e. blocks that directly
 * access dram). this pass calls the register allocator algorithm
 * on a modified graph that only contains megablocks
 */
Op::Graph Pass::reassign_registers(Op::Graph graph) {
  Op::Graph megablock_graph = create_megablock_graph(graph);
  Op::RegisterAllocator allocatr(megablock_graph);
  return megablock_graph;
}

/* Remove QLinearAdd nodes created during kernel decomposition.
   These are not needed as we uses accumulant addition in FPGA hardware. */
void Pass::absorb(Op::Graph &graph) {
  Op::VertexIterator vi, vi_end, next;
  std::tie(vi, vi_end) = boost::vertices(graph);

  for (next = vi; vi != vi_end; vi = next) {
    ++next;
    Op::Vertex v = *vi;
    Op::LayerBase *l = graph[v];

    if (l->op_type() == "QLinearEltwise" &&
        l->input_type[0] == onnx::TensorProto_DataType_INT32 ) {

      if (l->output_type[0] == onnx::TensorProto_DataType_INT8) {

        Op::Vertex last_parent;
        for (auto [in_begin, in_end] = boost::in_edges(v, graph);
             in_begin != in_end; ++in_begin) {
          last_parent = boost::source(*in_begin, graph);
        }

        Op::Vertex child =
            boost::target(*boost::out_edges(v, graph).first, graph);

        boost::add_edge(last_parent, child, graph);
      }

      boost::clear_vertex(v, graph);
      boost::remove_vertex(v, graph);
    }
  }
}

/* In onnx, a QLinearConv can be followed by Relu, Maxpool, etc.
 * These (miniblocks) are available only for float operations as
 * a result of which a QLinearConv's (or any other megablock's) output
 * (traditionally, int8/uint8) will be Dequantized to fp32, operated on relu,
 * maxpool etc. and requantized back to lower precision. This
 * dequantization-quantization introduces a shift in the values that the FPGA
 * must account for. We do this by consuming scale values from following dq-q
 * layers into QLinearConv's y_scale
 */
void Pass::adjust_scale_shift(Op::Graph graph) {
  std::queue<Op::Vertex> S;
  /* all nodes on which shape inference is done */
  std::unordered_set<Op::Vertex> done_set;
  auto vitr = boost::vertices(graph);
  Op::Vertex v = *(vitr.first);
  S.push(v);
  Op::LayerBase *latest_megablock = nullptr;
  if (is_megablock(graph[v])) {
    latest_megablock = graph[v];
  }
  done_set.insert(v);
  while (!S.empty()) {
    Op::Vertex n = S.front();
    S.pop();

    auto out_edges = boost::out_edges(n, graph);
    std::vector<std::pair<Op::Vertex, Op::Vertex>> edges_to_remove;
    for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
      edges_to_remove.push_back({n, boost::target(*itr, graph)});
    }

    for (auto [src, dest] : edges_to_remove) {
      /* make sure all parents of 'dest' have underwent infer_shape */
      auto in_edges = boost::in_edges(dest, graph);
      bool dest_parents_done = 1;
      for (auto itr = in_edges.first; itr != in_edges.second; ++itr) {
        Op::Vertex dsource = boost::source(*itr, graph);
        auto present = done_set.find(dsource);
        if (present == done_set.end()) {
          dest_parents_done = 0;
        }
      }

      if (dest_parents_done) {
        if (is_megablock(graph[dest])) {
          latest_megablock = graph[dest];
        } else {
          Op::LayerBase *l = graph[dest];
          if (is_op_type(l, "DequantizeLinear") && latest_megablock != nullptr && l->device != DEVICE_CPU) {
            std::vector<float> mega_scale = latest_megablock->get_output_scale();
            std::vector<float> dl_scale = broadcast_vec(l->get_output_scale(), mega_scale.size());
            std::vector<float> ret(mega_scale.size());
            for (size_t i = 0; i < mega_scale.size(); ++i) {
              ret.at(i) = mega_scale.at(i) / dl_scale.at(i);
            }
            latest_megablock->set_output_scale(ret);
          } else if (is_op_type(l, "QuantizeLinear") && latest_megablock != nullptr && l->device != DEVICE_CPU) {
            std::vector<float> mega_scale = latest_megablock->get_output_scale();
            std::vector<float> dl_scale = broadcast_vec(l->get_output_scale(), mega_scale.size());
            std::vector<float> ret(mega_scale.size());
            for (size_t i = 0; i < mega_scale.size(); ++i) {
              ret.at(i) = mega_scale.at(i) * dl_scale.at(i);
            }
            latest_megablock->set_output_scale(ret);
          }
        } 
        done_set.insert(dest);
        boost::remove_edge(src, dest, graph);
        if (boost::in_degree(dest, graph) == 0) {
          S.push(dest);
        }
      } else {
        S.push(n);
      }
    }
  }
}

/* Megablocks like convolution are followed by miniblocks
 * like relu and/or maxpool in pipeline. relu does not change
 * the shape of its outputs but maxpool does. in case, where
 * maxpool is present in the pipeline, convolution's true
 * output shape would be that of maxpool and not convolution
 *
 * this pass traverses a megablock's miniblock pipeline
 * to calculate and store the true output dims
 *
 * does a depth first traversal over nodes
 */
void Pass::extract_conv_true_odims(Op::Graph gcopy) {
  /* will contain megablock nodes */
  std::stack<Op::Vertex> candidates;
  std::set<Op::Vertex> discovered;

  Op::Vertex root = Op::get_root_node(&gcopy);
  Op::LayerBase *cc = nullptr;
  candidates.push(root);

  while (!candidates.empty()) {
    Op::Vertex v = candidates.top();
    Op::LayerBase *l = gcopy[v];

    if (is_op_type(l, "QLinearConv") || is_op_type(l, "QLinearEltwise")) {
      cc = l;
    } else if (is_megablock(l) || changes_dimension_count(l)) {
      cc = nullptr;
    } else if (cc != nullptr) {
      cc->pipelined_output_dims = l->output_dims;
    }

    candidates.pop();
    auto r = discovered.insert(v);
    if (r.second == true) { // v is undiscovered
      auto out_edges = boost::out_edges(v, gcopy);
      for (auto itr = out_edges.first; itr != out_edges.second; ++itr) {
        Op::Vertex v2 = boost::target(*itr, gcopy);
        candidates.push(v2);
      }
    }
  }
}

/* Find the pattern of layers conv -> flatten -> gemm and marks gemm with
 * details of conv
 */
void Pass::mark_cfg(const std::vector<Op::LayerBase *> &order) {
  bool flatten_pass = false;
  std::vector<int> former_layer_dims;
  for (Op::LayerBase *l : order) {
    if (is_op_type(l, "Flatten")) {
      if (l->input_dims[0].size() == 4) {
        flatten_pass = true;
        former_layer_dims = l->input_dims[0];
      } else {
        flatten_pass = false;
        former_layer_dims = std::vector<int>();
      }
    } else if (is_op_type(l, "QGemm")) {
      if (flatten_pass) {
        Op::Layer::QGemm *cc = dynamic_cast<Op::Layer::QGemm *>(l);
        cc->former_layer_dims = former_layer_dims;
        flatten_pass = false;
      } else {
        flatten_pass = false;
        former_layer_dims = std::vector<int>();
      }
    }
  }
}

void Op::Parser::pass_set_device(Op::Graph gcopy) {
  auto order = crt_exec_order(gcopy);
  /* prologue */
  int itr_frm_start = 0;
  for (; itr_frm_start < static_cast<int>(order.size()); ++itr_frm_start) {
    if (is_miniblock(order.at(itr_frm_start))) {
      break;
    } else {
      order.at(itr_frm_start)->device = DEVICE_CPU;
    }
  }
  int itr_from_end = order.size() - 1;
  for (; itr_from_end > 0; --itr_from_end) {
    if (is_miniblock(order.at(itr_from_end))) {
      break;
    } else {
      order.at(itr_from_end)->device = DEVICE_CPU;
    }
  }
  for (auto itr = itr_frm_start; itr <= itr_from_end; ++itr) {
    order.at(itr)->device = DEVICE_FPGA;
  }
}

int extract_opcode(const std::bitset<INST_SIZE_BITS> &inst) {
#ifndef NDEBUG
  /* assert if all opcodes are the same size */
  std::vector<int> all_opcodes{CONV_Opcode_COUNT, START_Opcode_COUNT,
                               FC_Opcode_COUNT, TailBlock_Opcode_COUNT,
                               OutputBlock_Opcode_COUNT};
  assert_all_equal(all_opcodes.data(), all_opcodes.size());
#endif
  return static_cast<int>(bitset_range_get<CONV_Opcode_COUNT, INST_SIZE_BITS>(
      inst, CONV_Opcode_LOW, CONV_Opcode_HIGH));
}

bool cmp_opcodes(std::bitset<INST_SIZE_BITS> i1,
                 std::bitset<INST_SIZE_BITS> i2) {
  int op1 = extract_opcode(i1);
  int op2 = extract_opcode(i2);
  return op1 != op2;
}

/* OR two instructions together, return the result */
std::bitset<INST_SIZE_BITS> or_inst(std::bitset<INST_SIZE_BITS> i1,
                                    std::bitset<INST_SIZE_BITS> i2) {
  std::bitset<INST_SIZE_BITS> ret = i1 | i2;
  return ret;
}


static std::bitset<INST_SIZE_BITS> gen_start_inst(int layer_num,
                                                  int total_layers) {
  std::bitset<INST_SIZE_BITS> start_inst;

  std::bitset<START_Opcode_COUNT> opcode{OP_START};
  bitset_range_set(start_inst, opcode, START_Opcode_LOW, START_Opcode_HIGH);

  std::bitset<START_LayerNumber_COUNT> lnum{layer_num};
  bitset_range_set(start_inst, lnum, START_LayerNumber_LOW,
                   START_LayerNumber_HIGH);

  std::bitset<START_TotalLayers_COUNT> tnum{total_layers};
  bitset_range_set(start_inst, tnum, START_TotalLayers_LOW,
                   START_TotalLayers_HIGH);

  return start_inst;
}

static int count_total_megablocks(const InstBlob &insts) {
  int cnt = 0;
  for (const auto &i : insts) {
    int opcode = extract_opcode(i);
    if (is_megablock_op_code(opcode)) {
      cnt++;
    }
  }
  return cnt;
}

static IVec2D aligned_qle_dims(const IVec2D &d) {
  IVec2D ret;
  auto sa_arch = get_sa_arch();
  for (int i = 0; i < d.size(); ++i) {
    auto dd = d.at(i);
    dd[TENSOR_4D_CHANNELS] = ceil_mod(dd[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);
    ret.push_back(dd);
  }
  return ret;
}

static std::vector<int> aligned_qle(const IVec2D& d) {
  auto ad = aligned_qle_dims(d);
  std::vector<int> ret;
  auto sa_arch = get_sa_arch();
  for (int i = 0; i < ad.size(); ++i) {
    int p = ad.at(i).at(TENSOR_4D_BATCH) * ad.at(i).at(TENSOR_4D_CHANNELS);
    p *= ceil_mod(ad.at(i).at(TENSOR_4D_HEIGHT) * ad.at(i).at(TENSOR_4D_WIDTH), get_conv_out_mod());
    ret.push_back(p);
  }
  return ret;
}

InstBlob Pass::insert_start_inst(const InstBlob &insts) {
  InstBlob ret;
  int total_layers = count_total_megablocks(insts);
  int layer_num = 1;
  for (size_t i = 0; i < insts.size(); ++i) {
    int op_code = extract_opcode(insts.at(i));
    if (is_megablock_op_code(op_code) && i != 0) {
      std::bitset<INST_SIZE_BITS> start_inst =
          gen_start_inst(layer_num, total_layers);
      layer_num++;
      ret.push_back(start_inst);
    }
    ret.push_back(insts.at(i));
  }
  std::bitset<INST_SIZE_BITS> last_start_inst = gen_start_inst(layer_num, total_layers);
  ret.push_back(last_start_inst);
  return ret;
}

InstGen::InstGen(const Op::Parser &parser) {
  /* TODO: redo this. consider making a new execution specific IR */
  Op::Graph graph = parser.get_graph();
  /* pass_reassign_registers is being called for its side-effect
   * which is the modification of LayerBase->{inputs,outputs} registers.
   */
  Pass::reassign_registers(graph);
  /* This function is called by its side-effect that adjusts
   * a megablocks' y_scale to account of shift introduced
   * by dequantize-quantize layers following a QLinearConv.
   */
  Pass::adjust_scale_shift(graph);

  AddressGen generator(graph);
  auto exec_order = generator.get_exec_order();
  if (gbl_args.has_option("print-exec-graph")) {
    std::cout << "== Execution Graph (in topological order) ==\n";
    for (const auto l : exec_order) {
      print_node(l);
      std::cout << '\n';
    }
  }
  total_model_size_cpu = generator.get_model_size_cpu();
  total_model_size_fpga = generator.get_model_size_fpga();
  /* Includes the instructions blob */
  total_dwp_packets = 1;

  Op::Graph megablock_graph = Pass::create_megablock_graph(graph);
  DispatchTable dispatch_table(megablock_graph);
  Op::RegisterAllocator allocatr(megablock_graph);

  if (gbl_args.has_option("print-megablock-graph")) {
    std::cout << "== Megablock Graph ==\n";
    auto megablock_order = crt_exec_order(megablock_graph);
    for (const auto l : megablock_order) {
      print_node(l);
      std::cout << '\n';
    }
  }

  InstBlob instructions;
  for (Op::LayerBase *l : exec_order) {
    /* push generated instructions and initializers to
     * 'instructions' and 'tbl' respectively
     */
    l->dispatch = dispatch_table.should_dispatch(l);
    int rr = l->get_inst(instructions, generator, init_tbl);
    total_dwp_packets += rr;
    insert_io_addr_tbl(l);
  }

  CmpFunc<std::bitset<INST_SIZE_BITS>> cmp = cmp_opcodes;
  CmpApplyFunc<std::bitset<INST_SIZE_BITS>> cmp_apply = or_inst;
  auto collapsed_insts =
      collapse_identical_adjacent(instructions, cmp, cmp_apply);
  ret_inst = Pass::insert_start_inst(collapsed_insts);
}

void InstGen::insert_io_addr_tbl(Op::LayerBase *l) {
  io_addr_tbl.insert({l->name, {l->inputs, l->outputs}});
}

InstBlob InstGen::get_blob() { return ret_inst; }

IOAddrTbl InstGen::get_io_addr_tbl() { return io_addr_tbl; }

InitializerTable InstGen::get_tbl() { return init_tbl; }

int InstGen::model_size_cpu() { return total_model_size_cpu; }

int InstGen::model_size_fpga() { return total_model_size_fpga; }

int InstGen::dwp_packets() { return total_dwp_packets; }

int Op::Layer::QuantizeLinear::get_inst(InstBlob &, AddressGen &,
                                        InitializerTable &) {
  return 0;
}

/* Generic gen_quant, used by conv and fc as their quantization routines
 * are same
 */
static std::bitset<INST_SIZE_BITS>
gen_quant(const std::vector<float> &x_scale, const std::vector<float> &w_scale,
          const std::vector<float> &y_scale,
          const std::vector<int> &zero_points) {
  std::bitset<INST_SIZE_BITS> quant_inst;
  std::vector<float> scales = compute_output_scale(x_scale, w_scale, y_scale);
  if (scales.size() != 1) {
    log_fatal("unsupported: per-channel quantization\n");
  }
  if (scales[0] == 0) {
    log_fatal("scales[0] = 0, need non-zero scales\n");
  }
  auto assert_zero = [](int i) {
    if (i != 0) {
      log_fatal("unsupported: non-zero zero-points\n");
    }
  };
  std::for_each(zero_points.begin(), zero_points.end(), assert_zero);

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(quant_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* TODO: deduce logically */
  float inverted_scale = 1 / scales[0];
  int shift_val = calc_shift_val(inverted_scale);
  int calib_scale = std::round((1 / scales[0]) * std::pow(2, shift_val));

  check_overflow(calib_scale, TailBlock_QuantScale_COUNT);
  std::bitset<TailBlock_QuantScale_COUNT> qscale{calib_scale};
  bitset_range_set(quant_inst, qscale, TailBlock_QuantScale_LOW,
                   TailBlock_QuantScale_HIGH);

  std::bitset<TailBlock_QuantShift_COUNT> qshift{shift_val};
  bitset_range_set(quant_inst, qshift, TailBlock_QuantShift_LOW,
                   TailBlock_QuantShift_HIGH);

  /* enable quant, ofcourse */
  std::bitset<TailBlock_QuantEn_COUNT> qen{1};
  bitset_range_set(quant_inst, qen, TailBlock_QuantEn_LOW,
                   TailBlock_QuantEn_HIGH);

  return quant_inst;
}

void decomp_inst(std::bitset<INST_SIZE_BITS> &conv_inst, const Op::Layer::QLinearConv *cc,
                 uint32_t &input_addr_start) {

  auto sa_arch = get_sa_arch();
  int pad_cnt = cc->m_cp.pad[I_LEFT];

  check_overflow(cc->m_cp.pad[I_UP], CONV_PadTop_COUNT);
  inst_set(conv_inst, (cc->m_cp.pad[I_UP] - (cc->m_cp.ki - 1)) < 0 ? 0 : (cc->m_cp.pad[I_UP] - (cc->m_cp.ki - 1)), CONV_PadTop);

  check_overflow(cc->m_cp.pad[I_DOWN], CONV_PadBottom_COUNT);
  inst_set(conv_inst, ((cc->m_cp.ki - 1) - cc->m_cp.pad[I_DOWN]) < 0 ? 0 : ((cc->m_cp.ki - 1) - cc->m_cp.pad[I_DOWN]), CONV_PadBottom);

  inst_set(conv_inst, ((cc->m_cp.ki - 1) - pad_cnt) < 0 ? 0 : (cc->m_cp.ki - 1) - pad_cnt, CONV_StartRowSkip);
  inst_set(conv_inst, (pad_cnt - (cc->m_cp.ki - 1)) < 0 ? 0 : pad_cnt - (cc->m_cp.ki - 1), CONV_EndRowSkip);

}

static std::bitset<INST_SIZE_BITS>
gen_conv_inst(const Op::Layer::QLinearConv *cc, AddressGen &gen,
              InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> conv_inst;
  inst_set(conv_inst, OP_CONV, CONV_Opcode);
  check_overflow(cc->input_dims[0][TENSOR_4D_WIDTH], CONV_IW_COUNT);
  inst_set(conv_inst, cc->input_dims[0][TENSOR_4D_WIDTH], CONV_IW);
  check_overflow(cc->input_dims[0][TENSOR_4D_HEIGHT], CONV_IH_COUNT);
  inst_set(conv_inst, cc->input_dims[0][TENSOR_4D_HEIGHT], CONV_IH);
  check_overflow(cc->input_dims[0][TENSOR_4D_CHANNELS], CONV_IC_COUNT);
  inst_set(conv_inst, cc->input_dims[0][TENSOR_4D_CHANNELS], CONV_IC);
  check_overflow(cc->weights->dims()[TENSOR_4D_CHANNELS], CONV_KC_COUNT);
  inst_set(conv_inst, cc->weights->dims()[TENSOR_4D_CHANNELS], CONV_KC);
  check_overflow(cc->m_cp.kn, CONV_KN_COUNT);
  inst_set(conv_inst, cc->m_cp.kn, CONV_KN);
  check_overflow(cc->m_cp.k[TENSOR_2D_WIDTH], CONV_KW_COUNT);
  inst_set(conv_inst, cc->m_cp.k[TENSOR_2D_WIDTH], CONV_KW);
  check_overflow(cc->m_cp.k[TENSOR_2D_HEIGHT], CONV_KH_COUNT);
  inst_set(conv_inst, cc->m_cp.k[TENSOR_2D_HEIGHT], CONV_KH);
  if (cc->m_cp.stride[TENSOR_2D_HEIGHT] != cc->m_cp.stride[TENSOR_2D_WIDTH]) {
    log_fatal("In layer {}, strides need to be symmetrical (same), got {}x{}\n",
              cc->name, cc->m_cp.stride[TENSOR_2D_HEIGHT],
              cc->m_cp.stride[TENSOR_2D_WIDTH]);
  }
  check_overflow(cc->m_cp.stride[TENSOR_2D_HEIGHT], CONV_Stride_COUNT);
  check_overflow(cc->m_cp.stride[TENSOR_2D_HEIGHT], CONV_Stride_COUNT);
  inst_set(conv_inst, cc->m_cp.stride[TENSOR_2D_HEIGHT], CONV_Stride);
  int pad_cnt = cc->m_cp.pad[I_LEFT];

  check_overflow(cc->m_cp.pad[I_LEFT], CONV_PadLeft_COUNT);
  inst_set(conv_inst, cc->m_cp.pad[I_LEFT], CONV_PadLeft);

  check_overflow(cc->m_cp.pad[I_RIGHT], CONV_PadRight_COUNT);
  inst_set(conv_inst, cc->m_cp.pad[I_RIGHT], CONV_PadRight);

  check_overflow(cc->m_cp.pad[I_UP], CONV_PadTop_COUNT);
  inst_set(conv_inst, cc->m_cp.pad[I_UP], CONV_PadTop);

  check_overflow(cc->m_cp.pad[I_DOWN], CONV_PadBottom_COUNT);
  inst_set(conv_inst, cc->m_cp.pad[I_DOWN], CONV_PadBottom);

  assert(cc->inputs.size() == 1);
  auto sa_arch = get_sa_arch();
  uint32_t input_addr_start = gen.io_addr_from_register(cc->inputs.at(0));
  uint32_t input_bytes =
      aligned_conv_input(cc->input_dims, cc->weights->dims()) * Op::tpdt_sizeof(cc->input_type[0]);
  uint32_t input_addr_end = input_addr_start + input_bytes;

  if (cc->m_cp.ki > 0) {
    decomp_inst(conv_inst, cc, input_addr_start);
  }

  uint32_t weight_bytes = aligned_conv_weight(cc) * Op::tensorproto_sizeof(cc->weights);
  uint32_t weight_addr_start = gen.alloc(weight_bytes);
  uint32_t weight_addr_end = ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);
  tbl.push_back(cc->weights->name(), weight_addr_start);
  inst_set(conv_inst, input_addr_start, CONV_ImageStartAddress);
  inst_set(conv_inst, input_addr_end, CONV_ImageEndAddress);
  inst_set(conv_inst, weight_addr_start, CONV_WeightStartAddress);
  inst_set(conv_inst, weight_addr_end, CONV_WeightEndAddress);

  if (!gbl_args.has_option("im2colbuf-size")) {
    log_fatal("--im2colbuf-size has to be provided. None found.\n");
  }
  int im2col_buf = gbl_args["im2colbuf-size"].as<int>();
  auto od = cc->output_dims.at(0);
  if (im2col_buf > od[TENSOR_4D_HEIGHT] * od[TENSOR_4D_WIDTH]) {
    inst_set(conv_inst, 1, CONV_Im2colPrefetch);
  }

  if (is_pointwise_conv(cc->weights->dims())) {
    inst_set(conv_inst, CONV_TYPE_PW, CONV_ConvType);
  } else if (is_depthwise_conv(cc->weights->dims(), cc->input_dims.at(0))) {
    inst_set(conv_inst, CONV_TYPE_DW, CONV_ConvType);
  } else {
    inst_set(conv_inst, CONV_TYPE_REGULAR, CONV_ConvType);
    if (!is_sa_regular_optimal(sa_arch)) {
      inst_set(conv_inst, 1, CONV_ChannelDuplicate);
    }
  }
  return conv_inst;
}

static std::bitset<INST_SIZE_BITS> gen_bias_inst(const Op::LayerBase* cc, 
                                                AddressGen& gen, 
                                                InitializerTable& tbl, 
                                                uint32_t bias_bytes,
                                                const onnx::TensorProto *bias) {
    std::bitset<INST_SIZE_BITS> bias_inst;

    uint32_t bias_addr_start = gen.alloc(bias_bytes);
    uint32_t bias_addr_end = ceil_mod(bias_addr_start + bias_bytes, WORD_SIZE);
    tbl.push_back(bias->name(), bias_addr_start);

    inst_set(bias_inst, OP_TailBlock, TailBlock_Opcode);
    inst_set(bias_inst, bias_addr_start, TailBlock_BiasStartAddress);
    inst_set(bias_inst, bias_addr_end, TailBlock_BiasEndAddress);
    inst_set(bias_inst, 1, TailBlock_BiasEn);

    int bias_width = Op::tensorproto_sizeof(bias) * 8;
    if (bias_width == 8 || bias_width == 32) {
        inst_set(bias_inst, bias_width, TailBlock_BiasWidth);
    } else {
        log_fatal("found an instruction with invalid bias width {} for layer {}\n", 
                  bias_width, cc->name);
    }

    return bias_inst;
}

static std::bitset<INST_SIZE_BITS> gen_conv_bias(const Op::Layer::QLinearConv* cc, 
                                                AddressGen& gen, 
                                                InitializerTable& tbl) {
    auto bias_dims = cc->bias->dims();
    uint32_t bias_bytes = aligned_conv_bias(bias_dims) * Op::tensorproto_sizeof(cc->bias);
    return gen_bias_inst(cc, gen, tbl, bias_bytes, cc->bias);
}

static std::bitset<INST_SIZE_BITS>
gen_output(uint32_t acc_addr, uint32_t out_addr, int citr, int kitr,
           int imgdimout, int imgdimacc, int accen, int dispatchen,
           int dispatchid, int onchip, int oh, int ow, int accreadfirst = 0, int flat_ctrl = 0) {
  std::bitset<INST_SIZE_BITS> output_inst;
  inst_set(output_inst, OP_OutputBlock, OutputBlock_Opcode);
  inst_set(output_inst, out_addr, OutputBlock_OutputAddr);
  inst_set(output_inst, acc_addr, OutputBlock_AccumulantAddr);
  inst_set(output_inst, citr, OutputBlock_ChannelItr);
  inst_set(output_inst, kitr, OutputBlock_KernelItr);
  inst_set(output_inst, imgdimout, OutputBlock_ImageDimOutput);
  inst_set(output_inst, imgdimacc, OutputBlock_ImageDimAcc);
  inst_set(output_inst, accen, OutputBlock_AccEn);
  inst_set(output_inst, accreadfirst, OutputBlock_AccumulantReadFirst);
  if (dispatchen) {
    inst_set(output_inst, 1, OutputBlock_DispatchEn);
    inst_set(output_inst, dispatchid, OutputBlock_DispatchID);
  }
  inst_set(output_inst, onchip, OutputBlock_OnChipAcc);
  inst_set(output_inst, oh, OutputBlock_OH);
  inst_set(output_inst, ow, OutputBlock_OW);
  inst_set(output_inst, flat_ctrl, OutputBlock_FlatController);
  return output_inst;
}

static std::bitset<INST_SIZE_BITS>
gen_conv_output(const Op::Layer::QLinearConv *cc, AddressGen &gen) {
  auto sa_arch = get_sa_arch();
  assert(cc->outputs.size() == 1);
  uint32_t acc_addr = gen.ps_addr_from_register(cc->inputs.at(0));
  uint32_t out_addr = gen.io_addr_from_register(cc->outputs.at(0));
  if (cc->m_cp.ki > 1) {
    acc_addr = gen.io_addr_from_register(cc->m_cp.ki);
  }

  auto odims = cc->output_dims.at(0);
  int citr = 0;
  int kitr = 0;
  if (is_regular_conv(cc->weights->dims(), cc->input_dims.at(0)) &&
      !is_sa_regular_optimal(sa_arch)) {
    kitr = ceil_div((int)cc->weights->dims(TENSOR_4D_BATCH),
                    (int)sa_arch[SA_ARCH_N]);
    citr = cc->weights->dims(TENSOR_4D_CHANNELS);
  } else {
    std::tie(kitr, citr) = cc->get_iterations();
  }

  auto pod = cc->pipelined_output_dims.at(0);
  int ido = ceil_mod(pod[TENSOR_4D_WIDTH] * pod[TENSOR_4D_HEIGHT],
                     get_conv_out_mod());
  int ida = ceil_mod(odims.at(TENSOR_4D_WIDTH) * odims.at(TENSOR_4D_HEIGHT),
                     get_conv_acc_mod());
  bool accen = true;
  bool accreadfirst = false;
  if (!is_sa_regular_optimal(sa_arch) && is_regular_conv(cc->weights->dims(), cc->input_dims.at(0))) {
    accen = true;
  } else if (cc->input_dims[0][TENSOR_4D_CHANNELS] <= sa_arch[SA_ARCH_N] ||
             is_depthwise_conv(cc->weights->dims(), cc->input_dims.at(0))) {
    accen = false;
  }

  if (cc->m_cp.ki > 1) {
    accen = true;
    accreadfirst = true;
  }
  int accbuf_size = 0;
  if (gbl_args.has_option("accbuf-size")) {
    /* division with ACC_SIZE/8 returns the depth of the acc fifo */
    accbuf_size = gbl_args["accbuf-size"].as<int>() / (ACC_SIZE / 8);
  } else {
    log_fatal("don't know accbuf-size, use option --accbuf-size to provide "
              "one\n");
  }
  int on_chip = 0;
  int acc_count = odims.at(TENSOR_4D_WIDTH) * odims.at(TENSOR_4D_HEIGHT);
  if (accbuf_size >= acc_count) {
    on_chip = 1;
  }
  int oh = odims.at(TENSOR_4D_HEIGHT);
  int ow = odims.at(TENSOR_4D_WIDTH);
  auto oi = gen_output(acc_addr, out_addr, citr, kitr, ido, ida, accen,
                       cc->dispatch, string_hash(cc->name), on_chip, oh, ow, accreadfirst);

  return oi;
}

static std::bitset<INST_SIZE_BITS>
gen_conv_quant(const Op::Layer::QLinearConv *cc, AddressGen &) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  return gen_quant(cc->x_scale, cc->w_scale, cc->y_scale, zero_points);
}

int Op::Layer::NoOp::get_inst(InstBlob &insts, AddressGen &,
                              InitializerTable &) {
  return 0;
}

int Op::Layer::QLinearConv::get_inst(InstBlob &insts, AddressGen &gen,
                                     InitializerTable &tbl) {
  auto conv_inst = gen_conv_inst(this, gen, tbl);
  /* there'll always be weights */
  int dwp_packets = 1;
  auto output_inst = gen_conv_output(this, gen);

  std::bitset<INST_SIZE_BITS> bias_inst;
  std::bitset<INST_SIZE_BITS> quant_inst;

  if (this->m_cp.ki > 0) {
    quant_inst.reset();
    inst_set(quant_inst, OP_TailBlock, TailBlock_Opcode);
  } else {
    quant_inst = gen_conv_quant(this, gen);
  }
  if (this->bias == nullptr) {
    bias_inst.reset();
    inst_set(bias_inst, OP_TailBlock, TailBlock_Opcode);
  } else {
    bias_inst = gen_conv_bias(this, gen, tbl);
  }

  int has_bias = inst_get(bias_inst, TailBlock_BiasEn);
  if (has_bias) {
    dwp_packets++;
  }

  /* order matters, be careful when messing with this */
  insts.push_back(conv_inst);
  insts.push_back(output_inst);
  insts.push_back(bias_inst);
  insts.push_back(quant_inst);
  return dwp_packets;
}

int Op::Layer::Relu::get_inst(InstBlob &insts, AddressGen &,
                              InitializerTable &) {
  std::bitset<INST_SIZE_BITS> relu_inst;
  inst_set(relu_inst, OP_TailBlock, TailBlock_Opcode);
  inst_set(relu_inst, 1, TailBlock_ActEn);
  inst_set(relu_inst, ACT_RELU, TailBlock_ActType);
  insts.push_back(relu_inst);
  return 0;
}

int Op::Layer::Maxpool::get_inst(InstBlob &insts, AddressGen &,
                                 InitializerTable &) {
  std::bitset<INST_SIZE_BITS> maxpool_inst;

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  inst_set(maxpool_inst, OP_TailBlock, TailBlock_Opcode);
  inst_set(maxpool_inst, 1, TailBlock_PoolEn);
  inst_set(maxpool_inst, POOL_MAX, TailBlock_PoolType);

  check_overflow(m_cp.k[TENSOR_2D_WIDTH], TailBlock_PoolWidth_COUNT);
  inst_set(maxpool_inst, m_cp.k[TENSOR_2D_WIDTH], TailBlock_PoolWidth);
  check_overflow(m_cp.k[TENSOR_2D_HEIGHT], TailBlock_PoolHeight_COUNT);
  inst_set(maxpool_inst, m_cp.k[TENSOR_2D_HEIGHT], TailBlock_PoolHeight);

  if (m_cp.stride[TENSOR_2D_HEIGHT] != m_cp.stride[TENSOR_2D_WIDTH]) {
    log_fatal("Strides need to be symmetric for layer {}\n", this->name);
  }
  check_overflow(m_cp.stride[TENSOR_2D_HEIGHT], TailBlock_PoolStride_COUNT);
  inst_set(maxpool_inst, m_cp.stride[TENSOR_2D_HEIGHT], TailBlock_PoolStride);

  int pad_cnt = m_cp.pad[I_LEFT];
  for (int i = 0; i < 4; ++i) {
    if (m_cp.pad[I_LEFT] != pad_cnt) {
      log_fatal("Pads for layer {} should all be equal\n", this->name);
    }
  }
  check_overflow(m_cp.pad[I_LEFT], TailBlock_PoolPadding_COUNT);
  inst_set(maxpool_inst, m_cp.pad[I_LEFT], TailBlock_PoolPadding);
  inst_set(maxpool_inst, input_dims[0][TENSOR_4D_HEIGHT] % m_cp.k[TENSOR_2D_HEIGHT], TailBlock_PoolModCount);
  inst_set(maxpool_inst, input_dims[0][TENSOR_4D_WIDTH] % m_cp.k[TENSOR_2D_WIDTH], TailBlock_PoolModCountCols);

  insts.push_back(maxpool_inst);
  return 0;
}

static std::bitset<INST_SIZE_BITS> gen_fc_inst(const Op::Layer::QGemm *cc, AddressGen &gen, InitializerTable &tbl) {
    std::bitset<INST_SIZE_BITS> fc_inst;
    inst_set(fc_inst, OP_FC, FC_Opcode);

    /* get the dimensions if transB is applied */
    std::vector<int> rows_cols = get_true_rc_weights(cc);
    check_overflow(rows_cols[0], FC_WeightRows_COUNT);
    inst_set(fc_inst, rows_cols[0], FC_WeightRows);
    check_overflow(rows_cols[1], FC_WeightCols_COUNT);
    inst_set(fc_inst, rows_cols[1], FC_WeightCols);
    std::vector<int> input_rows_cols = get_true_rc_inputs(cc);
    assert(input_rows_cols[0] == 1 && "input must be a vector");
    check_overflow(input_rows_cols[1], FC_InputRows_COUNT);
    if (!gbl_args.has_option("fcbuf-size")) {
        log_fatal("option --fcbuf-size missing from the command line, see help manual\n");
    }
    int fcbuf_size = gbl_args["fcbuf-size"].as<int>();
    if (input_rows_cols[1] > fcbuf_size) {
        log_fatal("In fc, input_row_size {}, exceeds provided FC input buffer size {}\n",
                  input_rows_cols[1], fcbuf_size);
    }
    inst_set(fc_inst, input_rows_cols[1], FC_InputRows);
    log_info("ignoring dropout constant while generating inst for QGemm\n");
    bool former_layer_conv = (cc->former_layer_dims.size() != 0);
    inst_set(fc_inst, former_layer_conv, FC_Flatten);
    int image_dims = 0;
    if (former_layer_conv) {
        image_dims = cc->former_layer_dims[TENSOR_4D_WIDTH] *
                    cc->former_layer_dims[TENSOR_4D_HEIGHT];
    }
    check_overflow(image_dims, FC_ImageDim_COUNT);
    inst_set(fc_inst, image_dims, FC_ImageDim);
    int vasize = get_va_size();
    int vec2mat_cols = 0;
    if (former_layer_conv) {
        IVec2D former_layer_dims_wrapper = {cc->former_layer_dims};
        vec2mat_cols = ceil_div(aligned_conv_output(former_layer_dims_wrapper), vasize);
    } else {
        vec2mat_cols = ceil_div(aligned_fc_io(cc->input_dims), vasize);
    }
    inst_set(fc_inst, vec2mat_cols, FC_Vec2MatCols);
    uint32_t input_addr_start = gen.io_addr_from_register(cc->inputs.at(0));
    uint32_t input_bytes = 0;
    if (cc->former_layer_dims.size() == 4) {
        IVec2D former_layer_dims_wrapper = {cc->former_layer_dims};
        input_bytes = aligned_conv_output(former_layer_dims_wrapper) *
                     Op::tpdt_sizeof(cc->input_type[0]);
    } else if (cc->former_layer_dims.size() == 0) {
        input_bytes = aligned_fc_io(cc->input_dims) * Op::tpdt_sizeof(cc->input_type[0]);
    } else {
        log_fatal("unknown size info in former layer dims of size {}, could potentially be dangerous \n",
                  cc->former_layer_dims.size());
    }
    uint32_t input_addr_end = ceil_mod(input_addr_start + input_bytes, WORD_SIZE);
    inst_set(fc_inst, input_addr_start, FC_ImageStartAddress);
    inst_set(fc_inst, input_addr_end, FC_ImageEndAddr);
    uint32_t weight_bytes = aligned_fc_weight(cc->weights->dims()) *
                           Op::tensorproto_sizeof(cc->weights);
    uint32_t weight_addr_start = gen.alloc(weight_bytes);
    uint32_t weight_addr_end = ceil_mod(weight_addr_start + weight_bytes, WORD_SIZE);
    tbl.push_back(cc->weights->name(), weight_addr_start);
    inst_set(fc_inst, weight_addr_start, FC_WeightStartAddress);
    inst_set(fc_inst, weight_addr_end, FC_WeightEndAddress);
    return fc_inst;
}

static std::bitset<INST_SIZE_BITS> gen_fc_output(const Op::Layer::QGemm *cc, AddressGen &gen) {
    assert(cc->outputs.size() == 1);
    uint32_t output_addr_start = gen.io_addr_from_register(cc->outputs.at(0));
    auto true_inputs = get_true_rc_weights(cc);
    int kern_itr = 0; int chan_itr = 0;
    std::tie(kern_itr, chan_itr) = cc->get_iterations();
    int va_size = get_va_size();
    auto sa_arch = get_sa_arch();
    int img_dim_output = va_size / sa_arch[SA_ARCH_N];

    return gen_output(
        0,                         // acc_addr (not used in FC, set to 0)
        output_addr_start,         // out_addr
        chan_itr,                  // citr
        kern_itr,                  // kitr
        img_dim_output,            // imgdimout
        0,                         // imgdimacc (not used in FC, set to 0)
        0,                         // accen (not used in FC, set to 0)
        cc->dispatch ? 1 : 0,      // dispatchen
        cc->dispatch ? string_hash(cc->name) : 0, // dispatchid
        0,                         // onchip (not used in FC, set to 0)
        cc->output_dims.at(0).at(TENSOR_2D_HEIGHT), // oh
        cc->output_dims.at(0).at(TENSOR_2D_WIDTH)   // ow
    );
}

static std::bitset<INST_SIZE_BITS> gen_fc_bias(const Op::Layer::QGemm* cc, 
                                              AddressGen& gen, 
                                              InitializerTable& tbl) {
    auto bias_dims = cc->bias->dims();
    uint32_t bias_bytes = aligned_fc_bias(bias_dims) * Op::tensorproto_sizeof(cc->bias);
    return gen_bias_inst(cc, gen, tbl, bias_bytes, cc->bias);
}

static std::bitset<INST_SIZE_BITS> gen_fc_quant(const Op::Layer::QGemm *cc,
                                                AddressGen &) {
  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->y_zero_point);
  return gen_quant(cc->a_scale, cc->b_scale, cc->y_scale, zero_points);
}

int Op::Layer::QGemm::get_inst(InstBlob &insts, AddressGen &gen,
                               InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> fc_inst = gen_fc_inst(this, gen, tbl);
  int dwp_packets = 1;
  std::bitset<INST_SIZE_BITS> output_inst = gen_fc_output(this, gen);
  std::bitset<INST_SIZE_BITS> bias_inst = gen_fc_bias(this, gen, tbl);
  std::bitset<INST_SIZE_BITS> quant_inst = gen_fc_quant(this, gen);

  int has_bias = inst_get(bias_inst, TailBlock_BiasEn);

  if (has_bias) {
    dwp_packets++;
  }

  insts.push_back(fc_inst);
  insts.push_back(output_inst);
  insts.push_back(bias_inst);
  insts.push_back(quant_inst);
  return dwp_packets;
}

int Op::Layer::Flatten::get_inst(InstBlob &, AddressGen &, InitializerTable &) {
  // TODO: ideally, flatten should be removed completely from the
  // graph and this function should not be present at all
  return 0;
}

int Op::Layer::DequantizeLinear::get_inst(InstBlob &, AddressGen &,
                                          InitializerTable &) {
  return 0;
}

void Op::Layer::QuantizeLinear::get_opcodes(std::vector<int> &) {}

void Op::Layer::QLinearConv::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_CONV);
  opcodes.push_back(OP_OutputBlock);
  if (bias != nullptr) {
    /* for bias */
    opcodes.push_back(OP_TailBlock);
  }
  /* for quantization */
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::Relu::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::DequantizeLinear::get_opcodes(std::vector<int> &) {}

void Op::Layer::Flatten::get_opcodes(std::vector<int> &) {}

void Op::Layer::Maxpool::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_TailBlock);
}


void Op::Layer::QGemm::get_opcodes(std::vector<int> &opcodes) {
  opcodes.push_back(OP_FC);
  opcodes.push_back(OP_OutputBlock);
  if (bias != nullptr) {
    opcodes.push_back(OP_TailBlock);
  }
  /* for quantization */
  opcodes.push_back(OP_TailBlock);
}

void Op::Layer::NoOp::get_opcodes(std::vector<int> &opcodes) {}

uint32_t Op::Layer::NoOp::get_weight_size() { return 0; }

uint32_t Op::Layer::Relu::get_weight_size() { return 0; }

uint32_t Op::Layer::Maxpool::get_weight_size() { return 0; }

uint32_t Op::Layer::Flatten::get_weight_size() { return 0; }

uint32_t Op::Layer::DequantizeLinear::get_weight_size() { return 0; }

uint32_t Op::Layer::QuantizeLinear::get_weight_size() { return 0; }

uint32_t Op::Layer::QLinearConv::get_weight_size() {
  uint32_t w = aligned_conv_weight(this) * Op::tensorproto_sizeof(weights);
  w = ceil_mod(w, WORD_SIZE);
  if (bias != nullptr) {
    uint32_t b = aligned_conv_bias(bias->dims()) * Op::tensorproto_sizeof(bias);
    b = ceil_mod(b, WORD_SIZE);
    return w + b;
  } else {
    return w;
  }
}

uint32_t Op::Layer::QGemm::get_weight_size() {
  uint32_t w =
      aligned_fc_weight(weights->dims()) * Op::tensorproto_sizeof(weights);
  w = ceil_mod(w, WORD_SIZE);

  uint32_t b = aligned_fc_bias(bias->dims()) * Op::tensorproto_sizeof(bias);
  b = ceil_mod(b, WORD_SIZE);

  return w + b;
}

void Op::Layer::LogSoftmax::get_opcodes(std::vector<int> &) {
  if (this->device != DEVICE_CPU) {
    log_fatal("Operator LogSoftmax can't run on the FPGA\n");
  }
}

uint32_t Op::Layer::LogSoftmax::get_weight_size() { return 0; }

int Op::Layer::LogSoftmax::get_inst(InstBlob &, AddressGen &,
                                    InitializerTable &) {
  if (this->device != DEVICE_CPU) {
    log_fatal("Operator LogSoftmax can't run on the FPGA\n");
  }
  return 0;
}

void Op::Layer::QLinearAveragePool::get_opcodes(std::vector<int> &op_codes) {
  op_codes.push_back(OP_TailBlock);
}

uint32_t Op::Layer::QLinearAveragePool::get_weight_size() {
  /* as average pool is a weight-less operation */
  return 0;
}

int Op::Layer::QLinearAveragePool::get_inst(InstBlob &insts, AddressGen &,
                                            InitializerTable &) {
  assert(this->device == DEVICE_FPGA);
  std::bitset<INST_SIZE_BITS> average_pool_inst;

  std::bitset<TailBlock_Opcode_COUNT> opcode{OP_TailBlock};
  bitset_range_set(average_pool_inst, opcode, TailBlock_Opcode_LOW,
                   TailBlock_Opcode_HIGH);

  /* enable relu */
  std::bitset<TailBlock_PoolEn_COUNT> poolen{1};
  bitset_range_set(average_pool_inst, poolen, TailBlock_PoolEn_LOW,
                   TailBlock_PoolEn_HIGH);

  if (this->output_dims.at(0).at(TENSOR_4D_HEIGHT) == 1 &&
      this->output_dims.at(0).at(TENSOR_4D_WIDTH) == 1) {
    std::bitset<TailBlock_PoolType_COUNT> pool_type{POOL_GLOBAL_AVG};
    bitset_range_set(average_pool_inst, pool_type, TailBlock_PoolType_LOW,
                     TailBlock_PoolType_HIGH);
  } else {
    std::bitset<TailBlock_PoolType_COUNT> pool_type{POOL_AVERAGE};
    bitset_range_set(average_pool_inst, pool_type, TailBlock_PoolType_LOW,
                     TailBlock_PoolType_HIGH);
  }

  std::bitset<TailBlock_PoolWidth_COUNT> pool_width{m_cp.k[TENSOR_2D_WIDTH]};
  bitset_range_set(average_pool_inst, pool_width, TailBlock_PoolWidth_LOW,
                   TailBlock_PoolWidth_HIGH);

  std::bitset<TailBlock_PoolHeight_COUNT> pool_height{m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(average_pool_inst, pool_height, TailBlock_PoolHeight_LOW,
                   TailBlock_PoolHeight_HIGH);

  assert_all_equal(m_cp.stride, 2);
  std::bitset<TailBlock_PoolStride_COUNT> pool_stride{
      m_cp.stride[TENSOR_2D_HEIGHT]};
  bitset_range_set(average_pool_inst, pool_stride, TailBlock_PoolStride_LOW,
                   TailBlock_PoolStride_HIGH);

  assert_all_equal(m_cp.pad, 4);
  std::bitset<TailBlock_PoolPadding_COUNT> pool_pad{m_cp.pad[I_LEFT]};
  bitset_range_set(average_pool_inst, pool_pad, TailBlock_PoolPadding_LOW,
                   TailBlock_PoolPadding_HIGH);

  std::bitset<TailBlock_PoolModCount_COUNT> modcount{
      input_dims[0][TENSOR_4D_HEIGHT] % m_cp.k[TENSOR_2D_HEIGHT]};
  bitset_range_set(average_pool_inst, modcount, TailBlock_PoolModCount_LOW,
                   TailBlock_PoolModCount_HIGH);

  insts.push_back(average_pool_inst);

  /* as average pool does not insert any dwp packets in the blob */
  return 0;
}

void Op::Layer::QLinearEltwise::get_opcodes(std::vector<int> &op_codes) {
  op_codes.push_back(OP_EltWise);
  op_codes.push_back(OP_OutputBlock);
  /* for quantization */
  op_codes.push_back(OP_TailBlock);
}

uint32_t Op::Layer::QLinearEltwise::get_weight_size() {
  log_warn("Treating QLinearEltwise as a weight-less operator consisting of "
           " only inputs and outputs\n");
  return 0;
}

static std::bitset<INST_SIZE_BITS> gen_eltwise(const Op::LayerBase *l,
                                               AddressGen &gen,
                                               InitializerTable &,
                                               int elt_type) {
  std::bitset<INST_SIZE_BITS> add_inst;
  inst_set(add_inst, OP_EltWise, EltWise_Opcode);
  inst_set(add_inst, elt_type, EltWise_EltType);
  if (l->inputs.size() < 2) {
    log_fatal("Need eltwise operator {} ({}) to have more than two inputs, "
              "found {} inputs\n",
              l->name, l->op_type(), l->inputs.size());
  }
  inst_set(add_inst, l->input_dims.at(0).at(TENSOR_4D_WIDTH), EltWise_IW);
  inst_set(add_inst, l->input_dims.at(0).at(TENSOR_4D_HEIGHT), EltWise_IH);
  inst_set(add_inst, l->input_dims.at(0).at(TENSOR_4D_CHANNELS), EltWise_IC);
  std::vector<int> ad = aligned_qle(l->input_dims);
  uint32_t left_start = gen.io_addr_from_register(l->inputs.at(0));
  uint32_t left_size = ad.at(0) * Op::tpdt_sizeof(l->input_type.at(0));
  uint32_t left_end = left_start + left_size;
  uint32_t right_start = gen.io_addr_from_register(l->inputs.at(1));
  uint32_t right_size = ad.at(1) * Op::tpdt_sizeof(l->input_type.at(1));
  uint32_t right_end = right_start + right_size;
  inst_set(add_inst, left_start, EltWise_LeftOperandStartAddress);
  inst_set(add_inst, left_end, EltWise_LeftOperandEndAddress);
  inst_set(add_inst, right_start, EltWise_RightOperandStartAddress);
  inst_set(add_inst, right_end, EltWise_RightOperandEndAddress);
  return add_inst;
}

static void gen_eltwise_input_quant(std::bitset<INST_SIZE_BITS> &add_inst,
                                    float a_scale, float b_scale, int a_zp,
                                    int b_zp) {
  int fp_ascale = fp_t(a_scale).raw();
  check_overflow(fp_ascale, EltWise_AScale_COUNT);
  inst_set(add_inst, fp_ascale, EltWise_AScale);
  int fp_bscale = fp_t(b_scale).raw();
  check_overflow(fp_bscale, EltWise_BScale_COUNT);
  inst_set(add_inst, fp_bscale, EltWise_BScale);
}

static std::bitset<INST_SIZE_BITS> gen_eltwise_output(const Op::LayerBase *l,
                                                      AddressGen &gen,
                                                      InitializerTable &) {
  std::bitset<INST_SIZE_BITS> output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  bitset_range_set(output_inst, ob_opcode, OutputBlock_Opcode_LOW,
                   OutputBlock_Opcode_HIGH);

  uint32_t output_addr_start = gen.io_addr_from_register(l->outputs.at(0));

  std::bitset<OutputBlock_OutputAddr_COUNT> ostart{output_addr_start};
  bitset_range_set(output_inst, ostart, OutputBlock_OutputAddr_LOW,
                   OutputBlock_OutputAddr_HIGH);

  std::bitset<OutputBlock_ChannelItr_COUNT> citr{1};
  bitset_range_set(output_inst, citr, OutputBlock_ChannelItr_LOW,
                   OutputBlock_ChannelItr_HIGH);

  auto sa_arch = get_sa_arch();
  int kernel_iterations = ceil_div(l->input_dims.at(0).at(TENSOR_4D_CHANNELS), sa_arch[SA_ARCH_N]);
  std::bitset<OutputBlock_KernelItr_COUNT> kitr{kernel_iterations};
  bitset_range_set(output_inst, kitr, OutputBlock_KernelItr_LOW,
                   OutputBlock_KernelItr_HIGH);

  auto pod = l->pipelined_output_dims.at(0);
  int image_dim_output = ceil_mod(pod[TENSOR_4D_WIDTH] * pod[TENSOR_4D_HEIGHT],
                                  get_conv_out_mod());

  std::bitset<OutputBlock_ImageDimOutput_COUNT> ido{image_dim_output};
  bitset_range_set(output_inst, ido, OutputBlock_ImageDimOutput_LOW,
                   OutputBlock_ImageDimOutput_HIGH);

  if (l->dispatch) {
    std::bitset<OutputBlock_DispatchEn_COUNT> dispatch_en{1};
    bitset_range_set(output_inst, dispatch_en, OutputBlock_DispatchEn_LOW,
                     OutputBlock_DispatchEn_HIGH);

    std::bitset<OutputBlock_DispatchID_COUNT> dispatch_id{string_hash(l->name)};
    bitset_range_set(output_inst, dispatch_id, OutputBlock_DispatchID_LOW,
                     OutputBlock_DispatchID_HIGH);
  }

  std::bitset<OutputBlock_OH_COUNT> oh_bs {l->output_dims.at(0).at(TENSOR_4D_HEIGHT)};
  bitset_range_set(output_inst, oh_bs, OutputBlock_OH_LOW, OutputBlock_OH_HIGH);

  std::bitset<OutputBlock_OW_COUNT> ow_bs {l->output_dims.at(0).at(TENSOR_4D_WIDTH)};
  bitset_range_set(output_inst, ow_bs, OutputBlock_OW_LOW, OutputBlock_OW_HIGH);

  return output_inst;
}

static std::bitset<INST_SIZE_BITS>
gen_eltwise_quant(const Op::Layer::QLinearEltwise *cc) {
  std::bitset<INST_SIZE_BITS> quant_inst;

  using variantT = std::variant<int8_t, uint8_t>;
  std::vector<int> zero_points = variant2vec<variantT, int>(cc->zero_point);
  std::vector<float> scales = cc->o_scale;
  if (scales.size() != 1) {
    log_fatal("unsupported: per-channel quantization\n");
  }
  if (scales[0] == 0) {
    log_fatal("scales[0] = 0, need non-zero scales\n");
  }
  auto assert_zero = [](int i) {
    if (i != 0) {
      log_fatal("unsupported: non-zero zero-points\n");
    }
  };
  std::for_each(zero_points.begin(), zero_points.end(), assert_zero);
  inst_set(quant_inst, OP_TailBlock, TailBlock_Opcode);

  float inverted_scale = 1 / scales[0];
  int shift_val = calc_shift_val(inverted_scale);
  int calib_scale = std::round((1 / scales[0]) * std::pow(2, shift_val));

  check_overflow(calib_scale, TailBlock_QuantScale_COUNT);
  inst_set(quant_inst, calib_scale, TailBlock_QuantScale);
  /* For Element wise operations, the intermidiate results are
   * FixedPoint on the FPGA. The addition of FIXED_POINT_SPLIT
   * to shift_val is essentially casting the result back to
   * int from FixedPoint 
   */
  int adjusted_shift_val = shift_val + FIXED_POINT_SPLIT;
  check_overflow(adjusted_shift_val, TailBlock_QuantShift_COUNT);
  inst_set(quant_inst, adjusted_shift_val, TailBlock_QuantShift);

  /* enable quant, ofcourse */
  inst_set(quant_inst, 1, TailBlock_QuantEn);
  return quant_inst;
}

/* Since EltWise is a megablock, i.e. it reads/writes to DRAM and is not
 * a part of any pipeline, get_inst() for QLinearAdd pushes multiple
 * instructions just like other megablocks like QLinearConv and QGemm
 */
int Op::Layer::QLinearEltwise::get_inst(InstBlob &blob, AddressGen &gen,
                                    InitializerTable &tbl) {
  assert(this->device == DEVICE_FPGA);
  auto add_inst = gen_eltwise(this, gen, tbl, this->operator_type);
  gen_eltwise_input_quant(add_inst, this->a_scale, this->b_scale, this->a_zp, this->b_zp);
  auto out_inst = gen_eltwise_output(this, gen, tbl);
  auto quant_inst = gen_eltwise_quant(this);
  blob.push_back(add_inst);
  blob.push_back(out_inst);
  blob.push_back(quant_inst);
  /* as qlinearadd does not insert any dwp packets in the blob */
  return 0;
}

int Op::Layer::Clip::get_inst(InstBlob &insts, AddressGen &gen, InitializerTable &tbl){
  std::bitset<INST_SIZE_BITS> clip_inst;
  inst_set(clip_inst, OP_TailBlock, TailBlock_Opcode);
  inst_set(clip_inst, 1, TailBlock_ActEn);
  inst_set(clip_inst, ACT_CLIP, TailBlock_ActType);
  inst_set(clip_inst, m_max, TailBlock_ActParam);
  insts.push_back(clip_inst);
  return 0;
}

void Op::Layer::Clip::get_opcodes(std::vector<int> &op_codes){
  op_codes.push_back(OP_TailBlock);
}

uint32_t Op::Layer::Clip::get_weight_size(){return 0;}

IVec2D Op::LayerBase::aligned_input() const { return input_dims; }

IVec2D Op::LayerBase::aligned_output() const { return output_dims; }

IVec2D Op::Layer::QLinearConv::aligned_input() const {
  return aligned_conv_input_dims(input_dims, this->weights->dims());
}

IVec2D Op::Layer::QLinearConv::aligned_output() const {
  return aligned_conv_output_dims(output_dims);
}

IVec2D Op::Layer::QGemm::aligned_input() const {
  return aligned_fc_io_dims(&input_dims[0]);
}

IVec2D Op::Layer::QGemm::aligned_output() const {
  return aligned_fc_io_dims(&output_dims.at(0));
}


IVec2D Op::Layer::QLinearEltwise::aligned_input() const {
  return aligned_qle_dims(input_dims);
}

IVec2D Op::Layer::QLinearEltwise::aligned_output() const {
  return aligned_qle_dims(input_dims);
}

std::pair<int,int> Op::Layer::QLinearConv::get_iterations() const {
  auto sa_arch = get_sa_arch();
  int kern_itr = 0; int chan_itr = 0;
  auto idims = this->input_dims.at(0);
  auto w = aligned_conv_weight_dims(this->weights->dims(), idims);
  if (is_pointwise_conv(w)) {
    kern_itr = ceil_div(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
    chan_itr = ceil_div(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_ROW]);
  } else if (is_depthwise_conv(this->weights->dims(), idims)) {
    kern_itr = ceil_div(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
    chan_itr = ceil_div(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_COLS]);
  } else {
    if (is_sa_regular_optimal(sa_arch)) {
      kern_itr = ceil_div(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_COLS]);
      chan_itr = ceil_div(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_N]);
    } else {
      /* just like depthwise */
      kern_itr = ceil_div(w[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
      chan_itr = ceil_div(w[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_COLS]);
    }
  }
  return std::pair(kern_itr, chan_itr);
}

std::pair<int,int> Op::Layer::QGemm::get_iterations() const {
  auto true_inputs = get_true_rc_weights(this);
  auto va_size = get_va_size();
  int kern_itr = ceil_div(true_inputs.at(TENSOR_2D_COLS), va_size);
  /* chan_itr always 1 for FC */
  return std::pair(kern_itr, 1);
}

void Op::Layer::Transpose::get_opcodes(std::vector<int> &op_codes) {
  op_codes.push_back(OP_TRANSPOSE); 
  op_codes.push_back(OP_OutputBlock); 
}

uint32_t Op::Layer::Transpose::get_weight_size() {
  return 0;
}

int Op::Layer::Transpose::get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) {
  std::bitset<INST_SIZE_BITS> tinst;
  inst_set(tinst, OP_TRANSPOSE, TRANSPOSE_Opcode);
  auto dims = aligned_channels(this->input_dims.at(0));
  inst_set(tinst, dims.at(TENSOR_4D_CHANNELS), TRANSPOSE_IC);
  inst_set(tinst, dims.at(TENSOR_4D_HEIGHT), TRANSPOSE_IH);
  inst_set(tinst, dims.at(TENSOR_4D_WIDTH), TRANSPOSE_IW);
  uint32_t addr = gen.io_addr_from_register(this->inputs.at(0));
  inst_set(tinst, addr, TRANSPOSE_ImageStartAddress);
  blob.push_back(tinst);

  uint32_t out_addr = gen.io_addr_from_register(this->outputs.at(0));
  int ido = ceil_mod(prod(dims), WORD_SIZE);
  auto odims = this->output_dims.at(0);
  std::bitset<INST_SIZE_BITS> oinst = gen_output(0, out_addr, 1, 1, ido, 0, 0, this->dispatch, string_hash(this->name), 0, odims.at(TENSOR_4D_HEIGHT), odims.at(TENSOR_4D_WIDTH), 0, 1);
  blob.push_back(oinst);
  return 0;
}

void Op::Layer::Reshape::get_opcodes(std::vector<int> &op_codes) {
  return;
}

uint32_t Op::Layer::Reshape::get_weight_size() {
  return 0;
}

int Op::Layer::Reshape::get_inst(InstBlob &blob, AddressGen &gen, InitializerTable &tbl) {
  return 0;
}

AddressGen::AddressGen(Op::Graph graph) : current_address{0} {
  m_exec_order = crt_exec_order(graph);

  Pass::extract_conv_true_odims(graph);
  Pass::mark_cfg(m_exec_order);

  if (!gbl_args.has_option("ramsize")) {
    log_fatal(
        "ramsize unknown, use option --ramsize to specify or see --help\n");
  }
  ram_size_max = gbl_args["ramsize"].as<int>() * 1024 * 1024;
  ram_size_max = ceil_mod(ram_size_max, WORD_SIZE);

  int total_instructions = get_total_instructions(m_exec_order);
  // std::cout << "total instructions " << total_instructions << '\n';
  /* size in bytes occupied by all instructions + one extra byte at the
   * top
   */
  inst_region_size =
      (total_instructions * (INST_SIZE_BITS / 8)) + (INST_SIZE_BITS / 8);

  io_region_register_size = get_io_region_register_size(m_exec_order);
  weight_region_size = get_weight_size(m_exec_order);

  max_io_reg = get_max_io_reg(m_exec_order);

  addr_incr(inst_region_size);
}

/* Calculate total instructions of size INST_SIZE_BITS
 *
 * Number of layers in a model != Total instructions
 * as some instructions, for example, tailblock contain
 * information corresponding to more than one layer
 */
int AddressGen::get_total_instructions(
    const std::vector<Op::LayerBase *> &order) {
  std::vector<int> op_codes;
  for (Op::LayerBase *l : order) {
    l->get_opcodes(op_codes);
  }
  auto cmp = [](int a, int b) -> bool { return a != b; };
  auto cmp_apply = [](int a, int) -> int { return a; };
  auto ret = collapse_identical_adjacent<int>(op_codes, cmp, cmp_apply);
  auto ret2 = insert_inst<int>(ret, is_megablock_op_code, OP_START);
  /* +1 for the last start instruction */
  return ret2.size() + 1;
}

int AddressGen::get_io_region_register_size(
    const std::vector<Op::LayerBase *> &order) {
  /* get largest dim in network */
  uint32_t largest_dim = 0;
  for (Op::LayerBase *l : order) {
    if (is_megablock(l)) {
      auto inp_dims = l->aligned_input()[0];
      uint32_t tmp_inp = prod(inp_dims.begin(), inp_dims.end(), 1) *
                         Op::tpdt_sizeof(l->input_type[0]);
      if (tmp_inp > largest_dim) {
        largest_dim = tmp_inp;
      }
      auto outp_dims = l->aligned_output()[0];
      uint32_t tmp_outp = prod(outp_dims.begin(), outp_dims.end(), 1) *
                          Op::tpdt_sizeof(l->output_type[0]);
      if (tmp_outp > largest_dim) {
        largest_dim = tmp_outp;
      }
    }
  }
  return largest_dim;
}

int AddressGen::get_weight_size(const std::vector<Op::LayerBase *> &order) {
  int sum = 0;
  for (Op::LayerBase *l : order) {
    sum += l->get_weight_size();
  }
  return sum;
}

void AddressGen::addr_incr(uint32_t size) {
  uint32_t i = ceil_mod(size, WORD_SIZE);
  if (current_address + i > ram_size_max) {
    log_fatal("OOM: cannot allocate memory of size {}, already occupied {}\n",
              size, current_address);
  }
  current_address += i;
}

uint32_t AddressGen::alloc(uint32_t size) {
  uint32_t ret = current_address;
  addr_incr(size);
  return ret;
}

uint32_t AddressGen::io_addr_from_register(Op::VirtualAddress reg) {
  uint32_t i =
      inst_region_size + weight_region_size + (reg * io_region_register_size);
  uint32_t ret = std::ceil((float)i / (float)WORD_SIZE) * WORD_SIZE;
  return ret;
}

int AddressGen::io_reg_size() const { return io_region_register_size; }

/* size in bytes occipied by inst and weight statically
 * while the model is being allocated on the cpu
 */
int AddressGen::get_model_size_cpu() const {
  int size = 0;
  size += inst_region_size;
  size += weight_region_size;
  return size;
}

/* size occupied on fpga is the size on the cpu i.e.
 * static model size (weights and instructions) +
 * dynamic size required for intermidiate inputs
 * and outputs
 */
int AddressGen::get_model_size_fpga() const {
  int size = get_model_size_cpu();
  size += (max_io_reg * io_region_register_size);
  size += io_region_register_size;
  return size;
}

std::vector<Op::LayerBase *> AddressGen::get_exec_order() const {
  return m_exec_order;
}

int AddressGen::get_max_io_reg(const std::vector<Op::LayerBase *> &order) {
  Op::VirtualAddress max_reg = 0;
  for (Op::LayerBase *l : order) {
    for (Op::VirtualAddress i : l->inputs) {
      if (i > max_reg) {
        max_reg = i;
      }
    }
    for (Op::VirtualAddress i : l->outputs) {
      if (i > max_reg) {
        max_reg = i;
      }
    }
  }
  return max_reg;
}

uint32_t AddressGen::ps_addr_from_register(Op::VirtualAddress reg) {
  uint32_t ps_base_addr = inst_region_size + weight_region_size +
                          ((max_io_reg + 1) * io_region_register_size);
  ps_base_addr = ceil_mod(ps_base_addr, WORD_SIZE);
  uint32_t ps_reg_offset = reg * (ACC_SIZE / 8) * io_region_register_size;
  uint32_t ps_reg_addr = ps_base_addr + ps_reg_offset;
  return ps_reg_addr;
}

void Table::clear() {
  tbl.clear();
  order.clear();
}

bool Table::is_empty() const { return tbl.empty() && order.empty(); }

/* bitset to hex */
template <std::size_t sz>
static std::string b2h(const std::bitset<sz> &binary) {
  std::stringstream hex_stream;
  hex_stream << std::hex << std::setfill('0');
  for (int i = sz - 1; i >= 0; i -= 8) {
    uint32_t value = 0;
    for (int j = i; j > (i - 8); --j) {
      value <<= 1;
      value |= binary[j];
    }
    hex_stream << std::setw(2) << value;
  }
  return hex_stream.str();
}

void pretty_print_inst_raw(const InstBlob &blob) {
  for (const auto &i : blob) {
    std::cout << b2h(i) << '\n';
  }
}

void pretty_print_html(const InstBlob &blob) {
  std::vector<pretty_data> data;
  pretty_data inst_data;
  for (const std::bitset<INST_SIZE_BITS> &i : blob) {
    pretty_print_html(i, data, inst_data);
  }
  generate_html(data, "pretty-print.html");
}

void pretty_print(const InstBlob &blob) {
  for (const std::bitset<INST_SIZE_BITS> &i : blob) {
    pretty_print(i);
    std::cout << '\n';
  }
}

void print_table(const Table &tbl) {
  std::map<std::string, int> maxes;
  for (const auto &i : tbl.tbl) {
    maxes.insert(
        {i.first, std::max((int)i.first.size(), count_digits(i.second))});
  }
  for (const auto &elem : tbl.order) {
#ifdef PRINT_COLOR
    std::cout << "\e[93m";
#endif
    std::cout << elem;
    int max = maxes[elem];
    if (static_cast<int>(elem.size()) < max) {
      for (int i = 0; i < (max - static_cast<int>(elem.size())); ++i) {
        std::cout << ' ';
      }
    }
    std::cout << '\t';
#ifdef PRINT_COLOR
    std::cout << "\e[39m";
#endif
  }
  std::cout << '\n';
  for (const auto &elem : tbl.order) {
    int elem_second = tbl.tbl.at(elem);
    std::cout << elem_second;
    int max = maxes[elem];
    if (count_digits(elem_second) < max) {
      for (int i = 0; i < (max - count_digits(elem_second)); ++i) {
        std::cout << ' ';
      }
    }
    std::cout << '\t';
  }
  std::cout << '\n';
}

void InitializerTable::push_back(const std::string& name, uint32_t addr) {
  tbl.insert({name, addr});
}

uint32_t InitializerTable::get(const std::string& name) {
  auto itr = tbl.find(name);
  if (itr == tbl.end()) {
    log_fatal("Could not find initializer {} in tbl\n", name);
  }
  return itr->second;
}

BinBlob::BinBlob(size_t size) {
  m_data = new char[size];
  m_size = size;
  m_ptr = 0;
}

BinBlob::~BinBlob() { delete[] m_data; }

void BinBlob::print() const {
  for (size_t i = 0; i < m_ptr; ++i) {
    std::cout << std::hex << m_data[i] << ' ';
  }
  std::cout << '\n';
}

void BinBlob::pretty_print() const {
}

void BinBlob::write(const std::string &filename) const {
  std::ofstream of(filename, std::ios::binary);
  of.write(m_data, m_ptr);
  of.close();
}

size_t BinBlob::size() const { return m_ptr; }

void BinBlob::append(int a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(uint32_t a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(uint8_t a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(int8_t a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  generic_append(a);
}

void BinBlob::append(float a) {
  assert(sizeof(a) <= (m_size - m_ptr));
  uint8_t bytes[sizeof(a)];
  std::memcpy(bytes, &a, sizeof(a));
  for (int i = 0; i < sizeof(a); ++i) {
    generic_append(bytes[i]);
  }
}

void BinBlob::append_dwp_header(uint32_t size, uint32_t addr) {
  uint32_t dwp_sop = DWP_SOP;
  append(dwp_sop);
  append(size);
  append(addr);
}

void BinBlob::append(const InstBlob &instblob, uint32_t addr) {
  uint32_t payload_size = (instblob.size() + 1) * (INST_SIZE_BITS / 8);
  append_dwp_header(payload_size, addr);

  assert(payload_size > 0);
  assert(payload_size <= (m_size - m_ptr));
  /* add the zeroth instruction itself */
  uint32_t inst_start = GATI_INST_ORG + (INST_SIZE_BITS / 8);
  append_zeroth_inst(inst_start, payload_size);
  for (const auto &inst : instblob) {
    generic_append(inst);
  }
}

template <typename T>
static void sa_align_aux_regular(const Op::LayerBase *l, BinBlob &blob, const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  auto strides = tensor->get_strides();
  auto aligned_dims = aligned_conv_weight_dims(dims, l->input_dims.at(0));
  auto sa_arch = get_sa_arch();
  auto aligned_size = aligned_conv_weight(l) * sizeof(T);
  auto deficit_zeros = ceil_mod(aligned_size, WORD_SIZE) - aligned_size;
  T zero = 0;
  if (aligned_dims[TENSOR_4D_HEIGHT] * aligned_dims[TENSOR_4D_WIDTH] >
      sa_arch[SA_ARCH_ROW]) {
    log_fatal(
        "not enough rows in sa for this convolution of kernel size {},{}\n",
        aligned_dims[TENSOR_4D_HEIGHT], aligned_dims[TENSOR_4D_WIDTH]);
  }
  assert(WORD_SIZE % 4 == 0);
  int chan_dim = 0; int kern_dim = 0;
  if (is_depthwise_conv(dims, l->input_dims.at(0))) {
    kern_dim = sa_arch[SA_ARCH_N];
    chan_dim = sa_arch[SA_ARCH_COLS];
  } else {
    if (is_sa_regular_optimal(sa_arch)) {
      kern_dim = sa_arch[SA_ARCH_COLS];
      chan_dim = sa_arch[SA_ARCH_N];
    } else {
      kern_dim = sa_arch[SA_ARCH_N];
      chan_dim = sa_arch[SA_ARCH_COLS];
    }
  }
  int kern_iterations = 0; int chan_iterations = 0;
  std::tie(kern_iterations, chan_iterations) = l->get_iterations();

  int count = 0;
  for (int kern = 0; kern < kern_iterations; ++kern) {
    for (int chan = 0; chan < chan_iterations; ++chan) {
      for (int srow = sa_arch[SA_ARCH_ROW] - 1; srow >= 0; srow--) {
        for (int schan = 0; schan < chan_dim; schan++) {
          for (int skern = 0; skern < kern_dim; skern++) {
            int k = kern * kern_dim + skern;
            int c = chan * chan_dim + schan;
            if (srow >= dims[TENSOR_4D_HEIGHT] * dims[TENSOR_4D_WIDTH] ||
                c >= dims[TENSOR_4D_CHANNELS] ||
                k >= dims[TENSOR_4D_BATCH]) {
              blob.append(zero);
            } else {
              int index = k * strides[0] + c * strides[1] + srow;
              blob.append(tensor->at(index));
            }
            count++;
          }
        }
      }
    }
  }
  for (decltype(deficit_zeros) i = 0; i < deficit_zeros; ++i) {
    blob.append(zero);
    count++;
  }
  log_info2("Inserted {} elements\n", count * sizeof(T));
}
template <typename T>
static void sa_align_aux_pointwise(const Op::LayerBase *l, BinBlob &blob, const Tensor<T> *tensor) {
  auto sa_arch = get_sa_arch();
  auto dims = tensor->get_dims();
  auto aligned_dims = aligned_conv_weight_dims(dims, l->input_dims.at(0));
  int kern_itr = ceil_div(aligned_dims[TENSOR_4D_BATCH], sa_arch[SA_ARCH_N]);
  int chan_itr = ceil_div(aligned_dims[TENSOR_4D_CHANNELS], sa_arch[SA_ARCH_ROW]);
  auto strides = tensor->get_strides();

  T zero = 0;
  int count = 0;
  for (int ki = 0; ki < kern_itr; ++ki) {
    for (int ci = 0; ci < chan_itr; ++ci) {
      for (int c = sa_arch[SA_ARCH_N] - 1; c >= 0; --c) {
        for (int r = 0; r < sa_arch[SA_ARCH_ROW]; ++r) {
          int kern_i = ki * sa_arch[SA_ARCH_N] + r;
          int chan_i = ci * sa_arch[SA_ARCH_ROW] + c;
          int index = kern_i * strides[0] + chan_i * strides[1];
          //std::cout << " kern " << kern_i << " chan " << chan_i << " index " << index << '\n';
          if (kern_i >= dims[TENSOR_4D_BATCH] || chan_i >= dims[TENSOR_4D_CHANNELS]) {
            blob.append(zero);
          } else {
            blob.append(tensor->at(index));
          }
          count++;
        }
      }
    }
  }
  log_info2("Inserted {} elements\n", count * sizeof(T));
}

template <typename T> static void sa_align_aux(const Op::LayerBase *l, BinBlob &blob, const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  auto sa_arch = get_sa_arch();
  assert(dims.size() == 4);
  if (is_pointwise_conv(dims) && !is_sa_regular_optimal(sa_arch)) {
    sa_align_aux_pointwise(l, blob, tensor);
  } else {
    sa_align_aux_regular(l, blob, tensor);
  }
}

template <typename T>
static void conv_bias_align_aux(BinBlob &blob, const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  assert(dims.size() == 1);
  size_t size = dims[TENSOR_4D_BATCH];
  size_t aligned_size =
      ceil_mod(aligned_conv_bias(dims) * sizeof(T), WORD_SIZE);
  size_t bytes = size * sizeof(T);
  size_t deficit_bytes = aligned_size - bytes;
  for (size_t i = 0; i < size; ++i) {
    blob.append(tensor->at(i));
  }
  uint8_t zero = 0;
  for (size_t i = 0; i < deficit_bytes; ++i) {
    blob.append(zero);
  }
  int count = size + deficit_bytes;
  log_info2("Inserted {} elements\n", count * sizeof(T));
}

template <typename T>
static void fc_bias_align_aux(BinBlob &blob, const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  assert(dims.size() == 1);
  int size = static_cast<int>(dims[0]);
  int aligned_dims = aligned_fc_bias(dims);

  auto sa_arch = get_sa_arch();
  auto va_size = get_va_size();
  int tail_blocks = sa_arch[SA_ARCH_N];

  if (tail_blocks > va_size) {
    log_fatal("Tailblocks != vasize; found tail_blocks ({}), vasize ({})\n",
              tail_blocks, va_size);
  }

  int dk = ceil_div(va_size, tail_blocks);
  int iterations = ceil_div(aligned_dims, tail_blocks * dk);

  T zero = 0;
  int count = 0;
  for (int i = 0; i < iterations; ++i) {
    for (int j = 0; j < dk; ++j) {
      for (int k = 0; k < tail_blocks; ++k) {
        int index = i * tail_blocks * dk + j + k * dk;
        if (index >= size) {
          blob.append(zero);
        } else {
          blob.append(tensor->at(index));
        }
        count++;
      }
    }
  }
  log_info2("Inserted {} elements\n", count * sizeof(T));
}

template <typename T>
static void fc_weight_align_aux(BinBlob &blob, const Tensor<T> *tensor, bool transpose) {
  auto dims = tensor->get_dims();
  assert(dims.size() == 2);
  auto aligned_dims = aligned_fc_weight_dims(dims);
  int va_size = get_va_size();
  int hiterations = 0;
  int viterations = 0;
  if (transpose) {
    hiterations = std::ceil(aligned_dims[0] / va_size);
    viterations = aligned_dims[1];
  } else {
    hiterations = std::ceil(aligned_dims[1] / va_size);
    viterations = aligned_dims[0];
  }
  std::vector<int> index(2);
  T zero = 0;
  int count = 0;
  for (int i = 0; i < hiterations; ++i) {
    for (int j = 0; j < viterations; ++j) {
      for (int k = 0; k < va_size; ++k) {
        index[0] = k + (i * va_size);
        index[1] = j;
        // std::cout << "index[0] " << index[0] << "index[1] " << index[1] <<
        // '\n';
        if (is_out_of_bounds(index, dims)) {
          blob.append(zero);
        } else {
          blob.append(tensor->at(index));
        }
        count++;
      }
    }
  }
  log_info2("Inserted {} elements\n", count * sizeof(T));
}

static void sa_align(const Op::LayerBase *l, BinBlob &blob, const onnx::TensorProto *tensor) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    sa_align_aux(l, blob, t1.get());
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    sa_align_aux(l, blob, t1.get());
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

static void conv_bias_align(BinBlob &blob, const onnx::TensorProto *tensor) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    conv_bias_align_aux(blob, t1.get());
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    conv_bias_align_aux(blob, t1.get());
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    std::unique_ptr<Tensor<int32_t>> t1{new TensorExtant<int32_t>(tensor)};
    conv_bias_align_aux(blob, t1.get());
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

static void fc_bias_align(BinBlob &blob, const onnx::TensorProto *tensor) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    fc_bias_align_aux(blob, t1.get());
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    fc_bias_align_aux(blob, t1.get());
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    std::unique_ptr<Tensor<int32_t>> t1{new TensorExtant<int32_t>(tensor)};
    fc_bias_align_aux(blob, t1.get());
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

static void fc_weight_align(BinBlob &blob, const onnx::TensorProto *tensor, bool transpose) {
  int32_t type = tensor->data_type();
  switch (type) {
  case onnx::TensorProto_DataType_INT8: {
    std::unique_ptr<Tensor<int8_t>> t1{new TensorExtant<int8_t>(tensor)};
    fc_weight_align_aux(blob, t1.get(), transpose);
    break;
  }
  case onnx::TensorProto_DataType_UINT8: {
    std::unique_ptr<Tensor<uint8_t>> t1{new TensorExtant<uint8_t>(tensor)};
    fc_weight_align_aux(blob, t1.get(), transpose);
    break;
  }
  case onnx::TensorProto_DataType_INT32: {
    std::unique_ptr<Tensor<int32_t>> t1{new TensorExtant<int32_t>(tensor)};
    fc_weight_align_aux(blob, t1.get(), transpose);
    break;
  }
  default:
    log_fatal("Cant generate weight blob, unsupported data type {} "
              "for tensor {}\n",
              Op::get_tensorproto_dtype_name((TPDT)type), tensor->name());
    break;
  }
}

void Op::Layer::QLinearConv::align_weights(BinBlob &blob, InitializerTable &tbl) {
    uint32_t aligned_sz = aligned_conv_weight(this);
    aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(weights->data_type()));
    aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
    log_info2("Appending initializer {} for size: {}, addr: {}\n", weights->name(), aligned_sz, tbl.get(weights->name()));
    blob.append_dwp_header(aligned_sz, tbl.get(weights->name()));
    sa_align(this, blob, weights);

    if (this->bias != nullptr) {
      aligned_sz = aligned_conv_bias(bias->dims());
      aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(bias->data_type()));
      aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
      log_info2("Appending initializer {} for size: {}, addr: {}\n",
                bias->name(), aligned_sz, tbl.get(bias->name()));
      blob.append_dwp_header(aligned_sz, tbl.get(bias->name()));
      conv_bias_align(blob, bias);
    }
}

void Op::Layer::QGemm::align_weights(BinBlob &blob, InitializerTable &tbl) {
    uint32_t aligned_sz = aligned_fc_weight(weights->dims());
    aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(weights->data_type()));
    aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
    log_info2("Appending initializer {} for size: {}, addr: {}\n", weights->name(), aligned_sz, tbl.get(weights->name()));
    blob.append_dwp_header(aligned_sz, tbl.get(weights->name()));
    fc_weight_align(blob, weights, m_cp.transB);

    aligned_sz = aligned_fc_bias(bias->dims());
    aligned_sz *= Op::tpdt_sizeof(static_cast<TPDT>(bias->data_type()));
    aligned_sz = ceil_mod(aligned_sz, WORD_SIZE);
    log_info2("Appending initializer {} for size: {}, addr: {}\n", bias->name(), aligned_sz, tbl.get(bias->name()));
    blob.append_dwp_header(aligned_sz, tbl.get(bias->name()));
    fc_bias_align(blob, bias);
}

char *BinBlob::get_data() { return m_data; }

const char *BinBlob::get_cdata() const { return m_data; }

void BinBlob::append_zeroth_inst(uint32_t start_addr, uint32_t end_addr) {
  std::bitset<INST_SIZE_BITS> inst{0};
  std::bitset<WORD_SIZE> start_addr_bs{start_addr};
  bitset_range_set(inst, start_addr_bs, ZerothStartAddress_LOW,
                   ZerothStartAddress_HIGH);
  std::bitset<WORD_SIZE> end_addr_bs{end_addr};
  bitset_range_set(inst, end_addr_bs, ZerothEndAddress_LOW,
                   ZerothEndAddress_HIGH);
  generic_append(inst);
}

GmlGen::GmlGen(uint32_t org) : m_org{org} {}

BinBlob GmlGen::generate_gml(Op::Parser &parser) {
  InstGen instgen(parser);
  uint32_t size = instgen.model_size_cpu();
  int tdp = instgen.dwp_packets();
  log_info("Total DWP Packets: {}\n", tdp);
  size += (tdp * DWP_HEADER_BYTES);

  log_info("Calculated GML size: {}\n", size);
  BinBlob blob(size);
  InstBlob instblob = instgen.get_blob();
  log_info("InstBlob Length: {}\n", instblob.size());
  if (gbl_args.has_option("pretty-print-inst")) {
    pretty_print(instblob);
  }
  if (gbl_args.has_option("pretty-print-inst-html")) {
    pretty_print_html(instblob);
  }
  if (gbl_args.has_option("pretty-print-inst-raw")) {
    pretty_print_inst_raw(instblob);
  }
  log_info("Appending instblob\n");
  blob.append(instblob, m_org);
  log_info("Appending initializers\n");
  //blob.append(tbl);

  InitializerTable tbl = instgen.get_tbl();

  Op::Graph graph = parser.get_graph();
  auto m_exec_order = crt_exec_order(graph);
  for (Op::LayerBase *l : m_exec_order) {
    l->align_weights(blob, tbl);
  }

  GmlCheck gmlcheck(instblob, blob);
  /* enfore NRVO at call site */
  return blob;
}

GmlCheck::GmlCheck() {
}

GmlCheck::GmlCheck(const InstBlob &instblob, const BinBlob &binblob) {
  check_citr_kitr(instblob);
  // check_addresses(instblob);
  check_weight_address_continuity(instblob);
  check_fc_flatten(instblob);
  check_dwp(binblob);
}

void GmlCheck::check_citr_kitr(const InstBlob &instblob) const {
  auto sa_arch = get_sa_arch();
  auto va_size = get_va_size();

  std::stack<const std::bitset<INST_SIZE_BITS> *> megablocks;

  for (const auto &i : instblob) {
    int op = extract_opcode(i);

    if (is_megablock_op_code(op)) {
      megablocks.push(&i);
    }

    if (op == OP_OutputBlock) {
      if (megablocks.empty()) {
        log_fatal("GmlCheck: Found output instruction without any parent "
                  "megablock instruction\n");
      }
      const auto *previous_inst = megablocks.top();
      int p_op = extract_opcode(*previous_inst);

      int expected_chan_itr = 0;
      int expected_kern_itr = 0;

      if (p_op == OP_CONV) {
        int ic = inst_get(*previous_inst, CONV_IC);
        int kh = inst_get(*previous_inst, CONV_KH);
        int kw = inst_get(*previous_inst, CONV_KW);
        int chan = inst_get(*previous_inst, CONV_KC);
        int kern = inst_get(*previous_inst, CONV_KN);
        if (kh == 1 && kw == 1) {
          expected_chan_itr = ceil_div(chan, sa_arch[SA_ARCH_ROW]);
          expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_N]);
        } else if (chan == 1 && ic > 1) {
          expected_chan_itr = ceil_div(chan, sa_arch[SA_ARCH_COLS]);
          expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_N]);
        } else {
          if (is_sa_regular_optimal(sa_arch)) {
            expected_chan_itr = ceil_div(chan, sa_arch[SA_ARCH_N]);
            expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_COLS]);
          } else {
            expected_chan_itr = chan;
            expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_N]);
          }
        }
      } else if (p_op == OP_FC) {
        expected_chan_itr = 1;
        /* FC processes va_size number of columns at a time, kernel
         * iterations for FC mean the iterations of weight cols to
         * process the weight matrix completely i.e.
         * WeightCols/va_size
         */
        int weight_cols = inst_get(*previous_inst, FC_WeightCols);
        expected_kern_itr = ceil_div(weight_cols, va_size);
      } else if (p_op == OP_EltWise) {
        expected_chan_itr = 1;
        int kern = inst_get(*previous_inst, EltWise_IC);
        expected_kern_itr = ceil_div(kern, sa_arch[SA_ARCH_N]);
      } else if (p_op == OP_NMS) {
        expected_chan_itr = 0;
        expected_kern_itr = 0;
      } else if (p_op == OP_TRANSPOSE) {
        expected_chan_itr = 1;
        expected_kern_itr = 1;
      } else {
        log_fatal("GmlCheck: megablock of opcode {} cannot be handled\n", p_op);
      }

      int computed_chan_itr = inst_get(i, OutputBlock_ChannelItr);
      int computed_kern_itr = inst_get(i, OutputBlock_KernelItr);

      if (computed_chan_itr != expected_chan_itr) {
        log_fatal("GmlCheck: computed channel iteration ({}) does not match "
                  "expected channel iteration ({})\n",
                  computed_chan_itr, expected_chan_itr);
      }

      if (computed_kern_itr != expected_kern_itr) {
        log_fatal("GmlCheck: computed kernel iteration ({}) does not match "
                  "expected kernel iteration ({})\n",
                  computed_kern_itr, expected_kern_itr);
      }
    }
  }
}

void GmlCheck::check_addresses(const InstBlob &instblob) const {
  auto sa_arch = get_sa_arch();
  std::stack<const std::bitset<INST_SIZE_BITS> *> op_insts;

  int index = 0;
  for (size_t i = 0; i < instblob.size(); ++i) {
    int op = extract_opcode(i);
    if (is_megablock_op_code(op)) {
      index = i + 1;
      break;
    }
  }

  for (size_t i = index; i < instblob.size(); ++i) {
    const auto &inst = instblob.at(i);
    int op = extract_opcode(inst);
    if (op == OP_OutputBlock) {
      op_insts.push(&inst);
    }
    if (is_megablock_op_code(op)) {
      if (op_insts.empty()) {
        log_fatal("Found an empty output stack i.e. this megablock {} at index "
                  "{} does not "
                  " have a preceding output instruction\n",
                  op, i);
      }
      int input_addr = 0;
      if (op == OP_CONV) {
        input_addr = inst_get(inst, CONV_ImageStartAddress);
      } else if (op == OP_FC) {
        input_addr = inst_get(inst, FC_ImageStartAddress);
      } else if (op == OP_EltWise) {
        /* continue as eltwise has two inputs, does not necessarily write to
         * its outputs
         */
        continue;
      } else {
        log_fatal("Unhandled megablock of opcode {} at index {}\n", op, i);
      }
      check_alignment(input_addr);
      const auto preceding_inst = op_insts.top();
      op_insts.pop();
      int output_addr = inst_get(*preceding_inst, OutputBlock_OutputAddr);
      check_alignment(output_addr);

      if (input_addr != output_addr) {
        log_fatal("GmlCheck: input_address != output_addr for output inst at "
                  "index {}\n",
                  i);
      }
    }
  }
}

/* Corollary: check if weight addresses do not overlap */
void GmlCheck::check_weight_address_continuity(const InstBlob &instblob) const {
  int current_address = 0;
  int ret = 0;
  for (size_t i = 0; i < instblob.size(); ++i) {
    const auto &inst = instblob.at(i);
    int op = extract_opcode(inst);
    if (op == OP_CONV) {
      ret = check_conv_weight_continuity(inst);
    } else if (op == OP_TailBlock) {
      ret = check_bias_continuity(inst);
    } else if (op == OP_FC) {
      ret = check_fc_weight_continuity(inst);
    } else if (op == OP_OutputBlock || op == OP_START || op == OP_EltWise ||
               op == OP_NMS || op == OP_TRANSPOSE || op == OP_RESHAPE) {
      // do nothing
    } else {
      log_fatal("Unhandled instruction in check_weight_address_continuity {}\n",
                op);
    }
    if (ret == -1) {
      continue;
    }
    if (ret < current_address) {
      log_fatal(
          "weight address continuity broken at current_address {} and ret {}\n",
          current_address, ret);
    } else {
      current_address = ret;
    }
  }
}

int GmlCheck::check_conv_weight_continuity(
    const std::bitset<INST_SIZE_BITS> &inst) const {
  auto sa_arch = get_sa_arch();
  int start = inst_get(inst, CONV_WeightStartAddress);
  int end = inst_get(inst, CONV_WeightEndAddress);
  check_alignment(start);
  check_alignment(end);
  if (start >= end) {
    log_fatal("Layer has WeightStartAddress {} >= WeightEndAddress {}", start,
              end);
  }
  int type = inst_get(inst, CONV_ConvType);
  int kn = inst_get(inst, CONV_KN);
  int ic = inst_get(inst, CONV_KC);
  int chan_itr = 0; int kern_itr = 0;
  if (type == CONV_TYPE_PW) {
    kern_itr = ceil_div(ceil_mod(kn, sa_arch[SA_ARCH_N]), sa_arch[SA_ARCH_N]);
    chan_itr = ceil_div(ceil_mod(ic, sa_arch[SA_ARCH_ROW]), sa_arch[SA_ARCH_ROW]);
  } else if (type == CONV_TYPE_DW) {
    kern_itr = ceil_div(ceil_mod(kn, sa_arch[SA_ARCH_N]), sa_arch[SA_ARCH_N]);
    chan_itr = ceil_div(ceil_mod(ic, sa_arch[SA_ARCH_COLS]), sa_arch[SA_ARCH_COLS]);
  } else {
    if (is_sa_regular_optimal(sa_arch)) {
      kern_itr = ceil_div(ceil_mod(kn, sa_arch[SA_ARCH_COLS]), sa_arch[SA_ARCH_COLS]);
      chan_itr = ceil_div(ceil_mod(ic, sa_arch[SA_ARCH_N]), sa_arch[SA_ARCH_N]);
    } else {
      kern_itr = ceil_div(ceil_mod(kn, sa_arch[SA_ARCH_N]), sa_arch[SA_ARCH_N]);
      chan_itr = ceil_div(ceil_mod(ic, sa_arch[SA_ARCH_COLS]), sa_arch[SA_ARCH_COLS]);
    }
  }
  int expected_weight_size = ceil_mod(kern_itr * chan_itr * prod(sa_arch), WORD_SIZE);

  int computed_weight_size = end - start;
  if (computed_weight_size != expected_weight_size) {
    log_fatal("For conv instruction, computed_weight_size {} does not match "
              "expected_weight_size {}\n",
              computed_weight_size, expected_weight_size);
  }
  return end;
}

int GmlCheck::check_bias_continuity(
    const std::bitset<INST_SIZE_BITS> &inst) const {
  if (!inst_get(inst, TailBlock_BiasEn)) {
    return -1;
  }
  int start = inst_get(inst, TailBlock_BiasStartAddress);
  int end = inst_get(inst, TailBlock_BiasEndAddress);
  check_alignment(start);
  check_alignment(end);
  if (start >= end) {
    log_fatal("Layer has BiasStartAddress {} >= BiasEndAddress {}", start, end);
  }
  return end;
}

int GmlCheck::check_fc_weight_continuity(
    const std::bitset<INST_SIZE_BITS> &inst) const {
  auto va_size = get_va_size();
  int start = inst_get(inst, FC_WeightStartAddress);
  int end = inst_get(inst, FC_WeightEndAddress);
  check_alignment(start);
  check_alignment(end);
  if (start >= end) {
    log_fatal("Layer has WeightStartAddress {} >= WeightEndAddress {}", start,
              end);
  }
  int wr = inst_get(inst, FC_WeightRows);
  int wc = inst_get(inst, FC_WeightCols);
  int expected_weight_size =
      ceil_mod(ceil_mod(wr, va_size) * ceil_mod(wc, va_size), WORD_SIZE);
  int computed_weight_size = end - start;
  if (computed_weight_size != expected_weight_size) {
    log_fatal("For FC instruction, computed_weight_size {} does not match "
              "expected_weight_size {}\n",
              computed_weight_size, expected_weight_size);
  }
  return end;
}

void GmlCheck::check_fc_flatten(const InstBlob &instblob) const {
  std::stack<const std::bitset<INST_SIZE_BITS> *> megablocks;
  for (size_t i = 0; i < instblob.size(); ++i) {
    const auto &inst = instblob.at(i);
    int op = extract_opcode(inst);
    if (op == OP_CONV) {
      if (!megablocks.empty()) {
        megablocks.pop();
      }
      megablocks.push(&inst);
    }

    if (op == OP_FC) {
      int expected_flatten = 0;
      if (megablocks.empty()) {
        expected_flatten = 0;
      } else {
        const auto *i_ptr = megablocks.top();
        megablocks.pop();
        if (extract_opcode(*i_ptr) == OP_CONV) {
          expected_flatten = 1;
        }
      }
      int computed_flatten = inst_get(inst, FC_Flatten);
      if (expected_flatten != computed_flatten) {
        log_fatal("GmlCheck: expected flatten for layer {} to be {} but the "
                  "instruction says it "
                  "ought to be {}\n",
                  i, expected_flatten, computed_flatten);
      }
    }
  }
}

void GmlCheck::check_alignment(int addr) const {
  if (addr % WORD_SIZE != 0) {
    log_fatal("Address {} is not aligned to WORD_SIZE {}\n", addr, WORD_SIZE);
  }
}

void GmlCheck::check_dwp(const BinBlob &binblob) const {
  const char *data = binblob.get_cdata();
  int size = static_cast<int>(binblob.size());

  std::vector<std::string> payloads;

  int packet_count = 0;
  for (int i = 0; i < size;) {
    uint32_t sop = bytes2int(data + i);
    uint32_t ds = bytes2int(data + i + 4);
    uint32_t addr = bytes2int(data + i + 8);
    log_info2("PACKET #{}\n", packet_count++);
    log_info2("OFFSET: {}\n", i);
    log_info2("DWP: {}\n", sop);
    log_info2("DS: {}\n", ds);
    log_info2("ADDR: {}\n", addr);
    if (sop != DWP_SOP) {
      log_fatal(
          "GmlCheck: sop at index {} with value {} does not match DWP_SOP {}\n",
          i, sop, DWP_SOP);
    }
    i += DWP_HEADER_BYTES;
    if ((size - i) < static_cast<int>(ds)) {
      log_fatal(
          "GmlCheck: Not enough bytes, starting at {}, ds: {}, size: {}\n", i,
          ds, size);
    }

    if (get_verbose2()) {
      if (ds >= 16) {
        for (int j = i; j < i+8; ++j) {
          std::cout << (int) data[j] << ' ';
        }
        std::cout << "... ";
        for (int j = i+ds-1; j > i+ds-1-8; --j) {
          std::cout << (int) data[j] << ' '; 
        }
      } else {
        for (int j = i; j < i+ds; ++j) {
          std::cout << (int) data[j] << ' '; 
        }
      }
      std::cout << '\n';
    }

    std::string ss;
    int range = i + ds;
    for (; i < range; ++i) {
      ss.push_back(data[i]);
    }
    payloads.push_back(ss);
    /* check for spare ff's and warn when found */
  }
}

void Op::Layer::NMS::get_opcodes(std::vector<int> &op_codes) {
  op_codes.push_back(OP_NMS);
  op_codes.push_back(OP_OutputBlock);
}

uint32_t Op::Layer::NMS::get_weight_size() {
  // No weights for NMS
  return 0;
}

uint32_t get_total_in_box(Op::LayerBase *layer) {
  return layer->input_dims[I_NMS_INPUT_BOXES][I_INPUT_BOXES_COUNT];
}
int get_total_classes(Op::LayerBase *layer) {
  return layer->input_dims[I_NMS_INPUT_SCORES][I_CLASSES_COUNT];
}
uint32_t get_box_start_address(AddressGen &gen, Op::LayerBase *layer) {
  return gen.io_addr_from_register(layer->inputs[I_NMS_INPUT_BOXES]);
}
uint32_t get_scores_start_address(AddressGen &gen, Op::LayerBase *layer) {
  return gen.io_addr_from_register(layer->inputs[I_NMS_INPUT_SCORES]);
}
Op::LayerBase *get_nms_layer(AddressGen &gen, int nms_i) {
  auto order = gen.get_exec_order();
  return order[nms_i];
}

int get_nms_index(AddressGen &gen) {
  auto order = gen.get_exec_order();
  for (int i = 0; i < order.size(); i++) {
    if (order[i]->name == "/NonMaxSuppression") {
      return i;
    }
  }
  return -1;
}

int Op::Layer::NMS::get_inst(InstBlob &insts, AddressGen &gen,
                             InitializerTable &) {
  std::bitset<INST_SIZE_BITS> nms_inst;
  int nms_i = get_nms_index(gen);
  assert(nms_i != -1);
  auto layer = get_nms_layer(gen, nms_i);
  const uint32_t box_memory_size =
      get_total_in_box(layer) * 4 *
      Op::tpdt_sizeof(layer->input_type[I_NMS_INPUT_BOXES]);
  const uint32_t scores_memory_size =
      get_total_in_box(layer) * get_total_classes(layer) *
      Op::tpdt_sizeof(layer->input_type[I_NMS_INPUT_SCORES]);
  std::bitset<NMS_Opcode_COUNT> opcode{OP_NMS};
  inst_set(nms_inst, opcode, NMS_Opcode);

  std::bitset<NMS_IOU_COUNT> iou{iou_threshold};
  inst_set(nms_inst, iou, NMS_IOU);

  std::bitset<NMS_ScoreThresh_COUNT> score_thresh{score_threshold};
  inst_set(nms_inst, score_thresh, NMS_ScoreThresh);

  std::bitset<NMS_TotalInBoxes_COUNT> total_in_boxes{get_total_in_box(layer)};
  inst_set(nms_inst, total_in_boxes, NMS_TotalInBoxes);

  std::bitset<NMS_MaxOutBoxes_COUNT> max_out_boxes{
      static_cast<unsigned int>(this->max_output_boxes)};
  inst_set(nms_inst, max_out_boxes, NMS_MaxOutBoxes);

  // 0-corner-cord  1-center-point
  std::bitset<NMS_CornerCord_COUNT> corner_cord{center_point_box};
  inst_set(nms_inst, corner_cord, NMS_CornerCord);

  int total_class = get_total_classes(layer);
  assert(total_class >= 0 && total_class <= 255);
  std::bitset<NMS_TotalClasses_COUNT> total_classes{
      static_cast<unsigned int>(total_class)};
  inst_set(nms_inst, total_classes, NMS_TotalClasses);

  const uint32_t box_start_address = get_box_start_address(gen, layer);
  const uint32_t box_end_address =
      box_start_address + ceil_mod(box_memory_size, 32);
  const uint32_t score_start_address = get_scores_start_address(gen, layer);
  const uint32_t score_end_address =
      score_start_address + ceil_mod(scores_memory_size, 32);

  std::bitset<NMS_BoxStartAddr_COUNT> box_start_addr{box_start_address};
  inst_set(nms_inst, box_start_addr, NMS_BoxStartAddr);

  std::bitset<NMS_BoxEndAddr_COUNT> box_end_addr{box_end_address};
  inst_set(nms_inst, box_end_addr, NMS_BoxEndAddr);

  std::bitset<NMS_ScoreStartAddr_COUNT> score_start_addr{score_start_address};
  inst_set(nms_inst, score_start_addr, NMS_ScoreStartAddr);

  std::bitset<NMS_ScoreEndAddr_COUNT> score_end_addr{score_end_address};
  inst_set(nms_inst, score_end_addr, NMS_ScoreEndAddr);

  insts.push_back(nms_inst);

  std::bitset<INST_SIZE_BITS> nms_output_inst;

  std::bitset<OutputBlock_Opcode_COUNT> ob_opcode{OP_OutputBlock};
  inst_set(nms_output_inst, ob_opcode, OutputBlock_Opcode);

  uint32_t output_addr_start = gen.io_addr_from_register(layer->outputs.at(0));

  std::bitset<OutputBlock_OutputAddr_COUNT> ostart{output_addr_start};
  inst_set(nms_output_inst, ostart, OutputBlock_OutputAddr);

  insts.push_back(nms_output_inst);
  return 0;
}
