#include "pch.h"

#include "optimization.h"

bool is_large_conv(Op::Layer::QLinearConv *cc) {
  if (cc->m_cp.k[0] > 3 && cc->m_cp.k[1] > 3) {
    return true;
  }
  return false;
}

Op::Vertex create_qconv(Op::Graph &g, const Op::Layer::QLinearConv *cc,
                        onnx::TensorProto *tensor, int n, int i) {
  Op::Vertex new_vertex = boost::add_vertex(g);
  auto *new_conv = new Op::Layer::QLinearConv(*cc);

  new_conv->name = "decompose_qconv_" + std::to_string(i);

  if (i == n - 1) {
    new_conv->bias = cc->bias;
  } else {
    new_conv->bias = nullptr;
  }

  new_conv->output_dims = cc->output_dims;
  new_conv->weights = tensor;
  new_conv->m_cp.k[0] = tensor->dims(2);
  new_conv->m_cp.k[1] = tensor->dims(3);
  new_conv->m_cp.ki = i+1;

  for (auto &output_type : new_conv->output_type) {
    output_type = onnx::TensorProto_DataType_INT32;
  }

  g[new_vertex] = new_conv;

  return new_vertex;
}

Op::Vertex create_qadd(Op::Graph &g,
                       std::vector<Op::Vertex> &new_decomposed_conv,
                       const Op::Layer::QLinearConv *cc, int n, int i) {
  Op::Vertex new_vertex = boost::add_vertex(g);
  auto *new_add = new Op::Layer::QLinearEltwise(ELTWISE_ADD); 

  new_add->name = "qadd_" + std::to_string(i);

  new_add->a_scale = cc->x_scale[0];
  new_add->b_scale = cc->w_scale[0];
  new_add->o_scale = cc->y_scale;

  new_add->a_zp = std::visit([](auto zp) { return static_cast<int>(zp); },
                             cc->x_zero_point[0]);
  new_add->b_zp = std::visit([](auto zp) { return static_cast<int>(zp); },
                             cc->w_zero_point[0]);
  new_add->zero_point = cc->y_zero_point;

  new_add->input_dims = g[new_decomposed_conv[i]]->output_dims;
  new_add->output_dims = new_add->input_dims;
  new_add->input_names.push_back(new_add->name + "inputs");
  new_add->output_names.push_back(new_add->name + "outputs");

  new_add->input_type.push_back(onnx::TensorProto_DataType_INT32);
  if (i == n - 2) {
    new_add->output_type.push_back(onnx::TensorProto_DataType_INT8);
  } else {
    new_add->output_type = new_add->input_type;
  }

  new_add->device = DEVICE_UNKNOWN;
  g[new_vertex] = new_add;
  return new_vertex;
}

template <typename T>
std::vector<onnx::TensorProto *>
slice_large_convolution(const onnx::TensorProto &initializer) {
  std::vector<onnx::TensorProto *> kernel_proto;

  TensorExtant<T> tensor(&initializer);
  const std::vector<int> &dims = tensor.get_dims();

  int N = dims[0];
  int C = dims[1];
  int H = dims[2];
  int W = dims[3];

  for (int h = 0; h < H; ++h) {
    std::vector<T> slice;
    slice.reserve(N * C * W);

    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int w = 0; w < W; ++w) {
          slice.push_back(tensor.at({n, c, h, w}));
        }
      }
    }

    auto *sliced_tensor = new onnx::TensorProto();
    sliced_tensor->set_data_type(initializer.data_type());
    sliced_tensor->set_name(initializer.name() + "_slice_" + std::to_string(h));
    sliced_tensor->add_dims(N);
    sliced_tensor->add_dims(C);
    sliced_tensor->add_dims(1);
    sliced_tensor->add_dims(W);
    sliced_tensor->set_raw_data(slice.data(), slice.size() * sizeof(T));

    kernel_proto.push_back(sliced_tensor);
  }

  return kernel_proto;
}

void split_large_kernel(Op::Graph &g) {
  std::vector<Op::Vertex> vertices_to_remove;

  for (auto vp = boost::vertices(g); vp.first != vp.second; ++vp.first) {
    auto v = *vp.first;

    if (strcmp(g[v]->op_type(), "QLinearConv") == 0) {
      Op::Layer::QLinearConv *cc = dynamic_cast<Op::Layer::QLinearConv *>(g[v]);

      if (cc && is_large_conv(cc)) {
        std::vector<Op::Vertex> predecessors = get_parents(v, g);
        std::vector<Op::Vertex> successors = get_children(v, g);

        std::vector<onnx::TensorProto *> sliced_tensors;
        if (cc->weights->data_type() == onnx::TensorProto_DataType_INT8) {
          sliced_tensors = slice_large_convolution<int8_t>(*(cc->weights));
        } else if (cc->weights->data_type() ==
                   onnx::TensorProto_DataType_UINT8) {
          sliced_tensors = slice_large_convolution<uint8_t>(*(cc->weights));
        }

        vertices_to_remove.push_back(v);
        boost::clear_vertex(v, g);

        std::vector<Op::Vertex> new_decomposed_conv;
        for (size_t i = 0; i < sliced_tensors.size(); i++) {
          Op::Vertex new_vertex =
              create_qconv(g, cc, sliced_tensors[i], sliced_tensors.size(), i);
          new_decomposed_conv.push_back(new_vertex);
          for (auto pred : predecessors) {
            boost::add_edge(pred, new_vertex, g);
          }
        }

        std::vector<Op::Vertex> qadd;
        for (size_t i = 0; i < sliced_tensors.size() - 1; i++) {
          Op::Vertex new_vertex =
              create_qadd(g, new_decomposed_conv, cc, sliced_tensors.size(), i);
          qadd.push_back(new_vertex);
        }

        if (new_decomposed_conv.size() >= 2 && qadd.size() >= 1) {
          boost::add_edge(new_decomposed_conv[0], qadd[0], g);
          boost::add_edge(new_decomposed_conv[1], qadd[0], g);

          for (size_t i = 1; i < qadd.size(); i++) {
            boost::add_edge(qadd[i - 1], qadd[i], g);
            boost::add_edge(new_decomposed_conv[i + 1], qadd[i], g);
          }

          for (auto succ : successors) {
            boost::add_edge(qadd.back(), succ, g);
          }
        }
      }
    }
  }

  for (auto v : vertices_to_remove) {
    boost::remove_vertex(v, g);
  }

  auto vp = boost::vertices(g);
  Op::Vertex v = *(vp.first);
  Op::Vertex new_vertex = boost::add_vertex(g);

  auto *dum = new Op::Layer::NoOp();
  dum->name = "NoOp";
  dum->input_dims = g[v]->input_dims;
  dum->output_dims = g[v]->input_dims;
  dum->device = DEVICE_CPU;
  dum->input_type = g[v]->input_type;
  dum->output_type = g[v]->input_type;
  dum->input_names.push_back("noop_inputs");
  dum->output_names.push_back("noop_outputs");
  g[new_vertex] = dum;
  boost::add_edge(new_vertex, v, g);

  Op::RegisterAllocator allocator(g);
}
