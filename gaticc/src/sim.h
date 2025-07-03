#pragma once

#include "onnx_parser.h"
#include "tensor.h"
#include "utils.h"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/graph_traits.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

constexpr int FIXED_POINT_BASE_TYPE = 18;
constexpr int FIXED_POINT_SPLIT = 10;

template <typename T> class Relu {
  int clip_val;

public:
  Relu(int clip_val);
  Relu();
  void exec(const Tensor<T> *input, Tensor<T> *output);
};

template <typename T> Relu<T>::Relu(int clip_val) : clip_val{clip_val} {}

template <typename T> Relu<T>::Relu() : clip_val{INT_MAX} {}

template <typename T>
void Relu<T>::exec(const Tensor<T> *input, Tensor<T> *output) {
  for (int i = 0; i < input->size(); ++i) {
    T x = input->at(i);
    T v = (x < 0) ? 0 : ((x > clip_val) ? clip_val : x);
    output->set(i, v);
  }
}

template <typename T>
void maxpool(const Tensor<T> *input, Tensor<T> *output,
             const Op::PoolParams &mp) {
  int pad_present = 0;
  for (int i = 0; i < 4; ++i) {
    if (mp.pad[i] != 0) {
      pad_present = 1;
      break;
    }
  }

  const Tensor<T> *padded_input;
  if (pad_present) {
    padded_input = tensor_pad(
        input, std::vector<int>{mp.pad[0], mp.pad[1], mp.pad[2], mp.pad[3]});
  } else {
    padded_input = input;
  }

  int output_depth = padded_input->dims_at(TENSOR_4D_CHANNELS);
  int output_height = mp_odims_row(mp, input->get_dims());
  int output_width = mp_odims_cols(mp, input->get_dims());

  for (int ici = 0; ici < output_depth; ++ici) {
    for (int ihi = 0; ihi < output_height * mp.stride[TENSOR_2D_HEIGHT];
         ihi += mp.stride[TENSOR_2D_HEIGHT]) {
      for (int iwi = 0; iwi < output_width * mp.stride[TENSOR_2D_WIDTH];
           iwi += mp.stride[TENSOR_2D_WIDTH]) {
        T max_val = std::numeric_limits<T>::min();
        for (int khi = 0; khi < mp.k[TENSOR_2D_HEIGHT]; ++khi) {
          for (int kwi = 0; kwi < mp.k[TENSOR_2D_WIDTH]; ++kwi) {
            std::vector<int> in_index{0, ici, (ihi + khi), (iwi + kwi)};
            // print_vec("in index ", in_index);
            max_val = std::max(max_val, padded_input->at(in_index));
          }
        }
        std::vector<int> out_index{0, ici, ihi / mp.stride[TENSOR_2D_HEIGHT],
                                   iwi / mp.stride[TENSOR_2D_WIDTH]};
        // std::cout << "ihi iwi " << ihi << ' ' << iwi << '\n';
        // print_vec("out index ", out_index);
        output->insert(out_index, max_val);
      }
    }
  }

  if (pad_present) {
    delete padded_input;
  }
}

template <typename T> void flatten(const Tensor<T> *input, Tensor<T> *output) {
  std::vector<int> new_dims = {1, input->dims_iterator(-1)};
  *output = *input;
  output->set_dims(new_dims);
}

void increment_shape(std::vector<int> &ii, const std::vector<int> &limit_shape);
int calc_shift_val(float inverted_scale);

template <typename T>
void transpose(Tensor<T> *input, Tensor<T> *output, const std::vector<int> &perm) {
  std::vector<int> in_dims = input->get_dims();
  std::vector<int> out_dims = permute(in_dims, perm);
  output->set_dims(out_dims);
  std::vector<int> istride = get_stride_from_shape(in_dims);
  std::vector<int> ostride = get_stride_from_shape(out_dims);
  std::vector<int> idx(in_dims.size(), 0);
  int total_elements = input->dims_iterator(-1);
  for (int i = 0; i < total_elements; ++i) {
    int iindex = dot(idx, istride);
    std::vector<int> permuted_idx(in_dims.size());
    for (size_t d = 0; d < in_dims.size(); ++d) {
      permuted_idx[d] = idx[perm[d]];
    }
    int oindex = dot(permuted_idx, ostride);
    output->set(oindex, input->at(iindex));
    increment_shape(idx, in_dims);
  }
}

/* Vector Arrays
 * Used by Gemm/Matmul routines */
template <typename inputT, typename weightT, typename biasT, typename outputT>
class VA {
  int wrows;
  int wcols;
  int isize;
  Tensor<weightT> *weights;
  Tensor<biasT> *bias;

  int a_zero_point;
  int b_zero_point;

public:
  VA(const Op::Layer::Gemm &gp);
  VA(const Op::Layer::MatMul &gp);
  VA(const Op::Layer::QLinearMatMul &gp);
  VA(const Op::Layer::QGemm &gp);
  void run(const Tensor<inputT> *input, Tensor<outputT> *output);
  ~VA() {
    delete weights;
    delete bias;
  }
};

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::Gemm &gp) {
  wrows = gp.m_cp.wr;
  wcols = gp.m_cp.wc;
  isize = gp.input_dims[0][TENSOR_2D_WIDTH];
  if (gp.m_cp.transB) {
    auto tmp = std::make_unique<TensorExtant<weightT>>(gp.weights);
    auto dims = tmp.get()->get_dims();
    std::vector<int> new_dims{dims[1], dims[0]};
    weights = new TensorCreate<weightT>(new_dims);
    transpose(tmp.get(), weights, std::vector<int>{1, 0});
  } else {
    weights = new TensorExtant<weightT>(gp.weights);
  }
  bias = new TensorExtant<biasT>(gp.bias);
  a_zero_point = 0;
  b_zero_point = 0;
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::MatMul &gp) {
  wrows = gp.m_cp.wc;
  wcols = gp.m_cp.wr;
  isize = gp.input_dims[0][TENSOR_2D_WIDTH];
  weights = new TensorExtant<weightT>(gp.weights);
  bias = nullptr;
  a_zero_point = 0;
  b_zero_point = 0;
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::QLinearMatMul &gp) {
  wrows = gp.m_cp.wc;
  wcols = gp.m_cp.wr;
  isize = gp.input_dims[0][TENSOR_2D_WIDTH];
  weights = new TensorExtant<weightT>(gp.weights);
  bias = nullptr;
  using variantT = std::variant<int8_t, uint8_t>;
  auto azps = variant2vec<variantT, int>(gp.a_zero_point);
  auto bzps = variant2vec<variantT, int>(gp.b_zero_point);
  assert(azps.size() == 1);
  a_zero_point = azps[0];
  assert(bzps.size() == 1);
  b_zero_point = bzps[0];
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
VA<inputT, weightT, biasT, outputT>::VA(const Op::Layer::QGemm &gp) {
  wrows = gp.m_cp.wr;
  wcols = gp.m_cp.wc;
  isize = gp.input_dims[0][TENSOR_2D_WIDTH];
  if (gp.m_cp.transB) {
    auto tmp = std::make_unique<TensorExtant<weightT>>(gp.weights);
    auto dims = tmp.get()->get_dims();
    std::vector<int> new_dims{dims[1], dims[0]};
    weights = new TensorCreate<weightT>(new_dims);
    transpose(tmp.get(), weights, std::vector<int>{1, 0});
  } else {
    weights = new TensorExtant<weightT>(gp.weights);
  }
  bias = new TensorExtant<biasT>(gp.bias);
  using variantT = std::variant<int8_t, uint8_t>;
  auto azps = variant2vec<variantT, int>(gp.a_zero_point);
  auto bzps = variant2vec<variantT, int>(gp.b_zero_point);
  assert(azps.size() == 1);
  a_zero_point = azps[0];
  assert(bzps.size() == 1);
  b_zero_point = bzps[0];
}

template <typename inputT, typename weightT, typename biasT, typename outputT>
void VA<inputT, weightT, biasT, outputT>::run(const Tensor<inputT> *input,
                                              Tensor<outputT> *output) {
  assert(input->dims_size() == 2 && weights->dims_size() == 2);

  int N = input->dims_at(0);
  int M = input->dims_at(1);
  int K = weights->dims_at(1);
  outputT dst = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      for (int k = 0; k < M; ++k) {
        /* TODO: use Tensor->at that returns a reference and += operator
         * part of tensor refactor
         */
        outputT a_int = static_cast<outputT>(input->at(i * M + k));
        outputT b_int = static_cast<outputT>(weights->at(k * K + j));
        dst += (a_int - a_zero_point) * (b_int - b_zero_point);
      }
      /* For gemm */
      if (bias != nullptr) {
        dst += bias->at(i * K + j);
      }
      output->set(i * K + j, dst);
      dst = 0;
    }
  }
}


template <typename T>
void reshape(const Tensor<T> *input, Tensor<T> *output,
             const std::vector<int> &new_shape) {
  /* atmost 1 dimension can be -1 */
  std::vector<int> deduced_shape =
      deduce_new_shape(new_shape, input->dims_iterator(-1));
  *output = *input;
  std::vector<int> dims(deduced_shape.size());
  std::copy(deduced_shape.begin(), deduced_shape.end(), dims.begin());
  output->set_dims(dims);
}

/* Element wise tensor addition */
template <typename inputT, typename outputT>
void tensor_eltwise(Tensor<outputT> *output, const Tensor<inputT> *input1,
                    const Tensor<inputT> *input2, int op) {
  assert(input1->dims_iterator(-1) == input2->dims_iterator(-1));

  if (op == ELTWISE_ADD) {
    for (int i = 0; i < input1->dims_iterator(-1); ++i) {
      output->set(i, input1->at(i) + input2->at(i));
    }
  } else if (op == ELTWISE_MULT) {
    for (int i = 0; i < input1->dims_iterator(-1); ++i) {
      output->set(i, input1->at(i) * input2->at(i));
    }
  } else if (op == ELTWISE_SUB) {
    for (int i = 0; i < input1->dims_iterator(-1); ++i) {
      output->set(i, input1->at(i) - input2->at(i));
    }
  } else {
    log_fatal("Unsupported eltwise operation %d\n", op);
  }
}

template <int FRAC_BITS, int TOTAL_BITS>
class FixedPoint {
  static_assert(FRAC_BITS > 0 && FRAC_BITS < TOTAL_BITS, "FRAC_BITS must be positive and less than TOTAL_BITS");
  static_assert(TOTAL_BITS <= 64, "TOTAL_BITS > 64 not supported");

  using BaseType = typename std::conditional<
      (TOTAL_BITS <= 8), int8_t,
      typename std::conditional<
          (TOTAL_BITS <= 16), int16_t,
          typename std::conditional<
              (TOTAL_BITS <= 32), int32_t,
              int64_t
          >::type
      >::type
  >::type;

  static constexpr BaseType MASK = (BaseType(1) << TOTAL_BITS) - 1;
  static constexpr BaseType SIGN_BIT = BaseType(1) << (TOTAL_BITS - 1);
  static BaseType mask(BaseType v) {
    v &= MASK;
    if (v & SIGN_BIT) { v |= ~MASK; }
    return v;
  }

  using WideType = typename std::conditional<(sizeof(BaseType) <= 4), int64_t, __int128_t>::type;

  BaseType value;

public:
  constexpr static BaseType SCALE = BaseType(1) << FRAC_BITS;

  FixedPoint() : value(0) {}
  FixedPoint(float f) : value(mask(static_cast<BaseType>(f * SCALE))) {}
  FixedPoint(int i) : value(mask(static_cast<BaseType>(i) << FRAC_BITS)) {}

  BaseType raw() const { return value; }

  static FixedPoint fromRaw(BaseType rawVal) {
    FixedPoint fp;
    fp.value = rawVal;
    return fp;
  }

  operator int() const { return static_cast<int>(toFloat()); }
  operator int8_t() const { return static_cast<int8_t>(toFloat()); }
  operator uint8_t() const { return static_cast<uint8_t>(toFloat()); }
  operator float() const { return toFloat(); }

  float toFloat() const { return static_cast<float>(value) / SCALE; }

  FixedPoint operator+(const FixedPoint &other) const {
    return fromRaw(value + other.value);
  }

  FixedPoint operator-(const FixedPoint &other) const {
    return fromRaw(value - other.value);
  }

  FixedPoint operator*(const FixedPoint &other) const {
    WideType prod = static_cast<WideType>(value) * other.value;
    return fromRaw(static_cast<BaseType>(prod >> FRAC_BITS));
  }

  FixedPoint operator/(const FixedPoint &other) const {
    WideType dividend = (static_cast<WideType>(value) << FRAC_BITS);
    return fromRaw(static_cast<BaseType>(dividend / other.value));
  }

  bool operator<(const FixedPoint &other) const {
    return value < other.value;
  }

  friend std::ostream &operator<<(std::ostream &os, const FixedPoint &fp) {
    return os << fp.toFloat();
  }
};
using fp_t = FixedPoint<FIXED_POINT_SPLIT, FIXED_POINT_BASE_TYPE>;

/* Element wise tensor addition with scales and zp
 *
 * returns: (i1_scale * (i1[i] - i1_zp) + i2_scale * (i2[i] - i2_zp))
 */
template <typename inputT, typename outputT>
void tensor_qeltwise(Tensor<outputT> *output, const Tensor<inputT> *input1,
                     const Tensor<inputT> *input2, float i1_scale,
                     float i2_scale, int i1_zp, int i2_zp, int op) {
  assert(input1->dims_iterator(-1) == input2->dims_iterator(-1));
  if (op == ELTWISE_ADD) {
    for (int i = 0; i < input1->dims_iterator(-1); ++i) {
      outputT v = ((fp_t(i1_scale) * fp_t(input1->at(i) - i1_zp)) + (fp_t(i2_scale) * fp_t((input2->at(i) - i2_zp))));
      output->set(i, v);
    }
  } else if (op == ELTWISE_MULT) {
    for (int i = 0; i < input1->dims_iterator(-1); ++i) {
      outputT v = ((fp_t(i1_scale) * fp_t(input1->at(i) - i1_zp)) * (fp_t(i2_scale) * fp_t((input2->at(i) - i2_zp))));
      output->set(i, v);
    }
  } else if (op == ELTWISE_SUB) {
    for (int i = 0; i < input1->dims_iterator(-1); ++i) {
      outputT v = ((fp_t(i1_scale) * fp_t(input1->at(i) - i1_zp)) - (fp_t(i2_scale) * fp_t((input2->at(i) - i2_zp))));
      output->set(i, v);
    }
  } else {
    log_fatal("Unsupported eltwise operation %d\n", op);
  }
}

/* Add a tensor and a vector. Each element of the
 * vector is added to all elements of each channel
 * of the tensor
 *
 *  input_tensor.shape = (_, C, _, _)
 *  input_vector.shape = (_, C)
 */
template <typename inputT, typename outputT>
void tensor_vector_add(Tensor<outputT> *output,
                       const Tensor<inputT> *input_tensor,
                       const Tensor<inputT> *input_vector) {
  assert(input_vector->dims_size() == 1);
  assert(input_vector->dims_at(0) == input_tensor->dims_at(TENSOR_4D_CHANNELS));
  assert(input_tensor->dims_size() == 4);

  for (int i = 0; i < output->dims_at(0); ++i) {
    for (int j = 0; j < output->dims_at(1); ++j) {
      for (int k = 0; k < output->dims_at(2); ++k) {
        for (int l = 0; l < output->dims_at(3); ++l) {
          std::vector<int> index{i, j, k, l};
          outputT t1 = input_tensor->at(index) + input_vector->at(j);
          output->insert(index, t1);
        }
      }
    }
  }
}

std::vector<float> compute_output_scale(const std::vector<float> &x_scale,
                                        const std::vector<float> &w_scale,
                                        const std::vector<float> &y_scale);

template <typename inputT, typename outputT>
inline outputT clip(inputT v, int min_lim, int max_lim) {
  if (v < min_lim) {
    return min_lim;
  } else if (v > max_lim) {
    return max_lim;
  } else {
    return v;
  }
}

template <typename inputT, typename outputT>
inline outputT quantize_fn(inputT v, float scale, int zero_point, int min_lim,
                           int max_lim, int shift_val) {
#if 1 /* switch this off for debugging with regular float quantization */
  constexpr int fpwidth = 16;
  /* FPGA style quantization (this is how it's implemented on the FPGA) */
  if constexpr ((std::is_same<outputT, int8_t>() || std::is_same<outputT, uint8_t>()) && (std::is_same<inputT, int32_t>())) {
    /* fpga style quantization */
    float inverted = 1 / scale;
    int int_scale = (int)((float)inverted * (float)(1 << shift_val));
    int ret = (int)((((int)v * int_scale) + (1 << (shift_val - 1))) >> shift_val);
    ret += zero_point;
    outputT r = (outputT)std::clamp<inputT>(ret, min_lim, max_lim);
    return r;
  } else if constexpr ((std::is_same<outputT, int8_t>() || std::is_same<outputT, uint8_t>()) && std::is_same<inputT, fp_t>()) {
    float inverted = 1 / scale;
    int int_scale = (int)((float)inverted * (float)(1 << shift_val));
    int64_t r1 = (int64_t) v.raw() * int_scale;
    int64_t r2 = r1 + (1 << (shift_val - 1));
    int64_t r3 = r2 >> shift_val;
    int64_t r4 = r3 + (1 << (FIXED_POINT_SPLIT-1));
    int r5 = r4 >> FIXED_POINT_SPLIT;
    r5 += zero_point;
    outputT r = (outputT)std::clamp<inputT>(r5, min_lim, max_lim);
    return r;
  } else {
    inputT rounded = std::round(((float)v / scale + zero_point));
    return (outputT)std::clamp<inputT>(rounded, min_lim, max_lim);
  }
#else
  inputT rounded = std::round(((float)v / scale + zero_point));
  return (outputT)std::clamp<inputT>(rounded, min_lim, max_lim);
#endif
}

template <typename inputT, typename outputT>
inline outputT dequantize_fn(inputT v, float scale, int zero_point) {
  return ((v - zero_point) * scale);
}

template <typename inputT, typename outputT>
void quantize(const Tensor<inputT> *input, Tensor<outputT> *output,
              const std::vector<float> &scales,
              const std::vector<int> &zero_point) {

  int min_lim = 0;
  int max_lim = 0;
  if (typeid(outputT) == typeid(uint8_t)) {
    min_lim = 0;
    max_lim = 255;
  } else if (typeid(outputT) == typeid(int8_t)) {
    min_lim = -128;
    max_lim = 127;
  } else {
    log_fatal("cant find saturation values for quantization (unimplemented)\n");
  }


  if (input->dims_size() == 4) {
    const auto bscales =
        broadcast_vec(scales, input->dims_at(TENSOR_4D_CHANNELS));
    float inverted = 1 / bscales[0];
    int shift_val = calc_shift_val(inverted);
    const auto bzero_points =
        broadcast_vec(zero_point, input->dims_at(TENSOR_4D_CHANNELS));
    auto out_strides = output->get_strides();
    auto quant_aux = [&bscales, &bzero_points, &min_lim, &max_lim, &shift_val, &output, &input, &out_strides](const int batch, const int channel) {
      int bc = batch * out_strides[0] + channel * out_strides[1];
      for (int k = 0; k < input->dims_at(TENSOR_4D_HEIGHT); ++k) {
        for (int l = 0; l < input->dims_at(TENSOR_4D_WIDTH); ++l) {
          int in_index = bc + k * out_strides[2] + l * out_strides[3];
          inputT val = input->at(in_index);
          outputT new_val = quantize_fn<inputT, outputT>(
              val, bscales[channel], bzero_points[channel], min_lim, max_lim, shift_val);
          output->set(in_index, new_val);
        }
      }
    };
    std::vector<std::thread> tc;
    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        tc.push_back(std::thread(quant_aux, i, j));
      }
    }
    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        tc[i * input->dims_at(TENSOR_4D_CHANNELS) + j].join();
      }
    }
  } else if (input->dims_size() == 2) {
    assert(scales.size() == 1);
    assert(zero_point.size() == 1);
    float inverted = 1 / scales[0];
    int shift_val = calc_shift_val(inverted);
    for (int i = 0; i < input->dims_iterator(-1); ++i) {
      inputT val = input->at(i);
      outputT new_val = quantize_fn<inputT, outputT>(
          val, scales[0], zero_point[0], min_lim, max_lim, shift_val);
      output->set(i, new_val);
    }
  }
}

template <typename inputT, typename weightT, typename outputT>
class ConvEngine {
  // const Op::Layer::Conv *cc;
  const Tensor<weightT> *weights;
  const Tensor<outputT> *bias;
  int kn;
  int kh;
  int kw;
  int m_stride_h;
  int m_stride_w;
  int ki;
  std::vector<int> pad_vec;

  std::vector<int> w_zero_points;
  std::vector<int> x_zero_points;
  std::vector<int> y_zero_points;

  std::vector<float> x_scales;
  std::vector<float> y_scales;
  std::vector<float> w_scales;

  void _kernel(int k, const Tensor<inputT> *input, Tensor<outputT> *output);
  void _dw_kernel(int k, const Tensor<inputT> *input, Tensor<outputT> *output);

public:
  ConvEngine(const Op::Layer::Conv *cc);
  ConvEngine(const Op::Layer::QLinearConv *cc);
  ~ConvEngine();
  void run(const Tensor<inputT> *input, Tensor<outputT> *output);
};

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::ConvEngine(const Op::Layer::Conv *cc) {
  weights = new TensorExtant<weightT>(cc->weights);
  if (cc->bias) {
    bias = new TensorExtant<outputT>(cc->bias);
  } else {
    bias = nullptr;
  }
  kn = cc->m_cp.kn;
  kh = cc->m_cp.k[TENSOR_2D_HEIGHT];
  kw = cc->m_cp.k[TENSOR_2D_WIDTH];
  const int *pad = cc->m_cp.pad;
  pad_vec = std::vector<int>{pad[0], pad[1], pad[2], pad[3]};
  w_zero_points = std::vector<int>(cc->output_dims[0][TENSOR_4D_CHANNELS], 0);
  x_zero_points = std::vector<int>(cc->input_dims[0][TENSOR_4D_CHANNELS], 0);
  y_zero_points = std::vector<int>(cc->output_dims[0][TENSOR_4D_CHANNELS], 0);

  w_scales = std::vector<float>(cc->output_dims[0][TENSOR_4D_CHANNELS], 0);
  x_scales = std::vector<float>(cc->input_dims[0][TENSOR_4D_CHANNELS], 0);
  y_scales = std::vector<float>(cc->output_dims[0][TENSOR_4D_CHANNELS], 0);
  m_stride_h = cc->m_cp.stride[TENSOR_2D_HEIGHT];
  m_stride_w = cc->m_cp.stride[TENSOR_2D_WIDTH];
}

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::ConvEngine(
    const Op::Layer::QLinearConv *cc) {
  weights = new TensorExtant<weightT>(cc->weights);
  if (cc->bias) {
    bias = new TensorExtant<outputT>(cc->bias);
  } else {
    bias = nullptr;
  }
  kn = cc->m_cp.kn;
  kh = cc->m_cp.k[TENSOR_2D_HEIGHT];
  kw = cc->m_cp.k[TENSOR_2D_WIDTH];
  /* 'ki' is the starting row for the convolution operation on the input.
   *  In the original convolution layer, the offset is 0 (starts from the first row).
   *  But in the new decomposed convolution layer, the offset may start from 1.
   *  Since indexing starts at 0, we subtract 1 when the offset is greater than 0.
   */
  ki = cc->m_cp.ki > 0 ? cc->m_cp.ki - 1 : cc->m_cp.ki;
  const int *pad = cc->m_cp.pad;
  pad_vec = std::vector<int>{pad[0], pad[1], pad[2], pad[3]};
  using variantT = std::variant<int8_t, uint8_t>;
  w_zero_points = broadcast_vec(variant2vec<variantT, int>(cc->w_zero_point),
                                cc->output_dims[0][TENSOR_4D_CHANNELS]);
  x_zero_points = broadcast_vec(variant2vec<variantT, int>(cc->x_zero_point),
                                cc->input_dims[0][TENSOR_4D_CHANNELS]);
  y_zero_points = broadcast_vec(variant2vec<variantT, int>(cc->y_zero_point),
                                cc->output_dims[0][TENSOR_4D_CHANNELS]);

  w_scales = cc->w_scale;
  x_scales = cc->x_scale;
  y_scales = cc->y_scale;
  m_stride_h = cc->m_cp.stride[TENSOR_2D_HEIGHT];
  m_stride_w = cc->m_cp.stride[TENSOR_2D_WIDTH];
}

template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::_kernel(int k,
                                                   const Tensor<inputT> *input,
                                                   Tensor<outputT> *output) {
  int nb = input->dims_at(TENSOR_4D_BATCH);
  int ic = input->dims_at(TENSOR_4D_CHANNELS);
  int oh = output->dims_at(TENSOR_4D_HEIGHT);
  int ow = output->dims_at(TENSOR_4D_WIDTH);
  int out_index = 0;
  int w_index = 0;
  int in_index = 0;

  std::vector<int> o_strides = output->get_strides();
  std::vector<int> w_strides = weights->get_strides();
  std::vector<int> i_strides = input->get_strides();

  auto w_zp = w_zero_points.at(k);
  auto x_zp = x_zero_points.at(0);

  for (int ibi = 0; ibi < nb; ++ibi) {
    for (int ici = 0; ici < ic; ++ici) {
      for (int ohi = ki, toh = 0; toh < oh; ohi += m_stride_h, toh += 1) {
        for (int owi = 0, tow = 0; tow < ow; owi += m_stride_w, tow += 1) {
          out_index = ibi * o_strides[0] + k * o_strides[1] +
                      toh * o_strides[2] + tow * o_strides[3];
          outputT acc = output->at(out_index);
          outputT x_int_sum = 0;
          outputT w_int_sum = 0;
          for (int khi = 0; khi < kh; ++khi) {
            for (int kwi = 0; kwi < kw; ++kwi) {
              w_index = k * w_strides[0] + ici * w_strides[1] + khi * w_strides[2] + kwi * w_strides[3];
              in_index = ibi * i_strides[0] + ici * i_strides[1] +
                         (ohi + khi) * i_strides[2] +
                         (owi + kwi) * i_strides[3];

              outputT x_int = static_cast<outputT>(input->at(in_index));
              x_int_sum += x_int;
              outputT w_int = static_cast<outputT>(weights->at(w_index));
              w_int_sum += w_int;
              outputT val2 = x_int * w_int;
              acc += val2;
            }
          }
          acc -= (w_zp * x_int_sum);
          acc -= (x_zp * w_int_sum);
          acc += (x_zp * w_zp * kh * kw);
          output->set(out_index, acc);
        }
      }
    }
  }
}

template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::_dw_kernel(
    int k, const Tensor<inputT> *input, Tensor<outputT> *output) {
  int nb = input->dims_at(TENSOR_4D_BATCH);
  int ic = input->dims_at(TENSOR_4D_CHANNELS);
  int oh = output->dims_at(TENSOR_4D_HEIGHT);
  int ow = output->dims_at(TENSOR_4D_WIDTH);
  int out_index = 0;
  int w_index = 0;
  int in_index = 0;
  std::vector<int> o_strides = output->get_strides();
  std::vector<int> w_strides = weights->get_strides();
  std::vector<int> i_strides = input->get_strides();
  auto w_zp = w_zero_points.at(k);
  auto x_zp = x_zero_points.at(0);

  for (int ibi = 0; ibi < nb; ++ibi) {
    for (int ohi = ki, toh = 0; toh < oh; ohi += m_stride_h, toh += 1) {
      for (int owi = 0, tow = 0; tow < ow; owi += m_stride_w, tow += 1) {
        out_index = ibi * o_strides[0] + k * o_strides[1] + toh * o_strides[2] +
                    tow * o_strides[3];
        outputT acc = output->at(out_index);
        outputT x_int_sum = 0;
        outputT w_int_sum = 0;
        for (int khi = 0; khi < kh; ++khi) {
          for (int kwi = 0; kwi < kw; ++kwi) {
            w_index = k * w_strides[0] + 0 * w_strides[1] +
                      khi * w_strides[2] + kwi * w_strides[3];
            in_index = ibi * i_strides[0] + k * i_strides[1] +
                       (ohi + khi) * i_strides[2] + (owi + kwi) * i_strides[3];

            outputT x_int = static_cast<outputT>(input->at(in_index));
            x_int_sum += x_int;
            outputT w_int = static_cast<outputT>(weights->at(w_index));
            w_int_sum += w_int;
            outputT val2 = x_int * w_int;
            acc += val2;
          }
        }
        acc -= (w_zp * x_int_sum);
        acc -= (x_zp * w_int_sum);
        acc += (x_zp * w_zp * kh * kw);
        output->set(out_index, acc);
      }
    }
  }
}

template <typename inputT, typename weightT, typename outputT>
void ConvEngine<inputT, weightT, outputT>::run(const Tensor<inputT> *input,
                                               Tensor<outputT> *output) {
  Tensor<inputT> *padded_input =
      tensor_pad(input, pad_vec, static_cast<inputT>(x_zero_points.at(0)));

  bool dw = false;
  if (is_depthwise_conv(weights->get_dims(), input->get_dims())) {
    dw = true;
  }
  std::vector<std::thread> tc;
#if 1
  for (int k = 0; k < kn; ++k) {
    if (dw) {
      tc.push_back(std::thread(&ConvEngine<inputT, weightT, outputT>::_dw_kernel,
                               this, k, padded_input, output));
    } else {
      tc.push_back(std::thread(&ConvEngine<inputT, weightT, outputT>::_kernel,
                               this, k, padded_input, output));
    }
  }
  for (int k = 0; k < kn; ++k) {
    tc[k].join();
  }
#else
  for (int k = 0; k < kn; ++k) {
    ConvEngine<inputT, weightT, outputT>::_kernel(k, padded_input, output);
  }
#endif
  delete padded_input;
  if (bias != nullptr) {
    tensor_vector_add(output, output, bias);
  }
}

template <typename inputT, typename weightT, typename outputT>
ConvEngine<inputT, weightT, outputT>::~ConvEngine() {
  delete weights;
  delete bias;
}

template <typename inputT, typename outputT>
void dequantize(const Tensor<inputT> *input, Tensor<outputT> *output,
                const std::vector<float> &scales,
                const std::vector<int> &zero_point) {
  /* TODO: refactor this */
  if (input->dims_size() == 4) {
    auto bscales = broadcast_vec(scales, input->dims_at(TENSOR_4D_CHANNELS));
    auto bzero_points =
        broadcast_vec(zero_point, input->dims_at(TENSOR_4D_CHANNELS));

    for (int i = 0; i < input->dims_at(TENSOR_4D_BATCH); ++i) {
      for (int j = 0; j < input->dims_at(TENSOR_4D_CHANNELS); ++j) {
        for (int k = 0; k < input->dims_at(TENSOR_4D_HEIGHT); ++k) {
          for (int l = 0; l < input->dims_at(TENSOR_4D_WIDTH); ++l) {
            std::vector<int> in_index{i, j, k, l};
            inputT val = input->at(in_index);
            outputT new_val = dequantize_fn<inputT, outputT>(val, bscales[j],
                                                             bzero_points[j]);
            output->insert(in_index, new_val);
          }
        }
      }
    }
  } else if (input->dims_size() == 2) {
    assert(scales.size() == 1);
    assert(zero_point.size() == 1);
    for (int i = 0; i < input->dims_iterator(-1); ++i) {
      inputT val = input->at(i);
      outputT new_val =
          dequantize_fn<inputT, outputT>(val, scales[0], zero_point[0]);
      output->set(i, new_val);
    }
  }
}

template <typename T>
void logsoftmax(Tensor<T> *output, Tensor<T> *input, int axis) {
  if (output->get_dims() != input->get_dims()) {
    log_fatal("logsoftmax: input, output dims do not match");
  }
  int dims_sz = input->dims_size();
  if (abs(axis) >= dims_sz) {
    log_fatal(
        "logsoftmax: received out of bounds axis value {}. total dims {}\n",
        axis, dims_sz);
  }

  int true_axis = axis;
  if (axis < 0) {
    true_axis = dims_sz + axis;
  }
  std::vector<int> axis_v(true_axis, 0);
  TensorSlice<T> slice(input, axis_v);
  std::vector<int> exp_dims;
  exp_dims.push_back(slice.size());
  TensorCreate<T> exp_v(exp_dims);

  for (int i = 0; i < slice.size(); ++i) {
    exp_v.set(i, std::exp(slice.at(i)));
  }

  T reduced_sum =
      std::accumulate(exp_v.begin(), exp_v.end(), static_cast<T>(1.0));
  for (int i = 0; i < output->size(); ++i) {
    output->set(i, input->at(i));
  }
  TensorSlice<T> out_slice(output, axis_v);
  assert(out_slice.size() == slice.size());
  for (int i = 0; i < out_slice.size(); ++i) {
    out_slice.set(i, std::log(exp_v.at(i) / reduced_sum));
  }
}

/* FPGA style binary average calculation without division
 * operations
 */
template <typename T> static T avg(const std::vector<T>& vec) {
  auto v = vec;
  if (v.size() == 1) {
    return v.at(0);
  }
  int iterations = ceil(log2f(v.size()));
  for (int j = 0; j < iterations; ++j) {
    std::vector<T> new_vec;
    for (size_t i = 0; i < v.size() - (v.size() % 2); i += 2) {
      int tmp = v.at(i) + v.at(i + 1);
      tmp >>= 1;
      new_vec.push_back(tmp);
    }
    if (v.size() % 2 != 0) {
      new_vec.push_back(v.at(v.size() - 1));
    }
    v = new_vec;
  }
  return v.at(0);
}

/* floats and doubles dont go well with log-based averages */
template <> float avg<float>(const std::vector<float>& v) {
  float sum = std::accumulate(v.cbegin(), v.cend(), (float)0);
  return static_cast<float>(static_cast<float>(sum) / v.size());
}

template <> double avg<double>(const std::vector<double>& v) {
  double sum = std::accumulate(v.cbegin(), v.cend(), (double)0);
  return static_cast<double>(static_cast<double>(sum) / v.size());
}

template <typename T>
void average_pool(const Tensor<T> *input, Tensor<T> *output,
                  const Op::PoolParams &mp) {
  int pad_present = 0;
  for (int i = 0; i < 4; ++i) {
    if (mp.pad[i] != 0) {
      pad_present = 1;
      break;
    }
  }

  const Tensor<T> *padded_input;
  if (pad_present) {
    padded_input = tensor_pad(
        input, std::vector<int>{mp.pad[0], mp.pad[1], mp.pad[2], mp.pad[3]});
  } else {
    padded_input = input;
  }

  int output_depth = padded_input->dims_at(TENSOR_4D_CHANNELS);
  int output_height = mp_odims_row(mp, input->get_dims());
  int output_width = mp_odims_cols(mp, input->get_dims());

  for (int ici = 0; ici < output_depth; ++ici) {
    for (int ihi = 0; ihi < output_height * mp.stride[TENSOR_2D_HEIGHT];
         ihi += mp.stride[TENSOR_2D_HEIGHT]) {
      for (int iwi = 0; iwi < output_width * mp.stride[TENSOR_2D_WIDTH];
           iwi += mp.stride[TENSOR_2D_WIDTH]) {
        std::vector<T> vals;
        for (int khi = 0; khi < mp.k[TENSOR_2D_HEIGHT]; ++khi) {
          for (int kwi = 0; kwi < mp.k[TENSOR_2D_WIDTH]; ++kwi) {
            std::vector<int> in_index{0, ici, (ihi + khi), (iwi + kwi)};
            vals.push_back(padded_input->at(in_index));
          }
        }
        T avg_val = avg<T>(vals);
        std::vector<int> out_index{0, ici, ihi / mp.stride[TENSOR_2D_HEIGHT],
                                   iwi / mp.stride[TENSOR_2D_WIDTH]};
        output->insert(out_index, avg_val);
      }
    }
  }

  if (pad_present) {
    delete padded_input;
  }
}

template <typename T>
void batchnorm(const Tensor<T> *input, Tensor<T> *output, float epsilon,
               float momentum, const Tensor<T> *scale, const Tensor<T> *bias,
               const Tensor<T> *mean, const Tensor<T> *var) {
  ignore_unused(momentum);
  std::vector<int> index(input->dims_size(), 0);
  std::vector<int> dims{input->get_dims()};
  if (dims.size() != 4) {
    log_fatal("BatchNorm does only supports dims of 4\n");
  }
  for (int i = 0; i < input->size(); ++i) {
    int chan_n = index.at(TENSOR_4D_CHANNELS);
    T v = ((input->at(index) - mean->at(chan_n)) /
           sqrt(var->at(chan_n) + epsilon)) *
              scale->at(chan_n) +
          bias->at(chan_n);
    output->insert(index, v);
    increment_shape(index, dims);
  }
}

template <typename T> void xabs(const Tensor<T> *input, Tensor<T> *output) {
  std::vector<int> index(input->dims_size(), 0);
  std::vector<int> dims{input->get_dims()};
  for (int i = 0; i < input->size(); ++i) {
    output->insert(index, std::abs(input->at(index)));
    increment_shape(index, dims);
  }
}

template <typename T>
void reduce_mean(const Tensor<T> *input, Tensor<T> *output, int axis,
                 int keepdims) {
  log_warn("Ignoring axis {} parameter to reduce_mean\n", axis);
  log_warn("Ignoring keepdims {} parameter to reduce_mean\n", keepdims);
  *output = *input;
}
