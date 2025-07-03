#include "utils.h"
#include "instructions.h"
#include "pch.h"
#include "tensor.h"
#include "version.h"
#include <cstdarg>
#include <regex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// #include <cstdint>
// #include <typeinfo>

/* Used by run_* functions in executor to free under-lying Tensor
 * pointers. This could very well be templated by that requires the
 * caller to know the type of the under-lying data that is being
 * abstracted by std::any. This is not true for us, thus the if-else
 * ladder.
 */
void TensorPool::free(int index) {
  std::any v = pool.at(index);
  if (v.type() == typeid(Tensor<int8_t> *)) {
    Tensor<int8_t> *dd = std::any_cast<Tensor<int8_t> *>(v);
    /* TODO: temporary hack, find a cleaner workaround */
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int16_t> *)) {
    Tensor<int16_t> *dd = std::any_cast<Tensor<int16_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int> *)) {
    Tensor<int> *dd = std::any_cast<Tensor<int> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int64_t> *)) {
    Tensor<int64_t> *dd = std::any_cast<Tensor<int64_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<int32_t> *)) {
    Tensor<int32_t> *dd = std::any_cast<Tensor<int32_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<float> *)) {
    Tensor<float> *dd = std::any_cast<Tensor<float> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<double> *)) {
    Tensor<double> *dd = std::any_cast<Tensor<double> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else if (v.type() == typeid(Tensor<uint8_t> *)) {
    Tensor<uint8_t> *dd = std::any_cast<Tensor<uint8_t> *>(v);
    if (dd->freeable()) {
      delete dd;
    }
  } else {
    log_fatal("Unknown type: {}, cannot free. Support has to be added\n",
              v.type().name());
  }
  pool.at(index).reset();
}

void TensorPool::free() {
  for (size_t i = 0; i < pool.size(); ++i) {
    pool.at(i).reset();
  }
}

void TensorPool::print() const {
  for (size_t i = 0; i < pool.size(); ++i) {
    std::cout << "At index: " << i << " Type: " << pool.at(i).type().name()
              << '\n';
  }
}

bool TensorPool::has_value(int index) { return pool.at(index).has_value(); }

void TensorPool::resize(int size) { pool.resize(size); }

std::vector<std::any>::iterator TensorPool::begin() {
  return pool.begin();
}
std::vector<std::any>::iterator TensorPool::end() {
  return pool.end();
}

struct meta_info {
  void *data;
  int itemsize;
  std::string format;
  int ndim;
  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;
  std::string name;
};

template <typename T>
static meta_info get_meta_info(const std::any& val) {
  meta_info ret;
  Tensor<T> *t = std::any_cast<Tensor<T>*>(val);
  auto dims = t->get_dims();
  auto strides = t->get_strides();
  ret.data = t->data();
  ret.itemsize = sizeof(T);
  ret.format = py::format_descriptor<T>::format();
  ret.ndim = dims.size();
  ret.shape.resize(dims.size());
  std::copy(dims.begin(), dims.end(), ret.shape.begin());
  ret.strides.resize(strides.size());
  for (int i = 0; i < strides.size(); ++i) {
    ret.strides[i] = strides[i] * sizeof(T);
  }
  ret.name = t->name();
  return ret;
}

py::list extract_pool(TensorPool &pool) {
  py::list ret;
  for (const auto &v : pool) {
    meta_info meta;
    if (v.type() == typeid(Tensor<int8_t> *)) {
      meta = get_meta_info<int8_t>(v);
    } else if (v.type() == typeid(Tensor<int16_t> *)) {
      meta = get_meta_info<int16_t>(v);
    } else if (v.type() == typeid(Tensor<int> *)) {
      meta = get_meta_info<int>(v);
    } else if (v.type() == typeid(Tensor<int64_t> *)) {
      meta = get_meta_info<int64_t>(v);
    } else if (v.type() == typeid(Tensor<float> *)) {
      meta = get_meta_info<float>(v);
    } else if (v.type() == typeid(Tensor<double> *)) {
      meta = get_meta_info<double>(v);
    } else if (v.type() == typeid(Tensor<uint8_t> *)) {
      meta = get_meta_info<uint8_t>(v);
    } else {
      log_fatal("Unknown type: {}, cannot free. Support has to be added\n",
                v.type().name());
    }
    py::array arr(py::buffer_info(nullptr, meta.itemsize, meta.format,
                                  meta.ndim, meta.shape, meta.strides));
    memcpy(arr.mutable_data(), meta.data, prod(meta.shape) * meta.itemsize);
    ret.append(py::make_tuple(meta.name, arr));
  }
  return ret;
}

/* path: such as "/usr/bin/file.txt"
 * returns: "file.txt"
 */
std::filesystem::path extract_basename(const std::string &path) {
  std::filesystem::path fs_path(path);
  return fs_path.filename();
}

/* path: such as "/usr/bin/file.txt"
 * returns: "/usr/bin"
 */
std::filesystem::path extract_dirname(const std::string &path) {
  std::filesystem::path fs_path(path);
  return fs_path.remove_filename();
}

/* true if two shapes are broadcastable
 * see https://numpy.org/doc/stable/user/basics.broadcasting.html
 */
bool is_broadcastable(const std::vector<int> &shape1,
                      const std::vector<int> &shape2) {
  if (shape1.size() == 1 || shape2.size() == 1) {
    return true;
  }

  if (shape1.size() == shape2.size()) {
    /* iterate from rhs */
    for (int i = shape1.size() - 1; i > 0; --i) {
      if (shape1[i] != shape2[i]) {
        return false;
      }
    }
    return true;
  }

  return false;
}

std::vector<float> compute_output_scale(const std::vector<float> &x_scale,
                                        const std::vector<float> &w_scale,
                                        const std::vector<float> &y_scale) {
  auto new_x_scale = broadcast_vec(x_scale, w_scale.size());
  auto new_y_scale = broadcast_vec(y_scale, w_scale.size());
  std::vector<float> ret(w_scale.size());
  for (size_t i = 0; i < w_scale.size(); ++i) {
    ret[i] = new_y_scale[i] / (new_x_scale[i] * w_scale[i]);
  }
  return ret;
}

std::vector<int> get_dims_after_pad(std::vector<int> current_dims,
                                    const std::vector<int> &pad) {
  auto last_dim = current_dims.rbegin();
  auto second_last_dim = current_dims.rbegin() + 1;

  *second_last_dim = *second_last_dim + pad[I_UP] + pad[I_DOWN];
  *last_dim = *last_dim + pad[I_LEFT] + pad[I_RIGHT];
  return current_dims;
}

bool islying(int i, int j, int rows, int cols, const std::vector<int> &pad) {
  if (((j >= 0 && j < pad[I_LEFT]) ||
       (j >= (cols + pad[I_LEFT]) && j < (cols + pad[I_LEFT] + pad[I_RIGHT]))) ||
      ((i >= 0 && i < pad[I_UP]) ||
       (i >= (rows + pad[I_UP]) && i < (rows + pad[I_UP] + pad[I_DOWN])))) {
    return true;
  } else {
    return false;
  }
}

int cmp_dims(const std::vector<int> &dim1, const std::vector<int> &dim2) {
  int p1 = prod(dim1.begin(), dim1.end(), 1);
  int p2 = prod(dim2.begin(), dim2.end(), 2);

  if (p1 > p2) {
    return 1;
  } else if (p1 == p2) {
    return 0;
  } else {
    return -1;
  }
}

int count_digits(int a) {
  int count = 0;
  while (a > 0) {
    a /= 10;
    count++;
  }
  return count;
}

std::vector<int> get_sa_arch() {
  if (!gbl_args.has_option("sa-arch")) {
    log_fatal("cant get architecture for sa, please use --sa-arch option\n");
  }
  std::string arch_list = gbl_args["sa-arch"].as<std::string>();
  std::vector<int> mnk = parse_csv_string<int>(arch_list);
  if (mnk.size() == 0 || mnk.size() != 3) {
    log_fatal("Ill formatted dimension string to --sa-arch: {}, expects "
              "3-vector string like m,n,k",
              arch_list);
  }
  return mnk;
}

int get_verbose() {
  return gbl_args.has_option("verbose") || gbl_args.has_option("verbose2");
}

int get_verbose2() {
  return gbl_args.has_option("verbose2");
}

int get_va_size() {
  if (!gbl_args.has_option("vasize")) {
    log_fatal(
        "can't deduce vector array size, use option --vasize to provide one\n");
  }
  int va_size = gbl_args["vasize"].as<int>();
  return va_size;
}

void Argparse::parse(int argc, char *argv[]) {
  try {
    args = argparser.parse(argc, argv);
  } catch (const std::exception &e) {
    print_usage();
    std::cerr << "FATAL: " << e.what() << '\n';
    exit(1);
  }
}

argagg::option_results &Argparse::operator[](const std::string &name) {
  return args[name];
}

bool Argparse::has_option(const std::string &name) const {
  return args.has_option(name);
}

/* NOTE: the 'values' are not ~set~ per se but appended to a vector
 * of values inside. when values are retrieved with [] operator,
 * by default, the last updated value is retrieved
 */
void Argparse::set_option(const char* opt_name, const char* val) {
  argagg::option_result ag; 
  ag.arg = val;
  args.options[opt_name].all.push_back(ag);
}

#define BOLD(str) ("\033[1m" str "\033[0m")

void Argparse::print_usage() const {
  auto color_options = [](const std::string &s) -> std::string {
    std::regex opt_reg(" +--[a-zA-Z0-9-]+");
    std::string result = std::regex_replace(s, opt_reg, "\033[34m$&\033[0m");
    return result;
  };

  auto print_ss_vector = [&color_options](const auto &ssv) {
    for (const auto &i : ssv) {
      std::cout << "  " << "\033[33m" << color_options(i.first) << "\033[0m"
                << '\n';
      std::cout << "  " << color_options(i.second) << '\n';
      std::cout << '\n';
    }
  };

  std::cout << BOLD("USAGE: gaticc [OPTIONS]\n\n");
  std::cout << argparser << '\n';
  std::cout << BOLD("USAGE EXAMPLES:\n\n");
  print_ss_vector(_usage_examples);
  std::cout << BOLD("CONCEPTS:\n\n");
  print_ss_vector(_concepts);
}

void Argparse::print_version() const {
  std::cerr << "Gaticc: " << GATICC_VERSION << '\n';
  std::cerr << "Boost: " << GATICC_BOOST_VERSION << '\n';
  std::cerr << "Protobuf: " << GATICC_PROTOBUF_VERSION << '\n';
  std::cerr << "ISA Version: " << ISA_VERSION << '\n';
}

void check_c_return_val(int val, const char *err) {
  if (val != 0) {
    log_fatal("{}: {}\n", err, strerror(errno));
  }
}

void check_c_return_val(void *val, const char *err) {
  if (val == NULL) {
    log_fatal("{}: {}\n", err, strerror(errno));
  }
}

/* args:
 *  vector<string> {
 *  "-c", "build/tests/models/fcv_1_20_int8.onnx",
 *  "--ramsize", "512",
 *  "--sa-arch", "9,4,4",
 *  "--vasize", "32",
 *  "--accbuf-size", "4096",
 *  "--pretty-print-inst"
 *  };
 *
 * Return a char** that can be passed to gbl_args.parse()
 */
std::pair<int, char **> argv_create(const std::vector<std::string> &opts) {
  char **ptr = new char *[opts.size()];
  for (size_t i = 0; i < opts.size(); ++i) {
    const char *p = opts.at(i).c_str();
    ptr[i] = new char[opts.at(i).size()];
    strcpy(ptr[i], p);
  }
  return std::pair<int, char **>{opts.size(), ptr};
}

void argv_delete(int argc, char **argv) {
  for (int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete argv;
}

std::vector<int> reduced_shape(const std::vector<int> &dims, int reduction_axis,
                               int keepdims) {
  if (reduction_axis == -1) {
    return std::vector<int>{1};
  }
  if (reduction_axis >= static_cast<int>(dims.size())) {
    log_fatal("in reduced_shape(), axis {} greater than total dims {}\n",
              reduction_axis, dims.size());
  }
  std::vector<int> new_shape;
  for (int i = 0; i < static_cast<int>(dims.size()); ++i) {
    if (i == reduction_axis) {
      if (keepdims) {
        new_shape.push_back(1);
      }
      continue;
    }
    new_shape.push_back(dims.at(i));
  }
  return new_shape;
}

std::vector<int> unsqueeze_shape(const std::vector<int> &dims,
                                 const std::vector<int> &indices) {
  std::vector<int> new_shape;
  for (size_t i = 0, j = 0; i < dims.size() + indices.size(); ++i) {
    if (std::find(indices.cbegin(), indices.cend(), i) != indices.cend()) {
      new_shape.push_back(1);
    } else {
      if (j >= dims.size()) {
        log_fatal("Index {} is out of bounds of tensors of dim size {}\n", j,
                  dims.size());
      }
      new_shape.push_back(dims.at(j++));
    }
  }
  return new_shape;
}

std::vector<int> concat_shape(const IVec2D &dims, int axis) {
  const auto &first_dims = dims.at(0);
  std::vector<int> new_shape{first_dims};
  for (size_t i = 1; i < dims.size(); ++i) {
    if (dims.at(i).size() != first_dims.size()) {
      log_fatal(
          "all dims must be of the same size, got dim of size {} at index {}\n",
          dims.at(i).size(), i);
    }

    for (int j = 0; j < static_cast<int>(new_shape.size()); ++j) {
      if (j == axis) {
        new_shape.at(j) += dims.at(i).at(j);
      } else {
        if (new_shape.at(j) != dims.at(i).at(j)) {
          log_fatal(
              "all the input array dimensions except for the concatenation "
              "axis must match exactly, but along dimension {}, the array at "
              "index {} has size {} and the array at index 0 has size {}",
              i, j, dims.at(i).at(j), new_shape.at(j));
        }
      }
    }
  }
  return new_shape;
}

std::string sed(const std::string& src, char c, char r) {
  std::string ret(src);
  std::replace(ret.begin(), ret.end(), c, r);
  return ret;
}

std::vector<int> permute(const std::vector<int> &v, std::vector<int> perm) {
  std::for_each(perm.begin(), perm.end(), [&v](int i) {
    ignore_unused(i);
    assert((i < v.size()) ? true : false);
  });
  std::vector<int> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    ret.at(i) = v.at(perm.at(i));
  }
  return ret;
}

int dot(const std::vector<int> &a, const std::vector<int> &b) {
  int sum = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}
