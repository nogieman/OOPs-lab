#pragma once
#include "onnx.pb.h"
#include "utils.h"
#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

/* A general purpose interface to an n-dimensional tensor
 *
 * Implementation Details:
 *
 * The Tensor Base Class is abstract and defines a
 * blueprint for underlying implementations. The
 * implementations inherit and override neccessarily
 * the pure functions and optionally the regular virtual
 * functions. All 'read' type of functions i.e. functions
 * that do not mutate the underlying tensor are pure and
 * need to be defined by every derived class. Regular
 * virtual functions are 'read+write' mutating functions, they
 * should only be implemented if the derived class wishes
 * to.
 *
 * Derived classes implement (or wrap around) different
 * types of concrete data structures to create a common
 * interface that of the Tensor base class. See, currently
 * implemented derived classes TensorExtant and TensorCreate
 * below.
 */
template <typename T> class Tensor {
public:
  virtual std::string name() const;
  /* Read functions */
  virtual T at(std::vector<int> &at) const = 0;
  virtual T at(std::vector<int> &&at) const = 0;
  virtual T at(int index) const = 0;
  virtual int dims_size() const = 0;
  virtual int dims_at(int index) const = 0;
  virtual std::vector<int> get_dims() const = 0;
  virtual int dims_iterator(int index) const = 0;
  virtual int size() const = 0;
  virtual std::vector<T> get() const = 0;
  virtual void print() const = 0;
  virtual std::vector<int> get_strides() const = 0;
  /* Derived classes implement this and return whether delete can be called
   * on the underlying tensor. For derived types (such as TensorExtant and
   * TensorSlice) that wrap around some other type and do not fully own their
   * tensors, freeable() returns false. For TensorCreate(), true is returned
   * as it fully own the underlying Tensor
   * */
  virtual bool freeable() const = 0;
  virtual ~Tensor() = 0;

  /* Write functions */

  /* insert one element at a time */
  virtual T* data();
  virtual void insert(std::vector<int> &at, T data);
  virtual void push_back(T data);
  virtual void push_back(const std::vector<T> &data);
  virtual void set_dims(std::vector<int> const &temp_dims);
  virtual void clear();
  virtual void shrink_to_fit();
  virtual void set(int index, T val);
  virtual Tensor<T> &operator=(const Tensor<T> &rhs);
  virtual Tensor<T> &operator=(Tensor<T> &&rhs);
  virtual typename std::vector<T>::iterator begin();
  virtual typename std::vector<T>::iterator end();
  virtual std::vector<T>&& rget();
};

template <typename T> std::string Tensor<T>::name() const {
  return "(null)";
}

template <typename T> T* Tensor<T>::data() {
  log_fatal("Un-implemented function\n");
}

template <typename T> void Tensor<T>::insert(std::vector<int> &, T) {
  log_fatal("Un-implemented function\n");
}
template <typename T> void Tensor<T>::push_back(T) {
  log_fatal("Un-implemented function\n");
}
template <typename T> void Tensor<T>::push_back(const std::vector<T> &) {
  log_fatal("Un-implemented function\n");
}
template <typename T> void Tensor<T>::set_dims(std::vector<int> const &) {
  log_fatal("Un-implemented function\n");
}
template <typename T> void Tensor<T>::clear() {
  log_fatal("Un-implemented function\n");
}

template <typename T> void Tensor<T>::shrink_to_fit() {
  log_fatal("Un-implemented function\n");
}
template <typename T> void Tensor<T>::set(int, T) {
  log_fatal("Un-implemented function\n");
}

template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor<T> &) {
  log_fatal("Un-implemented function\n");
}

template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor<T> &&) {
  log_fatal("Un-implemented function\n");
}

template <typename T> typename std::vector<T>::iterator Tensor<T>::begin() {
  log_fatal("Un-implemented function\n");
  return std::vector<T>().begin();
}

template <typename T> typename std::vector<T>::iterator Tensor<T>::end() {
  log_fatal("Un-implemented function\n");
  return std::vector<T>().end();
}

template <typename T> std::vector<T>&& Tensor<T>::rget() {
  log_fatal("Un-implemented function\n");
  return std::vector<T>();
}

template <typename T> Tensor<T>::~Tensor() {}

/* TensorExtant - Wrapper around onnx::TensorProto
 *
 * TensorExtant deduces where actual data is stored
 * in a onnx::TensorProto object (where weights and biases
 * of a NN are stored) and keeps a pointer
 * to it. It is read-only, does not allow mutating
 * weights
 */
template <typename T> class TensorExtant : public Tensor<T> {
private:
  std::vector<int> dims;
  std::vector<int> stride;
  const onnx::TensorProto *ptr;
  /* Where the actual data resides in memory */
  const T *data;
  /* Initialize `dims` and `ptr`, `data` is initialized
   * by template specialized constructors
   */
  void init_dims(const onnx::TensorProto *ptr);

public:
  /* There are no generic constructors for TensorExtant,
   * all are specialized. See tensor.cpp.
   */
  TensorExtant(const onnx::TensorProto *ptr);
  T at(std::vector<int> &at) const override;
  T at(std::vector<int> &&at) const override;
  T at(int index) const override;
  int dims_size() const override;
  int dims_at(int index) const override;
  std::vector<int> get_dims() const override;
  int dims_iterator(int index) const override;
  int size() const override;
  bool freeable() const override;
  /* Expensive function, creates a copy of the
   * underlying data
   */
  std::vector<T> get() const override;
  std::vector<int> get_strides() const override;
  void print() const override;
  ~TensorExtant();
};

template <typename T>
void TensorExtant<T>::init_dims(const onnx::TensorProto *ptr) {
  dims.resize(ptr->dims_size());
  std::copy(ptr->dims().begin(), ptr->dims().end(), dims.begin());
  this->ptr = ptr;
  this->stride = get_stride_from_shape(dims);
}

template <typename T> T TensorExtant<T>::at(int index) const {
  assert(index < this->dims_iterator(-1));
  return data[index];
}

template <typename T> T TensorExtant<T>::at(std::vector<int> &index) const {
  assert(index.size() == dims.size());
  int sum = 0;
  for (size_t i = 0; i < index.size(); i++) {
    assert(index[i] <= dims[i]);
    sum += index[i] * stride[i];
  }
  return at(sum);
}

template <typename T> T TensorExtant<T>::at(std::vector<int> &&index) const {
  return at(index);
}

template <typename T> void TensorExtant<T>::print() const {
  for (int i = 0; i < dims_iterator(-1); ++i) {
    if (i % 9 == 0) {
      std::cout << '\n';
    }
    std::cout << data[i] << '\t';
  }
}

template <typename T> std::vector<int> TensorExtant<T>::get_dims() const {
  return dims;
}

template <typename T> int TensorExtant<T>::dims_size() const {
  return dims.size();
}

template <typename T> int TensorExtant<T>::dims_at(int index) const {
  assert(index < dims.size());
  return dims[index];
}

template <typename T> int TensorExtant<T>::dims_iterator(int index) const {
  int a = 1;
  for (size_t i = 1; i < dims.size() - index; i++) {
    a *= dims[index + i];
  }
  return a;
}

template <typename T> int TensorExtant<T>::size() const {
  return dims_iterator(-1);
}

template <typename T> std::vector<T> TensorExtant<T>::get() const {
  std::vector<T> ret(dims_iterator(-1));
  for (int i = 0; i < this->size(); ++i) {
    ret[i] = data[i];
  }
  return ret;
}

template <typename T> std::vector<int> TensorExtant<T>::get_strides() const {
  return stride;
}

template <typename T> bool TensorExtant<T>::freeable() const { return false; }

template <typename T> TensorExtant<T>::~TensorExtant() {
  // frees nothing as it owns nothing
}

template <typename T> class TensorCreate : public Tensor<T> {
  std::vector<int> dims;
  std::vector<int> stride;
  std::vector<T> vec;

  std::string m_name{"null"};

public:
  TensorCreate() = delete;

  TensorCreate(std::vector<T> const &v, std::vector<int> const &dim) {
    dims = dim;
    vec = v;
    stride = get_stride_from_shape(dim);
  }

  TensorCreate(std::vector<int> const &dim) {
    dims = dim;
    vec.resize(dims_iterator(-1), 0);
    stride = get_stride_from_shape(dim);
  }

  TensorCreate(std::vector<int> const &dim, const std::string& name): TensorCreate(dim) {
    this->m_name = name;
  }

  TensorCreate(py::array arr) {
    if (!py::isinstance<py::array_t<T>>(arr)) {
      log_fatal("input array type mismatch: Expected array of type {}\n", typeid(T).name());
    }
    auto buf = arr.request();
    T *udata = static_cast<T *>(buf.ptr);
    vec.assign(udata, udata + buf.size);
    for (int i = 0; i < buf.shape.size(); ++i) {
      dims.push_back(buf.shape.at(i));
    }
    stride = get_stride_from_shape(dims);
  }

  TensorCreate(py::array arr, const std::string &name) : TensorCreate(arr) {
    this->m_name = name;
  }

  TensorCreate(const Tensor<T> *t) {
    vec = t->get();
    dims = t->get_dims();
    stride = t->get_strides();
  }

  TensorCreate(const Tensor<T> *t, const std::string &name) : TensorCreate(t) {
    this->m_name = name;
  }

  T at(std::vector<int> &at) const override {
    assert(at.size() == dims.size());
    int sum = 0;
    for (size_t i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum += at[i] * stride[i];
    }
    return vec.at(sum);
  }

  T at(std::vector<int> &&at) const override { return this->at(at); }

  T at(int index) const override { return vec.at(index); }

  int dims_size() const override { return dims.size(); }

  int dims_at(int index) const override { return dims.at(index); }
  void push_back(T data) override { vec.push_back(data); }

  void push_back(const std::vector<T> &data) {
    for (const T &i : data) {
      this->push_back(i);
    }
  }

  void insert(std::vector<int> &at, T data) override {
    assert(at.size() <= dims.size());
    int sum = 0;
    for (size_t i = 0; i < at.size(); i++) {
      assert(at[i] <= dims[i]);
      sum += at[i] * stride[i];
    }
    vec[sum] = data;
  }

  void set_dims(std::vector<int> const &temp_dims) override {
    dims = temp_dims;
    stride = get_stride_from_shape(temp_dims);
    return;
  }
  std::vector<int> get_dims() const override { return dims; }
  int dims_iterator(int index) const override {
    int a = 1;
    for (size_t i = 1; i < dims.size() - index; i++) {
      a *= dims[index + i];
    }
    return a;
  }

  void clear() override { vec.clear(); }

  void shrink_to_fit() override { vec.shrink_to_fit(); }

  int size() const override { return vec.size(); }

  std::vector<T> get() const override { return vec; }
  std::vector<T>&& rget() override { return std::move(vec); }

  std::vector<int> get_strides() const override { return stride; }

  void set(int index, T val) override { vec.at(index) = val; }

  virtual Tensor<T> &operator=(const Tensor<T> &rhs) override {
    this->dims = rhs.get_dims();
    this->vec = rhs.get();
    return *this;
  }

  virtual Tensor<T> &operator=(Tensor<T> &&rhs) override {
    this->dims = rhs.get_dims();
    this->vec = std::move(rhs.rget());
    return *this;
  }

  void print() const override { print_vec("tensor", vec); }

  typename std::vector<T>::iterator begin() override { return vec.begin(); }
  typename std::vector<T>::iterator end() override { return vec.end(); }

  bool freeable() const override { return true; }

  T *data() override { return vec.data(); }

  std::string name() const override { return m_name; };

  ~TensorCreate();
};

template <typename T> TensorCreate<T>::~TensorCreate() {}

template <typename T> class TensorSlice : public Tensor<T> {
  Tensor<T> *src;
  std::vector<int> slice;
  /* Linear offset wrt the original linear representation
   * of src tensor
   */
  int offset;
  /* Linear size upper bound of this slice */
  int slice_size;

  std::vector<int> dims;

public:
  TensorSlice(Tensor<T> *src, std::vector<int> slice);
  T at(std::vector<int> &index) const override;
  T at(std::vector<int> &&index) const override;
  T at(int index) const override;
  int dims_size() const override;
  int dims_at(int index) const override;
  std::vector<int> get_dims() const override;
  int dims_iterator(int index) const override;
  int size() const override;
  std::vector<T> get() const override;
  void print() const override;
  bool freeable() const override;
  std::vector<int> get_strides() const override;
  std::string name() const override;

  ~TensorSlice();

  /* Write functions */

  void set(int index, T val);
#if 0
  void insert(std::vector<int> &at, T data);
  void push_back(T data);
  void push_back(const std::vector<T>& data);
  void clear();
  void shrink_to_fit();
  Tensor<T>& operator=(Tensor<T>& rhs);
  typename std::vector<T>::iterator begin();
  typename std::vector<T>::iterator end();
#endif
};

template <typename T>
TensorSlice<T>::TensorSlice(Tensor<T> *src, std::vector<int> slice) {
  assert(slice.size() <= src->dims_size());

  this->slice = slice;
  this->src = src;
  this->offset = 0;
  std::vector<int> strides = get_stride_from_shape(src->get_dims());
  for (size_t i = 0; i < slice.size(); ++i) {
    this->offset += (strides[i] * slice[i]);
  }
  for (int i = static_cast<int>(slice.size()); i < src->dims_size(); ++i) {
    this->dims.push_back(src->dims_at(i));
  }
  this->slice_size = prod(dims.begin(), dims.end(), 1);
}

template <typename T>
std::string TensorSlice<T>::name() const {
  return src->name();
}

template <typename T> T TensorSlice<T>::at(std::vector<int> &index) const {
  std::vector<int> new_index = concat(slice, index);
  return src->at(new_index);
}

template <typename T> T TensorSlice<T>::at(std::vector<int> &&index) const {
  return at(index);
}

template <typename T> T TensorSlice<T>::at(int index) const {
  assert(index >= 0);
  assert(index < slice_size);
  return src->at(offset + index);
}

template <typename T> int TensorSlice<T>::dims_size() const {
  return dims.size();
}
template <typename T> int TensorSlice<T>::dims_at(int index) const {
  return dims.at(index);
}
template <typename T> std::vector<int> TensorSlice<T>::get_dims() const {
  return dims;
}
template <typename T> int TensorSlice<T>::dims_iterator(int index) const {
  int a = 1;
  for (size_t i = 1; i < dims.size() - index; i++) {
    a *= dims[index + i];
  }
  return a;
}
template <typename T> int TensorSlice<T>::size() const { return slice_size; }

template <typename T> std::vector<T> TensorSlice<T>::get() const {
  /* TODO: expensive function, remove get completely from tensor's
   * interface
   */
  std::vector<T> ret(slice_size);
  for (int i = 0; i < slice_size; ++i) {
    ret[i] = at(i);
  }
  return ret;
}

template <typename T> std::vector<int> TensorSlice<T>::get_strides() const {
  log_fatal("get_stride() for TensorSlice is unimplemented\n");
  return std::vector<int>{};
}

template <typename T> void TensorSlice<T>::print() const {
  for (int i = 0; i < slice_size; ++i) {
    std::cout << at(i) << ' ';
  }
  std::cout << '\n';
}

template <typename T> bool TensorSlice<T>::freeable() const { return false; }

template <typename T> TensorSlice<T>::~TensorSlice() {
  // frees nothing as it owns nothing
}

template <typename T> void TensorSlice<T>::set(int index, T data) {
  assert(index >= 0);
  assert(index < slice_size);
  return src->set(offset + index, data);
}

template <typename T>
Tensor<T> *tensor_sub_zp(const Tensor<T> *input, const std::vector<int> &zp) {
  assert(input->dims_size() == 4 && "tensor_pad assumes 4d inputs");
  std::vector<int> new_dims = input->get_dims();
  Tensor<T> *output = new TensorCreate<T>(new_dims);
  for (int i = 0; i < new_dims[0]; ++i) {
    for (int j = 0; j < new_dims[1]; ++j) {
      for (int k = 0; k < new_dims[2]; ++k) {
        for (int l = 0; l < new_dims[3]; ++l) {
          std::vector<int> out_index{i, j, k, l};
          T v = input->at(out_index) - zp[j];
          output->insert(out_index, v);
        }
      }
    }
  }
  return output;
}

template <typename T>
Tensor<T> *tensor_pad(const Tensor<T> *input, const std::vector<int> &pads,
                      T pad_val = 0) {
  assert(input->dims_size() == 4 && "tensor_pad assumes 4d inputs");
  std::vector<int> new_dims = get_dims_after_pad(input->get_dims(), pads);
  Tensor<T> *output = new TensorCreate<T>(new_dims);
  for (int i = 0; i < new_dims[0]; ++i) {
    for (int j = 0; j < new_dims[1]; ++j) {
      for (int k = 0; k < new_dims[2]; ++k) {
        for (int l = 0; l < new_dims[3]; ++l) {
          std::vector<int> out_index{i, j, k, l};
          if (islying(k, l, input->dims_at(2), input->dims_at(3), pads)) {
            output->insert(out_index, pad_val);
          } else {
            std::vector<int> in_index{i, j, k - pads[0], l - pads[1]};
            output->insert(out_index, input->at(in_index));
          }
        }
      }
    }
  }
  return output;
}

template <typename T> Tensor<T> *get_slice(Tensor<T> *src, std::vector<int> s) {
  std::vector<int> dd = src->get_dims();
  assert(dd.size() == 4);
  dd.at(0) = 1;
  Tensor<T> *ret = new TensorCreate<T>(dd);
  TensorSlice<T> slice(src, s);
  for (int i = 0; i < slice.size(); ++i) {
    ret->set(i, slice.at(i));
  }
  return ret;
}

template<typename T> std::string numpy_dtype();

/* Homemade .npy pickle function */
template <typename T>
void pickle_tensor(const Tensor<T> *t, std::string filename) {
  /* replace '/' with '_' */
  auto mod_filename = sed(filename, '/', '_') + std::string(".npy");
  static_assert(std::is_arithmetic<T>::value,
                "Only arithmetic types supported");
  // Compute total number of elements
  size_t total_elems = t->size();
  std::vector<int> shape = t->get_dims();

  std::ofstream out(mod_filename, std::ios::binary);
  if (!out) {
    log_fatal("pickle_tensor: Failed to open file {}\n", mod_filename);
  }
  out.write("\x93NUMPY", 6);
  out.put(1); // major version
  out.put(0); // minor version
  std::string shape_str = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_str += std::to_string(shape[i]);
    if (i + 1 < shape.size())
      shape_str += ", ";
  }
  if (shape.size() == 1)
    shape_str += ",";
  shape_str += ")";
  std::string header = "{'descr': '" + numpy_dtype<T>() +
                       "', 'fortran_order': False, 'shape': " + shape_str +
                       ", }";
  size_t header_len = header.size() + 1; // +1 for '\n'
  size_t total_len = 10 + header_len;
  size_t pad = (16 - (total_len % 16)) % 16;
  header += std::string(pad, ' ');
  header += '\n';
  uint16_t header_size = static_cast<uint16_t>(header.size());
  out.put(static_cast<char>(header_size & 0xFF));
  out.put(static_cast<char>((header_size >> 8) & 0xFF));
  out.write(header.c_str(), header.size());
  /* FIXME: this copies the tensor twice, avoid this */
  const char *data = reinterpret_cast<const char *>(t->get().data());
  out.write(data, total_elems * sizeof(T));
}
