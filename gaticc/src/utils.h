#pragma once

#include <any>
#include <array>
#include <bitset>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <typeinfo>
#include <unistd.h>
#include <variant>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;
/* from https://github.com/vietjtnguyen/argagg
 * for options parsing. See class Argparse for more info
 */
#include "argagg.h"

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &obj) {
  os << '(';
  for (size_t i = 0; i < obj.size(); ++i) {
    os << obj.at(i);
    if (i < (obj.size() - 1)) {
      os << ',';
    }
  }
  os << ')';
  return os;
}

/* shutoff -Wunused-variable
 * see https://herbsutter.com/2009/10/18/mailbag-shutting-up-compiler-warnings/
 */
template <typename T> void ignore_unused(const T &) {}

/* Wrapper over argagg library */
class Argparse {
  argagg::parser_results args;
  /* To extend, add a new definition here */
  argagg::parser argparser{
      /* name      invokation         description              expected
       *                                                       args */
      {
          {"help", {"-h", "--help"}, "print help and exit", 0},
          {"verbose", {"-v", "--v", "--verbose"}, "print log messages to standard out", 0},
          {"verbose2", {"--vv", "--verbose2"}, "print more log messages to standard out", 0},
          {"timeest",
           {"--timeest"},
           "print estimated time that a model would take based on FLOP counts "
           "(this does not account for latencies such as that of DRAM)"
           "Args: [frequency (int)]",
           1},
          {"info",
           {"-i", "--info"},
           "Query information from model. Args: <onnx_model>",
           1},
          {"sim", {"-s", "--sim"}, "Simulate inference on an input.", 1},
          {"venv-path",
           {"--venv-path"},
           "Append venv-path to sys.path while loading the interpreter. Args: "
           "[ : "
           "separated path list]",
           1},
          {"loadpy",
           {"--loadpy"},
           "Load the python script mentioned in arg. Usually the script "
           "that'll "
           "contain pre/post process functions for --sim"
           "\n\tArgs: [script_name.py]",
           1},
          {"input_path",
           {"--inputpath"},
           "specify input to model as a resident file path (one file at a "
           "time)",
           1},
          {"sa-arch",
           {"--sa-arch"},
           "systolic array architecture. Args: [comma sep values]. accepts "
           "args ",
           1},
          {"ramsize",
           {"--ramsize"},
           "ram size in MB. Args: int. For ex, --ramsize 512",
           1},
          {"vasize",
           {"--vasize"},
           "Vector Array size. Usually equivalent to per cycle DRAM bandwidth "
           "in "
           "bytes. Args: int. For ex, --vasize 32",
           1},
          {"pretty-print-blob",
           {"--pretty-print-blob"},
           "pretty print entire blob",
           0},
          {"pretty-print-inst",
           {"--pretty-print-inst"},
           "pretty print only instructions",
           0},
          {"pretty-print-inst-html",
           {"--pretty-print-inst-html"},
           "generate html file to store instructions",
           0},
          {"pretty-print-inst-raw",
           {"--pretty-print-inst-raw"},
           "pretty print instructions in raw hex",
           0},
          {"print-megablock-graph",
           {"--print-megablock-graph"},
           "megablock graph represents the nodes that actually execute on the "
           "FPGA, after fusing all pipelinable nodes. This option should be "
           "used with -c option",
           0},
          {"print-exec-graph",
           {"--print-exec-graph"},
           "exec graph contains all nodes (in topological order) with which "
           "instructions are ordered. This option should be used with -c "
           "option",
           0},
          {"output",
           {"--output", "-o"},
           "write output to file. Args: filename. For ex, -o model.gml",
           1},
          {"run",
           {"-r", "--run"},
           "run inference on model. Args: <gml_file>.",
           1},
          {"compile",
           {"-c", "--compile"},
           "Compile onnx model into gml file. Args: <onnx_model>",
           1},
          {"run_onnx",
           {"--run-onnx"},
           "onnx model thorough which model.gml was generated. TODO: remove "
           "this",
           1},
          {"dispatch",
           {"--dispatch"},
           "comma separated list of layers for which outputs are required. "
           "Args: "
           "[all | none | comma separated layer names]",
           1},
          {"dispatch_fn",
           {"--dispatch-fn"},
           "python function that'll be passed tensors returned by dipatchable "
           "nodes",
           1},
          {"receive-over-uart",
           {"--receive-over-uart"},
           "receive over /dev/ttyUSB0, via scripts/uart_server.py",
           1},
          {"accbuf-size",
           {"--accbuf-size"},
           "size of accumulant buffer for a single channel. Args: size in "
           "bytes "
           "(int)",
           1},
          {"dry-run",
           {"--dry-run"},
           "pretend to run by bypassing RAH (useful only for debugging)",
           0},
          {"fcbuf-size",
           {"--fcbuf-size"},
           "size of input buffer of the Vector Array that carries out the FC "
           "operation. "
           "Args: size in bytes (int)",
           1},
          {"remote",
           {"--remote"},
           "Send data to a remote vaaman running a gaticc-capable server. "
           "Args: <ip_address>",
           1},
          {"im2colbuf-size",
           {"--im2colbuf-size"},
           "size of input buffer of the im2col. "
           "Args: size in bytes (int)",
           1},
          {"version", {"--version"}, "print version info and exit", 0},
          {"summary", {"--summary"}, "print a summary of the model", 0},
      }};

  /*                                     description   command     */
  using SSVector = std::vector<std::pair<std::string, std::string>>;

  SSVector _usage_examples = {
      {"Run simulation given an onnx model and inputs. The simulation is run "
       "on the CPU. "
       "Useful to extract intermidiate outputs of layers",
       "gaticc -s path/to/model.onnx --loadpy <py_file> "
       "--preprocfn <preprocess_fn> --postprocfn <postprocess_fn> "
       "--venv-path ~/path/to/lib/python{version}/site-packages"},
      {"Run simulation but dispatch an intermidiate layer. Find layer "
       "names through model summary",
       "gaticc -s path/to/model.onnx --loadpy <py_file> "
       "--preprocfn <preprocess_fn> --postprocfn <postprocess_fn> "
       "--venv-path ~/path/to/lib/python{version}/site-packages "
       "--dispatch <layer1>,<layer2>,<layer3>"},
      {"Run simulation but dispatch all layers. Dispatch also stores "
       "an numpy pickled tensor for intermidiate outputs can be found "
       "in <layer_name>.tensor.npy",
       "gaticc -s path/to/model.onnx --loadpy <py_file> "
       "--preprocfn <preprocess_fn> --postprocfn <postprocess_fn> "
       "--venv-path ~/path/to/lib/python{version}/site-packages "
       "--dispatch all"},
      {"Get a summary of a onnx model",
       "gaticc -i path/to/model.onnx --summary"},
      {"Get a theoretical time estimate for a model",
       "gaticc -i path/to/model.onnx --timeest <frequency>  "
       "(e.g. --timeest 100)"},
      {"Compile a model into a gml file",
       "gaticc -c path/to/model.onnx --ramsize 512 --sa-arch 9,4,4 --vasize 32 "
       "--accbuf-size 4096 --fcbuf-size 32768 -o model.gml"},
      {"Pretty print generated instructions",
       "gaticc -c path/to/model.onnx --ramsize 512 --sa-arch 9,4,4 --vasize 32 "
       "--accbuf-size 4096 --fcbuf-size 32768 --pretty-print-inst"},
      {"Print the complete blob in hex",
       "gaticc -c path/to/model.onnx --ramsize 512 --sa-arch 9,4,4 --vasize 32 "
       "--accbuf-size 4096 --fcbuf-size 32768 --pretty-print-blob"},
      {"Run an inference",
       "gaticc -r model.gml --run-onnx model.onnx --loadpy "
       "<py_file> --preprocfn <preprocess_fn> "
       "--postprocfn <postprocess_fn> --venv-path "
       "~/path/to/lib/python{version}/site-packages "
       "--sa-arch <sa-arch> --ramsize <ramsize> --vasize "
       "<vasize> --accbuf-size <accbuf-size> --fcbuf-size {fcbuf-size}"},
      {"Run an inference but provide immediate image as input",
       "gaticc -r model.gml --run-onnx model.onnx --inputpath img.jpg --loadpy "
       "<py_file> --preprocfn <preprocess_fn>(img) "
       "--postprocfn <postprocess_fn> --venv-path "
       "~/path/to/lib/python{version}/site-packages "
       "--sa-arch <sa-arch> --ramsize <ramsize> --vasize "
       "<vasize> --accbuf-size <accbuf-size> --fcbuf-size <fcbuf-size>"},
      {"Run an inference but receive outputs over UART",
       "gaticc -r model.gml --run-onnx model.onnx --loadpy <py_file> "
       "--preprocfn <preprocess_fn> "
       "--postprocfn <postprocess_fn> --venv-path "
       "~/path/to/lib/python{version}/site-packages "
       "--sa-arch <sa-arch> --ramsize <ramsize> --vasize <vasize> "
       "--accbuf-size <accbuf-size> --fcbuf-size <fcbuf-size> "
       "--receive-over-uart"},
      {"Run an inference but dispatch intermidiate layers and receive over "
       "UART",
       "gaticc -r model.gml --run-onnx model.onnx --loadpy <py_file> "
       "--preprocfn <preprocess_fn> "
       "--postprocfn <postprocess_fn> --venv-path "
       "~/path/to/lib/python{version}/site-packages "
       "--sa-arch <sa-arch> --ramsize <ramsize> --vasize <vasize> "
       "--accbuf-size <accbuf-size> --fcbuf-size <fcbuf-size> "
       "--dispatch <layer1>,<layer2>,<layer3> --receive-over-uart"},
  };

  SSVector _concepts = {
      {"What does gaticc do and how to use it for inference",
       "Gaticc's main purpose is to do two things: compile and run. "
       "In compile, gaticc creates a .gml file from an .onnx file. The gml "
       "file contains re-ordered tensors and instructions. Gml can be "
       "generated "
       "by following the 'Compile model' usage example above. "
       "A gml file is used by gaticc to start a run, in which it takes inputs, "
       "sends it to the FPGA along with the model (gml), then receives the "
       "outputs. "
       "Run can be executed by following the 'Run an Inference' example above"},
      {"What are the --loadpy, --preprocfn, --postprocfn options",
       "Gaticc has an integrated python interpreter which allows calling into "
       "python functions "
       "as most people use python for ML work, they already have scripts to "
       "pre/post process their "
       "inputs/outputs, gaticc allows users to use the same scripts without "
       "re-writing them. "
       "--loadpy is used to provide the script and --pre/postproc are used to "
       "provide the "
       "functions to be called for pre/post processing. An example python file "
       "is provided in "
       "src/ml_inference.py and can be used"},
      {"What are the primary options",
       "Gaticc has primary options (-s, -i, -c, -r). Almost all other options "
       "are suboptions "
       "of these primary options. These are related to the four primary "
       "features of "
       "gaticc: simulation, info, compilation and run."},
      {"Where to get the models from?", "Refer to the gaticc project README"},
      {"What are --ramsize, --sa-arch --vasize?",
       "These are parameter based on which Gati is organized. These parameters "
       "affect address generation, alignment etc. They come directly from the "
       "architecture, thus, in order to answer why --accbuf-size is set to "
       "4096 "
       "in many examples, you need to understand the architecture or talk to "
       "architecture "
       "people"},
      {"How to view pretty-print.html?",
       "To view this HTML file, start a simple HTTP server in the file's "
       "directory on the remote server using the command: python3 -m "
       "http.server 8080 "
       "Then, open your browser and enter the URL "
       "http://<server-ip>:8080/pretty-print.html "
       "replacing <server-ip> with your machine's IP address. "
       "This will display the HTML content in your browser"},
      {"How to use --remote for running inference over LAN",
       "Make sure both the host and the device are on the same network. Find "
       "the "
       "ip of the device by running `ip a` on the device or by setting up mDNS "
       "resolution (vicharak.local) and 'ping'ing it. Next, start the server with "
       "`sudo python scripts/server.py` "
       "On the host, run the command as you would on the device, and append "
       "`--remote <ip>` along "
       "with the command"},
  };

public:
  void parse(int argc, char *argv[]);
  argagg::option_results &operator[](const std::string &name);
  bool has_option(const std::string &name) const;
  void print_usage() const;
  void print_version() const;
  void set_option(const char *opt_name, const char *val);
};

/* This is globally available for all functions. Alternatively,
 * an Argparse object could have been passed to each and every
 * contructor but this is the way I've decided to do it
 *
 * The Argparse::parse method is called on this object by main()
 * which in turn calls the underlying argagg functions.
 *
 * Functions looking to use args can simply call the subscript
 * operator[] on gbl_args.
 */
extern Argparse gbl_args;

/* returns true if -v or --vv are provided on the cli */
int get_verbose();
int get_verbose2();

/* has format specifier i.e. {} */
inline bool has_fs(const char *p) {
  if (*p == '\0' || *(p + 1) == '\0') {
    return false;
  }
  if (*p == '{' && *(p + 1) == '}') {
    return true;
  }
  return false;
}

inline void log(std::ostream &out, const char *p) {
  while (*p) {
    if (has_fs(p)) {
      std::cerr << "insufficient arguments for print";
      abort();
    }
    out << *p;
    ++p;
  }
}

template <typename T> void log(std::ostream &out, const char *p, T v) {
  while (*p) {
    const char s = *p;
    ++p;
    if (has_fs(p - 1)) {
      out << v;
      ++p;
      log(out, p);
      break;
    } else {
      out << s;
    }
  }
}

template <typename T, typename... Args>
void log(std::ostream &out, const char *p, T v, Args... args) {
  while (*p) {
    const char s = *p;
    p++;
    if (has_fs(p - 1)) {
      out << v;
      ++p;
      log(out, p, args...);
      break;
    } else {
      out << s;
    }
  }
}

template <typename T, typename... Args>
void log(std::ostream &out, const char *type, const char *p, T v, Args... args) {
  out << type << " ";
  log(out, p, v, args...);
}

template <typename T, typename... Args>
void log_info(const char *p, T v, Args... args) {
  if (get_verbose()) {
    std::stringstream out;
    log(out, "INFO:", p, v, args...);
    std::cout << out.str();
  }
}

template <typename T, typename... Args>
void log_info2(const char *p, T v, Args... args) {
  if (get_verbose2()) {
    std::stringstream out;
    log(out, "INFO:", p, v, args...);
    std::cout << out.str();
  }
}

inline void log_info(const char *p) { log_info("{}", p); }
inline void log_info2(const char *p) { log_info2("{}", p); }

template <typename T, typename... Args>
[[noreturn]] void log_fatal(const char *p, T v, Args... args) {
  std::stringstream out;
  log(out, "FATAL:", p, v, args...);
  throw std::runtime_error(out.str());
}

[[noreturn]] inline void log_fatal(const char *p) { log_fatal("{}", p); }

template <typename T, typename... Args>
void log_warn(const char *p, T v, Args... args) {
  std::stringstream out;
  log(out, "WARNING:", p, v, args...);
  std::cout << out.str();
}

inline void log_warn(const char *p) { log_warn("{}", p); }

void check_c_return_val(int val, const char *err);
void check_c_return_val(void *val, const char *err);

template <typename T>
void print_vec_vec(const char *s, std::vector<std::vector<T>> const &v) {
  std::cout << s << '\n';
  for (auto i : v) {
    for (auto j : i) {
      std::cout << j << '\t';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

/* TODO: use type_traits here */
/* Check if v belongs to the signed int family */
template <typename T> inline bool is_int_like(T v) {
  return typeid(v) == typeid(int) || typeid(v) == typeid(int8_t) ||
         typeid(v) == typeid(int16_t) || typeid(v) == typeid(int64_t) ||
         typeid(v) == typeid(long) || typeid(v) == typeid(long long);
}

template <typename T> inline bool is_unsigned_int_like(T v) {
  return typeid(v) == typeid(uint32_t) || typeid(v) == typeid(uint8_t) ||
         typeid(v) == typeid(uint16_t) || typeid(v) == typeid(uint64_t) ||
         typeid(v) == typeid(unsigned long) ||
         typeid(v) == typeid(unsigned long long);
}

template <typename T> inline bool is_float_like(T v) {
  return typeid(v) == typeid(float) || typeid(v) == typeid(double);
}

/* any container that overloads std::begin and std::end and operator<< on
 * its elements should be printable. the name has been kept for legacy
 * reasons, makes sense to use on linear containers.
 */
template <typename Container>
void print_vec(const char *s, Container const &v) {
  std::cout << s << "[";
  int newline_cnt = 0;
  std::cout << std::setprecision(8) << std::fixed;
  for (auto itr = std::begin(v); itr != std::end(v); ++itr) {
    /* print only 16 number on a single line */
    if (newline_cnt >= 16) {
      std::cout << '\n';
      newline_cnt = 0;
    }
    if (is_int_like<decltype(*itr)>(*itr) ||
        is_unsigned_int_like<decltype(*itr)>(*itr)) {
      std::cout << (int)*itr << ", " << ' ';
    } else {
      std::cout << *itr << ", " << ' ';
    }
    newline_cnt++;
  }
  std::cout << "]";
}

template <typename T = std::chrono::seconds> class Timer {
  using Tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
  Tp m_start;
  Tp m_stop;

public:
  void start() { m_start = std::chrono::high_resolution_clock::now(); }
  void stop() { m_stop = std::chrono::high_resolution_clock::now(); }

  T difference() { return std::chrono::duration_cast<T>(m_stop - m_start); }
  void report(std::string msg) {
    std::cout << msg << difference().count() << '\n';
  }
  // TODO: reset function
};

class MemProf {
  double m_start;
  double m_stop;
  double m_vm;

public:
  void process_mem_usage(double &vm_usage, double &resident_set) {
    vm_usage = 0.0;
    resident_set = 0.0;

    // the two fields we want
    unsigned long vsize;
    long rss;
    {
      std::string ignore;
      std::ifstream ifs("/proc/self/stat");
      ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
          ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
          ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
          ignore >> ignore >> vsize >> rss;
    }

    long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                        1024; // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024.0;
    resident_set = rss * page_size_kb;
  }

  void start() { this->process_mem_usage(m_vm, m_start); }

  void stop() { this->process_mem_usage(m_vm, m_stop); }

  /* Difference in KB */
  long difference() { return m_stop - m_start; }

  void report() {
    std::cout << "RSS: " << this->difference() << " KB, VM: " << m_vm
              << " KB\n";
  }
};

/* Parse a csv string of the form "9,8,8" and return a vector
 */
template <typename T> inline std::vector<T> parse_csv_string(std::string &s) {
  std::vector<T> result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    result.push_back(token);
  }
  return result;
}

template <> inline std::vector<int> parse_csv_string(std::string &s) {
  std::vector<int> result;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    result.push_back(std::stoi(token));
  }
  return result;
}

class TensorPool {
  std::vector<std::any> pool;

public:
  template <typename T> void set(int index, T data);
  template <typename T> T get(int index);
  template <typename T> void push_back(T data);
  void free();
  void free(int index);
  bool has_value(int index);
  void resize(int size);
  void print() const;
  std::vector<std::any>::iterator begin();
  std::vector<std::any>::iterator end();
};

template <typename T> void TensorPool::set(int index, T data) {
  pool.at(index) = data;
}

template <typename T> void TensorPool::push_back(T data) {
  pool.push_back(data);
}

template <typename T> T TensorPool::get(int index) {
  assert(pool.at(index).has_value() && "pool at index does not have a value");
  if (pool.at(index).type() != typeid(T)) {
    log_fatal("at index {}, expected type {} but got {}\n", index,
              pool.at(index).type().name(), typeid(T).name());
  }
  return std::any_cast<T>(pool.at(index));
}

/* like std::accumulate but calculates products
 * TODO: use this in tensor.h
 */
template <class InputIt, class T> T prod(InputIt first, InputIt last, T init) {
  T product = init;
  for (InputIt i = first; i != last; ++i) {
    product *= *i;
  }
  return product;
}

template <typename T> T prod(const std::vector<T>& v) {
  T product = static_cast<T>(1);
  for (const auto& i : v) {
    product *= i;
  }
  return product;
}

/* Add v1 and v2 and store into v1 */
template <typename T>
void add_vec(std::vector<T> &v1, const std::vector<T> &v2) {
  assert(v1.size() == v2.size());
  std::vector<T> ret(v1.size());
  for (int i = 0; i < v1.size(); ++i) {
    v1[i] = v1[i] + v2[i];
  }
}

/* path: such as "/usr/bin/file.txt"
 * returns: "file.txt"
 */
std::filesystem::path extract_basename(const std::string &path);
/* path: such as "/usr/bin/file.txt"
 * returns: "/usr/bin"
 */
std::filesystem::path extract_dirname(const std::string &path);

/* Container Concatenate */
template <typename Container>
Container concat(const Container &v1, const Container &v2) {
  Container ret;
  ret.insert(ret.begin(), v1.begin(), v1.end());
  ret.insert(ret.end(), v2.begin(), v2.end());
  return ret;
}

/* Element-wise multiplication */
template <typename T>
std::vector<T> operator*(const std::vector<T> &v1, const std::vector<T> &v2) {
  assert(v1.size() == v2.size());
  std::vector<T> ret(v1.size());
  for (int i = 0; i < v1.size(); ++i) {
    ret[i] = v1[i] * v2[i];
  }
  return ret;
}

/* https://stackoverflow.com/questions/53097952/how-to-understand-numpy-strides-for-layman */
template <typename Container>
inline Container get_stride_from_shape(const Container &shape) {
  Container ret(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    ret[i] = prod(std::begin(shape) + i + 1, std::end(shape), 1);
  }
  return ret;
}

template <typename Container>
inline Container get_stride_from_shape(const Container &&shape) {
  return get_stride_from_shape(shape);
}

bool is_broadcastable(const std::vector<int> &shape1,
                      const std::vector<int> &shape2);

std::vector<int> get_dims_after_pad(std::vector<int> current_dims,
                                    const std::vector<int> &pad);

/* return true if i,j lie in pad section of a 2d segment */
bool islying(int i, int j, int rows, int cols, const std::vector<int> &pad);

template <typename variantT, typename vectorT>
std::vector<vectorT> variant2vec(const std::vector<variantT> &var) {
  std::vector<vectorT> ret;
  for (variantT i : var) {
    if (std::holds_alternative<uint8_t>(i)) {
      ret.push_back((vectorT)std::get<uint8_t>(i));
    } else if (std::holds_alternative<int8_t>(i)) {
      ret.push_back((vectorT)std::get<int8_t>(i));
    } else {
      log_fatal("cant deduce type for zero point\n");
    }
  }
  return ret;
}

template <typename T>
std::vector<T> broadcast_vec(const std::vector<T> &in, int new_size) {
  if (in.size() == 1) {
    std::vector<T> ret(new_size);
    for (int i = 0; i < new_size; ++i) {
      ret[i] = in[0];
    }
    return ret;
  } else {
    assert(in.size() == new_size);
    return in;
  }
}
std::vector<float> compute_output_scale(const std::vector<float> &x_scale,
                                        const std::vector<float> &w_scale,
                                        const std::vector<float> &y_scale);

/* example:
 *  bitset_range_set(dest, src, 0, 3)
 * will set the first 4 least significant bits of dest by copying first four lsb
 * from src
 */
template <std::size_t b1N, std::size_t b2N>
void bitset_range_set(std::bitset<b1N> &dest, const std::bitset<b2N> &src,
                      int start, int stop) {
  ignore_unused(stop);
  assert(stop - start + 1 == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dest[start] = src[i];
    start++;
  }
}

template <std::size_t b1N, std::size_t b2N>
unsigned long bitset_range_get(const std::bitset<b2N> &src, int start,
                               int stop) {
  std::bitset<b1N> ret;
  for (int i = start, j = 0; i < stop + 1; ++i, ++j) {
    ret[j] = src[i];
  }
  return ret.to_ulong();
}

template <std::size_t b2N>
uint8_t bitset_byte_get(const std::bitset<b2N> &src, int n) {
  return static_cast<uint8_t>(
      bitset_range_get<8, b2N>(src, n * 8, (n * 8 + 8 - 1)));
}

template <typename T> void assert_all_equal(const T *arr, int size) {
  assert(size > 0);
  T a = arr[0];
  ignore_unused(a);
  for (int i = 0; i < size; ++i) {
    assert(arr[i] == a);
  }
}

int cmp_dims(const std::vector<int> &dim1, const std::vector<int> &dim2);

template <typename T> inline T ceil_mod(T i, int m) {
  return (T)(std::ceil((float)i / (float)m) * m);
}

template <typename T> inline T ceil_div(T i, T j) {
  return static_cast<T>(
      std::ceil(static_cast<float>(i) / static_cast<float>(j)));
}

int count_digits(int a);
void print_table(const std::map<std::string, int> &tbl);


std::vector<int> get_sa_arch();
int get_va_size();

template <typename T> uint32_t bytes2int(const T *data) {
  uint32_t value = 0;
  value |= (unsigned char)(data[0]);
  value <<= 8;
  value |= (unsigned char)(data[1]);
  value <<= 8;
  value |= (unsigned char)(data[2]);
  value <<= 8;
  value |= (unsigned char)(data[3]);
  return value;
}

inline int string_hash(const std::string &s) {
  return std::accumulate(s.begin(), s.end(), 0);
}

template <size_t sz, typename T>
std::bitset<sz> extract_bitset(const T *data, size_t size, int n, int m) {
  ignore_unused(size);
  assert(m - n == (sz / 8));
  assert(m - n < size);
  std::bitset<sz> ret{0};
  for (int i = n, j = ((sz / 8) - 1); i < m; ++i, --j) {
    std::bitset<sz> tmp{data[i]};
    tmp <<= (j * 8);
    ret |= tmp;
  }
  return ret;
}

std::pair<int, char **> argv_create(const std::vector<std::string> &opts);
void argv_delete(int argc, char **argv);

std::vector<int> reduced_shape(const std::vector<int> &dims, int reduction_axis,
                               int keepdims);
std::vector<int> unsqueeze_shape(const std::vector<int> &dims,
                                 const std::vector<int> &indices);
std::vector<int> concat_shape(const std::vector<std::vector<int>> &dims,
                              int axis);

template <int sz>
constexpr std::array<int8_t, sz / 8>
get_byte_vector(const std::bitset<sz> num) {
  static_assert(sz % 8 == 0, "Size must be a multiple of 8");
  std::array<int8_t, sz / 8> ret = {};
  for (size_t i = 0; i < ret.size(); ++i) {
    int8_t byte = 0;
    for (int bit = 0; bit < 8; ++bit) {
      if (num[i * 8 + bit]) {
        byte |= (1 << bit);
      }
    }
    ret[ret.size() - 1 - i] = byte;
  }
  return ret;
}

/* replace 'c' with 'r' */
std::string sed(const std::string& src, char c, char r);
py::list extract_pool(TensorPool &pool);
std::vector<int> permute(const std::vector<int> &v, std::vector<int> perm);
int dot(const std::vector<int>& a, const std::vector<int>& b);
