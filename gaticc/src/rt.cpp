#include "rt.h"
#include "pch.h"
#include "executor.h"
#include "instructions.h"
#include "instgen.h"
#include "onnx_parser.h"
#include "tensor.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cerrno>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <optimization.h>

Fstream::Fstream(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  check_c_return_val(fp, filename.c_str());
  struct stat sbuf;
  int err = stat(filename.c_str(), &sbuf);
  check_c_return_val(err, filename.c_str());
  m_size = sbuf.st_size;
  m_buf = (char *)malloc(sizeof(*m_buf) * m_size);
  check_c_return_val(m_buf, "malloc");
  size_t size_read = fread(m_buf, sizeof(*m_buf), m_size, fp);
  if (size_read != m_size) {
    log_fatal("couldn't read all {} bytes, {} bytes read\n", m_size, size_read);
  }
  fclose(fp);
}

Fstream::~Fstream() { free(m_buf); }

const char *Fstream::get_data() const { return m_buf; }
size_t Fstream::get_size() const { return m_size; }

/* convert a 32 bit integer into a 48 bit byte stream */
std::vector<char> cvt_32248(int v) {
  std::vector<char> buf;
  buf.push_back(static_cast<char>(0x00));
  buf.push_back(static_cast<char>(0x00));
  buf.push_back(static_cast<char>((v & 0xFF000000) >> 24));
  buf.push_back(static_cast<char>((v & 0x00FF0000) >> 16));
  buf.push_back(static_cast<char>((v & 0x0000FF00) >> 8));
  buf.push_back(static_cast<char>((v & 0x000000FF)));
  return buf;
}

std::vector<char> get_meta_packet(const std::bitset<META_WIDTH_BITS> type,
                                  const std::vector<char> &data) {
  constexpr std::bitset<META_WIDTH_BITS> meta_sop_set{META_SOP};
  constexpr auto meta_sop_arr{get_byte_vector<META_WIDTH_BITS>(meta_sop_set)};

  const auto type_arr{get_byte_vector<META_WIDTH_BITS>(type)};

  std::vector<char> size_buf{cvt_32248(static_cast<int>(data.size()))};
  std::vector<char> packet;
  packet.insert(packet.end(), meta_sop_arr.begin(), meta_sop_arr.end());
  packet.insert(packet.end(), size_buf.begin(), size_buf.end());
  packet.insert(packet.end(), type_arr.begin(), type_arr.end());

  for (char i : data) {
    packet.push_back(i);
  }
  return packet;
}

RealRah::RealRah() {
  m_handle = dlopen(RAH_SO_STRING, RTLD_LAZY);
  if (m_handle == NULL) {
    log_fatal(
        "dlopen(): {}: could not open {}, check if you've installed rah. \n"
        "Additionally, check "
        "if vaaman-fpga communication overlay has been configured "
        "properly (see "
        "https://docs.vicharak.in/vicharak_sbcs/vaaman/vaaman-linux/"
        "linux-configuration-guide/vicharak-config-tool/) ",
        dlerror(), RAH_SO_STRING);
  }
}

RealRah::~RealRah() { dlclose(m_handle); }

int RealRah::write(const char *data, size_t size) {
  /* clear buffers before writing */
  typedef int (*clear_fn_t)(const uint8_t);
  clear_fn_t clear_fn = get_dlsym<clear_fn_t>(m_handle, "rah_clear_buffer");
  log_info("clear buffers before read\n");
  (*clear_fn)(RAH_APP_ID);

  typedef int (*write_fn_t)(const uint8_t, const char *, const unsigned long);
  write_fn_t write_fn = get_dlsym<write_fn_t>(m_handle, "rah_write");

  log_info("writing meta app, size {}\n", size);
  std::vector<char> size_buf = cvt_32248(size);
  write_meta(META_TYPE_PAYLOAD_SIZE, size_buf);

  log_info("writing via rah, size {}\n", size);
  /* send the actual data */
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  int r = (*write_fn)(RAH_APP_ID, data, size);
  tt.stop();
  log_info("Data write time: {}\n", tt.difference().count());
  if (r < static_cast<int>(size)) {
    log_fatal(
        "Failed to write all data to rah. Expected size: {}, Actual size: {}",
        size, r);
  }
  return r;
}

/*
 * Lowest level MetaApp write.
 * TODO: document the META protocol
 * 'size' here is the size of payload in bytes
 */
int RealRah::write_meta(const std::bitset<META_WIDTH_BITS> type,
                        const std::vector<char> &data) {

  std::vector<char> packet = get_meta_packet(type, data);

  typedef int (*clear_fn_t)(const uint8_t);
  clear_fn_t clear_fn = get_dlsym<clear_fn_t>(m_handle, "rah_clear_buffer");
  (*clear_fn)(META_APP_ID);

  typedef int (*write_fn_t)(const uint8_t, const char *, const unsigned long);
  write_fn_t write_fn = get_dlsym<write_fn_t>(m_handle, "rah_write");
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  int r = (*write_fn)(META_APP_ID, packet.data(), packet.size());
  tt.stop();
  log_info("Meta packet write time: {}\n", tt.difference().count());
  return r;
}

int RealRah::read(char *data, size_t size) {

  typedef int (*read_fn_t)(const uint8_t, const char *, const unsigned long);
  read_fn_t read_fn;
  read_fn = (read_fn_t)dlsym(m_handle, "rah_read");
  char *error = dlerror();
  if (error != NULL) {
    log_fatal("{}\n", error);
  }
  log_info("reading via rah, size {}\n", size);
  return (*read_fn)(RAH_APP_ID, data, size);
}

int FakeRah::write_meta(const std::bitset<META_WIDTH_BITS>,
                        const std::vector<char> &data) {
  return static_cast<int>(data.size());
}

int FakeRah::write(const char *, size_t size) { return size; }

int FakeRah::read(char *data, size_t size) {
  int m_ptr = 0;
  auto append_int = [&](uint32_t a) {
    /* reverse iteration for big endian */
    for (int i = sizeof(uint32_t) - 1; i >= 0; --i) {
      char c = get_byte(a, i);
      data[m_ptr++] = c;
    }
  };
  memset(data, 0, size);
  append_int(DWP_SOP);
  append_int((uint32_t)(size - (DWP_HEADER_BYTES * 2)));
  append_int((uint32_t)2108);

  int8_t c = 1;
  for (size_t i = m_ptr; i < size; ++i) {
    data[i] = c;
    c++;
  }
  return size;
}

void FakeRah::check_version() {}

AirRah::AirRah(const std::string &server_ip) {
  log_info("Resetting AirRah servers...\n");
  int reset_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (reset_sock == -1) {
    log_fatal("Reset socket creation failed: {}\n", strerror(errno));
  }
  sockaddr_in reset_addr{};
  reset_addr.sin_family = AF_INET;
  reset_addr.sin_port = htons(9090);
  if (inet_pton(AF_INET, server_ip.c_str(), &reset_addr.sin_addr) <= 0) {
    log_fatal("Invalid reset address: {}\n", server_ip);
  }
  if (connect(reset_sock, (struct sockaddr *)&reset_addr, sizeof(reset_addr)) <
      0) {
    log_fatal("Failed to connect to reset server at {}:9090\n", server_ip);
  }
  send(reset_sock, "reset", 5, 0);
  char ack_buf[4] = {};
  int ack_len = recv(reset_sock, ack_buf, sizeof(ack_buf) - 1, 0);
  if (ack_len <= 0 || std::string(ack_buf) != "OK") {
    log_fatal("No valid acknowledgment received from reset server\n");
  }
  close(reset_sock);
  log_info("Received reset acknowledgment from server\n");

  log_info("Reading/Writing over AirRah\n");
  const int port_no = 8080;
  const int max_retries = 10;
  const int retry_delay_ms = 200;

  for (int attempt = 1; attempt <= max_retries; ++attempt) {
    m_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (m_sock == -1)
      log_fatal("Socket creation failed: {}\n", strerror(errno));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_no);
    if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
      log_fatal("Invalid server address: {}\n", server_ip);
    }

    if (connect(m_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) ==
        0) {
      log_info("Connected to server at ip {}, port {}\n", server_ip, port_no);
      break;
    }

    close(m_sock);
    if (attempt == max_retries) {
      log_fatal("Failed to connect after {} attempts\n", max_retries);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
  }
  log_info("Connected to server at ip {}, port {}\n", server_ip, port_no);
}

void AirRah::serv_send(int app_id, const char *data, int size) {
  /* TODO: check for errors */
  uint32_t id = htonl(static_cast<uint32_t>(app_id));
  uint32_t length = htonl(static_cast<uint32_t>(size));
  send(m_sock, &id, sizeof(id), 0);
  send(m_sock, &length, sizeof(length), 0);
  send(m_sock, data, size, 0);
}

int AirRah::write(const char *data, size_t size) {
  log_info("writing meta app, size {}\n", size);
  std::vector<char> size_buf = cvt_32248(size);
  write_meta(META_TYPE_PAYLOAD_SIZE, size_buf);

  log_info("writing via rah, size {}\n", size);
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  serv_send(RAH_APP_ID, data, size);
  tt.stop();
  log_info("Data write time: {}\n", tt.difference().count());
  return size;
}

int AirRah::read(char *data, size_t size) {
  uint32_t sz = htonl(static_cast<uint32_t>(size));
  send(m_sock, &sz, sizeof(sz), 0);

  uint32_t server_length;
  int bytes_received =
      recv(m_sock, &server_length, sizeof(server_length), MSG_WAITALL);
  if (bytes_received <= 0) {
    log_fatal("Server disconnected\n");
  }
  server_length = ntohl(server_length);

  size_t total_received = 0;
  while (total_received < server_length) {
    bytes_received = recv(m_sock, data + total_received,
                          server_length - total_received, 0);
    if (bytes_received <= 0) {
      log_fatal("Server disconnected during message\n");
    }
    total_received += bytes_received;
  }
  return total_received;
}

void AirRah::check_version() {
}

int AirRah::write_meta(const std::bitset<META_WIDTH_BITS> type,
               const std::vector<char> &data) {
  std::vector<char> packet = get_meta_packet(type, data);
  Timer<std::chrono::milliseconds> tt;
  tt.start();
  serv_send(META_APP_ID, packet.data(), packet.size());
  tt.stop();
  log_info("Meta packet write time: {}\n", tt.difference().count());
  return 0;
}

AirRah::~AirRah() {
  close(m_sock);
}

void Runner::tensor_pool_init() {
  int total_regs = m_parser->get_total_registers() + 1;
  tensor_pool.resize(total_regs);
  tensor_pool.free();
}

std::string Runner::get_run_arg() {
  assert(gbl_args.has_option("run"));
  return gbl_args["run"].as<std::string>();
}

Runner::Runner() {}

TensorPool Runner::infer(const std::string& onnx_path, const std::string& gml_path, py::array arr) {
  Op::Parser parser(onnx_path);
  m_parser = &parser;
  split_large_kernel(m_parser->get_graph());
  Pass::absorb(m_parser->get_graph());
  tensor_pool_init();
  Fstream fp(gml_path);
  /* TODO: use uniqueptr */
  Rah *rah;
  if (gbl_args.has_option("dry-run")) {
    rah = new FakeRah();
  } else if (gbl_args.has_option("remote")) {
    std::string ip_addr = gbl_args["remote"].as<std::string>();
    rah = new AirRah(ip_addr);
  } else {
    rah = new RealRah();
  }
  load_model(*rah, fp);
  HashedDispatchTable hdt(fp);
  TPDT input_type = parser.get_model_input_type();
  TPDT output_type = parser.get_model_output_type();

  if (input_type == onnx::TensorProto_DataType_FLOAT &&
      output_type == onnx::TensorProto_DataType_FLOAT) {
    Tensor<float> *input = new TensorCreate<float>(arr);
    return infer_aux<float>(*rah, hdt, input);
  } else if (input_type == onnx::TensorProto_DataType_INT8 &&
             output_type == onnx::TensorProto_DataType_INT32) {
    Tensor<int8_t> *input = new TensorCreate<int8_t>(arr);
    return infer_aux<int8_t>(*rah, hdt, input);
  } else {
    log_fatal("Unsupported type combo: {}, {}\n",
              Op::get_tensorproto_dtype_name(input_type),
              Op::get_tensorproto_dtype_name(output_type));
    return TensorPool();
  }
}

/* make sure correct bitstream is loaded & rah.service
 * is running
 * TODO: implement this, will probably require bitman?
 */

void RealRah::check_version() {
  typedef int (*version_check_fn_t)();
  version_check_fn_t version_check_fn =
      get_dlsym<version_check_fn_t>(m_handle, "rah_check_version_compatible");
  if (!version_check_fn) {
    log_fatal("CPU Rah version is not compatible with FPGA CPU\n");
  }
}

void Runner::scan(Rah &rah) {
  log_info("scanning for rah services no cap fr\n");
  rah.check_version();
  log_info("Version check passed!\n");
  std::vector<char> empty;
  rah.write_meta(META_TYPE_RESET, empty);
}

/* Loads aligned and padded weights to the FPGA's DRAM */
void Runner::load_model(Rah &rah, const Fstream &fp) {
  scan(rah);
  const char *data = fp.get_data();
  size_t size = fp.get_size();

  constexpr std::bitset<META_WIDTH_BITS> d_uart{META_CONST_DISPATCH_UART};
  constexpr std::bitset<META_WIDTH_BITS> d_rah{META_CONST_DISPATCH_RAH};

  constexpr auto d_uart_arr{get_byte_vector<META_WIDTH_BITS>(d_uart)};
  constexpr auto d_rah_arr{get_byte_vector<META_WIDTH_BITS>(d_rah)};

  std::vector<char> d_uart_vec(d_uart_arr.begin(), d_uart_arr.end());
  std::vector<char> d_rah_vec(d_rah_arr.begin(), d_rah_arr.end());

  log_info("setting dispatch type\n");
  if (gbl_args.has_option("receive-over-uart")) {

    rah.write_meta(META_TYPE_DISPATCH, d_uart_vec);
  } else {
    rah.write_meta(META_TYPE_DISPATCH, d_rah_vec);
  }

  log_info("writing model weights to FPGA dram\n");
  rah.write(data, size);
  log_info("write model weights complete\n");
  /* TODO: no way to know if it went through
   * successfully to the fpga
   */
}

void Runner::fake_exec(Op::LayerBase *l) {
  if (tensor_pool.has_value(l->outputs.at(0))) {
    tensor_pool.free(l->outputs.at(0));
  }
}

void Runner::read_uart(BinBlob &blob, std::string handler_ip,
                       int expected_size) {
  log_info("[gaticc] Connecting to Handler...\n");

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    log_fatal("Socket creation failed: {}\n", strerror(errno));
    return;
  }

  sockaddr_in server_addr{};
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(HANDLER_PORT);
  inet_pton(AF_INET, handler_ip.c_str(), &server_addr.sin_addr);

  if (connect(sock, (sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
    log_fatal("Connection to Handler failed: {}\n", strerror(errno));
    close(sock);
    return;
  }

  std::string request = "{\"size\": " + std::to_string(expected_size) + "}\n";
  send(sock, request.c_str(), request.size(), 0);
  log_info("[gaticc] Sent request for {}  bytes\n", expected_size);

  std::vector<char> buffer(expected_size);
  int received_bytes = 0;

  while (received_bytes < expected_size) {
    int len = recv(sock, buffer.data() + received_bytes,
                   expected_size - received_bytes, 0);
    if (len > 0) {
        received_bytes += len;
    } else if (len == 0) {
        log_fatal("Connection closed by server (received {} / {})\n", received_bytes, expected_size);
        close(sock);
        return;
    } else {
        log_fatal("Socket recv() failed: {}\n", strerror(errno));
        close(sock);
        return;
    }
  }
  log_info("[gaticc] Received {} bytes", received_bytes);

  if (received_bytes != expected_size) {
    log_fatal("Incomplete data! Expected {}, received {}\n", expected_size,
              received_bytes);
    close(sock);
    return;
  }

  memcpy(blob.get_data(), buffer.data(), received_bytes);
  close(sock);
}

void Runner::receive_output(Rah &rah, const Op::LayerBase *l, bool is_last_layer) {
  int expected_hash = string_hash(l->name);
  uint32_t expected_data_size = 0;

  if (strcmp(l->op_type(), "QLinearConv") == 0) {
    expected_data_size = aligned_conv_output(l->pipelined_output_dims) *
                         Op::tpdt_sizeof(l->output_type[0]);
  } else if (strcmp(l->op_type(), "QLinearMatMul") == 0 ||
             strcmp(l->op_type(), "QGemm") == 0) {
    expected_data_size =
        aligned_fc_io(&l->output_dims[0]) * Op::tpdt_sizeof(l->output_type[0]);
  } else if (strcmp(l->op_type(), "Transpose") == 0) {
    expected_data_size = ceil_mod(prod(l->output_dims.at(0)) * Op::tpdt_sizeof(l->output_type.at(0)), WORD_SIZE);
  } else {
    log_fatal("Unhandled layer of type: {}\n", l->op_type());
  }
  auto expected_dims = l->aligned_output()[0];
  uint32_t expected_packet_size = io_tensor_packet_size(expected_data_size);

  log_info("expected packet size in receive output: {}\n",
           expected_packet_size);
  log_info("expected data size in receive output: {}\n", expected_data_size);

  BinBlob blob(expected_packet_size);

  Timer<std::chrono::milliseconds> tt;
  tt.start();
  if (gbl_args.has_option("receive-over-uart")) {
    if (gbl_args.has_option("remote")) {
      std::string handler_ip = gbl_args["remote"].as<std::string>();
      read_uart(blob, handler_ip, expected_packet_size);
    } else {
      log_fatal("UART receive requires --remote option!\nPlease run with remote server and make sure UART server (port 5001) is running.\n");
    }
  } else {
    rah.read(blob.get_data(), expected_packet_size);
  }
  tt.stop();
  log_info("Data read time: {}\n", tt.difference().count());

  const unsigned char *data = (const unsigned char *)blob.get_data();

  if (!gbl_args.has_option("dry-run")) {
    /* dry-run is a false traversal of the run loop used for debugging,
     * correctness is not really needed all that much
     */
    check_dwp_header(data, expected_packet_size, expected_data_size,
                     expected_hash);
  }

  // check_dwp_footer(data, expected_packet_size, 0 /* expected data size */, 0
  // /* expected hash */);
  if (l->output_type[0] == onnx::TensorProto_DataType_INT8) {
    const int8_t *real_data =
        reinterpret_cast<const int8_t *>(data + DWP_HEADER_BYTES);
    receive_output_aux<int8_t>(real_data, l, is_last_layer);
  } else if (l->output_type[0] == onnx::TensorProto_DataType_UINT8) {
    const uint8_t *real_data =
        reinterpret_cast<const uint8_t *>(data + DWP_HEADER_BYTES);
    receive_output_aux<uint8_t>(real_data, l, is_last_layer);
  } else {
    log_fatal("can't compute with tensor of type {}\n",
              Op::get_tensorproto_dtype_name(l->output_type[0]));
  }
}

HashedDispatchTable::HashedDispatchTable(const Fstream &fp) {
  const unsigned char *data = (const unsigned char *)fp.get_data();
  size_t size = fp.get_size();
  assert(size > DWP_HEADER_BYTES);
  uint32_t dwp_header = bytes2int(data);
  uint32_t ds = bytes2int(data + 4);
  ignore_unused(
      dwp_header); // in Release, when the following assert is unavailable
  assert(dwp_header == DWP_SOP);
  int total_instructions = (ds / (INST_SIZE_BITS / 8));
  /* i starts at 1 to skip the zeroth instruction */
  int inst_bytes = (INST_SIZE_BITS / 8);
  assert(size >= (DWP_HEADER_BYTES + (total_instructions * inst_bytes)));
  int ptr = DWP_HEADER_BYTES + inst_bytes;
  for (int i = 1; i < total_instructions; ++i) {
    std::bitset<INST_SIZE_BITS> inst =
        extract_bitset<INST_SIZE_BITS>(data, size, ptr, ptr + inst_bytes);
    int opcode = extract_opcode(inst);
    if (opcode == OP_OutputBlock) {
      int dispatch_en = bitset_range_get<OutputBlock_DispatchEn_COUNT>(
          inst, OutputBlock_DispatchEn_LOW, OutputBlock_DispatchEn_HIGH);
      if (dispatch_en) {
        int dispatch_id = bitset_range_get<OutputBlock_DispatchID_COUNT>(
            inst, OutputBlock_DispatchID_LOW, OutputBlock_DispatchID_HIGH);
        tbl.push_back(dispatch_id);
      }
    }
    ptr = ptr + inst_bytes;
  }
}

bool HashedDispatchTable::should_dispatch(const Op::LayerBase *l) const {
  int hashed = string_hash(l->name);
  auto itr = std::find(tbl.begin(), tbl.end(), hashed);
  if (itr != tbl.end()) {
    return true;
  }
  return false;
}

void Op::LayerBase::send_input(TensorPool &, AddressGen &, Rah &, IOAddrTbl &) const {
  log_fatal("Cannot send inputs for layer {}: send_input() override not implemented\n", this->name);
}

template <typename T>
static void sa_align_input(BinBlob &blob, const Op::Layer::QLinearConv *l, uint32_t data_size, uint32_t addr,
                              const Tensor<T> *tensor) {
  Timer<std::chrono::microseconds> tt;
  tt.start();
  blob.append_dwp_header(data_size, addr);
  assert(tensor->dims_size() == 4 && "Expected a 4 dimensional array (NCHW)");
  IVec2D og_dims_v {tensor->get_dims()};
  auto og_dims = og_dims_v.at(0);
  auto aligned_dims = aligned_conv_input_dims(og_dims_v, l->weights->dims())[0];
  auto sa_arch = get_sa_arch();
  int og_frame_sz = og_dims[TENSOR_4D_HEIGHT] * og_dims[TENSOR_4D_WIDTH];
  int frame_sz = aligned_dims[TENSOR_4D_HEIGHT] * aligned_dims[TENSOR_4D_WIDTH];
  int batch_size = aligned_dims[TENSOR_4D_CHANNELS] * frame_sz;
  int chan_dim = 0;
  if (is_pointwise_conv(l->weights->dims())) {
    chan_dim = sa_arch[SA_ARCH_ROW];
  } else {
    chan_dim = sa_arch[SA_ARCH_N];
  }

  int dk = WORD_SIZE / chan_dim;
  T zero = 0;

  for (int b = 0; b < aligned_dims[TENSOR_4D_BATCH]; ++b) {
    for (int c = 0; c < aligned_dims[TENSOR_4D_CHANNELS] / chan_dim; ++c) {
      for (int e = 0; e < ceil_mod(frame_sz, dk) / dk; ++e) {
        for (int ci = 0; ci < chan_dim; ++ci) {
          for (int ei = 0; ei < dk; ++ei) {
            int chan_n = (c * chan_dim) + ci;
            int elem_n = (e * dk) + ei;
            int index = (b * batch_size) + (chan_n * og_frame_sz) + elem_n;
            if (chan_n >= og_dims[TENSOR_4D_CHANNELS] || elem_n >= og_frame_sz) {
              blob.append(zero);
            } else {
              blob.append(tensor->at(index));
            }
          }
        }
      }
    }
  }
  tt.stop();
  log_info("SA Align time: {} us\n", tt.difference().count());
}

template <typename T>
static void sa_send_input(const Op::LayerBase *l, TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &tbl) {
  Tensor<T> *input_tensor = tensor_pool.get<Tensor<T> *>(l->inputs.at(0));
  auto ireg = tbl.at(l->name).first.at(0);
  uint32_t addr = generator.io_addr_from_register(ireg);
  log_info("sending input for register {}, addr is {}\n", ireg, addr);
  const Op::Layer::QLinearConv *cc = dynamic_cast<const Op::Layer::QLinearConv*>(l);
  auto dims = input_tensor->get_dims();
  IVec2D dims_wrapper = {dims};
  uint32_t og_aligned_size = aligned_conv_input(dims_wrapper, cc->weights->dims()) * sizeof(T);
  uint32_t total_size_with_packets = io_tensor_packet_size(og_aligned_size);
  BinBlob blob(total_size_with_packets);
  sa_align_input<T>(blob, cc, og_aligned_size, addr, input_tensor);
  blob.append_dwp_header(0, 0);
  if (get_verbose()) {
    blob.write("input_data.bin");
  }
  GmlCheck gmlcheck;
  gmlcheck.check_dwp(blob);
  log_info("Start writing images to FPGA\n");
  rah.write(blob.get_data(), blob.size());
  log_info("finish writing images to FPGA\n");
}

void Op::Layer::QLinearConv::send_input(TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &tbl) const {
  if (input_type.at(0) == onnx::TensorProto_DataType_INT8) {
    sa_send_input<int8_t>(this, tensor_pool, generator, rah, tbl);
  } else if (input_type.at(0) == onnx::TensorProto_DataType_UINT8) {
    sa_send_input<uint8_t>(this, tensor_pool, generator, rah, tbl);
  } else {
    log_fatal("QLinearConv::send_input() can't handle {} type", get_tensorproto_dtype_name(input_type.at(0)));
  }
}

/* align inputs to WORD_SIZE and push them to blob */
template <typename T>
static void word_align_input(BinBlob &blob, const Tensor<T> *tensor) {
  auto dims = tensor->get_dims();
  auto real_size = prod(dims);
  auto aligned_size = ceil_mod(real_size, WORD_SIZE);
  for (int i = 0; i < real_size; ++i) {
    blob.append(tensor->at(i));
  }
  T zero = 0;
  for (int i = 0; i < aligned_size-real_size; ++i) {
    blob.append(zero);
  }
}

template <typename T>
static void generic_send(const Op::LayerBase *l, TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &tbl) {
  Tensor<T> *input_tensor = tensor_pool.get<Tensor<T> *>(l->inputs.at(0));
  auto ireg = tbl.at(l->name).first.at(0);
  uint32_t addr = generator.io_addr_from_register(ireg);
  log_info("sending input for register {}, addr is {}\n", ireg, addr);
  auto dims = input_tensor->get_dims();
  uint32_t payload_size = ceil_mod(prod(dims), WORD_SIZE) * sizeof(T);
  uint32_t packet_size = io_tensor_packet_size(payload_size);
  BinBlob blob(packet_size);
  blob.append_dwp_header(payload_size, addr);
  word_align_input(blob, input_tensor);
  blob.append_dwp_header(0, 0);
  GmlCheck gmlcheck;
  gmlcheck.check_dwp(blob);
  log_info("Start writing images to FPGA\n");
  rah.write(blob.get_data(), blob.size());
  log_info("finish writing images to FPGA\n");
}

void Op::Layer::Transpose::send_input(TensorPool &tensor_pool, AddressGen &generator, Rah &rah, IOAddrTbl &io_tbl) const {
  if (input_type.at(0) == onnx::TensorProto_DataType_INT8) {
    generic_send<int8_t>(this, tensor_pool, generator, rah, io_tbl);
  } else if (input_type.at(0) == onnx::TensorProto_DataType_UINT8) {
    generic_send<uint8_t>(this, tensor_pool, generator, rah, io_tbl);
  } else {
    log_fatal("QLinearConv::send_input() can't handle {} type", get_tensorproto_dtype_name(input_type.at(0)));
  }
}
