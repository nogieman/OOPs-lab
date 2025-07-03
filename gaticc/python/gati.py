import os
import shutil
import numpy as np
import socket
import struct
import _gati


# Using gati.py

# This Python module provides a set of utility functions to interact with the
# `gaticc` tool.

# The functions allow you to compile ONNX models, flash bitstreams, run
# inference, and evaluate results. Here's a list of functions of importance:
# 1. version
# 2. set_arch
# 3. get_arch
# 4. compile
# 5. run

# To understand their functionality, open examples/imagenet.py in a separate 
# window, and search up function names present in the imagenet.py in this 
# doc. Docstring based comments on each function should explain their purpose.

keep_quiet = False
gbl_arch = {"ramsize": 512, "sa-arch": "9,4,4", "vasize": 32, "accbuf-size": 4096, "fcbuf-size": 32768, "im2colbuf-size": 1024}
dispatch_arg = []
remote_ip = ""
remote_arg = []

def set_arch(ramsize=None, sa_arch=None, vasize=None, accbuf_size=None, fcbuf_size=None, im2colbuf_size=None, config=None):
  """
  Update the global architecture configuration. Parameters are optional and 
  will retain existing values if not specified.
  
  Args:
    ramsize: Size of RAM (optional)
    sa_arch: System architecture specification (optional)
    vasize: Virtual address space size (optional)
    accbuf_size: Accumulator buffer size (optional)
    fcbuf_size: Function call buffer size (optional)
    config: Optional dictionary containing configuration parameters
  
  Raises:
    TypeError: If config is provided but not a dictionary
  """
  global gbl_arch
  
  if 'gbl_arch' not in globals():
      gbl_arch = {}
  
  if config is not None:
      if not isinstance(config, dict):
          raise TypeError("config must be a dictionary")
      gbl_arch.update(config)
  else:
      updates = {}
      if ramsize is not None:
          updates["ramsize"] = ramsize
      if sa_arch is not None:
          updates["sa-arch"] = sa_arch
      if vasize is not None:
          updates["vasize"] = vasize
      if accbuf_size is not None:
          updates["accbuf-size"] = accbuf_size
      if fcbuf_size is not None:
          updates["fcbuf-size"] = fcbuf_size
      if im2colbuf_size is not None:
          updates["im2colbuf-size"] = im2colbuf_size
      if updates:
          gbl_arch.update(updates)

def set_dispatch(dispatch_list):
  """Sets a global dispatch comparison argument based on a provided dispatch list.
  Args:
    dispatch_list: A list of layer names
  Raises:
    ValueError: If `dispatch_list` is not a list, is empty, or contains unsupported types.
  Examples:
    >>> set_dispatch(["layer1", "layer2"])
  """
  global dispatch_arg
  if not isinstance(dispatch_list, list) or len(dispatch_list) < 1:
      raise ValueError(f"provided dispatch list {dispatch_list} should be a list with size greater than 0")
  dispatch_arg = []
  dispatch_arg = [("dispatch", ",".join(map(str, dispatch_list)))]

def set_keep_quiet(val=True):
  global keep_quiet
  keep_quiet = val

def set_remote(ip):
  global remote_ip
  global remote_arg
  remote_ip = os.popen(f"ping -c 1 {ip}").read().split('(')[1].split(')')[0] if "local" in ip else ip
  remote_arg = [("remote", str(remote_ip))]

def get_arch(): return gbl_arch
def get_arch_list(arch) -> list[tuple]: return [(str(i), str(arch[i])) for i in arch]
def kwargs2list(**kwargs) -> list[tuple]: return [(str(i).replace('_','-'), str(kwargs[i])) for i in kwargs]
def args2list(*args) -> list[tuple]: return [(str(i), "") for i in args]

remove_dupes = lambda pairs: {next(d for d in pairs if d[0] == k) for k in dict(pairs)}

def version():
    """
    Prints version info on stdout
    """
    _gati.version()
    return 0

def compile(
        onnx_path: str,
        out_path: str,
        *args,
        **kwargs,
        ):
    """
    Compile an ONNX model for the target hardware architecture.

    Args:
        onnx_path (str): Path to the input ONNX model file.
        out_path (str): Path where the compiled model will be saved.
        *args: Additional command-line flags to pass to the gaticc compiler.
        ramsize (int, optional): Size of the RAM in MB. Defaults to 512.
        sa_arch (str, optional): Systolic array architecture (e.g., "9,4,4"). Defaults to "9,4,4".
        vasize (int, optional): Vector ALU size. Defaults to 32.
        accbuf_size (int, optional): Accumulation buffer size in bytes. Defaults to 4096.
        fcbuf_size (int, optional): Fully-connected buffer size in bytes. Defaults to 32768.

    Prints:
        A message showing the architecture configuration being used.

    Raises:
        OSError: If the PYTHONPATH environment variable is not set.
    """
    rest = remove_dupes(kwargs2list(**kwargs) + args2list(*args) +
                        get_arch_list(get_arch()) + dispatch_arg)
    if not keep_quiet:
        print(f"GATICC COMPILE: Using arch: {rest}")
    _gati.compile(onnx_path, out_path, rest)
    return 0 # FIXME: need to return the exit status of the compile

def _flash_remote(ip, bitstream_file):
    PORT_BITSTREAM = 8081
    BUFFER_SIZE = 3 * 1024 * 1024
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, PORT_BITSTREAM))
    with open(bitstream_file, "rb") as f: data = f.read()
    s.send(struct.pack('>I', len(data))); sent = 0
    while sent < len(data): sent += s.send(data[sent:sent + BUFFER_SIZE])
    length = struct.unpack('>I', s.recv(4))[0]; print(f"Bitstream ack: {s.recv(length).decode()}")
    s.close()

def flash(
        bitstream_path: str
        ):
    if remote_ip != "":
        _flash_remote(remote_ip, bitstream_path)
    else:
        if shutil.which("bitman"):
            return os.system(f"sudo bitman -f {bitstream_path}")
        elif shutil.which("vaaman-ctl"):
            return os.system(f"sudo vaaman-ctl -i {bitstream_path}")
        else:
            OSError("Could not find any program to flash bitstream")

def run(
        onnx_path: str,
        gml_path: str,
        arr: np.ndarray,
        *args,
        **kwargs,
        ):
    """
    Run a compiled model on the target hardware.

    Args:
        onnx_path (str): Path to the original ONNX model file.
        gml_path (str): Path to the compiled model file (e.g., GML format).
        arr (np.ndarray): Input to the model.
        *args: Additional command-line flags to pass to the gaticc runtime.

    Prints:
        A message showing the architecture configuration being used.
    """
    rest = remove_dupes(kwargs2list(**kwargs) + args2list(*args) +
                        get_arch_list(get_arch()) + dispatch_arg + remote_arg)
    if not keep_quiet:
        print(f"GATICC RUN: Using arch: {rest}")
    return _gati.run(onnx_path, gml_path, arr, rest)

def summary(onnx_path: str):
  return _gati.info(onnx_path, [("summary", "")])

def match(label_file: str, predicted_labels: list) -> float:
  """
  Compare predicted labels against ground truth labels and calculate the match percentage.
  Args:
    label_file (str): Path to a file containing ground truth labels (one integer per line).
    prediction_file (str): Path to a file containing predicted labels (one integer per line).

  Returns:
    float: The percentage of matching labels (0.0 to 100.0).

  Prints:
    A list of indices where mismatches occurred, if any.

  Raises:
    ValueError: If the number of labels in the two files does not match.
    FileNotFoundError: If either file cannot be opened.
  """
  with open(label_file, "r") as f:
    file_labels = [int(line.strip()) for line in f]
  if len(file_labels) != len(predicted_labels):
    raise ValueError("Label file and array must have the same number of elements.")
  mismatches = []
  matches = 0
  for idx, (file_label, pred_label) in enumerate(zip(file_labels, predicted_labels)):
    if file_label == pred_label:
        matches += 1
    else:
        mismatches.append(idx)
  match_percentage = (matches / len(file_labels)) * 100
  if mismatches:
    print(f"Mismatched indices: {mismatches}")
  return match_percentage


def sim(
    onnx_path: str,
    arr: np.ndarray,
    *args,
    **kwargs,
    ) -> np.ndarray:
  """
  Run the ONNX file entirely on the CPU (simulation)

  Args:
    onnx_path (str): Path to the original ONNX model file.
    arr (np.ndarray): Input to the model.
    *args: Additional flags.
    **kwargs: Additional key-value flags.

  Returns:
    arr (np.ndarray): an array of the shape [N, ...] where is the
    batch size of the input
  """
  rest = remove_dupes(args2list(*args) + kwargs2list(**kwargs) + dispatch_arg)
  return _gati.sim(onnx_path, arr, rest)

def sim_npy_load(layer_names: list[str]) -> list[tuple[str, np.ndarray]]:
  """ load and return an npy file as a list[tuple[str, str]] """
  if len(layer_names) == 1 and layer_names[0] == "all":
    return [("".join(file_name.split('.')[:-2]), np.load(file_name)) for file_name in os.listdir('.') if ".tensor.npy" in file_name]
  else:
    return [(layer_name, np.load(layer_name.replace('/','_') + ".tensor.npy")) for layer_name in layer_names]

def compare_layer(sim_arr: list[tuple[str, np.ndarray]], run_arr: list[tuple[str, np.ndarray]], layer_names: list[tuple[str, str]]):
  def matcher(a1, a2):
    match_p = 0
    for index, (i, j) in enumerate(zip(a1.flatten(), a2.flatten())):
      print(f"Index: {index} Sim: {i}, Run: {j}")
      if i == j:
        match_p += 1
    return (match_p / len(a1.flatten())) * 100
  sim_d = dict(sim_arr)
  run_d = dict(run_arr)
  if not isinstance(layer_names, list) or not isinstance(layer_names[0], tuple): raise ValueError("layer_names must be a list of tuples")
  for sa, ra in layer_names:
    print(f"Matching {sa} with {ra}")
    ret = matcher(sim_d[sa], run_d[ra])
    print(f"Match Percent: {ret}")
