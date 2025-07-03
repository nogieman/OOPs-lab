import numpy as np
import os
import gati

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  path = "/home/metal/dev/datasets/gati/"
  onnx_path = f"{path}/models/mnist_6_28_int8.onnx"
  bitstream = "../../hex/gati_0.7.0_944_c4.hex"
  gml_path = "model.gml"
  gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
  gati.compile(onnx_path, gml_path)
  gati.set_remote("v11.local")
  gati.flash(bitstream)
  ret = post(gati.run(onnx_path, gml_path, np.load(f"{path}/mnist_2.npy")))
  print(f"Match: {gati.match(f'{path}/mnist_2_labels.txt', ret)}%")
