import gati
import numpy as np

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "/home/metal/dev/gaticc/tests/models/mnist_6_28_int8.onnx"
  ret = post(gati.sim(onnx_path, np.load("mnist_10.npy")))
  print(f"Match: {gati.match('mnist_10_labels.txt', ret)}%")
