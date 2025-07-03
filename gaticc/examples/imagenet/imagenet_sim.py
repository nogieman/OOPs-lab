import numpy as np
import classes
import gati

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "tests/models/imagenet_vgg_16_224_int8.onnx"
  ret = post(gati.sim(onnx_path, np.load("imagenet_2.npy"), "verbose"))
  print(f"Match: {gati.match('imagenet_2_labels.txt', ret)}%")
