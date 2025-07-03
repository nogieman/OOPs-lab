import gati
import numpy as np

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "imagenet_vgg_16_224_int8.onnx"
  bitstream = "rah.hex"
  gml_path = "model.gml"
  gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
  gati.compile(onnx_path, gml_path)
  gati.set_remote("192.168.10.69")
  gati.flash(bitstream)
  ret = post(gati.run(onnx_path, gml_path, np.load("imagenet_100.npy"),"verbose","verbose2","receive-over-uart"))
  print(f"Match: {gati.match('imagenet_100_labels.txt', ret)}%")