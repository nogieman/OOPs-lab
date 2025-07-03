import numpy as np
import classes
import os
import gati

#from PIL import Image
#def preprocess(image):
#  if not os.path.exists(image):
#    raise OSError("File not found: {}".format(image))
#  img = Image.open(image)
#  # resize to (256,256)
#  img = img.resize((256,256))
#  img = np.array(img.convert('RGB'))
#  # scale b/w 0 and 1
#  img = img / 255.
#  # take a (224,224) center crop of the image
#  h, w = img.shape[0], img.shape[1]
#  y0 = (h - 224) // 2
#  x0 = (w - 224) // 2
#  img = img[y0 : y0+224, x0 : x0+224, :]
#  # Normalize (values obtained from millions of imagenet images)
#  img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#  img = np.transpose(img, axes=[2, 0, 1])
#  img = img.astype(np.float32)
#  img = np.expand_dims(img, axis=0)
#  return img

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__ == "__main__":
  onnx_path = "../../tests/models/imagenet_vgg_16_224_int8.onnx"
  bitstream = "../../hex/gati_0.7.0_944_c4.hex"
  gml_path = "model.gml"
  gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
  gati.compile(onnx_path, gml_path)
  #gati.set_remote("v11.local")
  gati.flash(bitstream)
  ret = post(gati.run(onnx_path, gml_path, np.load("imagenet_2.npy")))
  print(f"Match: {gati.match('imagenet_2_labels.txt', ret)}%")
