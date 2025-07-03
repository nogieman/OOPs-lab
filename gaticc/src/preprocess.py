#!/usr/bin/env python3

# How to write a python function that can be called by gaticc
# 1. define a python function here as you normally would
# 2. make sure that the return value of the function that will
#    be exported in one of builtin types (list, tuple, int, etc.)
# 3. if, say, the return value is a numpy array, convert it to
#    a list explicitly (with the tolist() function, for example)
# 4. define a similarly named function by following the guide
#    in ffi.cpp 

#from keras.utils import load_img, img_to_array
#from keras.applications.vgg16 import preprocess_input

import numpy as np

import onnx

def quantize_f32i8(array):
    s = 255 / (np.max(array) - np.min(array))
    f = lambda v: np.clip(np.round(s*v), -128, 127)
    quantized_array = f(array)
    return quantized_array.astype(np.int8)

# See https://github.com/tensorflow/tensorflow/issues/24976
# in order to use tensorflow
#def preprocess(image):
#    image = load_img(image, target_size=(224, 224))
#    image = img_to_array(image)
#    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#    image = preprocess_input(image)
#    return image

def preprocess(image):
    rng = np.random.default_rng()
    return rng.integers(0, 255, 224*224*3)

def read_img(s):
    a = preprocess(s)
    q = quantize_f32i8(a)
    return q.flatten().tolist()

model = onnx.load("/home/metal/dev/vaaman-cnn/onnx/vgg16/vgg16-12-int8.onnx")
graph_def = model.graph
initializers = graph_def.initializer

# layer: layer number
# n: nth kernel
# c: cth channel in that kernel
def fetch_kernel(layer, n, c):
    lname = 'vgg0_conv{}_weight_quantized'.format(layer)
    for i in initializers:
        if ((i.name) == lname):
            a = np.frombuffer(i.raw_data, dtype=np.uint8).reshape(i.dims)
            return (a[n,c,:,:]).flatten().tolist()
