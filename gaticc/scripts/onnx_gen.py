import torch
import torch.nn as nn
import random
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import logging
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnx.checker



# model_description specifies the structure of the model as a list of layers. 
# Each layer is defined as a dictionary with its type (e.g., "Conv2d", "MaxPool2d", "Linear") 
# and required parameters like "filters" and "kernel_size" for Conv2d, or "kernel_size" for MaxPool2d.
# The last layer must be "Linear" to indicate the fully connected layer if needed.
# for eg. below model_description will create a model with 2 Conv2d layers followed by a MaxPool2d layer and a Linear layer.
# if model_description is None, a random model will be generated.
model_description = [
    {"type": "Conv2d", "filters": 64, "kernel_size": 3},
    {"type": "MaxPool2d", "kernel_size": 2},
    {"type": "Conv2d", "filters": 128, "kernel_size": 3},
    {"type": "MaxPool2d", "kernel_size": 2},
    {"type": "Linear"}
]
# input_dims specifies the input shape of the model as a tuple of 4 integers: (batch_size, channels, height, width)
input_dims = (1, 3, 224, 224) 


precision = "int8"


class RandomDataReader(CalibrationDataReader):
    def __init__(self, num_samples=10):
        self.data = iter([{"input": torch.randn(input_dims).numpy()} for _ in range(num_samples)])

    def get_next(self):
        return next(self.data, None)


def gen(num_layers, num_channels=input_dims[1], num_classes=random.randint(10,1000), description=None):
    layers = []
    feature_map_size = input_dims[2];  
    layer_count = 0
    has_linear = False
    
    if num_channels<1:
        logging.error("Number of channels must be greater than 0")
        return
    if description:
        for desc in description: 
            if desc["type"] == "Conv2d":
                num_filters = desc["filters"]
                kernel_size = desc["kernel_size"]
                print(f"Creating Conv2d: num_channels={num_channels}, num_filters={num_filters}")
                layers.append(nn.Conv2d(num_channels, num_filters, kernel_size, padding=1))
                layers.append(nn.ReLU(inplace=True))
                num_channels = num_filters 
                layer_count += 1
            elif desc["type"] == "MaxPool2d":
                kernel_size = desc["kernel_size"]
                layers.append(nn.MaxPool2d(kernel_size))
                feature_map_size = feature_map_size // kernel_size  
            elif desc["type"] == "Linear":
                layers.append(nn.Flatten())
                input_size = num_channels * feature_map_size * feature_map_size
                layers.append(nn.Linear(input_size, num_classes))
                has_linear = True
                layer_count += 1

    else:            
        for i in range(num_layers):
            a = random.randint(0, 100)
            num_filters = random.randint(32, 512)
            kernel_size = 3
            layers.append(nn.Conv2d(num_channels, num_filters, kernel_size, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layer_count += 1
            
            if feature_map_size > 1 and a % 2 == 0:
                layers.append(nn.MaxPool2d(2))
                feature_map_size = feature_map_size // 2  

            num_channels = num_filters
        
        layers.append(nn.Flatten())
        input_size = num_channels * feature_map_size * feature_map_size
        layers.append(nn.Linear(input_size, num_classes))
        layer_count += 1

    return nn.Sequential(*layers), layer_count, has_linear


def gen_onnx(num_models, nums_layers=(3, 100), description=None):
    onnx_models = []
    for i in range(num_models):
        num_layers = random.randint(nums_layers[0], nums_layers[1]) if description is None else None
        model, layers, linear = gen(num_layers=num_layers, description=description)
        dummy_input = torch.randn(input_dims)
        
        if linear:
            model_name = f"cfc_{layers}_{input_dims[3]}_{precision}.onnx"
        else:
            model_name = f"fcv_{layers}_{input_dims[3]}_{precision}.onnx"

        onnx_path = f"onnx/{model_name}"
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])
        dr = RandomDataReader()  
        # if needed to quantize in different precision, change the weight_type and activation_type.
        quantize_static(
            onnx_path, onnx_path, dr,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,  
            quant_format=QuantFormat.QOperator,
            extra_options={"ActivationSymmetric": True}
        )
        onnx_models.append(onnx_path)

    return onnx_models

class DefaultDataReader(CalibrationDataReader):
    def __init__(self, idims, num_samples=10):
        self.data = iter([{"input": torch.randn(idims).numpy()} for _ in range(num_samples)])

    def get_next(self):
        return next(self.data, None)

def foo():
    dummy_input = torch.randn((1, 144))
    model = nn.Sequential(*[nn.Linear(144, 10)])
    opath =  "fc_1_144_int8.onnx"
    torch.onnx.export(model, dummy_input, opath, opset_version=11, input_names=["input"], output_names=["output"])
    ddr = DefaultDataReader((1,144), num_samples=1)  
    # if needed to quantize in different precision, change the weight_type and activation_type.
    quantize_static(
        opath, opath, ddr,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,  
        quant_format=QuantFormat.QOperator,
        extra_options={"ActivationSymmetric": True}
    )

def infer_shape(dims, target):
    total = 1
    for d in dims: total *= d
    known = 1
    out = []
    infer = -1
    for i, d in enumerate(target):
        if d == -1:
            if infer != -1: raise ValueError("Multiple -1s")
            infer = i
            out.append(1)
        else:
            known *= d
            out.append(d)
    if infer != -1:
        if total % known: raise ValueError("Incompatible shape")
        out[infer] = total // known
    elif known != total:
        raise ValueError("Mismatched element count")
    return out

def stringize(li):
  s = ""
  for i in li:
    s += str(i) + "_"
  return s


def gen_transpose_reshape(input_dims, transpose_perm, reshape_shape):
    scale = helper.make_tensor('scale', TensorProto.FLOAT, [1], [0.1])
    zp = helper.make_tensor('zp', TensorProto.UINT8, [1], [0])
    shape = numpy_helper.from_array(np.array(reshape_shape, dtype=np.int64), name='shape_tensor')

    nodes = [
        helper.make_node('QuantizeLinear', ['input', 'scale', 'zp'], ['q'], name="q1"),
        helper.make_node('Transpose', ['q'], ['t'], perm=transpose_perm, name="t1"),
        helper.make_node('Reshape', ['t', 'shape_tensor'], ['r'], name="r1"),
        helper.make_node('DequantizeLinear', ['r', 'scale', 'zp'], ['output'], name="d1"),
    ]

    graph = helper.make_graph(
        nodes, 'tr_graph',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, input_dims)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, infer_shape(input_dims, reshape_shape))],
        [scale, zp, shape]
    )

    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    onnx.save(model, f'transpose_reshape_{stringize(input_dims)}.onnx')

#num_models = 1 # Number of models to generate
#onnx_models = gen_onnx(num_models, description=model_description if model_description else None)
#print(onnx_models)

gen_transpose_reshape([1,546,2,2], [0,2,3,1], [1,-1,91])
gen_transpose_reshape([1,546,3,3], [0,2,3,1], [1,-1,91])
gen_transpose_reshape([1,546,5,5], [0,2,3,1], [1,-1,91])
gen_transpose_reshape([1,546,10,10], [0,2,3,1], [1,-1,91])
gen_transpose_reshape([1,546,19,19], [0,2,3,1], [1,-1,91])
