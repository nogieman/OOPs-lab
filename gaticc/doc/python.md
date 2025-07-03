# Comparing Layers

```
import numpy as np
import gati

def gen():
  return np.expand_dims(np.load(f"/home/metal/dev/datasets/gati/{ds}.npy")[0], axis=0)

if __name__ == "__main__":
  onnx_path = "/home/metal/dev/datasets/gati/models/mnist_6_28_int8.onnx"
  gml_path = "model.gml"

  # Part 1: run simulation and store all tensors
  gati.set_dispatch(["all"])
  sim_ret = gati.sim(onnx_path, gen())

  # Part 2: run on the FPGA and compare the layers
  sim_ret = gati.sim_npy_load(["all"])
  gati.set_dispatch(["/conv1/Conv_quant"])
  gati.set_remote("v11.local")
  gati.flash("hex/gati_0.7.0_944_c4.hex")
  gati.compile(onnx_path,gml_path)
  ret = gati.run(onnx_path, gml_path, gen(), "verbose")
  gati.compare_layer(sim_ret, ret, [("_Relu_output_0_QuantizeLinear", "/conv1/Conv_quant")])
```

Make sure that you only run either part 1 first by commenting part 2, then run
part 2. Reason for this is, the hardware does not support dumping all layers
at the same time. `sim_npy_load` accepts a list of layer names and loads them
as a list of `(layer_name, layer_arr)` tuples. If `"all"` is the argument, all
layers with a `.tensor.npy` extension are loaded in. Later in the
`compare_layer` function, make sure you provide the right layer name, as its 
present in the array returned by `sim_npy_load`.
