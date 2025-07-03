import onnx, onnxruntime as ort, numpy as np, argparse, sys
from onnx import helper, TensorProto, shape_inference
import gati

ONNX_DTYPE_MAP = {k: getattr(TensorProto, k) for k in dir(TensorProto) if not k.startswith("_")}
NUMPY_DTYPE_MAP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
    "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8, "tensor(bool)": np.bool_
}

def get_dtype_from_graph(graph, tensor_name, fallback_dtype=TensorProto.INT8):
    for vi in list(graph.output) + list(graph.value_info) + list(graph.input):
        if vi.name == tensor_name:
            return vi.type.tensor_type.elem_type
    for init in graph.initializer:
        if init.name == tensor_name:
            return init.data_type
    return fallback_dtype

def expose_node_outputs(graph, node_names):
    exposed_outputs = []
    if len(node_names) == 1 and node_names[0] == "all":
        target_nodes = graph.node
    else:
        target_nodes = []
        for name in node_names:
            matched = [n for n in graph.node if n.name == name]
            if not matched:
                sys.exit(f"Error: Node '{name}' not found.")
            target_nodes.extend(matched)
    for node in target_nodes:
        for out in node.output:
            if not any(go.name == out for go in graph.output):
                dtype = get_dtype_from_graph(graph, out)
                if dtype is None:
                    sys.exit(f"Error: Cannot infer dtype (or fallback type) for output '{out}' of node '{node.name}'.")
                graph.output.append(helper.make_tensor_value_info(out, dtype, None))
            exposed_outputs.append((node.name, out))
    return exposed_outputs

def load_samples(npy, shape, sample_idx):
    data = np.load(npy)
    if data.ndim == len(shape) - 1:
        if sample_idx not in [None, 0]: sys.exit(f"sample_index {sample_idx} out of bounds.")
        return [data]
    if data.ndim != len(shape) or data.shape[0] == 0:
        sys.exit(f"Invalid NPY shape {data.shape} vs input shape {shape}")
    if sample_idx is not None:
        if not (0 <= sample_idx < data.shape[0]): sys.exit(f"sample_index {sample_idx} out of range.")
        return [data[sample_idx]]
    return [data[i] for i in range(data.shape[0])]

def run_inference(model_bytes, input_data, input_name, out_n, dtype):
    sess = ort.InferenceSession(model_bytes)
    out_names = [i[1] for i in out_n]
    layer_names = [i[0] for i in out_n]
    for i, x in enumerate(input_data):
        x_b = np.expand_dims(x, 0).astype(dtype)
        try:
            outs = sess.run(out_names, {input_name: x_b})
        except Exception as e:
            print(f"Error on sample {i}: {e}", file=sys.stderr); continue
        #print(f"\n--- Sample {i} ---")
        #for name, out in zip(out_names, outs):
        #    print(f"Output '{name}' {out.shape}:")
        #    for ind, vv in enumerate(out.flatten()):
        #      if ind % 32 == 0 and ind != 0:
        #        print()
        #      print(f"{vv} ", end='')
        #    #print(f"{name} {out.shape}:\n", ' '.join(map(str, out.flatten()))
    return list(zip(layer_names, outs))

def run_gati_sim(onnx_path, in_data, layers):
  gati.set_dispatch(layers)
  return gati.sim(onnx_path, in_data)

def compare_layers(ort_ret, sim_ret, dump=[]):
    sim_dict = dict(sim_ret)
    ort_dict = dict(ort_ret)
    for l in dump:
      sim_arr = sim_dict[l]
      ort_arr = ort_dict[l]
      for i,j in zip(sim_arr.flatten(), ort_arr.flatten()):
        print(f"Sim: {i}, Ort: {j}")
    return [
        (name, 100 * (abs(ort_arr - sim_dict[name]) < 1e-4).sum() / ort_arr.size)
        for name, ort_arr in ort_ret
        if name in sim_dict and ort_arr.shape == sim_dict[name].shape
    ]

def main():
    # Usage examples:
    # python ort_sim_cmp.py <onnx> <npy> <layer_name>|"all" --sample_index <which image in npy>
    p = argparse.ArgumentParser(description="Expose ONNX node outputs via node names.")
    p.add_argument("original_onnx_path"), p.add_argument("npy_path"), p.add_argument("intermediate_names", nargs='+')
    p.add_argument("--dtype", default="FLOAT", type=str.upper, choices=list(ONNX_DTYPE_MAP))
    p.add_argument("--sample_index", type=int, default=None)
    args = p.parse_args()

    try: model = shape_inference.infer_shapes(onnx.load(args.original_onnx_path)); graph = model.graph
    except Exception as e: sys.exit(f"Load error: {e}")
    
    dtype = ONNX_DTYPE_MAP[args.dtype]
    output_tensor_names = expose_node_outputs(graph, args.intermediate_names)
    model_bytes = model.SerializeToString()

    try:
        sess = ort.InferenceSession(model_bytes)
        input_info = sess.get_inputs()[0]
        input_name, input_type, input_shape = input_info.name, input_info.type, input_info.shape
        dtype_np = NUMPY_DTYPE_MAP.get(input_type)
        if not input_shape: sys.exit("Model input shape undefined.")
        samples = load_samples(args.npy_path, input_shape, args.sample_index)
    except Exception as e: sys.exit(f"Inference prep error: {e}")

    run_inf_ret = run_inference(model_bytes, samples, input_name, output_tensor_names, dtype_np)
    run_sim_ret = run_gati_sim(args.original_onnx_path, np.expand_dims(samples[0], 0), args.intermediate_names)
    rr = compare_layers(run_inf_ret, run_sim_ret)
    for name, pct in rr:
      print(f"{pct:6.2f}% {name}")
if __name__ == "__main__":
    main()
