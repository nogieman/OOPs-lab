"""
This script modifies an ONNX model by exposing specified intermediate outputs.

Usage:
    python script_name.py <model_path> <intermediate_output_name1> [<intermediate_output_name2> ...]

Arguments:
    <model_path>                 : The path to the ONNX model file to be modified.
    <intermediate_output_name1>  : The name of the first intermediate output to expose.
    [<intermediate_output_name2> ...] : Additional intermediate output names (optional).

Functionality:
1. Loads the ONNX model from the specified path.
2. Searches for the provided intermediate output names in the model's graph:
   - If an output is found, it logs the node where it is located.
   - If not found, it exits with an error message.
3. Adds the specified intermediate outputs to the graph's output nodes, 
   if they are not already present.
4. Dynamically generates a modified model file name by appending "_modified" 
   to the input model file name and saves the updated model in the same directory.

Example:
    Input:
        Model path: /models/example.onnx
        Intermediate outputs: output1, output2

    Command:
        python script_name.py /models/example.onnx output1 output2

    Output:
        Modified model saved as: /models/example_modified.onnx

Notes:
- The script assumes the intermediate outputs are of type FLOAT. Adjust this if needed.
- Ensure the model path and intermediate output names are correct to avoid unexpected behavior.
"""

import onnx
import sys
import os

if len(sys.argv) < 3:
    print("Usage: python script_name.py <model_path> <intermediate_output_name1> [<intermediate_output_name2> ...]")
else:
    user_input = sys.argv[1]
    path = user_input
    intermediate_output_names = sys.argv[2:]  # Take the remaining arguments as intermediate output names

    model = onnx.load(path)
    graph = model.graph

    print(f"Model: {user_input}")
    print(f"Intermediate Outputs to Search: {intermediate_output_names}")
    

    # Check for the existence of all intermediate outputs
    for output_name in intermediate_output_names:
        found = False
        for node in graph.node:
            if output_name in node.output:
                print(f"Found intermediate output: {output_name} in node: {node.name}")
                found = True
                break
        if not found:
            print(f"Intermediate output not found: {output_name}")
            sys.exit(1)

    # Add intermediate outputs to the graph's outputs
    for output_name in intermediate_output_names:
        # Check if the output is already in the graph's outputs
        if output_name not in [output.name for output in graph.output]:
            print(f"Adding intermediate output to graph: {output_name}")
            # Create a new ValueInfoProto for the output
            new_output = onnx.helper.make_tensor_value_info(
                name=output_name,
                elem_type=onnx.TensorProto.FLOAT,  # Assuming the output is float (adjust if needed)
                shape=None,  # Shape can be left as None or inferred from the model
            )
            graph.output.append(new_output)

    # Dynamically set the modified ONNX path
    base, ext = os.path.splitext(path)
    modified_onnx_path = f"{base}_modified{ext}"

    # Save the modified model
    onnx.save(model, modified_onnx_path)
    print(f"Modified model saved to: {modified_onnx_path}")
