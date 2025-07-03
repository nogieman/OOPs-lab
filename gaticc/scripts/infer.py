import argparse
import os
import numpy as np
import onnxruntime as ort

def match(label_file: str, prediction_file: str) -> float:
  with open(label_file, "r") as f:
      file_labels = [int(line.strip()) for line in f]
  with open(prediction_file, "r") as f:
      predicted_labels = [int(line.strip()) for line in f]
  if len(file_labels) != len(predicted_labels):
      raise ValueError("Label file and array must have the same number of elements.")
  mismatches = []
  matches = 0
  for idx, (file_label, pred_label) in enumerate(zip(file_labels, predicted_labels)):
      if file_label == pred_label:
          matches += 1
      else:
          mismatches.append(idx)
  match_percentage = (matches / len(file_labels)) * 100
  if mismatches:
      print(f"Mismatched indices: {mismatches}")
  return match_percentage

def run_inference(onnx_path, npy_path, labels_file, output_file="results.txt"):
    """Run ONNX model inference on NumPy input data and compare results.

    Args:
        onnx_path (str): Path to the ONNX model file.
        npy_path (str): Path to the NumPy (.npy) input data file.
        output_file (str): File to write inference results (default: 'results.txt').
        labels_file (str): File with ground truth labels for comparison (default: 'mnist_1000_labels.txt').

    Raises:
        FileNotFoundError: If ONNX or NPY file doesn't exist.
    """
    # Load input and model
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file '{onnx_path}' not found")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"NPY file '{npy_path}' not found")
    
    input_data = np.load(npy_path)
    session = ort.InferenceSession(onnx_path)

    # Get input name and run inference
    input_name = session.get_inputs()[0].name
    with open(output_file, "w"): pass  # Clear file
    for i in range(input_data.shape[0]):
        idata = np.expand_dims(input_data[i], axis=0)
        outputs = session.run(None, {input_name: idata})
        pred = np.argmax(outputs)
        print(pred)
        with open(output_file, "a") as f:
            f.write(f"{pred}\n")

    # Print match percentage
    match_pct = match(labels_file, output_file)
    print(f"Match: {match_pct}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX inference on NumPy data.")
    parser.add_argument("--onnx_path", help="Path to ONNX model file")
    parser.add_argument("--npy_path", help="Path to NumPy input data file")
    parser.add_argument("--output", default="results.txt", help="Output file for results (default: results.txt)")
    parser.add_argument("--labels", help="Labels file for comparison (default: mnist_1000_labels.txt)")
    args = parser.parse_args()

    run_inference(args.onnx_path, args.npy_path, args.labels, args.output)
