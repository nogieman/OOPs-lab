import os
import sys
import argparse
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from math import sqrt
import torch


# Utility functions
def cxcy_to_xy(cxcy):
    return np.concatenate([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], axis=1)

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    return np.concatenate([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2], np.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], axis=1)

def create_prior_boxes():
    fmap_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
    obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
    aspect_ratios = {'conv4_3': [1., 2., 0.5], 'conv7': [1., 2., 3., 0.5, .333], 
                     'conv8_2': [1., 2., 3., 0.5, .333], 'conv9_2': [1., 2., 3., 0.5, .333], 
                     'conv10_2': [1., 2., 0.5], 'conv11_2': [1., 2., 0.5]}
    fmaps = list(fmap_dims.keys())
    prior_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx, cy = (j + 0.5) / fmap_dims[fmap], (i + 0.5) / fmap_dims[fmap]
                for ratio in aspect_ratios[fmap]:
                    prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes).to("cpu")
    prior_boxes.clamp_(0, 1)
    return prior_boxes

def iou(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1_p, y1_p, x2_p, y2_p = box2[:4]
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image, axis=0)

def add_intermediate_outputs(model_path, intermediate_output_names):
    model = onnx.load(model_path)
    graph = model.graph
    print(f"Model: {model_path}")
    print(f"Intermediate Outputs to Search: {intermediate_output_names}")

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

    for output_name in intermediate_output_names:
        if output_name not in [output.name for output in graph.output]:
            print(f"Adding intermediate output to graph: {output_name}")
            new_output = onnx.helper.make_tensor_value_info(
                name=output_name,
                elem_type=onnx.TensorProto.FLOAT,
                shape=None,
            )
            graph.output.append(new_output)

    base, ext = os.path.splitext(model_path)
    modified_onnx_path = f"{base}_modified{ext}"
    onnx.save(model, modified_onnx_path)
    print(f"Modified model saved to: {modified_onnx_path}")
    return modified_onnx_path

def run_inference(image_path, onnx_model_path):
    ort.set_default_logger_severity(4)
    input_image = preprocess_image(image_path)
    iou_threshold = np.array([0.45], dtype=np.float32)
    session = ort.InferenceSession(onnx_model_path)
    input_names = [input.name for input in session.get_inputs()]
    inputs = {
        input_names[0]: input_image.astype(np.float32),
        input_names[1]: iou_threshold,
    }
    outputs = session.run(None, inputs)
    intermediate_outputs = {}
    for i, output in enumerate(session.get_outputs()):
        intermediate_outputs[output.name] = outputs[i]
    return intermediate_outputs

def inference_1000_images(onnx_model_path, dataset_dir):
    priors_cxcy = create_prior_boxes().numpy()
    image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()
    assert len(image_files) == 1000, "There should be exactly 1000 images in the dataset directory."

    all_boxes = []
    all_class_scores = []

    for i, image_path in enumerate(image_files):
        print(f"Processing image {i + 1}/{len(image_files)}: {image_path}")
        intermediate_outputs = run_inference(image_path, onnx_model_path)
        locs = intermediate_outputs["/pred_convs/Concat_12_output_0"][0]
        boxes_cxcy = gcxgcy_to_cxcy(locs, priors_cxcy)
        boxes_xy = cxcy_to_xy(boxes_cxcy)
        class_scores = intermediate_outputs["/pred_convs/Concat_13_output_0"][0]
        class_scores_softmax = softmax(class_scores, axis=1)
        all_boxes.append(boxes_xy)
        all_class_scores.append(class_scores_softmax)

    all_boxes = np.stack(all_boxes, axis=0)
    all_class_scores = np.stack(all_class_scores, axis=0)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    np.save(f"{output_dir}/{model_name}_bboxes.npy", all_boxes)
    np.save(f"{output_dir}/{model_name}_class_scores.npy", all_class_scores)
    print(f"Inference completed! Outputs saved to {output_dir}.")

def inference_single_nms(model_path, class_scores_path, bboxes_path):
    session = ort.InferenceSession(model_path)
    class_scores_all = np.load(class_scores_path).astype(np.float32)
    decoded_boxes_all = np.load(bboxes_path).astype(np.float32)
    nms_results = []

    for image_idx in range(class_scores_all.shape[0]):
        print(f"Processing image {image_idx + 1}/{class_scores_all.shape[0]}")
        scores = class_scores_all[image_idx]
        boxes = decoded_boxes_all[image_idx]
        num_classes = scores.shape[1]
        image_detections = []

        for class_id in range(1, num_classes):
            class_scores = scores[:, class_id]
            class_boxes = boxes
            class_boxes = np.expand_dims(class_boxes, axis=0)
            class_scores = np.expand_dims(class_scores, axis=0)
            class_scores = np.expand_dims(class_scores, axis=0)

            ort_inputs = {
                "boxes": class_boxes,
                "scores": class_scores
            }
            ort_outs = session.run(None, ort_inputs)
            nms_output = ort_outs[0]

            for detection in nms_output:
                batch_idx, class_idx, box_idx = detection
                box = class_boxes[batch_idx, box_idx]
                score = class_scores[batch_idx, class_idx, box_idx]
                image_detections.append([*box, class_id, score])

        image_detections = np.array(image_detections) if image_detections else np.empty((0, 6))
        nms_results.append(image_detections)

    nms_results = np.array(nms_results, dtype=object)
    np.save("outputs/nms_results.npy", nms_results)
    print("NMS results saved.")

def draw_bboxes(image_dir, nms_results_path, index):
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    results = np.load(nms_results_path, allow_pickle=True)
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if index >= len(image_files):
        print(f"Error: The specified index {index} is out of range. There are only {len(image_files)} images in the directory.")
        return

    image_path = image_files[index - 1]
    image = cv2.imread(image_path)
    original_image = image
    image_results = results[index - 1]

    if image_results.size == 0:
        print(f"No detections found for the image at index {index}.")
        return

    bboxes = image_results[:, :4]
    labels = image_results[:, 4]
    scores = image_results[:, 5]

    for bbox, label, score in zip(bboxes, labels, scores):
        xmin, ymin, xmax, ymax = bbox
        xmin = int(xmin * original_image.shape[1])
        xmax = int(xmax * original_image.shape[1])
        ymin = int(ymin * original_image.shape[0])
        ymax = int(ymax * original_image.shape[0])
        class_name = class_names[int(label)]
        color = (0, 255, 0)
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(original_image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = os.path.join(f"detected_image_{index}.jpg")
    cv2.imwrite(output_path, original_image)
    print(f"Detection results saved to {output_path}")

def compare_npy(file1, file2, iou_threshold=0.9):
    data1 = np.load(file1, allow_pickle=True)
    data2 = np.load(file2, allow_pickle=True)
    
    if len(data1) != len(data2):
        print("Mismatch in number of images!")
        return False
    
    success = True
    
    for img_idx, (image_boxes1, image_boxes2) in enumerate(zip(data1, data2)):
        if len(image_boxes1) != len(image_boxes2):
            print(f"Mismatch in number of detections for image {img_idx}!")
            success = False
            continue
        
        for i, (box1, box2) in enumerate(zip(image_boxes1, image_boxes2)):
            bbox1, class1 = box1[:4], box1[4]
            bbox2, class2 = box2[:4], box2[4]
            
            if class1 != class2:
                print(f"Class mismatch at image {img_idx}, detection {i}!")
                success = False
                continue
            
            if iou(bbox1, bbox2) < iou_threshold:
                print(f"Bounding boxes too different at image {img_idx}, detection {i}!")
                success = False
    
    if success:
        print("Yahoo! Your comparison ended successfully 🎉 ")


def main():
    parser = argparse.ArgumentParser(description="SSD300 Object Detection Pipeline")
    parser.add_argument("--run_pipeline", action="store_true", help="Run the full pipeline")
    parser.add_argument("--run_inference", action="store_true", help="Run only the inference step")
    parser.add_argument("--compare", nargs=2, metavar=("generated", "golden"), help="Compare two .npy files")
    parser.add_argument("--run_inference_and_compare", action="store_true", help="Run inference and then compare")
    parser.add_argument("--golden_values", metavar="golden_values", help="Path to golden values for comparison")
    parser.add_argument("--draw_bboxes", metavar="bbox_file", help="Draw bounding boxes from given .npy file")
    parser.add_argument("--image_index", metavar="image_index", help="Index of Image which you want to draw Bbox")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    if args.compare:
        compare_npy(args.compare[0], args.compare[1])
        sys.exit(0)
    
    onnx_model_path = "ssd_vgg_35_300_uint8.onnx"
    intermediate_outputs = ["/pred_convs/Concat_12_output_0", "/pred_convs/Concat_13_output_0"]
    modified_onnx_path = add_intermediate_outputs(onnx_model_path, intermediate_outputs)
    
    if args.run_pipeline:
        dataset_dir = "dataset"
        inference_1000_images(modified_onnx_path, dataset_dir)
    
    if args.run_pipeline or args.run_inference:
        class_scores_path = "outputs/ssd_vgg_35_300_uint8_modified_class_scores.npy"
        bboxes_path = "outputs/ssd_vgg_35_300_uint8_modified_bboxes.npy"
        nms_model_path = "nms_operator.onnx"
        inference_single_nms(nms_model_path, class_scores_path, bboxes_path)

    if args.run_inference_and_compare and args.golden_values:
        class_scores_path = "outputs/ssd_vgg_35_300_uint8_modified_class_scores.npy"
        bboxes_path = "outputs/ssd_vgg_35_300_uint8_modified_bboxes.npy"
        nms_model_path = "nms_operator.onnx"
        inference_single_nms(nms_model_path, class_scores_path, bboxes_path)
        compare_npy("outputs/nms_results.npy", args.golden_values)
        sys.exit(0)
    
    if args.run_pipeline:
        nms_results_path = "outputs/nms_results.npy"
        if 0 <= args.image_index < 1000:
            image_index = args.image_index 
        else:
            image_index=10
        draw_bboxes(dataset_dir, nms_results_path, image_index)
    
    if args.draw_bboxes:
        if 0 <= args.image_index < 1000:
            image_index = args.image_index 
        else:
            print("Image index out of range going with default value")
            image_index=10
        draw_bboxes("dataset", args.draw_bboxes, image_index)
    
if __name__ == "__main__":
    main()