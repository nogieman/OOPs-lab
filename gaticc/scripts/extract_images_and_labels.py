import numpy as np
from PIL import Image
import os.path
from os.path import join
import json

# for mnist and cifar
from torchvision import datasets, transforms

# store imagenet_*.npy 
# imagenet-sample-images are from https://github.com/EliSchwartz/imagenet-sample-images/
def load_images(directory, preprocess, save_file_prefix):
    image_list = []

    for filename in sorted(os.listdir(directory)):  # Sort to maintain order
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            processed_image = preprocess(file_path)  # Apply user-defined function
            image_list.append(processed_image)

    rr = np.concatenate(np.stack(image_list), axis=0)
    r2 = rr[:2]
    r5 = rr[:5]
    r10 = rr[:10]
    r100 = rr[:100]
    r500 = rr[:500]
    r1000 = rr[:1000]

    np.save(save_file_prefix + "_2", r2)
    np.save(save_file_prefix + "_5", r5)
    np.save(save_file_prefix + "_10", r10)
    np.save(save_file_prefix + "_100", r100)
    np.save(save_file_prefix + "_500", r500)
    np.save(save_file_prefix + "_1000", r1000)

# extract imagenet labels into *labels.txt file
# labels.json from 
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
def extract_labels(directory, labels_json, output_txt):
    # Load the labels mapping
    with open(labels_json, "r") as f:
        labels_dict = json.load(f)

    # Create a reverse mapping from class ID to index
    id_to_index = {v[0]: int(k) for k, v in labels_dict.items()}

    print(id_to_index)
    indices = []  # Store extracted indices

    # Iterate over image files
    for filename in sorted(os.listdir(directory)):  
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  
            class_id = filename.split("_")[0]  # Extract ID from filename
            if class_id in id_to_index:
                print(filename, id_to_index[class_id])
                indices.append(id_to_index[class_id])  # Store index

    # Convert to NumPy array and save to a text file
    indices_array = np.array(indices, dtype=np.int32)
    np.savetxt(output_txt + "_2_labels.txt", indices_array[:2], fmt="%d")
    np.savetxt(output_txt + "_5_labels.txt", indices_array[:5], fmt="%d")
    np.savetxt(output_txt + "_10_labels.txt", indices_array[:10], fmt="%d")
    np.savetxt(output_txt + "_100_labels.txt", indices_array[:100], fmt="%d")
    np.savetxt(output_txt + "_500_labels.txt", indices_array[:500], fmt="%d")
    np.savetxt(output_txt + "_1000_labels.txt", indices_array[:1000], fmt="%d")

    print(f"Saved {len(indices)} labels to {output_txt}")

def save_mnist_samples(output_dir="mnist_samples", sample_sizes=[2, 5, 10, 100, 500, 1000, 10000]):
    os.makedirs(output_dir, exist_ok=True)

    # Load MNIST Test Dataset
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    for size in sample_sizes:
        images, labels = zip(*[(mnist_test[i][0].numpy(), mnist_test[i][1]) for i in range(size)])

        # Save images as .npy
        np.save(os.path.join(output_dir, f"mnist_{size}.npy"), np.array(images))

        # Save labels as .txt
        np.savetxt(os.path.join(output_dir, f"mnist_{size}_labels.txt"), np.array(labels, dtype=np.int32), fmt="%d")

        print(f"Saved {size} images & labels in {output_dir}")

def save_cifar_samples(output_dir="cifar_samples", sample_sizes=[2, 5, 10, 100, 500, 1000, 10000]):
    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR-10 Test Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    for size in sample_sizes:
        images, labels = zip(*[(cifar_test[i][0].numpy(), cifar_test[i][1]) for i in range(size)])

        # Save images as .npy
        np.save(os.path.join(output_dir, f"cifar_{size}.npy"), np.array(images))

        # Save labels as .txt
        np.savetxt(os.path.join(output_dir, f"cifar_{size}_labels.txt"), np.array(labels, dtype=np.int32), fmt="%d")

        print(f"Saved {size} images & labels in {output_dir}")

#load_images("images/imagenet-sample-images/", preprocess, "imagenet")
#extract_labels("images/imagenet-sample-images/", "labels.json", "imagenet")

#save_mnist_samples(output_dir=".")

save_cifar_samples(output_dir=".")
