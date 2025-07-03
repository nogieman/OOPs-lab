import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat 
from onnxruntime.quantization.shape_inference import quant_pre_process
import logging
from torch.utils.data import DataLoader, Subset, Dataset
import os
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class TinyMnist(nn.Module):
    def __init__(self):
        super(TinyMnist, self).__init__()
        self.conv = nn.Conv2d(1, 20, 3, 1)
        self.fc = nn.Linear(3380, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = nn.Flatten()(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc(x)
        output = x
        return output

class PadLessMnist(nn.Module):
    def __init__(self):
        super(PadLessMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*512, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.log_softmax(x)
        return x

class AvgPoolMnist(nn.Module):
    def __init__(self):
        super(AvgPoolMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, stride=2)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# global average pool
class GapMnist(nn.Module):
    def __init__(self):
        super(GapMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 10, 3, 1, padding=1)
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 28, stride=2)
        x = nn.Flatten()(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class MnistQLinearAdd(nn.Module):
    def __init__(self):
        super(MnistQLinearAdd, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, padding=1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        z1 = F.relu(self.conv2(y))
        z2 = F.relu(self.conv3(y))
        z = z1 + z2
        z = F.max_pool2d(z, 2, stride=2)
        z = self.dropout1(z)
        z = self.flatten(z)
        z = self.fc1(z)
        z = F.relu(z)
        z = self.dropout2(z)
        z = self.fc2(z)
        output = F.log_softmax(z, dim=1)
        return output

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

# DATASET

def get_mnist_dataset(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST stats
    ])
    train_set = datasets.MNIST(
        root="../data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="../data", train=False, download=True, transform=transform
    )
    return (train_set, test_set)

def get_cifar10_dataset(batch_size=1):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    return (train_dataset, test_dataset)

# CALIB READERS

def get_mnist_calib_reader(batch_size=1):
    class MnistCalibrationDataReader(CalibrationDataReader):
        def __init__(self, data_loader):
            self.data_iter = iter(data_loader)

        def get_next(self):
            try:
                images, _ = next(self.data_iter)  # Get the next batch of images (ignoring labels)
                return {"input": images.numpy()}  # Convert to NumPy array
            except StopIteration:
                return None
    calibration_dataset = get_mnist_dataset()[0]
    calib_subset = Subset(calibration_dataset, indices=range(20000))
    calibration_loader = DataLoader(calib_subset, batch_size=batch_size)
    calibration_reader = MnistCalibrationDataReader(calibration_loader)
    return calibration_reader

def get_cifar10_calib_reader(batch_size=1):
    class CIFAR10CalibrationDataReader(CalibrationDataReader):
        def __init__(self, data_loader):
            self.data_loader = data_loader
            self.data_iter = iter(data_loader)

        def get_next(self):
            try:
                data = next(self.data_iter)[0]  # Get only the inputs
                return {"input": data.numpy()}
            except StopIteration:
                return None
    calibration_dataset = get_cifar10_dataset()[0]
    calib_subset = Subset(calibration_dataset, indices=range(5000))
    calibration_loader = DataLoader(calib_subset, batch_size=batch_size)
    calibration_reader = CIFAR10CalibrationDataReader(calibration_loader)
    return calibration_reader

def get_imagenet_calib_reader(image_dir, batch_size=1):
    class ImageNetCalibrationDataReader:
        def __init__(self, data_loader):
            self.data_iter = iter(data_loader)
        def get_next(self):
            try:
                images, _ = next(self.data_iter)
                return {"input": images.numpy()}
            except StopIteration:
                return None
                
    class ImageNetDataset(Dataset):
        def __init__(self, image_dir):
            self.image_paths = [f for f in os.listdir(image_dir) if f.endswith('.JPEG')]
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            img_path = os.path.join(image_dir, self.image_paths[idx])
            image = Image.open(img_path).convert('RGB')
            return self.transform(image), 0

    calibration_loader = DataLoader(ImageNetDataset(image_dir), batch_size=batch_size)
    return ImageNetCalibrationDataReader(calibration_loader)

def train(args, model, device, train_loader, optimizer, epoch):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# QUANTIZATION

def model_quantize(onnx_path, out_model, data_reader):
    quant_pre_process(onnx_path, "/tmp/pre_proc.onnx")
    quantize_static(
        model_input="/tmp/pre_proc.onnx",
        model_output=out_model,
        calibration_data_reader=data_reader,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, 
        quant_format=QuantFormat.QOperator,
        extra_options={"ActivationSymmetric": True}
    )

def quantize_mnist(in_model, out_model):
    if type(in_model) is str:
        model = torch.load(in_model)
    else:
        model = in_model
    model = torch.load(in_model)
    model.to('cpu')
    input_dims = (1, 1, 28, 28)
    dummy_input = torch.randn(input_dims)
    onnx_path = "intermidiate_tmp.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])
    # Perform quantization
    model_quantize(onnx_path, out_model, get_mnist_calib_reader())

def quantize_cifar_10(in_model, out_model):
    if type(in_model) is str:
        model = torch.load(in_model)
    else:
        model = in_model
    model = torch.load(in_model)
    model.to('cpu')
    input_dims = (1, 3, 32, 32)
    dummy_input = torch.randn(input_dims)
    onnx_path = "intermidiate_tmp.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])
    # Perform quantization
    model_quantize(onnx_path, out_model, get_cifar10_calib_reader())

def quantize_imagenet(in_model, out_model):
    if type(in_model) is str:
        model = torch.load(in_model)
    else:
        model = in_model
    model.to('cpu')
    input_dims = (1, 3, 224, 224)
    dummy_input = torch.randn(input_dims)
    onnx_path = "intermidiate_tmp.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])
    # Perform quantization
    model_quantize(onnx_path, out_model, get_imagenet_calib_reader(image_dir="/home/metal/dev/datasets/imagenet-sample-images/"))


# GENERATE DATASET NPY

def generate_cifar_npy():
    cifar_ds = get_cifar10_dataset()
    img_array = []
    for i in range(100):
        img, label = cifar_ds[i]
        img_array.append(img.numpy())

    ndar = np.stack(img_array, axis=0)
    np.save("cifar_100_images.npy", ndar)

# dataset, model def, train=true, onnx=true
def main(dataset, model):
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='For loading torch.pt')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.load_model:
        model = torch.load("torch.pt", map_location=device)

    train_ds, test_ds = dataset
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch_out_name = "torch.pt"
    torch.save(model, torch_out_name)
    return torch_out_name

def get_mobilenet(name):
    if name == "v2":
        return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    elif name == "v3_small":
        return torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
    elif name == "v3_large":
        return torchvision.models.mobilenetv3.mobilenet_v3_small(pretrained=True)
    else:
        raise ValueError(f"Dont know any mobilenet of this name: {name}")

def hub():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True)
    torch.save(model, "hub.pt")
    def quantize():
        quantize_cifar_10("hub.pt", "cifar10_vgg11.onnx")
    def test_acc(model):
        model.to("cuda")
        train_ds, test_ds = get_cifar10_dataset()
        test_loader = DataLoader(test_ds, batch_size=128)
        test(model, "cuda", test_loader)
    quantize()


if __name__ == '__main__':
    ds = get_mnist_dataset()
    model = Net()
    pt_model = main(ds, model)
    quantize_mnist(pt_model, "mnist_int8_stride2_pad2.onnx")
