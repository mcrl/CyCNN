# Environment

Below are environments that authors used.

- OS: CentOS Linux 7 (Core)
- CUDA Driver Version: 410.48 
- **gcc: 7.3.0**
- **nvcc(CUDA): release 10.0, V10.0.130**
- CPU: Intel(R) Xeon(R) Gold 6130F CPU @ 2.10GHz
- GPU: NVIDIA Tesla V100
- **Python: 3.6**
- **PyTorch: 1.2.0**
- **Torchvision: 0.4.0**

## Docker

We provide a docker image to simplify the setting. **Docker version >= 19.03 and CUDA driver of version at least 384.00 on the host is required for CUDA 10.0 functionality to work.**  [Link](http://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/) might be helpful if docker and GPU related error occurs. Make sure that Nvidia GPU driver and container runtime are properly installed.


```plaintext
# Pull docker image (~4.9GB)
docker pull cycnn/dockerimage:version3

# Run the container
docker run --gpus all -it -d cycnn/dockerimage:version3 /bin/bash

# Check the container id and name
docker ps -a

# Move the CyCNN source code to the container
docker cp cycnn-codes.zip <CONTAINER-ID>:/root/

# Attach to the container
docker attach <CONTAINER-NAME>

# Unzip file
cd root
unzip cycnn-codes.zip -d cycnn

# See the instructions (How to run) step 3.
```


# Code Structure
```plaintext
cycnn/
├── cycnn/
│   ├── data/                   (Directory for training data)
│   ├── logs/                   (Directory for training logs)
│   ├── models/                 (PyTorch implementations of CNN and CyCNN models)
│   │   ├── cyconvlayer.py
│   │   ├── cyresnet.py
│   │   ├── cyvgg.py
│   │   ├── getmodel.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── saves/                  (Directory for savining model parameters from training)
│   ├── main.py                 (Main script for model training/testing)
│   ├── data.py                 (Script for loading datasets)
│   ├── image_transforms.py     (Script for transforming images in various ways)
│   └── utils.py
├── cycnn-extension/            (Directory for CyCNN CUDA extension)
│   ├── cycnn.cpp
│   ├── cycnn_cuda.cu           (CUDA kernel code of cyconv layer)
│   └── setup.py
├── ReadMe.md

```

# How to run

**Make sure to have compatible gcc, nvcc(CUDA), Python, and PyTorch versions installed. We recommend to use the docker image that we provide. If you want to use another version of gcc or nvcc, you should check compatibility with Python, PyTorch, and Torchvision versions.**

You can skip step 1 and 2 if you use the provided docker image.

## 1. Install Requirements

**You can skip this step if you use provided docker image.** All needed pakcages are already installed in the image.
```plaintext
cycnn/cycnn> pip install -r requirements.txt
```

## 2. Install CyCNN extension

**You can skip this step if you use provided docker image.** CyCNN extension is already installed in the image.

Install the CyCNN extension using:

```plaintext
~/cycnn/cycnn-extension> python setup.py install
```

Then, you can use the CyCNN extension as follows: 

```plaintext
import CyConv2d_cuda

output = CyConv2d_cuda.forward(
        input, weight, workspace, stride, padding, dilation)
```

We implement `CyConv2d` wrapper in `cycnn/models/cyconvlayer.py`. It has almost same interface with `torch.nn.Conv2d`. 


## 3. Model Training / Testing

`~/cycnn/cycnn/main.py` is the main scipt for training/testing CNN/CyCNN models. 

```plaintext
usage: main.py [-h] [--model MODEL] [--train] [--test]
               [--polar-transform POLAR_TRANSFORM]
               [--augmentation AUGMENTATION] [--data-dir DATA_DIR]
               [--batch-size BATCH_SIZE] [--num-epochs NUM_EPOCHS] [--lr LR]
               [--dataset DATASET] [--redirect]
               [--early-stop-epochs EARLY_STOP_EPOCHS] [--test-while-training]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to train.
  --train               If used, run the script with training mode.
  --test                If used, run the script with test mode.
  --polar-transform POLAR_TRANSFORM
                        Polar transformation. Should be one of
                        linearpolar/logpolar.
  --augmentation AUGMENTATION
                        Training data augmentation. Should be one of
                        rot/trans/rottrans.
  --data-dir DATA_DIR   Directory path to save datasets.
  --batch-size BATCH_SIZE
                        Batch size used in training.
  --num-epochs NUM_EPOCHS
                        Number of maximum epochs.
  --lr LR               learning rate.
  --dataset DATASET     Dataset. Should be one of mnist/svhn/cifar10/cifar100
  --redirect            If used, redirect stdout to log file in logs/ .
  --early-stop-epochs EARLY_STOP_EPOCHS
                        Epochs to wait until early stopping.
  --test-while-training
                        If used with --train, run tests at every training
                        epoch.

```

For example, following script will perform CyVGG19 training on MNIST dataset with linearpolar transformation. It will save model parameters with highest validation accuracy in the `~/cycnn/cycnn/saves` directory.

```plaintext
~/cycnn/cycnn> python main.py --train --model=cyvgg19 --dataset=mnist \
				--polar-transform=linearpolar 

configuration:  {'model': 'cyvgg19', 'train': True, 'test': False, 
'polar_transform': 'linearpolar', 'augmentation': None, 
'data_dir': './data', 'batch_size': 128, 'num_epochs': 9999999,
'lr': 0.05, 'dataset': 'mnist', 'redirect': False, 'early_stop_epochs': 15, 
'test_while_training': False}
Using device:  cuda
1 devices available
# Parameters: 20559.2K
54000 Train data. 6000 Validation data. 10000 Test data.
===> Training mnist-cyvgg19-linearpolar begin
[Epoch 0] Train Loss: 0.508112
[Epoch 0] Validation loss: 0.1685, Accuracy: 5669/6000 (94.48%)
Saving model checkpoint to saves/mnist-cyvgg19-linearpolar.pt
Elapsed time: 18.0 sec
...
[Epoch 52] Train Loss: 0.000117
[Epoch 52] Validation loss: 0.0367, Accuracy: 5966/6000 (99.43%)
Elapsed time: 18.4 sec
Training Done!
```

Then we can run the test using the saved checkpoint. Note that testing always use rotated version of each datasets.

```plaintext
~/cycnn/cycnn> python main.py --test --model=cyvgg19 --dataset=mnist \
			   --polar-transform=linearpolar

configuration:  {'model': 'cyvgg19', 'train': False, 'test': True, 
'polar_transform': 'linearpolar', 'augmentation': None, 
'data_dir': './data', 'batch_size': 128, 'num_epochs': 9999999, 
'lr': 0.05, 'dataset': 'mnist', 'redirect': False, 'early_stop_epochs': 15,
'test_while_training': False}
Using device:  cuda
1 devices available
# Parameters: 20559.2K
54000 Train data. 6000 Validation data. 10000 Test data.
===> Testing mnist-cyvgg19-linearpolar with rotated dataset begin
Test loss: 1.0565, Accuracy: 8343/10000 (83.43%)
```

