# PyTorch MNIST Example

This document introduces **PyTorch** and the **MNIST dataset**, and demonstrates how to run a distributed training job on a Kubernetes cluster using **NVIDIA Run:ai**.

---

## Introduction

**PyTorch** is a popular open-source machine learning framework for building deep learning models.  
**MNIST** is a classic dataset of handwritten digits (0-9), commonly used for benchmarking image classification algorithms.

This example is designed to work with the **Fake GPU operator** for testing purposes, so no actual GPU hardware is required.

---

## Container Image

We will use a prebuilt container image.


<details>
<summary>If you want to build your own image, you could start from a PyTorch base image and load the MNIST dataset using Python.</summary>

#### Load the data
```
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
```

#### Create the Dockerfile
```
FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

ADD </path/to/dataset/locally> /opt/pytorch-mnist
WORKDIR /opt/pytorch-mnist

# Add folder for the logs.
RUN mkdir /katib

RUN chgrp -R 0 /opt/pytorch-mnist \
  && chmod -R g+rwX /opt/pytorch-mnist \
  && chgrp -R 0 /katib \
  && chmod -R g+rwX /katib

ENTRYPOINT ["python3", "/opt/pytorch-mnist/mnist.py"]
```

#### Build the image
```
docker build -t my-pytorch-mnist:latest .
```
</details>

## Runai Command
```bash
runai pytorch submit \
-p admin \
-i docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727 \
-g 1 \
--workers=2 \
--working-dir /opt/pytorch-mnist \
--command -- python3 "/opt/pytorch-mnist/mnist.py" "--epochs=1"
```
| Flag            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `-p`            | Run:ai project                                                              |
| `-i`            | Docker/Podman image to use                                                  |
| `-g`            | Number of whole GPUs                                                        |
| `--workers`     | Number of worker pods (2 in this case; 1 master pod is implied)             |
| `--working-dir` | Default directory when entering the container                               |
| `--command`     | Overrides the container's entrypoint; the command after `--` is executed    |