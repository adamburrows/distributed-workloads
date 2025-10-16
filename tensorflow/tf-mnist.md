# Tensorflow MNIST Example

This document introduces **Tensorflow** and **Tensorboard**, and demonstrates how to run a distributed training job on a Kubernetes cluster using **NVIDIA Run:ai**.

---

## Introduction

**Tensorflow** is an open-source machine learning framework developed by Google that enables building, training, and deploying deep learning models across CPUs, GPUs, and distributed systems.
**Tensorboard** is TensorFlow’s built-in visualization tool that lets you monitor and analyze model metrics, such as loss, accuracy, and computational graphs.
**MNIST** is a classic dataset of handwritten digits (0-9), commonly used for benchmarking image classification algorithms.


This example is designed to work with the **Fake GPU operator** for testing purposes, so no actual GPU hardware is required.

---

## Container Image

We will use a prebuilt container image.

## Runai Command - TF workload
```bash
runai tensorflow submit \
-p admin \
-i kubeflow/tf-mnist-with-summaries:latest \
-g 1 \
--workers=3 \
--existing-pvc claimname=dist-datasets-v2-project-c1vmr,path=/tmp/tensorflow/mnist/logs/mnist_with_summaries \
--command -- python /var/tf_mnist/mnist_with_summaries.py
```

| Flag               | Description                                                                  |
|-------------------|----------------------------------------------------------------------------|
| `-p`               | Run:ai project                                                             |
| `-i`               | Docker/Podman image to use                                                 |
| `-g`               | Number of whole GPUs                                                       |
| `--workers`        | Number of worker pods (2 in this case; 1 master pod is implied)           |
| `--existing-pvc`   | Mount persistent storage so we can save the accuracy/loss metrics          |
| `--command`        | Overrides the container's entrypoint; the command after `--` is executed   |

## Runai Command - Tensorboard
```bash
runai workspace submit \
-p testproject \
-i tensorflow/tensorflow:latest \
--gpu-portion-request 0.5 \
--external-url container=6006 \
--existing-pvc claimname=dist-datasets-v2-project-c1vmr,path=/opt/data \
--command -- tensorboard --logdir /opt/data --host 0.0.0.0 --path_prefix /\${RUNAI_PROJECT}/\${RUNAI_JOB_NAME}
```

| Flag                     | Description                                                                    |
|---------------------------|-------------------------------------------------------------------------------|
| `-p`                     | Run:ai project                                                                 |
| `-i`                     | Docker/Podman image to use                                                     |
| `--workers`              | Number of worker pods (2 in this case; 1 master pod is implied)                |
| `--gpu-portion-request`  | Fractional GPU request                                                         |
| `--external-url`         | Create an ingress connection available from the UI                             |
| `--existing-pvc`         | Mount persistent storage so we can save the accuracy/loss metrics              |
| `--command`              | Overrides the container's entrypoint; the command after `--` is executed       |