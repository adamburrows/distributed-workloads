# distributed-workloads

This repository provides a reference workflow for running distributed AI workloads on a Kubernetes cluster managed with **NVIDIA Run:ai**. It assumes you have Python installed along with necessary libraries, and access to Podman or Docker for container management.

---

## Prerequisites

Before getting started, ensure you have:

- Access to a Kubernetes cluster with **NVIDIA Run:ai** installed. Optionally understand how to convert the runai commands into podman/docker syntax.
- `kubectl` configured to access your cluster.
- Podman or Docker engine installed.
- Python environment with required libraries (e.g., `torch`, `torchvision`, `numpy`, etc.).
