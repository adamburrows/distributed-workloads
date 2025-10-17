# MPI Simple Example

This document introduces **MPI**, and demonstrates how to run a distributed training job on a Kubernetes cluster using **NVIDIA Run:ai**.

---

## Introduction

**MPI** or Message Passing Interface, is a standard for parallel computing where processes communicate by sending messages.

This example is designed to work with the **Fake GPU operator** for testing purposes, so no actual GPU hardware is required.

## Container Image

We will use a prebuilt container image.

<details>
<summary>Interested in building your own image?</summary>

1. Download the files in the build-image directory.
2. Optionally make modifications to the Dockerfile and/or .py file.
3. Build, tag, and push the image.
```
podman build -t my-test-image:latest --platform linux/amd64 .
podman tag localhost/my-test-image:latest docker.io/<user>/my-test-image:latest
podman push docker.io/<user>/my-test-image:latest
```
</details>

Run the MPI workload.  

## Runai Command - MPI workload
```bash
runai mpi submit \
-p testproject \
-i docker.io/hireamb/mpi-simple-example:latest \
-g 1 \
--workers=2 \
--master-no-pvcs \
--existing-pvc claimname=dist-datasets-v3-project-hpzdp,path=/opt/data \
--command -- mpirun python3 /home/myuser/mpi-train.py
```

| Flag            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `-p`            | Run:ai project                                                              |
| `-i`            | Docker/Podman image to use                                                  |
| `-g`            | Number of whole GPUs                                                        |
| `--workers`     | Number of worker pods (2 in this case; 1 master pod is implied)             |
| `--master-no-pvcs` | Don't mount volume to launcher pod                               |
| `--existing-pvc`| Mount persistent storage so we can save the accuracy/loss metrics                       |
| `--command`     | Overrides the container's entrypoint; the command after `--` is executed    |