# XGBoost Simple Example

This document introduces **XGBoost**, and demonstrates how to run a distributed training job on a Kubernetes cluster using **NVIDIA Run:ai**.

---

## Introduction

**XGBoost** or eXtreme Gradient Boosting, is an optimized machine learning library that uses gradient-boosted decision trees. It is known for its high performance, speed, and ability to handle large datasets efficiently.  

This example is designed to work with the **Fake GPU operator** for testing purposes, so no actual GPU hardware is required.

## Container Image

We will use a prebuilt container image.

<details>
<summary>Interested in building your own image?</summary>

1. Download the files in the build-image directory.
2. Optionally make modifications to the Dockerfile and/or .py file.
3. Build, tag, and push the image.
```
podman build -t my-test-image:latest .
podman tag localhost/my-test-image:latest docker.io/<user>/my-test-image:latest
podman push docker.io/<user>/my-test-image:latest
```
</details>

Run the XGBoost workload.  

## Runai Command - XGBoost workload
```bash
runai xgboost submit \
-p testproject \
-i docker.io/hireamb/xgboost-simple-example:latest \
-g 1 \
--workers=2 \
--existing-pvc claimname=dist-datasets-v3-project-hpzdp,path=/opt/data \
--command -- python3 "/app/xgboost-train.py"
```