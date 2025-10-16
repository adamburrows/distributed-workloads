# Use an NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts
ENV FAKE_GPU=1

# Install Python and essential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install XGBoost with GPU support, and common ML libraries
RUN pip install --no-cache-dir \
    xgboost[cuda] \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    wandb

# Copy your training script
RUN mkdir -p /app
ADD xgboost-train.py /app/xgboost-train.py

# Set working directory
WORKDIR /app

# Default command
ENTRYPOINT ["python3", "/app/xgboost-train.py"]