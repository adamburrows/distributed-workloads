# Use a small Python base that supports both x86_64 and arm64
FROM python:3.10-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Add a User to allow non-root ---
RUN useradd -m myuser
USER myuser
WORKDIR /home/myuser

# --- Copy code ---
COPY mpi-train.py /home/myuser/mpi-train.py

# --- Python dependencies ---
RUN pip install --no-cache-dir mpi4py numpy

# --- Fake GPU env ---
ENV CUDA_VISIBLE_DEVICES="0"

# --- Default command ---
CMD ["mpirun", "python3", "/home/myuser/mpi-train.py"]