from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(42 + rank)

# Simulate fake GPU environment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)

# ==== CONFIG ====
num_samples = 2000
num_features = 10
epochs = 20
lr = 0.05

# ==== DATA SPLIT ====
# Each rank gets its own chunk
samples_per_rank = num_samples // size
X_local = np.random.randn(samples_per_rank, num_features)
true_w = np.arange(1, num_features + 1)
y_local = X_local.dot(true_w) + np.random.randn(samples_per_rank) * 0.5

# ==== MODEL INIT ====
w = np.zeros(num_features)

# ==== TRAIN LOOP ====
for epoch in range(epochs):
    # Predictions and gradient
    y_pred = X_local.dot(w)
    error = y_pred - y_local
    grad = (2 / samples_per_rank) * X_local.T.dot(error)

    # Average gradients across all ranks
    avg_grad = np.zeros_like(grad)
    comm.Allreduce(grad, avg_grad, op=MPI.SUM)
    avg_grad /= size

    # Update weights
    w -= lr * avg_grad

    # Compute global loss
    local_loss = np.mean(error ** 2)
    total_loss = comm.allreduce(local_loss, op=MPI.SUM) / size

    if rank == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

# ==== RESULTS ====
if rank == 0:
    print("\nTraining complete.")
    print("True weights:", true_w)
    print("Learned weights:", np.round(w, 2))