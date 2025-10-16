#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# ===============================
# Fake GPU support
# ===============================
FAKE_GPU = os.environ.get("FAKE_GPU", "0") == "1"
USE_GPU = False

if FAKE_GPU:
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        if gpu_count > 0:
            USE_GPU = True
        else:
            print("No real GPU detected, falling back to CPU.")
    except Exception:
        print("pynvml not available or no GPU, falling back to CPU.")
else:
    # Detect CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        USE_GPU = True

# ===============================
# Generate sample data
# ===============================
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ===============================
# XGBoost parameters
# ===============================
params = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "tree_method": "gpu_hist" if USE_GPU else "hist",
    "device": "cuda" if USE_GPU else "cpu",
    "verbosity": 2,
}

print(f"Running XGBoost with tree_method={params['tree_method']} on device={params['device']}")

# ===============================
# Training
# ===============================
num_boost_round = 20
evals = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_boost_round=num_boost_round,evals=evals,verbose_eval=1 )

# ===============================
# Evaluation
# ===============================
y_pred = bst.predict(dtest)
y_pred_label = (y_pred > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_label)
loss = log_loss(y_test, y_pred)
print(f"\nFinal metrics -> Test accuracy: {acc:.4f}, Test logloss: {loss:.4f}")
