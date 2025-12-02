import os
import numpy as np
import torch
import config

def load_scaler_params():
    """Reads mean and std from text file."""
    if os.path.exists(config.SCALER_PARAMS_PATH):
        with open(config.SCALER_PARAMS_PATH, "r") as f:
            c = f.read().strip().split(',')
            return float(c[0]), float(c[1])
    else:
        print("Warning: Scaler params not found. Defaulting to 12.0, 10.0")
        return 12.0, 10.0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cka_score(X, Y):
    """Computes Linear CKA between two matrices X and Y."""
    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Frobenius norms
    norm_X = np.linalg.norm(X @ X.T, 'fro')
    norm_Y = np.linalg.norm(Y @ Y.T, 'fro')
    return np.trace((X @ X.T) @ (Y @ Y.T)) / (norm_X * norm_Y)