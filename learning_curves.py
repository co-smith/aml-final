"""
1) learning_curves.py (Lecture 23)
Goal: diagnose bias vs variance.
What to do: retrain the model using 20%, 40%, 60%, 80%, and 100% of the training data
and plot Training RMSE and Validation RMSE for each run.
Why: if the training and validation curves converge we likely have high bias (need a better model).
If there's a persistent gap we likely have high variance (need more data).
This answers the rubric's "error analyses" requirement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import config
import utils
import os

print("--- Learning Curve Analysis: Bias vs Variance Diagnosis ---\n")

# Load scaler params for unscaling
MEAN, STD = utils.load_scaler_params()

# Training data percentages to test
train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

# Storage for results
results = {
    'Linear Regression': {'train': [], 'val': []},
    'XGBoost': {'train': [], 'val': []},
    'Transformer': {'train': [], 'val': []},
    'FNO': {'train': [], 'val': []}
}

# ========== TRADITIONAL ML MODELS (Tabular Data) ==========

print("Training Traditional ML Models (Linear Regression & XGBoost)...\n")

# Load tabular data
df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
y = df['target_temp_t+1']
X = df.drop(columns=['target_temp_t+1'])

# Full splits
X_train_full = X.loc[X.index < config.TEST_START_DATE]
y_train_full = y.loc[y.index < config.TEST_START_DATE]
X_val = X.loc[(X.index >= config.VAL_START_DATE) & (X.index < config.TEST_START_DATE)]
y_val = y.loc[(y.index >= config.VAL_START_DATE) & (y.index < config.TEST_START_DATE)]

for size_pct in train_sizes:
    n_samples = int(len(X_train_full) * size_pct)
    X_train_subset = X_train_full.iloc[:n_samples]
    y_train_subset = y_train_full.iloc[:n_samples]

    print(f"Training with {size_pct*100:.0f}% of data ({n_samples} samples)...")

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_subset, y_train_subset)

    # Predictions
    y_pred_train_lr = lr.predict(X_train_subset)
    y_pred_val_lr = lr.predict(X_val)

    # RMSE (unscaled)
    rmse_train_lr = np.sqrt(mean_squared_error(y_train_subset * STD + MEAN, y_pred_train_lr * STD + MEAN))
    rmse_val_lr = np.sqrt(mean_squared_error(y_val * STD + MEAN, y_pred_val_lr * STD + MEAN))

    results['Linear Regression']['train'].append(rmse_train_lr)
    results['Linear Regression']['val'].append(rmse_val_lr)

    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train_subset, y_train_subset)

    # Predictions
    y_pred_train_xgb = xgb.predict(X_train_subset)
    y_pred_val_xgb = xgb.predict(X_val)

    # RMSE (unscaled)
    rmse_train_xgb = np.sqrt(mean_squared_error(y_train_subset * STD + MEAN, y_pred_train_xgb * STD + MEAN))
    rmse_val_xgb = np.sqrt(mean_squared_error(y_val * STD + MEAN, y_pred_val_xgb * STD + MEAN))

    results['XGBoost']['train'].append(rmse_train_xgb)
    results['XGBoost']['val'].append(rmse_val_xgb)

    print(f"  Linear Reg - Train: {rmse_train_lr:.4f}°C, Val: {rmse_val_lr:.4f}°C")
    print(f"  XGBoost    - Train: {rmse_train_xgb:.4f}°C, Val: {rmse_val_xgb:.4f}°C\n")

# ========== DEEP LEARNING MODELS (Sequential Data) ==========

print("Training Deep Learning Models (Transformer & FNO)...\n")

# Load datasets
train_ds_full = WeatherDataset(end_date=config.TRAIN_END_DATE)
val_ds = WeatherDataset(start_date=config.VAL_START_DATE, end_date=config.VAL_END_DATE)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

def train_dl_model(model, train_loader, val_loader, epochs=20):
    """Train a deep learning model and return train/val RMSE"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)

            # No input masking for fair comparison
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

    # Compute training RMSE
    model.eval()
    train_loss = 0.0
    with torch.no_grad():
        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            pred = model(X)
            train_loss += criterion(pred, y).item() * X.size(0)
    train_rmse = np.sqrt(train_loss / len(train_loader.dataset)) * STD

    # Compute validation RMSE
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            val_loss += criterion(model(X), y).item() * X.size(0)
    val_rmse = np.sqrt(val_loss / len(val_loader.dataset)) * STD

    return train_rmse, val_rmse

for size_pct in train_sizes:
    n_samples = int(len(train_ds_full) * size_pct)
    train_subset = Subset(train_ds_full, range(n_samples))
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"Training with {size_pct*100:.0f}% of data ({n_samples} samples)...")

    # --- Transformer ---
    tf = TimeSeriesTransformer(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
    rmse_train_tf, rmse_val_tf = train_dl_model(tf, train_loader, val_loader, epochs=20)

    results['Transformer']['train'].append(rmse_train_tf)
    results['Transformer']['val'].append(rmse_val_tf)

    # --- FNO ---
    fno = SimpleFNO(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
    rmse_train_fno, rmse_val_fno = train_dl_model(fno, train_loader, val_loader, epochs=20)

    results['FNO']['train'].append(rmse_train_fno)
    results['FNO']['val'].append(rmse_val_fno)

    print(f"  Transformer - Train: {rmse_train_tf:.4f}°C, Val: {rmse_val_tf:.4f}°C")
    print(f"  FNO         - Train: {rmse_train_fno:.4f}°C, Val: {rmse_val_fno:.4f}°C\n")

# ========== PLOTTING ==========

print("Generating learning curve plots...\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

models = ['Linear Regression', 'XGBoost', 'Transformer', 'FNO']
colors = ['blue', 'green', 'orange', 'red']

for idx, (model_name, color) in enumerate(zip(models, colors)):
    ax = axes[idx]

    train_rmse = results[model_name]['train']
    val_rmse = results[model_name]['val']
    sample_counts = [int(len(train_ds_full) * pct) for pct in train_sizes]

    # Plot curves
    ax.plot(sample_counts, train_rmse, 'o-', color=color, label='Training RMSE', linewidth=2, markersize=8)
    ax.plot(sample_counts, val_rmse, 's--', color=color, alpha=0.7, label='Validation RMSE', linewidth=2, markersize=8)

    # Styling
    ax.set_xlabel('Training Set Size (samples)', fontsize=11)
    ax.set_ylabel('RMSE (°C)', fontsize=11)
    ax.set_title(f'{model_name} Learning Curve', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Diagnosis text
    final_gap = val_rmse[-1] - train_rmse[-1]
    if final_gap < 0.1:
        diagnosis = "High Bias (converged)"
        suggestion = "Need better model/features"
    else:
        diagnosis = "High Variance (gap persists)"
        suggestion = "Need more training data"

    ax.text(0.98, 0.97, f'{diagnosis}\n{suggestion}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "learning_curves.png"), dpi=300)
print(f"Saved learning_curves.png to {config.OUTPUT_DIR}/")

# ========== SUMMARY ==========

print("\n" + "="*60)
print("LEARNING CURVE ANALYSIS SUMMARY")
print("="*60)

for model_name in models:
    print(f"\n{model_name}:")
    print(f"  Training RMSE:   {results[model_name]['train']}")
    print(f"  Validation RMSE: {results[model_name]['val']}")

    final_train = results[model_name]['train'][-1]
    final_val = results[model_name]['val'][-1]
    gap = final_val - final_train

    print(f"  Final Gap: {gap:.4f}°C")

    if gap < 0.1:
        print("  → DIAGNOSIS: High Bias (curves converged)")
        print("  → RECOMMENDATION: This model has reached its capacity. Need better features or more complex model.")
    else:
        print("  → DIAGNOSIS: High Variance (persistent gap)")
        print("  → RECOMMENDATION: This model could benefit from more training data or regularization.")

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("- If train/val curves are close together: HIGH BIAS")
print("  (Model is too simple, can't capture the pattern)")
print("- If there's a large gap: HIGH VARIANCE")
print("  (Model overfits training data, needs more data)")
print("- If curves are still decreasing: MORE DATA HELPS")
print("- If curves have plateaued: BETTER MODEL NEEDED")
print("="*60)
