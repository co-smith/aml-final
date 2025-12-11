import os
# Fix threading issues that cause segfaults on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import config
import utils
import joblib

def get_reconstructed_preds(model, loader):
    model.eval()
    preds_real = []
    actuals_real = []
    
    with torch.no_grad():
        for X, y_diff in loader:
            X = X.to(config.DEVICE)
            
            # 1. Get Prediction (This is the DELTA)
            pred_diff = model(X).cpu().numpy().flatten()
            
            # 2. Get Current Temp (t-0) to add delta to
            # X shape: (Batch, Seq, Feat). Seq=-1 is t-0. Feat=0 is Temp.
            current_temp = X[:, -1, 0].cpu().numpy().flatten()
            
            # 3. Reconstruct: Next = Current + Delta
            pred_val = current_temp + pred_diff
            
            # 4. Reconstruct Actuals too: Actual Next = Current + Actual Delta
            actual_val = current_temp + y_diff.numpy().flatten()
            
            preds_real.extend(pred_val)
            actuals_real.extend(actual_val)
            
    return np.array(preds_real), np.array(actuals_real)

print("--- Step 4: Deep Learning Evaluation (Delta Reconstruction) ---")
MEAN, STD = utils.load_scaler_params()

test_ds = WeatherDataset(start_date=config.TEST_START_DATE)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Load Models
tf = TimeSeriesTransformer(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
tf.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "Transformer_best.pth"), map_location=config.DEVICE))

fno = SimpleFNO(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
fno.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "FNO_best.pth"), map_location=config.DEVICE))

# 1. Standard Forecast - Deep Learning Models
p_tf, y_true = get_reconstructed_preds(tf, test_loader)
p_fno, _ = get_reconstructed_preds(fno, test_loader)

# Unscale (Apply Mean/Std to the Reconstructed Values)
p_tf_real = p_tf * STD + MEAN
p_fno_real = p_fno * STD + MEAN
y_real = y_true * STD + MEAN

# 2. Load Traditional ML Models and Get Predictions
print("\nLoading traditional ML models...")
lr = joblib.load(os.path.join(config.CHECKPOINT_DIR, "linear_regression.pkl"))
xgb = joblib.load(os.path.join(config.CHECKPOINT_DIR, "xgboost.pkl"))

# Load tabular test data for ML models
df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
X_test_tabular = df.loc[df.index >= config.TEST_START_DATE].drop(columns=['target_temp_t+1'])
y_test_tabular = df.loc[df.index >= config.TEST_START_DATE]['target_temp_t+1']
test_dates = df.loc[df.index >= config.TEST_START_DATE].index[:len(test_ds)]

# Get predictions (deltas in scaled space)
p_lr_delta = lr.predict(X_test_tabular[:len(test_ds)])
p_xgb_delta = xgb.predict(X_test_tabular[:len(test_ds)])

# Reconstruct absolute temperatures: pred = current_temp + delta
current_temps_tabular = X_test_tabular['temp_2m_t-0'].values[:len(test_ds)]
p_lr = current_temps_tabular + p_lr_delta
p_xgb = current_temps_tabular + p_xgb_delta

# Unscale
p_lr_real = p_lr * STD + MEAN
p_xgb_real = p_xgb * STD + MEAN

# Metrics for all models
rmse_tf = np.sqrt(np.mean((p_tf_real - y_real)**2))
rmse_fno = np.sqrt(np.mean((p_fno_real - y_real)**2))
rmse_lr = np.sqrt(np.mean((p_lr_real - y_real)**2))
rmse_xgb = np.sqrt(np.mean((p_xgb_real - y_real)**2))

print("\nAll Models Test RMSE:")
print(f"Linear Regression: {rmse_lr:.2f}°C")
print(f"XGBoost:           {rmse_xgb:.2f}°C")
print(f"Transformer:       {rmse_tf:.2f}°C")
print(f"FNO:               {rmse_fno:.2f}°C")

# Plot Forecast Comparison - ALL MODELS
plt.figure(figsize=(14, 7))
plt.plot(y_real[:150], label='Actual', color='black', linewidth=2.5, zorder=5)
plt.plot(p_lr_real[:150], label=f'Linear Reg ({rmse_lr:.2f}°C)', linestyle=':', linewidth=1.5, alpha=0.8)
plt.plot(p_xgb_real[:150], label=f'XGBoost ({rmse_xgb:.2f}°C)', linestyle='-.', linewidth=2, alpha=0.9)
plt.plot(p_tf_real[:150], label=f'Transformer ({rmse_tf:.2f}°C)', linestyle='--', linewidth=1.5, alpha=0.8)
plt.plot(p_fno_real[:150], label=f'FNO ({rmse_fno:.2f}°C)', linestyle='-', linewidth=2, alpha=0.9)
plt.legend(loc='best', fontsize=10)
plt.xlabel('Time Step (Hours)')
plt.ylabel('Temperature (°C)')
plt.title("Comprehensive Forecast Comparison: All Models")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(config.OUTPUT_DIR, "forecast_compare.png"), dpi=300)
print("\nSaved comprehensive forecast comparison to forecast_compare.png")

# ========== ALTERNATIVE VISUALIZATIONS ==========
print("\nGenerating alternative visualization options...")

# Side-by-Side Comparison (Traditional ML vs Deep Learning)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Traditional ML
ax1.plot(y_real[:150], label='Actual', color='black', linewidth=2.5, zorder=5)
ax1.plot(p_lr_real[:150], label=f'Linear Reg ({rmse_lr:.2f}°C)', linestyle=':', linewidth=2, alpha=0.8)
ax1.plot(p_xgb_real[:150], label=f'XGBoost ({rmse_xgb:.2f}°C)', linestyle='-.', linewidth=2, alpha=0.9)
ax1.set_xlabel('Time Step (Hours)', fontsize=11)
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.set_title('Traditional ML Models', fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Right: Deep Learning
ax2.plot(y_real[:150], label='Actual', color='black', linewidth=2.5, zorder=5)
ax2.plot(p_tf_real[:150], label=f'Transformer ({rmse_tf:.2f}°C)', linestyle='--', linewidth=2, alpha=0.8)
ax2.plot(p_fno_real[:150], label=f'FNO ({rmse_fno:.2f}°C)', linestyle='-', linewidth=2, alpha=0.9)
ax2.set_xlabel('Time Step (Hours)', fontsize=11)
ax2.set_ylabel('Temperature (°C)', fontsize=11)
ax2.set_title('Deep Learning Models', fontsize=13, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.suptitle('Side-by-Side Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "forecast_compare_sidebyside.png"), dpi=300)
print("  ✓ Saved Option 1: forecast_compare_sidebyside.png")



# 2x2 Grid (Each Model Individually)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

models = [
    ('Linear Regression', p_lr_real, rmse_lr, 'blue', ':'),
    ('XGBoost', p_xgb_real, rmse_xgb, 'green', '-.'),
    ('Transformer', p_tf_real, rmse_tf, 'orange', '--'),
    ('FNO', p_fno_real, rmse_fno, 'red', '-')
]

for idx, (name, preds, rmse, color, linestyle) in enumerate(models):
    ax = axes[idx // 2, idx % 2]

    ax.plot(y_real[:150], label='Actual', color='black', linewidth=2.5, zorder=5)
    ax.plot(preds[:150], label=f'{name} ({rmse:.2f}°C)',
            linestyle=linestyle, linewidth=2.5, alpha=0.9, color=color)

    ax.set_xlabel('Time Step (Hours)', fontsize=11)
    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_title(f'{name} vs Actual', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Individual Model Comparisons', fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "forecast_compare_grid.png"), dpi=300)
print("  ✓ Saved Option 3: forecast_compare_grid.png")

print("\nAll visualization options generated! Check outputs/ folder:")
print("  - forecast_compare.png (original, all models)")
print("  - forecast_compare_sidebyside.png (Option 1)")
print("  - forecast_compare_grid.png (Option 3)")

# 2. Robustness (Autoregressive Rollout with Delta)
print("Running 24-hr Autoregressive Rollout check...")
start_idx = 100
current_input, _ = test_ds[start_idx]
current_input = current_input.to(config.DEVICE).unsqueeze(0)
tf_rollout = []

# Get the starting absolute temp for the rollout
current_abs_temp = current_input[0, -1, 0].item()

with torch.no_grad():
    for i in range(24):
        # Predict Delta
        pred_diff = tf(current_input).item()
        
        # Apply Delta to get new Absolute Temp
        current_abs_temp = current_abs_temp + pred_diff
        tf_rollout.append(current_abs_temp)
        
        # Prepare Next Input
        next_real_input, _ = test_ds[start_idx + i + 1]
        new_row = next_real_input.to(config.DEVICE)[-1, :].clone()
        
        # We must inject the PREDICTED temp (not the real one) into the input
        # Note: current_abs_temp is scaled or unscaled? 
        # The model operates in Scaled space. So we keep it scaled.
        new_row[0] = current_abs_temp 
        
        current_input = torch.cat((current_input[:, 1:, :], new_row.view(1, 1, 5)), dim=1)

# Plot Rollout
plt.figure()
# Get real actuals for next 24 steps
_, real_deltas_seq = zip(*[test_ds[start_idx+i] for i in range(24)])
# We have to reconstruct the "Real" rollout series for plotting comparison
# This is tricky without the absolute values, so we approximate with y_real
actuals_rollout = y_real[start_idx:start_idx+24]

plt.plot(actuals_rollout, label='Actual')
plt.plot(np.array(tf_rollout) * STD + MEAN, label='Transformer Rollout')
plt.legend()
plt.title("24-Hour Rollout Stability (Delta Method)")
plt.savefig(os.path.join(config.OUTPUT_DIR, "rollout_check.png"))
print("Done.")

# ... [Previous imports and code remain the same] ...

# 2. Robustness (Autoregressive Rollout with Delta) - ALL 4 MODELS
print("Running 24-hr Autoregressive Rollout check (All Models)...")
start_idx = 100

# Initialize independent starting states for neural models
input_tf, _ = test_ds[start_idx]
input_tf = input_tf.to(config.DEVICE).unsqueeze(0)
input_fno = input_tf.clone()

# For traditional ML, we need to track the sequential window and flatten it
# Get initial tabular row from the processed data
initial_idx_in_df = np.where(X_test_tabular.index == test_dates[start_idx])[0][0]

# Initialize rollout storage
tf_rollout = []
fno_rollout = []
lr_rollout = []
xgb_rollout = []

# Get starting absolute temps (Scaled)
curr_temp_tf = input_tf[0, -1, 0].item()
curr_temp_fno = input_fno[0, -1, 0].item()
curr_temp_lr = X_test_tabular.iloc[initial_idx_in_df]['temp_2m_t-0']
curr_temp_xgb = X_test_tabular.iloc[initial_idx_in_df]['temp_2m_t-0']

# For traditional ML, maintain a sliding window of features
window_lr = X_test_tabular.iloc[initial_idx_in_df].copy()
window_xgb = X_test_tabular.iloc[initial_idx_in_df].copy()

with torch.no_grad():
    for i in range(24):
        # --- A. TRANSFORMER STEP ---
        pred_diff_tf = tf(input_tf).item()
        curr_temp_tf = curr_temp_tf + pred_diff_tf
        tf_rollout.append(curr_temp_tf)

        # --- B. FNO STEP ---
        pred_diff_fno = fno(input_fno).item()
        curr_temp_fno = curr_temp_fno + pred_diff_fno
        fno_rollout.append(curr_temp_fno)

        # --- C. LINEAR REGRESSION STEP ---
        pred_diff_lr = lr.predict(window_lr.values.reshape(1, -1))[0]
        curr_temp_lr = curr_temp_lr + pred_diff_lr
        lr_rollout.append(curr_temp_lr)

        # --- D. XGBOOST STEP ---
        pred_diff_xgb = xgb.predict(window_xgb.values.reshape(1, -1))[0]
        curr_temp_xgb = curr_temp_xgb + pred_diff_xgb
        xgb_rollout.append(curr_temp_xgb)

        # --- E. PREPARE NEXT INPUTS ---
        if i < 23:  # Don't need to prepare for the last iteration
            # Get the "Real" next weather features (Wind, Pressure, etc.) from dataset
            next_real_input, _ = test_ds[start_idx + i + 1]

            # Update neural model inputs
            new_row_tf = next_real_input.to(config.DEVICE)[-1, :].clone()
            new_row_tf[0] = curr_temp_tf

            new_row_fno = next_real_input.to(config.DEVICE)[-1, :].clone()
            new_row_fno[0] = curr_temp_fno

            input_tf = torch.cat((input_tf[:, 1:, :], new_row_tf.view(1, 1, 5)), dim=1)
            input_fno = torch.cat((input_fno[:, 1:, :], new_row_fno.view(1, 1, 5)), dim=1)

            # Update traditional ML windows (shift lags and inject predictions)
            # Get next real features from tabular data
            next_tabular_idx = initial_idx_in_df + i + 1
            if next_tabular_idx < len(X_test_tabular):
                next_features = X_test_tabular.iloc[next_tabular_idx].copy()

                # For Linear Regression: shift temperature lags, inject prediction
                for lag in range(config.SEQ_LEN - 1, 0, -1):
                    window_lr[f'temp_2m_t-{lag}'] = window_lr[f'temp_2m_t-{lag-1}']
                window_lr['temp_2m_t-0'] = curr_temp_lr
                # Update other features (wind, pressure, etc.) from real data
                for col in window_lr.index:
                    if not col.startswith('temp_2m_t-'):
                        window_lr[col] = next_features[col]

                # For XGBoost: same process
                for lag in range(config.SEQ_LEN - 1, 0, -1):
                    window_xgb[f'temp_2m_t-{lag}'] = window_xgb[f'temp_2m_t-{lag-1}']
                window_xgb['temp_2m_t-0'] = curr_temp_xgb
                for col in window_xgb.index:
                    if not col.startswith('temp_2m_t-'):
                        window_xgb[col] = next_features[col]

# Plot Rollout - ALL 4 MODELS
plt.figure(figsize=(12, 7))

# Get Ground Truth (Unscaled)
actuals_rollout = y_real[start_idx:start_idx+24]

# Unscale Predictions
tf_rollout_real = np.array(tf_rollout) * STD + MEAN
fno_rollout_real = np.array(fno_rollout) * STD + MEAN
lr_rollout_real = np.array(lr_rollout) * STD + MEAN
xgb_rollout_real = np.array(xgb_rollout) * STD + MEAN

plt.plot(actuals_rollout, label='Actual Ground Truth', color='black', linewidth=3, zorder=5)
plt.plot(lr_rollout_real, label='Linear Regression', linestyle=':', linewidth=2, alpha=0.8, color='blue')
plt.plot(xgb_rollout_real, label='XGBoost', linestyle='-.', linewidth=2, alpha=0.9, color='green')
plt.plot(tf_rollout_real, label='Transformer', linestyle='--', linewidth=2, alpha=0.8, color='orange')
plt.plot(fno_rollout_real, label='FNO', linestyle='-', linewidth=2, alpha=0.9, color='red')

plt.legend(loc='best', fontsize=10)
plt.title(f"24-Hour Autoregressive Stability: All Models\n(Start: {config.TEST_START_DATE} + {start_idx}h)", fontsize=13, fontweight='bold')
plt.xlabel("Hours into Future", fontsize=11)
plt.ylabel("Temperature (°C)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "rollout_compare_all.png"), dpi=300)
print("Saved rollout_compare_all.png (All 4 models)")

# ========== DIRECTIONAL ACCURACY ANALYSIS (Classification Metrics) ==========
print("\n--- Directional Accuracy: Treating Regression as Classification ---")
print("Goal: Assess if models predict the correct TREND (up/down) even if magnitude is off.\n")

def compute_directional_metrics(predictions, actuals, current_temps, model_name="Model"):
    """
    Compute precision, recall, and accuracy for trend prediction.
    Binary classification: 1 if temp went up, 0 if temp went down.
    """
    # Calculate actual and predicted changes
    actual_changes = actuals - current_temps
    pred_changes = predictions - current_temps

    # Create binary labels: 1 = up, 0 = down/flat
    # Use a small threshold to handle noise around 0
    threshold = 0.01  # 0.01°C threshold

    y_true = (actual_changes > threshold).astype(int)
    y_pred = (pred_changes > threshold).astype(int)

    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Additional stats
    correct_direction = np.sum(y_true == y_pred)
    total = len(y_true)

    print(f"\n{model_name} Directional Performance:")
    print("="*60)
    print(f"  Accuracy:  {accuracy:.4f} ({correct_direction}/{total} correct trends)")
    print(f"  Precision: {precision:.4f} (of predicted increases, % actually increased)")
    print(f"  Recall:    {recall:.4f} (of actual increases, % we predicted)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted Down  Predicted Up")
    print(f"  Actual Down        {cm[0,0]:6d}        {cm[0,1]:6d}")
    print(f"  Actual Up          {cm[1,0]:6d}        {cm[1,1]:6d}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Get current temperatures (t-0) from test set for both models
current_temps_test = []
with torch.no_grad():
    for X, _ in test_loader:
        X = X.to(config.DEVICE)
        current_temp = X[:, -1, 0].cpu().numpy().flatten()
        current_temps_test.extend(current_temp)

current_temps_test = np.array(current_temps_test) * STD + MEAN

# Compute directional metrics for all models
metrics_lr = compute_directional_metrics(p_lr_real, y_real, current_temps_test, "Linear Regression")
metrics_xgb = compute_directional_metrics(p_xgb_real, y_real, current_temps_test, "XGBoost")
metrics_tf = compute_directional_metrics(p_tf_real, y_real, current_temps_test, "Transformer")
metrics_fno = compute_directional_metrics(p_fno_real, y_real, current_temps_test, "FNO")

print("\n" + "="*60)
print("DIRECTIONAL ACCURACY SUMMARY:")
print("="*60)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score'}")
print("-"*60)
print(f"{'Linear Regression':<20} {metrics_lr['accuracy']:<12.4f} {metrics_lr['precision']:<12.4f} {metrics_lr['recall']:<12.4f} {metrics_lr['f1']:.4f}")
print(f"{'XGBoost':<20} {metrics_xgb['accuracy']:<12.4f} {metrics_xgb['precision']:<12.4f} {metrics_xgb['recall']:<12.4f} {metrics_xgb['f1']:.4f}")
print(f"{'Transformer':<20} {metrics_tf['accuracy']:<12.4f} {metrics_tf['precision']:<12.4f} {metrics_tf['recall']:<12.4f} {metrics_tf['f1']:.4f}")
print(f"{'FNO':<20} {metrics_fno['accuracy']:<12.4f} {metrics_fno['precision']:<12.4f} {metrics_fno['recall']:<12.4f} {metrics_fno['f1']:.4f}")
print("="*60)
print("\nKEY INSIGHT:")
print("Even when magnitude errors are large (RMSE), directional accuracy shows")
print("whether the model understands TRENDS - critical for decision-making.")
print("High precision = few false alarms. High recall = catches most increases.")
print("="*60)