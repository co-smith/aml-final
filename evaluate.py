import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import config
import utils
import os

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

# 1. Standard Forecast
p_tf, y_true = get_reconstructed_preds(tf, test_loader)
p_fno, _ = get_reconstructed_preds(fno, test_loader)

# Unscale (Apply Mean/Std to the Reconstructed Values)
p_tf_real = p_tf * STD + MEAN
p_fno_real = p_fno * STD + MEAN
y_real = y_true * STD + MEAN

# Metrics
rmse_tf = np.sqrt(np.mean((p_tf_real - y_real)**2))
rmse_fno = np.sqrt(np.mean((p_fno_real - y_real)**2))
print(f"Transformer RMSE: {rmse_tf:.2f}°C")
print(f"FNO RMSE:         {rmse_fno:.2f}°C")

# Plot Forecast
plt.figure(figsize=(12, 6))
plt.plot(y_real[:150], label='Actual', color='k')
plt.plot(p_tf_real[:150], label=f'Transformer ({rmse_tf:.2f})', linestyle='--')
plt.plot(p_fno_real[:150], label=f'FNO ({rmse_fno:.2f})', linestyle='-.')
plt.legend()
plt.title("Forecast Comparison (Reconstructed from Delta)")
plt.savefig(os.path.join(config.OUTPUT_DIR, "forecast_compare.png"))

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

# 2. Robustness (Autoregressive Rollout with Delta)
print("Running 24-hr Autoregressive Rollout check (Transformer vs FNO)...")
start_idx = 100

# Initialize independent starting states for both models
# We clone them so modifications to one don't affect the other
input_tf, _ = test_ds[start_idx]
input_tf = input_tf.to(config.DEVICE).unsqueeze(0)
input_fno = input_tf.clone()

tf_rollout = []
fno_rollout = []

# Get starting absolute temp (Scaled)
curr_temp_tf = input_tf[0, -1, 0].item()
curr_temp_fno = input_fno[0, -1, 0].item()

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

        # --- C. PREPARE NEXT INPUTS ---
        # Get the "Real" next weather features (Wind, Pressure, etc.) from dataset
        # We need the row at t+1. 
        # Note: We use the ground truth for auxiliary features (Wind/Pressure)
        # because we are not forecasting those, only Temperature.
        next_real_input, _ = test_ds[start_idx + i + 1]
        
        # Create next row for Transformer (injecting TF prediction)
        new_row_tf = next_real_input.to(config.DEVICE)[-1, :].clone()
        new_row_tf[0] = curr_temp_tf # <--- Inject TF's specific hallucination
        
        # Create next row for FNO (injecting FNO prediction)
        new_row_fno = next_real_input.to(config.DEVICE)[-1, :].clone()
        new_row_fno[0] = curr_temp_fno # <--- Inject FNO's specific hallucination

        # Update sliding windows
        input_tf = torch.cat((input_tf[:, 1:, :], new_row_tf.view(1, 1, 5)), dim=1)
        input_fno = torch.cat((input_fno[:, 1:, :], new_row_fno.view(1, 1, 5)), dim=1)

# Plot Rollout
plt.figure(figsize=(10, 6))

# Get Ground Truth (Unscaled)
actuals_rollout = y_real[start_idx:start_idx+24]

# Unscale Predictions
tf_rollout_real = np.array(tf_rollout) * STD + MEAN
fno_rollout_real = np.array(fno_rollout) * STD + MEAN

plt.plot(actuals_rollout, label='Actual Ground Truth', color='black', linewidth=2)
plt.plot(tf_rollout_real, label='Transformer Rollout', linestyle='--', color='blue')
plt.plot(fno_rollout_real, label='FNO Rollout', linestyle='-.', color='green')

plt.legend()
plt.title(f"24-Hour Autoregressive Stability Comparison\n(Start Date: {config.TEST_START_DATE} + {start_idx}h)")
plt.xlabel("Hours into Future")
plt.ylabel("Temperature (°C)")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(config.OUTPUT_DIR, "rollout_compare_dual.png"))
print("Saved rollout_compare_dual.png")