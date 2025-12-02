import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def autoregressive_rollout(model, dataset, start_idx, steps=12):
    """
    Predicts 'steps' hours into the future using the model's own predictions
    for temperature, but using actual ground truth for exogenous vars (wind, pressure).
    """
    model.eval()
    predictions = []
    actuals = []
    
    # Get the initial input sequence (6 hours history)
    # Shape: (Seq_Len, Features)
    current_input, _ = dataset[start_idx] 
    current_input = current_input.to(DEVICE).unsqueeze(0) # Add batch dim -> (1, 6, 5)
    
    # We need the FUTURE ground truth for exogenous variables to feed in
    # (In real life we'd need a separate wind model, but this tests Temp capability)
    
    with torch.no_grad():
        for i in range(steps):
            # 1. Predict T+1
            pred_temp = model(current_input) # Output shape (1, 1)
            pred_val = pred_temp.item()
            predictions.append(pred_val)
            
            # 2. Get the ACTUAL T+1 value for ground truth comparison
            _, actual_target = dataset[start_idx + i]
            actuals.append(actual_target.item())
            
            # 3. Prepare input for T+2
            # We need to shift the window and append the new prediction
            # But we also need the exogenous vars (Dew, Wind, Pressure) for the next step.
            # We grab them from the dataset at start_idx + i + 1 (the next timestep)
            
            # Get next real frame to steal exogenous vars from
            next_real_input, _ = dataset[start_idx + i + 1]
            next_real_input = next_real_input.to(DEVICE)
            
            # Create the new row: [Pred_Temp, Real_Dew, Real_WindU, Real_WindV, Real_Press]
            # Assuming Temp is at index 0
            new_row = next_real_input[-1, :].clone() # Get the newest data point
            new_row[0] = pred_val # REPLACE actual temp with PREDICTED temp
            
            # Roll the window: Drop first, append new row
            current_input = torch.cat((current_input[:, 1:, :], new_row.view(1, 1, 5)), dim=1)
            
    return predictions, actuals

if __name__ == "__main__":
    print("--- 1. Model Efficiency Check ---")
    
    # Load Models
    tf_model = TimeSeriesTransformer(input_dim=5, d_model=64).to(DEVICE)
    fno_model = SimpleFNO(input_dim=5, d_model=64).to(DEVICE)
    
    # Count Params
    tf_params = count_parameters(tf_model)
    fno_params = count_parameters(fno_model)
    
    print(f"Transformer Parameters: {tf_params:,}")
    print(f"FNO Parameters:         {fno_params:,}")
    if fno_params < tf_params:
        print(f"-> FNO is {tf_params/fno_params:.1f}x smaller.")
    
    print("\n--- 2. Autoregressive Rollout (12-Hour Forecast) ---")
    
    # Load Scaler to plot real values
    try:
        with open("scaler_params.txt", "r") as f:
            c = f.read().strip().split(',')
            MEAN, STD = float(c[0]), float(c[1])
    except:
        MEAN, STD = 12.0, 10.0

    test_dataset = WeatherDataset('processed_nyc_weather_data.csv', mode='test')
    
    # Load Weights
    tf_model.load_state_dict(torch.load("Transformer_best.pth", map_location=DEVICE))
    fno_model.load_state_dict(torch.load("FNO_best.pth", map_location=DEVICE))
    
    # Run rollout on a specific challenging start point (e.g., index 100)
    start_idx = 100 
    steps = 24
    
    tf_preds, tf_actuals = autoregressive_rollout(tf_model, test_dataset, start_idx, steps)
    fno_preds, _ = autoregressive_rollout(fno_model, test_dataset, start_idx, steps)
    
    # Unscale
    tf_preds = np.array(tf_preds) * STD + MEAN
    fno_preds = np.array(fno_preds) * STD + MEAN
    actuals = np.array(tf_actuals) * STD + MEAN
    
    # Calculate RMSE over the rollout
    tf_rollout_rmse = np.sqrt(np.mean((tf_preds - actuals)**2))
    fno_rollout_rmse = np.sqrt(np.mean((fno_preds - actuals)**2))
    
    print(f"Transformer 24-hr Rollout RMSE: {tf_rollout_rmse:.2f}")
    print(f"FNO 24-hr Rollout RMSE:         {fno_rollout_rmse:.2f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, steps+1), actuals, label='Actual', color='black', marker='o')
    plt.plot(range(1, steps+1), tf_preds, label=f'Transformer (Err: {tf_rollout_rmse:.2f})', linestyle='--')
    plt.plot(range(1, steps+1), fno_preds, label=f'FNO (Err: {fno_rollout_rmse:.2f})', linestyle='-.')
    plt.title("24-Hour Autoregressive Forecast Stability")
    plt.xlabel("Hours into Future")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rollout_comparison.png")
    print("Saved rollout_comparison.png")