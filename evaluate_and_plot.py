import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    # Load Transformer
    tf = TimeSeriesTransformer(input_dim=5, d_model=64).to(DEVICE)
    tf.load_state_dict(torch.load("Transformer_best.pth", map_location=DEVICE))
    tf.eval()
    
    # Load FNO
    fno = SimpleFNO(input_dim=5, d_model=64).to(DEVICE)
    fno.load_state_dict(torch.load("FNO_best.pth", map_location=DEVICE))
    fno.eval()
    
    return tf, fno

def get_predictions(model, loader):
    preds = []
    actuals = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            output = model(X)
            preds.extend(output.cpu().numpy().flatten())
            actuals.extend(y.numpy().flatten())
    return np.array(preds), np.array(actuals)

def load_scaler_params():
    """
    Tries to read the mean and std from the file saved by preprocess_data.py.
    If not found, returns generic NYC approximation.
    """
    filename = "scaler_params.txt"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read().strip().split(',')
            # Assuming format: mean,std
            return float(content[0]), float(content[1])
    else:
        print(f"WARNING: {filename} not found. Using approximate un-scaling params.")
        # Approx stats for NYC Temp (Celsius)
        return 12.0, 10.0

if __name__ == "__main__":
    print("Evaluating models on Test Set (Dec 2023)...")
    
    # 1. Load Scaler Params to convert back to Celsius
    TEMP_MEAN, TEMP_STD = load_scaler_params()
    print(f"Un-scaling using -> Mean: {TEMP_MEAN:.2f}, Std: {TEMP_STD:.2f}")

    # 2. Load Data
    test_dataset = WeatherDataset('processed_nyc_weather_data.csv', mode='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3. Load Models
    tf_model, fno_model = load_models()
    
    # 4. Get Scaled Predictions (Z-Scores)
    pred_tf_scaled, y_true_scaled = get_predictions(tf_model, test_loader)
    pred_fno_scaled, _ = get_predictions(fno_model, test_loader)
    
    # 5. UN-SCALE to Real Celsius
    # Formula: Real = (Scaled * Std) + Mean
    y_true_real = (y_true_scaled * TEMP_STD) + TEMP_MEAN
    pred_tf_real = (pred_tf_scaled * TEMP_STD) + TEMP_MEAN
    pred_fno_real = (pred_fno_scaled * TEMP_STD) + TEMP_MEAN
    
    # 6. Calculate Real Metrics
    rmse_tf = np.sqrt(mean_squared_error(y_true_real, pred_tf_real))
    mae_tf = mean_absolute_error(y_true_real, pred_tf_real)
    
    rmse_fno = np.sqrt(mean_squared_error(y_true_real, pred_fno_real))
    mae_fno = mean_absolute_error(y_true_real, pred_fno_real)
    
    print("\n========================================")
    print("   FINAL RESULTS (Degrees Celsius)")
    print("========================================")
    print(f"Transformer | RMSE: {rmse_tf:.4f}°C | MAE: {mae_tf:.4f}°C")
    print(f"FNO         | RMSE: {rmse_fno:.4f}°C | MAE: {mae_fno:.4f}°C")
    print("========================================")
    
    # --- PLOTTING ---
    
    # Plot 1: Forecast Zoom (First 150 Hours)
    # We zoom in to see the daily cycle clearly
    plt.figure(figsize=(14, 7))
    plt.plot(y_true_real[:150], label='Actual Temp', color='black', linewidth=2, alpha=0.8)
    plt.plot(pred_tf_real[:150], label=f'Transformer (RMSE={rmse_tf:.2f})', linestyle='--', color='blue', alpha=0.8)
    plt.plot(pred_fno_real[:150], label=f'FNO (RMSE={rmse_fno:.2f})', linestyle='-.', color='orange', alpha=0.8)
    
    plt.title("NYC Temperature Forecast: Deep Learning Comparison")
    plt.xlabel("Hours into Test Set (Dec 2023)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("forecast_comparison_celsius.png")
    print("Saved plot: forecast_comparison_celsius.png")
    
    # Plot 2: Scatter Plot (Actual vs Predicted)
    # Good for seeing if models struggle at extremes (high/low temps)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_real, pred_tf_real, alpha=0.3, s=10, label='Transformer', color='blue')
    plt.scatter(y_true_real, pred_fno_real, alpha=0.3, s=10, label='FNO', color='orange')
    
    # Perfect prediction line
    min_val = min(y_true_real.min(), pred_tf_real.min())
    max_val = max(y_true_real.max(), pred_tf_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    plt.title("Actual vs Predicted Temperature")
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scatter_plot_celsius.png")
    print("Saved plot: scatter_plot_celsius.png")