import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

print("Running baseline models and creating plots...")

# 1. Load processed data
try:
    df = pd.read_csv('processed_nyc_weather_data.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    print("ERROR: 'processed_nyc_weather_data.csv' not found.")
    print("Please run 'preprocess_data.py' first.")
    exit()

# 2. Define Features (X) and Target (y)
target_col = 'target_temp_t+1'
y = df[target_col]
X = df.drop(columns=[target_col])

# 3. Split data by time (Train: Oct/Nov, Test: Dec)
X_train = X.loc[X.index < '2023-12-01']
y_train = y.loc[y.index < '2023-12-01']
X_test = X.loc[X.index >= '2023-12-01']
y_test = y.loc[y.index >= '2023-12-01']

# 4. Train the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# 5. Create a results DataFrame for plotting
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_lr,
    'Error': y_test - y_pred_lr
})

# --- Plot 1: Actual vs. Predicted (First 5 Days) ---
print("Generating Plot 1: Actual vs. Predicted...")
plot_limit = '2023-12-05'
plt.figure(figsize=(14, 7))
plt.plot(results_df.loc[:plot_limit].index, results_df.loc[:plot_limit, 'Actual'], label='Actual Temperature', color='blue')
plt.plot(results_df.loc[:plot_limit].index, results_df.loc[:plot_limit, 'Predicted'], label='Linear Regression Prediction', color='red', linestyle='--')
plt.title('Actual vs. Predicted Temperature (First 5 Days of Dec 2023)', fontsize=16)
plt.ylabel('Temperature (°C)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png')
print("Saved 'predictions_vs_actual.png'")

# --- Plot 2: Histogram of Errors ---
print("Generating Plot 2: Error Distribution...")
plt.figure(figsize=(10, 6))
plt.hist(results_df['Error'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(results_df['Error'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {results_df["Error"].mean():.2f}°C')
plt.title('Distribution of Prediction Errors (Error = Actual - Predicted)', fontsize=16)
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('error_histogram.png')
print("Saved 'error_histogram.png'")

print("\nDone. You can now add these .png files to your report.")