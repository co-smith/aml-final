import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("Running baseline model and generating assumption-checking plots...")

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
# We check assumptions on the *test* set to see how it generalizes.
X_train = X.loc[X.index < '2023-12-01']
y_train = y.loc[y.index < '2023-12-01']
X_test = X.loc[X.index >= '2023-12-01']
y_test = y.loc[y.index >= '2023-12-01']

# 4. Train the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# 5. Calculate errors (residuals)
errors = y_test - y_pred_lr

# --- Plot 1: Residuals vs. Predicted Values ---
# This checks for homoscedasticity (constant variance)
print("Generating Plot 1: Residuals vs. Predicted Plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, errors, alpha=0.3)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs. Predicted Values (Test Set)', fontsize=16)
plt.xlabel('Predicted Temperature (°C)')
plt.ylabel('Residuals (Error)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('residuals_vs_predicted.png')
print("Saved 'residuals_vs_predicted.png'")

# --- Plot 2: Histogram of Errors ---
# This checks for normality of errors
print("Generating Plot 2: Error Distribution...")
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
plt.title('Distribution of Prediction Errors (Test Set)', fontsize=16)
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Density')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('error_distribution_histogram.png')
print("Saved 'error_distribution_histogram.png'")

print("\nDone. These two plots are the standard diagnostics for a linear regression report.")