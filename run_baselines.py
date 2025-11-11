import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("Establishing baseline performance for 1-hour temp forecast...")

df = pd.read_csv('processed_nyc_weather_data.csv', index_col=0, parse_dates=True)

print(f"Loaded {len(df)} rows of processed data.")

# Define the feature matrix (X) and the target vector (y)
target_col = 'target_temp_t+1' # The target is the temperature at the next timestep (t+1)
y = df[target_col]
X = df.drop(columns=[target_col]) # Features are all columns *except* the target

# Create a temporal train/test split (no shuffling!)
# This is crucial for time-series forecasting.
X_train = X.loc[X.index < '2023-12-01'] # Use data before Dec 2023 for training
y_train = y.loc[y.index < '2023-12-01']
X_test = X.loc[X.index >= '2023-12-01'] # Hold out Dec 2023 for testing
y_test = y.loc[y.index >= '2023-12-01']

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Baseline 1: Persistence (Naive) Model
# Assumes the best forecast for t+1 is the value at t (which is 'temp_2m_t-0')
# Any complex model must beat this.
y_pred_persistence = X_test['temp_2m_t-0']
rmse_persistence = np.sqrt(mean_squared_error(y_test, y_pred_persistence))

# Baseline 2: Linear Regression
# A simple model using the 6-hour lookback window as features.
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Report RMSE for both baselines in a Markdown format
# This is for easy copy/paste into the project report.
print("\n--- Baseline Model Performance ---")
print("| Model | RMSE (1-hr forecast) |")
print("| :--- | :--- |")
print(f"| Persistence Baseline (T+1 = T) | {rmse_persistence:.2f}°C |")
print(f"| Linear Regression (6-hr window) | {rmse_lr:.2f}°C |")