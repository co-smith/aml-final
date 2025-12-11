import os
# Fix threading issues that cause segfaults
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import config
import utils
from xgboost import XGBRegressor
import joblib

print("--- Step 3: Baselines & Assumptions ---")

# Load Data directly for sklearn (bypassing the PyTorch loader for speed)
print("Loading data...")
df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
print(f"Data loaded: {df.shape}")

y = df['target_temp_t+1']
X = df.drop(columns=['target_temp_t+1'])
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Splits
print("Creating train/test splits...")
X_train = X.loc[X.index < config.TEST_START_DATE]
y_train = y.loc[y.index < config.TEST_START_DATE]
X_test = X.loc[X.index >= config.TEST_START_DATE]
y_test = y.loc[y.index >= config.TEST_START_DATE]
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 1. Linear Regression
print("Fitting Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression fitted successfully!")
y_pred_lr = lr.predict(X_test)

# 2. Persistence (Naive)
y_pred_persist = X_test['temp_2m_t-0']

# 3. XGBoost
xgb = XGBRegressor(
    n_estimators=100,      # Number of boosting rounds
    max_depth=6,           # Maximum tree depth
    learning_rate=0.1,     # Step size shrinkage
    subsample=0.8,         # Fraction of samples per tree
    colsample_bytree=0.8,  # Fraction of features per tree
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Save trained models for later comparison
joblib.dump(lr, os.path.join(config.CHECKPOINT_DIR, "linear_regression.pkl"))
joblib.dump(xgb, os.path.join(config.CHECKPOINT_DIR, "xgboost.pkl"))
print(f"\nSaved Linear Regression to {config.CHECKPOINT_DIR}/linear_regression.pkl")
print(f"Saved XGBoost to {config.CHECKPOINT_DIR}/xgboost.pkl")

# Unscale metrics
MEAN, STD = utils.load_scaler_params()
rmse_lr = np.sqrt(mean_squared_error(y_test * STD + MEAN, y_pred_lr * STD + MEAN))
rmse_per = np.sqrt(mean_squared_error(y_test * STD + MEAN, y_pred_persist * STD + MEAN))
rmse_xgb = np.sqrt(mean_squared_error(y_test * STD + MEAN, y_pred_xgb * STD + MEAN))

print(f"Baseline Results (Unscaled °C):")
print(f"Persistence RMSE: {rmse_per:.2f}")
print(f"Linear Reg RMSE:  {rmse_lr:.2f}")
print(f"XGBoost RMSE:     {rmse_xgb:.2f}")

# 4. Plot Residuals (Assumption Check)
residuals = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals, alpha=0.3)
plt.axhline(0, color='r', linestyle='--')
plt.title("Linear Regression: Residuals vs Predicted")
plt.savefig(os.path.join(config.OUTPUT_DIR, "baseline_residuals.png"))
print("Saved baseline_residuals.png")

# XGBoost Residuals
residuals_xgb = y_test - y_pred_xgb
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_xgb, residuals_xgb, alpha=0.3, color='green')
plt.axhline(0, color='r', linestyle='--')
plt.title("XGBoost: Residuals vs Predicted")
plt.xlabel("Predicted Temperature Delta (scaled)")
plt.ylabel("Residuals (scaled)")
plt.savefig(os.path.join(config.OUTPUT_DIR, "xgboost_residuals.png"))
print("Saved xgboost_residuals.png")