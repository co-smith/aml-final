import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import config
import utils
import os

print("--- Step 3: Baselines & Assumptions ---")

# Load Data directly for sklearn (bypassing the PyTorch loader for speed)
df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
y = df['target_temp_t+1']
X = df.drop(columns=['target_temp_t+1'])

# Splits
X_train = X.loc[X.index < config.TEST_START_DATE]
y_train = y.loc[y.index < config.TEST_START_DATE]
X_test = X.loc[X.index >= config.TEST_START_DATE]
y_test = y.loc[y.index >= config.TEST_START_DATE]

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 2. Persistence (Naive)
y_pred_persist = X_test['temp_2m_t-0']

# Unscale metrics
MEAN, STD = utils.load_scaler_params()
rmse_lr = np.sqrt(mean_squared_error(y_test * STD + MEAN, y_pred_lr * STD + MEAN))
rmse_per = np.sqrt(mean_squared_error(y_test * STD + MEAN, y_pred_persist * STD + MEAN))

print(f"Baseline Results (Unscaled °C):")
print(f"Persistence RMSE: {rmse_per:.2f}")
print(f"Linear Reg RMSE:  {rmse_lr:.2f}")

# 3. Plot Residuals (Assumption Check)
residuals = y_test - y_pred_lr
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals, alpha=0.3)
plt.axhline(0, color='r', linestyle='--')
plt.title("Linear Regression: Residuals vs Predicted")
plt.savefig(os.path.join(config.OUTPUT_DIR, "baseline_residuals.png"))
print("Saved baseline_residuals.png")