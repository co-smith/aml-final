import pandas as pd
import xarray as xr
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
import config
import os

print("--- Step 1: Preprocessing (Delta Strategy) ---")

# 1. Load NetCDF
file_list = glob.glob(config.DATA_DIR + "/*.nc")
if not file_list:
    print(f"ERROR: No .nc files found in {config.DATA_DIR}")
    exit()

ds = xr.open_mfdataset(file_list, combine='by_coords')
df = ds.mean(dim=['latitude', 'longitude']).to_dataframe()

# 2. Rename & Convert
df = df.rename(columns={'t2m': 'temp_2m', 'd2m': 'dewpoint_2m', 'u10': 'wind_u_10m', 'v10': 'wind_v_10m', 'sp': 'surface_pressure'})
df['temp_2m'] -= 273.15
df['dewpoint_2m'] -= 273.15
df = df[['temp_2m', 'dewpoint_2m', 'wind_u_10m', 'wind_v_10m', 'surface_pressure']]

# 3. Standardize
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Save scaler params
temp_mean = scaler.mean_[0]
temp_std = scaler.scale_[0]
with open(config.SCALER_PARAMS_PATH, "w") as f:
    f.write(f"{temp_mean},{temp_std}")
print(f"Saved scaler params: Mean={temp_mean:.2f}, Std={temp_std:.2f}")

# 4. Lags
df_out = pd.DataFrame(index=df.index)
for col in df.columns:
    for i in range(config.SEQ_LEN):
        df_out[f'{col}_t-{i}'] = df[col].shift(i)

# --- THE FIX: PREDICT DELTA (CHANGE), NOT RAW VALUE ---
# Target = Temp(t+1) - Temp(t-0)
# Since data is standardized, this is "Change in Standard Deviations"
df_out['target_temp_t+1'] = df['temp_2m'].shift(-1) - df['temp_2m']
df_out = df_out.dropna()

df_out.to_csv(config.PROCESSED_DATA_PATH)
print(f"Saved processed data (Delta Targets) to {config.PROCESSED_DATA_PATH}")