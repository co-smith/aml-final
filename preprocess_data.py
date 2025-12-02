
import pandas as pd
import xarray as xr
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler

print("Starting preprocessing for NYC weather forecasting model...")

# 1. Load Data
file_pattern = 'data/nyc_era5_data_q*.nc'
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"ERROR: No files found matching pattern '{file_pattern}'")
    exit()

ds = xr.open_mfdataset(file_list, combine='by_coords')
ds_mean = ds.mean(dim=['latitude', 'longitude'])
df = ds_mean.to_dataframe()

# 2. Rename and Basic Convert
try:
    df = df.rename(columns={
        't2m': 'temp_2m',
        'd2m': 'dewpoint_2m',
        'u10': 'wind_u_10m',
        'v10': 'wind_v_10m',
        'sp': 'surface_pressure'
    })
    
    # Convert Kelvin to Celsius
    df['temp_2m'] = df['temp_2m'] - 273.15
    df['dewpoint_2m'] = df['dewpoint_2m'] - 273.15
    
    # Select features
    df = df[['temp_2m', 'dewpoint_2m', 'wind_u_10m', 'wind_v_10m', 'surface_pressure']]

except KeyError:
    print("Error renaming columns.")
    exit()

# --- THE FIX: STANDARDIZE DATA ---
print("Applying Standard Scaling (Mean=0, Std=1)...")
scaler = StandardScaler()
# We scale ALL columns so the Transformer treats them equally
df[df.columns] = scaler.fit_transform(df[df.columns])

# Important: We need to save the scaler mean/std for Temperature 
# so we can un-scale the predictions later for the plot!
temp_mean = scaler.mean_[0] # Assuming temp_2m is first column
temp_std = scaler.scale_[0]
print(f"Temperature Mean: {temp_mean:.2f}, Std: {temp_std:.2f}")

# 3. Create Lags (Feature Engineering)
print("Engineering features (6-hour lookback window)...")
df_out = pd.DataFrame(index=df.index)
n_lookback = 6 

for col in df.columns:
    for i in range(n_lookback):
        df_out[f'{col}_t-{i}'] = df[col].shift(i)

# Target is t+1
df_out['target_temp_t+1'] = df['temp_2m'].shift(-1)
df_out = df_out.dropna()

# Save scaler info to a small text file or print it clearly
# so we can hardcode it in the plotting script
with open("scaler_params.txt", "w") as f:
    f.write(f"{temp_mean},{temp_std}")

output_file = 'processed_nyc_weather_data.csv'
df_out.to_csv(output_file)

print(f"Preprocessing complete. Data standardized.")