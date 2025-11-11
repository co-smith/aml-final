import pandas as pd
import xarray as xr
import numpy as np
import glob  # For file pattern matching

print("Starting preprocessing for NYC weather forecasting model...")

# Discover all quarterly ERA5 data files
# Glob pattern to find all 'nyc_era5_data_q*.nc' files
file_pattern = 'nyc_era5_data_q*.nc'
file_list = glob.glob(file_pattern)

if not file_list:
    print(f"ERROR: No files found matching pattern '{file_pattern}'")
    print("Please make sure your quarterly files (e.g., 'nyc_era5_data_q1.nc') are in this directory.")
    exit()

print(f"Found {len(file_list)} files to merge:")
for f in sorted(file_list): # Sorting ensures correct time order, though open_mfdataset often handles this
    print(f"  - {f}")

# Lazily load and merge files into a single xarray Dataset
# open_mfdataset is ideal here; it combines by coords (primarily 'time')
ds = xr.open_mfdataset(file_list, combine='by_coords')

# Aggregate spatial dimensions (lat/lon)
# We want a single time series for the NYC grid, not a spatial map.
ds_mean = ds.mean(dim=['latitude', 'longitude'])

# Convert from xarray Dataset to pandas DataFrame for easier manipulation
df = ds_mean.to_dataframe()

# Clean, normalize, and select features
try:
    # Rename variables to be more descriptive/Pythonic
    df = df.rename(columns={
        't2m': 'temp_2m',
        'd2m': 'dewpoint_2m',
        'u10': 'wind_u_10m',
        'v10': 'wind_v_10m',
        'sp': 'surface_pressure'
    })
    
    # Convert temperatures from Kelvin (standard for ERA5) to Celsius
    df['temp_2m'] = df['temp_2m'] - 273.15
    df['dewpoint_2m'] = df['dewpoint_2m'] - 273.15
    print("\nConverted temperatures from Kelvin to Celsius.")
    
    # Select and re-order final features
    df = df[['temp_2m', 'dewpoint_2m', 'wind_u_10m', 'wind_v_10m', 'surface_pressure']]

except KeyError:
    print("Error renaming columns. The variables in the NetCDF might be different.")
    print("Available variables:", list(df.columns))
    exit()


# Feature Engineering: Create time-series lookback features
print("Engineering features (6-hour lookback window)...")
df_out = pd.DataFrame(index=df.index)
n_lookback = 6 # Using a 6-hour lookback window

# Create shifted features (e.g., temp_t-1, temp_t-2, ...) for all variables
for col in df.columns:
    for i in range(n_lookback):
        df_out[f'{col}_t-{i}'] = df[col].shift(i)

# Define the target variable: 2m temp at t+1
df_out['target_temp_t+1'] = df['temp_2m'].shift(-1)

# Drop rows with NaN values introduced by the shift/lookback operations
df_out = df_out.dropna()

# Save the final, model-ready dataset
output_file = 'processed_nyc_weather_data.csv'
df_out.to_csv(output_file)

print(f"\nPreprocessing complete.")
print(f"Model-ready dataset with {len(df_out)} samples saved to '{output_file}'")