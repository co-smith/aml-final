import torch
import os

# --- PATHS ---
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "nyc_era5_data_q4.nc")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_nyc_weather.csv")
SCALER_PARAMS_PATH = os.path.join(DATA_DIR, "scaler_params.txt")

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
SEQ_LEN = 6      # Lookback window
NUM_FEATURES = 5 # Temp, Dew, Wind U/V, Pressure
D_MODEL = 64     # Model internal dimension

# --- SPLITS ---
# Train: Start -> Nov 15
# Val:   Nov 15 -> Dec 01
# Test:  Dec 01 -> End
TRAIN_END_DATE = '2023-11-15' 
VAL_START_DATE = '2023-11-15'
VAL_END_DATE   = '2023-12-01'
TEST_START_DATE = '2023-12-01'

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")