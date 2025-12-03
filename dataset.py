import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import config

class WeatherDataset(Dataset):
    def __init__(self, start_date=None, end_date=None):
        
        # Load data
        # Optimization: use float32 during load to save memory
        df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        
        # Filter Dates
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index < end_date]
            
        # Extract Targets
        self.y = torch.tensor(df['target_temp_t+1'].values, dtype=torch.float32).unsqueeze(1)
        
        # Extract Features
        # The dataframe columns are flat: [T_t-5...T_t-0, D_t-5...D_t-0, ...]
        X_df = df.drop(columns=['target_temp_t+1'])
        
        # ---------------------------------------------------------
        # VECTORIZED RESHAPING (Removes the slow for-loops)
        # ---------------------------------------------------------
        # Current shape: (N_samples, Num_Features * Seq_Len)
        # We want:       (N_samples, Seq_Len, Num_Features)
        
        raw_values = X_df.values.astype(np.float32)
        N = raw_values.shape[0]
        F = config.NUM_FEATURES
        L = config.SEQ_LEN
        
        # Reshape to (N, Features, Seq_Len) first because of how we generated columns
        # (All lags for Feat 1 are contiguous, then all lags for Feat 2)
        X_reshaped = raw_values.reshape(N, F, L)
        
        # Permute to get (N, Seq_Len, Features) which is what Transformers expect
        X_reshaped = np.transpose(X_reshaped, (0, 2, 1))
        
        self.X = torch.from_numpy(X_reshaped)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]