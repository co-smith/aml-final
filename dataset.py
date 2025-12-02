import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import config

class WeatherDataset(Dataset):
    def __init__(self, start_date=None, end_date=None):
        
        df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        
        # Apply Date Filters
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index < end_date]
            
        self.y = torch.tensor(df['target_temp_t+1'].values, dtype=torch.float32).unsqueeze(1)
        X_df = df.drop(columns=['target_temp_t+1'])
        
        # Reconstruct (N, Seq_Len, Features)
        self.seq_len = config.SEQ_LEN
        self.num_features = config.NUM_FEATURES
        feature_names = ['temp_2m', 'dewpoint_2m', 'wind_u_10m', 'wind_v_10m', 'surface_pressure']
        
        self.X = np.zeros((len(X_df), self.seq_len, self.num_features))
        
        for t in range(self.seq_len):
            lag_idx = (self.seq_len - 1) - t 
            for feat_idx, feat_name in enumerate(feature_names):
                self.X[:, t, feat_idx] = X_df[f"{feat_name}_t-{lag_idx}"].values
                
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]