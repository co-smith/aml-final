import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class WeatherDataset(Dataset):
    def __init__(self, csv_file, mode='train', split_date='2023-12-01'):
        """
        Args:
            csv_file (string): Path to the processed csv file.
            mode (string): 'train' or 'test'.
            split_date (string): Date to split train/test.
        """
        # Load raw dataframe
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        
        # Split based on date
        if mode == 'train':
            df = df[df.index < split_date]
        else:
            df = df[df.index >= split_date]
            
        self.y = torch.tensor(df['target_temp_t+1'].values, dtype=torch.float32).unsqueeze(1)
        
        # Remove target to get features
        X_df = df.drop(columns=['target_temp_t+1'])
        
        # RECONSTRUCT SEQUENCE from flattened columns
        # Our cols are like: temp_t-0, temp_t-1... 
        # We need shape (N, 6, 5) -> (Batch, Lookback, Features)
        
        # 1. Identify base variables
        # We know we have 5 variables and 6 lags (0 to 5)
        self.seq_len = 6
        self.num_features = 5
        
        # Hardcoded for robustness based on preprocess_data.py
        # We want order: t-5, t-4, t-3, t-2, t-1, t-0 (Chronological)
        feature_names = ['temp_2m', 'dewpoint_2m', 'wind_u_10m', 'wind_v_10m', 'surface_pressure']
        
        # Create a container for the 3D data
        num_samples = len(X_df)
        self.X = np.zeros((num_samples, self.seq_len, self.num_features))
        
        for t in range(self.seq_len):
            # t=0 in loop corresponds to lag t-5 (furthest back) if we want chronological
            # But the CSV has t-0 as "now" and t-5 as "5 hours ago".
            # Let's order it Chronologically: [t-5, t-4, ..., t-0]
            lag_idx = (self.seq_len - 1) - t 
            
            for feat_idx, feat_name in enumerate(feature_names):
                col_name = f"{feat_name}_t-{lag_idx}"
                self.X[:, t, feat_idx] = X_df[col_name].values
                
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]