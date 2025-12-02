import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import config
import copy
import os
import numpy as np

def train(model, name):
    print(f"\n--- Training {name} with Input Masking ---")
    
    train_ds = WeatherDataset(end_date=config.TRAIN_END_DATE)
    val_ds = WeatherDataset(start_date=config.VAL_START_DATE, end_date=config.VAL_END_DATE)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_loss = float('inf')
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            
            # --- THE FIX: INPUT DROPOUT (Masking t-0) ---
            # 50% chance to zero out the most recent temperature
            # This forces the model to look at Wind/Pressure and history
            if np.random.rand() > 0.5:
                # Shape: (Batch, Seq, Feat). Last seq (-1), First feat (0 is Temp)
                X[:, -1, 0] = 0.0 
            
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        # Validation (No masking, we want to see real performance)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.DEVICE), y.to(config.DEVICE)
                val_loss += criterion(model(X), y).item() * X.size(0)
                
        avg_val = val_loss / len(val_ds)
        print(f"Epoch {epoch+1} | Train: {train_loss/len(train_ds):.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            best_wts = copy.deepcopy(model.state_dict())
            
    save_path = os.path.join(config.CHECKPOINT_DIR, f"{name}_best.pth")
    torch.save(best_wts, save_path)
    print(f"Saved best {name} model to {save_path}")

if __name__ == "__main__":
    tf = TimeSeriesTransformer(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
    train(tf, "Transformer")
    
    fno = SimpleFNO(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
    train(fno, "FNO")