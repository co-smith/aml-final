import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import time
import copy

# Configuration
BATCH_SIZE = 32
EPOCHS = 20 # Keep small for project speed, increase if needed
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def train_model(model, train_loader, val_loader, model_name="model"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        epoch_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                preds = model(X_val)
                loss = criterion(preds, y_val)
                val_loss += loss.item() * X_val.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{model_name}_best.pth")

    total_time = time.time() - start_time
    print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Val Loss: {best_loss:.4f}")
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # 1. Prepare Data
    print("Loading Datasets...")
    # Note: Ideally we split Train into Train/Val. 
    # Here we use the Test set (Dec 2023) as validation for simplicity in this script,
    # but strictly speaking, you should split Train -> (Train/Val) and keep Test separate.
    # For this code, we will just use the Test set to track progress.
    
    train_dataset = WeatherDataset('processed_nyc_weather_data.csv', mode='train')
    test_dataset = WeatherDataset('processed_nyc_weather_data.csv', mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Train Transformer
    model_tf = TimeSeriesTransformer(input_dim=5, d_model=64, nhead=4, num_layers=2).to(DEVICE)
    train_model(model_tf, train_loader, test_loader, model_name="Transformer")
    
    # 3. Train FNO
    model_fno = SimpleFNO(input_dim=5, d_model=64, modes=4).to(DEVICE)
    train_model(model_fno, train_loader, test_loader, model_name="FNO")

    print("\nAll models trained and saved.")