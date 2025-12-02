import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from models import TimeSeriesTransformer, SimpleFNO
import config
import utils
import os

# Ensure seaborn is installed for heatmaps
# pip install seaborn

print("--- Step 5: Advanced Analysis ---")

# --- Helper 1: Extract Internal Representations (for CKA) ---
def extract_reps(model, loader, is_transformer=True):
    model.eval()
    reps = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(config.DEVICE)
            if is_transformer:
                # Transformer: Extract from the encoder memory (last time step)
                src = model.input_linear(X)
                src = model.pos_encoder(src)
                mem = model.transformer_encoder(src)
                reps.append(mem[:, -1, :].cpu().numpy())
            else:
                # FNO: Extract from the projection before the final head
                x = model.fc0(X).permute(0, 2, 1)
                x = model.act(model.conv0(x) + model.w0(x))
                x = model.act(model.conv1(x) + model.w1(x))
                # Take last time step
                reps.append(x.permute(0, 2, 1)[:, -1, :].cpu().numpy())
    return np.vstack(reps)

# --- Helper 2: Compute Saliency (Input Gradients) ---
def compute_global_saliency(model, loader):
    """
    Computes the average absolute gradient for every input feature.
    High gradient = High importance.
    """
    model.eval() # Eval mode, but we allow gradients for the input
    total_saliency = None
    count = 0
    
    # Feature names for plotting
    features = ['Temp', 'Dew', 'Wind U', 'Wind V', 'Pressure']
    lags = [f't-{i}' for i in range(config.SEQ_LEN)][::-1] # t-5, t-4 ... t-0
    
    print(f"Computing saliency for {model.__class__.__name__}...")
    
    for X, _ in loader:
        X = X.to(config.DEVICE)
        X.requires_grad_() # <--- Key step: track gradients on INPUT
        
        output = model(X)
        
        # We want the gradient of the prediction w.r.t the input
        # Summing output allows backward() on a batch
        output.sum().backward()
        
        # Saliency = |Gradient|
        saliency = X.grad.abs().data.cpu().numpy()
        
        if total_saliency is None:
            total_saliency = np.sum(saliency, axis=0)
        else:
            total_saliency += np.sum(saliency, axis=0)
            
        count += X.shape[0]
        
        # Zero grads for next batch (though we create new X each time, good practice)
        model.zero_grad()

    # Average over all samples
    avg_saliency = total_saliency / count
    return avg_saliency, features, lags

def plot_saliency(saliency_matrix, features, lags, title, filename):
    plt.figure(figsize=(8, 6))
    # Y-axis: Time (Lags), X-axis: Features
    # Note: saliency_matrix shape is (Seq_Len, Features)
    sns.heatmap(saliency_matrix, xticklabels=features, yticklabels=lags, cmap="viridis", annot=False)
    plt.title(title)
    plt.xlabel("Weather Features")
    plt.ylabel("Time Lag")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, filename))
    print(f"Saved {filename}")

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Load Data
# We use the Test set to see what features the model uses for unseen data
test_loader = DataLoader(WeatherDataset(start_date=config.TEST_START_DATE), batch_size=64, shuffle=False)

# 2. Load Models
tf = TimeSeriesTransformer(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
tf.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "Transformer_best.pth"), map_location=config.DEVICE))

fno = SimpleFNO(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
fno.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "FNO_best.pth"), map_location=config.DEVICE))

# ------------------------------------------
# ANALYSIS PART A: CKA Similarity
# ------------------------------------------
print("\n[Part A] Running CKA Representation Analysis...")
rep_tf = extract_reps(tf, test_loader, True)
rep_fno = extract_reps(fno, test_loader, False)

score = utils.cka_score(rep_tf, rep_fno)

print(f"---------------------------------")
print(f"CKA Similarity Score: {score:.4f}")
print(f"---------------------------------")

if score > 0.9:
    print(">> Interpretation: High similarity. Both models learned very similar feature representations.")
elif score < 0.5:
    print(">> Interpretation: Low similarity. The models fundamentally 'think' differently.")
else:
    print(">> Interpretation: Moderate similarity.")

# ------------------------------------------
# ANALYSIS PART B: Saliency Maps
# ------------------------------------------
print("\n[Part B] Generating Saliency Maps (Feature Importance)...")

# Transformer Saliency
sal_tf, feats, lags = compute_global_saliency(tf, test_loader)
plot_saliency(sal_tf, feats, lags, "Transformer Feature Importance", "saliency_transformer.png")

# FNO Saliency
sal_fno, _, _ = compute_global_saliency(fno, test_loader)
plot_saliency(sal_fno, feats, lags, "FNO Feature Importance", "saliency_fno.png")

print("\nDone. Check the 'outputs/' folder for heatmaps.")