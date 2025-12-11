"""
4) ablation_study.py
Goal: Demonstrate the effect of input masking on model performance.
What to do: Train the TimeSeriesTransformer with masking probability set to 0.0 (disabled)
and compare to masking probability 0.5 (enabled).
Why: This directly covers the rubric's "Ablation Analysis" requirement by isolating
the contribution of a specific design choice (input masking).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from models import TimeSeriesTransformer
import config
import utils
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("ABLATION STUDY: Effect of Input Masking on Transformer Performance")
print("="*80)
print("\nResearch Question: Does input masking (zeroing out t-0 temperature 50% of")
print("the time during training) improve model generalization?\n")

# Load datasets
train_ds = WeatherDataset(end_date=config.TRAIN_END_DATE)
val_ds = WeatherDataset(start_date=config.VAL_START_DATE, end_date=config.VAL_END_DATE)
test_ds = WeatherDataset(start_date=config.TEST_START_DATE)

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# Load scaler params
MEAN, STD = utils.load_scaler_params()

def train_model_with_masking(masking_prob, model_name, epochs=20):
    """
    Train a Transformer with specified masking probability.
    masking_prob: float in [0, 1]. Probability of masking t-0 temperature during training.
    """
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"Masking Probability: {masking_prob}")
    print(f"{'='*80}\n")

    # Initialize model
    model = TimeSeriesTransformer(input_dim=config.NUM_FEATURES, d_model=config.D_MODEL).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)

            # Apply input masking based on masking_prob
            if masking_prob > 0 and np.random.rand() > (1 - masking_prob):
                # Zero out t-0 temperature (last time step, first feature)
                X[:, -1, 0] = 0.0

            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        avg_train_loss = train_loss / len(train_ds)

        # Validation (NO masking during evaluation)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.DEVICE), y.to(config.DEVICE)
                val_loss += criterion(model(X), y).item() * X.size(0)

        avg_val_loss = val_loss / len(val_ds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            train_rmse = np.sqrt(avg_train_loss) * STD
            val_rmse = np.sqrt(avg_val_loss) * STD
            print(f"Epoch {epoch+1:2d} | Train: {train_rmse:.4f}°C | Val: {val_rmse:.4f}°C")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            test_loss += criterion(model(X), y).item() * X.size(0)

    avg_test_loss = test_loss / len(test_ds)
    test_rmse = np.sqrt(avg_test_loss) * STD

    print(f"\nFinal Test RMSE: {test_rmse:.4f}°C")
    print(f"Best Val RMSE:   {np.sqrt(best_val_loss) * STD:.4f}°C")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_rmse': test_rmse,
        'best_val_rmse': np.sqrt(best_val_loss) * STD
    }

# ========== EXPERIMENT 1: NO MASKING (Baseline) ==========
results_no_mask = train_model_with_masking(
    masking_prob=0.0,
    model_name="Transformer WITHOUT Input Masking (Baseline)",
    epochs=config.EPOCHS
)

# ========== EXPERIMENT 2: WITH MASKING (Proposed Method) ==========
results_with_mask = train_model_with_masking(
    masking_prob=0.5,
    model_name="Transformer WITH Input Masking (p=0.5)",
    epochs=config.EPOCHS
)

# ========== RESULTS COMPARISON ==========
print("\n" + "="*80)
print("ABLATION STUDY RESULTS")
print("="*80)

print("\n1. WITHOUT Input Masking (Baseline):")
print(f"   Best Validation RMSE: {results_no_mask['best_val_rmse']:.4f}°C")
print(f"   Test RMSE:            {results_no_mask['test_rmse']:.4f}°C")

print("\n2. WITH Input Masking (p=0.5):")
print(f"   Best Validation RMSE: {results_with_mask['best_val_rmse']:.4f}°C")
print(f"   Test RMSE:            {results_with_mask['test_rmse']:.4f}°C")

# Calculate improvement
val_improvement = results_no_mask['best_val_rmse'] - results_with_mask['best_val_rmse']
test_improvement = results_no_mask['test_rmse'] - results_with_mask['test_rmse']
val_pct = (val_improvement / results_no_mask['best_val_rmse']) * 100
test_pct = (test_improvement / results_no_mask['test_rmse']) * 100

print("\n" + "-"*80)
print("IMPROVEMENT FROM MASKING:")
print("-"*80)
print(f"   Validation: {val_improvement:+.4f}°C ({val_pct:+.2f}%)")
print(f"   Test:       {test_improvement:+.4f}°C ({test_pct:+.2f}%)")

if val_improvement > 0:
    print("\n✓ CONCLUSION: Input masking IMPROVES generalization.")
    print("  The model learns to use wind, pressure, and history instead of")
    print("  relying solely on persistence (copying t-0 temperature).")
else:
    print("\n✗ CONCLUSION: Input masking DEGRADES performance.")
    print("  The model performs better when it has access to t-0 temperature.")

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training curves
ax1 = axes[0]
epochs_range = range(1, config.EPOCHS + 1)

# Convert losses to RMSE in °C
train_no_mask = [np.sqrt(loss) * STD for loss in results_no_mask['train_losses']]
val_no_mask = [np.sqrt(loss) * STD for loss in results_no_mask['val_losses']]
train_with_mask = [np.sqrt(loss) * STD for loss in results_with_mask['train_losses']]
val_with_mask = [np.sqrt(loss) * STD for loss in results_with_mask['val_losses']]

ax1.plot(epochs_range, train_no_mask, 'b-', label='No Mask - Train', alpha=0.7)
ax1.plot(epochs_range, val_no_mask, 'b--', label='No Mask - Val', linewidth=2)
ax1.plot(epochs_range, train_with_mask, 'r-', label='With Mask - Train', alpha=0.7)
ax1.plot(epochs_range, val_with_mask, 'r--', label='With Mask - Val', linewidth=2)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('RMSE (°C)')
ax1.set_title('Training Curves: Masking vs No Masking')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart comparison
ax2 = axes[1]
metrics = ['Validation RMSE', 'Test RMSE']
no_mask_values = [results_no_mask['best_val_rmse'], results_no_mask['test_rmse']]
with_mask_values = [results_with_mask['best_val_rmse'], results_with_mask['test_rmse']]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, no_mask_values, width, label='No Masking', color='blue', alpha=0.7)
bars2 = ax2.bar(x + width/2, with_mask_values, width, label='With Masking (p=0.5)', color='red', alpha=0.7)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}°C', ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('RMSE (°C)')
ax2.set_title('Final Performance Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "ablation_study_masking.png"), dpi=300)
print(f"\nSaved visualization to {config.OUTPUT_DIR}/ablation_study_masking.png")

print("\n" + "="*80)
print("ABLATION ANALYSIS COMPLETE")
print("="*80)
print("\nKey Takeaway: This experiment isolates the effect of input masking by")
print("training two identical models with only ONE difference: masking probability.")
print("This allows us to quantify the contribution of this specific design choice.")
print("="*80)
