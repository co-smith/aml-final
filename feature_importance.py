"""
Feature Importance Analysis for Traditional ML Models

Unlike neural network saliency maps (which use gradients), traditional ML models
have built-in feature importance metrics:
- Linear Regression: Coefficient magnitudes
- XGBoost: Information gain from tree splits

This script visualizes which features each traditional ML model relies on most.
"""

import os
# Fix threading issues that cause segfaults on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import config

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS: Traditional ML Models")
print("="*80)

# Load trained models
print("\nLoading models...")
lr = joblib.load(os.path.join(config.CHECKPOINT_DIR, "linear_regression.pkl"))
xgb = joblib.load(os.path.join(config.CHECKPOINT_DIR, "xgboost.pkl"))

# Load feature names from processed data
df = pd.read_csv(config.PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
feature_names = [col for col in df.columns if col != 'target_temp_t+1']

print(f"Number of features: {len(feature_names)}")

# ========== LINEAR REGRESSION FEATURE IMPORTANCE ==========
print("\n" + "="*80)
print("LINEAR REGRESSION: Coefficient Analysis")
print("="*80)

# Get coefficients
lr_coeffs = lr.coef_

# Create DataFrame for easier manipulation
lr_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_coeffs,
    'abs_coefficient': np.abs(lr_coeffs)
}).sort_values('abs_coefficient', ascending=False)

print("\nTop 10 Most Important Features (by absolute coefficient):")
print(lr_importance.head(10).to_string(index=False))

# Visualize - Top 20 features
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Bar chart of top 20 absolute coefficients
top20_lr = lr_importance.head(20)
colors = ['red' if c < 0 else 'green' for c in top20_lr['coefficient']]
ax1.barh(range(len(top20_lr)), top20_lr['abs_coefficient'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top20_lr)))
ax1.set_yticklabels(top20_lr['feature'], fontsize=9)
ax1.set_xlabel('Absolute Coefficient Value', fontsize=11)
ax1.set_title('Linear Regression: Top 20 Features by Importance', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Positive (increases temp)'),
    Patch(facecolor='red', alpha=0.7, label='Negative (decreases temp)')
]
ax1.legend(handles=legend_elements, loc='lower right')

# Right: Heatmap of temperature coefficients across lags
temp_features = [f for f in feature_names if 'temp_2m_t-' in f]
temp_coeffs = lr_importance[lr_importance['feature'].isin(temp_features)].sort_values('feature')

# Extract lag numbers
lags = [int(f.split('t-')[1]) for f in temp_coeffs['feature']]
coeffs = temp_coeffs['coefficient'].values

ax2.bar(range(len(lags)), coeffs, color=['red' if c < 0 else 'green' for c in coeffs], alpha=0.8)
ax2.set_xticks(range(len(lags)))
ax2.set_xticklabels([f't-{lag}' for lag in lags], fontsize=10)
ax2.set_xlabel('Time Lag', fontsize=11)
ax2.set_ylabel('Coefficient', fontsize=11)
ax2.set_title('Linear Regression: Temperature Coefficients by Lag', fontsize=13, fontweight='bold')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "feature_importance_linear_regression.png"), dpi=300)
print(f"\n✓ Saved: feature_importance_linear_regression.png")

# ========== XGBOOST FEATURE IMPORTANCE ==========
print("\n" + "="*80)
print("XGBOOST: Information Gain Analysis")
print("="*80)

# Get feature importances (based on information gain from tree splits)
xgb_importances = xgb.feature_importances_

# Create DataFrame
xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (by information gain):")
print(xgb_importance.head(10).to_string(index=False))

# Visualize - Top 20 features
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Bar chart of top 20 importances
top20_xgb = xgb_importance.head(20)
ax1.barh(range(len(top20_xgb)), top20_xgb['importance'], color='green', alpha=0.7)
ax1.set_yticks(range(len(top20_xgb)))
ax1.set_yticklabels(top20_xgb['feature'], fontsize=9)
ax1.set_xlabel('Feature Importance (Information Gain)', fontsize=11)
ax1.set_title('XGBoost: Top 20 Features by Importance', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Right: Temperature importance across lags
temp_features_xgb = xgb_importance[xgb_importance['feature'].str.contains('temp_2m_t-')].sort_values('feature')
lags_xgb = [int(f.split('t-')[1]) for f in temp_features_xgb['feature']]
importances_xgb = temp_features_xgb['importance'].values

ax2.bar(range(len(lags_xgb)), importances_xgb, color='green', alpha=0.8)
ax2.set_xticks(range(len(lags_xgb)))
ax2.set_xticklabels([f't-{lag}' for lag in lags_xgb], fontsize=10)
ax2.set_xlabel('Time Lag', fontsize=11)
ax2.set_ylabel('Importance', fontsize=11)
ax2.set_title('XGBoost: Temperature Feature Importance by Lag', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "feature_importance_xgboost.png"), dpi=300)
print(f"✓ Saved: feature_importance_xgboost.png")

# ========== COMPARISON PLOT ==========
print("\n" + "="*80)
print("COMPARISON: Linear Regression vs XGBoost (Temperature Features)")
print("="*80)

fig, ax = plt.subplots(figsize=(12, 6))

# Get temperature features for both models
temp_lr = lr_importance[lr_importance['feature'].str.contains('temp_2m_t-')].sort_values('feature')
temp_xgb = xgb_importance[xgb_importance['feature'].str.contains('temp_2m_t-')].sort_values('feature')

lags = [int(f.split('t-')[1]) for f in temp_lr['feature']]
x = np.arange(len(lags))
width = 0.35

# Normalize for comparison (both on 0-1 scale)
lr_norm = temp_lr['abs_coefficient'].values / temp_lr['abs_coefficient'].max()
xgb_norm = temp_xgb['importance'].values / temp_xgb['importance'].max()

bars1 = ax.bar(x - width/2, lr_norm, width, label='Linear Regression (Coeff)', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, xgb_norm, width, label='XGBoost (Info Gain)', color='green', alpha=0.7)

ax.set_xlabel('Time Lag', fontsize=11)
ax.set_ylabel('Normalized Importance', fontsize=11)
ax.set_title('Feature Importance Comparison: Temperature Across Time Lags', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f't-{lag}' for lag in lags], fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "feature_importance_comparison.png"), dpi=300)
print(f"✓ Saved: feature_importance_comparison.png")

# ========== SUMMARY ==========
print("\n" + "="*80)
print("SUMMARY & KEY INSIGHTS")
print("="*80)

print("\nLinear Regression Insights:")
print(f"  - Most important feature: {lr_importance.iloc[0]['feature']}")
print(f"  - Coefficient magnitude: {lr_importance.iloc[0]['abs_coefficient']:.4f}")
print(f"  - Sign: {'Positive' if lr_importance.iloc[0]['coefficient'] > 0 else 'Negative'}")

print("\nXGBoost Insights:")
print(f"  - Most important feature: {xgb_importance.iloc[0]['feature']}")
print(f"  - Importance score: {xgb_importance.iloc[0]['importance']:.4f}")

print("\nKey Differences:")
print("  - Linear Reg coefficients show LINEAR relationships (sign matters)")
print("  - XGBoost importance shows NON-LINEAR importance (always positive)")
print("  - Both models can identify which features drive predictions")

print("\n" + "="*80)
print("All feature importance visualizations saved to outputs/")
print("="*80)
