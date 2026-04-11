#!/usr/bin/env python3
"""
Step 3: Train XGBoost v6 WITH FIGURES — Non-Overlapping Sources Classifier
===========================================================================
Identical to train_xgboost_v6.py, but WITH figure generation during training.

Trains a single XGBoost multi-class classifier on 1.18M real-world arrays.

KEY DIFFERENCE FROM v5:
  • v5 uses stratified split (can leak same source into train & test)
  • v6 uses GroupShuffleSplit by source_id (ensures NO source overlap)
  → This is the "Non-Overlapping Sources" evaluation

Balance strategy (3-pronged):
  1. Undersample majority: cap timsort at ~3× the minority class count
  2. sample_weight: inverse-frequency weighting on the undersampled set
  3. eval_metric: mlogloss (proper multi-class metric)

Split: 70% train / 15% val / 15% test (group-aware, by source_id)

DIFFERENCES FROM train_xgboost_v6.py:
  ✅ Adds matplotlib imports for figure generation
  ✅ Tracks metrics per epoch
  ✅ Generates ROC curves
  ✅ Generates PR curves
  ✅ Generates learning curves
  ✅ Outputs to results/xgboost_v6_with_figures/ (doesn't overwrite v6)

Inputs:  data/training_dataset.csv  (from Step 2)
Outputs: models/xgboost_v6_with_figures/        (model JSON)
         results/xgboost_v6_with_figures/       (metrics + FIGURES)

Usage:
    python3 scripts/train_xgboost_v6_with_figures.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "diverse_training_data.csv"
MODEL_DIR   = ROOT / "models" / "xgboost_v6_with_figures"
RESULTS_DIR = ROOT / "results" / "xgboost_v6_with_figures"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42

# ── XGBoost hyperparameters ──────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.1,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
)


# ── Balance the dataset ──────────────────────────────────────────────────

def balanced_undersample(df: pd.DataFrame, label_col: str,
                         max_ratio: float = 3.0) -> pd.DataFrame:
    """
    Undersample majority classes so no class exceeds max_ratio × minority count.
    """
    counts = df[label_col].value_counts()
    min_count = counts.min()
    cap = int(min_count * max_ratio)

    parts = []
    for cls in counts.index:
        subset = df[df[label_col] == cls]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=SEED)
        parts.append(subset)

    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# ── Figure generation during training ─────────────────────────────────────

class TrainingTracker:
    """Track metrics during training for figure generation."""
    
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epochs = []
    
    def add_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Record metrics for one epoch."""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc * 100)  # Convert to percentage
        self.val_acc.append(val_acc * 100)
    
    def plot_training_curves(self, output_path):
        """Generate training curve figures."""
        print(f"\n[FIG] Generating training curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        ax1.plot(self.epochs, self.train_loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
        ax1.plot(self.epochs, self.val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss (mlogloss)', fontsize=11, fontweight='bold')
        ax1.set_title('Training History: Loss Curves', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.epochs, self.train_acc, label='Training Accuracy', linewidth=2, marker='o', markersize=3)
        ax2.plot(self.epochs, self.val_acc, label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Training History: Accuracy Curves', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        output_file = output_path / "01_training_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_file}")


def generate_roc_curves(y_true, y_proba, output_path):
    """Generate ROC curves for each class."""
    print(f"[FIG] Generating ROC curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    classes = np.unique(y_true)
    for i, class_label in enumerate(classes):
        y_binary = (y_true == class_label).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{ALGORITHMS[class_label]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('ROC Curves (Test Set)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / "02_roc_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_pr_curves(y_true, y_proba, output_path):
    """Generate Precision-Recall curves for each class."""
    print(f"[FIGerating Precision-Recall curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    classes = np.unique(y_true)
    for i, class_label in enumerate(classes):
        y_binary = (y_true == class_label).astype(int)
        precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, linewidth=2, label=f'{ALGORITHMS[class_label]} (AP = {pr_auc:.3f})')
    
    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Precision-Recall Curves (Test Set)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_file = output_path / "03_precision_recall_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_tsne_plot(X, y, output_path):
    """Generate t-SNE visualization of features."""
    print(f"[FIG] Generating t-SNE plot...")
    
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, algo in enumerate(ALGORITHMS):
        mask = y == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[i], label=algo, alpha=0.6, s=30)
    
    ax.set_xlabel('t-SNE 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=11, fontweight='bold')
    ax.set_title('t-SNE: Feature Space Visualization (Test Set)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / "04_tsne_features.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_feature_distributions(X, y, feature_names, output_path):
    """Generate box plots and violin plots of feature distributions."""
    print(f"[FIG] Generating feature distribution plots...")
    
    # Select top 6 most important features (for readability)
    top_features_idx = list(range(min(6, len(feature_names))))
    top_features = [feature_names[i] for i in top_features_idx]
    X_top = X[:, top_features_idx]
    
    # Create box plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (feat_idx, feat_name) in enumerate(zip(top_features_idx, top_features)):
        data_by_class = [X_top[y == i, idx] for i in range(len(ALGORITHMS))]
        
        bp = axes[idx].boxplot(data_by_class, labels=ALGORITHMS, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[idx].set_ylabel('Value', fontsize=10, fontweight='bold')
        axes[idx].set_title(feat_name, fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_path / "05_feature_distributions_boxplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")
    
    # Create violin plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (feat_idx, feat_name) in enumerate(zip(top_features_idx, top_features)):
        data_by_class = [X_top[y == i, idx] for i in range(len(ALGORITHMS))]
        
        parts = axes[idx].violinplot(data_by_class, positions=range(len(ALGORITHMS)), 
                                      showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#1f77b4')
            pc.set_alpha(0.6)
        
        axes[idx].set_xticks(range(len(ALGORITHMS)))
        axes[idx].set_xticklabels(ALGORITHMS)
        axes[idx].set_ylabel('Value', fontsize=10, fontweight='bold')
        axes[idx].set_title(feat_name, fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_path / "06_feature_distributions_violin.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_confusion_matrix(y_true, y_pred, output_path):
    """Generate confusion matrix heatmap."""
    print(f"[FIG] Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ALGORITHMS, 
                yticklabels=ALGORITHMS, cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_path / "07_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_feature_importance(model, feature_names, output_path):
    """Generate feature importance bar plot."""
    print(f"[FIG] Generating feature importance plot...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_pos = np.arange(len(indices))
    ax.bar(x_pos, importances[indices], color='#1f77b4', alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Feature Importance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_path / "08_feature_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_training_history(model, output_path):
    """Generate training history plot from eval_result."""
    print(f"[FIG] Generating training history plot...")
    
    # Get evaluation results from model
    results = model.evals_result()
    
    # Extract metrics for train and validation
    train_loss = results['validation_0']['mlogloss']
    val_loss = results['validation_1']['mlogloss']
    epochs = list(range(len(train_loss)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (mlogloss)', fontsize=12, fontweight='bold')
    ax.set_title('Training History: Loss per Epoch', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / "01_training_history_loss.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def generate_training_accuracy_history(X_train, y_train_enc, X_val, y_val_enc, model, output_path, le):
    """Generate training and validation loss + accuracy curves per epoch."""
    print(f"[FIG] Generating training history plots...")
    
    results = model.evals_result()
    
    # Get loss curves from XGBoost
    train_loss = results['validation_0']['mlogloss']
    val_loss = results['validation_1']['mlogloss']
    epochs = list(range(len(train_loss)))
    
    # Compute accuracy at sampled epochs (every 5 epochs for speed)
    sample_every = max(1, len(epochs) // 100)  # At most 100 points
    sampled_epochs = list(range(0, len(epochs), sample_every))
    
    train_accs = []
    val_accs = []
    for ep in sampled_epochs:
        if ep == 0:
            # Epoch 0 accuracy
            y_train_pred = model.predict(X_train, iteration_range=(0, 1))
            y_val_pred = model.predict(X_val, iteration_range=(0, 1))
        else:
            y_train_pred = model.predict(X_train, iteration_range=(0, ep + 1))
            y_val_pred = model.predict(X_val, iteration_range=(0, ep + 1))
        
        train_accs.append(accuracy_score(y_train_enc, y_train_pred) * 100)
        val_accs.append(accuracy_score(y_val_enc, y_val_pred) * 100)
    
    sampled_epochs_list = [epochs[i] for i in sampled_epochs if i < len(epochs)]
    
    # Create dual-axis plot: loss (left) and accuracy (right)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot loss on primary axis
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2.5, alpha=0.7, color='#1f77b4')
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2.5, alpha=0.7, color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (mlogloss)', fontsize=12, fontweight='bold', color='#555')
    ax1.tick_params(axis='y', labelcolor='#555')
    ax1.grid(True, alpha=0.2)
    
    # Plot accuracy on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(sampled_epochs_list, train_accs, label='Train Accuracy', linewidth=2.5, alpha=0.7, 
             color='#2ca02c', linestyle='--', marker='o', markersize=4)
    ax2.plot(sampled_epochs_list, val_accs, label='Val Accuracy', linewidth=2.5, alpha=0.7, 
             color='#d62728', linestyle='--', marker='s', markersize=4)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#555')
    ax2.tick_params(axis='y', labelcolor='#555')
    ax2.set_ylim([0, 100])
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    ax1.set_title('Training History: Loss vs Accuracy per Epoch', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_file = output_path / "01b_training_accuracy.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("XGBOOST v6 WITH FIGURES — Training Non-Overlapping Sources Classifier")
    print("=" * 80)
    
    start_time = time.time()
    
    # Load
    print(f"\n[LOAD] Loading {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    print(f"✓ Loaded {len(df):,} rows")
    print(f"  Class distribution: {dict(df['best_algorithm'].value_counts())}")
    
    print(f"\n[PREPROCESS] Applying margin >= 5% filter...")
    margin_filtered = df[(df['timing_margin'] >= 0.05) | (df['n_elements'] >= 2000)].copy()
    print(f"✓ After margin filter: {len(margin_filtered):,} rows")
    
    print(f"[BALANCE] Undersampling with max_ratio=3.0...")
    df_balanced = balanced_undersample(margin_filtered, 'best_algorithm', max_ratio=3.0)
    print(f"✓ After undersampling: {len(df_balanced):,} rows")
    print(f"  Class distribution: {dict(df_balanced['best_algorithm'].value_counts())}")
    
    # Encode labels
    le = LabelEncoder()
    df_balanced['label'] = le.fit_transform(df_balanced['best_algorithm'])
    label_to_algo = {i: algo for i, algo in enumerate(le.classes_)}
    print(f"✓ Label encoding: {label_to_algo}")
    
    # Extract features and groups
    X = df_balanced[FEATURE_NAMES].values
    y = df_balanced['label'].values
    groups = df_balanced['source_id'].values
    print(f"✓ Features: {len(FEATURE_NAMES)} features, {X.shape[0]} samples")
    print(f"✓ Groups (source_id): {len(np.unique(groups))} unique sources")
    
    # Split: 70/15/15 (GROUP-aware by source_id to avoid source leakage)
    print(f"\n[SPLIT] Group-aware split by source_id (70/15/15)...")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    train_val_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val = groups[train_val_idx]
    
    # Further split train_val into train/val
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))
    
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    print(f"✓ Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    print(f"  Train sources: {len(np.unique(groups_train_val[train_idx]))}")
    print(f"  Val sources: {len(np.unique(groups_train_val[val_idx]))}")
    print(f"  Test sources: {len(np.unique(groups[test_idx]))}")
    
    # Compute sample weights (inverse frequency)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    
    # Train
    print(f"\n[TRAIN] Training XGBoost...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    print(f"✓ Training complete in {time.time() - start_time:.1f}s")
    
    # Evaluate on all splits
    print(f"\n[EVAL] Evaluating...")
    results = {}
    predictions = {}
    
    for split_name, X, y in [
        ('train', X_train, y_train),
        ('val', X_val, y_val),
        ('test', X_test, y_test),
    ]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        acc = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
        
        results[split_name] = {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0),
        }
        
        predictions[split_name] = {
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': y_proba.tolist(),
        }
        
        print(f"  {split_name.upper()}: acc={acc:.3f}, balanced_acc={balanced_acc:.3f}")
    
    # Generate figures
    print(f"\n[FIGURES] Generating training visualizations...")
    
    y_test_np = np.array(predictions['test']['y_true'])
    y_pred_test = np.array(predictions['test']['y_pred'])
    y_proba_test = np.array(predictions['test']['y_proba'])
    
    generate_training_history(model, FIGURES_DIR)
    generate_training_accuracy_history(X_train, y_train_enc, X_val, y_val_enc, model, FIGURES_DIR, le)
    generate_roc_curves(y_test_np, y_proba_test, FIGURES_DIR)
    generate_pr_curves(y_test_np, y_proba_test, FIGURES_DIR)
    generate_tsne_plot(X_test, y_test_np, FIGURES_DIR)
    generate_feature_distributions(X_test, y_test_np, FEATURE_NAMES, FIGURES_DIR)
    generate_confusion_matrix(y_test_np, y_pred_test, FIGURES_DIR)
    generate_feature_importance(model, FEATURE_NAMES, FIGURES_DIR)
    
    # Save model
    print(f"\n[SAVE] Saving model and results...")
    
    model_file = MODEL_DIR / "xgb_v6_with_figures.json"
    model.get_booster().save_model(str(model_file))
    print(f"✓ Model saved: {model_file}")
    
    # Save evaluation results
    results_file = RESULTS_DIR / "evaluation_results.json"
    output = {
        'timestamp': datetime.now().isoformat(),
        'xgb_params': XGB_PARAMS,
        'features': FEATURE_NAMES,
        'algorithms': ALGORITHMS,
        'split_method': 'GroupShuffleSplit by source_id (Non-Overlapping Sources)',
        'dataset': {
            'total_raw': len(df),
            'after_margin_filter': len(margin_filtered),
            'after_undersample': len(df_balanced),
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test),
        },
        'results': results,
        'feature_importance': [
            {'feature': name, 'importance': float(score)}
            for name, score in sorted(
                zip(FEATURE_NAMES, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
        ],
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Results saved: {results_file}")
    
    # Save predictions
    predictions_file = RESULTS_DIR / "predictions_test.csv"
    pred_df = pd.DataFrame({
        'predicted': predictions['test']['y_pred'],
        'actual': predictions['test']['y_true'],
    })
    pred_df.to_csv(predictions_file, index=False)
    print(f"✓ Test predictions saved: {predictions_file}")
    
    print(f"\n" + "=" * 80)
    print(f"✓ COMPLETE. Results in: {RESULTS_DIR}")
    print(f"✓ Figures in: {FIGURES_DIR}")
    print(f"=" * 80)


if __name__ == "__main__":
    main()
