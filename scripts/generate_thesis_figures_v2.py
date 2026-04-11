#!/usr/bin/env python3
"""
Generate comprehensive thesis figures v2 with improved labeling.

Improvements from v1:
- Enhanced x-axis labels throughout
- Better font sizes and rotation for readability
- Academic naming (v6 = Non-Overlapping Sources)
- Formal language (the model, the selector, the framework)
- Better spacing and annotations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Setup
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "thesis_figures_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10

def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def load_csv(path):
    """Load CSV file."""
    return pd.read_csv(path)

def save_fig(name, tight_layout=True):
    """Save figure with standard settings."""
    if tight_layout:
        plt.tight_layout()
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()

# ============================================================================
# FIGURE 1: Model Version Comparison (v1 -> v6)
# ============================================================================
def fig_model_history():
    """Compare accuracy across all model versions."""
    print("\n[FIG 1] Model history comparison...")
    
    versions = {
        'v1\n(Regression)': {'test': 44.4, 'real': 57.9, 'domain': None},
        'v2\n(Baseline Classifier)': {'test': 62.5, 'real': 60.2, 'domain': None},
        'v3\n(Leaky)': {'test': 100.0, 'real': 94.5, 'domain': None},
        'v5\n(Production)': {'test': 76.1, 'real': 76.1, 'domain': 88.0},
        'v6\n(Non-Overlapping Sources)': {'test': 71.2, 'real': 71.2, 'domain': None},
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Test accuracy progression
    versions_list = list(versions.keys())
    test_accs = [versions[v]['test'] for v in versions_list]
    colors = ['#d62728' if v in ['v1', 'v2'] else '#2ca02c' if 'Leaky' in v else '#1f77b4' for v in versions_list]
    
    ax1.bar(range(len(versions_list)), test_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=50, color='gray', linestyle='--', label='Baseline (random)', linewidth=1)
    ax1.set_xticks(range(len(versions_list)))
    ax1.set_xticklabels(versions_list, rotation=0, ha='center', fontsize=10)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Evolution: Test Accuracy Across Versions', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_accs):
        ax1.text(i, v + 2, f'{v}%', ha='center', fontweight='bold', fontsize=10)
    
    # Plot 2: v5 vs v6 (production vs source-aware)
    models = ['v5\n(Production)', 'v6\n(Non-Overlapping Sources)']
    accuracies = [76.1, 71.2]
    bar_colors = ['#1f77b4', '#ff7f0e']
    
    ax2.bar(models, accuracies, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Production vs Non-Overlapping Sources Evaluation', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    for i, acc in enumerate(accuracies):
        ax2.text(i, acc + 1.5, f'{acc}%', ha='center', fontweight='bold', fontsize=11)
    
    save_fig("01_model_history")

# ============================================================================
# FIGURE 2: v5 Accuracy Metrics (Train/Val/Test)
# ============================================================================
def fig_v5_accuracy():
    """v5 accuracy across splits."""
    print("[FIG 2] v5 accuracy metrics...")
    
    v5_eval = load_json(RESULTS_DIR / "xgboost_v5" / "evaluation_results.json")
    
    splits = ['Training\nSet', 'Validation\nSet', 'Test\nSet']
    accuracies = [
        v5_eval['results']['train']['accuracy'] * 100,
        v5_eval['results']['val']['accuracy'] * 100,
        v5_eval['results']['test']['accuracy'] * 100,
    ]
    balanced_accs = [
        v5_eval['results']['train']['balanced_accuracy'] * 100,
        v5_eval['results']['val']['balanced_accuracy'] * 100,
        v5_eval['results']['test']['balanced_accuracy'] * 100,
    ]
    
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, balanced_accs, width, label='Balanced Accuracy', color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost v5: Accuracy Across Dataset Splits', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    save_fig("02_v5_accuracy_metrics")

# ============================================================================
# FIGURE 3: Per-Algorithm Recall (v5)
# ============================================================================
def fig_v5_recall():
    """Per-algorithm recall on test set."""
    print("[FIG 3] Per-algorithm recall...")
    
    v5_eval = load_json(RESULTS_DIR / "xgboost_v5" / "evaluation_results.json")
    
    algorithms = ['Introsort', 'Heapsort', 'Timsort']
    recalls = [
        v5_eval['results']['test']['classification_report']['introsort']['recall'] * 100,
        v5_eval['results']['test']['classification_report']['heapsort']['recall'] * 100,
        v5_eval['results']['test']['classification_report']['timsort']['recall'] * 100,
    ]
    precisions = [
        v5_eval['results']['test']['classification_report']['introsort']['precision'] * 100,
        v5_eval['results']['test']['classification_report']['heapsort']['precision'] * 100,
        v5_eval['results']['test']['classification_report']['timsort']['precision'] * 100,
    ]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, recalls, width, label='Recall', color='#2ca02c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, precisions, width, label='Precision', color='#d62728', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('v5 Test Set: Per-Algorithm Recall & Precision', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    save_fig("03_v5_per_algorithm_recall")

# ============================================================================
# FIGURE 4: Feature Importance (Top 16)
# ============================================================================
def fig_feature_importance():
    """Feature importance from v5."""
    print("[FIG 4] Feature importance...")
    
    v5_eval = load_json(RESULTS_DIR / "xgboost_v5" / "evaluation_results.json")
    
    importances_list = v5_eval['feature_importance']
    features = [item['feature'] for item in importances_list]
    values = [item['importance'] for item in importances_list]
    
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(range(len(features)))
    # Improve feature names for display
    feature_labels = [f.replace('_', ' ').title() for f in features]
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost v5: Feature Importance Ranking (All 16 Structural Features)', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels with better spacing
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.008, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    save_fig("04_feature_importance")

# ============================================================================
# FIGURE 5: Regret Analysis
# ============================================================================
def fig_regret_analysis():
    """Regret distribution from v5."""
    print("[FIG 5] Regret analysis...")
    
    regret = load_json(RESULTS_DIR / "xgboost_v5" / "regret_analysis.json")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Regret metrics summary
    metrics = ['Mean', 'Median', 'P95', 'P99', 'Maximum']
    values = [
        regret['per_instance_regret_us']['mean'],
        regret['per_instance_regret_us']['median'],
        regret['per_instance_regret_us']['p95'],
        regret['per_instance_regret_us']['p99'],
        regret['per_instance_regret_us']['max'],
    ]
    colors_regret = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    bars = ax1.bar(metrics, values, color=colors_regret, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Per-Instance Regret (μs)', fontsize=11, fontweight='bold')
    ax1.set_title('Regret Statistics on Test Set', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2., val + max(values)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Top-right: Algorithm timings
    algs = ['VBS\n(Oracle)', 'Model v5\n(Framework)', 'SBS\n(Heapsort)', 'Introsort', 'Timsort']
    times = [
        regret['vbs_total_s'],
        regret['model_total_s'],
        regret['sbs_total_s'],
        regret['algorithm_totals_s']['introsort'],
        regret['algorithm_totals_s']['timsort'],
    ]
    colors_time = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e', '#9467bd']
    bars = ax2.bar(algs, times, color=colors_time, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Total Runtime (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Algorithm Comparison on Full Dataset (1.18M Arrays)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2., val + max(times)*0.01,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Bottom-left: Gap closed and key metrics
    metrics_box = ['VBS-SBS\nGap', 'Model Regret\nvs VBS', 'Gap\nClosed', 'Perfect\nPicks']
    percentages = [
        regret['vbs_sbs_gap_pct'],
        regret['model_regret_vs_vbs_pct'],
        regret['gap_closed_pct'],
        regret['per_instance_regret_us']['zero_pct'],
    ]
    colors_box = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax3.bar(metrics_box, percentages, color=colors_box, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, percentages):
        ax3.text(bar.get_x() + bar.get_width()/2., val + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Bottom-right: Zero regret percentage
    zero_regret_pct = regret['per_instance_regret_us']['zero_pct']
    ax4.pie([zero_regret_pct, 100-zero_regret_pct], 
            labels=[f'Zero Regret\n{zero_regret_pct:.1f}%', f'With Regret\n{100-zero_regret_pct:.1f}%'],
            colors=['#2ca02c', '#ffcccc'], autopct='', startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=1.5), textprops=dict(fontsize=10, fontweight='bold'))
    ax4.set_title('Perfect Algorithm Selection Rate', fontsize=12, fontweight='bold')
    
    save_fig("05_regret_analysis", tight_layout=True)

# ============================================================================
# FIGURE 6: VBS vs SBS vs Model Comparison
# ============================================================================
def fig_vbs_sbs_model():
    """Compare VBS, SBS, and Model performance."""
    print("[FIG 6] VBS vs SBS vs Model...")
    
    regret = load_json(RESULTS_DIR / "xgboost_v5" / "regret_analysis.json")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    strategies = ['VBS\n(Oracle\nAlgorithm)', 'Model v5\n(Framework\nSelector)', 'SBS\n(Single\nAlgorithm)']
    times = [
        regret['vbs_total_s'],
        regret['model_total_s'],
        regret['sbs_total_s'],
    ]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    
    bars = ax.bar(strategies, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
    
    ax.set_ylabel('Total Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Comparison on Full Dataset (1.18M Arrays)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # Add time labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width()/2., time + max(times)*0.02,
               f'{time:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add annotations
    gap_vbs_sbs = regret['vbs_sbs_gap_pct']
    overhead_model_vbs = regret['model_regret_vs_vbs_pct']
    
    ax.text(0.5, 0.95, f'VBS-SBS Gap: {gap_vbs_sbs:.1f}%\nModel Overhead: {overhead_model_vbs:.1f}%',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    save_fig("06_vbs_sbs_model_comparison")

# ============================================================================
# FIGURE 7: Domain Holdout Analysis
# ============================================================================
def fig_domain_holdout():
    """Domain holdout evaluation."""
    print("[FIG 7] Domain holdout analysis...")
    
    domain_results = load_json(RESULTS_DIR / "domain_holdout" / "domain_holdout_results.json")
    
    domains = list(domain_results['existing_model_per_domain'].keys())
    accuracies = [domain_results['existing_model_per_domain'][d]['accuracy'] * 100 for d in domains]
    gap_closed = [domain_results['existing_model_per_domain'][d]['regret']['gap_closed_pct'] for d in domains]
    
    x = np.arange(len(domains))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, gap_closed, width, label='Gap Closed', color='#2ca02c', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Domain Evaluation: Leave-One-Domain-Out', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    save_fig("07_domain_holdout")

# ============================================================================
# FIGURE 8: Confusion Matrix (v5 Test)
# ============================================================================
def fig_confusion_matrix():
    """Confusion matrix for v5 test set."""
    print("[FIG 8] Confusion matrix...")
    
    preds = load_csv(RESULTS_DIR / "xgboost_v5" / "predictions_test.csv")
    
    if 'predicted' in preds.columns and 'actual' in preds.columns:
        actual = preds['actual'].values
        predicted = preds['predicted'].values
    else:
        actual = preds.iloc[:, 1].values
        predicted = preds.iloc[:, 0].values
    
    classes = ['Introsort', 'Heapsort', 'Timsort']
    cm = confusion_matrix(actual, predicted, labels=['introsort', 'heapsort', 'timsort'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes,
               cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5, linecolor='black')
    
    ax.set_ylabel('True Algorithm', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Algorithm', fontsize=12, fontweight='bold')
    ax.set_title('v5 Test Set: Prediction Confusion Matrix', fontsize=13, fontweight='bold')
    
    save_fig("08_confusion_matrix")

# ============================================================================
# FIGURE 9: Model Predictions Distribution (v5)
# ============================================================================
def fig_predictions_distribution():
    """Distribution of predicted algorithms."""
    print("[FIG 9] Predictions distribution...")
    
    preds = load_csv(RESULTS_DIR / "xgboost_v5" / "predictions_test.csv")
    
    if 'predicted' in preds.columns:
        predicted = preds['predicted'].values
    else:
        predicted = preds.iloc[:, 0].values
    
    classes = ['Introsort', 'Heapsort', 'Timsort']
    counts = [np.sum(predicted == c.lower()) for c in classes]
    percentages = [c / len(predicted) * 100 for c in counts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#ff7f0e', '#d62728', '#2ca02c']
    
    # Bar chart
    bars = ax1.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax1.set_title('Model Selector: Algorithm Distribution (Test Set)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, count, pct in zip(bars, counts, percentages):
        ax1.text(bar.get_x() + bar.get_width()/2., count + len(predicted)*0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=[f'{c}\n{p:.1f}%' for c, p in zip(classes, percentages)],
           colors=colors, autopct='', startangle=90,
           wedgeprops=dict(edgecolor='black', linewidth=1.5), textprops=dict(fontsize=10, fontweight='bold'))
    ax2.set_title('Selection Bias Across Algorithm Classes', fontsize=12, fontweight='bold')
    
    save_fig("09_predictions_distribution", tight_layout=True)

# ============================================================================
# FIGURE 10: v5 vs v6 Comparison
# ============================================================================
def fig_v5_vs_v6():
    """Compare v5 and v6 metrics."""
    print("[FIG 10] v5 vs v6 comparison...")
    
    v5 = load_json(RESULTS_DIR / "xgboost_v5" / "evaluation_results.json")
    v6 = load_json(RESULTS_DIR / "xgboost_v6" / "evaluation_results.json")
    
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'F1-Score\n(Weighted)']
    v5_vals = [
        v5['results']['test']['accuracy'] * 100,
        v5['results']['test']['balanced_accuracy'] * 100,
        v5['results']['test']['classification_report']['weighted avg']['f1-score'] * 100,
    ]
    v6_vals = [
        v6['results']['test']['accuracy'] * 100,
        v6['results']['test']['balanced_accuracy'] * 100,
        v6['results']['test']['classification_report']['weighted avg']['f1-score'] * 100,
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, v5_vals, width, label='v5 (Production)', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, v6_vals, width, label='v6 (Non-Overlapping Sources)', color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison: Production vs Non-Overlapping Sources Evaluation', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    save_fig("10_v5_vs_v6_comparison")

# ============================================================================
# FIGURE 11: Balanced Accuracy Across Splits
# ============================================================================
def fig_balanced_accuracy():
    """Balanced accuracy progression through splits."""
    print("[FIG 11] Balanced accuracy...")
    
    v5 = load_json(RESULTS_DIR / "xgboost_v5" / "evaluation_results.json")
    
    splits = ['Training\nSet', 'Validation\nSet', 'Test\nSet']
    balanced_accs = [
        v5['results']['train']['balanced_accuracy'] * 100,
        v5['results']['val']['balanced_accuracy'] * 100,
        v5['results']['test']['balanced_accuracy'] * 100,
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(splits, balanced_accs, marker='o', markersize=10, linewidth=2.5, color='#2ca02c', label='Balanced Accuracy')
    ax.fill_between(range(len(splits)), balanced_accs, alpha=0.3, color='#2ca02c')
    
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('v5: Balanced Accuracy Across Dataset Splits', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    for i, (split, acc) in enumerate(zip(splits, balanced_accs)):
        ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=11)
    
    save_fig("11_balanced_accuracy")

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Generate all thesis figures v2."""
    print("\n" + "="*70)
    print("GENERATING THESIS FIGURES v2 (IMPROVED LABELING)")
    print("="*70)
    
    try:
        fig_model_history()
        fig_v5_accuracy()
        fig_v5_recall()
        fig_feature_importance()
        fig_regret_analysis()
        fig_vbs_sbs_model()
        fig_domain_holdout()
        fig_confusion_matrix()
        fig_predictions_distribution()
        fig_v5_vs_v6()
        fig_balanced_accuracy()
        
        print("\n" + "="*70)
        print(f"✓ ALL FIGURES GENERATED: {OUTPUT_DIR}")
        print("="*70)
        
        # Create summary
        summary = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "version": "v2",
            "improvements": [
                "Enhanced x-axis labels for all figures",
                "Better font sizes and spacing",
                "Academic naming (v6 = Non-Overlapping Sources)",
                "Formal language (model, framework, selector)",
                "Improved readability throughout"
            ],
            "output_dir": str(OUTPUT_DIR),
            "figures": [
                "01_model_history.png",
                "02_v5_accuracy_metrics.png",
                "03_v5_per_algorithm_recall.png",
                "04_feature_importance.png",
                "05_regret_analysis.png",
                "06_vbs_sbs_model_comparison.png",
                "07_domain_holdout.png",
                "08_confusion_matrix.png",
                "09_predictions_distribution.png",
                "10_v5_vs_v6_comparison.png",
                "11_balanced_accuracy.png",
            ],
            "total_figures": 11,
        }
        
        with open(OUTPUT_DIR / "generation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary: {OUTPUT_DIR / 'generation_summary.json'}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
