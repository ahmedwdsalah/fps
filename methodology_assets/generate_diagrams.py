#!/usr/bin/env python3
"""Generate conceptual diagrams F1, F2, F3, F6, F8 for Chapter 3 Methodology."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path("/sessions/funny-eager-cray/mnt/My-Master-thesis/methodology_assets")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
})

# Color palette
C_PRIMARY = '#2c7bb6'
C_SECONDARY = '#d7191c'
C_ACCENT = '#fdae61'
C_GREEN = '#1a9641'
C_LIGHT = '#e0ecf4'
C_GREY = '#f0f0f0'
C_DARK = '#333333'


def draw_box(ax, xy, w, h, text, color=C_PRIMARY, fontsize=9, text_color='white', alpha=1.0, style='round'):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.08",
                          facecolor=color, edgecolor=C_DARK, linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', wrap=True)
    return box


def draw_arrow(ax, start, end, color=C_DARK, style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════════════════
# F1: System Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════
print("Generating F1: System architecture diagram...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Input
draw_box(ax, (0.3, 2.3), 1.6, 1.0, 'Input\nNumeric Array', C_GREY, text_color=C_DARK)

# Feature Extraction
draw_box(ax, (2.5, 2.3), 1.6, 1.0, 'Feature\nExtraction\n(16 features, O(n))', C_ACCENT, text_color=C_DARK, fontsize=8)

# Layer 1 box
layer1_box = FancyBboxPatch((4.6, 1.2), 2.4, 3.2, boxstyle="round,pad=0.1",
                             facecolor='#e8f4f8', edgecolor=C_PRIMARY, linewidth=2, linestyle='-')
ax.add_patch(layer1_box)
ax.text(5.8, 4.15, 'Layer 1: Offline', ha='center', fontsize=10, color=C_PRIMARY, fontweight='bold')

draw_box(ax, (4.9, 2.3), 1.8, 1.0, 'XGBoost\nClassifier\n(v5)', C_PRIMARY, fontsize=9)

# Layer 2 box
layer2_box = FancyBboxPatch((4.6, 0.0), 2.4, 1.0, boxstyle="round,pad=0.1",
                             facecolor='#fef0e0', edgecolor=C_ACCENT, linewidth=2, linestyle='--')
ax.add_patch(layer2_box)
ax.text(5.8, 0.8, 'Layer 2: Online (Future)', ha='center', fontsize=8, color='#b87000', fontstyle='italic')
draw_box(ax, (4.9, 0.15), 1.8, 0.55, 'LinUCB\nContextual Bandit', '#e8d4a0', text_color=C_DARK, fontsize=8)

# Output
draw_box(ax, (7.6, 2.3), 1.8, 1.0, 'Predicted\nBest Algorithm\n(introsort/heapsort/\ntimsort)', C_GREEN, fontsize=8)

# Arrows
draw_arrow(ax, (1.9, 2.8), (2.5, 2.8))
draw_arrow(ax, (4.1, 2.8), (4.9, 2.8))
draw_arrow(ax, (6.7, 2.8), (7.6, 2.8))

# Feature vector label
ax.text(3.3, 3.5, 'x ∈ R¹⁶', ha='center', fontsize=9, color=C_DARK, fontstyle='italic')

# Feedback arrow (dashed, from output back to Layer 2)
ax.annotate('', xy=(7.0, 0.45), xytext=(7.6, 2.3),
            arrowprops=dict(arrowstyle='->', color=C_ACCENT, lw=1.2, linestyle='--'))
ax.text(7.8, 1.3, 'Runtime\nfeedback', ha='center', fontsize=7.5, color='#b87000', fontstyle='italic')

plt.tight_layout()
fig.savefig(OUT / "F1_system_architecture.png", bbox_inches='tight')
plt.close()
print("  Saved: F1_system_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════
# F2: Feature Extraction Flowchart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating F2: Feature extraction flowchart...")
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Input array
draw_box(ax, (0.2, 3.0), 1.5, 0.8, 'Raw Array\n[x₁, x₂, ..., xₙ]', C_GREY, text_color=C_DARK, fontsize=9)

# Arrow to groups
draw_arrow(ax, (1.7, 3.4), (2.4, 3.4))

# Feature groups (7 groups stacked)
groups = [
    ("Size (2)", ["length_norm", "n_elements"], '#2c7bb6'),
    ("Ordering (5)", ["adj_sorted_ratio", "runs_ratio",\
     "inversion_ratio", "longest_run_ratio", "mean_abs_diff_norm"], '#fdae61'),
    ("Repetition (3)", ["duplicate_ratio", "top1_freq", "top5_freq"], '#d7191c'),
    ("Distribution (5)", ["dispersion_ratio", "entropy_ratio",\
     "skewness_t", "kurtosis_t", "outlier_ratio"], '#abd9e9'),
    ("Robust Scale (2)", ["iqr_norm", "mad_norm"], '#1a9641'),
]

y_positions = [5.8, 4.6, 3.4, 2.0, 0.7]
for i, (group_name, features, color) in enumerate(groups):
    y = y_positions[i]
    # Group box
    draw_box(ax, (2.4, y-0.1), 1.8, 0.8, group_name, color, fontsize=8.5,
             text_color='white' if color not in ['#abd9e9', '#fdae61'] else C_DARK)
    # Feature list
    feat_text = ", ".join(features)
    ax.text(4.5, y + 0.3, feat_text, fontsize=7, color=C_DARK, va='center',
            fontstyle='italic')

    # Arrow from input to each group
    if i != 2:  # middle one is direct
        ax.annotate('', xy=(2.4, y+0.3), xytext=(1.7, 3.4),
                    arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

# Output vector
draw_box(ax, (7.5, 2.8), 2.0, 1.2, 'Feature Vector\nx ∈ R¹⁶\n(all bounded,\nmostly [0,1])', C_PRIMARY, fontsize=8.5)

# Arrows from groups to output
for y in y_positions:
    # Find right edge of feature text area
    ax.annotate('', xy=(7.5, 3.4), xytext=(6.8, y+0.3),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

# O(n) label
ax.text(5.0, 6.6, 'All features computed in O(n) time', ha='center',
        fontsize=11, color=C_PRIMARY, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LIGHT, edgecolor=C_PRIMARY))

plt.tight_layout()
fig.savefig(OUT / "F2_feature_extraction.png", bbox_inches='tight')
plt.close()
print("  Saved: F2_feature_extraction.png")


# ═══════════════════════════════════════════════════════════════════════════
# F3: Model Evolution Timeline
# ═══════════════════════════════════════════════════════════════════════════
print("Generating F3: Model evolution timeline...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(0, 110)
ax.axis('off')

versions = [
    ("v1", 44.4, "Regression\n(failed)", C_SECONDARY),
    ("v2", 62.5, "Classifier\n(synthetic)", C_ACCENT),
    ("v3", 100.0, "Log+pairwise\n(data leakage)", '#999999'),
    ("v5", 76.1, "Production\n(1.18M arrays)", C_PRIMARY),
    ("v6", 71.2, "Source-aware\n(strict split)", '#7570b3'),
    ("v7", 72.9, "Regret-weighted\n(rejected)", C_SECONDARY),
    ("v8", 72.6, "Binary cascade\n(intro≈heap)", C_ACCENT),
]

x_positions = list(range(len(versions)))

# Draw bars
bar_width = 0.6
for i, (ver, acc, label, color) in enumerate(versions):
    bar = ax.bar(i, acc, width=bar_width, color=color, edgecolor=C_DARK,
                 linewidth=1, alpha=0.85)
    # Version label on top
    ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold', color=C_DARK)
    # Version name below bar
    ax.text(i, -4, ver, ha='center', fontsize=10, fontweight='bold', color=C_DARK)
    # Description below version
    ax.text(i, -14, label, ha='center', fontsize=7.5, color='#555555')

# Highlight v5 as production
ax.bar(3, 76.1, width=bar_width, color=C_PRIMARY, edgecolor=C_DARK,
       linewidth=2.5, alpha=0.95)
ax.text(3, 76.1 + 2, '76.1%', ha='center', fontsize=10, fontweight='bold', color=C_PRIMARY)

# Add horizontal line for v5 reference
ax.axhline(y=76.1, color=C_PRIMARY, linestyle='--', linewidth=0.8, alpha=0.5, xmin=0.05, xmax=0.95)

# Mark v3 as not deployable
ax.text(2, 95, '* Not\ndeployable', ha='center', fontsize=7, color='#666666', fontstyle='italic')

# Add key finding annotations
ax.annotate('Key finding:\nintro ≈ heap\n(AUC 0.603)', xy=(6, 72.6), xytext=(6.5, 90),
            fontsize=7, ha='center', color='#555555',
            arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

# Axis labels
ax.text(3, 108, 'Model Evolution: Test Accuracy by Version', ha='center',
        fontsize=12, fontweight='bold', color=C_DARK)

# Simple y-axis guide lines
for y in [25, 50, 75, 100]:
    ax.axhline(y=y, color='#e0e0e0', linewidth=0.5, zorder=0)
    ax.text(-0.4, y, f'{y}%', fontsize=8, color='#999999', ha='right', va='center')

plt.tight_layout()
fig.savefig(OUT / "F3_model_evolution.png", bbox_inches='tight')
plt.close()
print("  Saved: F3_model_evolution.png")


# ═══════════════════════════════════════════════════════════════════════════
# F6: Channel-Flag Routing Flowchart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating F6: Channel-flag routing flowchart...")
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.axis('off')

# Input
draw_box(ax, (0.2, 2.6), 1.4, 0.9, 'F1 Telemetry\nArray', C_GREY, text_color=C_DARK, fontsize=9)

# Channel flag
draw_box(ax, (0.2, 4.2), 1.4, 0.7, 'Channel\nFlag', C_ACCENT, text_color=C_DARK, fontsize=9)

# Feature extraction
draw_box(ax, (2.2, 2.6), 1.5, 0.9, 'Feature\nExtraction\n(16 features)', '#abd9e9', text_color=C_DARK, fontsize=8)

# Router
draw_box(ax, (4.2, 2.8), 1.3, 0.7, 'Router\n(by flag)', C_PRIMARY, fontsize=9)

# Arrows to router
draw_arrow(ax, (1.6, 3.1), (2.2, 3.1))
draw_arrow(ax, (3.7, 3.1), (4.2, 3.1))
draw_arrow(ax, (1.6, 4.55), (4.8, 3.5), color=C_ACCENT)

# Channel models (show 4 representative ones)
channels = [
    ("Speed\n(5 classes)", 5.2),
    ("RPM\n(5 classes)", 4.1),
    ("DRS\n(3 classes)", 3.0),
    ("nGear\n(4 classes)", 1.9),
]

colors_ch = ['#2c7bb6', '#7570b3', '#d95f02', '#1b9e77']
for i, ((name, _y), col) in enumerate(zip(channels, colors_ch)):
    y = channels[i][1]
    draw_box(ax, (6.0, y-0.25), 1.4, 0.7, name, col, fontsize=7.5)
    # Arrow from router
    ax.annotate('', xy=(6.0, y+0.1), xytext=(5.5, 3.15),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1))

# "..." for other channels
ax.text(6.7, 0.9, '... (9 channels total)', ha='center', fontsize=8,
        color='#777777', fontstyle='italic')

# Output
draw_box(ax, (8.0, 2.6), 1.5, 0.9, 'Predicted\nAlgorithm\n(per-channel)', C_GREEN, fontsize=8)

# Arrows from channel models to output
for i, (name, y) in enumerate(channels):
    ax.annotate('', xy=(8.0, 3.1), xytext=(7.4, y+0.1),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

# Title annotation
ax.text(5.0, 6.2, 'Domain-Routed Selection: Channel Flag as Routing Signal',
        ha='center', fontsize=11, fontweight='bold', color=C_DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LIGHT, edgecolor=C_PRIMARY))

# Key insight box
ax.text(5.0, 0.3, 'Each channel model is specialised to its signal characteristics\n'
        'Dynamic class sets: only algorithms with ≥20 samples per channel',
        ha='center', fontsize=8, color='#555555', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#cccccc'))

plt.tight_layout()
fig.savefig(OUT / "F6_channel_routing.png", bbox_inches='tight')
plt.close()
print("  Saved: F6_channel_routing.png")


# ═══════════════════════════════════════════════════════════════════════════
# F8: Data Collection Pipeline
# ═══════════════════════════════════════════════════════════════════════════
print("Generating F8: Data collection pipeline...")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis('off')

# Domain sources (left column)
domains = [
    ("F1 Telemetry", '#2c7bb6'),
    ("Stock Market", '#d7191c'),
    ("Cryptocurrency", '#fdae61'),
    ("Earthquake", '#1a9641'),
    ("Weather", '#7570b3'),
]

for i, (name, color) in enumerate(domains):
    y = 4.2 - i * 0.85
    draw_box(ax, (0.2, y), 1.5, 0.6, name, color, fontsize=8)
    draw_arrow(ax, (1.7, y+0.3), (2.3, 2.55))

# Fetch & Parse
draw_box(ax, (2.3, 2.1), 1.3, 0.9, 'Fetch\n& Parse\nRaw Arrays', C_GREY, text_color=C_DARK, fontsize=8)

# Structural Transforms
draw_box(ax, (4.0, 1.6), 1.5, 1.8, 'Structural\nTransforms\n\nRAW\nREV\nSHUF\nQBIN50\nPSORT10', C_ACCENT, text_color=C_DARK, fontsize=7.5)
draw_arrow(ax, (3.6, 2.55), (4.0, 2.55))

# Timing & Labelling
draw_box(ax, (5.9, 2.1), 1.3, 0.9, 'Timing &\nLabelling\n(best-of-K)', '#abd9e9', text_color=C_DARK, fontsize=8)
draw_arrow(ax, (5.5, 2.55), (5.9, 2.55))

# Feature Extraction
draw_box(ax, (7.6, 2.1), 1.3, 0.9, 'Feature\nExtraction\n(16 features)', C_PRIMARY, fontsize=8)
draw_arrow(ax, (7.2, 2.55), (7.6, 2.55))

# Output dataset
draw_box(ax, (7.6, 0.5), 1.3, 0.9, 'Training\nDataset\n(1.18M rows)', C_GREEN, fontsize=8)
draw_arrow(ax, (8.25, 2.1), (8.25, 1.4))

# Data quality box
ax.text(5.0, 0.5, '8 Quality Controls:\nDuplicate removal, NaN/Inf filter,\nconstant array filter, source_id\nfor leakage prevention',
        ha='center', fontsize=7.5, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor='#d7191c', linewidth=0.8))

# Stats
ax.text(5.0, 5.1, 'Data Collection Pipeline: 5 Domains, 5 Transforms, 1.18M Arrays',
        ha='center', fontsize=11, fontweight='bold', color=C_DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_LIGHT, edgecolor=C_PRIMARY))

plt.tight_layout()
fig.savefig(OUT / "F8_data_pipeline.png", bbox_inches='tight')
plt.close()
print("  Saved: F8_data_pipeline.png")

print("\nAll conceptual diagrams generated successfully!")
