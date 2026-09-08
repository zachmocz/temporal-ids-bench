"""
Regenerate fig_perclass_recall_by_split with proper spacing.
Reads the same two CSVs the V4 notebook used; outputs PDF + PNG with
pdf.fonttype=42 so text is selectable. Wider figure + larger annotations
+ shorter class labels to remove the cramped look.
"""
from pathlib import Path
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# Prefer the 3-seed mean (written by the V5 notebook); fall back to the V4
# seed-42 snapshot so the script still works against the archived V4 folder.
random_csv = HERE / 'random_per_class_recall_mean.csv'
RANDOM_TITLE = 'Per-class recall — random split (mean across seeds)'
if not random_csv.exists():
    random_csv = HERE / 'random_per_class_recall_seed42.csv'
    RANDOM_TITLE = 'Per-class recall — random split (seed=42)'
group_csv  = HERE / 'group_per_class_recall_mean.csv'

# Model order (matches V4 figure)
MODEL_ORDER = ['SVM', 'RF', 'MLP', 'Image-CNN',
               'LSTM', 'GRU', '1D-CNN', 'Transformer', 'Transformer-small']

# Shorter class labels for readability
CLASS_RENAME = {
    'BENIGN': 'BENIGN',
    'Bot': 'Bot',
    'DDoS': 'DDoS',
    'DoS GoldenEye': 'DoS GoldenEye',
    'DoS Hulk': 'DoS Hulk',
    'DoS slowloris': 'DoS slowloris',
    'FTP-Patator': 'FTP-Patator',
    'Infiltration': 'Infiltration',
    'PortScan': 'PortScan',
    'SSH-Patator': 'SSH-Patator',
    'Web Attack – Brute Force': 'WA Brute Force',
    'Web Attack – Sql Injection': 'WA SQL Inj.',
    'Web Attack – XSS': 'WA XSS',
}

def load(csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index('model').reindex(MODEL_ORDER)
    df.columns = [CLASS_RENAME.get(c, c) for c in df.columns]
    return df

random_df = load(random_csv)
group_df  = load(group_csv)

print(f"random shape: {random_df.shape}, group shape: {group_df.shape}")
print(f"models: {list(random_df.index)}")
print(f"classes: {list(random_df.columns)}")

# ---- Render ---------------------------------------------------------------
# Side-by-side layout (user preference). Wide aspect ratio so when scaled to
# \textwidth in LaTeX the figure stays vertically compact while keeping each
# cell readable. Each panel ~9.5 inches wide.
fig, axes = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)

cmap = plt.get_cmap('viridis')

def draw(ax, df, title):
    arr = df.values.astype(float)
    im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')
    n_rows, n_cols = arr.shape
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(df.columns, rotation=40, ha='right', fontsize=12)
    ax.set_yticklabels(df.index, fontsize=12)
    ax.set_xlabel('Class', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)
    ax.set_title(title, fontsize=14)
    for i in range(n_rows):
        for j in range(n_cols):
            v = arr[i, j]
            text_color = 'white' if v < 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=text_color, fontsize=11)
    return im

im1 = draw(axes[0], random_df, RANDOM_TITLE)
im2 = draw(axes[1], group_df,  'Per-class recall — group-by-5-tuple split (mean across seeds)')

cbar = fig.colorbar(im2, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label('Recall', fontsize=13)
cbar.ax.tick_params(labelsize=11)

# ---- Save ----------------------------------------------------------------
out_pdf = HERE / 'fig_perclass_recall_by_split_v4.pdf'
out_png = HERE / 'fig_perclass_recall_by_split_v4.png'
fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
fig.savefig(out_png, bbox_inches='tight', dpi=150)
print(f"\nWrote:\n  {out_pdf}\n  {out_png}")
