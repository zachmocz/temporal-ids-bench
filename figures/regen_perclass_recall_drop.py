"""
Regenerate fig_perclass_recall_drop at a compact size so it takes less
vertical space in the paper. Reads the same two CSVs and computes
random_per_class - group_per_class.
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
if not random_csv.exists():
    random_csv = HERE / 'random_per_class_recall_seed42.csv'
group_csv  = HERE / 'group_per_class_recall_mean.csv'

MODEL_ORDER = ['SVM', 'RF', 'MLP', 'Image-CNN',
               'LSTM', 'GRU', '1D-CNN', 'Transformer', 'Transformer-small']
CLASS_RENAME = {
    'BENIGN': 'BENIGN', 'Bot': 'Bot', 'DDoS': 'DDoS',
    'DoS GoldenEye': 'DoS GoldenEye', 'DoS Hulk': 'DoS Hulk',
    'DoS slowloris': 'DoS slowloris', 'FTP-Patator': 'FTP-Patator',
    'Infiltration': 'Infiltration', 'PortScan': 'PortScan',
    'SSH-Patator': 'SSH-Patator',
    'Web Attack – Brute Force': 'WA Brute Force',
    'Web Attack – Sql Injection': 'WA SQL Inj.',
    'Web Attack – XSS': 'WA XSS',
}

def load(csv):
    df = pd.read_csv(csv).set_index('model').reindex(MODEL_ORDER)
    df.columns = [CLASS_RENAME.get(c, c) for c in df.columns]
    return df

random_df = load(random_csv)
group_df  = load(group_csv)
drop_df   = random_df - group_df  # positive = lost recall

# Compact figure: 9 inches wide x 4.2 inches tall.
# When rendered at \columnwidth (~3.5in), each cell still readable;
# at 0.55\textwidth it stays compact too.
fig, ax = plt.subplots(1, 1, figsize=(9, 4.2), constrained_layout=True)

cmap = plt.get_cmap('RdBu_r')
arr = drop_df.values.astype(float)
im = ax.imshow(arr, cmap=cmap, vmin=-1.0, vmax=1.0, aspect='auto')
n_rows, n_cols = arr.shape
ax.set_xticks(np.arange(n_cols))
ax.set_yticks(np.arange(n_rows))
ax.set_xticklabels(drop_df.columns, rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(drop_df.index, fontsize=9)
ax.set_xlabel('Class', fontsize=10)
ax.set_ylabel('Model', fontsize=10)
ax.set_title('Per-class recall drop: random $-$ group', fontsize=11, pad=6)

for i in range(n_rows):
    for j in range(n_cols):
        v = arr[i, j]
        text_color = 'white' if abs(v) > 0.55 else 'black'
        sign = '+' if v > 0 else ''
        ax.text(j, i, f'{sign}{v:.2f}', ha='center', va='center',
                color=text_color, fontsize=8.5)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
cbar.set_label(r'$\Delta$ recall (random $-$ group)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

out_pdf = HERE / 'fig_perclass_recall_drop.pdf'
out_png = HERE / 'fig_perclass_recall_drop.png'
fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
fig.savefig(out_png, bbox_inches='tight', dpi=150)
print(f"Wrote: {out_pdf}\n       {out_png}\nshape: {arr.shape}")
