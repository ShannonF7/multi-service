import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(__file__)
REPORT_JSON = os.path.join(ROOT, "test_dataset", "eval_report.json")
OUT_DIR = os.path.join(ROOT, "test_dataset", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

with open(REPORT_JSON, "r", encoding="utf-8") as f:
    report = json.load(f)

metrics = report.get("metrics", {})
top1_dist = report.get("top1_distribution", {})
details = report.get("details", [])
seed_summary = report.get("seed_summary", {})

# 1) Metrics bar (Top1 / Top3 accuracy for normal & hard; negative-top1-rate)
labels = ["normal", "hard"]
norm_vals = []
hard_vals = []
for split in labels:
    d = metrics.get(split, {})
    total = d.get("total", 0)
    t1 = d.get("top1_hit", 0)
    t3 = d.get("top3_hit", 0)
    if total:
        norm_vals.append((t1 / total, t3 / total))
    else:
        norm_vals.append((0.0, 0.0))

# negative ratio
neg = metrics.get("negative", {})
neg_total = neg.get("total", 0)
neg_like = neg.get("top1_is_negative_like", 0)
neg_ratio = (neg_like / neg_total) if neg_total else 0.0

# Plot metrics
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
t1_vals = [norm_vals[i][0] for i in range(len(labels))]
t3_vals = [norm_vals[i][1] for i in range(len(labels))]
ax.bar(x - width/2, t1_vals, width, label='Top1 Acc')
ax.bar(x + width/2, t3_vals, width, label='Top3 Acc')
ax.set_ylabel('Accuracy')
ax.set_title('Top1/Top3 Accuracy by Split')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()
for i, v in enumerate(t1_vals):
    ax.text(i - width/2, v + 0.02, f"{v:.0%}", ha='center')
for i, v in enumerate(t3_vals):
    ax.text(i + width/2, v + 0.02, f"{v:.0%}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'metrics_accuracy.png'))
plt.close()

# Plot negative ratio
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(['negative_top1_like_ratio'], [neg_ratio], color='tab:orange')
ax.set_ylim(0, 1)
ax.set_ylabel('Ratio')
ax.set_title('Negative queries: top1 in {animal,vehicle}')
ax.text(0, neg_ratio + 0.02, f"{neg_ratio:.0%}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'negative_ratio.png'))
plt.close()

# 2) Top1 distribution (pie + bar)
labels = list(top1_dist.keys())
values = [top1_dist[k] for k in labels]
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(values, labels=labels, autopct='%1.0f%%', startangle=90)
ax.set_title('Top1 Label Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'top1_distribution_pie.png'))
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, values, color='tab:green')
ax.set_title('Top1 Label Counts')
ax.set_ylabel('Count')
for i, v in enumerate(values):
    ax.text(i, v + 0.02, str(v), ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'top1_distribution_bar.png'))
plt.close()

# 3) Distance distribution for top1 results
# Collect top1 distances
top1_distances = []
for d in details:
    ranked = d.get('ranked_results', [])
    if ranked:
        top1_distances.append(ranked[0].get('distance', 0.0))

if top1_distances:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(top1_distances, bins=10, color='tab:blue')
    ax.set_title('Top1 Distance Distribution')
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'top1_distance_hist.png'))
    plt.close()

print('Plots written to', OUT_DIR)
print('You can open the PNG files to inspect the visualizations.')
