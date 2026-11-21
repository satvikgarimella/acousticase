"""Create a shareable benchmark figure from the recorded evaluation results."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

report = json.loads((ROOT / "evaluation/artifacts/benchmark.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
fig.suptitle("AcoustiCase · initial held-out audio evaluation", fontsize=17, fontweight="bold")
classification = report["closed_set_classification"]["test"]
counts = [round(classification[k]["accuracy"]*64) for k in ("pretrained_grouped_baseline", "trained_linear_head")]
axes[0].bar([0, 1], counts, color=["#94a3b8", "#2563eb"], width=.55)
axes[0].set(ylim=(0, 70), xticks=[0, 1], xticklabels=["Pretrained baseline", "Trained clip classifier"], ylabel="Correct labels / 64 test recordings", title="Eight-category clip classification")
for i, count in enumerate(counts):
    axes[0].text(i, count+1, f"{count}/64", ha="center", fontsize=12, fontweight="bold")
keys = ["horn", "siren", "engine", "glass"]
detectors = report["clip_presence_detection"]["test"]
x = np.arange(4)
for shift, name, label, color in [(-.19, "legacy_v0_1", "Original detector", "#94a3b8"), (.19, "grouped_tuned", "Updated detector", "#2563eb")]:
    vals = [detectors[name]["per_class"][k]["f1"] for k in keys]
    axes[1].bar(x+shift, vals, width=.36, color=color, label=label)
axes[1].set(xticks=x, xticklabels=["Horn", "Siren", "Engine", "Glass"], ylim=(0, 1.1), ylabel="F1 score", title="Target sound presence in each clip")
axes[1].legend(loc="upper right", fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
fig.supxlabel("320 ESC-50 clips: 192 train / 64 validation / 64 test. Small development benchmark; no timing, collision, or speed accuracy claim.", fontsize=9)
output = ROOT.parent / "docs/ml-validation.png"
fig.savefig(output, dpi=170)
print(output)
