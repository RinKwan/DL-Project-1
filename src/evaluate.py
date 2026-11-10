"""
evaluate.py
-----------
Check a checkpoint on the held-out validation split and save a confusion
matrix. This is also the file where the final accuracy is printed.

Example:
    python src/evaluate.py --checkpoint runs/main/best.pt
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_train, to_tensor_batch, CLASS_NAMES
from model import build_model, count_parameters
from train import evaluate


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, path, title="Confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9.5)
    for i in range(10):
        for j in range(10):
            v = cm_norm[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 60 else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="runs/main/best.pt")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--out-fig", default="figures/confusion.png")
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    images, labels = load_train(args.data_dir)
    # exactly the same split as in train.py
    order = np.random.RandomState(42).permutation(len(labels))
    val_idx = order[:args.val_size]
    x_val, y_val = images[val_idx], labels[val_idx]

    for tta in ([False, True] if not args.no_tta else [False]):
        loss, acc = evaluate(model, x_val, y_val, device=device, tta=tta)
        print(f"validation accuracy ({'with' if tta else 'without'} TTA): "
              f"{acc * 100:.2f}%   loss {loss:.4f}")

    # confusion matrix of the plain predictions
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_val), 500):
            xb = to_tensor_batch(x_val[i:i + 500]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    cm = confusion_matrix(y_val, preds)
    plot_confusion(cm, args.out_fig)

    print(f"\nnumber of trainable parameters: {count_parameters(model):,}")
    print(f"final validation accuracy: {(preds == y_val).mean() * 100:.2f}%")
    with open(os.path.join(os.path.dirname(args.out_fig) or ".", "per_class.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy_%"])
        for c in range(10):
            w.writerow([CLASS_NAMES[c], round(cm[c, c] / cm[c].sum() * 100, 1)])


if __name__ == "__main__":
    main()
