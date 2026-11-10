"""
ensemble_predict.py
-------------------
Average the predictions of several checkpoints and write submission.csv.

Example:
    python src/ensemble_predict.py --checkpoints runs/robust_w48/best.pt \
        runs/main/best.pt --tta full --out data/submission.csv

Averaging the probabilities (not the labels) of a few models that were trained
in different ways removes part of the random error of each single model. A small
test-time augmentation (the image and its mirror image) is added for free.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import load_test, to_tensor_batch, CLASS_NAMES
from model import build_model


def tta_views(xb, mode):
    """Return the list of augmented versions of a batch we average over."""
    views = [xb]
    if mode in ("flip", "shift", "full"):
        views.append(torch.flip(xb, dims=[3]))
    if mode in ("shift", "full"):
        views.append(torch.roll(xb, shifts=1, dims=2))
        views.append(torch.roll(xb, shifts=-1, dims=2))
    if mode == "full":
        views.append(torch.roll(torch.flip(xb, dims=[3]), shifts=1, dims=2))
        views.append(torch.roll(torch.flip(xb, dims=[3]), shifts=-1, dims=2))
    return views


def predict_probs(model, images, device, mode, batch_size=500):
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            xb = to_tensor_batch(images[i:i + batch_size]).to(device)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                p = torch.stack([torch.softmax(model(v).float(), 1)
                                 for v in tta_views(xb, mode)]).mean(0)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--tta", default="full", choices=["none", "flip", "shift", "full"])
    p.add_argument("--out", default="data/submission.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, ids = load_test(args.data_dir)
    weights = args.weights or [1.0] * len(args.checkpoints)

    total = np.zeros((len(ids), len(CLASS_NAMES)), dtype=np.float64)
    for path, w in zip(args.checkpoints, weights):
        ck = torch.load(path, map_location=device, weights_only=False)
        model = build_model(ck["model"]).to(device)
        model.load_state_dict(ck["state_dict"])
        probs = predict_probs(model, images, device, args.tta)
        total += w * probs
        print(f"  {path}  ({ck['model']}, epoch {ck.get('epoch')})  weight {w}", flush=True)

    preds = total.argmax(1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for i, lab in zip(ids, preds):
            writer.writerow([int(i), int(lab)])
    print(f"wrote {args.out}  ({len(preds)} rows)")
    counts = np.bincount(preds, minlength=len(CLASS_NAMES))
    print("predicted class counts:", dict(zip(CLASS_NAMES, counts.tolist())))


if __name__ == "__main__":
    main()
