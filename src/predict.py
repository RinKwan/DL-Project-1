"""
predict.py
----------
Use a trained checkpoint to label the test images and write a submission file
in the same format as data/sample_submission.csv (two columns: id,label).

Example:
    python src/predict.py --checkpoint runs/main/best.pt --out data/submission.csv

We use simple test-time augmentation (TTA): the prediction of an image and the
prediction of its mirrored copy are averaged. This usually adds about 0.2-0.4%
accuracy and costs nothing.
"""

import argparse
import os
import sys

import csv

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_test, to_tensor_batch, CLASS_NAMES
from model import build_model, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="runs/main/best.pt")
    p.add_argument("--out", default="data/submission.csv")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")
    print(f"trainable parameters: {count_parameters(model):,}")

    images, ids = load_test(args.data_dir)
    print(f"test images: {len(ids)}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), args.batch_size):
            xb = to_tensor_batch(images[i:i + args.batch_size]).to(device)
            logits = model(xb)
            if not args.no_tta:
                logits = (logits + model(torch.flip(xb, dims=[3]))) / 2
            preds.append(logits.argmax(1).cpu())

    preds = torch.cat(preds).numpy()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for i, p in zip(ids, preds):
            writer.writerow([int(i), int(p)])
    print(f"wrote {args.out}  ({len(preds)} rows)")

    # a small sanity check: the class distribution should be roughly uniform
    print("predicted class counts:")
    for c in range(len(CLASS_NAMES)):
        print(f"  {c} {CLASS_NAMES[c]:10s} {int((preds == c).sum())}")


if __name__ == "__main__":
    main()
