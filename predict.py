"""
predict.py
----------
Use a trained checkpoint to label the test images and write a submission file
in the same format as data/sample_submission.csv (two columns: id,label).

Example:
    python src/predict.py --checkpoint runs/main/best.pt --out data/submission.csv

By default, test-time augmentation (TTA) averages the logits for each image
and its horizontal mirror. Pass --no-tta to use only the original images.
"""

import argparse
import csv
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES, load_test, to_tensor_batch
from model import build_model, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint", default="runs/main/best.pt")
    p.add_argument("--out", default="data/submission.csv")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--no-tta", action="store_true")
    return p.parse_args()


def predict_batches(model, images, num_images, batch_size, device, tta):
    batches = []
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            xb = to_tensor_batch(images[start:start + batch_size]).to(device)
            logits = model(xb)
            if tta:
                logits = (logits + model(torch.flip(xb, dims=[3]))) / 2
            batches.append(logits.argmax(1).cpu())
    return torch.cat(batches).numpy()


def write_submission(path, ids, predictions):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for image_id, prediction in zip(ids, predictions):
            writer.writerow([int(image_id), int(prediction)])


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

    preds = predict_batches(
        model, images, len(ids), args.batch_size, device, tta=not args.no_tta
    )
    write_submission(args.out, ids, preds)
    print(f"wrote {args.out}  ({len(preds)} rows)")

    # a small sanity check: the class distribution should be roughly uniform
    print("predicted class counts:")
    for c in range(len(CLASS_NAMES)):
        print(f"  {c} {CLASS_NAMES[c]:10s} {int((preds == c).sum())}")


if __name__ == "__main__":
    main()
