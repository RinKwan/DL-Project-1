"""Evaluate a trained ResNet on the project's held-out validation images.

Initial working version: checkpoint loading, accuracy/loss, optional horizontal
flip TTA, confusion matrix and per-class CSV.  It reads the competition's
train_labels.csv and PNGs directly, so it also works in the early repository
folder before dataset.py is introduced.

Examples (from the repository root):
    python FirstMeeting_corecode/evaluate.py --checkpoint runs/main/best.pt
    python FirstMeeting_corecode/evaluate.py --checkpoint runs/main/best.pt --no-tta

The validation subset must match the training split: NumPy RandomState(42),
first 5000 indices in the permutation of CSV row order.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from model import PARAMETER_LIMIT, build_model, count_parameters

CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck")
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def load_validation(data_dir, val_size=5000, split_seed=42):
    """Load only validation PNGs (not the 45k training images)."""
    data_dir = Path(data_dir)
    with (data_dir / "train_labels.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not 1 <= val_size <= len(rows):
        raise ValueError(f"val_size must be between 1 and {len(rows)}")

    indices = np.random.RandomState(split_seed).permutation(len(rows))[:val_size]
    images = np.empty((val_size, 32, 32, 3), dtype=np.uint8)
    labels = np.empty(val_size, dtype=np.int64)
    for j, idx in enumerate(indices):
        row = rows[int(idx)]
        filename = data_dir / "train_images" / row["filename"]
        with Image.open(filename) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (32, 32, 3):
            raise ValueError(f"unexpected image shape {rgb.shape}: {filename}")
        images[j] = rgb
        labels[j] = int(row["label"])
    if not np.all((0 <= labels) & (labels < len(CLASS_NAMES))):
        raise ValueError("label outside the 10 CIFAR-10 classes")
    return images, labels


def to_tensor_batch(images):
    """Match the normalization used by the project's training data loader."""
    x = images.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(x))


@torch.no_grad()
def run_evaluation(model, images, labels, device, batch_size=256, tta=False):
    """Return average CE loss, accuracy, and integer predictions."""
    model.eval()
    predictions = []
    loss_sum = 0.0
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        inputs = to_tensor_batch(images[start:stop]).to(device)
        targets = torch.from_numpy(labels[start:stop]).to(device)
        logits = model(inputs)
        if tta:
            # Two views of the SAME model; does not increase parameter count.
            logits = (logits + model(torch.flip(inputs, dims=(3,)))) / 2
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(predictions)
    return loss_sum / len(labels), float(np.mean(pred == labels)), pred


def confusion_matrix(labels, predictions, n_classes=10):
    """Rows are true labels; columns are predicted labels."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    return cm


def save_confusion_figure(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm_pct = 100.0 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(image, ax=ax, label="% within true class")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Validation confusion matrix")
    for row in range(10):
        for col in range(10):
            if cm_pct[row, col] >= 1:
                ax.text(col, row, f"{cm_pct[row, col]:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_pct[row, col] >= 60 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved confusion matrix: {path}")


def save_per_class_csv(cm, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "samples", "correct", "accuracy_%"])
        for i, name in enumerate(CLASS_NAMES):
            count = int(cm[i].sum())
            correct = int(cm[i, i])
            acc = 100.0 * correct / count if count else 0.0
            writer.writerow([name, count, correct, f"{acc:.2f}"])
    print(f"saved per-class accuracy: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="runs/main/best.pt")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-fig", default="figures/confusion.png")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("expected checkpoint dict with 'model' and 'state_dict' keys")
    dropout = checkpoint.get("args", {}).get("dropout", 0.0)
    model = build_model(checkpoint["model"], dropout=dropout)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()

    params = count_parameters(model)
    print(f"model: {checkpoint['model']} (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"trainable parameters: {params:,} / {PARAMETER_LIMIT:,}")
    if params > PARAMETER_LIMIT:
        print("WARNING: model exceeds the project's parameter budget")

    images, labels = load_validation(args.data_dir, args.val_size, args.split_seed)
    print(f"held-out validation images: {len(labels)} (split seed {args.split_seed})")
    plain_loss, plain_acc, pred = run_evaluation(model, images, labels, device,
                                                 args.batch_size, tta=False)
    print(f"validation accuracy (no TTA): {plain_acc * 100:.2f}%  loss {plain_loss:.4f}")
    if not args.no_tta:
        flip_loss, flip_acc, _ = run_evaluation(model, images, labels, device,
                                                args.batch_size, tta=True)
        print(f"validation accuracy (flip TTA): {flip_acc * 100:.2f}%  loss {flip_loss:.4f}")

    cm = confusion_matrix(labels, pred)
    save_confusion_figure(cm, args.out_fig)
    save_per_class_csv(cm, Path(args.out_fig).with_name("per_class.csv"))
    print(f"final validation accuracy (no TTA): {plain_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
