"""
train.py
--------
Train the ResNet from scratch on the project data.

Example:
    python src/train.py --model resnet18_w40 --epochs 150 --out-dir runs/main

What the script does:
  1. reads all images into RAM,
  2. splits the training data into train (90%) and validation (10%),
  3. trains with SGD + cosine learning rate schedule,
  4. every epoch prints train loss / accuracy and validation accuracy,
  5. saves a checkpoint, a CSV with all numbers and a PNG with the curves.

The number of trainable parameters is printed, so we can check the 5M limit.
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (load_train, random_crop_and_flip, cutout,
                     to_tensor_batch, mixup, cutmix, color_jitter, add_noise)
from model import build_model, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model", default="resnet18_w40")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--cutout", type=int, default=8, help="0 turns cutout off")
    p.add_argument("--mixup", type=float, default=0.0, help="alpha, 0 turns it off")
    p.add_argument("--cutmix", type=float, default=0.0, help="alpha, 0 turns it off")
    p.add_argument("--color-jitter", type=float, default=0.0,
                   help="strength of brightness/contrast/saturation jitter, 0 = off")
    p.add_argument("--noise", type=float, default=0.0,
                   help="max Gaussian noise sigma in [0,1] pixel units, 0 = off")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)  # in-RAM data, not needed
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--out-dir", default="runs/exp")
    return p.parse_args()


def evaluate(model, x_u8, y, batch_size=500, device="cuda", tta=False):
    """Return (loss, accuracy) on a fixed set of images."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            xb = to_tensor_batch(x_u8[i:i + batch_size]).to(device)
            yb = torch.from_numpy(y[i:i + batch_size]).to(device)
            if tta:  # average the logits of the image and its mirror image
                logits = (model(xb) + model(torch.flip(xb, dims=[3]))) / 2
            else:
                logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":                      # a bit faster on a fixed input size
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- data -------------------------------------------------------------
    images, labels = load_train(args.data_dir)
    rng = np.random.RandomState(42)          # fixed split, same for all runs
    order = rng.permutation(len(labels))
    val_idx = order[:args.val_size]
    tr_idx = order[args.val_size:]
    x_tr, y_tr = images[tr_idx], labels[tr_idx]
    x_val, y_val = images[val_idx], labels[val_idx]
    print(f"train images: {len(y_tr)}   validation images: {len(y_val)}")

    # ---- model ------------------------------------------------------------
    model = build_model(args.model, dropout=args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"model: {args.model}")
    print(f"trainable parameters: {n_params:,}  (limit is 5,000,000)")
    assert n_params < 5_000_000, "model is too big"

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    steps_per_epoch = math.ceil(len(y_tr) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:                     # linear warmup
            return args.lr * (step + 1) / warmup_steps
        # cosine decay from lr to 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * progress))

    # ---- training loop ----------------------------------------------------
    log_path = os.path.join(args.out_dir, "metrics.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc",
                                "val_loss", "val_acc", "seconds"])

    best_acc = 0.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(y_tr))
        run_loss, run_correct, seen = 0.0, 0, 0

        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb_u8 = random_crop_and_flip(x_tr[idx])
            if args.color_jitter > 0:
                xb_u8 = color_jitter(xb_u8, args.color_jitter)
            if args.noise > 0:
                xb_u8 = add_noise(xb_u8, args.noise)
            if args.cutout > 0:
                xb_u8 = cutout(xb_u8, args.cutout)
            xb = to_tensor_batch(xb_u8).to(device, non_blocking=True)
            yb = torch.from_numpy(y_tr[idx]).to(device, non_blocking=True)

            lr = lr_at(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if args.mixup > 0:
                xb, target = mixup(xb, yb, args.mixup)
            elif args.cutmix > 0:
                xb, target = cutmix(xb, yb, args.cutmix)
            else:
                target = yb

            with torch.amp.autocast("cuda", enabled=args.amp and device == "cuda"):
                logits = model(xb)
                loss = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
            step += 1

        val_loss, val_acc = evaluate(model, x_val, y_val, device=device)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  lr {lr:.4f}  "
              f"train_loss {run_loss / seen:.4f}  train_acc {run_correct / seen:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  ({dt:.0f}s)",
              flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(lr, 6),
                                    round(run_loss / seen, 5),
                                    round(run_correct / seen, 5),
                                    round(val_loss, 5), round(val_acc, 5),
                                    round(dt, 1)])

        torch.save({"model": args.model, "args": vars(args), "epoch": epoch,
                    "state_dict": model.state_dict()},
                   os.path.join(args.out_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": args.model, "args": vars(args), "epoch": epoch,
                        "state_dict": model.state_dict()},
                       os.path.join(args.out_dir, "best.pt"))

    print(f"\nbest validation accuracy: {best_acc:.4f}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "params": n_params,
                   "best_val_acc": best_acc, "epochs": args.epochs,
                   "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
