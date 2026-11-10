"""
dataset.py
----------
Loading the CIFAR-10 style data that comes with the project.

The data folder looks like this:
    data/train_images/*.png   + data/train_labels.csv   (50,000 images, 10 classes)
    data/test_images/*.png    + data/test.csv           (2,000 images, no labels)

All images are 32x32 RGB. Because the whole training set is small (about
150 MB), we read every picture once into one numpy array and keep it in RAM.
Random augmentation is then done on a whole mini-batch at a time, which is
much faster than reading PNG files again in every epoch.

Only the standard `csv` module is used for the label files, so the code needs
very few dependencies (torch, numpy, pillow).
"""

import csv
import os

import numpy as np
from PIL import Image

# CIFAR-10 mean / std (computed over the training set)
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def read_csv(path):
    """Read a small csv file with a header into a list of dictionaries."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_images(folder, filenames, cache_name):
    """Read all PNG files into a uint8 array of shape (N, 32, 32, 3).

    Reading 50,000 small PNG files takes about one minute, so the array is
    saved once in a `.cache` folder and loaded from there next time.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name + ".npy")

    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == len(filenames):
            return np.array(arr)          # fast copy into RAM

    out = np.empty((len(filenames), 32, 32, 3), dtype=np.uint8)
    for i, name in enumerate(filenames):
        img = Image.open(os.path.join(folder, name)).convert("RGB")
        out[i] = np.asarray(img, dtype=np.uint8)
    np.save(cache_path, out)
    return out


def load_train(data_dir):
    """Return (images, labels) for the training set."""
    rows = read_csv(os.path.join(data_dir, "train_labels.csv"))
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    images = _load_images(os.path.join(data_dir, "train_images"),
                          [r["filename"] for r in rows], "train_images")
    return images, labels


def load_test(data_dir):
    """Return (images, ids) for the test set."""
    rows = read_csv(os.path.join(data_dir, "test.csv"))
    ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)
    images = _load_images(os.path.join(data_dir, "test_images"),
                          [r["filename"] for r in rows], "test_images")
    return images, ids


# ----------------------------------------------------------------------------
# Augmentation, done on a numpy batch of shape (B, 32, 32, 3), dtype uint8
# ----------------------------------------------------------------------------

def random_crop_and_flip(batch, pad=4):
    """Pad with reflection, take a random 32x32 crop, randomly flip left/right."""
    b = batch.shape[0]
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                    mode="reflect")
    size = batch.shape[1]
    ys = np.random.randint(0, 2 * pad + 1, size=b)
    xs = np.random.randint(0, 2 * pad + 1, size=b)
    out = np.empty_like(batch)
    for i in range(b):
        out[i] = padded[i, ys[i]:ys[i] + size, xs[i]:xs[i] + size]
    flip = np.random.rand(b) < 0.5
    out[flip] = out[flip][:, :, ::-1]
    return out


def cutout(batch, size=8):
    """Cutout regularisation: put a black square at a random place."""
    b, h, w, _ = batch.shape
    for i in range(b):
        y = np.random.randint(0, h - size + 1)
        x = np.random.randint(0, w - size + 1)
        batch[i, y:y + size, x:x + size] = 0
    return batch


def to_tensor_batch(batch_u8):
    """uint8 (B,32,32,3) -> normalized float32 torch tensor (B,3,32,32)."""
    import torch
    x = batch_u8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(x))


def color_jitter(batch, strength=0.35):
    """Small random changes of brightness, contrast and saturation.

    Each image gets its own random factor, so one batch covers many
    different "lighting conditions". This is cheap because it is done for the
    whole batch at once with numpy.
    """
    b = batch.shape[0]
    x = batch.astype(np.float32)

    gain = np.random.uniform(1 - strength, 1 + strength, (b, 1, 1, 1))
    x = x * gain.astype(np.float32)

    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    contrast = np.random.uniform(1 - strength, 1 + strength, (b, 1, 1, 1))
    x = (x - mean) * contrast.astype(np.float32) + mean

    gray = (x * np.array([0.299, 0.587, 0.114], np.float32)).sum(-1, keepdims=True)
    sat = np.random.uniform(1 - strength, 1 + strength, (b, 1, 1, 1))
    x = gray + (x - gray) * sat.astype(np.float32)

    return np.clip(x, 0, 255).astype(np.uint8)


def add_noise(batch, sigma=0.04):
    """Add a little Gaussian noise (sigma is in units of 1/255 * 255 = pixel).

    The leaderboard images are a bit noisier than the training images, so we
    show the network noisy pictures during training as well.
    """
    x = batch.astype(np.float32)
    s = np.random.uniform(0.0, sigma * 255.0, (batch.shape[0], 1, 1, 1)).astype(np.float32)
    x = x + np.random.randn(*x.shape).astype(np.float32) * s
    return np.clip(x, 0.0, 255.0).astype(np.uint8)



def mixup(x, y, alpha=1.0, num_classes=10):
    """MixUp: blend two random images and their one-hot labels."""
    import torch
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[index]
    y1 = torch.nn.functional.one_hot(y, num_classes).float()
    y_mix = lam * y1 + (1.0 - lam) * y1[index]
    return x_mix, y_mix


def cutmix(x, y, alpha=1.0, num_classes=10):
    """CutMix: paste a random rectangle of another image, mix the labels."""
    import torch
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(x.size(0), device=x.device)
    h, w = x.size(2), x.size(3)
    cut_h, cut_w = int(h * np.sqrt(1.0 - lam)), int(w * np.sqrt(1.0 - lam))
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    y1h = torch.nn.functional.one_hot(y, num_classes).float()
    y_mix = lam * y1h + (1.0 - lam) * y1h[index]
    return x, y_mix

