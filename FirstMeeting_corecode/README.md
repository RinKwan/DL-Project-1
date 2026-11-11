# FirstMeeting Core Code — Version 1 (V1)

**Project:** Deep Learning Project 1 — CIFAR-10 Classification  
**Status:** Initial working implementation / development baseline  
**Scope of this version:** `model.py` and `evaluate.py`

## 1. Overview

This folder contains the **first working version (V1)** of the model-definition and evaluation components for our CIFAR-10 image classification project. Our aim is to build a compact ResNet-style classifier while keeping each individual model within the assignment's **5,000,000 trainable-parameter limit**.

V1 focuses on establishing a clear, testable interface before later training experiments and performance improvements. It is a **development milestone**, not a claim that the final Kaggle results have already been reproduced.

## 2. Files and Responsibilities

| File | Purpose |
| --- | --- |
| `model.py` | Defines the residual block, CIFAR-adapted ResNet architecture, model presets, and trainable-parameter counting. |
| `evaluate.py` | Loads a trained model checkpoint, evaluates a held-out validation split, and exports evaluation summaries. |
| `train.py` | Training component maintained separately; it is not implemented or changed in this V1 contribution. |
| `predict.py` | Kaggle prediction component maintained separately; it is not implemented or changed in this V1 contribution. |

### `model.py`: initial model design

- Uses a **3×3 convolutional stem** for 32×32 RGB images instead of the larger ImageNet-style stem.
- Implements a **BasicBlock** with two 3×3 convolutions, batch normalization, ReLU, and a residual skip connection.
- Uses identity shortcuts when feature shapes match; otherwise, a projection shortcut aligns dimensions.
- Organizes the network into four residual stages, followed by global average pooling and a 10-class linear classifier.
- Exposes `build_model(name)`, `count_parameters(model)`, and `PARAMETER_LIMIT` so other scripts can use a consistent interface.
- Returns **raw logits**, suitable for PyTorch `CrossEntropyLoss` (no softmax in the training forward pass).

Supported architecture names:

| Preset | Channels by stage | Trainable parameters | Meets 5M limit? |
| --- | --- | ---: | --- |
| `resnet18_w32` | 32 / 64 / 128 / 256 | 2,797,610 | Yes |
| `resnet18_w40` | 40 / 80 / 160 / 320 | 4,368,690 | Yes |
| `resnet18_w48` | 48 / 88 / 176 / 320 | 4,664,330 | Yes |
| `resnet18_w44` | 44 / 88 / 176 / 352 | 5,284,982 | **No — comparison only** |
| `resnet18_w64` | 64 / 128 / 256 / 512 | 11,173,962 | **No — comparison only** |

### `evaluate.py`: initial evaluation pipeline

- Reads the competition's `train_labels.csv` and corresponding PNG files in `train_images/`.
- Creates a reproducible held-out subset using NumPy `RandomState(42)` and the first 5,000 shuffled CSV rows **by default**. For a valid comparison, this split must match the one used during training.
- Loads an existing PyTorch checkpoint containing `model` and `state_dict` entries.
- Reports trainable parameter count, validation cross-entropy loss, and accuracy.
- Optionally evaluates **horizontal-flip test-time augmentation (TTA)** on the *same* network.
- Saves a validation confusion matrix and per-class accuracy CSV.

**Important:** The confusion matrix and per-class CSV use **non-TTA predictions**; flip-TTA accuracy is reported separately in the console.

## 3. Data and Environment

From the **repository root**, the expected data layout is:

```text
DL-Project-1/
├── FirstMeeting_corecode/
│   ├── model.py
│   ├── evaluate.py
│   └── README.md
├── data/
│   ├── train_labels.csv
│   └── train_images/
│       ├── ... .png
│       └── ...
└── runs/
    └── main/
        └── best.pt    # produced by training; not included in V1
```

Dependencies for these two scripts: **Python 3**, `torch`, `numpy`, `Pillow`, and `matplotlib`.

```bash
pip install torch numpy pillow matplotlib
```

## 4. Quick Start

Run these commands from the **repository root**.

**Check model parameter counts:**

```bash
python FirstMeeting_corecode/model.py
```

**Evaluate a checkpoint with default flip TTA:**

```bash
python FirstMeeting_corecode/evaluate.py \
  --data-dir data \
  --checkpoint runs/main/best.pt \
  --out-fig figures/confusion.png
```

**Evaluate without TTA:**

```bash
python FirstMeeting_corecode/evaluate.py \
  --data-dir data \
  --checkpoint runs/main/best.pt \
  --no-tta
```

The evaluation script creates `figures/confusion.png` and `figures/per_class.csv` unless another `--out-fig` location is supplied. It can use a CPU, or CUDA if available.

## 5. V1 Scope and Limitations

This is the **initial implementation**, so the following are *not* included as results of this version:

- No pretrained weights or completed training checkpoints are bundled with these two files.
- No final Kaggle submission score or claimed test accuracy is produced by this V1 package.
- Evaluation requires a checkpoint trained with the same model definition, preprocessing convention, and validation split.
- V1 evaluates only the project training-data validation subset; the Kaggle unlabeled test set cannot provide local classification accuracy.
- The model presets exceeding five million parameters are included for parameter comparisons **only**, not as compliant submission configurations.

## 6. Planned Follow-Up Work

After the initial version, the team can integrate training and prediction scripts, confirm split and normalization consistency, run controlled experiments on architecture width and training settings, and record real validation metrics. Later commits should document actual changes and results rather than retroactively presenting V1 as a completed final model.

---

**Version note:** `v1` documents the first functional model/evaluation interface and its reproducible evaluation workflow. Subsequent revisions should be captured in separate Git commits with specific change descriptions.
