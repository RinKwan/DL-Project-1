"""CIFAR-10 ResNet model definitions (initial working version).

The project uses 32x32 RGB images and permits at most 5 million trainable
parameters.  This module exposes the same build_model/count_parameters API
as the team's training and prediction scripts.

The forward method returns raw logits.  CrossEntropyLoss expects logits;
softmax is only needed when class probabilities are requested at inference.
"""

import torch
from torch import nn
from torch.nn import functional as F

PARAMETER_LIMIT = 5_000_000


class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus an identity or projected shortcut."""

    def __init__(self, in_ch, out_ch, stride=1, skip_kernel=1, dropout=0.0):
        super().__init__()
        if skip_kernel not in (1, 3):
            raise ValueError("skip_kernel must be 1 or 3")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # A projection is only required if the input/output shapes differ.
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, skip_kernel, stride=stride,
                          padding=skip_kernel // 2, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.dropout = dropout

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return F.relu(out + identity, inplace=True)


class ResNetCIFAR(nn.Module):
    """ResNet-18-style network with four configurable residual stages."""

    def __init__(self, num_classes=10, channels=(40, 80, 160, 320),
                 blocks=(2, 2, 2, 2), dropout=0.0, skip_kernel=1,
                 stem_channels=None):
        super().__init__()
        if len(channels) != 4 or len(blocks) != 4:
            raise ValueError("channels and blocks must each have four entries")
        if any(c <= 0 for c in channels) or any(b < 1 for b in blocks):
            raise ValueError("channels and blocks must contain positive integers")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        stem = channels[0] if stem_channels is None else stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
        )

        # Images shrink from 32 -> 16 -> 8 -> 4 across stages 2, 3, 4.
        layers = []
        in_ch = stem
        for stage, (n_blocks, out_ch) in enumerate(zip(blocks, channels)):
            first_stride = 1 if stage == 0 else 2
            for block_idx in range(n_blocks):
                layers.append(BasicBlock(in_ch, out_ch,
                                         stride=first_stride if block_idx == 0 else 1,
                                         skip_kernel=skip_kernel,
                                         dropout=dropout))
                in_ch = out_ch
        self.stages = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def count_parameters(model):
    """Count only trainable parameters (including classifier biases)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name="resnet18_w40", num_classes=10, dropout=0.0):
    """Build an architecture by the name stored in training checkpoints."""
    presets = {
        "resnet18_w32": (32, 64, 128, 256),
        "resnet18_w40": (40, 80, 160, 320),
        "resnet18_w44": (44, 88, 176, 352),  # over budget: comparison only
        "resnet18_w48": (48, 88, 176, 320),
        "resnet18_w64": (64, 128, 256, 512),  # over budget: comparison only
    }
    if name not in presets:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(presets)}")
    return ResNetCIFAR(num_classes=num_classes, channels=presets[name],
                       dropout=dropout)


if __name__ == "__main__":
    for name in ("resnet18_w32", "resnet18_w40", "resnet18_w48",
                 "resnet18_w44", "resnet18_w64"):
        model = build_model(name)
        params = count_parameters(model)
        status = "OK" if params <= PARAMETER_LIMIT else "OVER LIMIT (do not submit)"
        print(f"{name:15s}: {params:>11,} trainable parameters  {status}")
