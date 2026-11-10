"""
model.py
--------
A small ResNet for CIFAR-10, written from scratch.

The design follows the ResNet-18 block diagram in the assignment:
  conv 3x3 -> [stage 1: 2 residual blocks] -> [stage 2] -> [stage 3] -> [stage 4]
  -> global average pool (P x P) -> fully connected layer -> softmax

The only difference from the original ResNet-18 is that we start with a 3x3
convolution (stride 1) instead of a 7x7 convolution + max pool, because
CIFAR-10 images are only 32x32 pixels. This is the usual CIFAR version of
ResNet and is also what the figure in the assignment shows.

All design variables from the assignment can be changed:
  N  : number of residual layers (stages)
  B  : number of residual blocks in each stage
  C  : number of channels in each stage
  F  : convolution kernel size (3x3 here)
  K  : skip connection kernel size (1x1 here, 3x3 is also possible)
  P  : average pool size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """One residual block:  ReLU( S(x) + F(x) ),  F = conv-BN-ReLU-conv-BN."""

    def __init__(self, in_ch, out_ch, stride=1, skip_kernel=1, dropout=0.0):
        super().__init__()
        # F(x): conv -> BN -> relu -> conv -> BN
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # S(x): identity, or a small conv when the shape changes
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            pad = skip_kernel // 2
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=skip_kernel, stride=stride,
                          padding=pad, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.dropout = dropout

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        out = out + identity          # <- the skip connection
        return F.relu(out, inplace=True)


class ResNetCIFAR(nn.Module):
    """ResNet-18 style network whose width can be scaled with `width`."""

    def __init__(self, num_classes=10, channels=(40, 80, 160, 320),
                 blocks=(2, 2, 2, 2), dropout=0.0, skip_kernel=1,
                 stem_channels=None):
        super().__init__()
        c1, c2, c3, c4 = channels
        stem = stem_channels if stem_channels is not None else c1

        # stem: 3x3 conv, no max pool (images are only 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
        )

        # the four residual layers (stages). stride 2 halves the feature map
        # at the beginning of stages 2, 3 and 4: 32 -> 16 -> 8 -> 4
        stage_channels = [c1, c2, c3, c4]
        stage_strides = [1, 2, 2, 2]
        in_ch = stem
        stages = []
        for n_blocks, out_ch, stride in zip(blocks, stage_channels, stage_strides):
            for b in range(n_blocks):
                stages.append(BasicBlock(in_ch, out_ch,
                                         stride=stride if b == 0 else 1,
                                         skip_kernel=skip_kernel,
                                         dropout=dropout))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # P x P average pool. After 3 stride-2 layers a 32x32 image gives 4x4.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

        # weight init (He init for conv layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    """Number of trainable parameters (the value must stay below 5,000,000)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name="resnet18_w40", num_classes=10, dropout=0.0):
    """Small factory so the training script only needs a short name."""
    presets = {
        # small baseline for the ablation study
        "resnet18_w32": dict(channels=(32, 64, 128, 256)),
        # main model: ResNet-18 with width factor 0.625  (about 4.4M params)
        "resnet18_w40": dict(channels=(40, 80, 160, 320)),
        "resnet18_w44": dict(channels=(44, 88, 176, 352)),
        # wider early stages, last stage kept at 320 channels (4.66M params)
        "resnet18_w48": dict(channels=(48, 88, 176, 320)),
        # the plain ResNet-18 (too big, only used to show the parameter limit)
        "resnet18_w64": dict(channels=(64, 128, 256, 512)),
    }
    if name not in presets:
        raise ValueError(f"unknown model {name}")
    return ResNetCIFAR(num_classes=num_classes, dropout=dropout, **presets[name])


if __name__ == "__main__":
    for name in ["resnet18_w32", "resnet18_w40", "resnet18_w44", "resnet18_w64"]:
        m = build_model(name)
        print(f"{name:14s} params = {count_parameters(m):,}")
