"""Coordinate Attention module and helper activations.

This module is the single source of truth for the CoordAtt layer used in
both training (scripts/train_final_phase1.py) and inference/profiling
(src/edge/profiler.py, src/edge/vlm_trigger.py).

Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network
Design", CVPR 2021.
"""

import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    """Hard Sigmoid activation: relu6(x + 3) / 6."""

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.0) / 6.0


class HSwish(nn.Module):
    """Hard Swish activation: x * HSigmoid(x)."""

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.hsigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.hsigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention block with lazy initialisation.

    Performs separate X-axis and Y-axis average pooling, runs through a
    shared 1x1 bottleneck, then produces two multiplicative attention
    maps (Ax, Ay) that modulate the input feature map.

    Lazy init allows the same class definition to work with varying
    channel counts when parsed from a YAML architecture config.
    """

    def __init__(self, inp: int = 0, reduction: int = 32):
        super().__init__()
        self.inp = int(inp) if inp else None
        self.reduction = max(1, int(reduction))

        self.conv1: nn.Conv2d | None = None
        self.bn1: nn.BatchNorm2d | None = None
        self.act = HSwish()
        self.conv_h: nn.Conv2d | None = None
        self.conv_w: nn.Conv2d | None = None

    def _build(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        mip = max(8, channels // self.reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0).to(
            device=device, dtype=dtype
        )
        self.bn1 = nn.BatchNorm2d(mip).to(device=device, dtype=dtype)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0).to(
            device=device, dtype=dtype
        )
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0).to(
            device=device, dtype=dtype
        )
        self.inp = channels

    def _ensure_built(self, x) -> None:
        channels = int(x.shape[1])
        if self.conv1 is None or self.inp != channels:
            self._build(channels, x.device, x.dtype)

    def forward(self, x):
        self._ensure_built(x)
        assert self.conv1 is not None
        assert self.bn1 is not None
        assert self.conv_h is not None
        assert self.conv_w is not None

        identity = x
        _, _, h, w = x.size()

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


def register_coordatt() -> None:
    """Register CoordAtt (and helpers) with the ultralytics task parser.

    Must be called before loading a YAML config or a .pt checkpoint that
    contains CoordAtt layers.
    """
    import ultralytics.nn.tasks as tasks

    for cls in (HSigmoid, HSwish, CoordAtt):
        setattr(tasks, cls.__name__, cls)
