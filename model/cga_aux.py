"""CGA-v2 auxiliary heads.

The heads are used only during training.  Inference still consumes only the
MSHNet final logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CGAAuxHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, use_bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        self.shared = nn.Sequential(*layers)
        self.center_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.scale_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.peak_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(feat)
        return {
            "cga_center_logit": self.center_head(h),
            "cga_boundary_logit": self.boundary_head(h),
            "cga_scale_logit": self.scale_head(h),
            "cga_peak_logit": self.peak_head(h),
        }
