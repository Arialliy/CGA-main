"""Clean MSHNet backbone used by CGA-v2.

This file mirrors the MSHNet-style single-forward segmentation network used by
OHCM-MSHNet.  It keeps the original inference contract:

    image -> final logit -> sigmoid -> threshold 0.5

The optional ``return_dict=True`` path exposes decoder features for training-time
CGA auxiliary heads; the final inference output is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + residual)


class MSHNet(nn.Module):
    """MSHNet with optional decoder feature export."""

    def __init__(self, input_channels: int = 1, block: type[nn.Module] = ResNet) -> None:
        super().__init__()
        channels = [16, 32, 64, 128, 256]
        blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, channels[0], 1, 1)
        self.encoder_0 = self._make_layer(channels[0], channels[0], block)
        self.encoder_1 = self._make_layer(channels[0], channels[1], block, blocks[0])
        self.encoder_2 = self._make_layer(channels[1], channels[2], block, blocks[1])
        self.encoder_3 = self._make_layer(channels[2], channels[3], block, blocks[2])
        self.middle_layer = self._make_layer(channels[3], channels[4], block, blocks[3])
        self.decoder_3 = self._make_layer(channels[3] + channels[4], channels[3], block, blocks[2])
        self.decoder_2 = self._make_layer(channels[2] + channels[3], channels[2], block, blocks[1])
        self.decoder_1 = self._make_layer(channels[1] + channels[2], channels[1], block, blocks[0])
        self.decoder_0 = self._make_layer(channels[0] + channels[1], channels[0], block)
        self.output_0 = nn.Conv2d(channels[0], 1, 1)
        self.output_1 = nn.Conv2d(channels[1], 1, 1)
        self.output_2 = nn.Conv2d(channels[2], 1, 1)
        self.output_3 = nn.Conv2d(channels[3], 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, block: type[nn.Module], block_num: int = 1) -> nn.Sequential:
        layers = [block(in_channels, out_channels)]
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        return_feature: bool = False,
        return_dict: bool = False,
    ):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            masks = [mask0, mask1, mask2, mask3]
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
        else:
            masks = []
            output = self.output_0(x_d0)

        if return_dict:
            if len(masks) == 4:
                scale_logits = [masks[0], self.up(masks[1]), self.up_4(masks[2]), self.up_8(masks[3])]
            else:
                scale_logits = []
            return {
                "masks": masks,
                "base_logit": output,
                "base_logits": output,
                "scale_logits": scale_logits,
                "scale_logits_up": scale_logits,
                "decoder_feature": x_d0,
                "decoder_features": {"x_d0": x_d0, "x_d1": x_d1, "x_d2": x_d2, "x_d3": x_d3},
            }
        if return_feature:
            return masks, output, x_d0
        return masks, output
