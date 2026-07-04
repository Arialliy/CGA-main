"""Model factory for OHCM-MSHNet / CGA-v2."""
from __future__ import annotations

import torch.nn as nn

from model.MSHNet import MSHNet
from model.CGA_MSHNet import MSHNetCGA


def build_model(model_name: str = "MSHNetCGA", input_channels: int = 1, **kwargs) -> nn.Module:
    name = str(model_name).lower()
    if name in {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}:
        return MSHNetCGA(input_channels=input_channels, aux_hidden_channels=int(kwargs.get("aux_hidden_channels", 32)))
    if name in {"mshnet", "mshnetohem", "ohem"}:
        return MSHNet(input_channels=input_channels)
    raise ValueError(f"Unknown model_name={model_name!r}")


class Net(nn.Module):
    """Compatibility wrapper matching the legacy `Net(model_name=...)` pattern."""

    def __init__(self, model_name: str = "MSHNetCGA", input_channels: int = 1, **kwargs) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_model(model_name=model_name, input_channels=input_channels, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
