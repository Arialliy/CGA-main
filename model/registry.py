"""Backbone registry for explicit CGA adapters."""
from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from model.backbones.mshnet_adapter import MSHNetAdapter

BackboneBuilder = Callable[..., nn.Module]

BACKBONE_BUILDERS: dict[str, BackboneBuilder] = {
    "mshnet": MSHNetAdapter,
}


def available_backbones() -> list[str]:
    return sorted(BACKBONE_BUILDERS)


def get_backbone_builder(backbone_name: str) -> BackboneBuilder:
    name = str(backbone_name).lower()
    if name not in BACKBONE_BUILDERS:
        raise ValueError(
            f"Unknown backbone_name={backbone_name!r}. "
            f"Available: {available_backbones()}. "
            "Add an audited explicit adapter before using this backbone for paper evidence."
        )
    return BACKBONE_BUILDERS[name]
