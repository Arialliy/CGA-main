"""Model factory for fail-closed CGA paper-evidence experiments."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.CGA_MSHNet import MSHNetCGA
from model.cga_wrapper import CGAWrapper
from model.registry import available_backbones, get_backbone_builder


_CGA_MODEL_ALIASES = {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}
_MSHNET_BASE_ALIASES = {"mshnet", "mshnetohem", "ohem"}


def resolve_model_config(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    use_cga: bool = False,
) -> tuple[str, bool]:
    """Resolve legacy model names into the explicit backbone/CGA switches."""
    if model_name is not None:
        name = str(model_name).lower()
        if name in _CGA_MODEL_ALIASES:
            return "mshnet", True
        if name in _MSHNET_BASE_ALIASES:
            return "mshnet", False
        if name.endswith("_cga"):
            return name[: -len("_cga")], True
        if name in available_backbones():
            return name, bool(use_cga)
        return name, bool(use_cga)
    return str(backbone_name).lower(), bool(use_cga)


def build_model(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    input_channels: int = 1,
    use_cga: bool = False,
    aux_hidden_channels: int = 32,
    evidence_mode: str = "paper",
    legacy_model_factory: bool = False,
    **kwargs: Any,
) -> nn.Module:
    if evidence_mode not in {"paper", "smoke"}:
        raise ValueError(f"Unknown evidence_mode={evidence_mode!r}")
    if evidence_mode == "paper" and kwargs.get("allow_fallback_regularizer", False):
        raise RuntimeError(
            "Fallback regularizer is forbidden for paper evidence. "
            "Run smoke tests under evidence_mode='smoke' only."
        )

    resolved_backbone, resolved_use_cga = resolve_model_config(
        model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
    )

    if legacy_model_factory:
        if resolved_backbone == "mshnet" and resolved_use_cga:
            return MSHNetCGA(
                input_channels=input_channels,
                aux_hidden_channels=int(aux_hidden_channels),
            )
        raise ValueError(
            "legacy_model_factory only supports the legacy MSHNetCGA path. "
            "Use the explicit adapter registry for other configurations."
        )

    builder = get_backbone_builder(resolved_backbone)
    backbone = builder(input_channels=input_channels)
    if not resolved_use_cga:
        return backbone

    feature_channels = getattr(backbone, "FEATURE_CHANNELS", None)
    if feature_channels is None:
        raise AttributeError(f"{resolved_backbone} adapter must define FEATURE_CHANNELS for CGAWrapper")
    return CGAWrapper(
        backbone,
        backbone_name=resolved_backbone,
        feature_channels=int(feature_channels),
        aux_hidden_channels=int(aux_hidden_channels),
    )


class Net(nn.Module):
    """Compatibility wrapper matching the legacy `Net(model_name=...)` pattern."""

    def __init__(self, model_name: str | None = "MSHNetCGA", input_channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_model(model_name=model_name, input_channels=input_channels, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
