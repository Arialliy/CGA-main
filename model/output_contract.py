"""Explicit detector output contract for paper-evidence CGA experiments.

Adapters must state which tensors are logits and CGA features.  This module
intentionally does not inspect arbitrary tuples/lists or pick the first 4D
tensor, because that can silently train on the wrong detector output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureSpec:
    source: str
    stride: int
    channels: int
    resolution: tuple[int, int]


def _as_hw(x: Any) -> tuple[int, int]:
    if isinstance(x, torch.Tensor):
        return int(x.shape[-2]), int(x.shape[-1])
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise TypeError(f"Invalid resolution format: {type(x)!r}")


def validate_detector_output(
    output: dict[str, Any],
    *,
    backbone_name: str,
    require_feature: bool,
) -> dict[str, Any]:
    """Validate explicit detector output without guessing tensor semantics."""
    if not isinstance(output, dict):
        raise TypeError(
            f"{backbone_name} adapter must return a dict, got {type(output)!r}. "
            "Raw tuple/list/tensor outputs are not allowed in paper mode."
        )

    if "logits" not in output:
        raise KeyError(
            f"{backbone_name} adapter output lacks required key 'logits'. "
            "Do not infer logits from arbitrary tensors."
        )

    logits = output["logits"]
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"{backbone_name} logits must be Tensor[B,1,H,W], got "
            f"{getattr(logits, 'shape', None)}"
        )

    if "adapter_meta" not in output:
        raise KeyError(f"{backbone_name} adapter output lacks required key 'adapter_meta'.")
    adapter_meta = output["adapter_meta"]
    if not isinstance(adapter_meta, dict):
        raise TypeError(f"{backbone_name} adapter_meta must be a dict.")
    required_adapter_meta = {"backbone", "logits_source", "feature_source"}
    missing_adapter_meta = required_adapter_meta - set(adapter_meta.keys())
    if missing_adapter_meta:
        raise KeyError(
            f"{backbone_name} adapter_meta missing fields: {sorted(missing_adapter_meta)}"
        )

    if not require_feature:
        return output

    if "features" not in output or "feature_meta" not in output:
        raise KeyError(
            f"{backbone_name}+CGA requires explicit 'features' and 'feature_meta'."
        )

    features = output["features"]
    metas = output["feature_meta"]
    if not isinstance(features, (list, tuple)) or len(features) != 1:
        raise ValueError(
            f"{backbone_name}+CGA expects exactly one selected feature for controlled protocol."
        )
    if not isinstance(metas, (list, tuple)) or len(metas) != 1:
        raise ValueError(f"{backbone_name}+CGA expects exactly one feature_meta entry.")

    feat = features[0]
    meta = metas[0]
    if not torch.is_tensor(feat) or feat.ndim != 4:
        raise ValueError(
            f"{backbone_name} selected CGA feature must be Tensor[B,C,H,W], got "
            f"{getattr(feat, 'shape', None)}"
        )
    if not isinstance(meta, dict):
        raise TypeError(f"{backbone_name} feature_meta entry must be a dict.")

    required_meta = {"source", "stride", "channels", "resolution"}
    missing = required_meta - set(meta.keys())
    if missing:
        raise KeyError(f"{backbone_name} feature_meta missing fields: {sorted(missing)}")

    if int(meta["channels"]) != int(feat.shape[1]):
        raise ValueError(
            f"{backbone_name} feature_meta channels={meta['channels']} "
            f"but feature.shape[1]={feat.shape[1]}"
        )

    if _as_hw(meta["resolution"]) != (int(feat.shape[-2]), int(feat.shape[-1])):
        raise ValueError(
            f"{backbone_name} feature_meta resolution={meta['resolution']} "
            f"but feature.shape[-2:]={tuple(feat.shape[-2:])}"
        )

    return output
