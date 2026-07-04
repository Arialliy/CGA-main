"""MSHNet adapter with explicit logits/feature sources."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.MSHNet import MSHNet
from model.output_contract import validate_detector_output


class MSHNetAdapter(nn.Module):
    BACKBONE_NAME = "mshnet"
    LOGITS_SOURCE = "base_logits/base_logit"
    FEATURE_SOURCE = "decoder_features.x_d0"
    FEATURE_STRIDE = 1
    FEATURE_CHANNELS = 16

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.net = MSHNet(input_channels=input_channels)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mshnet_warm_flag: bool | None = None,
        warm_flag: bool | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        del return_dict
        if mshnet_warm_flag is None:
            mshnet_warm_flag = True if warm_flag is None else bool(warm_flag)

        raw = self.net(x, warm_flag=bool(mshnet_warm_flag), return_dict=True)
        if not isinstance(raw, dict):
            raise TypeError(
                "MSHNetAdapter requires MSHNet return_dict=True output. "
                "Do not silently parse tuple/list outputs in paper mode."
            )

        logits = raw.get("base_logits", raw.get("base_logit"))
        if logits is None:
            raise KeyError("MSHNet raw output lacks base_logits/base_logit")

        feat = raw.get("decoder_feature")
        if feat is None:
            decoder_features = raw.get("decoder_features", {})
            if not isinstance(decoder_features, dict) or "x_d0" not in decoder_features:
                raise KeyError("MSHNet raw output lacks decoder_feature or decoder_features['x_d0']")
            feat = decoder_features["x_d0"]

        output = {
            "logits": logits,
            "final_logit": logits,
            "final_logits": logits,
            "base_logit": logits,
            "base_logits": logits,
            "features": [feat],
            "feature_meta": [
                {
                    "source": self.FEATURE_SOURCE,
                    "stride": self.FEATURE_STRIDE,
                    "channels": int(feat.shape[1]),
                    "resolution": [int(feat.shape[-2]), int(feat.shape[-1])],
                }
            ],
            "adapter_meta": {
                "backbone": self.BACKBONE_NAME,
                "logits_source": self.LOGITS_SOURCE,
                "feature_source": self.FEATURE_SOURCE,
            },
            "masks": raw.get("masks", []),
            "scale_logits": raw.get("scale_logits", raw.get("scale_logits_up", [])),
            "scale_logits_up": raw.get("scale_logits_up", raw.get("scale_logits", [])),
        }
        return validate_detector_output(
            output,
            backbone_name=self.BACKBONE_NAME,
            require_feature=True,
        )
