"""Backbone-agnostic real CGA auxiliary-head wrapper."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.cga_aux import CGAAuxHead
from model.output_contract import validate_detector_output


class CGAWrapper(nn.Module):
    """Backbone + center/boundary/scale/peak CGA auxiliary heads.

    This is the paper-evidence CGA implementation.  It deliberately contains no
    fallback regularizer path.
    """

    REGULARIZER_IMPL = "center_boundary_scale_peak"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_name: str,
        feature_channels: int,
        aux_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.cga_aux_head = CGAAuxHead(
            in_channels=int(feature_channels),
            hidden_channels=int(aux_hidden_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        output = self.backbone(x, **kwargs)
        output = validate_detector_output(
            output,
            backbone_name=self.backbone_name,
            require_feature=True,
        )

        feat = output["features"][0]
        aux = self.cga_aux_head(feat)

        output = dict(output)
        output["aux_outputs"] = aux
        output.update(aux)
        output.setdefault("regularizer_meta", {})
        output["regularizer_meta"].update(
            {
                "use_cga": True,
                "regularizer_impl": self.REGULARIZER_IMPL,
                "paper_evidence_allowed": True,
            }
        )
        return output
