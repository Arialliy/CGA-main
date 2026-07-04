"""MSHNetCGA / CGA-v2 model.

Training-time component-geometry auxiliary supervision, unchanged single-forward
MSHNet inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .MSHNet import MSHNet
from .cga_aux import CGAAuxHead


def _extract_evidence_output(evidence):
    if isinstance(evidence, dict):
        final_logit = evidence.get("base_logits", evidence.get("base_logit"))
        if final_logit is None:
            raise KeyError("MSHNet return_dict=True output lacks base_logits/base_logit")
        decoder_feature = evidence.get("decoder_feature")
        if decoder_feature is None:
            decoder_feature = evidence.get("decoder_features", {}).get("x_d0")
        if decoder_feature is None:
            raise KeyError("MSHNet return_dict=True output lacks decoder_feature")
        masks = evidence.get("masks", [])
        scale_logits = evidence.get("scale_logits", evidence.get("scale_logits_up", []))
        return masks, final_logit, decoder_feature, scale_logits
    if isinstance(evidence, (tuple, list)) and len(evidence) >= 3:
        masks, final_logit, decoder_feature = evidence[:3]
        return masks, final_logit, decoder_feature, []
    raise TypeError("MSHNetCGA requires MSHNet to expose return_dict=True or return_feature=True")


class MSHNetCGA(nn.Module):
    """CGA-v2 target-preserving auxiliary-training wrapper."""

    def __init__(self, input_channels: int = 1, aux_hidden_channels: int = 32) -> None:
        super().__init__()
        self.evidence_net = MSHNet(input_channels)
        self.cga_aux_head = CGAAuxHead(in_channels=16, hidden_channels=aux_hidden_channels)

    def forward(self, x: torch.Tensor, warm_flag: bool = True, return_dict: bool = True):
        try:
            evidence = self.evidence_net(x, warm_flag=warm_flag, return_dict=True)
        except TypeError:
            evidence = self.evidence_net(x, warm_flag=warm_flag, return_feature=True)
        masks, final_logit, decoder_feature, scale_logits = _extract_evidence_output(evidence)
        aux_outputs = self.cga_aux_head(decoder_feature)
        out = {
            "final_logit": final_logit,
            "final_logits": final_logit,
            "base_logit": final_logit,
            "base_logits": final_logit,
            "masks": masks,
            "scale_logits": scale_logits,
            "scale_logits_up": scale_logits,
            "decoder_feature": decoder_feature,
            "aux_outputs": aux_outputs,
            **aux_outputs,
        }
        if return_dict:
            return out
        return masks, final_logit, aux_outputs


def configure_cga_trainable(model: nn.Module, mode: str = "all") -> list[str]:
    """Optionally freeze parts of CGA for controlled experiments."""
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "aux_only":
        for name, p in model.named_parameters():
            p.requires_grad = "cga_aux_head" in name
    elif mode == "decoder_aux":
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in ["cga_aux_head", "decoder_0", "output_0", "final"])
    else:
        raise ValueError(f"Unknown CGA trainable mode: {mode}")
    return [name for name, p in model.named_parameters() if p.requires_grad]


def extract_final_logit(output) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("final_logit", "final_logits", "base_logits", "base_logit", "logits"):
            if key in output:
                return output[key]
        raise KeyError(f"Could not find final logit key in output keys: {sorted(output.keys())}")
    if isinstance(output, (tuple, list)):
        if len(output) >= 2 and torch.is_tensor(output[1]):
            return output[1]
        if len(output) >= 1 and torch.is_tensor(output[-1]):
            return output[-1]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported model output type: {type(output)!r}")
