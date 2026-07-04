import pytest
import torch

from loss import build_loss
from net import build_model


def test_paper_mode_forbids_fallback_regularizer():
    with pytest.raises(RuntimeError, match="Fallback regularizer is forbidden"):
        build_model(
            model_name=None,
            backbone_name="mshnet",
            use_cga=True,
            evidence_mode="paper",
            allow_fallback_regularizer=True,
        )


def test_cga_loss_requires_all_four_heads_in_paper_mode():
    criterion = build_loss(use_cga=True, strict_cga_heads=True)
    output = {
        "logits": torch.randn(2, 1, 64, 64),
        "cga_center_logit": torch.randn(2, 1, 64, 64),
    }
    target = torch.zeros(2, 1, 64, 64)

    with pytest.raises(KeyError, match="cga_boundary_logit"):
        criterion(output, target, epoch=1)
