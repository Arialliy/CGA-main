import pytest
import torch

from loss import build_loss
from net import build_model
from train import compute_paper_evidence_allowed


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


def test_paper_evidence_allowed_requires_p1_p1a_and_no_fallback():
    assert not compute_paper_evidence_allowed(
        evidence_mode="smoke",
        p1_preflight_passed=True,
        p1a_hcval_source_audit_passed=True,
        fallback_regularizer_used=False,
    )
    assert not compute_paper_evidence_allowed(
        evidence_mode="paper",
        p1_preflight_passed=False,
        p1a_hcval_source_audit_passed=True,
        fallback_regularizer_used=False,
    )
    assert not compute_paper_evidence_allowed(
        evidence_mode="paper",
        p1_preflight_passed=True,
        p1a_hcval_source_audit_passed=False,
        fallback_regularizer_used=False,
    )
    assert not compute_paper_evidence_allowed(
        evidence_mode="paper",
        p1_preflight_passed=True,
        p1a_hcval_source_audit_passed=True,
        fallback_regularizer_used=True,
    )
    assert compute_paper_evidence_allowed(
        evidence_mode="paper",
        p1_preflight_passed=True,
        p1a_hcval_source_audit_passed=True,
        fallback_regularizer_used=False,
    )
