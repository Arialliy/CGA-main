import pytest
import torch

from net import build_model, resolve_model_config


def test_factory_builds_mshnet_baseline_without_cga_heads():
    model = build_model(model_name=None, backbone_name="mshnet", use_cga=False)
    out = model(torch.randn(1, 1, 64, 64), mshnet_warm_flag=True)

    assert out["logits"].shape == (1, 1, 64, 64)
    assert out["adapter_meta"]["backbone"] == "mshnet"
    assert "cga_center_logit" not in out


def test_factory_builds_mshnet_cga_wrapper_with_real_cga_heads():
    model = build_model(model_name=None, backbone_name="mshnet", use_cga=True)
    out = model(torch.randn(1, 1, 64, 64), mshnet_warm_flag=True)

    assert out["regularizer_meta"]["regularizer_impl"] == "center_boundary_scale_peak"
    for key in ("cga_center_logit", "cga_boundary_logit", "cga_scale_logit", "cga_peak_logit"):
        assert key in out
        assert out[key].shape == (1, 1, 64, 64)


def test_legacy_names_resolve_to_explicit_config():
    assert resolve_model_config("MSHNetCGA") == ("mshnet", True)
    assert resolve_model_config("MSHNetOHEM") == ("mshnet", False)


def test_unaudited_backbone_fails_closed():
    with pytest.raises(ValueError, match="Add an audited explicit adapter"):
        build_model(model_name=None, backbone_name="dnanet", use_cga=True)


def test_unknown_model_name_fails_closed():
    with pytest.raises(ValueError, match="Unknown backbone_name"):
        build_model("not_a_registered_model")
