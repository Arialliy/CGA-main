import torch
from net import build_model
from model.CGA_MSHNet import extract_final_logit


def test_mshnet_cga_forward_contract():
    model = build_model("MSHNetCGA")
    x = torch.randn(2, 1, 64, 64)
    out = model(x, warm_flag=True, return_dict=True)
    assert extract_final_logit(out).shape == (2, 1, 64, 64)
    for key in ["cga_center_logit", "cga_boundary_logit", "cga_scale_logit", "cga_peak_logit"]:
        assert key in out
        assert out[key].shape == (2, 1, 64, 64)


def test_mshnet_eval_still_final_logit_only_available():
    model = build_model("MSHNetCGA")
    x = torch.randn(1, 1, 64, 64)
    model.eval()
    with torch.no_grad():
        out = model(x, warm_flag=False, return_dict=True)
    assert extract_final_logit(out).shape == (1, 1, 64, 64)
