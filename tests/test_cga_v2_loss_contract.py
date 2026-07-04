import torch
from loss import build_loss
from net import build_model


def test_cga_loss_is_finite():
    model = build_model("MSHNetCGA")
    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.98).float()
    out = model(x, warm_flag=True, return_dict=True)
    loss_out = build_loss("MSHNetCGA")(out, y, epoch=1)
    assert torch.isfinite(loss_out["total"])
    assert "cga_center" in loss_out
