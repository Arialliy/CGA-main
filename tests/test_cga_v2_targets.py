import torch
from utils.cga_targets import build_cga_targets


def test_cga_targets_nonempty_for_component():
    y = torch.zeros(1, 1, 32, 32)
    y[:, :, 10:13, 11:14] = 1
    targets = build_cga_targets(y)
    assert targets["cga_center_target"].sum() > 0
    assert targets["cga_boundary_target"].sum() > 0
    assert targets["cga_peak_target"].sum() > 0
    assert targets["cga_scale_target"].max() > 0
