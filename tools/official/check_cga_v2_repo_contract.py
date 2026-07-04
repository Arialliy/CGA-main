"""Smoke-test the repository-level CGA-v2 code contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loss import build_loss
from model.CGA_MSHNet import MSHNetCGA, extract_final_logit
from net import build_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="docs/internal/cga_v2/repo_contract_summary.json")
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    args = p.parse_args()
    x = torch.randn(2, 1, args.height, args.width)
    y = (torch.rand(2, 1, args.height, args.width) > 0.98).float()
    model = build_model("MSHNetCGA")
    model.train()
    out = model(x, warm_flag=True, return_dict=True)
    logit = extract_final_logit(out)
    loss_out = build_loss("MSHNetCGA")(out, y, epoch=1)
    model.eval()
    with torch.no_grad():
        eval_out = model(x, warm_flag=False, return_dict=True)
        eval_logit = extract_final_logit(eval_out)
    checks = {
        "is_mshnet_cga": isinstance(model, MSHNetCGA),
        "train_logit_shape_ok": list(logit.shape) == [2, 1, args.height, args.width],
        "eval_logit_shape_ok": list(eval_logit.shape) == [2, 1, args.height, args.width],
        "aux_keys_present": all(k in out for k in ["cga_center_logit", "cga_boundary_logit", "cga_scale_logit", "cga_peak_logit"]),
        "loss_finite": bool(torch.isfinite(loss_out["total"]).item()),
    }
    summary = {"gate": "Gate-CGA-v2-repo-contract", "gate_pass": all(checks.values()), "checks": checks}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
