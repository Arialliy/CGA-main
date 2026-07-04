"""Training entry point for controlled fail-closed CGA experiments."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TrainSetLoader
from loss import build_loss
from net import build_model, resolve_model_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _loss_value(loss_out):
    return loss_out["total"] if isinstance(loss_out, dict) else loss_out


def compute_paper_evidence_allowed(
    *,
    evidence_mode: str,
    p1_preflight_passed: bool,
    p1a_hcval_source_audit_passed: bool,
    fallback_regularizer_used: bool,
) -> bool:
    if evidence_mode == "smoke":
        return False
    if evidence_mode != "paper":
        raise ValueError(f"Unknown evidence_mode={evidence_mode!r}")
    return bool(
        p1_preflight_passed
        and p1a_hcval_source_audit_passed
        and not fallback_regularizer_used
    )


def _output_fallback_regularizer_used(output) -> bool:
    if not isinstance(output, dict):
        return False
    regularizer_meta = output.get("regularizer_meta", {})
    return bool(
        output.get("fallback_regularizer_used", False)
        or (isinstance(regularizer_meta, dict) and regularizer_meta.get("fallback_regularizer_used", False))
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train MSHNet/CGA-v2")
    p.add_argument("--model_name", default=None)
    p.add_argument("--backbone_name", default="mshnet", choices=["mshnet", "dnanet", "alcnet", "acm", "isnet"])
    p.add_argument("--use_cga", action="store_true")
    p.add_argument("--evidence_mode", default="paper", choices=["paper", "smoke"])
    p.add_argument("--protocol", default="controlled", choices=["controlled", "official"])
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warm_epoch", type=int, default=None, help="Deprecated alias for --mshnet_warm_epoch")
    p.add_argument("--mshnet_warm_epoch", type=int, default=None)
    p.add_argument("--cga_start_epoch", type=int, default=1)
    p.add_argument("--cga_ramp_epochs", type=int, default=40)
    p.add_argument("--lambda_center", type=float, default=0.05)
    p.add_argument("--lambda_boundary", type=float, default=0.03)
    p.add_argument("--lambda_scale", type=float, default=0.02)
    p.add_argument("--lambda_peak", type=float, default=0.03)
    p.add_argument("--aux_hidden_channels", type=int, default=32)
    p.add_argument("--allow_fallback_regularizer", action="store_true")
    p.add_argument("--p1_preflight_passed", action="store_true")
    p.add_argument("--p1a_hcval_source_audit_passed", action="store_true")
    p.add_argument("--ohem_ratio", type=float, default=0.01)
    p.add_argument("--output_dir", default="results/official")
    p.add_argument("--resume", default="")
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def _run_model_name(model_name: str | None, backbone_name: str, use_cga: bool) -> str:
    if model_name:
        return str(model_name)
    return f"{backbone_name}_cga" if use_cga else backbone_name


def main() -> None:
    args = parse_args()
    if args.mshnet_warm_epoch is None:
        args.mshnet_warm_epoch = int(args.warm_epoch if args.warm_epoch is not None else 5)
    if args.evidence_mode == "paper" and args.allow_fallback_regularizer:
        raise RuntimeError(
            "Fallback regularizer is forbidden for paper evidence. "
            "Use --evidence_mode smoke for smoke-only plumbing tests."
        )
    if args.evidence_mode == "paper" and (
        not args.p1_preflight_passed or not args.p1a_hcval_source_audit_passed
    ):
        print("[WARN] paper_evidence_allowed will remain false until P1 and P1A pass.")
    if args.evidence_mode == "paper" and args.protocol != "controlled":
        print("[WARN] official/literature protocol is contextual only; do not use it as main CGA claim.")

    backbone_name, use_cga = resolve_model_config(
        args.model_name,
        backbone_name=args.backbone_name,
        use_cga=args.use_cga,
    )
    run_model_name = _run_model_name(args.model_name, backbone_name, use_cga)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / run_model_name / f"seed{args.seed}" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TrainSetLoader(args.dataset_dir, args.dataset_name, patch_size=args.patch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    model = build_model(
        model_name=args.model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        evidence_mode=args.evidence_mode,
        input_channels=1,
        aux_hidden_channels=args.aux_hidden_channels,
        allow_fallback_regularizer=args.allow_fallback_regularizer,
    ).to(device)
    criterion = build_loss(
        args.model_name or backbone_name,
        use_cga=use_cga,
        ohem_ratio=args.ohem_ratio,
        mshnet_warm_epoch=args.mshnet_warm_epoch,
        cga_start_epoch=args.cga_start_epoch,
        cga_ramp_epochs=args.cga_ramp_epochs,
        lambda_center=args.lambda_center,
        lambda_boundary=args.lambda_boundary,
        lambda_scale=args.lambda_scale,
        lambda_peak=args.lambda_peak,
        strict_cga_heads=(args.evidence_mode == "paper" and use_cga),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    log_path = out_dir / "train_log.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        stats = []
        fallback_regularizer_used = bool(args.allow_fallback_regularizer)
        for img, mask in loader:
            img = img.float().to(device)
            mask = mask.float().to(device)
            optim.zero_grad(set_to_none=True)
            forward_kwargs = {}
            if backbone_name == "mshnet":
                forward_kwargs["mshnet_warm_flag"] = epoch <= args.mshnet_warm_epoch
            output = model(img, **forward_kwargs)
            fallback_regularizer_used = fallback_regularizer_used or _output_fallback_regularizer_used(output)
            loss_out = criterion(output, mask, epoch=epoch)
            loss = _loss_value(loss_out)
            loss.backward()
            optim.step()
            row = {k: float(v.detach().cpu()) for k, v in loss_out.items() if torch.is_tensor(v) and v.numel() == 1}
            stats.append(row)
        mean_stats = {}
        if stats:
            for key in stats[0].keys():
                mean_stats[key] = float(np.mean([s.get(key, 0.0) for s in stats]))
        paper_evidence_allowed = compute_paper_evidence_allowed(
            evidence_mode=args.evidence_mode,
            p1_preflight_passed=args.p1_preflight_passed,
            p1a_hcval_source_audit_passed=args.p1a_hcval_source_audit_passed,
            fallback_regularizer_used=fallback_regularizer_used,
        )
        evidence_meta = {
            "epoch": epoch,
            "dataset": args.dataset_name,
            "model": run_model_name,
            "backbone": backbone_name,
            "use_cga": bool(use_cga),
            "regularizer_impl": "center_boundary_scale_peak" if use_cga else "none",
            "evidence_mode": args.evidence_mode,
            "p1_preflight_passed": bool(args.p1_preflight_passed),
            "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
            "fallback_regularizer_used": bool(fallback_regularizer_used),
            "paper_evidence_allowed": bool(paper_evidence_allowed),
            "protocol": args.protocol,
            "seed": args.seed,
            "mshnet_warm_epoch": args.mshnet_warm_epoch,
            "cga_start_epoch": args.cga_start_epoch,
            "cga_ramp_epochs": args.cga_ramp_epochs,
        }
        mean_stats.update(evidence_meta)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(mean_stats, sort_keys=True) + "\n")
        if epoch == args.epochs or epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_name": run_model_name,
                "backbone": backbone_name,
                "use_cga": bool(use_cga),
                "regularizer_impl": "center_boundary_scale_peak" if use_cga else "none",
                "evidence_mode": args.evidence_mode,
                "p1_preflight_passed": bool(args.p1_preflight_passed),
                "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
                "fallback_regularizer_used": bool(fallback_regularizer_used),
                "paper_evidence_allowed": bool(paper_evidence_allowed),
                "protocol": args.protocol,
                "dataset": args.dataset_name,
                "seed": args.seed,
                "mshnet_warm_epoch": args.mshnet_warm_epoch,
                "cga_start_epoch": args.cga_start_epoch,
                "cga_ramp_epochs": args.cga_ramp_epochs,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
            }, out_dir / f"{run_model_name}_{epoch}.pth.tar")
        print(json.dumps(mean_stats, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
