"""Training entry point for MSHNetOHEM and MSHNetCGA."""
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
from net import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _loss_value(loss_out):
    return loss_out["total"] if isinstance(loss_out, dict) else loss_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train MSHNet/CGA-v2")
    p.add_argument("--model_name", default="MSHNetCGA")
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--warm_epoch", type=int, default=5)
    p.add_argument("--ohem_ratio", type=float, default=0.01)
    p.add_argument("--output_dir", default="results/official")
    p.add_argument("--resume", default="")
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / args.model_name / f"seed{args.seed}" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = TrainSetLoader(args.dataset_dir, args.dataset_name, patch_size=args.patch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    model = build_model(args.model_name).to(device)
    criterion = build_loss(args.model_name, ohem_ratio=args.ohem_ratio, warm_epoch=args.warm_epoch).to(device)
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
        for img, mask in loader:
            img = img.float().to(device)
            mask = mask.float().to(device)
            optim.zero_grad(set_to_none=True)
            output = model(img, warm_flag=(epoch <= args.warm_epoch), return_dict=True)
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
        mean_stats.update({"epoch": epoch, "dataset": args.dataset_name, "model": args.model_name, "seed": args.seed})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(mean_stats, sort_keys=True) + "\n")
        if epoch == args.epochs or epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_name": args.model_name,
                "dataset": args.dataset_name,
                "seed": args.seed,
                "state_dict": model.state_dict(),
                "optimizer": optim.state_dict(),
            }, out_dir / f"{args.model_name}_{epoch}.pth.tar")
        print(json.dumps(mean_stats, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
