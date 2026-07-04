"""Inference/test entry point for MSHNetCGA and MSHNetOHEM."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import TestSetLoader
from metrics import IRSTDMetrics
from model.CGA_MSHNet import extract_final_logit
from net import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Test MSHNet/CGA-v2")
    p.add_argument("--model_name", default="MSHNetCGA")
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--train_dataset_name", default=None)
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--output_dir", default="results/official")
    return p.parse_args()


def crop_to_size(arr: torch.Tensor, size) -> torch.Tensor:
    h, w = int(size[0]), int(size[1])
    return arr[..., :h, :w]


def first_size(size) -> tuple[int, int]:
    if torch.is_tensor(size):
        if size.ndim == 2:
            size = size[0]
        return int(size[0].item()), int(size[1].item())
    if isinstance(size, (list, tuple)) and len(size) == 2 and torch.is_tensor(size[0]):
        return int(size[0][0].item()), int(size[1][0].item())
    if isinstance(size, (list, tuple)) and len(size) == 1:
        return first_size(size[0])
    return int(size[0]), int(size[1])


def save_prob_png(prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    train_name = args.train_dataset_name or args.dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()

    ds = TestSetLoader(args.dataset_dir, train_name, args.dataset_name, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    pred_dir = Path(args.output_dir) / args.model_name / f"seed{args.seed}" / args.dataset_name / args.split / "predictions"
    metric = IRSTDMetrics(threshold=args.threshold)
    with torch.no_grad():
        for img, mask, size, image_id in loader:
            img = img.float().to(device)
            output = model(img, warm_flag=False, return_dict=True)
            logit = extract_final_logit(output)
            prob = torch.sigmoid(logit).cpu()
            original_size = first_size(size)
            prob = crop_to_size(prob, original_size).squeeze().numpy()
            gt = crop_to_size(mask.float(), original_size).squeeze().numpy()
            metric.update(prob, gt, size=original_size)
            save_prob_png(prob, pred_dir / f"{image_id[0]}.png")
    summary = metric.get()
    checkpoint_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    summary.update({
        "model": args.model_name,
        "train_dataset": train_name,
        "dataset": args.dataset_name,
        "split": args.split,
        "seed": args.seed,
        "epoch": checkpoint_epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "threshold": args.threshold,
        "checkpoint": str(args.checkpoint),
        "prediction_dir": str(pred_dir),
    })
    out_path = Path(args.output_dir) / args.model_name / f"seed{args.seed}" / args.dataset_name / args.split / "summary_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.split == "test":
        compat_path = Path(args.output_dir) / args.model_name / f"seed{args.seed}" / args.dataset_name / "summary_metrics.json"
        compat_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
