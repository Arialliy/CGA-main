#!/usr/bin/env python3
"""Export final-logit probability maps for HC-Val mining.

This is the CGA-main-native counterpart of OHCM's export_step0_predictions.py.
It keeps the same output contract needed by analyze_step1_hard_clutter.py:
`probs/<id>.npy` and `logits/<id>.npy`, plus PNG masks/probability maps.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import TestSetLoader, get_dataset_entry, normalize_item_id
from metrics import IRSTDMetrics
from model.CGA_MSHNet import extract_final_logit
from net import build_model, resolve_model_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export final-logit probability maps")
    p.add_argument("--model_name", default=None)
    p.add_argument("--backbone_name", default="mshnet", choices=["mshnet", "dnanet", "alcnet", "acm", "isnet"])
    p.add_argument("--use_cga", action="store_true")
    p.add_argument("--evidence_mode", default="paper", choices=["paper", "smoke"])
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--train_dataset_name", default=None)
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--split", default="test")
    p.add_argument("--image_list", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def _run_model_name(model_name: str | None, backbone_name: str, use_cga: bool) -> str:
    if model_name:
        return str(model_name)
    return f"{backbone_name}_cga" if use_cga else backbone_name


def _first_size(size: Any, index: int) -> tuple[int, int]:
    if torch.is_tensor(size):
        if size.ndim == 2:
            return int(size[index, 0].item()), int(size[index, 1].item())
        return int(size[0].item()), int(size[1].item())
    if isinstance(size, (list, tuple)) and len(size) == 2 and torch.is_tensor(size[0]):
        return int(size[0][index].item()), int(size[1][index].item())
    if isinstance(size, (list, tuple)) and len(size) == 1:
        return _first_size(size[0], index)
    return int(size[0]), int(size[1])


def _image_id_at(image_id: Any, index: int) -> str:
    if isinstance(image_id, (list, tuple)):
        return str(image_id[index])
    if isinstance(image_id, np.ndarray):
        return str(image_id.reshape(-1)[index])
    return str(image_id)


def _crop(array: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return array[..., :h, :w]


def _save_prob_png(prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8)).save(path)


def _save_mask_png(prob: np.ndarray, threshold: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((prob > threshold).astype(np.uint8) * 255)).save(path)


def _read_image_list(path: Path, dataset_name: str) -> list[str]:
    entry = get_dataset_entry(dataset_name)
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = normalize_item_id(raw, entry)
        if item:
            items.append(item)
    return items


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_name = args.train_dataset_name or args.dataset_name
    backbone_name, use_cga = resolve_model_config(
        args.model_name,
        backbone_name=args.backbone_name,
        use_cga=args.use_cga,
    )
    run_model_name = _run_model_name(args.model_name, backbone_name, use_cga)
    output_dir = Path(args.output_dir)
    for subdir in ("probs", "logits", "masks"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=args.model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        evidence_mode=args.evidence_mode,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = TestSetLoader(args.dataset_dir, train_name, args.dataset_name, split=args.split)
    if args.image_list:
        dataset.items = _read_image_list(Path(args.image_list), args.dataset_name)
        if not dataset.items:
            raise ValueError(f"Empty image_list: {args.image_list}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metric = IRSTDMetrics(threshold=args.threshold)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for img, mask, size, image_id in loader:
            img = img.float().to(device)
            forward_kwargs: dict[str, Any] = {}
            if backbone_name == "mshnet":
                forward_kwargs["mshnet_warm_flag"] = False
            output = model(img, **forward_kwargs)
            logit_batch = extract_final_logit(output).detach().cpu()
            prob_batch = torch.sigmoid(logit_batch)
            mask_batch = mask.float()
            for index in range(prob_batch.shape[0]):
                h, w = _first_size(size, index)
                name = _image_id_at(image_id, index)
                logit = _crop(logit_batch[index, 0], h, w).numpy().astype(np.float32)
                prob = _crop(prob_batch[index, 0], h, w).numpy().astype(np.float32)
                gt = _crop(mask_batch[index, 0], h, w).numpy().astype(np.float32)
                np.save(output_dir / "logits" / f"{name}.npy", logit)
                np.save(output_dir / "probs" / f"{name}.npy", prob)
                _save_prob_png(prob, output_dir / "probs" / f"{name}.png")
                _save_mask_png(prob, args.threshold, output_dir / "masks" / f"{name}.png")
                metric.update(prob, gt, size=(h, w))
                rows.append(
                    {
                        "image_name": name,
                        "height": h,
                        "width": w,
                        "mean_prob": float(prob.mean()),
                        "max_prob": float(prob.max()),
                    }
                )

    checkpoint_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    summary = metric.get()
    summary.update(
        {
            "model": run_model_name,
            "backbone": backbone_name,
            "use_cga": bool(use_cga),
            "train_dataset": train_name,
            "dataset": args.dataset_name,
            "split": args.split,
            "image_list": str(Path(args.image_list).resolve()) if args.image_list else None,
            "seed": args.seed,
            "epoch": checkpoint_epoch,
            "checkpoint_epoch": checkpoint_epoch,
            "threshold": args.threshold,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "num_images": len(rows),
            "outputs": {
                "prob": str(output_dir / "probs"),
                "logit": str(output_dir / "logits"),
                "pred_mask": str(output_dir / "masks"),
                "per_image_export": str(output_dir / "per_image_export.csv"),
            },
        }
    )
    _write_csv(output_dir / "per_image_export.csv", rows, ["image_name", "height", "width", "mean_prob", "max_prob"])
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
