"""Evaluate saved prediction PNGs against masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import read_split_items, resolve_item_paths
from metrics import IRSTDMetrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate prediction directory")
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--prediction_dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", required=True)
    return p.parse_args()


def read_gray(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return arr / 255.0


def main() -> None:
    args = parse_args()
    ids = read_split_items(args.dataset_dir, args.dataset_name, args.split, required=True)
    metric = IRSTDMetrics(threshold=args.threshold)
    missing = []
    for image_id in ids:
        pred_path = Path(args.prediction_dir) / f"{image_id}.png"
        _, mask_path = resolve_item_paths(args.dataset_dir, args.dataset_name, image_id)
        if not pred_path.exists() or not mask_path.exists():
            missing.append(image_id)
            continue
        pred = read_gray(pred_path)
        gt = read_gray(mask_path)
        metric.update(pred, gt, size=gt.shape)
    summary = metric.get()
    summary.update({
        "dataset": args.dataset_name,
        "split": args.split,
        "threshold": args.threshold,
        "prediction_dir": str(args.prediction_dir),
        "missing_predictions_or_masks": missing,
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
