"""Dataset-name-selectable integrity preflight for CGA-v2."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset import (
    dataset_registry_sha256,
    dataset_root,
    get_dataset_entry,
    normalize_item_id,
    resolve_item_paths,
    sha256_file,
    split_path,
)


def _read_list(path: Path, entry: dict[str, Any]) -> list[str]:
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        item = normalize_item_id(raw, entry)
        if item:
            items.append(item)
    return items


def _open_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def _inspect_pair(
    dataset_dir: Path,
    dataset_name: str,
    item_id: str,
    allowed_values: set[int],
    registry_path: Path,
) -> dict[str, int]:
    image_path, mask_path = resolve_item_paths(dataset_dir, dataset_name, item_id, registry_path)
    row = {
        "missing_images": 0,
        "missing_masks": 0,
        "unreadable_images": 0,
        "unreadable_masks": 0,
        "size_mismatch": 0,
        "illegal_mask_value_files": 0,
        "empty_mask_files": 0,
    }
    if not image_path.exists():
        row["missing_images"] = 1
        return row
    if not mask_path.exists():
        row["missing_masks"] = 1
        return row
    try:
        image = _open_gray(image_path)
    except Exception:
        row["unreadable_images"] = 1
        return row
    try:
        mask = _open_gray(mask_path)
    except Exception:
        row["unreadable_masks"] = 1
        return row
    if image.shape != mask.shape:
        row["size_mismatch"] = 1
    values = set(int(v) for v in np.unique(mask))
    if not values.issubset(allowed_values):
        row["illegal_mask_value_files"] = 1
    if int((mask > 0).sum()) == 0:
        row["empty_mask_files"] = 1
    return row


def run_preflight(dataset_dir: Path, dataset_name: str, registry_path: Path) -> dict[str, Any]:
    entry = get_dataset_entry(dataset_name, registry_path)
    root = dataset_root(dataset_dir, dataset_name, registry_path).resolve()
    mask_policy = entry.get("mask_policy", {})
    allowed_values = set(int(v) for v in mask_policy.get("allowed_values", [0, 255]))
    summary: dict[str, Any] = {
        "gate": "Gate-CGA-v2-P1-dataset-preflight",
        "dataset_name": dataset_name,
        "dataset_root": str(root),
        "dataset_registry_sha256": dataset_registry_sha256(registry_path),
    }
    split_items: dict[str, list[str]] = {}
    errors: list[str] = []
    declared_splits = set(entry.get("splits", {}))
    for split in ("train", "test", "hcval"):
        count_key = f"{split}_count"
        sha_key = f"{split}_list_sha256"
        split_items[split] = []
        try:
            path = split_path(dataset_dir, dataset_name, split, registry_path)
        except KeyError:
            summary[count_key] = 0
            summary[sha_key] = None
            continue
        if not path.exists():
            summary[count_key] = 0
            summary[sha_key] = None
            if split in declared_splits:
                errors.append(f"missing_{split}_list")
            continue
        items = _read_list(path, entry)
        split_items[split] = items
        summary[count_key] = len(items)
        summary[sha_key] = sha256_file(path)
        expected = entry.get("expected_counts", {}).get(split)
        if expected is not None and int(expected) != len(items):
            errors.append(f"{split}_count_expected_{expected}_got_{len(items)}")

    train_set = set(split_items.get("train", []))
    test_set = set(split_items.get("test", []))
    all_items = split_items.get("train", []) + split_items.get("test", []) + split_items.get("hcval", [])
    counts = Counter(all_items)
    unique_items = sorted(counts)
    summary["train_test_overlap_count"] = len(train_set & test_set)
    summary["duplicate_item_count"] = sum(v - 1 for v in counts.values() if v > 1)
    if summary["train_test_overlap_count"]:
        errors.append("train_test_overlap")
    if summary["duplicate_item_count"]:
        errors.append("duplicate_items")

    aggregate = {
        "missing_images": 0,
        "missing_masks": 0,
        "unreadable_images": 0,
        "unreadable_masks": 0,
        "size_mismatch": 0,
        "illegal_mask_value_files": 0,
        "empty_mask_files": 0,
    }
    examples: dict[str, list[str]] = {k: [] for k in aggregate}
    for item_id in unique_items:
        row = _inspect_pair(dataset_dir, dataset_name, item_id, allowed_values, registry_path)
        for key, value in row.items():
            aggregate[key] += value
            if value and len(examples[key]) < 20:
                examples[key].append(item_id)
    summary.update(aggregate)
    summary["examples"] = examples
    for key, value in aggregate.items():
        if value:
            errors.append(key)
    summary["gate_pass"] = not errors
    summary["decision"] = "DATASET_PREFLIGHT_PASS" if summary["gate_pass"] else "STOP_NEW_REPO_CGA_V2_AT_DATASET_PREFLIGHT"
    summary["errors"] = errors
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--registry", default="configs/datasets.yaml")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    output = args.output or f"docs/internal/cga_v2/dataset_preflight/{args.dataset_name}/summary.json"
    summary = run_preflight(Path(args.dataset_dir), args.dataset_name, Path(args.registry))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
