"""Lightweight artifact manifest helpers that do not import training utils."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def assert_identity(summary: dict, *, dataset: str, seed: int, threshold: float, epoch: int, model: str) -> None:
    errors = []
    if str(summary.get("dataset")) != dataset:
        errors.append(f"dataset mismatch: {summary.get('dataset')} != {dataset}")
    if int(summary.get("seed", -999)) != int(seed):
        errors.append(f"seed mismatch: {summary.get('seed')} != {seed}")
    if abs(float(summary.get("threshold", -1.0)) - float(threshold)) > 1e-12:
        errors.append(f"threshold mismatch: {summary.get('threshold')} != {threshold}")
    if int(summary.get("epoch", summary.get("checkpoint_epoch", -999))) != int(epoch):
        errors.append(f"epoch mismatch: {summary.get('epoch', summary.get('checkpoint_epoch'))} != {epoch}")
    if str(summary.get("model", summary.get("model_name", ""))) != model:
        errors.append(f"model mismatch: {summary.get('model', summary.get('model_name'))} != {model}")
    if errors:
        raise AssertionError("; ".join(errors))
