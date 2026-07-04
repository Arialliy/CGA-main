"""Metrics and diagnostics for IRSTD segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    from skimage import measure
except Exception:  # pragma: no cover
    measure = None


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_binary(x: Any, threshold: float = 0.5) -> np.ndarray:
    arr = _to_numpy(x)
    arr = np.squeeze(arr)
    return (arr > threshold).astype(np.uint8)


def _label(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if measure is not None:
        return measure.label(mask, connectivity=2)
    # fallback BFS labels
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.int32)
    lab = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or out[y, x] != 0:
                continue
            lab += 1
            stack = [(y, x)]
            out[y, x] = lab
            while stack:
                cy, cx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and out[ny, nx] == 0:
                            out[ny, nx] = lab
                            stack.append((ny, nx))
    return out


def component_count(mask: Any, threshold: float = 0.5) -> int:
    lab = _label(as_binary(mask, threshold))
    return int(lab.max())


@dataclass
class MetricState:
    inter: float = 0.0
    union: float = 0.0
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    pixels: float = 0.0
    niou_sum: float = 0.0
    niou_count: int = 0
    pd_targets: float = 0.0
    pd_detected: float = 0.0
    fa_pixels: float = 0.0
    fp_components: float = 0.0


class IRSTDMetrics:
    def __init__(self, threshold: float = 0.5, distance_threshold: float = 3.0) -> None:
        self.threshold = float(threshold)
        self.distance_threshold = float(distance_threshold)
        self.state = MetricState()

    def update(self, pred_prob: Any, gt_mask: Any, size: tuple[int, int] | list[int] | None = None) -> None:
        pred = as_binary(pred_prob, self.threshold)
        gt = as_binary(gt_mask, 0.0)
        if pred.shape != gt.shape:
            raise AssertionError(f"Predict and Label Shape Don't Match: {pred.shape} vs {gt.shape}")
        inter = float(np.logical_and(pred, gt).sum())
        union = float(np.logical_or(pred, gt).sum())
        tp = inter
        fp = float(np.logical_and(pred > 0, gt == 0).sum())
        fn = float(np.logical_and(pred == 0, gt > 0).sum())
        self.state.inter += inter
        self.state.union += union
        self.state.tp += tp
        self.state.fp += fp
        self.state.fn += fn
        self.state.pixels += float((size[0] * size[1]) if size is not None else gt.size)
        self.state.niou_sum += inter / (union + np.spacing(1))
        self.state.niou_count += 1
        pd_info = target_detection_audit(pred, gt, distance_threshold=self.distance_threshold)
        self.state.pd_targets += float(pd_info["target_count"])
        self.state.pd_detected += float(pd_info["detected_target_count"])
        self.state.fa_pixels += float(pd_info["false_alarm_pixels"])
        self.state.fp_components += float(pd_info["fp_component_count"])

    def get(self) -> dict[str, float]:
        s = self.state
        precision = s.tp / (s.tp + s.fp + np.spacing(1))
        recall = s.tp / (s.tp + s.fn + np.spacing(1))
        f1 = 2.0 * precision * recall / (precision + recall + np.spacing(1))
        return {
            "mIoU": float(s.inter / (s.union + np.spacing(1))),
            "nIoU": float(s.niou_sum / max(1, s.niou_count)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Pd": float(s.pd_detected / (s.pd_targets + np.spacing(1))),
            "FA": float(s.fa_pixels / (s.pixels + np.spacing(1))),
            "FA_ppm": float(1e6 * s.fa_pixels / (s.pixels + np.spacing(1))),
            "FP_components": float(s.fp_components),
        }


def _region_centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def target_detection_audit(pred_mask: Any, gt_mask: Any, distance_threshold: float = 3.0) -> dict[str, Any]:
    pred = as_binary(pred_mask, 0.0)
    gt = as_binary(gt_mask, 0.0)
    pred_lab = _label(pred)
    gt_lab = _label(gt)
    pred_ids = [i for i in range(1, int(pred_lab.max()) + 1)]
    gt_ids = [i for i in range(1, int(gt_lab.max()) + 1)]
    used_pred: set[int] = set()
    target_details = []
    detected = 0
    for tid in gt_ids:
        tcoords = np.argwhere(gt_lab == tid)
        tc = _region_centroid(tcoords)
        best_pid = None
        best_dist = float("inf")
        best_iou = 0.0
        for pid in pred_ids:
            if pid in used_pred:
                continue
            pcoords = np.argwhere(pred_lab == pid)
            pc = _region_centroid(pcoords)
            dist = float(np.linalg.norm(tc - pc))
            inter = np.logical_and(gt_lab == tid, pred_lab == pid).sum()
            union = np.logical_or(gt_lab == tid, pred_lab == pid).sum()
            iou = float(inter / (union + np.spacing(1)))
            if dist < best_dist:
                best_pid = pid
                best_dist = dist
                best_iou = iou
        is_detected = best_pid is not None and best_dist < distance_threshold
        if is_detected:
            used_pred.add(int(best_pid))
            detected += 1
        target_details.append({
            "target_id": int(tid - 1),
            "area": int(tcoords.shape[0]),
            "detected": bool(is_detected),
            "nearest_pred_component_id": None if best_pid is None else int(best_pid - 1),
            "nearest_distance": None if best_pid is None else float(best_dist),
            "nearest_iou": float(best_iou),
        })
    fp_component_count = len([pid for pid in pred_ids if pid not in used_pred])
    matched_pred = np.isin(pred_lab, [pid for pid in used_pred])
    false_alarm_pixels = int(np.logical_and(pred > 0, ~matched_pred).sum())
    return {
        "target_count": len(gt_ids),
        "detected_target_count": detected,
        "missed_target_count": len(gt_ids) - detected,
        "pred_component_count": len(pred_ids),
        "fp_component_count": int(fp_component_count),
        "false_alarm_pixels": false_alarm_pixels,
        "target_details": target_details,
    }


def stratify_targets_by_area(gt_mask: Any) -> dict[str, int]:
    gt_lab = _label(as_binary(gt_mask, 0.0))
    bins = {"tiny_1_4": 0, "small_5_16": 0, "medium_17_64": 0, "large_gt64": 0}
    for tid in range(1, int(gt_lab.max()) + 1):
        area = int((gt_lab == tid).sum())
        if area <= 4:
            bins["tiny_1_4"] += 1
        elif area <= 16:
            bins["small_5_16"] += 1
        elif area <= 64:
            bins["medium_17_64"] += 1
        else:
            bins["large_gt64"] += 1
    return bins

# Legacy class names used by BasicIRSTD scripts.
class mIoU:
    def __init__(self):
        self.metric = IRSTDMetrics(threshold=0.5)
    def update(self, preds, labels):
        self.metric.update(preds, labels)
    def get(self):
        m = self.metric.get()
        return m["Recall"], m["mIoU"]
    def reset(self):
        self.metric = IRSTDMetrics(threshold=0.5)

class PD_FA:
    def __init__(self):
        self.metric = IRSTDMetrics(threshold=0.5)
    def update(self, preds, labels, size=None):
        self.metric.update(preds, labels, size=size)
    def get(self):
        m = self.metric.get()
        return m["Pd"], m["FA"]
    def reset(self):
        self.metric = IRSTDMetrics(threshold=0.5)
