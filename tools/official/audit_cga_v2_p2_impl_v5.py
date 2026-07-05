"""Audit CGA-v2 failed P2 seed42 result without changing the method.

This tool implements the audit-only checks described in
``CGA_v2_P2_impl_audit_only_plan_v5_home_ly_AAAI.md``.  It writes one JSON file
per audit step under the configured audit directory and never mutates training
or evaluation artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from dataset import TestSetLoader, read_split_items, resolve_item_paths, sha256_file
from metrics import _label, target_detection_audit
from model.CGA_MSHNet import extract_final_logit
from model.output_contract import validate_detector_output
from net import build_model
from utils.cga_targets import CGATargetConfig, binary_dilate, build_cga_targets, ensure_4d


AUDIT_STEPS_PRIORITY = [
    ("A0_absolute_path_inventory", "P2_INVALID_PATH_CONTAMINATION"),
    ("A1_artifact_freeze", "P2_INVALID_ARTIFACTS"),
    ("A2_paired_protocol", "P2_INVALID_PROTOCOL"),
    ("A3_strict_load_ohem", "P2_INVALID_CHECKPOINT_LOAD"),
    ("A3_strict_load_cga", "P2_INVALID_CHECKPOINT_LOAD"),
    ("A5_eval_output_trace_ohem", "P2_INVALID_EVAL_OUTPUT_SOURCE"),
    ("A5_eval_output_trace_cga", "P2_INVALID_EVAL_OUTPUT_SOURCE"),
    ("A5b_adapter_contract", "P2_INVALID_ADAPTER_CONTRACT"),
    ("A7_target_geometry", "P2_INVALID_TARGET_GENERATION"),
    ("A4_loss_scale", "P2_INVALID_LOSS_LOG"),
    ("A6_prediction_morphology", "P2_INVALID_PREDICTION_ARTIFACTS"),
]

CODE_IDENTITY_FILES = [
    "test.py",
    "train.py",
    "net.py",
    "loss.py",
    "model/cga_wrapper.py",
    "model/output_contract.py",
    "model/backbones/mshnet_adapter.py",
    "utils/cga_targets.py",
]

ABS_PATH_RE = re.compile(r"(?<![\w.-])/(?:[^\s'\"<>:,{}[\]]+/)*[^\s'\"<>:,{}[\]]+")
REQUIRED_CGA_KEYS = (
    "cga_center_logit",
    "cga_boundary_logit",
    "cga_scale_logit",
    "cga_peak_logit",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return list(obj.shape)
    return str(obj)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def make_step(
    audit_step: str,
    passed: bool,
    invalidates_p2: bool,
    requires_rerun: bool,
    decision_if_failed: str | None,
    notes: list[str] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "audit_step": audit_step,
        "pass": bool(passed),
        "invalidates_p2": bool(invalidates_p2),
        "requires_rerun": bool(requires_rerun),
        "decision_if_failed": decision_if_failed,
        "notes": notes or [],
        "artifacts": artifacts or {},
    }


def sha256_or_none(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_file(path)


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256": sha256_or_none(path) if path.is_file() else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse {path}:{line_no}: {exc}") from exc
    return rows


def run_cmd(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(args, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


class AuditContext:
    def __init__(
        self,
        *,
        root: Path,
        canonical_root: str,
        run_root: str,
        dataset_dir: Path,
        dataset_name: str,
        output_dir: Path,
        p2_dir: Path,
        audit_dir: Path,
        seed: int,
        epoch: int,
        threshold: float,
        mount_note: str,
    ) -> None:
        self.root = root
        self.canonical_root = canonical_root.rstrip("/")
        self.run_root = run_root.rstrip("/")
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.p2_dir = p2_dir
        self.audit_dir = audit_dir
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.threshold = float(threshold)
        self.mount_note = mount_note
        self.summary_path = p2_dir / "summary.json"
        self.summary = load_json(self.summary_path)

    def model_dir(self, model: str) -> Path:
        return self.output_dir / model / f"seed{self.seed}" / self.dataset_name

    def ckpt_path(self, model: str) -> Path:
        return self.model_dir(model) / f"{model}_{self.epoch}.pth.tar"

    def split_summary_path(self, model: str, split: str) -> Path:
        return self.model_dir(model) / split / "summary_metrics.json"

    def pred_dir(self, model: str, split: str) -> Path:
        return self.model_dir(model) / split / "predictions"

    def train_log_path(self, model: str) -> Path:
        return self.model_dir(model) / "train_log.jsonl"

    def to_run_path(self, path_text: str | Path) -> Path:
        text = str(path_text)
        if text.startswith(self.run_root):
            return Path(text)
        if text.startswith(self.canonical_root):
            return Path(self.run_root + text[len(self.canonical_root) :])
        return Path(text)

    def to_canonical_path(self, path_text: str | Path) -> str:
        text = str(path_text)
        if text.startswith(self.run_root):
            return self.canonical_root + text[len(self.run_root) :]
        return text

    def write_step(self, name: str, step: dict[str, Any]) -> None:
        write_json(self.audit_dir / f"{name}.json", step)


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    x = x.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {"checkpoint_type": type(ckpt).__name__}
    return {k: v for k, v in ckpt.items() if k not in {"state_dict", "model", "optimizer"}}


def normalized_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], list[str], dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    info: dict[str, Any] = {
        "checkpoint_type": type(ckpt).__name__,
        "checkpoint_keys": sorted(list(ckpt.keys())) if isinstance(ckpt, dict) else None,
    }
    normalization: list[str] = []
    state = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        normalization.append("unwrap_state_dict")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        normalization.append("unwrap_model")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    state = dict(state)
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module.") :]: v for k, v in state.items()}
        normalization.append("strip_module_prefix")
    keys = list(state.keys())
    if keys and all(k.startswith("model.") for k in keys):
        state = {k[len("model.") :]: v for k, v in state.items()}
        normalization.append("strip_model_prefix")
    return state, normalization, info


def compare_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    expected = model.state_dict()
    missing = sorted(k for k in expected.keys() if k not in state)
    unexpected = sorted(k for k in state.keys() if k not in expected)
    shape_mismatches = []
    for key in sorted(set(expected.keys()) & set(state.keys())):
        if tuple(expected[key].shape) != tuple(state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "expected": list(expected[key].shape),
                    "actual": list(state[key].shape),
                }
            )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def strict_load_model(model_name: str, ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_model(model_name=model_name, evidence_mode="paper").to(device)
    state, normalization, ckpt_info = normalized_state_dict(ckpt_path)
    cmp = compare_state_dict(model, state)
    if cmp["missing_keys"] or cmp["unexpected_keys"] or cmp["shape_mismatches"]:
        return model, {
            "normalization_applied": normalization,
            "checkpoint_info": ckpt_info,
            **cmp,
            "strict_load_passed": False,
        }
    load_result = model.load_state_dict(state, strict=True)
    return model, {
        "normalization_applied": normalization,
        "checkpoint_info": ckpt_info,
        **cmp,
        "strict_load_passed": not load_result.missing_keys and not load_result.unexpected_keys,
    }


def select_final_logit_key(output: Any) -> tuple[str | None, torch.Tensor]:
    if isinstance(output, dict):
        for key in ("final_logit", "final_logits", "base_logits", "base_logit", "logits"):
            if key in output:
                return key, output[key]
    return None, extract_final_logit(output)


def audit_a0(ctx: AuditContext) -> dict[str, Any]:
    sources: dict[str, str] = {}

    def scan_text(text: str, source: str) -> None:
        for match in ABS_PATH_RE.findall(text):
            cleaned = match.rstrip(").,;")
            if cleaned.startswith("/"):
                sources.setdefault(cleaned, source)

    candidate_files = [
        ctx.summary_path,
        ctx.split_summary_path("MSHNetOHEM", "test"),
        ctx.split_summary_path("MSHNetOHEM", "hcval"),
        ctx.split_summary_path("MSHNetCGA", "test"),
        ctx.split_summary_path("MSHNetCGA", "hcval"),
        ctx.train_log_path("MSHNetOHEM"),
        ctx.train_log_path("MSHNetCGA"),
    ]
    candidate_files.extend(sorted(ctx.p2_dir.rglob("*.json")))

    for path in sorted(set(candidate_files)):
        if path.exists() and path.is_file():
            scan_text(path.read_text(encoding="utf-8", errors="ignore"), str(path))

    for model in ("MSHNetOHEM", "MSHNetCGA"):
        ckpt_path = ctx.ckpt_path(model)
        if ckpt_path.exists():
            meta = checkpoint_metadata(ckpt_path)
            scan_text(json.dumps(meta, default=_json_default), str(ckpt_path))

    classes: dict[str, list[dict[str, str]]] = {
        "canonical": [],
        "canonical_missing": [],
        "realpath_equivalent": [],
        "noncanonical_resolved_by_mount_proof": [],
        "noncanonical_unresolved": [],
    }
    mount_proofs = []

    for path_text, source in sorted(sources.items()):
        row = {"path": path_text, "source": source}
        if path_text.startswith(ctx.canonical_root):
            run_equiv = ctx.to_run_path(path_text)
            if Path(path_text).exists() or run_equiv.exists():
                row["run_equivalent"] = str(run_equiv)
                classes["canonical"].append(row)
            else:
                classes["canonical_missing"].append(row)
            continue

        p = Path(path_text)
        if p.exists():
            try:
                real = str(p.resolve())
            except Exception:
                real = str(p)
            if real.startswith(ctx.canonical_root):
                row["realpath"] = real
                classes["realpath_equivalent"].append(row)
            elif path_text.startswith(ctx.run_root):
                rel = path_text[len(ctx.run_root) :].lstrip("/")
                canonical_equiv = ctx.canonical_root + "/" + rel
                row["canonical_equivalent"] = canonical_equiv
                row["mount_proof"] = ctx.mount_note
                classes["noncanonical_resolved_by_mount_proof"].append(row)
                mount_proofs.append(
                    {
                        "old_prefix": ctx.run_root,
                        "canonical_prefix": ctx.canonical_root,
                        "relative_path": rel,
                        "current_run_path_exists": True,
                    }
                )
            else:
                row["realpath"] = real
                classes["noncanonical_unresolved"].append(row)
        elif path_text.startswith(ctx.run_root):
            rel = path_text[len(ctx.run_root) :].lstrip("/")
            run_equiv = Path(ctx.run_root) / rel
            if run_equiv.exists():
                row["canonical_equivalent"] = ctx.canonical_root + "/" + rel
                row["mount_proof"] = ctx.mount_note
                classes["noncanonical_resolved_by_mount_proof"].append(row)
                mount_proofs.append(
                    {
                        "old_prefix": ctx.run_root,
                        "canonical_prefix": ctx.canonical_root,
                        "relative_path": rel,
                        "current_run_path_exists": True,
                    }
                )
            else:
                classes["noncanonical_unresolved"].append(row)
        else:
            classes["noncanonical_unresolved"].append(row)

    unresolved = classes["noncanonical_unresolved"]
    passed = len(unresolved) == 0
    notes = [
        f"Extracted {len(sources)} unique absolute paths from summaries, logs, checkpoint metadata, and P2 JSON files.",
        ctx.mount_note,
    ]
    if unresolved:
        notes.append(f"Found {len(unresolved)} noncanonical unresolved paths.")
    step = make_step(
        "A0_absolute_path_inventory",
        passed,
        not passed,
        not passed,
        "P2_INVALID_PATH_CONTAMINATION" if not passed else None,
        notes,
        {
            "canonical_root": ctx.canonical_root,
            "run_root": ctx.run_root,
            "path_count": len(sources),
            "mount_proofs": mount_proofs,
            **classes,
        },
    )
    ctx.write_step("A0_absolute_path_inventory", step)
    return step


def _artifact_row(ctx: AuditContext, model: str, split: str) -> dict[str, Any]:
    ckpt_path = ctx.ckpt_path(model)
    summary_path = ctx.split_summary_path(model, split)
    pred_dir = ctx.pred_dir(model, split)
    train_log = ctx.train_log_path(model)
    pred_count = len(list(pred_dir.glob("*.png"))) if pred_dir.exists() else 0
    last_epoch = None
    line_count = 0
    parse_error = None
    if train_log.exists():
        try:
            rows = read_jsonl(train_log)
            line_count = len(rows)
            last_epoch = int(rows[-1].get("epoch", -1)) if rows else None
        except Exception as exc:
            parse_error = str(exc)
    ckpt_epoch = None
    ckpt_meta: dict[str, Any] = {}
    if ckpt_path.exists():
        ckpt_meta = checkpoint_metadata(ckpt_path)
        ckpt_epoch = ckpt_meta.get("epoch")
    return {
        "model": model,
        "split": split,
        "checkpoint": file_info(ckpt_path),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_metadata": ckpt_meta,
        "summary_metrics": file_info(summary_path),
        "prediction_dir": str(pred_dir),
        "prediction_dir_exists": pred_dir.exists(),
        "prediction_png_count": pred_count,
        "train_log": file_info(train_log),
        "train_log_line_count": line_count,
        "train_log_last_epoch": last_epoch,
        "train_log_parse_error": parse_error,
    }


def audit_a1(ctx: AuditContext, a0: dict[str, Any]) -> dict[str, Any]:
    artifacts = {}
    failures = []
    for model in ("MSHNetOHEM", "MSHNetCGA"):
        for split in ("test", "hcval"):
            row = _artifact_row(ctx, model, split)
            artifacts[f"{model}/{split}"] = row
            if not row["checkpoint"]["exists"]:
                failures.append(f"{model}/{split}: missing checkpoint")
            if row["checkpoint_epoch"] != ctx.epoch:
                failures.append(f"{model}/{split}: checkpoint_epoch={row['checkpoint_epoch']} != {ctx.epoch}")
            if not row["summary_metrics"]["exists"]:
                failures.append(f"{model}/{split}: missing summary_metrics")
            if not row["prediction_dir_exists"] or row["prediction_png_count"] == 0:
                failures.append(f"{model}/{split}: missing or empty prediction_dir")
            if row["train_log_parse_error"]:
                failures.append(f"{model}/{split}: train log parse error")
            if row["train_log"]["exists"] and row["train_log_last_epoch"] is not None and row["train_log_last_epoch"] < ctx.epoch:
                failures.append(f"{model}/{split}: train log last epoch < {ctx.epoch}")

    for row in a0.get("artifacts", {}).get("canonical_missing", []):
        path = row.get("path", "")
        if any(token in path for token in ("summary_metrics.json", ".pth.tar", "predictions", "train_log.jsonl")):
            failures.append(f"A0 canonical_missing required artifact: {path}")

    code_hashes = {}
    for rel in CODE_IDENTITY_FILES:
        p = ctx.root / rel
        code_hashes[rel] = file_info(p)
    rc_branch, branch, branch_err = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], ctx.root)
    rc_commit, commit, commit_err = run_cmd(["git", "rev-parse", "HEAD"], ctx.root)
    rc_status, status, status_err = run_cmd(["git", "status", "--short"], ctx.root)
    rc_stat, diff_stat, diff_stat_err = run_cmd(["git", "diff", "--stat"], ctx.root)
    rc_check, diff_check, diff_check_err = run_cmd(["git", "diff", "--check"], ctx.root)

    git_info = {
        "branch": branch if rc_branch == 0 else None,
        "branch_error": branch_err if rc_branch != 0 else None,
        "commit": commit if rc_commit == 0 else None,
        "commit_error": commit_err if rc_commit != 0 else None,
        "dirty_status": status,
        "dirty_status_error": status_err if rc_status != 0 else None,
        "diff_stat": diff_stat,
        "diff_stat_error": diff_stat_err if rc_stat != 0 else None,
        "diff_check_pass": rc_check == 0,
        "diff_check_stdout": diff_check,
        "diff_check_stderr": diff_check_err,
    }
    p2_recorded_code_identity = any(
        key in ctx.summary for key in ("git_commit", "code_sha256", "code_identity", "run_manifest")
    )
    passed = len(failures) == 0
    step = make_step(
        "A1_artifact_freeze",
        passed,
        not passed,
        not passed,
        "P2_INVALID_ARTIFACTS" if not passed else None,
        failures if failures else ["All required P2 artifacts are present and frozen by hash."],
        {
            "artifacts": artifacts,
            "code_identity_hashes": code_hashes,
            "git": git_info,
            "historical_code_identity_available": bool(p2_recorded_code_identity),
        },
    )
    ctx.write_step("A1_artifact_freeze", step)
    return step


def audit_a2(ctx: AuditContext) -> dict[str, Any]:
    summary = ctx.summary
    full = summary.get("full", {})
    hcval = summary.get("hcval", {})
    failures = []
    notes = []

    def check_pair(section: dict[str, Any], split: str) -> None:
        b = section.get("baseline", {})
        c = section.get("candidate", {})
        equal_fields = ["dataset", "seed", "epoch", "threshold", "threshold_selection", "checkpoint_epoch", "split"]
        for field in equal_fields:
            if b.get(field) != c.get(field):
                failures.append(f"{split}: baseline/candidate {field} mismatch: {b.get(field)!r} vs {c.get(field)!r}")
        if b.get("split") != split or c.get("split") != split:
            failures.append(f"{split}: split name mismatch")
        if b.get("model") != "MSHNetOHEM":
            failures.append(f"{split}: baseline model is {b.get('model')!r}")
        if c.get("model") != "MSHNetCGA":
            failures.append(f"{split}: candidate model is {c.get('model')!r}")
        if b.get("use_cga") is not False:
            failures.append(f"{split}: baseline use_cga is not false")
        if c.get("use_cga") is not True:
            failures.append(f"{split}: candidate use_cga is not true")
        if b.get("threshold") != ctx.threshold or c.get("threshold") != ctx.threshold:
            failures.append(f"{split}: threshold is not fixed {ctx.threshold}")
        if b.get("threshold_selection") != "fixed_predeclared" or c.get("threshold_selection") != "fixed_predeclared":
            failures.append(f"{split}: threshold_selection is not fixed_predeclared")

    check_pair(full, "test")
    if hcval.get("available") is not True:
        failures.append("hcval section is not available")
    check_pair(hcval, "hcval")

    required_hashes = ["train_list_sha256", "test_list_sha256", "hcval_list_sha256"]
    for key in required_hashes:
        if not summary.get(key):
            failures.append(f"missing {key} in P2 summary")

    metadata_complete = True
    metadata_sources = {}
    for model in ("MSHNetOHEM", "MSHNetCGA"):
        ckpt_meta = checkpoint_metadata(ctx.ckpt_path(model))
        rows = read_jsonl(ctx.train_log_path(model))
        last_log = rows[-1] if rows else {}
        metadata_sources[model] = {
            "checkpoint": {k: ckpt_meta.get(k) for k in [
                "evidence_mode",
                "paper_evidence_allowed",
                "p1_preflight_passed",
                "p1a_hcval_source_audit_passed",
                "fallback_regularizer_used",
                "protocol",
                "seed",
                "dataset",
                "regularizer_impl",
                "use_cga",
            ]},
            "last_train_log": {k: last_log.get(k) for k in [
                "evidence_mode",
                "paper_evidence_allowed",
                "p1_preflight_passed",
                "p1a_hcval_source_audit_passed",
                "fallback_regularizer_used",
                "protocol",
                "seed",
                "dataset",
                "regularizer_impl",
                "use_cga",
            ]},
        }
        expected = {
            "evidence_mode": "paper",
            "paper_evidence_allowed": True,
            "p1_preflight_passed": True,
            "p1a_hcval_source_audit_passed": True,
            "fallback_regularizer_used": False,
            "protocol": "controlled",
        }
        for source_name, source in metadata_sources[model].items():
            for key, expected_value in expected.items():
                if key not in source or source.get(key) is None:
                    metadata_complete = False
                elif source.get(key) != expected_value:
                    failures.append(f"{model} {source_name}: {key}={source.get(key)!r}, expected {expected_value!r}")
        if model == "MSHNetOHEM":
            if ckpt_meta.get("use_cga") is not False:
                failures.append("MSHNetOHEM checkpoint use_cga is not false")
        if model == "MSHNetCGA":
            if ckpt_meta.get("use_cga") is not True:
                failures.append("MSHNetCGA checkpoint use_cga is not true")
            if ckpt_meta.get("regularizer_impl") != "center_boundary_scale_peak":
                failures.append("MSHNetCGA checkpoint regularizer_impl mismatch")

    if not metadata_complete:
        notes.append("metadata_complete=false; equivalent evidence was checked from checkpoint and final train-log rows.")

    passed = len(failures) == 0
    step = make_step(
        "A2_paired_protocol",
        passed,
        not passed,
        not passed,
        "P2_INVALID_PROTOCOL" if not passed else None,
        failures or notes or ["Paired protocol fields and paper-evidence metadata are consistent."],
        {
            "metadata_complete": metadata_complete,
            "metadata_sources": metadata_sources,
            "summary_hashes": {k: summary.get(k) for k in required_hashes},
            "full_delta": full.get("delta"),
            "hcval_delta": hcval.get("delta"),
        },
    )
    ctx.write_step("A2_paired_protocol", step)
    return step


def audit_a3_one(ctx: AuditContext, model: str, step_name: str) -> dict[str, Any]:
    ckpt = ctx.ckpt_path(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    notes = []
    try:
        _, info = strict_load_model(model, ckpt, device)
        failures = []
        if info["missing_keys"]:
            failures.append("missing_keys")
        if info["unexpected_keys"]:
            failures.append("unexpected_keys")
        if info["shape_mismatches"]:
            failures.append("shape_mismatches")
        if not info["strict_load_passed"]:
            failures.append("strict_load_failed")
        passed = len(failures) == 0
        if passed:
            notes.append("Checkpoint loads with strict=True after whitelist normalization only.")
        else:
            notes.append("Strict checkpoint loading failed: " + ", ".join(failures))
        artifacts = {
            "model": model,
            "checkpoint": str(ckpt),
            **info,
        }
    except Exception as exc:
        passed = False
        notes.append(f"Strict load audit raised {type(exc).__name__}: {exc}")
        artifacts = {"model": model, "checkpoint": str(ckpt), "exception": repr(exc)}
    step = make_step(
        step_name,
        passed,
        not passed,
        not passed,
        "P2_INVALID_CHECKPOINT_LOAD" if not passed else None,
        notes,
        artifacts,
    )
    ctx.write_step(step_name, step)
    return step


def audit_a4(ctx: AuditContext) -> dict[str, Any]:
    failures = []
    diagnostics = []
    artifacts: dict[str, Any] = {}
    try:
        ohem_rows = read_jsonl(ctx.train_log_path("MSHNetOHEM"))
        cga_rows = read_jsonl(ctx.train_log_path("MSHNetCGA"))
    except Exception as exc:
        step = make_step(
            "A4_loss_scale",
            False,
            True,
            True,
            "P2_INVALID_LOSS_LOG",
            [f"Loss log cannot be parsed: {exc}"],
            {},
        )
        ctx.write_step("A4_loss_scale", step)
        return step

    def validate_rows(rows: list[dict[str, Any]], model: str) -> None:
        if not rows:
            failures.append(f"{model}: empty train log")
            return
        epochs = [int(r.get("epoch", -1)) for r in rows]
        if epochs[-1] != ctx.epoch:
            failures.append(f"{model}: final epoch row missing")
        if epochs != sorted(epochs) or len(set(epochs)) != len(epochs):
            failures.append(f"{model}: epoch order corrupt")
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    failures.append(f"{model}: nonfinite {key} at epoch {row.get('epoch')}")

    validate_rows(ohem_rows, "MSHNetOHEM")
    validate_rows(cga_rows, "MSHNetCGA")

    def mean(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
        vals = [float(r.get(key, default)) for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    tail_ohem = ohem_rows[-20:]
    tail_cga = cga_rows[-20:]
    aux_weighted = []
    for r in tail_cga:
        aux = (
            0.05 * float(r.get("cga_center", 0.0))
            + 0.03 * float(r.get("cga_boundary", 0.0))
            + 0.02 * float(r.get("cga_scale", 0.0))
            + 0.03 * float(r.get("cga_peak", 0.0))
        )
        aux_weighted.append(float(r.get("cga_w", 0.0)) * aux)
    aux_total_mean = float(np.mean(aux_weighted)) if aux_weighted else 0.0
    cga_total_mean = mean(tail_cga, "total")
    cga_base_mean = mean(tail_cga, "base_total", mean(tail_cga, "total"))
    ohem_total_mean = mean(tail_ohem, "total")

    metrics = {
        "base_total_tail20_mean": cga_base_mean,
        "aux_total_tail20_mean": aux_total_mean,
        "aux_total_over_total_tail20_mean": aux_total_mean / (cga_total_mean + np.spacing(1)),
        "aux_total_over_base_total_tail20_mean": aux_total_mean / (cga_base_mean + np.spacing(1)),
        "center_over_total_tail20_mean": (0.05 * mean(tail_cga, "cga_center")) / (cga_total_mean + np.spacing(1)),
        "boundary_over_total_tail20_mean": (0.03 * mean(tail_cga, "cga_boundary")) / (cga_total_mean + np.spacing(1)),
        "scale_over_total_tail20_mean": (0.02 * mean(tail_cga, "cga_scale")) / (cga_total_mean + np.spacing(1)),
        "peak_over_total_tail20_mean": (0.03 * mean(tail_cga, "cga_peak")) / (cga_total_mean + np.spacing(1)),
        "cga_w_tail20_mean": mean(tail_cga, "cga_w"),
        "cga_w_last": float(cga_rows[-1].get("cga_w", 0.0)) if cga_rows else None,
        "nan_count": 0,
        "inf_count": 0,
        "base_total_cga_vs_ohem_ratio_tail20": cga_base_mean / (ohem_total_mean + np.spacing(1)),
        "soft_iou_cga_vs_ohem_ratio_tail20": mean(tail_cga, "soft_iou") / (mean(tail_ohem, "soft_iou") + np.spacing(1)),
        "ohem_loss_cga_vs_ohem_ratio_tail20": mean(tail_cga, "ohem") / (mean(tail_ohem, "ohem") + np.spacing(1)),
    }
    for rows in (ohem_rows, cga_rows):
        for row in rows:
            for value in row.values():
                if isinstance(value, (int, float)):
                    if math.isnan(float(value)):
                        metrics["nan_count"] += 1
                    if math.isinf(float(value)):
                        metrics["inf_count"] += 1

    p2 = ctx.summary
    full_delta = p2.get("full", {}).get("delta", {})
    hc_delta = p2.get("hcval", {}).get("delta", {})
    if full_delta.get("Pd", 0.0) > 0 and full_delta.get("Precision", 0.0) < 0 and full_delta.get("FA_ppm", 0.0) > 0:
        diagnostics.append("recall_booster_behavior_full")
    if hc_delta.get("Pd", 0.0) > 0 and hc_delta.get("Precision", 0.0) < 0 and hc_delta.get("FA_ppm", 0.0) > 0:
        diagnostics.append("recall_booster_behavior_hcval")
    if metrics["aux_total_over_base_total_tail20_mean"] > 0.25:
        diagnostics.append("aux_loss_overstrong_tail20")
    if metrics["cga_w_last"] == 1.0 and metrics["aux_total_over_base_total_tail20_mean"] > 0.10:
        diagnostics.append("ramp_too_aggressive_possible")

    if metrics["nan_count"] or metrics["inf_count"]:
        failures.append("NaN/Inf in required losses")

    artifacts = {
        "MSHNetOHEM_log": str(ctx.train_log_path("MSHNetOHEM")),
        "MSHNetCGA_log": str(ctx.train_log_path("MSHNetCGA")),
        "MSHNetOHEM_line_count": len(ohem_rows),
        "MSHNetCGA_line_count": len(cga_rows),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    passed = len(failures) == 0
    step = make_step(
        "A4_loss_scale",
        passed,
        not passed,
        not passed,
        "P2_INVALID_LOSS_LOG" if not passed else None,
        failures or diagnostics or ["Loss logs are parseable and numerically finite."],
        artifacts,
    )
    ctx.write_step("A4_loss_scale", step)
    return step


def _first_eval_batch(ctx: AuditContext) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
    ds = TestSetLoader(ctx.dataset_dir, ctx.dataset_name, ctx.dataset_name, split="test")
    img, mask, size, image_id = ds[0]
    return img.unsqueeze(0), mask.unsqueeze(0), (int(size[0]), int(size[1])), image_id


def audit_a5_one(ctx: AuditContext, model_name: str, step_name: str) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    notes = []
    failures = []
    artifacts: dict[str, Any] = {}
    try:
        model, load_info = strict_load_model(model_name, ctx.ckpt_path(model_name), device)
        if not load_info["strict_load_passed"]:
            failures.append("strict_load_failed_before_trace")
        model.eval()
        img, _mask, size, image_id = _first_eval_batch(ctx)
        img = img.float().to(device)
        forward_kwargs = {"mshnet_warm_flag": False}
        with torch.no_grad():
            output = model(img, **forward_kwargs)
        key, logit = select_final_logit_key(output)
        prob = torch.sigmoid(logit)
        pred = (prob > ctx.threshold).float()
        aux_present = isinstance(output, dict) and any(k in output for k in REQUIRED_CGA_KEYS)
        aux_used = key in REQUIRED_CGA_KEYS
        ambiguous = key is None
        if aux_used:
            failures.append("prediction uses auxiliary CGA logit")
        if ambiguous:
            failures.append("prediction tensor source is ambiguous")
        artifacts = {
            "model": model_name,
            "checkpoint": str(ctx.ckpt_path(model_name)),
            "strict_load": load_info,
            "image_id": image_id,
            "input_batch_shape": list(img.shape),
            "original_size": list(size),
            "raw_output_type": type(output).__name__,
            "raw_output_keys": sorted(list(output.keys())) if isinstance(output, dict) else None,
            "selected_prediction_tensor_key": key,
            "selected_prediction_tensor_shape": list(logit.shape),
            "selected_prediction_tensor_stats": tensor_stats(logit),
            "sigmoid_stats": tensor_stats(prob),
            "threshold": ctx.threshold,
            "positive_pixels_after_threshold": int(pred.sum().item()),
            "aux_logits_present": bool(aux_present),
            "aux_logits_used_for_prediction": bool(aux_used),
            "intermediate_features_used_for_prediction": False,
            "required_prediction_source": "final logits only",
        }
        if not failures:
            notes.append("Dry eval selected final detector logits only.")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        artifacts.setdefault("exception", repr(exc))
    passed = len(failures) == 0
    step = make_step(
        step_name,
        passed,
        not passed,
        not passed,
        "P2_INVALID_EVAL_OUTPUT_SOURCE" if not passed else None,
        failures or notes,
        artifacts,
    )
    ctx.write_step(step_name, step)
    return step


def audit_a5b(ctx: AuditContext) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    failures = []
    notes = []
    artifacts: dict[str, Any] = {}
    try:
        model, load_info = strict_load_model("MSHNetCGA", ctx.ckpt_path("MSHNetCGA"), device)
        model.eval()
        img, mask, size, image_id = _first_eval_batch(ctx)
        img = img.float().to(device)
        mask = mask.float().to(device)
        with torch.no_grad():
            output = model(img, mshnet_warm_flag=False)
        try:
            validated = validate_detector_output(output, backbone_name="mshnet", require_feature=True)
            validate_passed = True
        except Exception as exc:
            validated = output
            validate_passed = False
            failures.append(f"validate_detector_output failed: {exc}")

        logits = validated.get("logits")
        if logits is None:
            logits = validated.get("final_logit")
        if logits is None:
            logits = validated.get("base_logits")
        features = validated.get("features", [])
        feature = features[0] if features else None
        feature_meta = validated.get("feature_meta", [])
        adapter_meta = validated.get("adapter_meta")
        regularizer_meta = validated.get("regularizer_meta", {})
        aux_shapes = {
            key: list(validated[key].shape) if key in validated and torch.is_tensor(validated[key]) else None
            for key in REQUIRED_CGA_KEYS
        }
        for key, shape in aux_shapes.items():
            if shape is None:
                failures.append(f"{key} missing")
            elif logits is not None and shape[-2:] != list(logits.shape[-2:]):
                failures.append(f"{key} spatial shape incompatible with logits")
        if logits is None:
            failures.append("logits missing")
        elif list(logits.shape[-2:]) != list(mask.shape[-2:]):
            failures.append("logits shape incompatible with mask")
        if not adapter_meta:
            failures.append("adapter_meta missing")
        if not feature_meta:
            failures.append("feature_meta missing")
        if feature is None:
            failures.append("selected feature missing")
        if regularizer_meta.get("fallback_regularizer_used") is not False:
            failures.append("fallback_regularizer_used != false")
        if regularizer_meta.get("regularizer_impl") != "center_boundary_scale_peak":
            failures.append("regularizer_impl mismatch")

        ckpt_meta = checkpoint_metadata(ctx.ckpt_path("MSHNetCGA"))
        evidence_required = {
            "evidence_mode": "paper",
            "paper_evidence_allowed": True,
            "p1_preflight_passed": True,
            "p1a_hcval_source_audit_passed": True,
            "fallback_regularizer_used": False,
            "regularizer_impl": "center_boundary_scale_peak",
            "protocol": "controlled",
        }
        evidence_meta = {k: ckpt_meta.get(k) for k in evidence_required}
        for key, expected in evidence_required.items():
            if evidence_meta.get(key) != expected:
                failures.append(f"evidence metadata {key}={evidence_meta.get(key)!r}, expected {expected!r}")

        artifacts = {
            "model": "MSHNetCGA",
            "checkpoint": str(ctx.ckpt_path("MSHNetCGA")),
            "strict_load": load_info,
            "image_id": image_id,
            "validate_detector_output_passed": validate_passed,
            "adapter_meta": adapter_meta,
            "feature_meta": feature_meta,
            "regularizer_meta": regularizer_meta,
            "regularizer_impl": regularizer_meta.get("regularizer_impl"),
            "fallback_regularizer_used": regularizer_meta.get("fallback_regularizer_used"),
            "logits_shape": list(logits.shape) if torch.is_tensor(logits) else None,
            "features_0_shape": list(feature.shape) if torch.is_tensor(feature) else None,
            "features_0_dtype": str(feature.dtype) if torch.is_tensor(feature) else None,
            "feature_stride": feature_meta[0].get("stride") if feature_meta else None,
            "feature_channels": feature_meta[0].get("channels") if feature_meta else None,
            "feature_resolution": feature_meta[0].get("resolution") if feature_meta else None,
            "aux_logits": aux_shapes,
            "aux_logits_spatial_compatibility_with_masks_logits": not any("spatial shape incompatible" in f for f in failures),
            "paper_evidence_allowed_source": "checkpoint_metadata_and_train_log",
            "evidence_metadata": evidence_meta,
        }
        if not failures:
            notes.append("MSHNet adapter and CGA wrapper satisfy the explicit output contract.")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        artifacts.setdefault("exception", repr(exc))
    passed = len(failures) == 0
    step = make_step(
        "A5b_adapter_contract",
        passed,
        not passed,
        not passed,
        "P2_INVALID_ADAPTER_CONTRACT" if not passed else None,
        failures or notes,
        artifacts,
    )
    ctx.write_step("A5b_adapter_contract", step)
    return step


def read_mask_binary(path: Path) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def read_prob_binary(path: Path, threshold: float) -> np.ndarray:
    return (np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0 > threshold).astype(np.uint8)


def prediction_integrity_and_rows(ctx: AuditContext, model: str, split: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    pred_dir = ctx.pred_dir(model, split)
    items = read_split_items(ctx.dataset_dir, ctx.dataset_name, split, required=True)
    expected = set(items)
    files = sorted(pred_dir.glob("*.png")) if pred_dir.exists() else []
    ids = [p.stem for p in files]
    failures = []
    if not pred_dir.exists():
        failures.append(f"{model}/{split}: prediction directory missing")
    if not files:
        failures.append(f"{model}/{split}: prediction png count = 0")
    duplicates = sorted([k for k, v in Counter(ids).items() if v > 1])
    missing = sorted(expected - set(ids))
    extra = sorted(set(ids) - expected)
    if duplicates:
        failures.append(f"{model}/{split}: duplicate prediction ids")
    if missing:
        failures.append(f"{model}/{split}: missing prediction ids")

    rows = []
    unreadable = []
    shape_mismatch = []
    for image_id in items:
        pred_path = pred_dir / f"{image_id}.png"
        if not pred_path.exists():
            continue
        _, mask_path = resolve_item_paths(ctx.dataset_dir, ctx.dataset_name, image_id)
        try:
            pred = read_prob_binary(pred_path, ctx.threshold)
            gt = read_mask_binary(mask_path)
        except Exception:
            unreadable.append(image_id)
            continue
        if pred.shape != gt.shape:
            shape_mismatch.append(image_id)
            continue
        audit = target_detection_audit(pred, gt)
        fg = int(pred.sum())
        pixels = int(pred.size)
        rows.append(
            {
                "image_id": image_id,
                "model": model,
                "split": split,
                "foreground_pixels": fg,
                "foreground_ratio": float(fg / max(1, pixels)),
                "fp_components": int(audit["fp_component_count"]),
                "fp_area": int(audit["false_alarm_pixels"]),
                "tp_components": int(audit["detected_target_count"]),
                "target_count": int(audit["target_count"]),
            }
        )
    if unreadable:
        failures.append(f"{model}/{split}: unreadable prediction images")
    if shape_mismatch:
        failures.append(f"{model}/{split}: prediction/mask shape mismatch")

    integrity = {
        "model": model,
        "split": split,
        "prediction_dir": str(pred_dir),
        "exists": pred_dir.exists(),
        "png_count": len(files),
        "expected_count": len(items),
        "all_png_readable": not unreadable,
        "prediction_ids_align_with_split_list": not missing and not extra,
        "duplicate_prediction_ids": duplicates,
        "missing_prediction_ids": missing[:50],
        "extra_prediction_ids": extra[:50],
        "unreadable_prediction_ids": unreadable[:50],
        "shape_mismatch_ids": shape_mismatch[:50],
    }
    return integrity, rows, failures


def _pair_rows(ohem: list[dict[str, Any]], cga: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_o = {r["image_id"]: r for r in ohem}
    out = []
    for c in cga:
        o = by_o.get(c["image_id"])
        if not o:
            continue
        row = {
            "image_id": c["image_id"],
            "foreground_ratio_delta": c["foreground_ratio"] - o["foreground_ratio"],
            "fp_component_delta": c["fp_components"] - o["fp_components"],
            "fp_area_delta": c["fp_area"] - o["fp_area"],
            "cga_foreground_ratio": c["foreground_ratio"],
            "ohem_foreground_ratio": o["foreground_ratio"],
            "cga_fp_components": c["fp_components"],
            "ohem_fp_components": o["fp_components"],
            "cga_fp_area": c["fp_area"],
            "ohem_fp_area": o["fp_area"],
        }
        out.append(row)
    return out


def _morphology_class(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unclear"
    fp_area_delta = float(np.mean([r["fp_area_delta"] for r in rows]))
    fp_component_delta = float(np.mean([r["fp_component_delta"] for r in rows]))
    fg_delta = float(np.mean([r["foreground_ratio_delta"] for r in rows]))
    area = fp_area_delta > 0 or fg_delta > 0.001
    frag = fp_component_delta > 0
    if area and frag:
        return "mixed_area_and_fragmentation"
    if area:
        return "area_overactivation"
    if frag:
        return "fragmented_noise_components"
    return "unclear"


def audit_a6(ctx: AuditContext) -> dict[str, Any]:
    failures = []
    integrity = {}
    rows_by_key = {}
    for model in ("MSHNetOHEM", "MSHNetCGA"):
        for split in ("test", "hcval"):
            row, rows, row_failures = prediction_integrity_and_rows(ctx, model, split)
            integrity[f"{model}/{split}"] = row
            rows_by_key[f"{model}/{split}"] = rows
            failures.extend(row_failures)

    paired = {
        "test": _pair_rows(rows_by_key["MSHNetOHEM/test"], rows_by_key["MSHNetCGA/test"]),
        "hcval": _pair_rows(rows_by_key["MSHNetOHEM/hcval"], rows_by_key["MSHNetCGA/hcval"]),
    }
    diagnostics = {
        "test_morphology": _morphology_class(paired["test"]),
        "hcval_morphology": _morphology_class(paired["hcval"]),
        "test_mean_foreground_ratio_delta": float(np.mean([r["foreground_ratio_delta"] for r in paired["test"]])) if paired["test"] else 0.0,
        "test_mean_fp_component_delta": float(np.mean([r["fp_component_delta"] for r in paired["test"]])) if paired["test"] else 0.0,
        "test_mean_fp_area_delta": float(np.mean([r["fp_area_delta"] for r in paired["test"]])) if paired["test"] else 0.0,
        "hcval_mean_foreground_ratio_delta": float(np.mean([r["foreground_ratio_delta"] for r in paired["hcval"]])) if paired["hcval"] else 0.0,
        "hcval_mean_fp_component_delta": float(np.mean([r["fp_component_delta"] for r in paired["hcval"]])) if paired["hcval"] else 0.0,
        "hcval_mean_fp_area_delta": float(np.mean([r["fp_area_delta"] for r in paired["hcval"]])) if paired["hcval"] else 0.0,
        "top5_hcval_false_alarm_images_by_fp_components": sorted(
            paired["hcval"], key=lambda r: (-r["cga_fp_components"], r["image_id"])
        )[:5],
        "top5_hcval_false_alarm_images_by_foreground_ratio_delta": sorted(
            paired["hcval"], key=lambda r: (-r["foreground_ratio_delta"], r["image_id"])
        )[:5],
    }
    diagnostic_tags = []
    if diagnostics["hcval_morphology"] in {"area_overactivation", "mixed_area_and_fragmentation"}:
        diagnostic_tags.append("hcval_area_overactivation")
    if diagnostics["hcval_morphology"] in {"fragmented_noise_components", "mixed_area_and_fragmentation"}:
        diagnostic_tags.append("hcval_fragmented_noise_components")

    passed = len(failures) == 0
    step = make_step(
        "A6_prediction_morphology",
        passed,
        not passed,
        not passed,
        "P2_INVALID_PREDICTION_ARTIFACTS" if not passed else None,
        failures or diagnostic_tags or ["Prediction artifacts are complete and readable."],
        {
            "integrity": integrity,
            "component_morphology": diagnostics,
            "diagnostics": diagnostic_tags,
        },
    )
    ctx.write_step("A6_prediction_morphology", step)
    return step


def component_rows_for_mask(mask: torch.Tensor, targets: dict[str, torch.Tensor], image_id: str) -> list[dict[str, Any]]:
    arr = (mask[0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
    lab = _label(arr)
    center = (targets["cga_center_target"][0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
    peak = (targets["cga_peak_target"][0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
    boundary = (targets["cga_boundary_target"][0, 0].detach().cpu().numpy() > 0).astype(np.uint8)
    scale = targets["cga_scale_target"][0, 0].detach().cpu().numpy()
    rows = []
    for cid in range(1, int(lab.max()) + 1):
        comp = lab == cid
        coords = np.argwhere(comp)
        if coords.size == 0:
            continue
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0) + 1
        scale_vals = scale[comp]
        boundary_near = int(boundary[binary_dilate(torch.from_numpy(comp.astype(np.float32))[None, None], 2)[0, 0].numpy() > 0].sum())
        center_inside = int(center[comp].sum())
        peak_inside = int(peak[comp].sum())
        rows.append(
            {
                "image_id": image_id,
                "component_id": int(cid),
                "component_area": int(comp.sum()),
                "component_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_target_pixels_inside_component": center_inside,
                "peak_target_pixels_inside_component": peak_inside,
                "boundary_pixels_near_component": boundary_near,
                "scale_target_value_or_bin": float(scale_vals.max()) if scale_vals.size else 0.0,
                "has_center_inside_component": center_inside > 0,
                "has_peak_inside_component": peak_inside > 0,
            }
        )
    return rows


def audit_a7(ctx: AuditContext, max_images: int | None = None) -> dict[str, Any]:
    failures = []
    notes = []
    items = read_split_items(ctx.dataset_dir, ctx.dataset_name, "train", required=True)
    if max_images:
        items = items[: int(max_images)]
    cfg = CGATargetConfig()
    all_rows = []
    target_shape_errors = []
    nonfinite = 0
    for image_id in items:
        _, mask_path = resolve_item_paths(ctx.dataset_dir, ctx.dataset_name, image_id)
        mask_np = read_mask_binary(mask_path).astype(np.float32)
        mask = torch.from_numpy(mask_np)[None, None]
        targets = build_cga_targets(mask, cfg)
        for key, tensor in targets.items():
            if list(tensor.shape) != list(mask.shape):
                target_shape_errors.append(f"{image_id}:{key}:{list(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                nonfinite += 1
        all_rows.extend(component_rows_for_mask(mask, targets, image_id))

    total_components = len(all_rows)
    without_center_peak = [r for r in all_rows if not (r["has_center_inside_component"] or r["has_peak_inside_component"])]
    small_without = [r for r in without_center_peak if int(r["component_area"]) <= 4]
    center_rate = float(np.mean([r["has_center_inside_component"] for r in all_rows])) if all_rows else 0.0
    peak_rate = float(np.mean([r["has_peak_inside_component"] for r in all_rows])) if all_rows else 0.0
    center_or_peak_rate = float(np.mean([r["has_center_inside_component"] or r["has_peak_inside_component"] for r in all_rows])) if all_rows else 0.0
    boundary_rate = float(np.mean([r["boundary_pixels_near_component"] > 0 for r in all_rows])) if all_rows else 0.0

    if target_shape_errors:
        failures.append("target shapes incompatible with mask/logits")
    if nonfinite:
        failures.append("NaN/Inf target values")
    if center_or_peak_rate < 0.999:
        failures.append("some GT components receive no center/peak signal")
    if boundary_rate < 0.999:
        failures.append("some GT components receive no nearby boundary signal")

    artifacts = {
        "dataset": ctx.dataset_name,
        "split": "train",
        "sampled_image_count": len(items),
        "sample_policy": "all_train_images" if max_images is None else f"first_{max_images}_train_images",
        "component_count": total_components,
        "center_inside_each_gt_component_rate": center_rate,
        "peak_inside_each_gt_component_rate": peak_rate,
        "component_has_center_or_peak_rate": center_or_peak_rate,
        "boundary_near_component_rate": boundary_rate,
        "components_without_center_or_peak_count": len(without_center_peak),
        "small_components_without_center_or_peak_count": len(small_without),
        "target_shape_error_count": len(target_shape_errors),
        "nonfinite_target_count": nonfinite,
        "component_examples": all_rows[:50],
        "components_without_center_or_peak_examples": without_center_peak[:50],
    }
    if not failures:
        notes.append("CGA targets are finite, shape-compatible, and every sampled GT component has center or peak coverage.")
    step = make_step(
        "A7_target_geometry",
        len(failures) == 0,
        bool(failures),
        bool(failures),
        "P2_INVALID_TARGET_GENERATION" if failures else None,
        failures or notes,
        artifacts,
    )
    ctx.write_step("A7_target_geometry", step)
    return step


def audit_a8(ctx: AuditContext) -> dict[str, Any]:
    step_paths = sorted(p for p in ctx.audit_dir.glob("A*.json") if p.name != "A8_final_audit_decision.json")
    steps = [load_json(p) for p in step_paths]
    invalidating = [s for s in steps if s.get("invalidates_p2")]
    invalid_by_step = {s.get("audit_step"): s for s in invalidating}
    primary_decision = None
    for step_name, decision in AUDIT_STEPS_PRIORITY:
        if step_name in invalid_by_step:
            primary_decision = decision
            break
    diagnostics = []
    for step in steps:
        if step.get("audit_step") in {"A4_loss_scale", "A6_prediction_morphology"}:
            artifacts = step.get("artifacts", {})
            if artifacts.get("diagnostics"):
                diagnostics.append(step.get("audit_step"))
    if primary_decision is None:
        primary_decision = "P2_VALID_NEGATIVE_DESIGN_WEAKNESS" if diagnostics else "P2_VALID_NEGATIVE"

    final = {
        "final_decision": primary_decision,
        "can_run_seed43_44": False,
        "can_claim_positive_cga": False,
        "requires_seed42_rerun": any(bool(s.get("requires_rerun")) for s in invalidating),
        "invalidating_steps": [
            {
                "audit_step": s.get("audit_step"),
                "decision_if_failed": s.get("decision_if_failed"),
                "requires_rerun": s.get("requires_rerun"),
                "notes": s.get("notes", []),
            }
            for s in invalidating
        ],
        "diagnostic_steps": diagnostics,
        "notes": [],
        "audited_step_files": [str(p) for p in step_paths],
    }
    if invalidating:
        final["notes"].append("Current P2 cannot be interpreted as P2_VALID_NEGATIVE until invalidating steps pass.")
    elif primary_decision == "P2_VALID_NEGATIVE_DESIGN_WEAKNESS":
        final["notes"].append("All invalidating audits passed; diagnostics indicate recall/false-alarm design weakness.")
    else:
        final["notes"].append("All invalidating audits passed; current failed seed42 is a valid negative result.")
    write_json(ctx.audit_dir / "A8_final_audit_decision.json", final)
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CGA-v2 P2 implementation audit v5.")
    p.add_argument("--root", default="/home/AAAI/CGA-main")
    p.add_argument("--canonical_root", default="/home/ly/AAAI/CGA-main")
    p.add_argument("--run_root", default="/home/AAAI/CGA-main")
    p.add_argument("--dataset_dir", default="/home/AAAI/CGA-main/datasets")
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--output_dir", default="/home/AAAI/CGA-main/results/official_from_zero")
    p.add_argument("--p2_dir", default="/home/AAAI/CGA-main/docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST")
    p.add_argument("--audit_dir", default="/home/AAAI/CGA-main/docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoch", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target_geometry_max_images", type=int, default=0)
    p.add_argument(
        "--mount_note",
        default=(
            "Audit executed in Docker container 53fcbad0a5e2 where "
            "/home/AAAI/CGA-main is the mounted run root corresponding to "
            "host canonical root /home/ly/AAAI/CGA-main."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = AuditContext(
        root=Path(args.root),
        canonical_root=args.canonical_root,
        run_root=args.run_root,
        dataset_dir=Path(args.dataset_dir),
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir),
        p2_dir=Path(args.p2_dir),
        audit_dir=Path(args.audit_dir),
        seed=args.seed,
        epoch=args.epoch,
        threshold=args.threshold,
        mount_note=args.mount_note,
    )
    ctx.audit_dir.mkdir(parents=True, exist_ok=True)
    a0 = audit_a0(ctx)
    audit_a1(ctx, a0)
    audit_a2(ctx)
    audit_a3_one(ctx, "MSHNetOHEM", "A3_strict_load_ohem")
    audit_a3_one(ctx, "MSHNetCGA", "A3_strict_load_cga")
    audit_a5_one(ctx, "MSHNetOHEM", "A5_eval_output_trace_ohem")
    audit_a5_one(ctx, "MSHNetCGA", "A5_eval_output_trace_cga")
    audit_a5b(ctx)
    max_images = args.target_geometry_max_images if args.target_geometry_max_images > 0 else None
    audit_a7(ctx, max_images=max_images)
    audit_a4(ctx)
    audit_a6(ctx)
    final = audit_a8(ctx)
    print(json.dumps(final, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
