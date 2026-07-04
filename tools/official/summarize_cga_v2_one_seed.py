"""Write the P2-style one-seed paired reproduction summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "full_delta_mIoU_min": 0.02,
    "full_delta_precision_min": 0.01,
    "full_delta_pd_min": -0.001,
    "full_delta_fa_ppm_max": 0.0,
    "hcval_delta_miou_min": 0.0,
    "hcval_delta_pd_min": -0.001,
}


def _load_optional(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    keys = ["mIoU", "nIoU", "Precision", "Recall", "F1", "Pd", "FA", "FA_ppm", "FP_components"]
    return {k: float(candidate.get(k, 0.0)) - float(baseline.get(k, 0.0)) for k in keys}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epoch", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--baseline", default="MSHNetOHEM")
    p.add_argument("--candidate", default="MSHNetCGA")
    p.add_argument("--preflight_summary", required=True)
    p.add_argument("--baseline_full", required=True)
    p.add_argument("--candidate_full", required=True)
    p.add_argument("--baseline_hcval", default="")
    p.add_argument("--candidate_hcval", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    preflight = json.loads(Path(args.preflight_summary).read_text(encoding="utf-8"))
    baseline_full = json.loads(Path(args.baseline_full).read_text(encoding="utf-8"))
    candidate_full = json.loads(Path(args.candidate_full).read_text(encoding="utf-8"))
    baseline_hcval = _load_optional(args.baseline_hcval)
    candidate_hcval = _load_optional(args.candidate_hcval)
    full_delta = _delta(baseline_full, candidate_full)
    hcval_delta = _delta(baseline_hcval, candidate_hcval) if baseline_hcval and candidate_hcval else None
    full_pass = (
        full_delta["mIoU"] >= THRESHOLDS["full_delta_mIoU_min"]
        and full_delta["Precision"] >= THRESHOLDS["full_delta_precision_min"]
        and full_delta["Pd"] >= THRESHOLDS["full_delta_pd_min"]
        and full_delta["FA_ppm"] <= THRESHOLDS["full_delta_fa_ppm_max"]
    )
    hcval_available = hcval_delta is not None
    hcval_pass = bool(
        hcval_available
        and hcval_delta["mIoU"] >= THRESHOLDS["hcval_delta_miou_min"]
        and hcval_delta["Pd"] >= THRESHOLDS["hcval_delta_pd_min"]
    )
    gate_pass = bool(full_pass and hcval_pass)
    summary = {
        "gate": "Gate-CGA-v2-P2-seed42-reproduction" if args.seed == 42 else "Gate-CGA-v2-one-seed-paired-reproduction",
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "epoch": args.epoch,
        "threshold": args.threshold,
        "decision_rule_predeclared": True,
        "decision_rule_name": "seed42 reproduction decision rule",
        "thresholds": THRESHOLDS,
        "dataset_preflight_summary": args.preflight_summary,
        "dataset_registry_sha256": preflight.get("dataset_registry_sha256"),
        "train_list_sha256": preflight.get("train_list_sha256"),
        "test_list_sha256": preflight.get("test_list_sha256"),
        "hcval_list_sha256": preflight.get("hcval_list_sha256"),
        "full": {"baseline": baseline_full, "candidate": candidate_full, "delta": full_delta},
        "hcval": {
            "available": hcval_available,
            "baseline": baseline_hcval,
            "candidate": candidate_hcval,
            "delta": hcval_delta,
        },
        "pass_conditions": {
            "full_rule_pass": bool(full_pass),
            "hcval_rule_pass": bool(hcval_pass),
            "hcval_available": bool(hcval_available),
        },
        "gate_pass": gate_pass,
        "decision": "MAY_PROCEED_TO_NUDT_MULTISEED_REPRODUCTION" if gate_pass else "P2_FAIL_IMPL_AUDIT_ALLOWED",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
