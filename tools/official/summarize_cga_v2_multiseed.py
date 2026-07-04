"""Aggregate seed42/43/44 paired summaries for the P3 NUDT gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _mean_delta(seed_rows: dict[str, dict[str, Any]], split: str, key: str) -> float | None:
    values = []
    for row in seed_rows.values():
        delta = row.get(split, {}).get("delta")
        if delta is None:
            return None
        values.append(float(delta[key]))
    return float(mean(values)) if values else None


def _ge_zero_count(seed_rows: dict[str, dict[str, Any]], split: str, key: str) -> int:
    count = 0
    for row in seed_rows.values():
        delta = row.get(split, {}).get("delta")
        if delta is not None and float(delta[key]) >= 0.0:
            count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--epoch", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--baseline", default="MSHNetOHEM")
    p.add_argument("--candidate", default="MSHNetCGA")
    p.add_argument("--seed_summaries", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    per_seed = {}
    for path_s in args.seed_summaries:
        row = json.loads(Path(path_s).read_text(encoding="utf-8"))
        per_seed[str(row["seed"])] = row
    seeds = sorted(int(s) for s in per_seed)
    full_mean = {k: _mean_delta(per_seed, "full", k) for k in ["mIoU", "Precision", "Pd", "FA_ppm"]}
    hcval_mean = {k: _mean_delta(per_seed, "hcval", k) for k in ["mIoU", "Pd", "FA_ppm"]}
    full_pass_conditions = {
        "mean_delta_mIoU_gt_0": full_mean["mIoU"] is not None and full_mean["mIoU"] > 0.0,
        "mean_delta_Precision_gt_0": full_mean["Precision"] is not None and full_mean["Precision"] > 0.0,
        "mean_delta_Pd_ge_minus_0_001": full_mean["Pd"] is not None and full_mean["Pd"] >= -0.001,
        "mean_delta_FA_ppm_lt_0": full_mean["FA_ppm"] is not None and full_mean["FA_ppm"] < 0.0,
    }
    hcval_nonnegative_miou_count = _ge_zero_count(per_seed, "hcval", "mIoU")
    hcval_pass_conditions = {
        "mean_delta_mIoU_gt_0": hcval_mean["mIoU"] is not None and hcval_mean["mIoU"] > 0.0,
        "mean_delta_Pd_ge_minus_0_001": hcval_mean["Pd"] is not None and hcval_mean["Pd"] >= -0.001,
        "at_least_2_of_3_seed_delta_mIoU_ge_0": hcval_nonnegative_miou_count >= 2,
        "all_seed_FA_decrease_required": False,
    }
    gate_pass = all(full_pass_conditions.values()) and all(
        v for k, v in hcval_pass_conditions.items() if k != "all_seed_FA_decrease_required"
    )
    summary = {
        "gate": "Gate-CGA-v2-P3-NUDT-multiseed-reproduction",
        "dataset_name": args.dataset_name,
        "seeds": seeds,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "epoch": args.epoch,
        "threshold": args.threshold,
        "decision_rule_predeclared": True,
        "decision_rule_name": "NUDT multiseed main evidence rule",
        "full_mean_delta": full_mean,
        "hcval_mean_delta": hcval_mean,
        "full_pass_conditions": full_pass_conditions,
        "hcval_pass_conditions": hcval_pass_conditions,
        "per_seed": per_seed,
        "gate_pass": gate_pass,
        "decision": "NUDT_MAIN_EVIDENCE_FORMED" if gate_pass else "STOP_NEW_REPO_CGA_V2_AS_AAAI_MAIN_METHOD",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
