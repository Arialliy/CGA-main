"""Compare paired MSHNetOHEM vs MSHNetCGA summaries for one seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def get_metric(summary: dict[str, Any], key: str, split: str | None = None) -> float:
    """Read current top-level metrics or future nested split metrics."""
    candidates = (key, key.lower(), key.upper())
    for candidate in candidates:
        if candidate in summary:
            return float(summary[candidate])
        if split and isinstance(summary.get(split), dict) and candidate in summary[split]:
            return float(summary[split][candidate])
    raise KeyError(f"Missing metric {key!r} in summary; available keys={list(summary.keys())}")


def compute_delta_gate(base: dict[str, Any], cga: dict[str, Any], *, split: str | None = None) -> dict[str, Any]:
    base_miou = get_metric(base, "mIoU", split=split)
    cga_miou = get_metric(cga, "mIoU", split=split)
    base_precision = get_metric(base, "Precision", split=split)
    cga_precision = get_metric(cga, "Precision", split=split)
    base_fa = get_metric(base, "FA", split=split)
    cga_fa = get_metric(cga, "FA", split=split)

    delta = {
        "baseline_mIoU": base_miou,
        "candidate_mIoU": cga_miou,
        "delta_mIoU": cga_miou - base_miou,
        "baseline_Precision": base_precision,
        "candidate_Precision": cga_precision,
        "delta_Precision": cga_precision - base_precision,
        "baseline_FA": base_fa,
        "candidate_FA": cga_fa,
        "delta_FA": cga_fa - base_fa,
    }
    delta["seed_primary_gate_pass"] = bool(
        delta["delta_mIoU"] >= 0.020
        and delta["delta_Precision"] >= 0.010
        and delta["delta_FA"] <= 0.0
    )
    return delta


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired CGA vs OHEM seed summaries.")
    parser.add_argument("--baseline_full", required=True)
    parser.add_argument("--candidate_full", required=True)
    parser.add_argument("--baseline_hcval", default="")
    parser.add_argument("--candidate_hcval", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_full = _load(args.baseline_full)
    candidate_full = _load(args.candidate_full)
    full = compute_delta_gate(baseline_full, candidate_full)

    hcval = None
    if args.baseline_hcval and args.candidate_hcval:
        hcval = compute_delta_gate(_load(args.baseline_hcval), _load(args.candidate_hcval))

    summary = {
        "gate": "CGA-v2-evidence-first-seed-paired-delta",
        "seed": args.seed,
        "baseline": baseline_full.get("model", "MSHNetOHEM"),
        "candidate": candidate_full.get("model", "MSHNetCGA"),
        "threshold": candidate_full.get("threshold", baseline_full.get("threshold")),
        "threshold_selection": candidate_full.get(
            "threshold_selection",
            baseline_full.get("threshold_selection", "fixed_predeclared"),
        ),
        "full": full,
        "hcval": hcval,
        "seed42_primary_gate_pass": bool(full["seed_primary_gate_pass"]) if args.seed == 42 else None,
        "decision": "RUN_SEED43_44" if full["seed_primary_gate_pass"] else "STOP_AAAI_MAIN_ROUTE",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not full["seed_primary_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
