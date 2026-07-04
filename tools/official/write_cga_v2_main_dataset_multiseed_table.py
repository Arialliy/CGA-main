"""Create a Markdown table from a Gate-D multiseed summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x):
    return f"{float(x):+.5f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate_d_summary", required=True)
    p.add_argument("--output", default="docs/paper/cga_v2_aaai/main_dataset_multiseed_table.md")
    args = p.parse_args()
    data = json.loads(Path(args.gate_d_summary).read_text(encoding="utf-8"))
    lines = ["# CGA-v2 NUDT-SIRST Multiseed Table", ""]
    for split in ["full", "hcval"]:
        lines.append(f"## {split.upper()} per-seed deltas")
        lines.append("")
        lines.append("| Seed | mIoU | Precision | Pd | FA ppm |")
        lines.append("|---:|---:|---:|---:|---:|")
        for seed, row in data.get("per_seed", {}).items():
            d = row.get(split, {}).get("delta")
            if d is None:
                lines.append(f"| {seed} | n/a | n/a | n/a | n/a |")
            else:
                lines.append(f"| {seed} | {fmt(d['mIoU'])} | {fmt(d.get('Precision', 0.0))} | {fmt(d['Pd'])} | {fmt(d['FA_ppm'])} |")
        lines.append("")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
