"""Write a diagnostic/failure-case Markdown pack from target audit JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostic_summary", required=True)
    p.add_argument("--output", default="docs/paper/cga_v2_aaai/failure_cases.md")
    args = p.parse_args()
    data = json.loads(Path(args.diagnostic_summary).read_text(encoding="utf-8"))
    lines = ["# CGA-v2 Failure / Diagnostic Cases", ""]
    lines.append(f"Decision: `{data.get('decision', 'unknown')}`")
    lines.append(f"Retain as: `{data.get('retain_as', 'unknown')}`")
    lines.append("")
    for item in data.get("missed_target_details", []):
        lines.append(f"## {item.get('item')} target {item.get('target_id')}")
        lines.append("")
        lines.append(f"- Severity: `{item.get('severity')}`")
        lines.append(f"- Target area: `{item.get('target_area_pixels')}` pixels")
        lines.append(f"- OHEM detected: `{item.get('ohem_detected')}`")
        lines.append(f"- CGA detected: `{item.get('cga_detected')}`")
        lines.append(f"- OHEM IoU: `{item.get('ohem_iou_with_target')}`")
        lines.append(f"- CGA IoU: `{item.get('cga_iou_with_target')}`")
        lines.append(f"- CGA max prob on target: `{item.get('cga_max_probability_on_target')}`")
        lines.append("")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
