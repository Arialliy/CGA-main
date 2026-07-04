"""Create the P1.5 closest prior-art threat table without borrowing results."""
from __future__ import annotations

from pathlib import Path

import yaml


BASELINES = {
    "fair_paired_comparison": [
        {
            "name": "MSHNetOHEM",
            "status": "same-repo paired baseline",
            "comparison_use": "Fair paired comparison against MSHNetCGA under the frozen dataset, seed, epoch, and threshold protocol.",
            "numbers_policy": "Use only current-repo regenerated metrics.",
        }
    ],
    "literature_only_threats": [
        {
            "name": "MSHNet / SLS",
            "status": "closest architecture and supervision threat",
            "comparison_use": "Discuss as prior-art motivation and reviewer threat.",
            "numbers_policy": "Do not present literature numbers as same-protocol fair comparison.",
        },
        {
            "name": "ISNet / shape-edge reconstruction",
            "status": "shape-aware IRSTD threat",
            "comparison_use": "Separate from the current-repo paired MSHNetOHEM comparison.",
            "numbers_policy": "Literature-only unless reproduced in this repository under the frozen protocol.",
        },
        {
            "name": "PConv + SD Loss",
            "status": "loss/design threat",
            "comparison_use": "Use to frame component/shape regularization differences.",
            "numbers_policy": "Literature-only unless reproduced in this repository under the frozen protocol.",
        },
        {
            "name": "Other IRSTD SOTA",
            "status": "broad benchmark threat",
            "comparison_use": "Mention only as external context, not as a claim of superiority.",
            "numbers_policy": "No SOTA claim from cross-paper numbers.",
        },
    ],
}


def main() -> None:
    cfg_path = Path("configs/closest_baselines.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(BASELINES, sort_keys=False), encoding="utf-8")

    lines = [
        "# Closest Prior-Art Threat Table",
        "",
        "This table separates current-repo fair paired evidence from literature-only threats.",
        "",
        "## Fair Paired Comparison",
        "",
        "| Method | Status | Use | Numbers policy |",
        "|---|---|---|---|",
    ]
    for row in BASELINES["fair_paired_comparison"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['comparison_use']} | {row['numbers_policy']} |")
    lines.extend([
        "",
        "## Literature-Only Threats",
        "",
        "| Method | Status | Use | Numbers policy |",
        "|---|---|---|---|",
    ])
    for row in BASELINES["literature_only_threats"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['comparison_use']} | {row['numbers_policy']} |")
    lines.extend([
        "",
        "Do not present literature-only numbers as fair same-protocol comparisons.",
    ])
    out = Path("docs/paper/cga_v2_aaai/08_closest_prior_art_threat_table.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(cfg_path)


if __name__ == "__main__":
    main()
