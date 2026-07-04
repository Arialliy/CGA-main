"""Fail if manuscript text contains rejected CGA-v2 claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REJECTED_PHRASES = [
    "state-of-the-art",
    "sota",
    "external validation",
    "externally validated",
    "transfer learning",
    "fully closes hard-clutter",
    "complete hard-clutter closure",
    "blind validated",
    "all auxiliary targets are necessary",
    "all auxiliary heads are necessary",
    "validated on NUAA-SIRST and IRSTD-1K",
]

ALLOWED_CONTEXT_MARKERS = [
    "do not claim",
    "rejected claims",
    "limitations",
    "reviewer risks",
    "unsafe wording",
    "failure analysis",
    "diagnostic-only evidence",
    "literature-only",
    "not claim",
    "forbidden",
]


def _line_allowed(lines: list[str], idx: int, heading_context: str = "") -> bool:
    if any(marker in heading_context for marker in ALLOWED_CONTEXT_MARKERS):
        return True
    window = lines[max(0, idx - 4) : idx + 1]
    text = "\n".join(window).lower()
    return any(marker in text for marker in ALLOWED_CONTEXT_MARKERS)


def find_rejected_phrases(text: str) -> list[dict[str, object]]:
    findings = []
    lines = text.splitlines()
    heading_context = ""
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.lstrip().startswith("#"):
            heading_context = lower
        for phrase in REJECTED_PHRASES:
            if phrase.lower() in lower and not _line_allowed(lines, idx, heading_context):
                findings.append({"line": idx + 1, "phrase": phrase})
    return findings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--output", default="docs/internal/cga_v2/claim_guard_summary.json")
    args = p.parse_args()
    findings = []
    for path_s in args.paths:
        path = Path(path_s)
        if path.is_dir():
            files = list(path.rglob("*.md")) + list(path.rglob("*.tex"))
        else:
            files = [path]
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for finding in find_rejected_phrases(text):
                findings.append({"file": str(f), **finding})
    summary = {"gate": "Gate-CGA-v2-claim-guard", "gate_pass": not findings, "findings": findings}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
