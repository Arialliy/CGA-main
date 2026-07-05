#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"\bCGA-v2\s+improves\b",
    r"\bCGA\s+improves\b",
    r"\bCGA-v2\s+reduces\s+false\s+alarms\b",
    r"\bCGA\s+reduces\s+false\s+alarms\b",
    r"\bhard[- ]clutter\s+robust\b",
    r"\bpositive\s+Full\s+and\s+mean\s+HC-Val\s+performance\b",
    r"\bready\s+for\s+seed43/44\b",
    r"\bAAAI-main\s+ready\b",
    r"\buniversally\s+improves\b",
    r"\bplug-and-play\s+across\s+IRSTD\s+detectors\b",
]

ALLOW_CONTEXT = [
    "Do not claim",
    "Forbidden",
    "Blocked interpretation",
    "Blocked actions",
    "not valid as a positive",
    "does not support",
]


def is_allowed_context(text: str, start: int) -> bool:
    lo = max(0, start - 1000)
    ctx = text[lo:start]
    return any(marker.lower() in ctx.lower() for marker in ALLOW_CONTEXT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/ly/AAAI/CGA-main")
    p.add_argument(
        "--paths",
        nargs="*",
        default=["README.md", "docs", "scripts"],
        help="Files or directories to scan.",
    )
    p.add_argument("--output", default="")
    args = p.parse_args()

    root = Path(args.root).resolve()
    violations = []

    files = []
    for item in args.paths:
        path = (root / item).resolve()
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        else:
            files.extend(
                f for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in {".md", ".txt", ".py", ".sh"}
            )

    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(file.relative_to(root))
        for pat in FORBIDDEN_PATTERNS:
            for match in re.finditer(pat, text, flags=re.IGNORECASE):
                if is_allowed_context(text, match.start()):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                violations.append({
                    "file": rel,
                    "line": line,
                    "pattern": pat,
                    "match": match.group(0),
                })

    result = {
        "check": "no_positive_cga_v2_claims",
        "pass": len(violations) == 0,
        "violations": violations,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
