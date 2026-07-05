#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"

A8=${A8:-docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/A8_final_audit_decision.json}

if [[ ! -f "${A8}" ]]; then
  echo "[BLOCK] Missing A8 audit decision: ${A8}" >&2
  exit 2
fi

python3 - "$A8" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

decision = data.get("final_decision") or data.get("decision")
can_run = bool(data.get("can_run_seed43_44", False))
can_claim = bool(data.get("can_claim_positive_cga", False))
invalidating_steps = data.get("invalidating_steps", [])

if can_run:
    print(json.dumps({
        "guard": "cga_v2_seed43_44",
        "pass": True,
        "decision": decision,
        "can_run_seed43_44": can_run,
        "can_claim_positive_cga": can_claim,
        "invalidating_steps": invalidating_steps,
    }, indent=2, sort_keys=True))
    sys.exit(0)

print(json.dumps({
    "guard": "cga_v2_seed43_44",
    "pass": False,
    "blocked": True,
    "reason": "Current CGA-v2 P2 audit does not allow seed43/44.",
    "decision": decision,
    "can_run_seed43_44": can_run,
    "can_claim_positive_cga": can_claim,
    "invalidating_steps": invalidating_steps,
}, indent=2, sort_keys=True))
sys.exit(1)
PY
