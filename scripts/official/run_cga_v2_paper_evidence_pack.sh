#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
: "${GATE_D_SUMMARY:=docs/internal/cga_v2/gate_d_seed43_44_NUDT-SIRST/summary.json}"
if [[ -f "${GATE_D_SUMMARY}" ]]; then
  "${PYTHON}" -m tools.official.write_cga_v2_main_dataset_multiseed_table \
    --gate_d_summary "${GATE_D_SUMMARY}" \
    --output docs/paper/cga_v2_aaai/main_dataset_multiseed_table.md
else
  echo "Missing ${GATE_D_SUMMARY}; skip multiseed table generation" >&2
fi
