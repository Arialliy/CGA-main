#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/OHCM-MSHNet-main" >&2
  exit 2
fi
TARGET=$1
if [[ ! -d "${TARGET}" ]]; then
  echo "Target repo does not exist: ${TARGET}" >&2
  exit 2
fi
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="${TARGET}.backup.${STAMP}"
echo "Backing up ${TARGET} -> ${BACKUP}"
cp -a "${TARGET}" "${BACKUP}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "Applying overlay from ${SCRIPT_DIR} to ${TARGET}"
rsync -av --exclude '__pycache__' --exclude '.pytest_cache' "${SCRIPT_DIR}/" "${TARGET}/"
echo "Done. Run: cd ${TARGET} && bash scripts/official/run_cga_v2_contract.sh && python3 -m pytest tests/test_cga_v2_* -q"
