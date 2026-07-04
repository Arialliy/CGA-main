#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
: "${DATASET_DIR:=datasets}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${DATASET_REGISTRY:=configs/datasets.yaml}"
: "${OUTPUT:=docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json}"
"${PYTHON}" -m tools.official.check_cga_v2_dataset_preflight \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --registry "${DATASET_REGISTRY}" \
  --output "${OUTPUT}" \
  "$@"
