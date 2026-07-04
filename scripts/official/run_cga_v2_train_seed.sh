#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
: "${DATASET_DIR:=datasets}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${MODEL_NAME:=MSHNetCGA}"
: "${SEED:=42}"
: "${EPOCHS:=400}"
"${PYTHON}" train.py \
  --model_name "${MODEL_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  "$@"
