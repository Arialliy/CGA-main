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
: "${EPOCH:=400}"
: "${SPLIT:=test}"
: "${CHECKPOINT:=results/official/${MODEL_NAME}/seed${SEED}/${DATASET_NAME}/${MODEL_NAME}_${EPOCH}.pth.tar}"
"${PYTHON}" test.py \
  --model_name "${MODEL_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --seed "${SEED}" \
  --checkpoint "${CHECKPOINT}" \
  --threshold 0.5 \
  "$@"
