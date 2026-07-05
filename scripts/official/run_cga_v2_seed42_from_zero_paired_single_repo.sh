#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

DATASET_DIR=${DATASET_DIR:-/home/ly/AAAI/CGA-main/datasets}
DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-400}
EPOCH=${EPOCH:-${EPOCHS}}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ly/AAAI/CGA-main/results/official_from_zero}
CUDA_DEVICE=${CUDA_DEVICE:-1}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

PREFLIGHT_SUMMARY="docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json"
P1A_SUMMARY="docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json"
P2_OUTPUT="docs/internal/cga_v2/gate_p2_from_zero_seed${SEED}_${DATASET_NAME}/summary.json"

mkdir -p "$(dirname "${P2_OUTPUT}")"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 2
  fi
}

check_json_gate_pass() {
  local path="$1"
  "${PYTHON}" - "$path" <<'PY'
import json
import sys

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f)
if obj.get("gate_pass") is not True:
    raise SystemExit(f"gate_pass is not true in {p}: {obj.get('gate_pass')!r}")
PY
}

require_file "${DATASET_DIR}/${DATASET_NAME}/img_idx/train_${DATASET_NAME}.txt"
require_file "${DATASET_DIR}/${DATASET_NAME}/img_idx/test_${DATASET_NAME}.txt"
require_file "${DATASET_DIR}/${DATASET_NAME}/img_idx/hcval_${DATASET_NAME}.txt"

DATASET_DIR="${DATASET_DIR}" \
DATASET_NAME="${DATASET_NAME}" \
OUTPUT="${PREFLIGHT_SUMMARY}" \
bash scripts/official/run_cga_v2_dataset_preflight.sh
check_json_gate_pass "${PREFLIGHT_SUMMARY}"

require_file "${P1A_SUMMARY}"
check_json_gate_pass "${P1A_SUMMARY}"

mkdir -p "${OUTPUT_DIR}"

run_train() {
  local model_name="$1"
  echo "[RUN] Train ${model_name}, seed=${SEED}, epochs=${EPOCHS}"
  "${PYTHON}" train.py \
    --model_name "${model_name}" \
    --evidence_mode paper \
    --protocol controlled \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch_size 8 \
    --patch_size 256 \
    --num_workers 4 \
    --mshnet_warm_epoch 5 \
    --cga_start_epoch 1 \
    --cga_ramp_epochs 40 \
    --p1_preflight_passed \
    --p1a_hcval_source_audit_passed \
    --output_dir "${OUTPUT_DIR}"
}

run_eval() {
  local model_name="$1"
  local split="$2"
  local ckpt="${OUTPUT_DIR}/${model_name}/seed${SEED}/${DATASET_NAME}/${model_name}_${EPOCH}.pth.tar"
  require_file "${ckpt}"
  echo "[RUN] Eval ${model_name}, split=${split}, checkpoint=${ckpt}"
  MODEL_NAME="${model_name}" \
  DATASET_DIR="${DATASET_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  SEED="${SEED}" \
  EPOCH="${EPOCH}" \
  CHECKPOINT="${ckpt}" \
  SPLIT="${split}" \
  bash scripts/official/run_cga_v2_test_seed.sh --output_dir "${OUTPUT_DIR}"
}

run_train MSHNetOHEM
run_train MSHNetCGA

run_eval MSHNetOHEM test
run_eval MSHNetOHEM hcval
run_eval MSHNetCGA test
run_eval MSHNetCGA hcval

BASE_FULL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
CAND_FULL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
BASE_HCVAL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"
CAND_HCVAL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"

require_file "${BASE_FULL}"
require_file "${CAND_FULL}"
require_file "${BASE_HCVAL}"
require_file "${CAND_HCVAL}"

"${PYTHON}" -m tools.official.summarize_cga_v2_one_seed \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epoch "${EPOCH}" \
  --threshold 0.5 \
  --baseline MSHNetOHEM \
  --candidate MSHNetCGA \
  --preflight_summary "${PREFLIGHT_SUMMARY}" \
  --baseline_full "${BASE_FULL}" \
  --candidate_full "${CAND_FULL}" \
  --baseline_hcval "${BASE_HCVAL}" \
  --candidate_hcval "${CAND_HCVAL}" \
  --output "${P2_OUTPUT}"

echo "[DONE] Paired from-zero seed${SEED} summary: ${P2_OUTPUT}"
