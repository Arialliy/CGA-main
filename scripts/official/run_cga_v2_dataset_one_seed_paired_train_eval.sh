#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
: "${DATASET_DIR:=datasets}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${SEED:=42}"
: "${EPOCHS:=400}"
: "${EPOCH:=${EPOCHS}}"
: "${OUTPUT_DIR:=results/official}"
: "${PREFLIGHT_SUMMARY:=docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json}"
: "${P1A_SUMMARY:=docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json}"
: "${P2_OUTPUT:=docs/internal/cga_v2/gate_p2_seed${SEED}_${DATASET_NAME}/summary.json}"

DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" OUTPUT="${PREFLIGHT_SUMMARY}" \
  bash scripts/official/run_cga_v2_dataset_preflight.sh

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

check_json_gate_pass "${PREFLIGHT_SUMMARY}"
check_json_gate_pass "${P1A_SUMMARY}"

split_exists() {
  local split_name="$1"
  "${PYTHON}" - "$DATASET_DIR" "$DATASET_NAME" "$split_name" <<'PY'
import sys
from dataset import split_exists
raise SystemExit(0 if split_exists(sys.argv[1], sys.argv[2], sys.argv[3]) else 1)
PY
}

run_train_eval() {
  local model_name="$1"
  local ckpt="${OUTPUT_DIR}/${model_name}/seed${SEED}/${DATASET_NAME}/${model_name}_${EPOCH}.pth.tar"
  if [[ ! -f "${ckpt}" || "${FORCE_TRAIN:-0}" == "1" ]]; then
    MODEL_NAME="${model_name}" DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" \
      SEED="${SEED}" EPOCHS="${EPOCHS}" \
      bash scripts/official/run_cga_v2_train_seed.sh \
        --evidence_mode paper \
        --protocol controlled \
        --mshnet_warm_epoch 5 \
        --cga_start_epoch 1 \
        --cga_ramp_epochs 40 \
        --p1_preflight_passed \
        --p1a_hcval_source_audit_passed \
        --output_dir "${OUTPUT_DIR}"
  fi
  MODEL_NAME="${model_name}" DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" \
    SEED="${SEED}" EPOCH="${EPOCH}" CHECKPOINT="${ckpt}" SPLIT="test" \
    bash scripts/official/run_cga_v2_test_seed.sh --output_dir "${OUTPUT_DIR}"
  if split_exists hcval; then
    MODEL_NAME="${model_name}" DATASET_DIR="${DATASET_DIR}" DATASET_NAME="${DATASET_NAME}" \
      SEED="${SEED}" EPOCH="${EPOCH}" CHECKPOINT="${ckpt}" SPLIT="hcval" \
      bash scripts/official/run_cga_v2_test_seed.sh --output_dir "${OUTPUT_DIR}"
  fi
}

run_train_eval MSHNetOHEM
run_train_eval MSHNetCGA

BASE_FULL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
CAND_FULL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
BASE_HCVAL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"
CAND_HCVAL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"

HCVAL_ARGS=()
if [[ -f "${BASE_HCVAL}" && -f "${CAND_HCVAL}" ]]; then
  HCVAL_ARGS=(--baseline_hcval "${BASE_HCVAL}" --candidate_hcval "${CAND_HCVAL}")
fi

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
  "${HCVAL_ARGS[@]}" \
  --output "${P2_OUTPUT}"
