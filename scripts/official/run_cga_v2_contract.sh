#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
"${PYTHON}" -m py_compile \
  dataset.py \
  model/MSHNet.py model/cga_aux.py model/CGA_MSHNet.py \
  utils/cga_targets.py loss.py net.py train.py test.py evaluate.py metrics.py \
  tools/official/check_cga_v2_repo_contract.py \
  tools/official/check_cga_v2_dataset_preflight.py \
  tools/official/check_cga_v2_claim_guard.py \
  tools/official/summarize_cga_v2_one_seed.py \
  tools/official/summarize_cga_v2_multiseed.py \
  tools/official/write_cga_v2_closest_prior_art_threat_table.py
"${PYTHON}" -m pytest \
  tests/test_cga_v2_model_contract.py \
  tests/test_cga_v2_loss_contract.py \
  tests/test_cga_v2_targets.py \
  tests/test_cga_v2_metrics_diagnostics.py \
  tests/test_cga_v2_dataset_registry.py \
  tests/test_cga_v2_claim_guard.py -q
"${PYTHON}" -m tools.official.check_cga_v2_repo_contract "$@"
