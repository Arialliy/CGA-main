# CGA-v2 Repository-Style Code Overlay

This ZIP contains a repository-level CGA-v2 code overlay for OHCM-MSHNet.  It is not a single `model.py`; it includes the model, loss, net factory, train/test/evaluate scripts, metrics, official tools, shell runners, tests, configs, and internal docs.

## Included paths

```text
model/MSHNet.py
model/CGA_MSHNet.py
model/cga_aux.py
utils/cga_targets.py
loss.py
net.py
train.py
test.py
evaluate.py
metrics.py
tools/official/*.py
scripts/official/*.sh
tests/*.py
configs/cga_v2_default.json
docs/internal/cga_v2/repo_style_code_package.md
docs/paper/cga_v2_aaai/README.md
```

## Apply strategy

Use this as an overlay, not a blind destructive replacement.  Back up the current repository first:

```bash
cp -a /home/ly/AAAI/OHCM-MSHNet-main /home/ly/AAAI/OHCM-MSHNet-main.backup.$(date +%Y%m%d_%H%M%S)
rsync -av cga_v2_repo_grade_code_overlay/ /home/ly/AAAI/OHCM-MSHNet-main/
```

Then run:

```bash
cd /home/ly/AAAI/OHCM-MSHNet-main
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
bash scripts/official/run_cga_v2_contract.sh
"${PYTHON}" -m pytest \
  tests/test_cga_v2_model_contract.py \
  tests/test_cga_v2_loss_contract.py \
  tests/test_cga_v2_targets.py \
  tests/test_cga_v2_metrics_diagnostics.py \
  tests/test_cga_v2_claim_guard.py -q
git diff --check
```

## Training example

```bash
DATASET_DIR=datasets DATASET_NAME=NUDT-SIRST MODEL_NAME=MSHNetCGA SEED=42 EPOCHS=400 \
  bash scripts/official/run_cga_v2_train_seed.sh
```

## Evaluation example

```bash
DATASET_DIR=datasets DATASET_NAME=NUDT-SIRST MODEL_NAME=MSHNetCGA SEED=42 EPOCH=400 \
  bash scripts/official/run_cga_v2_test_seed.sh
```

## Claim boundary

This package implements a target-preserving component-geometry regularizer.  It should not be presented as a complete hard-clutter closure mechanism or as external/blind validation.
