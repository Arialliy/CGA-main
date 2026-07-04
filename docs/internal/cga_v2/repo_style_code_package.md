# CGA-v2 Repository-Style Paper-Grade Code Package

This overlay follows the OHCM-MSHNet repository layout:

```text
model/
utils/
loss.py
net.py
train.py
test.py
evaluate.py
metrics.py
tools/official/
scripts/official/
tests/
docs/internal/
```

## Method contract

- Candidate: `MSHNetCGA / CGA-v2 base`.
- Baseline: `MSHNetOHEM`.
- Inference path: unchanged MSHNet final logit -> sigmoid -> threshold 0.5.
- Training path: component-geometry auxiliary heads for center, boundary, scale, and peak targets.

## Claim boundary

Use the safe claim:

> target-preserving component-geometry regularization

Do not claim:

- complete hard-clutter closure;
- every auxiliary head is necessary;
- external / blind validation;
- validation on NUAA-SIRST or IRSTD-1K unless separately supported by frozen protocol.

## Minimal checks

```bash
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON}" -m py_compile \
  model/MSHNet.py model/cga_aux.py model/CGA_MSHNet.py \
  utils/cga_targets.py loss.py net.py train.py test.py evaluate.py metrics.py \
  tools/official/check_cga_v2_repo_contract.py

"${PYTHON}" -m pytest \
  tests/test_cga_v2_model_contract.py \
  tests/test_cga_v2_loss_contract.py \
  tests/test_cga_v2_targets.py \
  tests/test_cga_v2_metrics_diagnostics.py \
  tests/test_cga_v2_claim_guard.py -q
```
