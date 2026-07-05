# CGA-v2 P2 Seed42 Failure: Implementation Audit Plan and Minimal Code Changes

## 0. Verdict

The seed42 from-zero paired run has failed the predeclared P2 gate.

```text
Gate: Gate-CGA-v2-P2-seed42-reproduction
Decision: P2_FAIL_IMPL_AUDIT_ALLOWED
Gate pass: false
```

Therefore:

```text
Do not run seed43/seed44.
Do not start ablation.
Do not build failure-pack narrative yet.
Do not write paper-evidence claims.
Enter implementation audit.
```

The failure is not borderline positive. The Full split is slightly worse, and HC-Val shows a severe false-alarm / precision collapse.

---

## 1. Result interpretation

### 1.1 Full test split

| Metric | OHEM | CGA | Delta | Interpretation |
|---|---:|---:|---:|---|
| mIoU | 0.916185 | 0.913089 | -0.003096 | Fails +0.020 gate |
| Precision | 0.955114 | 0.952974 | -0.002140 | Fails +0.010 gate |
| Pd | 0.984127 | 0.989418 | +0.005291 | Target preservation improves slightly |
| FA_ppm | 12.317 | 15.236 | +2.918 | False alarms increase |

Conclusion:

```text
CGA is slightly more aggressive: Pd improves, but precision and FA worsen.
This is not acceptable under the predeclared rescue rule.
```

### 1.2 HC-Val split

| Metric | OHEM | CGA | Delta | Interpretation |
|---|---:|---:|---:|---|
| mIoU | 0.781553 | 0.509346 | -0.272208 | Severe degradation |
| Precision | 0.836364 | 0.527419 | -0.308944 | Severe false-positive behavior |
| Pd | 0.833333 | 1.000000 | +0.166667 | CGA finds all targets |
| FA_ppm | 111.898 | 706.991 | +595.093 | False-alarm explosion |

Conclusion:

```text
The current CGA configuration behaves like target-preserving but clutter-amplifying supervision.
This is useful as a diagnostic, but not as a paper claim.
```

---

## 2. Immediate decision

Follow the predeclared rule:

```text
If seed42 fails the primary gate, stop the AAAI-main route.
If seed42 passes, only then run seed43/44.
```

Current seed42 fails both Full and HC-Val rules.

Decision:

```text
STOP seed43/44.
START implementation audit.
```

---

## 3. Important path issue to audit

The canonical repository path is:

```text
/home/md0/ly/CGA-main
```

But the provided P2 summary contains result paths like:

```text
/home/AAAI/CGA-main/results/official_from_zero/...
```

This may be harmless if `/home/AAAI/CGA-main` is a Docker/container mount or symlink to `/home/md0/ly/CGA-main`; otherwise, the result manifest violates the canonical path contract.

Run:

```bash
cd /home/md0/ly/CGA-main

readlink -f /home/md0/ly/CGA-main || true
readlink -f /home/AAAI/CGA-main || true

find /home/md0/ly/CGA-main/results/official_from_zero -maxdepth 4 -type f | head || true
find /home/AAAI/CGA-main/results/official_from_zero -maxdepth 4 -type f | head || true

grep -R "/home/AAAI/CGA-main\|/home/ly/AAAI/CGA-main\|OHCM-MSHNet-main" -n \
  docs results scripts tools train.py test.py 2>/dev/null || true
```

Decision rule:

```text
If both paths resolve to the same real location: record this in the audit note.
If they do not: mark this P2 result as path-contaminated and rerun only after path correction.
```

---

## 4. Minimal audit scope

Do not change the CGA method yet.

Audit in this order:

```text
A1. Artifact/path integrity
A2. Checkpoint metadata integrity
A3. Strict checkpoint-load compatibility
A4. Train-log loss-scale audit
A5. Evaluation symmetry audit
A6. Prediction-mass / false-positive audit
A7. Adapter and feature-source audit
A8. Decision: implementation bug vs negative method result
```

---

## 5. A1 artifact/path integrity

Expected artifacts:

```text
/home/md0/ly/CGA-main/results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar
/home/md0/ly/CGA-main/results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar
/home/md0/ly/CGA-main/results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/test/summary_metrics.json
/home/md0/ly/CGA-main/results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/test/summary_metrics.json
/home/md0/ly/CGA-main/results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/hcval/summary_metrics.json
/home/md0/ly/CGA-main/results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/hcval/summary_metrics.json
```

Run the audit helper:

```bash
cd /home/md0/ly/CGA-main

python3 tools/official/audit_cga_v2_p2_failure.py \
  --root /home/md0/ly/CGA-main \
  --canonical_root /home/md0/ly/CGA-main \
  --p2_summary docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/summary.json \
  --output docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/impl_audit.json
```

If your P2 summary is under a different directory, pass the exact path with `--p2_summary`.

---

## 6. A2 checkpoint metadata integrity

For both checkpoints, verify:

```text
epoch = 400
seed = 42
dataset = NUDT-SIRST
protocol = controlled
evidence_mode = paper
p1_preflight_passed = true
p1a_hcval_source_audit_passed = true
paper_evidence_allowed = true
fallback_regularizer_used = false
```

For OHEM:

```text
use_cga = false
regularizer_impl = none
```

For CGA:

```text
use_cga = true
regularizer_impl = center_boundary_scale_peak
```

Manual command:

```bash
cd /home/md0/ly/CGA-main

python3 - <<'PY'
import json
import torch
from pathlib import Path

paths = [
    Path('results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar'),
    Path('results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar'),
]
keys = [
    'epoch', 'model_name', 'model', 'backbone', 'use_cga', 'regularizer_impl',
    'evidence_mode', 'p1_preflight_passed', 'p1a_hcval_source_audit_passed',
    'fallback_regularizer_used', 'paper_evidence_allowed', 'protocol', 'dataset',
    'seed', 'mshnet_warm_epoch', 'cga_start_epoch', 'cga_ramp_epochs'
]
for p in paths:
    print('\n##', p)
    ckpt = torch.load(p, map_location='cpu')
    print(json.dumps({k: ckpt.get(k) for k in keys if k in ckpt}, indent=2, sort_keys=True))
PY
```

---

## 7. A3 strict checkpoint-load compatibility

`test.py` currently loads checkpoints with `strict=False` in the public snapshot. This can hide missing or unexpected model weights during evaluation.

Add this audit script:

```text
tools/official/check_cga_v2_strict_checkpoint_load.py
```

Then run:

```bash
cd /home/md0/ly/CGA-main

python3 tools/official/check_cga_v2_strict_checkpoint_load.py \
  --model_name MSHNetOHEM \
  --checkpoint results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar \
  --evidence_mode paper \
  --output docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/strict_load_ohem.json

python3 tools/official/check_cga_v2_strict_checkpoint_load.py \
  --model_name MSHNetCGA \
  --checkpoint results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar \
  --evidence_mode paper \
  --output docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/strict_load_cga.json
```

Pass condition:

```text
missing_keys = []
unexpected_keys = []
```

If either checkpoint has missing or unexpected keys, the current P2 result is not reliable and should be marked as implementation-invalid.

### Recommended `test.py` patch

Patch the load section in `test.py`:

```python
state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
load_result = model.load_state_dict(state_dict, strict=False)
missing_keys = list(getattr(load_result, "missing_keys", []))
unexpected_keys = list(getattr(load_result, "unexpected_keys", []))

if args.evidence_mode == "paper" and (missing_keys or unexpected_keys):
    raise RuntimeError(
        "Paper-evidence checkpoint load must be exact. "
        f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
    )
```

And add to the evaluation summary:

```python
summary.update({
    "checkpoint_missing_keys": missing_keys,
    "checkpoint_unexpected_keys": unexpected_keys,
    "checkpoint_strict_load_pass": bool(not missing_keys and not unexpected_keys),
})
```

This is an audit-quality code change. It does not alter training or the method.

---

## 8. A4 train-log loss-scale audit

The current result pattern suggests CGA may be pushing the model toward higher sensitivity and lower precision.

Inspect `train_log.jsonl` for:

```text
base_total
total
cga_w
cga_center
cga_boundary
cga_scale
cga_peak
```

Run:

```bash
cd /home/md0/ly/CGA-main

python3 - <<'PY'
import json
from pathlib import Path
from statistics import mean

for model in ['MSHNetOHEM', 'MSHNetCGA']:
    path = Path(f'results/official_from_zero/{model}/seed42/NUDT-SIRST/train_log.jsonl')
    print('\n##', model, path)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    for r in rows[-5:]:
        keys = ['epoch', 'total', 'base_total', 'ohem', 'soft_iou', 'scale', 'cga_w', 'cga_center', 'cga_boundary', 'cga_scale', 'cga_peak']
        print({k: r.get(k) for k in keys if k in r})
    tail = rows[-20:]
    print('tail20_mean')
    for k in ['total', 'base_total', 'ohem', 'soft_iou', 'scale', 'cga_w', 'cga_center', 'cga_boundary', 'cga_scale', 'cga_peak']:
        vals = [float(r[k]) for r in tail if isinstance(r.get(k), (int, float))]
        if vals:
            print(k, mean(vals))
PY
```

Audit questions:

```text
1. Is total much larger than base_total for CGA after ramp?
2. Are any auxiliary heads numerically dominating training?
3. Does cga_w reach 1.0 as expected?
4. Are all four CGA losses present and finite?
5. Did baseline and CGA use the same OHEM ratio and MSHNet warm epoch?
```

If loss values are finite and reasonable, the negative result is more likely a real method/configuration failure than an implementation crash.

---

## 9. A5 evaluation symmetry audit

Verify for both OHEM and CGA:

```text
same dataset = NUDT-SIRST
same seed = 42
same checkpoint_epoch = 400
same threshold = 0.5
same threshold_selection = fixed_predeclared
same split names = test / hcval
same metric implementation
same prediction crop behavior
```

Run:

```bash
cd /home/md0/ly/CGA-main

python3 - <<'PY'
import json
from pathlib import Path

files = {
    'ohem_test': 'results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/test/summary_metrics.json',
    'cga_test': 'results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/test/summary_metrics.json',
    'ohem_hcval': 'results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/hcval/summary_metrics.json',
    'cga_hcval': 'results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/hcval/summary_metrics.json',
}
keys = ['model', 'dataset', 'split', 'seed', 'epoch', 'checkpoint_epoch', 'threshold', 'threshold_selection', 'checkpoint']
for name, path in files.items():
    d = json.loads(Path(path).read_text())
    print('\n##', name)
    print(json.dumps({k: d.get(k) for k in keys}, indent=2, sort_keys=True))
PY
```

---

## 10. A6 prediction-mass / false-positive audit

Because CGA has much larger FA on HC-Val, quantify whether it simply predicts more foreground mass at threshold 0.5.

Run:

```bash
cd /home/md0/ly/CGA-main

python3 - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np
from statistics import mean, median

for model in ['MSHNetOHEM', 'MSHNetCGA']:
    for split in ['test', 'hcval']:
        d = Path(f'results/official_from_zero/{model}/seed42/NUDT-SIRST/{split}/predictions')
        ratios = []
        for p in sorted(d.glob('*.png')):
            arr = np.array(Image.open(p).convert('L'))
            ratios.append(float((arr >= 128).mean()))
        print(model, split, {
            'n': len(ratios),
            'mean_positive_ratio': mean(ratios),
            'median_positive_ratio': median(ratios),
            'max_positive_ratio': max(ratios),
        })
PY
```

Interpretation:

```text
If CGA positive ratio is much higher than OHEM, the failure is over-detection.
If positive ratio is similar but FA components are higher, the failure is fragmentation / clutter-localization.
```

---

## 11. A7 adapter and feature-source audit

Current intended MSHNet CGA adapter contract:

```text
logits source: base_logits / base_logit
feature source: decoder_features.x_d0
feature stride: 1
feature channels: 16 expected by adapter constant, actual channel read from tensor
```

Audit one forward pass:

```bash
cd /home/md0/ly/CGA-main

python3 - <<'PY'
import torch
from dataset import TrainSetLoader
from net import build_model

root = '/home/md0/ly/CGA-main/datasets'
ds = TrainSetLoader(root, 'NUDT-SIRST', patch_size=256)
img, mask = ds[0]
img = img[None].float()

model = build_model(model_name='MSHNetCGA', evidence_mode='paper')
model.eval()
with torch.no_grad():
    out = model(img, mshnet_warm_flag=False)

print('keys:', sorted(out.keys()))
print('logits:', tuple(out['logits'].shape))
print('feature:', tuple(out['features'][0].shape))
print('feature_meta:', out['feature_meta'])
print('adapter_meta:', out['adapter_meta'])
print('aux:', {k: tuple(out[k].shape) for k in ['cga_center_logit', 'cga_boundary_logit', 'cga_scale_logit', 'cga_peak_logit']})
PY
```

Pass condition:

```text
logits spatial size matches final prediction size.
CGA aux logits spatial size matches selected feature size.
loss.py resizes targets to each aux head without silent fallback.
feature source is exactly decoder_features.x_d0 or an explicitly documented equivalent.
```

---

## 12. What not to do now

Do not do these after a failed seed42 gate:

```text
Do not run seed43/44.
Do not retune threshold to rescue Precision/FA.
Do not regenerate HC-Val after seeing outputs.
Do not change HC-Val membership.
Do not increase/decrease CGA weights and call it the same paper-evidence protocol.
Do not write AAAI main-table claims.
Do not add a second backbone to compensate for a failed MSHNet controlled pair.
```

---

## 13. Possible outcomes after audit

### Outcome A: implementation bug found

Examples:

```text
checkpoint key mismatch hidden by strict=False
wrong checkpoint path / stale output
OHEM and CGA trained with different protocol flags
CGA eval built a different model than CGA train
path mapping shows results from a different repo/root
missing P1/P1A metadata in checkpoint
```

Decision:

```text
Fix the bug.
Invalidate current P2 result.
Rerun seed42 paired from zero once under corrected protocol.
```

### Outcome B: no implementation bug found

Examples:

```text
strict load passes
metadata passes
path mapping is valid
losses are finite and expected
same threshold/split/checkpoint epoch
prediction mass confirms real over-detection behavior
```

Decision:

```text
Treat current CGA-v2 configuration as negative for AAAI-main rescue.
Do not run seed43/44.
Downgrade to negative analysis / workshop / method revision.
```

### Outcome C: method revision considered

If you revise the method after this result, label it as a new protocol/version.

Examples:

```text
CGA-v2.1 with delayed cga_start_epoch
CGA-v2.1 with smaller lambda_boundary/lambda_peak
CGA-v2.1 with feature-source change
CGA-v2.1 with precision-preserving auxiliary constraint
```

Rules:

```text
Do not mix CGA-v2 failed evidence with CGA-v2.1 claims.
Predeclare new protocol before rerunning.
Run OHEM and revised CGA from zero as paired evidence again.
```

---

## 14. Minimal code changes to commit now

Commit only audit tools and strict-load safeguards, not model/loss changes.

Recommended files:

```text
tools/official/audit_cga_v2_p2_failure.py
tools/official/check_cga_v2_strict_checkpoint_load.py
```

Optional small patch:

```text
test.py: raise on checkpoint missing/unexpected keys in evidence_mode=paper.
```

Do not change:

```text
model/cga_wrapper.py
model/cga_aux.py
utils/cga_targets.py
loss.py CGA weights
MSHNet adapter source
HC-Val split
```

---

## 15. Final one-line decision

```text
Seed42 failed the rescue gate. Stop multiseed and enter implementation audit. Only if the audit finds a concrete implementation bug should seed42 be rerun; otherwise, the current CGA-v2 configuration should not be pushed as an AAAI-main evidence route.
```
