# CGA-v2 P2 Seed42 Failure: Implementation Audit Plan with Audit-Only Safeguards

**Canonical root:** `/home/ly/AAAI/CGA-main`  
**Dataset root:** `/home/ly/AAAI/CGA-main/datasets`  
**Dataset:** `NUDT-SIRST`  
**Primary result root:** `/home/ly/AAAI/CGA-main/results/official_from_zero`  
**Current gate:** `Gate-CGA-v2-P2-seed42-reproduction`  
**Current decision:** `P2_FAIL_IMPL_AUDIT_ALLOWED`

This document supersedes any P2 audit plan that uses:

```text
/home/md0/ly/CGA-main
/home/AAAI/CGA-main
/home/ly/AAAI/OHCM-MSHNet-main/datasets
```

Those paths are **non-canonical** for the current audit. If they appear in logs, summaries, checkpoints, or prediction paths, they must be treated as possible path contamination until a `realpath` equivalence note proves that they resolve to the same physical repository or mount as `/home/ly/AAAI/CGA-main`.

---

## 0. Verdict

The seed42 from-zero paired result failed the predeclared P2 gate.

```text
Full test:
  OHEM: mIoU=0.916185, Precision=0.955114, Pd=0.984127, FA_ppm=12.317
  CGA : mIoU=0.913089, Precision=0.952974, Pd=0.989418, FA_ppm=15.236
  Δ   : mIoU=-0.003096, Precision=-0.002140, Pd=+0.005291, FA_ppm=+2.918

HC-Val:
  OHEM: mIoU=0.781553, Precision=0.836364, Pd=0.833333, FA_ppm=111.898
  CGA : mIoU=0.509346, Precision=0.527419, Pd=1.000000, FA_ppm=706.991
  Δ   : mIoU=-0.272208, Precision=-0.308944, Pd=+0.166667, FA_ppm=+595.093
```

Immediate decision:

```text
Do not run seed43/44.
Do not run ablation.
Do not create positive paper narrative.
Do not change CGA method parameters as part of this audit.
Enter implementation audit only.
```

Current safe interpretation:

```text
CGA increases Pd but worsens mIoU, Precision, and FA.
On HC-Val, the current implementation shows severe false-alarm collapse.
```

---

## 1. Scope

This is an **implementation audit plan**, not a method rescue plan.

Allowed now:

```text
1. Freeze current P2 artifacts.
2. Verify path consistency.
3. Verify paired-run symmetry.
4. Verify checkpoint identity and strict load.
5. Verify evaluation uses only final logits.
6. Verify train/eval adapter contract.
7. Verify loss-scale behavior.
8. Verify CGA target generation.
9. Verify prediction-level false-alarm morphology.
10. Produce a machine-readable final audit decision.
```

Not allowed now:

```text
1. Changing CGA loss weights.
2. Changing cga_start_epoch or cga_ramp_epochs.
3. Changing center/boundary/scale/peak target generation.
4. Changing adapter feature source.
5. Changing threshold.
6. Changing HC-Val split.
7. Running seed43/44.
8. Writing positive AAAI claims.
```

If any forbidden change is made, the result becomes a new protocol/method version, e.g. `CGA-v2.1`, and must be predeclared before rerunning from zero.

---

## 2. Canonical path contract

All commands must run from:

```bash
cd /home/ly/AAAI/CGA-main
```

Use:

```bash
ROOT=/home/ly/AAAI/CGA-main
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets
DATASET_NAME=NUDT-SIRST
P2_DIR=/home/ly/AAAI/CGA-main/docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST
RESULT_DIR=/home/ly/AAAI/CGA-main/results/official_from_zero
```

### 2.1 Realpath equivalence check

If old outputs contain `/home/AAAI/CGA-main` or `/home/md0/ly/CGA-main`, create:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/path_equivalence.json
```

Command:

```bash
cd /home/ly/AAAI/CGA-main
mkdir -p docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST

python3 - <<'PY'
import json
import os
from pathlib import Path

paths = [
    "/home/ly/AAAI/CGA-main",
    "/home/AAAI/CGA-main",
    "/home/md0/ly/CGA-main",
]

out = {}
for p in paths:
    pp = Path(p)
    out[p] = {
        "exists": pp.exists(),
        "is_dir": pp.is_dir(),
        "realpath": os.path.realpath(p) if pp.exists() else None,
    }

canonical = "/home/ly/AAAI/CGA-main"
canonical_real = out[canonical]["realpath"]
for p, item in out.items():
    item["equivalent_to_canonical"] = bool(
        item["exists"] and canonical_real and item["realpath"] == canonical_real
    )

Path("docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/path_equivalence.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(out, indent=2, sort_keys=True))
PY
```

Pass condition:

```json
{
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false
}
```

Fail condition:

```text
Any checkpoint, prediction, dataset, or summary path points outside canonical root
and is not realpath-equivalent to /home/ly/AAAI/CGA-main.
```

Failure decision:

```text
P2_INVALID_PATH_CONTAMINATION
```

---

## 3. Audit output contract

Every audit step must write one JSON file with this top-level schema:

```json
{
  "audit_step": "A1_artifact_freeze",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "notes": [],
  "artifacts": {}
}
```

Required meaning:

```text
pass:
  Whether this audit step passed.

invalidates_p2:
  Whether the current seed42 result is invalid as implementation evidence.

requires_rerun:
  Whether seed42 must be rerun after fixing a bug or contamination.

notes:
  Human-readable notes. Notes do not replace pass/fail fields.

artifacts:
  Step-specific hashes, counts, metrics, paths, and summary statistics.
```

Final aggregator output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/final_impl_audit_summary.json
```

---

## 4. A1 — Artifact freeze and run identity

Purpose: prevent ambiguity that checkpoint, summary, prediction, and train log may come from different runs.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A1_artifact_freeze.json
```

Must record:

```text
git commit
git branch
git dirty status
git status --short
checkpoint sha256
checkpoint mtime
checkpoint file size
summary_metrics.json sha256
prediction png count
train_log line count
train_log last epoch
train_log sha256
all artifact realpaths
inside_canonical_root=true/false
```

Minimal audit-only tool:

```text
tools/official/audit_cga_v2_artifact_freeze.py
```

Expected command:

```bash
cd /home/ly/AAAI/CGA-main

python3 tools/official/audit_cga_v2_artifact_freeze.py \
  --root /home/ly/AAAI/CGA-main \
  --output docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A1_artifact_freeze.json \
  --paths \
    results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar \
    results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar \
    results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/test/summary_metrics.json \
    results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/test/summary_metrics.json \
    results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/hcval/summary_metrics.json \
    results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/hcval/summary_metrics.json \
  --prediction_dirs \
    results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/test/predictions \
    results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/test/predictions \
    results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/hcval/predictions \
    results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/hcval/predictions \
  --train_logs \
    results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/train_log.jsonl \
    results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/train_log.jsonl
```

Pass condition:

```text
pass=true
invalidates_p2=false
requires_rerun=false
all required checkpoints exist
all required summaries exist
prediction png count > 0
git commit recorded
all required artifacts are inside canonical root or realpath-equivalent
```

Failure decision:

```text
P2_INVALID_ARTIFACT_MISMATCH
```

---

## 5. A2 — Paired protocol symmetry

Purpose: verify OHEM and CGA are a valid pair.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A2_protocol_symmetry.json
```

Required checks:

```text
1. Same dataset_name: NUDT-SIRST.
2. Same train_list_sha256.
3. Same test_list_sha256.
4. Same hcval_list_sha256.
5. Same dataset_registry_sha256.
6. Same seed: 42.
7. Same checkpoint_epoch: 400.
8. Same threshold: 0.5.
9. Same threshold_selection: fixed_predeclared.
10. Same metric implementation version, if logged.
11. Same image/mask root or realpath-equivalent root.
```

Pass JSON:

```json
{
  "audit_step": "A2_protocol_symmetry",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "notes": [],
  "artifacts": {
    "train_list_sha256_equal": true,
    "test_list_sha256_equal": true,
    "hcval_list_sha256_equal": true,
    "seed_equal": true,
    "epoch_equal": true,
    "threshold_equal": true
  }
}
```

Failure decision:

```text
P2_INVALID_PROTOCOL_ASYMMETRY
```

---

## 6. A3 — Strict checkpoint load with whitelist normalization

Purpose: verify evaluated checkpoints exactly match intended models.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A3_strict_load_ohem.json
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A3_strict_load_cga.json
```

### 6.1 Whitelisted normalization

Allowed transformations:

```text
1. unwrap checkpoint["state_dict"] if checkpoint is a dict and has state_dict.
2. unwrap checkpoint["model"] only if state_dict is absent and model is state_dict-like.
3. strip one leading "module." prefix from all keys if the checkpoint is DataParallel style.
4. strip one explicitly named wrapper prefix only if all unexpected keys share that prefix and the script records it.
```

Not allowed:

```text
1. strict=False silent pass.
2. dropping missing keys.
3. dropping unexpected keys.
4. dropping CGA auxiliary head keys.
5. dropping backbone keys.
6. partial loading.
```

Pass condition:

```json
{
  "audit_step": "A3_strict_load_cga",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "notes": [],
  "artifacts": {
    "missing_keys": [],
    "unexpected_keys": [],
    "normalization_applied": "unwrap_state_dict|strip_module|none",
    "checkpoint_sha256": "..."
  }
}
```

Failure decision:

```text
P2_INVALID_CHECKPOINT_LOAD
```

Important: a patch to make `test.py` use strict loading may be useful later, but it must be recorded as a **post-P2 audit-only safeguard**, not as part of the failed P2 result. For current P2, use an external strict-load checker so that the audit does not rewrite history.

---

## 7. A4 — Loss-scale audit

Purpose: determine whether CGA auxiliary terms overpower the main segmentation/OHEM losses and push the model toward recall-biased over-detection.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A4_loss_scale_audit.json
```

Compute over the last 20 epoch-level log rows, or the last 20 logged training summaries.

Required metrics:

```text
tail20_mean_base_total_ohem
tail20_mean_base_total_cga
tail20_mean_main_loss_ohem
tail20_mean_main_loss_cga
tail20_mean_ohem_loss_ohem
tail20_mean_ohem_loss_cga
tail20_mean_soft_iou_ohem
tail20_mean_soft_iou_cga

tail20_mean_cga_aux_total
tail20_mean_loss_center
tail20_mean_loss_boundary
tail20_mean_loss_scale
tail20_mean_loss_peak

tail20_mean_aux_total_over_total
tail20_mean_center_over_total
tail20_mean_boundary_over_total
tail20_mean_scale_over_total
tail20_mean_peak_over_total

tail20_mean_aux_total_over_base_total
tail20_mean_cga_w
tail20_last_cga_w
nan_or_inf_count
```

Suggested warnings:

```text
aux_total_over_total > 0.30:
  warn: CGA auxiliary supervision may dominate optimization.

aux_total_over_total > 0.50:
  strong_warn: CGA auxiliary supervision likely dominates optimization.

boundary_over_total > 0.25 or peak_over_total > 0.25:
  warn: boundary/peak may drive over-activation.

tail20_last_cga_w > 1.0:
  invalidates_p2=true unless explicitly predeclared.

nan_or_inf_count > 0:
  invalidates_p2=true.
```

Interpretation:

```text
If auxiliary loss is too strong but all implementation checks pass,
this is not a bug fix inside CGA-v2.
It means CGA-v2 is a valid negative or a design weakness.
Any reweighting is CGA-v2.1 and needs a new predeclared protocol.
```

---

## 8. A5 — Eval final-logit-only audit

Purpose: verify evaluation does not leak CGA auxiliary heads or intermediate features into prediction.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A5_eval_final_logit_only.json
```

Required checks:

```text
1. Prediction is generated from output["logits"] or normalized contract logits only.
2. cga_center_logit is never thresholded as final prediction.
3. cga_boundary_logit is never thresholded as final prediction.
4. cga_scale_logit is never thresholded as final prediction.
5. cga_peak_logit is never thresholded as final prediction.
6. model.eval() is used before inference.
7. torch.no_grad() or torch.inference_mode() is used.
8. threshold=0.5 is applied to final sigmoid/probability map.
```

Suggested grep:

```bash
cd /home/ly/AAAI/CGA-main

grep -R "cga_center_logit\|cga_boundary_logit\|cga_scale_logit\|cga_peak_logit" -n \
  test.py evaluate.py tools scripts utils 2>/dev/null || true
```

Failure decision:

```text
P2_INVALID_EVAL_AUX_LEAKAGE
```

---

## 9. A6 — Prediction morphology and component-level FA audit

Purpose: explain why HC-Val FA_ppm jumps from 111.898 to 706.991.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A6_prediction_morphology_audit.json
```

Required per-image metrics for OHEM and CGA on both Full and HC-Val:

```text
foreground_ratio
foreground_area
connected_component_count
fp_component_count
fp_area
max_fp_component_area
mean_fp_component_area
p95_fp_component_area
```

Required paired deltas:

```text
CGA - OHEM foreground_ratio
CGA - OHEM foreground_area
CGA - OHEM fp_component_count
CGA - OHEM fp_area
CGA - OHEM max_fp_component_area
```

Required outputs:

```text
top5_hcval_false_alarm_images
top5_full_false_alarm_images
per_image_fp_components.csv
per_image_foreground_ratio.csv
```

Required failure visualization directory:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/failure_top5_hcval/
  <id>_image.png
  <id>_mask.png
  <id>_ohem_pred.png
  <id>_cga_pred.png
  <id>_diff.png
```

Classify FA mode:

```text
area_overactivation:
  foreground ratio and FP area increase sharply.

fragmented_noise_components:
  FP component count increases sharply but FP area remains moderate.

edge_or_hotspot_confusion:
  false alarms align with boundaries, hot pixels, or bright background clutter.

mixed_overactivation_and_fragmentation:
  both FP area and FP component count increase.
```

A6 does not by itself invalidate P2 unless predictions are missing/unreadable. Its role is diagnosis.

Failure decision if artifacts are missing:

```text
P2_INVALID_MISSING_PREDICTIONS
```

---

## 10. A7 — CGA target generation audit

Purpose: verify deterministic center/boundary/scale/peak targets are correct and not overly expansive for tiny targets.

Output:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/A7_cga_target_audit.json
```

Required metrics over a fixed sample set:

```text
mask_fg_ratio
center_fg_ratio
boundary_fg_ratio
peak_fg_ratio
scale_finite_ratio
boundary_fg_ratio_over_mask_fg_ratio
peak_fg_ratio_over_mask_fg_ratio
center_inside_gt_rate
nan_or_inf_count
```

Warning thresholds:

```text
boundary_fg_ratio_over_mask_fg_ratio > 5:
  warn: boundary target may be too wide for tiny targets.

peak_fg_ratio_over_mask_fg_ratio > 2:
  warn: peak target may be insufficiently sparse.
```

Invalidating conditions:

```text
center_inside_gt_rate < 1.0 for non-empty single-component targets
nan_or_inf_count > 0
scale target contains invalid values
```

Failure decision:

```text
P2_INVALID_CGA_TARGET_GENERATION
```

---

## 11. Final decision truth table

After A1-A7, generate:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/final_impl_audit_summary.json
```

Truth table:

| Condition | Final decision | Next action |
|---|---|---|
| Path contamination | `P2_INVALID_PATH_CONTAMINATION` | Fix path contract and rerun seed42 from zero. |
| Artifact mismatch or missing files | `P2_INVALID_ARTIFACT_MISMATCH` | Freeze correct artifacts or rerun seed42. |
| OHEM/CGA split/seed/threshold/epoch mismatch | `P2_INVALID_PROTOCOL_ASYMMETRY` | Fix runner and rerun seed42. |
| Checkpoint strict load fails | `P2_INVALID_CHECKPOINT_LOAD` | Fix model/checkpoint mismatch and rerun seed42. |
| Eval uses CGA aux head or wrong tensor | `P2_INVALID_EVAL_AUX_LEAKAGE` | Fix evaluator and rerun seed42. |
| Adapter contract uses silent fallback or wrong source | `P2_INVALID_ADAPTER_CONTRACT` | Fix adapter and rerun seed42. |
| CGA target generation invalid | `P2_INVALID_CGA_TARGET_GENERATION` | Fix target generation and rerun seed42. |
| Prediction files missing/unreadable | `P2_INVALID_MISSING_PREDICTIONS` | Fix eval artifact generation and rerun/evaluate. |
| Loss aux is too strong, but not a bug | `P2_VALID_NEGATIVE_DESIGN_WEAKNESS` | Stop AAAI-main CGA-v2; any reweighting is CGA-v2.1. |
| All audits pass and result remains negative | `P2_VALID_NEGATIVE` | Stop AAAI-main CGA-v2; no seed43/44. |
| Clear bug found and fixed | `P2_INVALIDATED_BY_IMPLEMENTATION_BUG` | Predeclare fix and rerun seed42 from zero. |

---

## 12. Audit scripts only

This section replaces any old “Minimal Code Changes” wording.

Allowed audit-only additions:

```text
tools/official/audit_cga_v2_artifact_freeze.py
tools/official/audit_cga_v2_protocol_symmetry.py
tools/official/check_cga_v2_strict_checkpoint_load.py
tools/official/audit_cga_v2_loss_scale.py
tools/official/audit_cga_v2_eval_final_logit_only.py
tools/official/audit_cga_v2_prediction_morphology.py
tools/official/audit_cga_v2_targets.py
tools/official/summarize_cga_v2_p2_impl_audit.py
```

Do not patch `test.py` strict loading before labeling the current P2 result unless it is explicitly recorded as:

```text
post-P2 audit-only patch, not used to generate the failed seed42 result.
```

Reason:

```text
Otherwise reviewers or future readers cannot tell whether the failed P2 result was produced before or after the eval-code change.
```

External strict-load checking is allowed and recommended because it audits the existing checkpoint without changing the historical evaluator.

---

## 13. Paper claim downgrade

Delete or rewrite:

```text
CGA improves IRSTD.
CGA reduces false alarms.
CGA is robust on hard cases.
CGA improves hard-clutter behavior.
CGA is target-preserving and works under the current protocol.
CGA is AAAI-ready.
```

Allowed internal statement:

```text
Under the current CGA-main seed42 from-zero paired protocol, CGA increases Pd but worsens mIoU, Precision, and FA. On HC-Val, it causes severe false-alarm collapse. The result failed the predeclared P2 gate and triggers implementation audit rather than multiseed continuation.
```

If the final audit decision is `P2_VALID_NEGATIVE`, write:

```text
CGA-v2 as currently implemented behaves more like a recall booster than a false-alarm suppressor. The AAAI-main positive method route is stopped.
```

---

## 14. Recommended execution order

```text
1. Freeze artifacts with A1.
2. Resolve path equivalence if non-canonical paths appear.
3. Run A2 protocol symmetry.
4. Run A3 strict checkpoint load.
5. Run A4 loss-scale audit.
6. Run A5 eval final-logit-only audit.
7. Run A6 prediction morphology audit.
8. Run A7 target generation audit.
9. Generate final_impl_audit_summary.json.
10. Decide: invalid implementation vs valid negative.
```

Do not run seed43/44 unless:

```text
1. P2 is invalidated by a concrete implementation bug,
2. the bug is fixed,
3. the fix is predeclared,
4. seed42 is rerun from zero,
5. the rerun passes the original predeclared gate.
```

---

## 15. One-line conclusion

Use `/home/ly/AAAI/CGA-main` as the only canonical root. The current task is not method repair; it is machine-readable implementation audit of the failed seed42 P2 result.
