# CGA-v2 P2 Seed42 Failure: Implementation Audit Plan with Audit-Only Safeguards, v3

## 0. Verdict

The five suggested changes are correct and should be incorporated before running the implementation audit.

Current goal:

```text
Audit the failed P2 seed42 from-zero paired result.
Do not repair the method yet.
Do not run seed43/44.
Do not run ablations.
Do not write positive CGA paper claims.
```

The audit must decide between:

```text
P2_INVALID_IMPLEMENTATION / P2_INVALID_PROTOCOL / P2_INVALID_PATH_CONTAMINATION
```

and:

```text
P2_VALID_NEGATIVE
```

The invalidation decision has priority. If any hard invalidating audit condition is found, the current failed result cannot be interpreted as a valid negative method result.

---

## 1. Canonical path contract

The only canonical root for this audit is:

```text
/home/ly/AAAI/CGA-main
```

Use only:

```bash
ROOT=/home/ly/AAAI/CGA-main
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets
DATASET_NAME=NUDT-SIRST
RESULT_DIR=/home/ly/AAAI/CGA-main/results/official_from_zero
```

Do not use these as canonical roots:

```text
/home/md0/ly/CGA-main
/home/AAAI/CGA-main
/home/ly/AAAI/OHCM-MSHNet-main/datasets
```

If older summaries contain non-canonical paths, do not edit them manually. Treat them as audit evidence.

---

## 2. Correct P2 directory contract

The existing P2 summary path is:

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json
```

Therefore, the audit output should live under the same P2 gate directory:

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v3/
```

Do **not** silently switch to:

```text
docs/internal/cga_v2/gate_p2_seed42_NUDT-SIRST/
```

unless that directory is only a symlink or intentionally documented alias.

Recommended variables:

```bash
ROOT=/home/ly/AAAI/CGA-main
P2_DIR=${ROOT}/docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST
P2_SUMMARY=${P2_DIR}/summary.json
AUDIT_DIR=${P2_DIR}/audit_v3
```

If a new audit output directory is created outside `P2_DIR`, it must explicitly reference the original P2 summary:

```json
{
  "source_p2_summary": "docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json",
  "source_p2_summary_sha256": "..."
}
```

Otherwise the audit result and the original P2 gate become separated, which is not acceptable for evidence tracking.

---

## 3. Current failed P2 facts to freeze

The current P2 summary states:

```text
decision = P2_FAIL_IMPL_AUDIT_ALLOWED
gate_pass = false
epoch = 400
threshold = 0.5
```

Full split:

| Metric | OHEM | CGA | Delta |
|---|---:|---:|---:|
| mIoU | 0.916185 | 0.913089 | -0.003096 |
| Precision | 0.955114 | 0.952974 | -0.002140 |
| Pd | 0.984127 | 0.989418 | +0.005291 |
| FA_ppm | 12.317 | 15.236 | +2.918 |

HC-Val:

| Metric | OHEM | CGA | Delta |
|---|---:|---:|---:|
| mIoU | 0.781553 | 0.509346 | -0.272208 |
| Precision | 0.836364 | 0.527419 | -0.308944 |
| Pd | 0.833333 | 1.000000 | +0.166667 |
| FA_ppm | 111.898 | 706.991 | +595.093 |

Interpretation before audit:

```text
The run failed the predeclared P2 gate.
The result is not yet a valid negative method result until path, checkpoint, protocol, eval-output, and target-generation audits pass.
```

---

## 4. A0 — Pre-audit path-contamination hard stop

This must run before A1/A2.

The current P2 summary contains `/home/AAAI/CGA-main/...` paths. Since the canonical host path is `/home/ly/AAAI/CGA-main`, this is no longer a hypothetical risk.

Run:

```bash
cd /home/ly/AAAI/CGA-main

readlink -f /home/ly/AAAI/CGA-main
readlink -f /home/AAAI/CGA-main || true
readlink -f /home/md0/ly/CGA-main || true

python3 - <<'PY'
from pathlib import Path
for p in ["/home/ly/AAAI/CGA-main", "/home/AAAI/CGA-main", "/home/md0/ly/CGA-main"]:
    q = Path(p)
    print(p, "exists=", q.exists(), "realpath=", q.resolve() if q.exists() else None)
PY
```

Machine-readable output:

```text
${AUDIT_DIR}/A0_path_contamination_precheck.json
```

Required JSON schema:

```json
{
  "audit_step": "A0_path_contamination_precheck",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "canonical_root": "/home/ly/AAAI/CGA-main",
  "noncanonical_paths_seen_in_p2": ["/home/AAAI/CGA-main"],
  "mount_or_realpath_equivalence_proven": true,
  "notes": []
}
```

Hard rule:

```text
If /home/AAAI/CGA-main appears in the P2 summary but does not exist in the current environment,
and no original-run Docker mount / realpath equivalence proof is provided,
then mark:

  pass=false
  invalidates_p2=true
  requires_rerun=true
  decision=P2_INVALID_PATH_CONTAMINATION
```

This condition is higher priority than all metric interpretation.

Allowed proof examples:

```text
1. docker inspect record showing /home/AAAI/CGA-main was bind-mounted to /home/ly/AAAI/CGA-main.
2. saved original-run `readlink -f` output proving equivalence.
3. immutable run manifest from the original container containing both host and container root mappings.
```

Not sufficient:

```text
1. Manually editing old JSON paths.
2. Assuming the paths are equivalent because the file names look right.
3. Reconstructing paths after the fact without original-run evidence.
```

---

## 5. Audit-only policy boundary

The section title should not say `Minimal Code Changes`. The correct title is:

```text
Implementation Audit Plan with Audit-Only Safeguards
```

Allowed additions:

```text
tools/official/audit_cga_v2_p2_impl_v3.py
tools/official/check_cga_v2_strict_checkpoint_load_v3.py
tools/official/trace_cga_v2_eval_output_source.py
tools/official/audit_cga_v2_target_geometry_components.py
```

These are audit-only scripts. They must not modify:

```text
model/
loss.py
utils/cga_targets.py
train.py training behavior
test.py prediction behavior
threshold
HC-Val split
checkpoints
predictions
```

`test.py` strict-load hardening can be listed only as a future-run safeguard:

```text
Current P2 audit:
  external audit scripts only.

Future runs after P2 decision:
  optional test.py strict-load patch in a separate commit.
```

Do not mix a `test.py` patch into the current P2 interpretation. Otherwise the audit cannot answer whether the failed result was produced before or after the evaluation-code change.

---

## 6. Required JSON schema for every audit step

Every step A0-A7 must write one JSON file:

```json
{
  "audit_step": "A1_artifact_freeze",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "notes": [],
  "warnings": [],
  "artifacts": {}
}
```

Final summary:

```text
${AUDIT_DIR}/FINAL_P2_IMPL_AUDIT_SUMMARY.json
```

Minimum final schema:

```json
{
  "source_p2_summary": "docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json",
  "canonical_root": "/home/ly/AAAI/CGA-main",
  "audit_steps": {
    "A0_path_contamination_precheck": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A1_artifact_freeze": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A2_protocol_symmetry": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A3_strict_checkpoint_load": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A4_loss_scale": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A5_eval_output_source_trace": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A6_prediction_component_analysis": {"pass": true, "invalidates_p2": false, "requires_rerun": false},
    "A7_target_geometry_components": {"pass": true, "invalidates_p2": false, "requires_rerun": false}
  },
  "invalidation_priority_applied": true,
  "final_decision": "P2_VALID_NEGATIVE",
  "next_action": "stop_AAAI_main_CGA_v2"
}
```

---

## 7. A1 — Artifact freeze

A1 must record enough information to prove that checkpoint, summaries, predictions, and logs belong to the same run.

Required records:

```text
git commit
git branch
git dirty status
P2 summary sha256
checkpoint sha256 for OHEM and CGA
checkpoint mtime for OHEM and CGA
checkpoint size for OHEM and CGA
summary_metrics sha256 if separate files exist
prediction png count for Full and HC-Val, OHEM and CGA
prediction_dir realpath for Full and HC-Val, OHEM and CGA
train_log path
train_log sha256
train_log line count
train_log last epoch
train/test/hcval list sha256
checkpoint_epoch from summary
threshold and threshold_selection from summary
```

Output:

```text
${AUDIT_DIR}/A1_artifact_freeze.json
```

Invalidates P2 if:

```text
checkpoint missing
prediction_dir missing
prediction png count inconsistent with split list count
train_log last epoch incompatible with checkpoint_epoch
checkpoint mtime inconsistent with training log final timestamp if available
summary_metrics file belongs to a different run directory
required artifact sha256 cannot be computed
```

---

## 8. A2 — Protocol symmetry audit

A2 must prove OHEM and CGA are paired.

Required same fields:

```text
dataset_name
train_dataset
seed
epoch
checkpoint_epoch
split
threshold
threshold_selection
backbone
protocol
evidence_mode
train_list_sha256
test_list_sha256
hcval_list_sha256
```

Allowed differences:

```text
model: MSHNetOHEM vs MSHNetCGA
use_cga: false vs true
checkpoint path
prediction_dir
regularizer_impl
fallback_regularizer_used
```

Output:

```text
${AUDIT_DIR}/A2_protocol_symmetry.json
```

Invalidates P2 if seed, epoch, split, threshold, checkpoint epoch, or split hashes differ.

---

## 9. A3 — Strict checkpoint load with whitelist normalization

A3 must check that evaluated checkpoints strictly load into the intended model graphs.

Allowed checkpoint unwrapping:

```text
checkpoint["state_dict"]
checkpoint["model_state_dict"]
checkpoint["model"]
checkpoint["net"]
raw checkpoint dict as state_dict
```

Allowed prefix normalization, whitelist only:

```text
strip exactly one leading "module."
strip exactly one leading "model."
strip exactly one leading "net."
strip exactly one leading "network."
strip exactly one explicitly declared wrapper prefix if passed as --allowed_wrapper_prefix
```

Not allowed:

```text
strict=False
partial load
dropping unmatched keys
renaming arbitrary prefixes
ignoring CGA wrapper keys
ignoring prediction head keys
```

Commands:

```bash
cd /home/ly/AAAI/CGA-main

python3 tools/official/check_cga_v2_strict_checkpoint_load_v3.py \
  --root /home/ly/AAAI/CGA-main \
  --model_name MSHNetOHEM \
  --backbone_name mshnet \
  --checkpoint results/official_from_zero/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar \
  --evidence_mode paper \
  --output ${AUDIT_DIR}/A3_strict_load_ohem.json

python3 tools/official/check_cga_v2_strict_checkpoint_load_v3.py \
  --root /home/ly/AAAI/CGA-main \
  --model_name MSHNetCGA \
  --backbone_name mshnet \
  --use_cga \
  --checkpoint results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar \
  --evidence_mode paper \
  --output ${AUDIT_DIR}/A3_strict_load_cga.json
```

Pass condition:

```json
{
  "missing_keys": [],
  "unexpected_keys": [],
  "selected_transform": "raw|strip_module|strip_model|strip_net|strip_network|strip_allowed_wrapper_prefix",
  "selected_transform_is_whitelisted": true
}
```

Invalidates P2 if strict loading fails for either OHEM or CGA.

---

## 10. A4 — Quantitative loss-scale audit

A4 must not use subjective language such as `loss seems reasonable`.

Use tail20 epochs or tail20 logged iterations, depending on log granularity.

Required metrics:

```text
CGA aux_total / base_total tail20 mean
CGA aux_total / total_loss tail20 mean
CGA loss_center / total_loss tail20 mean
CGA loss_boundary / total_loss tail20 mean
CGA loss_scale / total_loss tail20 mean
CGA loss_peak / total_loss tail20 mean
CGA cga_w tail20 mean
CGA cga_w final value
CGA NaN/Inf count
OHEM NaN/Inf count
CGA base_total tail20 mean
OHEM base_total tail20 mean
CGA ohem/soft_iou tail20 mean if logged
OHEM ohem/soft_iou tail20 mean if logged
```

Output:

```text
${AUDIT_DIR}/A4_loss_scale.json
```

Invalidates P2 if:

```text
NaN/Inf appears in required losses
loss logging shows corrupted or impossible values
cga_w is not following the predeclared ramp due to implementation error
```

Does not automatically invalidate P2:

```text
aux_total/base_total is high but numerically valid.
```

That indicates configuration/design weakness, not necessarily an implementation bug. If this is the only finding, the correct outcome is `P2_VALID_NEGATIVE_DESIGN_WEAKNESS` or a new predeclared `CGA-v2.1` protocol.

---

## 11. A5 — Dynamic eval output-source trace

A5 cannot rely only on static grep. Static grep may miss helper-level key changes or wrapper-level selection changes.

Required approach:

```text
Run a trace-mode audit on the actual model/eval output selection path.
Record which tensor is selected for prediction.
Do not rewrite predictions.
Do not change threshold.
Do not patch the current P2 result.
```

Suggested audit script:

```text
tools/official/trace_cga_v2_eval_output_source.py
```

It should load one OHEM checkpoint and one CGA checkpoint, run a small deterministic batch from `test` and `hcval`, and write:

```text
${AUDIT_DIR}/A5_eval_output_source_trace.json
```

Required JSON fields:

```json
{
  "audit_step": "A5_eval_output_source_trace",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "traces": [
    {
      "model": "MSHNetCGA",
      "split": "hcval",
      "image_id": "...",
      "checkpoint_epoch": 400,
      "model_eval_called": true,
      "torch_no_grad_used": true,
      "selected_prediction_key": "logits",
      "selected_prediction_source": "final_logits",
      "selected_tensor_shape": [1, 1, 256, 256],
      "selected_tensor_dtype": "torch.float32",
      "logit_min": -0.0,
      "logit_max": 0.0,
      "logit_mean": 0.0,
      "prob_min": 0.0,
      "prob_max": 1.0,
      "prob_mean": 0.0,
      "threshold": 0.5,
      "positive_pixels_after_threshold": 0,
      "aux_keys_present": ["cga_center_logit", "cga_boundary_logit", "cga_scale_logit", "cga_peak_logit"],
      "aux_keys_used_for_prediction": []
    }
  ],
  "notes": []
}
```

Invalidates P2 if:

```text
prediction uses cga_center_logit, cga_boundary_logit, cga_scale_logit, or cga_peak_logit
prediction uses a decoder feature instead of final logits
model.eval() is not used
torch.no_grad() is not used
threshold is not 0.5
selected_prediction_key differs between OHEM and CGA without predeclared reason
```

Static grep can remain as a supporting check, but it is not sufficient by itself.

---

## 12. A6 — Prediction component-level false-alarm analysis

A6 must explain the HC-Val FA collapse at component level.

Required metrics for Full and HC-Val:

```text
per-image FP components
per-image FP pixels
per-image foreground ratio
CGA/OHEM foreground ratio delta
CGA/OHEM FP component delta
CGA/OHEM FP pixel delta
top-5 HC-Val false-alarm images by FP components
top-5 HC-Val false-alarm images by FP pixels
classification: large_area_overactivation / fragmented_noise_components / edge_clutter_activation / mixed / unknown
```

Output:

```text
${AUDIT_DIR}/A6_prediction_component_analysis.json
```

Required summary fields:

```json
{
  "hcval_failure_mode": "fragmented_noise_components|large_area_overactivation|edge_clutter_activation|mixed|unknown",
  "top5_hcval_fp_component_images": [],
  "top5_hcval_fp_pixel_images": [],
  "cga_mean_foreground_ratio_minus_ohem": 0.0,
  "cga_mean_fp_components_minus_ohem": 0.0,
  "cga_mean_fp_pixels_minus_ohem": 0.0
}
```

A6 usually does not invalidate P2. It explains whether the failure is:

```text
1. large-area overactivation,
2. fragmented noise components,
3. edge/highlight clutter activation,
4. mixed.
```

---

## 13. A7 — Per-component target geometry audit

A7 must check per-component behavior, not only single-component center placement.

Suggested audit script:

```text
tools/official/audit_cga_v2_target_geometry_components.py
```

Output:

```text
${AUDIT_DIR}/A7_target_geometry_components.json
```

For each audited image, record:

```text
image_id
GT component count
GT component area list
center target component count
peak target component count
boundary target area
scale target min/max
for each GT component:
  component_bbox
  component_area
  has_center_point_inside_gt_component
  has_peak_point_inside_gt_component
  center_points_inside_count
  peak_points_inside_count
  nearest_center_distance_to_component_centroid
  nearest_peak_distance_to_component_centroid
```

Required aggregate metrics:

```text
center_inside_each_gt_component_rate
peak_inside_each_gt_component_rate
component_has_center_or_peak_rate
boundary_area_over_gt_area_mean
boundary_area_over_gt_area_p95
peak_area_over_gt_area_mean
scale_value_range
multi_component_image_count
multi_component_component_count
```

Required JSON fragment:

```json
{
  "audit_step": "A7_target_geometry_components",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "target_generation_policy": "per_component|single_mask_centroid|unknown",
  "aggregate": {
    "center_inside_each_gt_component_rate": 1.0,
    "peak_inside_each_gt_component_rate": 1.0,
    "component_has_center_or_peak_rate": 1.0,
    "multi_component_image_count": 0
  },
  "examples": []
}
```

Invalidates P2 if:

```text
targets are shifted relative to masks
augmentation is applied to image/mask but not to CGA targets
center or peak points systematically fall outside GT components
boundary/peak targets are generated at the wrong resolution
multi-component GT masks are mishandled under the declared target policy
mask coordinate convention is wrong
```

If targets are aligned but too broad or too aggressive, that is a design weakness, not an implementation invalidation. It should lead to `CGA-v2.1`, not silent modification of P2.

---

## 14. Final decision truth table with invalidation priority

Priority rule:

```text
Invalidating conditions override negative diagnosis.
```

Therefore:

```text
If any of A0/A1/A2/A3/A5/A7 invalidates P2,
do not write P2_VALID_NEGATIVE.
```

Decision table:

| Condition | Decision | Next action |
|---|---|---|
| A0 path contamination cannot be resolved | `P2_INVALID_PATH_CONTAMINATION` | Obtain original mount proof or rerun seed42 from zero under canonical root. |
| A1 artifact freeze fails | `P2_INVALID_ARTIFACTS` | Fix artifact/run tracking, rerun seed42 from zero. |
| A2 protocol symmetry fails | `P2_INVALID_PROTOCOL` | Fix runner/protocol, rerun seed42 from zero. |
| A3 strict load fails | `P2_INVALID_CHECKPOINT_LOAD` | Fix checkpoint/model loading, rerun seed42 from zero. |
| A4 NaN/Inf or corrupted loss numerics | `P2_INVALID_TRAINING_NUMERICS` | Fix numerics, rerun seed42 from zero. |
| A5 eval uses aux/intermediate output | `P2_INVALID_EVAL_OUTPUT_SOURCE` | Fix eval path, rerun evaluation or seed42 as required. |
| A7 target geometry is misaligned | `P2_INVALID_TARGET_GENERATION` | Fix target generation, rerun seed42 from zero. |
| A0/A1/A2/A3/A5/A7 pass, A4 has no numerics bug, A6 explains FA collapse | `P2_VALID_NEGATIVE` | Stop AAAI-main CGA-v2 route. Do not run seed43/44. |
| Only issue is aux loss/config too aggressive, no bug | `P2_VALID_NEGATIVE_DESIGN_WEAKNESS` | Start `CGA-v2.1` only after predeclaring a new protocol. |

Final summary must compute this programmatically:

```python
hard_invalidating_steps = ["A0", "A1", "A2", "A3", "A5", "A7"]
if any(step.invalidates_p2 for step in hard_invalidating_steps):
    final_decision = first_invalidating_decision_by_priority
elif A4.invalidates_p2:
    final_decision = "P2_INVALID_TRAINING_NUMERICS"
elif metrics_gate_pass is False:
    final_decision = "P2_VALID_NEGATIVE"
else:
    final_decision = "UNEXPECTED_AUDIT_STATE"
```

---

## 15. Recommended execution order

```bash
cd /home/ly/AAAI/CGA-main

ROOT=/home/ly/AAAI/CGA-main
P2_DIR=${ROOT}/docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST
P2_SUMMARY=${P2_DIR}/summary.json
AUDIT_DIR=${P2_DIR}/audit_v3
mkdir -p "${AUDIT_DIR}"
```

Then:

```text
R0. Run A0 path-contamination precheck.
R1. If A0 invalidates P2, stop and do not run further interpretation.
R2. Run A1 artifact freeze.
R3. Run A2 protocol symmetry.
R4. Run A3 strict checkpoint load.
R5. Run A4 quantitative loss-scale audit.
R6. Run A5 dynamic eval output-source trace.
R7. Run A6 prediction component-level analysis.
R8. Run A7 per-component target geometry audit.
R9. Generate FINAL_P2_IMPL_AUDIT_SUMMARY.json with invalidation priority.
```

No seed43/44, no ablation, no model changes before R9.

---

## 16. Code-change summary

Allowed audit-only additions:

```text
tools/official/audit_cga_v2_p2_impl_v3.py
tools/official/check_cga_v2_strict_checkpoint_load_v3.py
tools/official/trace_cga_v2_eval_output_source.py
tools/official/audit_cga_v2_target_geometry_components.py
```

Suggested commit boundary:

```bash
git checkout -b cga-v2-p2-audit-only-v3

git add \
  tools/official/audit_cga_v2_p2_impl_v3.py \
  tools/official/check_cga_v2_strict_checkpoint_load_v3.py \
  tools/official/trace_cga_v2_eval_output_source.py \
  tools/official/audit_cga_v2_target_geometry_components.py \
  docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v3/.gitkeep

git commit -m "Add audit-only safeguards for CGA-v2 P2 seed42 failure"
```

Must not be included in this commit:

```text
model changes
loss changes
target-generation changes
threshold changes
HC-Val split changes
result JSON edits
prediction edits
checkpoint edits
```

---

## 17. Final one-line recommendation

```text
Use /home/ly/AAAI/CGA-main as the only canonical root, keep all P2 audit outputs under docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v3, treat unresolved /home/AAAI/CGA-main paths as P2_INVALID_PATH_CONTAMINATION, add dynamic eval trace and per-component target audit, and apply invalidation priority before deciding whether P2 is a valid negative result.
```
