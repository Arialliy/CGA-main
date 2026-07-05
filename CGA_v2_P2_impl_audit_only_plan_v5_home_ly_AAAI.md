# CGA-v2 P2 Seed42 Failure: Implementation Audit Plan with Audit-Only Safeguards v5

## 0. Scope

Canonical repository root:

```text
/home/ly/AAAI/CGA-main
```

Existing P2 summary directory:

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/
```

Audit output directory:

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/
```

This document is **not** a method-fix plan. It is an implementation audit plan for the failed seed42 paired result.

Do not change the following during this audit:

```text
model architecture
CGA heads
loss weights
target generation
threshold
HC-Val split
checkpoint selection
seed
training epoch
```

Allowed changes are audit-only scripts and post-P2 safeguards. Any patch to `test.py` strict loading must be labeled as a future-run safeguard and must not be mixed into the interpretation of the already generated P2 result.

---

## 1. Current P2 status

The current P2 summary reports:

```text
gate = Gate-CGA-v2-P2-seed42-reproduction
decision = P2_FAIL_IMPL_AUDIT_ALLOWED
gate_pass = false
seed = 42
epoch = 400
threshold = 0.5
```

Therefore:

```text
Do not run seed43/44.
Do not run ablation.
Do not write positive CGA claims.
Enter implementation audit.
```

The current result may eventually be classified as one of:

```text
P2_INVALID_IMPLEMENTATION
P2_INVALID_PATH_CONTAMINATION
P2_INVALID_ARTIFACTS
P2_INVALID_PROTOCOL
P2_INVALID_CHECKPOINT_LOAD
P2_INVALID_EVAL_OUTPUT_SOURCE
P2_INVALID_ADAPTER_CONTRACT
P2_INVALID_TARGET_GENERATION
P2_INVALID_LOSS_LOG
P2_INVALID_PREDICTION_ARTIFACTS
P2_VALID_NEGATIVE
P2_VALID_NEGATIVE_DESIGN_WEAKNESS
```

It cannot be classified as `P2_VALID_NEGATIVE` until every invalidating audit step has passed.

---

## 2. Unified audit JSON schema

Every audit step must write one JSON file with this minimum schema:

```json
{
  "audit_step": "A0_absolute_path_inventory",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": null,
  "notes": [],
  "artifacts": {}
}
```

Required fields:

| Field | Meaning |
|---|---|
| `audit_step` | Stable step id, for example `A5b_adapter_contract`. |
| `pass` | Whether this audit step passed. |
| `invalidates_p2` | Whether failure prevents interpreting current P2 as valid negative. |
| `requires_rerun` | Whether a fresh seed42 paired rerun is required after fixing the issue. |
| `decision_if_failed` | Concrete invalidation label, for example `P2_INVALID_ADAPTER_CONTRACT`. |
| `notes` | Human-readable audit notes. |
| `artifacts` | Machine-readable evidence paths, hashes, counts, and measurements. |

Final aggregation must read all step JSON files rather than rely on manual explanation.

---

## 3. A0: absolute path inventory and canonical-root classification

### Goal

Detect path contamination before interpreting the result.

The script must not hard-code only `/home/AAAI/CGA-main`. It must recursively extract all absolute paths from:

```text
summary.json
Full summary_metrics.json
HC-Val summary_metrics.json
train logs
checkpoint metadata
result manifests
any JSON under the P2 result directory
```

### Classification

Each absolute path must be classified as one of:

```text
canonical
canonical_missing
realpath_equivalent
noncanonical_resolved_by_mount_proof
noncanonical_unresolved
```

Definitions:

| Class | Meaning | Invalidates P2? |
|---|---|---|
| `canonical` | Path starts with `/home/ly/AAAI/CGA-main` and exists. | No. |
| `canonical_missing` | Path starts with `/home/ly/AAAI/CGA-main` but file/dir is missing. | Not A0. Hand off to A1 as artifact failure. |
| `realpath_equivalent` | Noncanonical path exists and `realpath` resolves under the canonical root. | No, but record mapping. |
| `noncanonical_resolved_by_mount_proof` | Noncanonical path does not exist now, but a documented original-run mount map proves equivalence and corresponding canonical artifacts exist. | No, but high scrutiny. |
| `noncanonical_unresolved` | Noncanonical path cannot be resolved and no mount proof exists. | Yes: `P2_INVALID_PATH_CONTAMINATION`. |

### Mount proof requirement

If a path such as:

```text
/home/AAAI/CGA-main/...
```

appears in the result summary but does not exist in the current host environment, it may only be accepted if an audit note provides:

```text
original run environment id
Docker/container id if available
mount table or bind-mount declaration
old_prefix -> canonical_prefix mapping
proof that the relative artifact exists under canonical root
```

Without this proof:

```text
A0 pass = false
invalidates_p2 = true
decision_if_failed = P2_INVALID_PATH_CONTAMINATION
requires_rerun = true
```

### A0 output

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/A0_absolute_path_inventory.json
```

Example output:

```json
{
  "audit_step": "A0_absolute_path_inventory",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_PATH_CONTAMINATION",
  "notes": [
    "Found noncanonical unresolved checkpoint path /home/AAAI/CGA-main/..."
  ],
  "artifacts": {
    "canonical_root": "/home/ly/AAAI/CGA-main",
    "canonical": [],
    "canonical_missing": [],
    "realpath_equivalent": [],
    "noncanonical_resolved_by_mount_proof": [],
    "noncanonical_unresolved": [
      "/home/AAAI/CGA-main/results/official_from_zero/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar"
    ]
  }
}
```

### Important execution rule

If A0 is invalid:

```text
Stop result interpretation.
Still allow A1 artifact freeze for archival evidence.
Do not enter P2_VALID_NEGATIVE.
```

---

## 4. A1: artifact freeze and code identity hash

### Goal

Freeze the exact result and code state used for audit.

A1 must record both artifacts and code identity. This prevents mixing files from different runs or tracing current code that differs from the P2 execution code.

### Required artifact freeze fields

For each model/split pair:

```text
MSHNetOHEM/test
MSHNetOHEM/hcval
MSHNetCGA/test
MSHNetCGA/hcval
```

record:

```text
checkpoint path
checkpoint exists
checkpoint sha256
checkpoint size
checkpoint mtime
summary_metrics path
summary_metrics sha256
prediction_dir path
prediction png count
train_log path
train_log exists
train_log line count
train_log last epoch
```

### Required code identity hash

Record current hashes for:

```text
test.py
train.py
net.py
loss.py
model/cga_wrapper.py
model/output_contract.py
model/backbones/mshnet_adapter.py
utils/cga_targets.py
```

Also record:

```text
git commit
git branch
git dirty status
git diff --stat
git diff --check result
```

If the P2 summary or run manifest contains a recorded code hash/commit, A1 must compare it against the current code. If absent, record:

```text
historical_code_identity_available = false
```

This does not automatically invalidate P2, but it limits how strongly A5/A5b dynamic tracing can be used to explain the historical P2 run.

### Failure conditions

A1 fails and invalidates P2 if:

```text
required checkpoint missing
required summary missing
prediction directory missing
prediction png count = 0
checkpoint epoch != 400
train log exists but last epoch < 400
summary checksum cannot be computed
canonical_missing paths from A0 point to required artifacts
```

Decision:

```text
P2_INVALID_ARTIFACTS
```

### A1 output

```text
audit_v5/A1_artifact_freeze.json
```

---

## 5. A2: paired protocol audit

### Goal

Ensure OHEM and CGA were compared under the same controlled protocol.

Required equal fields:

```text
dataset_name
train_list_sha256
test_list_sha256
hcval_list_sha256
seed
epoch
threshold
threshold_selection
split names
checkpoint_epoch
metric implementation if logged
```

Required model differences:

```text
baseline.model = MSHNetOHEM
candidate.model = MSHNetCGA
baseline.use_cga = false
candidate.use_cga = true
```

Required paper-evidence metadata, if available in run config or summary:

```text
evidence_mode = paper
paper_evidence_allowed = true
p1_preflight_passed = true
p1a_hcval_source_audit_passed = true
fallback_regularizer_used = false
protocol = controlled
```

If these metadata fields are missing from P2 result files, A2 should not silently pass; it should write:

```text
metadata_complete = false
```

and explain whether other files provide equivalent evidence.

### Failure conditions

A2 invalidates P2 if:

```text
OHEM and CGA use different split hashes
OHEM and CGA use different seed
epoch/checkpoint selection mismatch
threshold mismatch or threshold sweep used
paper-evidence flags contradict paper mode
fallback regularizer used
```

Decision:

```text
P2_INVALID_PROTOCOL
```

### A2 output

```text
audit_v5/A2_paired_protocol.json
```

---

## 6. A3: strict checkpoint-load audit with whitelist normalization

### Goal

Ensure checkpoints can be loaded into the intended model without hidden missing/unexpected keys.

### Allowed normalization only

The strict-load script may apply only these whitelist transformations:

```text
1. unwrap checkpoint["state_dict"]
2. unwrap checkpoint["model"]
3. strip one leading "module."
4. strip one explicitly configured wrapper prefix, for example "model."
```

The script must log every applied normalization:

```json
{
  "normalization_applied": ["unwrap_state_dict", "strip_module_prefix"]
}
```

Not allowed:

```text
silent strict=False
partial key dropping
shape-mismatch ignoring
regex-based broad deletion
loading MSHNetCGA checkpoint into MSHNetOHEM or vice versa
```

### Pass condition

After allowed normalization:

```text
missing_keys = []
unexpected_keys = []
shape_mismatches = []
```

### Failure decision

```text
P2_INVALID_CHECKPOINT_LOAD
```

### A3 output

```text
audit_v5/A3_strict_load_ohem.json
audit_v5/A3_strict_load_cga.json
```

---

## 7. A4: loss-scale and loss-log audit

### Goal

Determine whether the CGA auxiliary losses dominated the main loss or show invalid logging/numerical behavior.

A4 has two layers:

```text
integrity layer: can invalidate P2
interpretive layer: diagnoses design weakness but does not by itself invalidate P2
```

### Required metrics

For tail20 epochs or last available 20 log rows, compute:

```text
base_total_tail20_mean
aux_total_tail20_mean
aux_total_over_total_tail20_mean
aux_total_over_base_total_tail20_mean
center_over_total_tail20_mean
boundary_over_total_tail20_mean
scale_over_total_tail20_mean
peak_over_total_tail20_mean
cga_w_tail20_mean
cga_w_last
nan_count
inf_count
```

Also compare OHEM and CGA base losses:

```text
base_total_cga_vs_ohem_ratio_tail20
soft_iou_cga_vs_ohem_ratio_tail20
ohem_loss_cga_vs_ohem_ratio_tail20
```

### Integrity failures

A4 invalidates P2 if:

```text
required training log missing and no alternate loss log exists
loss log cannot be parsed
NaN/Inf in required losses
final epoch loss row missing
epoch order corrupt
```

Decision:

```text
P2_INVALID_LOSS_LOG
```

### Design diagnosis, not invalidation

A4 may diagnose:

```text
aux_loss_overstrong
ramp_too_aggressive
recall_booster_behavior
```

These do not invalidate P2 unless the log itself is invalid. They support:

```text
P2_VALID_NEGATIVE_DESIGN_WEAKNESS
```

if all hard audits pass.

### A4 output

```text
audit_v5/A4_loss_scale.json
```

---

## 8. A5: dynamic eval output-source trace

### Goal

Prove that evaluation uses the final detector logits, not auxiliary heads or intermediate tensors.

Static grep is not sufficient. A5 must run or instrument one eval forward pass for OHEM and CGA, then record:

```text
model name
checkpoint path
input batch shape
raw output type
selected prediction tensor key
selected prediction tensor shape
selected prediction tensor min/max/mean
sigmoid min/max/mean
threshold
positive pixels after threshold
aux logits present?
aux logits used for prediction?
intermediate features used for prediction?
```

Required prediction source:

```text
final logits only
```

Invalid if:

```text
prediction uses cga_center_logit
prediction uses cga_boundary_logit
prediction uses cga_scale_logit
prediction uses cga_peak_logit
prediction uses feature map instead of final logits
prediction tensor source is ambiguous
CGA and OHEM evaluator choose different output semantics
```

Decision:

```text
P2_INVALID_EVAL_OUTPUT_SOURCE
```

### A5 output

```text
audit_v5/A5_eval_output_trace_ohem.json
audit_v5/A5_eval_output_trace_cga.json
```

---

## 9. A5b: adapter and output-contract audit

### Goal

Explicitly verify that the MSHNet adapter and CGA wrapper satisfy the paper-evidence output contract.

This is distinct from strict checkpoint loading and eval output tracing.

### Required fields to record

For `MSHNetCGA` under a paper-mode dry forward pass, record:

```text
validate_detector_output passed?
adapter_meta
feature_meta
regularizer_impl
fallback_regularizer_used
logits shape
features[0] shape
features[0] dtype
feature stride
feature channels
feature resolution
cga_center_logit shape
cga_boundary_logit shape
cga_scale_logit shape
cga_peak_logit shape
aux logits spatial compatibility with masks/logits
```

### Required evidence metadata

Record and verify:

```text
evidence_mode = paper
paper_evidence_allowed = true
p1_preflight_passed = true
p1a_hcval_source_audit_passed = true
fallback_regularizer_used = false
regularizer_impl = center_boundary_scale_peak
protocol = controlled
```

If `paper_evidence_allowed` is not stored inside the model output by design, A5b must read it from the training/evidence metadata layer and record:

```text
paper_evidence_allowed_source = train_or_manifest_metadata
```

### Invalid conditions

A5b invalidates P2 if:

```text
validate_detector_output fails
adapter_meta missing
feature_meta missing
feature source ambiguous
feature stride/channel/resolution missing
fallback_regularizer_used != false
regularizer_impl != center_boundary_scale_peak
any of four aux logits missing
aux logit shapes incompatible
logits shape incompatible with masks
paper-mode metadata contradicts P1/P1A/fallback requirements
```

Decision:

```text
P2_INVALID_ADAPTER_CONTRACT
```

### A5b output

```text
audit_v5/A5b_adapter_contract.json
```

---

## 10. A6: prediction artifact and component-morphology audit

### Goal

Explain the HC-Val false-alarm explosion while separating artifact failures from diagnostic morphology.

A6 has two layers:

```text
artifact integrity: can invalidate P2
component morphology: diagnostic only unless artifacts are missing/corrupt
```

### Artifact integrity checks

For each prediction directory:

```text
exists
png count
all png readable
prediction shape matches mask shape
prediction ids align with split list
no duplicate prediction ids
no missing prediction ids
```

Invalid if:

```text
prediction directory missing
png count = 0
missing prediction ids
unreadable prediction images
prediction/mask shape mismatch
```

Decision:

```text
P2_INVALID_PREDICTION_ARTIFACTS
```

### Component-level diagnostics

For Full and HC-Val, compute:

```text
per-image FP components
per-image predicted foreground ratio
per-image FP area
per-image TP components if available
CGA - OHEM foreground ratio delta
CGA - OHEM FP component delta
CGA - OHEM FP area delta
top-5 HC-Val false-alarm images by FP components
top-5 HC-Val false-alarm images by foreground ratio delta
```

Classify FA morphology:

```text
area_overactivation
fragmented_noise_components
mixed_area_and_fragmentation
unclear
```

This explains whether HC-Val failure is caused by large-area over-activation, many noise fragments, or both.

### A6 output

```text
audit_v5/A6_prediction_morphology.json
```

---

## 11. A7: target-generation geometry audit with per-component checks

### Goal

Verify that center / boundary / scale / peak targets are geometrically aligned with GT components, especially for multi-component masks.

### Required per-component metrics

For every GT connected component in sampled train images, record:

```text
component_id
component_area
component_bbox
center_target_pixels_inside_component
peak_target_pixels_inside_component
boundary_pixels_near_component
scale_target_value_or_bin
has_center_inside_component
has_peak_inside_component
```

Aggregate:

```text
center_inside_each_gt_component_rate
peak_inside_each_gt_component_rate
component_has_center_or_peak_rate
boundary_near_component_rate
components_without_center_or_peak_count
small_components_without_center_or_peak_count
```

### Invalid if

```text
center target frequently outside GT component
peak target frequently outside GT component
some GT components systematically receive no center/peak signal
boundary target expands far outside tiny components beyond predeclared rule
target shapes incompatible with mask/logits
NaN/Inf target values
```

Decision:

```text
P2_INVALID_TARGET_GENERATION
```

### A7 output

```text
audit_v5/A7_target_geometry.json
```

---

## 12. A8: final aggregation and truth table

### Goal

Aggregate all A0-A7 outputs into a machine-readable final decision.

A8 must not rely on narrative judgment. It reads all JSON files under:

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/
```

and applies:

```text
If any step has invalidates_p2 = true, current P2 cannot be P2_VALID_NEGATIVE.
```

### Invalidation priority

If multiple steps invalidate P2, use this priority for the primary decision while still listing all failures:

```text
1. A0 -> P2_INVALID_PATH_CONTAMINATION
2. A1 -> P2_INVALID_ARTIFACTS
3. A2 -> P2_INVALID_PROTOCOL
4. A3 -> P2_INVALID_CHECKPOINT_LOAD
5. A5 -> P2_INVALID_EVAL_OUTPUT_SOURCE
6. A5b -> P2_INVALID_ADAPTER_CONTRACT
7. A7 -> P2_INVALID_TARGET_GENERATION
8. A4 -> P2_INVALID_LOSS_LOG
9. A6 -> P2_INVALID_PREDICTION_ARTIFACTS
```

A4 and A6 are primarily diagnostic, but if they report invalid artifacts/logs, they are full invalidating conditions.

### Truth table

| Condition | Final decision | Action |
|---|---|---|
| A0 unresolved noncanonical paths | `P2_INVALID_PATH_CONTAMINATION` | Provide mount proof or rerun seed42 from zero under canonical root. |
| A1 required artifacts missing/inconsistent | `P2_INVALID_ARTIFACTS` | Fix artifact/run management and rerun seed42 from zero. |
| A2 paired protocol mismatch | `P2_INVALID_PROTOCOL` | Fix runner/protocol and rerun seed42 paired. |
| A3 strict load fails after whitelist normalization | `P2_INVALID_CHECKPOINT_LOAD` | Fix checkpoint/model loading and rerun evaluation or training as needed. |
| A5 prediction source is aux/intermediate/ambiguous | `P2_INVALID_EVAL_OUTPUT_SOURCE` | Fix evaluator and rerun eval from frozen checkpoints if training artifacts are valid. |
| A5b adapter/output contract fails | `P2_INVALID_ADAPTER_CONTRACT` | Fix adapter/wrapper contract and rerun seed42 from zero. |
| A7 target geometry fails | `P2_INVALID_TARGET_GENERATION` | Fix target generation, label as new protocol/version, rerun from zero. |
| A4 loss log missing/corrupt or NaN/Inf | `P2_INVALID_LOSS_LOG` | Fix logging/numerical issue and rerun seed42 from zero. |
| A6 prediction artifacts missing/corrupt | `P2_INVALID_PREDICTION_ARTIFACTS` | Rerun eval or seed42 as needed. |
| All invalidating audits pass, metrics remain failed | `P2_VALID_NEGATIVE` | Stop AAAI-main CGA-v2 positive route. |
| All hard audits pass, A4/A6 diagnose over-recall/overactivation | `P2_VALID_NEGATIVE_DESIGN_WEAKNESS` | Treat as method/design failure; any rescue becomes CGA-v2.1 with predeclared protocol. |

### A8 output

```text
audit_v5/A8_final_audit_decision.json
```

Required final JSON fields:

```json
{
  "final_decision": "P2_VALID_NEGATIVE",
  "can_run_seed43_44": false,
  "can_claim_positive_cga": false,
  "requires_seed42_rerun": false,
  "invalidating_steps": [],
  "diagnostic_steps": ["A4_loss_scale", "A6_prediction_morphology"],
  "notes": []
}
```

---

## 13. Execution order

Recommended execution:

```text
A0 absolute path inventory
A1 artifact freeze, even if A0 invalid
A2 paired protocol audit
A3 strict checkpoint-load audit
A5 dynamic eval output-source trace
A5b adapter/output-contract audit
A7 target-generation audit
A4 loss-scale audit
A6 prediction morphology audit
A8 final aggregation
```

If A0 invalidates P2:

```text
Stop interpreting metrics.
Still run A1 to freeze evidence.
A8 final decision cannot be P2_VALID_NEGATIVE.
```

If any of A1/A2/A3/A5/A5b/A7 invalidates P2:

```text
Do not interpret current result as valid method failure.
Fix the implementation/protocol issue.
Rerun seed42 according to the corrected, predeclared protocol.
```

If only A4/A6 show diagnostic over-recall and no invalidation:

```text
Current P2 is a valid negative or design-weakness result.
Do not run seed43/44 for CGA-v2.
Any method rescue must be CGA-v2.1 with a new predeclared protocol.
```

---

## 14. Paper claim downgrade while audit is pending

Allowed statement:

```text
Under the current seed42 paired protocol, CGA failed the predeclared P2 gate and is under implementation audit.
```

Not allowed:

```text
CGA improves target preservation.
CGA reduces false alarms.
CGA is robust on HC-Val.
CGA is effective for MSHNet-style IRSTD.
CGA should proceed to seed43/44.
```

If A8 returns `P2_VALID_NEGATIVE`, allowed final statement:

```text
After implementation audit, the current CGA-v2 configuration is a valid seed42 negative: it increases detection probability but degrades precision and false-alarm behavior, especially on HC-Val. The AAAI-main CGA-v2 positive route is stopped.
```

If A8 returns any `P2_INVALID_*`, allowed final statement:

```text
The current P2 result is implementation-invalid and cannot be interpreted as either positive or negative method evidence. Fix the invalidating condition and rerun seed42 from zero or rerun evaluation from frozen valid checkpoints, depending on the failure type.
```
