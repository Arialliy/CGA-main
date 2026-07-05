# CGA-v2 P2 Seed42 Failure: Implementation Audit Plan with Audit-Only Safeguards v4

## 0. Scope

This document supersedes the v3 audit plan.

It is an **implementation audit plan**, not a method-fixing plan. It must not be used to justify continuing seed43/44, changing thresholds, changing HC-Val, changing CGA losses, or rewriting the paper claim as positive evidence.

The canonical project root is:

```text
/home/ly/AAAI/CGA-main
```

Do not use these paths as canonical roots:

```text
/home/md0/ly/CGA-main
/home/AAAI/CGA-main
/home/ly/AAAI/OHCM-MSHNet-main/datasets
```

If historical outputs contain `/home/AAAI/CGA-main`, it must be handled by the path-contamination audit. It is not automatically accepted as equivalent.

---

## 1. Current P2 state

Use the existing P2 directory:

```text
P2_DIR=docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST
P2_SUMMARY=docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json
AUDIT_DIR=docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v4
```

The P2 result has already failed the predeclared seed42 gate. Current action is:

```text
Do not run seed43/44.
Do not run ablation.
Do not write positive CGA paper claims.
Run implementation audit only.
```

---

## 2. Audit-only principles

Allowed:

```text
1. Read existing summaries, checkpoints, train logs, prediction masks, split lists, and code metadata.
2. Add audit-only scripts under tools/official/ or scripts/official/.
3. Generate machine-readable JSON audit outputs under AUDIT_DIR.
4. Freeze artifacts and compute hashes.
5. Run dry forward/eval traces to identify tensor sources.
```

Not allowed during current P2 interpretation:

```text
1. Changing model architecture.
2. Changing CGA target generation.
3. Changing loss weights, ramp schedule, threshold, split, or checkpoint selection.
4. Patching test.py and then mixing patched outputs with the current P2 result.
5. Calling this result P2_VALID_NEGATIVE before all hard invalidation audits pass.
```

`test.py` strict-load changes may be added later as **future-run safeguards**, but the current P2 audit must first answer whether the already-generated P2 artifacts are valid.

---

## 3. Required JSON schema for every audit step

Every audit script must write one JSON file with this minimum schema:

```json
{
  "audit_step": "A0_path_inventory",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_PATH_CONTAMINATION",
  "notes": [],
  "artifacts": {}
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `pass` | Whether this audit step passes. |
| `invalidates_p2` | Whether failure means the current P2 result cannot be interpreted as either method-positive or method-negative. |
| `requires_rerun` | Whether the next valid evidence step must rerun seed42 from zero after fixing the issue. |
| `decision_if_failed` | Machine-readable failure decision. |
| `notes` | Human-readable explanation. |
| `artifacts` | Hashes, counts, paths, and detailed diagnostic payloads. |

The final aggregator must read all audit JSONs and produce:

```text
AUDIT_DIR/final_p2_audit_decision.json
AUDIT_DIR/final_p2_audit_table.md
```

---

## 4. A0: absolute-path inventory and contamination audit

### Purpose

Do not hard-code only `/home/AAAI/CGA-main`. The audit must automatically extract **all absolute paths** from:

```text
P2 summary.json
Full/Hcval summary_metrics.json if referenced
result manifests if present
train logs if present
checkpoint metadata if inspectable
```

Then classify every path as:

```text
canonical
realpath_equivalent
noncanonical_unresolved
```

### Canonical root

```text
CANONICAL_ROOT=/home/ly/AAAI/CGA-main
```

### Classification rules

| Class | Rule |
|---|---|
| `canonical` | Path string starts with `/home/ly/AAAI/CGA-main` and `realpath(path)` is inside canonical root. |
| `realpath_equivalent` | Path string is noncanonical but exists and `realpath(path)` is inside canonical root, or a documented Docker mount proof maps it to canonical root. |
| `noncanonical_unresolved` | Path string is noncanonical and does not exist in the current environment, or no mount/realpath proof is provided. |

### Hard-stop rule

If any checkpoint, prediction directory, dataset path, split list, summary, or training log path is `noncanonical_unresolved`, output:

```json
{
  "audit_step": "A0_path_inventory",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_PATH_CONTAMINATION"
}
```

This stops **result interpretation**.

It does **not** prevent A1 artifact freeze from running for archival purposes.

### Required output fields

```json
{
  "audit_step": "A0_path_inventory",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_PATH_CONTAMINATION",
  "canonical_root": "/home/ly/AAAI/CGA-main",
  "absolute_paths": [
    {
      "path": "/home/AAAI/CGA-main/results/...",
      "source_file": "summary.json",
      "json_pointer": "/full/baseline/checkpoint",
      "exists": false,
      "realpath": null,
      "class": "noncanonical_unresolved"
    }
  ],
  "path_classes": {
    "canonical": 0,
    "realpath_equivalent": 0,
    "noncanonical_unresolved": 0
  },
  "mount_proof": null,
  "notes": []
}
```

### Suggested command

```bash
cd /home/ly/AAAI/CGA-main

mkdir -p docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v4

python3 tools/official/audit_p2_a0_path_inventory.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --p2_dir docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST \
  --summary docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json \
  --output docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v4/A0_path_inventory.json
```

---

## 5. A1: artifact freeze

### Purpose

Even if A0 fails, A1 should still run where possible. A1 creates an immutable audit snapshot so later analysis cannot accidentally mix artifacts from different runs.

### Required records

```text
git commit
git branch
git dirty status
git diff --stat
checkpoint sha256
checkpoint mtime
checkpoint size
summary.json sha256
summary_metrics.json sha256 if present
prediction png count
prediction png aggregate sha256 if feasible
train_log line count
train_log last epoch
train_log sha256
split list sha256
checkpoint epoch metadata
```

### Required output decision

A1 invalidates P2 if any referenced required artifact is missing, mutable without trace, or inconsistent with the summary.

```json
{
  "audit_step": "A1_artifact_freeze",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_ARTIFACTS",
  "artifacts": {
    "git": {
      "commit": "...",
      "branch": "...",
      "dirty": true,
      "diff_stat": "..."
    },
    "checkpoints": {
      "MSHNetOHEM": {
        "path": "...",
        "sha256": "...",
        "mtime_iso": "...",
        "size_bytes": 0
      },
      "MSHNetCGA": {
        "path": "...",
        "sha256": "...",
        "mtime_iso": "...",
        "size_bytes": 0
      }
    },
    "predictions": {
      "full_ohem_png_count": 0,
      "full_cga_png_count": 0,
      "hcval_ohem_png_count": 0,
      "hcval_cga_png_count": 0
    },
    "train_logs": {
      "ohem_line_count": 0,
      "ohem_last_epoch": 400,
      "cga_line_count": 0,
      "cga_last_epoch": 400
    }
  },
  "notes": []
}
```

---

## 6. A2: paired protocol symmetry audit

### Purpose

Verify OHEM and CGA differ only in the intended variable:

```text
use_cga=false vs use_cga=true
```

### Required symmetry checks

```text
same dataset_name
same train split hash
same test split hash
same HC-Val split hash
same dataset_registry_sha256
same seed
same epoch
same checkpoint epoch
same threshold=0.5
same threshold_selection=fixed_predeclared
same image resolution / patch size if logged
same checkpoint selection rule
same eval implementation
same evidence_mode=paper
same protocol=controlled
same P1/P1A flags
```

If a protocol mismatch exists, output:

```json
{
  "audit_step": "A2_paired_protocol_symmetry",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_PROTOCOL",
  "mismatches": [
    {
      "field": "threshold",
      "baseline": 0.5,
      "candidate": 0.4
    }
  ],
  "notes": []
}
```

---

## 7. A3: strict checkpoint-load audit with whitelisted normalization

### Purpose

`strict=False` can hide checkpoint problems. The audit must check whether the epoch400 OHEM and CGA checkpoints can be loaded into the intended model graph under a strict key match after **only whitelisted normalization**.

### Allowed state_dict extraction

Allowed:

```text
checkpoint itself is a state_dict
checkpoint["state_dict"]
checkpoint["model_state_dict"]
checkpoint["model"]
```

### Allowed key normalization

Allowed only if explicitly recorded:

```text
strip exactly one leading "module."
strip exactly one explicitly configured wrapper prefix, e.g. "model."
```

Not allowed:

```text
silent strict=False
ignoring missing keys
dropping unexpected keys
renaming arbitrary key substrings
partial load
loading only backbone while skipping CGA wrapper heads
```

### Required CGA-specific checks

For `MSHNetCGA`, strict load must include the four CGA auxiliary heads:

```text
cga_center_logit-related head parameters
cga_boundary_logit-related head parameters
cga_scale_logit-related head parameters
cga_peak_logit-related head parameters
```

If missing/unexpected keys remain after whitelisted normalization:

```json
{
  "audit_step": "A3_strict_checkpoint_load",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_CHECKPOINT_LOAD",
  "normalization_applied": ["unwrap:state_dict", "strip_prefix:module."],
  "missing_keys": ["..."],
  "unexpected_keys": ["..."],
  "notes": []
}
```

---

## 8. A4: loss-scale and ramp audit

### Purpose

Determine whether CGA behaves like an over-strong recall booster because auxiliary terms dominate the main loss.

This audit is diagnostic unless it reveals NaN/Inf, missing loss keys, impossible ramp values, or clearly inconsistent logging.

### Required metrics

Compute over the final 20 logged epochs, or the largest available suffix if fewer logs exist:

```text
aux_total / total tail20 mean
aux_total / base_total tail20 mean
center_loss / total tail20 mean
boundary_loss / total tail20 mean
scale_loss / total tail20 mean
peak_loss / total tail20 mean
cga_w tail20 mean
cga_w final value
NaN count
Inf count
CGA base_total vs OHEM base_total ratio
CGA ohem loss vs OHEM ohem loss ratio if logged
CGA soft_iou loss vs OHEM soft_iou loss ratio if logged
```

### Suggested diagnostic thresholds

These thresholds should not invalidate P2 by themselves unless they expose impossible or corrupted values:

```text
aux_total / base_total tail20 mean > 0.50  -> config_overstrong_warning
aux_total / total tail20 mean > 0.33      -> config_overstrong_warning
cga_w final != expected ramp target        -> ramp_mismatch_warning or invalid if due to bug
any NaN/Inf                                -> P2_INVALID_LOSS_LOG
```

### Required output example

```json
{
  "audit_step": "A4_loss_scale_ramp",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_LOSS_LOG",
  "diagnosis": "config_overstrong_warning",
  "metrics": {
    "tail_n": 20,
    "aux_total_over_base_total_mean": 0.0,
    "aux_total_over_total_mean": 0.0,
    "center_over_total_mean": 0.0,
    "boundary_over_total_mean": 0.0,
    "scale_over_total_mean": 0.0,
    "peak_over_total_mean": 0.0,
    "cga_w_tail_mean": 1.0,
    "cga_w_final": 1.0,
    "nan_count": 0,
    "inf_count": 0
  },
  "notes": []
}
```

---

## 9. A5: dynamic eval output-source trace

### Purpose

Static grep is not enough. The audit must dynamically trace what tensor is actually used for prediction.

### Required trace

Run the evaluator on a tiny deterministic sample set from test and HC-Val, without changing threshold or checkpoint.

Record:

```text
model name
checkpoint path
input id
input shape
raw output type
selected prediction tensor key
selected prediction tensor shape
selected tensor min/max/mean
sigmoid tensor min/max/mean
threshold=0.5
positive pixel count
whether aux logits exist
whether aux logits were used for prediction
whether intermediate feature tensor was used for prediction
```

### Hard invalidation rule

If prediction uses any of these instead of final logits:

```text
cga_center_logit
cga_boundary_logit
cga_scale_logit
cga_peak_logit
decoder feature
intermediate tensor
first 4D tensor guessed from tuple/list
```

then output:

```json
{
  "audit_step": "A5_dynamic_eval_output_trace",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_EVAL_OUTPUT_SOURCE",
  "notes": ["Evaluation used cga_peak_logit instead of final logits."]
}
```

### Required output example

```json
{
  "audit_step": "A5_dynamic_eval_output_trace",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_EVAL_OUTPUT_SOURCE",
  "traces": [
    {
      "split": "test",
      "image_id": "...",
      "selected_key": "logits",
      "shape": [1, 1, 256, 256],
      "logit_min": -10.0,
      "logit_max": 10.0,
      "sigmoid_min": 0.0,
      "sigmoid_max": 1.0,
      "positive_pixels_at_0p5": 0,
      "aux_keys_present": [
        "cga_center_logit",
        "cga_boundary_logit",
        "cga_scale_logit",
        "cga_peak_logit"
      ],
      "aux_used_for_prediction": false
    }
  ],
  "notes": []
}
```

---

## 10. A5b: adapter and output-contract audit

### Purpose

The previous audits check loading and prediction-source behavior, but they do not explicitly confirm that the CGA wrapper/adapter obeys the paper contract. This step must be added.

### Required checks

Run one forward pass in the same model construction path used by P2 and record:

```text
validate_detector_output passed
adapter_meta
feature_meta
regularizer_impl
fallback_regularizer_used=false
logits shape
features[0] shape
feature stride
feature channels
feature resolution
four auxiliary logits exist
four auxiliary logits shapes
aux logits spatial resolution relation to logits
paper/smoke evidence metadata fields
```

### Required CGA contract

For `MSHNetCGA`, all must hold:

```text
regularizer_impl == center_boundary_scale_peak
fallback_regularizer_used == false
output["logits"] exists
output["features"] is non-empty
output["features"][0] is a tensor
output["adapter_meta"] exists
output["feature_meta"] exists
cga_center_logit exists
cga_boundary_logit exists
cga_scale_logit exists
cga_peak_logit exists
all aux logits are finite tensors
aux logits batch/spatial dimensions are compatible with training loss targets
validate_detector_output passes without guessing first 4D tensor
```

### Hard invalidation rule

If feature source is wrong, fallback is used, aux head is missing, aux shape is incompatible, or output validation fails:

```json
{
  "audit_step": "A5b_adapter_contract",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_ADAPTER_CONTRACT",
  "notes": []
}
```

### Required output example

```json
{
  "audit_step": "A5b_adapter_contract",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_ADAPTER_CONTRACT",
  "contract": {
    "validate_detector_output_passed": true,
    "adapter_meta": {
      "backbone": "mshnet",
      "logits_source": "...",
      "feature_source": "..."
    },
    "feature_meta": [
      {
        "source": "...",
        "stride": 1,
        "channels": 16,
        "resolution": [256, 256]
      }
    ],
    "regularizer_impl": "center_boundary_scale_peak",
    "fallback_regularizer_used": false,
    "logits_shape": [1, 1, 256, 256],
    "features_0_shape": [1, 16, 256, 256],
    "aux_logits": {
      "cga_center_logit": [1, 1, 256, 256],
      "cga_boundary_logit": [1, 1, 256, 256],
      "cga_scale_logit": [1, 1, 256, 256],
      "cga_peak_logit": [1, 1, 256, 256]
    }
  },
  "notes": []
}
```

---

## 11. A6: prediction morphology and component-level false alarm audit

### Purpose

HC-Val FA increased sharply. Pixel-level foreground mass is not enough. The audit must identify whether the false alarms are caused by:

```text
area overactivation
fragmented noise components
background edge hallucination
hot-object/highlight activation
mixed behavior
```

### Required metrics

Compute for OHEM and CGA, per split and per image:

```text
foreground pixel count
foreground ratio
foreground ratio delta CGA-OHEM
FP pixel count
FP area ratio
FP connected components
FP components per image
mean FP component area
top-5 HC-Val images by FP component increase
top-5 HC-Val images by FP pixel increase
top-5 HC-Val images by foreground-ratio increase
classification: area_overactivation / fragmented_noise_components / mixed
```

### Required output example

```json
{
  "audit_step": "A6_prediction_morphology",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_PREDICTION_ARTIFACTS",
  "diagnosis": {
    "hcval_failure_type": "fragmented_noise_components",
    "cga_minus_ohem_mean_foreground_ratio": 0.0,
    "cga_minus_ohem_mean_fp_components": 0.0,
    "cga_minus_ohem_mean_fp_pixels": 0.0
  },
  "top5_hcval_false_alarm_images": [
    {
      "image_id": "...",
      "ohem_fp_components": 0,
      "cga_fp_components": 0,
      "delta_fp_components": 0,
      "ohem_fp_pixels": 0,
      "cga_fp_pixels": 0,
      "delta_fp_pixels": 0,
      "ohem_fg_ratio": 0.0,
      "cga_fg_ratio": 0.0,
      "delta_fg_ratio": 0.0
    }
  ],
  "notes": []
}
```

A6 is normally diagnostic. It invalidates P2 only if required prediction artifacts are missing, unreadable, or mismatched with summary paths/counts.

---

## 12. A7: per-component target-geometry audit

### Purpose

The CGA target generator must be checked at component level, not only image level. NUDT images may contain multiple connected target components.

### Required per-component checks

For every GT connected component in test and HC-Val, or a documented deterministic sample if full audit is expensive:

```text
component area
component bbox
component centroid
center target positive pixels inside this component
peak target positive pixels inside this component
boundary target pixels near this component
scale target finite and localized
whether component has at least one center or peak point inside GT
```

### Required aggregate metrics

```text
center_inside_each_gt_component_rate
peak_inside_each_gt_component_rate
component_has_center_or_peak_rate
boundary_overlap_with_gt_or_local_ring_rate
scale_finite_rate
scale_nonzero_near_component_rate
components_without_center_or_peak list
```

### Hard invalidation rule

If components systematically lack center/peak targets, or boundary/peak targets expand far outside small targets in a way inconsistent with the documented target-generation policy:

```json
{
  "audit_step": "A7_target_geometry",
  "pass": false,
  "invalidates_p2": true,
  "requires_rerun": true,
  "decision_if_failed": "P2_INVALID_TARGET_GENERATION",
  "notes": []
}
```

### Required output example

```json
{
  "audit_step": "A7_target_geometry",
  "pass": true,
  "invalidates_p2": false,
  "requires_rerun": false,
  "decision_if_failed": "P2_INVALID_TARGET_GENERATION",
  "metrics": {
    "num_images": 0,
    "num_components": 0,
    "center_inside_each_gt_component_rate": 1.0,
    "peak_inside_each_gt_component_rate": 1.0,
    "component_has_center_or_peak_rate": 1.0,
    "scale_finite_rate": 1.0,
    "scale_nonzero_near_component_rate": 1.0
  },
  "components_without_center_or_peak": [],
  "notes": []
}
```

---

## 13. A8: final audit aggregation and truth table

### Invalidation priority rule

Invalidating conditions have priority over negative diagnosis.

If any of the following fail with `invalidates_p2=true`, the final decision must **not** be `P2_VALID_NEGATIVE`:

```text
A0 path inventory / path contamination
A1 artifact freeze
A2 paired protocol symmetry
A3 strict checkpoint load
A5 dynamic eval output source
A5b adapter contract
A7 target geometry
```

A4 and A6 are primarily diagnostic unless they reveal corrupted artifacts, NaN/Inf, or missing required logs/predictions.

### Final truth table

| Condition | Final decision | Action |
|---|---|---|
| A0 unresolved noncanonical path | `P2_INVALID_PATH_CONTAMINATION` | Provide mount proof or rerun seed42 under canonical root. |
| A1 artifacts missing/inconsistent | `P2_INVALID_ARTIFACTS` | Freeze correct artifacts or rerun seed42. |
| A2 protocol mismatch | `P2_INVALID_PROTOCOL` | Fix runner/protocol and rerun seed42 from zero. |
| A3 strict load fails | `P2_INVALID_CHECKPOINT_LOAD` | Fix checkpoint/model loading and rerun seed42 from zero. |
| A5 prediction uses aux/intermediate tensor | `P2_INVALID_EVAL_OUTPUT_SOURCE` | Fix evaluator and rerun eval/training as required. |
| A5b adapter contract fails | `P2_INVALID_ADAPTER_CONTRACT` | Fix adapter/wrapper contract and rerun seed42 from zero. |
| A7 target geometry fails | `P2_INVALID_TARGET_GENERATION` | Fix target generator and predeclare new valid rerun. |
| A0/A1/A2/A3/A5/A5b/A7 pass, A4 shows aux over-strong, A6 shows overactivation | `P2_VALID_NEGATIVE_DESIGN_WEAKNESS` | Stop AAAI-main CGA-v2; any fix is CGA-v2.1 with new protocol. |
| All hard audits pass and metrics remain bad | `P2_VALID_NEGATIVE` | Stop AAAI-main CGA-v2. |
| Hard bug found and fixed | `P2_INVALID_IMPLEMENTATION_FIXED_PENDING_RERUN` | Rerun seed42 from zero before any paper claim. |

### Required final output

```json
{
  "gate": "Gate-CGA-v2-P2-seed42-reproduction-audit-v4",
  "canonical_root": "/home/ly/AAAI/CGA-main",
  "p2_summary": "docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json",
  "final_decision": "P2_INVALID_PATH_CONTAMINATION",
  "p2_valid_negative_allowed": false,
  "seed43_44_allowed": false,
  "ablation_allowed": false,
  "paper_positive_claim_allowed": false,
  "rerun_seed42_required": true,
  "audit_steps": {
    "A0_path_inventory": "fail",
    "A1_artifact_freeze": "pass_or_not_run",
    "A2_paired_protocol_symmetry": "not_run",
    "A3_strict_checkpoint_load": "not_run",
    "A4_loss_scale_ramp": "not_run",
    "A5_dynamic_eval_output_trace": "not_run",
    "A5b_adapter_contract": "not_run",
    "A6_prediction_morphology": "not_run",
    "A7_target_geometry": "not_run"
  },
  "notes": []
}
```

---

## 14. Recommended execution order

```bash
cd /home/ly/AAAI/CGA-main

export P2_DIR=docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST
export AUDIT_DIR=${P2_DIR}/audit_v4
mkdir -p "${AUDIT_DIR}"
```

Run A0 first:

```bash
python3 tools/official/audit_p2_a0_path_inventory.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --p2_dir "${P2_DIR}" \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A0_path_inventory.json"
```

If A0 invalidates P2:

```text
Stop result interpretation.
Still run A1 artifact freeze if possible.
Then generate final audit decision.
Do not run seed43/44.
Do not write P2_VALID_NEGATIVE.
```

Run A1 freeze even after A0 invalidation where files are reachable:

```bash
python3 tools/official/audit_p2_a1_artifact_freeze.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --p2_dir "${P2_DIR}" \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A1_artifact_freeze.json"
```

Only if A0/A1 do not invalidate P2, continue interpretive audits:

```bash
python3 tools/official/audit_p2_a2_protocol_symmetry.py \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A2_paired_protocol_symmetry.json"

python3 tools/official/audit_p2_a3_strict_checkpoint_load.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A3_strict_checkpoint_load.json"

python3 tools/official/audit_p2_a4_loss_scale_ramp.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A4_loss_scale_ramp.json"

python3 tools/official/audit_p2_a5_dynamic_eval_output_trace.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A5_dynamic_eval_output_trace.json"

python3 tools/official/audit_p2_a5b_adapter_contract.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A5b_adapter_contract.json"

python3 tools/official/audit_p2_a6_prediction_morphology.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A6_prediction_morphology.json"

python3 tools/official/audit_p2_a7_target_geometry.py \
  --canonical_root /home/ly/AAAI/CGA-main \
  --summary "${P2_DIR}/summary.json" \
  --output "${AUDIT_DIR}/A7_target_geometry.json"
```

Finally aggregate:

```bash
python3 tools/official/audit_p2_a8_aggregate.py \
  --audit_dir "${AUDIT_DIR}" \
  --summary "${P2_DIR}/summary.json" \
  --output_json "${AUDIT_DIR}/final_p2_audit_decision.json" \
  --output_md "${AUDIT_DIR}/final_p2_audit_table.md"
```

---

## 15. Paper-claim implications

Until the audit final decision is known, the only valid statement is:

```text
The current P2 seed42 paired result failed the predeclared gate and is under implementation audit.
```

If final decision is `P2_INVALID_*`:

```text
The current P2 result cannot be interpreted as method-positive or method-negative.
Fix the invalidating issue and rerun seed42 from zero.
```

If final decision is `P2_VALID_NEGATIVE`:

```text
The current CGA-v2 implementation is a valid negative under the frozen seed42 protocol.
It should not be submitted as an AAAI-main positive method.
```

If final decision is `P2_VALID_NEGATIVE_DESIGN_WEAKNESS`:

```text
CGA-v2 behaves as a recall booster and false-alarm amplifier under the current configuration.
Any rescue requires a new predeclared version such as CGA-v2.1.
```

---

## 16. One-line conclusion

The three proposed changes are accepted:

```text
Add A5b adapter/contract audit.
Make A0 automatically extract and classify all absolute paths.
Let A1 artifact freeze run even if A0 invalidates P2, while stopping result interpretation.
```

The priority rule is strict:

```text
Any hard invalidation prevents P2_VALID_NEGATIVE.
Only after A0/A1/A2/A3/A5/A5b/A7 all pass can the failed metrics be interpreted as a valid negative result.
```
