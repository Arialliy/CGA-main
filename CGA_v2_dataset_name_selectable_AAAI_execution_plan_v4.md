# CGA-v2 Dataset-Name-Selectable AAAI Execution Plan v4

## 0. Decision

This plan replaces the previous dataset-name-selectable v3 plan.

```text
Decision:
  ACCEPT_V3_WITH_P3_GATE_AND_FAIL_STOP_REVISIONS

Current execution scope:
  P0-P3 only.

Repository assumption:
  This is a new repo-grade implementation.
  It has no valid paper evidence until it regenerates its own artifacts.

Core rule:
  New repo does not borrow old results.
  Historical OHCM-MSHNet-main results are sanity references only.
  NUDT-SIRST seed42/43/44 paired reproduction is the paper-evidence prerequisite.
```

The method should remain narrowly positioned as:

```text
CGA-v2: target-preserving component-geometry regularization for MSHNet-style IRSTD.
```

Do **not** position it as:

```text
new SOTA IRSTD model
complete hard-clutter false-alarm closure
external validation
transfer learning
multi-benchmark validated method
```

## 1. Dataset-name-selectable repository principle

The repository must follow a BasicIRSTD-style dataset selection interface.

Do **not** create separate hard-coded training pipelines for NUDT / NUAA / IRSTD-1K.

Use one generic interface:

```bash
DATASET_DIR=/path/to/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
```

Change only:

```bash
DATASET_NAME=NUAA-SIRST
DATASET_NAME=IRSTD-1K
```

The same dataset registry, dataset loader, preflight, train runner, eval runner, and report writer must handle all datasets.

## 2. `configs/datasets.yaml` schema

Each dataset must be represented as a registry entry.

```yaml
datasets:
  NUDT-SIRST:
    root_name: NUDT-SIRST
    image_dir: images
    mask_dir: masks
    list_dir: img_idx
    splits:
      train: train_NUDT-SIRST.txt
      test: test_NUDT-SIRST.txt
      hcval: hcval_NUDT-SIRST.txt
    expected_counts:
      train: null
      test: null
      hcval: null
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true
    mask_policy:
      type: binary
      allowed_values: [0, 255]
      canonical_required: false
    claim_policy:
      role: main_evidence_after_three_seed_pass
      external_validation_claim_allowed: false
      benchmark_validation_claim_allowed: false

  NUAA-SIRST:
    root_name: NUAA-SIRST
    image_dir: images
    mask_dir: masks
    list_dir: img_idx
    splits:
      train: train_NUAA-SIRST.txt
      test: test_NUAA-SIRST.txt
    expected_counts:
      train: null
      test: null
      hcval: null
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true
    mask_policy:
      type: binary
      allowed_values: [0, 255]
      canonical_required: false
    claim_policy:
      role: blocked_until_integrity_preflight_pass
      external_validation_claim_allowed: false
      benchmark_validation_claim_allowed: false

  IRSTD-1K:
    root_name: IRSTD-1K
    image_dir: images
    mask_dir: masks
    list_dir: img_idx
    splits:
      train: train_IRSTD-1K.txt
      test: test_IRSTD-1K.txt
    expected_counts:
      train: null
      test: null
      hcval: null
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true
    mask_policy:
      type: canonical_binary
      allowed_values: [0, 255]
      canonical_required: true
    claim_policy:
      role: supervised_characterization_only_after_canonical_preflight_and_target_audit
      external_validation_claim_allowed: false
      benchmark_validation_claim_allowed: false
```

Important details:

```text
expected_counts should be null at first.
Do not hard-code 697 / 256 / 800 unless generated and frozen by current-repo preflight.
```

## 3. Dataset preflight output requirements

The generic dataset preflight must output dataset identity and split identity.

Required fields:

```json
{
  "gate": "Gate-CGA-v2-P1-dataset-preflight",
  "dataset_name": "NUDT-SIRST",
  "dataset_root": "/abs/path/to/datasets/NUDT-SIRST",
  "dataset_registry_sha256": "...",
  "train_list_sha256": "...",
  "test_list_sha256": "...",
  "hcval_list_sha256": "...",
  "train_count": 0,
  "test_count": 0,
  "hcval_count": 0,
  "train_test_overlap_count": 0,
  "duplicate_item_count": 0,
  "missing_images": 0,
  "missing_masks": 0,
  "unreadable_images": 0,
  "unreadable_masks": 0,
  "size_mismatch": 0,
  "illegal_mask_value_files": 0,
  "empty_mask_files": 0,
  "gate_pass": true,
  "decision": "DATASET_PREFLIGHT_PASS"
}
```

Preflight must inspect:

```text
list file existence
image/mask path resolution
item_format suffix behavior
image/mask readability
image/mask size equality
mask value validity under mask_policy
empty mask count
train/test overlap
duplicate items
split sha256
registry sha256
```

## 4. Dataset claim policy

The generic script may run any registered dataset, but paper claims do not automatically upgrade.

```text
NUDT-SIRST:
  Eligible for main paper evidence only after three-seed paired pass.

IRSTD-1K:
  Eligible only for supervised characterization after canonical preflight and target-level audit.
  Not external validation.
  Not multi-benchmark validation.
  Default if new repo has no IRSTD result: omit from paper evidence.

NUAA-SIRST:
  Blocked unless integrity preflight passes on official clean data.
  If any unresolved size mismatch remains, omit from paper evidence.
```

## 5. Execution order

```text
P0: repo identity and contract
P1: dataset registry + NUDT-SIRST preflight
P1.5: closest prior-art threat table
P2: NUDT-SIRST seed42 paired reproduction
P3: NUDT-SIRST seed43/44 paired reproduction + multiseed gate
P4a: seed42 ablation required for submission
P4b: multiseed ablation optional, not required
P5: failure / diagnostic pack
P6: closest baseline comparison pack
P7: claim linter + manuscript pack
```

Do not run NUAA-SIRST or IRSTD-1K before NUDT-SIRST P3 passes.

## 6. P0: repo identity and contract

### 6.1 Required output

Create:

```text
docs/internal/cga_v2/repo_identity/code_delta_statement.md
```

Content:

```markdown
# CGA-v2 Repository Identity Statement

This repository is a new repo-grade implementation.

Historical OHCM-MSHNet-main results are internal references only.
All paper evidence must be regenerated in this repository before being claimed.

This repo must pass model/loss/eval/data contract tests before any training result is accepted.
```

### 6.2 Required contract tests

```text
model import contract
model train/eval forward contract
loss finite contract
CGA target generation contract
eval final-logit-only contract
threshold fixed at 0.5 unless explicitly changed in a frozen protocol
no test-time auxiliary head usage
no post-processing or verifier
```

## 7. P1: NUDT-SIRST dataset preflight

Command:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

P1 must pass before any seed42 training.

If P1 fails:

```text
STOP_NEW_REPO_CGA_V2_AT_DATASET_PREFLIGHT
```

Do not train.
Do not edit split.
Do not delete samples.
Do not modify masks.

## 8. P1.5: closest prior-art threat table

This runs in parallel with P1/P2 because it does not depend on training results.

Required output:

```text
docs/paper/cga_v2_aaai/08_closest_prior_art_threat_table.md
configs/closest_baselines.yaml
```

Must separate:

```text
fair paired comparison:
  MSHNetOHEM vs MSHNetCGA in this repo

literature-only threat:
  MSHNet / SLS
  ISNet / shape-edge reconstruction
  PConv + SD Loss
  other IRSTD SOTA
```

Do not present literature-only numbers as fair same-protocol comparison.

## 9. P2: NUDT-SIRST seed42 paired reproduction

### 9.1 Name

Do not call this “trend strong enough.”

Use:

```text
seed42 reproduction decision rule
```

The thresholds are pre-declared before seeing new-repo seed42 results.

### 9.2 Command

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
```

This must train and evaluate:

```text
MSHNetOHEM seed42 epoch400
MSHNetCGA  seed42 epoch400
Full eval
HC-Val eval
threshold 0.5
```

### 9.3 P2 summary must record rule identity

```json
{
  "gate": "Gate-CGA-v2-P2-seed42-reproduction",
  "dataset_name": "NUDT-SIRST",
  "seed": 42,
  "baseline": "MSHNetOHEM",
  "candidate": "MSHNetCGA",
  "epoch": 400,
  "threshold": 0.5,
  "decision_rule_predeclared": true,
  "decision_rule_name": "seed42 reproduction decision rule",
  "thresholds": {
    "full_delta_mIoU_min": 0.02,
    "full_delta_precision_min": 0.01,
    "full_delta_pd_min": -0.001,
    "full_delta_fa_ppm_max": 0.0,
    "hcval_delta_miou_min": 0.0,
    "hcval_delta_pd_min": -0.001
  },
  "dataset_preflight_summary": "...",
  "dataset_registry_sha256": "...",
  "train_list_sha256": "...",
  "test_list_sha256": "...",
  "hcval_list_sha256": "...",
  "gate_pass": true,
  "decision": "MAY_PROCEED_TO_NUDT_MULTISEED_REPRODUCTION"
}
```

### 9.4 P2 pass rule

```text
Full:
  delta_mIoU      >= +0.020
  delta_Precision >= +0.010
  delta_Pd        >= -0.001
  delta_FA_ppm    <= 0.0

HC-Val:
  delta_mIoU      >= 0.0
  delta_Pd        >= -0.001
```

### 9.5 P2 failure handling

If P2 fails, do **not** immediately stop the method.

First allow exactly one implementation audit:

```text
P2_FAIL_IMPL_AUDIT_ALLOWED:
  dataset split identity
  dataset registry identity
  seed determinism
  model registry
  OHEM loss config
  CGA loss config
  auxiliary target generation
  checkpoint epoch
  eval threshold
  train/eval mode
  summary identity
  metric implementation identity
```

If an implementation error is found, fix only that error and rerun seed42 once.

If no implementation error is found:

```text
STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42
```

Forbidden after scientific P2 fail:

```text
architecture tuning
loss tuning
threshold search
checkpoint search
seed search
ablation winner promotion
NUAA / IRSTD expansion
```

## 10. P3: NUDT-SIRST seed43/44 multiseed paired reproduction

P3 starts only after P2 passes.

### 10.1 Command

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_multiseed_train_eval.sh
```

This must run:

```text
seed43 OHEM / CGA paired train + Full / HC-Val eval
seed44 OHEM / CGA paired train + Full / HC-Val eval
```

Then aggregate seed42/43/44.

### 10.2 P3 pass conditions

P3 is the main paper-evidence gate.

```text
Full split:
  mean delta_mIoU      > 0
  mean delta_Precision > 0
  mean delta_Pd        >= -0.001
  mean delta_FA_ppm    < 0

HC-Val:
  mean delta_mIoU      > 0
  mean delta_Pd        >= -0.001
  at least 2/3 seeds delta_mIoU >= 0
```

Do **not** require every seed HC-Val FA to decrease.

Rationale:

```text
The historical evidence has a known seed44 HC-Val tradeoff where mIoU and Pd improve but Precision and FA worsen.
The new repo should report per-seed caveats instead of forcing an all-seed FA rule.
```

### 10.3 P3 summary required fields

```json
{
  "gate": "Gate-CGA-v2-P3-NUDT-multiseed-reproduction",
  "dataset_name": "NUDT-SIRST",
  "seeds": [42, 43, 44],
  "baseline": "MSHNetOHEM",
  "candidate": "MSHNetCGA",
  "epoch": 400,
  "threshold": 0.5,
  "decision_rule_predeclared": true,
  "decision_rule_name": "NUDT multiseed main evidence rule",
  "full_pass_conditions": {
    "mean_delta_mIoU_gt_0": true,
    "mean_delta_Precision_gt_0": true,
    "mean_delta_Pd_ge_minus_0_001": true,
    "mean_delta_FA_ppm_lt_0": true
  },
  "hcval_pass_conditions": {
    "mean_delta_mIoU_gt_0": true,
    "mean_delta_Pd_ge_minus_0_001": true,
    "at_least_2_of_3_seed_delta_mIoU_ge_0": true,
    "all_seed_FA_decrease_required": false
  },
  "per_seed": {
    "42": {},
    "43": {},
    "44": {}
  },
  "gate_pass": true,
  "decision": "NUDT_MAIN_EVIDENCE_FORMED"
}
```

### 10.4 P3 failure handling

If P3 fails:

```text
STOP_NEW_REPO_CGA_V2_AS_AAAI_MAIN_METHOD
```

Then:

```text
keep repo as implementation / negative-analysis package
keep seed42 result as diagnostic if applicable
do not write NUDT multiseed main claim
do not move to ablation/manuscript as if main evidence passed
do not run NUAA / IRSTD as rescue
do not tune architecture / loss / threshold / checkpoint / seeds
```

P3 fail means the new repo did not regenerate the required main evidence.

## 11. P4: ablation

P4 starts only after P3 passes.

### 11.1 Required for submission

```text
P4a seed42 ablation is required.
```

Only run:

```text
full CGA
aux_off
no_center_peak
no_boundary_scale
```

Do not expand ablation variants.

### 11.2 Not required

```text
P4b multiseed ablation is not required.
```

If time allows, it may be reported as supplemental only, but not as a gate for the main paper.

### 11.3 Interpretation

Ablation must allow mixed attribution.

Do not promote:

```text
aux_off
no_center_peak
no_boundary_scale
```

as the final method.

## 12. P5: failure / diagnostic pack

P5 starts only after P3 passes.

Required outputs:

```text
NUDT per-seed caveat table
HC-Val tradeoff cases
failure-case visual pack
component FP analysis
target size stratification if metrics support it
```

IRSTD-1K default:

```text
Option B: omit IRSTD-1K from new-repo paper evidence.
```

Only include IRSTD-1K if the new repo reruns its own IRSTD-1K supervised train/test characterization and generates its own target-level diagnostic audit.

Do not import historical XDU788 / XDU850 diagnostic cases as new-repo evidence.

## 13. P6: closest baseline comparison pack

Required:

```text
MSHNetOHEM paired baseline in this repo
closest prior-art threat table
clear literature-only vs fair-paired status
```

Optional if time allows:

```text
same-protocol reproduction of one or more public baselines
```

Do not claim SOTA from literature-only comparisons.

## 14. P7: claim linter and manuscript pack

The claim linter must fail unsafe positive claims:

```text
state-of-the-art
external validation
transfer learning
validated on NUAA-SIRST and IRSTD-1K
all auxiliary heads are necessary
complete hard-clutter closure
```

But it must allow these phrases inside sections such as:

```text
Do not claim
Rejected claims
Limitations
Reviewer risks
Unsafe wording
Failure analysis
Diagnostic-only evidence
```

Safe claim:

```text
CGA-v2 is a target-preserving component-geometry regularization method for MSHNet-style IRSTD.
It preserves the final inference path and improves NUDT-SIRST fixed-seed paired results when P3 passes.
```

## 15. Minimal execution checklist

```bash
cd /home/ly/AAAI/OHCM-MSHNet-cga-v2-paper
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

# P0
bash scripts/official/run_cga_v2_repo_contract.sh

# P1
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh

# P1.5
bash scripts/official/run_cga_v2_closest_prior_art_threat_table.sh

# P2
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh

# P3 only after P2 passes
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_multiseed_train_eval.sh
```

## 16. Final execution rule

```text
先只做 NUDT-SIRST。
不跑 NUAA。
不跑 IRSTD-1K。
不调模型。
不调 loss。
不调 threshold。
不换 checkpoint。
不 seed search。
```

If P3 passes, the new repo has formed its NUDT-SIRST main paper evidence.

If P3 fails, stop CGA-v2 as an AAAI main-method route in this new repository.
