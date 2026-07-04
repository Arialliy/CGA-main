# CGA-v2 Dataset-Name-Selectable AAAI Execution Plan v2

## 0. Decision

This plan replaces the earlier hard-coded three-dataset route.

```text
Decision:
  USE_DATASET_NAME_SELECTABLE_REPO_STRUCTURE

Style:
  BasicIRSTD-style dataset selection

Core command shape:
  DATASET_DIR=/path/to/datasets \
  DATASET_NAME=NUDT-SIRST \
  SEED=42 \
  bash scripts/official/run_cga_v2_one_dataset_paired_seed.sh
```

Do **not** create three separate code paths for `NUDT-SIRST`, `NUAA-SIRST`, and `IRSTD-1K`. The three datasets are registry entries under the same training / testing / evaluation pipeline.

The immediate goal is not multi-benchmark validation. The immediate goal is:

```text
1. Make the new repo dataset-name selectable.
2. Run registry-driven data preflight.
3. Re-generate NUDT-SIRST evidence in the new repo.
4. Keep NUAA / IRSTD-1K behind dataset-specific claim and preflight policies.
```

---

## 1. Why this revision is necessary

The current AAAI route is a narrow-claim rescue route, not a new SOTA / multi-benchmark validation claim. The paper should be positioned as:

```text
CGA-v2: target-preserving component-geometry regularization for MSHNet-style IRSTD.
```

The uploaded narrow-claim plan explicitly states that CGA-v2 should follow a `revise / narrow claim / evidence-first` route, not be submitted as a new SOTA model with broad benchmark validation.

The new overlay repository has no experiment artifacts, so historical results from `OHCM-MSHNet-main` can only be used as sanity references. Paper evidence must be regenerated in the current repository.

---

## 2. Repository principle

### 2.1 Wrong structure

Do not implement:

```text
scripts/official/run_nudt_train_eval.sh
scripts/official/run_nuaa_train_eval.sh
scripts/official/run_irstd1k_train_eval.sh
```

Do not write dataset-specific training branches inside `train.py`, `test.py`, or `evaluate.py`.

### 2.2 Correct structure

Implement:

```text
configs/datasets.yaml

tools/official/dataset_registry.py

dataset.py
  accepts dataset_name and dataset_dir

train.py
  --dataset_name
  --dataset_dir

test.py / evaluate.py
  --dataset_name
  --dataset_dir

scripts/official/run_cga_v2_dataset_preflight.sh
scripts/official/run_cga_v2_one_dataset_paired_seed.sh
scripts/official/run_cga_v2_nudt_multiseed_train_eval.sh
```

Then switch datasets only by:

```bash
DATASET_NAME=NUDT-SIRST
DATASET_NAME=NUAA-SIRST
DATASET_NAME=IRSTD-1K
```

---

## 3. Dataset registry schema

Add:

```text
configs/datasets.yaml
```

Example schema:

```yaml
datasets:
  NUDT-SIRST:
    root_name: NUDT-SIRST
    image_dir: images
    mask_dir: masks
    index_dir: img_idx
    train_list: train_NUDT-SIRST.txt
    test_list: test_NUDT-SIRST.txt
    hcval_list: hcval_NUDT-SIRST.txt
    expected_counts:
      train: 697
      test: null
      hcval: null
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true

  NUAA-SIRST:
    root_name: NUAA-SIRST
    image_dir: images
    mask_dir: masks
    index_dir: img_idx
    train_list: train_NUAA-SIRST.txt
    test_list: test_NUAA-SIRST.txt
    hcval_list: null
    expected_counts:
      train: 256
      test: 86
      hcval: null
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true

  IRSTD-1K:
    root_name: IRSTD-1K
    image_dir: images
    mask_dir: masks
    index_dir: img_idx
    train_list: train_IRSTD-1K.txt
    test_list: test_IRSTD-1K.txt
    hcval_list: null
    expected_counts:
      train: 800
      test: 201
      hcval: null
    item_format:
      list_has_extension: false
      image_suffix: .png
      mask_suffix: .png
      id_prefix: ""
      id_suffix: ""
      strip_extension_from_list_item: true
```

### 3.1 Why `item_format` is required

BasicIRSTD-style lists may contain any of the following:

```text
000001
000001.png
Misc_111
XDU788
```

Therefore the registry must define how list entries become image / mask paths. Do not hard-code `.png` behavior inside `dataset.py`.

### 3.2 Required item resolution function

Add:

```text
tools/official/dataset_registry.py
```

Required behavior:

```python
from pathlib import Path


def normalize_item_id(raw_item: str, item_format: dict) -> str:
    item = raw_item.strip()
    if item_format.get("strip_extension_from_list_item", True):
        item = Path(item).stem
    prefix = item_format.get("id_prefix", "")
    suffix = item_format.get("id_suffix", "")
    return f"{prefix}{item}{suffix}"


def resolve_image_mask_paths(dataset_root: Path, item: str, spec: dict) -> tuple[Path, Path]:
    item_format = spec["item_format"]
    item_id = normalize_item_id(item, item_format)
    image = dataset_root / spec["image_dir"] / f"{item_id}{item_format['image_suffix']}"
    mask = dataset_root / spec["mask_dir"] / f"{item_id}{item_format['mask_suffix']}"
    return image, mask
```

---

## 4. Generic dataset preflight

Add:

```text
tools/official/check_cga_v2_dataset_preflight.py
scripts/official/run_cga_v2_dataset_preflight.sh
tests/test_cga_v2_dataset_registry.py
tests/test_cga_v2_dataset_preflight.py
```

Preflight must be generic and registry-driven:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
PYTHON=${PYTHON:-python3} \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

### 4.1 Required checks

For each split:

```text
list file exists
list file is readable
image exists
mask exists
image readable
mask readable
image/mask spatial size match
mask values valid or match declared mask policy
no path traversal
no duplicate list entries
no train/test overlap
```

### 4.2 Required split hash fields

Preflight output must include list hashes. `dataset_spec_sha256` alone is insufficient because registry config can stay fixed while list files change.

Required output identity:

```json
{
  "gate": "Gate-CGA-v2-dataset-name-preflight",
  "dataset_name": "NUDT-SIRST",
  "dataset_root": "/abs/path/to/datasets/NUDT-SIRST",
  "dataset_spec_sha256": "...",
  "train_list_sha256": "...",
  "test_list_sha256": "...",
  "hcval_list_sha256": "... or null",
  "train_test_overlap_count": 0,
  "duplicate_item_count": 0,
  "gate_pass": true
}
```

### 4.3 Output paths

```text
docs/internal/cga_v2/dataset_preflight/<DATASET_NAME>/summary.json
```

Examples:

```text
docs/internal/cga_v2/dataset_preflight/NUDT-SIRST/summary.json
docs/internal/cga_v2/dataset_preflight/NUAA-SIRST/summary.json
docs/internal/cga_v2/dataset_preflight/IRSTD-1K/summary.json
```

---

## 5. Dataset claim policy

Generic code can run all registry datasets. Paper claims cannot automatically upgrade just because a script can run a dataset.

Add:

```text
configs/dataset_claim_policy.yaml
```

Suggested policy:

```yaml
dataset_claim_policy:
  NUDT-SIRST:
    role: main_evidence_candidate
    eligible_for_main_table_after:
      - preflight_pass
      - seed42_paired_pass
      - seed43_44_paired_pass
    allowed_claim: "main fixed-seed paired evidence"

  IRSTD-1K:
    role: supervised_characterization_only
    eligible_after:
      - canonical_preflight_pass
      - target_level_audit
    allowed_claim: "single-dataset supervised train/test characterization"
    forbidden_claims:
      - external_validation
      - transfer_learning
      - multi_benchmark_validation

  NUAA-SIRST:
    role: blocked_until_integrity_pass
    blocked_reason: "Misc_111 image/mask size mismatch unless official clean data passes strict preflight."
    allowed_claim: null
    forbidden_claims:
      - include_in_paper_evidence_when_integrity_fails
      - validated_on_nuaa
```

The current route should not start with all three datasets. It should start with `NUDT-SIRST` only.

---

## 6. Code identity and delta statement

Because this is a new repo-grade implementation, add:

```text
docs/internal/cga_v2/repo_identity/code_delta_statement.md
```

Required content:

```markdown
# CGA-v2 Repo Identity Statement

This repository is a new repo-grade implementation of CGA-v2.

Historical OHCM-MSHNet-main results are internal references only.
They are not treated as current-repository paper evidence.

All paper evidence used for this repository must be regenerated here.

The intended method identity is:
- MSHNetCGA / CGA-v2 base
- training-time component-geometry regularization
- unchanged final mask inference path
- fixed threshold 0.5 for reported gates
```

Add checker:

```text
tools/official/check_cga_v2_repo_identity.py
tests/test_cga_v2_repo_identity.py
```

---

## 7. Generic one-dataset paired runner

Add:

```text
scripts/official/run_cga_v2_one_dataset_paired_seed.sh
scripts/official/run_cga_v2_one_model_seed.sh
```

Required call:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_one_dataset_paired_seed.sh
```

The paired runner must train/evaluate:

```text
MSHNetOHEM
MSHNetCGA
```

Fixed protocol:

```text
epoch = 400
threshold = 0.5
checkpoint = epoch400
```

### 7.1 Summary identity check

Every generated `summary_metrics.json` must include or be paired with a manifest containing:

```json
{
  "dataset_name": "NUDT-SIRST",
  "seed": 42,
  "model_name": "MSHNetCGA",
  "checkpoint_epoch": 400,
  "threshold": 0.5,
  "dataset_spec_sha256": "...",
  "train_list_sha256": "...",
  "test_list_sha256": "...",
  "hcval_list_sha256": "... or null"
}
```

A checker must reject summaries whose identity fields do not match the requested dataset / seed / threshold / epoch.

---

## 8. Execution order

### P0: repo identity + contract

```bash
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON}" -m py_compile \
  model/CGA_MSHNet.py \
  model/cga_aux.py \
  utils/cga_targets.py \
  loss.py \
  net.py \
  train.py \
  test.py \
  evaluate.py \
  metrics.py \
  tools/official/dataset_registry.py \
  tools/official/check_cga_v2_repo_identity.py

"${PYTHON}" -m pytest \
  tests/test_cga_v2_repo_identity.py \
  tests/test_cga_v2_model_contract.py \
  tests/test_cga_v2_loss_contract.py \
  tests/test_cga_v2_eval_contract.py -q

git diff --check
```

### P1: generic dataset registry + NUDT preflight

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

Stop if NUDT preflight fails. Do not train.

### P1.5: closest prior-art threat table

This can run in parallel with experiments because it does not depend on training results.

```bash
bash scripts/official/run_cga_v2_related_work_threat_table.sh
```

### P2: NUDT-SIRST seed42 paired run

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_one_dataset_paired_seed.sh
```

### P3: NUDT-SIRST seed43/44 paired run

Only after seed42 passes its predeclared reproduction decision rule.

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_nudt_multiseed_train_eval.sh
```

### P4a: seed42 ablation required for submission

Required ablations only:

```text
full
aux_off
no_center_peak
no_boundary_scale
```

Do not expand ablations unless paper reviewers explicitly require it later.

### P4b: multiseed ablation not required

Do not run multiseed ablation for the current AAAI rescue route.

### P5: failure / diagnostic pack

For this new repo, failure cases must be generated from this repo's artifacts.

IRSTD-1K has two options:

```text
Option A:
  New repo reruns IRSTD-1K supervised seed42 diagnostic, then includes it.

Option B:
  Omit IRSTD-1K from new-repo paper evidence.
  Historical internal diagnostic can be mentioned outside main result only.
```

Do not use old XDU788 / XDU850 diagnostic details as new-repo evidence unless this repo regenerates them.

### P6: closest baseline comparison pack

Generate:

```text
closest_baseline_threat_table.md
baseline_comparison_manifest.json
```

Separate:

```text
fair paired comparison:
  MSHNetOHEM vs MSHNetCGA in this repo

literature-only threat:
  MSHNet/SLS
  ISNet / shape-edge methods
  PConv + SD Loss
  other SOTA
```

### P7: claim linter + manuscript pack

Claim linter must block unsafe positive claims:

```text
state-of-the-art
external validation
transfer learning
validated on NUAA-SIRST and IRSTD-1K
all auxiliary heads are necessary
complete hard-clutter closure
```

But it must allow these strings inside:

```text
Do not claim
Rejected claims
Limitations
Reviewer risks
Unsafe wording
Forbidden claims
Failure analysis
Diagnostic-only evidence
```

---

## 9. Seed42 reproduction decision rule

Do not call this a “strong trend gate.” Call it:

```text
seed42 reproduction decision rule
```

State explicitly:

```text
The thresholds are pre-declared before seeing new-repo seed42 results.
```

Suggested predeclared sanity thresholds:

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

If the rule fails, do not immediately stop the method. First run an implementation audit.

---

## 10. P2 failure implementation audit

If seed42 fails, allow exactly one implementation audit:

```text
P2_FAIL_IMPL_AUDIT_ALLOWED
```

Allowed checks:

```text
dataset split
seed determinism
model registry
OHEM/CGA loss config
checkpoint epoch
eval threshold
train/eval mode
summary identity
dataset_spec_sha256
train/test/hcval list hashes
```

Only if the failure is confirmed not to be an implementation/configuration error, decide:

```text
STOP_NEW_REPO_CGA_V2_AT_NUDT_SEED42
```

Forbidden during audit:

```text
architecture tuning
loss tuning
threshold search
checkpoint search
seed search
post-hoc split change
```

---

## 11. Dataset-specific claim boundary

### NUDT-SIRST

```text
Role:
  main evidence dataset

Eligible for paper main table after:
  seed42 paired pass
  seed43/44 paired pass
```

### IRSTD-1K

```text
Role:
  supervised characterization only

Allowed only after:
  canonical preflight
  target-level audit

Not allowed:
  external validation claim
  transfer learning claim
  multi-benchmark validation claim
```

### NUAA-SIRST

```text
Role:
  blocked unless integrity preflight passes on official clean data

Current known issue:
  Misc_111 image/mask size mismatch

Not allowed:
  include in paper evidence if mismatch remains
  train on resized/deleted/imputed sample
```

---

## 12. Immediate minimal todo

```text
1. Add configs/datasets.yaml with item_format rules.
2. Add dataset_registry.py.
3. Add generic dataset preflight with split hashes.
4. Add code_delta_statement.md.
5. Add generic one-dataset paired runner.
6. Run P0 contract.
7. Run NUDT-SIRST preflight only.
8. Only then run NUDT-SIRST seed42.
```

Do not start NUAA or IRSTD-1K at the beginning of the new repo route.

