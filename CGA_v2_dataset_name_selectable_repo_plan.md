# CGA-v2 Dataset-Name-Selectable Repository Plan

## 0. Correction

The repository should **not** become a hard-coded “three public datasets” fork.

It should follow a BasicIRSTD-style design:

```text
One code path.
One dataset registry.
One dataset_name argument.
The same train / test / evaluate / preflight / report scripts work for any registered dataset.
```

The three immediate target datasets are:

```text
NUDT-SIRST
NUAA-SIRST
IRSTD-1K
```

But they should be selected as:

```bash
--dataset_name NUDT-SIRST
--dataset_name NUAA-SIRST
--dataset_name IRSTD-1K
```

not by creating three separate code branches.

This also matches the current execution fact: the new overlay repository has no experiment artifacts, so it must regenerate its own evidence, but that regeneration should be done through a generic `dataset_name` interface rather than through dataset-specific scripts. The earlier plan already notes that the new overlay folder cannot directly use historical artifacts as current-repo results and must generate its own artifacts.  

---

## 1. Core Design Rule

### Wrong design

```text
run_cga_v2_nudt_seed42_train_eval.sh
run_cga_v2_nuaa_seed42_train_eval.sh
run_cga_v2_irstd_seed42_train_eval.sh
check_nudt_preflight.py
check_nuaa_preflight.py
check_irstd_preflight.py
```

This becomes brittle and slow. Every dataset creates new scripts, new checks, new output conventions, and new failure modes.

### Correct design

```text
DATASET_NAME=NUDT-SIRST  bash scripts/official/run_cga_v2_one_dataset_one_seed.sh
DATASET_NAME=NUAA-SIRST  bash scripts/official/run_cga_v2_one_dataset_one_seed.sh
DATASET_NAME=IRSTD-1K    bash scripts/official/run_cga_v2_one_dataset_one_seed.sh
```

All scripts should read a shared dataset registry:

```text
configs/datasets.yaml
```

and all result summaries must record:

```text
dataset_name
dataset_dir
resolved_dataset_root
train_list
test_list
split protocol
mask policy
dataset_spec_sha256
```

---

## 2. Repository-Level Goal

The immediate goal is:

```text
Make the new CGA-v2 repo a dataset-name-selectable IRSTD training/evaluation repository.
```

Not:

```text
Make a one-off NUDT-only experiment folder.
Make a hard-coded NUAA/NUDT/IRSTD-1K benchmark fork.
Keep modifying model structure.
```

The method remains fixed:

```text
MSHNetCGA / CGA-v2 base
training-time component-geometry auxiliary supervision
single-forward final-logit inference
threshold = 0.5
checkpoint = epoch400
```

The safe paper positioning remains:

```text
Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection
```

---

## 3. Dataset Registry

### 3.1 New file

```text
configs/datasets.yaml
```

### 3.2 Schema

```yaml
datasets:
  NUDT-SIRST:
    dataset_name: NUDT-SIRST
    root_mode: dataset_dir_plus_name
    image_dir: images
    mask_dir: masks
    train_list: img_idx/train_NUDT-SIRST.txt
    test_list: img_idx/test_NUDT-SIRST.txt
    hcval_list: img_idx/hcval_NUDT-SIRST.txt
    full_eval_split: test
    has_hcval: true
    mask_policy:
      type: binary
      allowed_values: [0, 255]
    expected_counts:
      train: null
      test: null
      hcval: null

  NUAA-SIRST:
    dataset_name: NUAA-SIRST
    root_mode: dataset_dir_plus_name
    image_dir: images
    mask_dir: masks
    train_list: img_idx/train_NUAA-SIRST.txt
    test_list: img_idx/test_NUAA-SIRST.txt
    has_hcval: false
    mask_policy:
      type: binary
      allowed_values: [0, 255]
    integrity_policy:
      no_delete: true
      no_resize: true
      no_impute: true
      canonical_view_required_if_raw_fails: true
    expected_counts:
      train: 256
      test: 86

  IRSTD-1K:
    dataset_name: IRSTD-1K
    root_mode: dataset_dir_plus_name
    image_dir: images
    mask_dir: masks
    train_list: img_idx/train_IRSTD-1K.txt
    test_list: img_idx/test_IRSTD-1K.txt
    has_hcval: false
    mask_policy:
      type: canonical_binary
      rule_source: frozen_dataset_preflight
      allowed_values: [0, 255]
    expected_counts:
      train: 800
      test: 201
```

### 3.3 `DATASET_DIR` semantics

Keep BasicIRSTD-style parent-root semantics:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets
DATASET_NAME=NUDT-SIRST
```

The resolved root is:

```text
${DATASET_DIR}/${DATASET_NAME}
```

For canonical datasets:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets_canonical/F2R
DATASET_NAME=IRSTD-1K
```

The resolved root is:

```text
/home/ly/AAAI/OHCM-MSHNet-main/datasets_canonical/F2R/IRSTD-1K
```

Do **not** pass:

```bash
DATASET_DIR=/home/.../datasets_canonical/F2R/IRSTD-1K
DATASET_NAME=IRSTD-1K
```

because that resolves to:

```text
.../IRSTD-1K/IRSTD-1K
```

---

## 4. Dataset Registry Loader

### 4.1 New file

```text
tools/official/dataset_registry.py
```

Use `tools/official/` rather than `utils/` so lightweight preflight and checker tools do not accidentally trigger training-time imports.

### 4.2 Code skeleton

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    dataset_dir: Path
    resolved_root: Path
    image_dir: Path
    mask_dir: Path
    train_list: Path
    test_list: Path
    hcval_list: Optional[Path]
    has_hcval: bool
    raw: Dict[str, Any]
    spec_sha256: str


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset_spec(
    dataset_name: str,
    dataset_dir: str | Path,
    registry_path: str | Path = "configs/datasets.yaml",
) -> DatasetSpec:
    registry_path = Path(registry_path)
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    with registry_path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    datasets = registry.get("datasets", {})
    if dataset_name not in datasets:
        known = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset_name={dataset_name!r}. Known datasets: {known}")

    raw = dict(datasets[dataset_name])
    root_mode = raw.get("root_mode", "dataset_dir_plus_name")

    if root_mode == "dataset_dir_plus_name":
        resolved_root = dataset_dir / dataset_name
    elif root_mode == "dataset_dir_is_root":
        resolved_root = dataset_dir
    else:
        raise ValueError(f"Unsupported root_mode={root_mode!r}")

    image_dir = resolved_root / raw.get("image_dir", "images")
    mask_dir = resolved_root / raw.get("mask_dir", "masks")
    train_list = resolved_root / raw["train_list"]
    test_list = resolved_root / raw["test_list"]

    hcval_value = raw.get("hcval_list")
    hcval_list = resolved_root / hcval_value if hcval_value else None

    spec_text = yaml.safe_dump(raw, sort_keys=True, allow_unicode=True)

    return DatasetSpec(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        resolved_root=resolved_root,
        image_dir=image_dir,
        mask_dir=mask_dir,
        train_list=train_list,
        test_list=test_list,
        hcval_list=hcval_list,
        has_hcval=bool(raw.get("has_hcval", False)),
        raw=raw,
        spec_sha256=_sha256_text(spec_text),
    )
```

---

## 5. Dataset-Aware `dataset.py`

### 5.1 Principle

`dataset.py` should not contain separate hard-coded dataset implementations unless absolutely necessary.

It should accept:

```text
dataset_dir
dataset_name
split
```

and resolve paths through `configs/datasets.yaml`.

### 5.2 Minimal interface

```python
class IRSTDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str,
        split: str,
        registry_path: str = "configs/datasets.yaml",
        transform=None,
    ):
        self.spec = load_dataset_spec(dataset_name, dataset_dir, registry_path)
        self.split = split
        self.transform = transform

        if split == "train":
            list_path = self.spec.train_list
        elif split == "test":
            list_path = self.spec.test_list
        elif split == "hcval":
            if not self.spec.has_hcval or self.spec.hcval_list is None:
                raise ValueError(f"Dataset {dataset_name} has no hcval split")
            list_path = self.spec.hcval_list
        else:
            raise ValueError(f"Unsupported split={split!r}")

        self.items = read_split_items(list_path)
```

Do not create separate classes like:

```text
NUDTDataset
NUAADataset
IRSTD1KDataset
```

unless some dataset has a truly different file format. Even then, use a registry field:

```yaml
loader: default_image_mask_list
```

not a hard-coded branch in scripts.

---

## 6. CLI Contract

All main entry points must accept the same dataset arguments:

```text
--dataset_name
--dataset_dir
--dataset_registry
```

### 6.1 `train.py`

Add / standardize:

```python
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--dataset_registry", default="configs/datasets.yaml")
```

Summary identity must include:

```json
{
  "dataset_name": "NUDT-SIRST",
  "dataset_dir": "/.../datasets",
  "resolved_dataset_root": "/.../datasets/NUDT-SIRST",
  "dataset_spec_sha256": "...",
  "seed": 42,
  "model_name": "MSHNetCGA",
  "checkpoint_epoch": 400
}
```

### 6.2 `test.py` / `evaluate.py`

Add / standardize the same fields:

```python
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--dataset_dir", required=True)
parser.add_argument("--dataset_registry", default="configs/datasets.yaml")
parser.add_argument("--split", default="test", choices=["train", "test", "hcval"])
parser.add_argument("--threshold", type=float, default=0.5)
```

Evaluation summary identity must include:

```json
{
  "dataset_name": "IRSTD-1K",
  "split": "test",
  "threshold": 0.5,
  "checkpoint_epoch": 400,
  "model_name": "MSHNetCGA",
  "dataset_spec_sha256": "..."
}
```

### 6.3 `net.py`

Keep model selection independent from dataset selection:

```python
model = build_model(model_name=args.model_name)
```

Do not put dataset-specific model logic in `net.py`.

---

## 7. Official Scripts

### 7.1 One generic seed script

Create:

```text
scripts/official/run_cga_v2_one_dataset_one_seed.sh
```

Usage:

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
MODEL_NAME=MSHNetCGA \
bash scripts/official/run_cga_v2_one_dataset_one_seed.sh
```

The same script should work for:

```bash
DATASET_NAME=NUDT-SIRST
DATASET_NAME=NUAA-SIRST
DATASET_NAME=IRSTD-1K
```

### 7.2 Script skeleton

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

DATASET_DIR=${DATASET_DIR:?DATASET_DIR is required}
DATASET_NAME=${DATASET_NAME:?DATASET_NAME is required}
SEED=${SEED:?SEED is required}
MODEL_NAME=${MODEL_NAME:-MSHNetCGA}
DATASET_REGISTRY=${DATASET_REGISTRY:-configs/datasets.yaml}
THRESHOLD=${THRESHOLD:-0.5}
EPOCH=${EPOCH:-400}

"${PYTHON}" -m tools.official.check_cga_v2_dataset_preflight \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_registry "${DATASET_REGISTRY}"

"${PYTHON}" train.py \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_registry "${DATASET_REGISTRY}" \
  --model_name "${MODEL_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCH}"

"${PYTHON}" test.py \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_registry "${DATASET_REGISTRY}" \
  --model_name "${MODEL_NAME}" \
  --seed "${SEED}" \
  --checkpoint_epoch "${EPOCH}" \
  --split test \
  --threshold "${THRESHOLD}"
```

### 7.3 Baseline and candidate wrapper

Create:

```text
scripts/official/run_cga_v2_one_dataset_paired_seed.sh
```

This runs both:

```text
MSHNetOHEM
MSHNetCGA
```

for the same:

```text
dataset_name
seed
epoch
threshold
split
```

---

## 8. Preflight Tool

### 8.1 New file

```text
tools/official/check_cga_v2_dataset_preflight.py
```

### 8.2 Required checks

For the selected `dataset_name`:

```text
registry entry exists
resolved root exists
train/test list exists
image/mask file exists for every item
image/mask readable
image/mask size match
mask values match frozen policy
no path traversal
no duplicate list items
no train/test overlap
```

### 8.3 Output path

```text
docs/internal/cga_v2/dataset_preflight/{DATASET_NAME}/summary.json
```

### 8.4 Summary identity

```json
{
  "gate": "Gate-CGA-v2-dataset-name-preflight",
  "dataset_name": "NUAA-SIRST",
  "dataset_dir": "/.../datasets",
  "resolved_dataset_root": "/.../datasets/NUAA-SIRST",
  "dataset_spec_sha256": "...",
  "gate_pass": false,
  "decision": "STOP_DATASET_DUE_TO_INTEGRITY_FAILURE"
}
```

---

## 9. Dataset-Specific Gates Without Dataset-Specific Code

The code path is generic; the gate behavior is dataset-specific through registry metadata.

### 9.1 NUDT-SIRST

If `has_hcval=true`, evaluation includes:

```text
test / Full
hcval
```

### 9.2 NUAA-SIRST

If `has_hcval=false`, evaluation includes:

```text
test only
```

### 9.3 IRSTD-1K

If `mask_policy.type=canonical_binary`, preflight must verify that the canonical view is used and that `DATASET_DIR / DATASET_NAME` matches the manifest root.

---

## 10. Result Identity Checker

### 10.1 New file

```text
tools/official/check_cga_v2_one_dataset_seed_result.py
```

### 10.2 Must verify

```text
dataset_name
seed
threshold = 0.5
checkpoint_epoch = 400
baseline = MSHNetOHEM
candidate = MSHNetCGA
split
resolved_dataset_root
dataset_spec_sha256
model_name in summary
```

Do not allow a summary from the wrong dataset to pass simply because metrics look good.

---

## 11. Output Directory Layout

Use consistent dataset-name paths:

```text
results/official/{MODEL_NAME}/seed{SEED}/{DATASET_NAME}/{MODEL_NAME}_400.pth.tar

docs/internal/cga_v2/runs/{DATASET_NAME}/seed{SEED}/{MODEL_NAME}/train_summary.json

docs/internal/cga_v2/runs/{DATASET_NAME}/seed{SEED}/eval_{SPLIT}_{MODEL_NAME}/summary_metrics.json
```

Examples:

```text
results/official/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar
results/official/MSHNetCGA/seed42/NUAA-SIRST/MSHNetCGA_400.pth.tar
results/official/MSHNetCGA/seed42/IRSTD-1K/MSHNetCGA_400.pth.tar
```

---

## 12. Execution Order

### Step 0: Contract only

```bash
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON}" -m py_compile \
  tools/official/dataset_registry.py \
  tools/official/check_cga_v2_dataset_preflight.py \
  tools/official/check_cga_v2_one_dataset_seed_result.py \
  dataset.py train.py test.py evaluate.py metrics.py net.py loss.py

"${PYTHON}" -m pytest \
  tests/test_dataset_registry.py \
  tests/test_cga_v2_dataset_preflight.py \
  tests/test_cga_v2_one_dataset_seed_result.py \
  tests/test_cga_v2_dataset_name_cli_contract.py -q

git diff --check
```

### Step 1: Preflight selected dataset

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

Then:

```bash
DATASET_NAME=NUAA-SIRST
DATASET_NAME=IRSTD-1K
```

using the same script.

### Step 2: Seed42 paired run for selected dataset

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_one_dataset_paired_seed.sh
```

### Step 3: Repeat with same script for other datasets only after preflight pass

```bash
DATASET_NAME=NUAA-SIRST
DATASET_NAME=IRSTD-1K
```

No new code should be written just because the dataset changes.

---

## 13. What To Remove From The Previous Plan

Remove wording like:

```text
three-public-dataset version
NUDT script
NUAA script
IRSTD script
public3-specific runner
```

Replace with:

```text
dataset_name-selectable version
dataset registry
generic one-dataset-one-seed runner
generic preflight
generic paired result checker
```

---

## 14. Code Modification Checklist

### Add

```text
configs/datasets.yaml

tools/official/dataset_registry.py
tools/official/check_cga_v2_dataset_preflight.py
tools/official/check_cga_v2_one_dataset_seed_result.py
tools/official/write_cga_v2_dataset_result_table.py

scripts/official/run_cga_v2_dataset_preflight.sh
scripts/official/run_cga_v2_one_dataset_one_seed.sh
scripts/official/run_cga_v2_one_dataset_paired_seed.sh
scripts/official/run_cga_v2_dataset_name_multiseed.sh

tests/test_dataset_registry.py
tests/test_cga_v2_dataset_preflight.py
tests/test_cga_v2_one_dataset_seed_result.py
tests/test_cga_v2_dataset_name_cli_contract.py
```

### Modify

```text
dataset.py
train.py
test.py
evaluate.py
metrics.py
net.py only if model_name registry is incomplete
README.md
```

### Do not modify

```text
model architecture
CGA auxiliary head design
loss weights
threshold
checkpoint epoch
seed set
split definitions
```

---

## 15. Final Direction

The final execution direction is:

```text
Not a hard-coded three-dataset branch.
A dataset-name-selectable repository.
```

The same command shape should work for all datasets:

```bash
DATASET_DIR=/path/to/dataset_parent \
DATASET_NAME=<NUDT-SIRST|NUAA-SIRST|IRSTD-1K> \
SEED=42 \
bash scripts/official/run_cga_v2_one_dataset_paired_seed.sh
```

This makes the repository closer to BasicIRSTD-style usage and much more suitable for release, reproduction, and AAAI supplementary code.
