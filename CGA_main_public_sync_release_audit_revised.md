# CGA-main public-main sync / release audit checklist — revised scope

Date: 2026-07-04

Target repo:

```text
https://github.com/Arialliy/CGA-main
```

## 0. Scope boundary

This review targets the **public GitHub `main` branch snapshot / release-sync audit**.

It does **not** target the Docker 53f local fail-closed implementation that has already passed Gate-1A.

Do not use this document to overwrite the current Docker 53f local code if the local repository already has:

```text
model/output_contract.py
model/backbones/mshnet_adapter.py
model/cga_wrapper.py
model/registry.py
net.py with evidence_mode support
train.py with evidence_mode + mshnet_warm_epoch + mshnet_warm_flag support
```

This document is now only one of the following:

```text
1. public-main synchronization checklist
2. release-before-push audit checklist
3. stale-public-main patch review if a clone is still behind Gate-1A
```

It is **not** the next local experiment plan.

## 1. Correct local status assumption

For Docker 53f local `/home/AAAI/CGA-main`, treat the user-reported status as authoritative until contradicted by a fresh local audit:

```text
Gate-1A: PASS

Evidence:
  bash scripts/official/run_cga_v2_contract.sh --height 64 --width 64: 19 passed
  repo contract: gate_pass=true
  baseline 1 epoch smoke: pass, regularizer_impl=none
  MSHNet+CGA 1 epoch smoke: pass, regularizer_impl=center_boundary_scale_peak
  paper-mode fallback guard: failed closed as expected
  git diff --check: pass

Evidence role:
  implementation smoke / contract validation only

Paper evidence role:
  not yet allowed
```

Therefore, the next local gates are:

```text
Gate-1B: MSHNet legacy-CGA vs adapter-CGA semantic trend check
Gate-P1A: NUDT HC-Val frozen source audit
```

Do **not** rerun this public-main patch plan as though Gate-1A is missing locally.

## 2. Current naming convention to preserve

Use the implementation names already present in the fail-closed codebase:

```text
--evidence_mode smoke|paper
--mshnet_warm_epoch
mshnet_warm_flag
```

Do not reintroduce:

```text
--paper_mode
--mshnet_warm
```

The internal boolean equivalent is allowed only as a derived local variable:

```python
is_paper_evidence_mode = args.evidence_mode == "paper"
```

but the CLI, logs, scripts, and checkpoint metadata should stay on `evidence_mode`.

## 3. Remote/public sync audit

Use this only to check whether public `main` or a clone is synced to the local Gate-1A implementation.

```bash
cd /home/AAAI/CGA-main

git fetch origin

git ls-tree -r --name-only origin/main | grep -E \
  'model/output_contract.py|model/backbones/mshnet_adapter.py|model/cga_wrapper.py|model/registry.py|tests/test_cga_failclosed|tests/test_adapter_explicit|tests/test_multibackbone' || true

git show origin/main:net.py | grep -E 'evidence_mode|CGAWrapper|registry|allow_fallback_regularizer'

git show origin/main:train.py | grep -E -- '--evidence_mode|--mshnet_warm_epoch|mshnet_warm_flag|allow_fallback_regularizer'
```

Expected synced state:

```text
model/output_contract.py exists
model/backbones/mshnet_adapter.py exists
model/cga_wrapper.py exists
model/registry.py exists
net.py uses evidence_mode and fail-closed fallback guard
train.py uses --evidence_mode, --mshnet_warm_epoch, and mshnet_warm_flag
DNANet / ACM / ALCNet / ISNet remain unregistered or fail-closed until explicit adapter audits pass
```

If this check passes, this document should be treated as an audit checklist, not a patch-to-apply plan.

## 4. `paper_evidence_allowed` policy

Do not hard-code `paper_evidence_allowed = False` globally.

Do not let the model wrapper unconditionally decide paper evidence eligibility either, because the model cannot know whether the dataset and split gates passed.

Use this policy instead:

```python
def compute_paper_evidence_allowed(
    *,
    evidence_mode: str,
    p1_preflight_passed: bool,
    fallback_regularizer_used: bool,
) -> bool:
    if evidence_mode == "smoke":
        return False
    if evidence_mode != "paper":
        raise ValueError(f"Unknown evidence_mode={evidence_mode!r}")
    return bool(p1_preflight_passed and not fallback_regularizer_used)
```

Equivalent compact expression:

```python
paper_evidence_allowed = bool(
    args.evidence_mode == "paper"
    and p1_preflight_passed
    and not fallback_regularizer_used
)
```

Required runner behavior:

```text
smoke runner:
  evidence_mode=smoke
  paper_evidence_allowed=false

paper-evidence runner:
  evidence_mode=paper
  paper_evidence_allowed=true only after P1/P1A pass and fallback_regularizer_used=false
```

Recommended metadata fields:

```json
{
  "evidence_mode": "smoke_or_paper",
  "protocol": "controlled",
  "p1_preflight_passed": false,
  "p1a_hcval_source_audit_passed": false,
  "fallback_regularizer_used": false,
  "regularizer_impl": "center_boundary_scale_peak",
  "paper_evidence_allowed": false
}
```

For implementation-level smoke outputs, keep:

```json
{
  "evidence_mode": "smoke",
  "paper_evidence_allowed": false
}
```

For official paper-evidence outputs, require:

```json
{
  "evidence_mode": "paper",
  "protocol": "controlled",
  "p1_preflight_passed": true,
  "p1a_hcval_source_audit_passed": true,
  "fallback_regularizer_used": false,
  "paper_evidence_allowed": true
}
```

## 5. Recommended small code adjustment for metadata ownership

If `model/cga_wrapper.py` currently writes:

```python
"paper_evidence_allowed": True
```

move that decision out of the model wrapper. The wrapper should only say what it knows:

```python
output.setdefault("regularizer_meta", {})
output["regularizer_meta"].update(
    {
        "use_cga": True,
        "regularizer_impl": "center_boundary_scale_peak",
        "fallback_regularizer_used": False,
    }
)
```

Then `train.py`, the official runner, or the evidence manifest writer should compute:

```python
fallback_regularizer_used = bool(
    output.get("fallback_regularizer_used", False)
    or output.get("regularizer_meta", {}).get("fallback_regularizer_used", False)
)

paper_evidence_allowed = bool(
    args.evidence_mode == "paper"
    and p1_preflight_passed
    and not fallback_regularizer_used
)
```

This prevents a model-level field from accidentally upgrading smoke or pre-P1 runs into paper evidence.

## 6. Current command style

### 6.1 Contract tests

```bash
cd /home/AAAI/CGA-main

CUDA_VISIBLE_DEVICES=1 bash scripts/official/run_cga_v2_contract.sh \
  --height 64 \
  --width 64
```

Expected local Gate-1A result:

```text
19 passed
```

### 6.2 Baseline smoke

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_dir datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --model_name MSHNet \
  --evidence_mode smoke \
  --epochs 1 \
  --batch_size 2 \
  --patch_size 64 \
  --mshnet_warm_epoch 1 \
  --output_dir results/failclosed_smoke
```

Expected metadata:

```json
{
  "regularizer_impl": "none",
  "evidence_mode": "smoke",
  "paper_evidence_allowed": false
}
```

### 6.3 MSHNet+CGA smoke

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_dir datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --model_name MSHNetCGA \
  --use_cga \
  --evidence_mode smoke \
  --epochs 1 \
  --batch_size 2 \
  --patch_size 64 \
  --mshnet_warm_epoch 1 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 1 \
  --output_dir results/failclosed_smoke
```

Expected metadata:

```json
{
  "regularizer_impl": "center_boundary_scale_peak",
  "fallback_regularizer_used": false,
  "evidence_mode": "smoke",
  "paper_evidence_allowed": false
}
```

### 6.4 Paper mode fallback guard

This should fail closed:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_dir datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --use_cga \
  --evidence_mode paper \
  --allow_fallback_regularizer \
  --epochs 1 \
  --batch_size 2 \
  --patch_size 64
```

Expected:

```text
RuntimeError: Fallback regularizer is forbidden for paper evidence.
```

## 7. Forward-call convention

Training should pass the warm flag with the current implementation name:

```python
forward_kwargs = {}
if backbone_name == "mshnet":
    forward_kwargs["mshnet_warm_flag"] = epoch <= args.mshnet_warm_epoch

output = model(img, **forward_kwargs)
```

Do not switch back to:

```python
output = model(img, mshnet_warm=(epoch <= args.mshnet_warm_epoch))
```

unless the actual adapter has been renamed accordingly.

## 8. Actual next local gates

### Gate-1B: MSHNet legacy-CGA vs adapter-CGA semantic trend check

Purpose:

```text
Ensure the new adapter + CGAWrapper path is semantically consistent with legacy MSHNetCGA before migrating CGA to DNANet / ALCNet / ACM.
```

Required checks:

```text
same seed
same dataset smoke subset
same main loss
same CGA four heads
regularizer_impl=center_boundary_scale_peak
fallback_regularizer_used=false
loss_center/loss_boundary/loss_scale/loss_peak present and finite
train curve trend not obviously broken
```

Suggested outputs:

```text
docs/internal/cga_v2/gates/GATE1B_MSHNET_LEGACY_VS_ADAPTER_TREND.md
docs/internal/cga_v2/gates/gate1b_mshnet_legacy_vs_adapter_summary.json
results/gate1b_mshnet_legacy_vs_adapter/
```

### Gate-P1A: NUDT HC-Val frozen source audit

Purpose:

```text
Recover or register a frozen hcval_NUDT-SIRST.txt source before any paper-evidence training.
```

Required checks:

```text
hcval_NUDT-SIRST.txt exists
source was frozen before new-repo seed42 results
no post-hoc selection from current model errors
no copying full test list without documented pre-existing policy
all listed images/masks exist and pass strict preflight
split hash and source metadata are written
```

Only after Gate-P1A and P1 pass should seed42 paper-evidence training start.

## 9. What not to do now

Do not do any of the following from the current local state:

```text
overwrite local Gate-1A files with an older public-main patch plan
run DNANet+CGA paper training before DNANet adapter source audit
run ALCNet/ACM+CGA paper training before adapter source audit
start seed42 NUDT Full + HC-Val paper-evidence training before Gate-P1A/P1 pass
label results/failclosed_smoke as paper evidence
claim universal plug-and-play effectiveness
```

## 10. Safe paper-claim boundary

After Gate-1A only:

```text
We have a fail-closed, explicit-output CGA wrapper validated on MSHNet implementation smoke tests.
```

After Gate-1B only:

```text
The adapter-CGA path is semantically consistent with the legacy MSHNetCGA path in smoke/trend validation.
```

After Gate-P1A + P1 + controlled paired seed42 on MSHNet:

```text
CGA improves the audited MSHNet host under controlled paired protocol.
```

After MSHNet + DNANet + ALCNet/ACM controlled paired evidence is positive:

```text
CGA improves multiple host detectors under controlled paired protocols.
```

Do not write:

```text
CGA is universally effective for all IRSTD models.
CGA is fully plug-and-play across arbitrary backbones.
```

## 11. Final decision

This document supersedes the earlier public-main patch review as a **scope-corrected sync/release checklist**.

Current local next step is not Gate-1A. It is:

```text
1. Gate-1B: MSHNet legacy-CGA vs adapter-CGA trend check
2. Gate-P1A: NUDT HC-Val frozen source audit
3. P1 dataset preflight rerun
4. seed42 paper-evidence training only after P1A/P1 pass
```
