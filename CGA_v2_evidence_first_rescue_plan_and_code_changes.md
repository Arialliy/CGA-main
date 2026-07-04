# CGA-v2 Evidence-First Rescue Plan and Minimal Code Changes

## 0. Verdict

The opinion is correct.

For the current `CGA-main` state, the primary blocker is no longer metadata ownership or wrapper bookkeeping. The real blocker is missing paired paper evidence:

```text
MSHNetOHEM seed42 epoch400: existing result available.
MSHNetCGA seed42 epoch400: no complete current-repo epoch400 eval summary yet.
seed43/44 paired evidence: missing.
CGA vs OHEM current-repo gate result: missing.
```

Therefore, the next priority is not another large refactor. The next priority is to finish the current-repo MSHNetCGA seed42 epoch400 training/evaluation, compute the paired delta against MSHNetOHEM, and only then decide whether to run seed43/44.

This document supersedes tool-first or metadata-first plans for the current local Gate-1A-passed repo state.

---

## 1. Scope and applicability

This plan applies to the current `CGA-main` implementation after:

```text
Gate-1A fail-closed contract: passed
MSHNet adapter smoke: passed
MSHNet+CGA smoke: passed
fallback guard: passed
wrapper metadata ownership: fixed
P1 dataset preflight: passed
P1A HC-Val source audit: passed
```

It does **not** apply to an older public-main snapshot where `output_contract.py`, `cga_wrapper.py`, `registry.py`, or `mshnet_adapter.py` were missing.

---

## 2. Correct current priority

### Priority 1: complete MSHNetCGA seed42 epoch400

Run the current-repo CGA model to epoch400 under the frozen controlled protocol:

```text
model: MSHNetCGA / mshnet + CGAWrapper
regularizer_impl: center_boundary_scale_peak
fallback_regularizer_used: false
evidence_mode: paper
p1_preflight_passed: true
p1a_hcval_source_audit_passed: true
threshold: 0.5
seed: 42
epochs: 400
```

### Priority 2: evaluate Full and HC-Val

Use the same split, same threshold, and same evaluation path as the existing MSHNetOHEM seed42 result.

Required outputs:

```text
results/official/mshnet_cga/seed42/NUDT-SIRST/
  train_log.jsonl
  checkpoint epoch400
  full summary_metrics.json
  hcval summary_metrics.json
  evidence metadata / run config
```

### Priority 3: compute paired delta

Compare current-repo seed42:

```text
MSHNetOHEM seed42 epoch400
vs
MSHNetCGA seed42 epoch400
```

Primary gate:

```text
Full mIoU      >= +0.020
Full Precision >= +0.010
Full FA        <= baseline FA
```

HC-Val should be reported as a diagnostic / hard-clutter validation split. It may show a target-preservation vs false-alarm tradeoff, but the Full split should pass the main rescue gate before spending time on seed43/44.

### Priority 4: decide seed43/44

Only after seed42 passes the primary gate:

```text
Run seed43/44 paired evidence.
```

If seed42 fails the gate, stop the AAAI-main route and do not spend time on multiseed runs, ablations, or failure-pack storytelling.

### Priority 5: ablation and failure pack

Only after seed42 is positive:

```text
Run seed42 ablation.
Build target-preservation / hard-clutter failure pack.
Write mixed-mechanism caveat.
```

Do not do narrative-first failure analysis before positive evidence exists.

---

## 3. What is no longer the first blocker

| Item | Correct assessment | Action now |
|---|---|---|
| `model/cga_wrapper.py` metadata ownership | Correct direction, but no longer the main blocker if wrapper already does not write `paper_evidence_allowed=True`. | Do not refactor further unless grep reveals regression. |
| `regularizer_scope` / `regularizer_owner` | Useful audit enhancement. | Defer. |
| `train.py` P1/P1A hard fail | Conceptually correct, but hard-failing too early may break existing runners. | Keep metadata gate; do not block training if the run can safely mark `paper_evidence_allowed=false`. For paper-evidence runs, pass P1/P1A flags. |
| `legacy_model_factory` CLI | Useful for Gate-1B trend check. | Defer unless explicitly running Gate-1B. Not required for main paired evidence. |
| `eval_threshold` CLI | Good for traceability. Current default threshold is already 0.5. | Add metadata if cheap; not a blocker. |
| paired runner / manifest checker | Valuable, but checker schema must match current summary files. | Implement minimal delta checker after MSHNetCGA seed42 eval output exists. |
| failure pack | Required for paper narrative. | Do only after seed42 positive. |

---

## 4. Minimal code-change policy

The current recommendation is:

```text
Do not modify the model.
Do not change CGA heads.
Do not change target generation.
Do not add another backbone before seed42 paired evidence.
Do not spend first priority on manifest/checker refactors.
```

Allowed minimal code changes:

```text
1. Add or adjust a seed42 train/eval runner.
2. Add a small paired-delta checker that matches current summary_metrics.json schema.
3. Add missing metadata fields to eval summaries if they are absent.
4. Add threshold=0.5 to run config / summary metadata if not already logged.
```

Deferred code changes:

```text
1. legacy_model_factory CLI exposure.
2. broader manifest checker.
3. hard P1/P1A entry blocking.
4. multi-backbone registry extension.
5. DNANet / ALCNet / ACM adapters.
6. ablation automation.
```

---

## 5. Recommended seed42 CGA training command

Use the explicit adapter path, not legacy ambiguity:

```bash
cd /home/AAAI/CGA-main

CUDA_VISIBLE_DEVICES=1 python train.py \
  --backbone_name mshnet \
  --use_cga \
  --evidence_mode paper \
  --protocol controlled \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --seed 42 \
  --epochs 400 \
  --batch_size 8 \
  --patch_size 256 \
  --num_workers 4 \
  --mshnet_warm_epoch 5 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 40 \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --output_dir results/official
```

Expected train output directory:

```text
results/official/mshnet_cga/seed42/NUDT-SIRST/
```

Expected metadata in final epoch log/checkpoint:

```json
{
  "model": "mshnet_cga",
  "backbone": "mshnet",
  "use_cga": true,
  "regularizer_impl": "center_boundary_scale_peak",
  "fallback_regularizer_used": false,
  "evidence_mode": "paper",
  "p1_preflight_passed": true,
  "p1a_hcval_source_audit_passed": true,
  "paper_evidence_allowed": true,
  "protocol": "controlled",
  "seed": 42
}
```

If `paper_evidence_allowed` is false in a run intended as paper evidence, stop and fix the metadata/run flags before treating the result as evidence.

---

## 6. Evaluation requirement

Evaluate with:

```text
threshold = 0.5
same test split as OHEM
same HC-Val split as P1A-audited source
same metric implementation
same checkpoint epoch = 400
```

Do not sweep threshold for the main table.

If the evaluator already defaults to threshold 0.5, do not refactor it now. Just make sure the summary records:

```json
{
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared"
}
```

If the evaluator does not record threshold metadata, add only that logging field.

---

## 7. Minimal paired-delta checker

The checker should accept both current top-level metrics and possible future nested metrics.

New file suggestion:

```text
tools/official/compare_cga_vs_ohem_seed.py
```

Core metric reader:

```python
def get_metric(summary: dict, key: str, split: str | None = None) -> float:
    """Read current top-level metrics or future nested split metrics."""
    if key in summary:
        return float(summary[key])
    if split and isinstance(summary.get(split), dict) and key in summary[split]:
        return float(summary[split][key])
    # tolerate lower/upper variants
    for candidate in (key, key.lower(), key.upper()):
        if candidate in summary:
            return float(summary[candidate])
        if split and isinstance(summary.get(split), dict) and candidate in summary[split]:
            return float(summary[split][candidate])
    raise KeyError(f"Missing metric {key!r} in summary; available keys={list(summary.keys())}")
```

Gate logic:

```python
def compute_delta_gate(base: dict, cga: dict) -> dict:
    base_miou = get_metric(base, "mIoU")
    cga_miou = get_metric(cga, "mIoU")

    base_precision = get_metric(base, "Precision")
    cga_precision = get_metric(cga, "Precision")

    base_fa = get_metric(base, "FA")
    cga_fa = get_metric(cga, "FA")

    delta = {
        "delta_mIoU": cga_miou - base_miou,
        "delta_Precision": cga_precision - base_precision,
        "delta_FA": cga_fa - base_fa,
    }
    delta["seed42_primary_gate_pass"] = bool(
        delta["delta_mIoU"] >= 0.020
        and delta["delta_Precision"] >= 0.010
        and delta["delta_FA"] <= 0.0
    )
    return delta
```

Do not assume the summary schema is:

```json
{
  "Full": {
    "mIoU": 0.0
  }
}
```

if the current actual file is top-level:

```json
{
  "mIoU": 0.0,
  "Precision": 0.0,
  "FA": 0.0
}
```

---

## 8. Seed42 decision table

| Condition | Decision |
|---|---|
| Full mIoU < +0.020 | Stop AAAI-main route. |
| Full Precision < +0.010 | Stop AAAI-main route or downgrade claim. |
| Full FA increases | Stop AAAI-main route unless there is a very strong target-preservation story; still risky. |
| Full passes but HC-Val FA worsens | Continue cautiously; report as tradeoff, not universal hard-clutter closure. |
| Full passes and HC-Val also improves | Run seed43/44 immediately. |

Recommended seed42 decision:

```text
If seed42 fails the primary gate, do not run seed43/44.
If seed42 passes the primary gate, run seed43/44 paired evidence.
```

---

## 9. Three-seed requirement

Three-seed paired evidence means matched pairs:

```text
seed42: MSHNetOHEM vs MSHNetCGA
seed43: MSHNetOHEM vs MSHNetCGA
seed44: MSHNetOHEM vs MSHNetCGA
```

If only OHEM seed42 exists, then seed43/44 require both baseline and CGA runs.

Do not report:

```text
CGA seed42/43/44 vs OHEM seed42 only
```

as three-seed paired evidence.

---

## 10. Paper claim after seed42

Before seed42 CGA epoch400 summary exists:

```text
No paper evidence claim.
```

If seed42 passes but seed43/44 are not yet run:

```text
Preliminary paired seed42 evidence only.
```

If seed42/43/44 paired evidence is stable:

```text
CGA is a training-time component-geometry regularizer for MSHNet-style IRSTD that preserves the inference path and improves target preservation under a frozen controlled paired protocol.
```

Do not write:

```text
CGA is universally plug-and-play across IRSTD detectors.
CGA solves hard clutter.
Every CGA head is strictly necessary.
CGA is a new multi-backbone SOTA method.
```

---

## 11. Final recommended execution order

```text
R0. Freeze current code; do not modify model.
R1. Confirm existing MSHNetOHEM seed42 epoch400 summary path.
R2. Run MSHNetCGA seed42 epoch400.
R3. Evaluate Full and HC-Val at threshold 0.5.
R4. Compute paired delta: CGA seed42 vs OHEM seed42.
R5. If seed42 gate fails: stop AAAI-main route.
R6. If seed42 gate passes: run paired seed43/44.
R7. After positive evidence: run ablation and failure pack.
R8. Only then polish manifest/checker/metadata for release.
```

---

## 12. One-line conclusion

The current rescue route should be evidence-first, not metadata-first:

```text
Finish MSHNetCGA seed42 epoch400 -> evaluate Full/HC-Val -> compute paired delta -> decide seed43/44 -> then ablation/failure pack -> then release metadata polish.
```
