# CGA-v2.1 Predeclared Protocol Template

## Status

```text
protocol_name = CGA-v2.1
status = draft_not_active
inherits_from = CGA-v2 valid negative audit
```

CGA-v2.1 is a new protocol. It must not be presented as a continuation of the failed CGA-v2 P2 result.

## Why v2.1 is needed

CGA-v2 valid-negative audit found that the current component-geometry regularizer behaves as a recall/Pd booster and fails to suppress false alarms, especially on HC-Val.

## Allowed design directions

Choose and freeze before training:

```text
1. Add explicit hard-negative / background suppression term.
2. Add component-level false-positive penalty.
3. Rebalance center/boundary/scale/peak auxiliary weights.
4. Delay or weaken CGA ramp.
5. Modify component target geometry to reduce over-expansion.
6. Change selected adapter feature only after explicit source audit.
```

## Forbidden post-hoc rescue

```text
Do not change HC-Val after seeing CGA-v2 outputs.
Do not sweep threshold for the main table.
Do not selectively report seeds.
Do not reuse CGA-v2 P2 as positive evidence.
Do not call v2.1 results CGA-v2 results.
```

## Required frozen settings

```text
canonical_root = /home/ly/AAAI/CGA-main
dataset_dir = /home/ly/AAAI/CGA-main/datasets
dataset_name = NUDT-SIRST
seed42_first_gate = required
threshold = 0.5 fixed_predeclared
protocol = controlled
p1_preflight_passed = true
p1a_hcval_source_audit_passed = true
```

## v2.1 seed42 gate

Before seed43/44:

```text
Full delta mIoU      >= +0.020
Full delta Precision >= +0.010
Full delta FA_ppm    <= 0.0
HC-Val must not show catastrophic FA collapse
```

## Decision rule

```text
If v2.1 seed42 fails:
  stop v2.1 AAAI-main route.

If v2.1 seed42 passes:
  run paired seed43/44.

If three-seed paired evidence is stable:
  then run ablation and failure pack.
```
