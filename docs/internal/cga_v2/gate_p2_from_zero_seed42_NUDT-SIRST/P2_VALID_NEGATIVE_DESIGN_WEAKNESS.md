# Gate P2 from-zero seed42: Valid negative design weakness

## Decision

```text
final_decision = P2_VALID_NEGATIVE_DESIGN_WEAKNESS
invalidating_steps = []
requires_seed42_rerun = false
can_run_seed43_44 = false
can_claim_positive_cga = false
```

## Scope

This file records the post-P2 implementation audit outcome for the existing seed42 from-zero paired experiment.

It does not modify model code, loss code, target generation, threshold, checkpoint, dataset split, or historical P2 result artifacts.

## Interpretation

The audited CGA-v2 implementation is not invalidated by implementation bugs under the v5 audit. Therefore, the failed seed42 gate should be treated as a valid negative/design-weakness result.

## Main observation

```text
Full split:
  Pd improves slightly, but mIoU, Precision, and FA worsen.

HC-Val:
  Pd reaches 1.0, but false alarms increase sharply and Precision/mIoU collapse.
```

## Blocked actions

```text
Do not run seed43/44 for CGA-v2.
Do not write positive CGA-v2 paper claims.
Do not tune the current CGA-v2 protocol post hoc.
Do not modify HC-Val after observing the result.
```

## Allowed next step

Open a new predeclared protocol, e.g. `CGA-v2.1`, if continuing the project.
