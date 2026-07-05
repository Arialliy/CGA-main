# CGA-v2 Paper Notes

## Current status

CGA-v2 seed42 under the audited from-zero paired protocol is a **valid negative design-weakness result**, not positive paper evidence.

The current implementation increases Pd, but worsens Precision, mIoU, and false alarms, especially on HC-Val.

## Safe title

> Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection

This title may remain as a historical/rescue-title note, but the current CGA-v2 results do not support a positive AAAI-main method claim.

## Current evidence statement

> Under the audited seed42 controlled paired protocol on NUDT-SIRST, CGA-v2 behaves as a recall/Pd booster rather than a false-alarm-suppressing regularizer. It is therefore not valid as a positive AAAI-main method claim in its current form.

## Key P2 result

```text
Final audit decision:
  P2_VALID_NEGATIVE_DESIGN_WEAKNESS

Allowed interpretation:
  CGA-v2 increases Pd but substantially worsens false alarms and precision.

Blocked interpretation:
  CGA-v2 improves hard-clutter robustness or reduces false alarms.
```

## Forbidden CGA-v2 claims

Do not claim:

```text
CGA-v2 improves Full split performance.
CGA-v2 improves hard-clutter robustness.
CGA-v2 reduces false alarms.
CGA-v2 is positive paper evidence.
CGA-v2 is ready for seed43/44 multiseed validation.
CGA-v2 is AAAI-main ready.
CGA-v2 is a multi-backbone plug-and-play method.
CGA-v2 universally improves IRSTD detectors.
```

## Current decision

```text
final_decision = P2_VALID_NEGATIVE_DESIGN_WEAKNESS
invalidating_steps = []
requires_seed42_rerun = false
can_run_seed43_44 = false
can_claim_positive_cga = false
```

## Next route

Any rescue must be treated as a new predeclared protocol, for example `CGA-v2.1`.

The current CGA-v2 P2 result must not be retroactively rescued by tuning weights, changing targets, changing thresholds, modifying HC-Val, or selectively rerunning seeds.
