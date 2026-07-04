# Closest Prior-Art Threat Table

This table separates current-repo fair paired evidence from literature-only threats.

## Fair Paired Comparison

| Method | Status | Use | Numbers policy |
|---|---|---|---|
| MSHNetOHEM | same-repo paired baseline | Fair paired comparison against MSHNetCGA under the frozen dataset, seed, epoch, and threshold protocol. | Use only current-repo regenerated metrics. |

## Literature-Only Threats

| Method | Status | Use | Numbers policy |
|---|---|---|---|
| MSHNet / SLS | closest architecture and supervision threat | Discuss as prior-art motivation and reviewer threat. | Do not present literature numbers as same-protocol fair comparison. |
| ISNet / shape-edge reconstruction | shape-aware IRSTD threat | Separate from the current-repo paired MSHNetOHEM comparison. | Literature-only unless reproduced in this repository under the frozen protocol. |
| PConv + SD Loss | loss/design threat | Use to frame component/shape regularization differences. | Literature-only unless reproduced in this repository under the frozen protocol. |
| Other IRSTD SOTA | broad benchmark threat | Mention only as external context, not as a claim of superiority. | No SOTA claim from cross-paper numbers. |

Do not present literature-only numbers as fair same-protocol comparisons.
