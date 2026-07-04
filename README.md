# CGA

## Paper-Evidence CGA Protocol

Formal paper evidence must use the real CGA auxiliary heads only:

- `cga_center_logit`
- `cga_boundary_logit`
- `cga_scale_logit`
- `cga_peak_logit`

Fallback regularizers are forbidden for paper evidence. Any fallback/mock result
is smoke-test-only and cannot be used in paper tables, ablations, SOTA
comparisons, or AAAI main claims.

Detector adapters must explicitly declare logits, the selected CGA feature, and
feature metadata. Silent tensor inference, including selecting the first 4D
tensor from a raw model output, is forbidden.
