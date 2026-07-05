# CGA-v2.1 Required Fixes Before Seed42: False-Alarm-Controlled Protocol and Code Changes

**Canonical repo root:** `/home/ly/AAAI/CGA-main`  
**Status:** pre-seed42 protocol design for a new method version  
**Do not treat this as CGA-v2 post-hoc tuning.** CGA-v2.1 must be a new predeclared protocol with its own lock file, runner, summarizer, and gate.

---

## 0. Verdict

The critique is correct.

CGA-v2.1 is a reasonable rescue direction only if it is explicitly redefined as:

```text
False-alarm-controlled component-geometry regularization
```

rather than the failed CGA-v2 claim:

```text
target-preserving CGA improves hard-clutter behavior
```

But the current v2.1 plan is not yet executable as valid evidence. Before any seed42 from-zero paired training, the plan must add:

```text
1. A complete train -> test -> hcval -> summarize -> gate runner.
2. A v2.1-specific summarizer with Full and HC-Val FA/Precision gates.
3. A real protocol-lock checker that compares CLI/runtime args to the lock.
4. A safe-background loss based on logits BCE, not only sigmoid-prob power.
5. A detached-scale ratio cap so gradients are not killed when the cap is active.
6. Explicit ratio-cap diagnostics.
7. Strict/audited checkpoint loading for v2.1 paper-mode evaluation.
8. No automatic hyperparameter override in train.py.
```

Only after those are committed and contract/smoke checks pass should seed42 from-zero paired training start.

---

## 1. Why CGA-v2.1 is a new protocol

CGA-v2 failed the seed42 gate and the v5 implementation audit found no invalidating implementation error. Therefore CGA-v2 must remain frozen as a valid negative / design weakness result.

CGA-v2.1 must not reuse CGA-v2 positive language, seed43/44 route, or gate summary. It must get its own namespace:

```text
docs/internal/cga_v2_1/
results/official_cga_v21/
tools/official/summarize_cga_v21_one_seed.py
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

Recommended method name:

```text
CGA-v2.1: False-Alarm-Controlled Component-Geometry Regularization
```

Safe claim before results:

```text
CGA-v2.1 is a predeclared attempt to control the false-alarm collapse observed in CGA-v2 by adding safe-background hard-negative suppression and bounded auxiliary regularization.
```

Forbidden claim before seed42 passes:

```text
CGA-v2.1 improves IRSTD.
CGA-v2.1 solves hard clutter.
CGA-v2.1 is AAAI-ready.
```

---

## 2. Protocol lock: must be real, not just `test -f`

### 2.1 New file

Create:

```text
docs/internal/cga_v2_1/protocol_lock.json
```

Initial lock template:

```json
{
  "protocol_name": "CGA-v2.1 false-alarm-controlled component-geometry regularization",
  "protocol_version": "cga_v2_1_predeclared_001",
  "root": "/home/ly/AAAI/CGA-main",
  "dataset_dir": "/home/ly/AAAI/CGA-main/datasets",
  "dataset_name": "NUDT-SIRST",
  "baseline": "MSHNetOHEM",
  "candidate": "MSHNetCGA21",
  "backbone_name": "mshnet",
  "seed": 42,
  "epochs": 400,
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "evidence_mode": "paper",
  "protocol": "controlled",
  "cga_variant": "v2_1",
  "mshnet_warm_epoch": 5,
  "cga_start_epoch": 20,
  "cga_ramp_epochs": 80,
  "lambda_center": 0.02,
  "lambda_boundary": 0.01,
  "lambda_scale": 0.01,
  "lambda_peak": 0.02,
  "lambda_safe_bg": 0.05,
  "safe_bg_dilate_px": 5,
  "safe_bg_topk_ratio": 0.005,
  "aux_ratio_cap": 0.15,
  "ohem_ratio": 0.01,
  "full_delta_mIoU_min": 0.020,
  "full_delta_precision_min": 0.010,
  "full_delta_pd_min": -0.001,
  "full_delta_fa_ppm_max": 0.0,
  "hcval_delta_mIoU_min": -0.020,
  "hcval_delta_precision_min": -0.020,
  "hcval_delta_pd_min": -0.001,
  "hcval_delta_fa_ppm_max": 50.0,
  "strict_eval_load": true,
  "no_threshold_sweep": true,
  "do_not_reuse_cga_v2_checkpoints": true,
  "predeclared_before_seed42": true
}
```

These numbers are a **first locked protocol**, not defaults to be silently overwritten by code. If you change them, update the lock before running smoke/seed42 and treat that as a new lock revision.

### 2.2 New checker

Create:

```text
tools/official/check_cga_v21_protocol_lock.py
```

Required behavior:

```text
Input:
  --lock docs/internal/cga_v2_1/protocol_lock.json
  --args-json docs/internal/cga_v2_1/runs/seed42_runtime_args.json

Checks:
  dataset_dir
  dataset_name
  baseline
  candidate
  seed
  epochs
  threshold
  cga_variant
  cga_start_epoch
  cga_ramp_epochs
  lambda_center
  lambda_boundary
  lambda_scale
  lambda_peak
  lambda_safe_bg
  safe_bg_dilate_px
  safe_bg_topk_ratio
  aux_ratio_cap
  ohem_ratio
  strict_eval_load

Output:
  pass: true/false
  mismatches: []
  invalidates_run: true/false
```

Skeleton:

```python
# tools/official/check_cga_v21_protocol_lock.py
from __future__ import annotations
import argparse, json
from pathlib import Path

REQUIRED_KEYS = [
    "dataset_dir", "dataset_name", "baseline", "candidate", "seed", "epochs",
    "threshold", "evidence_mode", "protocol", "cga_variant", "mshnet_warm_epoch",
    "cga_start_epoch", "cga_ramp_epochs", "lambda_center", "lambda_boundary",
    "lambda_scale", "lambda_peak", "lambda_safe_bg", "safe_bg_dilate_px",
    "safe_bg_topk_ratio", "aux_ratio_cap", "ohem_ratio", "strict_eval_load",
]

FLOAT_KEYS = {
    "threshold", "lambda_center", "lambda_boundary", "lambda_scale", "lambda_peak",
    "lambda_safe_bg", "safe_bg_topk_ratio", "aux_ratio_cap", "ohem_ratio",
}


def same_value(key: str, a, b) -> bool:
    if key in FLOAT_KEYS:
        return abs(float(a) - float(b)) <= 1e-12
    return a == b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lock", required=True)
    p.add_argument("--args-json", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.args_json).read_text(encoding="utf-8"))

    mismatches = []
    missing = []
    for key in REQUIRED_KEYS:
        if key not in lock or key not in runtime:
            missing.append(key)
            continue
        if not same_value(key, lock[key], runtime[key]):
            mismatches.append({"key": key, "lock": lock[key], "runtime": runtime[key]})

    ok = not missing and not mismatches
    out = {
        "gate": "CGA-v2.1 protocol lock check",
        "pass": ok,
        "invalidates_run": not ok,
        "missing": missing,
        "mismatches": mismatches,
        "lock": args.lock,
        "args_json": args.args_json,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
```

---

## 3. Loss design: replace weak probability penalty with logits BCE hard-negative loss

### 3.1 Why

CGA-v2 failed mainly by increasing Pd but amplifying false alarms. A sigmoid-probability power penalty may have weak gradients for exactly the high-confidence false positives that need suppression.

CGA-v2.1 should add a safe-background hard-negative term on the **final logits**:

```text
safe background = pixels outside a dilated GT region
loss = top-k BCEWithLogits(final_logits, target=0) on safe background
```

This directly penalizes high-confidence background activation.

### 3.2 Add config

Modify `loss.py`:

```python
@dataclass(frozen=True)
class CGAV21LossConfig(CGALossConfig):
    lambda_safe_bg: float = 0.05
    safe_bg_dilate_px: int = 5
    safe_bg_topk_ratio: float = 0.005
    aux_ratio_cap: float = 0.15
```

### 3.3 Safe-background mask

```python
def _dilate_binary(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    mask = mask.float()
    if radius_px <= 0:
        return mask
    k = 2 * int(radius_px) + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius_px)


def _safe_background_mask(target: torch.Tensor, radius_px: int) -> torch.Tensor:
    dilated = _dilate_binary(target, radius_px)
    return (dilated <= 0).float()
```

### 3.4 Logits BCE top-k safe-background loss

```python
def _safe_bg_topk_bce_with_logits(
    final_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dilate_px: int,
    topk_ratio: float,
) -> torch.Tensor:
    target = _resize_like(target, final_logits)
    safe_bg = _safe_background_mask(target, dilate_px)
    loss_map = F.binary_cross_entropy_with_logits(
        final_logits,
        torch.zeros_like(final_logits),
        reduction="none",
    )

    losses = []
    for b in range(final_logits.shape[0]):
        vals = loss_map[b][safe_bg[b] > 0]
        if vals.numel() == 0:
            continue
        k = max(1, int(vals.numel() * float(topk_ratio)))
        k = min(k, vals.numel())
        losses.append(torch.topk(vals.flatten(), k=k, largest=True).values.mean())
    if not losses:
        return final_logits.sum() * 0.0
    return torch.stack(losses).mean()
```

---

## 4. Ratio cap: use detached scale, not `torch.minimum`

### 4.1 Problem

Do **not** use:

```python
reg_capped = torch.minimum(reg_raw, reg_cap)
```

When `reg_raw > reg_cap`, this cuts off the gradient through `reg_raw`. That is dangerous because false-alarm control may stop contributing exactly when it is too large.

### 4.2 Correct implementation

Use detached scaling:

```python
base_total = base["total"]
reg_raw = w * geom_aux_total + self.cfg.lambda_safe_bg * loss_safe_bg

reg_cap = self.cfg.aux_ratio_cap * base_total.detach().clamp_min(1e-6)
scale = (reg_cap / reg_raw.detach().clamp_min(1e-6)).clamp(max=1.0)
reg_capped = reg_raw * scale

total = base_total + reg_capped
```

This preserves the regularizer gradient direction while bounding its contribution.

### 4.3 Required diagnostics

Log all of these:

```text
reg_raw
reg_capped
reg_raw_over_base
reg_capped_over_base
cap_active
cap_active_rate
cap_scale
cap_scale_mean
safe_bg
geom_aux_total
cga_w
```

Interpretation:

```text
reg_capped_over_base > aux_ratio_cap + tolerance
  => cap implementation bug

reg_raw_over_base >> aux_ratio_cap and cap_active_rate high
  => protocol may be too aggressive, but not necessarily an implementation bug

NaN/Inf in any loss term
  => invalid run
```

---

## 5. New `CGAV21Loss`

Add to `loss.py`:

```python
class CGAV21Loss(CGALoss):
    def __init__(self, cfg: CGAV21LossConfig | None = None, target_cfg=None, *, strict_cga_heads: bool = True) -> None:
        super().__init__(cfg or CGAV21LossConfig(), target_cfg, strict_cga_heads=strict_cga_heads)
        self.cfg: CGAV21LossConfig

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        if not isinstance(output, dict):
            if self.strict_cga_heads:
                raise TypeError("Paper-mode CGA-v2.1 requires dict output with explicit auxiliary logits.")
            return self.base_loss(output, target, epoch=epoch)

        if self.strict_cga_heads:
            _require_cga_logits(output)

        final_logit = extract_final_logit(output)
        target = _resize_like(target, final_logit)
        base = self.base_loss(output, target, epoch=epoch)
        targets = build_cga_targets(target, self.target_cfg)

        loss_center = self._bce(output.get("cga_center_logit"), targets["cga_center_target"])
        loss_boundary = self._bce(output.get("cga_boundary_logit"), targets["cga_boundary_target"])
        loss_scale = self._bce(output.get("cga_scale_logit"), targets["cga_scale_target"])
        loss_peak = self._bce(output.get("cga_peak_logit"), targets["cga_peak_target"])

        w = _ramp_weight(epoch, self.cfg.start_epoch, self.cfg.ramp_epochs)
        geom_aux_total = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )

        loss_safe_bg = _safe_bg_topk_bce_with_logits(
            final_logit,
            target,
            dilate_px=self.cfg.safe_bg_dilate_px,
            topk_ratio=self.cfg.safe_bg_topk_ratio,
        )

        reg_raw = w * geom_aux_total + self.cfg.lambda_safe_bg * loss_safe_bg
        base_total = base["total"]
        reg_cap = self.cfg.aux_ratio_cap * base_total.detach().clamp_min(1e-6)
        cap_scale = (reg_cap / reg_raw.detach().clamp_min(1e-6)).clamp(max=1.0)
        reg_capped = reg_raw * cap_scale
        total = base_total + reg_capped

        eps = torch.tensor(1e-6, device=final_logit.device, dtype=final_logit.dtype)
        base_den = base_total.detach().abs().clamp_min(eps)
        return {
            "total": total,
            "base_total": base_total.detach(),
            "ohem": base["ohem"],
            "soft_iou": base["soft_iou"],
            "scale": base["scale"],
            "cga_w": torch.tensor(w, device=final_logit.device, dtype=final_logit.dtype),
            "cga_center": loss_center.detach(),
            "cga_boundary": loss_boundary.detach(),
            "cga_scale": loss_scale.detach(),
            "cga_peak": loss_peak.detach(),
            "safe_bg": loss_safe_bg.detach(),
            "geom_aux_total": geom_aux_total.detach(),
            "reg_raw": reg_raw.detach(),
            "reg_capped": reg_capped.detach(),
            "reg_raw_over_base": (reg_raw.detach() / base_den).detach(),
            "reg_capped_over_base": (reg_capped.detach() / base_den).detach(),
            "cap_active": (cap_scale.detach() < 0.999).float(),
            "cap_scale": cap_scale.detach(),
        }
```

---

## 6. `build_loss`: add `cga_variant`, do not auto-overwrite v2.1 params

Modify `build_loss(...)`:

```python
def build_loss(
    name: str | None = "MSHNet",
    *,
    use_cga: bool | None = None,
    cga_variant: str = "v2",
    ohem_ratio: float = 0.01,
    lambda_iou: float = 1.0,
    mshnet_warm_epoch: int | None = None,
    warm_epoch: int | None = None,
    cga_start_epoch: int = 1,
    cga_ramp_epochs: int = 40,
    lambda_center: float = 0.05,
    lambda_boundary: float = 0.03,
    lambda_scale: float = 0.02,
    lambda_peak: float = 0.03,
    lambda_safe_bg: float = 0.0,
    safe_bg_dilate_px: int = 5,
    safe_bg_topk_ratio: float = 0.005,
    aux_ratio_cap: float = 0.15,
    strict_cga_heads: bool = True,
    **kwargs,
) -> nn.Module:
    ...
    if use_cga:
        if cga_variant == "v2_1":
            cfg = CGAV21LossConfig(
                lambda_center=float(lambda_center),
                lambda_boundary=float(lambda_boundary),
                lambda_scale=float(lambda_scale),
                lambda_peak=float(lambda_peak),
                lambda_safe_bg=float(lambda_safe_bg),
                safe_bg_dilate_px=int(safe_bg_dilate_px),
                safe_bg_topk_ratio=float(safe_bg_topk_ratio),
                aux_ratio_cap=float(aux_ratio_cap),
                start_epoch=int(cga_start_epoch),
                ramp_epochs=int(cga_ramp_epochs),
                ohem_ratio=float(ohem_ratio),
                lambda_iou=float(lambda_iou),
                warm_epoch=int(mshnet_warm_epoch),
            )
            return CGAV21Loss(cfg, strict_cga_heads=strict_cga_heads)
        if cga_variant == "v2":
            ... existing CGALoss path ...
        raise ValueError(f"Unknown cga_variant={cga_variant!r}")
```

Important rule:

```text
train.py must not silently override v2.1 params.
The v2.1 runner must explicitly pass every protocol hyperparameter.
The protocol-lock checker must compare them before training.
```

---

## 7. `train.py`: add explicit v2.1 args and runtime args JSON

### 7.1 Add CLI args

```python
p.add_argument("--cga_variant", default="v2", choices=["v2", "v2_1"])
p.add_argument("--run_name", default="")
p.add_argument("--lambda_safe_bg", type=float, default=0.0)
p.add_argument("--safe_bg_dilate_px", type=int, default=5)
p.add_argument("--safe_bg_topk_ratio", type=float, default=0.005)
p.add_argument("--aux_ratio_cap", type=float, default=0.15)
p.add_argument("--protocol_lock", default="")
p.add_argument("--runtime_args_json", default="")
```

### 7.2 Do not auto-override defaults

Do **not** implement logic like:

```python
if args.cga_variant == "v2_1":
    args.lambda_center = 0.02
    ...
```

The runner must pass all values explicitly.

### 7.3 Run name

```python
def _run_model_name(model_name, backbone_name, use_cga, cga_variant="v2", run_name=""):
    if run_name:
        return str(run_name)
    if model_name:
        return str(model_name)
    if use_cga and cga_variant == "v2_1":
        return "MSHNetCGA21"
    return f"{backbone_name}_cga" if use_cga else backbone_name
```

### 7.4 Build loss

```python
criterion = build_loss(
    args.model_name or backbone_name,
    use_cga=use_cga,
    cga_variant=args.cga_variant,
    ohem_ratio=args.ohem_ratio,
    mshnet_warm_epoch=args.mshnet_warm_epoch,
    cga_start_epoch=args.cga_start_epoch,
    cga_ramp_epochs=args.cga_ramp_epochs,
    lambda_center=args.lambda_center,
    lambda_boundary=args.lambda_boundary,
    lambda_scale=args.lambda_scale,
    lambda_peak=args.lambda_peak,
    lambda_safe_bg=args.lambda_safe_bg,
    safe_bg_dilate_px=args.safe_bg_dilate_px,
    safe_bg_topk_ratio=args.safe_bg_topk_ratio,
    aux_ratio_cap=args.aux_ratio_cap,
    strict_cga_heads=(args.evidence_mode == "paper" and use_cga),
)
```

### 7.5 Evidence metadata

Add these to `evidence_meta` and checkpoint:

```python
"cga_variant": args.cga_variant,
"regularizer_impl": "center_boundary_scale_peak_safe_bg_bounded" if use_cga and args.cga_variant == "v2_1" else ("center_boundary_scale_peak" if use_cga else "none"),
"lambda_safe_bg": args.lambda_safe_bg,
"safe_bg_dilate_px": args.safe_bg_dilate_px,
"safe_bg_topk_ratio": args.safe_bg_topk_ratio,
"aux_ratio_cap": args.aux_ratio_cap,
"protocol_lock": args.protocol_lock,
```

### 7.6 Runtime args JSON

At startup, if `--runtime_args_json` is set, write:

```python
runtime = {
    "dataset_dir": args.dataset_dir,
    "dataset_name": args.dataset_name,
    "baseline": "MSHNetOHEM",
    "candidate": _run_model_name(...),
    "seed": args.seed,
    "epochs": args.epochs,
    "threshold": 0.5,
    "evidence_mode": args.evidence_mode,
    "protocol": args.protocol,
    "cga_variant": args.cga_variant,
    "mshnet_warm_epoch": args.mshnet_warm_epoch,
    "cga_start_epoch": args.cga_start_epoch,
    "cga_ramp_epochs": args.cga_ramp_epochs,
    "lambda_center": args.lambda_center,
    "lambda_boundary": args.lambda_boundary,
    "lambda_scale": args.lambda_scale,
    "lambda_peak": args.lambda_peak,
    "lambda_safe_bg": args.lambda_safe_bg,
    "safe_bg_dilate_px": args.safe_bg_dilate_px,
    "safe_bg_topk_ratio": args.safe_bg_topk_ratio,
    "aux_ratio_cap": args.aux_ratio_cap,
    "ohem_ratio": args.ohem_ratio,
    "strict_eval_load": True,
}
```

The runner should run the protocol-lock checker **after writing this JSON and before training**.

---

## 8. `test.py`: strict/audited load must be part of v2.1 eval

Current v2.1 evidence must not use silent `strict=False` loading.

### 8.1 Add CLI

```python
p.add_argument("--strict_load", action="store_true")
p.add_argument("--run_name", default="")
p.add_argument("--cga_variant", default="v2", choices=["v2", "v2_1"])
```

### 8.2 Audited load helper

Allow only whitelisted normalization:

```text
Allowed:
  checkpoint["state_dict"]
  checkpoint["model"]
  remove one leading "module."
  remove one explicitly configured wrapper prefix, if documented

Not allowed:
  strict=False paper-mode eval
  dropping arbitrary keys
  loading partial state dict silently
```

Minimal helper:

```python
def _normalize_state_dict(obj):
    state = obj.get("state_dict", obj.get("model", obj)) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    if all(str(k).startswith("module.") for k in state.keys()):
        state = {str(k)[7:]: v for k, v in state.items()}
    return state

state = _normalize_state_dict(ckpt)
if args.strict_load:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Strict/audited load failed: missing={missing}, unexpected={unexpected}")
else:
    model.load_state_dict(state, strict=False)
```

Preferred cleaner version:

```python
if args.strict_load:
    model.load_state_dict(state, strict=True)
else:
    model.load_state_dict(state, strict=False)
```

Use the explicit missing/unexpected version only if prefix normalization is needed before strict validation.

### 8.3 Output name

`test.py` should use `--run_name MSHNetCGA21` so candidate summaries do not overwrite v2 results.

---

## 9. Do not reuse existing v2 summarizer

Do not use:

```text
tools/official/summarize_cga_v2_one_seed.py
```

for v2.1. It does not enforce the v2.1 HC-Val false-alarm and precision guard.

Create:

```text
tools/official/summarize_cga_v21_one_seed.py
```

Required gates:

```python
THRESHOLDS = {
    "full_delta_mIoU_min": 0.020,
    "full_delta_precision_min": 0.010,
    "full_delta_pd_min": -0.001,
    "full_delta_fa_ppm_max": 0.0,
    "hcval_delta_mIoU_min": -0.020,
    "hcval_delta_precision_min": -0.020,
    "hcval_delta_pd_min": -0.001,
    "hcval_delta_fa_ppm_max": 50.0,
}
```

Decision logic:

```python
full_pass = (
    delta_full["mIoU"] >= THRESHOLDS["full_delta_mIoU_min"]
    and delta_full["Precision"] >= THRESHOLDS["full_delta_precision_min"]
    and delta_full["Pd"] >= THRESHOLDS["full_delta_pd_min"]
    and delta_full["FA_ppm"] <= THRESHOLDS["full_delta_fa_ppm_max"]
)

hcval_pass = (
    hcval_available
    and delta_hcval["mIoU"] >= THRESHOLDS["hcval_delta_mIoU_min"]
    and delta_hcval["Precision"] >= THRESHOLDS["hcval_delta_precision_min"]
    and delta_hcval["Pd"] >= THRESHOLDS["hcval_delta_pd_min"]
    and delta_hcval["FA_ppm"] <= THRESHOLDS["hcval_delta_fa_ppm_max"]
)

gate_pass = bool(full_pass and hcval_pass)
```

Output fields:

```json
{
  "gate": "Gate-CGA-v2.1-P2-seed42-from-zero-paired",
  "gate_pass": false,
  "decision": "CGA_V21_SEED42_FAIL_STOP" ,
  "can_run_seed43_44": false,
  "can_claim_positive_cga_v21": false,
  "thresholds": {},
  "full": {},
  "hcval": {},
  "protocol_lock": "docs/internal/cga_v2_1/protocol_lock.json"
}
```

If `gate_pass=true`, decision becomes:

```text
CGA_V21_SEED42_PASS_MAY_RUN_SEED43_44
```

---

## 10. Complete runner: train, eval, summarize, gate

Create:

```text
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

It must not stop after training. It must run:

```text
P1/P1A check
protocol-lock check
OHEM train
CGA21 train
OHEM test eval
CGA21 test eval
OHEM hcval eval
CGA21 hcval eval
v2.1 summarizer
gate JSON
```

Skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"
PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

DATASET_DIR=${DATASET_DIR:-/home/ly/AAAI/CGA-main/datasets}
DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-400}
EPOCH=${EPOCH:-${EPOCHS}}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ly/AAAI/CGA-main/results/official_cga_v21}
PREFLIGHT_SUMMARY=${PREFLIGHT_SUMMARY:-docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json}
P1A_SUMMARY=${P1A_SUMMARY:-docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-docs/internal/cga_v2_1/protocol_lock.json}
P2_DIR=${P2_DIR:-docs/internal/cga_v2_1/gate_p2_from_zero_seed42_${DATASET_NAME}}
RUNTIME_ARGS_JSON=${P2_DIR}/seed42_runtime_args.json
LOCK_CHECK_JSON=${P2_DIR}/protocol_lock_check.json
P2_SUMMARY=${P2_DIR}/summary.json

BASELINE=MSHNetOHEM
CANDIDATE=MSHNetCGA21

mkdir -p "${P2_DIR}"

test -f "${PROTOCOL_LOCK}"

check_json_gate_pass() {
  local path="$1"
  "${PYTHON}" - "$path" <<'PY'
import json, sys
p=sys.argv[1]
obj=json.load(open(p, encoding='utf-8'))
if obj.get('gate_pass') is not True:
    raise SystemExit(f'gate_pass is not true in {p}: {obj.get("gate_pass")!r}')
PY
}

check_json_gate_pass "${PREFLIGHT_SUMMARY}"
check_json_gate_pass "${P1A_SUMMARY}"

cat > "${RUNTIME_ARGS_JSON}" <<JSON
{
  "dataset_dir": "${DATASET_DIR}",
  "dataset_name": "${DATASET_NAME}",
  "baseline": "${BASELINE}",
  "candidate": "${CANDIDATE}",
  "seed": ${SEED},
  "epochs": ${EPOCHS},
  "threshold": 0.5,
  "evidence_mode": "paper",
  "protocol": "controlled",
  "cga_variant": "v2_1",
  "mshnet_warm_epoch": 5,
  "cga_start_epoch": 20,
  "cga_ramp_epochs": 80,
  "lambda_center": 0.02,
  "lambda_boundary": 0.01,
  "lambda_scale": 0.01,
  "lambda_peak": 0.02,
  "lambda_safe_bg": 0.05,
  "safe_bg_dilate_px": 5,
  "safe_bg_topk_ratio": 0.005,
  "aux_ratio_cap": 0.15,
  "ohem_ratio": 0.01,
  "strict_eval_load": true
}
JSON

"${PYTHON}" -m tools.official.check_cga_v21_protocol_lock \
  --lock "${PROTOCOL_LOCK}" \
  --args-json "${RUNTIME_ARGS_JSON}" \
  --output "${LOCK_CHECK_JSON}"

# From-zero guard: avoid accidentally reusing v2 or stale v2.1 checkpoints.
BASE_CKPT="${OUTPUT_DIR}/${BASELINE}/seed${SEED}/${DATASET_NAME}/${BASELINE}_${EPOCH}.pth.tar"
CAND_CKPT="${OUTPUT_DIR}/${CANDIDATE}/seed${SEED}/${DATASET_NAME}/${CANDIDATE}_${EPOCH}.pth.tar"
if [[ "${FORCE_TRAIN:-0}" != "1" ]]; then
  if [[ -f "${BASE_CKPT}" || -f "${CAND_CKPT}" ]]; then
    echo "Existing checkpoint found. Set FORCE_TRAIN=1 or use a fresh OUTPUT_DIR." >&2
    exit 1
  fi
fi

# Baseline OHEM
"${PYTHON}" train.py \
  --model_name MSHNetOHEM \
  --run_name "${BASELINE}" \
  --evidence_mode paper \
  --protocol controlled \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size 8 \
  --patch_size 256 \
  --num_workers 4 \
  --mshnet_warm_epoch 5 \
  --ohem_ratio 0.01 \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --output_dir "${OUTPUT_DIR}"

# Candidate CGA-v2.1
"${PYTHON}" train.py \
  --backbone_name mshnet \
  --use_cga \
  --cga_variant v2_1 \
  --run_name "${CANDIDATE}" \
  --evidence_mode paper \
  --protocol controlled \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size 8 \
  --patch_size 256 \
  --num_workers 4 \
  --mshnet_warm_epoch 5 \
  --cga_start_epoch 20 \
  --cga_ramp_epochs 80 \
  --lambda_center 0.02 \
  --lambda_boundary 0.01 \
  --lambda_scale 0.01 \
  --lambda_peak 0.02 \
  --lambda_safe_bg 0.05 \
  --safe_bg_dilate_px 5 \
  --safe_bg_topk_ratio 0.005 \
  --aux_ratio_cap 0.15 \
  --ohem_ratio 0.01 \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${RUNTIME_ARGS_JSON}" \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --output_dir "${OUTPUT_DIR}"

# Eval helper
run_eval() {
  local model_name="$1"
  local use_cga_flag="$2"
  local cga_variant="$3"
  local split="$4"
  local ckpt="${OUTPUT_DIR}/${model_name}/seed${SEED}/${DATASET_NAME}/${model_name}_${EPOCH}.pth.tar"

  if [[ "${use_cga_flag}" == "1" ]]; then
    "${PYTHON}" test.py \
      --backbone_name mshnet \
      --use_cga \
      --cga_variant "${cga_variant}" \
      --run_name "${model_name}" \
      --evidence_mode paper \
      --dataset_dir "${DATASET_DIR}" \
      --dataset_name "${DATASET_NAME}" \
      --split "${split}" \
      --seed "${SEED}" \
      --checkpoint "${ckpt}" \
      --threshold 0.5 \
      --strict_load \
      --output_dir "${OUTPUT_DIR}"
  else
    "${PYTHON}" test.py \
      --model_name MSHNetOHEM \
      --run_name "${model_name}" \
      --evidence_mode paper \
      --dataset_dir "${DATASET_DIR}" \
      --dataset_name "${DATASET_NAME}" \
      --split "${split}" \
      --seed "${SEED}" \
      --checkpoint "${ckpt}" \
      --threshold 0.5 \
      --strict_load \
      --output_dir "${OUTPUT_DIR}"
  fi
}

run_eval "${BASELINE}" 0 v2 test
run_eval "${CANDIDATE}" 1 v2_1 test
run_eval "${BASELINE}" 0 v2 hcval
run_eval "${CANDIDATE}" 1 v2_1 hcval

BASE_FULL="${OUTPUT_DIR}/${BASELINE}/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
CAND_FULL="${OUTPUT_DIR}/${CANDIDATE}/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
BASE_HCVAL="${OUTPUT_DIR}/${BASELINE}/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"
CAND_HCVAL="${OUTPUT_DIR}/${CANDIDATE}/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"

"${PYTHON}" -m tools.official.summarize_cga_v21_one_seed \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epoch "${EPOCH}" \
  --threshold 0.5 \
  --baseline "${BASELINE}" \
  --candidate "${CANDIDATE}" \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --preflight_summary "${PREFLIGHT_SUMMARY}" \
  --baseline_full "${BASE_FULL}" \
  --candidate_full "${CAND_FULL}" \
  --baseline_hcval "${BASE_HCVAL}" \
  --candidate_hcval "${CAND_HCVAL}" \
  --output "${P2_SUMMARY}"
```

The summarizer should exit nonzero when `gate_pass=false`. That is fine: the JSON is still written and the runner fails closed.

---

## 11. `net.py` / `model/cga_wrapper.py`: distinguish v2.1 metadata without changing aux heads

The architecture can stay the same: four CGA aux heads only.

But metadata should distinguish v2 and v2.1.

### 11.1 `CGAWrapper`

Optional small change:

```python
class CGAWrapper(nn.Module):
    def __init__(..., regularizer_impl: str = "center_boundary_scale_peak"):
        ...
        self.regularizer_impl = regularizer_impl

    def forward(...):
        ...
        output["regularizer_meta"].update({
            "use_cga": True,
            "regularizer_impl": self.regularizer_impl,
            "fallback_regularizer_used": False,
        })
```

### 11.2 `net.py`

Add `cga_variant` to `build_model(...)` and choose:

```python
regularizer_impl = (
    "center_boundary_scale_peak_safe_bg_bounded"
    if cga_variant == "v2_1"
    else "center_boundary_scale_peak"
)
return CGAWrapper(..., regularizer_impl=regularizer_impl)
```

This is metadata-only. The v2.1 method change lives in `loss.py` and protocol.

---

## 12. Tests required before seed42

Add:

```text
tests/test_cga_v21_protocol_lock.py
tests/test_cga_v21_loss_ratio_cap.py
tests/test_cga_v21_safe_bg_loss.py
tests/test_cga_v21_summarizer_gate.py
tests/test_cga_v21_no_v2_contamination.py
tests/test_cga_v21_strict_eval_load.py
```

Minimum assertions:

```text
protocol lock mismatch -> fail
safe-bg loss penalizes high positive background logits more than low logits
ratio cap keeps reg_capped_over_base <= cap + tolerance
reg_raw_over_base can exceed cap and is logged
cap_active becomes true when raw regularizer exceeds cap
v2 summarizer is not called by v2.1 runner
v2.1 summarizer checks HC-Val FA_ppm and Precision
strict_load catches missing/unexpected keys
CGA-v2 checkpoints/results are not used as CGA-v2.1 artifacts
```

---

## 13. Execution order

Do not start seed42 until this order is complete:

```text
R0. Commit/freeze CGA-v2 valid-negative branch.
R1. Create docs/internal/cga_v2_1/protocol_lock.json.
R2. Add CGAV21Loss with logits BCE safe-background loss.
R3. Add detached-scale ratio cap and diagnostics.
R4. Add train.py explicit v2.1 CLI args with no automatic override.
R5. Add test.py strict/audited load and run_name.
R6. Add check_cga_v21_protocol_lock.py.
R7. Add summarize_cga_v21_one_seed.py.
R8. Add complete train/eval/hcval/summarize runner.
R9. Add tests.
R10. Run contract/smoke only.
R11. Commit protocol lock and code.
R12. Start seed42 from-zero paired.
R13. If seed42 fails: stop, no seed43/44.
R14. If seed42 passes: run paired seed43/44.
```

---

## 14. Go / No-Go rule

### 14.1 Seed42 pass required

CGA-v2.1 seed42 must pass:

```text
Full:
  delta_mIoU      >= +0.020
  delta_Precision >= +0.010
  delta_Pd        >= -0.001
  delta_FA_ppm    <= 0.0

HC-Val:
  delta_mIoU      >= -0.020
  delta_Precision >= -0.020
  delta_Pd        >= -0.001
  delta_FA_ppm    <= +50.0
```

### 14.2 If seed42 fails

```text
Stop CGA-v2.1 AAAI route.
Do not run seed43/44.
Do not write positive claim.
Do not tune under the same lock.
```

### 14.3 If seed42 passes

```text
Run paired seed43/44 from zero.
Then run ablation/failure pack.
Then write paper claim.
```

---

## 15. One-line conclusion

The proposed v2.1 direction is right, but the plan must be tightened before execution:

```text
CGA-v2.1 is not allowed to train until the v2.1-specific summarizer, protocol-lock checker, complete runner, strict eval load, logits BCE safe-background loss, and detached-scale ratio cap are implemented and locked.
```
