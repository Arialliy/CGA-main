# CGA-v2.1 Pre-Seed42 Required Fixes

**Canonical root:** `/home/ly/AAAI/CGA-main`

**Verdict:** the critique is correct.

```text
Go: implement the v2.1 pre-seed42 fixes.
No-Go: seed42 from-zero paired training until every contract/smoke/protocol-lock check passes.
```

CGA-v2.1 must be treated as a **new predeclared protocol**, not a continuation of CGA-v2. Current CGA-v2 already failed the seed42 gate and was audited as a valid negative/design-weakness result. Therefore, v2.1 must not inherit CGA-v2 positive claims, thresholds, runner assumptions, or summarizer behavior.

---

## 0. Non-negotiable pre-seed42 requirements

Before launching v2.1 seed42 training, implement all of these:

```text
R1. test.py must support:
    --run_name
    --cga_variant
    --strict_load
    --write_eval_trace_json

R2. v2.1 eval must use strict/audited checkpoint loading:
    whitelist state_dict nesting and prefix normalization
    then strict=True equivalent
    no fallback load path in paper evidence

R3. v2.1 must have its own summarizer:
    tools/official/summarize_cga_v21_one_seed.py
    do not import or reuse v2 gate logic

R4. v2.1 runner must be complete:
    train OHEM
    train CGA21
    eval test/hcval for both
    run v2.1 summarizer
    write seed42 gate JSON

R5. protocol lock must be enforceable:
    runner_runtime_args.json checked before training
    train_runtime_args_candidate.json checked after training
    do not use the same JSON path for both

R6. from-zero protection must be hard:
    paper-evidence runner refuses existing output directories
    do not use FORCE_TRAIN=1

R7. safe-background loss must be explicitly declared:
    either separate ramp or shared ramp
    this plan uses a separate safe-bg ramp

R8. ratio cap must preserve gradients:
    use detached scale
    do not use torch.minimum(reg_raw, reg_cap)

R9. v2.1 paper evidence must not rely on default hyperparameters:
    runner passes every v2.1 hyperparameter explicitly
    protocol-lock checker verifies all of them
```

---

## 1. Add protocol lock

Create:

```text
docs/internal/cga_v2_1/protocol_lock.json
```

Use this exact file initially:

```json
{
  "protocol_name": "CGA-v2.1 false-alarm-controlled component-geometry regularization",
  "protocol_version": "v2.1.0-preseed42",
  "root": "/home/ly/AAAI/CGA-main",
  "dataset_name": "NUDT-SIRST",
  "baseline_model": "MSHNetOHEM",
  "candidate_model": "MSHNetCGA21",
  "backbone_name": "mshnet",
  "seed": 42,
  "epochs": 400,
  "batch_size": 8,
  "patch_size": 256,
  "lr": 0.0005,
  "num_workers": 4,
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "evidence_mode": "paper",
  "protocol": "controlled",
  "strict_load": true,
  "cga_variant": "v2_1",
  "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1",
  "mshnet_warm_epoch": 5,
  "cga_start_epoch": 1,
  "cga_ramp_epochs": 40,
  "lambda_center": 0.05,
  "lambda_boundary": 0.03,
  "lambda_scale": 0.02,
  "lambda_peak": 0.03,
  "lambda_safe_bg": 0.05,
  "safe_bg_topk_ratio": 0.01,
  "safe_bg_ignore_radius": 5,
  "safe_bg_start_epoch": 1,
  "safe_bg_ramp_epochs": 40,
  "aux_ratio_cap": 0.15,
  "ohem_ratio": 0.01,
  "lambda_iou": 1.0,
  "gate_thresholds": {
    "full_delta_mIoU_min": 0.02,
    "full_delta_precision_min": 0.01,
    "full_delta_pd_min": -0.001,
    "full_delta_fa_ppm_max": 0.0,
    "hcval_delta_mIoU_min": 0.0,
    "hcval_delta_precision_min": 0.0,
    "hcval_delta_pd_min": -0.001,
    "hcval_delta_fa_ppm_max": 50.0
  },
  "forbidden": {
    "reuse_cga_v2_positive_claims": true,
    "reuse_cga_v2_summarizer": true,
    "threshold_sweep_for_main_gate": true,
    "seed43_44_before_seed42_gate_pass": true,
    "force_train_in_paper_evidence_runner": true
  }
}
```

---

## 2. Modify `loss.py`

### 2.1 Add this import if missing

At the top of `loss.py`, make sure these imports exist:

```python
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 2.2 Add this code after `class CGALoss`

Insert this exact block after the existing `CGALoss` class and before `MSHNetCGALoss = CGALoss` or before `build_loss(...)`.

```python
@dataclass(frozen=True)
class CGAV21LossConfig(CGALossConfig):
    """False-alarm-controlled CGA-v2.1 loss config.

    Geometry auxiliary supervision and safe-background suppression are both
    explicitly declared in protocol_lock.json. The regularizer is ratio-capped
    with detached scale so gradient direction is preserved even when the cap is
    active.
    """

    lambda_safe_bg: float = 0.05
    safe_bg_topk_ratio: float = 0.01
    safe_bg_ignore_radius: int = 5
    safe_bg_start_epoch: int = 1
    safe_bg_ramp_epochs: int = 40
    aux_ratio_cap: float = 0.15


class CGAV21Loss(nn.Module):
    """CGA-v2.1: component geometry + safe-background false-alarm control."""

    REGULARIZER_IMPL = "center_boundary_scale_peak_safe_bg_v2_1"

    def __init__(
        self,
        cfg: CGAV21LossConfig | None = None,
        target_cfg: CGATargetConfig | None = None,
        *,
        strict_cga_heads: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or CGAV21LossConfig()
        self.target_cfg = target_cfg or CGATargetConfig()
        self.strict_cga_heads = bool(strict_cga_heads)
        self.base_loss = MSHNetOHEMLoss(
            ohem_ratio=self.cfg.ohem_ratio,
            lambda_iou=self.cfg.lambda_iou,
            warm_epoch=self.cfg.warm_epoch,
        )

        if self.cfg.safe_bg_topk_ratio <= 0:
            raise ValueError("safe_bg_topk_ratio must be > 0 for CGA-v2.1")
        if self.cfg.aux_ratio_cap <= 0:
            raise ValueError("aux_ratio_cap must be > 0 for CGA-v2.1")
        if self.cfg.lambda_safe_bg < 0:
            raise ValueError("lambda_safe_bg must be >= 0 for CGA-v2.1")

    @staticmethod
    def _bce(logit: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        if logit is None:
            return target.sum() * 0.0
        target = _resize_like(target, logit)
        return F.binary_cross_entropy_with_logits(logit, target)

    def _safe_background_mask(self, target: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        target = _resize_like(target, ref)
        target = (target > 0.5).float()
        radius = int(self.cfg.safe_bg_ignore_radius)
        if radius > 0:
            kernel = 2 * radius + 1
            dilated = F.max_pool2d(target, kernel_size=kernel, stride=1, padding=radius)
        else:
            dilated = target
        return dilated <= 0.5

    def _safe_background_topk_bce(self, final_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Hard-negative BCEWithLogits on safe background pixels.

        The target is zero on safe-background pixels. We choose top-k by loss
        magnitude, not by detached probability, so high-confidence false alarms
        receive strong gradients.
        """

        target = _resize_like(target, final_logit)
        safe_mask = self._safe_background_mask(target, final_logit)
        zero_target = torch.zeros_like(final_logit)
        loss_map = F.binary_cross_entropy_with_logits(final_logit, zero_target, reduction="none")

        losses: list[torch.Tensor] = []
        for b in range(final_logit.shape[0]):
            values = loss_map[b][safe_mask[b]]
            if values.numel() == 0:
                losses.append(final_logit[b].sum() * 0.0)
                continue
            k = max(1, int(values.numel() * float(self.cfg.safe_bg_topk_ratio)))
            k = min(k, values.numel())
            losses.append(torch.topk(values.flatten(), k=k, largest=True).values.mean())

        return torch.stack(losses).mean()

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
        base_total = base["total"]

        targets = build_cga_targets(target, self.target_cfg)
        loss_center = self._bce(output.get("cga_center_logit"), targets["cga_center_target"])
        loss_boundary = self._bce(output.get("cga_boundary_logit"), targets["cga_boundary_target"])
        loss_scale = self._bce(output.get("cga_scale_logit"), targets["cga_scale_target"])
        loss_peak = self._bce(output.get("cga_peak_logit"), targets["cga_peak_target"])

        geom_w = _ramp_weight(epoch, self.cfg.start_epoch, self.cfg.ramp_epochs)
        safe_bg_w = _ramp_weight(epoch, self.cfg.safe_bg_start_epoch, self.cfg.safe_bg_ramp_epochs)

        geom_aux_total = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )
        safe_bg_loss = self._safe_background_topk_bce(final_logit, target)

        reg_raw = geom_w * geom_aux_total + safe_bg_w * self.cfg.lambda_safe_bg * safe_bg_loss

        base_den = base_total.detach().clamp_min(1e-6)
        reg_cap = float(self.cfg.aux_ratio_cap) * base_den

        # Preserve regularizer gradients while capping magnitude.
        reg_scale = (reg_cap / reg_raw.detach().clamp_min(1e-6)).clamp(max=1.0)
        reg_capped = reg_raw * reg_scale

        total = base_total + reg_capped

        reg_raw_over_base = reg_raw.detach() / base_den
        reg_capped_over_base = reg_capped.detach() / base_den
        cap_active = (reg_scale.detach() < 0.999).float()

        return {
            "total": total,
            "base_total": base_total.detach(),
            "ohem": base["ohem"],
            "soft_iou": base["soft_iou"],
            "scale": base["scale"],
            "cga_w": torch.tensor(geom_w, device=final_logit.device, dtype=final_logit.dtype),
            "safe_bg_w": torch.tensor(safe_bg_w, device=final_logit.device, dtype=final_logit.dtype),
            "cga_center": loss_center.detach(),
            "cga_boundary": loss_boundary.detach(),
            "cga_scale": loss_scale.detach(),
            "cga_peak": loss_peak.detach(),
            "geom_aux_raw": geom_aux_total.detach(),
            "safe_bg": safe_bg_loss.detach(),
            "reg_raw": reg_raw.detach(),
            "reg_capped": reg_capped.detach(),
            "reg_raw_over_base": reg_raw_over_base.detach(),
            "reg_capped_over_base": reg_capped_over_base.detach(),
            "cap_active": cap_active.detach(),
            "cap_scale": reg_scale.detach(),
        }
```

### 2.3 Modify `build_loss(...)`

Add these parameters to the `build_loss(...)` signature:

```python
cga_variant: str = "v2",
lambda_safe_bg: float = 0.05,
safe_bg_topk_ratio: float = 0.01,
safe_bg_ignore_radius: int = 5,
safe_bg_start_epoch: int = 1,
safe_bg_ramp_epochs: int = 40,
aux_ratio_cap: float = 0.15,
```

Then replace the existing `if use_cga:` block in `build_loss(...)` with this exact block:

```python
    if use_cga:
        cga_variant_l = str(cga_variant).lower()
        if cga_variant_l in {"v2_1", "v21", "cga-v2.1", "cga_v2_1"}:
            cfg = CGAV21LossConfig(
                lambda_center=float(lambda_center),
                lambda_boundary=float(lambda_boundary),
                lambda_scale=float(lambda_scale),
                lambda_peak=float(lambda_peak),
                lambda_safe_bg=float(lambda_safe_bg),
                safe_bg_topk_ratio=float(safe_bg_topk_ratio),
                safe_bg_ignore_radius=int(safe_bg_ignore_radius),
                safe_bg_start_epoch=int(safe_bg_start_epoch),
                safe_bg_ramp_epochs=int(safe_bg_ramp_epochs),
                aux_ratio_cap=float(aux_ratio_cap),
                start_epoch=int(cga_start_epoch),
                ramp_epochs=int(cga_ramp_epochs),
                ohem_ratio=float(ohem_ratio),
                lambda_iou=float(lambda_iou),
                warm_epoch=int(mshnet_warm_epoch),
            )
            return CGAV21Loss(cfg, strict_cga_heads=strict_cga_heads)

        if cga_variant_l not in {"v2", "cga-v2", "legacy"}:
            raise ValueError(f"Unknown cga_variant={cga_variant!r}")

        cfg = CGALossConfig(
            lambda_center=float(lambda_center),
            lambda_boundary=float(lambda_boundary),
            lambda_scale=float(lambda_scale),
            lambda_peak=float(lambda_peak),
            start_epoch=int(cga_start_epoch),
            ramp_epochs=int(cga_ramp_epochs),
            ohem_ratio=float(ohem_ratio),
            lambda_iou=float(lambda_iou),
            warm_epoch=int(mshnet_warm_epoch),
        )
        return CGALoss(cfg, strict_cga_heads=strict_cga_heads)
```

---

## 3. Modify `train.py`

### 3.1 Add CLI arguments

In `parse_args()`, add these arguments:

```python
    p.add_argument("--run_name", default=None)
    p.add_argument("--cga_variant", default="v2", choices=["v2", "v2_1"])
    p.add_argument("--lambda_safe_bg", type=float, default=0.05)
    p.add_argument("--safe_bg_topk_ratio", type=float, default=0.01)
    p.add_argument("--safe_bg_ignore_radius", type=int, default=5)
    p.add_argument("--safe_bg_start_epoch", type=int, default=1)
    p.add_argument("--safe_bg_ramp_epochs", type=int, default=40)
    p.add_argument("--aux_ratio_cap", type=float, default=0.15)
    p.add_argument("--write_runtime_args_json", default="")
```

### 3.2 Replace `_run_model_name(...)`

Replace the current `_run_model_name(...)` with:

```python
def _run_model_name(
    model_name: str | None,
    backbone_name: str,
    use_cga: bool,
    run_name: str | None = None,
    cga_variant: str = "v2",
) -> str:
    if run_name:
        return str(run_name)
    if model_name:
        return str(model_name)
    if use_cga and str(cga_variant).lower() in {"v2_1", "v21"}:
        return f"{backbone_name}_cga21"
    return f"{backbone_name}_cga" if use_cga else backbone_name
```

### 3.3 Add helper to write runtime args

Add this function near the other helper functions:

```python
def write_runtime_args_json(
    path: str,
    *,
    args: argparse.Namespace,
    backbone_name: str,
    use_cga: bool,
    run_model_name: str,
) -> None:
    if not path:
        return
    payload = dict(vars(args))
    payload.update(
        {
            "resolved_backbone_name": backbone_name,
            "resolved_use_cga": bool(use_cga),
            "run_model_name": run_model_name,
        }
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
```

### 3.4 Add v2.1 paper evidence hard checks

After the existing fallback check in `main()`, add:

```python
    if args.cga_variant == "v2_1" and args.evidence_mode == "paper":
        if args.protocol != "controlled":
            raise RuntimeError("CGA-v2.1 paper evidence requires --protocol controlled.")
        if not args.p1_preflight_passed or not args.p1a_hcval_source_audit_passed:
            raise RuntimeError(
                "CGA-v2.1 paper evidence requires both "
                "--p1_preflight_passed and --p1a_hcval_source_audit_passed."
            )
```

### 3.5 Use `run_name` and write runtime args

Replace the line that computes `run_model_name` with:

```python
    run_model_name = _run_model_name(
        args.model_name,
        backbone_name,
        use_cga,
        run_name=args.run_name,
        cga_variant=args.cga_variant,
    )
```

After `run_model_name` is computed, add:

```python
    write_runtime_args_json(
        args.write_runtime_args_json,
        args=args,
        backbone_name=backbone_name,
        use_cga=use_cga,
        run_model_name=run_model_name,
    )
```

### 3.6 Pass v2.1 params into `build_loss(...)`

Modify the `criterion = build_loss(...)` call so it includes these arguments:

```python
        cga_variant=args.cga_variant,
        lambda_safe_bg=args.lambda_safe_bg,
        safe_bg_topk_ratio=args.safe_bg_topk_ratio,
        safe_bg_ignore_radius=args.safe_bg_ignore_radius,
        safe_bg_start_epoch=args.safe_bg_start_epoch,
        safe_bg_ramp_epochs=args.safe_bg_ramp_epochs,
        aux_ratio_cap=args.aux_ratio_cap,
```

### 3.7 Record v2.1 metadata

Before `evidence_meta = {...}`, compute:

```python
        regularizer_impl = "none"
        if use_cga:
            regularizer_impl = (
                "center_boundary_scale_peak_safe_bg_v2_1"
                if args.cga_variant == "v2_1"
                else "center_boundary_scale_peak"
            )
```

Then make sure `evidence_meta` contains:

```python
            "cga_variant": args.cga_variant,
            "regularizer_impl": regularizer_impl,
            "lambda_safe_bg": float(args.lambda_safe_bg),
            "safe_bg_topk_ratio": float(args.safe_bg_topk_ratio),
            "safe_bg_ignore_radius": int(args.safe_bg_ignore_radius),
            "safe_bg_start_epoch": int(args.safe_bg_start_epoch),
            "safe_bg_ramp_epochs": int(args.safe_bg_ramp_epochs),
            "aux_ratio_cap": float(args.aux_ratio_cap),
```

Use the same `regularizer_impl` in checkpoint metadata.

---

## 4. Modify `test.py`

### 4.1 Add CLI arguments

In `parse_args()`, add:

```python
    p.add_argument("--run_name", default=None)
    p.add_argument("--cga_variant", default="v2", choices=["v2", "v2_1"])
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--strip_state_prefix", action="append", default=[])
    p.add_argument("--write_eval_trace_json", default="")
```

### 4.2 Replace `_run_model_name(...)`

Replace the existing `_run_model_name(...)` with:

```python
def _run_model_name(
    model_name: str | None,
    backbone_name: str,
    use_cga: bool,
    run_name: str | None = None,
    cga_variant: str = "v2",
) -> str:
    if run_name:
        return str(run_name)
    if model_name:
        return str(model_name)
    if use_cga and str(cga_variant).lower() in {"v2_1", "v21"}:
        return f"{backbone_name}_cga21"
    return f"{backbone_name}_cga" if use_cga else backbone_name
```

### 4.3 Add strict checkpoint load helpers

Add this block before `main()`:

```python
def _extract_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value, key
    if isinstance(ckpt, dict):
        return ckpt, "checkpoint_dict_as_state_dict"
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> tuple[dict, bool]:
    if not prefix:
        return state_dict, False
    if not any(str(k).startswith(prefix) for k in state_dict.keys()):
        return state_dict, False
    out = {}
    for key, value in state_dict.items():
        key_s = str(key)
        out[key_s[len(prefix):] if key_s.startswith(prefix) else key_s] = value
    return out, True


def _normalize_state_dict_whitelist(state_dict: dict, strip_prefixes: list[str]) -> tuple[dict, list[str]]:
    normalized = dict(state_dict)
    applied: list[str] = []

    # Always allow exactly one common DataParallel prefix.
    normalized, used = _strip_prefix_if_present(normalized, "module.")
    if used:
        applied.append("module.")

    # Additional wrapper prefixes must be explicitly supplied by CLI.
    for prefix in strip_prefixes:
        normalized, used = _strip_prefix_if_present(normalized, prefix)
        if used:
            applied.append(prefix)

    return normalized, applied


def strict_audited_load(model: torch.nn.Module, checkpoint_path: str, device, strip_prefixes: list[str]) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_state, state_source = _extract_state_dict_from_checkpoint(ckpt)
    state, applied_prefixes = _normalize_state_dict_whitelist(raw_state, strip_prefixes)

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    missing = sorted(model_keys - state_keys)
    unexpected = sorted(state_keys - model_keys)

    report = {
        "checkpoint": str(checkpoint_path),
        "state_source": state_source,
        "applied_prefixes": applied_prefixes,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_load_pass": not missing and not unexpected,
    }

    if missing or unexpected:
        raise RuntimeError(
            "Strict checkpoint load failed. "
            f"missing={missing[:20]} unexpected={unexpected[:20]} "
            f"full_report={json.dumps(report, sort_keys=True)}"
        )

    model.load_state_dict(state, strict=True)
    return {"checkpoint_object": ckpt, "report": report}
```

### 4.4 Add eval output trace helper

Add this block before `main()`:

```python
def extract_final_logit_with_trace(output):
    if isinstance(output, dict):
        for key in ("final_logit", "final_logits", "base_logits", "base_logit", "logits"):
            if key in output:
                tensor = output[key]
                if not torch.is_tensor(tensor):
                    raise TypeError(f"Output key {key!r} is not a tensor: {type(tensor)!r}")
                trace = {
                    "output_type": "dict",
                    "tensor_source_key": key,
                    "aux_keys_present": sorted([k for k in output.keys() if str(k).startswith("cga_")]),
                    "aux_used_for_prediction": str(key).startswith("cga_"),
                    "logit_shape": list(tensor.shape),
                    "logit_min": float(tensor.detach().min().cpu()),
                    "logit_max": float(tensor.detach().max().cpu()),
                }
                return tensor, trace
        raise KeyError(f"Could not find final logit in output keys: {sorted(output.keys())}")

    if torch.is_tensor(output):
        trace = {
            "output_type": "tensor",
            "tensor_source_key": "tensor_output",
            "aux_keys_present": [],
            "aux_used_for_prediction": False,
            "logit_shape": list(output.shape),
            "logit_min": float(output.detach().min().cpu()),
            "logit_max": float(output.detach().max().cpu()),
        }
        return output, trace

    raise TypeError(f"Unsupported output type for audited eval: {type(output)!r}")
```

### 4.5 Replace checkpoint loading in `main()`

Replace:

```python
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
```

with:

```python
    if not args.strict_load and args.evidence_mode == "paper" and args.cga_variant == "v2_1":
        raise RuntimeError("CGA-v2.1 paper evidence requires --strict_load.")

    if args.strict_load:
        load_result = strict_audited_load(
            model,
            args.checkpoint,
            device=device,
            strip_prefixes=list(args.strip_state_prefix),
        )
        ckpt = load_result["checkpoint_object"]
        strict_load_report = load_result["report"]
    else:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        strict_load_report = {
            "checkpoint": str(args.checkpoint),
            "strict_load_pass": False,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "non_strict_load_used": True,
        }
```

### 4.6 Replace inference tensor extraction

Replace:

```python
            output = model(img, **forward_kwargs)
            logit = extract_final_logit(output)
            prob = torch.sigmoid(logit).cpu()
```

with:

```python
            output = model(img, **forward_kwargs)
            logit, trace = extract_final_logit_with_trace(output)
            prob_tensor = torch.sigmoid(logit)
            trace.update(
                {
                    "prob_min": float(prob_tensor.detach().min().cpu()),
                    "prob_max": float(prob_tensor.detach().max().cpu()),
                    "threshold": float(args.threshold),
                    "positive_pixels_after_threshold": int((prob_tensor.detach() >= args.threshold).sum().cpu()),
                }
            )
            eval_traces.append(trace)
            prob = prob_tensor.cpu()
```

Before the `with torch.no_grad():` loop, add:

```python
    eval_traces = []
```

### 4.7 Record metadata in summary

In `summary.update(...)`, add:

```python
        "run_name": run_model_name,
        "cga_variant": args.cga_variant,
        "regularizer_impl": (
            "center_boundary_scale_peak_safe_bg_v2_1"
            if bool(use_cga) and args.cga_variant == "v2_1"
            else ("center_boundary_scale_peak" if bool(use_cga) else "none")
        ),
        "strict_load": bool(args.strict_load),
        "strict_load_pass": bool(strict_load_report.get("strict_load_pass", False)),
        "strict_load_report": strict_load_report,
        "eval_trace_first_batch": eval_traces[0] if eval_traces else None,
```

After writing `summary_metrics.json`, add:

```python
    if args.write_eval_trace_json:
        trace_path = Path(args.write_eval_trace_json)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {
                    "summary_path": str(out_path),
                    "num_traces": len(eval_traces),
                    "first_trace": eval_traces[0] if eval_traces else None,
                    "strict_load_report": strict_load_report,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
```

---

## 5. Add protocol-lock checker

Create:

```text
tools/official/check_cga_v21_protocol_lock.py
```

with this exact content:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLOAT_KEYS = {
    "lr",
    "threshold",
    "lambda_center",
    "lambda_boundary",
    "lambda_scale",
    "lambda_peak",
    "lambda_safe_bg",
    "safe_bg_topk_ratio",
    "aux_ratio_cap",
    "ohem_ratio",
    "lambda_iou",
}

INT_KEYS = {
    "seed",
    "epochs",
    "batch_size",
    "patch_size",
    "num_workers",
    "mshnet_warm_epoch",
    "cga_start_epoch",
    "cga_ramp_epochs",
    "safe_bg_ignore_radius",
    "safe_bg_start_epoch",
    "safe_bg_ramp_epochs",
}

COMMON_KEYS = [
    "dataset_name",
    "seed",
    "epochs",
    "batch_size",
    "patch_size",
    "lr",
    "num_workers",
    "threshold",
    "evidence_mode",
    "protocol",
    "cga_variant",
    "mshnet_warm_epoch",
    "ohem_ratio",
]

CANDIDATE_KEYS = COMMON_KEYS + [
    "lambda_center",
    "lambda_boundary",
    "lambda_scale",
    "lambda_peak",
    "lambda_safe_bg",
    "safe_bg_topk_ratio",
    "safe_bg_ignore_radius",
    "safe_bg_start_epoch",
    "safe_bg_ramp_epochs",
    "aux_ratio_cap",
    "cga_start_epoch",
    "cga_ramp_epochs",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_value(key: str, value: Any) -> Any:
    if key in FLOAT_KEYS:
        return float(value)
    if key in INT_KEYS:
        return int(value)
    if key in {"strict_load"}:
        return bool(value)
    return value


def equal_values(key: str, a: Any, b: Any, tol: float) -> bool:
    a_n = normalize_value(key, a)
    b_n = normalize_value(key, b)
    if key in FLOAT_KEYS:
        return math.isclose(float(a_n), float(b_n), rel_tol=tol, abs_tol=tol)
    return a_n == b_n


def main() -> None:
    p = argparse.ArgumentParser("Check CGA-v2.1 runtime args against protocol lock")
    p.add_argument("--protocol_lock", required=True)
    p.add_argument("--runtime_args_json", required=True)
    p.add_argument("--role", required=True, choices=["baseline", "candidate"])
    p.add_argument("--output", required=True)
    p.add_argument("--float_tol", type=float, default=1e-9)
    args = p.parse_args()

    lock = load_json(args.protocol_lock)
    runtime = load_json(args.runtime_args_json)

    required_keys = CANDIDATE_KEYS if args.role == "candidate" else COMMON_KEYS

    mismatches = []
    missing = []
    for key in required_keys:
        if key not in lock:
            missing.append({"source": "protocol_lock", "key": key})
            continue
        if key not in runtime:
            missing.append({"source": "runtime_args_json", "key": key})
            continue
        if not equal_values(key, lock[key], runtime[key], args.float_tol):
            mismatches.append(
                {
                    "key": key,
                    "protocol_lock": lock[key],
                    "runtime": runtime[key],
                }
            )

    if args.role == "candidate":
        if runtime.get("run_model_name") != lock.get("candidate_model"):
            mismatches.append(
                {
                    "key": "run_model_name",
                    "protocol_lock": lock.get("candidate_model"),
                    "runtime": runtime.get("run_model_name"),
                }
            )
        if not bool(runtime.get("resolved_use_cga", runtime.get("use_cga", False))):
            mismatches.append(
                {
                    "key": "use_cga",
                    "protocol_lock": True,
                    "runtime": runtime.get("resolved_use_cga", runtime.get("use_cga")),
                }
            )
    else:
        if runtime.get("run_model_name") != lock.get("baseline_model"):
            mismatches.append(
                {
                    "key": "run_model_name",
                    "protocol_lock": lock.get("baseline_model"),
                    "runtime": runtime.get("run_model_name"),
                }
            )
        if bool(runtime.get("resolved_use_cga", runtime.get("use_cga", False))):
            mismatches.append(
                {
                    "key": "use_cga",
                    "protocol_lock": False,
                    "runtime": runtime.get("resolved_use_cga", runtime.get("use_cga")),
                }
            )

    out = {
        "checker": "check_cga_v21_protocol_lock",
        "protocol_lock": str(args.protocol_lock),
        "runtime_args_json": str(args.runtime_args_json),
        "role": args.role,
        "pass": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))

    if not out["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
```

---

## 6. Add v2.1 summarizer

Create:

```text
tools/official/summarize_cga_v21_one_seed.py
```

with this exact content:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], key: str) -> float:
    if key in summary:
        return float(summary[key])
    lower = key.lower()
    upper = key.upper()
    for candidate in (lower, upper):
        if candidate in summary:
            return float(summary[candidate])
    if key == "FA_ppm" and "FA" in summary:
        return float(summary["FA"]) * 1_000_000.0
    raise KeyError(f"Missing metric {key!r}; available keys={sorted(summary.keys())}")


def collect_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "mIoU": metric(summary, "mIoU"),
        "Precision": metric(summary, "Precision"),
        "Pd": metric(summary, "Pd"),
        "FA_ppm": metric(summary, "FA_ppm"),
        "F1": metric(summary, "F1") if "F1" in summary else float("nan"),
        "nIoU": metric(summary, "nIoU") if "nIoU" in summary else float("nan"),
    }


def delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {k: float(candidate[k] - baseline[k]) for k in ("mIoU", "Precision", "Pd", "FA_ppm")}


def pass_full(d: dict[str, float], thresholds: dict[str, float]) -> bool:
    return bool(
        d["mIoU"] >= float(thresholds["full_delta_mIoU_min"])
        and d["Precision"] >= float(thresholds["full_delta_precision_min"])
        and d["Pd"] >= float(thresholds["full_delta_pd_min"])
        and d["FA_ppm"] <= float(thresholds["full_delta_fa_ppm_max"])
    )


def pass_hcval(d: dict[str, float], thresholds: dict[str, float]) -> bool:
    return bool(
        d["mIoU"] >= float(thresholds["hcval_delta_mIoU_min"])
        and d["Precision"] >= float(thresholds["hcval_delta_precision_min"])
        and d["Pd"] >= float(thresholds["hcval_delta_pd_min"])
        and d["FA_ppm"] <= float(thresholds["hcval_delta_fa_ppm_max"])
    )


def require_metadata(summary: dict[str, Any], *, model: str, cga_variant: str, strict_load: bool) -> list[str]:
    problems = []
    if summary.get("model") != model and summary.get("run_name") != model:
        problems.append(f"expected model/run_name={model}, got model={summary.get('model')} run_name={summary.get('run_name')}")
    if summary.get("cga_variant") != cga_variant:
        problems.append(f"expected cga_variant={cga_variant}, got {summary.get('cga_variant')}")
    if bool(summary.get("strict_load_pass", False)) != bool(strict_load):
        problems.append(f"expected strict_load_pass={strict_load}, got {summary.get('strict_load_pass')}")
    if float(summary.get("threshold", -1.0)) != 0.5:
        problems.append(f"expected threshold=0.5, got {summary.get('threshold')}")
    if summary.get("threshold_selection") != "fixed_predeclared":
        problems.append(f"expected threshold_selection=fixed_predeclared, got {summary.get('threshold_selection')}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser("Summarize one seed for CGA-v2.1")
    p.add_argument("--protocol_lock", required=True)
    p.add_argument("--baseline_full", required=True)
    p.add_argument("--candidate_full", required=True)
    p.add_argument("--baseline_hcval", required=True)
    p.add_argument("--candidate_hcval", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = load_json(args.protocol_lock)
    thresholds = lock["gate_thresholds"]

    baseline_full_raw = load_json(args.baseline_full)
    candidate_full_raw = load_json(args.candidate_full)
    baseline_hcval_raw = load_json(args.baseline_hcval)
    candidate_hcval_raw = load_json(args.candidate_hcval)

    metadata_problems = []
    metadata_problems.extend(
        require_metadata(
            baseline_full_raw,
            model=lock["baseline_model"],
            cga_variant=lock["cga_variant"],
            strict_load=bool(lock["strict_load"]),
        )
    )
    metadata_problems.extend(
        require_metadata(
            candidate_full_raw,
            model=lock["candidate_model"],
            cga_variant=lock["cga_variant"],
            strict_load=bool(lock["strict_load"]),
        )
    )
    metadata_problems.extend(
        require_metadata(
            baseline_hcval_raw,
            model=lock["baseline_model"],
            cga_variant=lock["cga_variant"],
            strict_load=bool(lock["strict_load"]),
        )
    )
    metadata_problems.extend(
        require_metadata(
            candidate_hcval_raw,
            model=lock["candidate_model"],
            cga_variant=lock["cga_variant"],
            strict_load=bool(lock["strict_load"]),
        )
    )

    baseline_full = collect_metrics(baseline_full_raw)
    candidate_full = collect_metrics(candidate_full_raw)
    baseline_hcval = collect_metrics(baseline_hcval_raw)
    candidate_hcval = collect_metrics(candidate_hcval_raw)

    full_delta = delta(candidate_full, baseline_full)
    hcval_delta = delta(candidate_hcval, baseline_hcval)

    full_rule_pass = pass_full(full_delta, thresholds)
    hcval_rule_pass = pass_hcval(hcval_delta, thresholds)

    gate_pass = bool(full_rule_pass and hcval_rule_pass and not metadata_problems)

    out = {
        "gate": "Gate-CGA-v2.1-seed42-from-zero-paired",
        "decision_rule_name": "CGA-v2.1 seed42 false-alarm-controlled gate",
        "decision_rule_predeclared": True,
        "protocol_lock": str(args.protocol_lock),
        "dataset_name": lock["dataset_name"],
        "seed": int(lock["seed"]),
        "baseline": lock["baseline_model"],
        "candidate": lock["candidate_model"],
        "threshold": float(lock["threshold"]),
        "threshold_selection": lock["threshold_selection"],
        "full": {
            "baseline": baseline_full,
            "candidate": candidate_full,
            "delta": full_delta,
            "rule_pass": full_rule_pass,
        },
        "hcval": {
            "baseline": baseline_hcval,
            "candidate": candidate_hcval,
            "delta": hcval_delta,
            "rule_pass": hcval_rule_pass,
        },
        "metadata_problems": metadata_problems,
        "pass_conditions": {
            "full_rule_pass": full_rule_pass,
            "hcval_rule_pass": hcval_rule_pass,
            "metadata_pass": not metadata_problems,
        },
        "gate_pass": gate_pass,
        "can_run_seed43_44": gate_pass,
        "can_claim_positive_cga_v21": gate_pass,
        "decision": "P3_CGA_V21_SEED42_PASS_RUN_SEED43_44" if gate_pass else "P3_CGA_V21_SEED42_FAIL_STOP",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))

    if not gate_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
```

---

## 7. Add complete v2.1 runner

Create:

```text
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

with this exact content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-1}

DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets}
DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-400}
BATCH_SIZE=${BATCH_SIZE:-8}
PATCH_SIZE=${PATCH_SIZE:-256}
LR=${LR:-0.0005}
NUM_WORKERS=${NUM_WORKERS:-4}

RUN_ID=${RUN_ID:?RUN_ID is required for from-zero paper evidence, e.g. RUN_ID=seed42_20260705_v21_predeclared}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/results/official_cga_v21/${RUN_ID}}
AUDIT_DIR=${AUDIT_DIR:-${ROOT}/docs/internal/cga_v2_1/gate_seed42_${DATASET_NAME}/${RUN_ID}}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-${ROOT}/docs/internal/cga_v2_1/protocol_lock.json}

BASELINE_MODEL=MSHNetOHEM
CANDIDATE_MODEL=MSHNetCGA21
BACKBONE_NAME=mshnet

THRESHOLD=0.5
MSHNET_WARM_EPOCH=5
CGA_START_EPOCH=1
CGA_RAMP_EPOCHS=40
LAMBDA_CENTER=0.05
LAMBDA_BOUNDARY=0.03
LAMBDA_SCALE=0.02
LAMBDA_PEAK=0.03
LAMBDA_SAFE_BG=0.05
SAFE_BG_TOPK_RATIO=0.01
SAFE_BG_IGNORE_RADIUS=5
SAFE_BG_START_EPOCH=1
SAFE_BG_RAMP_EPOCHS=40
AUX_RATIO_CAP=0.15
OHEM_RATIO=0.01

export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

mkdir -p "${AUDIT_DIR}"

if [[ ! -f "${PROTOCOL_LOCK}" ]]; then
  echo "[CGA-v2.1][FAIL] Missing protocol lock: ${PROTOCOL_LOCK}" >&2
  exit 10
fi

P1_SUMMARY="${ROOT}/docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json"
P1A_SUMMARY="${ROOT}/docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
for name, path in [("P1", "${P1_SUMMARY}"), ("P1A", "${P1A_SUMMARY}")]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[CGA-v2.1][FAIL] Missing {name} summary: {p}")
    data = json.loads(p.read_text())
    if not bool(data.get("gate_pass", False)):
        raise SystemExit(f"[CGA-v2.1][FAIL] {name} gate_pass is not true: {p}")
print("[CGA-v2.1] P1/P1A gates pass.")
PY

if [[ -e "${OUTPUT_ROOT}/${BASELINE_MODEL}/seed${SEED}/${DATASET_NAME}" ]]; then
  echo "[CGA-v2.1][FAIL] Existing baseline output dir would contaminate from-zero evidence:" >&2
  echo "  ${OUTPUT_ROOT}/${BASELINE_MODEL}/seed${SEED}/${DATASET_NAME}" >&2
  exit 11
fi

if [[ -e "${OUTPUT_ROOT}/${CANDIDATE_MODEL}/seed${SEED}/${DATASET_NAME}" ]]; then
  echo "[CGA-v2.1][FAIL] Existing candidate output dir would contaminate from-zero evidence:" >&2
  echo "  ${OUTPUT_ROOT}/${CANDIDATE_MODEL}/seed${SEED}/${DATASET_NAME}" >&2
  exit 12
fi

write_runtime_args() {
  local role="$1"
  local model="$2"
  local use_cga="$3"
  local output_json="$4"

  "${PYTHON}" - "$role" "$model" "$use_cga" "$output_json" <<PY
import json
import sys
role, model, use_cga, output_json = sys.argv[1:5]
payload = {
    "role": role,
    "run_model_name": model,
    "model_name": None,
    "backbone_name": "${BACKBONE_NAME}",
    "resolved_backbone_name": "${BACKBONE_NAME}",
    "use_cga": use_cga.lower() == "true",
    "resolved_use_cga": use_cga.lower() == "true",
    "cga_variant": "v2_1",
    "evidence_mode": "paper",
    "protocol": "controlled",
    "dataset_dir": "${DATASET_DIR}",
    "dataset_name": "${DATASET_NAME}",
    "seed": int("${SEED}"),
    "epochs": int("${EPOCHS}"),
    "batch_size": int("${BATCH_SIZE}"),
    "patch_size": int("${PATCH_SIZE}"),
    "lr": float("${LR}"),
    "num_workers": int("${NUM_WORKERS}"),
    "threshold": float("${THRESHOLD}"),
    "mshnet_warm_epoch": int("${MSHNET_WARM_EPOCH}"),
    "cga_start_epoch": int("${CGA_START_EPOCH}"),
    "cga_ramp_epochs": int("${CGA_RAMP_EPOCHS}"),
    "lambda_center": float("${LAMBDA_CENTER}"),
    "lambda_boundary": float("${LAMBDA_BOUNDARY}"),
    "lambda_scale": float("${LAMBDA_SCALE}"),
    "lambda_peak": float("${LAMBDA_PEAK}"),
    "lambda_safe_bg": float("${LAMBDA_SAFE_BG}"),
    "safe_bg_topk_ratio": float("${SAFE_BG_TOPK_RATIO}"),
    "safe_bg_ignore_radius": int("${SAFE_BG_IGNORE_RADIUS}"),
    "safe_bg_start_epoch": int("${SAFE_BG_START_EPOCH}"),
    "safe_bg_ramp_epochs": int("${SAFE_BG_RAMP_EPOCHS}"),
    "aux_ratio_cap": float("${AUX_RATIO_CAP}"),
    "ohem_ratio": float("${OHEM_RATIO}"),
    "strict_load": True,
}
from pathlib import Path
p = Path(output_json)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(p)
PY
}

BASE_RUNNER_ARGS="${AUDIT_DIR}/runner_runtime_args_baseline.json"
CAND_RUNNER_ARGS="${AUDIT_DIR}/runner_runtime_args_candidate.json"
BASE_TRAIN_ARGS="${AUDIT_DIR}/train_runtime_args_baseline.json"
CAND_TRAIN_ARGS="${AUDIT_DIR}/train_runtime_args_candidate.json"

write_runtime_args baseline "${BASELINE_MODEL}" false "${BASE_RUNNER_ARGS}"
write_runtime_args candidate "${CANDIDATE_MODEL}" true "${CAND_RUNNER_ARGS}"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${BASE_RUNNER_ARGS}" \
  --role baseline \
  --output "${AUDIT_DIR}/A0_protocol_lock_baseline_pretrain.json"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${CAND_RUNNER_ARGS}" \
  --role candidate \
  --output "${AUDIT_DIR}/A0_protocol_lock_candidate_pretrain.json"

echo "[CGA-v2.1] Training baseline from zero: ${BASELINE_MODEL}"
"${PYTHON}" train.py \
  --run_name "${BASELINE_MODEL}" \
  --backbone_name "${BACKBONE_NAME}" \
  --cga_variant v2_1 \
  --evidence_mode paper \
  --protocol controlled \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --mshnet_warm_epoch "${MSHNET_WARM_EPOCH}" \
  --ohem_ratio "${OHEM_RATIO}" \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --write_runtime_args_json "${BASE_TRAIN_ARGS}" \
  --output_dir "${OUTPUT_ROOT}"

echo "[CGA-v2.1] Training candidate from zero: ${CANDIDATE_MODEL}"
"${PYTHON}" train.py \
  --run_name "${CANDIDATE_MODEL}" \
  --backbone_name "${BACKBONE_NAME}" \
  --use_cga \
  --cga_variant v2_1 \
  --evidence_mode paper \
  --protocol controlled \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --patch_size "${PATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --mshnet_warm_epoch "${MSHNET_WARM_EPOCH}" \
  --cga_start_epoch "${CGA_START_EPOCH}" \
  --cga_ramp_epochs "${CGA_RAMP_EPOCHS}" \
  --lambda_center "${LAMBDA_CENTER}" \
  --lambda_boundary "${LAMBDA_BOUNDARY}" \
  --lambda_scale "${LAMBDA_SCALE}" \
  --lambda_peak "${LAMBDA_PEAK}" \
  --lambda_safe_bg "${LAMBDA_SAFE_BG}" \
  --safe_bg_topk_ratio "${SAFE_BG_TOPK_RATIO}" \
  --safe_bg_ignore_radius "${SAFE_BG_IGNORE_RADIUS}" \
  --safe_bg_start_epoch "${SAFE_BG_START_EPOCH}" \
  --safe_bg_ramp_epochs "${SAFE_BG_RAMP_EPOCHS}" \
  --aux_ratio_cap "${AUX_RATIO_CAP}" \
  --ohem_ratio "${OHEM_RATIO}" \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --write_runtime_args_json "${CAND_TRAIN_ARGS}" \
  --output_dir "${OUTPUT_ROOT}"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${BASE_TRAIN_ARGS}" \
  --role baseline \
  --output "${AUDIT_DIR}/A1_protocol_lock_baseline_posttrain.json"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${CAND_TRAIN_ARGS}" \
  --role candidate \
  --output "${AUDIT_DIR}/A1_protocol_lock_candidate_posttrain.json"

BASE_CKPT="${OUTPUT_ROOT}/${BASELINE_MODEL}/seed${SEED}/${DATASET_NAME}/${BASELINE_MODEL}_${EPOCHS}.pth.tar"
CAND_CKPT="${OUTPUT_ROOT}/${CANDIDATE_MODEL}/seed${SEED}/${DATASET_NAME}/${CANDIDATE_MODEL}_${EPOCHS}.pth.tar"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[CGA-v2.1][FAIL] Missing baseline checkpoint: ${BASE_CKPT}" >&2
  exit 20
fi
if [[ ! -f "${CAND_CKPT}" ]]; then
  echo "[CGA-v2.1][FAIL] Missing candidate checkpoint: ${CAND_CKPT}" >&2
  exit 21
fi

for SPLIT in test hcval; do
  echo "[CGA-v2.1] Eval baseline split=${SPLIT}"
  "${PYTHON}" test.py \
    --run_name "${BASELINE_MODEL}" \
    --backbone_name "${BACKBONE_NAME}" \
    --cga_variant v2_1 \
    --evidence_mode paper \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --checkpoint "${BASE_CKPT}" \
    --threshold "${THRESHOLD}" \
    --strict_load \
    --num_workers 1 \
    --write_eval_trace_json "${AUDIT_DIR}/eval_trace_${BASELINE_MODEL}_${SPLIT}.json" \
    --output_dir "${OUTPUT_ROOT}"

  echo "[CGA-v2.1] Eval candidate split=${SPLIT}"
  "${PYTHON}" test.py \
    --run_name "${CANDIDATE_MODEL}" \
    --backbone_name "${BACKBONE_NAME}" \
    --use_cga \
    --cga_variant v2_1 \
    --evidence_mode paper \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --checkpoint "${CAND_CKPT}" \
    --threshold "${THRESHOLD}" \
    --strict_load \
    --num_workers 1 \
    --write_eval_trace_json "${AUDIT_DIR}/eval_trace_${CANDIDATE_MODEL}_${SPLIT}.json" \
    --output_dir "${OUTPUT_ROOT}"
done

BASE_FULL="${OUTPUT_ROOT}/${BASELINE_MODEL}/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
CAND_FULL="${OUTPUT_ROOT}/${CANDIDATE_MODEL}/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
BASE_HC="${OUTPUT_ROOT}/${BASELINE_MODEL}/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"
CAND_HC="${OUTPUT_ROOT}/${CANDIDATE_MODEL}/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"

"${PYTHON}" tools/official/summarize_cga_v21_one_seed.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --baseline_full "${BASE_FULL}" \
  --candidate_full "${CAND_FULL}" \
  --baseline_hcval "${BASE_HC}" \
  --candidate_hcval "${CAND_HC}" \
  --output "${AUDIT_DIR}/seed42_gate_summary.json"
```

Make it executable:

```bash
chmod +x scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

---

## 8. Add v2.1 no-contamination test

Create:

```text
tests/test_cga_v21_no_v2_contamination.py
```

with this exact content:

```python
from __future__ import annotations

import json
from pathlib import Path


def test_protocol_lock_forbids_v2_summarizer() -> None:
    lock_path = Path("docs/internal/cga_v2_1/protocol_lock.json")
    assert lock_path.exists()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["cga_variant"] == "v2_1"
    assert lock["forbidden"]["reuse_cga_v2_summarizer"] is True


def test_v21_tools_exist() -> None:
    assert Path("tools/official/summarize_cga_v21_one_seed.py").exists()
    assert Path("tools/official/check_cga_v21_protocol_lock.py").exists()


def test_v21_runner_exists_and_does_not_use_force_train() -> None:
    runner = Path("scripts/official/run_cga_v21_seed42_from_zero_paired.sh")
    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "FORCE_TRAIN" not in text
    assert "summarize_cga_v21_one_seed.py" in text
    assert "summarize_cga_v2_one_seed.py" not in text
    assert "--strict_load" in text
    assert "--write_eval_trace_json" in text
```

---

## 9. Validation commands before seed42

Run these first:

```bash
cd /home/ly/AAAI/CGA-main

python3 -m py_compile \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py \
  train.py \
  test.py \
  loss.py

bash -n scripts/official/run_cga_v21_seed42_from_zero_paired.sh

python3 - <<'PY'
import json
from pathlib import Path
lock = json.loads(Path("docs/internal/cga_v2_1/protocol_lock.json").read_text())
assert lock["cga_variant"] == "v2_1"
assert lock["strict_load"] is True
assert lock["gate_thresholds"]["hcval_delta_fa_ppm_max"] == 50.0
assert lock["forbidden"]["force_train_in_paper_evidence_runner"] is True
print("protocol_lock sanity pass")
PY

git diff --check
```

If `pytest` is available:

```bash
python3 -m pytest tests/test_cga_v21_no_v2_contamination.py -q
```

If `pytest` is unavailable, at least run:

```bash
python3 - <<'PY'
from pathlib import Path
for p in [
    "docs/internal/cga_v2_1/protocol_lock.json",
    "tools/official/summarize_cga_v21_one_seed.py",
    "tools/official/check_cga_v21_protocol_lock.py",
    "scripts/official/run_cga_v21_seed42_from_zero_paired.sh",
]:
    assert Path(p).exists(), p
print("v2.1 required files exist")
PY
```

---

## 10. Smoke-only dry run before paper seed42

Do not launch epoch400 immediately after editing. First run a small smoke in a separate output root and do **not** call it paper evidence.

Use a separate smoke runner or manually run 1 epoch with:

```bash
cd /home/ly/AAAI/CGA-main

CUDA_VISIBLE_DEVICES=1 python train.py \
  --run_name MSHNetCGA21_smoke \
  --backbone_name mshnet \
  --use_cga \
  --cga_variant v2_1 \
  --evidence_mode smoke \
  --protocol controlled \
  --dataset_dir /home/ly/AAAI/CGA-main/datasets \
  --dataset_name NUDT-SIRST \
  --seed 42 \
  --epochs 1 \
  --batch_size 2 \
  --patch_size 64 \
  --num_workers 0 \
  --mshnet_warm_epoch 1 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 1 \
  --lambda_center 0.05 \
  --lambda_boundary 0.03 \
  --lambda_scale 0.02 \
  --lambda_peak 0.03 \
  --lambda_safe_bg 0.05 \
  --safe_bg_topk_ratio 0.01 \
  --safe_bg_ignore_radius 3 \
  --safe_bg_start_epoch 1 \
  --safe_bg_ramp_epochs 1 \
  --aux_ratio_cap 0.15 \
  --output_dir /home/ly/AAAI/CGA-main/results/smoke_cga_v21
```

Smoke pass criteria:

```text
train.py runs
loss output contains:
  safe_bg
  safe_bg_w
  reg_raw
  reg_capped
  reg_raw_over_base
  reg_capped_over_base
  cap_active
  cap_scale

No NaN/Inf.
paper_evidence_allowed must be false in smoke mode.
```

---

## 11. Launch seed42 only after all checks pass

Only after code validation and smoke pass:

```bash
cd /home/ly/AAAI/CGA-main

RUN_ID=seed42_cga_v21_predeclared_001 \
CUDA_DEVICE=1 \
ROOT=/home/ly/AAAI/CGA-main \
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

Expected final gate file:

```text
docs/internal/cga_v2_1/gate_seed42_NUDT-SIRST/seed42_cga_v21_predeclared_001/seed42_gate_summary.json
```

Decision rule:

```text
gate_pass=false:
  stop
  do not run seed43/44
  do not write positive CGA-v2.1 claim

gate_pass=true:
  open seed43/44 paired route
```

---

## 12. Commit sequence

Use two commits before training:

```bash
cd /home/ly/AAAI/CGA-main

git add \
  docs/internal/cga_v2_1/protocol_lock.json \
  loss.py train.py test.py \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py \
  scripts/official/run_cga_v21_seed42_from_zero_paired.sh \
  tests/test_cga_v21_no_v2_contamination.py

git commit -m "Predeclare CGA-v2.1 false-alarm-controlled protocol"
```

After validation/smoke, commit smoke artifacts only if they are small metadata files; do not commit predictions/checkpoints:

```bash
git status --short | grep -E 'results/|predictions/|\.pth|\.pth\.tar|checkpoint' || true
```

If this prints anything, do not commit those files.

---

## 13. Final go/no-go

```text
GO:
  implement fixes
  run contract tests
  run smoke
  commit protocol lock and code

NO-GO:
  seed42 paper training before the above passes
  seed43/44 before seed42 gate pass
  reusing v2 summarizer
  using strict=False eval for v2.1 paper evidence
  using FORCE_TRAIN=1
  relying on train.py default hyperparameters as protocol
```
