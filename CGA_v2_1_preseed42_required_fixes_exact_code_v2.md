# CGA-v2.1 Pre-Seed42 Required Fixes — Exact Code v2

Canonical repo root:

```text
/home/ly/AAAI/CGA-main
```

Status:

```text
Go: implement protocol fixes, audit-only tests, contract/smoke.
No-Go: seed42 from-zero paired training until all fixes below pass.
```

This document supersedes `CGA_v2_1_preseed42_required_fixes_exact_code.md` for the pre-seed42 CGA-v2.1 implementation stage.

The CGA-v2 result must remain frozen as a valid negative design-weakness result. CGA-v2.1 is a new protocol, not a post-hoc retuning of CGA-v2.

---

## 0. Why this v2 revision is required

The previous v2.1 code plan had six evidence-validity gaps:

1. Protocol lock did not compare all locked fields, especially `strict_load`, `root`, `dataset_dir`, `threshold_selection`, `regularizer_impl`, and `lambda_iou`.
2. Baseline metadata was incorrectly allowed to use `cga_variant="v2_1"`; baseline must be `cga_variant="none"`.
3. The summarizer did not strongly audit baseline/candidate metadata.
4. `CGAWrapper` still reported `regularizer_impl=center_boundary_scale_peak`, while v2.1 logs wanted `center_boundary_scale_peak_safe_bg_v2_1`.
5. Strict-load prefix normalization was too broad.
6. Unit tests did not cover protocol mismatch, safe-bg loss, ratio cap, HC-Val FA gate, or strict-load failures.

Seed42 must not start before these are fixed.

---

## 1. Required protocol semantics

Use two separate concepts:

```text
protocol_variant = v2_1
```

This describes the experimental protocol.

```text
candidate_cga_variant = v2_1
```

This describes the candidate method.

Baseline metadata must be:

```json
{
  "model": "MSHNetOHEM",
  "use_cga": false,
  "cga_variant": "none",
  "regularizer_impl": "none"
}
```

Candidate metadata must be:

```json
{
  "model": "MSHNetCGA21",
  "use_cga": true,
  "cga_variant": "v2_1",
  "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"
}
```

Do not allow the baseline to inherit candidate CGA metadata.

---

## 2. Add protocol lock

Create:

```text
docs/internal/cga_v2_1/protocol_lock.json
```

Exact content:

```json
{
  "protocol_id": "CGA-v2.1-false-alarm-controlled-seed42",
  "protocol_variant": "v2_1",
  "root": "/home/ly/AAAI/CGA-main",
  "dataset_dir": "/home/ly/AAAI/CGA-main/datasets",
  "dataset_name": "NUDT-SIRST",
  "seed": 42,
  "epochs": 400,
  "checkpoint_epoch": 400,
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "strict_load": true,
  "evidence_mode": "paper",
  "protocol": "controlled",
  "p1_preflight_passed": true,
  "p1a_hcval_source_audit_passed": true,
  "backbone_name": "mshnet",
  "baseline": {
    "model_name": "MSHNetOHEM",
    "run_name": "MSHNetOHEM",
    "use_cga": false,
    "cga_variant": "none",
    "regularizer_impl": "none"
  },
  "candidate": {
    "model_name": "MSHNetCGA21",
    "run_name": "MSHNetCGA21",
    "use_cga": true,
    "cga_variant": "v2_1",
    "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"
  },
  "optimizer": {
    "lr": 0.0005,
    "batch_size": 8,
    "patch_size": 256,
    "num_workers": 4
  },
  "loss": {
    "ohem_ratio": 0.01,
    "lambda_iou": 1.0,
    "mshnet_warm_epoch": 5,
    "lambda_center": 0.05,
    "lambda_boundary": 0.03,
    "lambda_scale": 0.02,
    "lambda_peak": 0.03,
    "cga_start_epoch": 1,
    "cga_ramp_epochs": 40,
    "lambda_safe_bg": 0.03,
    "safe_bg_start_epoch": 1,
    "safe_bg_ramp_epochs": 40,
    "safe_bg_topk_ratio": 0.01,
    "safe_bg_ignore_radius": 3,
    "aux_ratio_cap": 0.15
  },
  "gate": {
    "full_delta_mIoU_min": 0.02,
    "full_delta_precision_min": 0.01,
    "full_delta_pd_min": -0.001,
    "full_delta_fa_ppm_max": 0.0,
    "hcval_delta_mIoU_min": 0.0,
    "hcval_delta_precision_min": 0.0,
    "hcval_delta_pd_min": -0.001,
    "hcval_delta_fa_ppm_max": 50.0
  }
}
```

---

## 3. Replace `model/cga_wrapper.py`

Replace the full file:

```python
"""Backbone-agnostic real CGA auxiliary-head wrapper."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.cga_aux import CGAAuxHead
from model.output_contract import validate_detector_output


class CGAWrapper(nn.Module):
    """Backbone + center/boundary/scale/peak CGA auxiliary heads.

    The wrapper owns only implementation metadata. It must not decide whether a
    result is paper evidence. Paper-evidence eligibility belongs to train/eval
    runner metadata.
    """

    DEFAULT_REGULARIZER_IMPL = "center_boundary_scale_peak"
    ALLOWED_REGULARIZER_IMPLS = {
        "center_boundary_scale_peak",
        "center_boundary_scale_peak_safe_bg_v2_1",
    }

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_name: str,
        feature_channels: int,
        aux_hidden_channels: int = 32,
        regularizer_impl: str = DEFAULT_REGULARIZER_IMPL,
        cga_variant: str = "v2",
    ) -> None:
        super().__init__()
        if regularizer_impl not in self.ALLOWED_REGULARIZER_IMPLS:
            raise ValueError(
                f"Unknown regularizer_impl={regularizer_impl!r}; "
                f"allowed={sorted(self.ALLOWED_REGULARIZER_IMPLS)}"
            )
        if cga_variant not in {"v2", "v2_1"}:
            raise ValueError(f"Unknown cga_variant={cga_variant!r}")

        self.backbone = backbone
        self.backbone_name = backbone_name
        self.regularizer_impl = str(regularizer_impl)
        self.cga_variant = str(cga_variant)
        self.cga_aux_head = CGAAuxHead(
            in_channels=int(feature_channels),
            hidden_channels=int(aux_hidden_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        output = self.backbone(x, **kwargs)
        output = validate_detector_output(
            output,
            backbone_name=self.backbone_name,
            require_feature=True,
        )
        feat = output["features"][0]
        aux = self.cga_aux_head(feat)

        output = dict(output)
        output["aux_outputs"] = aux
        output.update(aux)
        output.setdefault("regularizer_meta", {})
        output["regularizer_meta"].update(
            {
                "use_cga": True,
                "cga_variant": self.cga_variant,
                "regularizer_impl": self.regularizer_impl,
                "fallback_regularizer_used": False,
            }
        )
        return output
```

---

## 4. Patch `net.py`

Apply these exact edits.

### 4.1 Add CGA-v2.1 aliases

Replace the alias block with:

```python
_CGA_MODEL_ALIASES = {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}
_CGA21_MODEL_ALIASES = {"mshnetcga21", "mshnet_cga21", "cga-v2.1", "cga-v21", "cga21"}
_MSHNET_BASE_ALIASES = {"mshnet", "mshnetohem", "ohem"}
```

### 4.2 Replace `resolve_model_config`

```python
def resolve_model_config(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    use_cga: bool = False,
) -> tuple[str, bool]:
    """Resolve legacy model names into explicit backbone/CGA switches."""
    if model_name is not None:
        name = str(model_name).lower()
        if name in _CGA21_MODEL_ALIASES:
            return "mshnet", True
        if name in _CGA_MODEL_ALIASES:
            return "mshnet", True
        if name in _MSHNET_BASE_ALIASES:
            return "mshnet", False
        if name.endswith("_cga"):
            return name[: -len("_cga")], True
        if name in available_backbones():
            return name, bool(use_cga)
        return name, bool(use_cga)
    return str(backbone_name).lower(), bool(use_cga)
```

### 4.3 Replace `build_model`

```python
def build_model(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    input_channels: int = 1,
    use_cga: bool = False,
    aux_hidden_channels: int = 32,
    evidence_mode: str = "paper",
    legacy_model_factory: bool = False,
    cga_variant: str = "v2",
    regularizer_impl: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    if evidence_mode not in {"paper", "smoke"}:
        raise ValueError(f"Unknown evidence_mode={evidence_mode!r}")
    if evidence_mode == "paper" and kwargs.get("allow_fallback_regularizer", False):
        raise RuntimeError(
            "Fallback regularizer is forbidden for paper evidence. "
            "Run smoke tests under evidence_mode='smoke' only."
        )

    resolved_backbone, resolved_use_cga = resolve_model_config(
        model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
    )

    if cga_variant not in {"none", "v2", "v2_1"}:
        raise ValueError(f"Unknown cga_variant={cga_variant!r}")
    if resolved_use_cga and cga_variant == "none":
        raise ValueError("use_cga=True requires cga_variant to be 'v2' or 'v2_1'.")
    if (not resolved_use_cga) and cga_variant != "none":
        raise ValueError("use_cga=False requires cga_variant='none'.")

    if legacy_model_factory:
        if resolved_backbone == "mshnet" and resolved_use_cga and cga_variant in {"v2", "v2_1"}:
            return MSHNetCGA(
                input_channels=input_channels,
                aux_hidden_channels=int(aux_hidden_channels),
            )
        raise ValueError(
            "legacy_model_factory only supports the legacy MSHNetCGA path. "
            "Use the explicit adapter registry for other configurations."
        )

    builder = get_backbone_builder(resolved_backbone)
    backbone = builder(input_channels=input_channels)
    if not resolved_use_cga:
        return backbone

    feature_channels = getattr(backbone, "FEATURE_CHANNELS", None)
    if feature_channels is None:
        raise AttributeError(f"{resolved_backbone} adapter must define FEATURE_CHANNELS for CGAWrapper")

    if regularizer_impl is None:
        regularizer_impl = (
            "center_boundary_scale_peak_safe_bg_v2_1"
            if cga_variant == "v2_1"
            else "center_boundary_scale_peak"
        )

    return CGAWrapper(
        backbone,
        backbone_name=resolved_backbone,
        feature_channels=int(feature_channels),
        aux_hidden_channels=int(aux_hidden_channels),
        regularizer_impl=regularizer_impl,
        cga_variant=cga_variant,
    )
```

---

## 5. Patch `loss.py`

Append this code after the existing `CGALoss` class and before `MSHNetCGALoss = CGALoss`.

```python
@dataclass(frozen=True)
class CGAV21LossConfig:
    lambda_center: float = 0.05
    lambda_boundary: float = 0.03
    lambda_scale: float = 0.02
    lambda_peak: float = 0.03
    lambda_safe_bg: float = 0.03
    cga_start_epoch: int = 1
    cga_ramp_epochs: int = 40
    safe_bg_start_epoch: int = 1
    safe_bg_ramp_epochs: int = 40
    safe_bg_topk_ratio: float = 0.01
    safe_bg_ignore_radius: int = 3
    aux_ratio_cap: float = 0.15
    ohem_ratio: float = 0.01
    lambda_iou: float = 1.0
    warm_epoch: int = 5


def _validate_cga_v21_cfg(cfg: CGAV21LossConfig) -> None:
    if not (0.0 < float(cfg.safe_bg_topk_ratio) <= 1.0):
        raise ValueError("safe_bg_topk_ratio must be in (0, 1].")
    if int(cfg.safe_bg_ignore_radius) < 0:
        raise ValueError("safe_bg_ignore_radius must be >= 0.")
    if int(cfg.cga_start_epoch) < 0:
        raise ValueError("cga_start_epoch must be >= 0.")
    if int(cfg.cga_ramp_epochs) < 0:
        raise ValueError("cga_ramp_epochs must be >= 0.")
    if int(cfg.safe_bg_start_epoch) < 0:
        raise ValueError("safe_bg_start_epoch must be >= 0.")
    if int(cfg.safe_bg_ramp_epochs) < 0:
        raise ValueError("safe_bg_ramp_epochs must be >= 0.")
    if float(cfg.aux_ratio_cap) <= 0.0:
        raise ValueError("aux_ratio_cap must be > 0.")
    if float(cfg.lambda_iou) < 0.0:
        raise ValueError("lambda_iou must be >= 0.")
    for name in ("lambda_center", "lambda_boundary", "lambda_scale", "lambda_peak", "lambda_safe_bg"):
        if float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be >= 0.")


def _safe_background_mask(target: torch.Tensor, ref: torch.Tensor, ignore_radius: int) -> torch.Tensor:
    target = _resize_like(target, ref)
    fg = (target > 0.5).float()
    radius = int(ignore_radius)
    if radius > 0:
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(fg, kernel_size=kernel, stride=1, padding=radius)
    else:
        dilated = fg
    return (dilated <= 0.0).float()


def _topk_safe_bg_bce_with_logits(
    logits: torch.Tensor,
    safe_bg: torch.Tensor,
    *,
    topk_ratio: float,
) -> torch.Tensor:
    zeros = torch.zeros_like(logits)
    loss_map = F.binary_cross_entropy_with_logits(logits, zeros, reduction="none")
    losses: list[torch.Tensor] = []
    for b in range(logits.shape[0]):
        vals = loss_map[b][safe_bg[b] > 0.5].flatten()
        if vals.numel() == 0:
            continue
        k = max(1, int(vals.numel() * float(topk_ratio)))
        k = min(k, vals.numel())
        losses.append(torch.topk(vals, k=k, largest=True).values.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


class CGAV21Loss(nn.Module):
    """False-alarm-controlled CGA-v2.1 loss.

    This loss keeps the four CGA auxiliary heads unchanged and adds a
    safe-background hard-negative BCE term on the final logits. The regularizer
    is capped with a detached scale so the gradient direction remains available
    even when the raw regularizer exceeds the cap.
    """

    def __init__(
        self,
        cfg: CGAV21LossConfig | None = None,
        target_cfg: CGATargetConfig | None = None,
        *,
        strict_cga_heads: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg or CGAV21LossConfig()
        _validate_cga_v21_cfg(self.cfg)
        self.target_cfg = target_cfg or CGATargetConfig()
        self.strict_cga_heads = bool(strict_cga_heads)
        self.base_loss = MSHNetOHEMLoss(
            ohem_ratio=self.cfg.ohem_ratio,
            lambda_iou=self.cfg.lambda_iou,
            warm_epoch=self.cfg.warm_epoch,
        )

    @staticmethod
    def _bce(logit: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        if logit is None:
            return target.sum() * 0.0
        target = _resize_like(target, logit)
        return F.binary_cross_entropy_with_logits(logit, target)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        if not isinstance(output, dict):
            raise TypeError("CGA-v2.1 requires dict output with explicit auxiliary logits.")
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

        geom_aux_total = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )

        safe_bg = _safe_background_mask(target, final_logit, self.cfg.safe_bg_ignore_radius)
        loss_safe_bg = _topk_safe_bg_bce_with_logits(
            final_logit,
            safe_bg,
            topk_ratio=self.cfg.safe_bg_topk_ratio,
        )

        cga_w = _ramp_weight(epoch, self.cfg.cga_start_epoch, self.cfg.cga_ramp_epochs)
        safe_bg_w = _ramp_weight(epoch, self.cfg.safe_bg_start_epoch, self.cfg.safe_bg_ramp_epochs)

        reg_raw = cga_w * geom_aux_total + safe_bg_w * self.cfg.lambda_safe_bg * loss_safe_bg
        reg_cap = float(self.cfg.aux_ratio_cap) * base_total.detach().abs().clamp_min(1e-6)
        cap_scale = (reg_cap / reg_raw.detach().clamp_min(1e-6)).clamp(max=1.0)
        reg_capped = reg_raw * cap_scale
        total = base_total + reg_capped

        raw_over_base = reg_raw.detach() / base_total.detach().abs().clamp_min(1e-6)
        capped_over_base = reg_capped.detach() / base_total.detach().abs().clamp_min(1e-6)
        cap_active = (reg_raw.detach() > reg_cap).float()

        return {
            "total": total,
            "base_total": base_total.detach(),
            "ohem": base["ohem"],
            "soft_iou": base["soft_iou"],
            "scale": base["scale"],
            "cga_w": torch.tensor(cga_w, device=final_logit.device, dtype=final_logit.dtype),
            "safe_bg_w": torch.tensor(safe_bg_w, device=final_logit.device, dtype=final_logit.dtype),
            "cga_center": loss_center.detach(),
            "cga_boundary": loss_boundary.detach(),
            "cga_scale": loss_scale.detach(),
            "cga_peak": loss_peak.detach(),
            "geom_aux_total": geom_aux_total.detach(),
            "safe_bg": loss_safe_bg.detach(),
            "reg_raw": reg_raw.detach(),
            "reg_capped": reg_capped.detach(),
            "reg_raw_over_base": raw_over_base.detach(),
            "reg_capped_over_base": capped_over_base.detach(),
            "cap_active": cap_active.detach(),
            "cap_scale": cap_scale.detach(),
        }
```

Then replace the existing `build_loss(...)` function with:

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
    lambda_safe_bg: float = 0.03,
    safe_bg_start_epoch: int = 1,
    safe_bg_ramp_epochs: int = 40,
    safe_bg_topk_ratio: float = 0.01,
    safe_bg_ignore_radius: int = 3,
    aux_ratio_cap: float = 0.15,
    strict_cga_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    name_l = "" if name is None else str(name).lower()
    if use_cga is None:
        use_cga = "cga" in name_l

    if cga_variant not in {"none", "v2", "v2_1"}:
        raise ValueError(f"Unknown cga_variant={cga_variant!r}")
    if use_cga and cga_variant == "none":
        raise ValueError("use_cga=True requires cga_variant to be 'v2' or 'v2_1'.")
    if (not use_cga) and cga_variant != "none":
        raise ValueError("use_cga=False requires cga_variant='none'.")

    if mshnet_warm_epoch is None:
        mshnet_warm_epoch = int(warm_epoch if warm_epoch is not None else kwargs.get("warm_epoch", 5))
    if "start_epoch" in kwargs:
        cga_start_epoch = int(kwargs["start_epoch"])
    if "ramp_epochs" in kwargs:
        cga_ramp_epochs = int(kwargs["ramp_epochs"])

    if use_cga and cga_variant == "v2_1":
        cfg21 = CGAV21LossConfig(
            lambda_center=float(lambda_center),
            lambda_boundary=float(lambda_boundary),
            lambda_scale=float(lambda_scale),
            lambda_peak=float(lambda_peak),
            lambda_safe_bg=float(lambda_safe_bg),
            cga_start_epoch=int(cga_start_epoch),
            cga_ramp_epochs=int(cga_ramp_epochs),
            safe_bg_start_epoch=int(safe_bg_start_epoch),
            safe_bg_ramp_epochs=int(safe_bg_ramp_epochs),
            safe_bg_topk_ratio=float(safe_bg_topk_ratio),
            safe_bg_ignore_radius=int(safe_bg_ignore_radius),
            aux_ratio_cap=float(aux_ratio_cap),
            ohem_ratio=float(ohem_ratio),
            lambda_iou=float(lambda_iou),
            warm_epoch=int(mshnet_warm_epoch),
        )
        return CGAV21Loss(cfg21, strict_cga_heads=strict_cga_heads)

    if use_cga:
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

    return MSHNetOHEMLoss(
        ohem_ratio=float(ohem_ratio),
        lambda_iou=float(lambda_iou),
        warm_epoch=int(mshnet_warm_epoch),
    )
```

---

## 6. Patch `train.py`

### 6.1 Add imports

Add:

```python
from dataclasses import asdict
```

if not already present.

### 6.2 Add helper functions after `_output_fallback_regularizer_used`

```python
def _regularizer_impl_for_args(*, use_cga: bool, cga_variant: str) -> str:
    if not use_cga:
        return "none"
    if cga_variant == "v2_1":
        return "center_boundary_scale_peak_safe_bg_v2_1"
    if cga_variant == "v2":
        return "center_boundary_scale_peak"
    raise ValueError(f"Invalid cga_variant={cga_variant!r} for use_cga={use_cga!r}")


def _validate_cga_v21_train_args(args: argparse.Namespace) -> None:
    if args.safe_bg_topk_ratio <= 0.0 or args.safe_bg_topk_ratio > 1.0:
        raise ValueError("--safe_bg_topk_ratio must be in (0, 1].")
    if args.safe_bg_ignore_radius < 0:
        raise ValueError("--safe_bg_ignore_radius must be >= 0.")
    for name in ("cga_start_epoch", "cga_ramp_epochs", "safe_bg_start_epoch", "safe_bg_ramp_epochs"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be >= 0.")
    if args.aux_ratio_cap <= 0.0:
        raise ValueError("--aux_ratio_cap must be > 0.")
    if args.lambda_iou < 0.0:
        raise ValueError("--lambda_iou must be >= 0.")


def _write_runtime_args_json(args: argparse.Namespace, *, backbone_name: str, use_cga: bool, run_model_name: str) -> None:
    if not args.runtime_args_json:
        return
    regularizer_impl = _regularizer_impl_for_args(use_cga=use_cga, cga_variant=args.cga_variant)
    payload = {
        "root": str(Path.cwd()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": args.dataset_name,
        "model_name": args.model_name or run_model_name,
        "run_name": run_model_name,
        "backbone_name": backbone_name,
        "use_cga": bool(use_cga),
        "protocol_variant": args.protocol_variant,
        "cga_variant": args.cga_variant,
        "regularizer_impl": regularizer_impl,
        "evidence_mode": args.evidence_mode,
        "protocol": args.protocol,
        "p1_preflight_passed": bool(args.p1_preflight_passed),
        "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "threshold": float(args.threshold),
        "threshold_selection": args.threshold_selection,
        "strict_load": bool(args.strict_load),
        "ohem_ratio": float(args.ohem_ratio),
        "lambda_iou": float(args.lambda_iou),
        "mshnet_warm_epoch": int(args.mshnet_warm_epoch),
        "lambda_center": float(args.lambda_center),
        "lambda_boundary": float(args.lambda_boundary),
        "lambda_scale": float(args.lambda_scale),
        "lambda_peak": float(args.lambda_peak),
        "cga_start_epoch": int(args.cga_start_epoch),
        "cga_ramp_epochs": int(args.cga_ramp_epochs),
        "lambda_safe_bg": float(args.lambda_safe_bg),
        "safe_bg_start_epoch": int(args.safe_bg_start_epoch),
        "safe_bg_ramp_epochs": int(args.safe_bg_ramp_epochs),
        "safe_bg_topk_ratio": float(args.safe_bg_topk_ratio),
        "safe_bg_ignore_radius": int(args.safe_bg_ignore_radius),
        "aux_ratio_cap": float(args.aux_ratio_cap),
    }
    out = Path(args.runtime_args_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
```

### 6.3 Add parser arguments in `parse_args()`

Add these arguments after the existing CGA/loss arguments:

```python
    p.add_argument("--protocol_variant", default="v2", choices=["v2", "v2_1"])
    p.add_argument("--cga_variant", default=None, choices=["none", "v2", "v2_1"])
    p.add_argument("--lambda_iou", type=float, default=1.0)
    p.add_argument("--lambda_safe_bg", type=float, default=0.03)
    p.add_argument("--safe_bg_start_epoch", type=int, default=1)
    p.add_argument("--safe_bg_ramp_epochs", type=int, default=40)
    p.add_argument("--safe_bg_topk_ratio", type=float, default=0.01)
    p.add_argument("--safe_bg_ignore_radius", type=int, default=3)
    p.add_argument("--aux_ratio_cap", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold_selection", default="fixed_predeclared", choices=["fixed_predeclared"])
    p.add_argument("--strict_load", action="store_true", help="Training metadata flag; eval must use strict load.")
    p.add_argument("--runtime_args_json", default="")
```

### 6.4 Add this block after resolving `backbone_name, use_cga`

In `main()`, after:

```python
backbone_name, use_cga = resolve_model_config(...)
```

add:

```python
    if args.cga_variant is None:
        args.cga_variant = "v2" if use_cga else "none"
    if not use_cga:
        args.cga_variant = "none"
    if args.protocol_variant == "v2_1" and use_cga:
        args.cga_variant = "v2_1"
    if args.protocol_variant == "v2_1":
        _validate_cga_v21_train_args(args)
```

### 6.5 Patch `build_model(...)` call

Pass the new fields:

```python
    regularizer_impl = _regularizer_impl_for_args(use_cga=use_cga, cga_variant=args.cga_variant)
    model = build_model(
        model_name=args.model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        evidence_mode=args.evidence_mode,
        input_channels=1,
        aux_hidden_channels=args.aux_hidden_channels,
        allow_fallback_regularizer=args.allow_fallback_regularizer,
        cga_variant=args.cga_variant,
        regularizer_impl=regularizer_impl,
    ).to(device)
```

### 6.6 Patch `build_loss(...)` call

Pass these fields:

```python
    criterion = build_loss(
        args.model_name or run_model_name,
        use_cga=use_cga,
        cga_variant=args.cga_variant,
        ohem_ratio=args.ohem_ratio,
        lambda_iou=args.lambda_iou,
        mshnet_warm_epoch=args.mshnet_warm_epoch,
        cga_start_epoch=args.cga_start_epoch,
        cga_ramp_epochs=args.cga_ramp_epochs,
        lambda_center=args.lambda_center,
        lambda_boundary=args.lambda_boundary,
        lambda_scale=args.lambda_scale,
        lambda_peak=args.lambda_peak,
        lambda_safe_bg=args.lambda_safe_bg,
        safe_bg_start_epoch=args.safe_bg_start_epoch,
        safe_bg_ramp_epochs=args.safe_bg_ramp_epochs,
        safe_bg_topk_ratio=args.safe_bg_topk_ratio,
        safe_bg_ignore_radius=args.safe_bg_ignore_radius,
        aux_ratio_cap=args.aux_ratio_cap,
        strict_cga_heads=use_cga,
    )
```

### 6.7 Write runtime args JSON

After `run_model_name = _run_model_name(...)`, add:

```python
    _write_runtime_args_json(args, backbone_name=backbone_name, use_cga=use_cga, run_model_name=run_model_name)
```

---

## 7. Replace `test.py`

Replace the full file with this code:

```python
"""Inference/test entry point for MSHNetCGA and MSHNetOHEM."""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import TestSetLoader
from metrics import IRSTDMetrics
from net import build_model, resolve_model_config


_FINAL_LOGIT_KEYS = ("final_logit", "final_logits", "base_logits", "base_logit", "logits")
_ALLOWED_STATE_PREFIXES = ("module.", "model.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Test MSHNet/CGA-v2/v2.1")
    p.add_argument("--model_name", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--backbone_name", default="mshnet", choices=["mshnet", "dnanet", "alcnet", "acm", "isnet"])
    p.add_argument("--use_cga", action="store_true")
    p.add_argument("--cga_variant", default=None, choices=["none", "v2", "v2_1"])
    p.add_argument("--regularizer_impl", default=None)
    p.add_argument("--evidence_mode", default="paper", choices=["paper", "smoke"])
    p.add_argument("--protocol", default="controlled", choices=["controlled", "official"])
    p.add_argument("--protocol_variant", default="v2", choices=["v2", "v2_1"])
    p.add_argument("--dataset_dir", default="datasets")
    p.add_argument("--train_dataset_name", default=None)
    p.add_argument("--dataset_name", default="NUDT-SIRST")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold_selection", default="fixed_predeclared", choices=["fixed_predeclared"])
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--output_dir", default="results/official")
    return p.parse_args()


def _run_model_name(model_name: str | None, run_name: str | None, backbone_name: str, use_cga: bool) -> str:
    if run_name:
        return str(run_name)
    if model_name:
        return str(model_name)
    return f"{backbone_name}_cga" if use_cga else backbone_name


def _regularizer_impl_for_eval(*, use_cga: bool, cga_variant: str, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    if not use_cga:
        return "none"
    if cga_variant == "v2_1":
        return "center_boundary_scale_peak_safe_bg_v2_1"
    return "center_boundary_scale_peak"


def crop_to_size(arr: torch.Tensor, size) -> torch.Tensor:
    h, w = int(size[0]), int(size[1])
    return arr[..., :h, :w]


def first_size(size) -> tuple[int, int]:
    if torch.is_tensor(size):
        if size.ndim == 2:
            size = size[0]
        return int(size[0].item()), int(size[1].item())
    if isinstance(size, (list, tuple)) and len(size) == 2 and torch.is_tensor(size[0]):
        return int(size[0][0].item()), int(size[1][0].item())
    if isinstance(size, (list, tuple)) and len(size) == 1:
        return first_size(size[0])
    return int(size[0]), int(size[1])


def save_prob_png(prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8)).save(path)


def _select_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict) and value:
                return value
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            tensorish = [torch.is_tensor(v) for v in ckpt.values()]
            if tensorish and sum(tensorish) >= max(1, len(tensorish) // 2):
                return ckpt
    raise TypeError("Could not locate a checkpoint state_dict under state_dict/model_state_dict/model or a raw state dict.")


def _strip_prefix_if_all_keys(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor] | None:
    keys = list(state.keys())
    if not keys:
        return None
    if not all(k.startswith(prefix) for k in keys):
        return None
    stripped = [k[len(prefix):] for k in keys]
    if any(not k for k in stripped):
        return None
    if len(set(stripped)) != len(stripped):
        raise RuntimeError(f"Prefix stripping {prefix!r} would create duplicate keys.")
    return OrderedDict((k[len(prefix):], v) for k, v in state.items())


def normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    for prefix in _ALLOWED_STATE_PREFIXES:
        stripped = _strip_prefix_if_all_keys(state, prefix)
        if stripped is not None:
            return stripped, f"strip_all:{prefix}"
    return state, "none"


def audited_load_state_dict(model: torch.nn.Module, checkpoint_path: str, *, map_location, strict_load: bool) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = _select_state_dict(ckpt)
    state, normalization = normalize_state_dict_keys(state)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    strict_load_pass = not missing and not unexpected
    if strict_load and not strict_load_pass:
        raise RuntimeError(
            "Strict checkpoint load failed after whitelist normalization. "
            f"normalization={normalization}; missing={missing}; unexpected={unexpected}"
        )
    if strict_load:
        model.load_state_dict(state, strict=True)
    checkpoint_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "strict_load": bool(strict_load),
        "strict_load_pass": bool(strict_load_pass),
        "state_dict_normalization": normalization,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def extract_final_logit_with_trace(output: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    trace: dict[str, Any] = {
        "output_type": type(output).__name__,
        "logit_source": None,
        "aux_used_for_prediction": False,
    }
    if isinstance(output, dict):
        trace["output_keys"] = sorted(str(k) for k in output.keys())
        for key in _FINAL_LOGIT_KEYS:
            if key in output and torch.is_tensor(output[key]):
                trace["logit_source"] = key
                trace["aux_used_for_prediction"] = key.startswith("cga_") or key == "aux_outputs"
                return output[key], trace
        raise KeyError(f"Could not find final logit in output keys: {sorted(output.keys())}")
    if torch.is_tensor(output):
        trace["logit_source"] = "tensor_output"
        return output, trace
    raise TypeError("Paper-mode eval requires dict or tensor output; tuple/list silent selection is forbidden.")


def _tensor_range(x: torch.Tensor) -> list[float]:
    return [float(x.min().detach().cpu().item()), float(x.max().detach().cpu().item())]


def main() -> None:
    args = parse_args()
    train_name = args.train_dataset_name or args.dataset_name
    backbone_name, use_cga = resolve_model_config(
        args.model_name,
        backbone_name=args.backbone_name,
        use_cga=args.use_cga,
    )
    if args.cga_variant is None:
        args.cga_variant = "v2" if use_cga else "none"
    if not use_cga:
        args.cga_variant = "none"
    if args.protocol_variant == "v2_1" and use_cga:
        args.cga_variant = "v2_1"
    regularizer_impl = _regularizer_impl_for_eval(
        use_cga=use_cga,
        cga_variant=args.cga_variant,
        explicit=args.regularizer_impl,
    )
    run_model_name = _run_model_name(args.model_name, args.run_name, backbone_name, use_cga)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=args.model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        evidence_mode=args.evidence_mode,
        cga_variant=args.cga_variant,
        regularizer_impl=regularizer_impl,
    ).to(device)

    load_audit = audited_load_state_dict(model, args.checkpoint, map_location=device, strict_load=args.strict_load)
    model.eval()

    ds = TestSetLoader(args.dataset_dir, train_name, args.dataset_name, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    pred_dir = Path(args.output_dir) / run_model_name / f"seed{args.seed}" / args.dataset_name / args.split / "predictions"
    metric = IRSTDMetrics(threshold=args.threshold)
    first_eval_trace: dict[str, Any] | None = None

    with torch.no_grad():
        for img, mask, size, image_id in loader:
            img = img.float().to(device)
            forward_kwargs = {}
            if backbone_name == "mshnet":
                forward_kwargs["mshnet_warm_flag"] = False
            output = model(img, **forward_kwargs)
            logit, trace = extract_final_logit_with_trace(output)
            if first_eval_trace is None:
                prob_first = torch.sigmoid(logit.detach())
                first_eval_trace = dict(trace)
                first_eval_trace.update(
                    {
                        "logits_shape": list(logit.shape),
                        "logits_minmax": _tensor_range(logit),
                        "sigmoid_minmax": _tensor_range(prob_first),
                        "threshold": float(args.threshold),
                        "positive_pixels_after_threshold": int((prob_first > args.threshold).sum().detach().cpu().item()),
                    }
                )
                if first_eval_trace.get("aux_used_for_prediction"):
                    raise RuntimeError(f"Eval attempted to use auxiliary output as prediction: {first_eval_trace}")

            prob = torch.sigmoid(logit).cpu()
            original_size = first_size(size)
            prob = crop_to_size(prob, original_size).squeeze().numpy()
            gt = crop_to_size(mask.float(), original_size).squeeze().numpy()
            metric.update(prob, gt, size=original_size)
            save_prob_png(prob, pred_dir / f"{image_id[0]}.png")

    summary = metric.get()
    checkpoint_epoch = int(load_audit["checkpoint_epoch"])
    summary.update(
        {
            "model": run_model_name,
            "run_name": run_model_name,
            "backbone": backbone_name,
            "use_cga": bool(use_cga),
            "protocol_variant": args.protocol_variant,
            "cga_variant": args.cga_variant,
            "regularizer_impl": regularizer_impl,
            "evidence_mode": args.evidence_mode,
            "protocol": args.protocol,
            "train_dataset": train_name,
            "dataset": args.dataset_name,
            "split": args.split,
            "seed": args.seed,
            "epoch": checkpoint_epoch,
            "checkpoint_epoch": checkpoint_epoch,
            "threshold": args.threshold,
            "threshold_selection": args.threshold_selection,
            "checkpoint": str(args.checkpoint),
            "prediction_dir": str(pred_dir),
            "eval_trace": first_eval_trace or {},
            **load_audit,
        }
    )

    out_path = Path(args.output_dir) / run_model_name / f"seed{args.seed}" / args.dataset_name / args.split / "summary_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.split == "test":
        compat_path = Path(args.output_dir) / run_model_name / f"seed{args.seed}" / args.dataset_name / "summary_metrics.json"
        compat_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
```

---

## 8. Add protocol-lock checker

Create:

```text
tools/official/check_cga_v21_protocol_lock.py
```

Exact code:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOSS_FIELDS = (
    "ohem_ratio",
    "lambda_iou",
    "mshnet_warm_epoch",
    "lambda_center",
    "lambda_boundary",
    "lambda_scale",
    "lambda_peak",
    "cga_start_epoch",
    "cga_ramp_epochs",
    "lambda_safe_bg",
    "safe_bg_start_epoch",
    "safe_bg_ramp_epochs",
    "safe_bg_topk_ratio",
    "safe_bg_ignore_radius",
    "aux_ratio_cap",
)

ROOT_FIELDS = (
    "root",
    "dataset_dir",
    "dataset_name",
    "seed",
    "epochs",
    "threshold",
    "threshold_selection",
    "strict_load",
    "evidence_mode",
    "protocol",
    "protocol_variant",
    "p1_preflight_passed",
    "p1a_hcval_source_audit_passed",
    "backbone_name",
)

ROLE_FIELDS = (
    "model_name",
    "run_name",
    "use_cga",
    "cga_variant",
    "regularizer_impl",
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def equal_value(key: str, got: Any, expected: Any) -> bool:
    if key in {"root", "dataset_dir"}:
        return norm_path(got) == norm_path(expected)
    if isinstance(expected, bool):
        return bool(got) == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(got) == expected
    if isinstance(expected, float):
        return abs(float(got) - expected) <= 1e-12
    return str(got) == str(expected)


def compare_field(mismatches: list[dict[str, Any]], *, key: str, got: Any, expected: Any, scope: str) -> None:
    if not equal_value(key, got, expected):
        mismatches.append({"scope": scope, "field": key, "got": got, "expected": expected})


def main() -> None:
    p = argparse.ArgumentParser("Check CGA-v2.1 protocol lock against runtime args")
    p.add_argument("--protocol_lock", required=True)
    p.add_argument("--runtime_args_json", required=True)
    p.add_argument("--role", required=True, choices=["baseline", "candidate"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = load_json(args.protocol_lock)
    runtime = load_json(args.runtime_args_json)
    mismatches: list[dict[str, Any]] = []

    for key in ROOT_FIELDS:
        compare_field(mismatches, key=key, got=runtime.get(key), expected=lock.get(key), scope="root")

    for key in LOSS_FIELDS:
        compare_field(mismatches, key=key, got=runtime.get(key), expected=lock["loss"].get(key), scope="loss")

    role_lock = lock[args.role]
    for key in ROLE_FIELDS:
        compare_field(mismatches, key=key, got=runtime.get(key), expected=role_lock.get(key), scope=args.role)

    # Additional hard assertions that are too important to leave implicit.
    hard_expected = {
        "strict_load": True,
        "threshold_selection": "fixed_predeclared",
        "evidence_mode": "paper",
        "protocol": "controlled",
        "protocol_variant": "v2_1",
    }
    for key, expected in hard_expected.items():
        compare_field(mismatches, key=key, got=runtime.get(key), expected=expected, scope="hard")

    if args.role == "baseline":
        expected_role = {"use_cga": False, "cga_variant": "none", "regularizer_impl": "none"}
    else:
        expected_role = {
            "use_cga": True,
            "cga_variant": "v2_1",
            "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1",
        }
    for key, expected in expected_role.items():
        compare_field(mismatches, key=key, got=runtime.get(key), expected=expected, scope="hard_role")

    result = {
        "checker": "check_cga_v21_protocol_lock",
        "role": args.role,
        "pass": len(mismatches) == 0,
        "mismatches": mismatches,
        "protocol_lock": str(Path(args.protocol_lock).resolve()),
        "runtime_args_json": str(Path(args.runtime_args_json).resolve()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
```

---

## 9. Add v2.1 summarizer

Create:

```text
tools/official/summarize_cga_v21_one_seed.py
```

Exact code:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_CANDIDATE_IMPL = "center_boundary_scale_peak_safe_bg_v2_1"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], key: str) -> float:
    if key in summary:
        return float(summary[key])
    for alt in (key, key.lower(), key.upper()):
        if alt in summary:
            return float(summary[alt])
    raise KeyError(f"Missing metric {key!r}; available={sorted(summary.keys())}")


def audit_summary_meta(
    summary: dict[str, Any],
    *,
    role: str,
    dataset_name: str,
    seed: int,
    epoch: int,
    split: str,
) -> list[str]:
    errors: list[str] = []

    expected = {
        "dataset": dataset_name,
        "seed": seed,
        "checkpoint_epoch": epoch,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "evidence_mode": "paper",
        "protocol": "controlled",
        "protocol_variant": "v2_1",
        "split": split,
        "strict_load": True,
        "strict_load_pass": True,
    }
    for key, value in expected.items():
        got = summary.get(key)
        if isinstance(value, float):
            if abs(float(got) - value) > 1e-12:
                errors.append(f"{role}:{split}:{key} got={got!r} expected={value!r}")
        else:
            if got != value:
                errors.append(f"{role}:{split}:{key} got={got!r} expected={value!r}")

    if role == "baseline":
        role_expected = {
            "use_cga": False,
            "cga_variant": "none",
            "regularizer_impl": "none",
            "model": "MSHNetOHEM",
            "run_name": "MSHNetOHEM",
        }
    else:
        role_expected = {
            "use_cga": True,
            "cga_variant": "v2_1",
            "regularizer_impl": REQUIRED_CANDIDATE_IMPL,
            "model": "MSHNetCGA21",
            "run_name": "MSHNetCGA21",
        }
    for key, value in role_expected.items():
        got = summary.get(key)
        if got != value:
            errors.append(f"{role}:{split}:{key} got={got!r} expected={value!r}")

    if summary.get("missing_keys") not in ([], None):
        errors.append(f"{role}:{split}:missing_keys not empty: {summary.get('missing_keys')}")
    if summary.get("unexpected_keys") not in ([], None):
        errors.append(f"{role}:{split}:unexpected_keys not empty: {summary.get('unexpected_keys')}")
    trace = summary.get("eval_trace") or {}
    if trace.get("aux_used_for_prediction") is not False:
        errors.append(f"{role}:{split}:aux_used_for_prediction={trace.get('aux_used_for_prediction')!r}")
    if trace.get("logit_source") not in {"final_logit", "final_logits", "base_logits", "base_logit", "logits", "tensor_output"}:
        errors.append(f"{role}:{split}:invalid logit_source={trace.get('logit_source')!r}")
    return errors


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return metric(candidate, key) - metric(baseline, key)


def summarize_split(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": {k: metric(baseline, k) for k in ("mIoU", "Precision", "Pd", "FA_ppm")},
        "candidate": {k: metric(candidate, k) for k in ("mIoU", "Precision", "Pd", "FA_ppm")},
        "delta": {k: delta(candidate, baseline, k) for k in ("mIoU", "Precision", "Pd", "FA_ppm")},
    }


def main() -> None:
    p = argparse.ArgumentParser("Summarize CGA-v2.1 one-seed paired run")
    p.add_argument("--baseline_full", required=True)
    p.add_argument("--candidate_full", required=True)
    p.add_argument("--baseline_hcval", required=True)
    p.add_argument("--candidate_hcval", required=True)
    p.add_argument("--protocol_lock", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = load_json(args.protocol_lock)
    baseline_full = load_json(args.baseline_full)
    candidate_full = load_json(args.candidate_full)
    baseline_hcval = load_json(args.baseline_hcval)
    candidate_hcval = load_json(args.candidate_hcval)

    dataset_name = lock["dataset_name"]
    seed = int(lock["seed"])
    epoch = int(lock["checkpoint_epoch"])
    meta_errors: list[str] = []
    meta_errors += audit_summary_meta(baseline_full, role="baseline", dataset_name=dataset_name, seed=seed, epoch=epoch, split="test")
    meta_errors += audit_summary_meta(candidate_full, role="candidate", dataset_name=dataset_name, seed=seed, epoch=epoch, split="test")
    meta_errors += audit_summary_meta(baseline_hcval, role="baseline", dataset_name=dataset_name, seed=seed, epoch=epoch, split="hcval")
    meta_errors += audit_summary_meta(candidate_hcval, role="candidate", dataset_name=dataset_name, seed=seed, epoch=epoch, split="hcval")

    full = summarize_split(candidate_full, baseline_full)
    hcval = summarize_split(candidate_hcval, baseline_hcval)
    gate = lock["gate"]

    full_gate_pass = bool(
        full["delta"]["mIoU"] >= float(gate["full_delta_mIoU_min"])
        and full["delta"]["Precision"] >= float(gate["full_delta_precision_min"])
        and full["delta"]["Pd"] >= float(gate["full_delta_pd_min"])
        and full["delta"]["FA_ppm"] <= float(gate["full_delta_fa_ppm_max"])
    )
    hcval_gate_pass = bool(
        hcval["delta"]["mIoU"] >= float(gate["hcval_delta_mIoU_min"])
        and hcval["delta"]["Precision"] >= float(gate["hcval_delta_precision_min"])
        and hcval["delta"]["Pd"] >= float(gate["hcval_delta_pd_min"])
        and hcval["delta"]["FA_ppm"] <= float(gate["hcval_delta_fa_ppm_max"])
    )
    metadata_gate_pass = len(meta_errors) == 0
    gate_pass = bool(full_gate_pass and hcval_gate_pass and metadata_gate_pass)

    result = {
        "gate": "Gate-CGA-v2.1-P2-seed42-from-zero-paired",
        "decision_rule_predeclared": True,
        "protocol_variant": "v2_1",
        "baseline": "MSHNetOHEM",
        "candidate": "MSHNetCGA21",
        "dataset_name": dataset_name,
        "seed": seed,
        "epoch": epoch,
        "threshold": float(lock["threshold"]),
        "threshold_selection": lock["threshold_selection"],
        "full": full,
        "hcval": hcval,
        "pass_conditions": {
            "metadata_gate_pass": metadata_gate_pass,
            "full_gate_pass": full_gate_pass,
            "hcval_gate_pass": hcval_gate_pass,
        },
        "metadata_errors": meta_errors,
        "gate_pass": gate_pass,
        "decision": "P2_1_PASS_SEED42_ALLOW_SEED43_44" if gate_pass else "P2_1_FAIL_STOP_NO_SEED43_44",
        "can_run_seed43_44": gate_pass,
        "can_claim_positive_cga21": gate_pass,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
```

---

## 10. Add complete runner

Create:

```text
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

Exact code:

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
RUN_ID=${RUN_ID:-seed42_protocol_locked}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/official_cga_v21/${RUN_ID}}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-${ROOT}/docs/internal/cga_v2_1/protocol_lock.json}
AUDIT_DIR=${AUDIT_DIR:-${ROOT}/docs/internal/cga_v2_1/gate_p2_seed42_${DATASET_NAME}/${RUN_ID}}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  echo "ERROR: FORCE_TRAIN=1 is forbidden for CGA-v2.1 paper evidence. Use a fresh OUTPUT_DIR/RUN_ID." >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  echo "Use a fresh RUN_ID to preserve from-zero evidence." >&2
  exit 2
fi

if [[ ! -f "${PROTOCOL_LOCK}" ]]; then
  echo "ERROR: missing protocol lock: ${PROTOCOL_LOCK}" >&2
  exit 2
fi

mkdir -p "${AUDIT_DIR}"

BASE_RUNNER_ARGS="${AUDIT_DIR}/runner_runtime_args_baseline.json"
CAND_RUNNER_ARGS="${AUDIT_DIR}/runner_runtime_args_candidate.json"
BASE_TRAIN_ARGS="${AUDIT_DIR}/train_runtime_args_baseline.json"
CAND_TRAIN_ARGS="${AUDIT_DIR}/train_runtime_args_candidate.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path
root = Path("${ROOT}").resolve()
dataset_dir = Path("${DATASET_DIR}").resolve()
common = {
  "root": str(root),
  "dataset_dir": str(dataset_dir),
  "dataset_name": "${DATASET_NAME}",
  "seed": int("${SEED}"),
  "epochs": int("${EPOCHS}"),
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "strict_load": True,
  "evidence_mode": "paper",
  "protocol": "controlled",
  "protocol_variant": "v2_1",
  "p1_preflight_passed": True,
  "p1a_hcval_source_audit_passed": True,
  "backbone_name": "mshnet",
  "ohem_ratio": 0.01,
  "lambda_iou": 1.0,
  "mshnet_warm_epoch": 5,
  "lambda_center": 0.05,
  "lambda_boundary": 0.03,
  "lambda_scale": 0.02,
  "lambda_peak": 0.03,
  "cga_start_epoch": 1,
  "cga_ramp_epochs": 40,
  "lambda_safe_bg": 0.03,
  "safe_bg_start_epoch": 1,
  "safe_bg_ramp_epochs": 40,
  "safe_bg_topk_ratio": 0.01,
  "safe_bg_ignore_radius": 3,
  "aux_ratio_cap": 0.15
}
baseline = dict(common, model_name="MSHNetOHEM", run_name="MSHNetOHEM", use_cga=False, cga_variant="none", regularizer_impl="none")
candidate = dict(common, model_name="MSHNetCGA21", run_name="MSHNetCGA21", use_cga=True, cga_variant="v2_1", regularizer_impl="center_boundary_scale_peak_safe_bg_v2_1")
Path("${BASE_RUNNER_ARGS}").write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
Path("${CAND_RUNNER_ARGS}").write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")
PY

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${BASE_RUNNER_ARGS}" \
  --role baseline \
  --output "${AUDIT_DIR}/protocol_lock_baseline_pretrain.json"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${CAND_RUNNER_ARGS}" \
  --role candidate \
  --output "${AUDIT_DIR}/protocol_lock_candidate_pretrain.json"

"${PYTHON}" train.py \
  --model_name MSHNetOHEM \
  --backbone_name mshnet \
  --evidence_mode paper \
  --protocol controlled \
  --protocol_variant v2_1 \
  --cga_variant none \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size 8 \
  --patch_size 256 \
  --num_workers 4 \
  --lr 5e-4 \
  --ohem_ratio 0.01 \
  --lambda_iou 1.0 \
  --mshnet_warm_epoch 5 \
  --threshold 0.5 \
  --threshold_selection fixed_predeclared \
  --strict_load \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --runtime_args_json "${BASE_TRAIN_ARGS}" \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON}" train.py \
  --model_name MSHNetCGA21 \
  --backbone_name mshnet \
  --use_cga \
  --evidence_mode paper \
  --protocol controlled \
  --protocol_variant v2_1 \
  --cga_variant v2_1 \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch_size 8 \
  --patch_size 256 \
  --num_workers 4 \
  --lr 5e-4 \
  --ohem_ratio 0.01 \
  --lambda_iou 1.0 \
  --mshnet_warm_epoch 5 \
  --lambda_center 0.05 \
  --lambda_boundary 0.03 \
  --lambda_scale 0.02 \
  --lambda_peak 0.03 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 40 \
  --lambda_safe_bg 0.03 \
  --safe_bg_start_epoch 1 \
  --safe_bg_ramp_epochs 40 \
  --safe_bg_topk_ratio 0.01 \
  --safe_bg_ignore_radius 3 \
  --aux_ratio_cap 0.15 \
  --threshold 0.5 \
  --threshold_selection fixed_predeclared \
  --strict_load \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --runtime_args_json "${CAND_TRAIN_ARGS}" \
  --output_dir "${OUTPUT_DIR}"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${BASE_TRAIN_ARGS}" \
  --role baseline \
  --output "${AUDIT_DIR}/protocol_lock_baseline_posttrain.json"

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${CAND_TRAIN_ARGS}" \
  --role candidate \
  --output "${AUDIT_DIR}/protocol_lock_candidate_posttrain.json"

BASE_CKPT="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/MSHNetOHEM_${EPOCHS}.pth.tar"
CAND_CKPT="${OUTPUT_DIR}/MSHNetCGA21/seed${SEED}/${DATASET_NAME}/MSHNetCGA21_${EPOCHS}.pth.tar"

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "ERROR: missing baseline checkpoint: ${BASE_CKPT}" >&2
  exit 3
fi
if [[ ! -f "${CAND_CKPT}" ]]; then
  echo "ERROR: missing candidate checkpoint: ${CAND_CKPT}" >&2
  exit 3
fi

for SPLIT in test hcval; do
  "${PYTHON}" test.py \
    --model_name MSHNetOHEM \
    --run_name MSHNetOHEM \
    --backbone_name mshnet \
    --evidence_mode paper \
    --protocol controlled \
    --protocol_variant v2_1 \
    --cga_variant none \
    --regularizer_impl none \
    --dataset_dir "${DATASET_DIR}" \
    --train_dataset_name "${DATASET_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --checkpoint "${BASE_CKPT}" \
    --threshold 0.5 \
    --threshold_selection fixed_predeclared \
    --strict_load \
    --output_dir "${OUTPUT_DIR}"

  "${PYTHON}" test.py \
    --model_name MSHNetCGA21 \
    --run_name MSHNetCGA21 \
    --backbone_name mshnet \
    --use_cga \
    --evidence_mode paper \
    --protocol controlled \
    --protocol_variant v2_1 \
    --cga_variant v2_1 \
    --regularizer_impl center_boundary_scale_peak_safe_bg_v2_1 \
    --dataset_dir "${DATASET_DIR}" \
    --train_dataset_name "${DATASET_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --checkpoint "${CAND_CKPT}" \
    --threshold 0.5 \
    --threshold_selection fixed_predeclared \
    --strict_load \
    --output_dir "${OUTPUT_DIR}"
done

"${PYTHON}" tools/official/summarize_cga_v21_one_seed.py \
  --baseline_full "${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json" \
  --candidate_full "${OUTPUT_DIR}/MSHNetCGA21/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json" \
  --baseline_hcval "${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json" \
  --candidate_hcval "${OUTPUT_DIR}/MSHNetCGA21/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json" \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --output "${AUDIT_DIR}/P2_1_seed42_summary.json"
```

Make executable:

```bash
chmod +x /home/ly/AAAI/CGA-main/scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

---

## 11. Add tests

Create:

```text
tests/test_cga_v21_loss_protocol_summary_strict.py
```

Exact code:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from loss import CGAV21Loss, CGAV21LossConfig, build_loss
from test import normalize_state_dict_keys
from tools.official.summarize_cga_v21_one_seed import audit_summary_meta


def _fake_output(batch=2, h=32, w=32):
    x = torch.randn(batch, 1, h, w, requires_grad=True)
    return {
        "logits": x,
        "cga_center_logit": torch.randn(batch, 1, h, w, requires_grad=True),
        "cga_boundary_logit": torch.randn(batch, 1, h, w, requires_grad=True),
        "cga_scale_logit": torch.randn(batch, 1, h, w, requires_grad=True),
        "cga_peak_logit": torch.randn(batch, 1, h, w, requires_grad=True),
        "masks": [],
    }


def test_cga_v21_loss_outputs_ratio_cap_metrics():
    loss = CGAV21Loss(
        CGAV21LossConfig(
            lambda_safe_bg=0.03,
            safe_bg_topk_ratio=0.05,
            safe_bg_ignore_radius=2,
            aux_ratio_cap=0.15,
        )
    )
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 12:15, 12:15] = 1.0
    out = loss(_fake_output(), target, epoch=400)
    assert torch.isfinite(out["total"])
    assert "reg_raw_over_base" in out
    assert "reg_capped_over_base" in out
    assert float(out["reg_capped_over_base"]) <= 0.150001
    out["total"].backward()


def test_cga_v21_invalid_safe_bg_topk_ratio():
    with pytest.raises(ValueError):
        CGAV21Loss(CGAV21LossConfig(safe_bg_topk_ratio=0.0))
    with pytest.raises(ValueError):
        CGAV21Loss(CGAV21LossConfig(safe_bg_topk_ratio=1.5))


def test_build_loss_requires_none_variant_for_baseline():
    with pytest.raises(ValueError):
        build_loss("MSHNetOHEM", use_cga=False, cga_variant="v2_1")
    baseline = build_loss("MSHNetOHEM", use_cga=False, cga_variant="none")
    assert baseline is not None


def test_prefix_normalization_only_when_all_keys_share_prefix():
    state = {"module.a": torch.tensor(1), "module.b": torch.tensor(2)}
    stripped, mode = normalize_state_dict_keys(state)
    assert mode == "strip_all:module."
    assert sorted(stripped.keys()) == ["a", "b"]

    mixed = {"module.a": torch.tensor(1), "b": torch.tensor(2)}
    not_stripped, mode2 = normalize_state_dict_keys(mixed)
    assert mode2 == "none"
    assert sorted(not_stripped.keys()) == ["b", "module.a"]


def _summary(role: str, split: str, *, fa_ppm=0.0, precision=0.9):
    is_candidate = role == "candidate"
    return {
        "dataset": "NUDT-SIRST",
        "seed": 42,
        "checkpoint_epoch": 400,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "evidence_mode": "paper",
        "protocol": "controlled",
        "protocol_variant": "v2_1",
        "split": split,
        "strict_load": True,
        "strict_load_pass": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "use_cga": bool(is_candidate),
        "cga_variant": "v2_1" if is_candidate else "none",
        "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1" if is_candidate else "none",
        "model": "MSHNetCGA21" if is_candidate else "MSHNetOHEM",
        "run_name": "MSHNetCGA21" if is_candidate else "MSHNetOHEM",
        "eval_trace": {"aux_used_for_prediction": False, "logit_source": "logits"},
        "mIoU": 0.9,
        "Precision": precision,
        "Pd": 0.9,
        "FA_ppm": fa_ppm,
    }


def test_summarizer_meta_requires_baseline_none_variant():
    s = _summary("baseline", "test")
    assert audit_summary_meta(s, role="baseline", dataset_name="NUDT-SIRST", seed=42, epoch=400, split="test") == []
    s["cga_variant"] = "v2_1"
    errs = audit_summary_meta(s, role="baseline", dataset_name="NUDT-SIRST", seed=42, epoch=400, split="test")
    assert any("cga_variant" in e for e in errs)


def test_summarizer_meta_requires_strict_load_pass():
    s = _summary("candidate", "hcval")
    s["strict_load_pass"] = False
    errs = audit_summary_meta(s, role="candidate", dataset_name="NUDT-SIRST", seed=42, epoch=400, split="hcval")
    assert any("strict_load_pass" in e for e in errs)
```

Create:

```text
tests/test_cga_v21_protocol_checker.py
```

Exact code:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_protocol_checker_catches_strict_load_mismatch(tmp_path: Path):
    lock = {
        "root": str(tmp_path),
        "dataset_dir": str(tmp_path / "datasets"),
        "dataset_name": "NUDT-SIRST",
        "seed": 42,
        "epochs": 400,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "strict_load": True,
        "evidence_mode": "paper",
        "protocol": "controlled",
        "protocol_variant": "v2_1",
        "p1_preflight_passed": True,
        "p1a_hcval_source_audit_passed": True,
        "backbone_name": "mshnet",
        "loss": {
            "ohem_ratio": 0.01,
            "lambda_iou": 1.0,
            "mshnet_warm_epoch": 5,
            "lambda_center": 0.05,
            "lambda_boundary": 0.03,
            "lambda_scale": 0.02,
            "lambda_peak": 0.03,
            "cga_start_epoch": 1,
            "cga_ramp_epochs": 40,
            "lambda_safe_bg": 0.03,
            "safe_bg_start_epoch": 1,
            "safe_bg_ramp_epochs": 40,
            "safe_bg_topk_ratio": 0.01,
            "safe_bg_ignore_radius": 3,
            "aux_ratio_cap": 0.15,
        },
        "baseline": {"model_name": "MSHNetOHEM", "run_name": "MSHNetOHEM", "use_cga": False, "cga_variant": "none", "regularizer_impl": "none"},
        "candidate": {"model_name": "MSHNetCGA21", "run_name": "MSHNetCGA21", "use_cga": True, "cga_variant": "v2_1", "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"},
    }
    runtime = {
        **{k: lock[k] for k in ("root", "dataset_dir", "dataset_name", "seed", "epochs", "threshold", "threshold_selection", "evidence_mode", "protocol", "protocol_variant", "p1_preflight_passed", "p1a_hcval_source_audit_passed", "backbone_name")},
        **lock["loss"],
        **lock["candidate"],
        "strict_load": False,
    }
    lock_path = tmp_path / "lock.json"
    runtime_path = tmp_path / "runtime.json"
    out_path = tmp_path / "out.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "tools/official/check_cga_v21_protocol_lock.py",
            "--protocol_lock",
            str(lock_path),
            "--runtime_args_json",
            str(runtime_path),
            "--role",
            "candidate",
            "--output",
            str(out_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode == 2
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["pass"] is False
    assert any(m["field"] == "strict_load" for m in result["mismatches"])
```

---

## 12. Contract/smoke commands before seed42

Run these first:

```bash
cd /home/ly/AAAI/CGA-main

python3 -m py_compile \
  loss.py \
  train.py \
  test.py \
  net.py \
  model/cga_wrapper.py \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py

bash -n scripts/official/run_cga_v21_seed42_from_zero_paired.sh

python3 -m pytest \
  tests/test_cga_v21_loss_protocol_summary_strict.py \
  tests/test_cga_v21_protocol_checker.py
```

Then run a smoke with a fresh output directory and fewer epochs only if your runner supports smoke mode in a separate script. Do not use the paper runner for smoke unless you explicitly set a fresh `RUN_ID` and do not label the outputs as paper evidence.

---

## 13. Seed42 is still blocked until these pass

Only start seed42 after all conditions are true:

```text
1. protocol_lock.json committed.
2. protocol checker passes for baseline and candidate pretrain runtime args.
3. protocol checker passes again for baseline and candidate train runtime args.
4. test.py strict-load interface exists and py_compile passes.
5. summarizer enforces Full and HC-Val FA/Precision gates.
6. wrapper metadata reports center_boundary_scale_peak_safe_bg_v2_1 for candidate.
7. baseline metadata reports cga_variant=none and regularizer_impl=none.
8. pytest and bash -n pass.
9. no model/loss/target/threshold changes occur after committing the protocol lock.
```

Seed42 command only after the above:

```bash
cd /home/ly/AAAI/CGA-main

CUDA_DEVICE=1 \
ROOT=/home/ly/AAAI/CGA-main \
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
RUN_ID=seed42_protocol_locked_$(date +%Y%m%d_%H%M%S) \
bash scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

---

## 14. Final Go/No-Go

```text
Go now:
  implement the exact fixes above,
  run py_compile/bash -n/tests,
  commit protocol lock + code.

No-Go now:
  seed42 from-zero paired training.
```

CGA-v2.1 becomes seed42-ready only after protocol-lock coverage, baseline/candidate metadata separation, strict eval load, v2.1 summarizer, wrapper metadata consistency, and tests all pass.
