# CGA-v2.1 Pre-Seed42 Hard Fixes — Exact Code v4

Canonical repository root:

```text
/home/ly/AAAI/CGA-main
```

Current decision:

```text
Go: implement CGA-v2.1 pre-seed42 hard fixes.
No-Go: start CGA-v2.1 seed42 from-zero paired training.
```

This document supersedes the previous v3 exact-code plan. It fixes the remaining hard blockers:

```text
1. v2.1 summarizer must exit nonzero when metric gate fails.
2. Checkpoint metadata audit must compare paper evidence flags, fallback flag, P1/P1A flags, loss params, optimizer params, and resume.
3. resolve_model_config must remain backward compatible for old 2-tuple callers.
4. loss output must not contain metadata keys such as regularizer_impl.
5. protocol checker must compare root and dataset_dir by resolved paths.
6. runner runtime JSON must be generated from actual ROOT/DATASET_DIR environment values.
```

The CGA-v2.1 protocol is a new predeclared protocol. It must not be treated as a post-hoc modification of CGA-v2.

---

## 0. Why this is still No-Go for seed42

The prior CGA-v2 P2 result failed the seed42 gate. It should remain frozen as a valid-negative design-weakness result. CGA-v2.1 may only start seed42 after the code/protocol fixes below pass static checks, unit tests, protocol-lock checks, and smoke validation.

---

## 1. Target files

Edit or create these files only:

```text
net.py
model/cga_wrapper.py
loss.py
train.py
test.py
docs/internal/cga_v2_1/protocol_lock.json
tools/official/check_cga_v21_protocol_lock.py
tools/official/summarize_cga_v21_one_seed.py
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
tests/test_cga_v21_preseed42_hard_fixes.py
```

Do not change current CGA-v2 P2 outputs, checkpoints, thresholds, HC-Val split, or historical summaries.

---

## 2. `docs/internal/cga_v2_1/protocol_lock.json`

Create:

```bash
cd /home/ly/AAAI/CGA-main
mkdir -p docs/internal/cga_v2_1
cat > docs/internal/cga_v2_1/protocol_lock.json <<'JSON'
{
  "protocol_variant": "cga_v2_1",
  "dataset": {
    "root": "/home/ly/AAAI/CGA-main",
    "dataset_dir": "/home/ly/AAAI/CGA-main/datasets",
    "dataset_name": "NUDT-SIRST",
    "train_split": "train",
    "full_split": "test",
    "hcval_split": "hcval"
  },
  "roles": {
    "baseline": {
      "model_name": "MSHNetOHEM",
      "run_name": "MSHNetOHEM",
      "backbone": "mshnet",
      "use_cga": false,
      "cga_variant": "none",
      "regularizer_impl": "none"
    },
    "candidate": {
      "model_name": "MSHNetCGA21",
      "run_name": "MSHNetCGA21",
      "backbone": "mshnet",
      "use_cga": true,
      "cga_variant": "v2_1",
      "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"
    }
  },
  "training": {
    "seed": 42,
    "epochs": 400,
    "batch_size": 8,
    "patch_size": 256,
    "num_workers": 4,
    "resume": ""
  },
  "optimizer": {
    "optimizer": "Adam",
    "lr": 0.0005,
    "weight_decay": 0.0,
    "ohem_ratio": 0.01,
    "lambda_iou": 1.0
  },
  "loss": {
    "lambda_center": 0.05,
    "lambda_boundary": 0.03,
    "lambda_scale": 0.02,
    "lambda_peak": 0.03,
    "cga_start_epoch": 1,
    "cga_ramp_epochs": 40,
    "lambda_safe_bg": 0.05,
    "safe_bg_topk_ratio": 0.01,
    "safe_bg_ignore_radius": 2,
    "safe_bg_start_epoch": 1,
    "safe_bg_ramp_epochs": 40,
    "aux_ratio_cap": 0.15
  },
  "evaluation": {
    "threshold": 0.5,
    "threshold_selection": "fixed_predeclared",
    "strict_load": true,
    "strict_load_required": true
  },
  "evidence": {
    "evidence_mode": "paper",
    "protocol": "controlled",
    "paper_evidence_allowed": true,
    "p1_preflight_passed": true,
    "p1a_hcval_source_audit_passed": true,
    "fallback_regularizer_used": false
  },
  "gates": {
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
JSON
```

---

## 3. `net.py` exact replacement

Replace the whole `net.py` with this version. It preserves old 2-tuple callers by default and only returns the variant when `return_variant=True`.

```python
"""Model factory for fail-closed CGA paper-evidence experiments."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.CGA_MSHNet import MSHNetCGA
from model.cga_wrapper import CGAWrapper
from model.registry import available_backbones, get_backbone_builder

_CGA_V2_ALIASES = {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}
_CGA_V21_ALIASES = {
    "mshnetcga21",
    "mshnet_cga21",
    "cga21",
    "cga-v21",
    "cga_v21",
    "cga-v2.1",
    "cga_v2_1",
    "mshnet-cga-v2.1",
}
_MSHNET_BASE_ALIASES = {"mshnet", "mshnetohem", "ohem"}
_ALLOWED_CGA_VARIANTS = {None, "none", "v2", "v2_1"}


def _normalize_variant(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower().replace("-", "_").replace(".", "_")
    if v in {"", "none", "false", "0"}:
        return "none"
    if v in {"v2", "cga_v2"}:
        return "v2"
    if v in {"v21", "v2_1", "cga_v21", "cga_v2_1"}:
        return "v2_1"
    raise ValueError(f"Unknown cga_variant={value!r}; allowed: none, v2, v2_1")


def _resolve_alias(model_name: str | None, backbone_name: str, use_cga: bool) -> tuple[str, bool, str | None]:
    if model_name is None:
        return str(backbone_name).lower(), bool(use_cga), None

    name = str(model_name).lower()
    if name in _CGA_V21_ALIASES:
        return "mshnet", True, "v2_1"
    if name in _CGA_V2_ALIASES:
        return "mshnet", True, "v2"
    if name in _MSHNET_BASE_ALIASES:
        return "mshnet", False, "none"
    if name.endswith("_cga21") or name.endswith("_cga_v2_1"):
        return name.split("_cga", 1)[0], True, "v2_1"
    if name.endswith("_cga"):
        return name[: -len("_cga")], True, "v2"
    if name in available_backbones():
        return name, bool(use_cga), None
    return name, bool(use_cga), None


def resolve_model_config(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    use_cga: bool = False,
    cga_variant: str | None = None,
    return_variant: bool = False,
):
    """Resolve legacy model names into explicit backbone/CGA switches.

    Backward compatibility:
      - default return is the legacy 2-tuple: (backbone_name, use_cga)
      - set return_variant=True to receive: (backbone_name, use_cga, cga_variant)
    """
    resolved_backbone, resolved_use_cga, alias_variant = _resolve_alias(model_name, backbone_name, use_cga)
    explicit_variant = _normalize_variant(cga_variant)

    if not resolved_use_cga:
        if explicit_variant not in {None, "none"}:
            raise ValueError("use_cga=False requires cga_variant=None or 'none'")
        resolved_variant = "none"
    else:
        if alias_variant is not None and explicit_variant not in {None, alias_variant}:
            raise ValueError(
                f"model_name={model_name!r} resolves to cga_variant={alias_variant!r}, "
                f"but explicit cga_variant={explicit_variant!r} was provided"
            )
        resolved_variant = explicit_variant or alias_variant or "v2"
        if resolved_variant == "none":
            raise ValueError("use_cga=True requires cga_variant='v2' or 'v2_1'")

    if return_variant:
        return resolved_backbone, bool(resolved_use_cga), resolved_variant
    return resolved_backbone, bool(resolved_use_cga)


def regularizer_impl_for_variant(cga_variant: str | None, use_cga: bool) -> str:
    variant = _normalize_variant(cga_variant)
    if not use_cga or variant == "none":
        return "none"
    if variant == "v2":
        return "center_boundary_scale_peak"
    if variant == "v2_1":
        return "center_boundary_scale_peak_safe_bg_v2_1"
    raise ValueError(f"Unsupported cga_variant={cga_variant!r}")


def build_model(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    input_channels: int = 1,
    use_cga: bool = False,
    cga_variant: str | None = None,
    regularizer_impl: str | None = None,
    aux_hidden_channels: int = 32,
    evidence_mode: str = "paper",
    legacy_model_factory: bool = False,
    **kwargs: Any,
) -> nn.Module:
    if evidence_mode not in {"paper", "smoke"}:
        raise ValueError(f"Unknown evidence_mode={evidence_mode!r}")
    if evidence_mode == "paper" and kwargs.get("allow_fallback_regularizer", False):
        raise RuntimeError(
            "Fallback regularizer is forbidden for paper evidence. "
            "Run smoke tests under evidence_mode='smoke' only."
        )

    resolved_backbone, resolved_use_cga, resolved_variant = resolve_model_config(
        model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        cga_variant=cga_variant,
        return_variant=True,
    )
    resolved_impl = regularizer_impl_for_variant(resolved_variant, resolved_use_cga)
    if regularizer_impl is not None and str(regularizer_impl) != resolved_impl:
        raise ValueError(
            f"regularizer_impl={regularizer_impl!r} conflicts with "
            f"cga_variant={resolved_variant!r} -> {resolved_impl!r}"
        )

    if legacy_model_factory:
        if resolved_backbone == "mshnet" and resolved_use_cga and resolved_variant == "v2":
            return MSHNetCGA(input_channels=input_channels, aux_hidden_channels=int(aux_hidden_channels))
        raise ValueError("legacy_model_factory supports only legacy MSHNetCGA / cga_variant='v2'")

    builder = get_backbone_builder(resolved_backbone)
    backbone = builder(input_channels=input_channels)
    if not resolved_use_cga:
        return backbone

    feature_channels = getattr(backbone, "FEATURE_CHANNELS", None)
    if feature_channels is None:
        raise AttributeError(f"{resolved_backbone} adapter must define FEATURE_CHANNELS for CGAWrapper")

    return CGAWrapper(
        backbone,
        backbone_name=resolved_backbone,
        feature_channels=int(feature_channels),
        aux_hidden_channels=int(aux_hidden_channels),
        cga_variant=resolved_variant,
        regularizer_impl=resolved_impl,
    )


class Net(nn.Module):
    """Compatibility wrapper matching the legacy `Net(model_name=...)` pattern."""

    def __init__(self, model_name: str | None = "MSHNetCGA", input_channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_model(model_name=model_name, input_channels=input_channels, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
```

Required regression check after this change:

```bash
cd /home/ly/AAAI/CGA-main
python3 - <<'PY'
from net import resolve_model_config
assert resolve_model_config('MSHNetOHEM') == ('mshnet', False)
assert resolve_model_config('MSHNetCGA') == ('mshnet', True)
assert resolve_model_config('MSHNetCGA21', return_variant=True) == ('mshnet', True, 'v2_1')
try:
    resolve_model_config('MSHNetOHEM', cga_variant='v2_1', return_variant=True)
    raise AssertionError('expected conflict')
except ValueError:
    pass
print('resolve_model_config compatibility: pass')
PY
```

---

## 4. `model/cga_wrapper.py` metadata update

Patch the wrapper so the model output metadata matches v2.1 evidence metadata.

Replace the class initializer and metadata part of `forward` with this version. Keep the existing imports and `CGAAuxHead` call structure.

```python
class CGAWrapper(nn.Module):
    """Backbone wrapper with explicit center/boundary/scale/peak auxiliary heads."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_name: str,
        feature_channels: int,
        aux_hidden_channels: int = 32,
        cga_variant: str = "v2",
        regularizer_impl: str = "center_boundary_scale_peak",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_name = str(backbone_name)
        self.feature_channels = int(feature_channels)
        self.cga_variant = str(cga_variant)
        self.regularizer_impl = str(regularizer_impl)
        self.aux_head = CGAAuxHead(
            in_channels=int(feature_channels),
            hidden_channels=int(aux_hidden_channels),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> dict:
        raw = self.backbone(x, *args, **kwargs)
        out = validate_detector_output(raw)
        features = out.get("features") or []
        if not features:
            raise RuntimeError("CGAWrapper requires explicit features from the backbone adapter")
        feature = features[0]
        aux = self.aux_head(feature)
        out.update(aux)
        out["use_cga"] = True
        out["cga_variant"] = self.cga_variant
        out["regularizer_impl"] = self.regularizer_impl
        out["fallback_regularizer_used"] = False
        out["regularizer_meta"] = {
            "use_cga": True,
            "cga_variant": self.cga_variant,
            "regularizer_impl": self.regularizer_impl,
            "fallback_regularizer_used": False,
            "aux_heads": ["center", "boundary", "scale", "peak"],
            "feature_channels": self.feature_channels,
            "backbone_name": self.backbone_name,
        }
        return out
```

If the existing file uses a slightly different class body, apply the same fields exactly:

```text
self.cga_variant
self.regularizer_impl
out["cga_variant"]
out["regularizer_impl"]
out["regularizer_meta"]["cga_variant"]
out["regularizer_meta"]["regularizer_impl"]
```

---

## 5. `loss.py` v2.1 code

### 5.1 Add these imports

Ensure these imports exist at the top:

```python
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 5.2 Add config and loss class after `CGALoss`

Do not put metadata keys such as `regularizer_impl` in the returned loss dict.

```python
@dataclass(frozen=True)
class CGAV21LossConfig(CGALossConfig):
    lambda_safe_bg: float = 0.05
    safe_bg_topk_ratio: float = 0.01
    safe_bg_ignore_radius: int = 2
    safe_bg_start_epoch: int = 1
    safe_bg_ramp_epochs: int = 40
    aux_ratio_cap: float = 0.15

    def validate(self) -> None:
        if not (0.0 < float(self.safe_bg_topk_ratio) <= 1.0):
            raise ValueError("safe_bg_topk_ratio must be in (0, 1]")
        if int(self.safe_bg_ignore_radius) < 0:
            raise ValueError("safe_bg_ignore_radius must be >= 0")
        if int(self.start_epoch) < 0 or int(self.ramp_epochs) < 0:
            raise ValueError("CGA geometry start/ramp epochs must be non-negative")
        if int(self.safe_bg_start_epoch) < 0 or int(self.safe_bg_ramp_epochs) < 0:
            raise ValueError("safe-background start/ramp epochs must be non-negative")
        if float(self.aux_ratio_cap) <= 0.0:
            raise ValueError("aux_ratio_cap must be > 0")
        for name in (
            "lambda_center",
            "lambda_boundary",
            "lambda_scale",
            "lambda_peak",
            "lambda_safe_bg",
            "lambda_iou",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")


def _safe_background_mask(target: torch.Tensor, radius: int) -> torch.Tensor:
    """Return background pixels outside a small dilation around GT positives."""
    if target.dim() == 3:
        target = target[:, None]
    target = target.float()
    if radius <= 0:
        return target <= 0
    k = int(radius) * 2 + 1
    dilated = F.max_pool2d(target, kernel_size=k, stride=1, padding=int(radius))
    return dilated <= 0


class CGAV21Loss(nn.Module):
    """CGA-v2.1 loss: geometry auxiliary supervision + safe-background FA control.

    The safe-background term uses BCEWithLogits on final logits over GT-safe background
    pixels and selects top-k hard negatives. Auxiliary regularization is ratio-capped
    using a detached scale, so gradients are preserved when the cap is active.
    """

    def __init__(self, cfg: CGAV21LossConfig, strict_cga_heads: bool = True) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.base = MSHNetOHEMLoss(
            ohem_ratio=cfg.ohem_ratio,
            lambda_iou=cfg.lambda_iou,
            warm_epoch=cfg.warm_epoch,
        )
        self.strict_cga_heads = bool(strict_cga_heads)

    def _geometry_aux(self, output: dict[str, Any], target: torch.Tensor, final_logit: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.strict_cga_heads:
            _require_cga_logits(output)
        targets = build_cga_targets(target, CGATargetConfig(), size=final_logit.shape[-2:])

        center_logit = output["cga_center_logit"]
        boundary_logit = output["cga_boundary_logit"]
        scale_logit = output["cga_scale_logit"]
        peak_logit = output["cga_peak_logit"]

        t_center = _resize_like(targets["center"], center_logit)
        t_boundary = _resize_like(targets["boundary"], boundary_logit)
        t_scale = _resize_like(targets["scale"], scale_logit)
        t_peak = _resize_like(targets["peak"], peak_logit)

        loss_center = F.binary_cross_entropy_with_logits(center_logit, t_center)
        loss_boundary = F.binary_cross_entropy_with_logits(boundary_logit, t_boundary)
        loss_scale = F.smooth_l1_loss(torch.sigmoid(scale_logit), t_scale)
        loss_peak = F.binary_cross_entropy_with_logits(peak_logit, t_peak)

        geom = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )
        return {
            "geom_aux_total": geom,
            "loss_center": loss_center,
            "loss_boundary": loss_boundary,
            "loss_scale": loss_scale,
            "loss_peak": loss_peak,
        }

    def _safe_bg_loss(self, final_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _resize_like(target, final_logit)
        safe_mask = _safe_background_mask(target, self.cfg.safe_bg_ignore_radius)
        if safe_mask.shape != final_logit.shape:
            safe_mask = safe_mask.expand_as(final_logit)
        logits = final_logit[safe_mask]
        if logits.numel() == 0:
            return final_logit.sum() * 0.0
        zeros = torch.zeros_like(logits)
        loss = F.binary_cross_entropy_with_logits(logits, zeros, reduction="none")
        k = max(1, int(loss.numel() * float(self.cfg.safe_bg_topk_ratio)))
        return torch.topk(loss.flatten(), k=min(k, loss.numel()), largest=True).values.mean()

    def forward(self, output: Any, target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        if not isinstance(output, dict):
            raise TypeError("CGAV21Loss requires dict output with explicit CGA auxiliary logits")
        final_logit = extract_final_logit(output)
        base_out = self.base(output, target, epoch=epoch)
        base_total = base_out["total"]

        geom = self._geometry_aux(output, target, final_logit)
        geom_w = torch.tensor(
            _ramp_weight(epoch, self.cfg.start_epoch, self.cfg.ramp_epochs),
            dtype=base_total.dtype,
            device=base_total.device,
        )
        safe_w = torch.tensor(
            _ramp_weight(epoch, self.cfg.safe_bg_start_epoch, self.cfg.safe_bg_ramp_epochs),
            dtype=base_total.dtype,
            device=base_total.device,
        )
        safe_bg = self._safe_bg_loss(final_logit, target)
        reg_raw = geom_w * geom["geom_aux_total"] + safe_w * self.cfg.lambda_safe_bg * safe_bg

        reg_cap = base_total.detach().abs().clamp_min(1e-6) * float(self.cfg.aux_ratio_cap)
        cap_scale = (reg_cap / reg_raw.detach().abs().clamp_min(1e-6)).clamp(max=1.0)
        reg_capped = reg_raw * cap_scale
        total = base_total + reg_capped

        return {
            "total": total,
            "base_total": base_total.detach(),
            "ohem": base_out.get("ohem", base_total.detach()),
            "soft_iou": base_out.get("soft_iou", base_total.detach() * 0.0),
            "geom_aux_total": geom["geom_aux_total"].detach(),
            "loss_center": geom["loss_center"].detach(),
            "loss_boundary": geom["loss_boundary"].detach(),
            "loss_scale": geom["loss_scale"].detach(),
            "loss_peak": geom["loss_peak"].detach(),
            "loss_safe_bg": safe_bg.detach(),
            "cga_w": geom_w.detach(),
            "safe_bg_w": safe_w.detach(),
            "reg_raw": reg_raw.detach(),
            "reg_capped": reg_capped.detach(),
            "reg_raw_over_base": (reg_raw.detach().abs() / base_total.detach().abs().clamp_min(1e-6)),
            "reg_capped_over_base": (reg_capped.detach().abs() / base_total.detach().abs().clamp_min(1e-6)),
            "cap_active": (cap_scale.detach() < 0.999).float(),
            "cap_scale": cap_scale.detach(),
        }
```

### 5.3 Replace `build_loss(...)`

Replace the existing `build_loss` function with this version.

```python
def _normalize_loss_variant(model_name: str | None, use_cga: bool, cga_variant: str | None) -> str:
    if not use_cga:
        if cga_variant not in {None, "none"}:
            raise ValueError("use_cga=False requires cga_variant=None or 'none'")
        return "none"
    if cga_variant is None:
        name = str(model_name or "").lower().replace("-", "_").replace(".", "_")
        if any(token in name for token in ("cga21", "cga_v21", "cga_v2_1")):
            return "v2_1"
        return "v2"
    v = str(cga_variant).lower().replace("-", "_").replace(".", "_")
    if v in {"none", "false", "0"}:
        raise ValueError("use_cga=True cannot use cga_variant='none'")
    if v in {"v2", "cga_v2"}:
        return "v2"
    if v in {"v21", "v2_1", "cga_v21", "cga_v2_1"}:
        return "v2_1"
    raise ValueError(f"Unknown cga_variant={cga_variant!r}")


def build_loss(
    model_name: str | None = None,
    *,
    use_cga: bool = False,
    cga_variant: str | None = None,
    ohem_ratio: float = 0.01,
    lambda_iou: float = 1.0,
    mshnet_warm_epoch: int = 5,
    cga_start_epoch: int = 1,
    cga_ramp_epochs: int = 40,
    lambda_center: float = 0.05,
    lambda_boundary: float = 0.03,
    lambda_scale: float = 0.02,
    lambda_peak: float = 0.03,
    strict_cga_heads: bool = True,
    lambda_safe_bg: float = 0.05,
    safe_bg_topk_ratio: float = 0.01,
    safe_bg_ignore_radius: int = 2,
    safe_bg_start_epoch: int = 1,
    safe_bg_ramp_epochs: int = 40,
    aux_ratio_cap: float = 0.15,
) -> nn.Module:
    variant = _normalize_loss_variant(model_name, use_cga, cga_variant)
    if variant == "none":
        return MSHNetOHEMLoss(
            ohem_ratio=ohem_ratio,
            lambda_iou=lambda_iou,
            warm_epoch=mshnet_warm_epoch,
        )
    if variant == "v2":
        return CGALoss(
            CGALossConfig(
                lambda_center=lambda_center,
                lambda_boundary=lambda_boundary,
                lambda_scale=lambda_scale,
                lambda_peak=lambda_peak,
                start_epoch=cga_start_epoch,
                ramp_epochs=cga_ramp_epochs,
                ohem_ratio=ohem_ratio,
                lambda_iou=lambda_iou,
                warm_epoch=mshnet_warm_epoch,
            ),
            strict_cga_heads=strict_cga_heads,
        )
    if variant == "v2_1":
        return CGAV21Loss(
            CGAV21LossConfig(
                lambda_center=lambda_center,
                lambda_boundary=lambda_boundary,
                lambda_scale=lambda_scale,
                lambda_peak=lambda_peak,
                start_epoch=cga_start_epoch,
                ramp_epochs=cga_ramp_epochs,
                ohem_ratio=ohem_ratio,
                lambda_iou=lambda_iou,
                warm_epoch=mshnet_warm_epoch,
                lambda_safe_bg=lambda_safe_bg,
                safe_bg_topk_ratio=safe_bg_topk_ratio,
                safe_bg_ignore_radius=safe_bg_ignore_radius,
                safe_bg_start_epoch=safe_bg_start_epoch,
                safe_bg_ramp_epochs=safe_bg_ramp_epochs,
                aux_ratio_cap=aux_ratio_cap,
            ),
            strict_cga_heads=strict_cga_heads,
        )
    raise AssertionError(f"unreachable variant={variant!r}")
```

---

## 6. `train.py` required changes

### 6.1 Add arguments inside `parse_args()`

Add these arguments after the current CGA parameters:

```python
    p.add_argument("--protocol_variant", default="cga_v2", choices=["cga_v2", "cga_v2_1"])
    p.add_argument("--cga_variant", default=None, choices=[None, "none", "v2", "v2_1"])
    p.add_argument("--regularizer_impl", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold_selection", default="fixed_predeclared", choices=["fixed_predeclared"])
    p.add_argument("--strict_load_required", action="store_true")
    p.add_argument("--runtime_args_json", default="")
    p.add_argument("--optimizer", default="Adam", choices=["Adam"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_iou", type=float, default=1.0)
    p.add_argument("--lambda_safe_bg", type=float, default=0.05)
    p.add_argument("--safe_bg_topk_ratio", type=float, default=0.01)
    p.add_argument("--safe_bg_ignore_radius", type=int, default=2)
    p.add_argument("--safe_bg_start_epoch", type=int, default=1)
    p.add_argument("--safe_bg_ramp_epochs", type=int, default=40)
    p.add_argument("--aux_ratio_cap", type=float, default=0.15)
```

### 6.2 Add helper functions near `compute_paper_evidence_allowed`

```python
def _regularizer_impl_for_variant(use_cga: bool, cga_variant: str) -> str:
    if not use_cga or cga_variant == "none":
        return "none"
    if cga_variant == "v2":
        return "center_boundary_scale_peak"
    if cga_variant == "v2_1":
        return "center_boundary_scale_peak_safe_bg_v2_1"
    raise ValueError(f"Unsupported cga_variant={cga_variant!r}")


def _loss_params_from_args(args: argparse.Namespace) -> dict:
    return {
        "lambda_center": float(args.lambda_center),
        "lambda_boundary": float(args.lambda_boundary),
        "lambda_scale": float(args.lambda_scale),
        "lambda_peak": float(args.lambda_peak),
        "cga_start_epoch": int(args.cga_start_epoch),
        "cga_ramp_epochs": int(args.cga_ramp_epochs),
        "lambda_safe_bg": float(args.lambda_safe_bg),
        "safe_bg_topk_ratio": float(args.safe_bg_topk_ratio),
        "safe_bg_ignore_radius": int(args.safe_bg_ignore_radius),
        "safe_bg_start_epoch": int(args.safe_bg_start_epoch),
        "safe_bg_ramp_epochs": int(args.safe_bg_ramp_epochs),
        "aux_ratio_cap": float(args.aux_ratio_cap),
    }


def _optimizer_params_from_args(args: argparse.Namespace) -> dict:
    return {
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "patch_size": int(args.patch_size),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "ohem_ratio": float(args.ohem_ratio),
        "lambda_iou": float(args.lambda_iou),
        "resume": str(args.resume or ""),
    }


def _write_runtime_args_json(path: str, payload: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
```

### 6.3 Replace the resolver call

Replace:

```python
    backbone_name, use_cga = resolve_model_config(
        args.model_name,
        backbone_name=args.backbone_name,
        use_cga=args.use_cga,
    )
```

with:

```python
    backbone_name, use_cga, cga_variant = resolve_model_config(
        args.model_name,
        backbone_name=args.backbone_name,
        use_cga=args.use_cga,
        cga_variant=args.cga_variant,
        return_variant=True,
    )
    regularizer_impl = args.regularizer_impl or _regularizer_impl_for_variant(use_cga, cga_variant)

    if args.protocol_variant == "cga_v2_1":
        if args.evidence_mode == "paper" and args.resume:
            raise RuntimeError("CGA-v2.1 paper evidence is from-zero only; --resume is forbidden")
        if args.evidence_mode == "paper" and not args.strict_load_required:
            raise RuntimeError("CGA-v2.1 paper evidence requires --strict_load_required")
```

### 6.4 Replace model creation

Replace the `build_model(...)` call with:

```python
    model = build_model(
        model_name=args.model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        cga_variant=cga_variant,
        regularizer_impl=regularizer_impl,
        evidence_mode=args.evidence_mode,
        input_channels=1,
        aux_hidden_channels=args.aux_hidden_channels,
        allow_fallback_regularizer=args.allow_fallback_regularizer,
    ).to(device)
```

### 6.5 Replace loss creation

Replace the `build_loss(...)` call with:

```python
    criterion = build_loss(
        args.model_name or backbone_name,
        use_cga=use_cga,
        cga_variant=cga_variant,
        ohem_ratio=args.ohem_ratio,
        lambda_iou=args.lambda_iou,
        mshnet_warm_epoch=args.mshnet_warm_epoch,
        cga_start_epoch=args.cga_start_epoch,
        cga_ramp_epochs=args.cga_ramp_epochs,
        lambda_center=args.lambda_center,
        lambda_boundary=args.lambda_boundary,
        lambda_scale=args.lambda_scale,
        lambda_peak=args.lambda_peak,
        strict_cga_heads=(args.evidence_mode == "paper" and use_cga),
        lambda_safe_bg=args.lambda_safe_bg,
        safe_bg_topk_ratio=args.safe_bg_topk_ratio,
        safe_bg_ignore_radius=args.safe_bg_ignore_radius,
        safe_bg_start_epoch=args.safe_bg_start_epoch,
        safe_bg_ramp_epochs=args.safe_bg_ramp_epochs,
        aux_ratio_cap=args.aux_ratio_cap,
    ).to(device)
```

### 6.6 Replace optimizer creation

Replace:

```python
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
```

with:

```python
    if args.optimizer != "Adam":
        raise ValueError("Only Adam is locked for CGA-v2.1 protocol")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

### 6.7 Replace resume block

Replace the existing resume block with:

```python
    start_epoch = 1
    if args.resume:
        if args.protocol_variant == "cga_v2_1" and args.evidence_mode == "paper":
            raise RuntimeError("CGA-v2.1 paper evidence is from-zero only; --resume is forbidden")
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Resume checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
```

### 6.8 Add runtime metadata before the training loop

Add after `optim` creation:

```python
    paper_evidence_allowed_initial = compute_paper_evidence_allowed(
        evidence_mode=args.evidence_mode,
        p1_preflight_passed=args.p1_preflight_passed,
        p1a_hcval_source_audit_passed=args.p1a_hcval_source_audit_passed,
        fallback_regularizer_used=bool(args.allow_fallback_regularizer),
    )
    runtime_payload = {
        "protocol_variant": args.protocol_variant,
        "model_name": run_model_name,
        "backbone": backbone_name,
        "use_cga": bool(use_cga),
        "cga_variant": cga_variant,
        "regularizer_impl": regularizer_impl,
        "root": str(Path.cwd().resolve()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "dataset_name": args.dataset_name,
        "seed": int(args.seed),
        "evidence_mode": args.evidence_mode,
        "protocol": args.protocol,
        "paper_evidence_allowed": bool(paper_evidence_allowed_initial),
        "p1_preflight_passed": bool(args.p1_preflight_passed),
        "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
        "fallback_regularizer_used": bool(args.allow_fallback_regularizer),
        "threshold": float(args.threshold),
        "threshold_selection": args.threshold_selection,
        "strict_load_required": bool(args.strict_load_required),
        "loss_params": _loss_params_from_args(args),
        "optimizer_params": _optimizer_params_from_args(args),
    }
    _write_runtime_args_json(args.runtime_args_json, runtime_payload)
```

### 6.9 Update `evidence_meta`

Inside the epoch loop, replace the existing `evidence_meta = {...}` block with:

```python
        loss_params = _loss_params_from_args(args)
        optimizer_params = _optimizer_params_from_args(args)
        evidence_meta = {
            "epoch": epoch,
            "protocol_variant": args.protocol_variant,
            "dataset": args.dataset_name,
            "model": run_model_name,
            "backbone": backbone_name,
            "use_cga": bool(use_cga),
            "cga_variant": cga_variant,
            "regularizer_impl": regularizer_impl,
            "evidence_mode": args.evidence_mode,
            "p1_preflight_passed": bool(args.p1_preflight_passed),
            "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
            "fallback_regularizer_used": bool(fallback_regularizer_used),
            "paper_evidence_allowed": bool(paper_evidence_allowed),
            "protocol": args.protocol,
            "seed": args.seed,
            "threshold": float(args.threshold),
            "threshold_selection": args.threshold_selection,
            "strict_load_required": bool(args.strict_load_required),
            "mshnet_warm_epoch": args.mshnet_warm_epoch,
            "cga_start_epoch": args.cga_start_epoch,
            "cga_ramp_epochs": args.cga_ramp_epochs,
            "loss_params": loss_params,
            "optimizer_params": optimizer_params,
        }
```

### 6.10 Update checkpoint payload

Inside `torch.save({...})`, include these keys:

```python
                    "protocol_variant": args.protocol_variant,
                    "model_name": run_model_name,
                    "backbone": backbone_name,
                    "use_cga": bool(use_cga),
                    "cga_variant": cga_variant,
                    "regularizer_impl": regularizer_impl,
                    "evidence_mode": args.evidence_mode,
                    "p1_preflight_passed": bool(args.p1_preflight_passed),
                    "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
                    "fallback_regularizer_used": bool(fallback_regularizer_used),
                    "paper_evidence_allowed": bool(paper_evidence_allowed),
                    "protocol": args.protocol,
                    "dataset": args.dataset_name,
                    "seed": args.seed,
                    "threshold": float(args.threshold),
                    "threshold_selection": args.threshold_selection,
                    "strict_load_required": bool(args.strict_load_required),
                    "resume": str(args.resume or ""),
                    "loss_params": _loss_params_from_args(args),
                    "optimizer_params": _optimizer_params_from_args(args),
```

Keep the existing `state_dict` and `optimizer` keys.

---

## 7. `test.py` required code

### 7.1 Add arguments

Add these to `parse_args()`:

```python
    p.add_argument("--run_name", default=None)
    p.add_argument("--protocol", default="controlled", choices=["controlled", "official"])
    p.add_argument("--protocol_variant", default="cga_v2", choices=["cga_v2", "cga_v2_1"])
    p.add_argument("--cga_variant", default=None, choices=[None, "none", "v2", "v2_1"])
    p.add_argument("--regularizer_impl", default=None)
    p.add_argument("--strict_load", action="store_true")
```

### 7.2 Add strict-load helpers after `save_prob_png`

```python
_ALLOWED_PREFIXES = ("module.", "model.", "net.")


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
    return ckpt


def _strip_shared_prefix_if_safe(state_dict: dict, prefix: str) -> dict:
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    prefixed = [k for k in keys if k.startswith(prefix)]
    if len(prefixed) != len(keys):
        return state_dict
    stripped = [k[len(prefix):] for k in keys]
    if len(set(stripped)) != len(stripped):
        raise RuntimeError(f"Stripping prefix {prefix!r} would create duplicate keys")
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def normalize_state_dict_for_strict_load(raw_state: dict) -> dict:
    state = _unwrap_state_dict(raw_state)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state_dict is not a dict: {type(state)!r}")
    for prefix in _ALLOWED_PREFIXES:
        stripped = _strip_shared_prefix_if_safe(state, prefix)
        if stripped is not state:
            return stripped
    return state


def strict_load_model(model, ckpt: dict, *, strict_load: bool) -> dict:
    state = normalize_state_dict_for_strict_load(ckpt)
    if strict_load:
        result = model.load_state_dict(state, strict=True)
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if missing or unexpected:
            raise RuntimeError(f"strict load failed: missing={missing}, unexpected={unexpected}")
        return {"strict_load_pass": True, "missing_keys": [], "unexpected_keys": []}
    result = model.load_state_dict(state, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"non-strict load still found mismatches: missing={missing}, unexpected={unexpected}")
    return {"strict_load_pass": True, "missing_keys": [], "unexpected_keys": []}
```

### 7.3 Replace resolver/model/load block in `main()`

Replace the current resolver/model/checkpoint-load section with:

```python
    train_name = args.train_dataset_name or args.dataset_name
    backbone_name, use_cga, cga_variant = resolve_model_config(
        args.model_name,
        backbone_name=args.backbone_name,
        use_cga=args.use_cga,
        cga_variant=args.cga_variant,
        return_variant=True,
    )
    run_model_name = args.run_name or _run_model_name(args.model_name, backbone_name, use_cga)
    regularizer_impl = args.regularizer_impl
    if regularizer_impl is None:
        if not use_cga or cga_variant == "none":
            regularizer_impl = "none"
        elif cga_variant == "v2_1":
            regularizer_impl = "center_boundary_scale_peak_safe_bg_v2_1"
        else:
            regularizer_impl = "center_boundary_scale_peak"

    if args.protocol_variant == "cga_v2_1" and args.evidence_mode == "paper" and not args.strict_load:
        raise RuntimeError("CGA-v2.1 paper eval requires --strict_load")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=args.model_name,
        backbone_name=backbone_name,
        use_cga=use_cga,
        cga_variant=cga_variant,
        regularizer_impl=regularizer_impl,
        evidence_mode=args.evidence_mode,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    strict_meta = strict_load_model(model, ckpt, strict_load=args.strict_load)
    model.eval()
```

### 7.4 Add eval trace in loop

Before the loop:

```python
    eval_trace = None
```

Inside the inference loop, right after `logit = extract_final_logit(output)`, add:

```python
            if eval_trace is None:
                eval_trace = {
                    "prediction_tensor_source": "final_logit",
                    "logit_shape": list(logit.shape),
                    "logit_min": float(logit.detach().min().cpu()),
                    "logit_max": float(logit.detach().max().cpu()),
                    "uses_aux_heads_for_prediction": False,
                    "output_keys": sorted(list(output.keys())) if isinstance(output, dict) else str(type(output)),
                }
```

### 7.5 Update summary metadata

Replace `summary.update({...})` with:

```python
    checkpoint_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    ckpt_meta = ckpt if isinstance(ckpt, dict) else {}
    summary.update({
        "protocol_variant": args.protocol_variant,
        "model": run_model_name,
        "backbone": backbone_name,
        "use_cga": bool(use_cga),
        "cga_variant": cga_variant,
        "regularizer_impl": regularizer_impl,
        "train_dataset": train_name,
        "dataset": args.dataset_name,
        "split": args.split,
        "seed": args.seed,
        "epoch": checkpoint_epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "threshold": args.threshold,
        "threshold_selection": "fixed_predeclared",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "prediction_dir": str(pred_dir.resolve()),
        "evidence_mode": args.evidence_mode,
        "protocol": args.protocol,
        "strict_load": bool(args.strict_load),
        "strict_load_pass": bool(strict_meta["strict_load_pass"]),
        "strict_load_missing_keys": strict_meta["missing_keys"],
        "strict_load_unexpected_keys": strict_meta["unexpected_keys"],
        "paper_evidence_allowed": bool(ckpt_meta.get("paper_evidence_allowed", False)),
        "fallback_regularizer_used": bool(ckpt_meta.get("fallback_regularizer_used", True)),
        "p1_preflight_passed": bool(ckpt_meta.get("p1_preflight_passed", False)),
        "p1a_hcval_source_audit_passed": bool(ckpt_meta.get("p1a_hcval_source_audit_passed", False)),
        "checkpoint_metadata": {
            "protocol_variant": ckpt_meta.get("protocol_variant"),
            "cga_variant": ckpt_meta.get("cga_variant"),
            "regularizer_impl": ckpt_meta.get("regularizer_impl"),
            "strict_load_required": ckpt_meta.get("strict_load_required"),
            "threshold": ckpt_meta.get("threshold"),
            "threshold_selection": ckpt_meta.get("threshold_selection"),
            "paper_evidence_allowed": ckpt_meta.get("paper_evidence_allowed"),
            "fallback_regularizer_used": ckpt_meta.get("fallback_regularizer_used"),
            "p1_preflight_passed": ckpt_meta.get("p1_preflight_passed"),
            "p1a_hcval_source_audit_passed": ckpt_meta.get("p1a_hcval_source_audit_passed"),
            "loss_params": ckpt_meta.get("loss_params"),
            "optimizer_params": ckpt_meta.get("optimizer_params"),
            "resume": ckpt_meta.get("resume", ""),
        },
        "eval_trace": eval_trace or {},
    })
```

---

## 8. `tools/official/check_cga_v21_protocol_lock.py`

Create this file:

```bash
mkdir -p tools/official
cat > tools/official/check_cga_v21_protocol_lock.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolved(path_value: str) -> str:
    return str(Path(path_value).expanduser().resolve())


def eq_float(a: Any, b: Any, eps: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= eps


def check_equal(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        if not eq_float(actual, expected):
            errors.append(f"{name}: actual={actual!r}, expected={expected!r}")
    elif actual != expected:
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}")


def check_role(errors: list[str], runtime: dict[str, Any], lock: dict[str, Any], role: str) -> None:
    expected = lock["roles"][role]
    actual = runtime[role] if role in runtime else runtime
    for key in ("run_name", "backbone", "use_cga", "cga_variant", "regularizer_impl"):
        check_equal(errors, f"{role}.{key}", actual.get(key), expected[key])


def main() -> None:
    p = argparse.ArgumentParser("Check CGA-v2.1 protocol lock against runtime args")
    p.add_argument("--lock", required=True)
    p.add_argument("--runtime_args", required=True)
    p.add_argument("--role", choices=["baseline", "candidate", "paired"], default="paired")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = load_json(args.lock)
    runtime = load_json(args.runtime_args)
    errors: list[str] = []

    check_equal(errors, "protocol_variant", runtime.get("protocol_variant"), lock["protocol_variant"])
    check_equal(errors, "dataset_name", runtime.get("dataset_name"), lock["dataset"]["dataset_name"])
    check_equal(errors, "root", resolved(runtime.get("root", "")), resolved(lock["dataset"]["root"]))
    check_equal(errors, "dataset_dir", resolved(runtime.get("dataset_dir", "")), resolved(lock["dataset"]["dataset_dir"]))

    for key in ("seed", "epochs", "batch_size", "patch_size", "num_workers", "resume"):
        check_equal(errors, f"training.{key}", runtime.get(key), lock["training"][key])

    for key in ("optimizer", "lr", "weight_decay", "ohem_ratio", "lambda_iou"):
        check_equal(errors, f"optimizer.{key}", runtime.get(key), lock["optimizer"][key])

    for key in (
        "lambda_center",
        "lambda_boundary",
        "lambda_scale",
        "lambda_peak",
        "cga_start_epoch",
        "cga_ramp_epochs",
        "lambda_safe_bg",
        "safe_bg_topk_ratio",
        "safe_bg_ignore_radius",
        "safe_bg_start_epoch",
        "safe_bg_ramp_epochs",
        "aux_ratio_cap",
    ):
        check_equal(errors, f"loss.{key}", runtime.get(key), lock["loss"][key])

    for key in ("threshold", "threshold_selection", "strict_load", "strict_load_required"):
        check_equal(errors, f"evaluation.{key}", runtime.get(key), lock["evaluation"][key])

    for key in (
        "evidence_mode",
        "protocol",
        "paper_evidence_allowed",
        "p1_preflight_passed",
        "p1a_hcval_source_audit_passed",
        "fallback_regularizer_used",
    ):
        check_equal(errors, f"evidence.{key}", runtime.get(key), lock["evidence"][key])

    if args.role == "paired":
        check_role(errors, runtime, lock, "baseline")
        check_role(errors, runtime, lock, "candidate")
    else:
        check_role(errors, runtime, lock, args.role)

    result = {
        "checker": "check_cga_v21_protocol_lock",
        "pass": len(errors) == 0,
        "errors": errors,
        "lock": str(Path(args.lock).resolve()),
        "runtime_args": str(Path(args.runtime_args).resolve()),
        "role": args.role,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/check_cga_v21_protocol_lock.py
```

---

## 9. `tools/official/summarize_cga_v21_one_seed.py`

Create this file. It performs metric gate checks, metadata checks, checkpoint metadata checks, and exits nonzero when gate fails.

```bash
cat > tools/official/summarize_cga_v21_one_seed.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(d: dict[str, Any], key: str) -> float:
    if key in d:
        return float(d[key])
    for k in (key, key.lower(), key.upper()):
        if k in d:
            return float(d[k])
    raise KeyError(f"Missing metric {key!r}; keys={sorted(d.keys())}")


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return metric(candidate, key) - metric(baseline, key)


def approx_equal(a: Any, b: Any, eps: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return a == b


def require(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float) or isinstance(actual, float):
        if not approx_equal(actual, expected):
            errors.append(f"{name}: actual={actual!r}, expected={expected!r}")
    elif actual != expected:
        errors.append(f"{name}: actual={actual!r}, expected={expected!r}")


def audit_summary_role(errors: list[str], summary: dict[str, Any], lock: dict[str, Any], role: str) -> None:
    exp = lock["roles"][role]
    require(errors, f"{role}.model", summary.get("model"), exp["run_name"])
    require(errors, f"{role}.backbone", summary.get("backbone"), exp["backbone"])
    require(errors, f"{role}.use_cga", bool(summary.get("use_cga")), bool(exp["use_cga"]))
    require(errors, f"{role}.cga_variant", summary.get("cga_variant"), exp["cga_variant"])
    require(errors, f"{role}.regularizer_impl", summary.get("regularizer_impl"), exp["regularizer_impl"])
    require(errors, f"{role}.protocol_variant", summary.get("protocol_variant"), lock["protocol_variant"])
    require(errors, f"{role}.evidence_mode", summary.get("evidence_mode"), lock["evidence"]["evidence_mode"])
    require(errors, f"{role}.protocol", summary.get("protocol"), lock["evidence"]["protocol"])
    require(errors, f"{role}.seed", int(summary.get("seed")), int(lock["training"]["seed"]))
    require(errors, f"{role}.checkpoint_epoch", int(summary.get("checkpoint_epoch")), int(lock["training"]["epochs"]))
    require(errors, f"{role}.dataset", summary.get("dataset"), lock["dataset"]["dataset_name"])
    require(errors, f"{role}.threshold", float(summary.get("threshold")), float(lock["evaluation"]["threshold"]))
    require(errors, f"{role}.threshold_selection", summary.get("threshold_selection"), lock["evaluation"]["threshold_selection"])
    require(errors, f"{role}.strict_load_pass", bool(summary.get("strict_load_pass")), True)
    trace = summary.get("eval_trace", {})
    require(errors, f"{role}.eval_trace.uses_aux_heads_for_prediction", bool(trace.get("uses_aux_heads_for_prediction", True)), False)


def _checkpoint(path: str) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict: {path}")
    return ckpt


def audit_checkpoint_role(errors: list[str], summary: dict[str, Any], lock: dict[str, Any], role: str) -> None:
    ckpt_path = summary.get("checkpoint")
    if not ckpt_path:
        errors.append(f"{role}.checkpoint missing in summary")
        return
    ckpt = _checkpoint(ckpt_path)
    exp = lock["roles"][role]
    require(errors, f"{role}.ckpt.protocol_variant", ckpt.get("protocol_variant"), lock["protocol_variant"])
    require(errors, f"{role}.ckpt.model_name", ckpt.get("model_name"), exp["run_name"])
    require(errors, f"{role}.ckpt.backbone", ckpt.get("backbone"), exp["backbone"])
    require(errors, f"{role}.ckpt.use_cga", bool(ckpt.get("use_cga")), bool(exp["use_cga"]))
    require(errors, f"{role}.ckpt.cga_variant", ckpt.get("cga_variant"), exp["cga_variant"])
    require(errors, f"{role}.ckpt.regularizer_impl", ckpt.get("regularizer_impl"), exp["regularizer_impl"])
    require(errors, f"{role}.ckpt.evidence_mode", ckpt.get("evidence_mode"), lock["evidence"]["evidence_mode"])
    require(errors, f"{role}.ckpt.protocol", ckpt.get("protocol"), lock["evidence"]["protocol"])
    require(errors, f"{role}.ckpt.paper_evidence_allowed", bool(ckpt.get("paper_evidence_allowed")), True)
    require(errors, f"{role}.ckpt.fallback_regularizer_used", bool(ckpt.get("fallback_regularizer_used")), False)
    require(errors, f"{role}.ckpt.p1_preflight_passed", bool(ckpt.get("p1_preflight_passed")), True)
    require(errors, f"{role}.ckpt.p1a_hcval_source_audit_passed", bool(ckpt.get("p1a_hcval_source_audit_passed")), True)
    require(errors, f"{role}.ckpt.threshold", float(ckpt.get("threshold")), float(lock["evaluation"]["threshold"]))
    require(errors, f"{role}.ckpt.threshold_selection", ckpt.get("threshold_selection"), lock["evaluation"]["threshold_selection"])
    require(errors, f"{role}.ckpt.strict_load_required", bool(ckpt.get("strict_load_required")), True)
    require(errors, f"{role}.ckpt.resume", ckpt.get("resume", ""), lock["training"]["resume"])
    require(errors, f"{role}.ckpt.loss_params", ckpt.get("loss_params"), lock["loss"])
    expected_optimizer = dict(lock["optimizer"])
    expected_optimizer.update({
        "batch_size": lock["training"]["batch_size"],
        "patch_size": lock["training"]["patch_size"],
        "num_workers": lock["training"]["num_workers"],
        "epochs": lock["training"]["epochs"],
        "resume": lock["training"]["resume"],
    })
    require(errors, f"{role}.ckpt.optimizer_params", ckpt.get("optimizer_params"), expected_optimizer)


def main() -> None:
    p = argparse.ArgumentParser("Summarize CGA-v2.1 one-seed paired run")
    p.add_argument("--lock", required=True)
    p.add_argument("--baseline_full", required=True)
    p.add_argument("--candidate_full", required=True)
    p.add_argument("--baseline_hcval", required=True)
    p.add_argument("--candidate_hcval", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = load_json(args.lock)
    bf = load_json(args.baseline_full)
    cf = load_json(args.candidate_full)
    bh = load_json(args.baseline_hcval)
    ch = load_json(args.candidate_hcval)

    metadata_errors: list[str] = []
    for role, summary in (("baseline", bf), ("candidate", cf), ("baseline", bh), ("candidate", ch)):
        audit_summary_role(metadata_errors, summary, lock, role)
        audit_checkpoint_role(metadata_errors, summary, lock, role)

    full_delta = {
        "mIoU": delta(cf, bf, "mIoU"),
        "Precision": delta(cf, bf, "Precision"),
        "Pd": delta(cf, bf, "Pd"),
        "FA_ppm": delta(cf, bf, "FA_ppm"),
    }
    hcval_delta = {
        "mIoU": delta(ch, bh, "mIoU"),
        "Precision": delta(ch, bh, "Precision"),
        "Pd": delta(ch, bh, "Pd"),
        "FA_ppm": delta(ch, bh, "FA_ppm"),
    }
    g = lock["gates"]
    full_pass = bool(
        full_delta["mIoU"] >= float(g["full_delta_mIoU_min"])
        and full_delta["Precision"] >= float(g["full_delta_precision_min"])
        and full_delta["Pd"] >= float(g["full_delta_pd_min"])
        and full_delta["FA_ppm"] <= float(g["full_delta_fa_ppm_max"])
    )
    hcval_pass = bool(
        hcval_delta["mIoU"] >= float(g["hcval_delta_mIoU_min"])
        and hcval_delta["Precision"] >= float(g["hcval_delta_precision_min"])
        and hcval_delta["Pd"] >= float(g["hcval_delta_pd_min"])
        and hcval_delta["FA_ppm"] <= float(g["hcval_delta_fa_ppm_max"])
    )
    metadata_pass = len(metadata_errors) == 0
    gate_pass = bool(metadata_pass and full_pass and hcval_pass)

    result = {
        "gate": "Gate-CGA-v2.1-seed42-from-zero-paired",
        "decision_rule_predeclared": True,
        "metadata_pass": metadata_pass,
        "metadata_errors": metadata_errors,
        "full_rule_pass": full_pass,
        "hcval_rule_pass": hcval_pass,
        "gate_pass": gate_pass,
        "can_run_seed43_44": gate_pass,
        "can_claim_positive_cga_v21": gate_pass,
        "full": {"baseline": bf, "candidate": cf, "delta": full_delta},
        "hcval": {"baseline": bh, "candidate": ch, "delta": hcval_delta},
        "thresholds": g,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))

    if not metadata_pass:
        raise SystemExit(2)
    if not gate_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/summarize_cga_v21_one_seed.py
```

---

## 10. Runner: `scripts/official/run_cga_v21_seed42_from_zero_paired.sh`

Create a complete train/eval/summarize runner. It must write runtime JSON from actual environment variables, not hard-coded paths.

```bash
mkdir -p scripts/official
cat > scripts/official/run_cga_v21_seed42_from_zero_paired.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
ROOT=$(realpath "${ROOT}")
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-1}
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets}
DATASET_DIR=$(realpath "${DATASET_DIR}")
DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-400}
RUN_ID=${RUN_ID:?RUN_ID is required for from-zero paper evidence}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/official_cga_v21/${RUN_ID}}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-${ROOT}/docs/internal/cga_v2_1/protocol_lock.json}

if [[ "${SEED}" != "42" ]]; then
  echo "CGA-v2.1 first gate is seed42 only; got SEED=${SEED}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "OUTPUT_DIR already exists; from-zero evidence requires a new output dir: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}"

P1_SUMMARY=${P1_SUMMARY:-${ROOT}/docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json}
P1A_SUMMARY=${P1A_SUMMARY:-${ROOT}/docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json}

"${PYTHON}" - <<PY
import json, pathlib, sys
for name, path in [('P1', '${P1_SUMMARY}'), ('P1A', '${P1A_SUMMARY}')]:
    p = pathlib.Path(path)
    if not p.exists():
        raise SystemExit(f'{name} summary missing: {p}')
    data = json.loads(p.read_text())
    if not data.get('gate_pass', False):
        raise SystemExit(f'{name} gate_pass is not true: {p}')
PY

RUNNER_ARGS=${OUTPUT_DIR}/runner_runtime_args.json
cat > "${RUNNER_ARGS}" <<JSON
{
  "protocol_variant": "cga_v2_1",
  "root": "${ROOT}",
  "dataset_dir": "${DATASET_DIR}",
  "dataset_name": "${DATASET_NAME}",
  "seed": ${SEED},
  "epochs": ${EPOCHS},
  "batch_size": 8,
  "patch_size": 256,
  "num_workers": 4,
  "resume": "",
  "optimizer": "Adam",
  "lr": 0.0005,
  "weight_decay": 0.0,
  "ohem_ratio": 0.01,
  "lambda_iou": 1.0,
  "lambda_center": 0.05,
  "lambda_boundary": 0.03,
  "lambda_scale": 0.02,
  "lambda_peak": 0.03,
  "cga_start_epoch": 1,
  "cga_ramp_epochs": 40,
  "lambda_safe_bg": 0.05,
  "safe_bg_topk_ratio": 0.01,
  "safe_bg_ignore_radius": 2,
  "safe_bg_start_epoch": 1,
  "safe_bg_ramp_epochs": 40,
  "aux_ratio_cap": 0.15,
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "strict_load": true,
  "strict_load_required": true,
  "evidence_mode": "paper",
  "protocol": "controlled",
  "paper_evidence_allowed": true,
  "p1_preflight_passed": true,
  "p1a_hcval_source_audit_passed": true,
  "fallback_regularizer_used": false,
  "baseline": {
    "run_name": "MSHNetOHEM",
    "backbone": "mshnet",
    "use_cga": false,
    "cga_variant": "none",
    "regularizer_impl": "none"
  },
  "candidate": {
    "run_name": "MSHNetCGA21",
    "backbone": "mshnet",
    "use_cga": true,
    "cga_variant": "v2_1",
    "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"
  }
}
JSON

"${PYTHON}" tools/official/check_cga_v21_protocol_lock.py \
  --lock "${PROTOCOL_LOCK}" \
  --runtime_args "${RUNNER_ARGS}" \
  --role paired \
  --output "${OUTPUT_DIR}/protocol_lock_check_runner.json"

COMMON_TRAIN=(
  --protocol_variant cga_v2_1
  --evidence_mode paper
  --protocol controlled
  --dataset_dir "${DATASET_DIR}"
  --dataset_name "${DATASET_NAME}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch_size 8
  --patch_size 256
  --num_workers 4
  --lr 0.0005
  --optimizer Adam
  --weight_decay 0.0
  --ohem_ratio 0.01
  --lambda_iou 1.0
  --lambda_center 0.05
  --lambda_boundary 0.03
  --lambda_scale 0.02
  --lambda_peak 0.03
  --cga_start_epoch 1
  --cga_ramp_epochs 40
  --lambda_safe_bg 0.05
  --safe_bg_topk_ratio 0.01
  --safe_bg_ignore_radius 2
  --safe_bg_start_epoch 1
  --safe_bg_ramp_epochs 40
  --aux_ratio_cap 0.15
  --threshold 0.5
  --threshold_selection fixed_predeclared
  --strict_load_required
  --p1_preflight_passed
  --p1a_hcval_source_audit_passed
  --output_dir "${OUTPUT_DIR}"
)

"${PYTHON}" train.py \
  --model_name MSHNetOHEM \
  --backbone_name mshnet \
  --cga_variant none \
  --regularizer_impl none \
  --runtime_args_json "${OUTPUT_DIR}/train_runtime_args_baseline.json" \
  "${COMMON_TRAIN[@]}"

"${PYTHON}" train.py \
  --model_name MSHNetCGA21 \
  --backbone_name mshnet \
  --use_cga \
  --cga_variant v2_1 \
  --regularizer_impl center_boundary_scale_peak_safe_bg_v2_1 \
  --runtime_args_json "${OUTPUT_DIR}/train_runtime_args_candidate.json" \
  "${COMMON_TRAIN[@]}"

BASE_CKPT=${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/MSHNetOHEM_${EPOCHS}.pth.tar
CAND_CKPT=${OUTPUT_DIR}/MSHNetCGA21/seed${SEED}/${DATASET_NAME}/MSHNetCGA21_${EPOCHS}.pth.tar

test_one() {
  local model_name=$1
  local run_name=$2
  local use_cga_flag=$3
  local cga_variant=$4
  local regularizer_impl=$5
  local checkpoint=$6
  local split=$7
  if [[ "${use_cga_flag}" == "true" ]]; then
    USE_CGA_ARGS=(--use_cga)
  else
    USE_CGA_ARGS=()
  fi
  "${PYTHON}" test.py \
    --model_name "${model_name}" \
    --run_name "${run_name}" \
    --backbone_name mshnet \
    "${USE_CGA_ARGS[@]}" \
    --protocol_variant cga_v2_1 \
    --cga_variant "${cga_variant}" \
    --regularizer_impl "${regularizer_impl}" \
    --evidence_mode paper \
    --protocol controlled \
    --dataset_dir "${DATASET_DIR}" \
    --train_dataset_name "${DATASET_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${split}" \
    --seed "${SEED}" \
    --checkpoint "${checkpoint}" \
    --threshold 0.5 \
    --strict_load \
    --output_dir "${OUTPUT_DIR}"
}

test_one MSHNetOHEM MSHNetOHEM false none none "${BASE_CKPT}" test
test_one MSHNetCGA21 MSHNetCGA21 true v2_1 center_boundary_scale_peak_safe_bg_v2_1 "${CAND_CKPT}" test
test_one MSHNetOHEM MSHNetOHEM false none none "${BASE_CKPT}" hcval
test_one MSHNetCGA21 MSHNetCGA21 true v2_1 center_boundary_scale_peak_safe_bg_v2_1 "${CAND_CKPT}" hcval

"${PYTHON}" tools/official/summarize_cga_v21_one_seed.py \
  --lock "${PROTOCOL_LOCK}" \
  --baseline_full "${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json" \
  --candidate_full "${OUTPUT_DIR}/MSHNetCGA21/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json" \
  --baseline_hcval "${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json" \
  --candidate_hcval "${OUTPUT_DIR}/MSHNetCGA21/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json" \
  --output "${OUTPUT_DIR}/gate_cga_v21_seed42_summary.json"
SH
chmod +x scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

Important behavior:

```text
If summarizer metadata fails -> exit 2.
If metrics gate fails -> exit 3.
Therefore runner failure means seed42 gate failure, not script success.
```

---

## 11. Add tests

Create:

```bash
cat > tests/test_cga_v21_preseed42_hard_fixes.py <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from loss import build_loss
from net import resolve_model_config
from test import normalize_state_dict_for_strict_load


def test_build_model_resolver_backward_compatibility():
    assert resolve_model_config("MSHNetOHEM") == ("mshnet", False)
    assert resolve_model_config("MSHNetCGA") == ("mshnet", True)
    assert resolve_model_config("MSHNetCGA21", return_variant=True) == ("mshnet", True, "v2_1")


def test_build_loss_baseline_accepts_none_variant():
    loss = build_loss("MSHNetOHEM", use_cga=False, cga_variant="none")
    assert loss.__class__.__name__ == "MSHNetOHEMLoss"


def test_build_loss_rejects_invalid_safe_bg_ratio():
    with pytest.raises(ValueError):
        build_loss("MSHNetCGA21", use_cga=True, cga_variant="v2_1", safe_bg_topk_ratio=0.0)
    with pytest.raises(ValueError):
        build_loss("MSHNetCGA21", use_cga=True, cga_variant="v2_1", safe_bg_topk_ratio=1.1)


def test_strict_prefix_normalization_rejects_duplicate_after_strip():
    state = {"module.a": torch.tensor(1), "a": torch.tensor(2)}
    normalized = normalize_state_dict_for_strict_load(state)
    assert set(normalized.keys()) == {"module.a", "a"}


def test_strict_prefix_normalization_strips_only_all_shared_prefix():
    state = {"module.a": torch.tensor(1), "module.b": torch.tensor(2)}
    normalized = normalize_state_dict_for_strict_load(state)
    assert set(normalized.keys()) == {"a", "b"}


def test_loss_output_has_no_regularizer_metadata_key():
    loss = build_loss("MSHNetCGA21", use_cga=True, cga_variant="v2_1")
    output = {
        "logits": torch.randn(2, 1, 16, 16),
        "cga_center_logit": torch.randn(2, 1, 16, 16),
        "cga_boundary_logit": torch.randn(2, 1, 16, 16),
        "cga_scale_logit": torch.randn(2, 1, 16, 16),
        "cga_peak_logit": torch.randn(2, 1, 16, 16),
    }
    target = torch.zeros(2, 1, 16, 16)
    target[:, :, 7:9, 7:9] = 1
    out = loss(output, target, epoch=10)
    assert "regularizer_impl" not in out
    assert "reg_raw_over_base" in out
    assert "reg_capped_over_base" in out


def test_protocol_checker_optimizer_mismatch(tmp_path: Path):
    lock = {
        "protocol_variant": "cga_v2_1",
        "dataset": {"root": str(tmp_path), "dataset_dir": str(tmp_path / "datasets"), "dataset_name": "NUDT-SIRST"},
        "roles": {
            "baseline": {"run_name": "MSHNetOHEM", "backbone": "mshnet", "use_cga": False, "cga_variant": "none", "regularizer_impl": "none"},
            "candidate": {"run_name": "MSHNetCGA21", "backbone": "mshnet", "use_cga": True, "cga_variant": "v2_1", "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"},
        },
        "training": {"seed": 42, "epochs": 400, "batch_size": 8, "patch_size": 256, "num_workers": 4, "resume": ""},
        "optimizer": {"optimizer": "Adam", "lr": 0.0005, "weight_decay": 0.0, "ohem_ratio": 0.01, "lambda_iou": 1.0},
        "loss": {"lambda_center": 0.05, "lambda_boundary": 0.03, "lambda_scale": 0.02, "lambda_peak": 0.03, "cga_start_epoch": 1, "cga_ramp_epochs": 40, "lambda_safe_bg": 0.05, "safe_bg_topk_ratio": 0.01, "safe_bg_ignore_radius": 2, "safe_bg_start_epoch": 1, "safe_bg_ramp_epochs": 40, "aux_ratio_cap": 0.15},
        "evaluation": {"threshold": 0.5, "threshold_selection": "fixed_predeclared", "strict_load": True, "strict_load_required": True},
        "evidence": {"evidence_mode": "paper", "protocol": "controlled", "paper_evidence_allowed": True, "p1_preflight_passed": True, "p1a_hcval_source_audit_passed": True, "fallback_regularizer_used": False},
    }
    runtime = {
        "protocol_variant": "cga_v2_1",
        "root": str(tmp_path),
        "dataset_dir": str(tmp_path / "datasets"),
        "dataset_name": "NUDT-SIRST",
        "seed": 42,
        "epochs": 400,
        "batch_size": 8,
        "patch_size": 256,
        "num_workers": 4,
        "resume": "",
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 0.0,
        "ohem_ratio": 0.01,
        "lambda_iou": 1.0,
        "lambda_center": 0.05,
        "lambda_boundary": 0.03,
        "lambda_scale": 0.02,
        "lambda_peak": 0.03,
        "cga_start_epoch": 1,
        "cga_ramp_epochs": 40,
        "lambda_safe_bg": 0.05,
        "safe_bg_topk_ratio": 0.01,
        "safe_bg_ignore_radius": 2,
        "safe_bg_start_epoch": 1,
        "safe_bg_ramp_epochs": 40,
        "aux_ratio_cap": 0.15,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "strict_load": True,
        "strict_load_required": True,
        "evidence_mode": "paper",
        "protocol": "controlled",
        "paper_evidence_allowed": True,
        "p1_preflight_passed": True,
        "p1a_hcval_source_audit_passed": True,
        "fallback_regularizer_used": False,
        "baseline": lock["roles"]["baseline"],
        "candidate": lock["roles"]["candidate"],
    }
    (tmp_path / "datasets").mkdir()
    lock_path = tmp_path / "lock.json"
    runtime_path = tmp_path / "runtime.json"
    out_path = tmp_path / "out.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    proc = subprocess.run([
        "python3", "tools/official/check_cga_v21_protocol_lock.py",
        "--lock", str(lock_path),
        "--runtime_args", str(runtime_path),
        "--role", "paired",
        "--output", str(out_path),
    ], cwd=Path(__file__).resolve().parents[1])
    assert proc.returncode != 0
    data = json.loads(out_path.read_text())
    assert not data["pass"]
    assert any("optimizer.lr" in e for e in data["errors"])
PY
```

---

## 12. Required validation before seed42

Run:

```bash
cd /home/ly/AAAI/CGA-main

python3 -m py_compile \
  net.py \
  model/cga_wrapper.py \
  loss.py \
  train.py \
  test.py \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py

bash -n scripts/official/run_cga_v21_seed42_from_zero_paired.sh

python3 -m pytest tests/test_cga_v21_preseed42_hard_fixes.py

git diff --check
```

If `pytest` is unavailable in the environment:

```bash
python3 - <<'PY'
try:
    import pytest
except Exception as exc:
    raise SystemExit(f'pytest unavailable; install/use env with pytest before seed42: {exc}')
PY
```

Seed42 remains blocked until these checks pass.

---

## 13. Commit before training

After all checks pass:

```bash
cd /home/ly/AAAI/CGA-main

git status --short

git add \
  net.py \
  model/cga_wrapper.py \
  loss.py \
  train.py \
  test.py \
  docs/internal/cga_v2_1/protocol_lock.json \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py \
  scripts/official/run_cga_v21_seed42_from_zero_paired.sh \
  tests/test_cga_v21_preseed42_hard_fixes.py

git commit -m "Add CGA-v2.1 protocol-locked pre-seed42 safeguards"
```

Only after this commit exists should seed42 be launched.

---

## 14. Seed42 command after hard fixes pass

Do not run this until Section 12 and Section 13 are complete.

```bash
cd /home/ly/AAAI/CGA-main

CUDA_DEVICE=1 \
ROOT=/home/ly/AAAI/CGA-main \
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
RUN_ID=seed42_protocol_locked_$(date +%Y%m%d_%H%M%S) \
bash scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

Expected final output:

```text
results/official_cga_v21/<RUN_ID>/gate_cga_v21_seed42_summary.json
```

Decision:

```text
exit 0: metadata passed and metric gate passed; seed43/44 may be considered.
exit 2: metadata/protocol/checkpoint audit failed; fix implementation and rerun seed42 from zero.
exit 3: metrics gate failed; stop CGA-v2.1 positive route unless a new predeclared protocol is created.
```

---

## 15. Final decision rule

CGA-v2.1 seed42 is not a smoke test. It is the first new paper-evidence gate.

```text
If seed42 fails metadata audit:
  P21_INVALID_IMPLEMENTATION
  Fix bug and rerun seed42 from zero.

If seed42 passes metadata but fails metric gate:
  P21_VALID_NEGATIVE_OR_DESIGN_WEAKNESS
  Stop seed43/44.

If seed42 passes metadata and metric gate:
  P21_SEED42_PASS
  Then run seed43/44 paired evidence.
```
