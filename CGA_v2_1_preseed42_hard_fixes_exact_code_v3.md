# CGA-v2.1 Pre-Seed42 Hard Fixes — Exact Code v3

**Canonical root:** `/home/ly/AAAI/CGA-main`  
**Status:** `Go` for implementation fixes; `No-Go` for seed42 training.  
**Scope:** protocol/runner/evaluation/metadata/audit fixes only. Do not change the historical CGA-v2 P2 result, checkpoint, threshold, HC-Val split, or CGA-v2 evidence interpretation.

---

## 0. Why this v3 patch exists

CGA-v2 is already frozen as a valid-negative design-weakness result. CGA-v2.1 must therefore be treated as a **new predeclared protocol**, not as a retroactive rescue of CGA-v2.

Before starting any CGA-v2.1 seed42 from-zero paired training, the code must close the following hard gaps:

1. Protocol lock must cover optimizer/training hyperparameters, not only loss/role fields.
2. `build_model` / `build_loss` must not default non-CGA paths to `cga_variant="v2"`.
3. `MSHNetCGA21` alias must force `cga_variant="v2_1"`.
4. Training logs and checkpoints must carry v2.1 metadata, not old `center_boundary_scale_peak` metadata.
5. v2.1 paper-mode training must forbid `--resume`.
6. `test.py` must support strict audited loading and write strict-load metadata.
7. The v2.1 summarizer must audit baseline/candidate role metadata and HC-Val FA/Precision gates.
8. Tests must cover optimizer lock mismatch, alias resolution, non-CGA compatibility, checkpoint metadata, strict load, and HC-Val FA gate failure.

---

## 1. Create protocol lock

Create:

```text
docs/internal/cga_v2_1/protocol_lock.json
```

with this exact content:

```json
{
  "protocol_variant": "cga_v2_1",
  "protocol_version": "2026-07-05-preseed42-v3",
  "root": "/home/ly/AAAI/CGA-main",
  "dataset_dir": "/home/ly/AAAI/CGA-main/datasets",
  "dataset_name": "NUDT-SIRST",
  "seed": 42,
  "epochs": 400,
  "evidence_mode": "paper",
  "protocol": "controlled",
  "p1_preflight_passed": true,
  "p1a_hcval_source_audit_passed": true,
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "strict_load": true,
  "strict_load_required": true,
  "optimizer": {
    "name": "Adam",
    "lr": 0.0005,
    "weight_decay": 0.0,
    "batch_size": 8,
    "patch_size": 256,
    "num_workers": 4,
    "resume": ""
  },
  "loss": {
    "ohem_ratio": 0.01,
    "lambda_iou": 1.0,
    "mshnet_warm_epoch": 5,
    "cga_start_epoch": 1,
    "cga_ramp_epochs": 40,
    "lambda_center": 0.05,
    "lambda_boundary": 0.03,
    "lambda_scale": 0.02,
    "lambda_peak": 0.03,
    "lambda_safe_bg": 0.10,
    "safe_bg_topk_ratio": 0.01,
    "safe_bg_ignore_radius": 3,
    "safe_bg_start_epoch": 1,
    "safe_bg_ramp_epochs": 40,
    "aux_ratio_cap": 0.15
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
  "gates": {
    "full_delta_mIoU_min": 0.020,
    "full_delta_precision_min": 0.010,
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

## 2. Replace `model/cga_wrapper.py`

Replace the whole file:

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

    The wrapper owns the auxiliary heads and reports implementation metadata.
    It does not decide whether a run is paper evidence; that remains runner /
    evidence-manifest metadata.
    """

    V2_IMPL = "center_boundary_scale_peak"
    V21_IMPL = "center_boundary_scale_peak_safe_bg_v2_1"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_name: str,
        feature_channels: int,
        aux_hidden_channels: int = 32,
        cga_variant: str = "v2",
        regularizer_impl: str | None = None,
    ) -> None:
        super().__init__()
        if cga_variant not in {"v2", "v2_1"}:
            raise ValueError(f"CGAWrapper requires cga_variant in {{'v2', 'v2_1'}}, got {cga_variant!r}")

        expected_impl = self.V21_IMPL if cga_variant == "v2_1" else self.V2_IMPL
        if regularizer_impl is None:
            regularizer_impl = expected_impl
        if regularizer_impl != expected_impl:
            raise ValueError(
                f"regularizer_impl={regularizer_impl!r} does not match "
                f"cga_variant={cga_variant!r}; expected {expected_impl!r}"
            )

        self.backbone = backbone
        self.backbone_name = str(backbone_name)
        self.cga_variant = str(cga_variant)
        self.regularizer_impl = str(regularizer_impl)
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
        output["cga_variant"] = self.cga_variant
        output["regularizer_impl"] = self.regularizer_impl
        output["fallback_regularizer_used"] = False
        return output
```

---

## 3. Replace `net.py`

Replace the whole file:

```python
"""Model factory for fail-closed CGA paper-evidence experiments."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.CGA_MSHNet import MSHNetCGA
from model.cga_wrapper import CGAWrapper
from model.registry import available_backbones, get_backbone_builder

_CGA_V2_ALIASES = {"mshnetcga", "cga", "cga-v2", "mshnet_cga", "mshnetcga2"}
_CGA_V21_ALIASES = {
    "mshnetcga21",
    "mshnet_cga21",
    "cga-v2.1",
    "cga-v21",
    "cga_v2_1",
    "cga21",
}
_MSHNET_BASE_ALIASES = {"mshnet", "mshnetohem", "ohem"}

_IMPL_BY_VARIANT = {
    "none": "none",
    "v2": "center_boundary_scale_peak",
    "v2_1": "center_boundary_scale_peak_safe_bg_v2_1",
}


def normalize_cga_variant(value: str | None, *, use_cga: bool, alias_variant: str | None = None) -> str:
    if not use_cga:
        if value not in {None, "", "none", "None"}:
            raise ValueError(f"Non-CGA model requires cga_variant='none', got {value!r}")
        return "none"

    if alias_variant is not None:
        if value not in {None, "", alias_variant}:
            raise ValueError(
                f"Model alias requires cga_variant={alias_variant!r}, got conflicting value {value!r}"
            )
        return alias_variant

    if value in {None, ""}:
        return "v2"
    if value not in {"v2", "v2_1"}:
        raise ValueError(f"Unknown cga_variant={value!r}; expected 'v2', 'v2_1', or 'none'")
    return str(value)


def regularizer_impl_for_variant(cga_variant: str) -> str:
    if cga_variant not in _IMPL_BY_VARIANT:
        raise ValueError(f"Unknown cga_variant={cga_variant!r}")
    return _IMPL_BY_VARIANT[cga_variant]


def resolve_model_config(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    use_cga: bool = False,
    cga_variant: str | None = None,
) -> tuple[str, bool, str]:
    """Resolve legacy aliases into explicit backbone/CGA/variant switches."""
    alias_variant: str | None = None

    if model_name is not None:
        name = str(model_name).lower()
        if name in _CGA_V21_ALIASES:
            return "mshnet", True, normalize_cga_variant(cga_variant, use_cga=True, alias_variant="v2_1")
        if name in _CGA_V2_ALIASES:
            return "mshnet", True, normalize_cga_variant(cga_variant, use_cga=True, alias_variant="v2")
        if name in _MSHNET_BASE_ALIASES:
            return "mshnet", False, normalize_cga_variant(cga_variant, use_cga=False)
        if name.endswith("_cga21") or name.endswith("_cga_v21"):
            backbone = name.split("_cga")[0]
            return backbone, True, normalize_cga_variant(cga_variant, use_cga=True, alias_variant="v2_1")
        if name.endswith("_cga"):
            backbone = name[: -len("_cga")]
            return backbone, True, normalize_cga_variant(cga_variant, use_cga=True, alias_variant="v2")
        if name in available_backbones():
            resolved_use_cga = bool(use_cga)
            return name, resolved_use_cga, normalize_cga_variant(cga_variant, use_cga=resolved_use_cga)
        resolved_use_cga = bool(use_cga)
        return name, resolved_use_cga, normalize_cga_variant(cga_variant, use_cga=resolved_use_cga)

    resolved_use_cga = bool(use_cga)
    return str(backbone_name).lower(), resolved_use_cga, normalize_cga_variant(cga_variant, use_cga=resolved_use_cga)


def build_model(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    input_channels: int = 1,
    use_cga: bool = False,
    cga_variant: str | None = None,
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
    )

    if legacy_model_factory:
        if resolved_backbone == "mshnet" and resolved_use_cga and resolved_variant == "v2":
            return MSHNetCGA(
                input_channels=input_channels,
                aux_hidden_channels=int(aux_hidden_channels),
            )
        raise ValueError(
            "legacy_model_factory only supports the legacy MSHNetCGA v2 path. "
            "Use the explicit adapter registry for CGA-v2.1."
        )

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
        regularizer_impl=regularizer_impl_for_variant(resolved_variant),
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

---

## 4. Modify `loss.py`

### 4.1 Insert this block after `class CGALoss`

```python
@dataclass(frozen=True)
class CGAV21LossConfig(CGALossConfig):
    lambda_safe_bg: float = 0.10
    safe_bg_topk_ratio: float = 0.01
    safe_bg_ignore_radius: int = 3
    safe_bg_start_epoch: int = 1
    safe_bg_ramp_epochs: int = 40
    aux_ratio_cap: float = 0.15


class CGAV21Loss(nn.Module):
    """False-alarm-controlled CGA-v2.1 loss.

    Geometry auxiliary supervision remains center/boundary/scale/peak.
    v2.1 adds logits-level safe-background hard-negative suppression and
    caps the regularization magnitude with a detached scale, preserving the
    regularizer gradient direction when the cap is active.
    """

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
        self._validate_config()
        self.base_loss = MSHNetOHEMLoss(
            ohem_ratio=self.cfg.ohem_ratio,
            lambda_iou=self.cfg.lambda_iou,
            warm_epoch=self.cfg.warm_epoch,
        )

    def _validate_config(self) -> None:
        if not (0.0 < float(self.cfg.safe_bg_topk_ratio) <= 1.0):
            raise ValueError("safe_bg_topk_ratio must be in (0, 1]")
        if int(self.cfg.safe_bg_ignore_radius) < 0:
            raise ValueError("safe_bg_ignore_radius must be >= 0")
        if int(self.cfg.start_epoch) < 0 or int(self.cfg.ramp_epochs) < 0:
            raise ValueError("cga start/ramp epochs must be non-negative")
        if int(self.cfg.safe_bg_start_epoch) < 0 or int(self.cfg.safe_bg_ramp_epochs) < 0:
            raise ValueError("safe-bg start/ramp epochs must be non-negative")
        if float(self.cfg.aux_ratio_cap) <= 0.0:
            raise ValueError("aux_ratio_cap must be > 0")
        if float(self.cfg.lambda_safe_bg) < 0.0:
            raise ValueError("lambda_safe_bg must be >= 0")

    @staticmethod
    def _bce(logit: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
        if logit is None:
            return target.sum() * 0.0
        target = _resize_like(target, logit)
        return F.binary_cross_entropy_with_logits(logit, target)

    @staticmethod
    def _dilate_binary(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        k = 2 * int(radius) + 1
        return F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=int(radius))

    def _safe_background_loss(self, final_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _resize_like(target, final_logit)
        ignore = self._dilate_binary(target, int(self.cfg.safe_bg_ignore_radius))
        safe_bg = (ignore <= 0.0).float()
        zero = torch.zeros_like(final_logit)
        bce = F.binary_cross_entropy_with_logits(final_logit, zero, reduction="none")
        losses: list[torch.Tensor] = []
        for b in range(final_logit.shape[0]):
            vals = bce[b][safe_bg[b] > 0.5].flatten()
            if vals.numel() == 0:
                continue
            k = max(1, int(vals.numel() * float(self.cfg.safe_bg_topk_ratio)))
            losses.append(torch.topk(vals, k=min(k, vals.numel()), largest=True).values.mean())
        if not losses:
            return final_logit.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        if not isinstance(output, dict):
            raise TypeError("CGA-v2.1 paper-mode loss requires dict output with explicit auxiliary logits.")
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
        loss_safe_bg = self._safe_background_loss(final_logit, target)

        geom_w = _ramp_weight(epoch, self.cfg.start_epoch, self.cfg.ramp_epochs)
        safe_w = _ramp_weight(epoch, self.cfg.safe_bg_start_epoch, self.cfg.safe_bg_ramp_epochs)

        geom_aux_total = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )
        safe_aux_total = self.cfg.lambda_safe_bg * loss_safe_bg
        reg_raw = geom_w * geom_aux_total + safe_w * safe_aux_total

        base_total = base["total"]
        base_for_ratio = base_total.detach().abs().clamp_min(1e-6)
        reg_cap = float(self.cfg.aux_ratio_cap) * base_for_ratio
        cap_scale = (reg_cap / reg_raw.detach().abs().clamp_min(1e-6)).clamp(max=1.0)
        reg_capped = reg_raw * cap_scale
        total = base_total + reg_capped

        cap_active = (cap_scale < 0.999).float()
        return {
            "total": total,
            "base_total": base_total.detach(),
            "ohem": base["ohem"],
            "soft_iou": base["soft_iou"],
            "scale": base["scale"],
            "cga_w": torch.tensor(geom_w, device=final_logit.device, dtype=final_logit.dtype),
            "safe_bg_w": torch.tensor(safe_w, device=final_logit.device, dtype=final_logit.dtype),
            "cga_center": loss_center.detach(),
            "cga_boundary": loss_boundary.detach(),
            "cga_scale": loss_scale.detach(),
            "cga_peak": loss_peak.detach(),
            "safe_bg": loss_safe_bg.detach(),
            "geom_aux_total": geom_aux_total.detach(),
            "safe_aux_total": safe_aux_total.detach(),
            "reg_raw": reg_raw.detach(),
            "reg_capped": reg_capped.detach(),
            "reg_raw_over_base": (reg_raw.detach().abs() / base_for_ratio).detach(),
            "reg_capped_over_base": (reg_capped.detach().abs() / base_for_ratio).detach(),
            "cap_scale": cap_scale.detach(),
            "cap_active": cap_active.detach(),
            "regularizer_impl": torch.tensor(0.0, device=final_logit.device, dtype=final_logit.dtype),
        }
```

### 4.2 Replace `build_loss(...)` with this block

```python
def _infer_loss_variant(name: str | None, use_cga: bool | None, cga_variant: str | None) -> tuple[bool, str]:
    name_l = "" if name is None else str(name).lower()
    if use_cga is None:
        use_cga = "cga" in name_l
    if not use_cga:
        if cga_variant not in {None, "", "none", "None"}:
            raise ValueError(f"Non-CGA loss requires cga_variant='none', got {cga_variant!r}")
        return False, "none"
    if any(tok in name_l for tok in ("cga21", "cga-v2.1", "cga-v21", "cga_v2_1")):
        if cga_variant not in {None, "", "v2_1"}:
            raise ValueError(f"MSHNetCGA21 loss requires cga_variant='v2_1', got {cga_variant!r}")
        return True, "v2_1"
    if cga_variant in {None, ""}:
        return True, "v2"
    if cga_variant not in {"v2", "v2_1"}:
        raise ValueError(f"Unknown cga_variant={cga_variant!r}")
    return True, str(cga_variant)


def build_loss(
    name: str | None = "MSHNet",
    *,
    use_cga: bool | None = None,
    cga_variant: str | None = None,
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
    lambda_safe_bg: float = 0.10,
    safe_bg_topk_ratio: float = 0.01,
    safe_bg_ignore_radius: int = 3,
    safe_bg_start_epoch: int = 1,
    safe_bg_ramp_epochs: int = 40,
    aux_ratio_cap: float = 0.15,
    strict_cga_heads: bool = True,
    **kwargs: Any,
) -> nn.Module:
    resolved_use_cga, resolved_variant = _infer_loss_variant(name, use_cga, cga_variant)
    if mshnet_warm_epoch is None:
        mshnet_warm_epoch = int(warm_epoch if warm_epoch is not None else kwargs.get("warm_epoch", 5))
    if "start_epoch" in kwargs:
        cga_start_epoch = int(kwargs["start_epoch"])
    if "ramp_epochs" in kwargs:
        cga_ramp_epochs = int(kwargs["ramp_epochs"])

    if not resolved_use_cga:
        return MSHNetOHEMLoss(
            ohem_ratio=float(ohem_ratio),
            lambda_iou=float(lambda_iou),
            warm_epoch=int(mshnet_warm_epoch),
        )

    if resolved_variant == "v2_1":
        cfg21 = CGAV21LossConfig(
            lambda_center=float(lambda_center),
            lambda_boundary=float(lambda_boundary),
            lambda_scale=float(lambda_scale),
            lambda_peak=float(lambda_peak),
            start_epoch=int(cga_start_epoch),
            ramp_epochs=int(cga_ramp_epochs),
            ohem_ratio=float(ohem_ratio),
            lambda_iou=float(lambda_iou),
            warm_epoch=int(mshnet_warm_epoch),
            lambda_safe_bg=float(lambda_safe_bg),
            safe_bg_topk_ratio=float(safe_bg_topk_ratio),
            safe_bg_ignore_radius=int(safe_bg_ignore_radius),
            safe_bg_start_epoch=int(safe_bg_start_epoch),
            safe_bg_ramp_epochs=int(safe_bg_ramp_epochs),
            aux_ratio_cap=float(aux_ratio_cap),
        )
        return CGAV21Loss(cfg21, strict_cga_heads=strict_cga_heads)

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

## 5. Replace `tools/official/check_cga_v21_protocol_lock.py`

Create this file:

```python
#!/usr/bin/env python3
"""Strict protocol-lock checker for CGA-v2.1 pre-seed42 runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

COMMON_FIELDS = [
    "protocol_variant",
    "root",
    "dataset_dir",
    "dataset_name",
    "seed",
    "epochs",
    "evidence_mode",
    "protocol",
    "p1_preflight_passed",
    "p1a_hcval_source_audit_passed",
    "threshold",
    "threshold_selection",
    "strict_load",
    "strict_load_required",
]

OPTIMIZER_FIELDS = [
    "optimizer.name",
    "optimizer.lr",
    "optimizer.weight_decay",
    "optimizer.batch_size",
    "optimizer.patch_size",
    "optimizer.num_workers",
    "optimizer.resume",
]

LOSS_FIELDS = [
    "loss.ohem_ratio",
    "loss.lambda_iou",
    "loss.mshnet_warm_epoch",
    "loss.cga_start_epoch",
    "loss.cga_ramp_epochs",
    "loss.lambda_center",
    "loss.lambda_boundary",
    "loss.lambda_scale",
    "loss.lambda_peak",
    "loss.lambda_safe_bg",
    "loss.safe_bg_topk_ratio",
    "loss.safe_bg_ignore_radius",
    "loss.safe_bg_start_epoch",
    "loss.safe_bg_ramp_epochs",
    "loss.aux_ratio_cap",
]

ROLE_FIELDS = [
    "model_name",
    "run_name",
    "backbone",
    "use_cga",
    "cga_variant",
    "regularizer_impl",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dotted_get(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def runtime_get(runtime: dict[str, Any], lock_path: str) -> Any:
    aliases = {
        "optimizer.name": "optimizer",
        "optimizer.lr": "lr",
        "optimizer.weight_decay": "weight_decay",
        "optimizer.batch_size": "batch_size",
        "optimizer.patch_size": "patch_size",
        "optimizer.num_workers": "num_workers",
        "optimizer.resume": "resume",
        "loss.ohem_ratio": "ohem_ratio",
        "loss.lambda_iou": "lambda_iou",
        "loss.mshnet_warm_epoch": "mshnet_warm_epoch",
        "loss.cga_start_epoch": "cga_start_epoch",
        "loss.cga_ramp_epochs": "cga_ramp_epochs",
        "loss.lambda_center": "lambda_center",
        "loss.lambda_boundary": "lambda_boundary",
        "loss.lambda_scale": "lambda_scale",
        "loss.lambda_peak": "lambda_peak",
        "loss.lambda_safe_bg": "lambda_safe_bg",
        "loss.safe_bg_topk_ratio": "safe_bg_topk_ratio",
        "loss.safe_bg_ignore_radius": "safe_bg_ignore_radius",
        "loss.safe_bg_start_epoch": "safe_bg_start_epoch",
        "loss.safe_bg_ramp_epochs": "safe_bg_ramp_epochs",
        "loss.aux_ratio_cap": "aux_ratio_cap",
    }
    key = aliases.get(lock_path, lock_path)
    if key not in runtime:
        raise KeyError(key)
    return runtime[key]


def equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(actual) == expected
    if isinstance(expected, float):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return str(actual) == str(expected)


def compare_field(lock: dict[str, Any], runtime: dict[str, Any], field: str) -> dict[str, Any] | None:
    expected = dotted_get(lock, field) if "." in field else lock[field]
    try:
        actual = runtime_get(runtime, field)
    except KeyError:
        return {"field": field, "expected": expected, "actual": None, "reason": "missing_runtime_field"}
    if not equal(expected, actual):
        return {"field": field, "expected": expected, "actual": actual, "reason": "mismatch"}
    return None


def check_runtime_against_lock(lock: dict[str, Any], runtime: dict[str, Any], role: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for field in COMMON_FIELDS + OPTIMIZER_FIELDS + LOSS_FIELDS:
        mismatch = compare_field(lock, runtime, field)
        if mismatch:
            mismatches.append(mismatch)

    if role not in lock.get("roles", {}):
        mismatches.append({"field": "role", "expected": sorted(lock.get("roles", {})), "actual": role, "reason": "unknown_role"})
    else:
        expected_role = lock["roles"][role]
        for field in ROLE_FIELDS:
            expected = expected_role[field]
            actual = runtime.get(field)
            if not equal(expected, actual):
                mismatches.append({"field": f"roles.{role}.{field}", "expected": expected, "actual": actual, "reason": "mismatch"})

    if runtime.get("protocol_variant") != "cga_v2_1":
        mismatches.append({"field": "protocol_variant", "expected": "cga_v2_1", "actual": runtime.get("protocol_variant"), "reason": "not_v21"})
    if runtime.get("evidence_mode") == "paper" and runtime.get("resume", "") not in {"", None}:
        mismatches.append({"field": "resume", "expected": "", "actual": runtime.get("resume"), "reason": "paper_mode_resume_forbidden"})

    return {
        "checker": "check_cga_v21_protocol_lock",
        "role": role,
        "pass": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol_lock", required=True)
    p.add_argument("--runtime_args_json", required=True)
    p.add_argument("--role", required=True, choices=["baseline", "candidate"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = load_json(args.protocol_lock)
    runtime = load_json(args.runtime_args_json)
    result = check_runtime_against_lock(lock, runtime, args.role)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
```

---

## 6. Add strict-load helpers to `test.py`

Insert these functions near the top of `test.py` after imports:

```python
from collections import OrderedDict
from typing import Any

ALLOWED_STRIP_PREFIXES = ("module.", "model.")


def extract_checkpoint_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def strip_prefix_if_all_keys(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor] | None:
    keys = list(state.keys())
    if not keys or not all(k.startswith(prefix) for k in keys):
        return None
    stripped = OrderedDict()
    for key, value in state.items():
        new_key = key[len(prefix):]
        if new_key in stripped:
            raise RuntimeError(f"Prefix stripping {prefix!r} creates duplicate key {new_key!r}")
        stripped[new_key] = value
    return stripped


def normalize_state_dict_whitelist(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], str]:
    model_keys = list(state.keys())
    if not model_keys:
        return state, "raw_empty"
    for prefix in ALLOWED_STRIP_PREFIXES:
        stripped = strip_prefix_if_all_keys(state, prefix)
        if stripped is not None:
            return stripped, f"strip_all:{prefix}"
    return state, "raw"


def load_state_dict_audited(model: torch.nn.Module, ckpt: Any, *, strict_load: bool) -> dict[str, Any]:
    state = extract_checkpoint_state_dict(ckpt)
    normalized, normalization = normalize_state_dict_whitelist(state)
    if strict_load:
        model.load_state_dict(normalized, strict=True)
        return {
            "strict_load_requested": True,
            "strict_load_pass": True,
            "state_dict_normalization": normalization,
            "missing_keys": [],
            "unexpected_keys": [],
        }

    incompatible = model.load_state_dict(normalized, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    return {
        "strict_load_requested": False,
        "strict_load_pass": len(missing) == 0 and len(unexpected) == 0,
        "state_dict_normalization": normalization,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
```

### 6.1 Add parser args to `test.py`

Add these to `parse_args()`:

```python
    p.add_argument("--run_name", default=None)
    p.add_argument("--protocol_variant", default="cga_v2", choices=["cga_v2", "cga_v2_1"])
    p.add_argument("--cga_variant", default=None)
    p.add_argument("--regularizer_impl", default=None)
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--strict_load_required", action="store_true")
    p.add_argument("--protocol", default="controlled", choices=["controlled", "official"])
    p.add_argument("--threshold_selection", default="fixed_predeclared")
```

### 6.2 Replace checkpoint loading in `test.py`

Replace:

```python
ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
```

with:

```python
ckpt = torch.load(args.checkpoint, map_location=device)
if args.evidence_mode == "paper" and args.protocol_variant == "cga_v2_1" and not args.strict_load:
    raise RuntimeError("CGA-v2.1 paper evaluation requires --strict_load")
load_audit = load_state_dict_audited(model, ckpt, strict_load=bool(args.strict_load))
if args.strict_load_required and not load_audit["strict_load_pass"]:
    raise RuntimeError(f"Strict load required but failed: {load_audit}")
```

### 6.3 Update model resolution in `test.py`

Replace the existing `resolve_model_config(...)` call with:

```python
backbone_name, use_cga, resolved_cga_variant = resolve_model_config(
    args.model_name,
    backbone_name=args.backbone_name,
    use_cga=args.use_cga,
    cga_variant=args.cga_variant,
)
run_model_name = args.run_name or _run_model_name(args.model_name, backbone_name, use_cga)
```

and pass `cga_variant` into `build_model`:

```python
model = build_model(
    model_name=args.model_name,
    backbone_name=backbone_name,
    use_cga=use_cga,
    cga_variant=resolved_cga_variant,
    evidence_mode=args.evidence_mode,
).to(device)
```

### 6.4 Add eval output trace inside the inference loop

Before the loop:

```python
eval_trace: dict[str, Any] | None = None
```

After `logit = extract_final_logit(output)`:

```python
if eval_trace is None:
    aux_keys = [k for k in ("cga_center_logit", "cga_boundary_logit", "cga_scale_logit", "cga_peak_logit") if isinstance(output, dict) and k in output]
    eval_trace = {
        "prediction_source": "final_logit",
        "logit_shape": list(logit.shape),
        "logit_min": float(logit.detach().min().cpu()),
        "logit_max": float(logit.detach().max().cpu()),
        "sigmoid_min": float(torch.sigmoid(logit).detach().min().cpu()),
        "sigmoid_max": float(torch.sigmoid(logit).detach().max().cpu()),
        "aux_keys_present": aux_keys,
        "aux_used_for_prediction": False,
    }
```

### 6.5 Add metadata to summary in `test.py`

In `summary.update({...})`, add:

```python
        "run_name": run_model_name,
        "protocol_variant": args.protocol_variant,
        "cga_variant": resolved_cga_variant,
        "regularizer_impl": args.regularizer_impl or ckpt.get("regularizer_impl", "center_boundary_scale_peak" if use_cga else "none"),
        "evidence_mode": args.evidence_mode,
        "protocol": args.protocol,
        "threshold_selection": args.threshold_selection,
        "strict_load": bool(args.strict_load),
        "strict_load_required": bool(args.strict_load_required),
        "strict_load_pass": bool(load_audit["strict_load_pass"]),
        "state_dict_normalization": load_audit["state_dict_normalization"],
        "strict_load_missing_keys": load_audit["missing_keys"],
        "strict_load_unexpected_keys": load_audit["unexpected_keys"],
        "eval_trace": eval_trace or {},
        "checkpoint_metadata": {
            "model_name": ckpt.get("model_name") if isinstance(ckpt, dict) else None,
            "protocol_variant": ckpt.get("protocol_variant") if isinstance(ckpt, dict) else None,
            "cga_variant": ckpt.get("cga_variant") if isinstance(ckpt, dict) else None,
            "regularizer_impl": ckpt.get("regularizer_impl") if isinstance(ckpt, dict) else None,
            "paper_evidence_allowed": ckpt.get("paper_evidence_allowed") if isinstance(ckpt, dict) else None,
            "strict_load_required": ckpt.get("strict_load_required") if isinstance(ckpt, dict) else None,
            "threshold": ckpt.get("threshold") if isinstance(ckpt, dict) else None,
            "threshold_selection": ckpt.get("threshold_selection") if isinstance(ckpt, dict) else None,
            "loss_params": ckpt.get("loss_params") if isinstance(ckpt, dict) else None,
            "optimizer_params": ckpt.get("optimizer_params") if isinstance(ckpt, dict) else None,
        },
```

---

## 7. Modify `train.py`

### 7.1 Add imports

Add:

```python
from typing import Any
```

### 7.2 Add parser args

Add these in `parse_args()`:

```python
    p.add_argument("--run_name", default=None)
    p.add_argument("--protocol_variant", default="cga_v2", choices=["cga_v2", "cga_v2_1"])
    p.add_argument("--cga_variant", default=None)
    p.add_argument("--regularizer_impl", default=None)
    p.add_argument("--strict_load_required", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--threshold_selection", default="fixed_predeclared")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", default="Adam", choices=["Adam"])
    p.add_argument("--lambda_iou", type=float, default=1.0)
    p.add_argument("--lambda_safe_bg", type=float, default=0.10)
    p.add_argument("--safe_bg_topk_ratio", type=float, default=0.01)
    p.add_argument("--safe_bg_ignore_radius", type=int, default=3)
    p.add_argument("--safe_bg_start_epoch", type=int, default=1)
    p.add_argument("--safe_bg_ramp_epochs", type=int, default=40)
    p.add_argument("--aux_ratio_cap", type=float, default=0.15)
    p.add_argument("--write_runtime_args_json", default="")
```

### 7.3 Add helpers before `main()`

```python
def regularizer_impl_for_variant(cga_variant: str) -> str:
    if cga_variant == "none":
        return "none"
    if cga_variant == "v2":
        return "center_boundary_scale_peak"
    if cga_variant == "v2_1":
        return "center_boundary_scale_peak_safe_bg_v2_1"
    raise ValueError(f"Unknown cga_variant={cga_variant!r}")


def build_runtime_args(args: argparse.Namespace, *, run_model_name: str, backbone_name: str, use_cga: bool, cga_variant: str) -> dict[str, Any]:
    return {
        "protocol_variant": args.protocol_variant,
        "root": str(Path.cwd()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": args.dataset_name,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "evidence_mode": args.evidence_mode,
        "protocol": args.protocol,
        "p1_preflight_passed": bool(args.p1_preflight_passed),
        "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
        "threshold": float(args.threshold),
        "threshold_selection": args.threshold_selection,
        "strict_load": bool(args.strict_load_required),
        "strict_load_required": bool(args.strict_load_required),
        "optimizer": args.optimizer,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "patch_size": int(args.patch_size),
        "num_workers": int(args.num_workers),
        "resume": str(args.resume or ""),
        "model_name": str(args.model_name or run_model_name),
        "run_name": run_model_name,
        "backbone": backbone_name,
        "use_cga": bool(use_cga),
        "cga_variant": cga_variant,
        "regularizer_impl": args.regularizer_impl or regularizer_impl_for_variant(cga_variant),
        "ohem_ratio": float(args.ohem_ratio),
        "lambda_iou": float(args.lambda_iou),
        "mshnet_warm_epoch": int(args.mshnet_warm_epoch),
        "cga_start_epoch": int(args.cga_start_epoch),
        "cga_ramp_epochs": int(args.cga_ramp_epochs),
        "lambda_center": float(args.lambda_center),
        "lambda_boundary": float(args.lambda_boundary),
        "lambda_scale": float(args.lambda_scale),
        "lambda_peak": float(args.lambda_peak),
        "lambda_safe_bg": float(args.lambda_safe_bg),
        "safe_bg_topk_ratio": float(args.safe_bg_topk_ratio),
        "safe_bg_ignore_radius": int(args.safe_bg_ignore_radius),
        "safe_bg_start_epoch": int(args.safe_bg_start_epoch),
        "safe_bg_ramp_epochs": int(args.safe_bg_ramp_epochs),
        "aux_ratio_cap": float(args.aux_ratio_cap),
    }


def validate_v21_train_args(args: argparse.Namespace, *, use_cga: bool, cga_variant: str) -> None:
    if args.protocol_variant == "cga_v2_1" and args.evidence_mode == "paper":
        if args.resume:
            raise RuntimeError("CGA-v2.1 paper-mode from-zero training forbids --resume")
        if not args.strict_load_required:
            raise RuntimeError("CGA-v2.1 paper protocol requires --strict_load_required metadata")
        if not args.p1_preflight_passed or not args.p1a_hcval_source_audit_passed:
            raise RuntimeError("CGA-v2.1 paper protocol requires P1 and P1A flags")
        if use_cga and cga_variant != "v2_1":
            raise RuntimeError(f"CGA-v2.1 candidate must use cga_variant='v2_1', got {cga_variant!r}")
        if not use_cga and cga_variant != "none":
            raise RuntimeError(f"CGA-v2.1 baseline must use cga_variant='none', got {cga_variant!r}")
        if not (0.0 < float(args.safe_bg_topk_ratio) <= 1.0):
            raise RuntimeError("safe_bg_topk_ratio must be in (0, 1]")
        if int(args.safe_bg_ignore_radius) < 0:
            raise RuntimeError("safe_bg_ignore_radius must be >= 0")
        if int(args.safe_bg_start_epoch) < 0 or int(args.safe_bg_ramp_epochs) < 0:
            raise RuntimeError("safe-bg start/ramp epochs must be non-negative")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
```

### 7.4 Update model resolution in `main()`

Replace the existing resolution block:

```python
backbone_name, use_cga = resolve_model_config(
    args.model_name,
    backbone_name=args.backbone_name,
    use_cga=args.use_cga,
)
run_model_name = _run_model_name(args.model_name, backbone_name, use_cga)
```

with:

```python
backbone_name, use_cga, resolved_cga_variant = resolve_model_config(
    args.model_name,
    backbone_name=args.backbone_name,
    use_cga=args.use_cga,
    cga_variant=args.cga_variant,
)
run_model_name = args.run_name or _run_model_name(args.model_name, backbone_name, use_cga)
if args.regularizer_impl is None:
    args.regularizer_impl = regularizer_impl_for_variant(resolved_cga_variant)
validate_v21_train_args(args, use_cga=use_cga, cga_variant=resolved_cga_variant)
runtime_args = build_runtime_args(
    args,
    run_model_name=run_model_name,
    backbone_name=backbone_name,
    use_cga=use_cga,
    cga_variant=resolved_cga_variant,
)
if args.write_runtime_args_json:
    write_json(args.write_runtime_args_json, runtime_args)
```

### 7.5 Update `build_model(...)` call

Add `cga_variant=resolved_cga_variant`:

```python
model = build_model(
    model_name=args.model_name,
    backbone_name=backbone_name,
    use_cga=use_cga,
    cga_variant=resolved_cga_variant,
    evidence_mode=args.evidence_mode,
    input_channels=1,
    aux_hidden_channels=args.aux_hidden_channels,
    allow_fallback_regularizer=args.allow_fallback_regularizer,
).to(device)
```

### 7.6 Update `build_loss(...)` call

Replace the current call with:

```python
criterion = build_loss(
    args.model_name or backbone_name,
    use_cga=use_cga,
    cga_variant=resolved_cga_variant,
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
    safe_bg_topk_ratio=args.safe_bg_topk_ratio,
    safe_bg_ignore_radius=args.safe_bg_ignore_radius,
    safe_bg_start_epoch=args.safe_bg_start_epoch,
    safe_bg_ramp_epochs=args.safe_bg_ramp_epochs,
    aux_ratio_cap=args.aux_ratio_cap,
    strict_cga_heads=(args.evidence_mode == "paper" and use_cga),
).to(device)
```

### 7.7 Update optimizer

Replace:

```python
optim = torch.optim.Adam(model.parameters(), lr=args.lr)
```

with:

```python
optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

### 7.8 Replace evidence/checkpoint metadata blocks

Before writing log/checkpoint, build:

```python
loss_params = {
    "ohem_ratio": float(args.ohem_ratio),
    "lambda_iou": float(args.lambda_iou),
    "mshnet_warm_epoch": int(args.mshnet_warm_epoch),
    "cga_start_epoch": int(args.cga_start_epoch),
    "cga_ramp_epochs": int(args.cga_ramp_epochs),
    "lambda_center": float(args.lambda_center),
    "lambda_boundary": float(args.lambda_boundary),
    "lambda_scale": float(args.lambda_scale),
    "lambda_peak": float(args.lambda_peak),
    "lambda_safe_bg": float(args.lambda_safe_bg),
    "safe_bg_topk_ratio": float(args.safe_bg_topk_ratio),
    "safe_bg_ignore_radius": int(args.safe_bg_ignore_radius),
    "safe_bg_start_epoch": int(args.safe_bg_start_epoch),
    "safe_bg_ramp_epochs": int(args.safe_bg_ramp_epochs),
    "aux_ratio_cap": float(args.aux_ratio_cap),
}
optimizer_params = {
    "optimizer": args.optimizer,
    "lr": float(args.lr),
    "weight_decay": float(args.weight_decay),
    "batch_size": int(args.batch_size),
    "patch_size": int(args.patch_size),
    "num_workers": int(args.num_workers),
    "epochs": int(args.epochs),
    "resume": str(args.resume or ""),
}
evidence_meta = {
    "epoch": epoch,
    "dataset": args.dataset_name,
    "model": run_model_name,
    "model_name": run_model_name,
    "run_name": run_model_name,
    "backbone": backbone_name,
    "use_cga": bool(use_cga),
    "protocol_variant": args.protocol_variant,
    "cga_variant": resolved_cga_variant,
    "regularizer_impl": args.regularizer_impl,
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

Then use `evidence_meta` both in logs and checkpoints. The checkpoint dict must include:

```python
{
    **evidence_meta,
    "state_dict": model.state_dict(),
    "optimizer": optim.state_dict(),
}
```

---

## 8. Create `tools/official/summarize_cga_v21_one_seed.py`

```python
#!/usr/bin/env python3
"""Summarize one CGA-v2.1 seed with strict role and HC-Val FA gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(obj: dict[str, Any], key: str) -> float:
    if key in obj:
        return float(obj[key])
    variants = [key, key.lower(), key.upper()]
    for variant in variants:
        if variant in obj:
            return float(obj[variant])
    raise KeyError(f"Missing metric {key!r}; available={sorted(obj.keys())}")


def require(summary: dict[str, Any], field: str, expected: Any, errors: list[dict[str, Any]], where: str) -> None:
    actual = summary.get(field)
    if actual != expected:
        errors.append({"where": where, "field": field, "expected": expected, "actual": actual})


def require_checkpoint_meta(summary: dict[str, Any], field: str, expected: Any, errors: list[dict[str, Any]], where: str) -> None:
    ckpt = summary.get("checkpoint_metadata") or {}
    actual = ckpt.get(field)
    if actual != expected:
        errors.append({"where": where, "field": f"checkpoint_metadata.{field}", "expected": expected, "actual": actual})


def audit_role(summary: dict[str, Any], *, role: str, lock: dict[str, Any], split: str, errors: list[dict[str, Any]]) -> None:
    expected_role = lock["roles"][role]
    where = f"{role}:{split}"
    require(summary, "dataset", lock["dataset_name"], errors, where)
    require(summary, "train_dataset", lock["dataset_name"], errors, where)
    require(summary, "split", split, errors, where)
    require(summary, "seed", lock["seed"], errors, where)
    require(summary, "checkpoint_epoch", lock["epochs"], errors, where)
    require(summary, "threshold", lock["threshold"], errors, where)
    require(summary, "threshold_selection", lock["threshold_selection"], errors, where)
    require(summary, "evidence_mode", lock["evidence_mode"], errors, where)
    require(summary, "protocol", lock["protocol"], errors, where)
    require(summary, "strict_load", True, errors, where)
    require(summary, "strict_load_pass", True, errors, where)
    require(summary, "use_cga", expected_role["use_cga"], errors, where)
    require(summary, "cga_variant", expected_role["cga_variant"], errors, where)
    require(summary, "regularizer_impl", expected_role["regularizer_impl"], errors, where)

    require_checkpoint_meta(summary, "protocol_variant", lock["protocol_variant"], errors, where)
    require_checkpoint_meta(summary, "cga_variant", expected_role["cga_variant"], errors, where)
    require_checkpoint_meta(summary, "regularizer_impl", expected_role["regularizer_impl"], errors, where)
    require_checkpoint_meta(summary, "strict_load_required", True, errors, where)
    require_checkpoint_meta(summary, "threshold", lock["threshold"], errors, where)
    require_checkpoint_meta(summary, "threshold_selection", lock["threshold_selection"], errors, where)

    trace = summary.get("eval_trace") or {}
    if trace.get("prediction_source") != "final_logit" or trace.get("aux_used_for_prediction") is not False:
        errors.append({"where": where, "field": "eval_trace", "expected": "final_logit/no_aux", "actual": trace})


def delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return metric(candidate, key) - metric(baseline, key)


def summarize_v21(
    *,
    lock: dict[str, Any],
    baseline_test: dict[str, Any],
    candidate_test: dict[str, Any],
    baseline_hcval: dict[str, Any],
    candidate_hcval: dict[str, Any],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    audit_role(baseline_test, role="baseline", lock=lock, split="test", errors=errors)
    audit_role(candidate_test, role="candidate", lock=lock, split="test", errors=errors)
    audit_role(baseline_hcval, role="baseline", lock=lock, split="hcval", errors=errors)
    audit_role(candidate_hcval, role="candidate", lock=lock, split="hcval", errors=errors)

    full_delta = {
        "mIoU": delta(candidate_test, baseline_test, "mIoU"),
        "Precision": delta(candidate_test, baseline_test, "Precision"),
        "Pd": delta(candidate_test, baseline_test, "Pd"),
        "FA_ppm": delta(candidate_test, baseline_test, "FA_ppm"),
    }
    hcval_delta = {
        "mIoU": delta(candidate_hcval, baseline_hcval, "mIoU"),
        "Precision": delta(candidate_hcval, baseline_hcval, "Precision"),
        "Pd": delta(candidate_hcval, baseline_hcval, "Pd"),
        "FA_ppm": delta(candidate_hcval, baseline_hcval, "FA_ppm"),
    }
    gates = lock["gates"]
    full_pass = bool(
        full_delta["mIoU"] >= float(gates["full_delta_mIoU_min"])
        and full_delta["Precision"] >= float(gates["full_delta_precision_min"])
        and full_delta["Pd"] >= float(gates["full_delta_pd_min"])
        and full_delta["FA_ppm"] <= float(gates["full_delta_fa_ppm_max"])
    )
    hcval_pass = bool(
        hcval_delta["mIoU"] >= float(gates["hcval_delta_mIoU_min"])
        and hcval_delta["Precision"] >= float(gates["hcval_delta_precision_min"])
        and hcval_delta["Pd"] >= float(gates["hcval_delta_pd_min"])
        and hcval_delta["FA_ppm"] <= float(gates["hcval_delta_fa_ppm_max"])
    )
    metadata_pass = len(errors) == 0
    gate_pass = bool(metadata_pass and full_pass and hcval_pass)
    return {
        "gate": "Gate-CGA-v2.1-seed42-from-zero-paired",
        "decision_rule_predeclared": True,
        "protocol_variant": lock["protocol_variant"],
        "seed": lock["seed"],
        "epoch": lock["epochs"],
        "dataset_name": lock["dataset_name"],
        "baseline": lock["roles"]["baseline"]["model_name"],
        "candidate": lock["roles"]["candidate"]["model_name"],
        "metadata_pass": metadata_pass,
        "metadata_errors": errors,
        "full": {"baseline": baseline_test, "candidate": candidate_test, "delta": full_delta, "gate_pass": full_pass},
        "hcval": {"baseline": baseline_hcval, "candidate": candidate_hcval, "delta": hcval_delta, "gate_pass": hcval_pass},
        "gate_pass": gate_pass,
        "can_run_seed43_44": bool(gate_pass),
        "can_claim_positive_cga_v21": bool(gate_pass),
        "decision": "CGA_V21_SEED42_PASS_RUN_SEED43_44" if gate_pass else "CGA_V21_SEED42_FAIL_STOP",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol_lock", required=True)
    p.add_argument("--baseline_test", required=True)
    p.add_argument("--candidate_test", required=True)
    p.add_argument("--baseline_hcval", required=True)
    p.add_argument("--candidate_hcval", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    result = summarize_v21(
        lock=load_json(args.protocol_lock),
        baseline_test=load_json(args.baseline_test),
        candidate_test=load_json(args.candidate_test),
        baseline_hcval=load_json(args.baseline_hcval),
        candidate_hcval=load_json(args.candidate_hcval),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["metadata_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
```

---

## 9. Create complete seed42 runner

Create:

```text
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-1}
DATASET_DIR=${DATASET_DIR:-/home/ly/AAAI/CGA-main/datasets}
DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}
RUN_ID=${RUN_ID:?RUN_ID is required for from-zero paper evidence}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-docs/internal/cga_v2_1/protocol_lock.json}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/results/official_cga_v21/${RUN_ID}}
AUDIT_DIR=${AUDIT_DIR:-${ROOT}/docs/internal/cga_v2_1/seed42_${RUN_ID}}

if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
  echo "[FATAL] FORCE_TRAIN=1 is forbidden for CGA-v2.1 paper evidence. Use a new RUN_ID/OUTPUT_DIR." >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[FATAL] OUTPUT_DIR already exists: ${OUTPUT_DIR}" >&2
  echo "Use a fresh RUN_ID for from-zero paper evidence." >&2
  exit 2
fi

mkdir -p "${AUDIT_DIR}"

write_runtime_json() {
  local role="$1"
  local output="$2"
  local model_name="$3"
  local run_name="$4"
  local use_cga="$5"
  local cga_variant="$6"
  local regularizer_impl="$7"
  ${PYTHON} - "$output" "$role" "$model_name" "$run_name" "$use_cga" "$cga_variant" "$regularizer_impl" <<'PY'
import json, sys
path, role, model_name, run_name, use_cga_s, cga_variant, regularizer_impl = sys.argv[1:]
use_cga = use_cga_s.lower() == "true"
obj = {
  "protocol_variant": "cga_v2_1",
  "root": "/home/ly/AAAI/CGA-main",
  "dataset_dir": "/home/ly/AAAI/CGA-main/datasets",
  "dataset_name": "NUDT-SIRST",
  "seed": 42,
  "epochs": 400,
  "evidence_mode": "paper",
  "protocol": "controlled",
  "p1_preflight_passed": True,
  "p1a_hcval_source_audit_passed": True,
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "strict_load": True,
  "strict_load_required": True,
  "optimizer": "Adam",
  "lr": 0.0005,
  "weight_decay": 0.0,
  "batch_size": 8,
  "patch_size": 256,
  "num_workers": 4,
  "resume": "",
  "model_name": model_name,
  "run_name": run_name,
  "backbone": "mshnet",
  "use_cga": use_cga,
  "cga_variant": cga_variant,
  "regularizer_impl": regularizer_impl,
  "ohem_ratio": 0.01,
  "lambda_iou": 1.0,
  "mshnet_warm_epoch": 5,
  "cga_start_epoch": 1,
  "cga_ramp_epochs": 40,
  "lambda_center": 0.05,
  "lambda_boundary": 0.03,
  "lambda_scale": 0.02,
  "lambda_peak": 0.03,
  "lambda_safe_bg": 0.10,
  "safe_bg_topk_ratio": 0.01,
  "safe_bg_ignore_radius": 3,
  "safe_bg_start_epoch": 1,
  "safe_bg_ramp_epochs": 40,
  "aux_ratio_cap": 0.15,
  "role": role
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, sort_keys=True)
PY
}

BASE_RUNNER_ARGS="${AUDIT_DIR}/runner_runtime_args_baseline.json"
CAND_RUNNER_ARGS="${AUDIT_DIR}/runner_runtime_args_candidate.json"
BASE_TRAIN_ARGS="${AUDIT_DIR}/train_runtime_args_baseline.json"
CAND_TRAIN_ARGS="${AUDIT_DIR}/train_runtime_args_candidate.json"

write_runtime_json baseline "${BASE_RUNNER_ARGS}" MSHNetOHEM MSHNetOHEM false none none
write_runtime_json candidate "${CAND_RUNNER_ARGS}" MSHNetCGA21 MSHNetCGA21 true v2_1 center_boundary_scale_peak_safe_bg_v2_1

${PYTHON} tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${BASE_RUNNER_ARGS}" \
  --role baseline \
  --output "${AUDIT_DIR}/protocol_check_baseline_runner.json"

${PYTHON} tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${CAND_RUNNER_ARGS}" \
  --role candidate \
  --output "${AUDIT_DIR}/protocol_check_candidate_runner.json"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

COMMON_TRAIN_ARGS=(
  --protocol_variant cga_v2_1
  --evidence_mode paper
  --protocol controlled
  --dataset_dir "${DATASET_DIR}"
  --dataset_name "${DATASET_NAME}"
  --seed "${SEED}"
  --epochs 400
  --batch_size 8
  --patch_size 256
  --num_workers 4
  --optimizer Adam
  --lr 0.0005
  --weight_decay 0.0
  --ohem_ratio 0.01
  --lambda_iou 1.0
  --mshnet_warm_epoch 5
  --cga_start_epoch 1
  --cga_ramp_epochs 40
  --lambda_center 0.05
  --lambda_boundary 0.03
  --lambda_scale 0.02
  --lambda_peak 0.03
  --lambda_safe_bg 0.10
  --safe_bg_topk_ratio 0.01
  --safe_bg_ignore_radius 3
  --safe_bg_start_epoch 1
  --safe_bg_ramp_epochs 40
  --aux_ratio_cap 0.15
  --threshold 0.5
  --threshold_selection fixed_predeclared
  --strict_load_required
  --p1_preflight_passed
  --p1a_hcval_source_audit_passed
  --output_dir "${OUTPUT_DIR}"
  --save_every 400
)

${PYTHON} train.py \
  --model_name MSHNetOHEM \
  --run_name MSHNetOHEM \
  --cga_variant none \
  --regularizer_impl none \
  --write_runtime_args_json "${BASE_TRAIN_ARGS}" \
  "${COMMON_TRAIN_ARGS[@]}"

${PYTHON} tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${BASE_TRAIN_ARGS}" \
  --role baseline \
  --output "${AUDIT_DIR}/protocol_check_baseline_train.json"

${PYTHON} train.py \
  --model_name MSHNetCGA21 \
  --run_name MSHNetCGA21 \
  --cga_variant v2_1 \
  --regularizer_impl center_boundary_scale_peak_safe_bg_v2_1 \
  --write_runtime_args_json "${CAND_TRAIN_ARGS}" \
  "${COMMON_TRAIN_ARGS[@]}"

${PYTHON} tools/official/check_cga_v21_protocol_lock.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --runtime_args_json "${CAND_TRAIN_ARGS}" \
  --role candidate \
  --output "${AUDIT_DIR}/protocol_check_candidate_train.json"

BASE_CKPT="${OUTPUT_DIR}/MSHNetOHEM/seed42/${DATASET_NAME}/MSHNetOHEM_400.pth.tar"
CAND_CKPT="${OUTPUT_DIR}/MSHNetCGA21/seed42/${DATASET_NAME}/MSHNetCGA21_400.pth.tar"

COMMON_TEST_ARGS=(
  --protocol_variant cga_v2_1
  --evidence_mode paper
  --protocol controlled
  --dataset_dir "${DATASET_DIR}"
  --train_dataset_name "${DATASET_NAME}"
  --dataset_name "${DATASET_NAME}"
  --seed "${SEED}"
  --threshold 0.5
  --threshold_selection fixed_predeclared
  --num_workers 1
  --strict_load
  --strict_load_required
  --output_dir "${OUTPUT_DIR}"
)

for SPLIT in test hcval; do
  ${PYTHON} test.py \
    --model_name MSHNetOHEM \
    --run_name MSHNetOHEM \
    --cga_variant none \
    --regularizer_impl none \
    --checkpoint "${BASE_CKPT}" \
    --split "${SPLIT}" \
    "${COMMON_TEST_ARGS[@]}"

  ${PYTHON} test.py \
    --model_name MSHNetCGA21 \
    --run_name MSHNetCGA21 \
    --cga_variant v2_1 \
    --regularizer_impl center_boundary_scale_peak_safe_bg_v2_1 \
    --checkpoint "${CAND_CKPT}" \
    --split "${SPLIT}" \
    "${COMMON_TEST_ARGS[@]}"
done

${PYTHON} tools/official/summarize_cga_v21_one_seed.py \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --baseline_test "${OUTPUT_DIR}/MSHNetOHEM/seed42/${DATASET_NAME}/test/summary_metrics.json" \
  --candidate_test "${OUTPUT_DIR}/MSHNetCGA21/seed42/${DATASET_NAME}/test/summary_metrics.json" \
  --baseline_hcval "${OUTPUT_DIR}/MSHNetOHEM/seed42/${DATASET_NAME}/hcval/summary_metrics.json" \
  --candidate_hcval "${OUTPUT_DIR}/MSHNetCGA21/seed42/${DATASET_NAME}/hcval/summary_metrics.json" \
  --output "${AUDIT_DIR}/seed42_gate_summary.json"
```

---

## 10. Tests to add before seed42

Create:

```text
tests/test_cga_v21_preseed42_hard_fixes.py
```

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from loss import build_loss
from net import resolve_model_config
from test import normalize_state_dict_whitelist
from tools.official.check_cga_v21_protocol_lock import check_runtime_against_lock
from tools.official.summarize_cga_v21_one_seed import summarize_v21


def test_build_model_aliases_and_non_cga_variant_resolution():
    assert resolve_model_config("MSHNetOHEM") == ("mshnet", False, "none")
    assert resolve_model_config("MSHNetCGA21") == ("mshnet", True, "v2_1")
    assert resolve_model_config("MSHNetCGA") == ("mshnet", True, "v2")
    with pytest.raises(ValueError):
        resolve_model_config("MSHNetOHEM", cga_variant="v2_1")


def test_cga_v21_loss_safe_bg_and_ratio_cap_runs():
    loss_fn = build_loss(
        "MSHNetCGA21",
        use_cga=True,
        cga_variant="v2_1",
        safe_bg_topk_ratio=0.01,
        safe_bg_ignore_radius=1,
        aux_ratio_cap=0.15,
    )
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 12:14, 12:14] = 1.0
    output = {
        "logits": torch.randn(2, 1, 32, 32, requires_grad=True),
        "cga_center_logit": torch.randn(2, 1, 32, 32, requires_grad=True),
        "cga_boundary_logit": torch.randn(2, 1, 32, 32, requires_grad=True),
        "cga_scale_logit": torch.randn(2, 1, 32, 32, requires_grad=True),
        "cga_peak_logit": torch.randn(2, 1, 32, 32, requires_grad=True),
        "masks": [],
    }
    out = loss_fn(output, target, epoch=40)
    assert torch.isfinite(out["total"])
    assert "safe_bg" in out
    assert "reg_raw_over_base" in out
    assert "reg_capped_over_base" in out
    assert float(out["reg_capped_over_base"]) <= 0.150001
    out["total"].backward()
    assert output["logits"].grad is not None


def test_cga_v21_loss_rejects_bad_safe_bg_topk_ratio():
    with pytest.raises(ValueError):
        build_loss("MSHNetCGA21", use_cga=True, cga_variant="v2_1", safe_bg_topk_ratio=0.0)
    with pytest.raises(ValueError):
        build_loss("MSHNetCGA21", use_cga=True, cga_variant="v2_1", safe_bg_topk_ratio=1.5)


def make_lock() -> dict:
    return {
        "protocol_variant": "cga_v2_1",
        "root": "/home/ly/AAAI/CGA-main",
        "dataset_dir": "/home/ly/AAAI/CGA-main/datasets",
        "dataset_name": "NUDT-SIRST",
        "seed": 42,
        "epochs": 400,
        "evidence_mode": "paper",
        "protocol": "controlled",
        "p1_preflight_passed": True,
        "p1a_hcval_source_audit_passed": True,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "strict_load": True,
        "strict_load_required": True,
        "optimizer": {"name": "Adam", "lr": 0.0005, "weight_decay": 0.0, "batch_size": 8, "patch_size": 256, "num_workers": 4, "resume": ""},
        "loss": {"ohem_ratio": 0.01, "lambda_iou": 1.0, "mshnet_warm_epoch": 5, "cga_start_epoch": 1, "cga_ramp_epochs": 40, "lambda_center": 0.05, "lambda_boundary": 0.03, "lambda_scale": 0.02, "lambda_peak": 0.03, "lambda_safe_bg": 0.10, "safe_bg_topk_ratio": 0.01, "safe_bg_ignore_radius": 3, "safe_bg_start_epoch": 1, "safe_bg_ramp_epochs": 40, "aux_ratio_cap": 0.15},
        "roles": {
            "baseline": {"model_name": "MSHNetOHEM", "run_name": "MSHNetOHEM", "backbone": "mshnet", "use_cga": False, "cga_variant": "none", "regularizer_impl": "none"},
            "candidate": {"model_name": "MSHNetCGA21", "run_name": "MSHNetCGA21", "backbone": "mshnet", "use_cga": True, "cga_variant": "v2_1", "regularizer_impl": "center_boundary_scale_peak_safe_bg_v2_1"},
        },
        "gates": {"full_delta_mIoU_min": 0.02, "full_delta_precision_min": 0.01, "full_delta_pd_min": -0.001, "full_delta_fa_ppm_max": 0.0, "hcval_delta_mIoU_min": 0.0, "hcval_delta_precision_min": 0.0, "hcval_delta_pd_min": -0.001, "hcval_delta_fa_ppm_max": 50.0},
    }


def make_runtime(role="candidate") -> dict:
    lock = make_lock()
    role_obj = lock["roles"][role]
    runtime = {
        "protocol_variant": "cga_v2_1",
        "root": lock["root"],
        "dataset_dir": lock["dataset_dir"],
        "dataset_name": lock["dataset_name"],
        "seed": 42,
        "epochs": 400,
        "evidence_mode": "paper",
        "protocol": "controlled",
        "p1_preflight_passed": True,
        "p1a_hcval_source_audit_passed": True,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "strict_load": True,
        "strict_load_required": True,
        "optimizer": "Adam",
        "lr": 0.0005,
        "weight_decay": 0.0,
        "batch_size": 8,
        "patch_size": 256,
        "num_workers": 4,
        "resume": "",
        "ohem_ratio": 0.01,
        "lambda_iou": 1.0,
        "mshnet_warm_epoch": 5,
        "cga_start_epoch": 1,
        "cga_ramp_epochs": 40,
        "lambda_center": 0.05,
        "lambda_boundary": 0.03,
        "lambda_scale": 0.02,
        "lambda_peak": 0.03,
        "lambda_safe_bg": 0.10,
        "safe_bg_topk_ratio": 0.01,
        "safe_bg_ignore_radius": 3,
        "safe_bg_start_epoch": 1,
        "safe_bg_ramp_epochs": 40,
        "aux_ratio_cap": 0.15,
        **role_obj,
    }
    return runtime


def test_protocol_checker_catches_optimizer_mismatch():
    lock = make_lock()
    runtime = make_runtime("candidate")
    runtime["lr"] = 0.001
    result = check_runtime_against_lock(lock, runtime, "candidate")
    assert result["pass"] is False
    assert any(m["field"] == "optimizer.lr" for m in result["mismatches"])


def test_protocol_checker_catches_resume():
    lock = make_lock()
    runtime = make_runtime("candidate")
    runtime["resume"] = "old.pth.tar"
    result = check_runtime_against_lock(lock, runtime, "candidate")
    assert result["pass"] is False
    assert any(m["field"] == "optimizer.resume" or m["field"] == "resume" for m in result["mismatches"])


def test_strict_prefix_normalization_requires_all_keys_share_prefix():
    state = {"module.a": torch.tensor(1), "module.b": torch.tensor(2)}
    stripped, mode = normalize_state_dict_whitelist(state)
    assert mode == "strip_all:module."
    assert sorted(stripped.keys()) == ["a", "b"]

    mixed = {"module.a": torch.tensor(1), "b": torch.tensor(2)}
    unstripped, mode = normalize_state_dict_whitelist(mixed)
    assert mode == "raw"
    assert sorted(unstripped.keys()) == ["b", "module.a"]


def summary(role: str, split: str, *, fa_ppm: float, precision: float, miou: float, pd: float) -> dict:
    lock = make_lock()
    r = lock["roles"][role]
    return {
        "dataset": "NUDT-SIRST",
        "train_dataset": "NUDT-SIRST",
        "split": split,
        "seed": 42,
        "checkpoint_epoch": 400,
        "threshold": 0.5,
        "threshold_selection": "fixed_predeclared",
        "evidence_mode": "paper",
        "protocol": "controlled",
        "strict_load": True,
        "strict_load_pass": True,
        "use_cga": r["use_cga"],
        "cga_variant": r["cga_variant"],
        "regularizer_impl": r["regularizer_impl"],
        "mIoU": miou,
        "Precision": precision,
        "Pd": pd,
        "FA_ppm": fa_ppm,
        "eval_trace": {"prediction_source": "final_logit", "aux_used_for_prediction": False},
        "checkpoint_metadata": {"protocol_variant": "cga_v2_1", "cga_variant": r["cga_variant"], "regularizer_impl": r["regularizer_impl"], "strict_load_required": True, "threshold": 0.5, "threshold_selection": "fixed_predeclared"},
    }


def test_summarizer_hcval_fa_gate_fail():
    lock = make_lock()
    result = summarize_v21(
        lock=lock,
        baseline_test=summary("baseline", "test", fa_ppm=10.0, precision=0.9, miou=0.9, pd=0.9),
        candidate_test=summary("candidate", "test", fa_ppm=9.0, precision=0.92, miou=0.93, pd=0.9),
        baseline_hcval=summary("baseline", "hcval", fa_ppm=100.0, precision=0.8, miou=0.7, pd=0.8),
        candidate_hcval=summary("candidate", "hcval", fa_ppm=180.0, precision=0.82, miou=0.72, pd=0.8),
    )
    assert result["metadata_pass"] is True
    assert result["full"]["gate_pass"] is True
    assert result["hcval"]["gate_pass"] is False
    assert result["gate_pass"] is False


def test_summarizer_checkpoint_metadata_audit_fails():
    lock = make_lock()
    bad = summary("candidate", "test", fa_ppm=9.0, precision=0.92, miou=0.93, pd=0.9)
    bad["checkpoint_metadata"]["regularizer_impl"] = "center_boundary_scale_peak"
    result = summarize_v21(
        lock=lock,
        baseline_test=summary("baseline", "test", fa_ppm=10.0, precision=0.9, miou=0.9, pd=0.9),
        candidate_test=bad,
        baseline_hcval=summary("baseline", "hcval", fa_ppm=100.0, precision=0.8, miou=0.7, pd=0.8),
        candidate_hcval=summary("candidate", "hcval", fa_ppm=120.0, precision=0.82, miou=0.72, pd=0.8),
    )
    assert result["metadata_pass"] is False
    assert any("checkpoint_metadata.regularizer_impl" in e["field"] for e in result["metadata_errors"])
```

---

## 11. Pre-seed42 validation commands

Run these before any seed42 training:

```bash
cd /home/ly/AAAI/CGA-main

python3 -m py_compile \
  net.py \
  loss.py \
  train.py \
  test.py \
  model/cga_wrapper.py \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py

bash -n scripts/official/run_cga_v21_seed42_from_zero_paired.sh

python3 -m pytest \
  tests/test_cga_v21_preseed42_hard_fixes.py

git diff --check
```

Then commit the protocol lock and code:

```bash
git add \
  docs/internal/cga_v2_1/protocol_lock.json \
  model/cga_wrapper.py \
  net.py \
  loss.py \
  train.py \
  test.py \
  tools/official/check_cga_v21_protocol_lock.py \
  tools/official/summarize_cga_v21_one_seed.py \
  scripts/official/run_cga_v21_seed42_from_zero_paired.sh \
  tests/test_cga_v21_preseed42_hard_fixes.py

git commit -m "Predeclare CGA-v2.1 false-alarm-controlled protocol"
```

---

## 12. Seed42 is allowed only after commit

Only after the tests and commit pass:

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

## 13. Go / No-Go rules

| Condition | Decision |
|---|---|
| Protocol checker misses optimizer/resume/strict_load/regularizer_impl fields | No-Go seed42 |
| `build_model("MSHNetOHEM")` fails due to CGA variant logic | No-Go seed42 |
| `MSHNetCGA21` does not resolve to `cga_variant="v2_1"` | No-Go seed42 |
| Train log/checkpoint still says `regularizer_impl=center_boundary_scale_peak` for candidate | No-Go seed42 |
| v2.1 paper-mode train allows `--resume` | No-Go seed42 |
| v2.1 eval does not use strict audited load | No-Go seed42 |
| v2.1 summarizer does not enforce HC-Val FA/Precision gate | No-Go seed42 |
| All checks pass and code/protocol lock are committed | Go seed42 |

If seed42 fails, stop. Do not run seed43/44. If seed42 passes both Full and HC-Val gates, then run seed43/44 paired evidence.
