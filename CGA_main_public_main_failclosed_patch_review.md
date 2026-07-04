# CGA-main public `main` review and fail-closed patch plan

Target repo:

```text
https://github.com/Arialliy/CGA-main
```

Date: 2026-07-04

## 0. Verdict

The public GitHub `main` branch is **not yet the full Gate-1A fail-closed multi-backbone implementation** described from the Docker 53f local run.

It is currently a **partial CGA-v2 overlay**:

- It contains the legacy MSHNet / MSHNetCGA path.
- It contains the real four-head CGA auxiliary head: `center / boundary / scale / peak`.
- `loss.py` already contains strict four-head CGA loss logic and ramp parameters.
- But `model/` does **not** yet expose `output_contract.py`, `registry.py`, `cga_wrapper.py`, or `backbones/mshnet_adapter.py` on public `main`.
- `net.py` still builds only `MSHNetCGA` or `MSHNet` from `model_name`.
- `train.py` still uses `--model_name` and `--warm_epoch` and does not yet drive the explicit adapter/registry protocol.

Therefore, if the goal is AAAI-grade evidence, public `main` should still be treated as:

```text
implementation smoke / legacy CGA-v2 overlay only
not multi-backbone paper evidence
```

## 1. Current public-main state

### 1.1 Root repo

Public `main` shows these high-level items:

```text
configs/
docs/
model/
scripts/official/
tests/
tools/official/
utils/
dataset.py
evaluate.py
loss.py
metrics.py
net.py
test.py
train.py
```

This means some repo-level CGA-v2 scaffolding is present.

### 1.2 `model/` directory

Current public `model/` tree shows only:

```text
model/CGA_MSHNet.py
model/MSHNet.py
model/__init__.py
model/cga_aux.py
```

Missing from public `main`:

```text
model/output_contract.py
model/registry.py
model/cga_wrapper.py
model/backbones/mshnet_adapter.py
model/backbones/dnanet_adapter.py
model/backbones/alcnet_adapter.py
model/backbones/acm_adapter.py
model/backbones/isnet_adapter.py
```

### 1.3 `net.py`

Current public `net.py` still has a simple factory:

```python
from model.MSHNet import MSHNet
from model.CGA_MSHNet import MSHNetCGA


def build_model(model_name: str = "MSHNetCGA", input_channels: int = 1, **kwargs) -> nn.Module:
    name = str(model_name).lower()
    if name in {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}:
        return MSHNetCGA(input_channels=input_channels, aux_hidden_channels=int(kwargs.get("aux_hidden_channels", 32)))
    if name in {"mshnet", "mshnetohem", "ohem"}:
        return MSHNet(input_channels=input_channels)
    raise ValueError(f"Unknown model_name={model_name!r}")
```

This is not yet a multi-backbone registry.

### 1.4 `train.py`

Current public `train.py` still exposes:

```python
p.add_argument("--model_name", default="MSHNetCGA")
p.add_argument("--warm_epoch", type=int, default=5)
```

and trains by:

```python
model = build_model(args.model_name).to(device)
criterion = build_loss(args.model_name, ohem_ratio=args.ohem_ratio, warm_epoch=args.warm_epoch).to(device)
...
output = model(img, warm_flag=(epoch <= args.warm_epoch), return_dict=True)
loss_out = criterion(output, mask, epoch=epoch)
```

This is still MSHNet/MSHNetCGA-oriented. It does not yet separate:

```text
mshnet_warm_epoch
cga_start_epoch
cga_ramp_epochs
```

in the actual training entry point.

### 1.5 `CGA_MSHNet.py`

Current public `CGA_MSHNet.py` already wraps MSHNet and attaches `CGAAuxHead`. It extracts:

```text
final_logit
masks
decoder_feature
scale_logits
```

and then returns:

```python
{
    "final_logit": final_logit,
    "final_logits": final_logit,
    "base_logit": final_logit,
    "base_logits": final_logit,
    "masks": masks,
    "scale_logits": scale_logits,
    "scale_logits_up": scale_logits,
    "decoder_feature": decoder_feature,
    "aux_outputs": aux_outputs,
    **aux_outputs,
}
```

This is useful, but it is still a model-specific wrapper rather than a general adapter contract.

Important issue: `_extract_evidence_output()` currently falls back from:

```text
evidence["decoder_feature"]
```

to:

```text
evidence["decoder_features"]["x_d0"]
```

This is acceptable as legacy compatibility, but it should **not** be used as the final paper-mode adapter policy. In paper mode, the adapter must explicitly declare the feature source and fail if that source is absent.

### 1.6 `cga_aux.py`

Current public `model/cga_aux.py` correctly implements the four real CGA heads:

```text
center_head
boundary_head
scale_head
peak_head
```

This should be preserved.

### 1.7 `loss.py`

Current public `loss.py` is closer to the desired direction. It already defines:

```python
REQUIRED_CGA_LOGITS = (
    "cga_center_logit",
    "cga_boundary_logit",
    "cga_scale_logit",
    "cga_peak_logit",
)
```

and in strict mode it requires all four auxiliary logits.

It also already has:

```text
lambda_center
lambda_boundary
lambda_scale
lambda_peak
cga_start_epoch
cga_ramp_epochs
mshnet_warm_epoch
```

inside `build_loss()`.

But `train.py` does not yet pass these cleanly, so the training interface still needs to be updated.

## 2. Required patch boundary

The patch should not change the CGA method into a fallback regularizer. The official paper path must be:

```text
center / boundary / scale / peak
```

Forbidden for paper evidence:

```text
boundary-weighted BCE fallback
background suppression fallback
first-4D-tensor feature guessing
silent adapter fallback
unregistered backbone auto-routing
```

Every result generated by fallback or legacy compatibility mode must be labeled:

```text
smoke-test only
not paper evidence
```

## 3. Patch plan

### 3.1 Add `model/output_contract.py`

Create:

```text
model/output_contract.py
```

Recommended content:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


CONTRACT_VERSION = "cga_v2_failclosed_explicit_v1"


@dataclass(frozen=True)
class FeatureMeta:
    source: str
    stride: int
    channels: int
    resolution: tuple[int, int]


def _is_4d_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.dim() == 4


def require_output_contract(output: dict[str, Any], *, paper_mode: bool = True) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise TypeError("Detector adapter must return a dict in CGA-v2 paper mode.")

    required = ["logits", "features", "feature_meta", "adapter_meta"]
    missing = [k for k in required if k not in output]
    if missing:
        raise KeyError(f"Missing output contract fields: {missing}")

    logits = output["logits"]
    if not _is_4d_tensor(logits):
        raise TypeError("output['logits'] must be a BCHW tensor.")

    features = output["features"]
    if not isinstance(features, (list, tuple)) or len(features) != 1:
        raise TypeError("Paper mode requires exactly one explicitly selected CGA feature.")
    if not _is_4d_tensor(features[0]):
        raise TypeError("output['features'][0] must be a BCHW tensor.")

    feature_meta = output["feature_meta"]
    if not isinstance(feature_meta, (list, tuple)) or len(feature_meta) != 1:
        raise TypeError("output['feature_meta'] must describe exactly one feature.")

    meta = feature_meta[0]
    for key in ("source", "stride", "channels", "resolution"):
        if key not in meta:
            raise KeyError(f"feature_meta[0] missing key: {key}")

    adapter_meta = output["adapter_meta"]
    for key in ("backbone", "logits_source", "feature_source", "contract_version"):
        if key not in adapter_meta:
            raise KeyError(f"adapter_meta missing key: {key}")

    if paper_mode and adapter_meta["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            f"Wrong contract_version={adapter_meta['contract_version']!r}; "
            f"expected {CONTRACT_VERSION!r}"
        )

    return output
```

### 3.2 Add `model/backbones/mshnet_adapter.py`

Create:

```text
model/backbones/__init__.py
model/backbones/mshnet_adapter.py
```

Recommended content:

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.MSHNet import MSHNet
from model.output_contract import CONTRACT_VERSION, require_output_contract


class MSHNetAdapter(nn.Module):
    """Explicit MSHNet adapter for CGA-v2.

    Paper mode forbids silent fallback. The selected CGA feature source must be
    declared by `feature_source` and present in MSHNet's return_dict output.
    """

    def __init__(
        self,
        input_channels: int = 1,
        *,
        feature_source: str = "decoder_feature",
        feature_stride: int = 1,
        feature_channels: int = 16,
        paper_mode: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = MSHNet(input_channels)
        self.feature_source = feature_source
        self.feature_stride = int(feature_stride)
        self.feature_channels = int(feature_channels)
        self.paper_mode = bool(paper_mode)

        allowed = {"decoder_feature", "decoder_features.x_d0"}
        if self.feature_source not in allowed:
            raise ValueError(f"Unsupported MSHNet feature_source={self.feature_source!r}; allowed={sorted(allowed)}")

    def _get_feature(self, evidence: dict[str, Any]) -> torch.Tensor:
        if self.feature_source == "decoder_feature":
            if "decoder_feature" not in evidence or evidence["decoder_feature"] is None:
                raise KeyError("MSHNetAdapter expected evidence['decoder_feature']; no fallback allowed in paper mode.")
            return evidence["decoder_feature"]

        if self.feature_source == "decoder_features.x_d0":
            decoder_features = evidence.get("decoder_features")
            if not isinstance(decoder_features, dict) or "x_d0" not in decoder_features:
                raise KeyError("MSHNetAdapter expected evidence['decoder_features']['x_d0']; no fallback allowed in paper mode.")
            return decoder_features["x_d0"]

        raise AssertionError("unreachable")

    def forward(self, x: torch.Tensor, *, mshnet_warm: bool = True) -> dict[str, Any]:
        evidence = self.backbone(x, warm_flag=mshnet_warm, return_dict=True)
        if not isinstance(evidence, dict):
            raise TypeError("MSHNetAdapter requires MSHNet(..., return_dict=True) to return dict evidence.")

        logits = evidence.get("base_logits", evidence.get("base_logit"))
        if logits is None:
            raise KeyError("MSHNetAdapter requires base_logits/base_logit from MSHNet.")

        feature = self._get_feature(evidence)
        h, w = int(feature.shape[-2]), int(feature.shape[-1])

        out = {
            "logits": logits,
            "final_logit": logits,
            "final_logits": logits,
            "base_logit": logits,
            "base_logits": logits,
            "masks": evidence.get("masks", []),
            "scale_logits": evidence.get("scale_logits", evidence.get("scale_logits_up", [])),
            "features": [feature],
            "feature_meta": [
                {
                    "source": self.feature_source,
                    "stride": self.feature_stride,
                    "channels": self.feature_channels,
                    "resolution": [h, w],
                }
            ],
            "adapter_meta": {
                "backbone": "mshnet",
                "logits_source": "base_logits/base_logit",
                "feature_source": self.feature_source,
                "contract_version": CONTRACT_VERSION,
            },
        }
        return require_output_contract(out, paper_mode=self.paper_mode)
```

### 3.3 Add `model/cga_wrapper.py`

Create:

```text
model/cga_wrapper.py
```

Recommended content:

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.cga_aux import CGAAuxHead
from model.output_contract import require_output_contract


class CGAWrapper(nn.Module):
    """Attach real four-head CGA supervision to an explicit detector adapter."""

    regularizer_impl = "center_boundary_scale_peak"

    def __init__(self, adapter: nn.Module, *, aux_hidden_channels: int = 32, paper_mode: bool = True) -> None:
        super().__init__()
        self.adapter = adapter
        self.paper_mode = bool(paper_mode)

        feature_channels = getattr(adapter, "feature_channels", None)
        if feature_channels is None:
            raise ValueError("CGAWrapper requires adapter.feature_channels; no channel guessing allowed.")

        self.cga_aux_head = CGAAuxHead(
            in_channels=int(feature_channels),
            hidden_channels=int(aux_hidden_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        out = self.adapter(x, **kwargs)
        out = require_output_contract(out, paper_mode=self.paper_mode)

        feature = out["features"][0]
        aux = self.cga_aux_head(feature)

        out = dict(out)
        out["regularizer_impl"] = self.regularizer_impl
        out["fallback_regularizer_used"] = False
        out["aux_outputs"] = aux
        out.update(aux)
        return out
```

### 3.4 Add `model/registry.py`

Create:

```text
model/registry.py
```

Recommended content:

```python
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.backbones.mshnet_adapter import MSHNetAdapter
from model.cga_wrapper import CGAWrapper


AUDITED_BACKBONES = {
    "mshnet": MSHNetAdapter,
}


def build_backbone_adapter(backbone_name: str, *, input_channels: int = 1, paper_mode: bool = True, **kwargs: Any) -> nn.Module:
    name = str(backbone_name).lower()
    if name not in AUDITED_BACKBONES:
        raise ValueError(
            f"Backbone {backbone_name!r} is not audited/registered for paper-mode CGA-v2. "
            "Add an explicit adapter with logits_source, feature_source, stride, channels, and tests first."
        )
    cls = AUDITED_BACKBONES[name]
    return cls(input_channels=input_channels, paper_mode=paper_mode, **kwargs)


def build_detector(
    *,
    backbone_name: str = "mshnet",
    use_cga: bool = False,
    input_channels: int = 1,
    paper_mode: bool = True,
    aux_hidden_channels: int = 32,
    **kwargs: Any,
) -> nn.Module:
    adapter = build_backbone_adapter(
        backbone_name,
        input_channels=input_channels,
        paper_mode=paper_mode,
        **kwargs,
    )
    if use_cga:
        return CGAWrapper(adapter, aux_hidden_channels=aux_hidden_channels, paper_mode=paper_mode)
    return adapter
```

### 3.5 Replace `net.py` with a compatibility factory

Recommended replacement:

```python
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.MSHNet import MSHNet
from model.CGA_MSHNet import MSHNetCGA
from model.registry import build_detector


def _legacy_to_v2(model_name: str) -> tuple[str, bool] | None:
    name = str(model_name).lower()
    if name in {"mshnet", "mshnetohem", "ohem"}:
        return "mshnet", False
    if name in {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}:
        return "mshnet", True
    return None


def build_model(
    model_name: str = "MSHNetCGA",
    input_channels: int = 1,
    *,
    backbone_name: str | None = None,
    use_cga: bool | None = None,
    paper_mode: bool = True,
    legacy_mode: bool = False,
    **kwargs: Any,
) -> nn.Module:
    if legacy_mode:
        name = str(model_name).lower()
        if name in {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}:
            return MSHNetCGA(input_channels=input_channels, aux_hidden_channels=int(kwargs.get("aux_hidden_channels", 32)))
        if name in {"mshnet", "mshnetohem", "ohem"}:
            return MSHNet(input_channels=input_channels)
        raise ValueError(f"Unknown legacy model_name={model_name!r}")

    if backbone_name is None or use_cga is None:
        mapped = _legacy_to_v2(model_name)
        if mapped is None:
            raise ValueError(f"Unknown model_name={model_name!r}; use --backbone_name and --use_cga instead.")
        mapped_backbone, mapped_use_cga = mapped
        backbone_name = mapped_backbone if backbone_name is None else backbone_name
        use_cga = mapped_use_cga if use_cga is None else use_cga

    return build_detector(
        backbone_name=backbone_name,
        use_cga=bool(use_cga),
        input_channels=input_channels,
        paper_mode=paper_mode,
        **kwargs,
    )


class Net(nn.Module):
    def __init__(self, model_name: str = "MSHNetCGA", input_channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_model(model_name=model_name, input_channels=input_channels, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
```

### 3.6 Update `train.py`

Keep legacy compatibility, but add explicit protocol args:

```python
p.add_argument("--backbone_name", default="mshnet")
p.add_argument("--use_cga", action="store_true")
p.add_argument("--paper_mode", action="store_true", default=True)
p.add_argument("--legacy_mode", action="store_true", default=False)
p.add_argument("--mshnet_warm_epoch", type=int, default=5)
p.add_argument("--cga_start_epoch", type=int, default=1)
p.add_argument("--cga_ramp_epochs", type=int, default=40)
p.add_argument("--lambda_center", type=float, default=0.05)
p.add_argument("--lambda_boundary", type=float, default=0.03)
p.add_argument("--lambda_scale", type=float, default=0.02)
p.add_argument("--lambda_peak", type=float, default=0.03)
```

Build model:

```python
model = build_model(
    model_name=args.model_name,
    backbone_name=args.backbone_name,
    use_cga=args.use_cga,
    paper_mode=args.paper_mode,
    legacy_mode=args.legacy_mode,
).to(device)
```

Build loss:

```python
criterion = build_loss(
    args.model_name,
    use_cga=args.use_cga,
    ohem_ratio=args.ohem_ratio,
    mshnet_warm_epoch=args.mshnet_warm_epoch,
    cga_start_epoch=args.cga_start_epoch,
    cga_ramp_epochs=args.cga_ramp_epochs,
    lambda_center=args.lambda_center,
    lambda_boundary=args.lambda_boundary,
    lambda_scale=args.lambda_scale,
    lambda_peak=args.lambda_peak,
    strict_cga_heads=args.paper_mode,
).to(device)
```

Forward:

```python
output = model(img, mshnet_warm=(epoch <= args.mshnet_warm_epoch))
loss_out = criterion(output, mask, epoch=epoch)
```

Log mandatory fields:

```python
mean_stats.update({
    "epoch": epoch,
    "dataset": args.dataset_name,
    "model": args.model_name,
    "backbone_name": args.backbone_name,
    "use_cga": bool(args.use_cga),
    "regularizer_impl": "center_boundary_scale_peak" if args.use_cga else "none",
    "fallback_regularizer_used": False,
    "paper_evidence_allowed": False,  # set true only in paper-evidence runner after P1 passes
    "seed": args.seed,
})
```

### 3.7 Update `test.py` / `evaluate.py`

Evaluation should never require CGA targets. It should use only final logits:

```python
output = model(img, mshnet_warm=False)
if isinstance(output, dict):
    logits = output.get("logits", output.get("final_logits", output.get("final_logit")))
else:
    logits = output
```

Do not compute or report CGA auxiliary losses in evaluation.

## 4. Tests to add or keep

Required tests:

```text
tests/test_cga_failclosed_paper_mode.py
tests/test_adapter_explicit_contract.py
tests/test_multibackbone_factory.py
```

Minimum assertions:

```text
1. MSHNet adapter returns logits/features/feature_meta/adapter_meta.
2. MSHNet adapter fails if declared feature source is missing.
3. CGAWrapper returns cga_center_logit / cga_boundary_logit / cga_scale_logit / cga_peak_logit.
4. CGAWrapper sets fallback_regularizer_used = false.
5. build_model(backbone_name="dnanet", use_cga=True, paper_mode=True) fails closed until DNANet adapter is audited.
6. build_model(backbone_name="mshnet", use_cga=False) runs.
7. build_model(backbone_name="mshnet", use_cga=True) runs.
```

## 5. Execution plan

### 5.1 Public-main sync gate

Run:

```bash
cd /home/AAAI/CGA-main

git status --short
git branch --show-current
git log --oneline -5

git ls-tree -r --name-only HEAD | grep -E \
  'model/output_contract.py|model/backbones/mshnet_adapter.py|model/cga_wrapper.py|model/registry.py|test_cga_failclosed|test_adapter_explicit|test_multibackbone' || true
```

Expected before patch:

```text
missing fail-closed adapter files
```

Expected after patch:

```text
all required files present
```

### 5.2 Contract tests

Run inside Docker 53f:

```bash
cd /home/AAAI/CGA-main
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. pytest -q \
  tests/test_cga_failclosed_paper_mode.py \
  tests/test_adapter_explicit_contract.py \
  tests/test_multibackbone_factory.py
```

Expected:

```text
all passed
```

### 5.3 Smoke training only

Baseline smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_dir datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --model_name MSHNet \
  --epochs 1 \
  --batch_size 2 \
  --patch_size 64 \
  --mshnet_warm_epoch 1 \
  --output_dir results/failclosed_smoke
```

CGA smoke:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset_dir datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --model_name MSHNetCGA \
  --use_cga \
  --epochs 1 \
  --batch_size 2 \
  --patch_size 64 \
  --mshnet_warm_epoch 1 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 1 \
  --output_dir results/failclosed_smoke
```

These are still **smoke only**, not paper evidence.

## 6. Gates

### Gate-1A

```text
MSHNet fail-closed adapter contract and real CGA wrapper smoke validation
```

Allowed evidence role:

```text
implementation smoke / contract validation only
```

Not allowed:

```text
paper table
AAAI main evidence claim
multibackbone claim
```

### Gate-1B

```text
MSHNet legacy-CGA vs adapter-CGA semantic trend check
```

Required checks:

```text
same seed
same dataset smoke subset
same main loss
same four CGA heads
regularizer_impl=center_boundary_scale_peak
fallback_regularizer_used=false
loss_center/loss_boundary/loss_scale/loss_peak present
trend not obviously broken
```

### Gate-P1A

```text
NUDT HC-Val frozen source audit
```

Must pass before seed42 paper-evidence training.

### Gate-2

```text
DNANet adapter explicit source audit
```

DNANet remains fail-closed until it declares:

```text
logits_source
feature_source
feature_stride
feature_channels
feature_resolution
```

## 7. Paper claim boundary

Current safe claim after Gate-1A only:

```text
We refactor CGA into an explicitly contracted, fail-closed wrapper and validate the implementation path on MSHNet smoke tests.
```

Not yet safe:

```text
CGA improves multiple host detectors.
CGA is universally plug-and-play.
CGA supports AAAI main evidence tables on NUDT Full + HC-Val.
```

Safe claim after MSHNet + DNANet + ALCNet/ACM controlled paired seed42 are all positive:

```text
CGA improves multiple host detectors under controlled paired protocols.
```

## 8. Dataset gate remains blocking

Do not start seed42 paper-evidence training until frozen `hcval_NUDT-SIRST.txt` is recovered/audited and P1 passes.

If no frozen HC-Val exists, allowed:

```text
Full-only implementation smoke
repo release contract tests
negative/limitation note
```

Not allowed:

```text
NUDT Full + HC-Val paper claim
HC-Val result table
creating HC-Val after seeing model outputs
```
