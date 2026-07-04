# CGA-main 多 Backbone 改造方案：把 CGA 从 `MSHNetCGA` 解耦成可插拔正则化方法

> 目标：不要把论文和代码叙事写死为 `MSHNet + CGA`。把 CGA 改造成 **training-time plug-and-play regularization**，让同一个 CGA 机制能挂到 `MSHNet / DNANet / ALCNet / ACM / ISNet-style` 等多个 infrared small target detector/backbone 上。

---

## 0. 当前仓库诊断

当前 `CGA-main` 已经是一个较小的 CGA-v2/OHCM-MSHNet 风格仓库，核心入口大概是：

```text
train.py
net.py
loss.py
dataset.py
test.py
evaluate.py
metrics.py
model/
  MSHNet.py
  CGA_MSHNet.py
  cga_aux.py
configs/
scripts/official/
tools/official/
tests/
```

当前问题不是训练循环本身，而是 **模型工厂和 CGA 实现被绑定在 MSHNet 上**：

```python
# net.py 当前逻辑大意
from model.MSHNet import MSHNet
from model.CGA_MSHNet import MSHNetCGA

if model_name in {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}:
    return MSHNetCGA(...)

if model_name in {"mshnet", "mshnetohem", "ohem"}:
    return MSHNet(...)
```

`train.py` 当前训练入口也只通过 `model_name` 建模：

```python
model = build_model(args.model_name).to(device)
criterion = build_loss(args.model_name, ohem_ratio=args.ohem_ratio, warm_epoch=args.warm_epoch).to(device)

output = model(img, warm_flag=(epoch <= args.warm_epoch), return_dict=True)
loss_out = criterion(output, mask, epoch=epoch)
```

这会导致实验叙事天然变成：

```text
MSHNetCGA 是一个新网络
```

而不是：

```text
CGA 是一个可插拔正则化方法，可以作用到多个 ISTD detector/backbone
```

所以代码改造的核心是：

```text
model_name:     控制实验名字 / 兼容旧脚本
backbone_name:  控制宿主 detector/backbone
use_cga:        控制是否启用 CGA regularization
```

---

## 1. 总体改造原则

### 1.1 保留旧路径，不要直接删除 `MSHNetCGA`

保留这些旧入口：

```text
model/MSHNet.py
model/CGA_MSHNet.py
model/cga_aux.py
net.py 中的 MSHNet / MSHNetCGA 兼容分支
```

原因：旧路径能作为 reproduction / sanity check。新路径应该是增量添加：

```text
MSHNet legacy baseline
MSHNet legacy CGA
MSHNet adapter baseline
MSHNet adapter + generic CGA
DNANet adapter baseline
DNANet adapter + generic CGA
ALCNet/ACM/ISNet-style adapter baseline
ALCNet/ACM/ISNet-style adapter + generic CGA
```

### 1.2 不要让 CGA 依赖某一个 backbone 的私有 feature 命名

错误写法：

```python
# 不推荐
x1, x2, x3, x4 = mshnet.encoder(...)
loss_cga = cga(x3, x4, mask)
```

推荐写法：

```python
# 推荐
output = backbone_adapter(img, return_dict=True)
output = normalize_detector_output(output)
loss_cga = cga_regularizer(output, mask)
```

也就是说，CGA 只能依赖统一协议：

```python
output = {
    "logits": Tensor[B, 1, H, W],
    "pred": Tensor[B, 1, H, W],
    "features": list[Tensor] | tuple[Tensor],
    "losses": dict[str, Tensor],
    "raw": Any,
}
```

### 1.3 推理阶段不要计算 CGA loss

如果论文要说 CGA 是 training-time regularization，那么测试阶段必须是：

```python
model.eval()
with torch.no_grad():
    output = model(img, return_dict=True)
    logits = output["logits"]
```

不要在 test/eval 阶段传 `mask`，也不要计算 `loss_cga`。

---

## 2. 建议新增文件结构

在现有 `model/` 下新增：

```text
model/
  common_output.py              # 统一不同 backbone 的输出格式
  registry.py                   # backbone registry / model factory
  cga_regularizer.py            # backbone-agnostic CGA regularizer wrapper
  cga_wrapper.py                # 把任意 backbone 包成 backbone + CGA
  backbones/
    __init__.py
    base.py
    mshnet_adapter.py
    dnanet_adapter.py
    alcnet_adapter.py
    acm_adapter.py
    isnet_adapter.py
```

不要一开始就重写 `train.py/test.py/loss.py` 的全部逻辑；先做最小侵入式改造。

---

## 3. 新增 `model/common_output.py`

作用：把不同 detector 的输出统一成一个 dict。DNANet、ALCNet、ACM、ISNet-style 的 forward 输出可能是 tensor、tuple、list、dict。训练和评估不能到处写 `if isinstance(...)`，否则后面会非常难维护。

```python
# model/common_output.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

LOGIT_KEYS = (
    "logits",
    "pred",
    "prediction",
    "out",
    "output",
    "mask",
    "saliency",
)

FEATURE_KEYS = (
    "features",
    "feats",
    "multi_scale_features",
    "encoder_features",
    "decoder_features",
    "aux_features",
)


def _is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def _tensor_list(x: Any) -> List[torch.Tensor]:
    if x is None:
        return []
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, (list, tuple)):
        return [v for v in x if torch.is_tensor(v)]
    return []


def extract_logits(raw: Any) -> torch.Tensor:
    """Extract the final segmentation logits from common detector outputs."""
    if torch.is_tensor(raw):
        return raw

    if isinstance(raw, dict):
        for key in LOGIT_KEYS:
            value = raw.get(key, None)
            if torch.is_tensor(value):
                return value
        # Fallback: first 4D tensor in dict values.
        for value in raw.values():
            if torch.is_tensor(value) and value.ndim >= 3:
                return value

    if isinstance(raw, (list, tuple)):
        # Most ISTD segmentation models either return [logits, aux...] or [..., logits].
        # Prefer the first 4D tensor; if this mismatches your backbone, override in adapter.
        for value in raw:
            if torch.is_tensor(value) and value.ndim >= 3:
                return value

    raise TypeError(
        "Cannot extract logits from detector output. "
        "Please return a Tensor or dict with one of keys: "
        f"{LOGIT_KEYS}. Got type={type(raw)!r}."
    )


def extract_features(raw: Any) -> List[torch.Tensor]:
    """Extract optional feature tensors for feature-level CGA."""
    if isinstance(raw, dict):
        for key in FEATURE_KEYS:
            value = raw.get(key, None)
            tensors = _tensor_list(value)
            if tensors:
                return tensors

        # Conservative fallback: collect non-logit tensors with 4D shape.
        features: List[torch.Tensor] = []
        for key, value in raw.items():
            if key in LOGIT_KEYS:
                continue
            if torch.is_tensor(value) and value.ndim == 4:
                features.append(value)
            elif isinstance(value, (list, tuple)):
                features.extend([v for v in value if torch.is_tensor(v) and v.ndim == 4])
        return features

    if isinstance(raw, (list, tuple)):
        return [v for v in raw if torch.is_tensor(v) and v.ndim == 4]

    return []


def normalize_detector_output(raw: Any) -> Dict[str, Any]:
    """Return a stable dict protocol used by loss, CGA, test, and metrics."""
    logits = extract_logits(raw)
    features = extract_features(raw)

    if isinstance(raw, dict):
        output: Dict[str, Any] = dict(raw)
    else:
        output = {"raw": raw}

    output.setdefault("logits", logits)
    output.setdefault("pred", logits)
    output.setdefault("features", features if features else [logits])
    output.setdefault("losses", {})
    return output
```

---

## 4. 新增 backbone adapter 基类

### 4.1 `model/backbones/__init__.py`

```python
# model/backbones/__init__.py
from .base import BackboneAdapter

__all__ = ["BackboneAdapter"]
```

### 4.2 `model/backbones/base.py`

```python
# model/backbones/base.py
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.common_output import normalize_detector_output


class BackboneAdapter(nn.Module):
    """Adapter that converts arbitrary detector outputs to the CGA-main output protocol."""

    backbone_name: str = "generic"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x, return_dict: bool = True, **kwargs: Any):
        # Most external ISTD models only accept x.
        raw = self.backbone(x)
        output = normalize_detector_output(raw)
        output.setdefault("backbone_name", self.backbone_name)
        return output if return_dict else output["logits"]
```

---

## 5. 新增 `MSHNetAdapter`

先把当前 MSHNet 接入新协议，确保新旧路径能对齐。

```python
# model/backbones/mshnet_adapter.py
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.MSHNet import MSHNet
from model.common_output import normalize_detector_output


class MSHNetAdapter(nn.Module):
    backbone_name = "mshnet"

    def __init__(self, input_channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.backbone = MSHNet(input_channels=input_channels)

    def forward(
        self,
        x,
        warm_flag: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        # Current train.py already calls warm_flag/return_dict for MSHNet-style models.
        try:
            raw = self.backbone(x, warm_flag=warm_flag, return_dict=True)
        except TypeError:
            raw = self.backbone(x)

        output = normalize_detector_output(raw)
        output.setdefault("backbone_name", self.backbone_name)
        return output if return_dict else output["logits"]
```

---

## 6. 新增外部 backbone adapter 占位文件

当前仓库没有 DNANet / ALCNet / ACM / ISNet-style 的实现文件，所以不能只改 `net.py` 就直接跑这些模型。你需要把对应 backbone 的实现拷贝到下面这些文件之一：

```text
model/backbones/dnanet_impl.py
model/backbones/alcnet_impl.py
model/backbones/acm_impl.py
model/backbones/isnet_impl.py
```

然后让 adapter 调用它们。

### 6.1 `model/backbones/dnanet_adapter.py`

```python
# model/backbones/dnanet_adapter.py
from __future__ import annotations

from typing import Any

from model.backbones.base import BackboneAdapter


class DNANetAdapter(BackboneAdapter):
    backbone_name = "dnanet"

    def __init__(self, input_channels: int = 1, **kwargs: Any) -> None:
        try:
            from model.backbones.dnanet_impl import DNANet
        except Exception as exc:
            raise ImportError(
                "DNANet implementation is missing. "
                "Copy your DNANet code to model/backbones/dnanet_impl.py "
                "and expose a class named DNANet."
            ) from exc

        # Adjust this constructor to match your DNANet implementation.
        backbone = DNANet(input_channels=input_channels, **kwargs)
        super().__init__(backbone)
```

### 6.2 `model/backbones/alcnet_adapter.py`

```python
# model/backbones/alcnet_adapter.py
from __future__ import annotations

from typing import Any

from model.backbones.base import BackboneAdapter


class ALCNetAdapter(BackboneAdapter):
    backbone_name = "alcnet"

    def __init__(self, input_channels: int = 1, **kwargs: Any) -> None:
        try:
            from model.backbones.alcnet_impl import ALCNet
        except Exception as exc:
            raise ImportError(
                "ALCNet implementation is missing. "
                "Copy your ALCNet code to model/backbones/alcnet_impl.py "
                "and expose a class named ALCNet."
            ) from exc

        backbone = ALCNet(input_channels=input_channels, **kwargs)
        super().__init__(backbone)
```

### 6.3 `model/backbones/acm_adapter.py`

```python
# model/backbones/acm_adapter.py
from __future__ import annotations

from typing import Any

from model.backbones.base import BackboneAdapter


class ACMAdapter(BackboneAdapter):
    backbone_name = "acm"

    def __init__(self, input_channels: int = 1, **kwargs: Any) -> None:
        try:
            from model.backbones.acm_impl import ACM
        except Exception as exc:
            raise ImportError(
                "ACM implementation is missing. "
                "Copy your ACM code to model/backbones/acm_impl.py "
                "and expose a class named ACM."
            ) from exc

        backbone = ACM(input_channels=input_channels, **kwargs)
        super().__init__(backbone)
```

### 6.4 `model/backbones/isnet_adapter.py`

```python
# model/backbones/isnet_adapter.py
from __future__ import annotations

from typing import Any

from model.backbones.base import BackboneAdapter


class ISNetAdapter(BackboneAdapter):
    backbone_name = "isnet"

    def __init__(self, input_channels: int = 1, **kwargs: Any) -> None:
        try:
            from model.backbones.isnet_impl import ISNet
        except Exception as exc:
            raise ImportError(
                "ISNet-style implementation is missing. "
                "Copy your ISNet-style code to model/backbones/isnet_impl.py "
                "and expose a class named ISNet."
            ) from exc

        backbone = ISNet(input_channels=input_channels, **kwargs)
        super().__init__(backbone)
```

---

## 7. 新增 `model/cga_regularizer.py`

这里的关键原则：**不要重新发明一个假 CGA**。你现在仓库已有 `model/cga_aux.py`，所以正式实验应把里面的真实 CGA 逻辑迁移/封装到 `CGARegularizer`。

建议先写一个 wrapper 接口，真实 CGA 代码逐步迁入：

```python
# model/cga_regularizer.py
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common_output import normalize_detector_output


class CGARegularizer(nn.Module):
    """Backbone-agnostic CGA regularization wrapper.

    Formal use:
        Replace _fallback_regularization() with the real logic currently in model/cga_aux.py.

    Contract:
        input:  normalized detector output + binary mask
        output: scalar Tensor loss
    """

    def __init__(
        self,
        aux_hidden_channels: int = 32,
        boundary_weight: float = 1.0,
        background_weight: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.aux_hidden_channels = int(aux_hidden_channels)
        self.boundary_weight = float(boundary_weight)
        self.background_weight = float(background_weight)

        # If model/cga_aux.py already exposes a reusable module, import it here.
        # Example:
        #   from model.cga_aux import CGAAuxHead
        #   self.real_cga = CGAAuxHead(...)
        self.real_cga = None

    def forward(self, output: Dict[str, Any], mask: torch.Tensor) -> torch.Tensor:
        output = normalize_detector_output(output)

        if self.real_cga is not None:
            # Replace this block with the real signature of your existing cga_aux module.
            # Example:
            #   return self.real_cga(output["features"], output["logits"], mask)
            return self.real_cga(output, mask)

        # Only for pipeline smoke tests. Do not call this the final paper CGA.
        return self._fallback_regularization(output, mask)

    def _fallback_regularization(self, output: Dict[str, Any], mask: torch.Tensor) -> torch.Tensor:
        """Safe temporary regularizer for smoke tests.

        This is not a replacement for your real CGA. It only makes the multi-backbone
        plumbing executable before cga_aux.py is refactored.
        """
        logits = output["logits"]
        if logits.shape[-2:] != mask.shape[-2:]:
            target = F.interpolate(mask.float(), size=logits.shape[-2:], mode="nearest")
        else:
            target = mask.float()

        prob = torch.sigmoid(logits)
        target = (target > 0.5).float()

        # Boundary map from binary mask morphology.
        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - target, kernel_size=3, stride=1, padding=1)
        boundary = (dilated - eroded).clamp(0.0, 1.0)

        bce = F.binary_cross_entropy(prob, target, reduction="none")
        target_preserve = (bce * (1.0 + self.boundary_weight * boundary)).mean()

        background = (1.0 - dilated).clamp(0.0, 1.0)
        clutter_suppress = (prob * background).mean()

        return target_preserve + self.background_weight * clutter_suppress
```

正式版替换点：

```python
# 把 model/cga_aux.py 中与 MSHNet feature 名称绑定的逻辑
# 改成只接收 output["features"], output["logits"], mask。

return self.real_cga(output["features"], output["logits"], mask)
```

---

## 8. 新增 `model/cga_wrapper.py`

`CGAWrapper` 的作用：把任意 backbone 包成 `backbone + CGA`。训练阶段有 mask 时计算 CGA loss；测试阶段没有 mask 时只返回 logits。

```python
# model/cga_wrapper.py
from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.common_output import normalize_detector_output
from model.cga_regularizer import CGARegularizer


class CGAWrapper(nn.Module):
    """Wrap any ISTD detector/backbone with training-time CGA regularization."""

    use_cga = True

    def __init__(
        self,
        backbone: nn.Module,
        cga: CGARegularizer,
        lambda_cga: float = 1.0,
        warm_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.cga = cga
        self.lambda_cga = float(lambda_cga)
        self.warm_epoch = int(warm_epoch)

    def forward(
        self,
        x,
        mask=None,
        epoch: int | None = None,
        warm_flag: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        raw = self.backbone(x, warm_flag=warm_flag, return_dict=True, **kwargs)
        output = normalize_detector_output(raw)

        should_apply_cga = self.training and mask is not None
        if epoch is not None and self.warm_epoch > 0:
            should_apply_cga = should_apply_cga and (int(epoch) > self.warm_epoch)

        if should_apply_cga:
            loss_cga = self.cga(output, mask)
            output.setdefault("losses", {})
            output["losses"]["loss_cga"] = self.lambda_cga * loss_cga

        return output if return_dict else output["logits"]
```

---

## 9. 新增 `model/registry.py`

```python
# model/registry.py
from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn


def _norm(name: str | None) -> str:
    return str(name or "").lower().replace("-", "_").replace("+", "_")


def list_backbones() -> List[str]:
    return ["mshnet", "dnanet", "alcnet", "acm", "isnet"]


def build_backbone(backbone_name: str, input_channels: int = 1, **kwargs: Any) -> nn.Module:
    name = _norm(backbone_name)

    if name in {"mshnet", "msh", "mshnetohem"}:
        from model.backbones.mshnet_adapter import MSHNetAdapter

        return MSHNetAdapter(input_channels=input_channels, **kwargs)

    if name in {"dnanet", "dna"}:
        from model.backbones.dnanet_adapter import DNANetAdapter

        return DNANetAdapter(input_channels=input_channels, **kwargs)

    if name in {"alcnet", "alc"}:
        from model.backbones.alcnet_adapter import ALCNetAdapter

        return ALCNetAdapter(input_channels=input_channels, **kwargs)

    if name in {"acm", "acmnet"}:
        from model.backbones.acm_adapter import ACMAdapter

        return ACMAdapter(input_channels=input_channels, **kwargs)

    if name in {"isnet", "isnet_style", "isnetstyle"}:
        from model.backbones.isnet_adapter import ISNetAdapter

        return ISNetAdapter(input_channels=input_channels, **kwargs)

    raise ValueError(f"Unknown backbone_name={backbone_name!r}. Available: {list_backbones()}")
```

---

## 10. 修改 `net.py`

把 `net.py` 从“只认识 MSHNet / MSHNetCGA”改成“兼容旧入口 + 支持新 registry”。

推荐直接替换为：

```python
# net.py
"""Model factory for legacy MSHNet/CGA-v2 and multi-backbone CGA."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from model.MSHNet import MSHNet
from model.CGA_MSHNet import MSHNetCGA
from model.cga_regularizer import CGARegularizer
from model.cga_wrapper import CGAWrapper
from model.registry import build_backbone


def _norm(name: str | None) -> str:
    return str(name or "").lower().replace("-", "_").replace("+", "_")


def _infer_backbone_from_model_name(model_name: str) -> str:
    name = _norm(model_name)
    for token in ("dnanet", "alcnet", "acm", "isnet", "mshnet"):
        if token in name:
            return token
    return "mshnet"


def _infer_use_cga(model_name: str, use_cga: bool | None) -> bool:
    if use_cga is not None:
        return bool(use_cga)
    name = _norm(model_name)
    return "cga" in name


def build_model(
    model_name: str = "MSHNetCGA",
    input_channels: int = 1,
    backbone_name: str | None = None,
    use_cga: bool | None = None,
    lambda_cga: float = 1.0,
    aux_hidden_channels: int = 32,
    warm_epoch: int = 0,
    legacy: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Build model.

    Legacy examples:
        build_model("MSHNet")
        build_model("MSHNetCGA")

    New examples:
        build_model("mshnet", backbone_name="mshnet", use_cga=False)
        build_model("mshnet_cga", backbone_name="mshnet", use_cga=True)
        build_model("dnanet_cga", backbone_name="dnanet", use_cga=True)
    """
    name = _norm(model_name)

    # Keep old route unchanged for exact reproduction.
    if legacy or (backbone_name is None and use_cga is None):
        if name in {"mshnetcga", "cga", "cga_v2", "mshnet_cga"}:
            return MSHNetCGA(
                input_channels=input_channels,
                aux_hidden_channels=int(aux_hidden_channels),
            )
        if name in {"mshnet", "mshnetohem", "ohem"}:
            return MSHNet(input_channels=input_channels)

    if backbone_name is None:
        backbone_name = _infer_backbone_from_model_name(model_name)

    cga_enabled = _infer_use_cga(model_name, use_cga)
    backbone = build_backbone(backbone_name, input_channels=input_channels, **kwargs)

    if not cga_enabled:
        return backbone

    cga = CGARegularizer(aux_hidden_channels=aux_hidden_channels, **kwargs)
    return CGAWrapper(
        backbone=backbone,
        cga=cga,
        lambda_cga=lambda_cga,
        warm_epoch=warm_epoch,
    )


class Net(nn.Module):
    """Compatibility wrapper matching the legacy `Net(model_name=...)` pattern."""

    def __init__(self, model_name: str = "MSHNetCGA", input_channels: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_model(model_name=model_name, input_channels=input_channels, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)
```

---

## 11. 修改 `train.py`

### 11.1 新增训练参数

在 `parse_args()` 里增加：

```python
p.add_argument("--backbone_name", default="", help="mshnet/dnanet/alcnet/acm/isnet")
p.add_argument("--use_cga", type=int, default=-1, choices=[-1, 0, 1], help="-1 infer from model_name; 0 off; 1 on")
p.add_argument("--lambda_cga", type=float, default=1.0)
p.add_argument("--aux_hidden_channels", type=int, default=32)
p.add_argument("--legacy_model_factory", action="store_true", help="use old MSHNet/MSHNetCGA factory path")
```

### 11.2 新增辅助函数

放在 `_loss_value()` 后面：

```python
def _as_loss_dict(loss_out):
    if isinstance(loss_out, dict):
        return loss_out
    return {"main": loss_out, "total": loss_out}


def _output_aux_loss(output):
    if not isinstance(output, dict):
        return None, {}
    losses = output.get("losses", {}) or {}
    if not losses:
        return None, {}
    aux_total = None
    clean = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            clean[key] = value
            aux_total = value if aux_total is None else aux_total + value
    return aux_total, clean
```

### 11.3 修改模型构建

把原来的：

```python
model = build_model(args.model_name).to(device)
```

改成：

```python
use_cga = None if args.use_cga < 0 else bool(args.use_cga)
model = build_model(
    args.model_name,
    backbone_name=(args.backbone_name or None),
    use_cga=use_cga,
    lambda_cga=args.lambda_cga,
    aux_hidden_channels=args.aux_hidden_channels,
    warm_epoch=args.warm_epoch,
    legacy=args.legacy_model_factory,
).to(device)
```

### 11.4 修改 forward + loss 计算

把原来的：

```python
output = model(img, warm_flag=(epoch <= args.warm_epoch), return_dict=True)
loss_out = criterion(output, mask, epoch=epoch)
loss = _loss_value(loss_out)
loss.backward()
```

改成：

```python
forward_kwargs = {
    "warm_flag": (epoch <= args.warm_epoch),
    "return_dict": True,
}

# Only CGAWrapper needs mask/epoch during training.
if getattr(model, "use_cga", False):
    forward_kwargs.update({"mask": mask, "epoch": epoch})

output = model(img, **forward_kwargs)
loss_out = _as_loss_dict(criterion(output, mask, epoch=epoch))
loss = _loss_value(loss_out)

aux_total, aux_losses = _output_aux_loss(output)
if aux_total is not None:
    loss = loss + aux_total
    for key, value in aux_losses.items():
        loss_out[key] = value
    loss_out["total"] = loss

loss.backward()
```

### 11.5 修改日志字段

原来的日志里只有 `model_name`，建议补上 backbone 和 CGA 开关：

```python
mean_stats.update({
    "epoch": epoch,
    "dataset": args.dataset_name,
    "model": args.model_name,
    "backbone": args.backbone_name or "legacy",
    "use_cga": bool(getattr(model, "use_cga", False)),
    "lambda_cga": float(args.lambda_cga),
    "seed": args.seed,
})
```

### 11.6 checkpoint 里也写入 backbone / CGA 信息

把 checkpoint dict 改成：

```python
torch.save({
    "epoch": epoch,
    "model_name": args.model_name,
    "backbone_name": args.backbone_name,
    "use_cga": bool(getattr(model, "use_cga", False)),
    "lambda_cga": float(args.lambda_cga),
    "dataset": args.dataset_name,
    "seed": args.seed,
    "state_dict": model.state_dict(),
    "optimizer": optim.state_dict(),
}, out_dir / f"{args.model_name}_{epoch}.pth.tar")
```

---

## 12. 修改 `loss.py`

原则：`loss.py` 不应该强绑定 `MSHNetCGA`。CGA loss 由 `CGAWrapper` 放进 `output["losses"]`，训练循环统一加到 total loss 上。

所以 `build_loss()` 应该只关心主 segmentation loss，例如 SoftIoU / Dice / BCE / OHEM：

```python
# loss.py 中建议增加

def canonical_loss_name(model_name: str) -> str:
    name = str(model_name).lower().replace("-", "_").replace("+", "_")
    for suffix in ("_cga", "cga_"):
        name = name.replace(suffix, "_")
    name = name.replace("cga", "")
    if "mshnet" in name or "ohem" in name:
        return "mshnet"
    if "dnanet" in name:
        return "dnanet"
    if "alcnet" in name:
        return "alcnet"
    if "acm" in name:
        return "acm"
    if "isnet" in name:
        return "isnet"
    return "generic"
```

然后在 `build_loss()` 里：

```python
loss_name = canonical_loss_name(model_name)

# 初期最稳：所有 backbone 先用同一个主 loss，避免 loss 差异污染 CGA ablation。
return ExistingMainSegmentationLoss(ohem_ratio=ohem_ratio, warm_epoch=warm_epoch)
```

实验论文里要公平：

```text
DNANet baseline      = DNANet + same main loss
DNANet + CGA         = DNANet + same main loss + CGA regularization
ALCNet baseline      = ALCNet + same main loss
ALCNet + CGA         = ALCNet + same main loss + CGA regularization
MSHNet baseline      = MSHNet + same main loss
MSHNet + CGA         = MSHNet + same main loss + CGA regularization
```

---

## 13. 修改 `test.py` / `evaluate.py`

测试入口必须和训练入口一致地支持：

```text
--backbone_name
--use_cga
--lambda_cga
--aux_hidden_channels
--legacy_model_factory
```

但 eval 阶段不能传 `mask` 给模型：

```python
from model.common_output import normalize_detector_output

model.eval()
with torch.no_grad():
    raw = model(img, return_dict=True)
    output = normalize_detector_output(raw)
    logits = output["logits"]
    pred = torch.sigmoid(logits)
```

checkpoint load 时也要支持新字段：

```python
ckpt = torch.load(args.pth_path, map_location=device)
model_name = args.model_name or ckpt.get("model_name", "")
backbone_name = args.backbone_name or ckpt.get("backbone_name", "")
use_cga = args.use_cga
if use_cga < 0 and "use_cga" in ckpt:
    use_cga = int(bool(ckpt["use_cga"]))

model = build_model(
    model_name,
    backbone_name=(backbone_name or None),
    use_cga=(None if use_cga < 0 else bool(use_cga)),
    lambda_cga=args.lambda_cga,
    aux_hidden_channels=args.aux_hidden_channels,
).to(device)
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
```

---

## 14. 增加模型工厂测试

新增：

```text
tests/test_model_factory_multibackbone.py
```

内容：

```python
# tests/test_model_factory_multibackbone.py
import pytest
import torch

from net import build_model
from model.common_output import normalize_detector_output


def test_build_mshnet_adapter_baseline_forward():
    model = build_model("mshnet", backbone_name="mshnet", use_cga=False)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        out = normalize_detector_output(model(x, return_dict=True))
    assert "logits" in out
    assert out["logits"].shape[0] == 2


def test_build_mshnet_adapter_cga_forward_train():
    model = build_model("mshnet_cga", backbone_name="mshnet", use_cga=True, lambda_cga=0.1)
    model.train()
    x = torch.randn(2, 1, 256, 256)
    y = torch.zeros(2, 1, 256, 256)
    y[:, :, 100:105, 120:125] = 1.0
    out = model(x, mask=y, epoch=10, return_dict=True)
    assert "logits" in out
    assert "losses" in out
    assert "loss_cga" in out["losses"]
    assert out["losses"]["loss_cga"].ndim == 0


def test_missing_external_backbone_has_helpful_error():
    with pytest.raises(ImportError, match="DNANet implementation is missing"):
        build_model("dnanet", backbone_name="dnanet", use_cga=False)
```

执行：

```bash
python -m pytest tests/test_model_factory_multibackbone.py -q
```

---

## 15. 推荐执行顺序

### Step 0：新建分支

```bash
git checkout -b feature/cga-regularizer-multibackbone
```

### Step 1：先只接 MSHNet adapter

新增这些文件：

```text
model/common_output.py
model/backbones/__init__.py
model/backbones/base.py
model/backbones/mshnet_adapter.py
model/registry.py
model/cga_regularizer.py
model/cga_wrapper.py
```

然后改：

```text
net.py
train.py
```

先不要碰 DNANet/ALCNet/ACM/ISNet-style。

### Step 2：跑 MSHNet 新旧路径 sanity check

```bash
# 旧路径：保持原始 MSHNetCGA 可复现
python train.py \
  --legacy_model_factory \
  --model_name MSHNetCGA \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2

# 新路径：MSHNet baseline
python train.py \
  --model_name MSHNet \
  --backbone_name mshnet \
  --use_cga 0 \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2

# 新路径：MSHNet + generic CGA wrapper
python train.py \
  --model_name MSHNet_CGA \
  --backbone_name mshnet \
  --use_cga 1 \
  --lambda_cga 0.1 \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2
```

### Step 3：把真实 CGA 从 `model/cga_aux.py` 迁移进 `CGARegularizer`

正式实验前必须做这一步。不要用 fallback regularizer 当论文方法。

目标签名：

```python
loss_cga = self.real_cga(
    features=output["features"],
    logits=output["logits"],
    mask=mask,
)
```

如果 `cga_aux.py` 现在写死了 MSHNet 的 channel 数或 feature 层名，需要改成：

```python
# 不推荐
CGA(x2_from_mshnet, x3_from_mshnet, mask)

# 推荐
CGA(features: list[Tensor], logits: Tensor, mask: Tensor)
```

### Step 4：接入第二个 backbone

建议优先顺序：

```text
1. DNANet
2. ALCNet 或 ACM
3. ISNet-style
```

把代码复制到：

```text
model/backbones/dnanet_impl.py
```

然后修正 `DNANetAdapter.__init__()` 里的 constructor。

跑：

```bash
python train.py \
  --model_name DNANet \
  --backbone_name dnanet \
  --use_cga 0 \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2

python train.py \
  --model_name DNANet_CGA \
  --backbone_name dnanet \
  --use_cga 1 \
  --lambda_cga 0.1 \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2
```

### Step 5：接入第三个 backbone

建议选择 `ALCNet` 或 `ACM`，这样可以证明 CGA 不只对 MSHNet/DNANet 这种融合结构有效。

```bash
python train.py \
  --model_name ALCNet \
  --backbone_name alcnet \
  --use_cga 0 \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2

python train.py \
  --model_name ALCNet_CGA \
  --backbone_name alcnet \
  --use_cga 1 \
  --lambda_cga 0.1 \
  --dataset_dir /path/to/datasets \
  --dataset_name NUDT-SIRST \
  --epochs 1 \
  --batch_size 2
```

---

## 16. AAAI 实验矩阵建议

主文最少放 3 个 host detector/backbone：

| Backbone / Detector | Baseline | + CGA | 论文作用 |
|---|---:|---:|---|
| MSHNet | yes | yes | 当前强 baseline / continuity |
| DNANet | yes | yes | dense/nested feature-fusion 类验证 |
| ALCNet 或 ACM | yes | yes | local contrast / attention 类验证 |
| ISNet-style | optional | optional | shape/edge-aware 类补充验证 |

运行命名建议：

```text
MSHNet
MSHNet_CGA
DNANet
DNANet_CGA
ALCNet
ALCNet_CGA
ACM
ACM_CGA
ISNet
ISNet_CGA
```

注意：每一对 baseline/+CGA 必须保持以下因素相同：

```text
same dataset split
same seed
same epoch
same patch size
same optimizer
same lr schedule
same main segmentation loss
same evaluation script
```

CGA 只能是额外变量。

---

## 17. 当前 P1 / HC-Val 注意事项

你之前的 P1 preflight 结论是合理的：缺少冻结的 `hcval_NUDT-SIRST.txt` 时，不应该启动 seed42 paper-evidence training。

因此执行顺序应该是：

```text
1. 先完成代码层面的 multi-backbone smoke test
2. 找回/审计冻结的 hcval_NUDT-SIRST.txt
3. P1 dataset preflight 通过后
4. 再启动正式 seed42/多 seed 论文证据训练
```

不要把 test list 直接复制成 HC-Val；不要基于当前模型错误样本后验挑 HC-Val；不要用 zero-byte zip 替换正式数据。

---

## 18. 最小可交付 PR 切分

### PR-1：架构解耦，不接外部 backbone

包含：

```text
model/common_output.py
model/backbones/base.py
model/backbones/mshnet_adapter.py
model/registry.py
model/cga_regularizer.py
model/cga_wrapper.py
net.py 修改
train.py 修改
tests/test_model_factory_multibackbone.py
```

验收：

```bash
python -m pytest tests/test_model_factory_multibackbone.py -q
python train.py --model_name MSHNet --backbone_name mshnet --use_cga 0 --epochs 1 --batch_size 2
python train.py --model_name MSHNet_CGA --backbone_name mshnet --use_cga 1 --epochs 1 --batch_size 2
```

### PR-2：迁移真实 CGA

包含：

```text
把 model/cga_aux.py 里的真实 CGA 逻辑封装到 CGARegularizer
删除 fallback 在正式路径中的使用
增加 cga ablation 参数：lambda_cga / warm_epoch / aux_hidden_channels
```

验收：

```text
MSHNet legacy CGA 与 MSHNet adapter + CGA 的 trend 一致
loss_cga 能正常记录到 train_log.jsonl
测试阶段不计算 loss_cga
```

### PR-3：接入 DNANet

包含：

```text
model/backbones/dnanet_impl.py
model/backbones/dnanet_adapter.py constructor 修正
DNANet / DNANet_CGA smoke test
```

验收：

```text
DNANet baseline 可训练/可测
DNANet_CGA 可训练/可测
checkpoint 记录 backbone_name=dnanet, use_cga=true/false
```

### PR-4：接入 ALCNet/ACM/ISNet-style

包含：

```text
model/backbones/alcnet_impl.py 或 acm_impl.py 或 isnet_impl.py
对应 adapter constructor 修正
baseline/+CGA paired scripts
```

验收：

```text
至少 3 个 backbone 的 paired result 表可生成
```

---

## 19. 最后建议

论文代码最好不要让命令长这样：

```bash
python train.py --model_name MSHNetCGA
```

而是让实验命令长这样：

```bash
python train.py --model_name MSHNet --backbone_name mshnet --use_cga 0
python train.py --model_name MSHNet_CGA --backbone_name mshnet --use_cga 1

python train.py --model_name DNANet --backbone_name dnanet --use_cga 0
python train.py --model_name DNANet_CGA --backbone_name dnanet --use_cga 1

python train.py --model_name ALCNet --backbone_name alcnet --use_cga 0
python train.py --model_name ALCNet_CGA --backbone_name alcnet --use_cga 1
```

这样 reviewer 一眼能看出：

```text
CGA 不是 MSHNet 的一个私有结构补丁，而是可迁移的 ISTD regularization。
```
