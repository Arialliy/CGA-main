# CGA-main 多 Backbone 改造修正版：Fail-Closed Paper-Evidence 方案

> 目的：把 `CGA-main` 从 `MSHNetCGA` 单宿主实现，改成可审计的多 backbone CGA regularization 实验框架；同时避免两个致命风险：
>
> 1. fallback regularizer 混入正式实验；
> 2. adapter/normalizer 自动猜 tensor，导致“能跑但结果不可信”。

---

## 0. 结论

你的修正意见是对的。上一版方案里最需要改的是两处：

```text
必须删除 paper-evidence 路径里的 fallback regularizer。
必须禁止 normalize_detector_output() silent fallback 取第一个 4D tensor。
```

正式论文实验只能使用真实 CGA：

```text
center / boundary / scale / peak
```

不能退化成：

```text
boundary-weighted BCE + background suppression
```

否则方法定义变了，实验结果不能支撑论文里的 CGA claim。

---

## 1. 当前仓库状态判断

当前 `model/` 里已经有真实 CGA 组件：

```text
model/CGA_MSHNet.py
model/MSHNet.py
model/cga_aux.py
```

`cga_aux.py` 中的 `CGAAuxHead` 已经定义了四个 CGA auxiliary heads：

```text
cga_center_logit
cga_boundary_logit
cga_scale_logit
cga_peak_logit
```

`loss.py` 里也已经有对应的四项 CGA loss：

```text
loss_center
loss_boundary
loss_scale
loss_peak
```

所以多 backbone 改造不应该新造一个 fallback regularizer，而应该把现有的四头 CGA 逻辑从 `MSHNetCGA` 中抽出来，变成所有 adapter 共用的显式 wrapper。

---

## 2. Paper-evidence 强约束

新增全局约束：

```text
Fallback regularizer is forbidden for paper evidence.
Any result generated with fallback must be labeled smoke-test only.
```

建议写入三个地方：

```text
1. README / experiment protocol
2. train.py runtime guard
3. checkpoint / train_log.jsonl metadata
```

### 2.1 禁止项

正式结果禁止：

```text
--allow_fallback_regularizer
--evidence_mode smoke
--regularizer_impl fallback
silent tensor inference
post-hoc HC-Val construction
```

### 2.2 允许项

smoke test 可以使用 mock/fallback，但只能验证：

```text
forward shape
loss scalar existence
DataLoader/optimizer/checkpoint plumbing
```

不能用于：

```text
paper table
ablation
SOTA comparison
AAAI main evidence
```

---

## 3. 不要自动猜输出：adapter 必须显式声明

上一版 `normalize_detector_output()` 如果自动找第一个 4D tensor，必须改掉。

原因：不同 detector 可能同时输出：

```text
encoder feature
decoder feature
side logits
aux logits
final logits
multi-scale logits
```

取错 tensor 也可能 shape 正确、训练能跑，但结果是假的。

正式 adapter 必须显式声明：

```text
logits source
feature source
feature stride
feature channels
feature resolution
```

每个 adapter 的 forward 必须返回标准结构：

```python
{
    "logits": final_logits,
    "features": [selected_decoder_feature],
    "feature_meta": [
        {
            "source": "decoder_stage_x",
            "stride": 4,
            "channels": 16,
            "resolution": [Hf, Wf],
        }
    ],
    "adapter_meta": {
        "backbone": "mshnet",
        "logits_source": "base_logits",
        "feature_source": "decoder_features.x_d0",
    }
}
```

没有这些字段，paper mode 直接报错。

---

## 4. 推荐文件结构

新增或修改：

```text
model/
  output_contract.py              # 新增：显式输出协议，只校验，不猜 tensor
  cga_wrapper.py                  # 新增：任意 backbone + CGA aux head
  registry.py                     # 新增：backbone registry
  backbones/
    __init__.py
    mshnet_adapter.py             # 新增：先做
    dnanet_adapter.py             # 新增：第二阶段
    alcnet_adapter.py             # 新增：第三阶段，或用 ACM 替代
    acm_adapter.py                # optional
    isnet_adapter.py              # optional
  cga_aux.py                      # 保留：center/boundary/scale/peak heads
  CGA_MSHNet.py                   # 保留：legacy 对照，不作为未来主接口
net.py                            # 修改：build_model 支持 backbone_name/use_cga
loss.py                           # 修改：MSHNetCGALoss 泛化命名为 CGALoss，保留 alias
train.py                          # 修改：拆分 warm 参数，写 paper/smoke guard
```

---

## 5. 新增 `model/output_contract.py`

重点：只接受显式 adapter output；不做任何 silent fallback。

```python
# model/output_contract.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureSpec:
    source: str
    stride: int
    channels: int
    resolution: tuple[int, int]


def _as_hw(x: Any) -> tuple[int, int]:
    if isinstance(x, torch.Tensor):
        return int(x.shape[-2]), int(x.shape[-1])
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return int(x[0]), int(x[1])
    raise TypeError(f"Invalid resolution format: {type(x)!r}")


def validate_detector_output(
    output: dict[str, Any],
    *,
    backbone_name: str,
    require_feature: bool,
) -> dict[str, Any]:
    """Validate explicit detector output.

    Paper-evidence rule:
      - no automatic tensor guessing
      - no first-4D fallback
      - adapter must state logits/features/meta explicitly
    """
    if not isinstance(output, dict):
        raise TypeError(
            f"{backbone_name} adapter must return a dict, got {type(output)!r}. "
            "Raw tuple/list/tensor outputs are not allowed in paper mode."
        )

    if "logits" not in output:
        raise KeyError(
            f"{backbone_name} adapter output lacks required key 'logits'. "
            "Do not infer logits from arbitrary tensors."
        )

    logits = output["logits"]
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"{backbone_name} logits must be Tensor[B,1,H,W], got "
            f"{getattr(logits, 'shape', None)}"
        )

    if not require_feature:
        return output

    if "features" not in output or "feature_meta" not in output:
        raise KeyError(
            f"{backbone_name}+CGA requires explicit 'features' and 'feature_meta'."
        )

    features = output["features"]
    metas = output["feature_meta"]
    if not isinstance(features, (list, tuple)) or len(features) != 1:
        raise ValueError(
            f"{backbone_name}+CGA expects exactly one selected feature for controlled protocol."
        )
    if not isinstance(metas, (list, tuple)) or len(metas) != 1:
        raise ValueError(
            f"{backbone_name}+CGA expects exactly one feature_meta entry."
        )

    feat = features[0]
    meta = metas[0]
    if not torch.is_tensor(feat) or feat.ndim != 4:
        raise ValueError(
            f"{backbone_name} selected CGA feature must be Tensor[B,C,H,W], got "
            f"{getattr(feat, 'shape', None)}"
        )

    required_meta = {"source", "stride", "channels", "resolution"}
    missing = required_meta - set(meta.keys())
    if missing:
        raise KeyError(f"{backbone_name} feature_meta missing fields: {sorted(missing)}")

    if int(meta["channels"]) != int(feat.shape[1]):
        raise ValueError(
            f"{backbone_name} feature_meta channels={meta['channels']} "
            f"but feature.shape[1]={feat.shape[1]}"
        )

    if _as_hw(meta["resolution"]) != (int(feat.shape[-2]), int(feat.shape[-1])):
        raise ValueError(
            f"{backbone_name} feature_meta resolution={meta['resolution']} "
            f"but feature.shape[-2:]={tuple(feat.shape[-2:])}"
        )

    return output
```

---

## 6. 新增 `model/backbones/mshnet_adapter.py`

MSHNet 作为 Gate 1。adapter 要显式选择 `x_d0` 或等价 decoder feature。

```python
# model/backbones/mshnet_adapter.py
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.MSHNet import MSHNet
from model.output_contract import validate_detector_output


class MSHNetAdapter(nn.Module):
    BACKBONE_NAME = "mshnet"
    LOGITS_SOURCE = "base_logits/base_logit"
    FEATURE_SOURCE = "decoder_features.x_d0 or decoder_feature"
    FEATURE_STRIDE = 1
    FEATURE_CHANNELS = 16

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.net = MSHNet(input_channels)

    def forward(self, x: torch.Tensor, *, mshnet_warm_flag: bool = True) -> dict[str, Any]:
        try:
            raw = self.net(x, warm_flag=mshnet_warm_flag, return_dict=True)
        except TypeError:
            raw = self.net(x, warm_flag=mshnet_warm_flag, return_feature=True)

        if not isinstance(raw, dict):
            raise TypeError(
                "MSHNetAdapter requires MSHNet return_dict=True output. "
                "Do not silently parse tuple/list outputs in paper mode."
            )

        logits = raw.get("base_logits", raw.get("base_logit"))
        if logits is None:
            raise KeyError("MSHNet raw output lacks base_logits/base_logit")

        feat = raw.get("decoder_feature")
        if feat is None:
            decoder_features = raw.get("decoder_features", {})
            if not isinstance(decoder_features, dict) or "x_d0" not in decoder_features:
                raise KeyError("MSHNet raw output lacks decoder_feature or decoder_features['x_d0']")
            feat = decoder_features["x_d0"]

        output = {
            "logits": logits,
            "features": [feat],
            "feature_meta": [
                {
                    "source": self.FEATURE_SOURCE,
                    "stride": self.FEATURE_STRIDE,
                    "channels": int(feat.shape[1]),
                    "resolution": [int(feat.shape[-2]), int(feat.shape[-1])],
                }
            ],
            "adapter_meta": {
                "backbone": self.BACKBONE_NAME,
                "logits_source": self.LOGITS_SOURCE,
                "feature_source": self.FEATURE_SOURCE,
            },
            "masks": raw.get("masks", []),
            "scale_logits": raw.get("scale_logits", raw.get("scale_logits_up", [])),
        }
        return validate_detector_output(output, backbone_name=self.BACKBONE_NAME, require_feature=True)
```

备注：`FEATURE_STRIDE = 1` 只是示例。正式写入前要用真实输入/feature shape 审计确认。如果 `x_d0` 不是原图尺度，应改为实际 stride。

---

## 7. 新增 `model/cga_wrapper.py`

`CGAWrapper` 只接真实四头 CGA，不包含 fallback。

```python
# model/cga_wrapper.py
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model.cga_aux import CGAAuxHead
from model.output_contract import validate_detector_output


class CGAWrapper(nn.Module):
    """Backbone + real center/boundary/scale/peak CGA auxiliary heads.

    This wrapper is allowed for paper evidence.
    It must never call a fallback regularizer.
    """

    REGULARIZER_IMPL = "center_boundary_scale_peak"

    def __init__(self, backbone: nn.Module, *, backbone_name: str, feature_channels: int, aux_hidden_channels: int = 32) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.cga_aux_head = CGAAuxHead(in_channels=int(feature_channels), hidden_channels=int(aux_hidden_channels))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        output = self.backbone(x, **kwargs)
        output = validate_detector_output(output, backbone_name=self.backbone_name, require_feature=True)

        feat = output["features"][0]
        aux = self.cga_aux_head(feat)

        output = dict(output)
        output["aux_outputs"] = aux
        output.update(aux)
        output.setdefault("regularizer_meta", {})
        output["regularizer_meta"].update(
            {
                "use_cga": True,
                "regularizer_impl": self.REGULARIZER_IMPL,
                "paper_evidence_allowed": True,
            }
        )
        return output
```

---

## 8. 修改 `loss.py`：把 MSHNetCGALoss 泛化为 CGALoss

当前逻辑可以复用，但命名和参数要拆开。

### 8.1 参数命名

把：

```python
warm_epoch
start_epoch
ramp_epochs
```

改为：

```python
mshnet_warm_epoch
cga_start_epoch
cga_ramp_epochs
```

### 8.2 泛化类名

建议：

```python
class CGALoss(nn.Module):
    ...

MSHNetCGALoss = CGALoss  # backward-compatible alias
```

### 8.3 严格要求四个 CGA logit

当前 `_bce(logit=None)` 会返回 0，这对 ablation 可以，但对正式 CGA paper evidence 不够严格。

建议新增 `strict_cga_heads=True`：

```python
REQUIRED_CGA_LOGITS = (
    "cga_center_logit",
    "cga_boundary_logit",
    "cga_scale_logit",
    "cga_peak_logit",
)


def _require_cga_logits(output: dict) -> None:
    missing = [k for k in REQUIRED_CGA_LOGITS if k not in output or output[k] is None]
    if missing:
        raise KeyError(
            "Paper-mode CGA requires all four auxiliary logits: "
            f"{REQUIRED_CGA_LOGITS}. Missing: {missing}"
        )
```

在 `CGALoss.forward()` 中：

```python
if self.strict_cga_heads:
    _require_cga_logits(output)
```

### 8.4 `build_loss()` 改成显式开关

不要只靠 `"cga" in model_name` 判断。改成：

```python
def build_loss(
    name: str = "MSHNet",
    *,
    use_cga: bool = False,
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
) -> nn.Module:
    if use_cga:
        cfg = CGALossConfig(
            lambda_center=lambda_center,
            lambda_boundary=lambda_boundary,
            lambda_scale=lambda_scale,
            lambda_peak=lambda_peak,
            start_epoch=cga_start_epoch,
            ramp_epochs=cga_ramp_epochs,
            ohem_ratio=ohem_ratio,
            lambda_iou=lambda_iou,
            warm_epoch=mshnet_warm_epoch,
        )
        return CGALoss(cfg, strict_cga_heads=strict_cga_heads)

    return MSHNetOHEMLoss(
        ohem_ratio=ohem_ratio,
        lambda_iou=lambda_iou,
        warm_epoch=mshnet_warm_epoch,
    )
```

---

## 9. 修改 `net.py`

当前 `net.py` 只支持 `MSHNetCGA` 和 `MSHNet`。需要改成 registry。

```python
# net.py
from __future__ import annotations

import torch.nn as nn

from model.backbones.mshnet_adapter import MSHNetAdapter
from model.cga_wrapper import CGAWrapper


BACKBONE_BUILDERS = {
    "mshnet": MSHNetAdapter,
    # "dnanet": DNANetAdapter,
    # "alcnet": ALCNetAdapter,
    # "acm": ACMAdapter,
    # "isnet": ISNetAdapter,
}


def build_model(
    model_name: str | None = None,
    *,
    backbone_name: str = "mshnet",
    input_channels: int = 1,
    use_cga: bool = False,
    aux_hidden_channels: int = 32,
    evidence_mode: str = "paper",
    **kwargs,
) -> nn.Module:
    if evidence_mode not in {"paper", "smoke"}:
        raise ValueError(f"Unknown evidence_mode={evidence_mode!r}")

    if evidence_mode == "paper" and kwargs.get("allow_fallback_regularizer", False):
        raise RuntimeError(
            "Fallback regularizer is forbidden for paper evidence. "
            "Run smoke tests under evidence_mode='smoke' only."
        )

    # Backward compatibility.
    if model_name is not None:
        name = str(model_name).lower()
        if name in {"mshnetcga", "cga", "cga-v2", "mshnet_cga"}:
            backbone_name = "mshnet"
            use_cga = True
        elif name in {"mshnet", "mshnetohem", "ohem"}:
            backbone_name = "mshnet"
            use_cga = False

    backbone_name = backbone_name.lower()
    if backbone_name not in BACKBONE_BUILDERS:
        raise ValueError(
            f"Unknown backbone_name={backbone_name!r}. "
            f"Available: {sorted(BACKBONE_BUILDERS)}"
        )

    backbone = BACKBONE_BUILDERS[backbone_name](input_channels=input_channels)

    if not use_cga:
        return backbone

    # For MSHNet adapter, feature_channels is known after first audit.
    # Better: expose backbone.CGA_FEATURE_CHANNELS after adapter validation.
    feature_channels = getattr(backbone, "FEATURE_CHANNELS", None)
    if feature_channels is None:
        raise AttributeError(f"{backbone_name} adapter must define FEATURE_CHANNELS for CGAWrapper")

    return CGAWrapper(
        backbone,
        backbone_name=backbone_name,
        feature_channels=int(feature_channels),
        aux_hidden_channels=aux_hidden_channels,
    )
```

---

## 10. 修改 `train.py`

### 10.1 参数拆分

新增：

```python
p.add_argument("--backbone_name", default="mshnet", choices=["mshnet", "dnanet", "alcnet", "acm", "isnet"])
p.add_argument("--use_cga", action="store_true")
p.add_argument("--evidence_mode", default="paper", choices=["paper", "smoke"])
p.add_argument("--protocol", default="controlled", choices=["controlled", "official"])

p.add_argument("--mshnet_warm_epoch", type=int, default=5)
p.add_argument("--cga_start_epoch", type=int, default=1)
p.add_argument("--cga_ramp_epochs", type=int, default=40)

p.add_argument("--lambda_center", type=float, default=0.05)
p.add_argument("--lambda_boundary", type=float, default=0.03)
p.add_argument("--lambda_scale", type=float, default=0.02)
p.add_argument("--lambda_peak", type=float, default=0.03)

p.add_argument("--allow_fallback_regularizer", action="store_true")
```

把旧的：

```python
--warm_epoch
```

废弃或只作为兼容 alias，不再用于新实验日志。

### 10.2 Runtime guard

```python
if args.evidence_mode == "paper" and args.allow_fallback_regularizer:
    raise RuntimeError(
        "Fallback regularizer is forbidden for paper evidence. "
        "Use --evidence_mode smoke for smoke-only plumbing tests."
    )

if args.evidence_mode == "paper" and args.protocol != "controlled":
    print("[WARN] official/literature protocol is contextual only; do not use it as main CGA claim.")
```

### 10.3 Build model/loss

替换：

```python
model = build_model(args.model_name).to(device)
criterion = build_loss(args.model_name, ohem_ratio=args.ohem_ratio, warm_epoch=args.warm_epoch).to(device)
```

为：

```python
model = build_model(
    model_name=args.model_name,
    backbone_name=args.backbone_name,
    use_cga=args.use_cga,
    evidence_mode=args.evidence_mode,
    input_channels=1,
    aux_hidden_channels=32,
    allow_fallback_regularizer=args.allow_fallback_regularizer,
).to(device)

criterion = build_loss(
    args.model_name or args.backbone_name,
    use_cga=args.use_cga,
    ohem_ratio=args.ohem_ratio,
    mshnet_warm_epoch=args.mshnet_warm_epoch,
    cga_start_epoch=args.cga_start_epoch,
    cga_ramp_epochs=args.cga_ramp_epochs,
    lambda_center=args.lambda_center,
    lambda_boundary=args.lambda_boundary,
    lambda_scale=args.lambda_scale,
    lambda_peak=args.lambda_peak,
    strict_cga_heads=(args.evidence_mode == "paper" and args.use_cga),
).to(device)
```

### 10.4 Forward

替换：

```python
output = model(img, warm_flag=(epoch <= args.warm_epoch), return_dict=True)
```

为：

```python
forward_kwargs = {}
if args.backbone_name == "mshnet":
    forward_kwargs["mshnet_warm_flag"] = epoch <= args.mshnet_warm_epoch

output = model(img, **forward_kwargs)
```

### 10.5 日志必须写入证据属性

```python
mean_stats.update(
    {
        "epoch": epoch,
        "dataset": args.dataset_name,
        "backbone": args.backbone_name,
        "use_cga": bool(args.use_cga),
        "regularizer_impl": "center_boundary_scale_peak" if args.use_cga else "none",
        "evidence_mode": args.evidence_mode,
        "paper_evidence_allowed": bool(args.evidence_mode == "paper" and not args.allow_fallback_regularizer),
        "protocol": args.protocol,
        "seed": args.seed,
        "mshnet_warm_epoch": args.mshnet_warm_epoch,
        "cga_start_epoch": args.cga_start_epoch,
        "cga_ramp_epochs": args.cga_ramp_epochs,
    }
)
```

checkpoint 也写同样 metadata。

---

## 11. Adapter 示例：DNANet / ALCNet / ACM

不要一开始实现 5 个。最低目标：

```text
1. MSHNet
2. DNANet
3. ALCNet 或 ACM
```

ISNet-style 放 optional，因为它本身 shape/edge-aware，和 CGA novelty 容易发生解释冲突。

每个 adapter 都必须填：

```python
BACKBONE_NAME = "dnanet"
LOGITS_SOURCE = "..."
FEATURE_SOURCE = "..."
FEATURE_STRIDE = ...
FEATURE_CHANNELS = ...
```

模板：

```python
# model/backbones/dnanet_adapter.py
class DNANetAdapter(nn.Module):
    BACKBONE_NAME = "dnanet"
    LOGITS_SOURCE = "final segmentation head"
    FEATURE_SOURCE = "selected decoder fusion feature; must be audited"
    FEATURE_STRIDE = -1          # TODO: replace after audit
    FEATURE_CHANNELS = -1        # TODO: replace after audit

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        # self.net = DNANet(...)
        raise NotImplementedError("Import DNANet implementation, then fill explicit sources.")

    def forward(self, x, **kwargs):
        # raw = self.net(x, return_features=True)
        # final_logits = raw[EXPLICIT_KEY]
        # selected_feature = raw[EXPLICIT_KEY]
        # output = {...}
        # return validate_detector_output(output, backbone_name=self.BACKBONE_NAME, require_feature=True)
        raise NotImplementedError
```

原则：宁可 adapter 不能跑，也不能取错 feature 后静默跑通。

---

## 12. Protocol A / B

### Protocol A：controlled paired protocol

这是主 claim 使用的协议。

每个 backbone 内部做成 paired experiment：

```text
same dataset split
same seed
same main loss
same batch size
same crop/patch policy
same optimizer
same LR schedule
same training epochs
only difference: use_cga = false / true
```

结论只能写：

```text
CGA improves multiple host detectors under controlled paired protocols.
```

不能写：

```text
CGA universally improves all IRSTD models.
```

### Protocol B：official/literature protocol

只用于 contextual comparison。

允许每个 backbone 按原论文/公开实现 recipe 训练，但不能用于 CGA 主变量归因。

论文中建议写：

```text
Protocol B contextualizes our controlled gains against reported or reproduced detector-level baselines; it is not used to claim that CGA alone achieves SOTA under heterogeneous recipes.
```

---

## 13. 推荐 gate 顺序

不要直接全做。

```text
Gate 0: NUDT P1 + HC-Val preflight 通过
Gate 1: MSHNet legacy CGA vs MSHNet adapter CGA trend 一致
Gate 2: 真实 CGARegularizer/CGAWrapper 完成，fallback 禁止进入 paper evidence
Gate 3: DNANet baseline / DNANet + CGA seed42 跑通
Gate 4: ALCNet 或 ACM baseline / +CGA seed42 跑通
Gate 5: 三个 backbone 都有正向 seed42，再考虑 multi-seed
```

### Gate 1 通过标准

```text
MSHNetCGA legacy result and MSHNetAdapter+CGA result should have the same trend.
允许有小幅数值差异。
不允许方向相反且无法解释。
```

### Gate 2 通过标准

```text
paper mode 下：
- fallback flag 会直接报错
- 缺任意 cga_center/boundary/scale/peak logit 会报错
- 缺 feature_meta 会报错
- 缺 logits 会报错
- tuple/list/tensor raw output 会报错
```

---

## 14. 执行命令建议

### 14.1 Gate 1：MSHNet baseline

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --seed 42 \
  --epochs 400 \
  --batch_size 8 \
  --patch_size 256 \
  --mshnet_warm_epoch 5 \
  --protocol controlled \
  --evidence_mode paper \
  --output_dir results/protocol_A
```

### 14.2 Gate 1：MSHNet + CGA

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name mshnet \
  --use_cga \
  --seed 42 \
  --epochs 400 \
  --batch_size 8 \
  --patch_size 256 \
  --mshnet_warm_epoch 5 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 40 \
  --lambda_center 0.05 \
  --lambda_boundary 0.03 \
  --lambda_scale 0.02 \
  --lambda_peak 0.03 \
  --protocol controlled \
  --evidence_mode paper \
  --output_dir results/protocol_A
```

### 14.3 Gate 3：DNANet paired

```bash
# baseline
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name dnanet \
  --seed 42 \
  --protocol controlled \
  --evidence_mode paper \
  --output_dir results/protocol_A

# +CGA
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name dnanet \
  --use_cga \
  --seed 42 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 40 \
  --protocol controlled \
  --evidence_mode paper \
  --output_dir results/protocol_A
```

### 14.4 Gate 4：ALCNet 或 ACM paired

优先选一个，不要两个都同时开。

```bash
# 例如 ALCNet
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name alcnet \
  --seed 42 \
  --protocol controlled \
  --evidence_mode paper \
  --output_dir results/protocol_A

CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --backbone_name alcnet \
  --use_cga \
  --seed 42 \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 40 \
  --protocol controlled \
  --evidence_mode paper \
  --output_dir results/protocol_A
```

---

## 15. 必加测试

新增：

```text
tests/test_cga_failclosed_paper_mode.py
tests/test_adapter_explicit_contract.py
tests/test_multibackbone_factory.py
```

### 15.1 fallback 禁止进入 paper evidence

```python
def test_paper_mode_forbids_fallback_regularizer():
    import pytest
    from net import build_model

    with pytest.raises(RuntimeError, match="Fallback regularizer is forbidden"):
        build_model(
            backbone_name="mshnet",
            use_cga=True,
            evidence_mode="paper",
            allow_fallback_regularizer=True,
        )
```

### 15.2 禁止 silent 4D tensor fallback

```python
def test_validate_detector_output_does_not_guess_first_4d_tensor():
    import pytest
    import torch
    from model.output_contract import validate_detector_output

    bad_output = {
        "some_feature": torch.randn(2, 16, 64, 64),
        "another_feature": torch.randn(2, 32, 32, 32),
    }

    with pytest.raises(KeyError, match="logits"):
        validate_detector_output(bad_output, backbone_name="dummy", require_feature=True)
```

### 15.3 CGA 四头缺失直接失败

```python
def test_cga_loss_requires_all_four_heads_in_paper_mode():
    import pytest
    import torch
    from loss import build_loss

    criterion = build_loss(use_cga=True, strict_cga_heads=True)
    output = {
        "logits": torch.randn(2, 1, 256, 256),
        "cga_center_logit": torch.randn(2, 1, 256, 256),
        # boundary/scale/peak intentionally missing
    }
    target = torch.zeros(2, 1, 256, 256)

    with pytest.raises(KeyError, match="cga_boundary_logit"):
        criterion(output, target, epoch=1)
```

---

## 16. 论文 claim 改法

### seed42 只有 MSHNet 正向

写窄 claim：

```text
CGA regularization for MSHNet-style IRSTD.
```

### MSHNet + DNANet 正向，ALCNet/ACM 不稳

写中等 claim：

```text
CGA shows promising transferability across MSHNet and DNANet-style IRSTD detectors under controlled paired protocols.
```

### MSHNet + DNANet + ALCNet/ACM 都正向

写宽 claim：

```text
CGA improves multiple host detectors under controlled paired protocols, suggesting that component-geometry regularization is a useful training-time prior for IRSTD.
```

仍然不要写：

```text
universally effective
plug-and-play for all IRSTD models
state-of-the-art across all backbones
```

除非有足够多 backbone、多 seed、多 dataset 支撑。

---

## 17. 最终执行建议

当前应该按这个顺序推进：

```text
1. 先修 fail-closed 机制：fallback 从 paper path 删除。
2. 写 output_contract.py：adapter 输出必须显式声明，不许自动猜。
3. 用 MSHNetAdapter 复现 legacy MSHNetCGA trend。
4. 把 loss.py 的 MSHNetCGALoss 泛化为 CGALoss，但保留 center/boundary/scale/peak 四项。
5. DNANet adapter 只在 feature source 审计清楚后再跑。
6. ALCNet/ACM 二选一，不要同时铺太大。
7. 三个 backbone seed42 都正向后，再开 multi-seed。
```

这版才能支撑 AAAI 叙事：

```text
CGA is a component-geometry regularization strategy for infrared small target detection.
```

而不是：

```text
MSHNet + 一个不清楚来源的 auxiliary loss trick。
```
