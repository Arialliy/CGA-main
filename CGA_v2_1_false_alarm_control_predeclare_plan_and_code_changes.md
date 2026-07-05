# CGA-v2.1 新方案：False-Alarm-Controlled Component-Geometry Regularization

> Canonical repo path: `/home/ly/AAAI/CGA-main`  
> Branch suggestion: `cga-v2.1-fa-control-predeclare`  
> Status: **new predeclared protocol**, not a post-hoc modification of CGA-v2.

---

## 0. 一句话结论

CGA-v2 已经不能继续作为 AAAI 正向路线。CGA-v2.1 必须作为新协议启动：

```text
CGA-v2.1 = component-geometry auxiliary supervision
           + safe-background false-alarm control
           + bounded auxiliary regularization
           + late/ramped regularization schedule
```

目标不是继续证明原来的 CGA-v2 有效，而是修复已经暴露的核心失败模式：

```text
CGA-v2 提高 Pd，但明显放大 false alarms，尤其 HC-Val 上发生 hard-case false-alarm collapse。
```

因此 v2.1 的主目标必须从：

```text
target-preserving component-geometry regularization
```

升级为：

```text
false-alarm-controlled component-geometry regularization
```

推荐安全标题：

```text
CGA-v2.1: False-Alarm-Controlled Component-Geometry Regularization
for Target-Preserving Infrared Small Target Detection
```

---

## 1. 为什么必须新开 v2.1

当前 CGA-v2 的 P2 seed42 paired gate 已失败，且 v5 implementation audit 判定为 valid negative design weakness。结论边界如下：

```text
CGA-v2:
  final_decision = P2_VALID_NEGATIVE_DESIGN_WEAKNESS
  can_run_seed43_44 = false
  can_claim_positive_cga = false
```

所以不能再做：

```text
继续 seed43/44
继续写 CGA-v2 positive claim
在同一 protocol 下调 lambda / 改 target / 改 threshold 来救火
把失败后的新参数说成原方法
```

允许做的是：

```text
新开 CGA-v2.1
预声明新方法和新 protocol
冻结超参和 gate
从零训练 OHEM vs CGA-v2.1 seed42
seed42 过 gate 后才 seed43/44
```

---

## 2. 当前失败模式诊断

P2 seed42 结果显示：

### Full test split

```text
CGA-v2:
  Pd        +0.005291
  mIoU      -0.003096
  Precision -0.002140
  FA_ppm    +2.918
```

解释：

```text
target hit rate slightly improves,
but precision and mIoU degrade,
and false alarms increase.
```

### HC-Val

```text
CGA-v2:
  Pd        +0.166667
  mIoU      -0.272208
  Precision -0.308944
  FA_ppm    +595.093
```

解释：

```text
CGA-v2 behaves like a recall/Pd booster,
not a hard-clutter robust regularizer.
```

因此 v2.1 的设计必须显式解决：

```text
false-positive components
safe-background overactivation
auxiliary loss over-strength
boundary/scale supervision causing expansion
```

---

## 3. v2.1 方法定义

### 3.1 方法名

```text
FAC-CGA / CGA-v2.1
False-Alarm-Controlled Component-Geometry Regularization
```

### 3.2 核心思想

保留 CGA 的四个 component-geometry 辅助头：

```text
center
boundary
scale
peak
```

但不再只做 positive geometry regularization。新增一个 **safe-background false-alarm control branch in the loss**：

```text
safe background = pixels outside dilated GT target region
```

在 safe background 上对 final logits 加 hard-negative penalty，抑制模型在远离真实目标的背景杂波上产生高响应。

### 3.3 v2.1 总损失

```text
L_total =
  L_base
  + ramp(t) * clamp(
      L_geom_pos + L_fa_control,
      max = aux_ratio_cap * detach(L_base)
    )
```

其中：

```text
L_base:
  existing MSHNet OHEM + SoftIoU loss

L_geom_pos:
  lower-weight center / boundary / scale / peak auxiliary loss

L_fa_control:
  hard safe-background suppression loss

aux_ratio_cap:
  prevents the auxiliary regularizer from overpowering the main detection loss
```

---

## 4. 具体 loss 设计

### 4.1 保留 positive component geometry，但降低权重

CGA-v2 原始默认：

```text
lambda_center   = 0.05
lambda_boundary = 0.03
lambda_scale    = 0.02
lambda_peak     = 0.03
cga_start_epoch = 1
cga_ramp_epochs = 40
```

CGA-v2.1 预声明为更保守：

```text
lambda_center   = 0.020
lambda_boundary = 0.005
lambda_scale    = 0.005
lambda_peak     = 0.020

cga_start_epoch = 20
cga_ramp_epochs = 100
```

理由：

```text
center / peak: preserve tiny target localization
boundary / scale: keep but heavily down-weight, because v2 failure suggests expansion/overactivation risk
late ramp: let base detector first learn stable localization before auxiliary geometry enters
```

### 4.2 Safe-background hard-negative loss

定义：

```python
gt_dilated = dilate(gt_mask, radius=bg_ignore_radius)
safe_bg = 1 - gt_dilated
```

建议预声明：

```text
bg_ignore_radius = 5
bg_topk_ratio    = 0.002
bg_gamma         = 2.0
lambda_bg_hard   = 0.030
lambda_bg_area   = 0.005
```

损失：

```python
prob = sigmoid(final_logits)

L_bg_hard =
  mean(topk((prob ** bg_gamma) over safe_bg, ratio=bg_topk_ratio))

L_bg_area =
  mean((prob ** 2) over safe_bg)

L_fa_control =
  lambda_bg_hard * L_bg_hard
  + lambda_bg_area * L_bg_area
```

解释：

```text
L_bg_hard: directly suppresses the most suspicious background activations.
L_bg_area: prevents broad low-level overactivation across background.
safe_bg: ignores a buffer around targets, so tiny target neighborhoods are not over-penalized.
```

### 4.3 Auxiliary ratio cap

为了避免 v2.1 变成另一个过强 regularizer，新增上界：

```python
reg_raw = L_geom_pos + L_fa_control
reg_cap = aux_ratio_cap * detach(L_base)
reg = min(reg_raw, reg_cap)
L_total = L_base + ramp(t) * reg
```

预声明：

```text
aux_ratio_cap = 0.15
```

日志必须记录：

```text
reg_raw
reg_capped
reg_cap
reg_over_base_ratio
geom_pos_total
fa_control_total
bg_hard
bg_area
```

如果 `reg_over_base_ratio` tail20 mean 长期超过 0.15，说明 cap 逻辑没有生效，seed42 结果无效。

---

## 5. 禁止事项

CGA-v2.1 设计完成并 commit 后，禁止：

```text
1. 根据 seed42 结果再调 lambda。
2. 根据 HC-Val false-alarm 图再改 bg_ignore_radius。
3. 改 threshold，主结果仍固定 threshold=0.5。
4. 改 HC-Val split。
5. 复用 CGA-v2 checkpoint。
6. 只训练 CGA-v2.1，不重新训练 paired OHEM。
7. seed42 不过 gate 还继续 seed43/44。
8. 把 v2.1 结果说成 v2 结果。
```

---

## 6. 代码修改总览

只允许新加 v2.1 variant，不破坏 v2 负结果和已有 audit。

### 6.1 新增文件

```text
utils/cga_v21_targets.py
tools/official/check_cga_v21_protocol_lock.py
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
scripts/official/guard_cga_v21_no_seed43_44_until_seed42_pass.sh
docs/internal/cga_v2_1/PREDECLARED_PROTOCOL.md
docs/internal/cga_v2_1/protocol_lock.json
tests/test_cga_v21_loss_contract.py
tests/test_cga_v21_protocol_lock.py
```

### 6.2 修改文件

```text
loss.py
train.py
net.py
model/cga_wrapper.py
docs/paper/cga_v2_aaai/README.md
```

修改原则：

```text
v2 behavior must remain unchanged when --cga_variant v2.
v2.1 behavior only activates when --cga_variant v2_1 or model_name=MSHNetCGA21.
```

---

## 7. 代码修改细节

## 7.1 `model/cga_wrapper.py`

当前 wrapper 已经是四头 CGA。v2.1 不需要新增 heads，但要允许 metadata 区分 regularizer implementation。

### 修改建议

将 constructor 改成：

```python
class CGAWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        backbone_name: str,
        feature_channels: int,
        aux_hidden_channels: int = 32,
        regularizer_impl: str = "center_boundary_scale_peak",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.regularizer_impl = str(regularizer_impl)
        self.cga_aux_head = CGAAuxHead(
            in_channels=int(feature_channels),
            hidden_channels=int(aux_hidden_channels),
        )
```

forward metadata 改成：

```python
output["regularizer_meta"].update(
    {
        "use_cga": True,
        "regularizer_impl": self.regularizer_impl,
        "fallback_regularizer_used": False,
    }
)
```

v2 默认：

```text
regularizer_impl = center_boundary_scale_peak
```

v2.1：

```text
regularizer_impl = center_boundary_scale_peak_fa_control_v21
```

---

## 7.2 `net.py`

给 `build_model()` 增加一个可选参数：

```python
regularizer_impl: str = "center_boundary_scale_peak"
```

在构建 `CGAWrapper` 时传入：

```python
return CGAWrapper(
    backbone,
    backbone_name=backbone_name,
    feature_channels=feature_channels,
    aux_hidden_channels=aux_hidden_channels,
    regularizer_impl=regularizer_impl,
)
```

新增 model name resolution：

```python
if model_name_l in {"mshnetcga21", "mshnet_cga21", "cga-v2.1", "cga21"}:
    return "mshnet", True
```

注意：

```text
net.py 只负责构建同一四头 wrapper。
v2.1 的核心差异在 loss/protocol，不在 backbone architecture。
```

---

## 7.3 `loss.py`

新增 `CGAV21LossConfig`：

```python
@dataclass(frozen=True)
class CGAV21LossConfig(CGALossConfig):
    lambda_center: float = 0.020
    lambda_boundary: float = 0.005
    lambda_scale: float = 0.005
    lambda_peak: float = 0.020

    start_epoch: int = 20
    ramp_epochs: int = 100

    lambda_bg_hard: float = 0.030
    lambda_bg_area: float = 0.005
    bg_ignore_radius: int = 5
    bg_topk_ratio: float = 0.002
    bg_gamma: float = 2.0

    aux_ratio_cap: float = 0.15
```

新增工具函数：

```python
def _topk_mean(values: torch.Tensor, ratio: float) -> torch.Tensor:
    flat = values.flatten()
    if flat.numel() == 0:
        return values.sum() * 0.0
    k = max(1, int(flat.numel() * float(ratio)))
    return torch.topk(flat, k=min(k, flat.numel()), largest=True).values.mean()


def _safe_background_mask(target: torch.Tensor, radius: int) -> torch.Tensor:
    from utils.cga_targets import binary_dilate
    target = _resize_like(target, target)
    dilated = binary_dilate(target, radius=radius)
    return (1.0 - dilated).clamp(0.0, 1.0)
```

新增 `CGAV21Loss`：

```python
class CGAV21Loss(CGALoss):
    def __init__(
        self,
        cfg: CGAV21LossConfig | None = None,
        target_cfg: CGATargetConfig | None = None,
        *,
        strict_cga_heads: bool = True,
    ) -> None:
        super().__init__(
            cfg=cfg or CGAV21LossConfig(),
            target_cfg=target_cfg,
            strict_cga_heads=strict_cga_heads,
        )
        self.cfg: CGAV21LossConfig

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
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

        geom_pos = (
            self.cfg.lambda_center * loss_center
            + self.cfg.lambda_boundary * loss_boundary
            + self.cfg.lambda_scale * loss_scale
            + self.cfg.lambda_peak * loss_peak
        )

        prob = torch.sigmoid(final_logit)
        safe_bg = _safe_background_mask(target, radius=self.cfg.bg_ignore_radius)
        bg_values = (prob.clamp(0.0, 1.0) ** self.cfg.bg_gamma) * safe_bg

        bg_hard = _topk_mean(bg_values[safe_bg > 0], self.cfg.bg_topk_ratio)
        bg_area = ((prob ** 2) * safe_bg).sum() / safe_bg.sum().clamp_min(1.0)

        fa_control = (
            self.cfg.lambda_bg_hard * bg_hard
            + self.cfg.lambda_bg_area * bg_area
        )

        reg_raw = geom_pos + fa_control
        reg_cap = self.cfg.aux_ratio_cap * base["total"].detach()
        reg_capped = torch.minimum(reg_raw, reg_cap)

        w = _ramp_weight(epoch, self.cfg.start_epoch, self.cfg.ramp_epochs)
        total = base["total"] + w * reg_capped

        return {
            "total": total,
            "base_total": base["total"].detach(),
            "ohem": base["ohem"],
            "soft_iou": base["soft_iou"],
            "scale": base["scale"],
            "cga_w": torch.tensor(w, device=final_logit.device, dtype=final_logit.dtype),

            "cga_center": loss_center.detach(),
            "cga_boundary": loss_boundary.detach(),
            "cga_scale": loss_scale.detach(),
            "cga_peak": loss_peak.detach(),

            "v21_geom_pos": geom_pos.detach(),
            "v21_bg_hard": bg_hard.detach(),
            "v21_bg_area": bg_area.detach(),
            "v21_fa_control": fa_control.detach(),
            "v21_reg_raw": reg_raw.detach(),
            "v21_reg_cap": reg_cap.detach(),
            "v21_reg_capped": reg_capped.detach(),
            "v21_reg_over_base": (reg_capped.detach() / base["total"].detach().clamp_min(1e-6)),
        }
```

更新 `build_loss()`：

```python
def build_loss(..., cga_variant: str = "v2", ...):
    ...
    if use_cga:
        if str(cga_variant).lower() in {"v2_1", "v21", "cga-v2.1"}:
            cfg = CGAV21LossConfig(
                ohem_ratio=float(ohem_ratio),
                lambda_iou=float(lambda_iou),
                warm_epoch=int(mshnet_warm_epoch),
                # allow explicit overrides only before protocol lock
                lambda_center=float(lambda_center),
                lambda_boundary=float(lambda_boundary),
                lambda_scale=float(lambda_scale),
                lambda_peak=float(lambda_peak),
                start_epoch=int(cga_start_epoch),
                ramp_epochs=int(cga_ramp_epochs),
                lambda_bg_hard=float(kwargs.get("lambda_bg_hard", 0.030)),
                lambda_bg_area=float(kwargs.get("lambda_bg_area", 0.005)),
                bg_ignore_radius=int(kwargs.get("bg_ignore_radius", 5)),
                bg_topk_ratio=float(kwargs.get("bg_topk_ratio", 0.002)),
                bg_gamma=float(kwargs.get("bg_gamma", 2.0)),
                aux_ratio_cap=float(kwargs.get("aux_ratio_cap", 0.15)),
            )
            return CGAV21Loss(cfg, strict_cga_heads=strict_cga_heads)

        return CGALoss(...)
```

重要：

```text
v2 默认行为不能被 v2.1 参数污染。
```

---

## 7.4 `train.py`

新增 CLI：

```python
p.add_argument("--cga_variant", default="v2", choices=["v2", "v2_1"])
p.add_argument("--lambda_bg_hard", type=float, default=0.030)
p.add_argument("--lambda_bg_area", type=float, default=0.005)
p.add_argument("--bg_ignore_radius", type=int, default=5)
p.add_argument("--bg_topk_ratio", type=float, default=0.002)
p.add_argument("--bg_gamma", type=float, default=2.0)
p.add_argument("--aux_ratio_cap", type=float, default=0.15)
p.add_argument("--protocol_lock", default="")
```

如果是 v2.1，强制 metadata：

```python
regularizer_impl = (
    "center_boundary_scale_peak_fa_control_v21"
    if args.cga_variant == "v2_1"
    else "center_boundary_scale_peak"
)
```

build model：

```python
model = build_model(
    model_name=args.model_name,
    backbone_name=backbone_name,
    use_cga=use_cga,
    evidence_mode=args.evidence_mode,
    input_channels=1,
    aux_hidden_channels=args.aux_hidden_channels,
    allow_fallback_regularizer=args.allow_fallback_regularizer,
    regularizer_impl=regularizer_impl,
).to(device)
```

build loss：

```python
criterion = build_loss(
    args.model_name or backbone_name,
    use_cga=use_cga,
    ohem_ratio=args.ohem_ratio,
    mshnet_warm_epoch=args.mshnet_warm_epoch,
    cga_start_epoch=args.cga_start_epoch,
    cga_ramp_epochs=args.cga_ramp_epochs,
    lambda_center=args.lambda_center,
    lambda_boundary=args.lambda_boundary,
    lambda_scale=args.lambda_scale,
    lambda_peak=args.lambda_peak,
    strict_cga_heads=(args.evidence_mode == "paper" and use_cga),
    cga_variant=args.cga_variant,
    lambda_bg_hard=args.lambda_bg_hard,
    lambda_bg_area=args.lambda_bg_area,
    bg_ignore_radius=args.bg_ignore_radius,
    bg_topk_ratio=args.bg_topk_ratio,
    bg_gamma=args.bg_gamma,
    aux_ratio_cap=args.aux_ratio_cap,
).to(device)
```

evidence metadata 里新增：

```python
"cga_variant": args.cga_variant,
"regularizer_impl": regularizer_impl,
"lambda_bg_hard": args.lambda_bg_hard,
"lambda_bg_area": args.lambda_bg_area,
"bg_ignore_radius": args.bg_ignore_radius,
"bg_topk_ratio": args.bg_topk_ratio,
"bg_gamma": args.bg_gamma,
"aux_ratio_cap": args.aux_ratio_cap,
"protocol_lock": args.protocol_lock,
```

如果 `args.cga_variant == "v2_1"`，建议默认覆盖：

```python
if args.cga_variant == "v2_1":
    if args.cga_start_epoch == 1:
        args.cga_start_epoch = 20
    if args.cga_ramp_epochs == 40:
        args.cga_ramp_epochs = 100
    # v2.1 conservative defaults
    if args.lambda_center == 0.05:
        args.lambda_center = 0.020
    if args.lambda_boundary == 0.03:
        args.lambda_boundary = 0.005
    if args.lambda_scale == 0.02:
        args.lambda_scale = 0.005
    if args.lambda_peak == 0.03:
        args.lambda_peak = 0.020
```

更严格做法：

```text
v2.1 runner 显式传入所有超参，不依赖自动覆盖。
```

---

## 8. Protocol lock

新增：

```text
docs/internal/cga_v2_1/protocol_lock.json
```

内容：

```json
{
  "protocol_name": "CGA-v2.1-FAC",
  "status": "predeclared_before_seed42",
  "canonical_root": "/home/ly/AAAI/CGA-main",
  "baseline": "MSHNetOHEM",
  "candidate": "MSHNetCGA21",
  "dataset": "NUDT-SIRST",
  "splits": {
    "train": "train_NUDT-SIRST.txt",
    "test": "test_NUDT-SIRST.txt",
    "hcval": "hcval_NUDT-SIRST.txt"
  },
  "threshold": 0.5,
  "threshold_selection": "fixed_predeclared",
  "seed42_first": true,
  "seed43_44_allowed_only_after_seed42_gate_pass": true,
  "model_changes": {
    "backbone": "unchanged_mshnet_adapter",
    "cga_aux_heads": "unchanged_center_boundary_scale_peak",
    "new_loss": "safe_background_false_alarm_control_and_aux_ratio_cap"
  },
  "hyperparameters": {
    "lambda_center": 0.02,
    "lambda_boundary": 0.005,
    "lambda_scale": 0.005,
    "lambda_peak": 0.02,
    "cga_start_epoch": 20,
    "cga_ramp_epochs": 100,
    "lambda_bg_hard": 0.03,
    "lambda_bg_area": 0.005,
    "bg_ignore_radius": 5,
    "bg_topk_ratio": 0.002,
    "bg_gamma": 2.0,
    "aux_ratio_cap": 0.15
  },
  "primary_seed42_gate": {
    "delta_mIoU_min": 0.02,
    "delta_Precision_min": 0.01,
    "delta_FA_ppm_max": 0.0,
    "delta_Pd_min": -0.001
  },
  "hcval_guard": {
    "delta_mIoU_min": 0.0,
    "delta_FA_ppm_max": 50.0,
    "delta_Precision_min": -0.02,
    "delta_Pd_min": -0.001
  },
  "forbidden": [
    "change_threshold_after_seed42",
    "change_hcval_split_after_seed42",
    "reuse_cga_v2_checkpoint",
    "run_seed43_44_before_seed42_pass",
    "retune_v21_hyperparameters_after_seeing_seed42",
    "claim_cga_v2_positive_using_cga_v21_results"
  ]
}
```

---

## 9. Runner 设计

新增：

```text
scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

脚本逻辑：

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
EPOCHS=${EPOCHS:-400}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ly/AAAI/CGA-main/results/official_cga_v21_from_zero}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-docs/internal/cga_v2_1/protocol_lock.json}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

test -f "${PROTOCOL_LOCK}"

# guard: no seed43/44 before seed42 pass
if [ "${SEED}" != "42" ]; then
  bash scripts/official/guard_cga_v21_no_seed43_44_until_seed42_pass.sh
fi

# baseline from zero
"${PYTHON}" train.py \
  --model_name MSHNetOHEM \
  --backbone_name mshnet \
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
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --output_dir "${OUTPUT_DIR}"

# candidate from zero
"${PYTHON}" train.py \
  --model_name MSHNetCGA21 \
  --backbone_name mshnet \
  --use_cga \
  --cga_variant v2_1 \
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
  --cga_ramp_epochs 100 \
  --lambda_center 0.020 \
  --lambda_boundary 0.005 \
  --lambda_scale 0.005 \
  --lambda_peak 0.020 \
  --lambda_bg_hard 0.030 \
  --lambda_bg_area 0.005 \
  --bg_ignore_radius 5 \
  --bg_topk_ratio 0.002 \
  --bg_gamma 2.0 \
  --aux_ratio_cap 0.15 \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --protocol_lock "${PROTOCOL_LOCK}" \
  --output_dir "${OUTPUT_DIR}"

# evaluation and paired delta can reuse existing official eval/compare tools,
# but output must go under:
# docs/internal/cga_v2_1/gate_p21_seed42_NUDT-SIRST/
```

---

## 10. Seed42 gate

新增：

```text
docs/internal/cga_v2_1/gate_p21_seed42_NUDT-SIRST/summary.json
```

### 10.1 Primary AAAI gate

必须同时满足：

```text
Full ΔmIoU      >= +0.020
Full ΔPrecision >= +0.010
Full ΔFA_ppm    <= 0.0
Full ΔPd        >= -0.001
```

### 10.2 HC-Val guard

必须满足：

```text
HC-Val ΔmIoU      >= 0.0
HC-Val ΔFA_ppm    <= +50.0
HC-Val ΔPrecision >= -0.020
HC-Val ΔPd        >= -0.001
```

解释：

```text
v2 已经在 HC-Val false alarm 上 collapse。
v2.1 若仍显著放大 HC-Val FA，则不允许写 hard-clutter / false-alarm-control claim。
```

### 10.3 决策表

| Seed42 result | Decision |
|---|---|
| Primary gate fail | Stop CGA-v2.1 AAAI route. Do not run seed43/44. |
| Primary pass, HC-Val guard fail | Method may improve Full but cannot claim hard-clutter robustness; AAAI route still high-risk. |
| Primary pass, HC-Val guard pass | Run paired seed43/44. |
| seed43/44 unstable | No AAAI main positive claim. |
| seed42/43/44 stable | Write narrow MSHNet-style CGA-v2.1 claim. |

---

## 11. 代码测试

新增测试：

```text
tests/test_cga_v21_loss_contract.py
tests/test_cga_v21_protocol_lock.py
tests/test_cga_v21_no_v2_contamination.py
```

### 11.1 Loss contract test

检查：

```text
CGAV21Loss returns:
  total
  base_total
  cga_center
  cga_boundary
  cga_scale
  cga_peak
  v21_geom_pos
  v21_bg_hard
  v21_bg_area
  v21_fa_control
  v21_reg_raw
  v21_reg_cap
  v21_reg_capped
  v21_reg_over_base

No NaN/Inf.
v21_reg_over_base <= aux_ratio_cap + tolerance.
```

### 11.2 v2 不受污染

同一 dummy input：

```text
build_loss(... cga_variant=v2)
```

不得返回：

```text
v21_bg_hard
v21_fa_control
v21_reg_capped
```

### 11.3 protocol lock test

检查：

```text
protocol_lock.json exists
threshold == 0.5
seed42_first == true
forbidden contains no retuning / no seed43 before seed42
```

---

## 12. 执行顺序

```bash
cd /home/ly/AAAI/CGA-main

git checkout -b cga-v2.1-fa-control-predeclare
```

### R0. 冻结 v2 状态

```bash
git status --short
```

确认没有：

```text
results/
predictions/
*.pth
*.pth.tar
```

进入 git。

### R1. 写入 v2.1 protocol

```bash
mkdir -p docs/internal/cga_v2_1
# 写入 PREDECLARED_PROTOCOL.md 和 protocol_lock.json
```

### R2. 完成代码修改

```bash
git add \
  loss.py \
  train.py \
  net.py \
  model/cga_wrapper.py \
  docs/internal/cga_v2_1/PREDECLARED_PROTOCOL.md \
  docs/internal/cga_v2_1/protocol_lock.json \
  scripts/official/run_cga_v21_seed42_from_zero_paired.sh \
  scripts/official/guard_cga_v21_no_seed43_44_until_seed42_pass.sh \
  tests/test_cga_v21_loss_contract.py \
  tests/test_cga_v21_protocol_lock.py \
  tests/test_cga_v21_no_v2_contamination.py

git commit -m "Predeclare CGA-v2.1 false-alarm-controlled protocol"
```

### R3. 只跑 contract / smoke

```bash
python3 -m py_compile train.py loss.py net.py model/cga_wrapper.py

bash -n scripts/official/run_cga_v21_seed42_from_zero_paired.sh
bash -n scripts/official/guard_cga_v21_no_seed43_44_until_seed42_pass.sh

python3 -m pytest tests/test_cga_v21_loss_contract.py tests/test_cga_v21_protocol_lock.py
```

如果环境没有 pytest，先至少跑：

```bash
python3 - <<'PY'
import torch
from loss import build_loss
loss = build_loss(
    "MSHNetCGA21",
    use_cga=True,
    cga_variant="v2_1",
    strict_cga_heads=True,
)
print(type(loss).__name__)
PY
```

### R4. seed42 from-zero paired

只有 P1/P1A 已 pass 后运行：

```bash
CUDA_DEVICE=1 \
ROOT=/home/ly/AAAI/CGA-main \
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
EPOCHS=400 \
OUTPUT_DIR=/home/ly/AAAI/CGA-main/results/official_cga_v21_from_zero \
bash scripts/official/run_cga_v21_seed42_from_zero_paired.sh
```

### R5. seed42 决策

```text
如果 seed42 fail:
  stop
  不跑 seed43/44
  不写 AAAI positive claim

如果 seed42 pass:
  再跑 seed43/44 paired
```

---

## 13. 论文 claim 边界

### seed42 前

只能写：

```text
CGA-v2.1 is a predeclared rescue protocol motivated by the audited CGA-v2 valid-negative result.
```

### seed42 pass 但未 multiseed

只能写：

```text
Preliminary seed42 paired evidence suggests that false-alarm-controlled CGA-v2.1 may address the CGA-v2 over-detection failure mode.
```

### 三种子稳定后

可以写：

```text
CGA-v2.1 is a training-time false-alarm-controlled component-geometry regularizer for MSHNet-style IRSTD.
It preserves the inference path while improving target preservation without increasing false alarms under a frozen paired protocol.
```

仍然不能写：

```text
universal plug-and-play
multi-backbone proof
solves all hard clutter
SOTA across IRSTD
every head is strictly necessary
```

---

## 14. 最终建议

如果 AAAI deadline 很近，CGA-v2.1 是高风险路线。它只有在 seed42 迅速翻正时才值得继续。

推荐 Go/No-Go：

```text
T0: commit protocol lock
T1: run seed42 from-zero paired
T2: if seed42 primary + HC-Val guard pass -> seed43/44
T3: if seed42 fail -> stop AAAI-main route
```

不要再在 CGA-v2 上消耗时间；v2.1 必须作为新方法、新协议、新证据链处理。
