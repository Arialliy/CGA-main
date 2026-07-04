# CGA-v2 AAAI-27 Narrow-Claim Rescue Route：方案与代码修改

> 推荐论文定位：**Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection**  
> 当前策略：**pivot-with-rescue-route / revise-narrow-claim**  
> 当前重点：不要继续扩模型；先把 MSHNet-style controlled paired evidence 做成可信论文证据。

---

## 0. 适用范围

本文档适用于当前 `CGA-main` 的 AAAI-27 抢救路线：

```text
目标：用最小增量把 CGA-v2 从“工程正确”推进到“窄 claim 可投稿证据”。
对象：MSHNet-style IRSTD + 训练期 component-geometry regularization。
优先级：结果证据 > novelty 叙事 > 多 backbone 扩展。
```

不要把当前版本包装成：

```text
universal plug-and-play regularizer for all IRSTD detectors
new SOTA method across all backbones
full hard-clutter closure method
```

当前更安全的 claim 是：

```text
CGA is a training-time component-geometry regularizer for MSHNet-style IRSTD.
It preserves the inference path and improves target preservation / hard-clutter behavior under a frozen controlled paired protocol.
```

---

## 1. 当前评分与决策

| 维度 | 分数 |
|---|---:|
| Problem importance | 3.5 / 5 |
| Novelty | 2.5 / 5 |
| Conceptual innovation | 2.5 / 5 |
| Method soundness | 3.5 / 5 |
| Elegance | 3.5 / 5 |
| Feasibility before AAAI-27 | 2.5 / 5 |
| Experimental convincibility | 2.5 / 5 |
| Venue fit | 3.0 / 5 |
| Timeliness | 3.5 / 5 |
| Risk-adjusted acceptance potential | 2.5 / 5 |

```text
Weighted score: 2.95 / 5 ≈ 5.9 / 10
Recommendation: pivot-with-rescue-route / revise-narrow-claim
```

结论：

```text
不是 abandon。
但当前版本离 AAAI 主会还差：
1. 当前仓库重新生成的 paper-evidence 结果；
2. 与 SLS / ISNet / PConv-SD 的 novelty delta；
3. 至少 MSHNet-style paired evidence 的稳定三种子结果。
```

---

## 2. 当前论文主张应如何收缩

### 2.1 推荐标题

```text
Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection
```

### 2.2 摘要中的安全主张

推荐写法：

```text
We propose CGA, a training-time component-geometry regularization strategy for MSHNet-style infrared small target detection. CGA supervises complementary center, boundary, scale, and peak cues through auxiliary heads during training, while leaving the inference path unchanged. Under a frozen controlled paired protocol, CGA improves target-preservation behavior and reduces failure modes in hard clutter.
```

不要写：

```text
CGA is universally effective for all IRSTD backbones.
CGA is the first geometry/shape-aware supervision for IRSTD.
CGA solves hard-clutter infrared small target detection.
CGA establishes a new generic plug-and-play framework.
```

### 2.3 Claim ladder

| Evidence stage | 允许 claim | 禁止 claim |
|---|---|---|
| Gate-1A smoke only | implementation path validated | paper evidence |
| seed42 MSHNet paired positive | MSHNet-style single-seed evidence | robust improvement |
| seed42/43/44 MSHNet paired stable | controlled paired MSHNet-style improvement | universal plug-and-play |
| MSHNet + DNANet + ALCNet/ACM positive | multiple host detectors under controlled protocol | all IRSTD models |

当前推荐主线只需要先冲到：

```text
seed42/43/44 MSHNet paired stable
```

---

## 3. 最近 prior-art 风险与 novelty delta

| Work | 对 CGA 的威胁 | CGA 必须强调的差异 |
|---|---|---|
| MSHNet / SLS, CVPR 2024 | 已覆盖 MSHNet 框架、scale/location-sensitive loss、center/location 约束 | CGA 是 component-level auxiliary geometry，不是 scale/location loss 变体 |
| ISNet, CVPR 2022 | 已把 shape reconstruction / edge modeling 带入 IRSTD | boundary/peak 不能说是首次形状监督，只能说是 component-geometry regularization 的一部分 |
| PConv + SD Loss, AAAI 2025 | AAAI 已接受 IRSTD 的 spatial distribution + scale-aware dynamic loss + benchmark package | 只靠 auxiliary heads 不够，必须做 target preservation / hard clutter diagnostics |
| AAAI-26 domain-auxiliary IR moving small target | 表明 AAAI 对 IR small target 开放，但偏新 setting / domain shift / 多数据 | 单 NUDT + MSHNet anchor 必须窄 claim，不能假装大一统 |

---

## 4. Strict blockers

### Blocker A：没有当前仓库三种子结果

不能使用历史 OHCM-MSHNet 结果作为当前 repo evidence。

必须重新生成：

```text
seed42 / seed43 / seed44
MSHNetOHEM baseline vs MSHNetCGA
same dataset split
same training schedule
same threshold=0.5
same evaluation script
```

### Blocker B：novelty 被压薄

当前最强可守 claim 是：

```text
target-preserving component-geometry regularization
```

不要主打：

```text
new backbone
universal regularizer
new SOTA loss
first geometry-aware IRSTD method
```

### Blocker C：只注册了 MSHNet

在 `model/registry.py` 只有 `mshnet` 时，不能写：

```text
plug-and-play across multiple backbones
```

只能写：

```text
MSHNet-style host detector
```

### Blocker D：四头 ablation 很可能是 mixed attribution

如果 ablation 不能证明每个 head 都必要，就写：

```text
The four cues jointly provide a mixed target-preserving regularization effect.
```

不要写：

```text
Each head is independently necessary and sufficient.
```

---

## 5. 最小可投稿路线

不要再改模型结构。先走证据路线：

```text
Gate-R0: release-sync / metadata ownership check
Gate-R1: P1 + P1A dataset evidence gate
Gate-R2: seed42 paired run: MSHNetOHEM vs MSHNetCGA
Gate-R3: seed43/44 paired run
Gate-R4: seed42 ablation + failure pack
Gate-R5: paper table + claim freeze
```

核心判断标准：

```text
Full mIoU: CGA - baseline >= +0.020
Precision: CGA - baseline >= +0.010
FA: no increase
threshold: 0.5
```

如果 seed42 不过：

```text
停止 AAAI main route
改 workshop / journal extension / second-backbone rescue
```

如果 seed42 过但 seed43/44 不稳：

```text
保留 narrow claim
写 target-preservation diagnostics
不投 AAAI main，或改 weaker venue
```

如果三种子稳定：

```text
可以准备 AAAI narrow-claim submission
```

---

# Part I：必须修改/确认的代码

---

## 6. 修改 1：`model/cga_wrapper.py` 的 metadata ownership

### 6.1 问题

`CGAWrapper` 不应该写：

```python
"paper_evidence_allowed": True
```

因为 wrapper 只知道模型用了真实 CGA，不知道：

```text
P1 是否通过
P1A 是否通过
是否 smoke runner
是否用了 fallback
是否满足 paper evidence protocol
```

### 6.2 正确实现

`model/cga_wrapper.py` 应改成：

```python
output.setdefault("regularizer_meta", {})
output["regularizer_meta"].update(
    {
        "use_cga": True,
        "regularizer_impl": self.REGULARIZER_IMPL,  # center_boundary_scale_peak
        "fallback_regularizer_used": False,
        "regularizer_scope": "training_auxiliary_heads",
        "regularizer_owner": "model.cga_wrapper.CGAWrapper",
    }
)
```

不要在 `model/` 层出现：

```python
paper_evidence_allowed
```

### 6.3 grep guard

```bash
grep -R 'paper_evidence_allowed' -n model || true
```

理想结果：

```text
model/ 下无 paper_evidence_allowed
```

---

## 7. 修改 2：`train.py` 增加 P1/P1A paper evidence gate

### 7.1 新增 CLI

在 `parse_args()` 加：

```python
p.add_argument(
    "--p1_preflight_passed",
    action="store_true",
    help="Set only when Gate-P1 dataset preflight summary has gate_pass=true.",
)

p.add_argument(
    "--p1a_hcval_source_audit_passed",
    action="store_true",
    help="Set only when Gate-P1A HC-Val source audit summary has gate_pass=true.",
)

p.add_argument(
    "--legacy_model_factory",
    action="store_true",
    help="Use legacy MSHNetCGA factory for Gate-1B trend check only.",
)

p.add_argument(
    "--eval_threshold",
    type=float,
    default=0.5,
    help="Fixed binarization threshold for paper-evidence evaluation.",
)
```

继续使用当前命名：

```text
--evidence_mode smoke|paper
--mshnet_warm_epoch
mshnet_warm_flag
```

不要再新增：

```text
--paper_mode
--mshnet_warm
```

### 7.2 新增函数

```python
def compute_paper_evidence_allowed(
    args: argparse.Namespace,
    *,
    fallback_regularizer_used: bool,
) -> bool:
    """Single owner of paper-evidence eligibility metadata."""
    return bool(
        args.evidence_mode == "paper"
        and bool(args.p1_preflight_passed)
        and bool(args.p1a_hcval_source_audit_passed)
        and not bool(fallback_regularizer_used)
    )
```

### 7.3 fail-closed guard

```python
def assert_paper_evidence_gate(
    args: argparse.Namespace,
    *,
    paper_evidence_allowed: bool,
    fallback_regularizer_used: bool,
) -> None:
    if args.evidence_mode != "paper":
        return

    if fallback_regularizer_used:
        raise RuntimeError(
            "Fallback regularizer is forbidden for paper evidence. "
            "Use --evidence_mode smoke for smoke-only tests."
        )

    if not paper_evidence_allowed:
        missing = []
        if not args.p1_preflight_passed:
            missing.append("P1 dataset preflight")
        if not args.p1a_hcval_source_audit_passed:
            missing.append("P1A HC-Val source audit")
        raise RuntimeError(
            "Paper-evidence training is blocked. Missing gate(s): "
            + ", ".join(missing)
            + "."
        )
```

### 7.4 build_model 传入 legacy flag

当前 `net.py` 支持 `legacy_model_factory`，但 `train.py` 必须把 CLI 传进去：

```python
model = build_model(
    model_name=args.model_name,
    backbone_name=backbone_name,
    use_cga=use_cga,
    evidence_mode=args.evidence_mode,
    input_channels=1,
    aux_hidden_channels=args.aux_hidden_channels,
    allow_fallback_regularizer=args.allow_fallback_regularizer,
    legacy_model_factory=args.legacy_model_factory,
).to(device)
```

否则 Gate-1B 不能真正比较 legacy MSHNetCGA 和 adapter-CGA。

### 7.5 日志与 checkpoint metadata

训练日志和 checkpoint 必须写入：

```python
regularizer_meta = output.get("regularizer_meta", {}) if isinstance(output, dict) else {}
fallback_regularizer_used = bool(
    regularizer_meta.get("fallback_regularizer_used", False)
    or args.allow_fallback_regularizer
)
paper_evidence_allowed = compute_paper_evidence_allowed(
    args,
    fallback_regularizer_used=fallback_regularizer_used,
)
assert_paper_evidence_gate(
    args,
    paper_evidence_allowed=paper_evidence_allowed,
    fallback_regularizer_used=fallback_regularizer_used,
)

evidence_meta = {
    "epoch": epoch,
    "dataset": args.dataset_name,
    "model": run_model_name,
    "backbone": backbone_name,
    "use_cga": bool(use_cga),
    "regularizer_impl": regularizer_meta.get(
        "regularizer_impl",
        "center_boundary_scale_peak" if use_cga else "none",
    ),
    "fallback_regularizer_used": bool(fallback_regularizer_used),
    "evidence_mode": args.evidence_mode,
    "paper_evidence_allowed": bool(paper_evidence_allowed),
    "p1_preflight_passed": bool(args.p1_preflight_passed),
    "p1a_hcval_source_audit_passed": bool(args.p1a_hcval_source_audit_passed),
    "protocol": args.protocol,
    "seed": args.seed,
    "eval_threshold": float(args.eval_threshold),
    "mshnet_warm_epoch": args.mshnet_warm_epoch,
    "cga_start_epoch": args.cga_start_epoch,
    "cga_ramp_epochs": args.cga_ramp_epochs,
}
```

checkpoint 同步写入同样字段。

---

## 8. 修改 3：`test.py` / `evaluate.py` 固定 threshold=0.5

### 8.1 问题

AAAI route 中不能用 sweep threshold 得到最好结果，然后说是 fixed protocol。

### 8.2 修改

如果 `test.py` 没有该参数，新增：

```python
p.add_argument("--eval_threshold", type=float, default=0.5)
```

预测时固定：

```python
prob = torch.sigmoid(logits)
pred = (prob >= float(args.eval_threshold)).float()
```

日志/metrics JSON 中必须写：

```json
{
  "eval_threshold": 0.5,
  "threshold_selection": "fixed_predeclared"
}
```

不允许写：

```text
best_threshold
val_selected_threshold
threshold_sweep_best
```

除非该结果单独标为 diagnostic，不进主表。

---

## 9. 修改 4：新增 paired seed runner

新增：

```text
scripts/official/run_cga_v2_aaai_nudt_one_seed_paired_train_eval.sh
```

建议内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"

: "${DATASET_DIR:?DATASET_DIR is required}"
: "${DATASET_NAME:=NUDT-SIRST}"
: "${SEED:=42}"
: "${EPOCHS:=400}"
: "${BATCH_SIZE:=8}"
: "${PATCH_SIZE:=256}"
: "${NUM_WORKERS:=4}"
: "${EVAL_THRESHOLD:=0.5}"
: "${CUDA_VISIBLE_DEVICES:=1}"

export CUDA_VISIBLE_DEVICES

COMMON_ARGS=(
  --evidence_mode paper
  --protocol controlled
  --p1_preflight_passed
  --p1a_hcval_source_audit_passed
  --dataset_dir "${DATASET_DIR}"
  --dataset_name "${DATASET_NAME}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --patch_size "${PATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_threshold "${EVAL_THRESHOLD}"
  --mshnet_warm_epoch 5
  --output_dir results/official
)

python train.py \
  --model_name MSHNetOHEM \
  "${COMMON_ARGS[@]}"

python train.py \
  --model_name MSHNetCGA \
  --use_cga \
  --cga_start_epoch 1 \
  --cga_ramp_epochs 40 \
  "${COMMON_ARGS[@]}"

# If test.py/evaluate.py CLI names differ, adapt this block only.
python test.py \
  --model_name MSHNetOHEM \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --eval_threshold "${EVAL_THRESHOLD}" \
  --checkpoint results/official/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/MSHNetOHEM_${EPOCHS}.pth.tar \
  --output_dir results/official/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/eval

python test.py \
  --model_name MSHNetCGA \
  --dataset_dir "${DATASET_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --eval_threshold "${EVAL_THRESHOLD}" \
  --checkpoint results/official/MSHNetCGA/seed${SEED}/${DATASET_NAME}/MSHNetCGA_${EPOCHS}.pth.tar \
  --output_dir results/official/MSHNetCGA/seed${SEED}/${DATASET_NAME}/eval
```

设权限：

```bash
chmod +x scripts/official/run_cga_v2_aaai_nudt_one_seed_paired_train_eval.sh
```

---

## 10. 修改 5：新增 paper-evidence manifest checker

新增：

```text
tools/official/check_cga_v2_paper_evidence_manifest.py
```

用途：检查训练/评估结果是否能进论文表格。

最小逻辑：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_last_jsonl(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        raise RuntimeError(f"empty log: {path}")
    return rows[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_log", required=True)
    p.add_argument("--cga_log", required=True)
    p.add_argument("--baseline_metrics", required=True)
    p.add_argument("--cga_metrics", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_full_miou_gain", type=float, default=0.020)
    p.add_argument("--min_precision_gain", type=float, default=0.010)
    p.add_argument("--max_fa_gain", type=float, default=0.0)
    args = p.parse_args()

    b_log = load_last_jsonl(Path(args.baseline_log))
    c_log = load_last_jsonl(Path(args.cga_log))
    b_m = json.loads(Path(args.baseline_metrics).read_text())
    c_m = json.loads(Path(args.cga_metrics).read_text())

    required_log_flags = [
        "paper_evidence_allowed",
        "p1_preflight_passed",
        "p1a_hcval_source_audit_passed",
    ]

    errors = []
    for name, row in [("baseline", b_log), ("cga", c_log)]:
        for key in required_log_flags:
            if row.get(key) is not True:
                errors.append(f"{name}:{key}_not_true")
        if row.get("evidence_mode") != "paper":
            errors.append(f"{name}:evidence_mode_not_paper")
        if row.get("protocol") != "controlled":
            errors.append(f"{name}:protocol_not_controlled")

    if c_log.get("regularizer_impl") != "center_boundary_scale_peak":
        errors.append("cga:regularizer_impl_not_center_boundary_scale_peak")
    if c_log.get("fallback_regularizer_used") is not False:
        errors.append("cga:fallback_regularizer_used_not_false")

    delta = {
        "full_miou": float(c_m["Full"]["mIoU"]) - float(b_m["Full"]["mIoU"]),
        "precision": float(c_m["Full"]["Precision"]) - float(b_m["Full"]["Precision"]),
        "fa": float(c_m["Full"]["FA"]) - float(b_m["Full"]["FA"]),
    }

    if delta["full_miou"] < args.min_full_miou_gain:
        errors.append("full_miou_gain_below_gate")
    if delta["precision"] < args.min_precision_gain:
        errors.append("precision_gain_below_gate")
    if delta["fa"] > args.max_fa_gain:
        errors.append("fa_increased")

    result = {
        "gate": "Gate-CGA-v2-AAAI-NUDT-one-seed-paper-evidence",
        "gate_pass": not errors,
        "errors": errors,
        "delta": delta,
        "baseline_log": str(args.baseline_log),
        "cga_log": str(args.cga_log),
        "baseline_metrics": str(args.baseline_metrics),
        "cga_metrics": str(args.cga_metrics),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

注意：`b_m["Full"]["mIoU"]` 这类 key 要按你当前 `evaluate.py` 的实际 JSON schema 改一次。

---

## 11. 修改 6：新增三种子 runner

新增：

```text
scripts/official/run_cga_v2_aaai_nudt_three_seed_paired.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
cd "${ROOT}"

: "${DATASET_DIR:?DATASET_DIR is required}"
: "${DATASET_NAME:=NUDT-SIRST}"

for SEED in 42 43 44; do
  echo "[INFO] Running paired seed ${SEED}"
  DATASET_DIR="${DATASET_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  SEED="${SEED}" \
  EPOCHS=${EPOCHS:-400} \
  EVAL_THRESHOLD=${EVAL_THRESHOLD:-0.5} \
  bash scripts/official/run_cga_v2_aaai_nudt_one_seed_paired_train_eval.sh

  echo "[INFO] Checking paper-evidence manifest for seed ${SEED}"
  python -m tools.official.check_cga_v2_paper_evidence_manifest \
    --baseline_log results/official/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/train_log.jsonl \
    --cga_log results/official/MSHNetCGA/seed${SEED}/${DATASET_NAME}/train_log.jsonl \
    --baseline_metrics results/official/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/eval/metrics.json \
    --cga_metrics results/official/MSHNetCGA/seed${SEED}/${DATASET_NAME}/eval/metrics.json \
    --output docs/internal/cga_v2/gates/seed${SEED}_paper_evidence_summary.json

done
```

---

## 12. 修改 7：ablation runner，不改模型

四头 ablation 不需要改模型，用 lambda 置零：

```bash
# full CGA
--lambda_center 0.05 --lambda_boundary 0.03 --lambda_scale 0.02 --lambda_peak 0.03

# no center
--lambda_center 0.00 --lambda_boundary 0.03 --lambda_scale 0.02 --lambda_peak 0.03

# no boundary
--lambda_center 0.05 --lambda_boundary 0.00 --lambda_scale 0.02 --lambda_peak 0.03

# no scale
--lambda_center 0.05 --lambda_boundary 0.03 --lambda_scale 0.00 --lambda_peak 0.03

# no peak
--lambda_center 0.05 --lambda_boundary 0.03 --lambda_scale 0.02 --lambda_peak 0.00
```

新增：

```text
scripts/official/run_cga_v2_aaai_seed42_ablation.sh
```

Ablation 结论写法：

```text
The four components jointly regularize target-preserving representations. Because the auxiliary heads interact through the shared decoder feature, we report ablations as mixed-mechanism diagnostics rather than claiming each head is independently necessary.
```

---

## 13. 修改 8：failure pack / target-preservation diagnostics

AAAI 叙事不能只靠主表。必须给出 hard-clutter / target-preservation evidence。

新增：

```text
tools/official/build_cga_v2_failure_pack.py
```

最小输出：

```text
docs/paper/cga_v2_aaai/failure_pack/
  seed42_false_alarm_reduced.csv
  seed42_missed_target_recovered.csv
  seed42_tradeoff_cases.csv
  overlays/
    <case_id>_baseline.png
    <case_id>_cga.png
    <case_id>_gt.png
```

选择规则必须预注册：

```text
1. 从 fixed test/HC-Val split 中选择；
2. 不根据论文叙事手工挑图；
3. 排序依据写入 CSV，例如 baseline false positives minus CGA false positives；
4. 保留失败案例，不只展示成功案例。
```

论文图可以分三类：

```text
A. weak target preserved
B. clutter false alarm suppressed
C. tradeoff / failure case
```

---

# Part II：执行命令

---

## 14. R0：release-sync / metadata ownership check

```bash
cd /home/ly/AAAI/CGA-main

git diff --check
bash scripts/official/run_cga_v2_contract.sh --height 64 --width 64

grep -R 'paper_evidence_allowed' -n model || true

grep -R 'compute_paper_evidence_allowed\|p1_preflight_passed\|p1a_hcval_source_audit_passed\|legacy_model_factory' -n \
  train.py net.py tests scripts tools docs || true
```

通过标准：

```text
1. contract tests pass
2. model/ 下没有 paper_evidence_allowed
3. train.py 有 compute_paper_evidence_allowed
4. train.py 有 --p1_preflight_passed
5. train.py 有 --p1a_hcval_source_audit_passed
6. train.py 有 --legacy_model_factory 并传给 build_model
```

---

## 15. R1：P1/P1A gate

如果当前 `summary.json` 和 `hcval_source_summary.json` 已经 pass，仍建议跑一次确认：

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

如果 HC-Val source audit 需要重跑：

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
HCVAL_LIST=/path/to/frozen/hcval_NUDT-SIRST.txt \
SOURCE_NOTE='Recovered from pre-existing frozen HC-Val split before new-repo seed42 training.' \
bash scripts/official/run_cga_v2_nudt_hcval_list_source_audit.sh
```

---

## 16. R2：seed42 paired run

```bash
cd /home/ly/AAAI/CGA-main

DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
EPOCHS=400 \
EVAL_THRESHOLD=0.5 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/official/run_cga_v2_aaai_nudt_one_seed_paired_train_eval.sh
```

seed42 通过条件：

```text
Full mIoU >= +0.020
Precision >= +0.010
FA not increased
paper_evidence_allowed=true
fallback_regularizer_used=false
threshold=0.5
```

不通过则停，不跑三种子。

---

## 17. R3：seed43/44

seed42 过后再跑：

```bash
cd /home/ly/AAAI/CGA-main

DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
EPOCHS=400 \
EVAL_THRESHOLD=0.5 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/official/run_cga_v2_aaai_nudt_three_seed_paired.sh
```

三种子结果表至少输出：

```text
seed
model
Full mIoU
Full nIoU
Precision
Recall
F1
Pd
Fa
HC-Val mIoU
HC-Val Precision
HC-Val Fa
paper_evidence_allowed
fallback_regularizer_used
threshold
```

---

## 18. R4：ablation + failure pack

只在 seed42 主结果正向后做。

```bash
CUDA_VISIBLE_DEVICES=1 \
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_aaai_seed42_ablation.sh
```

Failure pack：

```bash
python -m tools.official.build_cga_v2_failure_pack \
  --dataset_dir /home/ly/AAAI/OHCM-MSHNet-main/datasets \
  --dataset_name NUDT-SIRST \
  --baseline_pred results/official/MSHNetOHEM/seed42/NUDT-SIRST/eval/predictions \
  --cga_pred results/official/MSHNetCGA/seed42/NUDT-SIRST/eval/predictions \
  --output_dir docs/paper/cga_v2_aaai/failure_pack \
  --top_k 20
```

---

# Part III：论文写法

---

## 19. 主表写法

主表标题：

```text
Controlled paired comparison on NUDT-SIRST with fixed threshold 0.5.
```

表中必须加：

```text
All results are regenerated in the current repository under a frozen controlled protocol. Historical OHCM-MSHNet artifacts are not reused as paper evidence.
```

---

## 20. ablation 写法

推荐：

```text
We ablate the auxiliary geometry components by zeroing their loss weights. Because all heads share the same decoder feature, the ablation measures mixed regularization effects rather than independent causal necessity of each component.
```

---

## 21. limitation 写法

推荐：

```text
CGA is currently validated as a training-time regularization strategy for MSHNet-style IRSTD under a controlled paired protocol. Extending the explicit adapter contract to additional host detectors such as DNANet, ALCNet, or ACM is left as future work unless all source-audited adapters and paired results are completed before submission.
```

---

## 22. Go / No-Go 决策

```text
GO for AAAI narrow claim if:
  seed42 passes gate
  seed43/44 do not reverse the trend
  paper_evidence_allowed=true for all main runs
  fallback_regularizer_used=false for all main runs
  ablation/failure pack supports target-preservation narrative

NO-GO for AAAI main if:
  seed42 fails mIoU/Precision/FA gate
  three seeds unstable or reversed
  paper evidence metadata missing
  results rely on historical artifacts
  CGA only improves one metric while FA rises materially
```

---

## 23. 最终推荐执行顺序

```text
1. 修正 cga_wrapper.py metadata ownership
2. 修正 train.py P1/P1A paper evidence gate + legacy_model_factory CLI
3. 固定 eval threshold=0.5
4. 添加 paired seed runner 和 manifest checker
5. R0 contract + grep guard
6. R1 P1/P1A preflight confirmation
7. R2 seed42 paired run
8. seed42 pass 后跑 seed43/44
9. 做 ablation + failure pack
10. 根据三种子结果决定 AAAI main / workshop / journal extension
```

---

## 24. 一句话结论

当前最优路线不是继续扩成 multi-backbone，而是先把 CGA-v2 写成：

```text
MSHNet-style target-preserving component-geometry regularization
```

并用当前仓库重新生成的 frozen controlled paired evidence 证明：

```text
inference path unchanged
center/boundary/scale/peak four-head regularization
fixed threshold=0.5
three-seed paired improvement
hard-clutter / target-preservation diagnostics
```

这条路线比提前写 plug-and-play / universal 更稳。
