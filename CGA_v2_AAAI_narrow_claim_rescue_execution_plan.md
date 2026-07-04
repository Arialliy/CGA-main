# CGA-v2 AAAI Narrow-Claim Rescue Execution Plan

## 0. Verdict

按 AAAI 主会投稿标准，当前 CGA-v2 不能直接按“新 SOTA 模型 + 多 benchmark validation”来投。正确路线是：

```text
revise / narrow claim / evidence-first
```

推荐论文定位：

```text
CGA-v2: target-preserving component-geometry regularization for MSHNet-style IRSTD
```

当前不继续改模型结构；先把新 overlay 仓库变成一个可复现实验仓库。由于新文件夹里没有实验产物，第一优先级是重新生成 NUDT-SIRST 主数据集证据。

---

## 1. 当前保留主张

可以写：

```text
CGA-v2 是一种训练期 component-geometry regularization。
它保持 MSHNet final mask inference path 不变。
它在 NUDT-SIRST fixed-seed paired protocol 上相对 MSHNetOHEM 提升 Full 指标和 mean HC-Val mIoU / Pd。
机制消融显示 mixed attribution，而不是 strict per-head necessity。
```

不能写：

```text
CGA-v2 是新的 SOTA IRSTD 模型。
CGA-v2 完全解决 hard clutter。
CGA-v2 已在 NUAA-SIRST / IRSTD-1K 上完成 multi-benchmark validation。
CGA-v2 是 transfer learning。
所有 auxiliary heads 都被证明必要。
每个 seed 都降低 HC-Val FA。
```

---

## 2. 现有证据边界

### 2.1 NUDT-SIRST 主证据

历史 frozen protocol：

```text
Candidate: MSHNetCGA / CGA-v2 base
Baseline: MSHNetOHEM
Dataset: NUDT-SIRST
Seeds: 42 / 43 / 44
Checkpoint: epoch400
Threshold: 0.5
```

历史主结果：

```text
Full mean delta vs MSHNetOHEM:
  mIoU      +0.04027
  Precision +0.02859
  Pd        +0.00282
  FA_ppm    -19.26499

HC-Val mean delta vs MSHNetOHEM:
  mIoU      +0.06759
  Precision +0.05519
  Pd        +0.05556
  FA_ppm    -100.02984
```

但新 overlay 文件夹没有这些实验产物，不能直接当作当前仓库结果。必须先在新仓库中生成自己的 NUDT artifacts。

### 2.2 IRSTD-1K

只能写成：

```text
single-dataset supervised train/test characterization / diagnostic evidence
```

不能写成 validation。原因是 seed42 aggregate 指标为正，但 target-level audit 发现 CGA 漏掉 2 个 OHEM 已检出的目标，并且都是 complete miss。

### 2.3 NUAA-SIRST

NUAA 目前不能进入主证据。数据完整性 gate 被 `Misc_111` image/mask size mismatch 阻断。除非找到官方正确版本并通过 canonical preflight，否则不要跑 NUAA 训练/测试。

---

## 3. 最近 prior-art 威胁

AAAI 写作必须正面处理 closest prior-art：

```text
MSHNet / SLS:
  已经覆盖 scale/location-sensitive loss 和 MSHNet 框架。
  CGA-v2 的 scale/center target 必须解释为 component-level regularization，而不是重复 SLS。

Shape / edge / boundary methods:
  已有方法使用 interior/boundary 或 shape-aware supervision。
  CGA-v2 的 boundary/peak/scale 不能被写成完全新观察。

PConv + SD Loss:
  已经覆盖 spatial distribution / scale-aware dynamic loss / stronger benchmark package。
  CGA-v2 不能只靠“加 auxiliary heads”作为 novelty。
```

因此 contribution 应收缩为：

```text
training-time component geometry regularization,
no inference-path change,
fixed-seed paired improvement on MSHNetOHEM anchor,
with honest mixed-mechanism evidence.
```

---

## 4. 执行路线总览

```text
P0: closest prior-art threat table
P1: 新 overlay 仓库 NUDT-SIRST seed42 复现
P2: 新 overlay 仓库 NUDT-SIRST seed43/44 多 seed 复现
P3: mixed ablation + failure analysis pack
P4: IRSTD-1K supervised characterization 改成 diagnostic-only
P5: closest baseline comparison pack
P6: claim linter + manuscript package
```

注意：P1/P2 必须优先于 P3/P4/P5，因为新文件夹没有实验数据。

---

## 5. P0：closest prior-art threat table

### 5.1 新增文件

```text
configs/closest_baselines.yaml

docs/paper/cga_v2_aaai/07_related_work_threats.md

tools/official/write_cga_v2_related_work_threat_table.py

tests/test_cga_v2_related_work_threat_table.py
```

### 5.2 `configs/closest_baselines.yaml` 示例

```yaml
closest_baselines:
  - name: MSHNet / SLS
    type: direct_baseline_and_novelty_threat
    relation: scale/location-sensitive loss and MSHNet framework
    fair_comparison_status: local_paired_baseline_available_as_MSHNetOHEM
    claim_guidance: "CGA-v2 must be framed as component-geometry regularization, not scale/location loss replacement."

  - name: shape_edge_boundary_prior_work
    type: mechanism_threat
    relation: shape / boundary / edge supervision in IRSTD
    fair_comparison_status: literature_only_unless_code_available
    claim_guidance: "Do not claim boundary/shape supervision is new by itself."

  - name: PConv + SD Loss
    type: recent_AAAI_scale_aware_package
    relation: spatial distribution convolution + scale-based dynamic loss + benchmark package
    fair_comparison_status: literature_only_initially
    claim_guidance: "Do not claim SOTA unless same-protocol reproduction exists."
```

### 5.3 输出

```text
docs/paper/cga_v2_aaai/07_related_work_threats.md
```

内容必须分成：

```text
fair paired comparison:
  MSHNetOHEM vs MSHNetCGA

literature-only threat:
  MSHNet/SLS
  ISNet/iSmallNet/shape-boundary methods
  PConv + SD Loss
  other SOTA
```

不要把 literature-only 数字写成 fair SOTA comparison。

---

## 6. P1：新 overlay 仓库 NUDT seed42 复现

### 6.1 目标

先证明新 repo overlay 不是只有代码，而能生成自己的主结果：

```text
MSHNetOHEM seed42 train epoch400 + Full / HC-Val eval
MSHNetCGA  seed42 train epoch400 + Full / HC-Val eval
```

固定：

```text
DATASET_NAME=NUDT-SIRST
SEED=42
checkpoint=epoch400
threshold=0.5
```

### 6.2 新增文件

```text
docs/internal/cga_v2/nudt_reproduction/nudt_reproduction_plan.json

tools/official/check_cga_v2_overlay_bootstrap.py
tools/official/check_cga_v2_nudt_dataset_preflight.py
tools/official/check_cga_v2_nudt_seed42_reproduction.py
tools/official/write_cga_v2_nudt_seed42_reproduction_report.py

scripts/official/run_cga_v2_overlay_bootstrap.sh
scripts/official/run_cga_v2_nudt_dataset_preflight.sh
scripts/official/run_cga_v2_nudt_seed42_train_eval.sh
scripts/official/run_cga_v2_nudt_one_model_seed.sh

tests/test_cga_v2_overlay_bootstrap.py
tests/test_cga_v2_nudt_dataset_preflight.py
tests/test_cga_v2_nudt_seed42_reproduction.py
```

### 6.3 输出路径

```text
results/official/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar
results/official/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar

docs/internal/cga_v2/nudt_reproduction/seed42/eval_full_ohem/summary_metrics.json
docs/internal/cga_v2/nudt_reproduction/seed42/eval_full_cga/summary_metrics.json
docs/internal/cga_v2/nudt_reproduction/seed42/eval_hcval_ohem/summary_metrics.json
docs/internal/cga_v2/nudt_reproduction/seed42/eval_hcval_cga/summary_metrics.json
```

### 6.4 seed42 gate

先用 sanity gate，不要求完全复现历史数字：

```text
Full:
  delta_mIoU      >= +0.020
  delta_Precision >= +0.010
  delta_Pd        >= -0.001
  delta_FA_ppm    <= 0.0

HC-Val:
  delta_mIoU      >= 0.0
  delta_Pd        >= -0.001
```

如果失败，先查：

```text
dataset root
model registry
loss config
seed handling
summary identity
checkpoint epoch
threshold
```

不要直接改模型。

### 6.5 命令

```bash
cd /home/ly/AAAI/OHCM-MSHNet-cga-v2-paper

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

bash scripts/official/run_cga_v2_overlay_bootstrap.sh

DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_nudt_dataset_preflight.sh

DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
bash scripts/official/run_cga_v2_nudt_seed42_train_eval.sh
```

---

## 7. P2：NUDT seed43/44 多 seed 复现

只有 seed42 通过后才跑 seed43/44。

### 7.1 新增文件

```text
scripts/official/run_cga_v2_nudt_multiseed_train_eval.sh

tools/official/check_cga_v2_nudt_multiseed_reproduction.py
tools/official/write_cga_v2_nudt_multiseed_table.py

tests/test_cga_v2_nudt_multiseed_reproduction.py
```

### 7.2 Gate

```text
Full:
  3/3 mIoU non-regression
  3/3 Precision non-regression
  3/3 Pd non-regression >= -0.001
  mean mIoU > 0
  mean Precision > 0

HC-Val:
  at least 2/3 seeds mIoU positive
  mean mIoU > 0
  mean Pd >= 0
```

不要强行要求每 seed HC-Val FA 都下降，因为历史 seed44 HC-Val 已有 Precision / FA tradeoff。

### 7.3 输出

```text
docs/internal/cga_v2/nudt_reproduction/multiseed/summary.json
docs/paper/cga_v2_aaai/03_experiment_tables.md
```

---

## 8. P3：mixed ablation + failure analysis pack

### 8.1 必须保留的解释

```text
Gate-E rejected strict auxiliary-mechanism claim.
Gate-E0 confirmed ablation controls are interpretable.
aux_off is CGA_ARCH_ONLY_CONTROL, not OHEM-equivalent.
No ablation winner promotion.
```

### 8.2 新增/更新工具

```text
tools/official/write_cga_v2_ablation_table.py
tools/official/write_cga_v2_failure_case_pack.py
tools/official/check_cga_v2_claim_linter.py

scripts/official/run_cga_v2_ablation_failure_pack.sh

tests/test_cga_v2_ablation_table.py
tests/test_cga_v2_failure_case_pack.py
tests/test_cga_v2_claim_linter.py
```

### 8.3 failure case 类型

```text
Positive NUDT examples:
  CGA improves compactness / target structure / false positives.

Tradeoff examples:
  seed44 HC-Val Precision and FA regression.

IRSTD diagnostic examples:
  XDU788 target_id=4 complete_miss
  XDU850 target_id=0 complete_miss
```

IRSTD 只进 diagnostic section，不进 validation table。

---

## 9. P4：IRSTD-1K supervised characterization 重写

建议写成：

```text
IRSTD-1K single-dataset supervised train/test characterization
```

不要写成：

```text
transfer learning
external validation
supplemental validation evidence
```

应保留：

```text
aggregate metrics positive
but target-level audit found 2 complete misses
retain as diagnostic_evidence_only
```

新增/更新：

```text
docs/paper/cga_v2_aaai/07_irstd_single_benchmark_diagnostic.md

tools/official/write_cga_v2_irstd_supervised_characterization_section.py

tests/test_cga_v2_irstd_supervised_characterization_section.py
```

---

## 10. P5：closest baseline comparison pack

### 10.1 两层比较

```text
Layer 1: Fair local paired comparison
  MSHNetOHEM vs MSHNetCGA
  same dataset, seed, threshold, epoch

Layer 2: Literature-only comparison
  MSHNet/SLS
  ISNet / iSmallNet / boundary-shape methods
  PConv + SD Loss
  other IRSTD SOTA
```

### 10.2 如果要做 SOTA 对比

只有在同协议、同数据、同 threshold、同 metrics 下复现，才可称为 fair comparison。否则只能写：

```text
Closest-work positioning and novelty threat analysis
```

---

## 11. 是否还需要改模型？

当前不改。

如果新 overlay NUDT 不能复现，先查 implementation/config，不改结构。

只有以下情况才考虑新分支：

```text
new branch: cga_v3_after_submission
```

CGA-v3 可以考虑：

```text
target-level preservation loss
small-target recall guard
component confidence calibration
```

但不要污染当前 CGA-v2 投稿分支。

---

## 12. 最终 AAAI 提交判断

如果完成：

```text
P0 related-work threat table
P1/P2 新 overlay NUDT 三 seed复现
P3 mixed ablation + failure analysis
P4 IRSTD diagnostic-only rewrite
P5 closest baseline positioning
P6 claim linter
```

则可以按保守论文投：

```text
Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection
```

如果不能完成 NUDT 新仓库复现，不建议投 AAAI 主会。

如果完成 NUDT 但没有 closest baseline positioning，仍然高风险，会被认为是 incremental MSHNet variant with limited generalization evidence。

---

## 13. 最小执行顺序

```bash
# P0
bash scripts/official/run_cga_v2_related_work_threat_table.sh

# P1
bash scripts/official/run_cga_v2_overlay_bootstrap.sh
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets DATASET_NAME=NUDT-SIRST \
  bash scripts/official/run_cga_v2_nudt_dataset_preflight.sh
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets DATASET_NAME=NUDT-SIRST SEED=42 \
  bash scripts/official/run_cga_v2_nudt_seed42_train_eval.sh

# P2
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets DATASET_NAME=NUDT-SIRST \
  bash scripts/official/run_cga_v2_nudt_multiseed_train_eval.sh

# P3-P5
bash scripts/official/run_cga_v2_ablation_failure_pack.sh
bash scripts/official/run_cga_v2_paper_evidence_pack.sh
```

## 14. One-line rule

```text
先把新 overlay 仓库的 NUDT 主结果跑出来；
再把 claim、ablation、failure、closest baseline 写准；
不要现在继续换模型或扩展不干净数据集。
```
