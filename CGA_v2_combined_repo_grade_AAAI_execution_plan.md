# CGA-v2 Combined AAAI Execution Plan: Repo-Grade Model Package + Narrow-Claim Evidence Route

## 0. 一句话结论

当前 CGA-v2 不应继续包装成“新 SOTA IRSTD 大模型”，也不应继续随意改结构、loss、threshold 或 checkpoint。正确路线是：

```text
revise / narrow claim / evidence-first
```

论文安全定位固定为：

```text
CGA-v2: target-preserving component-geometry regularization for MSHNet-style IRSTD
```

当前最重要的事情不是继续发明模块，而是把新 overlay 仓库整理成一个可复现、可审计、论文级的仓库：模型、loss、train/test/evaluate、metrics、official scripts、tests、docs、paper tables 全部闭环。由于新文件夹里没有实验产物，第一优先级是重新生成 NUDT-SIRST 主数据集证据。

---

## 1. 当前目标重新冻结

### 1.1 允许的 claim

可以写：

```text
CGA-v2 是一种训练期 component-geometry regularization。
CGA-v2 保持 MSHNet final mask inference path 不变。
CGA-v2 在 NUDT-SIRST fixed-seed paired protocol 上相对 MSHNetOHEM 提升 Full 指标和 mean HC-Val mIoU / Pd。
机制消融显示 mixed attribution，而不是 strict per-head necessity。
```

### 1.2 禁止的 claim

不能写：

```text
CGA-v2 是新的 SOTA IRSTD 模型。
CGA-v2 完全解决 hard clutter。
CGA-v2 已在 NUAA-SIRST / IRSTD-1K 上完成 multi-benchmark validation。
CGA-v2 是 transfer learning。
所有 auxiliary heads 都被证明必要。
每个 seed 都降低 HC-Val FA。
```

### 1.3 当前路线的核心原则

```text
继续润色：是。
润色对象：repo-grade paper model package + evidence package。
不做：architecture / loss / threshold / checkpoint / seed tuning。
```

换句话说，CGA-v2 的优化不是再加一个新 head，而是把它打磨成一套审稿人能复现、能检查、能理解 claim 边界的完整代码与实验包。

---

## 2. 为什么不能继续随意改模型

当前 CGA-v2 已经有一个清晰的可写方法形态：

```text
training:
  MSHNet segmentation evidence path
  + component geometry auxiliary supervision

inference:
  image -> final logit -> sigmoid -> threshold 0.5
```

如果现在继续改结构或 loss，会导致前面已经形成的 Gate-D / Gate-E / F0 / F1 证据链失效。更重要的是，已有机制消融不是 strict auxiliary necessity，而是 mixed attribution。因此现在继续做 architecture tuning，会把论文从“保守但干净”推回“实验搜索”。

允许修改的是工程级润色：

```text
- 文件结构清晰化
- 模型 contract 测试
- loss contract 测试
- eval/test 只用 final logit 的 contract
- summary identity 检查
- official runner/checker/test/doc 体系
- failure case / ablation / prior-art threat table
```

不允许修改的是结果导向调参：

```text
- 改 lambda 追 HC-Val
- 改 threshold
- 换 checkpoint
- 换 seed
- 重启 DPS / suppression loss
- promotion aux_off / no_boundary_scale
- 修改 split 或样本
```

---

## 3. 历史证据边界：只能作为目标，不是新 overlay 当前结果

历史 frozen protocol 是：

```text
Candidate: MSHNetCGA / CGA-v2 base
Baseline: MSHNetOHEM
Dataset: NUDT-SIRST
Seeds: 42 / 43 / 44
Checkpoint: epoch400
Threshold: 0.5
```

历史主结果是：

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

但新 overlay 仓库没有这些实验产物，因此不能直接把历史结果当作当前仓库结果。新仓库必须重新生成自己的 NUDT artifacts。

---

## 4. 仓库级代码结构要求

新 overlay 仓库应完整覆盖 OHCM-MSHNet 风格结构，而不是只给一个孤立 `model.py`。

### 4.1 必须存在的核心文件

```text
model/
  MSHNet.py
  CGA_MSHNet.py
  cga_aux.py

utils/
  cga_targets.py

loss.py
net.py
train.py
test.py
evaluate.py
metrics.py
```

### 4.2 必须存在的 official 工程文件

```text
tools/official/
  check_cga_v2_overlay_bootstrap.py
  check_cga_v2_model_contract.py
  check_cga_v2_loss_contract.py
  check_cga_v2_eval_contract.py
  check_cga_v2_nudt_dataset_preflight.py
  check_cga_v2_nudt_seed42_reproduction.py
  check_cga_v2_nudt_multiseed_reproduction.py
  write_cga_v2_related_work_threat_table.py
  write_cga_v2_nudt_seed42_reproduction_report.py
  write_cga_v2_nudt_multiseed_table.py
  write_cga_v2_ablation_table.py
  write_cga_v2_failure_case_pack.py
  check_cga_v2_closest_baseline_manifest.py
  check_cga_v2_claim_linter.py
```

### 4.3 必须存在的 scripts

```text
scripts/official/
  run_cga_v2_overlay_bootstrap.sh
  run_cga_v2_model_contract.sh
  run_cga_v2_nudt_dataset_preflight.sh
  run_cga_v2_nudt_one_model_seed.sh
  run_cga_v2_nudt_seed42_train_eval.sh
  run_cga_v2_nudt_multiseed_train_eval.sh
  run_cga_v2_ablation_failure_pack.sh
  run_cga_v2_paper_evidence_pack.sh
```

### 4.4 必须存在的 tests

```text
tests/
  test_cga_v2_overlay_bootstrap.py
  test_cga_v2_model_contract.py
  test_cga_v2_loss_contract.py
  test_cga_v2_eval_contract.py
  test_cga_v2_metrics_diagnostics.py
  test_cga_v2_nudt_dataset_preflight.py
  test_cga_v2_nudt_seed42_reproduction.py
  test_cga_v2_nudt_multiseed_reproduction.py
  test_cga_v2_related_work_threat_table.py
  test_cga_v2_ablation_table.py
  test_cga_v2_failure_case_pack.py
  test_cga_v2_claim_linter.py
```

### 4.5 必须存在的 docs

```text
docs/internal/cga_v2/
  repo_grade_model_contract_plan.json
  nudt_reproduction/nudt_reproduction_plan.json
  nudt_reproduction/seed42/...
  nudt_reproduction/multiseed/...

docs/paper/cga_v2_aaai/
  00_paper_outline.md
  01_title_abstract_contributions.md
  03_experiment_tables.md
  04_ablation_and_mechanism.md
  05_limitations_and_risk.md
  07_irstd_single_benchmark_diagnostic.md
  08_closest_baseline_threat_table.md
```

---

## 5. 模型代码 contract

### 5.1 `model/CGA_MSHNet.py`

职责：定义 MSHNetCGA 主模型。

必须满足：

```text
train mode:
  return final_logit, aux_outputs

eval mode:
  return final_logit only, or return object whose official eval path only uses final_logit
```

禁止：

```text
- test-time verifier
- test-time suppression
- post-processing
- checkpoint ensemble
- auxiliary heads 参与 inference final probability
```

### 5.2 `model/cga_aux.py`

职责：定义 auxiliary heads。

```text
CGAAuxHead:
  center_logit
  boundary_logit
  scale_logit
  peak_logit
```

要求：

```text
- input feature shape 和 decoder feature 匹配
- output spatial size 能对齐 GT mask
- 不包含 dataset-specific 逻辑
```

### 5.3 `utils/cga_targets.py`

职责：从 GT mask 生成辅助目标。

```text
center target
boundary target
scale target
peak target
valid mask
```

要求：

```text
- 只依赖 GT mask
- 不依赖预测结果
- 不依赖 validation/test split
- 不写入随机状态
```

### 5.4 `loss.py`

职责：组合 base loss 和 CGA auxiliary loss。

```text
L_total = L_MSHNetOHEM + lambda_center * L_center
                        + lambda_boundary * L_boundary
                        + lambda_scale * L_scale
                        + lambda_peak * L_peak
```

要求：

```text
- loss finite
- aux flags 可关闭，但不能 promotion ablation winner
- loss summary 写入各项 loss
```

### 5.5 `net.py`

职责：注册模型。

```text
MSHNetOHEM
MSHNetCGA
```

要求：

```text
- model name 到 class 的映射明确
- checkpoint identity 写入 summary
- eval path 不调用 aux outputs
```

### 5.6 `train.py`

职责：训练调度。

要求：

```text
- 固定 seed
- 固定 epoch400
- checkpoint path 标准化
- train summary 写 dataset/seed/model/epoch/threshold/config hash
- 不在 train.py 手写 CGA target 逻辑，只调用 loss / cga_targets
```

### 5.7 `test.py` / `evaluate.py`

职责：只做固定协议 evaluation。

要求：

```text
- threshold = 0.5
- checkpoint = epoch400
- eval summary 写 dataset / seed / model / checkpoint_epoch / threshold
- candidate 和 baseline summary identity 可被 checker 验证
```

### 5.8 `metrics.py`

除原指标外，增加诊断指标：

```text
connected component FP count
target-level detection audit
target area stratification
missed target / gained target audit
failure case selector
```

---

## 6. P0：closest prior-art threat table

### 6.1 目标

AAAI 审稿会追问 novelty，因此先建立 threat table，避免后面写作失控。

必须区分：

```text
fair paired comparison:
  MSHNetOHEM vs MSHNetCGA

literature-only threat:
  MSHNet/SLS
  ISNet / shape-boundary methods
  PConv + SD Loss
  other SOTA
```

不要把 literature-only 数字写成 fair SOTA comparison。

### 6.2 新增文件

```text
configs/closest_baselines.yaml

docs/paper/cga_v2_aaai/08_closest_baseline_threat_table.md

tools/official/write_cga_v2_related_work_threat_table.py

tests/test_cga_v2_related_work_threat_table.py
```

### 6.3 `configs/closest_baselines.yaml` 示例

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

---

## 7. P0.5：repo-grade model contract，不训练

### 7.1 目标

先证明 overlay 仓库代码自洽：能 import、能 forward、loss finite、eval path 不用 aux。

### 7.2 输出

```text
docs/internal/cga_v2/repo_grade_model_contract/summary.json
```

### 7.3 Gate

```text
gate_pass = true only if:
  py_compile pass
  model contract pass
  loss contract pass
  eval contract pass
  metrics diagnostics pass
  no training started
```

### 7.4 命令

```bash
cd /home/ly/AAAI/OHCM-MSHNet-cga-v2-paper

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON}" -m py_compile \
  model/CGA_MSHNet.py \
  model/cga_aux.py \
  utils/cga_targets.py \
  loss.py \
  net.py \
  train.py \
  test.py \
  evaluate.py \
  metrics.py

"${PYTHON}" -m pytest \
  tests/test_cga_v2_overlay_bootstrap.py \
  tests/test_cga_v2_model_contract.py \
  tests/test_cga_v2_loss_contract.py \
  tests/test_cga_v2_eval_contract.py \
  tests/test_cga_v2_metrics_diagnostics.py -q

git diff --check
```

---

## 8. P1：新 overlay 仓库 NUDT-SIRST seed42 复现

### 8.1 目标

先证明新 overlay 仓库不是只有代码，而能生成自己的主结果。

```text
MSHNetOHEM seed42 train epoch400 + Full / HC-Val eval
MSHNetCGA  seed42 train epoch400 + Full / HC-Val eval
```

固定：

```text
DATASET_NAME = NUDT-SIRST
SEED = 42
checkpoint = epoch400
threshold = 0.5
```

### 8.2 新增文件

```text
docs/internal/cga_v2/nudt_reproduction/nudt_reproduction_plan.json

tools/official/check_cga_v2_nudt_dataset_preflight.py
tools/official/check_cga_v2_nudt_seed42_reproduction.py
tools/official/write_cga_v2_nudt_seed42_reproduction_report.py

scripts/official/run_cga_v2_nudt_dataset_preflight.sh
scripts/official/run_cga_v2_nudt_one_model_seed.sh
scripts/official/run_cga_v2_nudt_seed42_train_eval.sh

tests/test_cga_v2_nudt_dataset_preflight.py
tests/test_cga_v2_nudt_seed42_reproduction.py
```

### 8.3 输出路径

```text
results/official/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar
results/official/MSHNetCGA/seed42/NUDT-SIRST/MSHNetCGA_400.pth.tar

docs/internal/cga_v2/nudt_reproduction/seed42/eval_full_ohem/summary_metrics.json
docs/internal/cga_v2/nudt_reproduction/seed42/eval_full_cga/summary_metrics.json
docs/internal/cga_v2/nudt_reproduction/seed42/eval_hcval_ohem/summary_metrics.json
docs/internal/cga_v2/nudt_reproduction/seed42/eval_hcval_cga/summary_metrics.json

docs/internal/cga_v2/nudt_reproduction/seed42/summary.json
```

### 8.4 seed42 sanity gate

不要求完全复现历史数字，但要求方向正确。

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

不要直接改模型结构。

### 8.5 命令

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

## 9. P2：NUDT-SIRST seed43/44 多 seed 复现

只有 seed42 sanity gate 通过后，才跑 seed43/44。

### 9.1 新增文件

```text
scripts/official/run_cga_v2_nudt_multiseed_train_eval.sh

tools/official/check_cga_v2_nudt_multiseed_reproduction.py
tools/official/write_cga_v2_nudt_multiseed_table.py

tests/test_cga_v2_nudt_multiseed_reproduction.py
```

### 9.2 Gate

```text
Full:
  all seeds delta_Pd >= -0.001
  all seeds delta_mIoU >= -0.001
  mean delta_mIoU > 0
  mean delta_Precision > 0

HC-Val:
  mean delta_mIoU >= 0
  mean delta_Pd >= -0.001
  at least 2/3 seeds delta_mIoU >= 0
```

不要求每个 seed 都降低 HC-Val FA，因为历史 seed44 HC-Val 已有 FA tradeoff。

### 9.3 输出

```text
docs/internal/cga_v2/nudt_reproduction/multiseed/summary.json
docs/paper/cga_v2_aaai/03_experiment_tables.md
```

### 9.4 命令

```bash
DATASET_DIR=/home/ly/AAAI/OHCM-MSHNet-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_nudt_multiseed_train_eval.sh
```

---

## 10. P3：mixed ablation + failure analysis pack

### 10.1 目标

把已有机制结论写诚实：

```text
full CGA positive on NUDT main performance
mechanism attribution mixed
aux_off = CGA_ARCH_ONLY_CONTROL, not OHEM-equivalent
no ablation winner promotion
```

### 10.2 新增/输出

```text
docs/paper/cga_v2_aaai/04_ablation_and_mechanism.md
docs/paper/cga_v2_aaai/05_limitations_and_risk.md
docs/paper/cga_v2_aaai/failure_cases/

tools/official/write_cga_v2_ablation_table.py
tools/official/write_cga_v2_failure_case_pack.py

tests/test_cga_v2_ablation_table.py
tests/test_cga_v2_failure_case_pack.py
```

### 10.3 Failure case 必须包括

```text
seed44 HC-Val Precision / FA regression caveat
IRSTD-1K XDU788 complete miss
IRSTD-1K XDU850 complete miss
positive examples where CGA fixes OHEM errors
FP component examples where CGA reduces clutter
```

---

## 11. P4：IRSTD-1K 只能写 supervised characterization / diagnostic evidence

当前 IRSTD-1K 不能写 validation。

原因：

```text
seed42 aggregate metrics positive
但 target-level audit 发现 CGA 漏掉 2 个 OHEM 已检出的目标
两个都是 complete miss
```

因此只能写：

```text
single-dataset supervised train/test characterization / diagnostic evidence
```

不能写：

```text
external validation
transfer learning
benchmark validation
multi-benchmark validation
```

相关输出应保留：

```text
docs/paper/cga_v2_aaai/07_irstd_single_benchmark_diagnostic.md
```

---

## 12. P5：closest baseline comparison pack

### 12.1 目标

AAAI 风险主要来自 novelty 和 closest baseline。因此必须把 baseline 风险写清楚。

### 12.2 分两类

```text
fair paired local comparison:
  MSHNetOHEM vs MSHNetCGA

literature-only closest work:
  MSHNet/SLS
  ISNet / shape / edge / boundary methods
  PConv + SD Loss
  DNANet / ACM / other IRSTD baselines
```

### 12.3 禁止

```text
不要把 literature-only 数字写成 fair SOTA comparison。
不要 claim CGA-v2 超过 PConv + SD Loss，除非同协议复现。
不要 claim boundary/shape supervision 是全新观察。
```

---

## 13. P6：claim linter + manuscript package

### 13.1 目标

机器检查论文 draft 是否越界。

### 13.2 必须拦截的短语

```text
state-of-the-art
fully solves hard clutter
external validation
blind validation
validated on NUAA-SIRST and IRSTD-1K
all auxiliary heads are necessary
complete hard-clutter closure
transfer learning
```

如果必须使用类似词，需要手动 waiver，并写明证据来源。

### 13.3 输出

```text
docs/paper/cga_v2_aaai/manuscript/main.tex
docs/paper/cga_v2_aaai/manuscript/sections/*.tex
docs/paper/cga_v2_aaai/manuscript/tables/*.tex
docs/internal/cga_v2/claim_linter/summary.json
```

---

## 14. 总执行顺序

```text
P0: related-work threat table
P0.5: repo-grade model contract
P1: NUDT seed42 reproduction
P2: NUDT seed43/44 multiseed reproduction
P3: ablation + failure pack
P4: IRSTD diagnostic-only section
P5: closest baseline comparison pack
P6: claim linter + manuscript package
```

P1/P2 必须优先于 P3/P4/P5，因为新文件夹没有实验数据。

---

## 15. 不同失败点的处理

### 15.1 P0.5 失败

```text
原因多半是代码接口、shape、import、loss contract 问题。
允许修代码。
不允许改模型思想。
```

### 15.2 P1 seed42 失败

先查：

```text
dataset path
model registry
loss config
seed
checkpoint epoch
threshold
summary identity
```

如果代码/配置正确但指标失败：

```text
STOP_CGA_V2_OVERLAY_AT_NUDT_SEED42_REPRODUCTION
```

不要调模型。

### 15.3 P2 multiseed 失败

如果 seed42 成功但 seed43/44 不稳：

```text
STOP_CGA_V2_AS_AAAI_MAIN_METHOD
保留 seed42 diagnostic，不写三种子主表。
```

### 15.4 P3/P4 失败

如果 ablation/failure pack 不能支持 claim：

```text
降级论文 claim。
不要 promotion ablation winner。
不要删除 failure case。
```

---

## 16. AAAI 论文写法

推荐题目：

```text
Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection
```

安全摘要核心：

```text
Infrared small target detection requires improving mask quality without suppressing weak target evidence. We introduce CGA-v2, a component-geometry regularization framework built on MSHNet, which preserves the final mask inference path while adding training-time geometric targets for component-level structure. Across fixed seeds on NUDT-SIRST, CGA-v2 improves paired Full metrics and mean hard-clutter validation mIoU/Pd over MSHNetOHEM. Ablation analysis indicates mixed attribution rather than strict necessity of every auxiliary target, so we position CGA-v2 as a target-preserving regularizer rather than a complete hard-clutter false-alarm closure mechanism.
```

---

## 17. 最终建议

```text
不要继续随意润色模型结构。
要继续润色仓库级模型包。
先在新 overlay 仓库复现 NUDT seed42。
seed42 过了再跑 seed43/44。
多 seed 主表复现后，再写 ablation、failure case、closest baseline 和 manuscript。
```

一句话：

> 当前最接近 AAAI 投稿的路线，不是再发明一个模型，而是把 MSHNetCGA / CGA-v2 base 打磨成可复现的 repo-grade paper model，并用 NUDT-SIRST 多 seed 主表、mixed ablation、failure case、closest prior-art threat table 构成一篇保守但完整的方法论文。
