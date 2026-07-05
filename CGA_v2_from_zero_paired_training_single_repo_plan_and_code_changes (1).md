# CGA-v2 从零 Paired 训练方案与代码修改（单仓库路径版）

## 0. 结论修正

上一版里“**不意外重训 OHEM，只要求已有 OHEM seed42 summaries**”这句话不适用于当前情况。

如果 `/home/ly/AAAI/CGA-main` 里没有正式数据、没有 `results/official/MSHNetOHEM/seed42/NUDT-SIRST/...`、没有可审计的 OHEM epoch400 summary，那么必须从零开始做 **paired evidence**：

```text
seed42: MSHNetOHEM epoch400 train -> test/hcval eval
seed42: MSHNetCGA  epoch400 train -> test/hcval eval
then:   paired delta = CGA - OHEM
```

不能把“没有结果”当成“已有 OHEM baseline”。

当前优先级应改成：

```text
1. 确认数据集真的在 /home/ly/AAAI/CGA-main/datasets
2. P1 dataset preflight pass
3. P1A HC-Val frozen source audit pass
4. 从零训练 MSHNetOHEM seed42 epoch400
5. 从零训练 MSHNetCGA seed42 epoch400
6. 用同一 test/hcval split、threshold=0.5 评估两边
7. 计算 paired delta
8. seed42 通过 gate 后，才跑 seed43/44
```

---

## 1. 单仓库路径 contract

所有路径统一在：

```text
/home/ly/AAAI/CGA-main
```

不要再使用：

```text
/home/ly/AAAI/OHCM-MSHNet-main/datasets
/home/AAAI/CGA-main
```

固定路径：

```text
ROOT=/home/ly/AAAI/CGA-main
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets
DATASET_NAME=NUDT-SIRST
RESULT_DIR=/home/ly/AAAI/CGA-main/results/official_from_zero
```

期望数据结构：

```text
/home/ly/AAAI/CGA-main/datasets/NUDT-SIRST/
  images/
  masks/
  img_idx/
    train_NUDT-SIRST.txt
    test_NUDT-SIRST.txt
    hcval_NUDT-SIRST.txt
```

如果 `datasets/NUDT-SIRST` 不存在，不能训练。

如果 `hcval_NUDT-SIRST.txt` 不存在，不能把结果标成 paper evidence。

---

## 2. 当前代码判断

当前代码方向基本正确：

```text
model/cga_wrapper.py:
  使用真实四头 CGA: center / boundary / scale / peak
  不包含 fallback regularizer
  不应声明 paper_evidence_allowed=True

train.py:
  paper_evidence_allowed 应由 evidence_mode + P1 + P1A + fallback 决定

scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh:
  逻辑上会跑 MSHNetOHEM 和 MSHNetCGA 两边
  如果 checkpoint 不存在，或者 FORCE_TRAIN=1，会训练对应模型
```

但是，当前主线不能再假设：

```text
MSHNetOHEM seed42 summary 已经存在
```

除非你能实际看到这些文件：

```text
/home/ly/AAAI/CGA-main/results/official/MSHNetOHEM/seed42/NUDT-SIRST/test/summary_metrics.json
/home/ly/AAAI/CGA-main/results/official/MSHNetOHEM/seed42/NUDT-SIRST/hcval/summary_metrics.json
/home/ly/AAAI/CGA-main/results/official/MSHNetOHEM/seed42/NUDT-SIRST/MSHNetOHEM_400.pth.tar
```

如果这些不存在，就必须从零训练 OHEM。

---

## 3. 训练前数据检查

先不要跑模型。先检查数据集是否真的存在：

```bash
cd /home/ly/AAAI/CGA-main

find /home/ly/AAAI/CGA-main/datasets -maxdepth 3 -type f | head -50

ls -lah /home/ly/AAAI/CGA-main/datasets/NUDT-SIRST
ls -lah /home/ly/AAAI/CGA-main/datasets/NUDT-SIRST/img_idx
```

必须至少看到：

```text
train_NUDT-SIRST.txt
test_NUDT-SIRST.txt
hcval_NUDT-SIRST.txt
```

然后跑 P1 preflight：

```bash
cd /home/ly/AAAI/CGA-main

DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
bash scripts/official/run_cga_v2_dataset_preflight.sh
```

检查输出：

```bash
cat /home/ly/AAAI/CGA-main/docs/internal/cga_v2/dataset_preflight/NUDT-SIRST/summary.json
```

必须满足：

```json
{
  "gate_pass": true
}
```

如果 P1 不通过，停止。

如果 P1A HC-Val source audit summary 已经存在，也要检查：

```bash
cat /home/ly/AAAI/CGA-main/docs/internal/cga_v2/dataset_preflight/NUDT-SIRST/hcval_source_summary.json
```

必须满足：

```json
{
  "gate_pass": true
}
```

如果 P1A 不通过，不能把训练结果标成 paper evidence。

---

## 4. 推荐执行方式：从零 paired seed42

推荐使用新的 fresh runner，避免误用旧 checkpoint 或旧 summary。

新增文件：

```text
scripts/official/run_cga_v2_seed42_from_zero_paired_single_repo.sh
```

内容建议如下：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"

PYTHON=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

DATASET_DIR=${DATASET_DIR:-/home/ly/AAAI/CGA-main/datasets}
DATASET_NAME=${DATASET_NAME:-NUDT-SIRST}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-400}
EPOCH=${EPOCH:-${EPOCHS}}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ly/AAAI/CGA-main/results/official_from_zero}
CUDA_DEVICE=${CUDA_DEVICE:-1}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

PREFLIGHT_SUMMARY="docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/summary.json"
P1A_SUMMARY="docs/internal/cga_v2/dataset_preflight/${DATASET_NAME}/hcval_source_summary.json"
P2_OUTPUT="docs/internal/cga_v2/gate_p2_from_zero_seed${SEED}_${DATASET_NAME}/summary.json"

mkdir -p "$(dirname "${P2_OUTPUT}")"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing required file: ${path}" >&2
    exit 2
  fi
}

check_json_gate_pass() {
  local path="$1"
  "${PYTHON}" - "$path" <<'PY'
import json, sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    obj = json.load(f)
if obj.get('gate_pass') is not True:
    raise SystemExit(f"gate_pass is not true in {p}: {obj.get('gate_pass')!r}")
PY
}

# 1. Dataset files must exist.
require_file "${DATASET_DIR}/${DATASET_NAME}/img_idx/train_${DATASET_NAME}.txt"
require_file "${DATASET_DIR}/${DATASET_NAME}/img_idx/test_${DATASET_NAME}.txt"
require_file "${DATASET_DIR}/${DATASET_NAME}/img_idx/hcval_${DATASET_NAME}.txt"

# 2. Run P1 preflight.
DATASET_DIR="${DATASET_DIR}" \
DATASET_NAME="${DATASET_NAME}" \
OUTPUT="${PREFLIGHT_SUMMARY}" \
bash scripts/official/run_cga_v2_dataset_preflight.sh

check_json_gate_pass "${PREFLIGHT_SUMMARY}"

# 3. P1A must already exist and pass for paper evidence.
require_file "${P1A_SUMMARY}"
check_json_gate_pass "${P1A_SUMMARY}"

# 4. Use a fresh output root by default.
mkdir -p "${OUTPUT_DIR}"

run_train() {
  local model_name="$1"
  echo "[RUN] Train ${model_name}, seed=${SEED}, epochs=${EPOCHS}"
  "${PYTHON}" train.py \
    --model_name "${model_name}" \
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
    --cga_start_epoch 1 \
    --cga_ramp_epochs 40 \
    --p1_preflight_passed \
    --p1a_hcval_source_audit_passed \
    --output_dir "${OUTPUT_DIR}"
}

run_eval() {
  local model_name="$1"
  local split="$2"
  local ckpt="${OUTPUT_DIR}/${model_name}/seed${SEED}/${DATASET_NAME}/${model_name}_${EPOCH}.pth.tar"
  require_file "${ckpt}"
  echo "[RUN] Eval ${model_name}, split=${split}, checkpoint=${ckpt}"
  MODEL_NAME="${model_name}" \
  DATASET_DIR="${DATASET_DIR}" \
  DATASET_NAME="${DATASET_NAME}" \
  SEED="${SEED}" \
  EPOCH="${EPOCH}" \
  CHECKPOINT="${ckpt}" \
  SPLIT="${split}" \
  bash scripts/official/run_cga_v2_test_seed.sh --output_dir "${OUTPUT_DIR}"
}

# 5. From-zero paired training.
run_train MSHNetOHEM
run_train MSHNetCGA

# 6. Evaluate both on Full/test and HC-Val.
run_eval MSHNetOHEM test
run_eval MSHNetOHEM hcval
run_eval MSHNetCGA test
run_eval MSHNetCGA hcval

BASE_FULL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
CAND_FULL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/test/summary_metrics.json"
BASE_HCVAL="${OUTPUT_DIR}/MSHNetOHEM/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"
CAND_HCVAL="${OUTPUT_DIR}/MSHNetCGA/seed${SEED}/${DATASET_NAME}/hcval/summary_metrics.json"

require_file "${BASE_FULL}"
require_file "${CAND_FULL}"
require_file "${BASE_HCVAL}"
require_file "${CAND_HCVAL}"

# 7. Summarize paired delta.
"${PYTHON}" -m tools.official.summarize_cga_v2_one_seed \
  --dataset_name "${DATASET_NAME}" \
  --seed "${SEED}" \
  --epoch "${EPOCH}" \
  --threshold 0.5 \
  --baseline MSHNetOHEM \
  --candidate MSHNetCGA \
  --preflight_summary "${PREFLIGHT_SUMMARY}" \
  --baseline_full "${BASE_FULL}" \
  --candidate_full "${CAND_FULL}" \
  --baseline_hcval "${BASE_HCVAL}" \
  --candidate_hcval "${CAND_HCVAL}" \
  --output "${P2_OUTPUT}"

echo "[DONE] Paired from-zero seed${SEED} summary: ${P2_OUTPUT}"
```

赋权：

```bash
chmod +x /home/ly/AAAI/CGA-main/scripts/official/run_cga_v2_seed42_from_zero_paired_single_repo.sh
```

运行：

```bash
cd /home/ly/AAAI/CGA-main

CUDA_DEVICE=1 \
ROOT=/home/ly/AAAI/CGA-main \
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
EPOCHS=400 \
OUTPUT_DIR=/home/ly/AAAI/CGA-main/results/official_from_zero \
bash scripts/official/run_cga_v2_seed42_from_zero_paired_single_repo.sh
```

---

## 5. 如果继续使用现有 paired runner，需要这样理解

当前已有脚本：

```text
scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
```

它的行为是：

```text
run_train_eval MSHNetOHEM
run_train_eval MSHNetCGA
```

并且当 checkpoint 不存在，或者 `FORCE_TRAIN=1` 时，会训练对应模型。

所以如果你想从零跑，也可以用：

```bash
cd /home/ly/AAAI/CGA-main

CUDA_VISIBLE_DEVICES=1 \
ROOT=/home/ly/AAAI/CGA-main \
DATASET_DIR=/home/ly/AAAI/CGA-main/datasets \
DATASET_NAME=NUDT-SIRST \
SEED=42 \
EPOCHS=400 \
EPOCH=400 \
OUTPUT_DIR=/home/ly/AAAI/CGA-main/results/official_from_zero \
FORCE_TRAIN=1 \
bash scripts/official/run_cga_v2_dataset_one_seed_paired_train_eval.sh
```

但是，建议检查这个脚本是否把 P1/P1A flags 传进了 `train.py`。

如果没有传，需要修改训练调用，把这些参数加上：

```bash
--evidence_mode paper \
--protocol controlled \
--p1_preflight_passed \
--p1a_hcval_source_audit_passed
```

否则训练可以跑，但 `paper_evidence_allowed` 可能会是 `false`，不能直接作为 paper evidence。

---

## 6. 最小代码修改清单

### 6.1 必改：新增 from-zero paired runner

新增：

```text
scripts/official/run_cga_v2_seed42_from_zero_paired_single_repo.sh
```

目的：

```text
明确从零训练 OHEM + CGA，避免误以为已有 OHEM summary。
```

### 6.2 建议改：修正旧计划文档

把所有这种说法删除：

```text
only require existing MSHNetOHEM seed42 summaries
```

改成：

```text
If no current-repo OHEM seed42 epoch400 checkpoint and summaries exist, train MSHNetOHEM from scratch under the same controlled protocol before training/comparing MSHNetCGA.
```

### 6.3 建议改：paired runner 传入 P1/P1A flags

如果现有 `run_cga_v2_dataset_one_seed_paired_train_eval.sh` 调用：

```bash
bash scripts/official/run_cga_v2_train_seed.sh --output_dir "${OUTPUT_DIR}"
```

建议改成：

```bash
bash scripts/official/run_cga_v2_train_seed.sh \
  --evidence_mode paper \
  --protocol controlled \
  --p1_preflight_passed \
  --p1a_hcval_source_audit_passed \
  --output_dir "${OUTPUT_DIR}"
```

前提是 P1/P1A summary 已经 pass。

更安全的版本是：先用 Python 检查 P1/P1A summary 的 `gate_pass=true`，再传这两个 flags。

### 6.4 不急着改

暂时不要改：

```text
model/cga_wrapper.py
model/cga_aux.py
utils/cga_targets.py
model/registry.py
DNANet / ALCNet / ACM adapter
ablation automation
failure-pack automation
```

现在主 blocker 是结果证据，不是模型结构。

---

## 7. seed42 gate

从零跑完后，检查：

```text
/home/ly/AAAI/CGA-main/docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/summary.json
```

核心 gate：

```text
Full mIoU      >= OHEM + 0.020
Full Precision >= OHEM + 0.010
Full FA        <= OHEM FA
```

如果 seed42 不过：

```text
停止 AAAI main route。
不要跑 seed43/44。
不要做 ablation/failure pack 叙事。
```

如果 seed42 通过：

```text
继续 seed43/44 paired evidence。
```

---

## 8. seed43/44 也必须 paired

三种子证据必须是：

```text
seed42: MSHNetOHEM vs MSHNetCGA
seed43: MSHNetOHEM vs MSHNetCGA
seed44: MSHNetOHEM vs MSHNetCGA
```

不能做：

```text
MSHNetCGA seed42/43/44 vs MSHNetOHEM seed42 only
```

如果 seed43/44 没有 OHEM，就也必须从零训练 OHEM。

---

## 9. 论文 claim 边界

在从零 paired seed42 完成前，不能写：

```text
CGA improves over OHEM under current-repo paper evidence.
```

seed42 通过但 seed43/44 未完成时，只能写：

```text
Preliminary paired seed42 evidence suggests a positive trend.
```

三种子都稳定后，才能写：

```text
CGA is a training-time component-geometry regularizer for MSHNet-style IRSTD that preserves the inference path and improves target preservation under a frozen controlled paired protocol.
```

仍然不要写：

```text
universal plug-and-play
multi-backbone SOTA
CGA solves hard clutter
each CGA head is independently necessary
```

---

## 10. 最终执行顺序

```text
R0. 不再改模型。
R1. 确认 /home/ly/AAAI/CGA-main/datasets/NUDT-SIRST 存在。
R2. 确认 train/test/hcval list 存在。
R3. 跑 P1 dataset preflight。
R4. 确认 P1A HC-Val source audit pass。
R5. 从零训练 MSHNetOHEM seed42 epoch400。
R6. 从零训练 MSHNetCGA seed42 epoch400。
R7. 同一 test/hcval split、threshold=0.5 评估两边。
R8. 计算 paired delta。
R9. seed42 不过 gate，停止 AAAI main route。
R10. seed42 通过，再跑 seed43/44 paired。
R11. 三种子稳定后，再做 ablation / failure pack / release metadata polish。
```

---

## 11. 一句话结论

现在应该改成：

```text
因为当前仓库没有数据和可复用结果，所以必须从零做 MSHNetOHEM vs MSHNetCGA paired seed42。只有 paired seed42 通过后，才考虑 seed43/44、ablation 和 failure-pack。
```
