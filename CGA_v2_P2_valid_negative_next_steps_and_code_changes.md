# CGA-v2 P2 Valid-Negative 后续方案与代码修改指南

**Canonical root**

```text
/home/ly/AAAI/CGA-main
```

本文档只使用上面的路径。历史结果 JSON 中出现的 `/home/AAAI/CGA-main/...` 只能作为 P2 audit 已审计过的历史运行路径，不应继续写入新的 runner、README 或 protocol。

---

## 0. 当前状态判定

### 0.1 已完成事实

P2 from-zero seed42 paired experiment 已完成：

```text
baseline  = MSHNetOHEM
candidate = MSHNetCGA
dataset   = NUDT-SIRST
seed      = 42
epoch     = 400
threshold = 0.5 fixed_predeclared
gate_pass = false
```

v5 implementation audit 已完成，用户态结论为：

```text
final_decision = P2_VALID_NEGATIVE_DESIGN_WEAKNESS
invalidating_steps = []
requires_seed42_rerun = false
can_run_seed43_44 = false
can_claim_positive_cga = false
```

这说明当前失败结果不是路径、strict-load、eval-output、adapter contract 或 target generation bug 导致；在当前 CGA-v2 设计和配置下，它是一个有效的 negative/design-weakness 结果。

### 0.2 当前结果含义

Full split：

```text
CGA Pd        +0.005291
CGA mIoU      -0.003096
CGA Precision -0.002140
CGA FA_ppm    +2.918
```

HC-Val：

```text
CGA Pd        +0.166667
CGA mIoU      -0.272208
CGA Precision -0.308944
CGA FA_ppm    +595.093
```

结论不是“CGA improves”，而是：

```text
CGA-v2 behaves as a recall/Pd booster, but it substantially worsens false alarms and precision, especially on HC-Val.
```

---

## 1. 立即停止项

当前不允许继续做：

```text
1. 不跑 seed43/44。
2. 不做 CGA-v2 ablation/failure-pack 作为 positive paper narrative。
3. 不写 AAAI positive main claim。
4. 不调 lambda/ramp/threshold 来 retroactively rescue 当前 P2。
5. 不改 HC-Val split。
6. 不把 CGA-v2 写成 hard-clutter robust 或 false-alarm suppressing method。
7. 不把当前 P2 失败结果覆盖或删除。
```

当前允许做：

```text
1. 归档 P2 valid-negative audit。
2. 降级 paper notes / README 里的 claim。
3. 增加 audit-only guard，防止误跑 seed43/44。
4. 增加 claim-check 脚本，防止 positive CGA-v2 文案回流。
5. 如果继续 rescue，另开 CGA-v2.1 预声明协议。
```

---

## 2. 下一步总路线

```text
R0. Freeze current CGA-v2 P2 artifacts.
R1. Commit v5 audit script and audit_v5 outputs.
R2. Downgrade docs/paper/cga_v2_aaai/README.md.
R3. Add seed43/44 guard based on A8_final_audit_decision.json.
R4. Add positive-claim grep/check script.
R5. Add CGA-v2.1 predeclaration template.
R6. Only after v2.1 protocol is frozen, decide whether to modify method.
```

---

## 3. Branch 建议

```bash
cd /home/ly/AAAI/CGA-main

git checkout -b cga-v2-p2-valid-negative-audit
git status --short
```

确认不要把 checkpoint、prediction png、large result dir 误提交：

```bash
git status --short | grep -E '(^.. results/|\.pth\.tar$|/predictions/)' || true
```

如果出现这些文件，先不要提交：

```bash
git reset results || true
```

---

## 4. 代码修改原则

这次修改是 **audit-only / documentation-only / guard-only**。

### 4.1 不改这些文件的算法逻辑

```text
model/cga_wrapper.py
loss.py
utils/cga_targets.py
model/backbones/mshnet_adapter.py
model/output_contract.py
test.py
evaluate.py
```

原因：当前 P2 已被 v5 audit 判为 valid negative design weakness。此时再改模型、loss、target 或 eval，会制造新的 protocol，不能用于解释当前 P2。

### 4.2 可以改这些文件

```text
docs/paper/cga_v2_aaai/README.md
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/P2_VALID_NEGATIVE_DESIGN_WEAKNESS.md
scripts/official/guard_cga_v2_no_seed43_44.sh
tools/official/check_no_positive_cga_v2_claims.py
docs/internal/cga_v2/cga_v2_1_predeclare_protocol_template.md
```

---

## 5. 修改 1：降级 paper notes

### 5.1 替换文件

路径：

```text
/home/ly/AAAI/CGA-main/docs/paper/cga_v2_aaai/README.md
```

建议直接替换为：

```bash
cd /home/ly/AAAI/CGA-main

cat > docs/paper/cga_v2_aaai/README.md <<'MD'
# CGA-v2 Paper Notes

## Current status

CGA-v2 seed42 under the audited from-zero paired protocol is a **valid negative design-weakness result**, not positive paper evidence.

The current implementation increases Pd, but worsens Precision, mIoU, and false alarms, especially on HC-Val.

## Safe title

> Component-Geometry Regularization for Target-Preserving Infrared Small Target Detection

This title may remain as a historical/rescue-title note, but the current CGA-v2 results do not support a positive AAAI-main method claim.

## Current evidence statement

> Under the audited seed42 controlled paired protocol on NUDT-SIRST, CGA-v2 behaves as a recall/Pd booster rather than a false-alarm-suppressing regularizer. It is therefore not valid as a positive AAAI-main method claim in its current form.

## Key P2 result

```text
Final audit decision:
  P2_VALID_NEGATIVE_DESIGN_WEAKNESS

Allowed interpretation:
  CGA-v2 increases Pd but substantially worsens false alarms and precision.

Blocked interpretation:
  CGA-v2 improves hard-clutter robustness or reduces false alarms.
```

## Forbidden CGA-v2 claims

Do not claim:

```text
CGA-v2 improves Full split performance.
CGA-v2 improves hard-clutter robustness.
CGA-v2 reduces false alarms.
CGA-v2 is positive paper evidence.
CGA-v2 is ready for seed43/44 multiseed validation.
CGA-v2 is AAAI-main ready.
CGA-v2 is a multi-backbone plug-and-play method.
CGA-v2 universally improves IRSTD detectors.
```

## Current decision

```text
final_decision = P2_VALID_NEGATIVE_DESIGN_WEAKNESS
invalidating_steps = []
requires_seed42_rerun = false
can_run_seed43_44 = false
can_claim_positive_cga = false
```

## Next route

Any rescue must be treated as a new predeclared protocol, for example `CGA-v2.1`.

The current CGA-v2 P2 result must not be retroactively rescued by tuning weights, changing targets, changing thresholds, modifying HC-Val, or selectively rerunning seeds.
MD
```

### 5.2 验证旧 positive 句子已删除

```bash
grep -R "positive Full\|improves hard-clutter\|reduces false alarms\|ready for seed43" -n \
  docs/paper/cga_v2_aaai README.md docs 2>/dev/null || true
```

理想结果：不应再出现正向 claim；如果出现，只能出现在 “Forbidden claims” 或 “Do not claim” 上下文中。

---

## 6. 修改 2：新增 P2 valid-negative 审计说明

路径：

```text
docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/P2_VALID_NEGATIVE_DESIGN_WEAKNESS.md
```

命令：

```bash
cd /home/ly/AAAI/CGA-main

cat > docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/P2_VALID_NEGATIVE_DESIGN_WEAKNESS.md <<'MD'
# Gate P2 from-zero seed42: Valid negative design weakness

## Decision

```text
final_decision = P2_VALID_NEGATIVE_DESIGN_WEAKNESS
invalidating_steps = []
requires_seed42_rerun = false
can_run_seed43_44 = false
can_claim_positive_cga = false
```

## Scope

This file records the post-P2 implementation audit outcome for the existing seed42 from-zero paired experiment.

It does not modify model code, loss code, target generation, threshold, checkpoint, dataset split, or historical P2 result artifacts.

## Interpretation

The audited CGA-v2 implementation is not invalidated by implementation bugs under the v5 audit. Therefore, the failed seed42 gate should be treated as a valid negative/design-weakness result.

## Main observation

```text
Full split:
  Pd improves slightly, but mIoU, Precision, and FA worsen.

HC-Val:
  Pd reaches 1.0, but false alarms increase sharply and Precision/mIoU collapse.
```

## Blocked actions

```text
Do not run seed43/44 for CGA-v2.
Do not write positive CGA-v2 paper claims.
Do not tune the current CGA-v2 protocol post hoc.
Do not modify HC-Val after observing the result.
```

## Allowed next step

Open a new predeclared protocol, e.g. `CGA-v2.1`, if continuing the project.
MD
```

---

## 7. 修改 3：新增 seed43/44 guard

这个 guard 防止有人在当前 CGA-v2 P2 已 valid-negative 的情况下误启动 multiseed。

新增文件：

```text
scripts/official/guard_cga_v2_no_seed43_44.sh
```

命令：

```bash
cd /home/ly/AAAI/CGA-main

cat > scripts/official/guard_cga_v2_no_seed43_44.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/AAAI/CGA-main}
cd "${ROOT}"

A8=${A8:-docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/A8_final_audit_decision.json}

if [[ ! -f "${A8}" ]]; then
  echo "[BLOCK] Missing A8 audit decision: ${A8}" >&2
  exit 2
fi

python3 - "$A8" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

decision = data.get("final_decision") or data.get("decision")
can_run = bool(data.get("can_run_seed43_44", False))
can_claim = bool(data.get("can_claim_positive_cga", False))
invalidating_steps = data.get("invalidating_steps", [])

if can_run:
    print(json.dumps({
        "guard": "cga_v2_seed43_44",
        "pass": True,
        "decision": decision,
        "can_run_seed43_44": can_run,
        "can_claim_positive_cga": can_claim,
        "invalidating_steps": invalidating_steps,
    }, indent=2, sort_keys=True))
    sys.exit(0)

print(json.dumps({
    "guard": "cga_v2_seed43_44",
    "pass": False,
    "blocked": True,
    "reason": "Current CGA-v2 P2 audit does not allow seed43/44.",
    "decision": decision,
    "can_run_seed43_44": can_run,
    "can_claim_positive_cga": can_claim,
    "invalidating_steps": invalidating_steps,
}, indent=2, sort_keys=True))
sys.exit(1)
PY
SH

chmod +x scripts/official/guard_cga_v2_no_seed43_44.sh
```

验证 guard 按预期阻止：

```bash
cd /home/ly/AAAI/CGA-main

if bash scripts/official/guard_cga_v2_no_seed43_44.sh; then
  echo "[ERROR] guard unexpectedly allowed seed43/44"
  exit 1
else
  echo "[PASS] guard blocks seed43/44 for current CGA-v2"
fi
```

---

## 8. 修改 4：新增 positive-claim 检查脚本

新增文件：

```text
tools/official/check_no_positive_cga_v2_claims.py
```

命令：

```bash
cd /home/ly/AAAI/CGA-main

cat > tools/official/check_no_positive_cga_v2_claims.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"\bCGA-v2\s+improves\b",
    r"\bCGA\s+improves\b",
    r"\bCGA-v2\s+reduces\s+false\s+alarms\b",
    r"\bCGA\s+reduces\s+false\s+alarms\b",
    r"\bhard[- ]clutter\s+robust\b",
    r"\bpositive\s+Full\s+and\s+mean\s+HC-Val\s+performance\b",
    r"\bready\s+for\s+seed43/44\b",
    r"\bAAAI-main\s+ready\b",
    r"\buniversally\s+improves\b",
    r"\bplug-and-play\s+across\s+IRSTD\s+detectors\b",
]

ALLOW_CONTEXT = [
    "Do not claim",
    "Forbidden",
    "Blocked interpretation",
    "Blocked actions",
    "not valid as a positive",
    "does not support",
]

def is_allowed_context(text: str, start: int) -> bool:
    lo = max(0, start - 240)
    ctx = text[lo:start]
    return any(marker.lower() in ctx.lower() for marker in ALLOW_CONTEXT)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/ly/AAAI/CGA-main")
    p.add_argument(
        "--paths",
        nargs="*",
        default=["README.md", "docs", "scripts"],
        help="Files or directories to scan.",
    )
    p.add_argument("--output", default="")
    args = p.parse_args()

    root = Path(args.root).resolve()
    violations = []

    files = []
    for item in args.paths:
        path = (root / item).resolve()
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        else:
            files.extend(
                f for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in {".md", ".txt", ".py", ".sh"}
            )

    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(file.relative_to(root))
        for pat in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                if is_allowed_context(text, m.start()):
                    continue
                line = text.count("\n", 0, m.start()) + 1
                violations.append({
                    "file": rel,
                    "line": line,
                    "pattern": pat,
                    "match": m.group(0),
                })

    result = {
        "check": "no_positive_cga_v2_claims",
        "pass": len(violations) == 0,
        "violations": violations,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)

if __name__ == "__main__":
    main()
PY

chmod +x tools/official/check_no_positive_cga_v2_claims.py
```

验证：

```bash
cd /home/ly/AAAI/CGA-main

python3 tools/official/check_no_positive_cga_v2_claims.py \
  --root /home/ly/AAAI/CGA-main \
  --output docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/no_positive_claims_check.json
```

通过条件：

```json
{
  "pass": true,
  "violations": []
}
```

---

## 9. 修改 5：新增 CGA-v2.1 预声明模板

新增文件：

```text
docs/internal/cga_v2/cga_v2_1_predeclare_protocol_template.md
```

命令：

```bash
cd /home/ly/AAAI/CGA-main

cat > docs/internal/cga_v2/cga_v2_1_predeclare_protocol_template.md <<'MD'
# CGA-v2.1 Predeclared Protocol Template

## Status

```text
protocol_name = CGA-v2.1
status = draft_not_active
inherits_from = CGA-v2 valid negative audit
```

CGA-v2.1 is a new protocol. It must not be presented as a continuation of the failed CGA-v2 P2 result.

## Why v2.1 is needed

CGA-v2 valid-negative audit found that the current component-geometry regularizer behaves as a recall/Pd booster and fails to suppress false alarms, especially on HC-Val.

## Allowed design directions

Choose and freeze before training:

```text
1. Add explicit hard-negative / background suppression term.
2. Add component-level false-positive penalty.
3. Rebalance center/boundary/scale/peak auxiliary weights.
4. Delay or weaken CGA ramp.
5. Modify component target geometry to reduce over-expansion.
6. Change selected adapter feature only after explicit source audit.
```

## Forbidden post-hoc rescue

```text
Do not change HC-Val after seeing CGA-v2 outputs.
Do not sweep threshold for the main table.
Do not selectively report seeds.
Do not reuse CGA-v2 P2 as positive evidence.
Do not call v2.1 results CGA-v2 results.
```

## Required frozen settings

```text
canonical_root = /home/ly/AAAI/CGA-main
dataset_dir = /home/ly/AAAI/CGA-main/datasets
dataset_name = NUDT-SIRST
seed42_first_gate = required
threshold = 0.5 fixed_predeclared
protocol = controlled
p1_preflight_passed = true
p1a_hcval_source_audit_passed = true
```

## v2.1 seed42 gate

Before seed43/44:

```text
Full delta mIoU      >= +0.020
Full delta Precision >= +0.010
Full delta FA_ppm    <= 0.0
HC-Val must not show catastrophic FA collapse
```

## Decision rule

```text
If v2.1 seed42 fails:
  stop v2.1 AAAI-main route.

If v2.1 seed42 passes:
  run paired seed43/44.

If three-seed paired evidence is stable:
  then run ablation and failure pack.
```
MD
```

---

## 10. 可选：future-run strict-load safeguard

这不是当前 P2 解释的一部分。当前 P2 已经由 v5 audit 处理。若未来开启 v2.1，可以单独在 future-run branch 里把 `test.py` 的 checkpoint load 从 `strict=False` 改成 strict/audited path。

不要把这个 patch 混入当前 P2 result explanation。

建议写法：

```python
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    raise RuntimeError(
        "Checkpoint load mismatch under paper evidence eval. "
        f"missing={missing}, unexpected={unexpected}"
    )
```

更严格的版本应该复用 v5 audit 的白名单 normalization 规则，而不是 silent partial load。

---

## 11. 验证命令

```bash
cd /home/ly/AAAI/CGA-main

# 1. 确认 no positive claim
python3 tools/official/check_no_positive_cga_v2_claims.py \
  --root /home/ly/AAAI/CGA-main \
  --output docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5/no_positive_claims_check.json

# 2. 确认 seed43/44 被阻止
if bash scripts/official/guard_cga_v2_no_seed43_44.sh; then
  echo "[ERROR] seed43/44 guard unexpectedly passed"
  exit 1
else
  echo "[PASS] seed43/44 guard blocks current CGA-v2"
fi

# 3. 确认没有误提交 checkpoint/prediction
git status --short | grep -E '(^.. results/|\.pth\.tar$|/predictions/)' || true

# 4. 只做语法检查
python3 -m py_compile tools/official/check_no_positive_cga_v2_claims.py
bash -n scripts/official/guard_cga_v2_no_seed43_44.sh
```

---

## 12. Git add / commit 建议

```bash
cd /home/ly/AAAI/CGA-main

git add \
  tools/official/audit_cga_v2_p2_impl_v5.py \
  docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/audit_v5 \
  docs/internal/cga_v2/gate_p2_from_zero_seed42_NUDT-SIRST/P2_VALID_NEGATIVE_DESIGN_WEAKNESS.md \
  docs/paper/cga_v2_aaai/README.md \
  scripts/official/guard_cga_v2_no_seed43_44.sh \
  tools/official/check_no_positive_cga_v2_claims.py \
  docs/internal/cga_v2/cga_v2_1_predeclare_protocol_template.md

# 防止误提交大结果
git reset results || true

git diff --check
git status --short

git commit -m "Record CGA-v2 P2 valid negative audit and block positive claims"
```

如果要推公开 GitHub：

```bash
git push -u origin cga-v2-p2-valid-negative-audit
```

建议先走 branch / PR，不要直接强推 `main`。

---

## 13. Final Go / No-Go

| 项目 | 当前决定 |
|---|---|
| CGA-v2 seed43/44 | No-Go |
| CGA-v2 AAAI positive paper | No-Go |
| CGA-v2 ablation/failure-pack positive narrative | No-Go |
| P2 audit archive | Go |
| paper notes claim downgrade | Go |
| seed43/44 guard | Go |
| positive-claim checker | Go |
| CGA-v2.1 predeclare template | Go |
| CGA-v2.1 method redesign | Only after protocol freeze |

---

## 14. 一句话结论

当前 CGA-v2 不是“还差 seed43/44”的状态，而是：

```text
P2_VALID_NEGATIVE_DESIGN_WEAKNESS
```

下一步应归档审计、降级论文表述、阻止误跑 multiseed，并把任何 rescue 明确切换到新的 `CGA-v2.1` 预声明协议。
