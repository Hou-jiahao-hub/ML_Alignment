# 交接：joint refit 的测量与替换（P2 / P2' / P2'' 完成）

**会话日期**：2026-08-26（接续同日的 damping 等价性修复）
**上一份交接**：`docs/handoff/2026-08-26-damping-equivalence-v2.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 18、19（含 §7 全量复现）
**分支**：`feat/material-aware-broken-helix`（**未推送、未 merge**）
**本轮提交**：`3977b51`（Iteration 18）、`9f6a951`（Iteration 19）
**工作树状态**：仅 Iteration 19 §7 的文档更新未 commit；代码全部已提交
**后续**：P2''' 与 P4a 均已完成，见 `docs/handoff/2026-08-26-joint-downstream-evaluation.md`

---

## 0. 一句话状态

joint 校准初始化的 measurement refit 从 Adam 换成 Gauss-Newton，并在全量上复现：
`chi2/component` **7595.34 → 0.763503**，非驻点行 **100% → 0%**，
`n_components` 两臂逐位相同（1154761），改善不是丢样本换来的。
~~**下游影响（P2'''）仍未评估**~~ → **已评估并关闭**，见 Iteration 20：全量上
tx/ty 恢复 corr ≥ 0.95、recovery ≈ 1.0，`17:8` 的 ty 从反号纠正为 +0.99993。

---

## 1. 本轮做完的三件事

| # | 内容 | Iteration | 提交 |
|---|---|---|---|
| P2 | joint refit 的驻点性**测量**（纯测量，零行为变更）| 18 | `3977b51` |
| P2' | 换成 Gauss-Newton（预注册变更）| 19 | `9f6a951` |
| P2'' | 全量 A/B 复现 | 19 §7 | 文档未提交 |

---

## 2. 实验结果

### 2.1 全量 A/B（可引用）

`--track-mode free --joint-iters 1`，两臂只差 `--joint-refit-solver` 一个 flag。
数据集 `Vol17_Lay8__Vol24_Lay4`：98958 条输入轨迹，保留 **61482** 条。

| 量 | Adam | **Gauss-Newton** |
|---|---|---|
| retained tracks | 61482 | 61482 |
| `n_components` | 1154761 | **1154761** |
| `invalid_fraction` | 0.0 | **0.0** |
| `chi2_per_component` | 7595.34 | **0.763503** |
| 非驻点行占比 | 100.00% | **0.00%** |
| Newton decrement p90 | 4.1e5 | **2.61e-17** |
| GN 收敛 | — | **True，5 次迭代**（上限 60）|

**按预注册顺序核对**：`n_components` 与 `invalid_fraction` 两臂逐位相同 → 样本量没变
→ 四个数量级的 chi2 改善不是丢行换来的。预注册判据以极大裕度满足（非驻点行精确为
0.0，上限 1e-3；非有限行 0）。

`chi2/component` 落到 0.763 有独立的物理佐证：正确归一的 per-component chi2 在测量噪声
主导时本就该在 1 附近。**7992 / 7595 是荒谬值，0.76 才是合理值。**

### 2.2 小样本对全量的代表性（已验证）

| 量 | 300 轨迹 | 全量（205 倍 component）|
|---|---|---|
| Adam chi2/component | 7992.76 | 7595.34 |
| Adam decrement p90 | 4.17e5 | 4.1e5 |
| GN chi2/component | 0.741103 | 0.763503 |
| GN 非驻点行 | 0.00% | 0.00% |

两个尺度同向且量级一致。**以后可以用 `--max-tracks 300` 做快速筛查**，但结论仍需全量确认。

### 2.3 Iteration 18 的测量（本轮起点）

Adam refit 在 200 次迭代后 decrement p90 = 4.17e5，阈值 1e-6，非驻点行 100%，
seed → refit 只降约 22%。这个数字在 Iteration 19 的 Adam 臂上**逐位复现**，
说明 Iteration 19 的重构没有改变 Adam 的行为。

---

## 3. 实验经验

### 3.1 「不收敛行丢弃」在这条路径上根本不存在（计划被实测推翻）

Iteration 19 的计划里写了「GN 不收敛行一律丢弃」，实施时核实发现无从落地：
`fit_broken_helix`（`validate_helix_model.py:2552`）对每行都返回参数（per-track 阻尼 +
步长接受），不产生 NaN 行；`converged` 是 **per-chunk 的 bool**，不是 per-row。
秩亏 NaN 丢弃是协方差路径 `_run_broken_arm_chunk` 的事，不在拟合路径上。
而且 joint 场景丢一条 nuisance 轨迹 = 改变 nuisance 集合，与 Gate 4 丢一行语义完全不同。

**per-row 判据只能落在 Newton decrement 上**，判据据此设计。

### 3.2 判据要能失败，测试要双向

只断言「GN 解满足判据」的话，一个**永不触发**的 gate 也能通过。因此用 `max_iters=1`
的真实截断 GN 解构造不达标输入，断言它确实 raise（实测 1 次迭代非驻点行 100%，
完整解 0%）。这与 Iteration 18 的分辨力测试是同一个道理。

### 3.3 `--max-tracks N` 时 `input_tracks` 报的是 4N，不是数据集规模

`ml_alignment_autograd.py:2342` 把 ROOT 读取上限设为 `max_tracks * 4`。所以
`--max-tracks 300` 的运行打印 `tracks=1200 retained=300`，**1200 是截断读取的规模**。
全量实际是 98958。本轮曾据此误判全量只有 1200 条轨迹。
**不要从 capped 运行里读数据集大小。**

### 3.4 预注册版本号要看它治理什么

`PREREGISTRATION_VERSION`（`validate_helix_model.py:2774`）的注释写明治理范围是
「this block」——Gate 4 的判决常量，且只写进 Gate 4 的 summary。P2' 改的是 joint 路径
求解器，不碰任何 Gate 4 常量，Gate 4 也不消费 joint 输出。升它会让此后每份 Gate 4
summary 声称判决规则变过，而实际未变。**故 joint 侧另立
`JOINT_REFIT_PREREGISTRATION_VERSION`，Gate 4 的保持 2。**

### 3.5 GN 分支必须 float64

预注册阈值 1e-6 低于 float32 分辨率，而 joint 管线在 GPU 上跑的就是 float32
（`seed_helices` 的 dtype 由 `--gpu` 决定）。GN 强制 float64 求解，再转回调用方 dtype。

### 3.6 关诊断 ≠ 关验收

验收读的是 GN 求解器自己算出的 decrement（`solver_stats["stationarity"]`），
不是 `--no-joint-refit-stationarity` 能关掉的那个诊断测量。

### 3.7 假桩要符合真实契约

`tests/test_component_mask_wiring.py` 的 monkeypatch 桩缺 `solver` 字段，被新增的打印
读到而失败。**补桩，而不是把打印改成防御式**——桩偏离契约就失去了保护作用。

### 3.8 沿用既有教训（未变）

- **任何让结果变好的改动，先查行数/样本量**（Iteration 17 §4.1）。
- 归因错了，所有候选修法都会错——改之前先量。
- 不救活秩亏行；gate 判据不为通过而放宽。
- **必须用 conda pytorch 3.10**，默认 `python` 3.9 会把干净仓库报成 35 个假失败。

---

## 4. 预注册（写在跑之前，事后未动）

| 常量（`scripts/ml_alignment_autograd.py` 常量区）| 值 |
|---|---|
| `JOINT_REFIT_SOLVER` | `"gauss-newton"` |
| `JOINT_REFIT_PREREGISTRATION_VERSION` | 1 |
| `JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION` | 1.0e-3 |

判据：非驻点行占比 ≤ 1e-3 **且** 非有限行 = 0 **且** 既有 `invalid_fraction` 守卫不变。
任一不满足 → raise，不产出结果。驻点阈值复用已预注册的
`NEWTON_DECREMENT_STATIONARY = 1e-6`。1e-3 是类比
`MAX_RANK_DEFICIENT_ROW_FRACTION` 定的，**不是看着输出挑的**。

Adam 保留在 `--joint-refit-solver {gauss-newton,adam}` 后面，默认 GN。
`best_iteration` / `improved` / `first_step` 是 Adam 语义，**只在 adam 分支出现**，
不给占位值——免得读 summary 的人把 GN 运行误认成「checkpoint 恰好没改善过」的 Adam 运行。

---

## 5. 未完成任务

### ~~P2''' — 下游影响未评估~~ → **已关闭**（Iteration 20）

全量两臂比较完成：GN 臂在 `17:8` 与 `24:4` 上均优于 Adam 臂，`17:8` 的 ty 从
corr −0.115 / recovery −0.275 纠正为 **+0.99993 / +0.99973**，
`best_total_loss` 2286.55 → 14.16。详见
`docs/handoff/2026-08-26-joint-downstream-evaluation.md`。

**本节原写的做法是错的**：`--joint-iters` 只被 `--joint-solver adam` 分支消费，
默认的 block-LM 用 `--lm-max-iters`（默认 20）。所以上面写的
「改用完整 `--joint-iters`」无效，而本轮 §2.1 的全量两臂**当时就已跑满 block-LM**——
P2''' 缺的是比较，不是运行。

### ~~P4a — `seed_helices` 的 Adam 预拟合未检验~~ → **已关闭**（Iteration 21）

测量确认**确实停在非驻点**（全量 `dec/F > 1%` 占比 99.98%），**但已被 GN refit 吸收，
不构成下游缺陷**。注意其目标是 SMT 几何距离平方和（mm²），与 `F = 0.5·chi2` 不同
gauge，`NEWTON_DECREMENT_STATIONARY = 1e-6` **不可移植**。详见新交接。

### P1' — `24:4|backward_extrapolation|LOC0`（未变）

唯一仍对 damping 真实敏感的组（敏感度 2.223 = 统计不确定度的 2.2 倍），
NLL 结论**不得引用**。它同时是唯一有秩亏行、Iteration 14/16 唯一回退的组。

### P3 — `min_extrapolation_fit_hits = 4` 过松（未变）

single 臂 4 hits 撑 5 个 gauge-fixed 参数（插值组是 8）。属另一次预注册变更。

### P4b — 其他（未变）

- 结论仍只在 20000 events 上，未在 100000 events 上复现。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动，**Gate 4 已确认 failed**
  （三次复现，插值组改善仅 0.3–1.8%）。

### P5 — 分支处置（未变）

`feat/material-aware-broken-helix` 未推送、未 merge。**需用户明确批准。**
另有未跟踪文件 `surface_alignment_logic.pptx`，历次提交均未纳入。

---

## 6. 环境（本轮实测）

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib   # 否则 awkward_cpp 缺 CXXABI_1.3.15
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q    # 3.10.20 / torch 2.11.0+cu126
```

**337 passed / 0 failed / 3 skipped**（Iteration 18 基线 333 + Iteration 19 新增 4）。
3 项 skip：AlignDerivLib pybind 未编译 ×1、**ACTS bindings 不可用 ×2**。

**注意**：memory 里记的「ACTS 2026-08-26 复核为可用」在本轮 pytest 下不成立——
大概率是该命令没有 source ACTS 环境脚本，而非环境退化。**要用 ACTS 时当场实测确认，
勿凭这两条 skip 或凭 memory 下结论。**

**不要用 `make test` 的默认 `python`**（3.9），它会把干净仓库报成 35 个失败。

全量单臂 `--joint-iters 1` 耗时超过 10 分钟，需后台运行；两臂并行有 OOM 风险
（本机可用内存约 7 GB，单臂峰值约 3 GB），建议串行。
