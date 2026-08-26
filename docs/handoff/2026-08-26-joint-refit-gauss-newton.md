# 交接：joint refit 换成 Gauss-Newton（P2' 完成）

**会话日期**：2026-08-26（接续同日的 Iteration 18 驻点性测量）
**上一份交接**：`docs/handoff/2026-08-26-joint-refit-stationarity.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 19
**分支**：`feat/material-aware-broken-helix`（**未推送、未 merge**）
**工作树状态**：本轮改动**尚未 commit**

---

## 0. 一句话状态

P2' **已完成**：joint 校准初始化的 measurement refit 换成 Gauss-Newton，
`chi2/component` 从 **7992.76 降到 0.741103**，非驻点行从 100% 降到 **0%**，
且 `n_components` 两臂相同（5616），改善不是丢样本换来的。

---

## 1. 预注册（写在跑之前，事后未动）

| 常量（`scripts/ml_alignment_autograd.py` 常量区）| 值 |
|---|---|
| `JOINT_REFIT_SOLVER` | `"gauss-newton"` |
| `JOINT_REFIT_PREREGISTRATION_VERSION` | 1 |
| `JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION` | 1.0e-3 |

判据：非驻点行占比 ≤ 1e-3 **且** 非有限行 = 0 **且** 既有 `invalid_fraction` 守卫不变。
任一不满足 → raise，不产出结果。驻点阈值复用已预注册的 `NEWTON_DECREMENT_STATIONARY = 1e-6`。

**`PREREGISTRATION_VERSION` 保持 2**：它治理的是 Gate 4 判决常量，本轮一个也没碰，
Gate 4 也不消费 joint 输出。joint 侧另立版本线，理由见 Iteration 19 §1。

---

## 2. 结果（`--track-mode free --max-tracks 300`，两臂只差一个 flag）

| 量 | Adam | **Gauss-Newton** |
|---|---|---|
| `n_components` | 5616 | **5616** |
| `chi2_per_component` | 7992.76 | **0.741103** |
| 非驻点行占比 | 100.00% | **0.00%** |
| decrement p90 | 4.17e5 | **1.87e-17** |

Adam 臂**逐位复现** Iteration 18 的数字，说明重构没有改变 Adam 的行为。

---

## 3. 三个先解决掉的坑（下轮不要重踩）

### 3.1 「不收敛行丢弃」在这条路径上不存在

`fit_broken_helix` 对每行都返回参数，不产生 NaN 行；`converged` 是 per-chunk 的 bool。
秩亏 NaN 丢弃是协方差路径 `_run_broken_arm_chunk` 的事。joint 场景丢一条 nuisance 轨迹
= 改变 nuisance 集合，与 Gate 4 丢一行语义完全不同。**per-row 判据只能是 Newton decrement。**

### 3.2 GN 分支必须 float64

阈值 1e-6 低于 float32 分辨率，而 joint 管线在 GPU 上跑的就是 float32
（`seed_helices` 的 dtype 由 `--gpu` 决定）。GN 强制 float64 求解，再转回调用方 dtype。

### 3.3 判据不受诊断开关影响

验收读的是 GN 求解器自己算的 decrement，不是 `--no-joint-refit-stationarity` 能关掉的
那个诊断测量。关诊断 ≠ 关验收。

---

## 4. 未完成任务

### P2'' — 全量复现（下一步）

本轮 A/B 只在 300 条轨迹上做过。全量 joint 运行未跑。

### P2''' — 下游影响未评估

只验证了 refit 本身。`target_metrics` 等下游对齐结果因 `refitted_parameters` 改变而
改变，**变化幅度与方向尚未系统评估**。这是引用任何 joint 对齐结论前必须做的。

### P1' — `24:4|backward_extrapolation|LOC0`（未变）

唯一仍对 damping 真实敏感的组，NLL 结论**不得引用**。

### P3 — `min_extrapolation_fit_hits = 4` 过松（未变）

single 臂 4 hits 撑 5 个 gauge-fixed 参数。改它属于另一次预注册变更。

### P4 — 其他（未变）

- `seed_helices` 的 Adam 预拟合仍未检验是否欠收敛。**它现在是唯一剩下的未检验 Adam 环节。**
- 结论仍只在 20000 events 上。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动，**Gate 4 已确认 failed**。

### P5 — 分支处置（未变）

`feat/material-aware-broken-helix` 未推送、未 merge，**且本轮改动尚未 commit**。
需用户明确批准。

---

## 5. 环境（本轮实测）

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q
```

**337 passed / 0 failed / 3 skipped**（Iteration 18 基线 333 + 本轮新增 4）。
3 项 skip：AlignDerivLib pybind 未编译 ×1、ACTS bindings 不可用 ×2。
**不要用 `make test` 的默认 `python`**（3.9），它会把干净仓库报成 35 个失败。
