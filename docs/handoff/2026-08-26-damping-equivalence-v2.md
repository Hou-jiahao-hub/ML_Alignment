# 交接：damping 两臂不等价的修复（预注册 v2）

**会话日期**：2026-08-26（接续同日上半场的 Gate 4 GN 重跑）
**上一份交接**：`docs/handoff/2026-08-26-gate4-gn-rerun.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 17
**分支**：`feat/material-aware-broken-helix`（6 个 commit，**未推送、未 merge**）

---

## 0. 一句话状态

P1（damping 两臂不等价）**已修复**：11/12 组的 damping 依赖被彻底消除
（6 组完全免疫），修复前是 8 个外推组跨全网格翻转。
**外推组的 NLL 现在可以引用了**，唯独 `24:4|backward_extrapolation|LOC0` 除外。
Gate 4 判定维持 failed，11/12 组与 Iteration 16 逐位相同。
P2（`ml_alignment_autograd.py` 同类缺陷）仍未动。

---

## 1. 上一份交接的归因是错的

`2026-08-25-broken-helix-gate4-and-solver.md` §3.3 记的
「`mean(diag H)` 被 `Q⁻¹` 块抬高」**实测不成立**：
kink 块与 reference 块的对角均值比是 **0.99**，`1/kink_sigma²` ≈ 3.7e5
比测量项的 ~5e9 小四个量级，`mean(diag H)` 随节点数 0→7 反而从 5.05e9 略降到 4.29e9。

真正的机制见 Iteration 17 §2：**padding 造成的结构零方向**。
broken 臂把 `node_s` 补到固定宽度，未用槽位不进入任何残差，H 与 J 的对应行列精确为零
（600/600 行验证：零特征值个数 = `2×(padded − valid)`，`|J @ nullvec|` 全为 0）。
阻尼给这些方向赋 `1/(λ·scale)` 的方差，解析上被 J 消掉，但把 Cholesky 条件数推到 ~2e13，
舍入误差污染全解。single 臂没有 padding 可 damp。

**候选修法 ①②③（按臂归一 / 分块阻尼 / pivoted Cholesky）全部基于错误归因，都不对。**

---

## 2. 改了什么（预注册 v2）

| # | 改动 |
|---|---|
| 1 | 结构零方向不再阻尼，改填 active 平均对角（对协方差是恒等变换）|
| 2 | `PRODUCTION_DAMPING` 1e-10 → **0.0**（修复 1 之后才首次可解）|
| 3 | 秩亏行返回 NaN 丢弃，不再正则化成一个数字 |
| 4 | 敏感性 gate 只用 `DAMPING_STABILITY_SCAN = (0.0, 1e-12)`，全网格仍记录为诊断 |
| 5 | 新增 `PREREGISTRATION_VERSION = 2`、`MAX_RANK_DEFICIENT_ROW_FRACTION` |

commit：`74dc0e7`（零方向 + damping=0）、`0e0f962`（秩亏行丢弃）、
`f0f9b9c`（逐行判据修正，见 §4）。

---

## 3. 全量验收结果

- **11/12 组的 N 与 RMSE 与 Iteration 16 逐位相同**
- 只有 24:4 backward 变（N 14324→14319，LOC0 RMSE 0.842698→0.786690），源于丢 5 行秩亏行
- **Gate 4 维持 failed**，插值组改善仍只有 0.3–1.8%
- damping 敏感度：6 组 ratio=0.000，5 组 ≤0.36，**1 组 = 2.223**

丢弃的 8 行（single 3 / broken 5）特征一致：`n_hits` 全是 **4 或 5**（健康行中位数 7），
single 臂 4 hits 撑 5 个 gauge-fixed 参数、μ_min 为负 → **结构性欠定**。
根因是预注册的 `min_extrapolation_fit_hits = 4` 过松（插值组是 8）。

---

## 4. 本轮最重要的教训

### 4.1 我自己写出了一个会悄悄美化结论的 bug

秩亏判据 `rank_floor = maximum * eps * n_parameters` 里，`maximum` 取的是
**整个 batch 的最大特征值**而非每行自己的。这等于用 chunk 里最硬的轨迹缩放所有行的门槛：
broken 臂 μ_max 达 5.7e12，floor 被抬到 ~0.04，而健康行 `smallest_active` 在 7.5e-7 量级
→ **50 行健康数据被整批误判为秩亏**。

而这 50 行**全部**落在 24:4 backward 组，丢掉它们把该组 LOC0 的 RMSE
从 **0.8427mm 改善到 0.4921mm（42%）**——正则化器差点悄悄剔掉了最不利的数据。

**发现它靠的是核对各组的 N**，不是看 RMSE 变好了没有。
**教训：任何让结果变好的改动，先查行数/样本量有没有变。**
修复后降到 5 行。回归测试双向验证过（bug 版本失败、修复版本通过）。

### 4.2 先实现后删除的 per-row 阻尼阶梯

Cholesky 失败的条件与数值秩亏判据 `μ_min ≤ ε·n·μ_max` 高度重合，
阶梯从来没有可服务的场景。凡是失败的行都是真秩亏，救活它 = 把正则化器的方差当结果报。

### 4.3 gate 判据没有为了让它通过而放宽

`24:4|backward_extrapolation|LOC0` 的 damping 敏感度是统计不确定度的 **2.2 倍**，
是真实敏感性而非符号判定噪声（对照：6 个组的 ratio 是 0.000）。
sensitivity gate 仍然 fail，但这次的 fail 有信息量：它精确指出唯一一个不可信的组。

---

## 5. 未完成任务

### P1' — `24:4|backward_extrapolation|LOC0`

唯一仍对 damping 真实敏感的组，其 NLL 结论**不得引用**。
这个组同时是：唯一有秩亏行的组、Iteration 14/16 里唯一回退的组。
数值极端行集中在此，值得单独查（外推距离最远、约束最弱的配置）。

### P2 — `ml_alignment_autograd.py:604` 的同类缺陷（未动）

另一个 Adam 版 `fit_helices_to_measurements`，block-LM 里 nuisance 轨迹的 refit，
**直接决定 joint 校准结果**。用户此前定的是「等 P0/P1 完成后再动」——**P1 现已完成**。
安全第一步仍是只测量 Newton decrement、不改行为，可复用
`validate_helix_model.py` 的 `newton_decrement` 与 `stationarity_report`。

### P3 — `min_extrapolation_fit_hits = 4` 过松

对 single 臂的 5 参数 gauge-fixed 拟合不足（4 hits 时 μ_min 为负）。
改它属于另一次预注册变更，本轮未动。

### P4 — 其他（未变）

- `seed_helices` 的 Adam 预拟合未检验是否欠收敛。
- 结论仍只在 20000 events 上，未在 100000 events 上复现。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动，**Gate 4 已确认 failed**。

### P5 — 分支处置

`feat/material-aware-broken-helix` 未推送、未 merge。**需用户明确批准**。

---

## 6. 数据产物（不要删）

```
data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/
  broken_helix_20260825/     # Adam 参考，Iteration 14
  gn_solver_20260825/        # Iteration 15
  broken_helix_gn_20260826/  # Iteration 16，GN 参考
  damping_v2_20260826/       # Iteration 17，预注册 v2 的现行参考
```

---

## 7. 环境（本轮实测）

与 `2026-08-26-gate4-gn-rerun.md` §7 相同：ACTS 可用，`python` = `/usr/bin/python` 3.9.25。
`make test`：**293 passed / 35 failed / 3 skipped**（35 项为既有失败，零新增）。
全量单次约 26 分钟。
