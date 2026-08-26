# 交接：Gate 4 在 Gauss–Newton 参考下的重跑结果

**会话日期**：2026-08-26
**上一份交接**：`docs/handoff/2026-08-25-broken-helix-gate4-and-solver.md`（仍然有效，本文件只记增量）
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 16
**分支**：`feat/material-aware-broken-helix`（3 个 commit，**未推送**）

---

## 0. 一句话状态

上一轮的工作树已固化成三个 commit；Gate 4 用 GN 新参考全量重跑完毕，
**判定维持 failed，方向未翻案，12 组里 10 组逐位不变**；
唯一实质变化是 24:4 backward 的回退归因翻案（+8.69% → +1.37%，落进容差内）。
P1（damping 两臂不等价）与 P2（`ml_alignment_autograd.py` 同类缺陷）按用户指示**本轮未动**。

---

## 1. 本轮做了什么

用户在会话开始时用 AskUserQuestion 定下两条，仍然有效：

| 决策项 | 选定 |
|---|---|
| 本轮范围 | **先提交工作树 + P0 Gate 4 重跑**；P1 / P2 暂不动 |
| 提交位置 | 新建分支 `feat/material-aware-broken-helix`（仓库历史此前从未用过分支）|

P2 在上一份交接里被标注为「挂在用户那里等答复」——**已答复：暂不动，等 P0/P1 完成**，
理由是避免同时改动两处求解器。

---

## 2. 三个 commit

| commit | 内容 |
|---|---|
| `d50aeda` | 材料 artifact 管线：`material_artifact.py` 三方 join / schema 1.1 / `material` 子命令 / 路径项 |
| `39f1597` | broken-helix 模型 + GN 求解器；含把「Adam 不达驻点」写成断言的测试 |
| `41acf3e` | 架构契约、Iteration 13/14/15、research 知识库、handoff、AGENTS.md 两节 |

合计 30 文件 / +7472 / −15。那 15 个删除与提交前工作树 diff 的 deletions 数一致 → 无意外删除。

`surface_alignment_logic.pptx` 按上一份交接的指示**未提交**，仍是未跟踪状态。

**测试基线**：`make test` → **285 passed / 35 failed / 3 skipped**。
35 项失败集中在 `compare_geometries` / `joint_alignment` / `plot_*` / `project_config`，
均为既有失败（`/usr/bin/python` 3.9 子进程），本轮改动文件对应的测试全部通过，零新增失败。
（上一份交接记的是 284 passed，本轮实测 285，方向无害，未追查。）

---

## 3. Gate 4 重跑结果

**产物**：`data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/broken_helix_gn_20260826/`
20000 events / 70352 行 / **26 分 18 秒**（比预估的 53 分钟快一半，因为 single 臂只需 2 次迭代）
`manifest.json` 已写出，`git_commit = 41acf3e`。

### 数值门控（先判这个）

| 臂 | 非驻点行占比 | decrement p50 | 迭代 |
|---|---|---|---|
| single | 1.279e-4 | 3.00e-23 | **2** |
| broken | 8.529e-5 | 3.02e-19 | 20 |

上限 1e-2，两臂通过。single 臂 2 次迭代即确认在最优 → 重跑要的「不动点」性质达成。

### 物理结论

- **Gate 4 维持 failed。** 插值组改善 0.3–1.8%，远不及 −20% 门槛。逐位复现。
- **12 组里 10 组与 Iteration 14 逐位相同**（6 位小数）。
- **24:4 backward LOC0 回退归因翻案**：CI 上界 +8.69% → **+1.37%**，
  已落进 `EXTRAPOLATION_RMSE_REGRESSION = 0.05`。旧跑的大回退主要是 Adam 参考未收敛的产物，
  不是「kink 吸收几何自由度」。但 CI 整体在正区间（下界 +0.07%），回退统计显著、幅度微小。
- **过散修正结论不变**：17:8 interp LOC1 pull_std 1.1258 → 0.9877，χ² 1.2674 → 0.9755。
- **damping 敏感性仍翻转 8 个外推组**，现被显式列为一条门控失败项。

---

## 4. 本轮新增的经验

### 4.1 比对两次实验的 holdout 一致性，比的是身份键而非文件哈希

新旧 `holdout_keys.csv` 的文件 SHA256 不同（`4ecf9804…` vs `d0490b85…`），一度像是两跑不可比。
实际差异**全在 `holdout_path_s_mm` 一列**——弧长依赖拟合参数，换求解器后必然变
（最大偏差 49.8 mm / 相对 48%，正是 Adam 未收敛的幅度）。
身份键 `(event_nr, track_nr, mode, holdout_sensor_idx)` 的哈希逐位一致，diff 0 行。

**这个文件里混入了依赖拟合结果的列，整体哈希对求解器变更没有不变性。**
上一份交接把整文件 SHA256 记为可比性凭据，是个会误导下一轮的坑。

### 4.2 「换了参考重跑」的正确预期是「大部分逐位不变」

10/12 组完全不变，正是因为 Iteration 14 的 GN single 臂本就从 Adam 点再优化到同一最优。
**变动集中在哪些组，本身就是信息**：只有 24:4 backward 变了，说明只有那个组的旧 single 臂
没到最优——这直接定位了旧结论里唯一不可信的部分。

### 4.3 长 markdown 追加的防截断做法有效

`cat >> file <<'EOF'` + 追加前 `grep -c '^## Iteration 16'` 唯一性断言 +
追加后 `git diff --numstat` 核对 deletions 为 0。本轮 86 insertions / **0 deletions**。
上一轮用 `t.index("### 标题")` 截断掉 234 行的事故没有重演。

---

## 5. 未完成任务（优先级已随本轮结果调整）

### P1 — 修 damping 对两臂不等价（现在是唯一的拦路项）

病灶：`scripts/validate_helix_model.py:2252`，`predictive_covariance` 用
`damping · mean(diag H) · I`。broken 臂 H 含 2N 个 kink 参数，`mean(diag H)` 被 `Q⁻¹` 块抬高，
同一相对 damping 对两臂的绝对扰动差很多。

`damping = 0` 时两臂 Cholesky 都失败，所以阻尼不是可选项。候选修法：
按臂各自 `mean(diag H)` 归一改为**绝对**阻尼；对 kink 块与 reference 块分别加阻尼；
或改用 pivoted Cholesky / LDLᵀ 免阻尼求解。

**在此之前，外推组的 NLL 数字一律不得引用。** 改动会 bump 预注册版本。

### P2 — `ml_alignment_autograd.py:604` 的同类缺陷

那里有另一个 Adam 版 `fit_helices_to_measurements`（block-LM 里 nuisance 轨迹的 refit），
**直接决定 joint 校准结果**。用户已定：等 P1 完成后再动。
**安全第一步仍然是只测量 Newton decrement、不改行为**，可复用
`validate_helix_model.py:3064` 的 `newton_decrement` 与 `:3102` 的 `stationarity_report`。

### P3 — 其他（未变）

- `seed_helices` 的 Adam 预拟合未检验是否欠收敛（只定起点，影响面小）。
- 17:8 forward LOC1 回退（+16.7%，Iteration 15 记）未做归因外的进一步验证。
- 结论仍只在 20000 events 上，未在计划书写的 100000 events 上复现。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动。**Gate 4 现已确认 failed。**

### P4 — 分支处置

`feat/material-aware-broken-helix` 未推送。Gate 4 结论已确定，可以考虑 merge 回 main，
但**这是用户的决定，未经明确批准不要 merge 或 push**。

---

## 6. 关键数据产物（不要删）

```
data/experiments/material/Vol17_Lay8__Vol24_Lay4/
  gate4-20260825/  gate4-20260825-eloss/  gate0-20260825/
data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/
  broken_helix_20260825/     # Adam 参考，Iteration 14
  gn_solver_20260825/        # Iteration 15 GN single
  broken_helix_gn_20260826/  # 本轮，Gate 4 的现行参考
```

---

## 7. 环境（本轮实测）

- `/cvmfs/sft.cern.ch/lcg/views/LCG_109` 已挂载，`../alignTest/build` 存在 → ACTS **可用**。
  memory 里 `acts-reconstruction-blocked.md` 记的「未挂载」已过期，但**每轮仍需实测**。
- `python` = `/usr/bin/python` 3.9.25（`make` 与全量跑都用它，与 Iteration 14 一致）。
  当前 shell 无 conda 环境激活（`CONDA_PREFIX` 为空），这是正常状态。
- 其余硬约束见上一份交接 §8，未变。
