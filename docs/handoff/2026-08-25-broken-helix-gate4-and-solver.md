# 交接：材料感知 broken-helix（Gate 0–4）与求解器修复

**会话日期**：2026-08-25
**计划文件**：`~/.claude/plans/docs-research-ml-alignment-autograd-py-jiggly-origami.md`（v3）
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 13 / 14 / 15
**本文件的定位**：下一会话冷启动用。只写代码和 git log 里读不出来的东西。

---

## 0. 一句话状态

工作包 ①②③ 做完了：Gate 0/1/2/3 全部通过，**Gate 4 判 failed（物理结论，非数值失败）**；
随后发现并修掉了一个更大的问题——测量 refit 的 Adam 求解器从未收敛，
换成 Gauss–Newton 后插值残差直接降 3.5 倍。
**Gate 4 需要用新参考重跑**，工作包 ④（joint 校准）按 Gate 5 规则仍不得启动。

---

## 1. 本轮范围（用户已定，勿擅自扩大）

会话早期用 AskUserQuestion 定下的三条，仍然有效：

| 决策项 | 选定 |
|---|---|
| 实现范围 | **仅 ①②③（模型 + 验证）**，不做 ④ joint 校准接入 |
| 材料预算来源 | ACTS `MaterialInteractor` 逐轨迹导出 |
| kink 自由度 | 非目标层 refit 时拟合，block-LM 时冻结 |

原始问题的第 (b) 层「应用到 joint 校准是否改善」**本轮明确不作答**。

---

## 2. 已取得的成果

### 2.1 Gate 0 —— 完全通过

- ACTS 逐轨迹材料导出跑通（`alignment_odd.py material` 子命令）。
- 三方 join（material / propsummary / tracksummary）全部 fail-closed 校验通过。
- **能损条目已关闭**（Iteration 14 用第二份 `energyLoss=True` artifact 复检）：
  段分配翻转率 **0.0**（上限 0.0）；p99 |Δpredicted|/σ = **2.03e-6**（LOC0）/
  **1.71e-5**（LOC1），上限 0.10。
  → 「记录 pass 用 `energyLoss=False` 与首版模型自洽」这一假设成立。

### 2.2 Gate 1/2/3 —— 通过

- artifact schema **1.1** 冻结（新增 `volume_geometry_id`）。
- 规范固定（gauge fixing）：reference helix 6→5 参数，条件数 **4.6e13 → 1.4e6**。
- 合成 MC ensemble closure：1200 成员，pull / 覆盖率均在预注册容差内。
- predictive covariance 解析对拍通过（排除了 ½χ² vs χ² 的 factor-of-two）。

### 2.3 Gate 4 —— **failed**（20000 events / 70352 holdout 行）

先判数值：Newton decrement 非驻点行占比 single **1.71e-4** / broken **8.53e-5**，
远低于预注册上限 1e-2 → **这是物理结论，不是数值失败**。

| 组 | N | RMSE single→broken | ΔRMSE 95% CI | 门槛 |
|---|---|---|---|---|
| 17:8 interp LOC0 † | 8346 | 0.021723→0.021470 | [−1.84%, −0.60%] | 上界 < −20% ✗ |
| 24:4 interp LOC0 † | 14197 | 0.050144→0.049868 | [−0.82%, −0.32%] | 上界 < −20% ✗ |
| 17:8 fwd LOC0 † | 5034 | 0.213453→0.213394 | [−0.06%, +0.00%] | NLL 上界 +0.009 ✗ |
| 24:4 bwd LOC0 † | 14324 | 0.846021→**0.864379** | [−0.89%, **+8.69%**] | 上界 < +5% ✗ |

†＝预注册必需组，全部 N ≥ 5034，远超 200 门槛 → 判定统计有效，非 inconclusive。

**可以宣称的**：broken 臂确实修正了一个真实的过散
（17:8 interp LOC1 pull_std 1.126→0.988，χ² 1.267→0.976）；插值 RMSE 改善统计显著但仅 0.3–1.8%。
**不得宣称**：broken-helix 优于 single helix。

### 2.4 Iteration 15 —— Adam→GN 求解器落差已修

**决定性证据**（同一批 70352 行、同一目标函数、同一掩码）：

| 求解器 | 非驻点行占比 | decrement p50 | 迭代 |
|---|---|---|---|
| Adam | **1.0000** | **12.6** | 200（预算耗尽）|
| Gauss–Newton | 1.28e-4 | 9.6e-23 | **16** |

Adam **每一行**都在 1e-6 门槛之上——不是个别行停滞，是整批没收敛。

全量 holdout 对照（关键行）：

| 层\|模式 | 分量 | Adam | GN | 比值 |
|---|---|---|---|---|
| 17:8 interp | LOC0 | 0.076261 | **0.021723** | **÷3.5** |
| 24:4 interp | LOC0 | 0.173330 | **0.050144** | **÷3.5** |
| 17:8 bwd | LOC0 | 0.567832 | **0.187964** | **÷3.0** |
| 24:4 bwd | LOC1 | 5.140685 | 3.247682 | ÷1.58 |
| 17:8 fwd | LOC1 | 0.060489 | **0.070582** | **×1.17（唯一回退）** |

门控失败项 **11 → 8**，其中三条是真的通过了（不是数字变小）：
17:8 interp LOC0（门槛 0.05）、24:4 interp LOC0（门槛 0.1）、17:8 bwd 方向角（门槛 0.005）。
**两个插值组第一次落进 Iteration 10 定的绝对阈值内。**

---

## 3. 实验经验（值得记的，含踩坑）

### 3.1 物理结论翻案：Iteration 10 的归因是错的

「RMSE 比 bias 高 1–2 个量级 ⇒ 未建模多重散射」——插值组上有 **3.5 倍来自优化器欠收敛**。
这直接解释了 broken-helix 收益为何这么小：GN 收敛后，留给材料 kink 解释的残差已所剩无几。

**教训**：再提「用更复杂的轨迹模型降残差」之前，先确认求解器到了驻点。

### 3.2 「converged: true」会骗人，要看 Newton decrement

`max_relative_decrease < 1e-10` 对**停滞的行同样成立**。
白化梯度范数也不行——它不是尺度无关的（σ=0.01mm 的行在最优点也有数值上很大的梯度）。
唯一可用的是 `λ² = gᵀH⁻¹g`。这条已固化成 `NEWTON_DECREMENT_STATIONARY = 1e-6` +
`MAX_NON_STATIONARY_ROW_FRACTION = 0.01` 的门控，主路径与 Gate 4 两臂共用。

### 3.3 damping 对两臂不等价（Gate 4 最重要的数值坑，**未解决**）

`damping = 0` 时两臂 Cholesky 都失败（H 在有限精度下并非严格正定），所以 damping 不是可选项。
但 `damping·mean(diag H)·I` 对两臂的绝对扰动差很多——broken 臂 H 含 2N 个 kink 参数，
`mean(diag H)` 被 `Q⁻¹` 块抬高：

| damping | 24:4 bwd LOC0 ΔNLL |
|---|---|
| 1e-12 | −0.0012 ✓ |
| **1e-10（production）** | **+0.106 ✗** |
| 1e-08 | +5.34 ✗ |
| 1e-06 | +29.2 ✗ |

**8 个外推组的 NLL 判定随预注册 damping 网格翻转** → 外推方向属于「不可测」，不是「无改善」。
插值组不在翻转名单里，那两条结论稳健。

候选修法：按臂各自 `mean(diag H)` 归一改为**绝对**阻尼；或对 kink 块与 reference 块分别加阻尼；
或改用 pivoted Cholesky / LDLᵀ 免阻尼求解。

### 3.4 小样本会给出反向结论

2000 轨迹子集上 24:4 bwd LOC0 显示 GN 回退 ×1.27；全量 70352 行下是 ÷1.12 **改善**。
**结论一律以全量为准**，子集只用来验证代码跑得通。

### 3.5 正确求解是有代价的

17:8 forward LOC1 从 0.0605 涨到 0.0706（+16.7%）。
Adam 的欠收敛相当于朝 SMT 种子的隐式收缩，对单边外推的曲率外插恰好起了正则作用；
GN 把窗内似然做到最优，曲率过拟合会被外推放大。
**这不是 bug**，但说明外推预测需要显式先验（如材料 kink 的 `Q`），不能靠优化器欠收敛白捡。

### 3.6 两个必须遵守的工程教训

- **改 markdown 长文件时，`t.index("### 标题")` 会匹配到第一处。**
  本轮曾因此把 `ml_optimization_log.md` 从 715 行截断到 481 行，删掉了 Iteration 12 尾部
  和整个 Iteration 13。已用 `git checkout HEAD --` + heredoc 重新追加完整恢复。
  之后的每次追加都改用「唯一性断言 + `git diff --stat` 核对 0 deletions」。
- **`if __name__ == "__main__"` 守卫一度被留在文件中部**，导致后面追加的 Gate 2/3/4 函数
  在脚本运行时根本没被定义。已移到文件末尾。

---

## 4. 改动文件清单（工作树，**全部未提交**）

| 文件 | 状态 | 内容 |
|---|---|---|
| `scripts/validate_helix_model.py` | M +1976 | broken-helix 数学、predictive covariance、Gate 4 双臂、`--fit-solver`、驻点门控 |
| `scripts/helix_model.py` | M +190 | 分段/kink/Highland；N=0 支持 |
| `scripts/material_artifact.py` | **?? 新建** | 三方 join、schema 1.1、聚合/剔除、fail-closed |
| `scripts/alignment_odd.py` | M +283 | `runMaterialExport()` + `material` 子命令 + 覆盖保护 + 导出端 manifest |
| `scripts/path_inventory.py` | M +24 | material ROOT / propsummary / NPZ / `holdout_keys.csv` 路径项 |
| `scripts/experiment_manifest.py` | M +32 | — |
| `scripts/alignment_artifact_layout.py` | M +1 | — |
| `tests/test_validate_helix_model.py` | M +1056 | Gate 2/3/4 契约、门控统计、求解器 9 项 |
| `tests/test_helix_model.py` | M +316 | closure 用例 |
| `tests/test_material_artifact.py` | **?? 新建** | schema / join / 聚合 / fail-closed |
| `tests/test_path_inventory.py` | M +57 | 新路径项 |
| `tests/test_experiment_manifest.py` | M +70 | — |
| `docs/architecture.md` | M +71 | Gate 4 paired holdout 契约 + Measurement refit solver 契约 |
| `docs/history/ml_optimization_log.md` | M +470 | Iteration 13/14/15 |
| `docs/research/` | ?? | 知识库（决策记录等） |

**注意**：`surface_alignment_logic.pptx` 是无关文件，勿一起提交。

`make test` 现状：**284 passed / 35 failed / 3 skipped**。
35 项失败是**既有的**，与本轮无关——它们用 `/usr/bin/python`（3.9）起子进程，导入即失败，
已用 stash 复核确认。**零新增失败。**

---

## 5. 关键数据产物（不要删）

```
data/experiments/material/Vol17_Lay8__Vol24_Lay4/
  gate4-20260825/material_nodes.npz          # 20000 events，19781 tracks / 268434 nodes
  gate4-20260825-eloss/material_nodes.npz    # energyLoss=True 对照，268436 nodes
  gate0-20260825/                            # 1000 events，Gate 0 用
data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/
  broken_helix_20260825/                     # Gate 4 全量（Adam 参考），含 holdout_keys.csv
  gn_solver_20260825/                        # Iteration 15 全量 GN（--model single）
```

`broken_helix_20260825/holdout_keys.csv` SHA256 = `4ecf9804f7e5980a427c9d62d88737e64093e6048da877abe41fe4e034a44398`

---

## 6. 预注册常量（Gate 1 冻结，改动需 bump 版本）

```
scripts/validate_helix_model.py
  FIT_SOLVERS = ("gauss-newton", "adam")     :88
  PRODUCTION_DAMPING = 1e-10                 :2205
  BOOTSTRAP_SEED = 20260825                  :2666
  MIN_GROUP_COMPONENTS = 200                 :2667
  INTERPOLATION_RMSE_IMPROVEMENT = 0.20      :2668
  EXTRAPOLATION_RMSE_REGRESSION = 0.05       :2669
  MAX_NON_STATIONARY_ROW_FRACTION = 0.01     :2679
  NEWTON_DECREMENT_STATIONARY = 1e-6         :3061
```

---

## 7. 未完成任务（按优先级）

### P0 — Gate 4 用新参考重跑

`fitted_parameters` 同时是 Gate 4 的冻结参考。换成 GN 后参考变了，
**Iteration 14 的门控数字不再逐位可复现**。约 53 分钟。

```bash
python scripts/validate_helix_model.py \
  --states data/reconstruction/nominal/Vol17_Lay8__Vol24_Lay4/trackstates_kf_mat1.root \
  --material-artifact data/experiments/material/Vol17_Lay8__Vol24_Lay4/gate4-20260825/material_nodes.npz \
  --material-artifact-energy-loss data/experiments/material/Vol17_Lay8__Vol24_Lay4/gate4-20260825-eloss/material_nodes.npz \
  --model both --max-tracks 0 \
  --output-dir data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/broken_helix_gn_<date>
```

**预期**：方向上不会翻案。Iteration 14 的 GN single 臂本就从 Adam 点再优化到同一最优
（两处 17:8 interp LOC0 都是 0.021723）。变的是 broken 臂的起点与线性化点，不是 single 臂水平。
新跑之后 single 臂成为自身最优的不动点，比较更干净。

### P1 — 修 damping 对两臂不等价

见 §3.3。**在此之前，外推组的 NLL 数字不要引用。**
这也是 24:4 backward 回退归因的前提——目前无法区分「kink 吸收几何自由度」与
「damping 不等价的产物」。

### P2 — `ml_alignment_autograd.py:604` 的同类缺陷

那里有**另一个** Adam 版 `fit_helices_to_measurements`，是 block-LM 里 nuisance 轨迹的 refit。
同样的欠收敛极可能存在，且它**直接决定 joint 校准结果**。

本轮未动的理由：参数化不同（position / direction / rho_scaled 三组独立 lr），
改动会改变 joint 校准行为，那属于工作包 ④，Gate 5 规定 Gate 4 未通过不得启动。

**建议的安全第一步**：只**测量**它的 Newton decrement，不改行为。
会话结束时这个问题正挂在用户那里等答复。

### P3 — 其他

- `seed_helices`（对 SMT 点的 Adam 几何预拟合）未检验是否也欠收敛。只定起点，影响面小。
- 17:8 forward LOC1 回退（+16.7%）未做归因外的进一步验证。
- 20000 events 的结论对必需组统计有效，但**未在计划书写的 100000 events 上复现**。
- 工作包 ④ 的接口按计划只留不接（`docs/...jiggly-origami.md` §④）。

---

## 8. 环境与硬约束

```bash
source setup_acts.sh     # ACTS 需新 shell，不与 conda 混用
```

- `/cvmfs/sft.cern.ch/lcg/views/LCG_109` 已挂载，`../alignTest/build` 存在。
  memory 里 `acts-reconstruction-blocked.md` 记的「未挂载」已过期，但**每轮仍需实测**。
- 只有 `alignment_odd.py` / `generate_odd_transform.py` 可 import `acts`；
  `helix_model.py`、`material_artifact.py`、`validate_helix_model.py` 必须无 ACTS 可导入。
- 大产物一律在 `data/` 下，**不提交**。
- 迁移 ROOT/JSON 源文件：复制→清点→校验→切配置→**取得明确批准后**才删源。
- 每个算法实验（含失败/中止）都要写 schema 化 `manifest.json`。
- 先写失败测试再实现。
- 破坏性操作（删除、覆盖、推送）前先确认。
