# 交接：P3 关闭（外推窗口改按残差数筛选）+ 文档与实验产物入库

**会话日期**：2026-08-27
**上一份交接**：`docs/handoff/2026-08-26-joint-downstream-evaluation.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 22
**分支**：`feat/material-aware-broken-helix`（**未推送、未 merge**）
**本轮提交**：`da425ed`（文档）、`9ef1b06`（脚本归档）、`afb20b7`（P3 代码 + 预注册）
**工作树状态**：Iteration 22 的结果章节与本文件未 commit

---

## 0. 一句话状态

P3 的挂账做法「抬高 `min_extrapolation_fit_hits`」被实测证伪——它会让
`17:8|forward_extrapolation` 从 5034 行掉到 42 行，击穿 `MIN_GROUP_COMPONENTS = 200`
而使判决变 `inconclusive`。真正的欠定条件是**残差分量数**（strip 只给 LOC0，
pixel 给 LOC0 + LOC1），全部 47809 个外推行里只有 **3 行**欠定，且全在 `24:4` backward。
改用 `DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS = 6` 后**秩亏行两臂归零**、不再事后丢行、
其余四组一行不动。**但 P1' 未关闭**：`24:4|bwd|LOC0` 的 damping 敏感度从 2.223
**升到 2.545**，Iteration 16 的一处归因据此被推翻。

---

## 1. 本轮做完的三件事

| # | 内容 | 提交 | 性质 |
|---|---|---|---|
| A | Iteration 20/21 与两份交接入库 | `da425ed` | 文档 |
| B | P2''' / P4a 产物归档 + 补 manifest + 分析脚本入库 | `9ef1b06` | 归档 |
| C | P3：外推窗口改按残差数筛选（Iteration 22）| `afb20b7` | **预注册变更** |

---

## 2. P3 结果（可引用）

### 2.1 达成的

- **秩亏行两臂归零**：single 3 → 0（`min_eigenvalue` 从 **−3.75e-10** 转正为 +3.70e-03，
  条件数从无法计算变为有限），broken 5 → 0。
- **不再事后丢行**：v2 的 24:4 bwd 在协方差路径上丢 5 行（14324 → 14319）；
  v3 的 `n_rows_in_group` = `n` = 14024。
- **只剔除 302 行**：24:4 bwd −300、17:8 bwd −2，其余四组**一行不动**，
  插值组行集合逐位不变。
- **意外收获**：Iteration 14/16 里 24:4 backward 是唯一 broken 臂回退的组，
  剔除这 300 行后两臂基本持平（LOC0 0.7875 vs 0.7874）。**那个回退来自欠定行，不是模型。**

### 2.2 未达成的（**必须照实读**）

**P1' 没关闭，而且更糟了。**

| 组 | v2 | v3 |
|---|---|---|
| `24:4|bwd|LOC0` | 2.223 | **2.545** |
| `24:4|bwd|LOC1` | 0.356 | **2.766** |

分解确认不是样本量减少的假象：LOC0 的分子（ΔNLL 在 damping 0 与 1e-12 之间的位移）
从 2.104e-03 升到 2.316e-03（+10%），分母只降 4%。

**推论：Iteration 16 把「唯一有秩亏行的组」与「唯一 damping 真实敏感的组」当成同一
根因的两个表现，这个归因不成立。** 秩亏行清零了，敏感度反而上升，是两件事。

### 2.3 Gate 4 判定

**仍是 failed，第四次复现。** 插值组一行未动，改善仍只有 0.18%–3.18%，
远不及 `INTERPOLATION_RMSE_IMPROVEMENT = 0.20`。判据未做任何放宽。

---

## 3. 预注册（写在跑之前，事后未动）

| 常量（`scripts/validate_helix_model.py`）| 值 |
|---|---|
| `DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS` | **6**（5 个 gauge-fixed 参数 + 1 冗余）|
| `PREREGISTRATION_VERSION` | 2 → **3** |
| `DEFAULT_MIN_EXTRAPOLATION_FIT_HITS` | **4，不变** |

版本号 bump 的理由：该常量虽不在 Gate 4 的常量 block 内，但它决定哪些行能进 gate，
**v2 与 v3 的 summary 不是一个序列**。版本守卫测试已重命名并扩展到这两个新常量。

---

## 4. 实验经验

### 4.1 判据要落在真正决定结果的量上，不是它的代理量

hits 数只是残差数的代理。4 个 hits 可以给 4 个残差（全 strip）也可以给 8 个（全 pixel）。
按 hits 筛，会砍掉 17:8 fwd 那 5034 个**残差数最小值就是 8** 的健康行去换 3 个病态行。
**改判据之前先量清楚病态行到底长什么样，别照着名字改。**

### 4.2 核对块本身也会写错，而且会以「失败」的面目出现

首轮核对报了两条失败，都是判据错，不是泄漏：

- **插值组 RMSE 不逐位相同**。行按 2048 行 chunk 拟合，在别处剔除 302 行会改变 chunk
  边界，float64 累加顺序随之改变，漂移 2.6e-10 ~ 1.2e-09。要求它逐位相同是错的判据；
  应判「行集合不变 且 漂移不超过舍入量级」。
- **24:4 bwd 报 −295 而非 −300**。拿它去比 v2 的 `n`（14319），而 v2 已在协方差路径
  事后丢过 5 行秩亏行。选择阶段的变更要比 `n_rows_in_group`（14324），
  否则等于把旧规则下的删除记到本次变更头上。

**核对块报失败时，先分清是判据错还是改动错。** 两次都是前者，但都必须查证过才能这么说。

### 4.3 全量 Gate 4 实测 28 分钟，不是 53 分钟

Iteration 14 记的 53 分钟。本轮同样配置 28 分钟。RSS 约 2 GB，单跑无内存压力。

### 4.4 预注册常量的守卫测试是会咬人的，这正是它的价值

`test_preregistration_v2_records_the_damping_decision` 断言
`PREREGISTRATION_VERSION == 2`，本轮 bump 到 3 时它按设计失败了。
**正确处理是更新并扩展它**（现已纳入新常量、并把「hit floor 保持 4」连同理由一起钉住），
而不是把断言放宽成 `>= 2`——那会让守卫失去意义。

### 4.5 沿用既有教训（未变）

- **任何让结果变好的改动，先查行数/样本量。** 本轮核对块的四步顺序即为此设计，
  且第 1、2 步都真的拦下了东西（虽然是判据错）。
- **归因错了，所有候选修法都会错。** 本轮又一次印证：Iteration 16 的归因错了，
  P3 挂账的做法就是错的。
- 不救活秩亏行；gate 判据不为通过而放宽。
- **必须用 conda pytorch 3.10**，默认 `python` 3.9 会把干净仓库报成 35 个假失败。

---

## 5. 未完成任务

### P1' — `24:4|backward_extrapolation|LOC0`（**仍开启，且已排除一个候选根因**）

NLL 结论**不得引用**。敏感度 2.223 → **2.545**。欠定行已被排除为根因，
下一个候选未定。该组仍是数值极端行的集中地。

### P4a' — 替换 seed 求解器（可选优化，非修复，未变）

参考实现在 `scripts/analysis/measure_seed.py` 的 `gn_solve`。判据不应是「SMT loss 更低」
（已确定会更低约三个数量级），而应是下游 `truth_metrics` 是否还有可测量的改善；
在 recovery 已 ≈ 1.0 的情况下，**必须先预注册「多大的变化才算真改善」**。

### P4b — 其他（未变）

- 结论仍只在 20000 events 上，未在 100000 events 上复现。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动，**Gate 4 已确认 failed**
  （四次复现）。

### P5 — 分支处置（未变）

`feat/material-aware-broken-helix` 未推送、未 merge。**需用户明确批准。**
另有未跟踪文件 `surface_alignment_logic.pptx`，历次提交均未纳入。

### 文档提交

Iteration 22 的 §5–§8 与本文件**尚未 commit**。

---

## 6. 环境（本轮实测）

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib   # 否则 awkward_cpp 缺 CXXABI_1.3.15
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q    # 3.10.20 / torch 2.11.0+cu126
```

**339 passed / 3 skipped / 11 deselected**（Iteration 19 基线 337 + 本轮新增 2）。
3 项 skip：AlignDerivLib pybind 未编译 ×1、**ACTS bindings 不可用 ×2**——
**要用 ACTS 时当场实测确认，勿凭这两条 skip 或凭 memory 下结论。**

**不要用 `make test` 的默认 `python`**（3.9），它会把干净仓库报成 35 个假失败。

### 本轮产物（都在 `data/` 下，不入 git，各带 manifest）

```
data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/
  residual_floor_20260827/     # Gate 4 v3 全量重跑（基线 damping_v2_20260826/ 未被覆盖）

data/experiments/joint_downstream/Vol17_Lay8__Vol24_Lay4/
  p2ppp-20260826/   # Iteration 20 的两臂比较依据 + manifest
  p4a-20260826/     # Iteration 21 的驻点性测量 + manifest
  p3-20260827/      # Iteration 22 的选择阶段测量与核对块输出 + manifest
```

入库的分析脚本（均标注为一次性脚本、无测试覆盖、不被产品路径导入）：

```
scripts/analysis/compare_summaries.py               # P2''' 两臂比较
scripts/analysis/measure_seed.py                    # P4a 驻点性测量
scripts/analysis/measure_extrapolation_selection.py # P3 选择阶段测量（hits/residuals/verify）
```

复跑命令：

```bash
# Gate 4 全量（约 28 分钟）；--output-dir 必须显式指定，否则覆盖基线
python scripts/validate_helix_model.py \
  --states data/reconstruction/nominal/Vol17_Lay8__Vol24_Lay4/trackstates_kf_mat1.root \
  --material-artifact             data/experiments/material/Vol17_Lay8__Vol24_Lay4/gate4-20260825/material_nodes.npz \
  --material-artifact-energy-loss data/experiments/material/Vol17_Lay8__Vol24_Lay4/gate4-20260825-eloss/material_nodes.npz \
  --model both --max-tracks 0 --output-dir <本次路径>

# P3 选择阶段测量（秒级，不跑拟合）
python scripts/analysis/measure_extrapolation_selection.py {hits|residuals|verify}

# 核对块
python data/experiments/joint_downstream/Vol17_Lay8__Vol24_Lay4/p3-20260827/check_block.py
```
