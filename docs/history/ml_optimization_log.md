# ML Alignment Optimization Log
# 文件：Examples/Scripts/Python/job/ml_optimization_log.md
# 用途：记录每次 ml_alignment.py 迭代的变更、训练指标、重建指标和分析
# 使用约定：每次优化前读本文件，每次训练完成后在末尾追加新条目

---

## Iteration 0 — 基线（有缺陷版本）
**日期**: 2026-05-xx（历史）
**配置**: v17l8_v24l4（vol=17 lay=8, vol=24 lay=4）
**EPOCHS**: 500 | **LAMBDA_L2**: 1e-7 | **优化器**: Adam + CosineAnnealingLR

### 变更内容
- 初始版本，未经系统调试

### 已知缺陷（本次迭代暴露）
1. **chi2 clamp = 50**：`physics_informed_loss` 中对 `raw_chi2` 裁剪至 50。vol=17 pixel 的 chi2≈(0.37/0.015)²≈608，全部被截断 → **梯度为 0**，像素层根本未训练
2. **残差方差公式错误**：用 `err_x_hit²` 替代理论值 `Σr = err_x_hit² - err_eLOC0_smt²`（残差协方差应扣除平滑轨迹贡献）
3. **L2 过弱**：LAMBDA_L2=1e-7，无法有效防止冗余修正

### 训练残差（`l_x_hit - eLOC0_smt`，来自 misaligned 轨迹）
| 层 | 训练前 Std | 训练后 Std |
|---|---|---|
| Vol17 X | 0.2626mm | 0.3300mm（变差） |
| Vol17 Y | 0.1350mm | 0.8100mm（严重变差） |

### 重建残差 Std（reconstruction ROOT 文件）
_未记录（已覆盖）_

---

## Iteration 1 — 修复 chi2 clamp + Σr 方差 + L2 增强
**日期**: 2026-06-01
**配置**: v17l8_v24l4
**EPOCHS**: 500 | **LAMBDA_L2**: 1e-5 | **优化器**: Adam + CosineAnnealingLR

### 变更内容（相对 Iteration 0）

**Fix 1（关键）：移除 chi2 clamp**
```python
# 旧代码（在 physics_informed_loss 中）：
chi2_per_hit = torch.clamp(raw_chi2, max=50.0)

# 新代码：
# 不裁剪，直接使用
chi2_loss = torch.mean(hit_weights * raw_chi2)
```

**Fix 2（重要）：Σr 残差协方差**
```python
# 旧代码（错误）：
var_eLOC0 = err_x_hit²
var_eLOC1 = err_y_hit²

# 新代码：Σr = V - H·P_smt·H^T = err_x_hit² - err_eLOC0_smt²
var_r0 = (err_x_hit² - err_eLOC0_smt²).clip(lower=1e-8)
var_r1 = (err_y_hit² - err_eLOC1_smt²).clip(lower=1e-8)
```

**Fix 3：L2 正则化增强**
```python
LAMBDA_L2 = 1e-5  # 旧值: 1e-7
```

**Fix 4：增加 per-layer 训练诊断输出**
```python
TARGET_LAYERS = [(17, 8), (24, 4)]  # 每 epoch 打印各层 chi2
```

### 训练残差（来自 misaligned trackstates，chi2/ndf<50 过滤后）
| 层 | 训练前 Std | 训练后 Std | 改善 |
|---|---|---|---|
| Vol17 X | 0.2626mm | 0.0519mm | **80%** |
| Vol17 Y | 0.1350mm | 0.0426mm | **68%** |
| Vol24 X | 0.4398mm | 0.0970mm | **78%** |
| Vol24 Y | 1.7747mm | 1.4422mm | **19%** |

### ML 修正量对比（vol17 lay8 sen1，相对 misaligned）
| 方向 | ML 修正 | ACTS 修正 | ML/ACTS 比例 |
|---|---|---|---|
| ΔX | 0.022mm | 0.080mm | 28% |
| ΔY | -0.175mm | -0.737mm | 24% |
| ΔZ | -0.010mm | -0.562mm | 2% |

### 重建残差 Std（Phase 4 ACTS 重建，`l_x_hit - t_eLOC0`）
| 层 | Nominal | Misaligned | ACTS-Aligned | ML-Aligned |
|---|---|---|---|---|
| Vol17 X | 0.0148mm | 4.01mm | **0.0148mm** | 3.97mm |
| Vol17 Y | 0.0148mm | 18.83mm | 0.021mm | 18.32mm |
| Vol24 X | 0.0422mm | 11.94mm | 3.82mm | 12.51mm |
| Vol24 Y | 1.176mm | 28.18mm | 11.60mm | 28.33mm |

### 问题分析
**训练指标好但重建无效**，根本原因如下：

1. **有偏残差问题**：训练用 `l_x_hit - eLOC0_smt`（KF 平滑态），KF 会将轨迹拉向测量点，使有偏残差远小于真实错位量。原始 ROOT 文件中 `res_eLOC0_smt` Std = **2.95mm**，但 chi2/ndf<50 过滤后只剩 **0.26mm** 的"好轨迹"
2. **chi2/ndf<50 过滤**：严重错位几何下大多数轨迹 chi2>>50，被丢弃，留下的是偶然拟合好的少数轨迹 → 训练信号不代表全局错位
3. **ML 修正量远小于真实值**：因为有偏残差信号弱，模型只找到真实修正量的 2-28%

### 下一步优化方向（候选）

**方向 A（最高优先级）：切换到无偏预测态残差**
```python
# 将训练信号从平滑态改为预测态（无偏）
merged_df["res_eLOC0"] = merged_df["l_x_hit"] - merged_df["eLOC0_prt"]
# 方差也换成预测态：
var_r0 = err_eLOC0_prt²  # 或 err_x_hit²（预测态不确定度已包含错位影响）
```
- ROOT 分支 `res_eLOC0_prt`（Std=2.97mm）和 `err_eLOC0_prt` 已在文件中，直接可用
- 预期：ML 模型看到真实 ~3mm 的错位信号，修正量会接近 ACTS 水平

**方向 B：放宽 chi2/ndf 过滤**
```python
sum_df = sum_df[sum_df["chi2_ndf"] < 200.0]  # 旧值: 50
```
- 引入更多错位轨迹（当前 50 阈值在严重错位下过于严格）
- 可与方向 A 配合使用

**方向 C：增加 epoch 数 + 调整 LR**
- 切换到预测态残差后，梯度信号变强，可能需要更小的初始 LR（1e-4→5e-5）
- Epoch 可从 500 增到 1000 观察收敛

**方向 D：加入旋转 Jacobian（AlignDerivLib 旋转项）**
- 目前只用平移分量（tx,ty,tz），旋转分量 Jacobian 可能对 Z 方向修正有帮助

---

## Iteration 2 — 修复选择偏差 + 切换无偏残差（P0 两点）
**日期**: 2026-06-03
**配置**: v17l8_v24l4
**EPOCHS**: 500 | **LAMBDA_L2**: 1e-5 | **优化器**: Adam + CosineAnnealingLR（均未改动）

### 背景：Iteration 1 “训练好但重建废” 的两个根因
1. **chi2/ndf<50 是选择偏差**：两层错位 ~0.5mm、像素分辨率 ~15μm，单 hit chi2≈(0.5/0.015)²≈1000，真正穿过错位模块的轨迹几乎全被切掉，留下的是“目标层残差偶然很小”的样本 → 训练信号系统性偏向零。
2. **平滑态残差有偏**：KF 平滑把轨迹拉向测量点，`l_x_hit − eLOC0_smt` 远小于真实错位（`res_eLOC0_smt` Std=2.95mm，过滤后仅 0.26mm）。

### 关键发现
ROOT trackstates 树**已自带无偏分支** `res_eLOC0_ubs / res_eLOC1_ubs`（把当前 hit 从平滑里剔除的无偏残差）及配套协方差 `err_eLOC0_ubs / err_eLOC1_ubs`，LOC0/LOC1 均存在。优于方向 A 的 `prt`：误差现成，无需手算 V±H·P·Hᵀ。

### 变更内容（相对 Iteration 1，仅 P0 两点）

**P0.1（`load_variable_information`，约 :121-126）：移除 chi2/ndf 过滤**
```python
# 旧：sum_df = sum_df[sum_df["chi2_ndf"] < 50.0]
# 新：不过滤；chi2_ndf 仍作为 per-hit 输入特征保留
```

**P0.2（残差与协方差，约 :138-208）：smt → ubs**
```python
# per_state_keys 新增：res_eLOC0_ubs, res_eLOC1_ubs, err_eLOC0_ubs, err_eLOC1_ubs
# 旧：res = l_x_hit - eLOC0_smt; var = Σr = err_x_hit² - err_eLOC0_smt²
# 新：
merged_df["res_eLOC0"] = merged_df["res_eLOC0_ubs"]
merged_df["res_eLOC1"] = merged_df["res_eLOC1_ubs"]
merged_df["var_eLOC0"] = (merged_df["err_eLOC0_ubs"]**2).clip(lower=1e-8)
merged_df["var_eLOC1"] = (merged_df["err_eLOC1_ubs"]**2).clip(lower=1e-8)
```

**保持不变**：优化器/LR/EPOCHS/LAMBDA_L2、Jacobian 每 20 epoch 重算、1D strip 处理、strip Y floor(1.44)、平移/旋转 clamp、hit_quality_net 仍锁 unit_weights。
> 注：旧的 `err_x_hit/err_y_hit` 每轨迹展开块仍保留（现已不被引用，无害但冗余），留待后续清理。

### 数据量变化（去 chi2 cut 的直接效果）
- 轨迹数：过滤版 ~数百条 → **98958 条**；有效 hit：**143697**（全 2D pixel；本配置 strip 在该筛选下无 1D hit）。

### 训练残差（无偏 `ubs` 残差，misaligned trackstates，无 chi2 过滤）
| 层 | 训练前 Std | 训练后 Std | 改善 |
|---|---|---|---|
| Vol17 X | 4.0241mm | 3.8704mm | 4% |
| Vol17 Y | 18.8618mm | 17.4283mm | 8% |
| Vol24 X | 11.9468mm | 11.5184mm | 4% |
| Vol24 Y | 28.2466mm | 28.1913mm | 0.2% |

> 训练残差已回到**真实错位尺度**（4–28mm），与 Iter1 的重建残差（4.01/18.83/11.94/28.18mm）几乎吻合 → **P0 成功消除了偏差信号**（Iter1 被压到 0.26mm 的假信号已修复）。但 chi2 在 epoch 40 即平台（497581→437058，仅 12%），残差只降 0.2–8%。

### ML 修正量对比（mean per-axis |δt|，相对 misaligned）
| 层 | 方向 | ML 修正 | ACTS 修正 | ML/ACTS |
|---|---|---|---|---|
| Vol17 L8 | X | 0.834mm | 0.319mm | 262% |
| Vol17 L8 | Y | 0.840mm | 0.322mm | 261% |
| Vol17 L8 | Z | 1.281mm | 0.504mm | 254% |
| Vol24 L4 | X | 1.172mm | 0.318mm | 368% |
| Vol24 L4 | Y | 1.205mm | 0.319mm | 378% |

单 sensor 样例 (17,8,1)：ML δt=[-1.299, 1.186, -1.320]　vs　ACTS δt=[0.080, -0.737, -0.562]
→ **符号相反、且大量饱和在 ±1.5mm clamp**。修正问题从 Iter1 的「过小（24–28%）」翻转为「过大且方向错误」。

### 重建残差 Std（未运行 Phase 4；训练残差仅降 ≤8% 已预示重建不会显著改善）
| 层 | Nominal | Misaligned | ACTS-Aligned | ML-Aligned(Iter2) |
|---|---|---|---|---|
| Vol17 X | 0.0148mm | 4.01mm | 0.0148mm | 未运行（预期 ≈misaligned） |
| Vol17 Y | 0.0148mm | 18.83mm | 0.021mm | 未运行 |
| Vol24 X | 0.0422mm | 11.94mm | 3.82mm | 未运行 |
| Vol24 Y | 1.176mm | 28.18mm | 11.60mm | 未运行 |

### 验收结论
- ✅ **P0 目标达成**：无偏残差 + 去过滤后，训练信号回到真实错位尺度（4–28mm），Iter1 的「偏差/选择偏差」根因已消除。
- ❌ **仍未追上 ACTS**：瓶颈已转移到**优化器**。无偏残差 per-hit 方差极大（如 strip Y：err~0.09mm 却有 ~28mm 残差扩散），单步梯度下降在逐 hit 噪声上拟合 → 修正量乱飘、撞 ±1.5mm clamp、方向与 ACTS 相反，chi2 12% 即平台。残差不在 J 的列空间内，凸 LSQ 也降不动。
- **诊断指向 Iteration 3 的 P1**：需用**每 surface 加权法方程** δα=(ΣJᵀσ⁻¹J+λI)⁻¹(ΣJᵀσ⁻¹r) 把 71600 个 hit 的噪声平均掉、提取相干位移；并放开 ±1.5mm clamp（真实修正 ~0.3–0.7mm，但中间迭代量可能更大）；同时做真值校验 `predicted_deltas` vs `−(misaligned−nominal)` 确认 J/JSON 符号一致。

### 下一步（Iteration 3 候选，承接上轮 P1/P2）
- P1：用每 surface 直接加权最小二乘 + Tikhonov 替代 500-epoch Adam（与 ACTS Kalman alignment 同源）。
- P1：外层闭环（写 JSON → `alignment_odd.py reconstruction` → 重读残差）迭代 2–3 轮。
- P2：tz 弱模显式正则/固定；启用 hit_quality_net。

---

## Iteration 3 — 直接加权 LSQ 求解器（P1）+ 真值结构发现
**日期**: 2026-06-03
**配置**: v17l8_v24l4 | **求解器**: 每 surface 加权法方程（替代 Adam）

### 变更内容
- 新增 `solve_alignment_weighted_lsq()`：A_s=ΣJᵀΣ⁻¹J, b_s=ΣJᵀΣ⁻¹r, δα_s=(A_s+λI)⁻¹b_s，per-surface 聚合。
- 新增 `validate_against_reference()`：把 ML 修正投影到 global 与 ACTS 对比方向（sign-agreement）。
- `run_alignment_pipeline` 加 `SOLVER="lsq"|"adam"` 开关；clamp 放宽到 ±5mm；LSQ 路径跳过 loss 曲线。

### 结果（misaligned 训练数据，无偏残差）
| 层 | X_before | X_after(LSQ) | Y_before | Y_after(LSQ) |
|---|---|---|---|---|
| Vol17 L8 | 4.0241mm | 3.9797mm | 18.8618mm | 17.7659mm |
| Vol24 L4 | 11.9468mm | 11.6845mm | 28.2466mm | 28.2360mm |

- 数学最优 LSQ **仍只降残差 1–6%**；sign-agreement 仅 **0.36–0.48**（≈随机），LSQ 修正方向与 ACTS 不相关。
- 排除“优化器问题”：瓶颈不在 P1。残差 r 基本**不在 J 的列空间内**。

### 关键真值发现（读 alignment_odd.py setupMisalignment）
1. **真实错位只有 3 个 DOF**：local X/Y 平移（±0.5mm，`shift_mag_mm=0.5`）+ local Z 旋转（±0.02rad）。`shift_local_z=0`、rx=ry=0 **恒为零** → 解 6-DOF 含 tz/rx/ry 三个纯噪声弱模。
2. **ACTS 不是真值**：Iter1 中 ACTS 自身对 vol24 只修到 3.82/11.60mm，未修干净，不能作为方向基准。
3. **nominal 几何 = `odd_transforms.json`**（setupMisalignment 输入），misaligned = `misalignment_applied.json` → 可重建逐元件真值 δα 做闭合检验。
4. **残差尺度存疑**：0.5mm 平移 + 0.02rad 旋转难以直接产生 4–28mm 残差 → 无偏残差 STD 大概率被 per-hit 外推噪声主导，**STD 不是正确的成功度量**。

### 结论与下一步（Iteration 4）
- 不再盲目换求解器。**先做纯几何 Jacobian 闭合检验**：对目标 sensor，用 `alignmentToResidualDerivative` 的 J 与“同一轨迹在 nominal/perturbed 面上重新求交得到的 local 残差有限差分”逐参数对比。J 正确 → 问题在估计/噪声/度量；J 错 → 修 AlignDerivLib frame/符号（最可能根因）。
- 之后：把解限制到 tx/ty/rz 3 DOF；成功度量改为“每 sensor 均值残差 + 加权 chi2 下降 + 对真值 δα 的吻合”。

---

## Iteration 4 — 鲁棒局部拟合（IRLS+3DOF）+ 真值校验 → 局部无监督路线判死
**日期**: 2026-06-03 | **求解器**: robust weighted LSQ (DOF=tx,ty,rz, IRLS pull_cut=5)

### 变更内容
- `solve_alignment_weighted_lsq` 重写为鲁棒版：IRLS 硬离群拒绝（pull>5）+ 限制 ACTIVE_DOF=(tx,ty,rz)（真值只有这些非零）。
- 新增 `validate_against_truth()`：用 nominal `odd_transforms.json` 算每 sensor 真实修正 δt_true=R_misᵀ(c_nom−c_mis)、rz_true=atan2(Rδ[1,0],Rδ[0,0])，报告 corr/recovery。
- 前置诊断（Iteration 3.5）：纯几何 **Jacobian 闭合检验**——解析 J 与有限差分逐元素吻合（tx[-1,0]/ty[0,-1]/tz[0.3,0.15]），`r−J·δα` 符号一致 → **J 完全正确**。

### 决定性结果：ML vs 真值相关性 ≈ 0
| 层 | DOF | corr | recovery | \|true\|rms | \|pred\|rms |
|---|---|---|---|---|---|
| Vol17 L8 | tx | +0.003 | +0.000 | 0.523mm | 0.043mm |
| Vol17 L8 | ty | −0.018 | −0.002 | 0.524mm | 0.048mm |
| Vol24 L4 | tx | −0.050 | −0.010 | 0.520mm | 0.104mm |
| Vol24 L4 | ty | −0.011 | −0.001 | 0.513mm | 0.044mm |

- 残差几乎不变（4.02→4.03mm 等）；sign-agreement vs ACTS 仍 0.4（随机）。
- IRLS 仅 12.3% inlier（err_ubs~0.009mm 太小，88% hit pull>5）；核里 corr 仍≈0。

### 结论（已实证，非推测）
1. **J 正确、优化器正确、鲁棒拟合也无效** → 排除所有"技术实现"嫌疑。
2. **corr≈0（非 −1）= 信号不存在**，不是符号/frame bug。
3. **根因（实证确认）**：misaligned 重建**几何自洽**，KF 把错位吸收进轨迹参数，逐 hit 无偏残差只剩分辨率尺度、不含逐面相干偏移。错位只在"对真值残差"（推断时不可得）里显现。
4. **结论**：逐面无偏残差 + 局部 Jacobian 的**无监督路线对本错位不可行**，无关求解器。

### Iteration 5 方向（二选一，需用户定）
- **A 监督学习**：用已知真值 δα（odd_transforms vs misalignment_applied）监督训练，特征=逐面残差分布/位置/角度。本 MC 可定量追平真值；`validate_against_truth` 已提供现成 truth 标签。
- **B 全局对齐导数（Millepede）**：实现"一个面移动→整条轨迹重拟合→各面残差变化"的全局导数，最小化全局轨迹 χ²。与 ACTS 同源、可用于真实数据；实现量大。

---

## Iteration 6 — 路线 A 闭环（求解器修复 + 迭代重建）：发散结论
**日期**: 2026-06-04

### 背景与路线 B 否决
- 决定性诊断（`diag_signal_check.py`）：misaligned 重建的 **unbiased 残差**逐 sensor 中位与真值 tx 的 corr 仅 **0.04–0.19**（不是之前误用 smoothed 残差时以为的 0）。信号弱但存在，被 KF 轨迹参数吸收。
- **路线 B（重新数字化）否决**：它是现有 misaligned 重建的符号镜像（错位 hits + 标称几何 vs 标称 hits + 错位几何），KF 吸收一样发生，无增益。

### 求解器修复（关键 bug）
旧 `solve_alignment_weighted_lsq` 的 pull-IRLS 把**信号 hit 当外点剔除**：像素 σ≈0.015mm，真实 0.5mm 错位 → pull≈33 ≫ cut=5，每个带信号的 hit 被丢，只剩 12% 近零残差 hit → 解出 δ≈0、corr 0.003。
调参实验（`tune_solver.py`，数据加载一次扫多配置）：

| 配置 | vol17 tx corr | \|pred\| |
|---|---|---|
| baseline pull=5 irls=4 | +0.003 | 0.04mm |
| no-reject + σfloor=0.2 | **−0.566** | 0.57mm |

修复：`PULL_CUT=1e9, N_IRLS=1, SIGMA_FLOOR=0.2, ACTIVE_DOF=(0,1)`（去 rz，真值 rz≈1e-4 rad，原来灌 0.02rad 噪声）。
**重要副产物**：解出的 δ 与"到真值的修正"**反号**（recov<0）——这解释了为何历来 ML 单发对齐打不过 ACTS（一直按错号应用）。闭环用 `gain=-0.5`（翻号+阻尼），单步 recovery 转正 +0.35。

### 路线 A 实现
- `ml_alignment.py` 改为单步求解器，CLI：`--input-geom --states --summary --output-geom --nominal --iter --gain --no-plot`（默认值向后兼容）。几何更新本就是"在当前几何叠加 δ"。
- `route_a_loop.sh N EVENTS [GAIN]`：每轮 (LCG 子shell) 用 G_k 重建 → (conda 子shell) 解一步 → 叠加得 G_{k+1}。两环境独立子 shell 隔离；**勿用 `set -u`**（LCG setup.sh 的 COMPILER 未定义变量会被 nounset 打断）。
- 收敛条件实测公式：`ρ²<2·recov`（ρ=‖g·δ‖/‖true‖）。5000 事件（~3 hit/sensor）噪声主导发散；定为 100k 事件（~65 hit/sensor）。

### 决定性结果：100k×4 轮仍发散
`|true|rms`（G_k 到 nominal 的剩余距离，mm）：

| 几何 | vol17 tx | vol17 ty | vol24 tx | vol24 ty |
|---|---|---|---|---|
| G_0 | 0.52 | 0.52 | 0.52 | 0.51 |
| G_1 | 0.43 | 1.30 | 0.92 | 1.61 |
| G_2 | 0.50 | 2.22 | 1.42 | 2.88 |
| G_3 | 0.58 | 2.91 | 1.78 | 4.10 |

除 vol17 tx 第一步小降后反弹，其余单调暴涨（ty→2.9/4.1mm）。**100k 高统计量没救回来。**

### 根因：符号逐轮翻转 → per-surface 方法原理性死路
含 gain=−0.5 的 corr 逐轮符号：iter0 **+0.18** → iter1 **−0.82** → iter2 −0.91 → iter3 −0.91。
即原始解与剩余真值的**符号关系不固定**：iter0 反号（gain 有效），但应用过冲修正后几何被推过头，iter1 起符号翻转，固定 gain 把几何越推越远。
本质：**per-surface 独立最小二乘的极小点 ≠ nominal**。KF 把错位吸收进轨迹参数，"最小化单面残差"与"逼近真值几何"是两个目标，方向关系随几何状态变化，固定增益/符号根治不了每轮变的符号。ty |pred| 每轮 1.3–1.6mm 过冲噪声加速发散。

### 结论
路线 A 架构正确实现，但搭配 per-surface 求解器**无法收敛**。被 KF 吸收的信号，单面独立求解（哪怕迭代）拿不回来。**唯一正解：Millepede 联立拟合**——求解时纳入 ∂res/∂(轨迹参数) 并解析消元，扣除吸收项，使求解目标真正对齐"逼近真值几何"，符号才稳定。

相关产物：`route_a_loop.sh`、`route_a_main.log`、`route_a_logs/`、`tune_solver.py`、`diag_signal_check.py`。

---

## Iteration 7 — standalone 无监督联立拟合对齐器 `global_align.py`：成功
**日期**: 2026-06-04

### 动机
用户重申目标：**无监督、不依赖 ACTS、可移植**的对齐代码（现实无真值标签）。唯一可行=轨迹+对齐联立最小化径迹 χ²（Millepede / CTD MLAligner），纯 torch。per-surface 迭代发散的根因是"最小化单面残差≠逼近真值"——把轨迹参数纳入联立优化即根治。

### 实现（`global_align.py`，纯 torch，推断不依赖 ACTS）
- 可微螺旋引擎：`helix_point/helix_tangent`（曲率参数化 (x0,y0,z0,psi0,tanl,rho)，弧长 s，ρ→0 稳定）、`intersect_phi`（Newton 平面求交）、`global_to_local`、`apply_align`（6-DOF）。
- **Stage 1 望远镜合成闭环**：联立拟合把 0.40mm 隐藏错位恢复到 **0.0017mm**；per-surface 冻结径迹仅 0.057mm 且有偏 → 方法成立。
- **Stage 2 数据接入** `load_real_data`：全几何用 odd_transforms.json（18824 面，固定参考约束径迹）+ misalignment_applied.json 覆盖 2268 目标面为 freed 起点；帧约定 `g2l(g_hit)≈(l_x,l_y)` 误差<6e-5mm 验证。
- **Stage 3 种子+拟合**：种子用矢量化 Adam 把螺旋拟合到 ACTS 干净点 g_smt（曲率符号自动学对，提取期用 ACTS 合规）；`real_fit` 联立优化全部径迹螺旋 + 2268 目标面 6-DOF（非目标面固定定 gauge），1/σ² 加权，1D strip 只用 u。

### 决定性结果（98958 径迹，每目标面 63 hit）
| 层/DOF | corr | recovery | \|true\| | \|fit\| |
|---|---|---|---|---|
| vol17 lay8 **tx** | **+0.997** | **+0.933** | 0.523mm | 0.489mm |
| vol24 lay4 **tx** | **+0.986** | **+0.899** | 0.520mm | 0.474mm |
| vol17 lay8 ty | +0.683 | +0.204 | 0.524mm | 0.156mm |
| vol24 lay4 ty | +0.321 | +0.491 | 0.513mm | 0.784mm |

- **tx（精测方向 σ=0.038mm）corr 0.99、恢复 90–93%**——基本解出对齐。vs per-surface corr 0.003 = 数量级突破。
- ty 较弱是**物理限制**：strip v 方向 σ~0.5mm（差 15×）+1D strip 不约束 v。
- 统计量 2000→99k 径迹，tx recovery 0.20→0.93，印证"机制对、靠统计量"。
- masked rms 底噪 ~0.85mm = 单螺旋模型无法拟合多次散射（KF 平滑径迹非单螺旋）；对齐信号靠按面平均的相干分量提取，不受随机 MS 影响。
- 结果存 `global_align_result.npy`（[18824,6] 局部 6-DOF 修正）。

### 结论
**standalone 无监督联立拟合是正解**，真实数据上 tx 恢复 corr 0.99。这是项目目标（不依赖 ACTS、可移植）的可行实现。后续：导出对齐几何 JSON → ACTS 重建验证残差改善（对标 ACTS alignment 基线）；ty/MS 底噪为已知物理限制。

---

## Iteration 7（续）— 重建验证：方法端到端成功
**日期**: 2026-06-04

### 关键修正：真实错位是 tx,ty 平移 + rx,ry 倾斜
全 6-DOF 分解真实错位（之前一直漏看旋转）：
- vol17/24：|tx|≈|ty|≈0.52mm，**|rx|≈|ry|≈0.0147rad（倾斜！）**，tz≈0.011mm、rz≈9e-5（可忽略）。
- **rx,ry 倾斜才是 2mm 残差的主导**（0.015rad×模块半长~100mm≈1.5mm）。早先 Iteration 6 和初版 global_align 只拟合平移/rz，漏了 rx,ry → 恢复 tx 对残差无改善。
- 修复：active_dof=(tx,ty,rx,ry)；alignment 拆平移/旋转**分组学习率**（旋转 lr=平移/15，解决尺度不匹配的过冲/欠收敛）。

### 联立拟合 score（98958 径迹，1000 迭代）
| 层/DOF | corr | recov | \|fit\|/\|true\| |
|---|---|---|---|
| vol17 tx | 0.998 | 0.96 | 0.50/0.52mm |
| vol17 rx | 0.852 | 0.47 | 0.008/0.015rad |
| vol24 tx | 0.995 | 0.98 | 0.51/0.52mm |
| vol24 rx | 0.934 | 1.02 | 0.016/0.015rad |
| ry(both) | 0.49–0.53 | 0.33–0.41 | 部分恢复 |
| ty(both) | 弱(strip v σ~0.5mm) | — | 噪声 |

### 决定性验证：ACTS 重建残差 res_eLOC0 (u 方向)
| 几何 | v17l8 MAD | v24l4 MAD |
|---|---|---|
| nominal | 0.0068 | 0.0117 |
| **misaligned** | **2.0496** | **7.2730** |
| ACTS-aligned(基线) | 0.0069 | 0.0155 |
| **GLOBAL-ours** | **0.0307** | **0.0429** |

**核心残差降 67×(v17)/170×(v24)，逼近 ACTS 基线（差 ~3–4×）。** rms 仍高是外点尾巴（ACTS 基线 v24 同样 rms 3.8/MAD 0.016）。

### 结论
**standalone 无监督联立拟合对齐器端到端成功**：纯 torch、推断不依赖 ACTS、可移植，在真实数据上把目标层残差核心压低近两个数量级，逼近 ACTS Kalman 对齐基线。这是项目目标的可行实现。
剩余差距（核心 0.031 vs 0.007、rms 尾巴、ry/ty 欠恢复）来自：单螺旋径迹模型的 MS 底噪、ry 可观测性弱、strip v 分辨率、无鲁棒损失压外点——均为已知可改进项，非方法缺陷。

---

## Iteration 8 — 鲁棒损失尝试 + GPU 加速 + ry 可观测性结论
**日期**: 2026-06-05

### GPU 加速（已落地）
- `seed_helices` 矢量化（删 99k 径迹 Python 循环）；`seed_helices/real_fit` 加 device/dtype 参数；CLI `--gpu`（cuda+float32）。
- 全量 fit：CPU float64 ~12min → **GPU float32 132s（~6×）**，recovery 与 float64 **逐项完全一致**（float32 无损）。RTX3050 FP64 跛，必须 float32。

### Huber 鲁棒损失：负结果（但有价值）
目标=压低重建 res0 的 rms 外点尾巴（GLOBAL rms 9.6 vs ACTS 3.8）。加 Huber（warmup 500 后切换）后**反而变差**：v17 MAD 0.031→0.081、外点 47.7%。
**诊断（决定性）**：尾巴**不是坏轨迹**，而是**残留 ry 倾斜**——`corr(res_eLOC0, l_x)=+0.74`（res_u∝u 是绕 v 轴倾斜的 signature）。残留 ~0.01rad 倾斜 × 模块边缘 ±100mm ≈ ±1mm，模块外围一半 hit 残差 >0.5mm。Huber 恰好下压这些**携带 ry 信号的边缘 hit** → rx/ry 恢复变差。**Huber 在此场景原理性不适用**（与 Iteration 6 IRLS 同类错误）。

### ry 可观测性是剩余瓶颈
- 加长训练（GPU 便宜，2000it=4.4min）：**vol24 ry recov 0.41→0.81**（短 strip，u 范围够）；但 **vol17 ry 过冲**（|fit|0.027 vs 真值0.015，recov 噪声化）→ v17 MAD 0.031→0.50 反而大坏。
- 根因：ry（res_u∝u）的 lever arm 在 **u 方向（模块窄边）**，vol17 像素模块 u 更窄 → ry 欠约束，多迭代从"欠恢复"滑向"过冲噪声"。**1000 迭代是 vol17 甜点**（ry 欠恢复但不过冲）。
- 结论：vol17 ry 受**探测器可观测性**限制，非算法 bug。ACTS 全 KF 用完整协方差/双坐标联合/MS 模型，对 ry 处理更好——这是与 ACTS 基线剩余差距的主因。

### 最终交付配置
`global_align.py --real --gpu`（1000 迭代，active_dof=(0,1,3,4)，分组 lr，无 Huber/无 reg）：
- score：tx recov 0.96/0.98、rx 0.85/0.93、ry 0.33/0.41（vol17 不过冲）。
- 重建 res_eLOC0 MAD：misaligned 2.05/7.27 → **GLOBAL 0.031/0.043mm**（降 67×/170×），核心逼近 ACTS(0.007/0.016)；rms 尾巴（残留 vol17 ry）为可观测性限制。

---

## Iteration 9 — `ml_alignment_autograd.py --track-mode free` 联立优化接入
**日期**: 2026-07-14

### 实现
- 默认 `surface` 路径保持不变；新增 `free` 分流，在完整 18824 面 nominal 几何上覆盖 2268 个输入目标面，所有有效层 hit 约束径迹，仅目标面 `tx,ty,rx,ry` 可训练。
- 新增稳定径迹分组/padding、1D LOC1 mask、原始测量方差、可微螺旋、12 步 Newton 平面求交、矢量化 smoothed-point warm start、目标面-only JSON 导出和六 DOF truth 指标。
- ODD 外层横向弧长实测最大约 1025mm；原草案 `S_MAX=600mm` 会误删 38% hit，修订为 **1200mm** 后全量求交剔除率为 **0%**。
- 曲率使用 `rho_scaled=rho/1e-4` 优化，避免物理曲率被 Adam 单步更新 `1e-3` 后放大约 24 倍。全量学习率 sweep（100 步）得到 track lr 0.01/0.03/0.1 的 chi2 分别为 452/71/763，因此 joint 默认采用 **0.03**。

### 全量结果（61702 条含目标层径迹，854904 hit，GPU float32，1000 iter）
| 层/DOF | corr | recovery | true RMS | fit RMS |
|---|---:|---:|---:|---:|
| vol17 tx | 1.000 | 1.000 | 0.5205mm | 0.5204mm |
| vol17 rx | 1.000 | 1.001 | 0.0148rad | 0.0148rad |
| vol17 ry | 0.996 | 0.996 | 0.0148rad | 0.0148rad |
| vol24 tx | 1.000 | 0.999 | 0.5229mm | 0.5226mm |
| vol24 rx | 0.995 | 1.011 | 0.0147rad | 0.0149rad |
| vol24 ry | 0.978 | 1.010 | 0.0146rad | 0.0154rad |

- chi2/valid-component：7480.17 → 42.99；无效求交率 0%。
- 输出：`alignment_output/Vol17_Lay8__Vol24_Lay4/aligned_transforms_joint_free_v17l8_v24l4.json`，确认仅含 2268 个目标面。
- 代码验收：`pytest tests/` 为 40 passed；`tests/test_run_ml_iterative.sh` 与 `py_compile` 通过。

### ACTS 重建状态
当前环境 `/cvmfs` 未挂载，`setup_acts.sh` 自动挂载需要交互式 sudo 密码，LCG_109/`acts.ActsPythonBindings` 无法加载，因此本轮未生成 `joint_free` 重建 ROOT，也未宣称新的 MAD。待 CVMFS 可用后执行 100k event aligned reconstruction 和 residual plotting，目标仍为 vol17/24 res_eLOC0 MAD 约 0.03–0.05mm。
- Review follow-up: `JOINT_LR_TRACK=3e-2` and `S_MAX=1200mm` are intentional empirical defaults; the original plan baseline remains reproducible through CLI overrides. Free-mode diagnostics now reuse `print_per_layer_residual_diagnostics`, and the toy closure covers tx/ty/rx/ry plus a masked strip component and frozen-track control.

---

## Iteration 10 — nominal measurement holdout：helix 内推/双向外推验证
**日期**: 2026-08-21 | **样本**: 每种模式最多 1000 个目标 hit | **拟合**: SMT seed + 非目标层 measurement refit，200+200 iter，CPU float64

### 与旧验证假设的区别
- 旧 `validate_helix_model.py` 以 `prediction-eLOC_smt` 为局部精度指标；本轮改为要求的 **`prediction-l_{x,y}_hit`**。目标层 measurement 和 SMT 均不进入 seed/refit，避免标签泄漏。
- 对同一目标 hit 主动构造 `interpolation`、`forward_extrapolation`（仅较小 pathLength）和 `backward_extrapolation`（仅较大 pathLength）。实际数据证明 8 hit/300mm 无法形成 forward 样本，因此外推采用独立最低门槛 **4 hit/100mm**；内推仍为 8 hit/300mm。
- 输入 SHA256：nominal trackstates `faed97457fd5af41880e25cad176e443914235779be8b9adddd0ef3a22443cd8`；nominal geometry `b409bccfbec3008a37e9aae57a66dad7c2cb4c79ab2b31071aff6a8f760c8680`。

### 结果：prediction − nominal local measurement
| 层 | 模式 | N | LOC0 bias / RMSE / P95 [mm] | LOC1 bias / RMSE / P95 [mm] | 方向 RMS [rad] |
|---|---|---:|---|---|---:|
| 17:8 | interpolation | 368 | -0.004 / 0.083 / 0.163 | -0.001 / 0.034 / 0.061 | 0.00048 |
| 17:8 | forward | 262 | +0.020 / 0.227 / 0.363 | -0.004 / 0.058 / 0.100 | 0.00310 |
| 17:8 | backward | 500 | -0.001 / 0.579 / 1.108 | +0.038 / 1.704 / 3.001 | 0.00501 |
| 24:4 | interpolation | 632 | +0.004 / 0.170 / 0.311 | +0.005 / 1.158 / 2.260 | 0.00032 |
| 24:4 | forward | 738 | -0.015 / 0.352 / 0.623 | -0.006 / 1.172 / 2.303 | 0.00273 |
| 24:4 | backward | 500 | +0.025 / 0.903 / 1.494 | -0.027 / 4.433 / 8.810 | 0.00863 |

所有分组求交失败率均为 **0%**，但绝对 RMSE 门槛未通过。内推精度最好，forward 外推约恶化 2–3 倍，backward 外推最差；各组 bias 接近零而 STD/RMSE 很大，说明主要是逐轨迹扩散而不是整体偏置。尤其 24:4 LOC1 的 measurement-normalized RMS 在内推/forward 约为 0.96/0.98，符合粗粒度 strip 分辨率；pixel LOC0 和 backward 模式远大于 1，单螺旋对 measurement 的预测误差明显超过测量误差。

### 验收结论
- **FAIL（有效负结果）**：当前 helix 模型不能满足 pixel 0.05mm、strip LOC0 0.10mm 的既定门槛；只有 17:8 interpolation LOC1 和 24:4 interpolation/forward LOC1 达到对应绝对门槛。
- 失败不是 surface intersection 数值问题，而与单螺旋忽略材料效应/多次散射、单侧窗口 lever arm 和传播方向不对称一致。本轮只验证模型，不调参或修改 helix 数学形式。
- 完整 manifest、summary、CSV 和 plots 位于 `data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/measurement_holdout_20260821/`；manifest 状态为 `completed`，validation `passed=false`。

---

## Iteration 11 — surface warm-start + non-target initialization + joint free-helix（实现阶段）
**日期**: 2026-08-24 | **状态**: implementation complete，ACTS 物理验收待运行

### 相对 Iteration 10 改变的假设
- Iteration 10 是逐目标 hit 的 nominal holdout 预测，且证明单螺旋单侧外推精度不足；本轮不重复该实验。新假设是先固定执行两轮 surface warm-start，再用整条轨迹的全部非目标层（至少 8 hit、300 mm）进行全局 SMT seed 和 measurement refit，最后在全部 measurement 上联合释放轨迹与双目标层几何。
- nominal/truth 不再提供门控、候选层或验收基准；物理验收改为候选几何重新 ACTS reconstruction 后的 measurement/smoothed-state `chi2/N <= 1.0`。
- 活动自由度按本轮固定配方设为 `(tx,ty,rz)`；这与历史 Iteration 7--9 的 `(tx,ty,rx,ry)` 实验假设不同，结果不得直接横向冒充复现。

### 已实现
- 抽取 ACTS-independent `helix_model.py`，统一六参数 helix、Newton 最近点/平面求交、曲率缩放和数值容差。
- free 模式新增非目标层 `fit_mask`、跨度筛选、SMT seed、measurement χ² refit、初始化统计和三阶段逐层 χ²；联合阶段仍使用全部有效 measurement，保留 checkpoint、patience 和 5% 无效求交上限。
- 新增 `reconstruction_acceptance.py`，对缺层、零分量、非有限值和 ROOT 字段不完整 fail closed；polish 使用固定字典序评分。
- 编排固定两轮 warm-start、一次 joint free-helix 和最多两轮 gain=0.2 surface polish；旧 nominal/candidate CLI 仅警告并忽略。每个候选使用独立文件名重建和验收，未改善时保留上一版。
- manifest 从运行开始创建，退出 trap 记录异常终止；最终 summary 包含初始化、逐层 χ²、验收、停止原因和最终选择。

### 尚未宣称的结果
- 本条不包含新的算法指标或 ACTS 验收结论。只有在独立 ACTS shell 完成 smoke/formal reconstruction、双层所有有效 LOC0/LOC1 均 `chi2/N <= 1.0` 且 manifest 完成后，才能追加物理成功结论。

---

## Iteration 12 — trial-006 失效分析与 block-LM 改版
**日期**: 2026-08-25 | **状态**: implementation complete，trial-007 待运行

### trial-006 的决定性证据
- 两轮 surface 将四个目标分量 χ²大幅降低，但 joint 输出与 warm2 JSON 的 SHA256
  完全相同。joint 最佳点停在 iteration `-1`：初始 loss `16.049`，iteration 100
  升至 `86.816`，所有更新 checkpoint 都比初始状态差，最终恢复零修正。
- 旧 `JOINT_LR_TRACK=0.03` 来自初始 χ²约 7480、GPU float32、不同 active DOF 的
  Iteration 9；trial-006 已经是 warm2 + measurement-refit 后的 CPU float64 近解。
  同一 Adam 步长同时作用于 mm 位置、rad 方向和无量纲 `tan_lambda`，首步方向变化在
  数百 mm 杠杆臂上可放大为多 mm，不能作为细校准默认。
- nominal 重建自身的 `17:8 LOC1 chi2/N=1.333`，证明统一绝对阈值 1.0 不是该统计量
  在当前重建模型下的自洽几何基准。trial-006 final 虽继续改善，仍以四分量失败结束。

### 与 Iteration 11 不同的新假设
- 轨迹是由非目标层确定的 nuisance parameter；目标层 measurement 只估计目标层几何，
  不再同时拉动轨迹。几何每 sensor 仅有三个自由度，采用聚合加权法方程比逐 hit Adam
  更符合其可观测结构。
- 事件固定切分为 80/10/10。calibration 用 nominal states 一次性冻结逐分量预测阈值并
  选择候选；validation 只对最终候选评估一次。truth geometry recovery 仍不进入控制流。

### 实现
- measurement refit 将位置、方向和缩放曲率学习率拆为 `1e-3/1e-5/1e-3`，保存初始
  checkpoint，并记录初始/最佳 χ²及三组参数更新 RMS。
- free 默认 geometry solver 改为固定轨迹的逐 sensor block-LM；Fisher 使用物理尺度
  归一化和 `1e-4` 特征值下限，只有实际目标 χ²下降且无效求交不超过 5% 才接受步骤。
- 新增稳定事件切分、nominal bootstrap acceptance profile、归一化 polish score 和
  blind-validation 编排。trial-007 只有四个 validation 分量全部通过且 joint 有非零
  有效更新时才可标记通过。
