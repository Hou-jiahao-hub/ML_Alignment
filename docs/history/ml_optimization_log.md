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
**日期**: 2026-08-25 | **状态**: implementation complete；trial-007 已于同日运行完毕并
**未通过**，结果补记在 Iteration 23

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

---

## Iteration 13 — 材料感知 broken-helix：Gate 0/1/2/3 实现与验收
**日期**: 2026-08-25 | **状态**: Gate 0 **有条件通过**（能损条目仍 open），Gate 1 契约冻结，
Gate 2/3 closure 通过，Gate 4 未做

### 相对 Iteration 12 改变的假设
- Iteration 10 已证明单 helix 的 holdout RMSE 下限来自未建模的多重散射（bias≈0、
  RMSE 大 1–2 量级）。本轮按 `docs/research/material_aware_tracks/decision_record.md`
  实现 GBL 风格「六参数 reference helix + 材料节点二维 kink」的前三个工作包。
- 本轮**不改** `ml_alignment_autograd.py`，**不回答 joint 校准是否改善**。

### 环境更正
`/cvmfs/sft.cern.ch/lcg/views/LCG_109` 已挂载、ACTS 绑定可用。Iteration 9–12 记录的
"ACTS 重建被阻塞"不再成立。

### 决定实现形态的 ACTS 源码事实
| 事实 | 出处 |
|---|---|
| 材料 ROOT 丢弃 track 索引且遍历 `unordered_map` | `RootMaterialTrackWriter.cpp:87` |
| `v_px/v_py/v_pz` 是 `parameters.direction()`，**单位向量**，无动量信息 | `RootMaterialTrackIo.cpp:189-191` |
| `mat_step_length` 已含 `pathCorrection` | `PointwiseMaterialInteraction.cpp:45` |
| `RootPropagationSummaryWriter` 提供 `(event_nr, track_nr)` + 5 个 start bound 参数 | `RootPropagationSummaryWriter.cpp:54-77` |
| `nMaterials` 只数 stepping summary 中有 surface material 的 step | 同上 `:141-151` |
| `sterileLogger=True` 会清空 `summary.steps` → `nMaterials` 恒为 0 | `PropagatorInterface.hpp:89,123` |
| volume material 节点 `sur_id` 写成 0 | `RootMaterialTrackIo.cpp:245` |
| ACTS Highland 用逐 slab `x/X0` 进 log 项 | `Interactions.cpp:484-491,516` |

因此三方 join 必须走 `material ↔ propsummary ↔ tracksummary`；`nMaterials` 只能作诊断。

### 实现
- `alignment_odd.py`：新增 `material` 子命令。**复用 `runReconstruction` 同一函数**
  （加 `recordMaterial` 参数）而非复制链路，保证 geometry/field/seed/KF/TrackSelector
  逐项一致；states/summary/material/propsummary 由同一次运行产出。
  新增 `reserve_reconstruction_output()` 覆盖保护（ACTS writer 全是 RECREATE）；
  Sequencer 启动前建 running manifest，异常时 finalize 为 `failed`。
  记录 pass 用 `energyLoss=False, multipleScattering=False`（与恒定曲率首版边界自洽）。
- `material_artifact.py`（新建，ACTS-free）：schema 1.1（新增 `volume_geometry_id`）、
  Join A 五参数硬校验、Join B 事件内双射、双向传播覆盖校验、`t_X0` closure、
  相对增量路径 closure、先聚合后剔除、按材料占比而非节点数的丢弃门槛、
  聚合 `Σθ0²` 变化上界、全部 fail-closed + manifest。
- `helix_model.py`：只新增 `highland_kink_sigma`（与 `Interactions.cpp:484-491` 逐项对齐，
  含 `q²/β²`）、`node_arc_lengths`、`segment_origin_path`、`assign_hit_segments`、
  `broken_helix_segments`、`broken_helix_local_path`、`broken_helix_track_parameters`；
  既有函数一行未改。新函数返回 `[T,H,6]`，正是现有 forward 的 `expand` 形状，下游不用改。

### 坐标契约（本轮第一个关键设计修正）
helix 的 `s` 原点由 `(x0,y0,z0)` 决定，**不是轨迹的全局不变量**；validation 每一行按
自己的 fit window 拟合各自的 reference。因此：
- artifact 的 `path_length_mm` 定义为**累计横向弦长**，仅用于排序与完整性诊断，
  不参与任何下游裁剪或段分配（manifest 中显式注明）。
- 消费方必须用自己冻结的 reference 从 `position_mm` 重算 `node_s` 与 `hit_s`。
- `node_arc_lengths` 返回 detached 值；`assign_hit_segments` 返回 detached int64。
- 回归测试 `test_reference_reparameterisation_leaves_prediction_invariant`：
  同一轨迹用两个不同 `s` 原点的 reference，段分配一致、预测点 atol 1e-9 一致。

### Gate 0 验收（1000 events，`--output-suffix mat0 --record-steps`）——**有条件通过**
| 硬门控 | 结果 |
|---|---|
| 传播失败数 | **0**（tracksummary 与 propsummary 逐事件计数完全相等） |
| Join A 五参数校验 | 100% 通过 |
| Join B 事件内双射 | 100% 成立，unmatched=0 |
| `mat_*` jagged 分支长度一致 | 通过（992/992 轨迹） |
| 方向归一 / `mat_X0>0` / `x/X0` 有限非负 | 通过 |
| `t_X0` closure | max 1.12e-7 ≤ 1e-4 |
| 三维路径 closure（相对增量） | max 2.27e-4 ≤ 1e-3 |
| 每轨迹丢弃材料占比 | max 1.66e-4 ≤ 1e-2 |
| 聚合 `Σθ0²` 变化 | 0.0 ≤ 2e-2 |

产物：992 轨迹 / 13508 节点（原始 13513，丢弃 5 个薄节点）。

**perigee 假设通过率 = 1.0** → 按计划升格为 Join B 硬校验
（`PERIGEE_HYPOTHESIS_REQUIRED = True`，`(v_x,v_y,v_z) == (-d0 sinφ, d0 cosφ, z0)`）。

**判为「有条件」的唯一原因见下一节：能损条目未关闭。**

### Gate 0 的 open 条目：能损对节点位置的影响（**未关闭，推迟到 Gate 4 判定**）

> **后续状态（Iteration 14）**：该条目已于 2026-08-25 的 Gate 4 复检中**关闭**，判据全部满足。

计划要求「`energyLoss=True/False` 的节点位置差 ≪ 传感器分辨率」。200 事件对照实测：

| 区域 | n | max 位置差 [mm] | p99 |
|---|---:|---|---|
| r<200（像素） | 1114 | 1.54e-3 | 2.57e-4 |
| 200<r<700 | 1100 | 5.23e-2 | 1.48e-2 |
| 700<r<1200（条形） | 494 | **2.44e-1** | 1.54e-1 |

**按原文措辞不通过**：条形层 0.244 mm 不是 ≪ 像素分辨率。节点位置只通过两条路径
进入模型，二者的间接估计都远小于该位置差本身：
1. **段分配**：最大偏差 / 最小节点间距 = 3.4e-2，量级上不可能翻转段归属；
2. **kink 到下游 hit 的杠杆臂**：θ0 × 0.24 mm ≈ 1e-3 rad × 0.24 mm ≈ **2.4e-4 mm**。

`x/X0` 差异 max 2.2e-5（可忽略）。

**但以上是量级论证，不是对模型输出的直接检验，因此该门控条目保持 open。**
生产配置暂用 `energyLoss=False`，消融开关 `--energy-loss` 与本表一并留在仓库中。
Gate 4 必须在**真实 holdout 行**上用两套节点（`energyLoss=True` / `False`）
直接复检，判据在此**预注册**：

| 复检量 | 定义 | 通过条件 |
|---|---|---|
| 段分配翻转率 | 两套节点各自算 `segment_of_hit`，逐 holdout 行比较 | **== 0** |
| 预测位移 | \|Δ 预测局部坐标\| 相对该分量测量 σ 的比值 | **p99 ≤ 10%** |

两条同时成立 → 该条目关闭、Gate 0 记为完全通过；任一不成立 →
`energyLoss=False` 的生产选择必须重新评估（改用 `energyLoss=True` 导出，
或把能损纳入模型），**不得沿用本轮的量级论证结论**。

### 本轮发现并修复的两个数据契约缺陷
- **D1 — manifest 互相覆盖（数据丢失）**：导出端（`alignment_odd.py`）与转换端
  （`material_artifact.py`）都把 `manifest.json` 写进 `<run_tag>/`，
  后跑的转换端**静默覆盖**了导出端 manifest（其中含 `exporter_config` 与
  ACTS 输入哈希）。`gate0-20260825` 与 `gate0-smoke` 的导出记录已实际丢失。
  修复：
  1. `experiment_manifest.write_experiment_manifest` 增加 `_guard_existing_manifest`，
     检测到目标 manifest 的 `command` 属于**另一个 stage 脚本**时抛 `FileExistsError`
     ——从根因上禁止跨阶段覆盖，而不只是改路径；
  2. 路径分层为 `<run_tag>/export/manifest.json` 与 `<run_tag>/convert/manifest.json`；
  3. 旧目录保留为 `*__pre-layout-fix`（未删除），导出重跑后记录已补齐。
- **D2 — 路径清单模板错误**：`path_inventory.py` 声明的材料 ROOT 名为
  `material_kf_{run_tag}.root`，而实际文件名由 `--output-suffix` 决定。
  更糟的是我最初写的测试**断言了这个错误行为**。已同时修正模板与测试，
  测试改为绑定 `artifact_filename`，不再复制常量。

### 本轮证据的已知局限（不得外推）
- 该仿真样本**每事件只有 0 或 1 条轨迹**（`max_tracks_per_event = 1`），
  因此 Join B 的歧义消解在真实数据上**从未被触发**，只有合成单测覆盖。
- ODD 材料**全部是 surface material**，volume 节点总数为 0；
  且无连续相同 `sur_id`（无 pre/post split）。聚合与体材料分组逻辑在真实数据上是 no-op，
  仅由单测覆盖，属防御性代码。
- `nMaterials`（13–18，均值 15.4）与 surface 节点数（9–17，均值 13.6）仅 8.97% 相等，
  最大差 6 —— 与 F5/F6 的语义差异一致，证实不能作为门控。

### Gate 2 — 规范固定（gauge fixing）：本轮第二个关键设计修正
broken-helix 的 MAP 拟合最初**不收敛**：60 次迭代后 Hessian 条件数 4.6e13、
最小特征值 7.4e-6、梯度范数仍约 1e4。

我最初把原因归给「toy 平面法向与轨迹近乎平行」，据此把平面改成径向——**这条归因是错的，
已撤回**：改完毫无改善。真实原因是 **reference helix 的六参数存在精确规范自由度**：
沿轨迹平移原点 `ds` 同时把 `psi0` 旋转 `rho·ds`，描出的是**同一条曲线**。
冻结的 `node_s` 只在 O(kink) ≈ 3e-4 的量级上打破这个简并，因此留下一个近零本征方向。

修复：`reference_gauge_frame(reference)` 用初始方向的正交补给出 `(origin, axis0, axis1)`，
参数化从 6 个降为 5 个 —— 位置写成 `origin + u·axis0 + v·axis1`，
沿轨迹方向的平移被显式消掉。参数向量为 `[u, v, psi0, tanl, rho_scaled, kinks...]`。

| 量 | 修复前 | 修复后 |
|---|---|---|
| 条件数 | 4.6e13 | 1.4e6 |
| 最小特征值 | 7.4e-6 | 262 |
| 收敛后 max\|grad\| | 384 | 1.2e-6 |
| 迭代次数 | 60（未收敛） | 3 |

相应地测试断言也从「拟合出的 reference 参数等于真值」改成**规范不变**的判据
（refit 的局部预测复现测量，atol 1e-4 —— 该量级由软方向决定，底噪 3.7e-5 mm）。
`predictive_covariance` 同时改为 batch-aware（`[T,P,P]`）。

### Gate 2 — 合成 MC ensemble closure（**通过**）
`MC_ENSEMBLE_SIZE = 1200`、`MC_SEED = 20260825`，kink σ = 3.2756e-4 rad、
测量 σ = 0.05 mm，每个成员按已知 `Q_j` 抽 kink → 生成测量 → 阻尼 Gauss–Newton 拟合
→ 在 holdout 面上取预测与预测方差。容差全部 Gate 1 预注册。

| 量 | 结果 | 预注册容差 |
|---|---|---|
| 拟合收敛 | 3 次迭代，max\|grad\| 5.166e-3 | — |
| holdout 残差 RMS | 0.12436 mm | — |
| 预测 σ（LOC0 / LOC1） | 0.15462 / 0.09015 mm | — |
| pull mean（LOC0 / LOC1） | −0.0093 / +0.0134 | \|·\| ≤ 0.06 |
| pull std（LOC0 / LOC1） | 0.9868 / 0.9716 | [0.93, 1.07] |
| 覆盖率 @1σ | 0.7033 / 0.6983 | 0.6827 ± 0.03 |
| 覆盖率 @1.96σ | 0.9542 / 0.9575 | 0.9500 ± 0.03 |

这是「单次已知 kink 恢复不算证明」的正式替代品：pull 的**均值与宽度同时**落在带内，
说明预测协方差的绝对尺度正确，而非仅偏置为零。

### Gate 3 — predictive covariance（**通过**）
`validate_helix_model.py` 新增 `gauss_newton_hessian` 与 `predictive_covariance`：
- 显式声明目标为 `F = ½χ²`，故 `H = JᵀV⁻¹J + blkdiag(Q⁻¹)` 即后验精度，无额外因子；
  测试用 `torch.autograd.functional.hessian` 对 `½χ²` 二阶求导逐元素对拍（atol 1e-9）。
- 禁止显式求逆：Cholesky 分解 + `cholesky_solve`；测试用源码检查钉死这一点。
- **factor-of-two 的独立验证**：无先验时 MAP 估计量的抽样协方差**就是**后验协方差，
  故用 40000 次噪声实现的经验协方差交叉检验绝对尺度——它不复用被测公式。
  结果：矩阵范数偏差 < 5%，对角比值落在 [0.95, 1.05]。差两倍会是 100% 的偏离。
- 诊断：对称性残差、Cholesky 成败、min/max 特征值、条件数。
  奇异 Hessian 在 `damping=0` 时如实抛 `LinAlgError`，加阻尼后可解。
- `PRODUCTION_DAMPING = 1e-10` 与 `DAMPING_SCAN = (0, 1e-12, 1e-10, 1e-8, 1e-6)`
  在 Gate 1 预注册；测试断言生产值在网格内、网格有序，且阻尼只会缩小方差不会放大。

### 测试
新增 `tests/test_material_artifact.py`（23 项，含合成三表 ROOT 端到端往返，验证行序
被打乱后仍落到正确轨迹）、`tests/test_helix_model.py` 11 项新用例
（Highland 与 ACTS 逐项对拍、`x/X0==0` masked limit 精确为 0 且梯度有限、
节点拆分的解析比值、zero-material 极限含梯度、两个 kink 分量等角度、
分段位置连续、padding 恒等传递、段分配确定性、坐标契约）、
`tests/test_validate_helix_model.py` 9 项（Gate 3 协方差 6 项、Gate 2 拟合与
MC ensemble 3 项）、`tests/test_experiment_manifest.py` 2 项（D1 跨阶段覆盖守卫）、
`tests/test_path_inventory.py` 1 项（D2 模板）。
`make test`：改动前 210 passed / 35 failed，改动后 **255 passed / 35 failed**
——新增 45 项全绿，零新增失败。既有 35 项失败与本轮无关
（它们用 `/usr/bin/python` 3.9 起子进程，导入即失败；已用 stash 复核确认）。

### 尚未宣称的结果
- Gate 0 的**能损条目仍 open**，判据已预注册，等 Gate 4 用真实 holdout 行复检。
- Gate 4（`--model both` 单进程双臂 paired holdout、bootstrap CI 门控、
  `holdout_keys.csv` + SHA256、必需组 `N≥200` 判定）**未实现**。
- 因此**本轮没有任何「broken-helix 优于 single helix」的结论**，也没有 joint 校准结论。
  Gate 2/3 只证明了数学实现与协方差尺度自洽，这是必要条件、不是物理结论。

## Iteration 14 — Gate 4：全量 paired holdout（**未通过**）与 Gate 0 能损条目关闭

**日期**：2026-08-25
**状态**：Gate 0 **完全通过**（能损条目已关闭），Gate 4 **执行完毕、判定 failed**。
**结论方向**：broken-helix 的改善是**真实但微小**的，与预注册门槛差一个量级；
外推组无改善甚至回退。**首版材料感知模型按预注册标准未通过。**

### 运行配置

```
python scripts/validate_helix_model.py \
  --states data/reconstruction/nominal/Vol17_Lay8__Vol24_Lay4/trackstates_kf_mat1.root \
  --material-artifact       .../gate4-20260825/material_nodes.npz \
  --material-artifact-energy-loss .../gate4-20260825-eloss/material_nodes.npz \
  --model both --max-tracks 0
```

- ACTS 导出 **20000 events**（非计划书写的 100000）；两次导出各约 4 分钟。
- artifact：19781 tracks / 268434 nodes（eloss 版 268436），schema 1.1，Gate 0 全部 fail-closed 检查通过。
- holdout 行数 **70352**，`n_rows_missing_material = 0`，`on_missing_material=fail` 未触发。
- `holdout_keys.csv` SHA256 `4ecf9804…4398`；两臂 key 清单逐行相同（镜像文件摘要相等断言通过）。
- 窗口内节点数 min/median/max = 0 / 7 / 13。
- 单次运行耗时约 53 分钟（35 个 2048 行 chunk）。

**样本量说明**：计划书写 100000 events，实跑 20000。但六个必需组的有效分量数为
5034–14324，全部远超预注册的 `N ≥ 200`，因此**判定在统计上有效**，
不构成 `inconclusive`。样本量只影响 CI 宽度，而失败原因是点估计本身离门槛太远。

### 收敛与稳定性（先判数值，再判物理）

| 臂 | iterations | 初始 objective | 最终 objective | max‖g‖ |
|---|---|---|---|---|
| single | 32 | 18.8299 | 2.3758 | 7678.7 |
| broken | 37 | 18.8299 | 2.2969 | 3527.2 |

`max_relative_decrease` 5.1e-11 / 9.2e-11，`converged: true`。但**「相对下降 < 1e-10」
对停滞行同样成立**，`max‖g‖` 高达数千说明存在个别未到驻点的行。为此新增
**Newton decrement** `λ² = gᵀH⁻¹g`（尺度无关，白化梯度范数不是）：

| 臂 | p50 | p90 | p99 | max | 非驻点行占比 |
|---|---|---|---|---|---|
| single | 4.5e-23 | 1.9e-15 | 3.4e-13 | 14.93 | **1.71e-4** |
| broken | 1.6e-19 | 3.9e-15 | 7.8e-13 | 0.538 | **8.53e-5** |

阈值 `NEWTON_DECREMENT_STATIONARY = 1e-6`，预注册上限
`MAX_NON_STATIONARY_ROW_FRACTION = 1e-2`。两臂都低两个量级
→ **本轮的 failed 是物理/模型结论，不是数值失败**，可以据此判定。
（该门控与阈值在读取结果**之前**写入代码并跑测试，非事后设定。）

### 分组结果（production damping 1e-10，paired bootstrap 1000 次，seed 20260825）

| 组 | N | RMSE single | RMSE broken | ΔRMSE 95% CI | ΔNLL 95% CI | pull_std single→broken |
|---|---|---|---|---|---|---|
| 17:8 interp LOC0 † | 8346 | 0.021723 | 0.021470 | [−1.84%, **−0.60%**] | [−0.0295, −0.0129] | 1.042 → 1.001 |
| 17:8 interp LOC1 | 8346 | 0.032497 | 0.031464 | [−5.29%, −1.40%] | [−0.127, −0.0615] | 1.126 → 0.988 |
| 17:8 fwd LOC0 † | 5034 | 0.213453 | 0.213394 | [−0.058%, +0.003%] | [−0.0004, **+0.0090**] | 0.987 → 0.992 |
| 17:8 fwd LOC1 | 5034 | 0.070582 | 0.070352 | [−0.90%, +0.13%] | [−0.102, −0.0378] | 1.115 → 1.000 |
| 17:8 bwd LOC0 † | 14190 | 0.187964 | 0.187950 | [−0.019%, +0.003%] | [−0.0038, −0.0006] | 0.998 → 0.982 |
| 17:8 bwd LOC1 | 14190 | 1.433637 | 1.433514 | [−0.027%, +0.009%] | [+0.0002, **+0.0038**] | 0.980 → 1.016 |
| 24:4 interp LOC0 † | 14197 | 0.050144 | 0.049868 | [−0.82%, **−0.32%**] | [−0.0094, −0.0037] | 1.003 → 0.984 |
| 24:4 interp LOC1 | 14197 | 1.187356 | 1.185249 | [−0.27%, −0.08%] | [−0.0025, −0.0008] | 0.985 → 0.981 |
| 24:4 fwd LOC0 † | 14261 | 0.259487 | 0.259476 | [−0.016%, +0.005%] | [−0.0011, **+0.0014**] | 0.999 → 0.990 |
| 24:4 fwd LOC1 | 14261 | 1.210019 | 1.210019 | [−0.039%, +0.037%] | [−0.0006, **+0.0003**] | 0.986 → 0.983 |
| 24:4 bwd LOC0 † | 14324 | 0.846021 | **0.864379** | [−0.89%, **+8.69%**] | [+0.0532, **+0.159**] | 0.991 → **1.102** |
| 24:4 bwd LOC1 | 14324 | 3.282800 | 3.299087 | [−1.33%, +2.23%] | [+0.0051, **+0.0136**] | 0.977 → 1.022 |

†＝预注册必需组。全部组 intersection failure rate = 0，|pull mean| ≤ 0.035，
predictive χ²/component ∈ [0.96, 1.27]（均在 [0.7, 1.5] 带内）。

### 失败原因（预注册判据，逐条）

1. **interpolation RMSE 改善量级不足**：门槛「CI 上界 < −20%」，实测
   17:8 −0.60%、24:4 −0.32%。**方向对、统计显著、幅度差 30–60 倍。**
2. **多个组 NLL CI 上界 ≥ 0**：17:8 fwd LOC0、17:8 bwd LOC1、24:4 fwd LOC0/LOC1、
   24:4 bwd LOC0/LOC1。
3. **24:4 backward LOC0 实质回退**：RMSE CI 上界 +8.69%，超出「回退 ≤ 5%」；
   NLL CI 完全在正侧 [+0.053, +0.159]；pull_std 从 0.991 恶化到 1.102。
4. **阻尼敏感性翻转**：8 个外推组的 NLL 判定随预注册 damping 网格翻转。

### 阻尼敏感性——本轮最值得记录的数值发现

`damping = 0` 时两臂 Cholesky 都失败（single 在 batch 元素 214 的 5 阶主子式，
broken 在元素 0 的 26 阶主子式），即 **H 在有限精度下并非严格正定**，
production damping 不是可选项。而 damping 对外推 NLL 的影响**远非中性**：

| damping | 17:8 fwd LOC0 | 17:8 bwd LOC0 | 24:4 fwd LOC0 | 24:4 bwd LOC0 |
|---|---|---|---|---|
| 1e-12 | −0.00066 ✓ | −0.00211 ✓ | −0.00027 ✓ | −0.00121 ✓ |
| **1e-10（production）** | +0.00384 ✗ | −0.00210 ✓ | +0.00024 ✗ | **+0.106 ✗** |
| 1e-08 | +1.144 ✗ | +0.0462 ✗ | +0.153 ✗ | +5.337 ✗ |
| 1e-06 | +15.20 ✗ | +3.947 ✗ | +3.441 ✗ | +29.18 ✗ |

ΔNLL 随 damping **单调恶化数个量级**——阻尼项 `damping·mean(diag H)·I` 在
外推方向上系统性放大 broken 臂的 predictive variance（broken 臂的 H 含
`2N` 个 kink 参数，`mean(diag H)` 被 `Q⁻¹` 块抬高，同一相对 damping 对
broken 的绝对扰动远大于 single）。**这意味着两臂在同一 damping 下并不公平。**

- **interpolation 组不在翻转名单里** → 那两条「改善但微小」的结论是稳健的。
- **所有外推组的 NLL 结论都不可信**，按预注册规则不得宣称任何方向。
- 后续必须先解决「damping 对两臂不等价」，否则外推 NLL 这条指标本身无意义。
  候选：按臂各自的 `mean(diag H)` 归一改为**绝对**阻尼、或对 kink 块与
  reference 块分别加阻尼、或改用 pivoted Cholesky / LDLᵀ 免阻尼求解。

### Gate 0 能损条目——**已关闭**

用第二份 `energyLoss=True` 的 artifact 在**同一批 70352 个 holdout 行**上复检：

| 判据 | 预注册上限 | 实测 |
|---|---|---|
| 段分配翻转率 | 0.0 | **0.0** |
| p99 \|Δpredicted\|/σ (LOC0) | 0.10 | **2.03e-6** |
| p99 \|Δpredicted\|/σ (LOC1) | 0.10 | **1.71e-5** |

max 比值分别 0.375 / 0.056（个别行），p99 低于上限 4–5 个量级。
**「记录 pass 用 `energyLoss=False` 与首版模型自洽」这一假设成立**，
Gate 0 由「有条件通过」升为**完全通过**。

### 意外发现：Gauss–Newton 求解器本身就把单 helix 的插值残差降了 3.5 倍

同一次运行同时产出旧路径（Adam refit）与新路径（GN）的 17:8 interpolation LOC0：

| 求解器 | 同一批 8346 行的 RMSE | 拟合 objective |
|---|---|---|
| Adam（Iteration 10 路径，200 iters） | **0.076261 mm** | 18.83 → — |
| Gauss–Newton single 臂 | **0.021723 mm** | 18.83 → **2.376** |

χ² 降到约 1/8，RMSE 降到 1/3.5。**Iteration 10 的诊断「RMSE 比 bias 高 1–2 个量级
⇒ 未建模的多重散射」在很大程度上其实是优化器未收敛**，不是物理。
这直接解释了本轮 broken-helix 收益为何这么小：GN 收敛之后，
留给材料 kink 去解释的残差已经所剩无几。

无泄漏复核：拟合掩码 `mask = stack((fit_mask, fit_mask & loc1_valid))`
不含 holdout hit，两条路径用的是同一批行、同一 holdout 定义。

### 本轮结论

- **不得宣称 broken-helix 优于 single helix。** 按预注册标准判定 **failed**。
- 可以宣称的：(a) broken 臂**确实修正了一个真实的过散**——
  17:8 interp LOC1 pull_std 1.126→0.988、17:8 fwd LOC1 1.115→1.000、
  interp χ² 1.267→0.976；(b) 插值 RMSE 改善统计显著但仅 0.3–1.8%；
  (c) 24:4 backward 明确回退。
- **优先级已经变了**：先修 Adam → GN 的求解器落差（收益 3.5 倍，已实证），
  再谈材料模型的边际收益（0.3–1.8%）。
- 工作包 ④（joint 校准）按 Gate 5 规则**不得启动**——Gate 4 未通过。

### 尚未宣称的结果（更新）

- 「材料感知模型能否改善 joint 校准」仍**无任何证据**，且本轮不具备启动条件。
- 外推方向的 NLL 优劣**未判定**（阻尼污染），不是「无改善」而是「不可测」。
- 24:4 backward 的回退**尚未归因**：可能是 strip 层 LOC1 弱约束下 kink 吸收了
  几何自由度，也可能纯粹是 damping 不等价的产物——两者本轮无法区分。
- 20000 events 的结论对必需组统计有效，但**未在 100000 events 上复现**。

## Iteration 15 — 测量 refit 换成 Gauss–Newton：求解器落差已消除

**日期**：2026-08-25
**状态**：实现完成、全量复跑完毕。Iteration 14 发现的 Adam↔GN 落差**已修掉**。
**这不是新模型**：目标函数、分量掩码、holdout 划分一律未动，只换了下降方式。

### 改动

`scripts/validate_helix_model.py`：

- `fit_helices_gauss_newton()`：主路径的测量 refit 改走 `fit_broken_helix`（`N == 0`），
  **与 Gate 4 single 臂是同一个求解器实例**，不是并排的第二份实现。
  测试 `test_gauss_newton_measurement_fit_matches_the_gate4_single_arm`
  断言两者输出逐位相等——防止主路径与门控路径日后悄悄分叉。
- `measurement_newton_decrement()`：对**任意**求解器的输出算 `λ² = gᵀH⁻¹g`。
  这把「Adam 提前停」从推断变成实测。两处均按 2048 行分块。
- `stationarity_report()` / `_merge_convergence()`：从 Gate 4 抽出复用，
  删掉了 `run_broken_arm` 里的重复实现。
- `--fit-solver {gauss-newton,adam}`，默认 `gauss-newton`。
  保留 `adam` 只为复现 Iteration 14 的发现，不是受支持的运行方式。
- **主路径新增驻点门控**：非驻点行占比 > `MAX_NON_STATIONARY_ROW_FRACTION` 直接判 fail。
  没到最优的 refit 不能当成对模型的陈述——这条门槛与 Gate 4 两臂用的是同一个。

`seed_helices`（对 SMT 点的几何预拟合）仍用 Adam，只提供 GN 的起点，本轮未动。

### 决定性证据：Adam 从未到过驻点

同一批 70352 行、同一目标函数：

| 求解器 | 非驻点行占比 | decrement p50 | decrement p99 | 迭代数 |
|---|---|---|---|---|
| Adam（2000 轨迹子集实测） | **1.0000** | **1.26e+01** | — | 200（预算耗尽） |
| Gauss–Newton（全量） | **1.28e-4** | 9.61e-23 | 7.40e-13 | **16** |

Adam **每一行**的 Newton decrement 都在 1e-6 门槛之上，中位数 12.6
——不是「个别行停滞」，是整批都没收敛。objective 4912.8 → 2.376。

### 全量 holdout 对照（70352 行，绝对同一批数据与划分）

| 层\|模式 | 分量 | Adam RMSE | GN RMSE | 比值 |
|---|---|---|---|---|
| 17:8 interp | LOC0 | 0.076261 | **0.021723** | **÷3.5** |
| 17:8 interp | LOC1 | 0.033937 | 0.032497 | ÷1.04 |
| 17:8 fwd | LOC0 | 0.237998 | 0.213453 | ÷1.12 |
| 17:8 fwd | LOC1 | 0.060489 | **0.070582** | **×1.17** |
| 17:8 bwd | LOC0 | 0.567832 | **0.187964** | **÷3.0** |
| 17:8 bwd | LOC1 | 1.727658 | 1.433637 | ÷1.21 |
| 24:4 interp | LOC0 | 0.173330 | **0.050144** | **÷3.5** |
| 24:4 interp | LOC1 | 1.198811 | 1.187356 | ÷1.01 |
| 24:4 fwd | LOC0 | 0.356660 | 0.259487 | ÷1.37 |
| 24:4 fwd | LOC1 | 1.209467 | 1.210019 | ×1.00 |
| 24:4 bwd | LOC0 | 0.945433 | 0.842530 | ÷1.12 |
| 24:4 bwd | LOC1 | 5.140685 | 3.247682 | **÷1.58** |

方向角 RMS 全线改善（17:8 bwd 0.005003→0.002968，24:4 bwd 0.009356→0.006670）。
bias 在外推组显著回中（24:4 fwd −0.0196→+0.0012，24:4 bwd −0.0214→+0.0024）
——Adam 的欠收敛此前把系统性偏移留在了外推方向上。

**门控**：失败项 11 → 8，其中**三条是真的通过了**，不是数字变小：
17:8 interpolation LOC0（0.0763→0.0217，门槛 0.05）、
24:4 interpolation LOC0（0.1733→0.0501，门槛 0.1）、
17:8 backward 方向角（0.005003→0.002968，门槛 0.005）。
**两个插值组第一次落进 Iteration 10 定的绝对阈值内。**
其余 8 条仍失败，但都是单边外推组——Iteration 10 已说明该阈值对外推不合适。

### 唯一的回退：17:8 forward LOC1（+16.7%）

0.060489 → 0.070582。这条必须照实记：Adam 的欠收敛相当于朝 SMT 种子的隐式收缩，
对单边外推的曲率外插恰好起了正则作用；GN 把拟合窗内的似然做到最优，
曲率的过拟合会被外推放大。**这是「正确求解」的代价，不是 bug**，
但它说明外推预测需要的是显式先验（如材料 kink 的 `Q`），不是靠优化器欠收敛白捡。

（2000 轨迹子集上 24:4 bwd LOC0 也曾回退 ×1.27，全量下变为 ÷1.12
——那是小样本假象，已被全量推翻。记在这里以免日后误引。）

### 对 Iteration 14 结论的影响

- **Gate 4 的数字不再逐位可复现**：`fitted_parameters` 同时是 Gate 4 的冻结参考，
  换成 GN 后参考变了。需要重跑（约 53 分钟）才能给出新的 broken-vs-single 判定。
- 但方向上不会翻案：Iteration 14 的 GN single 臂本来就是从 Adam 点再优化到同一最优
  （17:8 interp LOC0 两处都是 0.021723，24:4 interp LOC0 两处都是 0.050144）。
  变的是 broken 臂的起点与线性化点，不是 single 臂的水平。
- **Iteration 10 的物理归因正式作废**：「RMSE 比 bias 高 1–2 量级 ⇒ 未建模多重散射」
  在插值组上有 3.5 倍来自优化器，不是散射。

### 已知的同类缺陷（本轮未修，标记）

`scripts/ml_alignment_autograd.py:604` 有**另一个** Adam 版
`fit_helices_to_measurements`，是 block-LM 里 nuisance 轨迹的 refit。
同样的欠收敛极可能存在，且它直接决定 joint 校准结果。
本轮未动：参数化不同（position/direction/rho_scaled 三组独立 lr），
改动会改变 joint 校准行为，而那属于工作包 ④，Gate 5 规定 Gate 4 未通过不得启动。

### 测试

`tests/test_validate_helix_model.py` 新增 8 项（先写失败测试再实现）：
GN 目标值须比 Adam 低一个量级以上、GN 达驻点、**Adam 不达驻点**（把缺陷写成断言）、
GN 主路径 == Gate 4 single 臂、两处分块逐位一致、solver 名校验、默认值。
`make test`：**284 passed / 35 failed**，35 项为既有的 `/usr/bin/python` 3.9
子进程失败，与本轮无关；零新增失败。

### 尚未宣称的结果

- **Gate 4 尚未在新参考下重跑**，broken-vs-single 的判定暂时悬空。
- `seed_helices` 的 Adam 预拟合未检验是否也欠收敛（它只定起点，影响面小但未测）。
- `ml_alignment_autograd.py` 侧的同类缺陷**只是标记，未测量、未修**。
- 17:8 forward LOC1 的回退未做归因外的进一步验证。

---

## Iteration 16 — Gate 4 在 Gauss–Newton 参考下重跑：判定不变，24:4 backward 回退归因翻案

**运行**：`data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/broken_helix_gn_20260826/`
（20000 events / 70352 holdout 行 / 26 分 18 秒 / git commit `41acf3e`）

### 为什么要重跑

`fitted_parameters` 同时是 Gate 4 的冻结参考。Iteration 15 把测量 refit 从 Adam 换成
Gauss–Newton 之后参考点变了，Iteration 14 的门控数字不再逐位可复现。重跑之后
single 臂成为自身最优的不动点，两臂比较才干净。

### holdout 行可比性

新旧两跑的 `holdout_keys.csv` 文件 SHA256 **不同**（`4ecf9804…` vs `d0490b85…`），
但差异**全部**在 `holdout_path_s_mm` 一列：弧长依赖拟合参数，换求解器后必然变
（最大偏差 49.8 mm，相对 48%，正是 Adam 未收敛的幅度）。
身份键 `(event_nr, track_nr, mode, holdout_sensor_idx)` 的 SHA256 逐位一致
（`19c86cd3b4020ac3e7217681f9a7872e44257aef3a25e8987a87025d83254a15`，diff 0 行），
**同一批 70352 行，两跑可比**。

> 教训：比对两次实验的 holdout 一致性时，比的是**身份键**，不是整个文件的哈希。
> 文件里混入了依赖拟合结果的列，整体哈希对求解器变更没有不变性。

### 数值先行：驻点门控通过

| 臂 | 非驻点行占比 | decrement p50 | 迭代 |
|---|---|---|---|
| single | 1.279e-4 | 3.00e-23 | **2** |
| broken | 8.529e-5 | 3.02e-19 | 20 |

上限 `MAX_NON_STATIONARY_ROW_FRACTION = 0.01`，两臂均通过 → 可以解读物理。

**single 臂只用 2 次迭代**：从 GN 参考出发，求解器立刻确认已在最优。这正是重跑要的性质。

### 逐组对照（RMSE mm，与 Iteration 14 的 Adam 参考对比）

| 组 | N | single 旧→新 | broken 旧→新 | ΔRMSE CI 旧 | ΔRMSE CI 新 |
|---|---|---|---|---|---|
| 17:8 interp LOC0 | 8346 | 0.021723 → 0.021723 | 0.021470 → 0.021470 | [−1.84,−0.60]% | [−1.84,−0.60]% |
| 17:8 interp LOC1 | 8346 | 0.032497 → 0.032497 | 0.031464 → 0.031464 | [−5.28,−1.40]% | [−5.28,−1.40]% |
| 17:8 fwd LOC0 | 5034 | 0.213453 → 0.213453 | 0.213394 → 0.213394 | [−0.06,+0.00]% | [−0.06,+0.00]% |
| 17:8 fwd LOC1 | 5034 | 0.070582 → 0.070582 | 0.070352 → 0.070352 | [−0.90,+0.13]% | [−0.90,+0.13]% |
| 17:8 bwd LOC0 | 14190 | 0.187964 → 0.187964 | 0.187950 → 0.187950 | [−0.02,+0.00]% | [−0.02,+0.00]% |
| 17:8 bwd LOC1 | 14190 | 1.433637 → 1.433637 | 1.433514 → 1.433515 | [−0.03,+0.01]% | [−0.03,+0.01]% |
| 24:4 interp LOC0 | 14197 | 0.050144 → 0.050144 | 0.049868 → 0.049868 | [−0.82,−0.32]% | [−0.82,−0.32]% |
| 24:4 interp LOC1 | 14197 | 1.187356 → 1.187356 | 1.185249 → 1.185250 | [−0.27,−0.08]% | [−0.27,−0.08]% |
| 24:4 fwd LOC0 | 14261 | 0.259487 → 0.259487 | 0.259476 → 0.259476 | [−0.02,+0.00]% | [−0.02,+0.00]% |
| 24:4 fwd LOC1 | 14261 | 1.210019 → 1.210019 | 1.210019 → 1.210019 | [−0.04,+0.04]% | [−0.04,+0.04]% |
| **24:4 bwd LOC0** | 14324 | 0.846021 → **0.842698** | 0.864379 → **0.846829** | [−0.89,**+8.69**]% | [+0.07,**+1.37**]% |
| **24:4 bwd LOC1** | 14324 | 3.282800 → **3.247702** | 3.299087 → **3.298179** | [−1.33,+2.23]% | [+0.01,+3.53]% |

**12 组里 10 组逐位不变**（6 位小数）。只有 24:4 backward 两个分量动了。

### 主要结论

1. **Gate 4 维持 failed，方向未翻案。** 插值组改善仍只有 0.3–1.8%，远不及预注册的 −20%。
   `broken-helix 不优于 single helix` 这一条在新参考下**逐位复现**。

2. **24:4 backward LOC0 的回退归因翻案。** Iteration 14 记的 +8.69% CI 上界，
   在新参考下收窄到 **+1.37%**，已落进 `EXTRAPOLATION_RMSE_REGRESSION = 0.05` 的容差内
   （该组不再因 RMSE 判失败，只因 NLL 判失败）。
   旧跑那个大回退**主要是 Adam 参考未收敛的产物**，不是「kink 吸收几何自由度」。
   注意 CI 现在整体落在正区间（下界 +0.07%），回退**统计显著但幅度微小**。

3. **broken 臂修正真实过散的结论不变**，逐位一致：
   17:8 interp LOC1 pull_std 1.1258 → 0.9877，predictive χ² 1.2674 → 0.9755。

4. **damping 敏感性仍未解决。** 8 个外推组的 NLL 判定依旧随预注册 damping 网格翻转，
   现在被显式列为一条门控失败项。**外推方向仍属「不可测」，NLL 数字仍不得引用。**

### 可以宣称 / 不得宣称

- **可以**：broken helix 在插值方向上带来统计显著但幅度极小（0.3–1.8%）的 RMSE 改善，
  并把 17:8 interpolation LOC1 的过散修正到 χ² ≈ 1。
- **可以**：Iteration 14 报告的 24:4 backward 大幅回退不成立，它源于未收敛的比较基准。
- **不得**：broken helix 优于 single helix；任何外推方向的 NLL 比较。

### 尚未宣称

- damping 两臂不等价未修（`predictive_covariance` 用 `damping·mean(diag H)·I`，
  broken 臂的 `mean(diag H)` 被 `Q⁻¹` 块抬高）。修完之前外推组不可测。
- `ml_alignment_autograd.py` 的 Adam refit 同类缺陷仍**只是标记，未测量、未修**。
- 结论仍只在 20000 events 上取得，未在计划书写的 100000 events 上复现。

---

## Iteration 17 — damping 两臂不等价：归因翻案、修复与预注册 v2

**运行**：`data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/damping_v2_20260826/`
（20000 events / 70352 行 / git commit 见 manifest）

### 1. Iteration 14 记的归因是错的

交接文档 §3.3 与日志此前都记为：「broken 臂 H 含 2N 个 kink 参数，`mean(diag H)`
被 `Q⁻¹` 块抬高，所以同一相对 damping 对两臂的绝对扰动差很多」。**实测不成立**：

| 量 | 实测 |
|---|---|
| kink 块 / reference 块 对角均值比 | **0.99** |
| `1/kink_sigma²`（先验项） | 3.7e5 |
| 测量项对对角的贡献 | ~5e9 |
| `mean(diag H)` 随节点数 0→7 | 5.05e9 → 4.29e9（**略降**）|

先验比测量项小四个量级，根本不主导 kink 块的对角。`mean(diag H)` 从未被抬高。

### 2. 真正的机制：padding 造成的结构零方向

broken 臂把 `node_s` 补到固定宽度，**未使用的槽位不进入任何残差**，
H 与 J 的对应行列**精确为零**。600 行实测：

- 零特征值个数 **逐行精确等于** `2 × (padded_nodes − n_valid)`（600/600 吻合）
- `|J @ nullvec|` 的 p50 / p99 / max **全部为 0.000e+00**

所以 `J C Jᵀ` 在**解析上**完全不依赖阻尼赋给零方向的方差。
但阻尼给了它们 `1/(λ·mean(diag H))` 的巨大方差，把 Cholesky 的条件数推到 ~2e13
（float64 边缘），**舍入误差污染整个解**。single 臂没有 padding 可 damp，纹丝不动。
这才是两臂不等价的根源。

### 3. 第二层：非零相对阻尼无法对两臂等价

即使剔除 padding，broken 臂 active 子空间仍比 single 病态 100–300 倍
（cond p50 1.67e7 vs 1.07e5）——kink 确实是软方向。实测预测方差相对 `damping=0` 的偏差：

| λ | single p99 | broken p99 |
|---|---|---|
| 1e-12 | 4.5e-6 | 3.4e-3 |
| **1e-10（v1 production）** | **4.5e-4** | **1.9e-1** |
| 1e-6 | 8.1e-1 | 9.98e-1 |

**任何非零相对阻尼都不可能等价**，因为两臂的谱本质不同。

### 4. 修复（预注册 v2）

1. **结构零方向不再阻尼，改填 active 平均对角。** 对协方差是恒等变换（J 在其上为零），
   条件数回落到可解子空间的水平。预测若真的依赖非活跃方向则**拒绝**——那种情况下
   报出的方差纯粹是正则化器的产物。
2. **`PRODUCTION_DAMPING = 1e-10 → 0.0`。** 修复 ① 使 `damping = 0` 首次可解
   （此前 broken 臂必失败）。精确后验，两臂按构造等价。
3. **秩亏行丢弃而非正则化。** 返回 NaN，调用方既有的 finite 检查会把该行从**两臂**
   同时剔除（`compare_arms` 取交集），paired 行集契约自动保持。
   `MAX_RANK_DEFICIENT_ROW_FRACTION` 把关，防止行集悄悄缩水。
4. **敏感性 gate 只用 `DAMPING_STABILITY_SCAN = (0.0, 1e-12)`。** 全网格仍完整记录为
   诊断，但大阻尼就是大偏差，「加了偏差结论会变」平凡为真，不该判 gate 失败。
5. 新增 `PREREGISTRATION_VERSION = 2` 钉住这次变更。

### 5. 中途被推翻的两个自己的方案

- **per-row 阻尼阶梯**：先实现后删除。Cholesky 失败的条件与数值秩亏判据
  `μ_min ≤ ε·n·μ_max` 高度重合，阶梯**从来没有可服务的场景**——凡是失败的行都是
  真秩亏，救活它等于把正则化器的方差当成拟合结果报出去。
- **秩亏判据用了 batch 全局最大特征值**（`maximum` 是 `float(ev[...,-1].max())`）。
  这等于用 chunk 里最硬的轨迹缩放所有行的门槛：broken 臂 μ_max 达 5.7e12，
  floor 被抬到 ~0.04，而健康行 `smallest_active` 在 7.5e-7 量级 →
  **50 行健康数据被整批误判**，且这 50 行**全部**落在 24:4 backward 组，
  丢掉它们把该组 LOC0 的 RMSE 从 **0.8427mm 改善到 0.4921mm（42%）**。
  差点让正则化器悄悄剔掉最不利的数据。改为逐行判据后降到 5 行。
  回归测试双向验证过（bug 版本失败、修复版本通过）。

### 6. 全量验收

**11/12 组的 N 与 RMSE 与 Iteration 16 逐位相同。** 只有 24:4 backward 变化
（N 14324→14319，LOC0 RMSE 0.842698→0.786690），源于丢弃 5 行秩亏行。

丢弃的 8 行（single 3 / broken 5，交集后组内少 5）特征高度一致：

| arm | n_hits | n_active | μ_min/μ_max |
|---|---|---|---|
| single ×3 | **4** | 5 | −7.8e-18 / −3.0e-17（**负**）|
| broken ×5 | **4–5** | 11–15 | 2.9e-15 ~ 2.9e-17 |

健康行 n_hits 中位数是 7。这些行是**结构性欠定**（single 臂 4 hits 撑 5 个
gauge-fixed 参数，μ_min 甚至为负），残差反映的是欠定拟合的任意性而非模型误差。
根因是预注册的 `min_extrapolation_fit_hits = 4`（插值组是 8）过松。

**damping 敏感度**（|ΔNLL 变化| / CI 半宽，跨 0 与 1e-12）：

| ratio | 组数 |
|---|---|
| 0.000（完全免疫）| 6 |
| 0.004 – 0.36 | 5 |
| **2.223** | **1（24:4 bwd LOC0）** |

修复前是 **8 个外推组跨全网格翻转**；现在 11/12 组的 damping 依赖被彻底消除。

### 7. 结论

- **Gate 4 维持 failed。** 插值组改善仍只有 0.3–1.8%，逐位复现，broken 不优于 single。
- **外推组的 NLL 现在可以引用了**，唯独 `24:4|backward_extrapolation|LOC0` 除外——
  它的 damping 敏感度是统计不确定度的 2.2 倍，是**真实敏感性**，gate 正确地单独标记了它。
  判据未做任何放宽以让它通过。
- 24:4 backward 是数值极端行的集中地：唯一有秩亏行的组、唯一真实 damping 敏感的组、
  Iteration 14/16 里唯一回退的组。

### 尚未宣称

- `24:4|backward_extrapolation|LOC0` 的 NLL 结论仍不可引用。
- `min_extrapolation_fit_hits = 4` 对 single 臂的 5 参数拟合不足，但改它属于另一次
  预注册变更，本轮未动。
- `ml_alignment_autograd.py` 的 Adam refit 同类缺陷仍**只是标记，未测量、未修**。
- 结论仍只在 20000 events 上，未在 100000 events 上复现。

---

## Iteration 18 — joint 路径的 Adam refit：驻点性首次被测量

**改动**：`scripts/ml_alignment_autograd.py`、`tests/test_ml_alignment_autograd.py`
**性质**：**纯测量，不改行为**。求解器、预注册常量、`refitted_parameters` 全部未动。

### 1. 为什么现在做

`fit_helices_to_measurements` 是 joint 校准初始化阶段的 Adam 测量 refit，
在 `run_joint_alignment` 里调用一次，其输出是整个校准的 nuisance 轨迹起点。
Iteration 15 已在 Gate 4 路径上证明同类 Adam refit 从未到过驻点，但 joint 路径的
这一处从 Iteration 15 起就只被标记、从未测量。P1 完成后门槛解除。

### 2. 两个必须先解决的问题

**gauge 不一致。** `_measurement_refit_objective` 的 loss 是
`sum(r²/cov) / component_mask.sum()`，而预注册的 `NEWTON_DECREMENT_STATIONARY = 1e-6`
定义在 `F = 0.5·chi2` 上（`HALF_CHI2_OBJECTIVE`）。decrement 随目标线性缩放，
直接在归一化 loss 上测会把每个值缩小约 1e-5 倍，阈值随之失去意义。
→ 测量全部走 validation 模块，只把参数传过去。

**两个目标是否同一个。** validation 侧用冻结弧长 `hit_s`，autograd 侧每次重解
`intersect_plane`。实测在 nodeless 情况下**逐位相同**：

| 量 | 值 |
|---|---|
| autograd loss 还原为 `0.5·chi2` | 120264.615429 |
| validation `0.5·‖r‖²` | 120264.615429 |
| 相对差 | **0.000e+00** |

原因是三件事同时成立：`intersect_plane` 两边都用默认 `n_newton = 12`；
`EPS_VARIANCE == EPS_COVARIANCE == 1e-8`；零节点时 `broken_helix_track_parameters`
退化为 reference 的逐 hit 广播，`hit_s` 只用于选段，不影响预测。
component mask 也是同一个 `batches.fit_mask`。

### 3. 实现

- `_validation_batches_from_joint`：两个模块各有自己的 `TrackBatches`
  （`covariances`/`loc1_mask` 对 `measurement_variance`/`loc1_valid`）。构造真正的
  dataclass 而非 duck-typing，字段变更才会当场报错。`holdout_mask` 在 joint 场景没有
  意义（`newton_decrement` 不读任何 `holdout_*`），用第一个 fit hit 占位；
  `prediction_mode` 填哨兵 `joint-refit-diagnostic`，它不在 `PREDICTION_MODES` 里，
  因此这种 batches 永远选不进任何 Gate 4 结论。
- `measure_joint_refit_stationarity`：调 `measurement_newton_decrement` +
  `stationarity_report`，默认 CPU float64（与 1e-6 比较时 float32 精度不够）。
- 挂载在 `fit_helices_to_measurements` 的 stats dict 上，对 **seed 与 refit 各测一次**。
  只报 refit 说明不了 Adam 走了多远；seed→refit 的降幅才能区分「起点就近」和「优化器有效」。
- `--no-joint-refit-stationarity` 可关（默认开）。

### 4. 结果：joint 路径的 Adam refit 也从未接近驻点

真实数据，`--track-mode free --max-tracks 300`（1200 轨迹选中 300）：

| 量 | seed | 200 次 Adam 迭代后 |
|---|---|---|
| non-stationary 行占比 | 100.00% | **100.00%** |
| Newton decrement p90 | 5.36e5 | **4.17e5** |
| 非有限行 | 0 | 0 |

阈值是 1e-6。200 次迭代把 decrement 降了约 22%，终点距驻点仍有 **11 个数量级**。
这与 Iteration 15 在 Gate 4 路径上的发现同向，且这里的量级更极端。

### 5. 行为不变的验证

同一命令开/关 flag 各跑一次：

- `measurement chi2/component = 7992.76` 逐位相同；
- 最终导出的对齐几何 JSON **文件级完全一致**（`cmp` 无差异）。

不只是 refit 参数没变，整条下游都没变。

### 6. 测试

新增 5 项（`tests/test_ml_alignment_autograd.py`）：gauge 等价（相对差 ≤ 1e-9）、
适配器逐字段保真、target 泄漏仍 fail-closed、开关 flag 后 `torch.equal` 相同、
以及双向的分辨力测试。

最后一项是关键：只断言「1 次迭代比 20 次差」的话，一个恒返回常数的测量也能通过。
因此同时断言**同一目标的 Gauss-Newton 解必须被判为收敛**——实测
`non_stationary_row_fraction == 0.0` 且 `newton_decrement_max < 1e-6`。
测量工具本身是对的，所以 Adam 的大 decrement 是真的。

### 7. 环境更正（重要）

`docs/handoff/2026-08-26-damping-equivalence-v2.md` §7 记的
「`python` = `/usr/bin/python` 3.9.25，`make test` 293 passed / 35 failed」
**是解释器用错了**。`pyproject.toml` 声明 `requires-python = ">=3.10"`，
而 3.9 缺 `zip(..., strict=True)`，`fit_helices_to_measurements` 在它下面根本跑不完
第一次迭代。项目自己的 skip 消息也写着「Run in the conda pytorch Python 3.10 environment」。

正确解释器：`~/miniconda3/envs/pytorch/bin/python`（3.10.20，torch 2.11.0+cu126），
需要 `LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib`
（否则 `awkward_cpp` 找不到 `CXXABI_1.3.15`）。

| 解释器 | 全量测试 |
|---|---|
| `/usr/bin/python` 3.9.25 | 293 passed / **35 failed** / 3 skipped |
| conda `pytorch` 3.10.20（本轮前） | **328 passed / 0 failed** / 3 skipped |
| conda `pytorch` 3.10.20（本轮后） | **333 passed / 0 failed** / 3 skipped |

那 35 项「既有失败」不是代码欠债，是环境不匹配。本轮 +5 恰为新增测试数，零回归。

### 尚未宣称

- **没有**换求解器。把 refit 换成 `fit_helices_gauss_newton` 会改变
  `refitted_parameters`，属于下一次预注册变更。
- 上表的测量来自 300 条轨迹的小样本；全量 joint 运行上尚未复现。
- `seed_helices` 的 Adam 预拟合仍未检验（P4）。
- `24:4|backward_extrapolation|LOC0` 的 NLL 结论仍不可引用（P1'）。
- `min_extrapolation_fit_hits = 4` 仍未动（P3）。

---

## Iteration 19 — joint refit 换成 Gauss-Newton（P2'）

**改动**：`scripts/ml_alignment_autograd.py`、`tests/test_ml_alignment_autograd.py`、
`tests/test_joint_alignment.py`、`tests/test_component_mask_wiring.py`
**性质**：**求解器变更**。这是一次预注册变更，`refitted_parameters` 及全部下游产物改变。

### 1. 预注册在先

判据在跑之前写死，事后未做任何调整：

| 常量 | 值 | 来源 |
|---|---|---|
| `JOINT_REFIT_SOLVER` | `"gauss-newton"` | 本轮 |
| `JOINT_REFIT_PREREGISTRATION_VERSION` | 1 | 本轮新建的 joint 侧版本线 |
| `JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION` | 1.0e-3 | 类比 `MAX_RANK_DEFICIENT_ROW_FRACTION` |

驻点阈值本身**复用**已预注册的 `NEWTON_DECREMENT_STATIONARY = 1e-6`，未另定义。
非有限行要求为 0。任一不满足 → raise，不产出结果。

**`PREREGISTRATION_VERSION` 保持 2，未动。** 它的注释写明治理范围是「this block」——
Gate 4 的判决常量，且只写进 Gate 4 的 summary。本轮不碰任何 Gate 4 常量，Gate 4 也不
消费 joint 输出；升它会让此后每份 Gate 4 summary 声称判决规则变过，而实际未变。
故 joint 侧另立 `JOINT_REFIT_PREREGISTRATION_VERSION`。

### 2. 「不收敛行丢弃」在这条路径上不存在

本轮核实推翻了一个想当然的设计：`fit_broken_helix` 对每行都返回参数（per-track 阻尼 +
步长接受），不产生 NaN 行；`converged` 是 per-chunk 的 bool，不是 per-row。秩亏 NaN 丢弃
是协方差路径 `_run_broken_arm_chunk` 的事，不在拟合路径上。而且 joint 场景下丢一条
nuisance 轨迹 = 改变 nuisance 集合，与 Gate 4 丢一行语义完全不同。
**per-row 判据只能落在 Newton decrement 上**，判据据此设计。

### 3. 接线比预期机械

`fit_helices_gauss_newton` 与 Iteration 18 已在用的 `measurement_newton_decrement`
走**同一个** `_measurement_fit_inputs`，因此 `_validation_batches_from_joint` 这个适配器
早已被实测验证可直接喂给 GN 拟合路径。`prepare_broken_inputs` 确实读 `holdout_mask`，
但拟合用的 `component_mask` 只由 `fit_mask` 构成，holdout 只流向 `holdout_*` 输出字段
——这正是 Iteration 18 目标函数能逐位等价的原因，占位符对 GN 拟合同样无害。

**GN 分支强制 float64**：预注册阈值 1e-6 低于 float32 分辨率，而 joint 管线在 GPU 上
跑的就是 float32（`seed_helices` 的 dtype 由 `--gpu` 决定）。解完再转回调用方 dtype。

**判据不受 `--no-joint-refit-stationarity` 影响**：它读的是 GN 求解器自己算出的
decrement（`solver_stats["stationarity"]`），不是那个可关的诊断测量。关掉诊断不等于
关掉验收。

### 4. 结果

真实数据，`--track-mode free --max-tracks 300`，两臂只差一个 flag：

| 量 | Adam | **Gauss-Newton** |
|---|---|---|
| `n_components` | 5616 | **5616** |
| `chi2_per_component` | 7992.76 | **0.741103** |
| 非驻点行占比 | 100.00% | **0.00%** |
| Newton decrement p90 | 4.17e5 | **1.87e-17** |
| 非有限行 | 0 | 0 |

**先核对的是 `n_components`**（Iteration 17 §4.1 的教训）：两臂相同，说明没有行被悄悄
丢掉，chi2 的四个数量级改善不是样本变化换来的。

Adam 臂逐位复现 Iteration 18 的数字（7992.76 / 5.36e5 → 4.17e5 / 100%），说明重构没有
改变 Adam 的行为。

`chi2/component` 从 7992.76 降到 0.741 另有物理意义：一个正确归一的 per-component chi2
在测量噪声主导时应当在 1 附近。7992 是荒谬的，0.741 是合理的。

### 5. Adam 保留

`--joint-refit-solver {gauss-newton,adam}`，默认 GN。Adam 代码原封不动搬进
`_refit_helices_adam`，Iteration ≤18 的产物仍可复现，A/B 对照随时可跑。
`best_iteration` / `improved` / `first_step` 是 Adam 语义，**只在 adam 分支出现**——
不给占位值，免得读 summary 的人把 GN 运行误认成「checkpoint 恰好没改善过」的 Adam 运行。

### 6. 测试

新增 4 项，全量 **337 passed / 0 failed / 3 skipped**（基线 333 + 4）。

判据的双向测试是关键：单测「GN 解满足判据」的话，一个永不触发的 gate 也能通过。因此
同时用 `max_iters=1` 的截断 GN 解构造真实的不达标输入，断言它**确实 raise**
（实测 1 次迭代非驻点行 100%，完整解 0%）。

### 尚未宣称

- 本轮只在 300 条轨迹上做过 A/B，全量 joint 运行未复现。
- 下游对齐结果（`target_metrics` 等）的变化未系统评估，只验证了 refit 本身。
- `seed_helices` 的 Adam 预拟合仍未检验（P4）。
- `24:4|backward_extrapolation|LOC0` 的 NLL 结论仍不可引用（P1'）。
- `min_extrapolation_fit_hits = 4` 仍未动（P3）。
- Gate 4 维持 failed，工作包 ④ 仍被 Gate 5 阻塞。

### 7. 补充：全量复现（同日）

Iteration 19 §4 的 A/B 只在 300 条轨迹上做过。用同一对命令在全量上重跑，
两臂仍只差 `--joint-refit-solver` 一个 flag，`--joint-iters 1`：

| 量 | Adam | **Gauss-Newton** |
|---|---|---|
| retained tracks | 61482 | 61482 |
| `n_components` | 1154761 | **1154761** |
| `invalid_fraction` | 0.0 | **0.0** |
| `chi2_per_component` | 7595.34 | **0.763503** |
| 非驻点行占比 | 100.00% | **0.00%** |
| Newton decrement p90 | 4.1e5 | **2.61e-17** |
| GN 收敛 | — | **True，5 次迭代** |

98958 条输入轨迹选中 61482 条，1154761 个 component——是 300 轨迹小样本（5616）的
**205 倍**。`n_components` 与 `invalid_fraction` 两臂逐位相同，样本量没有变化。

预注册判据在全量上以极大裕度满足：非驻点行占比精确为 0.0（上限 1e-3），非有限行 0。

**小样本对全量的代表性得到验证**，两个尺度的结论同向且量级一致：

| 量 | 300 轨迹 | 全量 |
|---|---|---|
| Adam chi2/component | 7992.76 | 7595.34 |
| Adam decrement p90 | 4.17e5 | 4.1e5 |
| GN chi2/component | 0.741103 | 0.763503 |
| GN 非驻点行 | 0.00% | 0.00% |

GN 在 61482 行上只用 **5 次迭代**收敛（上限 60），说明这个目标对 Gauss-Newton 而言
是容易的——Adam 用 200 次迭代走不到的地方，不是问题难，是优化器不对。

至此 P2'' 关闭。**下游影响（P2'''）仍未评估**：本次两臂都用 `--joint-iters 1`，
只验证了 refit 本身，`target_metrics` 等对齐结果的变化尚未系统比较。

---

## Iteration 20 — joint refit 换 GN 后的下游影响评估（P2'''）

**性质**：**观测，不是变更**。不改产品代码，不设 raise 阈值，不动任何预注册常量
（`PREREGISTRATION_VERSION` 保持 2，`JOINT_REFIT_PREREGISTRATION_VERSION` 保持 1）。

Iteration 19 §7 的两臂都用 `--joint-iters 1`，只验证了初始化 refit 本身。GN 改变了
`refitted_parameters`，它是 joint 主循环的起点，因此 `target_metrics` 与
`truth_metrics` 都会随之改变，**幅度与方向尚未测量**。本轮就是这次测量。

### 1. 预注册比较协议（写在跑任何一臂之前，事后只填数字）

#### 1.1 核对顺序（强制：先样本量，后效果）

沿用 Iteration 17 §4.1 与 19 §4 的教训——任何让结果变好的改动先查样本量：

1. `n_tracks` / `n_hits` 两臂逐位相同。不同 → 停止比较，先查样本为什么变了。
2. `initialization_stats` 的选择统计两臂逐位相同（轨迹选择在 refit 之前，理应与
   solver 无关；不同即是 bug 信号）。
3. `invalid_fraction` 两臂都落在既有守卫内。
4. 以上全部通过，才读效果指标。

#### 1.2 效果指标（第 4 步之后才看）

| 层级 | 量 | summary JSON 键 |
|---|---|---|
| 主指标 | per-DOF correlation、recovery fraction，逐目标层 | `truth_metrics` |
| 次指标 | `rms_after_loc0/loc1`、`mean_after_loc0/loc1`，逐目标层 | `target_metrics` |
| 收敛 | `best_total_loss`、`last_total_loss`、`best_iteration`、`stopped_early` | 顶层 |
| 分阶段 | `smt_seed` / `measurement_refit` / `joint` 三阶段 layer chi2 | `stage_layer_chi2` |
| 验收留痕 | solver、stationarity、收敛迭代数 | `initialization_stats.measurement_refit` |

`truth_metrics` 作主指标的理由：`validate_against_truth`
（`scripts/ml_alignment_autograd.py:2260`）的 docstring 写明它才是「决定局域信号是否
可恢复」的量；`target_metrics` 的残差 RMS 可以在几何错的情况下被轨迹自由度吸收而
虚假变好。

#### 1.3 判读规则

- **不设 gate、不设 raise 阈值**：本轮是观测，不是验收。
- 结论三选一，逐目标层分别陈述：GN 臂**优于** / **劣于** / **与 Adam 臂无实质差异**。
- `24:4|backward_extrapolation|LOC0` 的任何数字**照旧不得引用**（P1'）。出现在结果里
  只作记录，不进结论。
- 两臂 `stopped_early` / `best_iteration` 不同属正常（起点不同），但必须显式报告：
  在不同 iteration 停下的两条曲线，终点比较要附上这个事实。

#### 1.4 两阶段执行

- **阶段 A**：`--max-tracks 300` 两臂跑满 `--joint-iters`，做方向筛查并标定单次主循环
  iteration 的墙钟（全量 1000 iters 的耗时目前无任何实测依据）。外推值只用于排期，
  不进结论。
- **阶段 B**：去掉 `--max-tracks`，全量两臂**串行**（本机可用约 7 GB，单臂峰值约 3 GB）。

两臂只差 `--joint-refit-solver` 一个 flag；`--output-geom` 与 `--output-summary`
分臂显式指定，避免第二臂覆盖第一臂并污染下游 reconstruction 的输入几何。

### 2. 环境与基线（本轮实测，2026-08-26）

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q
```

**337 passed / 3 skipped / 11 deselected**，与 Iteration 19 基线一致。3 项 skip 为
AlignDerivLib pybind 未编译 ×1、ACTS bindings ×2——**要用 ACTS 时当场实测确认，勿凭
这两条 skip 或凭 memory 下结论**。

本机无 `/usr/bin/time`，墙钟改由驱动脚本用 `date +%s` 差值记录。可用内存 8 GB，
两臂严格串行。GPU 存在（RTX 3050）但**两臂均不加 `--gpu`**：保持 CPU float64，与
Iteration 19 §7 的全量结果同配置可比。

默认输出几何 `data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_v17l8_v24l4_ml.json`
本轮前的 mtime 为 **2026-08-03 11:19:08**，用于事后确认它未被本轮污染。

### 3. 更正：`--joint-iters` 在默认求解器下是死参数

本轮实施时核实发现，交接文档给 P2''' 写的做法（「同一对命令改用完整 `--joint-iters`
（默认 1000）」）**无法落地**：

- 默认 `--joint-solver block-lm` 走 `run_block_lm_alignment`
  （`scripts/ml_alignment_autograd.py:1075`），它的迭代上限是 **`--lm-max-iters`
  （`LM_MAX_ITERS = 20`）**，**完全不消费 `config.joint_iters`**。
- 消费 `joint_iters` 的那个主循环（`scripts/ml_alignment_autograd.py:1332`）属于
  `--joint-solver adam` 分支。
- block-LM 没有 per-iteration 的进度打印（诊断只进 `fisher_diagnostics`），所以日志里
  不会出现 `Joint [iiii/1000]` 行——**日志安静不等于主循环没跑**。

由此得到两个推论：

1. 本轮两臂 `best_iteration = 19`、`stopped_early = False`，即 **20 次 LM 迭代全部跑满**，
   下游求解本来就是完整的，不需要「改用完整 `--joint-iters`」。
2. **Iteration 19 §7 的全量两臂同样如此**：那两次运行加的 `--joint-iters 1` 是死参数，
   实际也跑满了 block-LM。§7 说「只验证了 refit 本身」并不准确——下游其实已经算了，
   只是没有保存 summary、没有做比较。P2''' 缺的是**比较**，不是**运行**。

这条同时解释了 §7 记的「全量单臂 `--joint-iters 1` 耗时超过 10 分钟」：那 10 分钟里
本就包含完整的 block-LM 求解。

### 4. 阶段 A 结果：300 轨迹两臂（可引用）

两臂只差 `--joint-refit-solver`，其余全默认（`--target-layers 17:8,24:4`、
`--active-dof 0,1,5`、`--rng-seed 42`、`--joint-solver block-lm`、CPU float64）。
墙钟：adam 224 s，gauss-newton 217 s。

#### 4.1 核对块（先于效果，全部逐位相同）

| 量 | adam | gauss-newton | verdict |
|---|---|---|---|
| `n_tracks` | 300 | 300 | same |
| `n_hits` | 4151 | 4151 | same |
| `input_tracks` / `retained_tracks` / `excluded_tracks` | 1200 / 300 / 900 | 1200 / 300 / 900 | same |
| `fit_hits` | 3450 | 3450 | same |
| `invalid_fraction` | 0.0 | 0.0 | same |

`stage_layer_chi2` 各层 `n_components` 也逐位相同（17:8 → 349，24:4 → 352）。
**样本量没有变化，以下差异不是丢行换来的。**

初始化数字逐位复现 Iteration 19 §4：adam 臂 `chi2/component = 7992.76`、非驻点行
100%、decrement p90 4.17e5；GN 臂 0.741103、0.00%、1.87e-17。两臂
`initial_chi2_per_component` 同为 10918，起点一致。

#### 4.2 主指标 `truth_metrics`（GN 臂全面占优）

只列有信号的 DOF（tz/rx/ry 的 `true_rms` 为 1e-14 量级，无信号，两臂均为 0）：

| 层 | DOF | 量 | adam | gauss-newton |
|---|---|---|---|---|
| 17:8 | tx | corr / recovery | +0.346 / +0.212 | **+0.419 / +0.240** |
| 17:8 | ty | corr / recovery | **−0.100 / −0.114** | **+0.525 / +0.281** |
| 17:8 | rz | corr / recovery | +0.103 / +0.028 | **+0.316 / +0.056** |
| 24:4 | tx | corr / recovery | +0.256 / +0.215 | **+0.336 / +0.208** |
| 24:4 | ty | corr / recovery | +0.080 / +0.095 | **+0.237 / +0.224** |
| 24:4 | rz | corr / recovery | +0.216 / +0.063 | **+0.297 / +0.067** |

**最重要的一条**：`17:8` 的 `ty` 在 Adam 臂上 correlation 与 recovery **都是负的**——
预测的错位方向与真值相反。GN 臂把它翻正到 +0.525 / +0.281。这不是「精度提高」，
是**符号纠正**：Adam refit 的非驻点起点让下游把该自由度解到了反方向。

唯一不占优的是 `24:4` 的 `tx.recovery`（+0.215 → +0.208，−0.007），同层 corr 仍从
+0.256 升到 +0.336，`pred_rms` 从 0.434 降到 0.320（`true_rms` 0.517）——GN 臂的过冲
更小。这一项按预注册记为「无实质差异」。

#### 4.3 次指标 `target_metrics` 与分阶段 chi2

`joint` 阶段的 per-component chi2（两臂 `n_components` 相同）：

| 层/分量 | adam | gauss-newton | 倍数 |
|---|---|---|---|
| 17:8 LOC0 | 169.72 | **27.16** | 6.2× |
| 17:8 LOC1 | 7775.99 | **7.57** | **1027×** |
| 24:4 LOC0 | 48.76 | 8.03 | 6.1×（P1' 不可引用）|
| 24:4 LOC1 | 1.85 | **0.267** | 6.9× |

残差 RMS（after）：17:8 LOC0 0.195 → 0.078 mm，17:8 LOC1 1.323 → 0.041 mm，
24:4 LOC1 1.630 → 0.620 mm。

`best_total_loss`：**1990.63 → 10.73（185×）**，两臂 `best_iteration` 同为 19、
`stopped_early` 同为 False——**在同一迭代数下比较，不存在「一臂多跑了几步」的干扰**。

#### 4.4 阶段 A 判读

按预注册 §1.3，逐目标层：

- **17:8**：GN 臂**优于** Adam 臂。三个有信号 DOF 的 corr 全面提升，`ty` 从反号纠正为
  正相关，LOC1 的 joint chi2 降低三个数量级。
- **24:4**：GN 臂**优于** Adam 臂（`ty`、`rz`、`tx.corr` 提升；`tx.recovery` 无实质
  差异）。LOC0 的数字按 P1' 不进结论，判读只依据 LOC1 与 `truth_metrics`。

### 5. 阶段 B 结果：全量两臂（可引用）

命令与阶段 A 只差去掉 `--max-tracks 300`。墙钟：adam **5831 s**，gauss-newton
约 **5640 s**（各约 1.6 小时）。**耗时不线性于轨迹数**——205 倍的 component 只带来
约 26 倍的墙钟，因为 block-LM 按 sensor 分块，每个 sensor 只处理打到它的 hits。
单臂 RSS 峰值约 6.6 GB（交接 §6 记的「约 3 GB」偏低，两臂并行必然 OOM，**必须串行**）。

#### 5.1 核对块（全部逐位相同）

| 量 | adam | gauss-newton | verdict |
|---|---|---|---|
| `n_tracks` | 61482 | 61482 | same |
| `n_hits` | 852859 | 852859 | same |
| `input_tracks` / `retained_tracks` / `excluded_tracks` | 98958 / 61482 / 37476 | 同左 | same |
| `fit_hits` | 709672 | 709672 | same |
| `invalid_fraction` | 0.0 | 0.0 | same |

各层 `n_components` 也逐位相同（17:8 → 71345，24:4 → 71842）。初始化数字逐位复现
Iteration 19 §7：adam 7595.34 / 100% / 4.1e5，GN 0.763503 / 0.00% / 2.61e-17，
两臂 `initial_chi2_per_component` 同为 10349.3。

#### 5.2 主指标 `truth_metrics`（决定性结果）

| 层 | DOF | 量 | adam | gauss-newton |
|---|---|---|---|---|
| 17:8 | tx | corr / recovery | +0.967 / +0.874 | **+0.9998 / +0.9986** |
| 17:8 | ty | corr / recovery | **−0.115 / −0.275** | **+0.99993 / +0.99973** |
| 17:8 | rz | corr / recovery | +0.956 / +0.711 | **+0.987 / +0.832** |
| 24:4 | tx | corr / recovery | +0.941 / +0.938 | **+0.9998 / +0.9997** |
| 24:4 | ty | corr / recovery | +0.336 / +0.679 | **+0.954 / +1.017** |
| 24:4 | rz | corr / recovery | +0.982 / +0.811 | **+0.989 / +0.843** |

GN 臂在全量上把两个目标层的 **tx 与 ty 恢复到几乎完美**（corr ≥ 0.95，
recovery ≈ 1.0）。`pred_rms` 同时从过冲收敛到真值尺度：17:8 ty 的 `pred_rms`
1.2397 → **0.5214**（`true_rms` 0.5215），24:4 ty 1.0543 → 0.5538
（`true_rms` 0.5197）。**Adam 臂不只是解得不准，而是把 ty 解成了 2 倍幅度的过冲，
且 17:8 上符号还是反的。**

`24:4` 的 `ty.recovery` = 1.017 是轻微过冲（1.0 为完美），corr 0.954，仍属大幅改善。

#### 5.3 次指标与分阶段 chi2

`joint` 阶段 per-component chi2（两臂 `n_components` 相同）：

| 层/分量 | adam | gauss-newton | 倍数 |
|---|---|---|---|
| 17:8 LOC0 | 186.91 | **38.95** | 4.8× |
| 17:8 LOC1 | 8755.62 | **6.73** | **1302×** |
| 24:4 LOC0 | 231.50 | 10.10 | 22.9×（P1' 不可引用）|
| 24:4 LOC1 | 2.394 | **0.959** | 2.5× |

残差 RMS（after）：17:8 LOC0 0.205 → 0.094 mm，17:8 LOC1 1.404 → **0.039 mm**，
24:4 LOC1 1.857 → 1.175 mm。

`best_total_loss`：**2286.55 → 14.16（162×）**。两臂 `best_iteration` 同为 19、
`stopped_early` 同为 False——**同一迭代数下比较，无「一臂多跑了几步」的干扰**。

#### 5.4 小样本对全量的代表性（再次验证）

| 量 | 300 轨迹 | 全量 |
|---|---|---|
| adam `best_total_loss` | 1990.63 | 2286.55 |
| GN `best_total_loss` | 10.73 | 14.16 |
| adam 17:8 ty corr | −0.100 | −0.115 |
| GN 17:8 ty corr | +0.525 | +0.99993 |
| adam joint 17:8 LOC1 chi2 | 7775.99 | 8755.62 |
| GN joint 17:8 LOC1 chi2 | 7.57 | 6.73 |

两个尺度**结论同向**，`--max-tracks 300` 的快速筛查再次被证明可用。但注意
**改善幅度随样本量显著放大**（17:8 ty corr 从 +0.525 到 +0.99993）：小样本可以定
方向，**不能定幅度**。

### 6. 判读结论（按预注册 §1.3）

逐目标层，全量结果：

- **17:8**：GN 臂**优于** Adam 臂。tx/ty/rz 三个有信号 DOF 的 corr 与 recovery 全面
  提升；`ty` 从**反号**（−0.115 / −0.275）纠正为近乎完美恢复（+0.99993 / +0.99973）；
  LOC1 的 joint chi2 降低三个数量级。
- **24:4**：GN 臂**优于** Adam 臂。tx 近乎完美（+0.9998 / +0.9997），ty 从 +0.336 /
  +0.679 提升到 +0.954 / +1.017，rz 小幅提升。LOC0 的数字按 P1' 不进结论，判读只依据
  LOC1 与 `truth_metrics`。

**P2''' 关闭。** 下游影响已系统评估：GN 替换不仅修好了 refit 本身的驻点性，还使下游
几何恢复从「tx 尚可、ty 失败」变为「tx/ty 双双近乎完美」。Iteration 19 的求解器替换
**在下游得到确证，无需回退**。

**这也回答了 Iteration 18 提出的归因问题**：Adam refit 停在非驻点（decrement p90
4.1e5，阈值 1e-6）不是无害的收敛不足，它给 block-LM 提供了一个系统性偏斜的起点，
使下游把 `17:8` 的 ty 解到了反方向。**归因正确时，一个求解器替换就够了。**

### 7. 遗留

- `24:4|backward_extrapolation|LOC0` 的 NLL 结论仍**不得引用**（P1'，未变）。本轮
  该组的 LOC0 数字虽同向改善，但不进结论。
- `seed_helices` 的 Adam 预拟合仍未检验（P4a）——**现在是唯一剩下的未检验 Adam 环节**。
- Gate 4 维持 failed，工作包 ④ 仍被 Gate 5 阻塞（未变）。
- 结论仍只在 20000 events 上，未在 100000 events 上复现（P4b，未变）。


---

## Iteration 21 — `seed_helices` 的 Adam 预拟合驻点性测量（P4a）

**性质**：**纯测量，零行为变更**。不改产品代码，测量在
`scratchpad/p4a/measure_seed.py` 中离线完成。沿用 Iteration 18 的做法：先只测量，
确认有问题再谈替换。

### 1. 测什么，为什么不能直接套 1e-6

`initialization_stats.measurement_refit.stationarity` 里已有的 `seed` 项测的是
**seed 输出在 measurement 目标上**的驻点性（即 refit 的起点），全量 100% 非驻点、
decrement p90 5.38e5。那不是 seed 自己的目标。

`seed_helices`（`scripts/ml_alignment_autograd.py:548`）的目标是**非目标 SMT 全局点的
几何距离平方和**：`loss = Σ‖predicted − points‖² / count`，单位 mm²，**无协方差加权**。
预注册的 `NEWTON_DECREMENT_STATIONARY = 1e-6` 是在 `F = 0.5·chi2` gauge 下定的，
**gauge 不同，不能直接套用**。因此本轮采用两个无 gauge 争议的判据，外加一个绝对值仅
供记录：

1. **预算外延**：从 200 次迭代的 seed 继续跑 Adam，看 SMT loss 还能降多少。
2. **相对 Newton decrement** `dec/F`：per-track 的 `gᵀH⁻¹g` 除以该 track 的 `F`，
   无量纲，直接读作「目标函数还能降低百分之几」。
3. 绝对 decrement（mm², SMT gauge）：只记录，**不与 1e-6 比较**。

第三个判据是直接求解：用 per-track Levenberg-Marquardt 解同一个 SMT 目标，
看真正的最小值在哪。

### 2. 结果（300 轨迹）

`seed_helices(n_iter=200)` 的 SMT loss = **1.7712071**，与 pipeline 打印的
`SMT loss=1.77121` 一致——测量脚本确实复现了产品代码的行为。

**`n_effective` 全程恒为 3450，逐项核对未变**（Iteration 17 §4.1 的教训：
loss 是对 effective components 取平均的，helix 跑飞会让点变 invalid 从而缩小分母；
本轮分母没有动，下面的改善不是分母缩小换来的）。

| 判据 1：继续跑 Adam | SMT loss | n_effective | vs 200-iter |
|---|---|---|---|
| 基线（产品代码，200 iters）| 1.7712071 | 3450 | — |
| +200 | 1.6254709 | 3450 | +8.2% |
| +400 | 0.054073131 | 3450 | +96.9% |
| **+800** | **0.0030914347** | 3450 | **+99.8%** |
| +1800 | 38.230581 | 3450 | **−2058%（发散）** |
| +2800 | 102.92118 | 3450 | **−5711%（发散）** |

| 判据 2：相对 decrement `dec/F` | 值 |
|---|---|
| p50 | 0.2408 |
| p90 | 0.6495 |
| p99 | 1.064 |
| `dec/F > 1%` 的轨迹占比 | **100.00%** |
| `dec/F > 10%` 的轨迹占比 | **86.67%** |

| 判据 3：LM 直接求解同一目标 | 值 |
|---|---|
| GN 求解后 SMT loss（30 次迭代）| **0.0040002676**（−99.774%）|
| 求解后 `dec/F` p50 | **1.02e-08** |
| 求解后 `dec/F` p90 | 0.0744 |

绝对 decrement（mm²，仅记录，不与 1e-6 比较）：p50 1.93、p90 4.311、p99 9.50、
max 18.28，非有限 0。

### 3. 判读

**`seed_helices` 的 Adam 预拟合确实停在非驻点，且离真正的最小值差约三个数量级**
（1.771 vs 约 0.004）。中位轨迹上目标函数还有 24% 可降，**100% 的轨迹**相对
decrement 超过 1%。这与 Iteration 18 对 measurement refit 的测量结论同构：
**不是问题难，是优化器不对**——LM 用 30 次迭代就把 p50 的相对 decrement 压到 1e-8。

**但有一条重要限定：「多跑几次 Adam」不是安全的修法。** 同一个 lr = 0.5 在约 1000 次
迭代后**发散**（+1800 时 loss 涨到 38.2，+2800 涨到 102.9，而 `n_effective` 未变，
说明是真发散不是样本变化）。所以 `JOINT_SMT_SEED_ITERS` 调大是错的方向；
若要改，应换求解器而非加预算。

### 4. 全量复现

同一脚本 `--max-tracks 0`：61482 轨迹、709672 fit hits。基线 SMT loss =
**1.6941403**，与 pipeline 打印的 `SMT loss=1.69414` **逐位一致**；
`n_effective = 709672` 与 `fit_hits` 相同且**全程未变**。

| 量 | 300 轨迹 | 全量 |
|---|---|---|
| 基线 SMT loss（200 iters）| 1.7712071 | 1.6941403 |
| +400 Adam | 0.054073131（−96.9%）| 0.065015727（−96.2%）|
| **+800 Adam** | **0.0030914347（−99.8%）** | **0.0029251182（−99.8%）** |
| `dec/F` p50 | 0.2408 | 0.2384 |
| `dec/F` p90 | 0.6495 | 0.7083 |
| `dec/F > 1%` 占比 | 100.00% | **99.98%** |
| `dec/F > 10%` 占比 | 86.67% | **87.50%** |

绝对 decrement（mm²，仅记录）：p50 1.909、p90 4.567、p99 11.35、max 42.13。

LM 求解在全量上只跑了 **15** 次迭代（为控制耗时），loss 降到 0.044719603（−97.36%），
`dec/F` p50 = 0.0798。**这是迭代预算不足，不是全量更难**——300 轨迹用 30 次迭代时
p50 已到 1.02e-08。此项不作为幅度结论，只作为方向佐证。

**两个尺度结论同向且量级一致**，与 Iteration 20 §5.4 的观察一致：小样本可定方向。

### 5. 判读与下一步

**P4a 的测量结论：`seed_helices` 的 Adam 预拟合确实停在非驻点。** 全量上 99.98% 的
轨迹相对 decrement 超过 1%、87.50% 超过 10%，中位轨迹目标函数还有 23.8% 可降；
继续跑 800 次迭代即可把 SMT loss 降低约三个数量级（1.694 → 0.00293），
`n_effective` 不变。这与 Iteration 18 对 measurement refit 的诊断同构。

**但这一次「有问题」不等于「要改」，理由是下游已经被兜住了**：

- Iteration 20 已实测，GN refit 从这个非驻点 seed 出发，在全量上达到
  measurement 非驻点行 **0.00%**、5 次迭代收敛。
- 同一次运行的下游几何恢复已经近乎完美：两个目标层的 tx/ty
  corr ≥ 0.95、recovery ≈ 1.0，`17:8` 的 ty 达 +0.99993 / +0.99973。

也就是说，seed 的非驻点性**已被 GN refit 吸收**，可提升的空间被 recovery ≈ 1.0
的天花板限制。因此：

- **P4a（测量）到此关闭**，结论是「确有非驻点，但当前不构成下游缺陷」。
- **P4a'（替换 seed 求解器）降级为可选优化，不是修复。** 若要做，判据不应是
  「SMT loss 是否更低」（那已确定会更低），而应是**下游 `truth_metrics` 是否还有可
  测量的改善**——在 recovery 已 ≈ 1.0 的情况下，这需要先预注册「多大的变化才算真
  改善」，否则只会测到噪声。
- **不要调大 `JOINT_SMT_SEED_ITERS`**：同一 lr = 0.5 在约 1000 次迭代后发散
  （300 轨迹上 +1800 → 38.2，+2800 → 102.9，`n_effective` 未变），加预算会把一个
  「未收敛但稳定」的 seed 换成一个「可能发散」的 seed。


---

## Iteration 22 — 外推窗口的欠定行：hits 阈值被证伪，改用残差数判据（P3）

**改动**：`scripts/validate_helix_model.py`、`tests/test_validate_helix_model.py`
**性质**：**预注册变更**。`PREREGISTRATION_VERSION` 2 → 3。

### 1. 预注册（写在跑任何一次全量之前，事后不得改）

#### 1.1 起点

Iteration 16 §6 把 `24:4|backward_extrapolation` 的秩亏行归因到预注册的
`min_extrapolation_fit_hits = 4` 过松：single 臂 4 hits 撑 5 个 gauge-fixed 参数，
丢弃的 8 行 `n_hits` 全是 4–5。P3 就是这一条挂账。

#### 1.2 选值依据（唯一合法依据）

**gauge-fixed 参数计数 + 样本量保留。** 在定值阶段**不得查看** RMSE、NLL 或任何
效果指标；违反此条则本次变更作废重来。

#### 1.3 核对顺序（强制：先样本量，后效果）

沿用 Iteration 17 §4.1、19 §4、20 §1.1：

1. **插值组的 N 与 RMSE 逐位不变**——插值走 `min_fit_hits` / `min_fit_span`，
   本次改动只作用于外推路径，变了即是泄漏，停止并查 bug。
2. **未受影响的外推组 N 逐位不变**——本次判据只应剔除残差不足的行。
3. 各必需组 N ≥ `MIN_GROUP_COMPONENTS = 200`，否则判决 `inconclusive`。
4. `n_rank_deficient_rows` 全组为 **0**。
5. 以上全部通过，才读 RMSE / NLL / damping 敏感度。

#### 1.4 效果判据

- `24:4|backward_extrapolation|LOC0` 的 damping 敏感度（|ΔNLL 变化| / CI 半宽，
  跨 `DAMPING_STABILITY_SCAN = (0.0, 1e-12)`）**< 1** → P1' 关闭，该组 NLL 恢复可引用。
- Gate 4 判定**预期仍是 failed**。插值组不受影响，而通过与否由
  `INTERPOLATION_RMSE_IMPROVEMENT = 0.20` 主导，插值组改善只有 0.3–1.8%。
  **本次变更不是为了让 Gate 4 通过**，判据一律不放宽。

### 2. hits 阈值被实测证伪

C2 在 Gate 4 的实际输入（`trackstates_kf_mat1.root`，20000 events）上只跑选择阶段、
不跑拟合，统计各阈值下每组的 LOC0 分量数。`floor = 4` 一列与 Iteration 14 的分组表
**逐位相同**，证明该测量确实复现了 Gate 4 的选择阶段。

| 组 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| 17:8 fwd | 5034 | 999 | **42** | **0** | **0** |
| 17:8 bwd | 14190 | 14125 | 13527 | 12758 | 7843 |
| 24:4 fwd | 14261 | 9539 | 2768 | 270 | **1** |
| 24:4 bwd | 14324 | 13589 | 12869 | 6810 | 3405 |
| 17:8 interp | 8346 | 8346 | 8346 | 8346 | 8346 |
| 24:4 interp | 14197 | 14197 | 14197 | 14197 | 14197 |

**`floor = 6` 让 `17:8|forward_extrapolation` 掉到 42 < 200，判决直接变
`inconclusive`；7 与 8 让它归零。** 计划阶段的先验推荐 6 就此被否决。

两条同时被推翻的先验：

- 「健康行 `n_hits` 中位数是 7」不适用于外推窗口。`extrapolation_hits` 在
  `build_track_batches` 里**既是门槛又是窗口大小**（`before_all[-N:]` 是截取），
  所以外推行的 fit hits 恒等于 floor：实测 p50 = min = max = 4。
- 抬高 hits 阈值**不是**对症的工具。见下。

### 3. 真正的欠定条件是残差数，不是 hits 数

strip 命中只贡献 LOC0，pixel 命中贡献 LOC0 与 LOC1，所以 hits 数只是残差数的代理量。
在当前 `floor = 4` 下按 `sum(1 + loc1_valid)` 统计全部 47809 个外推行：

| 窗口内残差分量数 | 行数 |
|---|---|
| **4（欠定）** | **3** |
| 5（恰定，零冗余） | 299 |
| 6 | 9232 |
| 7 | 11766 |
| 8 | 26509 |

- 欠定行**只有 3 行，全部在 `24:4|backward_extrapolation`**——正是 Iteration 16 §6
  记录的「single ×3，`n_hits` = 4，`μ_min` 为负」。
- `17:8|forward_extrapolation` 的 5034 行**残差数最小值就是 8**，一个病态行都没有。
  `floor = 6` 要砍掉其中 4992 行去换那 3 行，代价是整组消失。

**结论：判据下错了维度。** 因此本轮不动 `min_extrapolation_fit_hits`，改为直接对
残差数设下界。

### 4. 本轮的预注册常量

| 常量（`scripts/validate_helix_model.py`）| 值 |
|---|---|
| `DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS` | **6** |
| `PREREGISTRATION_VERSION` | 2 → **3** |
| `DEFAULT_MIN_EXTRAPOLATION_FIT_HITS` | **4（不变）** |

6 = 5 个 gauge-fixed 参数 + 1 个自由度冗余。**不是看着输出挑的**：4 与 5 分别是欠定与
恰定，7 以上没有理由停在任何特定值。

按 §3 的分布，该判据只剔除 302 行（`24:4|bwd` 300、`17:8|bwd` 2），其余四组一行不动，
所有必需组仍远超 200。

`PREREGISTRATION_VERSION` 要 bump 的理由：该常量虽不在那个 block 内，但它决定 Gate 4
的输入样本集合，v2 与 v3 的 summary 不可混读。

### 5. 核对块（全量重跑，按 §1.3 顺序）

运行：`data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4/residual_floor_20260827/`
命令与 Iteration 17 基线只差 `--output-dir`（基线 `damping_v2_20260826/`，未被覆盖）。
墙钟 28 分钟（Iteration 14 记的 53 分钟，本轮更快）。

**1. 插值组的行集合逐位不变。** 17:8 interp `n` = 8346、24:4 interp `n` = 14197，两跑相同。

RMSE **不是**逐位相同，相对漂移 2.6e-10 ~ 1.2e-09。**这不是泄漏，而且要求它逐位相同
本身就是错的判据**：行按 2048 行 chunk 拟合，在别处剔除 302 行会改变 chunk 边界，
float64 的累加顺序随之改变。判据应是「行集合不变 且 漂移不超过舍入量级」。

**2. 只有含病态行的组移动。** 按 `n_rows_in_group`（选择阶段）：

| 组 | v2 | v3 | Δ |
|---|---|---|---|
| 17:8 interp | 8346 | 8346 | 0 |
| 17:8 fwd | 5034 | 5034 | 0 |
| 17:8 bwd | 14190 | 14188 | **−2** |
| 24:4 interp | 14197 | 14197 | 0 |
| 24:4 fwd | 14261 | 14261 | 0 |
| 24:4 bwd | 14324 | 14024 | **−300** |

与 §4 预注册的 −302 逐位吻合。

**首次核对时这一项报了 −295**，因为拿它去比 v2 的 `n` 而不是 `n_rows_in_group`：
v2 的 24:4 bwd 在协方差路径上**事后丢了 5 行秩亏行**（14324 → 14319），拿它作分母等于
把旧规则下的删除记到本次变更头上。**v3 的 `n_rows_in_group` = `n` = 14024，事后一行未丢。**

**3. 各组均 ≥ `MIN_GROUP_COMPONENTS` = 200。** 最小组 17:8 fwd = 5034。

**4. 秩亏行归零。**

| 臂 | v2 | v3 |
|---|---|---|
| single | **3 行**，`min_eigenvalue` = **−3.75e-10（负）**，条件数无法计算 | **0 行**，+3.6967e-03 |
| broken | **5 行**，条件数 7.51e+18 | **0 行**，+3.6968e-03 |

**P3 的直接目标达成**：欠定行在选择阶段就被挡住，不再需要事后丢弃。

### 6. 效果

#### 6.1 Gate 4 判定：仍是 failed（与预注册预期一致）

插值组一行未动，改善仍只有 0.18%–3.18%，远不及 `INTERPOLATION_RMSE_IMPROVEMENT = 0.20`。
判据未做任何放宽。

#### 6.2 意外的正面结果：24:4 backward 的 broken 回退消失了

| | v2 single | v2 broken | v3 single | v3 broken |
|---|---|---|---|---|
| 24:4 bwd LOC0 RMSE | 0.786690 | **0.790822** | 0.787503 | **0.787384** |
| 24:4 bwd LOC1 RMSE | 3.246447 | **3.296940** | 3.229907 | **3.243938** |

Iteration 14/16 里 24:4 backward 是**唯一 broken 臂回退的组**。剔除那 300 行之后两臂基本
持平（LOC0 0.7875 vs 0.7874）。**那个回退来自欠定行，不是模型。**

#### 6.3 P1' **未关闭**：damping 敏感度不降反升

| 组 | v2 | v3 |
|---|---|---|
| `24:4|bwd|LOC0` | 2.223 | **2.545** |
| `24:4|bwd|LOC1` | 0.356 | **2.766** |
| 其余十组 | ≤ 0.171 | ≤ 0.182 |

`unstable_groups` 仍是 `['24:4|backward_extrapolation|LOC0']`，`stable` 仍为 false。

分解（敏感度 = |ΔNLL 在 0 与 1e-12 之间的位移| / CI 半宽），确认不是样本量减少造成的假象：

| | 分子 | 分母（CI 半宽）|
|---|---|---|
| LOC0 v2 → v3 | 2.104e-03 → **2.316e-03（+10%）** | 9.466e-04 → 9.102e-04（−4%）|
| LOC1 v2 → v3 | 9.147e-05 → **2.590e-04（+183%）** | 2.572e-04 → 9.362e-05（−64%）|

**LOC0 的上升主要来自分子**，即 damping 引起的 ΔNLL 位移本身变大了。

### 7. 结论

- **P3 关闭。** 欠定行的根因是残差数不足，判据已落在残差数上，秩亏行两臂归零，
  且不再有事后丢行。`min_extrapolation_fit_hits` 保持 4，理由已随测量写进代码注释与
  版本守卫测试。
- **Iteration 16 的一处归因被推翻。** 它把「唯一有秩亏行的组」与「唯一 damping 真实
  敏感的组」当作同一根因的两个表现。秩亏行现已清零，**damping 敏感度反而上升**
  （2.223 → 2.545），所以这是两件事。
- **P1' 保持开启，判据不放宽。** `24:4|backward_extrapolation|LOC0` 的 NLL 结论
  仍**不得引用**。本轮该组的数字同样未进任何结论。
- **Gate 4 维持 failed**，插值组改善 0.18%–3.18%，第四次复现。

### 8. 尚未宣称

- **`24:4|backward_extrapolation` 的 damping 敏感度尚未归因。** 已排除欠定行这一项，
  下一个候选未定。该组仍是数值极端行的集中地。
- **hits 数与残差数的关系只在本数据集上测过。** 6 = 5 参数 + 1 冗余是参数计数得来的，
  但「4 hits 能给 4 到 8 个残差」的分布取决于 strip/pixel 的混合比例，换数据集需重测。
- **窗口大小仍等于 `min_extrapolation_fit_hits`。** 本轮只在门槛上加了残差判据，
  没有把窗口大小与门槛拆成两个参数；若将来要给外推臂更多冗余，那是另一次变更。
- 结论仍只在 20000 events 上，未在 100000 events 上复现（P4b 未变）。

---

## Iteration 23 — trial-007 的实际结果补记（主线端到端首次运行，未通过）

**日期**: 2026-08-27（补记 2026-08-25 的运行）
**状态**: **补记，无新实验、无代码改动**
**性质**: 文档修复。Iteration 12 记「trial-007 待运行」，实际它当日已跑完并 failed，
此后项目转去做 broken-helix 支线，结果始终未回填。任何只读文档的人都会以为
主线还没跑过端到端。

### 1. 运行标识

产物 `data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/trial-007/`，
manifest `status=completed`、`git_commit=191f1fa`、
`2026-08-25T05:25:55Z → 07:08:56Z`（约 1h43m），
command 为 `./run_surface_helix_experiment.sh --run-tag trial-007`
（即全部默认：`EVENTS=100000`、`EVENT_SPLIT_SEED=6006`、active DOF `(tx,ty,rz)`）。

### 2. 冻结阈值（`acceptance_baseline.json`）

calibration 9893 事件 / validation 9884 事件，bootstrap seed 6006 × 2000。

| 层:分量 | nominal_chi2 | predictive_upper_99 | threshold |
|---|---|---|---|
| 17:8 LOC0 | 0.6668 | 0.6931 | 1.0 |
| 17:8 LOC1 | 1.2577 | 1.6106 | 1.6106 |
| 24:4 LOC0 | 0.8208 | 0.8506 | 1.0 |
| 24:4 LOC1 | 0.9919 | 1.0303 | 1.0303 |

Iteration 12 提的「统一绝对阈值 1.0 不自洽」已由 bootstrap profile 解决：
两个 LOC1 阈值被抬到 nominal 之上；两个 LOC0 因 `max(·, 1.0)` 仍取 1.0，
比 nominal 基线宽 1.2–1.5 倍，属已知的保守选择。

### 3. 逐阶段 normalized_chi2（calibration 分区）

| 阶段 | 17:8 LOC0 | 17:8 LOC1 | 24:4 LOC0 | 24:4 LOC1 | acceptance_score |
|---|---|---|---|---|---|
| surface_warm2 | 15.54 | 20.13 | 3.23 | 1.042 | (4, 20.13, 35.94) |
| **joint_free** | **42.74** | **335.97** | **19.20** | **3.01** | **(4, 335.97, 396.90)** |
| surface_polish1 | 12.33 | 17.25 | 2.73 | 1.034 | (4, 17.25, 29.34) |
| surface_polish2 | 9.82 | 14.83 | 2.33 | 1.027 | (4, 14.83, 24.01) |

### 4. **joint 阶段是净损失，被守卫整段丢弃**

joint 把 17:8 LOC1 从 warm2 的 20.13 推到 **335.97**（差 16 倍）。
`run_surface_helix_experiment.sh:383-392` 的 `joint_improved` 守卫据此回退到 warm2
几何作为 polish 起点，内部 `stop_reason` 置为 `joint_not_improved`。

**最终几何 `aligned_transforms_trial-007_surface_polish2.json` 里没有任何 joint 贡献。**
主线迄今为止的最好结果，实际上是纯 surface 校准的结果。

根因是 measurement refit 未收敛：`chi2_per_component` 只从 10375 降到 **7595.64**，
`best_iteration=199` 即撞满迭代上限，`improved=true` 是 Adam 分支专有字段。
这与 Iteration 19 §7 的 Adam 臂 7595.34 同量级 —— **确认 trial-007 是 Adam 臂**。

`joint_free` 的 `truth_metrics`（`diagnostic_only`，不参与控制流）：

| 层 | tx corr/recovery | ty corr/recovery | rz corr/recovery |
|---|---|---|---|
| 17:8 | 0.514 / 0.897 | **−0.078 / −0.451** | 0.266 / 0.436 |
| 24:4 | 0.405 / 0.813 | 0.030 / 0.057 | — |

17:8 ty 是**负恢复**：joint 在这个自由度上比不做还糟。
`best_total_loss = 2323.02`，`correction_max_abs = 1.5`（**精确等于
`JOINT_CLAMP_T`，已饱和**），`best_iteration = 19`（**等于 `LM_MAX_ITERS - 1`，
预算 binding**），`stopped_early = false`。

### 5. blind validation（一次性，已用掉）

| 层:分量 | chi2/component | threshold | normalized | 判定 |
|---|---|---|---|---|
| 17:8 LOC0 | 9.821 | 1.0 | 9.82 | fail |
| 17:8 LOC1 | 24.852 | 1.611 | **15.43** | fail |
| 24:4 LOC0 | 2.473 | 1.0 | 2.47 | fail |
| 24:4 LOC1 | 1.028 | 1.030 | 0.998 | **pass** |

`failed_components = ["17:8:LOC0","17:8:LOC1","24:4:LOC0"]`，
`acceptance_score = [3, 15.43, 24.72]`。

`experiment_summary.json`：`passed = false`，
`stop_reason = "blind_validation_failed"`，`polish_attempts = 2`，
`joint_effective = true`。

### 6. 由此暴露的两个编排器缺陷（本轮只记录，修复见后续）

- **`stop_reason` 被无声丢弃**。`run_surface_helix_experiment.sh:467-470` 把
  `$stop_reason` 作为 `sys.argv[7]` 传入内联 Python，那段 Python **从不引用 argv[7]**，
  直接硬写 `"blind_validation_passed" if passed else "blind_validation_failed"`。
  于是 summary 写的是 `blind_validation_failed`，**看不出 joint 曾被回退**——
  真实内部状态是 `joint_not_improved`。要复原只能自己比对 5 个
  `acceptance_*.json` 的 score。这是主线唯一的成败记录文件。
- **`joint_effective` 与实际选择脱节**。同一行的
  `joint_effective = best_iteration>=0 and correction_max_abs>0` 只读 solver summary，
  不管 joint 几何是否被采纳。trial-007 里 joint 已被丢弃，该字段仍是 `true`。
  Iteration 12 §实现 定的通过条件「四个 validation 分量全部通过**且 joint 有非零
  有效更新**」，按现在的实现**可以在 joint 被完全丢弃的情况下满足**。

### 7. 结论

- **主线端到端跑过一次，未通过。** 4 个 validation 分量只过 1 个，最差差 15 倍。
- **joint 阶段在主线上迄今为零贡献**，且其失败可归因到一个已被修复的求解器缺陷。
- **该运行早于 `9f6a951`（2026-08-26 13:47，GN 成为 joint refit 默认）22 小时。
  主线从未用当前代码跑过。** 其 joint 结论不适用于当前代码。

### 8. 尚未宣称

- **没有任何证据说明换 GN 后主线 joint 会转正。** Iteration 19/20 的 GN 全量结果
  跑在主线配置**之外**（起点是原始错位几何而非 warm2、无 `--event-partition`、
  98958 条全用因而混入 calibration 与 validation 事件），
  按 §9「surface、SMT seed、measurement refit 和 block-LM 只读 train」的口径
  **不能充当主线业绩**。必须在编排器下重跑才算数。
- **block-LM 的几何解从未被证明收敛。** 三次全量运行全部 `best_iteration = 19`
  且 `stopped_early = false`，预算是 binding 约束。measurement refit 有预注册的
  驻点 raise，几何 block 一个都没有。尚未测量。
- **`correction_max_abs` 饱和的后果未评估。** trial-007 精确撞上
  `JOINT_CLAMP_T = 1.5 mm`，summary 无任何告警。
- 17:8 LOC1 的困难有一部分是物理的：nominal 自身 chi2/N 就有 1.2577，
  ShortStrip `ty` 弱可观测是已知限制，不是算法 bug。

---

## Iteration 24 — 主线复位：用当前代码重跑（trial-008）

**日期**: 2026-08-27
**状态**: **预注册已冻结，运行待执行**
**改动**: `scripts/run_surface_helix_experiment.sh`、`scripts/write_experiment_summary.py`（新增）、
`tests/test_write_experiment_summary.py`（新增）、`tests/test_run_surface_helix_experiment.sh`
**性质**: 编排层记录与守卫修复 + 一次单变量重跑。**不触碰任何求解器。**

### 1. 相对 Iteration 12/23 改变的假设

Iteration 23 确认：主线的 joint 阶段在 trial-007 里是净损失并被守卫整段丢弃，
而其 measurement refit 撞满迭代上限从未收敛（`chi2/component` 7595.64）。
Iteration 19 已把该求解器换成 Gauss--Newton（`9f6a951`，2026-08-26 13:47），
同一量降到 0.7635、非驻点行 100% → 0%。**该提交晚于 trial-008 之前唯一一次
主线运行 22 小时，主线从未在当前代码下跑过。**

新假设：**trial-007 的 joint 失败是求解器未收敛所致，而非「固定轨迹只由目标层
measurement 解几何」这个方法本身有问题。** 若成立，用当前代码重跑应使 joint
阶段首次产生正贡献。

### 2. 预注册（**写在跑之前，事后不得改**）

**唯一变更**：measurement refit 由 Adam 换为 Gauss--Newton。实操上即「用当前代码
重跑」——GN 已是 `ml_alignment_autograd.py:93` 的默认，编排器不传
`--joint-refit-solver`，自动继承。其余全部沿用 trial-007 的默认：
`EVENTS=100000`、`EVENT_SPLIT_SEED=6006`、active DOF `(tx,ty,rz)`、
`SURFACE_EPOCHS=200`、`LM_MAX_ITERS=20`、`JOINT_SMT_SEED_ITERS=200`。

**复用 trial-007 的 `acceptance_baseline.json`**，不重新标定
（同 split seed 6006、同 nominal states `sha256=317257a2c753d46a...`），
否则两次运行不可比。

**主判据（可证伪）**：joint_free 阶段在 calibration 上的 `acceptance_score`
必须**严格优于** warm2 的 `(4, 20.13, 35.94)`，即 joint 不再被
`run_surface_helix_experiment.sh` 的 `joint_improved` 守卫回退
（等价于 summary 的 `joint_adopted == true`）。

**次判据**：`joint_free_summary.json` 的 `measurement_refit.chi2_per_component`
应落在 1 附近（离线 GN 全量是 0.7635）而非 7596 量级；非驻点行占比为 0。

**不预注册任何 validation 结论。** 本轮按决策**跑到 calibration 为止**
（`--skip-blind-validation`），不产生 validation 数字，把 §9 那次一次性盲测留到
配置真正定稿之后。

### 3. 本轮的编排器改动（三处，都在 finalize 段）

Iteration 23 §6 记录的两个缺陷会让本轮的主判据在 summary 里读不出来，故先修：

- **`--skip-blind-validation` 开关**（新增，默认关闭，默认行为不变）。
  置位时跳过 blind validation 阶段，`current_acceptance` 保持为 calibration 那份，
  `workflow` 记为 `...calibration-only`，`passed` 写 `null`，
  `stop_reason` 写 `stopped_before_blind_validation`，
  且不把不存在的 validation 产物登记进 manifest。
- **`stop_reason` 不再被丢弃**。原 finalize 把 `$stop_reason` 作为 `sys.argv[7]`
  传入却从不引用。现拆为两个字段：`selection_stop_reason`（选择阶段的真实状态，
  如 `joint_not_improved`）与 `stop_reason`（验证结论）。
- **`joint_effective` 改为反映几何是否真被采纳**：
  `joint_effective = joint_solver_effective and joint_adopted`。
  solver 层的原语义保留为 `joint_solver_effective`，新增 `joint_adopted`。
  这样 Iteration 12 的通过条件不再能在 joint 被完全丢弃时满足。

实现上把那段一行流内联 Python 提成 `scripts/write_experiment_summary.py`，
使其可被测试覆盖（原一行流无法测试）。

**回归验证**：用新代码路径重建 trial-007 的 summary，与实际产物逐字段比对——
无字段丢失、schema 2 保持，唯一的取值变化是 `joint_effective` 由 `true`
纠正为 `false`（trial-007 的 joint 确实被丢弃了）。

### 4. 尚未宣称

- **本条不含任何 trial-008 的结果。** 运行完成前不得引用任何效果数字。
- 主判据若不通过，下一个候选是 `seed_helices` 的 Adam
  （`ml_alignment_autograd.py:548-579`，无 flag、无驻点检查，P4a 实测 99.98%
  轨迹停在非驻点）。**不得用调大 `--joint-smt-seed-iters` 来糊弄**——
  同一 lr=0.5 在约 1000 次迭代后发散（见 Iteration 21）。
- block-LM 的几何解仍未被证明收敛（三次全量运行全部 `best_iteration = 19`
  且 `stopped_early = false`）。本轮只记录读数，不改。

### 5. 核对块（按 §2 预注册顺序，全部通过）

1. **输入哈希**：10 个输入（geometry / states / summary / nominal_states /
   simulation_dataset / nominal_geometry / acts_setup / conda_setup /
   odd_digi_config / odd_material_map）与 trial-007 **全部相同**。
2. **event split**：seed 6006、80/10/10，两次一致。
3. **阈值 profile**：四个阈值逐位相同，nominal states sha256 相同
   （复用 trial-007 的 profile，未重新标定）。
4. **样本量一行未动**：`n_tracks` 49222、`n_hits` 682915、`input_tracks` 79181、
   `retained_tracks` 49222、`excluded_tracks` 29959、`fit_hits` 568283，
   **全部逐位相同**。改善不是丢样本换来的。

**附带发现：流水线不是逐位可复现的。** warm2 阶段的 chi2/component 在两次运行间
有 **1e-7 ~ 1e-6 相对漂移**（`n_components` 7155/7170 逐位相同，四个分量判定不变）。
warm1/warm2 的代码路径本轮未改，故这是运行间浮点非确定性（torch 归约顺序或
ACTS 线程），不是真实变化。**对照 joint 阶段的 22×–424×，两者差六个数量级**，
不影响结论。今后比较两次运行时应判「行集合不变 且 漂移不超过舍入量级」，
而非要求逐位相同（沿用 Iteration 22 §4.2 的教训）。

### 6. 效果

#### 6.1 主判据：**通过，且是压倒性的**

| 阶段 | acceptance_score（calibration） |
|---|---|
| trial-008 warm2 | (4, 20.131, 35.945) |
| trial-007 joint_free | (4, 335.97, 396.90) |
| **trial-008 joint_free** | **(0, 0.9843, 0.0)** |

**四个分量全部通过**，joint 几何被直接采纳，`polish_attempts = 0`
（polish 循环首轮即 break，因为已无失败分量）。

逐分量 normalized_chi2（joint_free 阶段，calibration）：

| 层:分量 | trial-007 | **trial-008** | 阈值 | 倍数 |
|---|---|---|---|---|
| 17:8 LOC0 | 42.74 | **0.7158** | 1.0 | 60× |
| 17:8 LOC1 | 335.97 | **0.7919** | 1.6106 | **424×** |
| 24:4 LOC0 | 19.20 | **0.8695** | 1.0 | 22× |
| 24:4 LOC1 | 3.01 | **0.9843** | 1.0303 | 3× |

summary 字段：`joint_adopted = true`、`joint_solver_effective = true`、
`joint_effective = true`、`selection_stop_reason = "joint_acceptance_passed"`。
**这是 joint 阶段第一次在主线上产生正贡献。**

#### 6.2 次判据：通过

`measurement_refit.chi2_per_component` **7595.638 → 0.763146**
（初值两次均为 10373~10375），`invalid_fraction = 0.0`。
与离线 GN 全量的 0.7635 一致。Adam 分支专有的
`best_iteration = 199 / improved = true` 字段不再出现。

#### 6.3 truth_metrics 全线转正（`diagnostic_only`，不参与控制流）

| 层/DOF | trial-007 corr / recovery | **trial-008** |
|---|---|---|
| 17:8 tx | +0.5143 / +0.8967 | **+0.9988 / +1.0006** |
| **17:8 ty** | **−0.0778 / −0.4511** | **+0.9996 / +0.9996** |
| 17:8 rz | +0.2663 / +0.4362 | **+0.9981 / +0.9967** |
| 24:4 tx | +0.4047 / +0.8131 | **+0.9961 / +0.9917** |
| 24:4 ty | +0.0296 / +0.0573 | **+0.9214 / +0.8817** |
| 24:4 rz | +0.3493 / +0.6970 | **+0.9892 / +0.9826** |

六个 DOF 的 corr 全部 ≥ 0.92、recovery 全部 ≥ 0.88。
17:8 ty 从**负恢复**（对齐比不对齐更糟）转为 +0.9996。

#### 6.4 意外收获：Iteration 23 §8 记的两个隐患自行消失

| 量 | trial-007 | trial-008 |
|---|---|---|
| `best_iteration` | 19（= `LM_MAX_ITERS − 1`） | **15** |
| `stopped_early` | false | **true** |
| `correction_max_abs` | **1.5**（精确等于 `JOINT_CLAMP_T`，饱和） | **1.3695**（未饱和）|
| `best_total_loss` | 2323.018 | **2.2143** |

**block-LM 首次在预算内收敛**——Iteration 23 §8 记的「几何解从未被证明收敛、
预算是 binding 约束」在 GN 下自动解除；几何修正也不再撞硬夹逼。
根因是同一个：refit 未收敛 ⇒ 轨迹参数错误 ⇒ 几何求解被迫做大修正、
撞上 clamp 且永远收敛不了。**这两条不需要单独修。**

### 7. 结论

- **Iteration 24 的假设成立。** trial-007 的 joint 失败是 measurement refit
  未收敛所致，不是「固定轨迹只由目标层 measurement 解几何」这个方法的问题。
- **主线 joint 阶段首次产生正贡献**，且 calibration 上四分量全过、无需 polish。
- **Iteration 23 §8 的两个待测隐患（block-LM 收敛性、clamp 饱和）同时消解**，
  根因与 refit 未收敛同源。

### 8. 尚未宣称

- **未通过 blind validation，因此不得声称主线通过。** 本轮按决策
  `--skip-blind-validation` 跑到 calibration 为止，`passed = null`、
  `stop_reason = "stopped_before_blind_validation"`。
  **calibration 是候选选择用的分区，joint 几何正是在其上评的，
  不构成盲测证据。** §9 的一次性 validation 仍未动用。
- **`24:4 ty` 是六个 DOF 里最弱的**（corr 0.9214 / recovery 0.8817）。
  ShortStrip `ty` 弱可观测是已知物理限制，本轮未改善其相对地位。
- **17:8 LOC1 的裕度最小之一**（0.7919 对阈值 1.6106 看似宽裕，
  但该阈值本身已因 nominal chi2/N = 1.2577 被抬高）。
- **流水线非逐位可复现**（§5），今后的两次运行比较需按舍入量级判定。
- **`seed_helices` 仍是 Adam**，本轮未动。它没有成为瓶颈，但也未被证明无害。

### 9. Blind validation（补跑，2026-08-27）

§6 的 calibration 结果出来后，用户决定动用 §9 那次一次性盲测。
**判据无需新预注册**：阈值 profile 已冻结且与 trial-007 共用，
通过条件即 Iteration 12 定的「四个 validation 分量全部通过 且 joint 有非零有效更新」。

评估对象是 trial-008 的最终几何 `aligned_transforms_trial-008_joint_free.json`
（joint 输出，未经 polish），命令为
`reconstruction_acceptance.py --event-partition validation --split-seed 6006`
复用 `trial-008/acceptance_baseline.json`。

| 层:分量 | chi2/component | threshold | normalized | 判定 |
|---|---|---|---|---|
| 17:8 LOC0 | 0.7042 | 1.0 | **0.7042** | pass |
| 17:8 LOC1 | 1.3602 | 1.6106 | **0.8445** | pass |
| 24:4 LOC0 | 0.8851 | 1.0 | **0.8851** | pass |
| 24:4 LOC1 | 0.9890 | 1.0303 | **0.9600** | pass |

`failed_components = []`，`acceptance_score` **[3, 15.43, 24.72] → [0, 0.9600, 0.0]**。
`n_components` 7096 / 7239，与 trial-007 的 validation 分区逐位相同。

`experiment_summary.json`：`passed = **true**`、
`stop_reason = "blind_validation_passed"`、`joint_effective = true`、
`polish_attempts = 0`。

### 10. 最终结论

**主线 free-helix 流程首次通过端到端盲验证。** 四个 validation 分量全部通过，
joint 阶段有非零有效更新且几何被采纳，polish 一轮未用。

从 trial-007 到 trial-008 的**唯一变更是 measurement refit 由 Adam 换为
Gauss--Newton**（实操上即用当前代码重跑，GN 已是默认）。输入哈希、event split、
阈值 profile、样本量（49222 轨迹 / 682915 hits / 568283 fit_hits）**全部逐位相同**。

`data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_trial-008_joint_free.json`
是主线迄今第一份通过验收的几何。

### 11. 尚未宣称（承 §8，按盲测结果更新）

- **一次性盲测已用掉。** 后续任何配置变更若要再评 validation，
  必须先按 §9 处理：或显式声明本次结果属于旧配方，或改 `--event-split-seed`。
  **本轮结果不得用于继续调参。**
- **`24:4 ty` 仍是六个 DOF 里最弱的**（corr 0.9214 / recovery 0.8817），
  且 `24:4 LOC1` 的 normalized 0.9600 是四个分量里裕度最小的。
  ShortStrip `ty` 弱可观测是已知物理限制。
- **结论只在 `Vol17_Lay8__Vol24_Lay4` 这一组目标层上成立**，
  未在 `Vol16_Lay6__Vol23_Lay4` 等其它组合上复现。
- **`seed_helices` 仍是 Adam**（99.98% 轨迹停在非驻点）。它没有成为瓶颈，
  但也未被证明无害；P4a' 仍是可选优化。
- **block-LM 虽首次早停收敛（`best_iteration = 15`），仍无驻点门控。**
  measurement refit 有预注册的驻点 raise，几何 block 没有。
- **流水线非逐位可复现**（§5），两次运行比较需按舍入量级判定。
- broken-helix 支线的 Gate 4 仍是 failed，P1' 仍开启。本轮与之无关。


---

## Iteration 25 — SMT helix seed：Adam 换成阻尼 Gauss--Newton/LM

**日期**：2026-09-01
**性质**：求解器修复；不改变 SMT 点集、mask、目标函数或 measurement refit。

### 1. 根因与既有证据

Iteration 21 已在 300 轨迹和全量 61482 轨迹上把问题复现为求解器欠收敛，而不是样本
变化：200 步 Adam 的全量 SMT loss 为 1.6941403，99.98% 的轨迹相对 Newton decrement
大于 1%，87.50% 大于 10%。继续到额外 800 步虽可把 loss 降至 0.002925，但相同
`lr=0.5` 再继续会发散；因此增加 `JOINT_SMT_SEED_ITERS` 不是可靠修法。

直接原因是旧路径只在固定 200 步预算处停止：位置、方向和缩放曲率使用固定 Adam 步长，
没有驻点门槛、候选下降检查或信赖域。该非线性最小二乘还带 helix 原点沿轨迹滑动的一维
gauge，固定步长一阶下降无法同时稳定处理各尺度。Iteration 21 的同目标 LM 在 30 步内
把 300 轨迹 loss 降到约 0.004，并把相对 decrement p50 压到约 1e-8。

### 2. 修改

- `scripts/helix_model.py` 新增共享 SMT 点拟合器：固定五参数可观测 gauge、物理曲率缩放、
  逐轨迹阻尼、只接受目标下降且不丢失既有有效点的候选步，并按 2048 行分块形成 Jacobian。
- `scripts/ml_alignment_autograd.py` 与 `scripts/validate_helix_model.py` 的两份
  `seed_helices` 均改调该共享求解器；非目标/非 holdout mask 和原函数默认返回值不变。
- joint summary 的 `initialization_stats.smt` 与 helix validation summary 的 `smt_seed`
  记录 solver、迭代、目标下降和相对 Newton decrement，避免以后只看到迭代预算。
- 驻点按 row 判断；已驻点 row 冻结，拒绝但未驻点的 row 继续提高 damping。非驻点比例超过
  0.1% 时 joint 与 validation 均 fail closed，不能把另一个预算截断结果交给 measurement refit。
  任一 non-finite decrement 直接失败，不计入 0.1% 容许量。
- measurement refit 仍走既有、已验证的 Gauss--Newton 实现；SMT seed 与它统一的是 solver
  策略，不是目标函数。SMT 保持全局点欧氏距离，measurement 保持 half-χ²。

### 3. 回归验证

无噪声曲线测试中，旧 Adam 30 步 RMS 为 1.924 mm；新 LM 30 步低于 1e-3 mm。
带不可约扰动的双轨迹测试中，目标下降超过四个数量级、相对 Newton decrement max
低于 1e-6；1 行与整批求解在 3e-8 参数绝对误差内一致。另有 invalid-point、单步预算
耗尽、float32 下已驻点/未驻点混合 row 的回归覆盖。

本轮未运行新的 ROOT/ACTS 全量算法实验，因此没有新增实验 manifest，也不宣称下游几何
或 blind-validation 指标发生改善；全量非驻点归因沿用 Iteration 21 已归档的测量结果。

## Iteration 26 — SMT seed 迭代预算 200 → 2000（预注册）

**日期**：2026-09-02
**性质**：预算修正；不改目标函数、mask、点集，**不改任何预注册阈值**。
**状态**：预注册，运行前写入。结果回填于本条 §5。

### 1. 触发事件

`trial-009-fullevents`（全量事件、`--no-event-split`）在 joint_free 阶段失败：

```
ValueError: SMT helix seed is non-stationary:
  4.6355% of rows exceed the Newton-decrement criterion (allowed 0.1000%)
```

这是 Iteration 25 加入的 `require_stationary_helix_point_fit` **第一次在真实数据上执行**。
`2af0d4f` 提交于 2026-09-01 17:23，而此前所有运行的 `git_commit`（trial-008 为
`8194ff0`，9-01 的三次 joint-only 为 `f8cd8cc`）均早于它。Iteration 25 §3 亦自述
「本轮未运行新的 ROOT/ACTS 全量算法实验」。

### 2. 归因（探针，不触发门控、不跑下游）

**2.1 与「全量事件」无关。** 同一次 ROOT 加载，分别在全量与 train 分区上跑 seed：

| 分区 | 轨迹数 | `non_stationary_row_fraction` | 门控 |
|---|---:|---:|---|
| 全量 | 61482 | 4.6355% | FAIL |
| train | 49222 | **4.6524%** | FAIL |

train 比全量还高。取消哈希分段不是原因；它只是让一个从未执行过的门控开始执行。

**2.2 是预算截断，不是发散。** 两组均报 `converged=false, iterations=200`（撞满
`--joint-smt-seed-iters 200`），`n_non_finite=0`，最差行相对 Newton decrement 仅
7.4e-3 / 5.2e-3。与门控 docstring 的 "budget-truncated rows" 一致。

**2.3 seed 求解器换代的收益已经兑现。** train 分区 `seed_n_points = 568283` 与
trial-008 逐位相同，可直接比较：smt loss **1.6488 → 0.0028561 mm²**（577×）。
门控拒绝的是一个远好于 trial-008 的 seed。

**2.4 预算扫描（10000 轨迹子样本，同一批轨迹，只变 `max_iters`）**：

| `max_iters` | 实际用 | 非驻点比例 | 门控 | `decrement_max` | `final_objective` |
|---:|---:|---:|---|---:|---:|
| 200 | 200 | 4.8500% | FAIL | 2.634e-03 | 0.0157037 |
| 600 | **367** | **0.0000%** | PASS | 2.064e-05 | 0.0157036 |
| 2000 | 367 | 0.0000% | PASS | 2.064e-05 | 0.0157036 |
| 6000 | 367 | 0.0000% | PASS | 2.064e-05 | 0.0157036 |

求解器在第 367 步自然收敛；600/2000/6000 逐位相同。目标函数无回升。

### 3. 为何这不违反 Iteration 25 的禁令

Iteration 25 §1 的原话是「相同 `lr=0.5` 再继续会发散；因此增加 `JOINT_SMT_SEED_ITERS`
不是可靠修法」。**该判断的主语是 Adam。** `f85aaf0` 已把 `seed_helices` 换成阻尼
Gauss--Newton/LM，无学习率、逐行 damping、只接受下降候选。§2.4 在四个预算上实测
无发散。禁令自身写明的例外「要动就换求解器」已由 `f85aaf0` 满足。

**本条不放宽 `POINT_FIT_MAX_NON_STATIONARY_ROW_FRACTION = 1e-3`。** 阈值不动，
改的是让求解器有机会达到它的预算。

### 4. 预注册判据（运行前固定）

**主判据（通过/失败）**：全量 61482 轨迹的 seed 报 `non_stationary_row_fraction`
≤ 1e-3 且 `n_non_finite = 0`，joint_free 阶段不再因驻点门控中止。

**次判据（不得劣化）**：seed loss 相对 200 步基线（train 0.00285611、全量
0.00282184 mm²）上升不得超过 1%。若上升，判定为发散，回滚本改动。

**证伪条件**：若全量运行报 `iterations ≥ 2000`，说明存在真正不收敛的行，
**须停下重新归因，不得再调大预算**。

**明确不主张的事**：
- seed loss 若有改善，**不作为算法改善的证据**。§2.4 显示 200→367 步只让目标函数
  动了 6.4e-6 相对量。
- 本次运行的 acceptance 与 truth_metrics **与 trial-007/008 不可比较**：样本集不同
  （全量 vs train 分区），且验收未分段、阈值仍按 validation 规模标定因而偏松。
  本次运行的唯一目的是让全量流水线跑通并产出 smt_seed / measurement_refit / joint
  三阶段的 chi2 记录。

### 5. 修改与结果

- `scripts/run_surface_helix_experiment.sh`：`JOINT_SMT_SEED_ITERS` 默认 200 → 2000。
- **有意未改** `ml_alignment_autograd.py:83` 的 `JOINT_SMT_SEED_ITERS = 200`：该常量
  同时被 `validate_helix_model.py` 消费，那条路径有自己的预注册数字，改它需要独立论证。
  本条只修生产编排器的配方。
- 结果：`trial-010-fullevents`，2026-09-02 13:04--15:47，`status=completed`。

**预注册判据逐条核对**：

| 判据 | 门槛 | 实测 | 结论 |
|---|---|---|---|
| 主判据 | `non_stationary_row_fraction ≤ 1e-3` | **0.0** | 通过 |
| 主判据 | `n_non_finite = 0` | 0 | 通过 |
| 主判据 | joint_free 不因驻点门控中止 | 未中止，全流程走完 | 通过 |
| 次判据 | seed loss 相对 0.00282184 上升 ≤ 1% | **0.00282182**（-8e-6 相对） | 通过 |
| 证伪条件 | `iterations ≥ 2000` 须停下重新归因 | **390** | 未触发 |

seed 在全量 61482 轨迹上用 390 步收敛（10000 轨迹子样本为 367 步），`converged=true`，
`relative_newton_decrement_max = 2.06e-05`。600 的预算本可通过但裕度仅 1.5×，2000 为
封顶值且实测不增加耗时。

**三阶段 chi2 / 有效计数**（`stage_layer_chi2`；被拟合量与目标层诊断量分母不同）：

| 阶段 | 范围 | 层:分量 | chi2/component | 有效计数 |
|---|---|---|---:|---:|
| smt_seed | 被拟合（非目标 SMT 点） | — | 0.00282182 mm² | 709672 |
| measurement_refit | 被拟合（非目标 measurement） | — | **0.7635026** | 1154761 |
| smt_seed | 诊断（目标层） | 17:8 LOC0 | 1233.729 | 71345 |
| smt_seed | 诊断（目标层） | 17:8 LOC1 | 322.850 | 71345 |
| smt_seed | 诊断（目标层） | 24:4 LOC0 | 324.465 | 71842 |
| smt_seed | 诊断（目标层） | 24:4 LOC1 | 1.858 | 71842 |
| measurement_refit | 诊断（目标层） | 17:8 LOC0 | 2041.547 | 71345 |
| measurement_refit | 诊断（目标层） | 17:8 LOC1 | 1263.078 | 71345 |
| measurement_refit | 诊断（目标层） | 24:4 LOC0 | 367.324 | 71842 |
| measurement_refit | 诊断（目标层） | 24:4 LOC1 | 1.219 | 71842 |
| joint | 诊断（目标层） | 17:8 LOC0 | **2.032** | 71345 |
| joint | 诊断（目标层） | 17:8 LOC1 | **4.631** | 71345 |
| joint | 诊断（目标层） | 24:4 LOC0 | **1.246** | 71842 |
| joint | 诊断（目标层） | 24:4 LOC1 | **0.958** | 71842 |

refit GN：`converged=true`，5 步，objective 299.223 -> 7.17009，驻点行占比
100.00% -> 0.00%，decrement p90 1.16e+03 -> 5.67e-14。
block-LM：`best_iteration=13`，`stopped_early=true`，total loss 915.744 -> 2.2126，
`correction_max_abs=1.3079`（未撞 `JOINT_CLAMP_T=1.5`）。
`joint_adopted=true`，`polish_attempts=0`，`selection_stop_reason=joint_acceptance_passed`。

**§4 的「不主张」条款依然有效**：本次 acceptance（四分量 normalized 0.6675 / 0.8307
/ 0.8237 / 0.9410，全部 pass）**不得与 trial-007/008 比较** —— 样本集不同（全量
61482 vs train 分区 49222），验收未分段（n≈71600 vs trial-008 validation 的 7096），
且阈值仍按 validation 规模标定因而相对偏松。truth_metrics 同理，仅作 `diagnostic_only`。
中间阶段的目标层诊断 chi2 与 trial-008 亦不可比：seed 求解器已由 Adam 换成 GN，
两者落点不同；唯一同口径可比的是 refit 自身目标函数 0.763503 vs trial-008 的
0.763146。

**已知记录缺口**：`manifest.json` 记 `git_commit=28b04edfa`，但运行时工作树含未提交
改动（本条的编排器默认值、`--no-event-split` 支持、本日志），manifest schema 无 dirty
标记。该 commit 不足以复现本次运行；复现需在 `28b04edfa` 之上叠加 `c52f8de`
（`feat: run the mainline on every event and unblock the truncated helix seed`），
即本条 §5 所列的编排器与 summary 改动。**未回改 `manifest.json`** —— 事后修改运行
记录等于伪造，缺口以本条说明的方式记录。

---

## Iteration 27 — block-LM 收敛与固定有效集契约（roadmap L1）

**性质**：数值契约修复，**不是物理改动**。新增 `BLOCK_LM_PREREGISTRATION_VERSION = 1`，
与 `JOINT_REFIT_PREREGISTRATION_VERSION` 及 Gate 4 的 `PREREGISTRATION_VERSION` 刻意
分开 —— 后两者治理的判据本次一条都没动。
**状态**：**预注册，写在跑真实数据之前。** 合成层已完成，真实两臂待运行，结果回填本条 §6。
**交接**：`docs/handoff/2026-09-08-block-lm-convergence-contract.md`
**上游**：`docs/handoff/2026-09-07-alignment-methodology-roadmap.md` §L1

### 1. 实验问题

在冻结的 train 径迹和原始错位几何上，当前 block-LM 的早停是否满足逐 sensor 驻点／
边界 KKT 条件？改为逐 sensor damping、拒绝后重试、固定有效集接受之后，几何是否改变？

### 2. 预注册（**写在跑之前，事后不得改**）

**冻结的常量**（`scripts/ml_alignment_autograd.py`）：

```
BLOCK_LM_DECREMENT_STATIONARY = 1.0e-6     # 每 component，F = 0.5*chi2 gauge
BLOCK_LM_MAX_DAMPING_RETRIES  = 8
BLOCK_LM_MAX_DAMPING          = 1.0e12
BLOCK_LM_MIN_DAMPING          = 1.0e-12
BLOCK_LM_BOUND_ATOL           = 1.0e-12
```

`BLOCK_LM_DECREMENT_STATIONARY` 沿用 `NEWTON_DECREMENT_STATIONARY` 的数值与 gauge，
口径改为每 component（那里是每径迹行）。两者独立冻结，**不得互相调整**。驻点判据
一律不加正则：`(H + εI)^-1 ≼ H^-1`，加正则只会让 decrement 偏小，把「没收敛」报成
「收敛」，正是本次要消灭的失效模式。

**两臂**：
- **A（baseline）**：commit `e9c63b2`。只加观测量，求解过程逐位不变。
- **B（contract-fixed）**：commit `428ef57`。逐 sensor damping；拒绝后提高 damping 并
  重试；接受比较使用当前固定有效 component；以驻点／边界 KKT 而非「本轮无接受步」
  宣告收敛。

**必须冻结**：同一 train event-key hash；同一 refitted track 参数文件及哈希；同一目标
sensor、active DOF、measurement、covariance、初始几何；同一求交容差、Fisher floor、
步长上限与硬边界。**不运行 calibration，不运行 validation，不运行 surface polish。**

**预注册主判据**：所有被标记为 `converged` 的**非边界** sensor 均满足
`lambda2 / n_components <= 1e-6`；合成的「首次拒绝、增大 damping 后可下降」案例继续
迭代并到达直接解。

**次判据**：真实 train 输入上，候选不得丢失当前有效 component
（`min_components_delta_accepted >= 0` 且 `min_components_end_minus_start >= 0`）；
改进臂／基线臂的逐 sensor 几何 truth 误差 RMS 比值，其 paired 95% CI 上界不超过 `1.01`。

**证伪条件**：任一非驻点块仍被报告 `converged`，或 loss 改善依赖有效 component 减少。

**宣称边界**：若生产几何与基线等价，只能宣称「收敛契约得到验证」，**不能宣称精度提升**。
本条**不解释也不修正已知的 9.9 倍 truth 距离偏差** —— 该偏差的主判据是 truth 距离而非
chi2，成因在固定径迹的 block-LM 之外。

**truth 的地位**：真实物理场景中不存在 truth。它只在带标注实验里可得，只能用于校准
完成之后核对模型，**绝不进入求解器的接受或停止判据**。`compare_truth_rms.py` 是唯一
可以谈 truth 的地方，其输出**不得反过来指导求解器调参**。

### 3. A 臂逐位不变的验证（已完成）

两个合成 case 的 `predicted_deltas` 与 `history` 序列化后，在 `e9c63b2` 与 `37ed4f5`
之间 `diff`：**完全相同**。

### 4. 合成层判据核对（已完成，全部通过）

主判据 A 的独立重算在**物理单位**下进行（求解器工作在归一化单位）。`lambda2` 对对角
重标度不变（`g → Sg, H → SHS`），所以这是一条独立路径却必须给出同一个数；测试同时
断言 `rel=1e-6` 内相等。

主判据 B 用确定性注入而非真实非线性：两平面 fixture 对 `tx` 严格线性，二次目标下任意
`λ ≥ 0` 的 LM 步都是 GN 步的收缩，**永远不会被拒**，真实非线性构造会退化成永真式。
断言锚定「重试确实发生在同一 sensor、同一外层迭代」（`attempt == 1`、`damping` 恰好
×10、随后 `accepted`），而不只是「最后收敛了」。

`test_block_lm_recovers_target_translation` 的容差由 `abs=1e-5` 改为**闭式驻点界**
`sqrt(24 · 1e-6 · 0.01 / 12) = 1.4142e-4`。**这是修正而非放松**：驻点判据说的是
「模型还能给的下降 ≤ 1e-6/component」，从未承诺参数精度到 1e-5 mm。实测终点
`lambda2_per_component = 7.811e-9`，与闭式 `1200·(1.2499e-5)²/24` 精确吻合。推导与
逐迭代轨迹见交接文档 §7。

`484 passed, 3 skipped`（conda pytorch 3.10.20）。

### 5. 待运行

1. 跑真实 train A 臂（`e9c63b2`），finalize manifest，存 `summary_A.json`，核对
   `predicted_deltas` 与既有基线逐位相同；
2. 跑真实 train B 臂（`428ef57`）；
3. `compare_truth_rms.py` 比 A vs B；
4. **先核对行数／component 数／sensor 数，再读 loss 和 truth。**

**预期读数**：真实数据上 `rank_deficient` 比例可能很高。这是审计输出，不是 bug ——
Fisher floor 撑起的方向本就不是被数据识别的模式。

### 6. 结果（待回填）

（尚未运行。）

**结果解释规则（预注册，事后不得改）**：
- B 通过且 truth 改善 → 进入生产候选，但仍需新的 calibration 选择协议；
- B 通过且两臂几何等价 → 证明当前结果的数值稳定性，**停止继续改 block-LM**；
- B 驻点更好但 truth 变差 → 保留 baseline，转入 M1 查弱模／正则；
- 两臂都非驻点 → 继续归因求交 Jacobian、尺度或边界，**禁止直接调大迭代数**；
- 改善依赖丢 hit → 实验失败，修复接受规则后重新预注册版本。

---

## Iteration 28 — block-LM convergence contract v2 评审加固（预注册）

**性质**：Iteration 27 的代码评审修复；不改变物理模型、输入集合或 truth 的地位。
**状态**：写在真实 A/B 运行之前；合成回归完成，真实两臂仍待运行。

评审发现 v1 的四条 fail-open／漏报路径，因此
`BLOCK_LM_PREREGISTRATION_VERSION` 从 1 升为 2：

1. `compare_truth_rms.py` 将未定义 CI 显示为 `n/a`，但顶层仍可能返回 `passed`；v2 对
   非有限输入、非有限 point/CI、有限 bootstrap 数不足和零基线 RMS 一律 fail closed，
   报 `invalidated` 并返回非零。
2. sensor 有目标 hit、但当前所有交点都无效的时候，v1 因 `n_components_end is None`
   把它归为 `no_data` 并从终态分母排除；v2 将其归为 `intersection_failed`。
3. 接受步会反复将逐 sensor damping 减半，v1 的八次 ×10 重试可能从 `1e-12` 只爬到
   `1e-4`，没有达到预注册论证中的 `1e5`；v2 用重试编号锚定生产初值，保持绝对阶梯。
4. v1 只累计 `lstsq_fallbacks`，终态没有读取；v2 将任一次 fallback 作为粘滞的
   `rank_deficient` 证据，保持 §4 的优先级。

四条回归均先在 v1 上观察到预期失败，再以最小修改转绿。A 臂继续冻结为 `e9c63b2`；
`428ef57` 只代表原 v1 B 臂，不得用于正式真实数据对照。真实运行前必须先提交本次 v2、
记录新的 B commit，并继续遵守 Iteration 27 的输入冻结、双 manifest 和结果解释规则。

验证：conda `pytorch` Python 3.10.20 下，定向套件 `82 passed, 1 deselected`；完整
`make test` 为 `488 passed, 3 skipped, 11 deselected`。3 个 skip 仍是既有的 ACTS／
AlignDerivLib 环境跳过。

本轮没有运行真实算法实验，因此没有实验 manifest，也没有新增物理或精度结论。

---

## Iteration 29 — 当前工作树 100k 全事件闭环与一次性几何 truth 否决门（预注册）

**日期**：2026-09-10
**性质**：工程回归闭环；验证当前修改后的生产编排、数值契约、ACTS 回灌和候选几何
闭合。**不构成 blind validation、独立泛化或新精度声明。**
**状态**：预注册；以下规则写于本轮真实 ROOT/ACTS 运行之前，结果只能在本条末尾追加，
不得据结果改阈值、调参或另选候选。

### 1. 冻结输入与生产配方

- 复用现有 100k misaligned reconstruction：
  `trackstates_kf_v17l8_v24l4.root`（98,958 ROOT entries）及配套 tracksummary；
  不重新生成错位输入。
- target layers 固定 `17:8,24:4`，active DOF 固定 `0,1,5`（`tx,ty,rz`）。
- run tag 固定 `closure-current-100k-20260910`；事件数 100000、surface epochs 200、
  joint iterations 1000、SMT seed cap 2000、measurement refit cap 200、
  block-LM cap 60、`max_tracks=0`、bootstrap seed 6006。
- 工作树允许 dirty，但主实验与 truth-veto 都必须捕获 schema-2 source snapshot；
  不以 commit 单独声称可复现。

生产选择严格保持：

```text
misaligned input acceptance
  -> joint free-helix / block-LM
  -> ACTS reconstruction + same-sample acceptance
  -> at most two gain-0.2 surface-polish candidates
  -> freeze exactly one final candidate
```

候选只能由不读取 truth 的 acceptance score 选择。最终候选路径只能从
`experiment_summary.json` 读取，不预设是 joint 还是 polish。

### 2. 一次性 truth 诊断

旧 holdout-track truth-distance closure 同时从两臂 refit 中排除目标层 hit，而两臂
非目标几何相同；收敛后它主要测到同一个非目标测量最优点。20-event 实测 CI 上界只因
浮点误差落在 `0.99999999998...` 附近，不能判别候选 geometry，本轮不得用它作门。

新 `run_truth_veto_closure.py` 在候选冻结后只运行一次，直接比较 nominal、原始
misaligned 与最终 candidate geometry。在 misaligned 局部帧中，对每个 sensor 形成
truth correction、candidate correction 和剩余误差；以 sensor 为配对 bootstrap 单位，
2000 resamples、seed 6006、95% 上尾。六个预注册指标为：

```text
17:8 tx, 17:8 ty, 17:8 rz,
24:4 tx, 24:4 ty, 24:4 rz
```

每项 CI 上界都必须严格小于 1.0。任一项失败即否决本轮最终候选，不允许读取其他候选的
truth 后换赢家。sensor 集合不一致、重复 ID、非有限量、零/非有限 baseline RMS 或
bootstrap 非有限均 fail closed。

### 3. 工程、数值与物理判据

- `make test` 与 `make test-shell` 必须通过；truth-veto 不得被 solver、
  calibration、reconstruction acceptance 或生产 shell 引用。
- 主实验和 truth-veto manifest 均须 finalize，source snapshot 状态为 `written`。
- retained tracks 必须为 61,482；不符时先判输入/选择身份失败，不读取效果指标。
- SMT seed 与 measurement refit 非驻点比例均不超过 0.1%，且无非有限 decrement。
- block-LM preregistration version 必须为 2；
  `min_components_delta_accepted >= 0` 且
  `min_components_end_minus_start >= 0`。rank-deficient、boundary、
  budget-exhausted 必须显式记账，不得伪装成 converged。
- 最终 acceptance score 必须严格优于原始 misaligned baseline；四个目标层分量均
  `passed=true`，同时保持 `profile_type=diagnostic_only`、
  `verdict_eligible=false`。
- §2 六个几何 truth-error RMS ratio 的 paired 95% CI 上界全部 `< 1.0`。
- `trial-010-fullevents` 只旁路展示轨迹数、迭代数、acceptance 与几何 RMS，
  不进入选择或门控。

### 4. 失败语义与宣称边界

truth 科学判据失败属于一次完整实验：truth-veto manifest 记 `completed` 且
`verdict.passed=false`；输入、配对、数值、异常或中断故障记 `failed` 并保留已有
指标。任一正式判据失败，本轮结论只能是「当前修改未通过闭环」，不得用 truth 调参后重跑。

### 5. 正式结果（只回填，不据 truth 调参）

正式主运行从 `2026-09-10T10:02:16Z` 至 `11:59:39Z`，run tag 为
`closure-current-100k-20260910`。主 manifest 为 schema 2、`completed`；输入哈希为：

- misaligned geometry：`b4b113222dd714c42d149508e8497542b29069a2a84a1b80e00b51307c82a63d`
- misaligned trackstates：`c7b377187c1eb10c2c258aaa1b6fb545940594c3d4bec027009dd5c8029edc2b`
- misaligned tracksummary：`c0b2c29ecc4fcd76d98cea1a7d40251772675fd5f551309224d213b3f2b27615`

主运行与 truth-veto 的 source snapshot 均为 `written`、各含 139 个文件；两个归档都已
用 `restore-source` 恢复到独立空目录并通过整包、文件数与逐文件哈希校验。恢复时只有
Python 3.9 `tarfile` 的安全默认行为提示，没有完整性错误。

数值身份与契约通过：98,958 input tracks 中保留 **61,482**；SMT seed 和 measurement
refit 终态非驻点比例均为 0，非有限 decrement 均为 0。block-LM preregistration
version 为 2，`best_iteration=13`、`stop_reason=all_blocks_terminal`；2,268 blocks
全部明确记为 interior `converged`，budget-exhausted、rank-deficient、boundary 和
intersection-failed 均为 0。`min_components_delta_accepted=0`、
`min_components_end_minus_start=0`，满足固定有效集判据。

生产 acceptance 将 score 从原始 misaligned 的
`[4, 1002.2362008145352, 1543.7135270312644]` 改善为
`[0, 0.9694442432274074, 0]`；四个 layer/component 均 `passed=true`。候选因此冻结为
summary 所列的 `aligned_transforms_closure-current-100k-20260910_joint_free.json`，
没有运行 polish。**但预注册的元数据判据未完全满足**：summary 明确记录
`acceptance_scope=diagnostic_only`，而 acceptance JSON 本身没有
`profile_type=diagnostic_only` 和 `verdict_eligible=false` 两个字段。既有单测还显式断言
无 event split 时省略这两个字段。因此本轮不能把语义近似当成字段契约通过；该工程缺口
不在 truth 后修补，也不重跑本轮。

冻结后仅对该候选执行了一次几何 truth-veto（seed 6006、2,000 paired sensor
bootstrap），truth manifest 为 schema 2、`completed`。六项 nominal ratio / 95% CI
上界如下，全部严格 `<1.0`：

| layer/DOF | ratio | CI upper |
|---|---:|---:|
| 17:8 tx | 0.00616720 | 0.00651634 |
| 17:8 ty | 0.00951357 | 0.01017928 |
| 17:8 rz | 0.00746449 | 0.00796759 |
| 24:4 tx | 0.01498277 | 0.01565493 |
| 24:4 ty | 0.31917439 | 0.33565704 |
| 24:4 rz | 0.01104834 | 0.01157299 |

`trial-010-fullevents` 旁路对照（未进入选择）：两次均为 61,482 tracks、best iteration
13、最终候选为 joint-free。trial-010 的 acceptance score 为
`[0, 0.940964439649173, 0]`，本轮为 `[0, 0.9694442432274074, 0]`；两次 profile schema
与阈值不同，不能把第二分量差值解释为物理退化。按 `compare_geometries.py` 的
misaligned/aligned RMS（tx、ty 单位 mm，rz 单位 rad）：

| layer/DOF | trial-010 | 本轮 |
|---|---:|---:|
| 17:8 tx | 0.51764482 / 0.00319084 | 0.51764482 / 0.00319073 |
| 17:8 ty | 0.52181491 / 0.00496369 | 0.52181491 / 0.00496371 |
| 17:8 rz | 0.02107606 / 0.000157321 | 0.02107606 / 0.000157322 |
| 24:4 tx | 0.51671589 / 0.00697580 | 0.51671589 / 0.00697582 |
| 24:4 ty | 0.52000645 / 0.16597611 | 0.52000645 / 0.16591260 |
| 24:4 rz | 0.02055998 / 0.000227157 | 0.02055998 / 0.000227154 |

**结论**：数值、重建 acceptance 和一次性几何 truth 判据均通过，未观察到“重建改善但
几何更差”；但 acceptance JSON 的两个预注册诊断元数据字段缺失，因此按 fail-closed
规则，**当前修改整体未完全通过本次工程闭环**。这是字段契约失败，不改变几何 truth
通过的事实，也不授权使用 truth 调参或改选候选。

---

## Iteration 30 — acceptance schema-3.1 修复后的 100k 全事件闭环

**日期**：2026-09-10
**性质**：Iteration 29 工程字段契约的修复验证；输入、算法参数、候选选择规则和一次性
truth-veto 阈值均保持不变。run tag 为 `closure-schema31-100k-20260910`。

### 修复与验证

根因是全事件流程移除 event split 时同时删掉了 acceptance JSON 的
`profile_type`／`verdict_eligible` 字段，只在顶层 summary 保留
`acceptance_scope=diagnostic_only`。修复将 acceptance schema 从 3.0 升为 3.1，成功和
fail-closed 产物均显式写入 `profile_type=diagnostic_only`、
`verdict_eligible=false`，且 `write_experiment_summary.py` 对两项执行 fail-closed 复查。
定向测试先观察到缺字段失败，修复后为 `38 passed, 1 deselected`；完整
`make test` 为 `570 passed, 3 skipped, 11 deselected`，`make test-shell` 通过。

### 闭环结果

主 manifest（schema 2）与 truth-veto manifest（schema 2）均为 `completed`，两者
source snapshot 均为 `written`、各含 139 个文件。主运行复用 Iteration 29 的冻结输入
哈希；98,958 input tracks 中保留 61,482。SMT seed 与 measurement refit 的终态
non-stationary fraction 均为 0，非有限 decrement 均为 0。

block-LM preregistration version 为 2，`best_iteration=13`、
`stop_reason=all_blocks_terminal`；2,268 blocks 全部为 interior `converged`，
budget-exhausted、rank-deficient、intersection-failed、boundary 均为 0；
`min_components_delta_accepted=0`、`min_components_end_minus_start=0`。

production acceptance 从 misaligned baseline
`[4, 1002.2362008145352, 1543.7135270312644]` 改善至
`[0, 0.9694442432274074, 0]`，四个 layer/component 均通过。最终候选冻结为
`aligned_transforms_closure-schema31-100k-20260910_joint_free.json`，没有运行 polish。
最终 acceptance JSON 为 schema 3.1，且明确记录 `profile_type=diagnostic_only`、
`verdict_eligible=false`、`profile_identity_verified=true`。

候选冻结后只执行一次 geometry truth-veto（seed 6006、2,000 paired sensor
bootstrap）。六项 nominal ratio / 95% CI 上界为：

| layer/DOF | ratio | CI upper |
|---|---:|---:|
| 17:8 tx | 0.00616720 | 0.00651634 |
| 17:8 ty | 0.00951357 | 0.01017928 |
| 17:8 rz | 0.00746449 | 0.00796759 |
| 24:4 tx | 0.01498277 | 0.01565493 |
| 24:4 ty | 0.31917439 | 0.33565704 |
| 24:4 rz | 0.01104834 | 0.01157299 |

六项 CI 上界均严格 `<1.0`，truth verdict 为 `passed=true`。因此本轮数值收敛、ACTS
回灌、production acceptance、诊断元数据契约及一次性几何 truth 判据全部通过；当前
多处修改通过本次工程闭环。acceptance 与 truth 仍分别属于 `diagnostic_only` 和
`post_hoc_truth_veto_only`，不构成 blind validation 或独立泛化声明。

---

## Iteration 33 — Schur P1 统计误差预算诊断与 L1 处置（预注册）

**日期**：2026-10-10。**状态**：预注册，写于任何 P1 数据运行之前。
**上游**：`plan.md` §4 Q1 / Phase P1；`docs/handoff/2026-10-08-schur-phase0-baseline-and-implementation-map.md`
§5.1、§6.2（F1、F4）。**性质**：`diagnostic_only`，不得回灌求解器调参；truth 不参与任何计算。

### A. 规格修订：P1 行集（关闭 F1）

plan.md §4 Q1 要求在 `--no-event-split` 行集上运行；该开关与整个 event split 机制已于
2026-09-10 移除，不再引用。P1 行集改为**当前全量事件口径 + 生产筛选**，与 P0.1 基线
`schur-p0-fixedtrack-20260922` 完全相同：

- 输入：P0.1 manifest 中的 `geometry` / `states` / `nominal_geometry`，SHA256 必须与之逐字节一致；
- 筛选：`build_track_batches`，target `17:8,24:4`，≥8 非 target hit，非 target 折线跨度 ≥300 mm，
  `max_tracks=0`；
- 初始化：`seed_helices`（**2000 次**，照抄 `run_surface_helix_experiment.sh` joint_free 阶段）
  → `require_stationary_helix_point_fit` → `fit_helices_to_measurements`（生产 GN，含非驻点闸门）；
  CPU float64；
- 身份闸门：retained tracks 必须为 61,482，refit `n_components` 必须为 1,154,761；
  任一不符即停止，不读任何比值。

### B. 计算定义

- 径迹为 ref-only refit 结果；在 `reference_gauge_frame(refit)` 的 5 参 gauge 帧中，用
  `helix_model.measurement_residual_jacobian` 求白化 Jacobian：ref 行（非 target、有效求交）给出
  `Γ_ref = BᵀWB`；target 行给出白化 `B_tgt`。target 行通过直接构造 `MeasurementFitInputs`
  得到，**不**放宽 `_non_target_fit_mask` 的断言。
- `P_k = (B Γ_ref⁻¹ Bᵀ)_kk`（mm²），经 `validate_helix_model.predictive_covariance`
  （`damping=PRODUCTION_DAMPING=0.0`）计算；rank-deficient 行比例上限沿用
  `MAX_RANK_DEFICIENT_ROW_FRACTION=1e-3`，超过即 `inconclusive`。
- target 分量集合：`hit_mask ∧ is_target ∧ 求交有效`；LOC1 另需 `loc1_mask`（1D strip LOC1 排除）。
- `V_k` 为生产所用方差（含 `STRIP_Y_FLOOR`）；`V_raw,k` 为未加 floor 的方差（F4 透传）。
  floor active ⇔ `V_raw,k < V_k`。
- 残差 `r_k = m_k − ŷ_k`（未白化，mm）在两套 target 几何上计算，非 target 几何均为 nominal：
  `input`（P0.1 输入错位几何）与 `aligned`（P0.1 fixed-track 输出几何
  `aligned_transforms_schur-p0-fixedtrack-20260922_joint_free.json`，SHA256
  `2b445fed…ef2eabb`）。ref-only refit 不读 target 几何，因此两套几何共用同一组径迹。
- 内插 / 外推：按 `build_track_batches` 的 hit 顺序（径向半径、|z|），target hit 前后都有非 target
  fit hit 为 `interpolation`，仅前有为 `forward_extrapolation`，仅后有为 `backward_extrapolation`。

### C. 输出（每个分层）

主单元为 {17:8, 24:4} × {LOC0, LOC1} 共 4 格（汇总所有 regime 与 floor 状态）；
子分层为 主单元 × regime × floor 状态。每格报告：分量数、径迹数；`V̄`、`V̄_raw`、`P̄`；
`R_stat = ΣP / ΣV`（即 `tr(BΓ_ref⁻¹Bᵀ)/tr(V)`）与 `ΣP / ΣV_raw`；per-component `P/V` 的中位数；
`M̄_aligned`、`M̄_input`（残差平方均值）；`U = M̄_aligned − V̄ − P̄`（未解释项）；
`mean r²/(V+P)` 与 `mean r²/V`（aligned）。另报每 sensor target 分量数的中位数。

### D. 判读规则（按主单元，固定于运行前）

- `statistical_important` ⇔ `R_stat ≥ 0.10`；
- `model_dominated` ⇔ `U > P̄` 且 `U > 0.10·V̄`；
- 总体结论：
  1. 任一主单元 `statistical_important ∧ ¬model_dominated` → **outcome 1**（径迹统计不确定度重要，
     继续 Schur，预期相对 fixed-track 信息量下降、不确定度更可信）；
  2. 否则若任一主单元 `model_dominated` → **outcome 2**（模型误差主导，Schur 仍可做但须下调
     truth 改善预期，且声明 `S⁻¹` 不是完整物理协方差）；
  3. 否则 → **outcome 3**（两项都小：Schur 的统计收益预期很小）。
- 已知偏差，不做修正、只报告：`aligned` 残差来自 fixed-track 对这些 target hit 的 3-DOF/sensor 拟合，
  `M̄_aligned` 约偏低 `3 / n_sensor` 量级，用每 sensor 分量数中位数界定。`M̄_input` 含注入错位，
  只作对照，不进入判读。
- P1 结果可以支撑「是否继续 Schur」以及**运行前写入**的 H1 判据修订，不得用于选择 damping、
  floor、收敛阈值、筛选或模型变体。H1「truth RMS ≥10%」判据本轮不修订。

### E. L1 真实两臂的处置

`2026-09-08-block-lm-convergence-contract.md` §11 要求 A/B 两臂共用「同一 train event-key hash」。
event split 移除后此条件无法满足。v2 收敛契约的代码已在 Iteration 29/30 的 100k 全事件闭环中运行
（2,268/2,268 interior `converged`，budget-exhausted / rank-deficient / intersection-failed 均为 0，
`min_components_delta_accepted=0`），P0.1 也证明 HEAD `2b8cd3e` 与之数值等价
（decrement/component p99 7.77e-7 < 1e-6）。**因此宣告 L1 真实两臂被 Iteration 29/30 替代，
不再单独执行**；roadmap M1 的「L1 完成」前置视为满足。

替代**不覆盖**的部分（不得引申）：

- 未运行 A 臂（`e9c63b2`），不能宣称 v1→v2 在真实数据上的 truth 差异或几何等价；
- `clipped_step_fraction=0.825`：`lm_max_step_t/r` 解后裁剪承担了大部分步长控制，这一点未经审计，
  留给 Schur P6（全局收敛契约）以先红后绿的回归处理。

### F. 结果（2026-10-10，本段只回填，不改上文任何定义或阈值）

运行 `schur-p1-error-budget-20261010`（manifest schema 2，`completed`，源码快照 `written`，
耗时 4,763 s）。产物：`data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/schur-p1-error-budget-20261010/`
（`error_budget.json`、`strata.csv`、`components.csv.gz`）。

**身份闸门 `passed`**：四项输入 SHA256 与 P0.1 一致；retained 61,482 / excluded 37,476；
refit `n_components=1,154,761`，`chi2_per_component=0.7635026065294721`（与 P0.1 逐位相同）；
seed 与 refit 非驻点行比例均为 0；rank-deficient 行 0，target 分量因求交失败或 rank 丢弃均为 0。
`Γ_ref` 条件数 p50 1.19e5、p99 2.47e5、max 6.4e8。

| 主单元 | n | V̄ (mm²) | P̄ (mm²) | R_stat | M̄_aligned | U | r²/(V+P) | r²/V | 判读 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17:8 LOC0 | 71,345 | 2.25e-4 | 1.30e-4 | 0.577 | 4.57e-4 | 1.02e-4 | 1.29 | 2.03 | stat |
| 17:8 LOC1 | 71,345 | 2.25e-4 | 7.42e-4 | 3.30 | 1.04e-3 | 7.47e-5 | 1.20 | 4.63 | stat |
| 24:4 LOC0 | 71,842 | 1.85e-3 | 4.39e-4 | 0.238 | 2.30e-3 | 1.49e-5 | 1.01 | 1.25 | stat |
| 24:4 LOC1 | 71,842 | 1.44 | 5.12e-3 | 0.0036 | 1.38 | −0.066 | 0.95 | 0.96 | — |

**总体结论：outcome 1（径迹统计不确定度重要）**。三个主单元 `statistical_important`，
无一 `model_dominated`。

必须同时记录的事实（不改变判读，但限制其解读）：

- **regime 分层退化**：两层 target 全部为内插，仅 17:8 有 5 个 backward extrapolation 分量，
  无 forward extrapolation。plan 要求的内插 / 外推对比在本行集上不可得。
- **floor 分层为空**：target 方差按层、按分量都是常数；24:4 LOC1 原始方差本身就是 1.44 mm²，
  因此 `STRIP_Y_FLOOR` 在 target 分量上从未生效。plan §3.2 关于「floor-active LOC1 pull」的保留意见
  对这两层 target 不适用。
- **17:8 LOC0 靠近 model 边界**：U/P̄ = 0.79。若按 `3/n_sensor·M̄` 粗略补回 fixed-track 自由度偏差
  （+2.3e-5），U ≈ 1.25e-4，仍 < P̄（1.30e-4），但余量只有约 4%。outcome 1 不依赖该单元：
  17:8 LOC1（U/P̄=0.10）与 24:4 LOC0（加偏差上界后 U ≲ 1.4e-4 < P̄=4.4e-4）各自独立满足。
- 24:4 LOC1（ty 方向）由测量方差主导（R_stat 0.0036），Schur 对它的统计修正预期可忽略；
  这与已知的 24:4 ty 在信息下限一致。
- 本结果是 `diagnostic_only`：不修订 H1 判据（WP3 待用户决定），不用于任何求解器参数选择。

---

## Iteration 34 — Schur P2 / P3 / P4：径迹块前置、Jacobian 闭合与三 oracle（预注册）

**日期**：2026-10-11。**状态**：预注册，写于 `scripts/joint_alignment_schur.py` 及任何相关测试、
任何数据运行之前。**上游**：`plan.md` Phase P2、P3、P4，§3.3–§3.7、§3.10、§8.1–§8.2、§8.4；
P1 决策笔记 `docs/handoff/2026-10-10-schur-p1-error-budget-decision.md`。
**性质**：合成验证 + 一次 `diagnostic_only` 小样本诊断。不实现 ρ、λ 自适应、`schur-lm` CLI、
稀疏累加、diagonal-only 消融；不修改 `run_block_lm_alignment`、`lm_max_step_t/r`、收敛常量、
`STRIP_Y_FLOOR`。H1 判据（WP3）本轮不修订，最迟在 Level D 全量运行前定下。truth 不参与任何计算；
合成测试里的「真值」只是生成数据用的已知参数。

### A. 共同约定

- 残差：`r = m − ŷ`，按分量白化 `r / sqrt(max(V, 1e-8))`，`V` 为生产方差（含 `STRIP_Y_FLOOR`）。
  `F = ½·Σ r_white²`。分量集合与生产一致：`hit_mask ∧ 求交有效`，LOC1 另需 `loc1_mask`；
  1D strip LOC1、padding、求交失败的位置在残差和全部 Jacobian 中精确为 0。
- 几何参数：仅 target sensor 的 active DOF `(tx, ty, rz)`，局部组合沿用 `apply_alignment_torch`
  （`c = c0 + R0·δt`，`R = R0·R(δr)`）。几何 Jacobian 以无量纲变量 `u` 表示：`a = s ⊙ u`，
  `s = (1 mm, 1 mm, 0.01 rad)`，与 block-LM 的 `physical_scales` 相同。非 target 几何固定，ref 行的 `A ≡ 0`。
- 径迹参数：生产 5 参 gauge 帧 `θ = (o0, o1, ψ0, tanλ, ρ/1e-4)`，帧由 `reference_gauge_frame(当前物理 helix)`
  构造；θ 不再额外缩放。
- `Γ = Γ_ref + Γ_tgt`，组成写死，不提供开关。
- 全部线性代数用 float64。局部 5×5 块只做 Cholesky 解（`cholesky_ex`），失败显式报告，不求显式逆。
- 数值秩：`torch.linalg.matrix_rank` 的默认容差（`max(m, n)·eps·σ_max`）。
- 本轮不启用 damping floor（ε = 0，plan §3.7 的 B 阶段）：`D_a = diag(C)`，`D_θ = diag(Γ)`，
  只取 active 方向。只有出现可复现的数值失败时，才另写预注册引入分组 floor。

### B. P3：Jacobian 闭合判据

- 有限差分：中心差分，float64，步长 `h = 1e-6`（`u` 与 `θ` 各自的单位下）。
- 逐元素通过条件：`|J_ad − J_fd| ≤ 1e-6·|J_ad| + 1e-8·max_col|J_ad|`，
  其中 `max_col` 是同一列（同一参数）全部行的最大绝对值。第二项防止结构上接近 0 的元素
  因舍入噪声被误判，同时远小于任何物理上有意义的导数。
- 结构零必须精确为 0（不走容差）：ref 行的 `A`、1D strip LOC1 行、padding、求交失败行。
- 分层统计（报告 max 与 p99 的相对误差、通过率）：sensor、DOF、LOC0/LOC1、strip 类型（1D/2D）、
  倾角（`|cos α|`，α 为径迹方向与面法向的夹角）；`|cos α| < 0.3` 记为近平行入射。
  **通过 ⇔ 所有分层的通过率均为 100%。**
- 与生产路径一致（相对 Frobenius 差 ≤ `1e-10`，只差 BLAS 归约顺序）：
  (1) 单 sensor 的 `C_s`、`c_s`（由 `A` 组装，已乘 `s`）与 `_block_lm_linearized_block` 在同一
  `fixed_mask` 上返回的 `fisher`、`gradient` 一致；
  (2) `B` 的 ref 行与 `helix_model.measurement_residual_jacobian` 一致；
  (3) `EPS_COVARIANCE == MEASUREMENT_VARIANCE_FLOOR == 1e-8`（两条白化路径的下限相同）。

### C. P2：径迹块与协方差前置

- **对称与正定**（合成）：`Γ_ref`、`Γ` 对称（相对 Frobenius 非对称度 ≤ `1e-12`），Cholesky 成功；
  `λ_min(Γ − Γ_ref) ≥ −1e-12·λ_max(Γ)`。
- **ref-only 驻点**：复用 `helix_model.measurement_newton_decrement`（`βᵀΓ_ref⁻¹β`），
  阈值沿用 `MEASUREMENT_NEWTON_DECREMENT_STATIONARY = 1e-6`。只作初始化质量诊断，
  不是联合求解开始后的逐步收敛要求。
- **gauge 生命周期**（合成）：
  (1) 在帧 F0 中接受 `δθ` 后 unpack 得物理 helix `p1`；用 `p1` 重建 F1 后 pack，再 unpack，与 `p1` 的差
  ≤ `1e-12`（相对）；
  (2) 负对照：把 F1 中算出的第二步增量错用到 F0 上，得到的物理 helix 与正确结果的差必须
  ≥ `1e-6`（相对），证明测试能抓出「跨帧复用增量」；
  (3) 模块 API 只接受物理 helix 作为线性化输入，不接受上一轮的 5 维增量。
- **合成 MC pull 标定**：真值 helix 穿过一组合成平面，按已知 `V` 加高斯噪声，
  `N_rep = 4000`，种子 `20261011`。帧固定为真值 helix 自身的 gauge 帧（真值 θ 的 offset 为 0），
  在该帧中做 ref-only GN 至 `βᵀΓ⁻¹β ≤ 1e-12`。
  pull 有两种：`(θ̂ − θ_true)_i / sqrt((Γ_ref(θ̂)⁻¹)_ii)`，以及 target 预测残差 `r_k / sqrt(V_k + P_k)`。
  分层：ref hit 数 {8, 16}、tanλ {0.2, 1.5}、target 位置 {内插, 外推}。
  **通过 ⇔ 每格每个量的 `|mean| ≤ 0.05` 且 `|std − 1| ≤ 0.05`**
  （N = 4000 时 std 的标准误约 0.011，0.05 约为 4.5σ，所以 30+ 个格子不会因偶然性误判）。
  另设一格 floor-active 的 strip LOC1 target 分量：只报告，不判定——floor 是有意的方差放大，pull 宽度预期 < 1。
- **小样本真实数据诊断**（`scripts/analysis/measure_track_block_conditioning.py`）：
  - 输入与 P0.1 / P1 相同（`geometry` / `states` / `nominal_geometry` 三项 SHA256 与 P0.1 manifest 逐字节一致，
    否则停止），生产筛选，`max_tracks=2000`（即 `build_track_batches` 的前 2000 条保留径迹，
    加载上限 `4×max_tracks`，与生产 CLI 相同）；seed 迭代 2000 次 → `require_stationary_helix_point_fit` →
    生产 GN refit（含非驻点闸门）；target 几何取输入错位几何、alignment = 0
    （即 Schur 求解器将来的起点）。
  - 报告：retained 径迹数、ref / target 分量数；`Γ_ref` 与 `Γ` 的条件数（p50/p90/p99/max）、
    数值秩、Cholesky 失败数、非有限数；ref-only Newton decrement 分布及超过 1e-6 的行数；
    `tr(Γ_tgt)/tr(Γ_ref)` 的分布。
  - P3 闭合在真实径迹上的复核：从保留径迹中按种子 `20261011` 无放回抽 200 条，判据同 §B。
  - 判定：非有限数 = 0、Cholesky 失败 = 0、真实径迹 P3 闭合 100% 通过。条件数与秩只报告，
    不设阈值（本轮不据此调任何参数）。manifest 的 `command` 必须是可原样执行的完整命令。

### D. P4：三 oracle、阻尼安全与归因

- 数据层级：Level A 为纯代数随机问题（控制条件数 ≤ `1e6`，并记录实际值）；
  Level B 为真实 helix 与求交代码在合成平面上生成的线性化（≥ 3 条径迹、≥ 2 个 target sensor，
  至少一对 sensor 被同一条径迹共同穿过，至少一个 sensor 不与其他 sensor 共享径迹）。
- **稠密 vs Schur**（λ = 0 与 λ = 1，λ = 1 时用 §A 的 `D_a`、`D_θ`）：
  `‖δa_S − δa_D‖/‖δa_D‖ < 1e-8`，`‖δθ_S − δθ_D‖/‖δθ_D‖ < 1e-8`（δθ 为所有径迹拼接；δa 用 `u` 单位）。
- **代数性质**（相对容差 `1e-10`）：`S_t` 对称；`G = 0 ⇒ S = C, b = c`；
  `Γ_ref ← κ·Γ_ref`，κ ∈ {1e4, 1e8} 时 `‖S − C‖/‖C‖` 随 κ 单调减小，κ = 1e8 时 ≤ `1e-6`（趋于 fixed-track）；
  多径迹累加等于逐径迹和；共享径迹的 sensor 对产生非零非对角块；不共享的 sensor 对的非对角块**精确为 0**
  （不构造该块）。
- **Woodbury**（λ = 0，通过公开调用 `schur_reduce(..., lam=0.0)`）：测试内独立组装稠密
  `R = V + B_tgt Γ_ref⁻¹ B_tgtᵀ`（不调用模块的组装函数），要求 `‖S − A_tgtᵀR⁻¹A_tgt‖_F/‖S‖_F < 1e-8`；
  在 `β_ref = 0` 的 ref-only 驻点上，要求 `‖b − A_tgtᵀR⁻¹r_tgt‖/‖b‖ < 1e-8`。
  （白化坐标下 `V = I`；这里的 `R⁻¹` 用稠密 `torch.linalg.solve` 求，与模块里逐径迹的 Cholesky 路径相互独立。）
- **大阻尼极限**：λ ∈ {1e2, 1e4, 1e6, 1e8}，`D_a = diag(C)`，`D_θ = diag(Γ)`。
  - 联合阻尼（生产形式）：`‖δ(1e6)‖/‖δ(1e4)‖ ≤ 0.02`，`‖δ(1e8)‖/‖δ(1e6)‖ ≤ 0.02`，对 δa、δθ 分别成立；
    且 `‖δ(1e8)‖ ≤ 1e-6·‖δ(0)‖`。λ = 1e2 只报告（问题维度较大时 1e2 可能还未进入渐近区）。
  - 负对照 `_geometry_only_damped_step`（只阻尼 C，**只在测试文件里定义**）：λ = 1e8 时
    `‖δa‖ ≤ 1e-6·‖δa(0)‖`，且 `‖δθ + Γ⁻¹β‖/‖Γ⁻¹β‖ ≤ 1e-4`（target-inclusive 的纯径迹吸收极限）。
  - 用源码 grep 断言 `scripts/joint_alignment_schur.py` 中不存在 `geometry_only` 字样。
- **归因**（Level B，纯几何注入，无噪声）：在 ref-only refit 后的径迹上，对两个 target sensor 注入
  `±(10 µm, 10 µm, 0.1 mrad)`。λ = 0 的联合 GN 步定义
  `f_geo = −⟨r_tgt, A_tgt δa⟩ / ‖r_tgt‖²`，`f_trk = −⟨r_tgt, B_tgt δθ⟩ / ‖r_tgt‖²`（白化内积）。
  **通过 ⇔** `f_geo ≥ 0.99`、`|f_trk| ≤ 0.01`、`‖s⊙δu − a_inj‖/‖a_inj‖ ≤ 1e-3`，
  且 `+` 与 `−` 两次注入恢复出的 δa 满足 `‖δa₊ + δa₋‖/‖δa₊‖ ≤ 1e-3`（符号反演）。
  负对照（几何-only 阻尼，λ = 1e8）的 `f_trk` 只报告；唯一断言是它严格大于联合步的 `f_trk`。

### E. 门出口与停止条件

P3 全部通过 + P4 全部通过 + P2 合成项全部通过 + 小样本诊断的 §C 判定通过，才记为
「P2–P4 通过」。任一失败即按 plan §18 停下来排查；**不放宽本条目中的任何容差**。
如确需修改容差，必须先在新的 Iteration 写明数值理由，再重跑。

### F. 结果（2026-10-11，本段只回填，不改上文任何定义或阈值）

**总体：P2–P4 通过**（§E 的四项全部满足）。有限差分方法按 Iteration 35 修订；原 `h = 1e-6` 的失败记录在那里。
`make test`：683 passed / 3 skipped / 14 deselected / 0 failed（基线 639，新增 44）；`make test-shell` exit 0。

**P3（`tests/test_joint_schur_jacobians.py`）**：Level B 合成问题共 3009 个元素，Richardson 有限差分下 0 失败，
近平行、1D strip、LOC0/LOC1、ref/target 各分层都不为空。与生产路径一致：
`C_s`/`c_s` 对 `_block_lm_linearized_block`、ref 行 `B` 对 `measurement_residual_jacobian`，相对差都 ≤ 1e-10；
`EPS_COVARIANCE == MEASUREMENT_VARIANCE_FLOOR`。结构零精确为 0。

**P2 合成**：`Γ_ref`、`Γ` 对称、Cholesky 成功、秩为 5，`Γ − Γ_ref` 半正定；gauge 生命周期三项通过
（跨帧复用增量的负对照能被抓出来）。MC pull 共 8 格 × 7 个量（5 个 θ 加 target LOC0/LOC1，每格 4000 次重复）：
`|mean|` 最大 0.042（16 个 ref hit、tanλ = 0.2、外推、tanλ 分量），std 落在 0.968–1.026，全部在 0.05 以内。
floor-active 那一格（建模方差是真实方差的 4 倍）的 target LOC1 pull std 为 0.590，只报告。

**P2/P3 真实小样本**（`schur-p2-track-blocks-20261011`，manifest `completed`，源码快照 `written`；
`-r2` 为同一输入重跑，只把报告量换成容差利用率，径迹块结果与第一次逐字段相同）：
加载 8000 条径迹条目，保留 2000 条；refit χ²/component 0.7494，seed 与 refit 非驻点行均为 0；
ref 分量 37,467，target 分量 9,230。

| 量 | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|
| `cond(Γ_ref)` | 1.19e5 | 1.78e5 | 2.38e5 | 2.80e5 |
| `cond(Γ)` | 1.28e5 | 1.71e5 | 2.18e5 | 2.85e5 |
| ref-only Newton decrement | 4.6e-19 | 1.5e-13 | 4.0e-12 | 2.7e-11 |
| `tr(Γ_tgt)/tr(Γ_ref)` | 0.216 | 0.355 | 0.497 | 0.898 |

所有径迹的两个块秩都是 5，非有限数 0，Cholesky 失败 0。`cond(Γ_ref)` 的 p50 与 P1 全量结果（1.19e5）一致。
P3 在 200 条真实径迹（421 个 target block）上复核：0 失败，容差利用率最大 0.28（target LOC0），
`|J| ≥ 1e-3·列最大值` 的元素最大相对差 2.9e-8。真实数据上近平行 target 分量有 1268 个元素检查；
target 层上没有 1D strip，所以 1D strip 分层只能在合成数据上覆盖。

**P4**：

| 检查 | 结果 | 判据 |
|---|---|---|
| 稠密 vs Schur，Level A（联合 Hessian 条件数 176） | δa ≤ 1.4e-15，δθ ≤ 4.5e-16 | < 1e-8 |
| 稠密 vs Schur，Level B（`cond(S)` = 64.5） | δa ≤ 1.8e-13，δθ ≤ 1.0e-13 | < 1e-8 |
| Woodbury `S` / `b`，Level A | 2.4e-16 / 4.4e-16 | < 1e-8 |
| Woodbury `S` / `b`，Level B | 2.6e-15 / 2.4e-15 | < 1e-8 |
| 联合阻尼 ‖δa‖，λ = 1e2…1e8 | 2.18e-3, 2.21e-5, 2.21e-7, 2.21e-9 | 相邻比 ≤ 0.02 |
| 联合阻尼 ‖δθ‖，λ = 1e2…1e8 | 1.00e-3, 1.03e-5, 1.03e-7, 1.03e-9 | 相邻比 ≤ 0.02 |
| 只阻尼几何的负对照，λ = 1e8 | δa → 0；δθ 收敛到 `−Γ⁻¹β` | ≤ 1e-6 / ≤ 1e-4 |
| 归因（±注入，无噪声） | `f_geo` = 1.000000，`|f_trk|` = 3.5e-8 | ≥ 0.99 / ≤ 0.01 |
| 恢复误差（物理单位 / `u` 单位） | 9.9e-5 / 8.1e-5 | ≤ 1e-3 |
| 正负注入的符号反演 | 1.6e-4 | ≤ 1e-3 |

负对照里，径迹吸收了 target 残差的 `f_trk` = 0.459：只阻尼几何的写法会把将近一半的错位信号转给径迹。
这是 plan §3.7 禁止这种写法的直接数值证据。

**变异检查**（不属于预注册，用来证明测试会变红）：`Γ` 漏掉 `Γ_tgt`、只阻尼几何、ref 行保留几何导数、
回代丢掉耦合项、旋转尺度写错，这 5 个人为错误分别被 Woodbury、阻尼极限、结构零、稠密 vs Schur、
block-LM 一致性测试抓出来。归因测试抓不住「漏掉 `Γ_tgt`」：无噪声注入时零残差解与 `Γ` 的组成无关，
这一项由 Woodbury 测试负责。

**不能引申的地方**：以上都是代数和线性化层面的正确性，不涉及 ρ、λ 自适应、非线性接受、稀疏累加或任何
truth 指标，不能用来宣称 Schur 会改善对齐。P5（跨 sensor 协方差 MC）、P6 收敛契约、M1 都还没做。

---

## Iteration 35 — Iteration 34 §B 有限差分方法修订（P3 首次运行后）

**日期**：2026-10-11。**状态**：修订，写于修改测试之前。**性质**：只改有限差分的**计算方法**；
通过判据（`|J_ad − J_fd| ≤ 1e-6·|J_ad| + 1e-8·max_col|J_ad|`）、结构零、生产路径一致性和分层要求一字不改。

### A. 触发事实

`tests/test_joint_schur_jacobians.py` 首次运行时，Level B 合成问题（10 条径迹、4 个 target，含 yaw 75°
的近平行面和 1D strip，在一般线性化点）共 3009 个元素，按 Iteration 34 §B 的 `h = 1e-6` 中心差分，
有 12 个失败，最大相对差 7.3e-4。失败元素分布在近平行和非近平行分层里，都是 `|J|` 远小于列最大值的元素
（例如 `|J| = 0.031`，而列最大值为 5.0）。其余三项 P3 测试（结构零、block-LM 一致、ref 行 `B` 一致）通过。

### B. 诊断：失败来自有限差分舍入，不是 autograd

对最差元素扫描步长：相对差在 h = 1e-3 时为 1.4e-8，随 h 减小单调变大（1e-5 时 6.3e-7，1e-6 时 7.4e-6，
1e-8 时 4.7e-4）。差值随 `1/h` 增长，是典型的舍入主导：白化残差约 30，求交位置约 650 mm，
双精度的舍入噪声经 `1/(2h)` 放大后，在 h = 1e-6 时约 1e-7 绝对量，超过了 `1e-8·max_col ≈ 5e-8`。
随步长变大，有限差分收敛到 autograd 的值，所以 autograd 正确。

全部元素的扫描：

| 方法 | 失败 | 最大相对差 |
|---|---:|---:|
| 中心差分 h = 1e-3 | 9 | 2.0e-5（截断主导） |
| 中心差分 h = 1e-4 | 0 | 7.0e-6 |
| 中心差分 h = 1e-5 | 0 | 1.1e-4 |
| 中心差分 h = 1e-6（原预注册） | 12 | 7.3e-4（舍入主导） |
| Richardson，h = 1e-3 与 5e-4 | 0 | 2.8e-6 |

### C. 修订

有限差分改为 **Richardson 外推**：`J_fd = (4·D(h/2) − D(h)) / 3`，其中 `D(h)` 为中心差分，
`h = 1e-3`（`u` 与 `θ` 各自的单位下）。截断误差降为 `O(h⁴)`，步长大所以舍入噪声小，
而且不依赖从这次扫描里挑出的某个单一步长。§B 的其他内容全部不变，也适用于 §C 的真实径迹复核。

原步长 `h = 1e-6` 的失败作为事实保留在本条目，不删除。
