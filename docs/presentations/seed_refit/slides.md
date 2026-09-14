# SMT seed 与 measurement refit：逐页展示文案

用途：探测器与算法组会中可独立插入的六页算法章节。正文 P1–P6；公式附录 A1–A14 见 [appendix.md](appendix.md)。

本文件的“页面文案”可直接复制到 PPT；“讲解备注”建议置于演讲者备注，不需要全部上屏。建议 16:9、白底、深灰正文，以青绿色表示 seed、靛蓝色表示 refit、赭色表示判断及验收。正文中文建议不小于 18 pt。图中文字不重复堆放整套推导，公式以可编辑公式对象插入。

核验对象为当前工作区生产 joint 路径，默认 Gauss–Newton measurement refit；不泛指仓库内所有历史 solver 或独立 holdout 实验。版本与源码定位见 [verification.md](verification.md) 和 [source_snapshot.json](source_snapshot.json)。

## P1｜径迹组批与拟合测量选择

**结论句：以非目标层约束径迹，以目标层测量约束后续几何更新。**

### 页面文案

**数据整理**

按 `(event_id, track_id)` 将逐 hit 数据组织为径迹；采用确定性排序，保留每个 hit 的 SMT 位置、方向、局部测量、测量方差及传感器标识。

**径迹筛选（默认条件）**

- 至少包含一个目标层 hit。
- 非目标层拟合 hit 数量不少于 8。
- 非目标层排序点的累计折线长度不少于 300 mm。

**批量表示**

不同长度的径迹填充至共同宽度：位置张量为 \(T\times H_{\max}\times3\)，测量张量为 \(T\times H_{\max}\times2\)。通过有效性掩码排除填充项及不可用的测量分量。

**数据使用边界**

非目标层：SMT seed 与 measurement refit；目标层：径迹固定后的几何优化。

### 流程图与布局

左侧 60%：`逐 hit 数据 → 确定性分组 → 目标/非目标层标记 → 径迹筛选 → padding + masks`。右侧 40% 放三条筛选条件及数据边界。图源：[p1_batches.mmd](diagrams/p1_batches.mmd)。

### 讲解备注

“累计折线长度”是排序后的非目标 SMT 点间欧氏距离之和，不是拟合 helix 的横向弧长，也不是首末点直线距离。实际排序优先级为 event、track、横向半径、\(|z|\)、volume、layer、module、sensor。LOC0 使用有效拟合 hit；LOC1 还要求 `loc1_mask`。目标层没有作为本阶段的独立验证集，因此这里不使用“holdout 验证”表述。排除目标层 SMT 点是拟合输入边界，不能据此声称非目标 SMT 平滑状态在重建来源上完全不受目标层测量影响。

**代码依据：** `ml_alignment_autograd.build_track_batches`、`_non_target_fit_mask`。**附录：** A1、A11、A14。

## P2｜Helix 初始化与五维规范固定

**结论句：六个物理参数包含一个原点选择冗余；规范固定保留五个局部独立的几何自由度。**

### 页面文案

**初始化：由首个非目标层拟合 hit 构造直线初值**

\[
\boldsymbol\eta=(x_0,y_0,z_0,\psi_0,\tau,\rho),\qquad \tau\equiv\tan\lambda,
\]
\[
\mathbf r_0=\mathbf p_{\mathrm{first}},\quad
\psi_0=\operatorname{atan2}(u_y,u_x),\quad
\tau=\frac{u_z}{\max(\sqrt{u_x^2+u_y^2},10^{-6})},\quad \rho=0.
\]

其中 \(\mathbf u\) 为该 hit 的 SMT 方向，\(\rho\) 为带符号曲率。

**为什么由六维转为五维？**

- 沿同一条 helix 移动参数原点，并同步调整初始位置和方位角，不改变几何轨迹。
- 对原点选择不敏感的几何目标，这一冗余可导致参数不可辨识及正规矩阵奇异或病态。
- 将原点限制在参考切向的法平面内，以两个平面内位移替代三个独立位置分量。

**优化变量与物理参数恢复**

\[
\boldsymbol\theta=(a_0,a_1,\psi_0,\tau,\rho_s),\qquad
\mathbf r_0=\mathbf r_{\mathrm{ref}}+a_0\mathbf e_0+a_1\mathbf e_1,
\qquad \rho=(10^{-4}\,\mathrm{mm}^{-1})\rho_s.
\]

规范固定移除表示冗余；曲率缩放改善数值尺度。输出仍为六个物理参数。

### 示意图与布局

上半页：初始化条件与公式。下半页：左侧放“同一曲线、两个原点”及参考平面示意；右侧放三条原因和五维映射。配图 `p2_gauge.svg/png`；参数映射图源：[p2_parameters.mmd](diagrams/p2_parameters.mmd)。

### 讲解备注

“首个”是排序后 `fit_mask` 的第一个真值，不是固定编号的探测器首层。参考法平面在每次拟合开始时建立，拟合过程中保持固定；refit 使用 seed 输出重新建立自己的参考系。规范固定是局部参数化选择，不是将 x、y 或 z 中任意一项固定为常数，也不是对几何真值施加约束。它消除原点表示冗余，但不能修复 hit 布局不足等其他不可辨识问题。seed 的 Jacobian 冻结最近点路径，因此其近似 Hessian 与完全消去路径变量后的几何 Hessian不同，不能直接将后者的精确零模结论逐字套到 seed 的实现矩阵。

**代码依据：** `seed_helices`、两模块的 `reference_gauge_frame` 与参数 pack/unpack。**附录：** A2–A5、A7。

## P3｜SMT seed：基于平滑状态的几何预拟合

**结论句：通过点到 helix 的几何预拟合，为局部测量重拟合提供初值。**

### 页面文案

**输入：**非目标层 SMT 全局点，以及由 SMT 方向构造的初始 helix。

**目标：**最小化有效 SMT 点到 helix 的无权重三维距离平方和。

**输出：**通过终态质量检查的六参数 helix，作为 measurement refit 初值。

\[
F_t^{\mathrm{seed}}=\frac12\sum_{i\in\mathcal E_t}
\|\mathbf h(\boldsymbol\eta_t,s_{ti}^{*})-\mathbf p_{ti}\|^2.
\]

**流程图节点**

1. 五维规范固定与曲率缩放。
2. 计算最近点路径、残差及 \(J,H,g\)。
3. 检查逐径迹驻点；已满足判据的径迹停止更新。
4. 对未驻点径迹求解 LM 候选步。
5. 重新计算候选最近点及目标函数。
6. 接受满足条件的候选并降低阻尼；否则保留参数并提高阻尼。
7. 全部驻点或预算耗尽后，重新计算终态诊断并执行质量检查。

**候选接受必须同时满足：**目标有限且严格下降；不得使初始有效拟合点失效。

### 流程图与布局

顶部 15%：输入／目标／输出短条。中部 70%：两列回路图；底部 15%：目标公式及候选条件。使用 [p3_seed.mmd](diagrams/p3_seed.mmd)。驻点阈值集中在 P5，避免重复长公式。

### 讲解备注

\(\mathcal E_t\) 表示当前有效的非目标 SMT 点集合。最近点通过有限次 Newton 迭代估计，图中 \(s^*\) 表示拟合所采用的最近点估计，不宣称得到全局最小距离。每次残差评估重新求路径，但本次 Jacobian 对路径停止梯度。该目标没有 measurement 方差权重，单位为 mm²，不能将其数值直接与 refit half-χ² 比较。库内“全部径迹驻点”的布尔结果与主线容许最多 0.1% 非驻点径迹的质量门限不同。

**代码依据：** `helix_model._single_point_fit_residual`、`_fit_helix_points_gauss_newton_block`、`damped_gauss_newton`、主线 `require_stationary_helix_point_fit`。**附录：** A6–A9、A13–A14。

## P4｜Measurement refit：基于局部测量的径迹重拟合

**结论句：在固定工作几何上，以非目标层局部测量重新约束径迹参数。**

### 页面文案

**输入：**seed helix、非目标层局部测量及方差、固定工作几何。

**目标：**最小化有效局部测量分量的 half-χ²。

**输出：**通过独立驻点验收的六参数 helix，用于后续固定径迹的几何优化。

\[
\widehat{\mathbf m}_{ti}=P R_{ti}^{\mathsf T}
\left[\mathbf h(\boldsymbol\eta_t,s_{ti}^{\cap})-\mathbf c_{ti}\right],
\qquad
r_{ti,d}=\frac{m_{ti,d}-\widehat m_{ti,d}}
{\sqrt{\max(V_{ti,d},10^{-8}\,\mathrm{mm}^2)}}.
\]
\[
F_t^{\mathrm{meas}}=\frac12\sum_{(i,d)\in\mathcal C_t}r_{ti,d}^{2}.
\]

**预测链：**helix–plane 求交 → 全局到局部坐标变换 → 提取 LOC0／LOC1 → 按测量方差标准化。

**优化链：**规范固定 → 构造残差与 Jacobian → LM 候选及接受判断 → 更新阻尼 → change 停止检查 → 独立驻点验收。

### 流程图与布局

顶部输入／目标／输出短条。左侧 60% 放 [p4_refit.mmd](diagrams/p4_refit.mmd)；右侧放预测链及残差定义，目标式可放图下。\(P\) 提取局部坐标的前两个分量；\(\mathcal C_t\) 包含拟合 mask、分量可用性和有效求交条件。

### 讲解备注

当前默认入口调用零材料节点的 `fit_broken_helix`，即单 helix 拟合，没有散射先验残差。几何中心与旋转在此阶段固定。无效求交分量在残差中置零；当前 refit 内层没有 seed 那样的“初始有效点不得丢失”候选检查，不能将该规则画入 refit。主线另有初始和最终求交失败率检查。SMT 在这里用于初值和预备路径信息，不作为额外的 SMT 距离惩罚项。

强制驻点验收针对内部 float64 拟合结果；随后输出转换回调用方 dtype，最终失效率检查基于转换后的参数。

**代码依据：** `fit_helices_to_measurements` → `_refit_helices_gauss_newton` → `fit_helices_gauss_newton` → `fit_broken_helix`。**附录：** A10–A14。

## P5｜LM 候选步、迭代停止与驻点验收

**结论句：目标下降用于接受候选；迭代停止与终态驻点验收承担不同职责。**

### 页面文案

**共同更新规则**

\[
H_t=J_t^{\mathsf T}J_t,\quad g_t=J_t^{\mathsf T}r_t,\quad
(H_t+\mu_t M_t)\Delta\theta_t=-g_t,\quad
M_t=\max\!\left(\tfrac15\operatorname{tr}H_t,10^{-30}\right)I_5.
\]

候选下降并满足该阶段接受条件：提交参数，\(\mu_t\leftarrow\max(0.1\mu_t,10^{-12})\)；否则保留参数，\(\mu_t\leftarrow10\mu_t\)。

| 检查层次 | SMT seed | Measurement refit |
|---|---|---|
| 循环停止 | 分块内全部径迹满足绝对或相对 decrement 判据 | 分块内最大 change 小于 \(10^{-10}\) |
| 驻点定义 | \(\nu_t\le10^{-12}\) 或 \(\nu_t/\max(F_t,10^{-12})\le10^{-6}\) | 终态 \(\nu_t\le10^{-6}\) |
| 预算耗尽 | 退出循环并重新评估终态 | 退出循环并执行独立验收 |
| 主线质量门限 | 非驻点比例不超过 0.1%；无非有限 decrement | 非驻点比例不超过 0.1%；无非有限 decrement |

\[
\nu_t=g_t^{\mathsf T}(H_t+10^{-12}M_t)^{-1}g_t.
\]

**注意：**候选全部被拒绝时，refit 的 change 也可能为零；这表示迭代停止，不表示通过驻点验收。

### 图表与布局

顶部共同公式，中央对照表，底部突出“停止 → 终态驻点检查 → 接受或失败”。使用 [p5_gate.mmd](diagrams/p5_gate.mmd) 作为小幅流程条。decrement 的单位随目标函数变化，两阶段阈值不可直接作为物理精度比较。

### 讲解备注

逆矩阵写法仅用于数学表示；代码使用线性方程求解。\(H\) 是 Gauss–Newton 近似 Hessian。\(\nu\) 按项目定义是二次型本身，没有取平方根，也不是标准二次模型下剩余下降量的完整系数。seed 的驻点径迹不再更新，refit 内层没有逐径迹驻点冻结。表中“分块”对应实际的径迹 chunk，默认最多 2048 行。初始阻尼为 \(10^{-8}\)，两阶段各有独立的逐径迹阻尼数组。另有 5% 默认失效率门限，seed 统计最近点失效，refit 检查求交失效。

5% 失效率按全部非目标层拟合 hit 汇总计算，并非对每条径迹分别应用的门限。

**代码依据：** `damped_gauss_newton`、`fit_broken_helix`、`newton_decrement`、主线质量检查。**附录：** A8–A9、A13–A14。

## P6｜两阶段拟合的差异与输出使用

**结论句：几何预拟合提供初值，测量重拟合确定固定工作几何下的径迹解。**

### 页面文案

| 比较项 | SMT seed | Measurement refit |
|---|---|---|
| 拟合数据 | 非目标层 SMT 三维点 | 非目标层 LOC0／有效 LOC1 |
| 目标与权重 | 无权重距离平方和；mm² | 方差标准化 half-χ²；无量纲 |
| 路径变量 | 点到 helix 的最近点估计 | helix 与测量面的交点 |
| Jacobian | 本次线性化冻结最近点路径 | 自动微分通过可微求交运算 |
| 候选有效性 | 保护初始有效点集合 | 无同等集合保护；另验求交失效率 |
| 终态用途 | measurement refit 初值 | 固定后用于目标层几何优化 |

**输出连接**

非目标层 measurement refit → 终态验收 → 固定径迹参数 → 目标层 measurement 几何优化。

**适用边界**

数值驻点是对当前目标及实现的检验，不等价于模型充分性或几何真值恢复的证明。

### 图表与布局

对照表占主体；底部使用 [p6_output.mmd](diagrams/p6_output.mmd) 展示数据边界。函数名和阈值索引放讲解备注，不继续增加正文页面。

### 讲解备注

当前主线的后续“joint”阶段固定径迹，联合求解目标几何；不要画成目标测量继续更新径迹的反馈回路。纯 helix 描述的适用性还取决于场与材料效应，驻点验收不能替代残差诊断和重建验证。当前 refit mask 随求交有效性改变，因此 half-χ² 下降的解释需结合有效分量检查，不能当作固定数据集上的无条件改进保证。所有数值效果需引用已有可复核实验，本章节不新增或暗示性能测量。

**代码依据：** seed/refit residual 与主线 fixed-track 调用链。**附录：** A7、A11–A14。
