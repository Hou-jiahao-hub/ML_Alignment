# Refit 数学推导：6 页紧凑审阅版

用途：审阅与复盘。正文与 PPT 同序；补充说明列于每页之后。配套 [seed 六页稿](seed_derivation_compact.md)。以当前主线固定几何、非目标层、单 Helix 的 GN refit 为准；实现位于 `scripts/helix_model.py`、`scripts/validate_helix_model.py` 与 `scripts/ml_alignment_autograd.py`。


## 1. 符号衔接与算法入口


### 沿用 seed 的定义与命名


| 符号 | 定义 · 维度 / 单位 |
|---|---|
| t, i；n, k | 径迹、hit；路径 Newton、外层 LM 索引 |
| Nₕ；T | 单轨 hit 槽位数；块内径迹数 |
| η；θ | 六维几何参数；五维优化变量 |
| h；t = hₛ | Helix 全局位置；横向弧长导数 |
| o；e₀, e₁ | 一次拟合内固定的参考原点与法平面基 |
| r；F；J | 残差；½‖r‖²；本阶段残差的 Jacobian |
| H；g；Δ | JᵀJ；Jᵀr；五维候选步 |
| α；M；μ | 平均对角尺度；αI₅；自适应阻尼 |
| A；ν；εμ | H + εμM；gᵀA⁻¹g；微阻尼 10⁻¹² |


### 入口：seed 输出 → 固定 gauge → refit


$$\boldsymbol\eta^{(0)}=\boldsymbol\eta_{\rm seed},\quad \rho_s^{(0)}=\rho_{\rm seed}/\kappa$$


$$\boldsymbol\theta=(a_0,a_1,\psi_0,\tau,\rho_s),\quad a_0^{(0)}=a_1^{(0)}=0$$

κ = 10⁻⁴ mm⁻¹；refit 初始曲率继承 seed，不重置为零。

参考系按 seed 输出构造，在本次拟合内保持固定。


### Refit 新增符号


| 符号 | 定义 · 维度 / 单位 |
|---|---|
| cᵢ；Rᵢ；nᵢ | 面中心 [mm]；局部→全局旋转；单位法向 |
| φᵢ(s)；ŝᵢ | 平面方程；求交路径估计（式中加 ∩） |
| mᵢ；m̂ᵢ | 局部测量与预测 [2 维，mm] |
| a；Vᵢ,ₐ；Wᵢ | LOC0/1 索引；方差；标准化矩阵 |
| r̃ᵢ,ₐ；rᵢ,ₐ | 标准化残差；应用有效 mask 后的残差 |
| P；Bᵢ | 取前两坐标的矩阵；固定路径 ∂h/∂θ |
| bᵢ；Lᵢ；wᵢ,ₐ | 非目标层 mask；LOC1 mask；有效分量 |
| ε∥；εφ；εV | 近平行、终态平面残差、方差保护阈值 |
| ζₜ；Ncomp | 逐轨相对下降量；有效测量分量数 |


### 同一数学角色，本阶段单位与维度改变


$$\mathbf r\in\mathbb R^{2N_h},\qquad J\in\mathbb R^{2N_h\times5}$$

refit：测量减预测，再按方差标准化。

seed：预测减测量，使用三维全局距离。

refit 的 r、F、ν 均无量纲；H 与 g 随优化坐标定标。


**补充说明**

表内求交路径 ŝᵢ 对应展示公式的 \hat s_i^\cap；只表示求交返回值。理想精确交点分支另记 s_i^\cap。平面函数 φ 不复用 seed 的单点距离目标 f，粗体局部测量 m 不与标量二次模型 m(Δ) 混淆。测量分量索引 a 仅取 0、1；不再使用 seed 推导 sinc 时的临时标量 a。t 为径迹索引时不加粗，粗体 t 为路径切向导数。本阶段 r、J 定义与 seed 相同角色但映射不同，不能将 seed 的冻结路径 J 直接移用。若在同一目标内把 r 与 J 同时反号，F、g、H 和 LM 步不变；两阶段还存在空间、白化、路径约束差异，不能仅以反号互相替代。当前主线通过零材料节点的 fit_broken_helix 复用求解器；零节点无散射先验项，内部自由度仍为 5。


## 2. Helix 与测量面求交


### 1  固定测量面，建立一维求根方程


$$\mathbf n_i=R_i\begin{bmatrix}0\\0\\1\end{bmatrix},\quad\phi_i(s)=\mathbf n_i^{\mathsf T}[\mathbf h(\boldsymbol\eta,s)-\mathbf c_i]$$


$$\phi_i(s)=0,\qquad\phi_i^{\prime}(s)=\mathbf n_i^{\mathsf T}\mathbf t(s)$$

这里求曲线穿过测量面的位置；不最小化点到曲线

距离。Rᵢ 为正交旋转矩阵，几何在 refit 中固定。


### 2  切线近似提供路径初值


$$\mathbf h(\boldsymbol\eta,s)\approx\mathbf r_0+s\mathbf t(0)$$


$$s_0=\frac{\mathbf n_i^{\mathsf T}(\mathbf c_i-\mathbf r_0)}{\mathbf n_i^{\mathsf T}\mathbf t(0)}$$


### 3  Newton 求根（原理同 seed 第 4 页）


$$s_{n+1}=s_n-\frac{\phi_i(s_n)}{\phi_i^{\prime}(s_n)}$$


### 4  实际数值保护与有效性

默认执行 12 次 Newton；ε∥ = 10⁻⁶。

初始化检查 t(0) 与 t(s₀) 的法向投影绝对值。


$$|\mathbf n_i^{\mathsf T}\mathbf t(0)|\ge\varepsilon_\parallel,\quad |\mathbf n_i^{\mathsf T}\mathbf t(s_0)|\ge\varepsilon_\parallel$$

每次候选须有限、|s| ≤ 1200 mm，且当前导数

绝对值 ≥ ε∥。失败即保留旧路径并永久标无效。

迭代除法使用保号分母保护；导数过小仍判无效，

不能把分母保护解释为允许近平行求交。


### 5  终态检查与适用边界


$$|\phi_i(\hat s_i^\cap)|\le\varepsilon_\phi=10^{-3}\ {\rm mm}$$

终态还要求路径、位置有限且未越界。

无效时返回路径 0、参考点 r₀ 和 valid = False。

近平行指切向与面近似平行；函数求无限平面交点，

不检查传感器有限边界，也不证明交点全局唯一。


**补充说明**

初始直线分母不满足近平行阈值时，代码用保号 ±1 避免非法除法，但 straight_valid=False，路径不会因此有效；随后还检查 s₀ 有限和 t(s₀) 的法向投影。每步更新通过 accept_newton_candidate 累积有效性，不复活失败路径。终态检查包含迭代有效性、路径与点有限、路径绝对值上界、平面残差阈值。无效位置实际调用 h(η,0)，即 r₀，不是全局坐标零点。求交不依赖 SMT 最近点路径作为 s₀；代码中的预计算 hit_s 在零节点路径不替代平面求交路径。


## 3. 局部预测、方差标准化与 half-χ²


### 1  全局交点 → 测量面的局部坐标


$$P=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}$$


$$\widehat{\mathbf m}_i=P R_i^{\mathsf T}[\mathbf h(\boldsymbol\eta,\hat s_i^\cap)-\mathbf c_i]$$

减去面中心后，用 Rᵢᵀ 转到局部坐标；P 取 LOC0、

LOC1。局部测量 mᵢ 与几何固定，预测随径迹更新。


### 2  测量减预测 → 按分量方差标准化


$$\widetilde r_{i,a}=\frac{m_{i,a}-\widehat m_{i,a}}{\sqrt{\max(V_{i,a},\varepsilon_V)}},\quad a\in\{0,1\}$$


$$W_i=\operatorname{diag}_{a=0,1}\left[\max(V_{i,a},\varepsilon_V)^{-1/2}\right]$$

εV = 10⁻⁸ mm²；按输入对角方差标准化。

没有对 LOC0–LOC1 完整协方差做逆平方根变换。

标准化后残差无量纲；平方和即本实现采用的 χ²。


### 3  分量有效性：非目标层 ∩ 求交成功


$$w_{i,0}=b_i\land\operatorname{valid}_i,\quad w_{i,1}=w_{i,0}\land L_i$$


$$r_{i,a}=\begin{cases}\widetilde r_{i,a},&w_{i,a}=1\\0,&w_{i,a}=0\end{cases}$$

Lᵢ 表示 LOC1 可用；无效项以 where 整体置零。

有效分量随当前求交结果变化。


### 4  堆叠残差 → 优化目标与报告量


$$\mathbf r=\operatorname{col}(\mathbf r_1,\ldots,\mathbf r_{N_h})\in\mathbb R^{2N_h}$$


$$F_t=\frac12\sum_{i,a}r_{ti,a}^2,\quad\chi^2=2\sum_t F_t$$


$$N_{\rm comp}=\sum_{t,i,a}w_{ti,a},\quad\overline{\chi^2}_{\rm comp}=\frac{2\sum_tF_t}{\max(N_{\rm comp},1)}$$

求解器报告逐轨平均 F；主线另报每有效分量 χ²。

后者没有扣除拟合自由度，不能称为 χ²/NDF。


**补充说明**

位置和局部测量使用 mm，方差使用 mm²，W 使用 mm⁻¹，r、F、χ² 无量纲。Fₜ 是逐径迹目标；全块统计用显式径迹索引 t。对角标准化对应代码 measured−local 除以 sqrt(clamp_min(variance,EPS_VARIANCE))；主线等价地计算残差平方除以 clamp_min(covariance,EPS_COVARIANCE)。本阶段 J 有 2Nₕ 行，其中无效分量行置零，不因单点缺少 LOC1 而改变张量宽度。零材料节点时 prior 展平长度为 0，因此没有额外材料残差。目标定义与 seed 第5页同为最小二乘，但数据空间、权重、残差方向和路径约束均已改变。


## 4. 交点路径对参数的导数与 Jacobian


### 1  理论：对精确平面约束隐式求导

以下为精确交点处、光滑分支及固定有效集的推导；

要求 nᵢᵀt ≠ 0。Bᵢ 表示固定路径的几何偏导。


$$B_i=\left.\frac{\partial\mathbf h(\boldsymbol\eta(\boldsymbol\theta),s)}{\partial\boldsymbol\theta}\right|_{s=s_i^\cap}\in\mathbb R^{3\times5}$$


$$0=\frac{d\phi_i}{d\boldsymbol\theta}=\mathbf n_i^{\mathsf T}\left(B_i+\mathbf t\frac{ds_i^\cap}{d\boldsymbol\theta}\right)$$


$$\frac{ds_i^\cap}{d\boldsymbol\theta}=-\frac{\mathbf n_i^{\mathsf T}B_i}{\mathbf n_i^{\mathsf T}\mathbf t}$$


### 2  路径变化 → 交点沿测量面移动


$$\frac{d\widehat{\mathbf m}_i}{d\boldsymbol\theta}=PR_i^{\mathsf T}\left(B_i+\mathbf t\frac{ds_i^\cap}{d\boldsymbol\theta}\right)$$


$$=PR_i^{\mathsf T}\left(I_3-\frac{\mathbf t\mathbf n_i^{\mathsf T}}{\mathbf n_i^{\mathsf T}\mathbf t}\right)B_i$$


### 3  标准化残差的理论 Jacobian


$$J_i=-\operatorname{diag}(w_{i,0},w_{i,1})W_i\frac{d\widehat{\mathbf m}_i}{d\boldsymbol\theta}$$


$$J=\operatorname{col}(J_1,\ldots,J_{N_h})\in\mathbb R^{2N_h\times5}$$

负号来自“测量减预测”；几何、测量与方差固定。

修正项使交点导数的法向分量为零。


### 4  实现：对有限次求交运算自动微分

vmap(jacrev(...)) 对每条径迹构造 J：

θ → η → 12 次 Newton 求交 → 局部坐标

→ 方差标准化 → mask → 堆叠残差。

求交路径没有像 seed 那样停止梯度；布尔有效性

不连续求导，where 对当前选中的数值分支求导。

代码没有显式代入左侧隐式公式。有限次迭代、

分支切换或近平行时，不保证与精确根公式相等。


**补充说明**

所有隐式公式在同一个精确交点处评估，省略了 t 的 η、sᵢᶜᵃᵖ 参数。由 nᵀ[B+t(ds/dθ)]=0 可见交点的全导数位于平面切空间；矩阵 I−tnᵀ/(nᵀt) 通常是斜投影，不能无条件称为正交投影。相比 seed 的 J=w∂h/∂θ|固定路径，refit 局部测量预测必须包含路径响应；不存在可直接照搬的“最近点残差与切向正交，所以路径项消失”结论。边界条件变化会改变有效分量集合，理论式仅描述固定有效分支，不对 mask 的离散变化求导。隐式函数公式是解释和核对依据，不是声称当前实现采用解析隐式求导。


## 5. LM 更新、候选重评估与接受规则


### 1  共同公式概述，不重复二次模型推导


$$\mathbf g=J^{\mathsf T}\mathbf r,\qquad H=J^{\mathsf T}J$$


$$\alpha=\max\left(\frac{\operatorname{tr}H}{5},10^{-30}\right),\quad M=\alpha I_5$$


$$(H+\mu M)\Delta=-\mathbf g,\quad\boldsymbol\theta^c=\boldsymbol\theta+\Delta$$

H 仍是 GN 近似；分母 5 来自单 Helix 五维变量。

每条径迹独立求步，各有自己的阻尼水平 μ。


### 2  候选必须重新求交与计算目标


$$F^c=\frac12\left\|\mathbf r(\boldsymbol\theta^c)\right\|^2$$

θᶜ → 新 Helix → 每个面的新交点 → 局部预测

→ 新有效 mask 与标准化残差 → Fᶜ。

不能用旧路径、旧预测或线性模型值代替 Fᶜ。


### 3  逐径迹严格下降才接受


$$\boldsymbol\theta^+=\begin{cases}\boldsymbol\theta^c,&F^c<F\\\boldsymbol\theta,&\text{otherwise}\end{cases}$$


$$\mu^{(0)}=10^{-8},\quad\mu^+=\begin{cases}\max(0.1\mu,10^{-12}),&F^c<F\\10\mu,&\text{otherwise}\end{cases}$$

采用自适应阻尼；不使用实际 / 预测下降比。

循环内不依据 ν 冻结已驻点径迹。


### 4  与 seed 接受规则不同的实现边界

内层只有 Fᶜ < F 的接受比较；没有独立的有限值

接受谓词，也没有“保留初始有效集”的条件。

有效分量集可随候选改变；目标下降可能伴随

样本变化，不能解释为固定样本目标的严格下降。

初始与终态另检查求交失效率；终态再验收驻点。


**补充说明**

这里“否则”直接对应 improved=candidate_objective<current。NaN 不会通过严格小于比较，但不能将代码描述为有完整的 isfinite 候选保护。与 seed 不同，该循环没有 candidate_ok_of 初始有效点保留约束，也没有逐行 stationary 掩码；所有行均计算候选并按 improved 更新参数和阻尼。下一轮重算 r、J、H、g。每条径迹独立求解，计算按最多 2048 行分块；默认每块最多 60 轮。H 的单位依赖五维坐标，αI₅ 是在该数值坐标系中的阻尼，不能因为残差无量纲就认为 H 所有元素单位相同。


## 6. 循环停止、独立驻点验收与完整流程


### 1  循环停止：当前块最大的相对下降量


$$\zeta_t=\frac{\max(F_t^{\rm old}-F_t^c,0)}{\max(|F_t^{\rm old}|,1)}$$


$$\operatorname{change}=\max_{t\in\mathrm{chunk}}\zeta_t$$

候选评估后，change < 10⁻¹⁰ 或预算耗尽则停止。

有限候选全被拒绝也会给出 change = 0。

因此 converged 只代表该循环停止条件成立，

不能替代独立的终态驻点检查。


### 2  驻点指标概述（原理见 seed 第 6 页）


$$\nu_t=\mathbf g_t^{\mathsf T}(H_t+\epsilon_\mu M_t)^{-1}\mathbf g_t,\quad\epsilon_\mu=10^{-12}$$


$$\operatorname{stat}_t=\operatorname{finite}(\nu_t)\land[\nu_t\le10^{-6}]$$

终态重新建 gauge、求交并计算 r、J、H、g；

不采用 seed 的相对 ν / F 判据。


### 3  主线验收：非有限、非驻点、求交失效


$$n_{\rm nonfinite}=0,\quad \frac{\#\{t:\nu_t>10^{-6}\}}{T}\le10^{-3}$$

非有限 ν 单独拒绝，不纳入允许比例。

初始及终态：非目标拟合 hit 的求交失效率 ≤ 5%。


### 4  实际预算与精度

默认 GN：每块最多 60 轮，内部 float64。

包装层 n_iter = 200 仅传给 Adam，不控制 GN。

返回参数按调用方 dtype 转换，再检查终态失效率。


### 5  从 seed 到 refit 输出

seed 参数 + 固定几何 + 非目标层测量

→ 初始求交失效率检查 → 固定 gauge、打包 θ

→ 求交 → 局部残差 → J、H、g → LM 候选

→ 重评目标 → 接受 / 拒绝、更新 μ → change

→ 未停止则循环；停止后独立验收 ν 与失效率

→ 恢复六维 η，作为后续固定径迹输入


**补充说明**

当 current 与候选目标均有限且候选未下降时，截断后的 decrease 为零；候选出现 NaN 时 change 本身可能非有限，因此不能笼统声称“任意全拒绝都使 change=0”。停止量分母 max(|F_old|,1) 是代码定义，不是 seed 的 ν/max(F,10⁻¹²)。newton_decrement 用拟合结果重建诊断参考系并重新计算 Jacobian，区别于一次拟合中固定 frame。主线 _refit_helices_gauss_newton 先检查 float64 解的非有限 ν 和非驻点比例，再由外层转换回调用方 dtype，重评目标和失效率；故不能把原 float64 驻点门限声称为对转换后参数逐项重新执行。失效率为非目标拟合 hit 中 valid=False 的比例，不是无效分量比例，默认上限 0.05，可由配置覆盖。包装层可选的 seed/refit stationarity 对比诊断不控制默认 GN 的强制驻点门限。ν 为无量纲二次型，不开平方，不等于实际目标下降，不保证全局最优。


## 核对与阅读边界

- 共同原理仅作概述：Helix / sinc 见 seed 第2页；五维 gauge 见第3页；GN / LM 二次模型见第5页；decrement 与模型下降量见第6页。
- 本稿描述主线默认 Gauss–Newton refit，通过零材料节点复用求解器；不展开 Adam、材料散射、对齐几何优化。
- 公式采用 PowerPoint 原生 OMML，并带有同式高清图片兼容回退分支；不支持原生公式的阅读器仍可正确显示。表格、正文、备注可编辑。
- 逐页补充说明同步写入 PPT 备注；关键差异、阈值和数学成立条件已在正文显示。
- 不修改算法，不运行新算法实验，不引用未经重新核验的数值精度或性能结论。

### 代码对应

| 环节 | 代码位置 |
|---|---|
| 平面求交与数值有效性 | `scripts/helix_model.py::intersect_plane / accept_newton_candidate` |
| 单 Helix 输入与固定参考系 | `scripts/validate_helix_model.py::_measurement_fit_inputs / reference_gauge_frame / _pack_parameters` |
| 局部预测、标准化残差与求导 | `scripts/validate_helix_model.py::broken_helix_local_prediction / _standardized_residual / _BATCHED_RESIDUAL_JACOBIAN` |
| LM、候选接受与循环停止 | `scripts/validate_helix_model.py::fit_broken_helix / fit_helices_gauss_newton` |
| 终态驻点指标 | `scripts/validate_helix_model.py::newton_decrement / stationarity_report` |
| 主线适配、精度、验收与报告 | `scripts/ml_alignment_autograd.py::_refit_helices_gauss_newton / fit_helices_to_measurements / _measurement_refit_objective` |
