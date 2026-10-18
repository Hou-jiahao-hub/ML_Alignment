# Seed 数学推导：6 页紧凑审阅版

用途：审阅与复盘。正文与 PPT 同序；补充说明列于每页之后。以当前 `scripts/helix_model.py` 和 `scripts/ml_alignment_autograd.py` 的 seed 路径为准。


## 1. 统一符号表


### 几何与参数


| 符号 | 定义 · 维度 / 单位 |
|---|---|
| i；n；k | hit；路径 Newton；外层 LM 的索引 |
| Nₕ；T | 单轨 hit 槽位数；批内径迹数 |
| pᵢ；vᵢ | SMT 全局点 [mm]；初始方向 [3 维] |
| s；ℓ | 横向有向弧长；三维弧长 [mm] |
| η；r₀ | 六维几何参数；s = 0 参考点 [mm] |
| ψ₀；τ | 初始方位角 [rad]；dz/ds = tanλ |
| ρ；ρₛ；κ | 曲率；无量纲缩放曲率；10⁻⁴ mm⁻¹ |
| θ；a₀, a₁ | 五维优化变量；法平面位移 [mm] |
| o；e₀, e₁ | 固定参考原点；两条正交单位轴 |
| h(η,s)；t(s) | 位置 [mm]；∂h/∂s [无量纲] |
| h″(s)；d(s) | 路径二阶导 [mm⁻¹]；h − p [mm] |
| f(s)；ŝᵢ | 单点距离目标 [mm²]；路径估计 [mm] |


### 残差与优化


| 符号 | 定义 · 维度 / 单位 |
|---|---|
| bᵢ；wᵢ；E | 非目标层 mask；有效 mask；有效点集 |
| rᵢ；r | 单点 3 维；堆叠 3Nₕ 维 [mm] |
| F；m(Δ) | 整轨目标；局部二次模型 [mm²] |
| J | 冻结路径 Jacobian [3Nₕ × 5] |
| H；g | JᵀJ [5 × 5]；Jᵀr [5 维] |
| Δ；θᶜ | 五维候选步；候选优化变量 |
| α；M | 平均对角尺度；M = αI₅ |
| μ；εμ | 自适应 LM 阻尼；固定微阻尼 10⁻¹² |
| A；ν | H + εμM；gᵀA⁻¹g [mm²] |
| νʳᵉˡ；stat | 相对驻点指标；逐轨驻点判定 |
| εc；Smax | 路径分母保护 10⁻⁹；路径界 1200 mm |
| Kmax | LM 预算；主线默认 200，底层 60 |


$$\boldsymbol\eta=(x_0,y_0,z_0,\psi_0,\tau,\rho)$$


$$\boldsymbol\theta=(a_0,a_1,\psi_0,\tau,\rho_s),\quad\rho=\kappa\rho_s$$

一符一义：f / F 区分单点与整轨目标；μ / M / ν 区分阻尼水平、尺度矩阵与驻点二次型。


**补充说明**

LM 迭代用上标 (k)，路径 Newton 用下标 n，正文在不混淆处省略迭代标记。ŝ 表示代码返回的有限次路径估计，s* 仅在讨论理想的内部最近点时使用。表内 w 是 0/1 有效性掩码，不是测量精度权重；seed 使用三维全局、未白化的欧氏残差。优化坐标混合了位移、角度、斜率和缩放曲率，J 各列及 H 各元素的单位不相同；αI₅ 是所选数值坐标中的阻尼，不应赋予统一物理单位。κ 赋予 mm⁻¹ 单位时，ρₛ 无量纲。vᵢ 对应 p_smt / hit_directions，只用于初始方向。


## 2. Helix 方程：积分、稳定表达与路径导数


### 1  横向弧长约定 → 坐标积分


$$\psi(s)=\psi_0+\rho s,\qquad\rho=\frac{d\psi}{ds}$$


$$\frac{dx}{ds}=\cos\psi(s),\quad\frac{dy}{ds}=\sin\psi(s),\quad\frac{dz}{ds}=\tau$$

以 r₀ 为 s = 0 的参考点。ρ ≠ 0 时积分：


$$x(s)=x_0+\frac{\sin(\psi_0+\rho s)-\sin\psi_0}{\rho}$$


$$y(s)=y_0+\frac{\cos\psi_0-\cos(\psi_0+\rho s)}{\rho}$$


$$z(s)=z_0+\tau s$$


### 2  和差化积 → 消去直接除以 ρ


$$a=\rho s/2,\qquad\operatorname{sinc}(a)=\frac{\sin a}{a},\quad\operatorname{sinc}(0)=1$$


$$\sin(\psi_0+2a)-\sin\psi_0=2\cos(\psi_0+a)\sin a$$


$$\cos\psi_0-\cos(\psi_0+2a)=2\sin(\psi_0+a)\sin a$$


### 3  实际计算采用的稳定式


$$\mathbf h(\boldsymbol\eta,s)=\begin{bmatrix}x_0+s\cos(\psi_0+a)\operatorname{sinc}(a)\\y_0+s\sin(\psi_0+a)\operatorname{sinc}(a)\\z_0+\tau s\end{bmatrix}$$


$$\rho\to0:\quad\mathbf h\to\mathbf r_0+s(\cos\psi_0,\sin\psi_0,\tau)^{\mathsf T}$$

初始化取 ρ = 0，sinc 形式直接给出直线极限。

代码 torch.sinc(a/π) 对应本文的 sin(a)/a。


### 4  路径一、二阶导数


$$\mathbf t(s)=\mathbf h_s=(\cos\psi(s),\sin\psi(s),\tau)^{\mathsf T}$$


$$\mathbf h_{ss}=(-\rho\sin\psi(s),\rho\cos\psi(s),0)^{\mathsf T}$$


$$\|\mathbf t\|^2=1+\tau^2,\qquad\frac{d\ell}{ds}=\sqrt{1+\tau^2}$$

t 是对横向弧长的导数，通常不是三维单位切向量。

ρ 为带符号几何拟合量，不由磁场公式赋初值。


**补充说明**

这里 dx/ds、dy/ds 是横向切向分量；三者不宜统称为三维方向余弦。积分式在 ρ = 0 处有可去奇点，并非曲线本身发散。非归一化 sinc 的展开是 1 − a²/6 + O(a⁴)。若仅讨论物理背景，SI 制中 |ρ| = |qB_z|/p_T，需统一动量、长度单位；本 seed 算法不使用该关系，故正文不引入磁场、电荷和动量符号。


## 3. 五维参数化与初始化


### 1  同一曲线的参考点冗余


$$\mathbf r_0^{\prime}=\mathbf h(\boldsymbol\eta,\delta),\quad\psi_0^{\prime}=\psi_0+\rho\delta$$


$$\mathbf h(\boldsymbol\eta^{\prime},s-\delta)=\mathbf h(\boldsymbol\eta,s)$$

同步移动参考点与路径原点，点集不变。

六维描述包含一维冗余；以参考平面选取局部代表。


### 2  构造固定参考平面


$$\widehat{\mathbf t}_0=\frac{\mathbf t(\boldsymbol\eta^{(0)},0)}{\|\mathbf t(\boldsymbol\eta^{(0)},0)\|}$$


$$\mathbf e_0=\frac{\mathbf u\times\widehat{\mathbf t}_0}{\|\mathbf u\times\widehat{\mathbf t}_0\|},\quad\mathbf e_1=\widehat{\mathbf t}_0\times\mathbf e_0$$

u 默认取 z 轴；若 |t̂₀,z| > 1 − 10⁻⁶，改用 x 轴。

o = 初始参考点；(o, e₀, e₁) 在整次求解中固定。


$$\mathbf r_0=\mathbf o+a_0\mathbf e_0+a_1\mathbf e_1,\quad\rho=\kappa\rho_s$$

降维来自参考点约束；曲率缩放只改变数值坐标。


### 3  按实际 seed 流程构造初值

bᵢ 为非目标层拟合 mask，j 为首个 bᵢ = 1 的槽位。

取该点的 SMT 全局位置 pⱼ 与方向 vⱼ。


$$\mathbf r_0^{(0)}=\mathbf p_j,\qquad\psi_0^{(0)}=\operatorname{atan2}(v_{j,y},v_{j,x})$$


$$\tau^{(0)}=\frac{v_{j,z}}{\max(\sqrt{v_{j,x}^2+v_{j,y}^2},10^{-6})}$$


$$\rho^{(0)}=0,\qquad a_0^{(0)}=a_1^{(0)}=\rho_s^{(0)}=0$$


### 4  五维求解 → 六维几何输出


$$\boldsymbol\theta=(a_0,a_1,\psi_0,\tau,\rho_s)\in\mathbb R^5$$


$$\boldsymbol\eta(\boldsymbol\theta)=\left(\mathbf r_0(\boldsymbol\theta),\psi_0,\tau,\kappa\rho_s\right)$$

κ = 10⁻⁴ mm⁻¹；内部 J 有 5 列，输出参数有 6 项。

初值只用非目标层数据；每条径迹须有拟合点。


**补充说明**

这里的 gauge（参考点规范）固定是局部参数化：初始切向穿过其法平面，局部可消去沿曲线平移原点的自由度；不声称任意远处曲线与该平面有唯一交点。几何目标的参数冗余不能直接用于断言“冻结路径 JᵀJ 必定奇异”，因为该 J 没有包含路径随参数变化的项。当前 _fit_helix_points_gauss_newton_block 在进入 LM 前构造 frame，迭代过程中不重建。初始方向分母的 10⁻⁶ 是代码数值保护，不是由几何推导出的物理常数。


## 4. 最近点求解：距离目标 → Newton 迭代


### 1  固定当前几何参数，建立单点目标


$$\mathbf d(s)=\mathbf h(\boldsymbol\eta,s)-\mathbf p,\quad f(s)=\tfrac12\mathbf d^{\mathsf T}\mathbf d$$


$$f^{\prime}(s)=\tfrac12(\mathbf d_s^{\mathsf T}\mathbf d+\mathbf d^{\mathsf T}\mathbf d_s)=\mathbf d^{\mathsf T}\mathbf t$$


$$f^{\prime\prime}(s)=\mathbf t^{\mathsf T}\mathbf t+\mathbf d^{\mathsf T}\mathbf h_{ss}$$


$$f^{\prime\prime}(s)=1+\tau^2+\mathbf d^{\mathsf T}\mathbf h_{ss}$$


### 2  理想内部最近点的局部条件


$$f^{\prime}(s^*)=0\quad\Longleftrightarrow\quad\mathbf d(s^*)\perp\mathbf t(s^*)$$

f′ = 0 是必要条件；再有 f″ > 0 则足以判定严格

局部极小。f″ = 0 不能据此排除极小。


### 3  线性化驻点方程


$$0\approx f^{\prime}(s_n)+f^{\prime\prime}(s_n)\Delta s_n$$


$$\Delta s_n=-f^{\prime}(s_n)/f^{\prime\prime}(s_n)$$


### 4  初值与实际更新


$$s_0=(p_x-x_0)\cos\psi_0+(p_y-y_0)\sin\psi_0$$


$$c_n=s_n-\frac{f^{\prime}(s_n)}{\operatorname{safe}(f^{\prime\prime}(s_n))}$$


$$\operatorname{safe}(v)=\begin{cases}v,&|v|>\varepsilon_c\\+\varepsilon_c,&0\le v\le\varepsilon_c\\-\varepsilon_c,&-\varepsilon_c\le v<0\end{cases}$$

εc = 10⁻⁹；固定执行 8 次；分母保护保留符号。


### 5  数值有效性与返回值

候选须有限且 |cₙ| ≤ 1200 mm。

若当前仍有效且候选有效，令 sₙ₊₁ = cₙ；

否则保留 sₙ，并将该路径永久标为无效。

返回前再检查 h 有限；无效预测用 r₀ 填充，

须结合有效 mask 使用。初值非有限亦标无效。

无终态 f′ / f″ 检查，不保证全局最近点。


**补充说明**

路径初值取 xy 切向投影，随后优化的仍是三维欧氏距离。到初始直线的三维最小二乘投影为 ((p−r₀)ᵀt(0))/(1+τ²)，数学上可作对照，但当前代码没有采用。路径边界用于候选有效性检查，不是对闭区间约束极小问题的完整求解，也不是把路径截断到边界。分母保护仍允许负二阶导，不能保证距离每步下降。上述 cₙ 是本页局部记号，表示 Newton 候选路径。


## 5. 残差与 Jacobian → Gauss–Newton → LM


### 1  有效残差与整轨目标


$$w_i=b_i\land\operatorname{valid}(\hat s_i),\quad\mathcal E=\{i:w_i=1\}$$


$$\mathbf r_i=w_i[\mathbf h(\boldsymbol\eta(\boldsymbol\theta),\hat s_i)-\mathbf p_i]$$


$$\mathbf r=\operatorname{col}(\mathbf r_1,\ldots,\mathbf r_{N_h}),\quad F=\tfrac12\mathbf r^{\mathsf T}\mathbf r$$

无效项用 where 整体置零；不按测量方差白化。


### 2  路径停止梯度：当前实现的 Jacobian


$$J_i=w_i\left.\frac{\partial\mathbf h(\boldsymbol\eta(\boldsymbol\theta),s)}{\partial\boldsymbol\theta}\right|_{s=\hat s_i},\quad J\in\mathbb R^{3N_h\times5}$$


$$\frac{d\mathbf r_i}{d\boldsymbol\theta}=w_i\left(\mathbf h_{\boldsymbol\theta}+\mathbf t\frac{ds_i^*}{d\boldsymbol\theta}\right)$$

全导数式仅针对有效集固定、精确且可微的

最近点分支；当前实现使用上式冻结路径偏导。


$$\mathbf r_i^{\mathsf T}\mathbf t=0\ \Rightarrow\ \mathbf r_i^{\mathsf T}\mathbf t\frac{ds_i^*}{d\boldsymbol\theta}=0$$


### 3  线性化残差 → 二次模型


$$\mathbf r(\boldsymbol\theta+\Delta)\approx\mathbf r+J\Delta$$


$$m(\Delta)=\tfrac12\|\mathbf r+J\Delta\|^2$$


$$m(\Delta)=F+\mathbf g^{\mathsf T}\Delta+\tfrac12\Delta^{\mathsf T}H\Delta$$


$$\mathbf g=J^{\mathsf T}\mathbf r,\quad H=J^{\mathsf T}J,\quad H\Delta=-\mathbf g$$

H 是 GN 近似，省略残差加权的二阶导数项。


### 4  二次模型加阻尼 → LM 方程


$$\alpha=\max\left(\frac{\operatorname{tr}H}{5},10^{-30}\right),\quad M=\alpha I_5$$


$$\min_{\Delta}\ m(\Delta)+\tfrac\mu2\Delta^{\mathsf T}M\Delta$$


$$(H+\mu M)\Delta=-\mathbf g,\quad\boldsymbol\theta^c=\boldsymbol\theta+\Delta$$

M 为平均对角元乘单位阵。精确最近点处路径项

不贡献目标一阶导，但一般会改变 JᵀJ。


**补充说明**

残差乘 w 的公式只表达有效项选择；程序用 torch.where，避免将无效数值简单乘零。每次目标评估重新求路径，在线性化中对路径和有效 mask 停止梯度。对固定路径残差映射，精确 Hessian 为 JᵀJ + Σⱼ rⱼ∇²rⱼ；GN 省略后一项。对完整最近点目标，还须考虑路径全导数及有效集变化。只有在精确内部最近点、有效集固定、分支可微时，包络性质使路径项对目标一阶导的贡献为零；有限次 Newton 仅近似满足。两种 Jacobian 的二次型一般不同，因此当前驻点指标是基于冻结路径局部模型的指标。


## 6. 候选接受、驻点判据与完整迭代流程


### 1  重算候选路径与目标，再决定是否接受

对每条非驻点径迹，重新评估 θᶜ 的路径、有效集

与 Fᶜ。以下条件同时满足才接受：


$$\operatorname{finite}(F^c),\qquad F^c<F,\qquad\mathcal E^c\supseteq\mathcal E^{(0)}$$

有效集锚定求解初始状态，避免通过丢失初始

有效点来降低目标；拒绝时保留当前参数。


$$\mu^{(0)}=10^{-8},\quad\mu^+=\begin{cases}\max(0.1\mu,10^{-12}),&\mathrm{accept}\\10\mu,&\mathrm{reject}\end{cases}$$

已驻点行保持参数与阻尼；不采用 gain ratio。


### 2  ν 与“剩余下降空间”的准确关系


$$A=H+\epsilon_\mu M,\quad\Delta_A=-A^{-1}\mathbf g,\quad\nu=\mathbf g^{\mathsf T}A^{-1}\mathbf g$$


$$m(0)-m(\Delta_A)=\tfrac12\nu+\tfrac{\epsilon_\mu}{2}\Delta_A^{\mathsf T}M\Delta_A$$

加微阻尼的二次模型下降量为 ν/2；ν 是二次型，

不是开方后的 decrement，也不是实际目标下降。


### 3  在求步前判断驻点


$$\epsilon_\mu=10^{-12},\quad\nu^{\rm rel}=\frac{\nu}{\max(F,10^{-12})}$$


$$\mathrm{stat}=\operatorname{finite}(\nu)\land\left[\nu\le10^{-12}\ \lor\ \nu^{\rm rel}\le10^{-6}\right]$$

全块均驻点则退出，否则更新活跃行；达到预算

亦停止。预算停止不代表驻点达标。


### 4  终态重新评估与验收


$$n_{\rm nonfinite}=0,\qquad\frac{\#\{\text{nonstationary tracks}\}}{T}\le10^{-3}$$

终态同时检查 ν 与 νʳᵉˡ 有限；不满足门限则报错。

主线默认 Kmax = 200；底层函数默认 60，可覆盖。


### 5  完整顺序

初始化 η、固定 gauge、打包 θ

→ 最近点 ŝ → r、J → H、g → ν 与驻点判断

→ 未全驻点：LM 求步 → 候选评估 → 接受 / 拒绝

→ 调整 μ 并循环 → 终态验收 → 输出六维 η


**补充说明**

εμ 用于驻点评估，与自适应步长阻尼 μ 不同。令 mε(Δ)=m(Δ)+(εμ/2)ΔᵀMΔ，则 ΔA 是该微阻尼模型的极小点，mε(0)−mε(ΔA)=ν/2；在原始 m 上下降还多出 (εμ/2)ΔAᵀMΔA。因此有微阻尼时不应把原始模型下降量无条件写成 ν/2。只有无微阻尼且 H 可逆时，Δ=−H⁻¹g 的原始模型下降才严格等于 gᵀH⁻¹g/2。当前实际 LM 步使用自适应 μ，亦不能把 ν/2 当作其实际目标下降。终态非驻点比例允许最多 0.1%，与循环内“全部驻点才提前退出”不同。当前主线 JOINT_SMT_SEED_ITERS=200，并传入求解器；底层 GAUSS_NEWTON_MAX_ITERS=60 只在调用未覆盖时生效。CPU 路径使用 float64；主线 GPU 路径可传入 float32，底层求解器不会自动升精度。主线还按配置检查无效点比例。这里不使用“10⁻⁶ 低于 float32 分辨率”这一不准确解释。


## 核对与阅读边界

- 本文依据当前源码核对，未修改算法，也未运行算法实验；不包含新数值精度或性能结论。
- 原始 `seed_derivation.md` / `.pptx` 保留。新稿修正参考系重建范围、实际调用迭代预算、极小条件和微阻尼下降量等表述。
- 所有展示公式在 PPT 中为 OMML 原生公式对象；表格和正文为可编辑文本。PowerPoint 使用原生公式分支；不支持该公式扩展的阅读器使用同式的高清图像回退分支，不显示 LaTeX 源码。
- PPT 六页与本文六节逐一对应，补充说明同时写入各页备注；关键实现条件已在正文显示。

### 实现索引

| 环节 | 代码位置 |
|---|---|
| 主线非目标层掩码、初值与 seed 调用 | `scripts/ml_alignment_autograd.py::_non_target_fit_mask / seed_helices` |
| 200 轮主线默认值与终态门限调用 | `scripts/ml_alignment_autograd.py::JOINT_SMT_SEED_ITERS` 与 seed 调用处 |
| 曲线与路径求解 | `scripts/helix_model.py::helix_point / helix_tangent / closest_helix_points` |
| 固定参考系与五维映射 | `scripts/helix_model.py::reference_gauge_frame / pack_gauge_fixed_track_parameters / unpack_gauge_fixed_track_parameters` |
| 残差、有效集与 Jacobian | `scripts/helix_model.py::_single_point_fit_residual / _single_point_fit_effective_mask / _fit_helix_points_gauss_newton_block` |
| LM 与验收 | `scripts/helix_model.py::damped_gauss_newton / require_stationary_helix_point_fit` |
