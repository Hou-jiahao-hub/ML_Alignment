# 公式附录：A1–A14

配套正文：[slides.md](slides.md)。每节为一页附录的完整上屏材料；“讲解备注”不要求上屏。公式以当前单 helix 主线路径为边界，不扩展至材料节点拟合。

## A1｜符号、参数与单位约定

**结论句：横向弧长、三维弧长及优化变量必须分别定义。**

| 符号 | 定义 | 单位或维度 |
|---|---|---|
| \(t,i,d,k\) | 径迹、hit、局部测量分量、外层迭代索引 | 索引 |
| \(s\) | 沿所选方向增加的横向弧长参数 | mm |
| \(\ell\) | 三维弧长，\(d\ell/ds=\sqrt{1+\tau^2}\) | mm |
| \(\boldsymbol\eta\) | \((x_0,y_0,z_0,\psi_0,\tau,\rho)\) | 6 个物理参数 |
| \(\tau\) | \(\tan\lambda=dz/ds\) | 无量纲 |
| \(\rho\) | \(d\psi/ds\)，带符号横向曲率 | mm\(^{-1}\) |
| \(\boldsymbol\theta\) | \((a_0,a_1,\psi_0,\tau,\rho_s)\) | 5 个优化变量 |
| \(\mathbf p,\mathbf h\) | SMT 全局点、helix 全局预测 | mm，三维 |
| \(m_d,V_d\) | 局部 measurement 及其方差 | mm，mm² |
| \(R,\mathbf c\) | 局部到全局旋转、测量面中心 | \(3\times3\)，mm |
| \(r,J,H,g\) | 残差、Jacobian、GN 近似 Hessian、梯度 | 随目标及参数定义 |
| \(\mu,M,\nu\) | LM 阻尼水平、阻尼尺度矩阵、decrement 二次型 | 不使用同一符号 |

**讲解备注：**下文为减少拥挤，在单径迹推导中省略 \(t\)。\(\tau\) 避免将倾角 \(\lambda\) 与原图 LM 阻尼 \(\lambda_t\) 混淆。数值下限按实现的 mm 参数单位使用，不意味着所有优化变量具有相同物理量纲。

**布局：**三列表格。**源码：** `helix_model` 模块说明、常量与 `helix_point`。

## A2｜Helix 位置参数方程

**结论句：由横向切向方位角积分得到带符号曲率的 helix。**

令 \(\psi(s)=\psi_0+\rho s\)，并定义
\[
\frac{dx}{ds}=\cos\psi(s),\qquad
\frac{dy}{ds}=\sin\psi(s),\qquad
\frac{dz}{ds}=\tau.
\]
对 \(\rho\ne0\)，从 \(s=0\) 积分至 \(s\)：
\[
x(s)=x_0+\frac{\sin(\psi_0+\rho s)-\sin\psi_0}{\rho},
\]
\[
y(s)=y_0+\frac{\cos\psi_0-\cos(\psi_0+\rho s)}{\rho},
\qquad z(s)=z_0+\tau s.
\]
令 \(a=\rho s/2\)，使用和差化积：
\[
\sin(\psi_0+2a)-\sin\psi_0=2\cos(\psi_0+a)\sin a,
\]
\[
\cos\psi_0-\cos(\psi_0+2a)=2\sin(\psi_0+a)\sin a.
\]
因此 x、y 的稳定式均为原点坐标加增量，y 增量前是正号。

**讲解备注：**这里 \(\rho\) 是几何拟合变量，不通过磁场、荷电量或动量强制计算。原图磁场关系含额外质量因子，应删去；本页不引入与当前求解无关的磁场—动量单位换算。

**布局：**左侧定义与积分，右侧和差化积；修正后的 y 式框选。**源码：** `helix_point`。

## A3｜稳定表达、切向量与直线极限

**结论句：sinc 表达在零曲率初值处连续且可微。**

定义非归一化 \(\operatorname{sinc}(a)=\sin a/a\)，\(\operatorname{sinc}(0)=1\)：
\[
\mathbf h(s)=\begin{bmatrix}
x_0+s\cos(\psi_0+\rho s/2)\operatorname{sinc}(\rho s/2)\\
y_0+s\sin(\psi_0+\rho s/2)\operatorname{sinc}(\rho s/2)\\
z_0+\tau s
\end{bmatrix}.
\]
\[
\operatorname{sinc}(a)=1-\frac{a^2}{6}+O(a^4)
\quad\Longrightarrow\quad
\lim_{\rho\to0}\mathbf h(s)=\mathbf r_0+s(\cos\psi_0,\sin\psi_0,\tau)^{\mathsf T}.
\]
\[
\mathbf t(s)=\mathbf h'(s)=
(\cos\psi(s),\sin\psi(s),\tau)^{\mathsf T},
\]
\[
\mathbf h''(s)=(-\rho\sin\psi(s),\rho\cos\psi(s),0)^{\mathsf T}.
\]
\[
\|\mathbf t(s)\|=\sqrt{1+\tau^2},\qquad
\widehat{\mathbf t}(s)=\mathbf t(s)/\sqrt{1+\tau^2}.
\]

**讲解备注：**代码 `_sinc(a)=torch.sinc(a/π)`，因为 PyTorch 的 sinc 采用归一化定义。上述一、二阶导数是对横向弧长的导数；不应称 \(\mathbf t\) 为“单位切向量”。零曲率初值仍允许通过可微表达更新曲率。

**布局：**位置式在左，导数与极限在右。**源码：** `_sinc`、`helix_point`、`helix_tangent`。

## A4｜参数原点冗余及其数值影响

**结论句：沿同一曲线改变参考原点不会引入新的几何自由度。**

沿原 helix 移动横向弧长 \(b\)，定义新参数：
\[
\mathbf r'_0=\mathbf h(\boldsymbol\eta,b),\qquad
\psi'_0=\psi_0+\rho b,\qquad \tau'=\tau,\qquad \rho'=\rho.
\]
新旧参数描述同一条曲线：
\[
\mathbf h(\boldsymbol\eta',u)=\mathbf h(\boldsymbol\eta,u+b).
\]
在 \(b=0\) 处，该参数变换的切向为
\[
v=\frac{d\boldsymbol\eta'}{db}\bigg|_{0}
=(\cos\psi_0,\sin\psi_0,\tau,\rho,0,0)^{\mathsf T}.
\]
对同一交点分支、无路径边界切换的原点不变预测 \(q(\boldsymbol\eta)\)：
\[
q(\boldsymbol\eta')=q(\boldsymbol\eta)
\Rightarrow J_qv=0
\Rightarrow (J_q^{\mathsf T}J_q)v=0.
\]
**后果：**六维表示存在一个不可辨识方向；规范固定选择该等价类的局部代表，消除原点冗余。

**讲解备注：**这是理想几何及相应完整导数的结论。有限 Newton 迭代、路径边界、有效性切换会影响数值不变性；seed 的冻结路径 Jacobian 也不必满足 \(Jv=0\)。不要写成“当前两个求解器的六维数值 Hessian 必然精确秩为 5”。该约束只处理原点冗余，剩余五维仍可能因观测不足而病态。

**布局：**等价曲线图与参数变换并排；底部三步蕴含式。**源码：** `reference_gauge_frame`；几何等价式由 A2 推导。

## A5｜五维规范固定与参数初始化

**结论句：用参考切向法平面内的两个位移参数表示优化原点。**

参考方向及平面正交基：
\[
\mathbf d=\frac{\mathbf t_{\rm ref}(0)}{\|\mathbf t_{\rm ref}(0)\|},\quad
\mathbf e_0=\frac{\mathbf b\times\mathbf d}{\|\mathbf b\times\mathbf d\|},\quad
\mathbf e_1=\mathbf d\times\mathbf e_0.
\]
默认 \(\mathbf b=(0,0,1)\)；若 \(|d_z|>1-10^{-6}\)，改用 \((1,0,0)\)。
\[
\mathbf r_0=\mathbf r_{\rm ref}+a_0\mathbf e_0+a_1\mathbf e_1
\Rightarrow (\mathbf r_0-\mathbf r_{\rm ref})\cdot\mathbf d=0.
\]
优化变量及恢复：
\[
\theta=(a_0,a_1,\psi_0,\tau,\rho_s),\qquad
\rho=c_\rho\rho_s,\quad c_\rho=10^{-4}\,\mathrm{mm}^{-1}.
\]
初始化：
\[
\mathbf r_{\rm ref}=\mathbf p_{\rm first},\quad
\psi_0=\operatorname{atan2}(u_y,u_x),\quad
\tau=\frac{u_z}{\max(\sqrt{u_x^2+u_y^2},10^{-6})},\quad
(a_0,a_1,\rho_s)=(0,0,0).
\]
链式法则示例：
\[
\frac{\partial r}{\partial a_j}=\frac{\partial r}{\partial\mathbf r_0}\mathbf e_j,
\qquad \frac{\partial r}{\partial\rho_s}=c_\rho\frac{\partial r}{\partial\rho}.
\]

**讲解备注：**refit 的参考 helix 是 seed 输出，故 refit 初始曲率一般不为零，且其 pack 中的 \(\rho_s\) 来自 seed。平面在单次拟合中固定；新拟合或终态诊断可重新建平面。曲率缩放不保证正规矩阵条件数良好，也不让所有参数具有相同物理单位。

**布局：**左侧平面约束及映射，右侧初始化和链式法则。**源码：** `seed_helices`、gauge frame、pack/unpack。

## A6｜点到 Helix 的最近点问题

**结论句：最近点估计通过距离平方的一维 Newton 迭代获得。**

给定 \(\boldsymbol\eta\) 与 SMT 点 \(\mathbf p\)，令
\[
\mathbf d(s)=\mathbf h(\boldsymbol\eta,s)-\mathbf p,\qquad
f(s)=\tfrac12\mathbf d(s)^{\mathsf T}\mathbf d(s).
\]
\[
f'(s)=\mathbf d(s)\cdot\mathbf t(s),\qquad
f''(s)=\mathbf t(s)\cdot\mathbf t(s)+\mathbf d(s)\cdot\mathbf h''(s).
\]
内部局部极小点需满足 \(f'(s^*)=0\)；Newton 更新为
\[
s_{n+1}=s_n-\frac{f'(s_n)}{\operatorname{safe}(f''(s_n))}.
\]
**代码采用的初值：**
\[
s_0=(p_x-x_0)\cos\psi_0+(p_y-y_0)\sin\psi_0.
\]
**作为对照的三维直线投影（当前未采用）：**
\[
s_0^{3D}=\frac{(p_x-x_0)\cos\psi_0+(p_y-y_0)\sin\psi_0+(p_z-z_0)\tau}{1+\tau^2}.
\]
**数值规则：**默认 8 次迭代；二阶导数按符号保护至至少 \(10^{-9}\) 的绝对值；非有限或 \(|s_{n+1}|>1200\) mm 的候选使该路径无效，并保留上一值。

**讲解备注：**单次路径迭代不恢复已失效路径，但下一次对另一组 helix 参数调用会重新初始化有效性。该函数没有以 \(|f'|\) 或 \(f''>0\) 作终态极小性证明；有限次 Newton 和有效性检查不能保证全局最近点。它返回 `(fitted, s, valid)`；无效 fitted 使用轨迹原点填充，必须结合 mask 使用。

**布局：**目标→导数→更新三行；初值与数值规则放侧栏。**源码：** `closest_helix_points`、`accept_newton_candidate`。

## A7｜SMT seed 残差与 Jacobian

**结论句：每次评估重新求路径，但本次线性化对路径停止梯度。**

令 \(w_{ti}\) 为非目标拟合 mask 与最近点有效性的交集，残差为
\[
\mathbf r_{ti}=w_{ti}\left[\mathbf h(\boldsymbol\eta(\theta_t),s_{ti}^{*})-\mathbf p_{ti}\right],
\quad r_t=\operatorname{vec}(\mathbf r_{ti}),\quad F_t=\tfrac12 r_t^{\mathsf T}r_t.
\]
代码的局部 Jacobian：
\[
J_{ti}^{\rm code}
=w_{ti}\left.\frac{\partial\mathbf h(\boldsymbol\eta(\theta_t),s)}{\partial\theta_t}\right|_{s=s_{ti}^{*}}.
\]
完整最近点预测的导数另含路径项：
\[
\frac{d\mathbf r_{ti}}{d\theta_t}
=w_{ti}\left(\frac{\partial\mathbf h}{\partial\theta_t}
+\mathbf t\frac{ds_{ti}^{*}}{d\theta_t}\right),
\]
其中忽略离散 mask 切换，且假定路径可微。在精确内部最近点处，
\[
\mathbf r_{ti}^{\mathsf T}\mathbf t=0
\quad\Rightarrow\quad
\mathbf r_{ti}^{\mathsf T}\mathbf t\frac{ds_{ti}^{*}}{d\theta_t}=0.
\]
因此路径项在目标一阶导数中消失，但不意味着两个 Jacobian 或 \(J^{\mathsf T}J\) 相同。

**候选检查：**重新求候选路径和目标，且要求
\[
\mathcal E_t(\theta_t^{\rm cand})\supseteq\mathcal E_t(\theta_t^{\rm init}).
\]

**讲解备注：**保护对象是“初始有效集合”，不是“上一轮有效集合”；后续新增的有效点可能再失效。有限 Newton 未必满足精确正交条件，上述一阶导数等价只是带条件的理论说明。代码以 `no_grad` 冻结路径，再用 `helix_point` 重新构造可微预测；后者不是无意义的重复计算。

**布局：**左侧实现，右侧完整导数对照；底部候选集合包含关系。**源码：** `_single_point_fit_residual`、`candidate_ok_of`。

## A8｜Gauss–Newton 正规方程

**结论句：用残差一阶近似构造局部二次优化问题。**

\[
F(\theta)=\tfrac12r(\theta)^{\mathsf T}r(\theta),\qquad
r(\theta+\Delta)\approx r+J\Delta.
\]
\[
m(\Delta)=\tfrac12\|r+J\Delta\|^2
=F+g^{\mathsf T}\Delta+\tfrac12\Delta^{\mathsf T}H\Delta,
\quad g=J^{\mathsf T}r,\quad H=J^{\mathsf T}J.
\]
令 \(\nabla_\Delta m=0\)，得到
\[
H\Delta=-g.
\]
对真正可微的原残差，其精确 Hessian 为
\[
\nabla^2F=J^{\mathsf T}J+\sum_jr_j\nabla^2r_j.
\]
Gauss–Newton 忽略残差二阶项，因此 \(H\) 应称为近似 Hessian，不能直接称“精确二阶导数”。

**实现维度：**当前单 helix 每条径迹有 5 个优化变量；seed 残差长度为 \(3H_{\max}\)，measurement 残差长度为 \(2H_{\max}\)，mask 分量为零。`vmap(jacrev(...))` 构造逐径迹 Jacobian，径迹之间独立求解。

**讲解备注：**seed 使用 A7 的冻结路径线性化，应理解为当前局部模型的 GN 方程，不能将精确 Hessian 恒等式直接作用到被停止梯度的整个原始目标。此处不是利用跨径迹均值目标产生耦合更新。

**布局：**目标→线性化→正规方程；旁注近似来源。**源码：** 两阶段 Jacobian 与 LM 循环。

## A9｜LM 阻尼与候选步接受

**结论句：逐径迹调节阻尼，并以重新计算的非线性目标决定是否提交候选。**

对五维优化变量定义
\[
\alpha_t=\max\left(\frac15\operatorname{tr}H_t,10^{-30}\right),\quad M_t=\alpha_t I_5.
\]
\[
(H_t+\mu_tM_t)\Delta_t=-g_t,\qquad
\theta_t^{\rm cand}=\theta_t+\Delta_t.
\]
\[
\theta_t^{\rm next}=
\begin{cases}\theta_t^{\rm cand},&\text{候选被接受},\\\theta_t,&\text{候选被拒绝},\end{cases}
\qquad
\mu_t^{\rm next}=\begin{cases}\max(0.1\mu_t,10^{-12}),&\text{接受},\\10\mu_t,&\text{拒绝}.\end{cases}
\]
初始 \(\mu_t=10^{-8}\)。阻尼形式是“平均对角元素乘单位阵”，不是 \(\operatorname{diag}(H)\)。

| 项目 | seed | refit |
|---|---|---|
| 目标要求 | 有限且严格下降 | 代码直接比较候选目标严格下降 |
| 额外候选条件 | 初始有效点不得失效；仅更新非驻点径迹 | 无相同的初始有效集合保护 |
| 拒绝后 | 参数不变，活跃径迹阻尼增大 | 参数不变，阻尼增大 |

**讲解备注：**该实现采用自适应阻尼 GN/LM 策略，没有用实际／预测下降比构造完整信赖域接受规则。数学公式中的矩阵求逆应通过线性方程求解实现。refit 内层没有独立 `isfinite` 接受谓词；NaN 不能通过严格小于比较，终态另有非有限诊断，不宜额外画入不存在的内层保护。

**布局：**共同方程与分支规则；底部小表。**源码：** `damped_gauss_newton`、`fit_broken_helix`。

## A10｜Helix 与测量平面的交点

**结论句：求交是平面约束的一维求根问题，与点到曲线最近点问题不同。**

平面中心 \(\mathbf c\)、单位法向 \(\mathbf n=R_{:,2}\)：
\[
f(s)=[\mathbf h(s)-\mathbf c]\cdot\mathbf n=0,\qquad
f'(s)=\mathbf t(s)\cdot\mathbf n.
\]
用原点处切线 \(\mathbf h(s)\approx\mathbf r_0+s\mathbf t(0)\) 得到初值：
\[
s_0=\frac{(\mathbf c-\mathbf r_0)\cdot\mathbf n}{\mathbf t(0)\cdot\mathbf n}.
\]
由一阶 Taylor 展开 \(f(s_n+\Delta s)\approx f(s_n)+f'(s_n)\Delta s\) 得到
\[
s_{n+1}=s_n-\frac{[\mathbf h(s_n)-\mathbf c]\cdot\mathbf n}
{\mathbf t(s_n)\cdot\mathbf n}.
\]
**当前数值条件：**默认 12 次迭代；初始及迭代导数的绝对值须不小于 \(10^{-6}\)；路径与坐标须有限；\(|s|\le1200\) mm；终态 \(|f(s)|\le10^{-3}\) mm。

**失效处理：**越界候选使路径无效，并保留上一迭代值；最终无效输出的路径置零、点置为原点，同时返回 `valid=False`。

**讲解备注：**近平行表示切向与平面近似平行，即与法向近似垂直；不是“切向与法向平行”。分母保护服务于数值计算，不能将其写成接受近平行解。共享函数签名不含 `s0` 参数。输出顺序为 `(s, point, valid)`，与最近点函数不同。求解的是无限平面约束，函数本身不检查传感器有限边界内包含关系。

**布局：**左侧四行推导，右侧近平行与有效性条件。配图 `a10_intersection.svg/png`。**源码：** `helix_model.intersect_plane`。

## A11｜局部测量残差与 half-χ²

**结论句：残差统一定义为测量减预测，并按有效分量的方差标准化。**

令 \(P=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}\)，则
\[
\widehat{\mathbf m}_{ti}=PR_{ti}^{\mathsf T}
\big[\mathbf h(\boldsymbol\eta_t,s_{ti}^{\cap})-\mathbf c_{ti}\big].
\]
\[
\widetilde r_{ti,d}=\frac{m_{ti,d}-\widehat m_{ti,d}}
{\sqrt{\max(V_{ti,d},\epsilon_V)}},\qquad \epsilon_V=10^{-8}\,\mathrm{mm}^2.
\]
\[
w_{ti,0}=\mathrm{fit}_{ti}\land\mathrm{valid}_{ti},\quad
w_{ti,1}=w_{ti,0}\land\mathrm{LOC1valid}_{ti},\quad
r_{ti,d}=\begin{cases}\widetilde r_{ti,d},&w_{ti,d},\\0,&\text{否则}.\end{cases}
\]
\[
F_t=\tfrac12\sum_{i,d}r_{ti,d}^2,\qquad
\chi^2=2\sum_tF_t,\qquad
\frac{\chi^2}{N_{\rm comp}}=\frac{2\sum_tF_t}{\sum_{t,i,d}w_{ti,d}}.
\]
求解器报告的“逐径迹平均 half-χ²”与主线诊断的“每有效分量 χ²”是不同统计量，后者也不是扣除拟合自由度后的 χ²/NDF。

**讲解备注：**这里使用输入中的有效测量方差；数值下限不等同于上游探测器分辨率策略。实现按对角方差标准化，没有额外求 LOC0–LOC1 完整协方差逆平方根。mask 中不包含目标层；有效性随当前求交变化，因此不是严格固定的残差样本集合。零节点路径没有额外材料先验。

**布局：**坐标变换、残差、mask、目标四层。**源码：** `_standardized_residual`、`_measurement_refit_objective`。

## A12｜Measurement Jacobian 的求导路径

**结论句：局部预测的参数导数包含交点随径迹参数的变化。**

固定几何，若交点分支光滑且 \(\mathbf n\cdot\mathbf t\ne0\)，由
\[
f(\theta,s(\theta))=\mathbf n^{\mathsf T}[\mathbf h(\theta,s)-\mathbf c]=0
\]
得到
\[
\frac{\partial s}{\partial\theta}
=-\frac{\mathbf n^{\mathsf T}\partial_\theta\mathbf h}
{\mathbf n^{\mathsf T}\mathbf t}.
\]
\[
\frac{d\widehat{\mathbf m}}{d\theta}
=PR^{\mathsf T}\left[\partial_\theta\mathbf h+
\mathbf t\frac{\partial s}{\partial\theta}\right]
=PR^{\mathsf T}\left[I-\frac{\mathbf t\mathbf n^{\mathsf T}}{\mathbf n^{\mathsf T}\mathbf t}\right]
\partial_\theta\mathbf h.
\]
令 \(W=\operatorname{diag}(1/\sqrt{\max(V_d,\epsilon_V)})\)，有效分量的理论 Jacobian 为
\[
J=-W\frac{d\widehat{\mathbf m}}{d\theta}.
\]
**当前实现：**`vmap(jacrev(...))` 对有限次 Newton 求交与局部变换的可微运算求导，并非显式使用上述隐式公式。布尔有效性、分段选择等离散判断不按连续变量求导。

**讲解备注：**隐式表达以精确根和固定分支为前提；有限迭代及分支边界处不保证与其严格相等。零材料节点的单 helix 路径中，预先计算的 `hit_s` 只涉及分段准备，不替代真正的平面交点弧长。与 seed 的停止路径梯度形成明确区别。

**布局：**左侧三步隐式推导，右侧“理论／实现”框。**源码：** `broken_helix_local_prediction`、`_BATCHED_RESIDUAL_JACOBIAN`。

## A13｜停止判据与驻点验收

**结论句：Newton decrement 检查局部模型中的剩余变化；change 只描述本轮目标变化。**

项目记录的 decrement 是二次型：
\[
\nu_t=g_t^{\mathsf T}(H_t+\epsilon_\mu M_t)^{-1}g_t,\qquad\epsilon_\mu=10^{-12}.
\]
若采用可逆的未阻尼二次模型及 \(\Delta=-H^{-1}g\)，预测下降为
\[
m(0)-m(\Delta)=\tfrac12g^{\mathsf T}H^{-1}g.
\]
因此 \(\nu\) 不是该下降量本身，也没有取平方根；加微小阻尼后的实际定义与未阻尼模型仍有区别。

**Seed：循环内逐径迹判据**
\[
\mathrm{finite}(\nu_t)\ \land\
\left[\nu_t\le10^{-12}\ \lor\
\frac{\nu_t}{\max(F_t,10^{-12})}\le10^{-6}\right].
\]
分块内全部满足则停止；否则仅更新未满足的径迹。预算耗尽后仍重新评估终态，终态要求 decrement 及相对量有限。

**Refit：循环停止量与独立验收**
\[
\delta_t=\frac{\max(F_t^{\rm old}-F_t^{\rm cand},0)}{\max(|F_t^{\rm old}|,1)},\qquad
\max_{t\in\mathrm{chunk}}\delta_t<10^{-10}.
\]
停止后重建终态诊断参考系并计算 \(\nu_t\)，按 \(\nu_t\le10^{-6}\) 判定驻点。

**主线门限：**两阶段各自的非驻点比例均须不超过 \(10^{-3}\)；非有限 decrement 立即失败，不纳入容许比例。

**讲解备注：**refit 的 `converged=True` 由 change 触发，不是独立驻点验收结果。预算耗尽也可能终态驻点合格；反之 change 很小也可能不合格。seed 的最大相对下降量只是诊断，不用于循环停止。固定点路径近似、微小阻尼和有效性分支使 decrement 应被描述为“项目采用的数值驻点指标”，不是全局最优证明。

**布局：**上部定义，下面 seed/refit 双栏。**源码：** `damped_gauss_newton`、`fit_broken_helix`、`newton_decrement`、`stationarity_report`。

## A14｜数值阈值、失败条件与代码对应

**结论句：预算、循环停止标记和主线质量门限必须分别解释。**

| 项目 | 当前默认值／行为 | 所属位置 |
|---|---|---|
| 非目标 hit 数／累计折线长度 | ≥8／≥300 mm | `build_track_batches` |
| seed 外层预算 | 主线默认 200；共享函数默认 60，由调用方覆盖 | `JOINT_SMT_SEED_ITERS`、`seed_helices` |
| refit GN 外层预算 | 每 chunk 默认 60 | `fit_helices_gauss_newton` |
| refit 配置的 200 轮 | 仅传给 Adam 路径，不控制当前默认 GN | `fit_helices_to_measurements` |
| 行分块 | 最多 2048 条／块 | seed 和 refit wrapper |
| 曲率尺度 | \(10^{-4}\) mm\(^{-1}\) | `CURVATURE_SCALE` |
| 初始／最低 LM 阻尼 | \(10^{-8}\)／\(10^{-12}\) | 两阶段 solver |
| 最近点迭代／分母保护 | 8／\(10^{-9}\) | `closest_helix_points` |
| 平面求交迭代／近平行门限 | 12／\(10^{-6}\) | `intersect_plane` |
| 路径范围／平面残差 | ±1200 mm／≤\(10^{-3}\) mm | `intersect_plane` |
| measurement 方差数值下限 | \(10^{-8}\) mm² | `EPS_VARIANCE` |
| seed 驻点 | 绝对 \(10^{-12}\) 或相对 \(10^{-6}\) | `damped_gauss_newton` |
| refit change／驻点 | \(<10^{-10}\)／\(\nu\le10^{-6}\) | 内层／终态诊断 |
| 非驻点比例 | ≤0.1%，两阶段分别检查 | 主线门限 |
| 失效率门限 | 默认 ≤5% | seed 最近点终态；refit 初始及终态求交 |

**失败出口：**无可用径迹／无有效 seed 点；终态非有限 decrement；非驻点比例超限；失效率超限；求解异常向调用方传播。不得将这些出口并入正常输出箭头。

失效率的分母是总体非目标层拟合 hit 数，并非逐径迹 hit 数；与逐径迹非驻点比例属于不同统计层次。

**讲解备注：**默认值不是每次实验的有效配置；具体运行需读取配置及 manifest。measurement GN 内部使用 float64，输出转换回调用方类型；生产 GPU seed 可使用 float32。当前任务只核验和制作文稿，未执行算法实验。源码版本和哈希记录在配套快照中，不将工作区内容无条件等同于 HEAD。

**布局：**表格为主，失败出口置底部；全部源码定位见 [verification.md](verification.md)。
