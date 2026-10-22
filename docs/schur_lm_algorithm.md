# 从零复现径迹—几何联合 Schur-LM 校准：数学原理与计算流程

> 阅读目标：在不知道现有代码结构的情况下，能够建立同样的数学模型，并独立实现、检查该算法。  
> 范围：固定非目标几何，以单螺旋线描述径迹，联合估计目标传感器的局部刚体修正和径迹参数。  
> 求导约定：具体函数的导数由 autograd 提供；本文不推导旋转、螺旋线或交点的解析 Jacobian。保留决定求解方法的线性化、正规方程、分块消元和约束计算。  
> 日期：2026-10-21。具体数值约定集中在附录 A，源码映射和验证记录放在附录 C—D。

## 1. 要复现这个算法，需要掌握哪些数学原理

这个校准问题可以从一句话开始：**已知传感器记录的局部测量，寻找一组传感器位置、姿态和粒子轨迹，使轨迹在传感器上的预测尽可能符合测量。**

困难在于，同一个残差既可以通过移动传感器改变，也可以通过改变径迹改变。算法需要同时处理这两类未知量，并利用固定参考层区分它们。autograd 负责计算模型的局部敏感度，建模、统计权重、消元、约束及是否收敛仍需由算法设计确定。

推荐按下表顺序学习。最后一列可以用来检查自己是否理解了相应原理。

| 数学基础 | 在算法中解决的问题 | 学习后的检查问题 |
|---|---|---|
| 向量、矩阵、基和坐标变换 | 将同一个点在全局与传感器局部坐标间转换 | 为什么局部到全局用 $R$，反向用 $R^\mathsf T$？ |
| 三维刚体运动与旋转矩阵 | 用有限平移和转角描述传感器几何 | 平移沿哪个基准坐标轴？三个转角按什么顺序组合？ |
| 空间曲线与几何约束 | 描述单螺旋线、最近点和平面交点 | 轨迹到 SMT 点最近，与轨迹穿过测量平面，为什么是两个问题？ |
| 数值求根与稳定计算 | 在有限精度中算出路径和交点 | Newton 迭代失效时怎样保持有效性和样本可比性？ |
| 参数化、尺度变换与 gauge | 去掉同一曲线的冗余表示，并统一数值坐标 | 六个物理参数为什么只用五个优化变量？ |
| 概率、方差与加权最小二乘 | 判断不同测量偏差应占多大权重 | 方差、标准差、白化残差和目标均值分别是什么？ |
| Jacobian 与局部线性化 | 把非线性残差近似为当前点附近的线性模型 | 交给 autograd 的函数输入和输出分别是什么？ |
| Gauss–Newton 与 LM | 从线性化模型产生可靠的非线性更新 | GN 解什么问题？阻尼与增益比各控制什么？ |
| 块矩阵与 Schur 补 | 消去大量径迹变量，保留全局几何耦合 | 为什么共享轨迹的两个传感器不能独立求解？ |
| 约束优化与 KKT | 在累计修正的允许范围内求解 | 位于边界时，哪个方向应冻结？ |
| 正定性、秩、特征值与 decrement | 区分已接近驻点和数据缺少约束 | 为什么目标下降或步很小，都不足以单独证明收敛？ |

本任务不要求先掌握神经网络、手写旋转生成矩阵导数、手写交点隐式微分或 ACTS 的内部传播 Jacobian。需要掌握的是：如何定义可计算的前向模型，怎样把它组织为一个数值自洽的联合最小二乘问题。

全文按数学依赖组织，而不是按现有文件划分。主要计算链是

~~~text
几何和轨迹参数
    → 螺旋线与测量平面的交点
    → 局部测量预测
    → 白化残差
    → autograd 给出 Jacobian
    → 联合 GN 正规方程
    → Schur 消元与回代
    → 满足约束的联合试探步
    → 真实残差比较与 LM 接受
    → 全局驻点、秩和终态判断
~~~

## 2. 先把校准问题定义完整

### 2.1 已知什么，估计什么

已知输入至少包含：

- 全部相关传感器的基准中心与姿态，其中非目标面提供固定参考；
- 每条径迹关联到哪些传感器，以及各 hit 的局部测量和测量误差；
- 用于初始化的非目标 SMT 三维点与方向；
- 目标层、允许估计的几何自由度、累计修正边界和数值控制参数。

SMT 表示上游重建提供的平滑径迹信息。它用于几何预拟合，不是粒子 MC 真值。复现优化器可以直接从整理好的数组开始；ROOT 的读取布局不属于数学模型的必要前提。

要估计的未知量是两组：每个有观测的目标 sensor 的局部几何修正，以及每条保留径迹的螺旋线参数。非目标几何始终固定。初始化虽只用非目标项，最终联合目标使用参考与目标的测量；这里没有把目标测量作为独立测试集。

### 2.2 模型的适用假设

每条径迹用一条常横向曲率、常纵横斜率的螺旋线描述。每个测量面按其局部平面处理。hit 与 sensor 的对应关系已知且固定，不在优化中重新关联；求交也不检查有限 sensor 边界。

模型没有材料散射节点、过程噪声或能损传播。统计权重取局部测量分量的对角方差。因此，“这个数学问题被正确求解”和“模型足以描述所有真实数据”是两项不同的判断。

固定参考几何提供了对齐的锚点。如果把整个探测器和所有径迹共同平移、转动，很多相对测量仍可能不变；允许所有对象一起自由移动会引入额外不可辨识模式。固定参考面和轨迹自身的五参数 gauge 处理的是两种不同层面的冗余。

### 2.3 全文核心符号

全文用列向量，长度为 mm，转角为 rad，曲率为 mm⁻¹。下标 $t,h,i,\ell$ 分别表示径迹、径迹内 hit、sensor、测量分量；$i=i(t,h)$，$\ell=0,1$。$k$ 为联合外迭代，$n$ 为路径求根迭代。

| 符号 | 含义 |
|---|---|
| $N_{\rm trk},N_{\rm hit},N_{\rm sen}$ | 保留径迹数、每条径迹补齐后的 hit 槽位数、实际参与求解的目标 sensor 数 |
| $\mathbf c_i^{(0)},R_i^{(0)}$ | 输入工作几何中的面中心、局部到全局旋转 |
| $\mathbf a_i=(\mathbf d_i,\boldsymbol\varphi_i)$ | 相对工作几何的累计局部平移和 Euler 转角，6 维 |
| $\mathcal D,K_a,E_a$ | 活动自由度的有序列表、其个数、无量纲步到六维物理步的映射 |
| $\boldsymbol\eta_t,\boldsymbol\theta_t$ | 六维物理 helix 参数、五维 gauge 优化坐标 |
| $\mathbf x_t(s),\tau_t,\kappa_t$ | helix 全局位置、纵横斜率、有符号横向曲率 |
| $\mathbf m_{th},\widehat{\mathbf m}_{th}$ | 局部测量、局部预测，均为 2 维 |
| $V_{th\ell}^{\rm raw},V_{th\ell},V_{th\ell}^{\rm eff}$ | 原始诊断方差、施加 strip 下限后的方差、白化实际方差 |
| $\mathbf e,\mathbf r,\mathbf d^{\rm seed}$ | 未白化测量残差、白化测量残差、三维 seed 残差 |
| $\mathcal I_k,\mathcal F$ | 当前联合比较冻结的有效分量集合、自由几何 DOF 集合 |
| $A,B$ | 白化残差对无量纲几何坐标、五维径迹坐标的 Jacobian |
| $C,G,\Gamma$ | 几何、几何—径迹交叉、径迹 GN 信息块 |
| $\mathbf g_a,\boldsymbol\beta$ | 几何梯度、径迹梯度；不与几何中心 $\mathbf c_i$ 混用 |
| $S_\mu,\mathbf b_\mu$ | 带相应阻尼的 Schur 约化信息与约化梯度 |
| $\Delta\mathbf u,\Delta\boldsymbol\theta,\boldsymbol\delta$ | 无量纲几何步、五维径迹步、二者堆叠的联合步 |
| $\mu_{\rm seed},\mu_{\rm ref},\mu_{\rm LM}$ | 三个阶段各自的阻尼水平 |
| $\rho_{\rm LM},\alpha_{\rm step}$ | 联合 LM 增益比、整条联合步的可行缩放比例 |
| $\nu_{\rm seed},\nu_{\rm ref},\nu_{\rm geo},\nu_{{\rm trk},t}$ | 各阶段 decrement 二次型，均不额外开方 |

变量维度和新引入的局部量在对应章节继续定义。$F_{\rm seed}$ 的单位为 mm²；白化后的 $F_{\rm ref},F_{\rm joint}$ 无量纲。这些目标均指未按样本数归一的半平方和，报告均值另行注明。

## 3. 需要怎样的线性代数基础

### 3.1 向量内积、范数和矩阵映射

对列向量 $\mathbf z$，$\|\mathbf z\|^2=\mathbf z^\mathsf T\mathbf z$ 表示各分量平方和。矩阵乘法表示从一种坐标或扰动空间映射到另一种空间。判断任何公式前，先检查输入维度、输出维度和单位。

例如 $A_{th}\in\mathbb R^{2\times K_a}$ 将一个 sensor 的几何步映射成该 hit 的两个白化残差分量变化；$B_{th}\in\mathbb R^{2\times5}$ 将所属轨迹的五维步映射到同样的残差空间。只有输出空间相同，二者才可以相加。

### 3.2 线性方程、正定性和秩

复现时最常见的任务是解 $H\mathbf z=\mathbf b$。记作 $H^{-1}\mathbf b$ 只是数学表达，实现应使用线性求解，不必形成显式逆。

对实对称矩阵，若任意非零 $\mathbf z$ 均满足 $\mathbf z^\mathsf TH\mathbf z>0$，则 $H$ 正定，解唯一，并可用 Cholesky 分解 $H=LL^\mathsf T$ 做两次三角求解。最小二乘中的 $J^\mathsf TJ$ 总是半正定；只有 $J$ 满列秩时才正定。

秩回答“多少个参数组合被独立约束”。很小的特征值或奇异值说明某些组合的约束很弱；阻尼能让数值步更稳，却不能因此证明数据增加了信息。后文会分别检查局部径迹块和全局约化几何块。

### 3.3 分块与稀疏性的含义

一个 hit 直接依赖一个 sensor 和一条轨迹，很多其他参数对它没有直接作用，因此 Jacobian 和正规方程有大量结构零。分块矩阵保留这种依赖关系；稀疏装配只存实际存在的块，不改变所解的方程。

消去一个共享轨迹会在它接触的 sensor 之间产生耦合。直接 Jacobian 中“没有共同出现的几何参数”，并不意味着最终几何系统对角。这是后文 Schur 补必须保留跨 sensor 块的原因。

## 4. 怎样用参数描述传感器几何

### 4.1 局部坐标与全局坐标

设 $\mathbf q$ 为相对面中心的局部位置，$\mathbf x$ 为全局位置，$R$ 为局部到全局旋转，则

$$
\mathbf x=\mathbf c+R\mathbf q,\qquad
\mathbf q=R^\mathsf T(\mathbf x-\mathbf c),\qquad
R^\mathsf TR=I,\quad\det R=1.
\tag{G-01}
$$

反向使用转置，是因为旋转矩阵正交。传感器只测量局部前两个坐标时，

$$
P_{\rm loc}=\begin{pmatrix}1&0&0\\0&1&0\end{pmatrix},\qquad
\widehat{\mathbf m}=P_{\rm loc}R^\mathsf T(\mathbf x-\mathbf c),\qquad
\mathbf n=R\mathbf e_z,\quad\mathbf e_z=(0,0,1)^\mathsf T.
\tag{G-02}
$$

$\mathbf n$ 为全局系中的平面法向。几何改变会改变局部坐标基以及轨迹交点，所以预测的变化不能仅理解为“从残差中减去一段平移”。

### 4.2 累计局部刚体修正

将传感器修正分成局部平移 $\mathbf d_i=(d_{ix},d_{iy},d_{iz})^\mathsf T$ 与转角 $\boldsymbol\varphi_i=(\varphi_{ix},\varphi_{iy},\varphi_{iz})^\mathsf T$，规定

$$
\mathbf c_i=\mathbf c_i^{(0)}+R_i^{(0)}\mathbf d_i,\qquad
R_i=R_i^{(0)}Q(\boldsymbol\varphi_i),\qquad
Q=R_z(\varphi_{iz})R_y(\varphi_{iy})R_x(\varphi_{ix}).
\tag{G-03}
$$

为复现相同几何，旋转顺序和右手符号必须一致：

$$
\begin{aligned}
R_x(u)&=\begin{pmatrix}1&0&0\\0&\cos u&-\sin u\\0&\sin u&\cos u\end{pmatrix},\\
R_y(u)&=\begin{pmatrix}\cos u&0&\sin u\\0&1&0\\-\sin u&0&\cos u\end{pmatrix},\\
R_z(u)&=\begin{pmatrix}\cos u&-\sin u&0\\\sin u&\cos u&0\\0&0&1\end{pmatrix}.
\end{aligned}
\tag{G-04}
$$

这些是有限旋转的前向计算公式，直接实现即可，具体导数交给 autograd。无需先手推 $L_x,L_y,L_z$ 或小角度旋转 Jacobian。

累计参数始终相对输入工作几何定义。平移使用固定 $R_i^{(0)}$；更新 Euler 参数后按式（G-03）重新计算整个旋转矩阵。对一般非零累计转角，不能任意改成“每步在当前矩阵后乘三个增量旋转”并声称模型未变。

### 4.3 活动自由度和无量纲求解坐标

六个刚体自由度不必全部估计。默认只选 $t_x,t_y,r_z$，用有序列表 $\mathcal D=(0,1,5)$ 表示。定义 $E_a\in\mathbb R^{6\times K_a}$：

$$
(E_a)_{dq}=\mathbf1[d=\mathcal D_q]\,s_d,\qquad
\mathbf s=(1\,{\rm mm},1\,{\rm mm},1\,{\rm mm},
0.01\,{\rm rad},0.01\,{\rm rad},0.01\,{\rm rad}),
\qquad
\Delta\mathbf a_i=E_a\Delta\mathbf u_i.
\tag{G-05}
$$

$\Delta\mathbf u_i$ 无量纲。默认 $\Delta\mathbf u_i=(0.02,-0.03,0.1)$ 对应局部平移 $20,-30\ \mu{\rm m}$ 和转角 $1\ {\rm mrad}$，其他分量增量为零。

$E_a$ 同时承担自由度选择和单位尺度转换。选择自由度改变允许估计的模型范围；在同一活动范围内一致重标度变量、Jacobian 和约束，是对同一问题换用数值坐标。尺度不是测量分辨率、物理先验或最大步长，数值重标度也不自动保证条件数改善。

## 5. 怎样描述径迹，并去掉参数冗余

### 5.1 单螺旋线模型

一条径迹的六维物理表示取为

$$
\boldsymbol\eta_t=(x_{0t},y_{0t},z_{0t},\psi_{0t},\tau_t,\kappa_t)^\mathsf T,
\qquad
\psi_t(s)=\psi_{0t}+\kappa_ts.
\tag{H-01}
$$

前三项是参考点，$\psi_0$ 是参考点切线在 $xy$ 平面的方位角，$\tau$ 是纵向位移相对于横向弧长的斜率，$\kappa$ 是有符号横向曲率。参考点不是强制的 perigee，也不要求位于束线。

令 $\operatorname{sinc}u=\sin u/u$，零点取连续极限 1，则全局位置为

$$
\mathbf x_t(s)=
\begin{pmatrix}
x_{0t}+s\cos(\psi_{0t}+\kappa_ts/2)\operatorname{sinc}(\kappa_ts/2)\\
y_{0t}+s\sin(\psi_{0t}+\kappa_ts/2)\operatorname{sinc}(\kappa_ts/2)\\
z_{0t}+\tau_ts
\end{pmatrix}.
\tag{H-02}
$$

这是稳定的前向参数式：零曲率时连续退化成直线，不需要计算不稳定的 $0/0$。若所用库定义 $\operatorname{sinc}_{\rm lib}u=\sin(\pi u)/(\pi u)$，调用时需将自变量除以 $\pi$。

曲线的路径方向可直接写成 $\mathbf v_t(s)=(\cos\psi_t(s),\sin\psi_t(s),\tau_t)^\mathsf T$。其模为 $\sqrt{1+\tau_t^2}$，所以 $s$ 是横向弧长，不是三维弧长。曲率直接从数据拟合，不需要由电荷、磁场或上游 $q/p$ 换算。

### 5.2 为什么六个物理参数只用五个优化变量

把参考点沿同一条 helix 移动 $s_0$，并同步改变起始方位角，会得到同一条空间曲线：

$$
\mathbf x_0'=\mathbf x(s_0),\qquad
\psi_0'=\psi_0+\kappa s_0,\qquad
\tau'=\tau,\quad\kappa'=\kappa,\qquad
\mathbf x'(s)=\mathbf x(s+s_0).
\tag{H-03}
$$

观测只能约束曲线，不能区分这些等价起点。如果把六个表示参数全部自由求解，就有一个与物理曲线无关的冗余方向。gauge 固定的目的，是为同一条曲线选择一个局部唯一的参数表示。

### 5.3 构造五参数参考帧

取参考原点 $\mathbf o=\mathbf x_0$，将当前三维路径方向归一化为 $\widehat{\mathbf v}$，在其法平面上构造两个正交轴：

$$
\widehat{\mathbf v}=\frac{\mathbf v(0)}{\|\mathbf v(0)\|},\qquad
\mathbf e_0=\frac{\mathbf h_{\rm aux}\times\widehat{\mathbf v}}
{\|\mathbf h_{\rm aux}\times\widehat{\mathbf v}\|},\qquad
\mathbf e_1=\widehat{\mathbf v}\times\mathbf e_0.
\tag{H-04}
$$

辅助轴通常取全局 $z$，接近平行时换全局 $x$。这个参考平面垂直于三维轨迹方向，不是全局 $xy$ 平面。

使用五维坐标并解包成物理 helix：

$$
\begin{aligned}
\boldsymbol\theta_t&=(o_{0t},o_{1t},\psi_{0t},\tau_t,\widetilde\kappa_t)^\mathsf T,
&\widetilde\kappa_t&=\kappa_t/\kappa_{\rm scale},\quad
\kappa_{\rm scale}=10^{-4}\ {\rm mm}^{-1},\\
\mathbf x_{0t}&=\mathbf o_t+o_{0t}\mathbf e_{0t}+o_{1t}\mathbf e_{1t},
&\kappa_t&=\kappa_{\rm scale}\widetilde\kappa_t.
\end{aligned}
\tag{H-05}
$$

打包当前物理轨迹时两个偏移初值为零，其他三个坐标从物理参数继承。偏移限制了参考点沿轨迹方向的冗余运动，保留五个局部可观测坐标。

seed 和 refit 各自在一次拟合内固定其起始帧；联合优化每次线性化都从当前物理 helix 重建帧。五维步只在产生它的帧里使用，提交后回到物理六维表示。把旧帧中的五维向量直接带到新帧，会改变它所代表的物理轨迹。

## 6. 怎样求最近点和平面交点

### 6.1 数值求根的共同原理

对于一维方程 $f(s)=0$，Newton 法利用当前点附近的线性近似产生新路径：

$$
s_{\rm cand}^{(n+1)}
=s^{(n)}-\frac{f(s^{(n)})}{f'(s^{(n)})}.
\tag{N-01}
$$

这里 $f'$ 可由 autograd 计算，不需要推导具体解析式。需要理解的是：分母太小、初值太远或分支改变时，Newton 可能失效，所以“写出迭代公式”还不足以定义可靠的求交算法。

本算法使用固定次数的迭代与明确的有效性检查。实际数值路径记为 $\widehat s$，理想精确根另记为 $s^\star$ 或 $s^\cap$；二者不能在未检查数值误差时直接等同。

### 6.2 seed 中的最近点问题

给定非目标 SMT 点 $\mathbf p_{th}^{\rm SMT}$，寻找 helix 上靠近它的位置。定义距离目标

$$
f_{{\rm cp},th}(s)=\tfrac12\|\mathbf x_t(s)-\mathbf p_{th}^{\rm SMT}\|^2.
\tag{N-02}
$$

局部极小点的一维必要条件是该目标对 $s$ 的一阶导数为零。复现时可让 autograd 返回 $f_{\rm cp}'$ 和 $f_{\rm cp}''$，然后计算

$$
s_{\rm cand}^{(n+1)}
=s^{(n)}
-\frac{f_{\rm cp}'(s^{(n)})}
{\operatorname{safe}_{10^{-9}}(f_{\rm cp}''(s^{(n)}))},
\quad
s^{(0)}=(p_x^{\rm SMT}-x_0)\cos\psi_0+(p_y^{\rm SMT}-y_0)\sin\psi_0.
\tag{N-03}
$$

$\operatorname{safe}_\epsilon(z)$ 在 $|z|>\epsilon$ 时取 $z$，否则取同号的 $\epsilon$；零按正号。初值是 $xy$ 投影，不是三维直线最小二乘投影。默认执行 8 次。

最近点函数没有保证全局最小，也不在终态另检查二阶导数为正或一阶导数达到指定容差。因此它严格说返回最近点的数值候选；可用性还取决于分支和数据范围。

### 6.3 测量预测中的平面交点

给定平面中心 $\mathbf c_i$、法向 $\mathbf n_i$，求满足下式的路径：

$$
\Phi_{{\rm plane},th}(s)
=\mathbf n_i^\mathsf T[\mathbf x_t(s)-\mathbf c_i]=0.
\tag{N-04}
$$

这里不最小化到某个三维点的距离，而是要求轨迹落在平面上。先用参考点处直线方向 $\mathbf v_t(0)$ 估计路径，再执行 Newton：

$$
s^{(0)}
=\frac{\mathbf n_i^\mathsf T(\mathbf c_i-\mathbf x_{0t})}
{\mathbf n_i^\mathsf T\mathbf v_t(0)},
\qquad
s_{\rm cand}^{(n+1)}
=s^{(n)}
-\frac{\Phi_{\rm plane}(s^{(n)})}
{\operatorname{safe}_{10^{-6}}(\Phi_{\rm plane}'(s^{(n)}))}.
\tag{N-05}
$$

导数由 autograd 提供；默认执行 12 次。直线初值并未把整个模型改成直线，后续每次评估仍使用式（H-02）的 helix。

最终局部预测为

$$
\widehat{\mathbf m}_{th}(\mathbf a_i,\boldsymbol\eta_t)
=P_{\rm loc}R_i(\mathbf a_i)^\mathsf T
\left[\mathbf x_t(\widehat s_{{\rm plane},th};\boldsymbol\eta_t)
-\mathbf c_i(\mathbf a_i)\right].
\tag{N-06}
$$

这就是后续最小二乘真正需要的前向模型。几何转角进入旋转和法向；法向影响路径，路径又影响交点。autograd 必须看到整条依赖链。

### 6.4 有效性是模型的一部分

两类路径候选都要求有限且绝对值不超过 1200 mm；不满足时保留上一数值路径，并将累计有效标志设为假。一次路径调用内，失效后不复活：

$$
v^{(n+1)}=v^{(n)}\land v_{\rm cand}^{(n+1)},\qquad
s^{(n+1)}=
\begin{cases}s_{\rm cand}^{(n+1)},&v^{(n+1)},\\s^{(n)},&\text{否则}.\end{cases}
\tag{N-07}
$$

平面求交另检查初始及迭代切向不近平行，终态平面残差不超过 $10^{-3}$ mm。无效结果可以返回有限占位，但占位不参与测量目标。最近点与平面求交的完整复现门限见附录 A。

自动微分计算的是有限次迭代程序的导数。在根已收敛、所选分支稳定且数值保护不主导时，它才能解释为理想几何约束附近的稳定敏感度。布尔有效性切换点不具有同样的光滑解释。

## 7. 怎样把测量偏差写成统计目标

### 7.1 方差决定权重

局部测量和预测之差记为

$$
e_{th\ell}=m_{th\ell}-\widehat m_{th\ell},\qquad
r_{th\ell}=\frac{e_{th\ell}}{\sqrt{V_{th\ell}^{\rm eff}}}.
\tag{W-01}
$$

$e$ 的单位为 mm；$V$ 为 mm²；白化后的 $r$ 无量纲。同样的 $0.1$ mm 偏差，对标准差 $0.01$ mm 的测量意味着 10 个标准差，对标准差 $1$ mm 的测量只意味着 0.1 个标准差，权重应不同。

若误差是独立零均值高斯，固定方差下的负对数似然除去常数项就是

$$
F=\tfrac12\sum r_{th\ell}^2
=\tfrac12\sum\frac{e_{th\ell}^2}{V_{th\ell}^{\rm eff}}.
\tag{W-02}
$$

这说明为什么采用白化最小二乘；它不证明真实误差一定高斯，也不自动补偿材料效应或模型误差。

复现时使用三个层次的方差：

$$
\begin{aligned}
V_{th\ell}^{\rm raw}&=\max(\sigma_{th\ell}^2,10^{-12}\ {\rm mm}^2),\\
V_{th\ell}&=
\begin{cases}
\max(V_{th1}^{\rm raw},1.44\ {\rm mm}^2),&\ell=1,\ \text{有效二维 strip 测量},\\
V_{th\ell}^{\rm raw},&\text{其他有效分量},
\end{cases}\\
V_{th\ell}^{\rm eff}&=\max(V_{th\ell},10^{-8}\ {\rm mm}^2).
\end{aligned}
\tag{W-03}
$$

原始诊断方差也已经过第一个数值保护。求解使用施加下限后的权重，且全程固定；不额外加入一份 track 预测方差再进行相同的 Schur 消元。

### 7.2 一维测量、padding 和分量集合

定义 $b_{th}^{\rm hit}$ 表示真实 hit，$b_{th}^{1}$ 表示 LOC1 测量存在，$v_{th}^{\rm plane}$ 表示求交有效。两个分量的有效性为

$$
w_{th0}=b_{th}^{\rm hit}\land v_{th}^{\rm plane},\qquad
w_{th1}=b_{th}^{\rm hit}\land b_{th}^{1}\land v_{th}^{\rm plane}.
\tag{W-04}
$$

无效分量的残差与 Jacobian 均置零。LOC1 缺失时可以准备数值占位，但不能把占位当作一个真实的零残差测量。补齐后的张量宽度也不是实际测量数。

联合外迭代在线性化点得到有效集合 $\mathcal I_k$，本轮所有阻尼试探都使用这个集合比较目标。冻结集合中的分量若在候选处消失，应拒步，而不是通过少计算几个坏残差降低目标。

### 7.3 三个目标及归一化的区别

| 阶段 | 残差 | 参与样本 | 目标单位 |
|---|---|---|---|
| 点 seed | helix 最近点预测减 SMT 三维点 | 非目标点 | mm² |
| 测量 refit | 白化的局部测量减预测 | 非目标测量 | 无量纲 |
| 联合校准 | 白化的局部测量减预测 | 参考与目标测量 | 无量纲 |

优化和驻点判断使用未归一的半平方和。报告里的 $\chi^2$ 每分量均值为 $2F/N_{\rm comp}$，没有扣除拟合参数数目，因此不等于 $\chi^2/{\rm NDF}$。

固定样本数时，将整个目标乘常数不改变最小点，却会改变梯度和 decrement 的数值口径。复现已有停止门限时必须使用同样的目标归一化。

## 8. autograd 已知时，仍需明确什么

### 8.1 定义可微的残差函数

autograd 的任务是从前向程序返回 Jacobian。联合线性化时，先由当前物理轨迹建立固定 gauge 帧 $\mathcal G_t^{(k)}$，再定义一个单 hit 函数

$$
\mathcal R_{th}(\mathbf u,\boldsymbol\theta)=
\operatorname{WhitenAndMask}\!\left(
\mathbf m_{th}
-\widehat{\mathbf m}_{th}
\left(\mathbf a_i^{(k)}+E_a\mathbf u,\
\operatorname{unpack}_{\mathcal G_t^{(k)}}(\boldsymbol\theta)\right)
\right).
\tag{AD-01}
$$

当前点对应 $\mathbf u=\mathbf0$、
$\boldsymbol\theta=\operatorname{pack}(\boldsymbol\eta_t^{(k)})$。几何基准、测量、方差、frame 和当前冻结 mask 在这次微分中作为固定数据。

通过 autograd 定义

$$
A_{th}=\operatorname{Jac}_{\mathbf u}\mathcal R_{th}
\in\mathbb R^{2\times K_a},\qquad
B_{th}=\operatorname{Jac}_{\boldsymbol\theta}\mathcal R_{th}
\in\mathbb R^{2\times5}.
\tag{AD-02}
$$

这里不展开任何旋转或求交的导数分量。需要复现的是函数本身、求导变量、数值尺度和固定条件。参考面不提供可调几何，因此参考行的 $A=0$；参考和目标行都可提供 $B$。

### 8.2 需要提供的自动微分接口

| 数学对象 | 输入变量 | 自动微分输出 | 用途 |
|---|---|---|---|
| 最近点一维距离目标 $f_{\rm cp}$ | 路径 $s$ | 一、二阶标量导数 | Newton 最近点更新 |
| 平面约束 $\Phi_{\rm plane}$ | 路径 $s$ | 一阶标量导数 | Newton 求交更新 |
| 展平的三维 seed 残差 | 五维 $\boldsymbol\theta_t$ | $3N_{\rm hit}\times5$ Jacobian | 点 seed |
| 展平的非目标白化残差 | 五维 $\boldsymbol\theta_t$ | $2N_{\rm hit}\times5$ Jacobian | 测量 refit |
| 单 hit 联合白化残差 | $\mathbf u_i,\boldsymbol\theta_t$ | $A_{th},B_{th}$ | 联合正规方程 |

求完整 Jacobian 后，后面的步骤都是矩阵乘法、线性求解和状态控制，不需要用 autograd 反向穿过整个 LM 迭代。

### 8.3 计算图与边界条件

最近点或交点路径依赖轨迹、几何参数，需保留这些依赖。把路径转成普通 Python 浮点数、在内部使用不可微的数组库，或在求交后人为断开其图，都会改变最终 Jacobian。若用嵌套 autograd 计算路径更新中的导数，还须保留外层 Jacobian 所需的高阶计算图。

有效性判定和根分支选择按当前分支处理，不对布尔比较求“平滑导数”。这意味着 Jacobian 是当前有效分支上的局部模型；候选步是否越过有效域，由后续真实前向评估决定。

每次扰动验证应重新计算完整路径和残差。只比较前向与反向两种 autograd 模式，无法排除它们共享同一个被错误断开的计算图。

## 9. 怎样先把每条径迹拟合好

### 9.1 Gauss–Newton 的矩阵级原理

先考虑一条径迹。设其残差向量为 $\mathbf f_t(\boldsymbol\theta_t)$，可以是三维 seed 残差，也可以是白化测量残差。autograd 给出 $J_t$ 后，在当前点作局部线性化：

$$
\mathbf f_t(\boldsymbol\theta_t+\Delta\boldsymbol\theta_t)
\simeq\mathbf f_t+J_t\Delta\boldsymbol\theta_t.
\tag{GN-01}
$$

把它代入半平方和，展开得到

$$
\tfrac12\|\mathbf f_t+J_t\Delta\boldsymbol\theta_t\|^2
=F_t+\mathbf g_t^\mathsf T\Delta\boldsymbol\theta_t
+\tfrac12\Delta\boldsymbol\theta_t^\mathsf TH_t\Delta\boldsymbol\theta_t,
\quad
F_t=\tfrac12\|\mathbf f_t\|^2,\qquad
\mathbf g_t=J_t^\mathsf T\mathbf f_t,\quad
H_t=J_t^\mathsf TJ_t.
\tag{GN-02}
$$

该二次模型的最小点由正规方程 $H_t\Delta\boldsymbol\theta_t=-\mathbf g_t$ 给出。GN 的近似是先线性化残差再求最小二乘；它没有保留残差自身的全部二阶非线性效应。小残差或局部近线性时，该模型通常更可信。

为避免一次走得过远，初始化阶段使用

$$
\omega_t=\max\{\operatorname{tr}(H_t)/5,10^{-30}\},\qquad
(H_t+\mu_t\omega_t I_5)\Delta\boldsymbol\theta_t=-\mathbf g_t.
\tag{GN-03}
$$

这里的五维数值坐标包含位置、角度、斜率和缩放曲率；平均对角尺度是数值约定，不是参数化无关的物理先验。

下面两个初始化阶段都使用这类阻尼步，但停止与有效性规则不同，复现时不能合并成一段相同控制逻辑。

### 9.2 第一步：非目标 SMT 点 seed

先选择每条径迹的首个非目标点，以其 SMT 位置和方向建立初值：

$$
\mathbf x_{0t}^{\rm init}=\mathbf p_{t,h_0}^{\rm SMT},\quad
\psi_{0t}^{\rm init}=\operatorname{atan2}(p_y^{\rm mom},p_x^{\rm mom}),\quad
\tau_t^{\rm init}=
\frac{p_z^{\rm mom}}{\max(\sqrt{(p_x^{\rm mom})^2+(p_y^{\rm mom})^2},10^{-6})},
\quad
\kappa_t^{\rm init}=0.
\tag{I-01}
$$

动量方向仅用来定初值；曲率随后由拟合估计。对有效非目标点定义

$$
\mathbf d_{th}^{\rm seed}
=\mathbf x_t(\widehat s_{{\rm cp},th})-\mathbf p_{th}^{\rm SMT},\qquad
F_{{\rm seed},t}=\tfrac12\sum_h\|\mathbf d_{th}^{\rm seed}\|^2.
\tag{I-02}
$$

无效或非拟合点的残差置零。固定本次 seed 起始 gauge 帧，autograd 计算展平残差的 Jacobian，用式（GN-03）更新。每条径迹独立控制阻尼：初值 $10^{-8}$，有效且有限、目标严格下降则接受并乘 0.1（下限 $10^{-12}$）；否则保留状态并乘 10。

记录初始有效点集合 $\mathcal E_t^{(0)}$，要求每个接受候选仍包含该集合。初始无效点可以后来变有效；这不是把整个拟合中曾有效的所有点永久冻结。

seed 用微阻尼 decrement 判断每行是否还需更新：

$$
\nu_{{\rm seed},t}
=\mathbf g_t^\mathsf T(H_t+10^{-12}\omega_tI_5)^{-1}\mathbf g_t,\qquad
\nu_{{\rm seed},t}^{\rm rel}
=\frac{\nu_{{\rm seed},t}}{\max(F_{{\rm seed},t},10^{-12})}.
\tag{I-03}
$$

decrement 有限，且绝对量不超过 $10^{-12}$ 或相对量不超过 $10^{-6}$ 时，该行退出更新。绝对量随 seed 目标为 mm²，不能解释为白化后的统计精度。预算默认 2000 次，最后仍需更新返回状态的诊断。

### 9.3 第二步：非目标局部测量 refit

将 seed 物理参数作为初值，保留其曲率，在固定非目标几何上最小化

$$
F_{{\rm ref},t}
=\tfrac12\sum_{h,\ell:\ {\rm ref\ valid}}
\frac{(m_{th\ell}-\widehat m_{th\ell})^2}{V_{th\ell}^{\rm eff}}.
\tag{I-04}
$$

本次 refit 起始 gauge 帧固定。仍用式（GN-03）的阻尼形式及相同初始阻尼/倍率，但默认最多 60 次，不逐行退休。接受条件是候选目标严格下降；循环退出条件是所有行的相对目标下降都很小：

$$
\zeta_t=
\frac{\max(F_{{\rm ref},t}-F_{{\rm ref},t}^{\rm cand},0)}
{\max(|F_{{\rm ref},t}|,1)},\qquad
\max_t\zeta_t<10^{-6}.
\tag{I-05}
$$

所有候选均不下降，也可能满足这一退出条件，所以退出后必须另做驻点检查。refit 在候选处重新判断求交有效性，没有联合 LM 那样逐试探冻结分量的保护。其末端检查不能替代或被描述为逐候选分量守恒。

### 9.4 初始化的秩与驻点门

测量 refit 起点和最终驻点计算均检查 Jacobian 有限、有效分量数至少 5、未正则化 Jacobian 数值秩为 5。数值秩按默认阈值 $\max(m,n)\epsilon_{\rm machine}\sigma_{\max}$ 判断，其中 $m=2N_{\rm hit},n=5$，含补齐的零行。

返回物理轨迹后重建 gauge 帧，重新计算

$$
\nu_{{\rm ref},t}
=(\mathbf g_t^{\rm ref})^\mathsf T
(H_t^{\rm ref}+10^{-12}\omega_t^{\rm ref}I_5)^{-1}\mathbf g_t^{\rm ref}.
\tag{I-06}
$$

这是未按分量数归一的白化半平方和口径。两个初始化阶段都要求非有限 decrement 行数为 0、非驻点行比例不超过 $10^{-3}$；seed 按式（I-03）判断，refit 按 $\nu_{{\rm ref},t}>10^{-6}$ 判断。关闭额外驻点报告不关闭这一生产验收门。

初始化显式排除目标项，但 SMT 来自上游重建，不能推断它与目标测量在统计上完全独立。联合校准开始后，轨迹继续作为未知量参与更新，refit 结果只是初始状态。

## 10. 怎样把几何和径迹放进同一个最小二乘问题

### 10.1 一条测量同时依赖两类未知量

对当前有效的 hit，autograd 给出 $A_{th}$、$B_{th}$ 后，局部线性模型是

$$
\mathbf r_{th}^{+}\simeq
\mathbf r_{th}
+A_{th}\Delta\mathbf u_{i(t,h)}
+B_{th}\Delta\boldsymbol\theta_t.
\tag{J-01}
$$

参考面的几何固定，故其 $A=0$，但它仍通过 $B$ 约束轨迹。目标测量同时约束几何和轨迹。把这些行全部堆叠，记 $\boldsymbol\delta=(\Delta\mathbf u,\Delta\boldsymbol\theta)$，则

$$
L_k(\boldsymbol\delta)
=F_{\rm joint}^{(k)}+\mathbf g^\mathsf T\boldsymbol\delta
+\tfrac12\boldsymbol\delta^\mathsf TH\boldsymbol\delta,
\qquad
\mathbf g=\mathcal J^\mathsf T\mathbf r,\quad H=\mathcal J^\mathsf T\mathcal J.
\tag{J-02}
$$

$\mathcal J$ 是联合 Jacobian，$H$ 在全文联合部分表示 GN 信息矩阵，不表示精确 Newton Hessian。这里开始只使用 autograd 已返回的矩阵，不再展开具体求导公式。

### 10.2 逐轨迹、逐 sensor 累加

设 $\mathcal H_{ti}$ 为轨迹 $t$ 在目标 sensor $i$ 上的 hit 集合，定义

$$
C_{ti}=\sum_{h\in\mathcal H_{ti}}A_{th}^\mathsf TA_{th},\quad
G_{ti}=\sum_{h\in\mathcal H_{ti}}A_{th}^\mathsf TB_{th},\quad
\mathbf g_{a,ti}=\sum_{h\in\mathcal H_{ti}}A_{th}^\mathsf T\mathbf r_{th}.
\tag{J-03}
$$

其维度依次是 $K_a\times K_a$、$K_a\times5$、$K_a$。几何块和梯度按轨迹求和：
$C_i=\sum_t C_{ti}$，$\mathbf g_{a,i}=\sum_t\mathbf g_{a,ti}$。

径迹块必须包含参考与目标的全部测量：

$$
\begin{aligned}
\Gamma_t&=\Gamma_t^{\rm ref}+\Gamma_t^{\rm tgt},
&\Gamma_t^{\rm ref/tgt}&=\sum_{h\in\mathcal H_t^{\rm ref/tgt}}B_{th}^\mathsf TB_{th},\\
\boldsymbol\beta_t&=\boldsymbol\beta_t^{\rm ref}+\boldsymbol\beta_t^{\rm tgt},
&\boldsymbol\beta_t^{\rm ref/tgt}&=\sum_{h\in\mathcal H_t^{\rm ref/tgt}}B_{th}^\mathsf T\mathbf r_{th}.
\end{aligned}
\tag{J-04}
$$

$\Gamma_t$ 为 $5\times5$，$\boldsymbol\beta_t$ 为 5 维。目标行既然留在目标函数中，它们的径迹信息也必须留在这里；只使用参考块会改变所解问题。

### 10.3 联合正规方程与阻尼

将几何块组成 $C=\operatorname{blockdiag}_i C_i$，径迹块组成 $\Gamma=\operatorname{blockdiag}_t\Gamma_t$，交叉矩阵 $G$ 的 $(i,t)$ 块为 $G_{ti}$。正规方程为

$$
\begin{pmatrix}C&G\\G^\mathsf T&\Gamma\end{pmatrix}
\begin{pmatrix}\Delta\mathbf u\\\Delta\boldsymbol\theta\end{pmatrix}
=-\begin{pmatrix}\mathbf g_a\\\boldsymbol\beta\end{pmatrix}.
\tag{J-05}
$$

联合 LM 采用按信息对角缩放的 Marquardt 阻尼：

$$
\begin{aligned}
D_a&=\operatorname{diag}(\operatorname{diag}C),&
D_\theta&=\operatorname{diag}(\operatorname{diag}\Gamma),\\
C_\mu&=C+\mu_{\rm LM}D_a,&
\Gamma_\mu&=\Gamma+\mu_{\rm LM}D_\theta.
\end{aligned}
\tag{J-06}
$$

在式（J-05）中用 $C_\mu,\Gamma_\mu$ 替换两块对角，即得到试探步方程。几何和径迹同时阻尼；只阻尼几何会允许轨迹继续吸收残差变化，具有不同的极限行为。

该阻尼形式不同于初始化的 $\omega_t I_5$。阻尼是数值步的控制机制，不向原测量目标增加一项物理先验损失。结构性无信息方向先删除，具体处理见第 13 节。

## 11. 为什么用 Schur 补，以及怎样消去径迹变量

### 11.1 从第二行消元，再求几何

大量径迹只通过共享传感器几何相互联系，而 $\Gamma_\mu$ 对径迹是块对角。可以先解每条轨迹方程：

$$
\Delta\boldsymbol\theta_t
=-\Gamma_{t,\mu}^{-1}
\left(\boldsymbol\beta_t+\sum_iG_{ti}^\mathsf T\Delta\mathbf u_i\right).
\tag{S-01}
$$

将它代入几何方程：

$$
C_\mu\Delta\mathbf u
-G\Gamma_\mu^{-1}
(\boldsymbol\beta+G^\mathsf T\Delta\mathbf u)
=-\mathbf g_a.
\tag{S-02}
$$

把几何步的系数和已知项分组，得到

$$
\boxed{
S_\mu\Delta\mathbf u=-\mathbf b_\mu,\qquad
S_\mu=C_\mu-G\Gamma_\mu^{-1}G^\mathsf T,\qquad
\mathbf b_\mu=\mathbf g_a-G\Gamma_\mu^{-1}\boldsymbol\beta
}
\tag{S-03}
$$

$S_\mu$ 就是 Schur 补。先解这个几何系统，再按式（S-01）回代全部径迹步。在相应局部块可逆时，这与直接解同一个联合线性方程等价；没有另外冻结轨迹，也没有额外引入新的物理近似。

### 11.2 跨传感器耦合来自哪里

逐块写出约化系统：

$$
(S_\mu)_{ij}
=\mathbf1[i=j](C_i+\mu_{\rm LM}D_{a,i})
-\sum_{t:\ i,j\text{均被}t\text{触及}}
G_{ti}\Gamma_{t,\mu}^{-1}G_{tj}^\mathsf T.
\tag{S-04}
$$

原来的几何块 $C$ 是块对角，消元后的 $S_\mu$ 一般不是。一个一维示意问题能看清原因：两个几何参数共享一个轨迹参数，直接信息为

$$
H_{\rm demo}=
\begin{pmatrix}
C_1&0&G_1\\
0&C_2&G_2\\
G_1&G_2&\gamma
\end{pmatrix},
\qquad
(S_{\rm demo})_{12}=-G_1G_2/\gamma.
\tag{S-05}
$$

这里各项是标量，仅用于说明结构。移动第一个 sensor 会改变共享轨迹的最优变化，继而影响第二个 sensor，因此二者不能分别独立求解。生产问题保留全部共触及块，不做“只留对角”的替换。

### 11.3 用分解求解，不形成显式逆

对每个 $5\times5$ 块做 Cholesky，分别求出
$\mathbf z_t^\beta=\Gamma_{t,\mu}^{-1}\boldsymbol\beta_t$、
$Y_{ti}=\Gamma_{t,\mu}^{-1}G_{ti}^\mathsf T$，即可累加式（S-03）并回代。无阻尼块也必须能通过所要求的正定分解；失败要报告，不静默换伪逆。

将每个约化 $K_a\times K_a$ 块展开到标量索引 $iK_a+p,jK_a+q$ 后构造 CSC，使用稀疏 LU 解几何系统。sensor 编号在这里指参与求解目标面中的连续块编号，应与全几何索引保持明确映射。

对小合成问题可以直接构造完整稠密联合系统，与 Schur 几何步和回代径迹步比较，这是非常有力的独立验证。大问题不需要构造完整稠密联合矩阵。

### 11.4 信息矩阵的统计解释需要条件

在线性/局部 GN、正确高斯权重、无活动约束和可辨识的条件下，无阻尼信息矩阵的逆可解释为局部协方差近似。令 $\Gamma^{\rm ref}=\operatorname{blockdiag}_t\Gamma_t^{\rm ref}$。固定几何时的 $\Gamma^{-1}$、只用参考测量的 $(\Gamma^{\rm ref})^{-1}$ 和联合估计后的径迹边缘协方差，是不同对象。

复现求解流程不要求先实现完整协方差输出或 Woodbury 恒等式。需要知道的是：Schur 已经处理了轨迹变化对几何信息的影响，不能把额外预测误差重复计入权重，也不能把带阻尼逆矩阵直接当作物理统计误差。

## 12. 怎样判断非线性试探步是否可信：联合 LM

### 12.1 阻尼选步，真实前向计算决定接受

当前局部模型记为式（J-02）的 $L_k$。先通过阻尼联合系统产生步，再按第 13 节缩放到可行域。实际执行步是

$$
\boldsymbol\delta_{\rm exec}
=\alpha_{\rm step}(\Delta\mathbf u,\Delta\boldsymbol\theta).
\tag{L-01}
$$

对这个实际步，预测下降由无阻尼模型给出：

$$
\operatorname{pred}
=-\mathbf g^\mathsf T\boldsymbol\delta_{\rm exec}
-\tfrac12\boldsymbol\delta_{\rm exec}^\mathsf TH\boldsymbol\delta_{\rm exec}.
\tag{L-02}
$$

计算时使用各块矩阵乘积即可，不必拼出完整 $H$。若先将步缩小再执行，却仍用未缩小的步算预测下降，增益比就失去了可比性。

重新求候选几何与轨迹的交点和残差，在当前冻结集合 $\mathcal I_k$ 上计算

$$
\operatorname{actual}
=F_{\rm joint}^{(k)}(\text{当前状态})
-F_{\rm joint}^{(k)}(\text{候选状态}),\qquad
\rho_{\rm LM}=\frac{\operatorname{actual}}{\operatorname{pred}}.
\tag{L-03}
$$

增益比衡量局部模型对真实下降的预测程度。残差为线性、集合固定且算术精确时，任意执行步都有 $\rho_{\rm LM}=1$。预测下降非正或非有限属于数值失败，不是普通拒步。

### 12.2 可比性与状态守恒

候选须保留 $\mathcal I_k$ 的全部分量，且冻结集合上的目标有限。分量失效不能靠置零取得下降。新变有效但不在 $\mathcal I_k$ 的分量，在本次比较中不加入；下一外迭代重新形成有效集合。

试探点执行的是已经解出的联合步，不在每个候选几何下额外进行一次“把轨迹完全拟合好”的内层优化。拒步时只改变阻尼和日志，保持当前几何、物理轨迹、Jacobian、权重与冻结集合不变；同一外迭代复用线性化。

接受后的新外迭代可以新增有效分量，因此下降保证针对每次固定集合的比较，不能无条件把跨集合的历史目标解释成同一个固定函数的单调序列。

### 12.3 确定的接受与阻尼规则

初始 $\mu_{\rm LM}=10^{-3}$，上下限为 $10^{-12},10^{12}$：

| 条件 | 决定 | 下一阻尼 |
|---|---|---|
| 候选无效、比值非有限，或 $\rho_{\rm LM}<10^{-4}$ | 拒绝 | $\min(10\mu_{\rm LM},10^{12})$ |
| $10^{-4}\le\rho_{\rm LM}<0.25$ | 接受 | $\min(10\mu_{\rm LM},10^{12})$ |
| $0.25\le\rho_{\rm LM}<0.75$ | 接受 | 保持 |
| $\rho_{\rm LM}\ge0.75$ | 接受 | $\max(0.1\mu_{\rm LM},10^{-12})$ |

每外迭代允许首次尝试加最多 8 次重试。阻尼已经在上限时仍被拒绝，或重试用尽，都结束该运行并报告原因。接受后保留物理状态，再从新状态重建下一轮 gauge 与线性化。

局部模型中的几何变化 $A\Delta\mathbf u$ 和径迹变化 $B\Delta\boldsymbol\theta$ 还含相互交叉作用。各自平方范数不能直接作为“几何误差占比”和“材料误差占比”，也不能忽略交叉项后相加解释总改进。

## 13. 怎样处理累计修正边界：KKT 与活动集

### 13.1 边界限制的是累计物理修正

默认平移半宽为 1.5 mm，转角半宽为 0.05 rad：

$$
|a_{i,\mathcal D_q}|\le B_q^{\rm box},\qquad
B_q^{\rm box}=
\begin{cases}1.5\ {\rm mm},&\mathcal D_q<3,\\0.05\ {\rm rad},&\mathcal D_q\ge3.\end{cases}
\tag{C-01}
$$

这些是相对工作几何的累计边界，径迹无箱约束。可行域和单次步长是两个概念；本算法没有每个自由度独立的固定最大步长裁剪。

对一个被约束变量，KKT 的直观含义是：内部最优点的梯度为零；下界最优点允许梯度指向“下降会越过下界”的方向；上界同理。轨迹无约束，联合驻点还要求径迹梯度接近零。

### 13.2 自由集合的具体判定

先构造无阻尼 Schur 系统。取某活动物理值 $x$、对应的无阻尼约化梯度分量 $b_0$ 和半宽 $B$，边界容差 $\epsilon_{\rm box}=10^{-12}$，定义

$$
\operatorname{free}_{\rm box}
=\neg\left[
(x\ge B-\epsilon_{\rm box}\land b_0<0)
\lor
(x\le-B+\epsilon_{\rm box}\land b_0>0)
\right].
\tag{C-02}
$$

下降方向取梯度反向，所以边界上继续指向域外的分量冻结。恰好零梯度不因符号冻结，仍留给秩检查。无量纲坐标与物理坐标之间的尺度为正，因此其梯度符号一致。

算法使用的是消去径迹增量后的二次模型约化梯度 $b_0$。当径迹梯度为零时，它也与原联合问题的几何梯度符号一致；在一般中间状态不应把两者无条件视为同一个数值。

此外删除结构性无信息的几何方向：

$$
\operatorname{free}_{{\rm struct},iq}=\mathbf1[(C_i)_{qq}>0],
\qquad
\mathcal F=\operatorname{free}_{\rm box}\land\operatorname{free}_{\rm struct}.
\tag{C-03}
$$

恰为零的信息方向不通过正则化伪装成可观测方向。很小但为正的方向继续进入第 14 节的谱检验。构造阻尼矩阵时，已删除方向的对角可以临时占位为 1；它随后被钉为零步，该值不影响自由解。

### 13.3 在自由系统求解，然后整体缩放

令 $\Pi_{\mathcal F}$ 为对角自由选择矩阵。可用以下固定大小系统实现“删除被冻结的 DOF”：

$$
S_{\mu,\rm pin}=
\Pi_{\mathcal F}S_\mu\Pi_{\mathcal F}+I-\Pi_{\mathcal F},
\qquad
\mathbf b_{\mu,\rm pin}=\Pi_{\mathcal F}\mathbf b_\mu.
\tag{C-04}
$$

解 $S_{\mu,\rm pin}\Delta\mathbf u=-\mathbf b_{\mu,\rm pin}$ 后，被冻结分量为零，自由部分与直接删除行列求解等价。再回代径迹步。几何自由集为空时，径迹仍可能需要移动。

对活动物理步 $\delta a_{iq}=s_{\mathcal D_q}\Delta u_{iq}$，计算到边界的最大比例：

$$
\alpha_{iq}^{\max}=
\begin{cases}
(B_q^{\rm box}-a_{i,\mathcal D_q})/\delta a_{iq},&\delta a_{iq}>0,\\
(-B_q^{\rm box}-a_{i,\mathcal D_q})/\delta a_{iq},&\delta a_{iq}<0,\\
+\infty,&\delta a_{iq}=0,
\end{cases}
\qquad
\alpha_{\rm step}=\min\left(1,\min_{i,q}\max(0,\alpha_{iq}^{\max})\right).
\tag{C-05}
$$

几何和径迹用同一个比例：

$$
\begin{aligned}
\mathbf a_i^+&=\mathbf a_i+\alpha_{\rm step}E_a\Delta\mathbf u_i,\\
\boldsymbol\eta_t^+
&=\operatorname{unpack}_{\mathcal G_t^{(k)}}
\left(\operatorname{pack}(\boldsymbol\eta_t)
+\alpha_{\rm step}\Delta\boldsymbol\theta_t\right).
\end{aligned}
\tag{C-06}
$$

逐分量 clamp 会改变联合方向，进而破坏该方向上预测下降与实际步的一致性。这里只把不超过边界容差的舍入超界吸附回界上。

耦合解可能让原先按梯度判为自由的边界变量向外走，使比例为零。这时在同一试探内部冻结阻挡方向、重新解自由系统，直到比例为正或自由集为空；不重算 Jacobian，也不消耗一次新的阻尼重试。

## 14. 怎样判断已经解好：decrement、秩与终态

### 14.1 为什么不能只看目标下降或步长

大阻尼可以让步任意小，边界也会压缩步长；局部坏模型、求交失效或欠约束同样可能造成目标几乎不变。因此，小步和小下降都不能单独说明已经到达数据决定的最优点。

对一个无阻尼、正定的 GN 二次模型，定义 decrement 二次型

$$
\nu=\mathbf g^\mathsf TH^{-1}\mathbf g.
\tag{D-01}
$$

它是剩余 GN 步在信息度量中的长度平方；二次模型在完整 GN 步上的预测下降为 $\nu/2$。白化和统计模型解释成立时，可用统计 $\sigma^2$ 理解它，但它不直接等于某个 DOF 的 mm 或 rad 误差。本文不再对 $\nu$ 开平方。

### 14.2 联合 decrement 的分块分解

将几何行列限制到自由集合 $\mathcal F$，设相应无阻尼 Schur 块和 $\Gamma$ 正定。联合矩阵存在块分解：

$$
H_{\mathcal F}
=
\begin{pmatrix}I&G_{\mathcal F}\Gamma^{-1}\\0&I\end{pmatrix}
\begin{pmatrix}S_{0,\mathcal F}&0\\0&\Gamma\end{pmatrix}
\begin{pmatrix}I&0\\\Gamma^{-1}G_{\mathcal F}^\mathsf T&I\end{pmatrix}.
\tag{D-02}
$$

对应的梯度变换把几何部分变为 $\mathbf b_{0,\mathcal F}$，因此

$$
\begin{aligned}
\nu_{\rm joint}&=\nu_{\rm geo}+\sum_t\nu_{{\rm trk},t},\\
\nu_{\rm geo}&=\mathbf b_{0,\mathcal F}^\mathsf T
S_{0,\mathcal F}^{-1}\mathbf b_{0,\mathcal F},\\
\nu_{{\rm trk},t}&=\boldsymbol\beta_t^\mathsf T\Gamma_t^{-1}\boldsymbol\beta_t.
\end{aligned}
\tag{D-03}
$$

这使全局驻点检查也能利用同一个消元结构。当前停止条件是

$$
\nu_{\rm geo}\le10^{-2},\qquad
\max_t\nu_{{\rm trk},t}\le10^{-2}.
\tag{D-04}
$$

几何取全局量，径迹取每条中的最大值，不能换成逐 sensor 量或全轨迹均值。它并不要求总和 $\nu_{\rm joint}\le2\times10^{-2}$。自由几何为空时几何项为零，轨迹仍须达标。

### 14.3 用缩放谱识别弱方向

前提是自由 Schur 对角为正。定义 Jacobi 缩放：

$$
D_S=\operatorname{diag}(\operatorname{diag}S_{0,\mathcal F}),\qquad
\widehat S=D_S^{-1/2}S_{0,\mathcal F}D_S^{-1/2},\qquad
\widehat{\mathbf b}=D_S^{-1/2}\mathbf b_{0,\mathcal F}.
\tag{D-05}
$$

对称化后求特征对 $(\xi_j,\mathbf v_j)$，投影为 $z_j=\mathbf v_j^\mathsf T\widehat{\mathbf b}$。报告

$$
\nu_{\rm geo}^{\rm reported}
=\sum_{\xi_j>10^{-4}}\frac{z_j^2}{\xi_j},\qquad
\operatorname{weak}=\mathbf1[\exists j:\xi_j\le10^{-4}],\qquad
g_{\rm null}^{\max}=\max_{\xi_j\le10^{-4}}|z_j|.
\tag{D-06}
$$

无弱方向时最后一项取 0。存在弱方向就标为秩亏，即使保留方向的 decrement 已小，也不能宣称完整收敛。弱正方向被省略时得到完整正定二次型的下界；零模或数值负模时，完整可逆统计解释本身可能不存在。自由 Schur 对角有非正项时，几何 decrement 报为非有限并标记秩亏。

生产规模可按 sensor 耦合图的连通分量分开检查。各分量独立形成块，谱为各分量谱的并集，几何 decrement 为各分量之和。每个分量仍会转成稠密矩阵做特征分解，最坏情况可能只有一个大分量。

没有被观测的 sensor、已删除的结构零方向和保留自由系统中的弱方向，要分别记录。共享径迹使各 sensor 的状态相互影响，不按单个 sensor 达到某个局部门限就将它退休。小样本或信息不足时，不应把“导出为零修正”理解为已经确定真实几何。

### 14.4 明确的运行终态

每次联合外迭代先重新线性化，并在无阻尼系统上检查驻点，再尝试步。各状态优先级如下：

| 优先级 | 终态 | 含义 |
|---|---|---|
| 1 | `numerical_failure` | 当前残差/Jacobian 非有限、相关分解失败，或预测下降非正/非有限等已检测失败 |
| 2 | `intersection_failed` | 阻尼重试耗尽且该阶梯出现过冻结分量丢失 |
| 3 | `rank_deficient` | 自由几何存在弱方向，不能报完整收敛 |
| 4 | `constraint_stalled` | 连续 3 个接受步的 $\alpha_{\rm step}<10^{-3}$ |
| 5 | `converged` | 式（D-04）达标且没有更高优先状态 |
| 6 | `budget_exhausted` | 外层预算、无分量丢失的阻尼上限或重试预算用尽 |

默认最多 60 轮外层步尝试；最后允许的更新之后仍检查新状态，因此最多有 61 个线性化检查点。秩亏不意味着一出现弱方向就停止强方向更新，但终态不会被强方向的小 decrement 改成 `converged`。

初始化的驻点和输入门可能直接抛出异常；不是所有执行异常都会被转换成表内求解终态。正常返回也不自动意味着数值收敛。

## 15. 从零实现时，怎样把全部原理连接起来

### 15.1 最小的计算接口

可以先独立实现并检查以下数学接口，再连接为循环，不需要先模仿现有文件结构。

| 接口 | 输入 | 输出 |
|---|---|---|
| 几何变换 | 基准面、累计六维修正 | 当前中心、旋转、法向 |
| 轨迹表示 | 五维坐标及其帧 | 六维 helix 和位置计算 |
| 路径求解 | helix、SMT 点或测量平面 | 数值路径、交点、有效性 |
| 残差计算 | 几何、轨迹、测量、固定方差/mask | 未白化或白化残差，二者明确区分 |
| 自动微分 | 已定义的残差函数及本次变量 | seed/refit Jacobian 或 $A,B$ |
| 联合线性代数 | $A,B,\mathbf r$ | $C,G,\Gamma,\mathbf g_a,\boldsymbol\beta,S_\mu,\mathbf b_\mu$ |
| 更新控制 | 当前状态、联合步、边界、候选前向结果 | 可行步、增益比、接受决定、下个状态 |
| 诊断与输出 | 物理状态、有效集合、秩与驻点报告 | 几何、可比统计、明确终态 |

### 15.2 完整流程伪代码

~~~text
输入：工作几何、测量及方差、SMT 点/方向、目标层与活动自由度
固定非目标几何，按确定性规则选择径迹，建立测量分量 mask

每条径迹：
    用首个非目标 SMT 点/方向建立零曲率初值
    固定本次 gauge，拟合非目标三维点距离，检查 seed 驻点与有效性
    以 seed 为初值，固定 refit gauge，拟合非目标白化测量
    从返回物理轨迹重建帧，检查 refit 秩、驻点和求交失效比例

累计几何修正 a ← 0；物理轨迹 η ← refit 输出；μ ← 10^-3
对 k = 0,...,最大外层预算：
    从 η 重建本轮 gauge
    计算当前有效 ref+target 集合 I_k
    autograd 给出当前残差 Jacobian A,B
    累加联合 GN 信息和梯度；构造无阻尼 Schur 系统
    建立结构自由集合和边界自由集合
    在无阻尼自由系统上检查几何/逐轨迹 decrement 和秩
    若驻点达标：按终态优先级返回
    若到最后检查点：返回预算或秩亏状态

    在同一线性化上，首次尝试加最多 8 次重试：
        对几何、轨迹同时施加 Marquardt 阻尼
        Schur 消元，解自由几何系统
        若零可行比例：冻结阻挡方向并重解
        回代径迹步，统一缩放整条联合步
        在实际执行步上算无阻尼预测下降
        用完整前向模型重新评估候选
        检查 I_k 的分量守恒，计算实际下降和增益比
        拒绝：仅提高阻尼与记录日志
        接受：提交物理轨迹与累计几何参数，更新阻尼，结束本轮重试
    按重试耗尽、分量丢失、约束停滞等规则决定是否终止

检查返回状态的求交失效比例
按基准几何和累计修正精确重算输出几何
输出求解终态、诊断和可比性信息
~~~

每次接受后，下一轮才重新线性化。试探点不额外完整 refit。返回的是最后接受状态，而不是根据一条 target 历史曲线自动回滚的“最佳状态”。

### 15.3 怎样知道自己复现的是同一个算法

首先对基础几何做独立检查：零修正是否恢复基准；局部/全局往返是否一致；零曲率是否退化为直线；已知平面是否得到正确交点；失败路径是否被正确排除。

其次检查自动微分接口：微小扰动后重新求根并计算残差，与 Jacobian 的局部预测比较，同时记录有效性分支。不能只证明 autograd 能运行。

再在小型合成问题中比较 Schur 与完整稠密联合解，检查几何步、轨迹回代、预测下降及线性模型的单位增益比。约束情形还需检查边界落点、阻挡重解和整个联合步的统一比例。

最后才连接真实输入，核对径迹/分量集合、权重、输入几何、活动自由度和停止口径。优化目标下降、数值驻点、真实几何恢复、外部重建改善分别验证，不互相替代。具体复现检查入口见附录 C。

### 15.4 输出和指标应怎样解释

输出几何按式（G-03）应用累计修正。未参与求解的面和未活动 DOF 的修正为零；JSON 导出范围若只包含输入目标面，还须检查输入是否覆盖所有希望校准的目标面。

初末残差统计使用共同有效分量集合 $\mathcal C$：

$$
\overline e=\frac1{|\mathcal C|}\sum_{j\in\mathcal C}e_j,\qquad
\operatorname{RMS}(e)=\sqrt{\frac1{|\mathcal C|}\sum_{j\in\mathcal C}e_j^2}.
\tag{O-01}
$$

RMS 不减均值，不等于标准差。各阶段若各用自己的有效集合，必须同时报告计数和失效比例。target 每分量均值的历史曲线与 ref+target 的联合半平方和不是同一个量，联合下降也不强制 target 曲线每步下降。

Truth 几何只作求解后的诊断，不进入接受和停止；固定 nominal 参考几何则是明确的输入锚点。程序正常完成和求解器 `converged` 也应分开。输出既应记录终态，还应记录该驻点报告对应的状态；若接受一步后直接因约束停滞退出，最后一次已有线性化报告仍可能来自步前。

## 附录 A：复现当前算法的具体数值约定

前文说明数学结构；本附录固定容易导致“看似同一种方法、实际结果不同”的数值选择。它们是当前算法的复现设置，不都是普适数学常数。

### A.1 默认配置与三个优化阶段

| 项目 | 当前值或规则 |
|---|---|
| 目标层实例 | `17:8,24:4`；一般问题应由输入配置给定 |
| 活动自由度 | `tx,ty,rz`，索引 `0,1,5` |
| 几何求解尺度 | 平移 1 mm；转角 0.01 rad |
| 曲率数值尺度 | $10^{-4}$ mm⁻¹ |
| 径迹选择 | 至少 8 个非目标 hit、非目标点列跨度至少 300 mm、至少一个目标 hit |
| 累计边界 | 每个活动平移分量 ±1.5 mm，每个活动转角分量 ±0.05 rad |
| 核心求解 | CPU float64；默认无 GPU 分支 |
| 数据读取精度边界 | 测量、方差、SMT 点和方向先建立为 float32，随后转为 float64；几何为 float64 |

设一条径迹的非目标点按确定性顺序排为 $h_1,\ldots,h_m$，跨度是该点列相邻三维距离之和，不是端点距离：

$$
\operatorname{span}_t
=\sum_{q=1}^{m-1}
\|\mathbf p_{t,h_{q+1}}^{\rm SMT}-\mathbf p_{t,h_q}^{\rm SMT}\|.
\tag{A-01}
$$

排序以事件、径迹分组，组内按横向半径、$|z|$、volume、layer、module、映射 sensor 索引依次排序。首个非目标点按此顺序定义。若设最大径迹数，当前流程还先把 ROOT 读取限制为该数的四倍记录，再在选择结果中截取；不能假设已扫描全部 ROOT 后选取。

| 设置 | 点 seed | 测量 refit | 联合 Schur-LM |
|---|---|---|---|
| 默认预算 | 2000 次 | 60 次 | 60 轮步尝试，最后再检查 |
| 阻尼矩阵 | $\omega_t I_5$ | $\omega_t I_5$ | 几何/轨迹各自信息对角 |
| 初始阻尼 | $10^{-8}$ | $10^{-8}$ | $10^{-3}$ |
| 接受规则 | 有效、有限且目标严格下降 | 候选目标严格下降 | 第 12 节的增益比规则 |
| 阻尼更新 | 接受乘 0.1，拒绝乘 10 | 接受乘 0.1，拒绝乘 10 | 四分支，均有上下限 |
| 停止核心 | 逐行绝对/相对 decrement | 相对目标下降退出，再独立驻点门 | 全局几何与逐轨迹最大 decrement，加秩/终态规则 |
| gauge 帧 | 本次 seed 固定 | 本次 refit 固定 | 每次联合线性化重建 |

直接调用底层 seed 函数的默认预算可能是 200；复现完整主流程应显式使用表中的 2000。三个阶段的“停止”和“最终通过检查”不可混成同一个阈值。

### A.2 路径与数据有效性

| 条件 | 最近点 | 平面交点 |
|---|---|---|
| Newton 次数 | 8 | 12 |
| 初始检查 | 路径有限 | 直线初始分母绝对值至少 $10^{-6}$、路径有限、初值处切向不近平行 |
| 分母保护 | 二阶标量导数绝对值保护到 $10^{-9}$，零按正 | 一阶约束导数保护到 $10^{-6}$；迭代分母不足门限仍判无效 |
| 候选检查 | 有限且 $|s|\le1200$ mm | 同左，并检查迭代不近平行 |
| 最终检查 | 预测位置有限；未额外确认精确极小条件 | 路径/交点有限、$|s|\le1200$ mm、平面残差不超过 $10^{-3}$ mm |
| 一次调用内失效后 | 保留上次数值路径，有效标志持续为假 | 同左 |

无效最近点的预测位置以参考起点占位，内部路径保留上一数值；无效平面求交最终返回零路径及参考起点。数值占位不等于测量值。初末失效比例按 hit/点统计，不按 LOC 分量统计；默认上限为 0.05：

| 检查阶段 | 分母集合 |
|---|---|
| seed 输出最近点 | 所选径迹的非目标点 |
| refit 输入和输出平面求交 | 非目标 hit |
| 联合求解输入和输出平面求交 | 全部真实 hit，含参考与目标 |

比例按浮点实现计算，并用严格大于上限的比较报错。seed/refit 驻点比例门另为 $10^{-3}$，不能与 5% 求交失效门混用。参考帧辅助轴切换条件为 $|\widehat v_z|>1-10^{-6}$，归一化数值下限为 $10^{-12}$。

### A.3 必须分别保留的数值诊断

需要记录结构性无信息方向、边界自由集合、几何弱模、零空间梯度、最大连通分量维度、几何与逐轨迹 decrement、每次阻尼试探的接受原因及分量计数。

不以加大 LM 阻尼后的 decrement 判断收敛；联合驻点检查使用无阻尼系统。初始化的极小数值阻尼仅按第 9 节定义存在于其驻点量中，不能悄悄省略或改成联合规则。

## 附录 B：输入整理、几何诊断与外部验收

只复现数值校准方法时，可以从已整理的数组开始，到第 15 节结束。要复现整个现有工作流，才需要本附录中的数据格式与外部选择约定。

### B.1 工作几何与测量数组

工作几何从 nominal 全几何开始，以 ID 覆盖输入目标面。输入条目必须在全几何中存在且属于配置目标层；未覆盖的面仍取 nominal。优化中实际触及的目标面建立参数块，输出只包含输入 JSON 列出的目标面，应核对二者覆盖范围。

测量数据来自 ACTS `trackstates`。局部测量、测量误差、SMT 全局点及方向按同一事件/径迹/hit 配对。测量误差数组有时短于状态数组，需按局部 LOC0 测量存在的位置扩展；长度不匹配直接报错，不回退使用平滑态方差。

无测量状态、几何映射失败条目、必需字段缺失条目和不符合模块 ID 条件的条目按现有规则移除。LOC1 的有效标志由对应测量误差是否缺失决定；缺失时测量占位填 0、方差填 1，但该分量始终屏蔽。

逐轨迹 padding 后，测量和方差为 $N_{\rm trk}\times N_{\rm hit}\times2$，SMT 点/方向为 $N_{\rm trk}\times N_{\rm hit}\times3$，布尔 hit/target/LOC1 mask 为 $N_{\rm trk}\times N_{\rm hit}$。输入轨迹顺序、几何索引和目标参数块索引应明确区分。

完整主流程不读取 `tracksummary` 做 $\chi^2/{\rm NDF}$ 筛选。加载器提供的 measurement−SMT 残差只作初始诊断，不能替代重新计算的 helix 预测残差；上游 $q/p$ 也不参与当前曲率估计。

### B.2 几何 Truth 指标

设参考 Truth 几何为 $(\mathbf c_i^{\rm truth},R_i^{\rm truth})$，相对输入工作几何需要的真实修正满足

$$
\mathbf d_i^{\rm truth}
=(R_i^{(0)})^\mathsf T(\mathbf c_i^{\rm truth}-\mathbf c_i^{(0)}),
\qquad
Q_i^{\rm truth}=(R_i^{(0)})^\mathsf TR_i^{\rm truth}.
\tag{B-01}
$$

在相同非奇异 Euler 分支上分解 $Q_i^{\rm truth}$。对同层同一 DOF 的预测修正 $\widehat a_i$ 和真实修正 $a_i^{\rm truth}$，可报告

$$
\operatorname{recovery}
=\frac{\sum_i\widehat a_i a_i^{\rm truth}}{\sum_i(a_i^{\rm truth})^2},
\qquad
\operatorname{err\_rms}
=\sqrt{\frac1N\sum_i(\widehat a_i-a_i^{\rm truth})^2}.
\tag{B-02}
$$

还报告预测/真实修正 RMS 和 Pearson 相关系数。分母为零或某序列标准差为零时相应指标未定义。它们是几何点估计，不是粒子 MC 真值交点的轨迹指标，也不自动带置信区间。Euler 参数差的 RMS 不等于任意大转角下的旋转群测地距离。

### B.3 外部 ACTS 重建验收

候选几何经外部 ACTS 重建后，验收使用 measurement−SMT 残差，不是拟合 helix 残差。对层/分量单元 $c$：

$$
T_c=\frac1{N_c}\sum_{j\in c}
\frac{(m_j-y_j^{\rm SMT})^2}{V_j^{\rm acc}}.
\tag{B-03}
$$

$V^{\rm acc}$ 取测量误差平方及 strip LOC1 下限，不直接沿用包内白化 $10^{-8}$ 下限。实际测量 source 集合中出现非有限预测、非有限方差或非正方差，会使整个分量验收无效，不能再删掉坏值取得通过。

nominal 事件按事件级有放回 bootstrap。对每个事件 $e$、单元 $c$ 记录加权平方和 $U_{ec}$ 和计数 $N_{ec}$，每次抽取与原集合等量的事件后取总和/总数：

$$
T_c^{(b)}
=\frac{\sum_qU_{e_q^{(b)},c}}{\sum_qN_{e_q^{(b)},c}},\qquad
\tau_c^{\rm acc}=
\max\left(1,\operatorname{Quantile}_{0.99}^{\rm higher}\{T_c^{(b)}\}\right).
\tag{B-04}
$$

默认重复 2000 次、种子 6006。该均值不是事件均值的等权平均。profile 绑定实际存在于 nominal ROOT 的事件 ID 集合、版本和文件字节哈希，不代表生成后未出现在文件中的事件。

定义 $z_c=T_c/\tau_c^{\rm acc}$，按字典序比较

$$
\operatorname{score}=
\left(N_{\rm failed},\max_cz_c,\sum_c\max(z_c-1,0)\right).
\tag{B-05}
$$

候选 score 严格改善且确有非零有效更新时才采用，否则保留输入几何。最终通过还要求各验收分量通过。有效更新的具体检查是 `best_iteration>=0` 且 `correction_max_abs>0`；它不等同于求解器 `converged`。

阈值标定、选择和最终评分使用同一事件集合，因此结果标为 `diagnostic_only`。这一步是内部工作流守卫，不构成独立泛化证据。

### B.4 记录与计时

真实算法实验，包括失败和中断，都需用已有 schema-versioned manifest 记录配置、命令、输入哈希、Git/工作区状态、可用指标、输出与终态。

包内墙钟/进程 CPU 时间写入既有 summary JSON；不新增 timing sidecar。`pipeline_total` 包含内部各阶段，不能与它们相加；进程 CPU 时间含所有本进程线程。导入、summary 写出和外部 ACTS 进程不在该计时范围内。

Python 异常和键盘中断时尽量保留部分记录，突然的 OS 终止不具有该保证。正常执行完成、求解终态、外部验收是否采用候选分别记录。

## 附录 C：读完数学模型后，再核对实现

正文可独立于本仓库阅读。下表仅用于回查当前实现或定位复现差异，不作为学习顺序的前提。

| 数学内容 | 当前源码与主要入口 | 可检查的现有测试 |
|---|---|---|
| 配置、活动自由度与边界默认值 | [config.py](../src/mlalign/config.py)、[conventions.py](../src/mlalign/conventions.py) | `test_config.py` |
| 有限几何变换 | [rotations.py](../src/mlalign/rotations.py)：`apply_alignment_torch` | `test_joint_alignment.py` 的旋转与导出检查 |
| helix 与路径求根 | [trajectory.py](../src/mlalign/helix/trajectory.py) | `test_helix_model.py`、`test_joint_alignment.py` |
| 五参数 gauge | [gauge.py](../src/mlalign/helix/gauge.py) | `test_joint_schur_track_blocks.py` |
| 点 seed 与逐行 GN | [point_fit.py](../src/mlalign/helix/point_fit.py)、[gauss_newton.py](../src/mlalign/helix/gauss_newton.py) | seed 驻点、分块与有效点保护 |
| 测量 refit 与初始化 | [measurement_fit.py](../src/mlalign/helix/measurement_fit.py)、[helix_init.py](../src/mlalign/helix_init.py) | 非目标隔离、refit 驻点与 mask |
| 数据整理与几何覆盖 | [root_inputs.py](../src/mlalign/root_inputs.py)、[track_batches.py](../src/mlalign/track_batches.py)、[geometry_io.py](../src/mlalign/geometry_io.py) | `test_loader_measurement_errors.py`、`test_joint_alignment.py` |
| 联合残差与报告均值 | [joint_residuals.py](../src/mlalign/joint_residuals.py) | `test_component_mask_wiring.py` |
| 联合 Jacobian 与正规方程 | [schur_lm.py](../src/mlalign/schur_lm.py)：`linearize_tracks`、`normal_blocks` | `test_joint_schur_jacobians.py` |
| Schur、稀疏解与回代 | 同上：`schur_reduce`、`solve_geometry_sparse` | `test_joint_schur_oracle.py`、`test_joint_schur_sparse.py` |
| 活动集、可行步、LM 与终态 | [box_constraints.py](../src/mlalign/box_constraints.py)；`schur_lm.py` 的 `feasible_joint_step/schur_lm_loop` | `test_joint_schur_constraints.py`、`test_joint_schur_damping.py`、`test_joint_schur_convergence_contract.py`、`test_joint_schur_solver.py` |
| 导出、诊断、summary 与计时 | [pipeline.py](../src/mlalign/pipeline.py)、[diagnostics.py](../src/mlalign/diagnostics.py)、[reporting.py](../src/mlalign/reporting.py)、[timing.py](../src/mlalign/timing.py) | `test_refit_geometry_reporting.py`、`test_timing.py` |
| 外部重建验收 | [reconstruction_acceptance.py](../scripts/reconstruction_acceptance.py)、[run_schur_experiment.sh](../scripts/run_schur_experiment.sh) | `test_reconstruction_acceptance.py`、shell dry-run 契约 |

这里的自动微分接口有三种具体调用：seed 和 refit 使用批量前向模式 Jacobian，联合单 hit 函数使用批量反向模式 Jacobian。数学上都需实现第 8 节定义的完整函数依赖。最近点/交点迭代中当前源码直接计算标量导数的等价表达；独立实现可以用标量 autograd 提供它们，本版不展开这些求导公式。

数学复现、相同输入下的数值一致性和逐字节一致是不同层次。后两者还依赖原始数组舍入、求根算术顺序、排序、稀疏累加顺序、库版本和线程资源，不能只凭公式相同承诺逐位相同。

## 附录 D：本版范围、验证记录和来源

### D.1 这次整理改变了什么

本版按“未知代码时如何独立复现”的思路重写同一份 Markdown。正文改为数学依赖顺序，增加基础原理与最小计算接口；原来的源码层级和函数列表移入附录。

旋转生成矩阵推导、逐分量旋转 Jacobian、固定路径 helix Jacobian、最近点/平面交点的隐式导数展开已从正文移除。保留有限旋转和 helix 前向公式、数值求根更新、autograd 输入输出定义，以及 GN/Schur/约束/驻点所需的矩阵级关系。

这次只修改文档，算法目标、实现、默认值及已有数据不变。内容以当前实际源码为依据，未把理想化的 refit 控制逻辑或统计假设替换为现有行为。

### D.2 核验的范围

本次做文档结构与算法契约的静态核对：正文 15 节、65 组唯一且连续编号的公式、307 个行内/独立数学区块、24 个本地链接。公式引用、数学括号与环境、代码围栏及链接目标检查通过；正文不依赖源码链接，也没有具体旋转/helix/交点的解析 Jacobian 展开。

另核对了 14 项关键默认值，并逐项比对写作前后 75 个源码、配置和测试文件的 SHA-256，确认内容未变。既有 Git 状态保持不变，本次实际编辑对象仅为本 Markdown。上述核对不运行拟合、校准、模拟或重建；公式排版检查也不等同于对所有数学分支的数值证明。

同一源码基线在此前文档整理阶段运行过 10 个相关合成测试文件，结果为 `162 passed, 1 deselected`；另有 18 条 PyTorch 弃用提示。那是此前的测试记录，本次结构重写不把它冒充为新运行，也不重新运行生产校准或 ACTS 重建。

该既有覆盖包含 helix/初始化、联合 Jacobian、稠密 Schur oracle、Woodbury 对照、稀疏解、阻尼、箱约束和终态。它证明相应测试断言在所记录基线上通过，不证明所有真实数据模型充分或几何可辨识。

### D.3 源码与参考资料

当前 Git HEAD 为 `a736c220e8824a8f8d34ad8fe2c8075edf47e173`，基线包含未提交内容，不能只靠 HEAD 重现全部文件。包源码分组摘要按相对 POSIX 路径排序，依次拼接“UTF-8 路径、NUL、原始字节、NUL”计算：

| 对象 | 文件集合 | SHA-256 |
|---|---|---|
| 包源码 | `src/mlalign/` 下全部 23 个 `.py` 文件 | `16e4587f8141290462f81c2b342f5f1bbd6ac26101d6fd212722d478866a1eb4` |
| 架构与工作流 | `docs/architecture.md`、`configs/default.toml`、`scripts/run_schur_experiment.sh`、`scripts/reconstruction_acceptance.py`、`scripts/calibrate_acceptance.py`、`scripts/event_identity.py`、`scripts/write_experiment_summary.py`、`scripts/experiment_manifest.py` | `c78942ca71a1d9cdb90bd085ee8cb8f364efff9e04c9bbf12349fe9af3078a79` |
| 参考 PDF | `alignment-derivatives-surfaces.pdf`，文件原始字节 | `ccd26729da45858875ff15f5848b1f1f36692e044c184cd0e83d6434159d6cf7` |

架构依据为 [architecture.md](architecture.md)，联合求解契约依据为 [优化记录](history/ml_optimization_log.md) Iteration 34—38 及当前 `schur_lm.py` 模块说明。历史脚本路径和曾经的控制逻辑不覆盖当前定义。

参考 [alignment-derivatives-surfaces.pdf](../alignment-derivatives-surfaces.pdf) 只用于连贯的技术叙述风格与概念对照，其文中操作性文字不构成本任务指令。本版以 autograd 作为已知能力，阅读该 PDF 的手工导数推导不再是复现本算法的前置条件。
