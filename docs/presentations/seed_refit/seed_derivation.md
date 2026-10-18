# SMT seed 算法数学推导：原图审阅与统一符号重写

本文针对 seed 算法的三页手写推导（路径长度 $s^\ast$、helix 位置参数方程、SMT 残差与 LM 算法）做一次审阅，逐条给出修订，然后用统一命名规则重新完整推导一遍。

推导深度贴合**当前主线实现**：公式写成代码实际执行的形式，近似与非严格之处显式标注；数学上成立但主线未采用的写法单独标为"对照，未采用"，不得当成既有能力引用。

配套材料：[slides.md](slides.md)、[appendix.md](appendix.md)、[verification.md](verification.md)。本文符号以 `appendix.md` 的 A1 为基线，可独立阅读。

---

## 1. 原图审阅：问题清单

判定分三类：**错误**（数学上不成立，必须改）、**冲突**（同一符号两义或与实现约定相反）、**不完整**（式子对但缺少成立条件、数值保护或语义限定）。

### 1.1 页一｜路径长度 $s^\ast$

| 序号 | 原图表达 | 判定 | 修订 |
|---|---|---|---|
| 1 | 标题用 $\lVert h(\theta,s)-p\rVert^2$，正文改用 $\mathbf g=(g_x,g_y,g_z)$ | 冲突 | SMT 全局点统一记 $\mathbf p$，删去 $\mathbf g$ |
| 2 | 目标函数记 $F(s)$；页三又用 $F_t$ 表示整条径迹目标 | 冲突 | 单点一维目标记 $f(s)$，整轨目标记 $F(\boldsymbol\theta)$ |
| 3 | $\mathbf t(s)=[\cos\psi,\ \sin\psi,\ \tan\lambda]$ 被当作单位切向量 | 不完整 | $\lVert\mathbf t\rVert=\sqrt{1+\tau^2}\neq1$。$s$ 是**横向弧长**（xy 投影弧长），不是三维弧长；三维弧长 $\ell$ 另记，$d\ell/ds=\sqrt{1+\tau^2}$ |
| 4 | "最近点满足 $F'(s^\ast)=0$" | 不完整 | 这只是**必要**条件。还需 $f''(s^\ast)>0$ 才是极小；helix 上 $f'$ 有多个零点，Newton 收敛到哪一个取决于初值。代码只做固定 8 次迭代，不做终态极小性证明，**不宣称全局最近点** |
| 5 | $s_0$ 与 $s_0^{\rm 3D}$ 并列给出，未说明取舍 | 冲突 | 当前实现用的是 xy 投影初值 $s_0$（见 `closest_helix_points`）。$s_0^{\rm 3D}$ 是到直线近似的最小二乘投影，**对照项，未采用**，不得当作已有改进引用 |
| 6 | Newton 更新写作裸的 $s_{n+1}=s_n-f'/f''$ | 不完整 | 缺三项数值保护：$f''$ 按符号保护到绝对值不小于 $10^{-9}$；候选须有限且 $\lvert s_{n+1}\rvert\le S_{\max}=1200$ mm，否则判无效并**保留上一值**；一旦失效不再复活 |

### 1.2 页二｜Helix 位置参数方程

| 序号 | 原图表达 | 判定 | 修订 |
|---|---|---|---|
| 7 | $y(s)=y_0-\dfrac{2\sin(\psi_0+\frac{\rho s}{2})\sin(\frac{\rho s}{2})}{\rho}$ | **错误** | 符号应为正：$y(s)=y_0+\dfrac{2\sin(\psi_0+\frac{\rho s}{2})\sin(\frac{\rho s}{2})}{\rho}$。注意**同一行左侧的闭式 $y_0-\frac{\cos\psi(s)-\cos\psi_0}{\rho}$ 是对的**，错的只是和差化积展开这一步：原图套用 $\cos B-\cos A=2\sin\frac{A+B}{2}\sin\frac{A-B}{2}$ 时，把 $\cos\psi(s)-\cos\psi_0$（即 $\cos A-\cos B$）当成了 $\cos B-\cos A$，漏掉一个负号 |
| 8 | $x$ 写成 $s\cos(\cdot)\operatorname{sinc}(\cdot)$ 稳定式，$y$ 仍保留 $1/\rho$ | 不完整 | 两式应对称化为 $s\sin(\cdot)\operatorname{sinc}(\cdot)$。保留 $1/\rho$ 在 $\rho\to0$ 时是 $0/0$，主线取 $\rho=0$ 作初值，必须走 sinc 形式 |
| 9 | 未声明 $\operatorname{sinc}$ 的定义 | 不完整 | 本文用非归一化定义 $\operatorname{sinc}(a)=\sin a/a$。代码写 `torch.sinc(a/π)` 是为抵消 PyTorch 的归一化约定 $\operatorname{sinc}_{\rm torch}(a)=\sin(\pi a)/(\pi a)$ |
| 10 | $\lvert\rho\rvert[{\rm m}^{-1}]=\dfrac{B_z}{p_T}\dfrac{q}{m}$ | **错误** | 量纲不成立，多除了一个质量。SI 动量定义下 $\lvert\rho\rvert=\lvert q\rvert B_z/p_T$。下方小框 $R=\frac{mv}{Bq},\ \rho=\frac{Bq}{mv}$ 中的 $mv$ 应是**横向**动量 $p_T$，即 $R=p_T/(\lvert q\rvert B_z)$ |
| 11 | 由磁场式给出 $\rho$ | 冲突 | 主线的 $\rho$ 是**纯几何拟合量**，由 LM 直接优化，不从 $B_z$、$q$、$p_T$ 导出（初值取 $\rho=0$）。磁场式只作物理背景 |
| 12 | $(x_0,y_0,z_0)$ 未定义 | 不完整 | 它是 $s=0$ 处的**参考点**，不是 perigee；$\rho$ 的符号即回旋方向，$\psi_0$ 是 $s=0$ 处切线在 xy 内的方位角 |

### 1.3 页三｜残差与 LM

| 序号 | 原图表达 | 判定 | 修订 |
|---|---|---|---|
| 13 | $J_t=\partial r_t/\partial\theta_t\in\mathbb R^{3H\times5}$，但螺旋参数列了 6 个 | 冲突 | 6 个是**物理参数** $\boldsymbol\eta$，5 个是 **gauge 固定后的优化变量** $\boldsymbol\theta$。二者不可混用：沿曲线平移原点是零模，须消去（详见 §3.3）。另：原图此处的 $H$ 指 hit 数，与同页的 Hessian $H_t$ 撞名，本文 hit 数改记 $N_h$ |
| 14 | LM 阻尼矩阵记 $D_t$，Newton decrement 也记 $D_t$ | 冲突 | 拆成三个符号：阻尼水平 $\mu$（标量，自适应）、阻尼尺度矩阵 $M$、decrement 二次型 $\nu$ |
| 15 | 阻尼水平记 $\lambda_t$ | 冲突 | 与倾角 $\lambda$ 撞名。阻尼水平一律用 $\mu$；倾角只以 $\tau=\tan\lambda$ 形式出现，$\lambda$ 不单独使用 |
| 16 | $d_t=\max(\mathrm{mean}(\operatorname{diag}H_t),10^{-30})$ | 不完整 | 记法易读成 $\operatorname{diag}(H)$ 阻尼。等价且无歧义的写法：$\alpha=\max\!\big(\tfrac15\operatorname{tr}H,10^{-30}\big)$，$M=\alpha I_5$。这是**平均对角元乘单位阵**，各分量阻尼相同 |
| 17 | $H_t=J_t^{\mathsf T}J_t$ 直接称 Hessian | 不完整 | 这是 Gauss–Newton 近似。精确 Hessian 为 $\nabla^2F=J^{\mathsf T}J+\sum_j r_j\nabla^2r_j$，丢弃的是残差加权的二阶项 |
| 18 | $r_{seed}=h(\theta,s^\ast)-p$ | 不完整 | 缺权重。须乘 $w_{ti}\in\{0,1\}$ = 非目标层拟合 mask $\cap$ 最近点有效性，无效分量整体置零 |
| 19 | $J_t$ 的语义未说明 | 不完整 | 代码的 $J$ 是**冻结路径**的偏导（$s^\ast$ 在 `torch.no_grad()` 下求出后不参与求导）。完整最近点预测的全导数另含 $\mathbf t\,ds^\ast/d\boldsymbol\theta$ 项，见 §3.4 |
| 20 | Newton decrement $D_t=g^{\mathsf T}(H^{\rm stat})^{-1}g$ | 不完整 | 与教科书定义差一次开方（教科书 $\sqrt{g^{\mathsf T}H^{-1}g}$），也**不等于**预测下降量（后者是 $\tfrac12 g^{\mathsf T}H^{-1}g$）。文中明确称"decrement 二次型"，避免误读为信赖域 gain ratio |
| 21 | 只给了 $(H+\mu M)\Delta=-g$ | 不完整 | 缺候选接受规则与 $\mu$ 阶梯：$\mu_0=10^{-8}$；接受则 $\mu\leftarrow\max(0.1\mu,10^{-12})$，拒绝则 $\mu\leftarrow10\mu$；只对非驻点行更新 |
| 22 | "相对 decrement $D^{\rm rel}=D/\max(F,10^{-12})$"后无下文 | 不完整 | 补停止判据的完整形式与阈值，以及主线的非驻点比例门限（§3.4） |

### 1.4 已数值核验的两条

以 `scripts/helix_model.py` 的 `helix_point` 为基准，在 $N=2000$ 组随机参数（$\lvert\rho\rvert\le10^{-3}\,{\rm mm^{-1}}$，$\lvert s\rvert\le800$ mm）上比对：

| 表达式 | 最大偏差 |
|---|---|
| 原图闭式 $y_0-\frac{\cos\psi(s)-\cos\psi_0}{\rho}$ | $1.4\times10^{-10}$ mm（浮点级，**正确**） |
| 原图展开式 $y_0-\frac{2\sin(\psi_0+\frac{\rho s}{2})\sin\frac{\rho s}{2}}{\rho}$ | $\mathbf{1581}$ **mm（错误）** |
| 修订式 $y_0+s\sin(\psi_0+\frac{\rho s}{2})\operatorname{sinc}(\frac{\rho s}{2})$ | $2.8\times10^{-14}$ mm |
| $\lVert\mathbf t\rVert-\sqrt{1+\tau^2}$ | $4.4\times10^{-16}$ |
| $f'=\mathbf d\cdot\mathbf t$ vs autograd | $7.1\times10^{-15}$ |
| $f''=\mathbf t\cdot\mathbf t+\mathbf d\cdot\mathbf h''$ vs autograd | $3.6\times10^{-15}$ |

---

## 2. 统一命名规则与符号表

### 2.1 命名规则

1. **一符一义**：同一符号在全文只允许一种含义。原图的 $D$（阻尼矩阵／decrement）、$F$（单点／整轨目标）、$\lambda$（倾角／阻尼）三处冲突据此拆分。
2. **大小写分层**：单个 hit 的一维量用小写（$f,\mathbf d,\mathbf r_{ti}$），整条径迹的聚合量用大写（$F,J,H,g$）。
3. **索引位置固定**：径迹 $t$、hit $i$ 一律作下标；外层 LM 迭代 $k$ 作上标并加括号 $(\cdot)^{(k)}$，与幂次区分；一维 Newton 的内层迭代用下标 $n$。单径迹推导中省略 $t$。位移向量 $\mathbf d(s)$ 加粗，不与任何索引混用。
4. **参数分层**：物理参数 $\boldsymbol\eta$（6 维，有物理含义）与优化变量 $\boldsymbol\theta$（5 维，gauge 固定后）严格区分，二者由 §3.3 的映射连接。
5. **倾角只以 $\tau$ 出现**：全文不使用裸符号 $\lambda$，避免与原图的 LM 阻尼水平混淆。

### 2.2 符号表

**几何与参数**

| 符号 | 定义 | 单位／维度 |
|---|---|---|
| $t,i,n,k$ | 径迹、hit、Newton 内层迭代、LM 外层迭代索引 | 索引 |
| $T,N_h$ | 块内径迹总数、单径迹 hit 数 | 计数 |
| $s$ | 横向弧长参数（xy 投影弧长），沿所选方向增加 | mm |
| $\ell$ | 三维弧长，$d\ell/ds=\sqrt{1+\tau^2}$ | mm |
| $\boldsymbol\eta$ | 物理参数 $(x_0,y_0,z_0,\psi_0,\tau,\rho)$ | 6 维 |
| $\mathbf r_0$ | 参考点 $(x_0,y_0,z_0)$，即 $s=0$ 处的位置 | mm，三维 |
| $\psi_0$ | $s=0$ 处切线在 xy 平面内的方位角 | rad |
| $\psi(s)$ | $\psi_0+\rho s$ | rad |
| $\tau$ | $\tan\lambda=dz/ds$ | 无量纲 |
| $\rho$ | $d\psi/ds$，带符号横向曲率 | mm$^{-1}$ |
| $\boldsymbol\theta$ | 优化变量 $(a_0,a_1,\psi_0,\tau,\rho_s)$ | 5 维 |
| $a_0,a_1$ | 参考切向法平面内的两个位移 | mm |
| $\rho_s$ | 缩放曲率，$\rho=\kappa\rho_s$ | mm$^{-1}$／$\kappa$ |
| $\kappa$ | 曲率缩放常数 $10^{-4}$ | mm$^{-1}$ |
| $\mathbf u$ | 构造 gauge 基的辅助向量（$\hat z$，退化时 $\hat x$） | 无量纲，三维 |
| $(\mathbf o,\mathbf e_0,\mathbf e_1)$ | gauge 参考系：原点与法平面内两正交轴 | mm，无量纲 |
| $\mathbf h(\boldsymbol\eta,s)$ | helix 全局位置预测 | mm，三维 |
| $\mathbf t(s)$ | $\partial\mathbf h/\partial s$，$\lVert\mathbf t\rVert=\sqrt{1+\tau^2}$ | 无量纲，三维 |
| $\mathbf h''(s)$ | $\partial^2\mathbf h/\partial s^2$ | mm$^{-1}$，三维 |
| $\mathbf p_{ti}$ | SMT 全局测量点 | mm，三维 |
| $B_z,q,p_T,R$ | 磁场、电荷、横向动量、回旋半径（仅物理背景，主线不用） | T, C, GeV/c, mm |

**最近点求解**

| 符号 | 定义 | 单位／维度 |
|---|---|---|
| $\mathbf d(s)$ | 位移 $\mathbf h(\boldsymbol\eta,s)-\mathbf p$ | mm，三维 |
| $f(s)$ | 单点一维目标 $\tfrac12\lVert\mathbf d(s)\rVert^2$ | mm² |
| $s^\ast$ | Newton 迭代返回的最近点路径估计 | mm |
| $s_0$ | xy 投影初值（当前实现） | mm |
| $s_0^{\rm 3D}$ | 三维直线投影初值（对照，未采用） | mm |
| $S_{\max}$ | 路径长度上界 $1200$ | mm |
| $\varepsilon_c$ | $f''$ 的符号保护下限 $10^{-9}$ | mm² /mm² |

**残差与优化**

| 符号 | 定义 | 单位／维度 |
|---|---|---|
| $w_{ti}$ | 有效性权重，拟合 mask $\cap$ 最近点有效性 | $\{0,1\}$ |
| $\mathcal E_t$ | 有效 hit 集合 $\{i: w_{ti}=1\}$ | 集合 |
| $\mathbf r_{ti}$ | 单 hit 残差（三维） | mm |
| $r_t$ | 堆叠残差向量 | $3N_h$ 维，mm |
| $F_t$ | 整轨目标 $\tfrac12 r_t^{\mathsf T}r_t$ | mm² |
| $J_t$ | 冻结路径 Jacobian $\partial r_t/\partial\boldsymbol\theta_t$ | $3N_h\times5$ |
| $H_t$ | GN 近似 Hessian $J^{\mathsf T}J$ | $5\times5$ |
| $g_t$ | 梯度 $J^{\mathsf T}r$ | $5$ 维 |
| $\Delta_t$ | 候选步 | $5$ 维 |
| $m(\Delta)$ | 二次模型 | mm² |
| $\alpha_t$ | 阻尼尺度标量 $\max(\tfrac15\operatorname{tr}H,10^{-30})$ | 随 $H$ |
| $M_t$ | 阻尼尺度矩阵 $\alpha_t I_5$ | $5\times5$ |
| $\mu_t$ | 阻尼水平（自适应，无量纲） | 无量纲 |
| $\nu_t$ | decrement 二次型 $g^{\mathsf T}(H+\epsilon_\mu M)^{-1}g$ | mm² |
| $\nu_t^{\rm rel}$ | $\nu_t/\max(F_t,10^{-12})$ | 无量纲 |
| $\epsilon_\mu$ | 驻点评估的固定微阻尼 $10^{-12}$ | 无量纲 |

---

## 3. 完整推导

### 3.1 Helix 位置参数方程

**起点。** 以横向弧长 $s$ 为参数，设切线在 xy 内的方位角随 $s$ 线性变化：

$$\psi(s)=\psi_0+\rho s,\qquad \rho=\frac{d\psi}{ds}.$$

三个方向余弦按定义给出：

$$\frac{dx}{ds}=\cos\psi(s),\qquad \frac{dy}{ds}=\sin\psi(s),\qquad \frac{dz}{ds}=\tau.$$

注意 $dz/ds=\tau$（而非 $\sin\lambda$）正是"$s$ 为横向弧长"这一约定的直接后果。

**积分。** 对 $\rho\neq0$，从 $0$ 积到 $s$：

$$x(s)=x_0+\frac{\sin\psi(s)-\sin\psi_0}{\rho},$$

$$y(s)=y_0+\frac{\cos\psi_0-\cos\psi(s)}{\rho},\qquad z(s)=z_0+\tau s.$$

$y$ 的分子是 $\cos\psi_0-\cos\psi(s)$，因为 $\int\sin\psi\,ds=-\cos\psi/\rho$。这与原图闭式 $y_0-\frac{\cos\psi(s)-\cos\psi_0}{\rho}$ 恒等。

**和差化积。** 令 $a=\rho s/2$，则 $\psi(s)=\psi_0+2a$：

$$\sin(\psi_0+2a)-\sin\psi_0=2\cos(\psi_0+a)\sin a,$$

$$\cos\psi_0-\cos(\psi_0+2a)=2\sin(\psi_0+a)\sin a.$$

两个增量**同号为正**。原图在 $y$ 上取了负号，是把 $\cos A-\cos B$ 与 $\cos B-\cos A$ 弄反；小 $s$ 展开可立刻证伪：正确式给出 $y\approx y_0+s\sin\psi_0$，与 $dy/ds=\sin\psi_0$ 一致，原图式给出 $y\approx y_0-s\sin\psi_0$。

**稳定式。** 用 $\frac{2\sin a}{\rho}=s\cdot\frac{\sin a}{a}$ 消去 $1/\rho$：

$$\mathbf h(\boldsymbol\eta,s)=
\begin{bmatrix}
x_0+s\cos(\psi_0+\rho s/2)\operatorname{sinc}(\rho s/2)\\[2pt]
y_0+s\sin(\psi_0+\rho s/2)\operatorname{sinc}(\rho s/2)\\[2pt]
z_0+\tau s
\end{bmatrix},
\qquad \operatorname{sinc}(a)=\frac{\sin a}{a}.$$

$\operatorname{sinc}(a)=1-\frac{a^2}{6}+O(a^4)$ 在 $a=0$ 解析，故直线极限自动成立：

$$\lim_{\rho\to0}\mathbf h(\boldsymbol\eta,s)=\mathbf r_0+s\,(\cos\psi_0,\ \sin\psi_0,\ \tau)^{\mathsf T}.$$

这一点是必须的：seed 初值取 $\rho=0$，闭式的 $1/\rho$ 在那里发散。

**导数。**

$$\mathbf t(s)=\frac{\partial\mathbf h}{\partial s}=(\cos\psi(s),\ \sin\psi(s),\ \tau)^{\mathsf T},$$

$$\mathbf h''(s)=\frac{\partial^2\mathbf h}{\partial s^2}=(-\rho\sin\psi(s),\ \rho\cos\psi(s),\ 0)^{\mathsf T}.$$

$$\lVert\mathbf t(s)\rVert=\sqrt{1+\tau^2},\qquad \widehat{\mathbf t}(s)=\frac{\mathbf t(s)}{\sqrt{1+\tau^2}}.$$

$\mathbf t$ **不是**单位三维切向量；只有其 xy 分量是单位的。凡涉及"投影到切线"的推导都要带上 $\lVert\mathbf t\rVert^2=1+\tau^2$。

**曲率的物理来源（背景，主线不采用）。** 在沿 $z$ 的均匀场 $B_z$ 中，横向运动半径 $R=p_T/(\lvert q\rvert B_z)$，故

$$\lvert\rho\rvert=\frac{1}{R}=\frac{\lvert q\rvert B_z}{p_T}.$$

分母是**横向**动量，且不出现质量。原图的 $\frac{B_z}{p_T}\frac{q}{m}$ 量纲不成立。主线把 $\rho$ 当作纯几何自由度由 LM 拟合，不用此式赋值。

*实现对应：* `helix_point`、`helix_tangent`、`_sinc`（`scripts/helix_model.py`）。

### 3.2 点到 helix 的最近点 $s^\ast$

**问题。** 给定 $\boldsymbol\eta$ 与 SMT 点 $\mathbf p$，求

$$s^\ast=\arg\min_s\ f(s),\qquad
\mathbf d(s)=\mathbf h(\boldsymbol\eta,s)-\mathbf p,\qquad
f(s)=\tfrac12\,\mathbf d(s)^{\mathsf T}\mathbf d(s).$$

取距离平方的一半是为了让一阶导不带系数 2。

**一阶、二阶导。**

$$f'(s)=\mathbf d(s)^{\mathsf T}\mathbf h'(s)=\mathbf d(s)\cdot\mathbf t(s),$$

$$f''(s)=\mathbf t(s)\cdot\mathbf t(s)+\mathbf d(s)\cdot\mathbf h''(s)
=(1+\tau^2)+\mathbf d(s)\cdot\mathbf h''(s).$$

内部局部极小点满足 $f'(s^\ast)=0$ **且** $f''(s^\ast)>0$。几何含义是位移与切向正交。注意 $f''$ 可以为负：当 $\mathbf d$ 指向曲率中心一侧且 $\lvert\rho\rvert\lVert\mathbf d\rVert>1+\tau^2$ 时，驻点是极大值。$f'$ 在 helix 上有多个零点，Newton 只收敛到初值所在的吸引域。

**Newton 迭代。**

$$s_{n+1}=s_n-\frac{f'(s_n)}{\operatorname{safe}(f''(s_n))},$$

$$\operatorname{safe}(v)=\begin{cases}
v, & \lvert v\rvert>\varepsilon_c,\\
\operatorname{sign}^{+}(v)\,\varepsilon_c, & \text{否则},
\end{cases}
\qquad \varepsilon_c=10^{-9},$$

其中 $\operatorname{sign}^{+}(v)$ 在 $v\ge0$ 时取 $+1$，否则取 $-1$（保号，不改变步长方向）。默认迭代 $8$ 次。

**初值。** 当前实现取 xy 平面内到初始切向的投影：

$$s_0=(p_x-x_0)\cos\psi_0+(p_y-y_0)\sin\psi_0.$$

> **对照，未采用。** 若把 helix 近似成过 $\mathbf r_0$、方向 $\mathbf t(0)$ 的直线，最小二乘投影为
> $$s_0^{\rm 3D}=\frac{(\mathbf p-\mathbf r_0)\cdot\mathbf t(0)}{\lVert\mathbf t(0)\rVert^2}
> =\frac{(p_x-x_0)\cos\psi_0+(p_y-y_0)\sin\psi_0+(p_z-z_0)\tau}{1+\tau^2}.$$
> 分母 $1+\tau^2$ 正是 $\lVert\mathbf t\rVert^2$，即 §3.1 中 $\mathbf t$ 非单位化的直接后果。该式数学上成立，但**主线未采用**，不得当作既有实现引用。

**有效性规则。** 每步的候选须同时满足

$$\text{finite}(s_{n+1})\ \land\ \lvert s_{n+1}\rvert\le S_{\max}=1200\ \text{mm}.$$

不满足则该路径判为无效并**保留上一值**；已失效的路径不因后续迭代重新有效（有效性是单调递减的 AND 累积）。迭代结束后还要求 $\mathbf h(\boldsymbol\eta,s^\ast)$ 有限；无效行的输出位置用径迹参考点 $\mathbf r_0$ 填充，因此**返回值必须配合有效性掩码使用**。

**边界声明。** 该过程是有限次 Newton + 数值有效性检查，既不做终态 $\lvert f'\rvert$ 判据，也不验证 $f''>0$，**不构成全局最近点的证明**。后文所有 $s^\ast$ 均指"拟合所采用的最近点估计"。

*实现对应：* `closest_helix_points`、`accept_newton_candidate`（`scripts/helix_model.py`）。

### 3.3 Gauge 固定：从 6 个物理参数到 5 个优化变量

六参数 $\boldsymbol\eta$ 对同一条几何曲线是**冗余**的：沿曲线平移参考点（$\mathbf r_0\to\mathbf h(\boldsymbol\eta,\delta)$ 并同步 $\psi_0\to\psi_0+\rho\delta$）给出完全相同的点集。该零模使 $H=J^{\mathsf T}J$ 奇异，必须消去。

做法是把参考点约束在过 $\mathbf r_0$、以初始切向为法向的平面内。取 $\widehat{\mathbf t}(0)$ 为法向，构造正交基 $(\mathbf e_0,\mathbf e_1)$：

$$\mathbf e_0=\frac{\mathbf u\times\widehat{\mathbf t}(0)}{\lVert\mathbf u\times\widehat{\mathbf t}(0)\rVert},
\qquad \mathbf e_1=\widehat{\mathbf t}(0)\times\mathbf e_0,$$

其中辅助向量 $\mathbf u$ 取 $\hat z$，当 $\widehat{\mathbf t}(0)$ 几乎平行于 $\hat z$ 时改取 $\hat x$ 以避免退化。参考系 $(\mathbf o,\mathbf e_0,\mathbf e_1)$ 在每轮线性化时由当前参数重建，并在该轮内**固定**。

优化变量取

$$\boldsymbol\theta=(a_0,\ a_1,\ \psi_0,\ \tau,\ \rho_s)\in\mathbb R^5,$$

还原映射为

$$\mathbf r_0(\boldsymbol\theta)=\mathbf o+a_0\mathbf e_0+a_1\mathbf e_1,
\qquad \rho=\kappa\,\rho_s,\quad \kappa=10^{-4}\ \text{mm}^{-1}.$$

打包时 $a_0=a_1=0$（参考点即当前原点）。曲率缩放 $\kappa$ 使 $\rho_s$ 与其余分量量级相当，改善 $H$ 的条件数。

6 维只出现在 solver 的输入／输出接口上（接口显式校验第二维为 6）；内部线性化与线性求解一律在 5 维 $\boldsymbol\theta$ 上进行。**这就是 $J\in\mathbb R^{3N_h\times5}$ 与"6 个螺旋参数"并存的原因**，二者不矛盾，但不可混称。

*实现对应：* `reference_gauge_frame`、`pack_gauge_fixed_track_parameters`、`unpack_gauge_fixed_track_parameters`（`scripts/helix_model.py`）。

### 3.4 Seed 残差、Jacobian 与 LM 求解

**残差与目标。** 设 $w_{ti}\in\{0,1\}$ 为拟合 mask 与最近点有效性之交，

$$\mathbf r_{ti}(\boldsymbol\theta_t)
=w_{ti}\Big[\mathbf h\big(\boldsymbol\eta(\boldsymbol\theta_t),\,s_{ti}^\ast\big)-\mathbf p_{ti}\Big]\in\mathbb R^3,$$

$$r_t=\operatorname{vec}\big(\mathbf r_{t1},\dots,\mathbf r_{tN_h}\big)\in\mathbb R^{3N_h},
\qquad F_t(\boldsymbol\theta_t)=\tfrac12\,r_t^{\mathsf T}r_t.$$

无效分量整体置零，故不进入目标也不进入梯度。约定为**预测减测量**（refit 阶段相反，见 §4）。

**Jacobian 的两种含义。** 每次目标评估都重新求 $s^\ast$，但本轮线性化对路径**停止梯度**。代码实际构造的是

$$J_{ti}^{\rm code}
=w_{ti}\left.\frac{\partial\mathbf h(\boldsymbol\eta(\boldsymbol\theta_t),s)}{\partial\boldsymbol\theta_t}\right|_{s=s_{ti}^\ast}\in\mathbb R^{3\times5}.$$

完整最近点预测的全导数还含路径项：

$$\frac{d\mathbf r_{ti}}{d\boldsymbol\theta_t}
=w_{ti}\left(\frac{\partial\mathbf h}{\partial\boldsymbol\theta_t}
+\mathbf t\,\frac{ds_{ti}^\ast}{d\boldsymbol\theta_t}\right).$$

在**精确**的内部最近点处 $\mathbf r_{ti}^{\mathsf T}\mathbf t=0$，于是

$$\mathbf r_{ti}^{\mathsf T}\,\mathbf t\,\frac{ds_{ti}^\ast}{d\boldsymbol\theta_t}=0,$$

路径项对目标的一阶导数 $g=J^{\mathsf T}r$ 无贡献。但这**不意味着**两个 Jacobian 相同，也不意味着两者的 $J^{\mathsf T}J$ 相同——$H$ 是 $J$ 的二次型，路径项不与 $r$ 缩并时不会消失。因此 seed 的驻点指标是"冻结路径目标"的二阶量，不是完整最近点目标的精确二阶量。（此外 $s^\ast$ 只是有限次 Newton 的估计，$\mathbf r^{\mathsf T}\mathbf t=0$ 本身也只近似成立。）

**Gauss–Newton 正规方程。** 在 $\boldsymbol\theta$ 处线性化 $r(\boldsymbol\theta+\Delta)\approx r+J\Delta$，得二次模型

$$m(\Delta)=\tfrac12\lVert r+J\Delta\rVert^2
=F+g^{\mathsf T}\Delta+\tfrac12\Delta^{\mathsf T}H\Delta,
\qquad g=J^{\mathsf T}r,\quad H=J^{\mathsf T}J.$$

令 $\nabla_\Delta m=g+H\Delta=0$ 得 $H\Delta=-g$。$H$ 是 GN 近似，精确 Hessian 为

$$\nabla^2F=J^{\mathsf T}J+\sum_j r_j\nabla^2 r_j,$$

丢弃的是残差加权二阶项；在小残差或近线性情形该项可忽略。

**LM 阻尼。** 对 5 维优化变量定义

$$\alpha_t=\max\!\Big(\tfrac15\operatorname{tr}H_t,\ 10^{-30}\Big),
\qquad M_t=\alpha_t I_5,$$

$$(H_t+\mu_t M_t)\,\Delta_t=-g_t,
\qquad \boldsymbol\theta_t^{\rm cand}=\boldsymbol\theta_t+\Delta_t.$$

阻尼形式是**平均对角元乘单位阵**，不是 $\operatorname{diag}(H)$，因此各分量受到相同的绝对阻尼；$\alpha$ 的 $10^{-30}$ 下限只防全零 $H$。

**候选接受。** 记当前目标 $F_t$、候选目标 $F_t^{\rm cand}$，接受当且仅当该径迹行

$$\underbrace{\neg\,\text{stationary}_t}_{\text{仍活跃}}\ \land\
\underbrace{\text{finite}(F_t^{\rm cand})}_{\text{数值有效}}\ \land\
\underbrace{F_t^{\rm cand}<F_t}_{\text{严格下降}}\ \land\
\underbrace{\mathcal E_t(\boldsymbol\theta_t^{\rm cand})\supseteq\mathcal E_t(\boldsymbol\theta_t^{\rm init})}_{\text{初始有效点不失效}}.$$

最后一条把有效集合锚定在**初始**集合上，防止 LM 通过让难拟合的 hit 失效来降低目标。

$$\boldsymbol\theta_t^{\rm next}=
\begin{cases}\boldsymbol\theta_t^{\rm cand},&\text{接受},\\[2pt]
\boldsymbol\theta_t,&\text{拒绝},\end{cases}
\qquad
\mu_t^{\rm next}=
\begin{cases}\max(0.1\,\mu_t,\ 10^{-12}),&\text{接受},\\[2pt]
10\,\mu_t,&\text{拒绝}.\end{cases}$$

初值 $\mu_t=10^{-8}$。$\mu$ 只对活跃（非驻点）行更新。这是**自适应阻尼**策略，没有用实际／预测下降比构造完整的信赖域接受规则。

**停止判据。** 每轮在求步之前先算 decrement 二次型

$$\nu_t=g_t^{\mathsf T}\big(H_t+\epsilon_\mu M_t\big)^{-1}g_t,
\qquad \epsilon_\mu=10^{-12},$$

$$\nu_t^{\rm rel}=\frac{\nu_t}{\max(F_t,\ 10^{-12})}.$$

逐径迹的驻点判据为

$$\text{stationary}_t=\text{finite}(\nu_t)\ \land\
\Big[\ \nu_t\le10^{-12}\ \ \lor\ \ \nu_t^{\rm rel}\le10^{-6}\ \Big].$$

全块所有行都满足才退出外层循环；最大迭代 $60$ 轮。

> $\nu$ 的读法。教科书的 Newton decrement 是 $\sqrt{g^{\mathsf T}H^{-1}g}$；这里记录的是**未开方的二次型**。它也不等于二次模型的预测下降量，后者为 $m(0)-m(\Delta)=\tfrac12 g^{\mathsf T}H^{-1}g$（取 $\Delta=-H^{-1}g$）。$\nu$ 恰是其两倍。不要把 $\nu$ 读成信赖域的 gain ratio。

**顺序。** 每轮的正确顺序是：求 $s^\ast$ → 组装 $r,J$ → 算 $H,g$ → 算 $\nu$ 并判驻点 → 若未全部驻点则求步、评候选、接受或拒绝、更新 $\mu$。**判驻点在求步之前**，原图把收敛判断画在最近点与残差计算之前是顺序错误。

**终态门限。** 循环停止后重新评估一次终态 $\nu_t$，输出非驻点行比例与非有限计数。主线要求

$$\text{n\_non\_finite}=0,
\qquad
\frac{\lvert\{t:\neg\text{stationary}_t\}\rvert}{T}\le10^{-3}.$$

非有限 decrement 立即失败；非驻点比例超限报错。注意主线**容许最多 0.1% 的非驻点行**，不是要求全部径迹达标。驻点阈值 $10^{-6}$ 低于 float32 分辨率，该评估须在 float64 下进行。

*实现对应：* `_single_point_fit_residual`、`_BATCHED_POINT_FIT_JACOBIAN`、`damped_gauss_newton`、`require_stationary_helix_point_fit`（`scripts/helix_model.py`）。

---

## 4. 附：与代码的对应关系

| 本文符号 | 代码变量 | 位置 |
|---|---|---|
| $\boldsymbol\eta$ | `track` / `reference` | `helix_model.py` |
| $\boldsymbol\theta$ | `theta` | `helix_model.py` |
| $\mathbf h,\mathbf t$ | `helix_point`, `helix_tangent` | `helix_model.py` |
| $s^\ast$, $w$ | `path`, `effective`(= `mask & valid`) | `_single_point_fit_residual` |
| $H,g$ | `hessian`, `gradient` | `damped_gauss_newton` |
| $\alpha$ | `scale` | `damped_gauss_newton` |
| $\mu$ | `level`（per-track `[T,1,1]`） | `damped_gauss_newton` |
| $M$ | `scale[:,None,None] * identity` | `damped_gauss_newton` |
| $\nu,\ \nu^{\rm rel}$ | `decrement`, `relative_decrement` | `damped_gauss_newton` |
| $\kappa$ | `CURVATURE_SCALE` = `1e-4` | `helix_model.py` |
| $S_{\max}$, $\varepsilon_c$ | `S_MAX`, `CLOSEST_POINT_CURVATURE_EPS` | `helix_model.py` |
| $\epsilon_\mu$ | `GAUSS_NEWTON_MIN_DAMPING` = `1e-12` | `helix_model.py` |

### 三个易踩的坑

1. **同名常量不同值。** `GAUSS_NEWTON_TOLERANCE` 在 `helix_model.py` 是 `1e-6`（seed 的相对 decrement 阈值），在 `validate_helix_model.py` 是 `1e-10`（refit 的 change 阈值）。引用时必须带模块名。
2. **`converged` 语义不同。** seed 的 `converged` 来自全行 decrement 驻点判据；refit 的来自"相对下降量 `change < 1e-10`"。**refit 的 `change` 为零也可能是候选全被拒绝**，表示迭代停止，不表示通过驻点验收——refit 须在循环停止后做独立的终态 decrement 检查。
3. **残差符号相反。** seed 残差是预测减测量 $\mathbf h-\mathbf p$（三维全局坐标，长度 $3N_h$）；refit 残差是测量减预测且按方差白化（二维局部坐标，长度 $2N_h$）。$F=\tfrac12\lVert r\rVert^2$ 不受符号影响，但 $g=J^{\mathsf T}r$ 的解读会反。

---

## 核验说明

§1.4 的数值结果由一次性核验脚本产生（$N=2000$ 组随机参数，float64，以 `helix_model.helix_point` 为基准，一二阶导以 `torch.autograd` 为基准）。脚本不入库。本文不含算法实验或性能结论。
