# Sensor block-LM 数学推导：6 页紧凑审阅版

用途：审阅与复盘。正文与 PPT 同序；补充说明列于每页之后。与 [seed](../seed_refit/seed_derivation_compact.md)、[refit](../seed_refit/refit_derivation_compact.md) 六页稿衔接；以当前 `scripts/ml_alignment_autograd.py` 的 sensor block-LM 为准。径迹固定，优化目标 sensor 位姿。


## 1. 符号衔接与 sensor 分块


### 沿用符号，区分径迹变量与几何变量


| 符号 | 定义 · 维度 / 单位 |
|---|---|
| η；θ；h；t | 六维 / 五维径迹参数；位置；路径导数 |
| cⱼ；Rⱼ；nⱼ | sensor 中心；局部→全局旋转；法向 |
| mᵢ；m̂ᵢ；Vᵢ,ₐ | 局部测量；预测；分量方差 |
| r；F；J | 标准化残差；½χ²；本阶段 Jacobian |
| H；g；Δ | JᵀJ；Jᵀr；归一化坐标的候选步 |
| μ；α；M | 阻尼水平；阻尼尺度；αI |
| ν | decrement 二次型；本阶段按自由集计算 |
| i, j；a；k | hit、sensor；LOC 分量；外层访问轮次 |


### 新增位姿与约束符号


| 符号 | 定义 · 维度 / 单位 |
|---|---|
| ξⱼ；xⱼ；zⱼ | 六维位姿修正；选定 DOF；归一化坐标 |
| Sⱼ；dⱼ | 正对角物理尺度；选定 DOF 数 ≤ 6 |
| Eⱼ；Nⱼ；Uⱼ | 固定比较分量集；其大小；自由变量集 |
| βⱼ；εeig | 谱下限补偿；特征值比例阈值 10⁻⁴ |


### 1  算法入口：refit 径迹固定

主线传入非目标层 refit 的 η，并停止梯度。

仅优化 target sensor 的位姿；局部测量保持固定。


$$F(\{\boldsymbol\xi_j\})=\sum_j F_j(\boldsymbol\xi_j),\quad F_j=\tfrac12\|\mathbf r_j\|^2$$


### 2  为什么可以逐 sensor 独立求解


$$\frac{\partial\mathbf r_j}{\partial\boldsymbol\xi_l}=0\quad(j\ne l)$$


$$H_{\rm all}=\operatorname{blockdiag}(H_1,H_2,\ldots)$$

一条径迹可穿过多个 sensor；径迹已固定，某块

残差只依赖本块位姿，因此当前目标精确解耦。


### 3  与前两阶段的关系

ξ 不复用径迹 θ；r 仍为“测量减预测”并标准化。

独立块各自维护 μ；驻点块可退出后续访问。

解耦成立于固定径迹、无跨 sensor 约束的本模型。


**补充说明**

本稿把代码 lambda2 统一记为 ν，避免将其误解为 LM 阻尼或倾角。μ 始终表示自适应阻尼；α、M 保持阻尼尺度角色，但 block-LM 的具体定义不同于 seed/refit 的平均对角元。J、H、g 均指归一化坐标 z 下的量，物理坐标导数显式标为 J_phys。Eⱼ 收集 sensor j 上的 (i,a) 有效分量；i 可汇总多条固定径迹的 hit，hᵢ 表示该 hit 对应的固定 Helix。函数 run_block_lm_alignment 在未传 initial_track_params 时会回退到 seed；这里描述已传入 refit 结果的主线流程，不能将该回退声称为自动执行 refit。默认 active_dof 从配置读取，不硬编码为六自由度。


## 2. 位姿参数化与固定集合目标


### 1  相对输入几何的局部平移与 Euler 旋转


$$\boldsymbol\xi_j=(d_u,d_v,d_w,\omega_x,\omega_y,\omega_z)$$


$$\mathbf c_j=\mathbf c_j^0+R_j^0\mathbf d_j,\quad \mathbf d_j=(d_u,d_v,d_w)^{\mathsf T}$$


$$R_j=R_j^0 R_z(\omega_z)R_y(\omega_y)R_x(\omega_x)$$


$$\mathbf n_j=R_j\mathbf e_z,\qquad\mathbf e_z=(0,0,1)^{\mathsf T}$$

平移沿输入面的局部轴；旋转右乘输入 R⁰。

优化的是相对输入几何的累计修正参数，不是每步

向当前旋转矩阵反复右乘一个小旋转。


### 2  改变平面后必须重新求交


$$\phi_{ij}(s,\boldsymbol\xi_j)=\mathbf n_j^{\mathsf T}[\mathbf h_i(s)-\mathbf c_j]=0$$


$$\widehat{\mathbf m}_{ij}=PR_j^{\mathsf T}[\mathbf h_i(\hat s_{ij}^{\cap})-\mathbf c_j]$$


### 3  残差与本次访问的比较集合


$$\widetilde r_{ij,a}=\frac{m_{i,a}-\widehat m_{ij,a}}{\sqrt{\max(V_{i,a},10^{-8})}}$$


$$\mathcal E_j^{(k)}=\{(i,a):\text{current valid component}\}$$


$$\mathbf r_j=\operatorname{col}_{(i,a)\in\mathcal E_j^{(k)}}(\widetilde r_{ij,a}),\quad N_j=|\mathcal E_j^{(k)}|$$


$$F_j=\tfrac12\mathbf r_j^{\mathsf T}\mathbf r_j,\quad \chi_j^2=2F_j$$

只选本 target sensor 的真实 hit；LOC1 还需其

可用标记。方差下限单位 mm²，r 与 F 无量纲。


### 4  目标固定到哪一层

Eⱼ 在本次线性化及全部阻尼重试中保持不变。

候选可增加有效分量，但不得丢失 Eⱼ 中的分量。

下次访问重新定义 Eⱼ，集合可增长，故跨轮历史

目标不保证单调；不能以删掉难拟合点换取下降。


**补充说明**

R_z R_y R_x 使用标准右手旋转，参数顺序对应代码三平移后三旋转。上标 0 是输入工作几何，不必等同于 nominal target 真值；本块残差不使用 nominal 黑点与预测点的全局距离。P 与 refit 相同，取局部三维坐标前两分量。求交复用 12 次 Newton、近平行阈值 10⁻⁶、路径界 1200 mm、面残差阈值 10⁻³ mm，不重复推导。Eⱼ 是每次访问当前有效分量的快照，与 seed 的初始有效集合锚点不同，也与 refit 内层动态比较集合不同。


## 3. 移动测量面时的几何 Jacobian


### 1  固定径迹，位姿参数改变平面

令 xⱼ 为选定的物理位姿变量，本页对其第 b 个

分量求导；η 固定，hᵢ 对该分量的显式偏导为零。


$$0=\partial_b\mathbf n_j^{\mathsf T}(\mathbf h_i-\mathbf c_j)+\mathbf n_j^{\mathsf T}(\mathbf t_i\partial_b s-\partial_b\mathbf c_j)$$


$$\frac{\partial s}{\partial x_b}=\frac{\mathbf n_j^{\mathsf T}\partial_b\mathbf c_j-(\partial_b\mathbf n_j)^{\mathsf T}(\mathbf h_i-\mathbf c_j)}{\mathbf n_j^{\mathsf T}\mathbf t_i}$$

要求 nⱼᵀtᵢ ≠ 0；所有量在同一精确交点处评估。

refit 改变曲线；此处改变中心、旋转与法向。


### 2  局部预测同时包含坐标系与交点变化


$$\partial_b\widehat{\mathbf m}_{ij}=P\left[(\partial_bR_j)^{\mathsf T}(\mathbf h_i-\mathbf c_j)\right.$$


$$\left.\qquad+R_j^{\mathsf T}(\mathbf t_i\partial_b s-\partial_b\mathbf c_j)\right]$$


### 3  位姿偏导与标准化残差的符号


$$\partial_b\mathbf n_j=(\partial_b R_j)\mathbf e_z$$

平移变量：∂c/∂dᵤ = R⁰eᵤ，∂R/∂dᵤ = 0。

旋转变量：∂c/∂ω = 0，∂R/∂ω 按乘积次序求导。


$$J_{j,\rm phys}=-\operatorname{col}_{\mathcal E_j}\left(W_i\frac{\partial\widehat{\mathbf m}_{ij}}{\partial\mathbf x_j}\right)$$

负号与 refit 一致：测量减预测。Wᵢ 为相同的

对角标准差逆矩阵，仅抽取 Eⱼ 的分量行。


### 4  实际求导路径与理论边界

xⱼ → 六维位姿行 → cⱼ、Rⱼ、nⱼ

→ 12 次 Newton 求交 → 局部预测 → rⱼ

→ autograd.functional.jacobian。

路径参与求导；固定集合与布尔有效性不连续求导。

有限迭代 / 分支边界处不保证等于精确根隐式导数。


**补充说明**

本页 ∂_b 表示对选定物理位姿变量 x_b 的偏导；它不同于对固定路径的径迹 θ 偏导。由 nᵀ(h(s(x))−c(x))=0 链式展开即可得到路径导数，分子两项分别来自中心移动和法向转动。局部预测还包含 ∂Rᵀ 项，不能只保留交点平移项。旋转偏导例如 ∂R/∂ω_x=R⁰R_zR_yR_x′，其余角在相应因子处求导；平移与旋转均相对固定输入几何定义。J_phys 的 col_E 记号表示先堆叠各 hit 两行，再只保留固定分量集合的行，不是把无效分量添零。实现中 fixed_mask 是常量索引，但有限次求交内部仍有 where 数值分支；隐式公式只作数学解释，不声称代码用了解析隐式求导。


## 4. 参数尺度与 block-LM 正规方程


### 1  物理位姿变量 → 无量纲优化坐标


$$\mathbf x_j=S_j\mathbf z_j,\quad S_j=\operatorname{diag}(s_b)_{b\in\mathrm{active}}$$

全六维尺度为 (1, 1, 1 mm；0.01, 0.01, 0.01 rad)，

按 active_dof 选取；Sⱼ 严格正对角。


$$J_j=J_{j,\rm phys}S_j,\quad H_j=J_j^{\mathsf T}J_j,\quad\mathbf g_j=J_j^{\mathsf T}\mathbf r_j$$


$$m_j(\Delta)=F_j+\mathbf g_j^{\mathsf T}\Delta+\tfrac12\Delta^{\mathsf T}H_j\Delta$$

J 的列尺度乘法就是链式法则；H、g、Δ 均在 z

坐标中。实现的 fisher 是该 GN 矩阵，不是真 Hessian。


### 2  本阶段的谱保护与阻尼尺度


$$\overline H_j=\tfrac12(H_j+H_j^{\mathsf T}),\quad e_{\min},e_{\max}=\text{extreme eigenvalues}$$


$$\beta_j=\max\{0,\varepsilon_{\rm eig}\max(e_{\max},0)-e_{\min}\}$$


### 3  LM 方程：谱下限补偿 + 自适应阻尼


$$\alpha_j=\max(e_{\max},1),\qquad M_j=\alpha_j I_{d_j}$$


$$(\overline H_j+\beta_j I_{d_j}+\mu_jM_j)\Delta_j=-\mathbf g_j$$


$$\Delta\mathbf x_j=S_j\Delta_j$$

εeig 默认 10⁻⁴，μ 初值 10⁻³；每个 sensor 独立维护。

这里 α 取最大特征值尺度，不是 seed/refit 的平均

对角元；β 为本阶段额外的谱下限补偿。


### 4  求步与诊断使用不同矩阵

求步对全部选定 active DOF 解正则化方程；

自由集只用于驻点 / KKT 检查，不缩减此求步系统。

solve 失败时回退 lstsq，并记录秩亏证据。

阻尼可使方程可解，却不能恢复数据缺少的约束；

不能用加大阻尼后的低 decrement 证明收敛。


**补充说明**

e_min、e_max 是对称 GN 矩阵的最小与最大特征值，与 seed 的法平面单位向量 e₀、e₁（粗体）区分。通过 z=S⁻¹x，模型所有列按实际物理尺度归一化；矩阵求解使用 torch.linalg.solve 而非显式求逆。β 抬升最小特征值至相对谱下限，μ αI 另外控制步幅，这与平均对角阻尼 M 的角色相同但具体定义不同。无正则化且满秩时 Δ=−H⁻¹g 最小化二次模型，其推导不重复。任何 lstsq 回退计数在最终状态中优先标为 rank_deficient，即使后续步可以接受。


## 5. 候选限制、固定集接受与阻尼重试


### 1  步长上限与可行域边界分别处理


$$\Delta\mathbf x_j^{\rm cap}=\operatorname{clip}(S_j\Delta_j,-\Delta_{\max},\Delta_{\max})$$


$$\mathbf x_j^c=\operatorname{clip}(\mathbf x_j+\Delta\mathbf x_j^{\rm cap},\mathbf l_j,\mathbf u_j)$$

逐分量步长上限：平移 0.1 mm，旋转 5×10⁻³ rad。

累计硬边界：平移 ±1.5 mm，旋转 ±0.05 rad。

步长上限不进入 KKT；硬边界进入自由集判定。


### 2  重新求交，在同一个 Eⱼ 上评分


$$(\chi_j^2)^c_{\mathcal E_j}< (\chi_j^2)_{\mathcal E_j},\quad\mathcal E_j\subseteq\mathcal E_j^c$$


$$\operatorname{finite}((\chi_j^2)^c_{\mathcal E_j}),\quad q_j^c\le\max(q_j,q_{\max})$$

qⱼ 为本 sensor 的求交失效率，qmax 默认 5%。

目标严格下降、保留原有效分量、失效率与有限值

检查全部通过才接受；拒绝则当前位姿不变。


### 3  接受后减半；拒绝后沿绝对阶梯增大


$$\text{accept}:\quad\mu_j^+=\max(\mu_j/2,10^{-12})$$


$$\mu_{\rm anchor}=\max(\mu_{\rm init},10^{-3})$$


$$\text{retry }b:\quad\mu_j^+=\min\{\max(10\mu_j,\mu_{\rm anchor}10^b),10^{12}\}$$

b = 1,…,8；最多 1 次初试 + 8 次重试。

拒绝时不重算 J；改变正则项后重新解步、求交评分。


### 4  一次块访问的结果

接受：提交 ξ，保留减小后的 μ，下一轮重新线性化。

重试耗尽：保留 ξ，记录失败原因并退出该块。

谱秩亏本身不会立即退出，仍可尝试正则化求步。

采用分量限步与阻尼阶梯，不计算 gain ratio。

新加入的有效分量在下一次线性化进入比较集合。


**补充说明**

Δxⱼᶜᵃᵖ 是截断后的物理步，Δmax 表示逐分量正步长上限；lⱼ、uⱼ 为累计修正硬边界。candidate 对全部六维行执行硬截断，未选 DOF 保持初始零值。失效率条件锚定 max(当前本块失效率,全局配置阈值)，不是每个块都无条件要求候选 ≤5%。阶梯锚点防止接受多次后 μ 极小，8 次乘十仍不足以试探合适阻尼；达到 10¹² 可提前结束重试。最大预算默认 60 轮，若全部块已退出则提前停止。代码每块每次接受即结束此次阶梯。


## 6. 自由集、驻点指标与终态分类


### 1  硬边界上的一阶 KKT 条件


$$\mathcal U_j=\{b:\neg[(x_b\ge u_b-\epsilon_b\land g_b<0)$$


$$\qquad\lor(x_b\le l_b+\epsilon_b\land g_b>0)]\}$$

εb = 10⁻¹²；边界上负梯度指向可行域外才冻结。

g_b = 0 不冻结；物理尺度正，归一化不改变梯度符号。


### 2  自由集谱分解，诊断不加阻尼


$$\overline H_{\mathcal U}=Q\operatorname{diag}(e_b)Q^{\mathsf T},\quad\mathbf a=Q^{\mathsf T}\mathbf g_{\mathcal U}$$


$$\nu_j=\sum_{e_b>\varepsilon_{\rm eig}\max(e_{\max,\mathcal U},0)}\frac{a_b^2}{e_b}$$

存在弱模即秩亏：截去弱模后的 ν 仅为保守下界，

不能凭小 ν 判收敛；满秩时才等于 gᵁᵀHᵁ⁻¹gᵁ。


$$\mathcal U_j=\varnothing\quad\text{or}\quad\nu_j/N_j\le10^{-6}$$


### 3  判据与状态的优先级

仅非秩亏块满足左下判据才按驻点退出。

自由集为空意味着箱约束 KKT 条件成立；其 ν = 0。

阈值按有效分量归一化，不是 refit 的逐轨绝对 ν。


| 情况 | 状态与含义 |
|---|---|
| 无 target hit | no_data；不进入有数据块统计 |
| 求交失败 | intersection_failed；无有效点 / 重试丢点 |
| 秩亏证据 | rank_deficient；弱模 / 曾用 lstsq |
| 驻点 / KKT | converged；在上述优先级之后判定 |
| 剩余情况 | budget_exhausted；轮数 / 重试耗尽 |


### 4  最后一次记录不等于重新验收终态

默认最多 60 轮；最后一轮接受后，代码不额外重算

该块的终态 ν。预算耗尽记录可能来自接受前线性化。

最终只另评总体残差与求交失效率；返回当前几何，

不回滚到 best_iteration。

驻点说明局部模型难以继续下降，不证明几何真值恢复。


**补充说明**

本页 a=Qᵀg_U 仅为谱坐标向量（粗体），与测量分量索引 a 不同；e_b 为自由集 GN 特征值。若存在零 / 负 / 低于相对阈值的特征值，完整逆矩阵指标可能无定义，实际 ν 应理解为保留强模的部分和，不能作为收敛证据；“下界”沿用代码诊断用语而不是对不存在的逆作断言。U 为空时返回 ν=0、rank_deficient=False；有自由变量但 g=0 仍检查秩，不把零梯度隐藏的弱模判满秩。自由集用于一阶箱约束判定，步求解仍在全部 active DOF 上，随后限步和投影。状态优先级为 no_data→intersection_failed→rank_deficient→converged→budget_exhausted；求交失败标志只在整条重试阶梯未获接受时汇总，不是曾出现一次丢点候选就永久失败。lstsq 回退证据则为累计记录。无 hit 与有 hit 但零有效分量必须区分。最终总体失效率按所有真实 hit 统计（并非仅 target），初始与最终均受配置阈值约束；本块候选失效率仅针对本 sensor。CPU 默认 float64，GPU 路径使用 float32，不继承 refit 内部强制 float64 的约定。


## 共用原理与实现核对

- 不重复 Helix / sinc、Newton 基础或 GN 二次模型推导。几何变换、移动平面隐式导数、参数尺度、谱正则、固定集合比较及边界驻点为本稿重点。
- 与 seed/refit 一致使用 r、F、J、H、g、Δ、μ、α、M、ν；每个阶段的具体目标、求导变量及阻尼尺度分别定义。ξ 用于几何，θ 专用于径迹。
- 公式为原生 OMML，含同式高清图像兼容回退；六页补充说明同步保存为备注。未修改任何算法，未运行算法实验或提出性能结论。

| 环节 | `scripts/ml_alignment_autograd.py` 对应函数 |
|---|---|
| 位姿约定 | `apply_alignment_torch`, `torch_euler_to_rotation_matrix` |
| sensor 目标与固定比较集 | `_sensor_residual_data`, `_block_lm_fixed_chi2` |
| Jacobian 与物理尺度 | `_block_lm_linearized_block` |
| 谱保护、LM 与重试 | `regularize_fisher_matrix`, `_block_lm_retry_damping`, `run_block_lm_alignment` |
| 边界、驻点与分类 | `_block_lm_free_dof_mask`, `_block_lm_decrement`, `_block_lm_block_status` |

### 复盘时的完整顺序

固定 refit 径迹 → 按 sensor 选择 target hit → 当前位姿求交并固定分量集 → 求标准化残差和归一化 J、H、g → 自由集 / 谱秩 / ν 检查 → 满足条件则退出该块，否则求正则化步 → 物理限步与硬边界投影 → 重算候选求交 → 在原集合上评分和检查不丢分量 → 接受并减半阻尼，或提高阻尼复用 J 重试 → 下次访问重线性化 → 汇总状态并检查最终总体失效率。
