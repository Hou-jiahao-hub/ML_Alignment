# Broken Lines 与 General Broken Lines

**主要来源：** [@blobel2006brokenlines]、[@kleinwort2012gbl]

## 文献事实

Broken Lines 将轨迹表示为一条参考轨迹及沿途离散散射点上的局部修正。General
Broken Lines（GBL）把磁场弯曲、能损和材料多重散射纳入同一线性化框架，并用
散射点处的局部偏移或等价 kink 描述随机轨迹扰动。[@blobel2006brokenlines;
@kleinwort2012gbl]

measurement 提供位置约束，scatterer 提供零均值、有限协方差的 kink 约束。由于
大多数局部参数只影响相邻少数区段，法方程具有带状或 bordered-band 结构，计算
量可随 measurement 数近似线性增长。GBL 还能给出沿轨迹的协方差，因而适合向
全局对齐提供局部拟合和导数。[@kleinwort2012gbl]

GBL 的“broken”不是让每段完全自由：若没有 scatterer covariance，局部方向变化
会失去物理代价，轨迹可吸收任意 measurement 或几何偏移。模型可识别性来自
measurement 约束、传播 Jacobian、散射先验和外部/gauge 约束的组合。

## 状态与目标函数视角

对参考轨迹参数 `q`、散射 kink `k_j` 和 measurement residual `r_i`，本项目可采用
与 GBL 相同的概率结构：

\[
\chi^2(q,k)=\sum_i r_i(q,k)^T V_i^{-1}r_i(q,k)
+\sum_j k_j^T Q_j^{-1}k_j.
\]

文献中的高效解法依赖围绕参考轨迹的线性化和专用矩阵结构。本项目计划使用
PyTorch 自动微分和批处理优化；二者共享统计模型，但数值算法不能未经验证地宣称
等价。

## 对当前项目的含义

- 保留现有六参数 helix 作为 reference trajectory，避免丢失已验证的磁场弯曲和
  surface intersection 实现。
- 在 ACTS 报告的材料节点加入二维局部角 kink，而不是为每个 hit 独立增加自由
  位移。
- kink prior 必须使用逐轨迹 `x/X0` 和动量；不能把散射尺度作为优化器降低 χ²的
  逃逸方向。
- 节点局部性适合 padded track batch，也为未来的带状求解器或 Schur 消元保留
  结构。
- nominal holdout 的完整预测协方差可由局部 Hessian/Laplace 近似计算；这比只看
  measurement variance 更接近 GBL 的协方差用途。

## 局限性

- 参考轨迹偏离真实轨迹过大时需要重新线性化；非线性 autograd 版本仍需 checkpoint
  和有限交点保护。
- Highland 高斯先验不能完整描述大角散射尾部。
- 未经约束的 kink 会与局部几何旋转产生相关性；必须用 synthetic closure 和
  misaligned alignment 的 paired 对照确认没有吸收对齐信号。
- 文献的线性复杂度来自专用带状代数；直接 Adam 优化不会自动获得该复杂度。

