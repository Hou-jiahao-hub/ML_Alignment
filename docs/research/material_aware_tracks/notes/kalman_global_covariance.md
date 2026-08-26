# Kalman 全局协方差与材料过程噪声

**主要来源：** [@hulsbergen2009covariance]、[@actsTracking]

## 文献事实

Kalman track fit 把相邻状态写成带过程噪声的传播关系，measurement 再通过测量模型
约束当前状态。材料散射因此不只是增加单点残差方差，还会在不同轨迹状态之间产生
相关性。ACTS 的概念文档用状态转移、过程噪声和 measurement noise 描述这一结构。
[@actsTracking]

Hulsbergen 给出了从 Kalman fit 构造整条轨迹状态全局协方差的方法，并说明该协方差
可用于 minimum-χ² detector alignment。关键不是复用某一个 smoothed state，而是
保留不同 measurement 状态由共同轨迹参数和过程噪声产生的相关性。
[@hulsbergen2009covariance]

## 对当前项目的含义

- 当前单 helix 联立优化包含共享轨迹参数造成的相关性，但没有材料过程噪声自由度，
  因而会把真实随机偏折留在 measurement residual 中。
- 仅使用 ACTS 导出的 `eLOC*_smt` 或其对角误差不能重建完整跨状态协方差；项目
  ROOT 文件也没有导出完整状态协方差矩阵。
- 两条可行路线是：实现可微 Kalman/smoother，或显式引入带先验的局部 kink。后者
  更贴合现有“轨迹参数 + 对齐参数”联合目标，同时避免在 core 中复刻全部 ACTS
  传播器。
- holdout pull 的分母应包含 measurement covariance 和由已拟合局部参数传播出的
  prediction covariance。对 MAP 拟合，可在固定 geometry 的验证阶段从局部
  Gauss–Newton Hessian 近似获得。

## 与 GBL 的关系

在线性、高斯且使用相同传播与材料假设时，GBL 和 Kalman/smoother 是同一类统计
问题的不同数值组织方式；它们都不能靠忽略跨状态相关性得到正确的 alignment
权重。实际实现是否数值一致取决于线性化点、传播 Jacobian、材料离散化和边界
条件，必须用同一 synthetic trajectory 做交叉验证。

## 局限性

- 局部 Hessian 逆要求处理弱模和数值条件数；阻尼值必须固定并作为验证配置记录。
- Laplace/Gauss–Newton covariance 是局部高斯近似，不保证描述大角散射尾部。
- ACTS smoothed state 可用于初始化和诊断，但若被当作 training label，会重新引入
  对 ACTS fit 的依赖和潜在信息泄漏。

