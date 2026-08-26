# 全局对齐中的 GBL 与 Millepede

**主要来源：** [@blobel2011fastalignment]、[@kleinwort2012gbl]

## 文献事实

track-based alignment 同时包含每条轨迹独有的 local parameters 和所有轨迹共享的
global geometry parameters。Millepede-II 类方法利用局部参数的块结构进行消元，
构造只含全局参数的方程。[@blobel2011fastalignment]

在复杂硅径迹器中，忽略多重散射相关性会给局部轨迹模型和全局对齐权重带来错误。
GBL 提供包含散射的局部轨迹重拟合，同时保持适合消元的稀疏/带状结构；这正是其
与大规模对齐结合的主要价值。[@blobel2011fastalignment; @kleinwort2012gbl]

## 对当前项目的映射

| 文献概念 | 当前仓库对象 | 升级后的对象 |
|---|---|---|
| global parameters | 目标 surface 的 alignment tensor | 保持不变 |
| local reference parameters | 每轨迹六参数 helix | 保持为 reference helix |
| scatterer local parameters | 当前不存在 | 每材料节点二维 kink |
| measurement covariance | LOC0/LOC1 方差和 mask | 保持现有 component contract |
| material covariance | 当前不存在 | 由 `x/X0`、动量和 muon 假设构造 |
| local elimination | Adam 联合优化 | 首版仍联合优化，保留未来 Schur 接口 |

## 识别性与约束

- 非目标 geometry 固定，继续承担 gauge/reference 作用。
- material kink 是零均值随机变量，不是无约束的附加 alignment DOF。
- geometry correction 与 kink 的相关性必须通过 truth-known toy closure 检验：材料
  存在时仍应恢复注入的 surface 位移/旋转；冻结轨迹控制组应呈现预期偏差。
- 不能用 nominal truth 选择 production stage 或 polish 候选。nominal geometry 只在
  模型 holdout 和候选冻结后的诊断中使用。

## 计算策略

首版沿用 autograd/Adam 是为了最小化架构变更并验证物理模型。若统计模型通过而
性能不足，再利用每轨迹局部 Hessian 做 Schur complement 或实现 bordered-band
求解器。不能在物理闭合前同时更换轨迹模型、求解器和 active DOF，否则无法归因
改善来源。

