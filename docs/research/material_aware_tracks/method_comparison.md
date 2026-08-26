# 候选方法比较

## 比较对象

### A. GBL 风格参考 helix + 材料 kink

保留当前六参数 helix 作为 reference trajectory，在 ACTS 导出的材料节点引入二维
方向 kink，用 `x/X0` 和动量确定零均值高斯先验。measurement residual 与 scattering
penalty 联合优化。统计结构来自 GBL，但首版数值实现继续使用 PyTorch autograd。
[@kleinwort2012gbl]

### B. 纯 PyTorch 可微 Kalman filter/smoother

在每个 surface 传播局部 track state 和 covariance，用材料 process noise 更新预测
协方差，再做 measurement update 和 smoothing。该方法直接产生状态协方差，但要求
在 core 中实现传播 Jacobian、协方差运输、材料更新和 smoothing。
[@hulsbergen2009covariance; @actsTracking]

### C. 经验分段 helix

按 measurement 或固定路径间距切段，为每段增加自由方向或曲率参数，使用统一 L2
或按 hit 间距设置经验正则，不读取材料图。

## 决策矩阵

评分为 1（差）到 5（优）。分数表示对当前仓库目标的适配度，不是方法的一般排名。

| 维度 | A: GBL kink | B: 可微 Kalman | C: 经验分段 | 判断依据 |
|---|---:|---:|---:|---|
| 多重散射物理正确性 | 5 | 5 | 2 | A/B 显式使用过程噪声；C 缺少材料尺度。 |
| 与现有 autograd 联立对齐兼容 | 5 | 3 | 4 | A 在现有 local/global 参数结构上增量扩展。 |
| 防止轨迹吸收几何错位 | 4 | 4 | 1 | A/B 有材料 covariance；C 的自由 kink 易退化。 |
| core 无 ACTS 依赖 | 5 | 4 | 5 | A 通过 adapter artifact 隔离；B 需复刻更多 ACTS 逻辑。 |
| predictive covariance | 4 | 5 | 2 | A 可由局部 Hessian 得到；B 原生传播；C 难物理解读。 |
| 首版实现与验证成本 | 4 | 1 | 5 | B 改写范围最大；C 快但证据价值低。 |
| GPU padded batch 可行性 | 4 | 3 | 5 | A 节点局部且可 padding；B 有顺序状态递推。 |
| 与全局对齐文献一致性 | 5 | 4 | 1 | GBL 已直接用于 Millepede/CMS。[@blobel2011fastalignment] |
| 当前 μ 子样本适配性 | 5 | 4 | 2 | A 聚焦高斯核心散射；B 能力超出首版需求。 |
| 总分 | **41/45** | **33/45** | **27/45** | A 为当前阶段首选。 |

## 关键权衡

### 为什么不先实现可微 Kalman

Kalman/smoother 在状态协方差输出上最自然，也能在相同线性高斯假设下表达与 GBL
相同的物理问题。但当前仓库已有可微 helix、surface intersection、padded tracks 和
联合 geometry optimization；为首版重写完整 covariance transport 会同时改变物理
模型和数值架构，难以定位失败。A 可以先验证“材料过程噪声是否解决下限”，以后再
决定是否需要 B 的顺序求解器。

### 为什么不把经验分段当作正式模型

C 可以快速降低 training residual，但没有逐节点物理 covariance 时，优化器无法
区分随机散射、坏 hit 和 geometry displacement。它可能复现历史上的“训练指标下降、
重建或 truth recovery 无改善”。C 只适合作为标记为 throwaway 的上限/消融实验，
不能成为 production candidate。

### 为什么首版不加入 trainable scattering scale

若全局尺度可训练，最大似然目标还必须包含 covariance 的 log-determinant；只优化
归一化残差会偏好无限放大方差。即使加入完整 likelihood，材料 scale 与 geometry
和 measurement error 仍高度相关。首版固定 PDG/ACTS 物理尺度，必要时只做预先
声明的离散 sensitivity scan，并为每个点写独立 manifest。

## 结论

选择 A。B 保留为当 A 的物理闭合通过但优化稳定性或性能无法接受时的升级方向；
C 仅允许用于明确标注、不可进入生产选择的消融实验。

