# 材料感知轨迹模型知识库

本目录汇总升级 `ml_alignment_autograd.py` 轨迹模型所需的理论、实验经验和
ACTS 数据边界。目标不是复述所有径迹拟合文献，而是回答一个具体问题：如何在
保持对齐核心不依赖 ACTS 的前提下，突破当前单一 helix 在 nominal holdout 上的
χ² 与预测精度下限。

## 已知项目事实

- 当前核心模型是六参数、恒定曲率 helix，整条轨迹共享一组参数。
- nominal measurement holdout 已观察到接近零的 bias，但 interpolation 和尤其
  one-sided extrapolation 的 RMSE 明显增大；这符合未建模随机散射而非整体平移
  偏置的特征。
- 历史上的 Huber 和 pull-based IRLS 会压低携带几何倾斜信息的边缘 hit，不能把
  它们作为材料模型的替代品。
- `trackstates` ROOT 含 measurement、预测、滤波、平滑状态和 `pathLength`，但不含
  每个材料交互的 `x/X0`。ODD 材料图由 ACTS 运行时持有。
- 仓库边界只允许 ACTS adapter 导入 `acts`；轨迹数学、验证和对齐求解必须能在
  无 ACTS 环境中导入。

这些事实来自仓库的 `docs/architecture.md`、`docs/ml_alignment_principles.md`、
`docs/history/ml_optimization_log.md` 和当前实现，属于项目观测，不是文献结论。

## 推荐阅读顺序

若只读一份文档，请读 [理论速读：从单 helix 到材料感知 broken-helix](theory_primer.md)——它把下列笔记串成一条从病症诊断到统计门控的完整推理链。

1. [粒子穿越材料与多重散射](notes/particle_material.md)：理解 `x/X0`、动量和
   散射方差的物理输入。
2. [Broken Lines 与 GBL](notes/general_broken_lines.md)：理解散射 kink、局部参数
   和带状法方程。
3. [Kalman 全局协方差](notes/kalman_global_covariance.md)：理解 process noise、
   跨状态相关性和预测协方差。
4. [全局对齐中的 GBL/Millepede](notes/global_alignment.md)：理解局部轨迹参数和
   全局几何参数如何共同进入对齐。
5. [CMS Run-2 经验](notes/cms_run2_alignment.md)：了解大规模真实探测器中的约束、
   弱模与验证边界。
6. [ACTS 材料与轨迹状态](notes/acts_material.md)：把方法映射到现有 ODD/ACTS
   输入与 adapter 边界。
7. 阅读 [方法比较](method_comparison.md) 和 [项目决策记录](decision_record.md)。

## 术语

| 术语 | 本知识库中的含义 |
|---|---|
| reference trajectory | 不含随机 kink 的名义 helix，用于线性化和节点排序。 |
| scatterer | 将一段材料等效为薄层后引入方向过程噪声的位置。 |
| kink | scatterer 前后轨迹方向之差的两个局部角分量。 |
| `x/X0` | 穿越厚度除以辐射长度，是小角多重散射方差的关键材料量。 |
| local parameters | 每条轨迹独有、可在构造全局对齐方程时消元的参数。 |
| global parameters | 多条轨迹共享的探测器对齐参数。 |
| predictive covariance | 在未参与拟合的 measurement 处，包含轨迹参数和过程噪声不确定度的预测协方差。 |
| calibrated pull | 使用完整预测协方差归一化后，中心与宽度符合模型预期的残差。 |

## 引用约定

正文使用方括号 citation key 指向 [sources.bib](sources.bib)。每个技术结论必须能追溯到一手
论文、PDG 或 ACTS 官方文档。检索过程和筛选标准记录在
[search_log.md](search_log.md)；仓库实验观测必须明确标注为“项目观测”。

## 当前结论

首选方案是 GBL 风格的“六参数参考 helix + 材料节点二维 kink”。measurement
残差和 Highland 散射先验共同组成目标函数；ACTS adapter 负责导出逐轨迹材料
交互，PyTorch core 只消费 schema-versioned 数值 artifact。详细理由、边界和下一
阶段所需接口见 [decision_record.md](decision_record.md)。
