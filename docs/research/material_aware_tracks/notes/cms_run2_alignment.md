# CMS Run-2 Tracker Alignment 经验

**主要来源：** [@cms2022run2alignment]

## 文献事实

CMS Run-2 tracker alignment 面向大量模块和随时间变化的 detector conditions，使用
track-based global alignment，并讨论了弱模、外部约束、时间区间和不同轨迹样本的
互补性。生产流程中的几何质量不能只由训练目标的下降判断，还需要独立的 track、
vertex 和 physics-observable 验证。[@cms2022run2alignment]

CMS 的 Millepede-II 路径使用 GBL 对 Kalman seed trajectory 做包含多重散射的局部
重拟合，并将所需局部信息与 global derivatives 交给全局求解。这说明 GBL 不是
替代 detector alignment，而是为 alignment 提供更正确的轨迹模型和协方差。

## 对当前项目的含义

- 必须分开“轨迹模型通过 nominal holdout”和“最终 alignment 改善 reconstruction”
  两级验收。
- 轨迹 χ²下降不是充分条件；最终仍需对同一输入样本做 ACTS reconstruction 的
  paired comparison。
- active DOF、数据样本、随机种子和优化预算在单 helix/broken-helix 比较中必须
  固定，避免把其他变化错误归因于材料模型。
- geometry truth 或 nominal 指标可用于冻结后的 MC 诊断，但不应参与 production
  candidate selection。
- 弱模与材料 kink 的相关性需要独立监控，不能只报告总 loss。

## 不直接照搬的部分

CMS 的探测器、磁场、数据组合、时间依赖和生产软件与 ODD 项目不同。该报告支持
“材料相关轨迹模型适合全局对齐”和“需要独立验证”这两个原则，不提供本项目的
学习率、kink 节点数或 acceptance 阈值。

