# 决策记录：材料感知分段轨迹模型

- 状态：已接受为下一阶段设计基线。
- 日期：2026-08-25。
- 决策范围：知识库结论与下一阶段接口；不代表模型已经实现或通过物理验收。

## 问题

当前 `ml_alignment_autograd.py` 用一组恒定曲率 helix 参数描述整条轨迹。nominal
measurement holdout 显示 bias 接近零但 RMSE、方向误差和 measurement-normalized
残差存在明显下限，且 backward extrapolation 最差。项目历史已排除交点数值失败，
也证明仅靠鲁棒损失可能误伤真实 geometry signal。

需要选择一种模型，在保持 core 无 ACTS 依赖的同时表示材料引起的随机方向变化，
并能继续与目标 surface alignment parameters 联合优化。

## 决策

采用 GBL 风格的“六参数 reference helix + 材料节点二维 kink”。

1. ACTS adapter 沿与 reconstruction 一致的 selected tracks 记录材料交互。
2. 每个有效材料节点引入两个与入射方向正交的小角 kink。
3. 相邻 helix 段位置连续；首版段间保持同一 signed curvature。
4. kink 均值固定为零，covariance 由节点 `x/X0`、muon 动量和 Highland 近似确定。
5. 优化目标同时包含 measurement χ²和 scattering χ²。
6. nominal holdout 使用局部 Hessian 传播 prediction covariance，评价 pull/χ²校准。
7. 模型通过 nominal gate 后，才进入 misaligned alignment 与 ACTS reconstruction。

该选择由 GBL 的材料散射轨迹表示和全局对齐应用支持
[@kleinwort2012gbl; @blobel2011fastalignment]，由 PDG 提供散射尺度
[@pdg2024passage]，并与 ACTS 的材料/process-noise 数据模型一致
[@actsTracking; @actsGeometryMaterial]。

## 最小材料 artifact 契约

artifact 使用 NPZ 数组和 companion manifest，schema 首版为 `1.0`：

| 字段 | 形状 | 单位/含义 |
|---|---:|---|
| `event_nr` | `[T]` | 与 trackstates 一致的事件编号。 |
| `track_nr` | `[T]` | 与 trackstates 一致的事件内轨迹编号。 |
| `q_over_p_gev_inv` | `[T]` | 节点间固定的带符号 `q/p`。 |
| `node_offsets` | `[T+1]` | flattened material nodes 的轨迹边界。 |
| `path_length_mm` | `[N]` | 沿 reference trajectory 单调递增。 |
| `position_mm` | `[N,3]` | global material-interaction position。 |
| `direction` | `[N,3]` | 归一化入射方向。 |
| `x_over_x0` | `[N]` | 已含 ACTS 路径修正的材料预算。 |
| `surface_geometry_id` | `[N]` | ACTS geometry identifier 的无符号整数编码。 |

manifest 必须记录 schema、单位、命令、Git commit、geometry/material-map/trackstates
SHA256、总轨迹/节点数、未匹配/重复/非法记录数和完成状态。任何键不唯一、路径非
单调、单位缺失或材料预算非有限/为负都中止消费。

## 模型验证口径

### Nominal model gate

single-helix 和 broken-helix 使用完全相同的稳定哈希 track split、holdout hit、随机
种子和迭代预算。每个具备足够统计量的 layer/mode/component 要求：

- intersection failure rate 不超过 5%；
- predictive χ²/component 位于 0.7–1.5；
- pull mean 绝对值不超过 0.2；
- interpolation LOC0 RMSE 相对 single helix 至少改善 20%；
- one-sided RMSE 不得回退超过 5%；
- held-out Gaussian negative log likelihood 严格改善。

旧的绝对 RMSE 阈值继续报告，但 one-sided extrapolation 的随机未来散射不可由均值
预测，因此它不再是唯一主门控。

### End-to-end alignment gate

保持 active DOF、输入 measurements、事件数、seed、优化预算和 candidate workflow
不变。broken-helix 的 reconstruction acceptance score 必须严格优于 single-helix，
且任何有效 component 的 χ²不得回退超过 5%。nominal geometry/truth 只在候选冻结
后报告，不参与 stage 或 candidate selection。

## 可证伪条件

出现任一情况时，不能宣称该决策成功：

- zero-material 极限不能复现现有 helix 的点、切向和梯度；
- synthetic scattering closure 不能恢复已知 kink 或产生校准 pull；
- 加入 kink 后 geometry injection 被轨迹系统性吸收；
- nominal held-out likelihood 没有 paired 改善；
- nominal 通过但 end-to-end reconstruction 不优于 single-helix；
- 改善依赖 nominal truth 选择候选或依赖可训练 covariance 膨胀。

若统计模型闭合但 autograd 的稳定性、内存或速度失败，再评估可微 Kalman/smoother
或 GBL bordered-band/Schur 求解；不得用无材料先验的自由分段替代正式模型。

## 明确不包含

- electron bremsstrahlung 和 Gaussian-sum filtering；
- 随机能损、段间可变曲率和非均匀磁场积分；
- active alignment DOF 的重新选择；
- 在本阶段修改 `ml_alignment_autograd.py` 或运行新的物理实验。

## 后续工作包

知识库评审通过后，后续按顺序独立规划和验收：

1. ACTS 逐轨迹材料导出与 schema loader。
2. broken-helix 数学、kink prior 和 synthetic closure。
3. nominal paired holdout 与 predictive covariance。
4. `ml_alignment_autograd.py` 联合对齐接入和 ACTS reconstruction。

