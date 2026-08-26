# 文献检索日志

## 检索边界

- 检索日期：2026-08-25。
- 目标：解释当前单 helix 的材料相关下限，并选择能与可微全局对齐结合的升级模型。
- 优先级：同行评审论文或预印本原文，其次是 PDG 和 ACTS 官方文档。
- 时间范围：Broken Lines 的奠基工作至当前 ACTS/CMS 实践；旧文献在仍定义核心方法时保留。
- 版本策略：论文引用已发表版本；软件行为引用带版本的 ACTS 文档，并用本地 ACTS checkout 核验接口是否存在。

## 查询记录

| 来源 | 查询词 | 纳入结果 | 用途 |
|---|---|---|---|
| arXiv | `General Broken Lines track fitting multiple scattering` | Kleinwort 2012 | GBL 参数化、矩阵结构、协方差。 |
| arXiv/CDS | `broken lines detector alignment Millepede CMS` | Blobel et al. 2011；CMS 2022 | 全局对齐可扩展性和生产经验。 |
| arXiv/CDS | `global covariance Kalman track detector alignment` | Hulsbergen 2009 | Kalman 状态间相关性和对齐应用。 |
| PDG | `2024 passage particles matter multiple scattering` | Groom and Klein 2024 | `x/X0`、Highland 近似和适用边界。 |
| ACTS docs | `material effects process noise track state` | Tracking；Track EDM | 过程噪声与 passive material state。 |
| ACTS docs/source | `surface volume material mapping interaction recording` | Geometry；Material mapping | ODD 材料图和 adapter 数据出口。 |

## 纳入标准

- 明确定义轨迹状态、散射参数或协方差传播。
- 能说明材料效应如何进入拟合目标或预测协方差。
- 对 track-based alignment 的局部/全局参数关系有直接贡献。
- 对当前仓库的 ACTS adapter 或纯 PyTorch core 有可操作映射。

## 排除或降级的方向

- 只以残差重加权处理尾部、但不表示材料过程噪声的方法：历史实验已显示可能压制真实几何信号。
- 纯经验自由分段而没有 `x/X0` 或固定散射先验的方法：kink 可无代价吸收对齐参数，识别性不足。
- 以电子 bremsstrahlung 为主的 GSF：当前样本是 `pT >= 2 GeV` 的 muon，首版问题是多重散射。
- 端到端复用 ACTS Kalman fitter 作为对齐 core：违反 core 必须无 ACTS 依赖的仓库边界。
- 二手博客和无方法细节的演示材料：可用于发现关键词，不用于支撑最终技术结论。

## 证据分级

- **文献事实**：由 `sources.bib` 的一手资料支撑。
- **官方软件契约**：由 ACTS 官方文档及本地 checkout 的公开接口共同支撑。
- **项目观测**：来自版本化代码、历史优化日志或 manifest，不能外推为普遍物理结论。
- **项目推断**：由文献事实和项目观测推导，必须在下一阶段通过测试或实验验证。

