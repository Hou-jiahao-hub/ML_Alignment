# 原图核验与实现对应

本记录用于备查，不新增正文页。核验限定生产 joint 初始化的 SMT seed 与默认 Gauss–Newton measurement refit；P7 的 block-LM 内容为后续补页，其核验对象是 `run_block_lm_alignment` 的收敛与接受契约（预注册版本 2）。源码定位由交付时快照提供；理论推导和当前近似实现明确区分。

## 原图逐项核验

| 原表达 | 判定 | 修订 |
|---|---|---|
| seed 在计算最近点及残差前判断收敛 | 顺序不完整 | 先计算 r、J、H、g、decrement，再判断驻点 |
| “更新 helix”作为无条件回路入口 | 易误解 | 候选只在接受时提交；拒绝保留当前参数 |
| LM 矩阵和 decrement 均记 D | 符号冲突 | 阻尼尺度 M，decrement ν，阻尼水平 μ |
| 六参数直接进行优化 | 缺少表示说明 | 两个位移、方位角、倾角和缩放曲率，共五个优化变量 |
| seed 接受条件只有目标下降 | 不完整 | 还要求有限、活跃径迹及保护初始有效点 |
| “J、H、g 同初始化阶段” | 表达不准确 | 构造形式相同，残差及求导路径不同 |
| measurement 文字为预测减测量 | 与公式和代码相反 | 统一为测量减预测 |
| refit change 小即可输出 | 缺独立验收 | 循环停止后检查 decrement、非驻点比例和求交失效率 |
| “frozen helix → joint alignment” | 主方向正确但含义须限定 | 主线固定径迹，仅由目标 measurement 求解几何 |
| 所有径迹均须满足 10⁻⁶ 才能通过主线 | 不精确 | 主线容许最多 0.1% 非驻点；非有限值不在容许范围内 |
| 最近点 Newton 导数 | 正确 | 补充有限迭代和非全局保证；s 为横向弧长 |
| 三维投影初值 s₀³ᴰ | 数学可作对照，非当前实现 | 当前使用 xy 投影初值；不得当作现有改进 |
| “首层上的第一个 hit” | 不准确 | 确定性排序后的第一个非目标层拟合 hit |
| y(s) 和差化积后负号 | 错误 | y₀ + s sin(ψ₀+ρs/2) sinc(ρs/2) |
| 曲率大小式中额外除以质量 m | 错误 | SI 动量定义下大小为 \lvert qB_z\rvert/p_T；本章节不使用磁场强制确定曲率 |
| helix_tangent 推导 | 正确 | 不是单位三维切向量，其模为 sqrt(1+τ²) |
| intersect_plane 签名带 s0 | 不符合共享函数 | 当前共享入口为 track, center, normal, n_newton=12 |
| 路径过大时“截断” | 不符合代码分支 | 越界候选判无效，保留上一值；最终输出使用无效占位 |
| t·n=0 的平行说明 | 应明确对象 | 切向平行于平面，垂直于法向 |
| Rz 第一行 cosγ, sinγ | 与项目约定相反 | 第一行为 cosγ, −sinγ；第二行为 sinγ, cosγ |
| build_track_batches 只按 hit 数筛选 | 缺条件 | 目标 hit 存在、非目标 hit 数≥8、累计折线长度≥300 mm |
| 排序写 z 和 model | 错误或拼写不准 | 使用绝对值 z、module，并保留末级 sensor 键 |
| 目标层只用于外推验证 | 不符合当前主线 | 固定径迹后用于目标几何优化，不是独立 holdout 验证集 |

## 源码对应索引

- `scripts/helix_model.py`：位置与切向、最近点、平面求交、gauge、seed Jacobian、LM、seed 终态门限。
- `scripts/ml_alignment_autograd.py`：生产组批、初始化、measurement 适配、主线失效率与驻点检查、几何更新旋转约定。
- `scripts/validate_helix_model.py`：零节点 measurement 残差、可微求交、LM change 停止、终态 decrement。
- `scripts/ml_alignment_autograd.py`（P7 block-LM）：`load_joint_geometry` 装配工作几何、`run_block_lm_alignment` 主循环、`_sensor_residual_data` 分块选择、`_block_lm_linearized_block` 线性化、`_block_lm_decrement` 与 `_block_lm_free_dof_mask` 驻点与自由集、`_block_lm_fixed_chi2` 与 `_block_lm_retry_damping` 接受与阻尼阶梯、`_block_lm_block_status` 与 `_block_lm_run_summary` 终态汇总，常量组 `BLOCK_LM_*`。

精确行号、文件 SHA256 与工作区状态见 [source_snapshot.json](source_snapshot.json)；P7 新增的函数索引尚未进入该快照，下次刷新快照时补齐。正式 PPT 页脚可只写“实现依据：ML_Alignment，版本见备注”，避免路径占据画面。

## 解释边界

1. seed 驻点指标基于冻结最近点路径的 Jacobian，不称为完整最近点目标的精确二阶量。
2. seed 的有效点保护只锚定初始集合。refit 内层无相同保护，不能增画该功能。
3. refit 的终态验收在重新建立的参考平面内计算；微小相对阻尼使有限数值结果不必完全参数化不变。
4. 目标层 SMT 点被排除，不等于重建的非目标平滑状态没有目标层信息来源。
5. 最近点与平面求交的 valid 都是实现的数值有效性，不构成全局最优或有限传感器面内包含证明。
6. 本交付没有算法实验、性能曲线或泛化结论，不需要也不伪造算法实验 manifest；源码追溯信息单独记录。

## 文案与图稿检查

交付前核查正文七页（P7 图稿待生成）、附录十四页、全部本地文档链接、Mermaid 解析、SVG/PNG 输出及独立视觉反馈。检查记录见 [qa.md](qa.md)。不修改核心算法，因此不运行耗时生产实验或为文案新增算法测试。
