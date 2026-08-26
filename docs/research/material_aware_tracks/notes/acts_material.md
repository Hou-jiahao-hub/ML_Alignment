# ACTS 材料描述与轨迹状态

**主要来源：** [@actsGeometryMaterial]、[@actsTrackEDM]、[@actsMaterialMapping]、
[@actsTracking]

## 官方软件契约

ACTS tracking geometry 同时支持 surface-based 和 volume-based material。材料基本
量包括辐射长度 `X0`、核相互作用长度 `L0`、原子量、原子序数和密度；surface
material 还包含代表性厚度，并可按局部坐标分 bin。[@actsGeometryMaterial]

ACTS track EDM 将轨迹表示为一系列离散 track states，除 measurement 和 hole 外，
也可以包含已知 passive material 的交点。材料传播可以在薄表面离散更新，也可以
在体材料中连续处理。[@actsTrackEDM; @actsTracking]

材料 mapping 将详细模拟中的材料投影并平均到选定的 surface/volume 表示。ODD
材料图因此是重建用的有效材料描述，不应被当作逐原子精确几何。
[@actsMaterialMapping]

## 本地 checkout 核验

本地 ACTS 源码包含 `MaterialInteractor`、`MaterialSlab::thicknessInX0()`、
`PropagationAlgorithm.recordMaterialInteractions` 和 `RootMaterialTrackWriter`。
`odd-material-maps.root` 包含按 geometry identifier 组织的 `t`、`x0`、`l0`、`A`、
`Z`、`rho` 数据。以上是 2026-08-25 对当前 checkout 的项目观测。

当前 nominal `trackstates` ROOT 列出了 `stateType` 和 `pathLength`，但 writer 输出中
没有逐状态 `x/X0`。现有 loader 还会通过 `l_x_hit.notna()` 删除 hole/material
state。因此 core 仅靠当前 measurement rows 无法恢复路径相关材料预算。

## 推荐数据边界

由现有 ACTS adapter 在与 reconstruction 相同的 geometry、material decorator、
particle hypothesis 和 selected-track 顺序下记录材料交互，并输出普通、版本化数值
artifact。最小字段为：

- `(event_nr, track_nr)` 和节点 offset；
- 节点 `pathLength`、global position、传播方向；
- 路径修正后的 `x/X0` 和 geometry identifier；
- 轨迹 `q/p` 或节点动量，以及明确单位；
- material map、geometry、trackstates 的输入哈希和映射统计。

纯 PyTorch core 只读取该 artifact，不导入 `acts`。导出阶段必须验证 selected-track
编号与 `RootTrackStatesWriter` 一致；缺失、重复、非单调路径、非有限或负材料预算
均 fail closed。

## 风险

- surface material 的 pre/post split、入射角修正和 volume material 不能通过简单
  ROOT histogram lookup 自动等价复现，因此优先使用 ACTS propagation 记录。
- 若材料导出使用 seed 而不是最终 selected track，节点位置和路径长度可能不一致；
  必须在接口测试中限制允许差异。
- material artifact 属于大数据输入，应位于 `data/` 并由 manifest 管理，不能提交
  到 Git。

