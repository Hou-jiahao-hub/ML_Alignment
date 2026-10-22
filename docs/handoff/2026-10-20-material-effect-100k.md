# 材料效应实验 100k 复现（2026-10-20）

用三组 100k 事件的 nominal 重建，复现 2026-10-14 材料效应实验
（`material-helix-20261015-10a9266`，每组 20k 事件）。旧脚本已在 efacf67 删除，
本次在 `tests/material_effect/` 下基于当前 mlalign 重写。不是 alignment 求解。

## 输入与身份

| 组 | 重建（nominal） | particles / hits |
|---|---|---|
| g1_pt10_material_on | `data/reconstruction/full-pt10-mat-100k/reconstruction/nominal/` | `full-pt10-mat-100k/simulation/` |
| g2_pt10_material_off | `data/reconstruction/full-pt10-Nomat-100k/reconstruction/nominal/` | `full-pt10-Nomat-100k/simulation/` |
| g3_pt100_material_on | `data/reconstruction/full-pt100-mat-100k-20261016/nominal/` | `data/simulation/full-pt100-mat-100k-20261016/` |

这些目录没有上游 sim/reco manifest，三组的身份改由 `run_material_effect.py verify` 核验（结果见 `arm_verification.json`）：
- 100,000 个 generated event 逐事件配对：顶点、η、φ、q 一致，g3 动量为 g1 的 10×；
- 材料开关：沿径迹 |p| 相对变化最大值 g1 0.69、g2 1.9e-7、g3 0.069。

## 运行

运行 ID 为 `material-helix-100k-20261020`。输出目录：
`data/experiments/helix_truth/Vol17_Lay8__Vol24_Lay4/material-helix-100k-20261020/`。
`preregistration.json` 写于任何数据运行之前，内含输入 SHA256 以及与 10-14 实验的已知差异。

```bash
PY=/home/houjiahao/miniconda3/envs/pytorch/bin/python
$PY tests/material_effect/run_material_effect.py verify  --run-root R --arm G=STATES,PARTICLES,HITS  # ×3
$PY tests/material_effect/run_material_effect.py arm     --run-root R --group G --states .. --summary .. --particles ..
$PY tests/material_effect/run_material_effect.py compare --run-root R --reference-contrasts <20k>/comparison/contrasts.csv
```

- 单组约 4 min，6 线程。P 的预算向量化后为 16 s；旧实现 20k 事件就要 1376 s。
- 三组都是 62.1k 条径迹、约 289k 个 target 分量。refit 非驻点、target 求交失败、rank 不足、truth 未连上的数量全部为 0。
- `STRIP_Y_FLOOR` 在这批数据上从未生效：原始方差与加 floor 后的方差相同。

## 结果（主单元 = 全部分量）

| 问题 | 结论 | 与 20k 对比 |
|---|---|---|
| 材料效应 g1−g2 | 四个单元的 M、U、truth_M 均 `increased`，ΔU ≈ ΔM | 估计值全部落在 20k CI 内；CI 宽度约为 20k 的 0.4×（预期 1/√5 = 0.45） |
| 高动量 g3−g1 | 四个单元均 `decreased` | 24:4 LOC1 的 M、U 从 `insufficient_evidence` 变为 `decreased` |
| 无材料闭合（g2） | 17:8 LOC0/LOC1、24:4 LOC0 的 U 和 truth_excess 由"与 0 相容"变为 `not_closed`；24:4 LOC1 的 U 仍为 `not_closed`，truth_excess 与 0 相容 | 点估计都在 20k CI 内，变化来自统计功效提高 |

无材料闭合的偏差幅度如下（g2，mm²）：

| 单元 | U | truth_excess | M − truth_M − V（≈ 测量方差的盈余） |
|---|---:|---:|---:|
| 17:8 LOC0 | −5.7e-6（−1.6% of V+P） | −2.9e-6（−2.2% of P） | −2.8e-6（−1.2% of V） |
| 17:8 LOC1 | −4.1e-5 | −2.7e-5（−4% of P） | −1.4e-5 |
| 24:4 LOC0 | −7.6e-5 | −1.3e-5（−2.9% of P） | −6.4e-5（−3.5% of V） |

原因已确认（2026-10-20）：ODD 数字化（`odd-digi-smearing-config.json`）对所有分量使用 `GaussClipped`，`range [-3, 3]`，即 ±3σ 截断。截断高斯的方差是 0.9733·σ²，而 err² 报告的是未截断的 σ²。
- 用 truth 拆分 g2 残差：⟨(m−t)²⟩/V − 1 为 −1.9%、−3.2%、−3.7%、−3.4%；⟨(p−t)²⟩/P − 1 为 −2.2%、−4.0%、−2.9%、−2.1%；交叉项约为 0。
- 预测 U ≈ (0.9733 − 1)(V + P)：四个单元分别为 −9.5e-6、−2.4e-5、−6.1e-5、−3.9e-2，观测值为 −5.7e-6、−4.1e-5、−7.6e-5、−5.0e-2，差值都在 CI 半宽以内。
- 结论：无材料时的负 U 是仿真误差模型的设定造成的（报告的 σ 未计入截断），不是 helix 模型误差；按 V_eff = 0.9733·σ² 修正后即闭合。

## 已知现象：17:8 LOC1 的 P 有重尾

- 三组的 P 中位数都是 6.78e-4，但 g3 的均值为 7.83e-4（g1、g2 约 6.64e-4）。差别来自 72k 个分量中最大的 20 个（最大值 1.5 mm²）；去掉它们后三组都约为 6.41e-4。
- 20k 运行里也有同样的现象（g3 7.49e-4 vs 6.42e-4）。
- 这会让 17:8 LOC1 的 high_momentum 对照出现以下表现：U 与 M 差别较大；CI 宽度比 > 1；truth_M 的估计落在 20k CI 之外。
- 这些少数径迹在 LOC1（z 向）上的 ref 拟合约束很弱。尚未逐条排查。

## 与 10-14 实验的已知差异（预注册中已列出）

- 筛选跨度用 SMT 折线长度，旧实验用 pathLength 极差。
- hit 排序按半径和 |z|。
- refit 用生产 GN；生产加载器把测量值存为 float32。
- P 用 Cholesky 向量化计算，damping 为 0，rank floor 为 5·eps·λmax。
- 不做 seed 点拟合的驻点闸门。

## 没做的

- 没有做 η 或内插/外推分层的 bootstrap；`group_metrics.csv` 里的 η 分层只有点估计。
- 没有排查 17:8 LOC1 重尾径迹。
- 代码未 commit。
