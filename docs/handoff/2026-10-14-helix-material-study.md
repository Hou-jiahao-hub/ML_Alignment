# 三组材料效应实验（2026-10-14）

本实验回答：10 GeV 下材料增加多少单 helix 预测误差、无材料时统计预算是否闭合、
100 GeV 下额外预测误差是否下降。不是 alignment 求解或独立泛化验证。

## 配置与入口

`alignment_odd.py` 的 `simulation` / `reconstruction` 子命令在代码中硬编码三组配置：
10 GeV 材料开、10 GeV 材料关、100 GeV 材料开；每组 20000 事件、seed 42。
这两个分支现在运行完整实验表，原有单组 CLI 路径/模式/事件数选项不控制本实验。
没有新增 CLI。`full` / `alignment` / `material` 的原调用方式不变。

当前运行 ID 为 `material-helix-20261014-r2`；r1 保留环境缺失的失败记录。
用户在中断后明确授权：使用优化提交 `b9f33df` 重新计算三组全量 helix，并直接覆盖 r2 被中断的
helix 输出。此次例外沿用 r2 目录，模拟和 nominal 重建继续复用；原中断记录保留，
旧 manifest、重启原因和当前源码哈希写入 `optimized_restart.json`。

2026-10-15：r2 的 `b9f33df` 三组全量验证已完成。用户随后指定 `10a9266` 重算三组 helix；
新输出 RUN 为 `material-helix-20261015-10a9266`，输入 RUN 仍为 `material-helix-20261014-r2`。
`preregistration.json` 的 `input_run_id` 明确关联原模拟/重建；新运行入口、状态和日志为
`run_full.py`、`status.json`、`run.log`，`source_pin.json` 锁定该提交的 helix 源码字节。
已完成的 r2 结果保留，统计与计时在新目录重新生成。
配置派生 target 目录和 suffix。模拟与重建目录分别为：

```
data/simulation/<RUN>/<group>/
data/reconstruction/nominal/<target-dir>/<RUN>/<group>/
data/experiments/helix_truth/<target-dir>/<RUN>/<group>/
```

ACTS 独立新 shell：

```bash
source setup_acts.sh
python scripts/alignment_odd.py simulation
python scripts/alignment_odd.py reconstruction
```

先在单独 smoke 运行 ID 下，通过 Python 内部调用将事件数设为 64，依次完成三组模拟与重建；
不新增 CLI、不改正式硬编码值。smoke 也必须保留 manifest。

退出 ACTS shell；在全新的 conda `pytorch` shell 中，对三个 group 依次执行：

```bash
python scripts/validate_helix_truth.py \
  --geometry nominal --nominal-states <group-reconstruction>/trackstates_kf_<suffix>.root \
  --nominal-geometry data/geometry/odd_transforms.json \
  --output-dir <group-helix> --max-events 0 --max-tracks 0 \
  --seed-iters 2000 --refit-iters 60 --batch-size 128 --threads 6 --prediction-budget
```

完成后比较，按 g1 / g2 / g3 顺序传递目录：

```bash
python scripts/analysis/compare_helix_material_effects.py \
  --run-dirs <g1-helix> <g2-helix> <g3-helix> --output-dir <RUN-helix>/comparison
```

## 数据与判读

模拟、重建、非 target refit、求交和评分几何均为 nominal。
helix seed/refit 不直接读取 target 测量、target SMT 和 Truth；上游 ACTS 仍保留
TruthSmeared seed 与 smoothing，因此不声称严格独立的 target holdout。

主统计只针对 target 层/LOC：`M=mean((prediction-measurement)^2)`、`U=M-V-P`，
并报告 `truth_M=mean((prediction-MC真值交点)^2)`、均值、标准差、RMSE 和 pull²。
`P` 是固定几何和测量噪声模型下的 ref-only 统计传播方差，不含材料过程噪声。
非 target / all-surface 图是补充诊断，不使用相同的独立预测预算判据。

生成粒子初态 1/2 一致，1/3 除动量放大 10 倍外一致；输出哈希、评分重算、覆盖检查必须通过。
每个单元用共同生成事件池 bootstrap 2000 次，seed 6006，保留各组完整队列；
记录身份重合率，但不按求交成功取交集。报告 eta 每单位区间及按 pathLength 定义的内插/外推。
空组或 bootstrap 中消失的极小组不可评估。

- 材料：报告 g1−g2 的 M/U/truth_M 差值与 95% CI；是整体材料作用，不单独分解散射。
- 无材料闭合：g2 的 U 与 truth_M−P 的 CI 是否包含 0；包含仅表示统计相容，不是等价证明。
- 高动量：g3−g1 的 U/truth_M CI 上界 <0 时才报告显著下降；动量对照也改变曲率与接受率。
- 最终 refit 非驻点、求交失败、预算无效或非有限值使相关主结论不可判定；保留所有记录。

## 时间与失败留痕

模拟/重建各记录 detector/流程准备、Sequencer.run 和两者总时间，包含 ROOT 写出，
排除环境加载和 provenance 哈希/归档。seed/refit 只累计各自求解器调用时间；
驻点检查、求交、预算和完整 helix 阶段独立计时。失败的批处理和逐轨迹重试均计入实际耗时。
每组保存 timings.json 和 manifest；总表在 comparison/timings.csv。

初次启动时 `/cvmfs/sft.cern.ch/lcg` 未挂载，r1 未执行物理实验。用户手动挂载后，r2 已通过
ACTS 加载以及三组 64 事件模拟/重建/helix smoke 检查，继续正式实验。代码测试、合成 ROOT I/O
验证与真实 Fatras 数据结果分开记录。
