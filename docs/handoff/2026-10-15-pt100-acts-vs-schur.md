# pT=100 GeV、材料开：ACTS-Alignment 与 Schur-LM 受控对照（2026-10-15）

本次回答的问题：在同一份 100 GeV、材料开的样本和同一次误校准抽样上，ACTS-Alignment 和 Schur-LM 的校准精度和耗时各是多少。数字和结论见优化日志 Iteration 48。

## 状态

已完成，所有 stage 的 manifest 都是 `completed`。冒烟 `pt100-align-20261015-smoke` 因宿主 OOM 被 SIGKILL，manifest 已补记为 failed；`-smoke2` 已完成。

## 入口与布局

常量（RUN、事件数、误校准 seed、ACTS 迭代上限、来源样本）都在 `scripts/pt100_alignment_layout.py`。这个文件不依赖 ACTS，由下面三处共用。

ACTS 侧用独立的新 shell：

```bash
source setup_acts.sh
python scripts/alignment_odd.py pt100-alignment acts-alignment   # ~4.5 h, ~6 GB, 单线程
python scripts/alignment_odd.py pt100-alignment misaligned
# Schur 跑完后：
python scripts/alignment_odd.py pt100-alignment aligned-acts
python scripts/alignment_odd.py pt100-alignment aligned-schur
```

conda `pytorch` 环境：

```bash
python scripts/analysis/run_pt100_schur_calibration.py --threads 1        # 主口径
python scripts/analysis/run_pt100_schur_calibration.py --threads default  # 副口径
python scripts/analysis/compare_pt100_alignment.py
python scripts/analysis/pt100_sensor_tail.py                              # 事后诊断
```

输出位置：
- 几何：`data/alignment/<RUN>/<target>/`，含误校准和 ACTS 结果。
- 重建：`data/reconstruction/<RUN>/{nominal→软链接到 g3,misaligned,aligned-acts,aligned-schur}/`。
- Schur 两个 arm、对比、尾部诊断：`data/experiments/pt100_alignment/<target>/<RUN>/`。

所有输出目录在已存在时都会拒绝写入。冒烟的做法是在 Python 内部修改 layout 常量（事件数 64、`PT100_ACTS_MAX_ITERATIONS=1`），具体见 smoke2 的 manifest 命令。

## 代码改动

- `alignment_odd.py`：
  - 新增 `pt100-alignment` 子命令和 `run_pt100_alignment_study`。
  - `runAlignment` 新增 `maxNumIterations` 参数，默认 100，不改变原行为。
  - `run_material_study_stage` 的 timings 增加 `process_cpu_seconds` 字段（只增不改）。
- 未提交的 material study 改动仍然把 `simulation` / `reconstruction` 子命令锁在 material study 上。本次没有恢复，所以 `run_candidate_acceptance.sh` 仍然不能用。
- 新增测试 `tests/test_compare_pt100_alignment.py`（13 项）。全部改动完成后 `make test` 为 903 passed / 3 skipped / 0 failed（conda pytorch 3.10）。

## 有意没做的事

- 没有做多次误校准抽样或重复计时。只有一次抽样、单次计时，所以没有 CI。
- 没有改 ACTS 的稠密求解，也没有测 ACTS 多线程的可能性。示例实现里 `execute` 没有加锁，强行开多线程会数据竞争。
- 尾部诊断里“剔除 hit ≤6 的 sensor”这个阈值（p10）是事后选定的，不能当作预注册判据。
- 没有和图二逐格比较，因为样本、事件数和 pT 分布都不同。

## 下一步可选

- 如果要正式声明精度等价或改善，需要预注册 sensor 筛选判据，并用独立样本、多次误校准抽样重跑。
- 如果要比较耗时扩展性，可以在 100k 事件上只重跑 Schur：ACTS 每次迭代的耗时与径迹数关系不大，但 event loop 和 KF 部分会增长。
