# 交接：全量四臂 helix-recovery 运行阻断与重跑前修复

**交接日期**：2026-09-07
**范围**：`Vol17_Lay8__Vol24_Lay4` 的 paired helix-recovery 四臂诊断
**代码提交**：`37ed4f5c1020f683dd2bf7c0f23f00474a9deb44`
**本轮产品代码改动**：无；本文只记录中断实验、根因证据和待实现的修复。
**相关实验记录**：`data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/full-all-events-20260907/manifest.json`

---

## 0. 一句话状态

全量 N-P / M-P / N-E / M-E 运行在构建 paired cohort 前停滞并由用户终止；它**没有**进入
seed、measurement refit、bootstrap 或任何科学判据计算。因此该次运行对“helix 是否恢复错位
影响”不产生通过、失败或 inconclusive 的物理结论。

直接阻断是本地 ROOT 文件经 `uproot.open(path)` 走到了 `FSSpecSource` 的 asyncio 读取路径，
其中 uproot 等待 basket 完成通知而通知未返回。对同一文件强制 `uproot.MemmapSource` 后读取
恢复为秒级，故根因不是 ROOT 数据体积、几何、严格配对或 helix 求解器。

---

## 1. 本次实验的冻结配置与终态

运行命令的有效参数如下：

```text
target layers          17:8,24:4
event partition        all
max events             0（全量）
max tracks             0（全量）
exposure               both（Protected + Exposed）
seed / refit iters     200 / 60
bootstrap              2000 replicates, seed 6006
device / dtype         CPU / float64
```

输入为 nominal/misaligned `trackstates_kf_v17l8_v24l4.root` 及其对应 geometry JSON；输入 SHA256
已写入 manifest。manifest 创建于 `2026-09-07T07:32:54Z`，在约四小时后由用户请求中断，
并被人工收尾为：

```text
status          failed
completed_at    2026-09-07T11:35:16Z
failure_reason  User-requested termination after the full-cohort process stalled
                before cohort construction; no paired metrics were emitted.
```

该目录没有 `paired_track_keys.csv`、`per_track_stage_metrics.csv`、summary 或图；这是符合上述
终态的，不能被误读为“空样本”或“模型失败”。

---

## 2. 直接根因：默认本地 uproot 读取路径失活

### 2.1 运行期证据

中断前主 Python 进程约运行四小时，RSS 约 681 MiB，CPU 约 0.1%。连续 10 秒采样中
`rchar`、`read_bytes` 与系统调用计数均不增长，所有 Python 线程都在 `futex_do_wait`；这不是
正在执行 Gauss--Newton 的形态。

受控单分支探针读取的 `volume_id` 只有 46 个 baskets、压缩 1.47 MB、解压 11.95 MB，却在
30 秒内未返回。faulthandler 给出主线程调用链：

```text
uproot.behaviors.TBranch._ranges_or_baskets_to_arrays
  -> queue.Queue.get
```

而后台 asyncio loop 在 `selectors.select`，没有任何 basket 完成通知到达队列。

环境为 `uproot 5.7.3`、`awkward 2.9.0`。检查 `uproot.open(path)` 返回的文件源为
`uproot.source.fsspec.FSSpecSource`；其本地分块读取通过 fsspec 全局 asyncio loop 提交。

### 2.2 对照实验

同一 nominal ROOT、同一 conda `pytorch` 解释器、同一 `volume_id` 分支：

| 打开方式 | 结果 |
|---|---:|
| 默认 `uproot.open(path)`（`FSSpecSource`） | 超过 30 s 未返回 |
| `uproot.open(path, handler=uproot.MemmapSource)` | 0.159 s 完成 |
| `MemmapSource` 读取 recovery 所需 18 个字段 | 2.235 s 完成 |

因此可证实：默认 fsspec/async 源是本次阻断的直接根因。不能把该故障归因于 ROOT 文件有
98,958 条 track entries、单文件约 1.09 GB 压缩 / 1.78 GB 解压，或归因于 helix 模型数值行为。

### 2.3 触发位置

`scripts/compare_helix_recovery.py` 的 `_load_recovery_branch` 将输入规范化为 `Path` 后调用：

```python
with uproot.open(path) as root_file:
```

该默认选择对本机普通 POSIX 文件不合适。应在 loader 边界显式区分“本地常规文件”和远程 URI，
而不是让 uproot 自动挑选 fsspec 源。

---

## 3. 修复直接根因的最小方案（尚未实现）

在 `compare_helix_recovery.py` 增加一个私有 ROOT 打开辅助函数：

1. 对 `Path.is_file()` 为真的本地普通文件，调用
   `uproot.open(path, handler=uproot.MemmapSource)`。
2. 对 URI、非本地文件或显式 file-like 对象，保留现有 uproot 默认源；不能把 Memmap 强加给
   远程 I/O。
3. `_load_recovery_branch` 的 nominal/misaligned 两个打开点都只能经该辅助函数，防止两臂来源
   不一致。

实现前必须先写失败回归测试：mock/spy 验证本地 `Path` 使用 `MemmapSource`、URI 不使用它，
并以当前小型 ROOT fixture 证明 loader 仍保留相同 identities、component mask 和严格配对失败
语义。随后运行 `make test-root-io` 与相关 helix-recovery 测试。

该修复只解决“读不回来”的直接故障；它并不使完整四臂计算天然变成快速任务。

---

## 4. 修复后仍存在的全量可扩展性风险

下列项不是本次停滞的直接原因（本次尚未离开 ROOT 读取），但会决定重跑是否能在可接受时间内
完成，必须在重跑全量前处理或至少量化。

### 4.1 `--max-events` / `--max-tracks` 发生得过晚

`load_paired_recovery_inputs` 先对 nominal/misaligned 分支完整执行：读取、展平、转 Tensor、
canonicalize 和 `_require_paired_contract`；之后才从 identities 构造 event/track mask。故当前：

```text
--max-events 20000 或 --max-tracks 300
```

会减少 frozen cohort 与 solver 的工作量，却**不会**减少 ROOT 读取、展平或全量严格配对工作量。
此前 20k 尝试在同一阶段耗时很长，正是这个顺序的必然结果。

后续应设计两阶段、分块读取：先低成本读取并配对 `(event_nr, track_nr)` 与每-entry state count，
确定确定性的 event/track cohort；再只物化该 cohort 的所需状态字段。若 event entries 不连续，必须
按 chunk 收集并严格比较，不能假定 nominal/misaligned 可以各自筛选后再求交。

### 4.2 全量 probe 行数与未分块 seed

当前 nominal 输入实测：

```text
measured states       1,308,537
target probe rows       143,697
```

`freeze_paired_cohort` 为每个 target hit 复制该物理 track 的完整 hit 行。随后
`run_four_arm_recovery` 对四臂依次执行 seed 200 次迭代与 refit 60 次迭代。measurement refit
内部有 2048-row chunk，但 `seed_helices` 调用的 point Gauss--Newton 仍一次接受完整 batch；并且
`_batches_for` 为每臂使用 Python 的 row × active-hit 双层循环重建 ID 张量。

所以即使 ROOT 打开恢复为秒级，143,697 行的 CPU float64 seed 和四次 batch 构造仍可能成为新的
长耗时点，甚至引起内存压力。不要把 Memmap 修复后的第一次全量运行当作“必然很快”的作业。

后续性能改造应：

1. 为 seed 增加 row chunk 参数；各 chunk 的参数和诊断按原有 row 顺序拼接。
2. 将 `_batches_for` 的 ID 填充张量化，或在 frozen cohort 中一次预计算、四臂复用。
3. 用小 cohort 逐位/容差比对未分块与分块输出，确认分块不改变四臂共享行集、mask、排序或
   stationarity 判据。

### 4.3 缺少可观测进度

manifest 仅在 cohort 完成后记录 `cohort_counts`，stdout 也只在全流程成功后打印输出目录。加载、
配对、cohort、每臂 seed/refit 和 bootstrap 都没有进度界限。故本次无法仅凭产物判断阶段，只能
通过进程采样诊断。

应在不改变统计输出的前提下，至少记录以下阶段及耗时：`manifest-hash`、`nominal-load`、
`misaligned-load`、`paired-contract`、`cohort`、每个 arm 的 `seed/refit`、`bootstrap`、`write`。
中断时将最近完成阶段和计数写入 manifest。

---

## 5. 推荐执行顺序

1. 先实现并测试本地 `MemmapSource` 选择；不要在默认 `FSSpecSource` 下再次启动全量任务。
2. 用新的输出目录运行 300 条严格匹配 track 的四臂 smoke：

   ```bash
   python scripts/compare_helix_recovery.py \
     --event-partition all --max-events 0 --max-tracks 300 --exposure both \
     --seed-iters 20 --refit-iters 10 --bootstrap-replicates 100 \
     --output-dir data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/smoke-300-tracks
   ```

   这只验证 I/O、严格身份键、四臂 mask、stationarity 和产物契约；样本不足不能形成恢复结论。

3. 在 smoke 通过后，先测量 2k/10k track 的加载、cohort、每臂 seed/refit 墙钟和峰值 RSS，决定
   是否必须先完成 §4.2 的 chunking。不得用 300-track 的线性外推承诺全量完成时间。
4. 性能/内存满足后，使用新 run directory 重跑全量：`--max-events 0 --max-tracks 0`、四臂、
   seed 200、refit 60、bootstrap 2000 / seed 6006。
5. 仅在生成 paired keys、per-stage metrics 和 summary 后，按预注册的 Q、10% practical equivalence、
   bias CI、至少 200 valid components 和 stationarity 门槛分别给 Protected/Exposed verdict。

---

## 6. 禁止的结论与工作树提醒

- 不能从本次 `failed` manifest 推断 Protected 或 Exposed 的恢复能力。
- 不能用既有 all-hit refit 结果替代该配对四臂设计的 Q / bootstrap / equivalence 结论。
- 不能因 `--max-tracks` 存在就声称它已经减少 ROOT 读取成本；当前实现不是这样。
- 不要覆盖或复用 `full-all-events-20260907` 输出目录；它是中断实验记录，必须保留。

创建本文档时工作树已有其他任务的未提交内容：

```text
M  scripts/compare_geometries.py
M  tests/test_compare_geometries.py
?? NUL
?? docs/handoff/2026-09-07-alignment-methodology-roadmap.md
?? docs/history/experiment_artifact_cleanup_2026-09-02.md
?? scripts/analysis/measure_all_hit_refit.py
?? surface_alignment_logic.pptx
```

后续实现只应新增本交接所述 loader/performance 改动及其测试；不要顺手提交、清理或覆盖上述文件。

---

## 7. 2026-09-09：修复、可观测性与分层正式实验（已完成）

### 7.1 已实施的修复

本交接原先列出的 ROOT I/O 直接根因已修复：本地普通 ROOT 文件由私有打开辅助函数显式使用
`uproot.MemmapSource`，而 URI、非本地路径和 file-like 输入仍保持 uproot 的默认处理。这使 nominal
和 misaligned 文件均在约 1--2 秒内完成 ROOT 打开，随后可以完成展平与严格配对；不再走本机失活的
`FSSpecSource` asyncio 路径。

此外已完成以下保障，均不改变四臂的 mask、严格配对或 event-cluster bootstrap 契约：

1. `occurrence` 改为每条轨迹内 **measured state** 的 ordinal，避免所有 ROOT state 的 ordinal
   造成 nominal/misaligned probe identity 错配。
2. manifest 与 stderr 增加实时阶段报告（ROOT、canonicalize、配对、cohort、每个 arm 的
   batch/seed/refit、统计、产物）；最近完成阶段会持久化到 `metrics.progress`。
3. frozen paired cohort 后按 `(target_volume, target_layer, prediction_mode)` 做确定性 SHA-256
   排序抽样。每层最多 2,000 rows、每个 event/stratum 最多 2 条不同 track、每 track/target-layer
   最多一个 probe；不足时保留全部而不重复补样。
4. primary verdict 可由 `--analysis-modes` 限定。未选 mode 仍输出 per-row metrics、groups 和
   coverage，但不污染主结论。

抽样不是 bootstrap：它固定本次四臂共享的 probe cohort；每个 bootstrap replicate 仍以 event 为
cluster，抽到某 event 时保留该 event 的全部**已入选** probe rows。

### 7.2 验证

实现后的相关测试与 smoke 均完成：

```text
python -m pytest tests/test_compare_helix_recovery.py -v  115 passed
make test                                               519 passed
100-row/stratum four-arm smoke                          completed
```

100-row smoke 目录为
`data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/smoke-sampled-100-20260909`。它只验证产物
契约；每个 interpolation stratum 仅 100 rows，故 coverage 正确标为 `underpowered`，不能作为
物理结论。

### 7.3 正式实验与结果

正式实验使用新的目录，未复用或覆盖此前失败的 `full-all-events-20260907`：

```text
data/experiments/helix_recovery/Vol17_Lay8__Vol24_Lay4/
  full-sampled-interpolation-2000-20260909-r2
```

实际命令的关键参数：

```text
--sample-rows-per-stratum 2000
--sampling-seed 6006
--max-probes-per-event-stratum 2
--analysis-modes interpolation
--seed-iters 600
--refit-iters 60
--bootstrap-replicates 2000
--bootstrap-seed 6006
```

运行完成（manifest `completed`，总耗时约 2,297 秒）。frozen cohort 有 4,003 rows：

| stratum | selected rows | coverage |
|---|---:|---|
| `17:8 × interpolation` | 2,000 | adequate（LOC0/LOC1 各 2,000） |
| `24:4 × interpolation` | 2,000 | adequate（LOC0/LOC1 各 2,000） |
| `17:8 × backward_extrapolation` | 3 | underpowered |
| 两层的 forward，以及 `24:4 × backward` | 0 | underpowered |

正式 primary analysis 只评价 interpolation。因此，不可从本次结果推断模型在首末 hit 的 forward 或
backward extrapolation 能力；这些是需要独立 boundary-target ROOT 协议补充的 coverage 缺口。

四臂统计结果如下：

* **Protected（N-P/M-P）为 `passed`。** 两层、LOC0/LOC1 的 2,000 个有效 components 均通过
  restoration、bias 和 practical-equivalence gates。这支持“排除 target hits 后，未观察到经 ACTS
  smoothing 传播到整条轨迹的间接污染”。
* **Exposed（N-E/M-E）为 `failed`。** 两层、LOC0/LOC1 都有 2,000 个有效 components，但均未通过
  practical-equivalence；例如 recovery ratio Q 分别为 `1.260`、`1.073`（17:8）和 `0.541`、
  `0.094`（24:4）。因此当前模型不能证明能从直接送入 seed/refit 的 target-hit 错位中自愈。

完整审计产物：`manifest.json`、`helix_recovery_summary.json`、`paired_track_keys.csv`、
`per_track_stage_metrics.csv` 及六张图均位于上述正式目录。该结论仅覆盖本次的 interpolation cohort；
不能外推到未覆盖的 extrapolation mode。

### 7.4 运行经验

* ROOT I/O 卡死已与 solver 长耗时区分：在 Memmap 修复后，读取/配对约两分钟完成；正式计算主要耗时
  是四个 4,003-row、600-iteration seed（每臂约 7--10 分钟）及随后 refit，而非 I/O 死锁。
* 执行环境回收前台子进程时会留下 `running` manifest。一次受此影响的临时目录
  `full-sampled-interpolation-2000-20260909` 已保留最后阶段并显式收尾为 `failed`；正式 r2 是唯一
  应用于本节科学结论的 completed run。
