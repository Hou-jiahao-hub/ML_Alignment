# 交接：主线复位 —— free-helix 首次通过端到端盲验证（trial-008）

**会话日期**：2026-08-27
**上一份交接**：`docs/handoff/2026-08-27-extrapolation-residual-floor.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 23（trial-007 补记）、
Iteration 24（trial-008 预注册 + 结果 + 盲测）
**分支**：`feat/material-aware-broken-helix`（**未推送、未 merge**，领先 main **19 个提交**）
**本轮提交**：`f0474a1`（交接修正）、`c4a2fff`（编排器修复）、`8194ff0`（Iteration 23/24）、
`0c25d2f`（trial-008 结果）
**工作树状态**：干净。唯一未跟踪的是 `surface_alignment_logic.pptx`

---

## 0. 一句话状态

**主线 free-helix 首次通过端到端盲验证**（trial-008，`passed = true`，
validation 四分量全过，`acceptance_score` 从 trial-007 的 `[3, 15.43, 24.72]`
变成 `[0, 0.96, 0.0]`）。**唯一变更是 measurement refit 由 Adam 换 GN**——
实操上就是「用当前代码重跑」，因为 GN 自 `9f6a951`（2026-08-26 13:47）起已是默认，
而此前唯一一次主线运行 trial-007 跑于 8-25 15:08，**早 22 小时**。
**§9 的一次性 validation 已用掉，本轮结果不得用于继续调参。**

---

## 1. 用户本轮的路线决策

- **free-helix 为主线**，broken-helix 降为后续方向。先完成主线优化与全线校准。
- **澄清一个长期误读**：交接文档里反复出现的「Gate 5 规定 Gate 4 未通过不得启动」
  **只封锁「把材料 kink 模型接进 joint」**，**不封锁主线**。主线用
  `helix_model.py` 的六参数单 helix，与 broken-helix 无关。**回主线不需要解锁任何 Gate。**
- trial-008 **先跑到 calibration 为止**（`--skip-blind-validation`），
  确认配置无需再调后，**再单独动用**那一次性盲测。

---

## 2. 本轮做完的四件事

| # | 内容 | 提交 | 性质 |
|---|---|---|---|
| A | 修正上一份交接被其自身提交改过期的三处表述 | `f0474a1` | 文档 |
| B | 编排器 finalize 段三处修复 + 新脚本 + 测试 | `c4a2fff` | **代码** |
| C | Iteration 23（trial-007 补记）+ 24（trial-008 预注册）| `8194ff0` | 文档 |
| D | trial-008 结果与盲测记录 | `0c25d2f` | 文档 |

---

## 3. 结果（可引用）

### 3.1 主线曾有一个两天没人知道的空缺

`ml_optimization_log.md` 一直写着「trial-007 待运行」。**它 2026-08-25 就跑完了，
并且 failed**（三分量失败，17:8 LOC1 差 15 倍）。此后项目转去做 broken-helix 支线，
结果从未回填。任何只读文档的人都会以为主线还没跑过端到端。已补记为 Iteration 23。

**trial-007 的 joint 阶段是净损失**：calibration 上 17:8 LOC1 从 warm2 的 20.13
被推到 **335.97**，守卫回退到 warm2，**最终几何完全绕过了 joint**。
根因是 measurement refit 撞满迭代上限从未收敛（`chi2/component = 7595.638`，
`best_iteration = 199`）。

### 3.2 核对块（trial-008，四步全过）

1. **输入哈希**：10 个输入与 trial-007 **全部相同**。
2. **event split**：seed 6006、80/10/10，两次一致。
3. **阈值 profile**：四个阈值逐位相同，nominal states sha256 相同（复用未重标定）。
4. **样本量一行未动**：`n_tracks` 49222、`n_hits` 682915、`fit_hits` 568283、
   `retained/excluded` 49222/29959 **全部逐位相同**。改善不是丢样本换来的。

### 3.3 主判据：通过，压倒性

| 阶段 | acceptance_score（calibration） |
|---|---|
| trial-008 warm2 | (4, 20.131, 35.945) |
| trial-007 joint_free | (4, 335.97, 396.90) |
| **trial-008 joint_free** | **(0, 0.9843, 0.0)** |

| 层:分量 | trial-007 | **trial-008** | 阈值 | 倍数 |
|---|---|---|---|---|
| 17:8 LOC0 | 42.74 | **0.7158** | 1.0 | 60× |
| 17:8 LOC1 | 335.97 | **0.7919** | 1.6106 | **424×** |
| 24:4 LOC0 | 19.20 | **0.8695** | 1.0 | 22× |
| 24:4 LOC1 | 3.01 | **0.9843** | 1.0303 | 3× |

`joint_adopted = true`，`polish_attempts = 0`（polish 首轮即 break，已无失败分量）。
**这是 joint 阶段第一次在主线上产生正贡献。**

### 3.4 次判据：通过

`measurement_refit.chi2_per_component` **7595.638 → 0.7631458**，
`invalid_fraction = 0.0`，与离线 GN 全量的 0.7635 一致。
Adam 分支专有的 `best_iteration = 199 / improved` 字段不再出现。

### 3.5 Blind validation：**四分量全过**

| 层:分量 | chi2/component | threshold | normalized | 判定 |
|---|---|---|---|---|
| 17:8 LOC0 | 0.7042 | 1.0 | **0.7042** | pass |
| 17:8 LOC1 | 1.3602 | 1.6106 | **0.8445** | pass |
| 24:4 LOC0 | 0.8851 | 1.0 | **0.8851** | pass |
| 24:4 LOC1 | 0.9890 | 1.0303 | **0.9600** | pass |

`failed_components = []`，`passed = true`，`stop_reason = "blind_validation_passed"`。
`n_components` 7096 / 7239，与 trial-007 的 validation 分区逐位相同。

**通过验收的几何**：
`data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_trial-008_joint_free.json`

### 3.6 truth_metrics 全线转正（`diagnostic_only`）

| 层/DOF | trial-007 corr / recovery | **trial-008** |
|---|---|---|
| 17:8 tx | +0.5143 / +0.8967 | **+0.9988 / +1.0006** |
| **17:8 ty** | **−0.0778 / −0.4511** | **+0.9996 / +0.9996** |
| 17:8 rz | +0.2663 / +0.4362 | **+0.9981 / +0.9967** |
| 24:4 tx | +0.4047 / +0.8131 | **+0.9961 / +0.9917** |
| 24:4 ty | +0.0296 / +0.0573 | **+0.9214 / +0.8817** |
| 24:4 rz | +0.3493 / +0.6970 | **+0.9892 / +0.9826** |

17:8 ty 从**负恢复**（对齐比不对齐更糟）转为 +0.9996。

### 3.7 意外收获：两个待测隐患自行消解

| 量 | trial-007 | trial-008 |
|---|---|---|
| `best_iteration` | 19（= `LM_MAX_ITERS − 1`）| **15** |
| `stopped_early` | false | **true** |
| `correction_max_abs` | **1.5**（精确等于 `JOINT_CLAMP_T`，饱和）| **1.3695** |
| `best_total_loss` | 2323.018 | **2.2143** |

**block-LM 首次在预算内收敛**，几何修正不再撞硬夹逼。根因与 refit 未收敛同源：
refit 不收敛 ⇒ 轨迹参数错误 ⇒ 几何求解被迫做大修正、撞 clamp 且永远收敛不了。
**这两条不需要单独修。**

### 3.8 编排器的三处修复（`c4a2fff`）

原 finalize 段是一行流内联 Python，已提取为
`scripts/write_experiment_summary.py`（因为一行流无法测试）：

- **`stop_reason` 不再被丢弃**。原代码把 `$stop_reason` 作为 `sys.argv[7]` 传入
  却**从不引用**，硬写验证结论。于是 trial-007 的 summary 写着
  `blind_validation_failed`，**完全看不出 joint 曾被回退**（真实状态是
  `joint_not_improved`）——复原它只能手工比对五个 acceptance 文件。
  现拆为 `selection_stop_reason`（选择阶段）与 `stop_reason`（验证结论）。
- **`joint_effective` 改为反映几何是否真被采纳**：
  `= joint_solver_effective and joint_adopted`。原实现只读 solver summary，
  于是被丢弃的 joint 仍报 `true`，**Iteration 12 的通过条件可以在 joint 被完全
  丢弃时满足**。solver 层原语义保留为 `joint_solver_effective`。
- **新增 `--skip-blind-validation`**（默认关闭，默认行为不变）：停在 calibration，
  `passed` 写 null，`workflow` 记为 `...calibration-only`，
  且不把不存在的 validation 产物登记进 manifest。

**回归验证**：用新路径重建 trial-007 的 summary，无字段丢失、schema 2 保持，
唯一变化是 `joint_effective` 由 `true` 纠正为 `false`。

---

## 4. 实验经验

### 4.1 「改进已落地」不等于「结论已更新」——要查它有没有真的跑过

GN 早就是默认值，编排器不传 flag 就继承它，代码层面无可挑剔。
但**编排器自 trial-007 之后再没跑过**，而 trial-007 早于 GN 落地 22 小时。
于是「主线 joint 无效」这个结论在文档里挂了两天，其实只是没人重跑。
**下次看到「某改进已合入默认」时，先去查最近一次端到端运行的时间戳与 git_commit，
而不是查代码。**

### 4.2 汇总文件会把自己的失败原因盖掉

trial-007 的 `stop_reason` 是假的——不是写错，是那段代码**根本没读**传进去的参数。
主线唯一的成败记录文件，在最关键的问题（joint 有没有被采纳）上给出了误导性答案。
**动手重跑之前先确认：这次运行的产物，能不能回答这次运行想问的问题。**
本轮正是因为先修了它，trial-008 的 `joint_adopted = true` 才是可直接引用的。

### 4.3 一行流内联 Python 是测试盲区

那三个缺陷全在同一个 `python3 -c '...'` 长表达式里，共存了很久。
提成脚本后 7 个测试立刻覆盖住，其中两个直接钉住了本轮修的语义。
**编排器里凡是带判断逻辑的内联 Python，都应视为未测试代码。**

### 4.4 要求两次运行逐位相同是错的判据（**再次**印证）

warm1/warm2 本轮代码未动，两次运行的 chi2/component 仍有 **1e-7 ~ 1e-6 相对漂移**
（`n_components` 7155/7170 逐位相同，四分量判定不变）。是浮点非确定性
（torch 归约顺序或 ACTS 线程），不是真实变化。
**对照 joint 阶段的 22×–424×，两者差六个数量级。**
应判「行集合不变 且 漂移不超过舍入量级」——与 Iteration 22 §4.2 得到的是同一条教训，
这次出现在完全不同的流水线上。

### 4.5 归因对了，一堆看似独立的问题会一起消失

进场时列了三个独立待办：joint 无效、block-LM 预算 binding、几何修正撞 clamp。
换 GN 之后**三个同时消失**，因为它们是同一个根因的三个表现。
**沿用 Iteration 16 的反面教训**（把两个现象错当同一根因）——正反两个方向都要小心：
既不要把无关的现象强行归成一因，也不要把同一因的多个表现当成多件事去逐个修。

### 4.6 沿用既有教训（未变）

- **任何让结果变好的改动，先查行数/样本量。** 本轮核对块第 4 步确认
  49222/682915/568283 逐位相同，才敢说改善不是丢样本换来的。
- **核对块先于效果。** 四步全过之前不引用任何效果数字。
- **必须用 conda pytorch 3.10**，默认 `python` 3.9 会把干净仓库报成 35 个假失败。
- **ACTS 可用性每轮当场实测**，不得凭 pytest 的两条 skip 或凭记忆。本轮实测可用。

---

## 5. 未完成任务

### ⚠️ M0 — 一次性 validation 已用掉（**不可逆状态，最先要知道**）

§9 规定「最终 validation 只运行一次，不得据此继续调参」。本轮已用掉。
**后续任何配置变更若要再评 validation，必须先按 §9 处理**：
或显式声明 trial-008 的结果属于旧配方，或改 `--event-split-seed`（默认 6006）
重新切分并重新标定阈值 profile（代价：与 trial-007/008 不再可直接比较）。
**trial-008 的结果本身不得用作继续调参的依据。**

### M1 — manifest 与 summary 不一致（小，但要么补要么记）

trial-008 以 `--skip-blind-validation` 跑完，manifest 在盲测之前就 finalize 了
（`status=completed`，`git_commit=8194ff0`），**其 outputs 列表不含
`acceptance_trial-008_validation.json`**，而 summary 现已含盲测结果。
未回改 manifest——那等于伪造运行记录。**正确做法是补一条说明，或把盲测记为
一次独立的、带自己 manifest 的运行。需要决定。**

### M2 — 主线剩余弱点（都在主线，非支线）

- **`24:4 ty` 是六个 DOF 里最弱的**（corr 0.9214 / recovery 0.8817），
  `24:4 LOC1` 的 normalized 0.9600 也是四分量里裕度最小的。
  ShortStrip `ty` 弱可观测是**已知物理限制，不是算法 bug**。
- **`seed_helices` 仍是 Adam**（`ml_alignment_autograd.py:548-579`，硬编码、
  无 flag、无驻点检查；P4a 实测 99.98% 轨迹停在非驻点）。
  它**没有成为瓶颈，但也未被证明无害**。参考实现
  `scripts/analysis/measure_seed.py:119` 的 `gn_solve`。
  **不得用调大 `--joint-smt-seed-iters` 来糊弄**——同一 lr=0.5 在约 1000 次迭代后
  发散（Iteration 21）。要动就换求解器，且必须**先预注册「多大变化才算真改善」**。
- **block-LM 虽首次早停收敛，仍无驻点门控**。measurement refit 有预注册的驻点
  raise（`ml_alignment_autograd.py:851-866`），几何 block 一个都没有。
  若要测，沿用 Iteration 18 的套路——先只测量，可复用
  `validate_helix_model.py:3218` 的 `newton_decrement` / `stationarity_report`。
- **`--joint-iters` 是死参数**（`run_block_lm_alignment` 只消费 `--lm-max-iters`；
  消费 `joint_iters` 的循环在 `--joint-solver adam` 分支）。编排器仍传
  `JOINT_ITERS=1000` 并做正数校验。误导性参数，未清理。
- **流水线非逐位可复现**（§4.4）。未定位、未记录到 `docs/architecture.md`。

### M3 — 泛化性未验证

**结论只在 `Vol17_Lay8__Vol24_Lay4` 这一组目标层上成立。**
`data/reconstruction/nominal/` 下另有 `Vol16_Lay6__Vol23_Lay4`、
`Vol17_Lay2__Vol24_Lay6` 等组合，均未复现。

### M4 — 分支处置（**需用户明确批准**）

`feat/material-aware-broken-helix` 领先 main **19 个提交**，无 upstream，
**全部工作只存在于本地一处**，其中包括主线唯一一份通过验收的几何的产出记录。
另有未跟踪文件 `surface_alignment_logic.pptx`，历次提交均未纳入
（文件名看起来正是主线的逻辑图，需决定是否纳入版本管理）。

### S1 — broken-helix 支线（**已挂起**）

- **P1'** `24:4|backward_extrapolation|LOC0` 的 damping 敏感度 **2.545** 未归因，
  该组 NLL **不得引用**。欠定行已被排除为根因，下一候选未定。
- **Gate 4 维持 failed，四次复现**（插值组改善 0.18%–3.18%，门槛 20%）。
- **P4b** 支线结论仍只在 20000 events 上。**注意：这条只适用于支线**——
  主线本来就跑 100000 events（`EVENTS=100000`）。
- **工作包 ④** joint 校准接入仍被 Gate 5 阻塞。与主线无关。

---

## 6. 环境与产物（本轮实测）

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib   # 否则 awkward_cpp 缺 CXXABI_1.3.15
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q    # 3.10.20
```

**346 passed / 3 skipped / 11 deselected**（上轮基线 339 + 本轮新增 7）。
3 项 skip：AlignDerivLib pybind 未编译 ×1、**ACTS bindings ×2**——
**要用 ACTS 时当场实测确认**（本轮实测可用：`/home/houjiahao/alignTest/build/python/acts`）。

**不要用 `make test` 的默认 `python`**（3.9），它会把干净仓库报成 35 个假失败。

### 本轮产物（`data/` 下，不入 git）

```
data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/
  trial-008/          # 本轮主线运行，manifest status=completed，git_commit=8194ff0
                      # 2026-08-27T06:01:24Z → 07:22:54Z（约 1h21m）
                      # 含 acceptance_trial-008_validation.json（盲测，manifest 未登记，见 M1）
  trial-007/          # 2026-08-25 的失败运行，未被覆盖

data/alignment/Vol17_Lay8__Vol24_Lay4/
  aligned_transforms_trial-008_joint_free.json   # ★ 主线首份通过验收的几何
```

磁盘：`data/reconstruction` 已占 108 GB，根分区余 732 GB。
单次重建约 2.5 分钟 / 100000 events，产出约 1.09 GB ROOT。

### 复跑命令

```bash
# 主线全流程（约 1h21m，5 次 ACTS 重建）；run tag 已存在会直接 die
./scripts/run_surface_helix_experiment.sh --run-tag <新 tag>

# 停在 calibration，保住盲测切分
./scripts/run_surface_helix_experiment.sh --run-tag <新 tag> --skip-blind-validation

# 单独评一次 validation（⚠️ 见 M0，一次性）
python scripts/reconstruction_acceptance.py \
  --states <最终 trackstates.root> \
  --output <acceptance_*_validation.json> \
  --target-layers 17:8,24:4 \
  --threshold-profile <acceptance_baseline.json> \
  --event-partition validation --split-seed 6006
```
