# 交接：joint 下游影响已评估 + 全部 Adam 环节检验完毕（P2''' / P4a 关闭）

**会话日期**：2026-08-26 ~ 08-27（接续同日的 P2 / P2' / P2''）
**上一份交接**：`docs/handoff/2026-08-26-joint-refit-gauss-newton.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 20（P2'''）、21（P4a）
**分支**：`feat/material-aware-broken-helix`（**未推送、未 merge**）
**本轮提交**：无。**本轮零产品代码改动**，两项工作都是观测
**工作树状态**：`docs/history/ml_optimization_log.md` 与两份交接文档未 commit

---

## 0. 一句话状态

GN 替换的下游影响已在全量上确证：目标层几何恢复从「tx 尚可、ty 失败」变为
**tx/ty 双双近乎完美**（corr ≥ 0.95，recovery ≈ 1.0），`17:8` 的 ty 从**反号**
（corr −0.115）纠正为 **+0.99993**，`best_total_loss` 2286.55 → 14.16。
`seed_helices` 的 Adam 预拟合经测量**确实停在非驻点**（全量 99.98% 的轨迹相对
decrement > 1%），**但已被 GN refit 吸收，不构成下游缺陷**。
**至此 seed → refit → 下游求解三个环节的 Adam 全部检验完毕。**
下一步由用户在 P4a'（可选优化）、P3、P4b、P5 之间选。

---

## 1. 本轮做完的两件事

| # | 内容 | Iteration | 性质 |
|---|---|---|---|
| P2''' | joint refit 换 GN 后的下游影响评估：两臂在 300 轨迹与全量上各跑完整下游求解，按预注册协议比较 | 20 | 观测，零代码改动 |
| P4a | `seed_helices` 的 Adam 预拟合驻点性测量（300 轨迹 + 全量）| 21 | 观测，零代码改动 |

两项的比较协议/判据都**写在跑之前**（Iteration 20 §1、21 §1），事后只填数字。

---

## 2. P2''' 结果（可引用）

全量 `Vol17_Lay8__Vol24_Lay4`，98958 条输入选中 61482 条、852859 hits。
两臂只差 `--joint-refit-solver` 一个 flag，墙钟各约 1.6 小时。

### 2.1 核对块先于效果（全部逐位相同）

`n_tracks` 61482、`n_hits` 852859、`input/retained/excluded_tracks`
98958/61482/37476、`fit_hits` 709672、`invalid_fraction` 0.0，各层 `n_components`
（17:8 → 71345，24:4 → 71842）**两臂逐位相同**。改善不是丢样本换来的。

初始化数字逐位复现 Iteration 19 §7：adam 7595.34 / 100% / 4.1e5，
GN 0.763503 / 0.00% / 2.61e-17。

### 2.2 主指标 `truth_metrics`

| 层 | DOF | adam (corr / recovery) | **gauss-newton** |
|---|---|---|---|
| 17:8 | tx | +0.967 / +0.874 | **+0.9998 / +0.9986** |
| 17:8 | ty | **−0.115 / −0.275** | **+0.99993 / +0.99973** |
| 17:8 | rz | +0.956 / +0.711 | **+0.987 / +0.832** |
| 24:4 | tx | +0.941 / +0.938 | **+0.9998 / +0.9997** |
| 24:4 | ty | +0.336 / +0.679 | **+0.954 / +1.017** |
| 24:4 | rz | +0.982 / +0.811 | **+0.989 / +0.843** |

`pred_rms` 同时从过冲收敛到真值尺度：17:8 ty **1.2397 → 0.5214**
（`true_rms` 0.5215）。**Adam 臂不只是解得不准，而是把 ty 解成了 2 倍幅度的过冲，
且 17:8 上符号还是反的。**

### 2.3 下游 chi2

`best_total_loss` **2286.55 → 14.16**（162×）；`joint` 阶段 17:8 LOC1 的
per-component chi2 **8755.62 → 6.73**（1302×）。两臂 `best_iteration` 同为 19、
`stopped_early` 同为 False——**同一迭代数下比较**。

### 2.4 判读

- **17:8**：GN 臂优于 Adam 臂（三个有信号 DOF 全面提升，ty 符号纠正）。
- **24:4**：GN 臂优于 Adam 臂（LOC0 数字按 P1' 不进结论）。

**Iteration 19 的求解器替换在下游得到确证，无需回退。** 这同时回答了 Iteration 18
提出的归因问题：Adam refit 停在非驻点不是无害的收敛不足，它给 block-LM 提供了系统性
偏斜的起点，使下游把 `17:8` 的 ty 解到了反方向。**归因对了，一个求解器替换就够了。**

---

## 3. P4a 结果（可引用）

### 3.1 测什么，为什么不能套 1e-6

`initialization_stats.measurement_refit.stationarity` 里已有的 `seed` 项测的是
**seed 输出在 measurement 目标上**的驻点性（即 refit 的起点），不是 seed 自己的目标。

`seed_helices`（`scripts/ml_alignment_autograd.py:548`）的目标是**非目标 SMT 全局点的
几何距离平方和**（mm²，**无协方差加权**），而预注册的
`NEWTON_DECREMENT_STATIONARY = 1e-6` 是在 `F = 0.5·chi2` gauge 下定的——
**gauge 不同，阈值不可移植**。故改用两个无量纲判据，绝对 decrement 只记录不比较。

### 3.2 结果

全量 61482 轨迹 / 709672 fit hits。基线 SMT loss **1.6941403** 与 pipeline 打印的
`SMT loss=1.69414` **逐位一致**（测量脚本确实复现了产品代码），
`n_effective = 709672` 与 `fit_hits` 相同且**全程未变**。

| 判据 | 300 轨迹 | 全量 |
|---|---|---|
| 基线 SMT loss（200 iters）| 1.7712071 | 1.6941403 |
| +400 Adam | 0.054073131（−96.9%）| 0.065015727（−96.2%）|
| **+800 Adam** | **0.0030914347（−99.8%）** | **0.0029251182（−99.8%）** |
| 相对 decrement `dec/F` p50 | 0.2408 | 0.2384 |
| `dec/F` p90 | 0.6495 | 0.7083 |
| `dec/F > 1%` 占比 | 100.00% | **99.98%** |
| `dec/F > 10%` 占比 | 86.67% | **87.50%** |

绝对 decrement（mm²，仅记录）：全量 p50 1.909、p90 4.567、p99 11.35、max 42.13。

LM 直接求解同一目标：300 轨迹 30 次迭代 → loss 0.0040002676，`dec/F` p50 **1.02e-08**；
全量为控耗时只跑 15 次迭代 → loss 0.044719603，`dec/F` p50 0.0798。
**全量这一项是迭代预算不足，不是全量更难**，只作方向佐证，不作幅度结论。

### 3.3 判读

**`seed_helices` 的 Adam 预拟合确实停在非驻点**，中位轨迹目标还有 23.8% 可降，
继续跑 800 次即可把 SMT loss 降低约三个数量级。与 Iteration 18 对 refit 的诊断同构。

**但「有问题」不等于「要改」，下游已被兜住**：同一次全量运行里，GN refit 从这个
非驻点 seed 出发达到 measurement 非驻点行 **0.00%**（5 次迭代收敛），下游 tx/ty
recovery ≈ 1.0。seed 的非驻点性**已被 GN refit 吸收**，可提升空间被 recovery ≈ 1.0
的天花板限制。故 P4a 关闭为「确有非驻点，当前不构成下游缺陷」。

---

## 4. 实验经验

### 4.1 `--joint-iters` 在默认求解器下是死参数（上一份交接的做法无法落地）

上一份交接给 P2''' 写的做法「改用完整 `--joint-iters`（默认 1000）」**落不了地**：

- 默认 `--joint-solver block-lm` 走 `run_block_lm_alignment`
  （`scripts/ml_alignment_autograd.py:1075`），迭代上限是 **`--lm-max-iters`
  （`LM_MAX_ITERS = 20`）**，**完全不消费 `config.joint_iters`**。
- 消费 `joint_iters` 的主循环（`scripts/ml_alignment_autograd.py:1332`）属于
  `--joint-solver adam` 分支。

推论：**Iteration 19 §7 的全量两臂也跑满了 block-LM**，那里的 `--joint-iters 1`
当时就是死参数。§7 说「只验证了 refit 本身」并不准确——下游其实已经算了，只是没保存
summary、没做比较。**P2''' 缺的是比较，不是运行。**

### 4.2 block-LM 不打印进度，日志安静不等于没跑

block-LM 的 per-iteration 诊断只进 `fisher_diagnostics`，不打印；加上 Python 重定向到
文件是块缓冲，**长跑期间日志可以一小时毫无更新而进程完全正常**。判断进度要看
`/proc/<pid>/stat` 的 utime（本轮实测 11 线程、CPU 时间约为 elapsed 的 2.9 倍）与
`/proc/<pid>/fdinfo`（ROOT 文件已关闭 = 数据加载阶段已过），**不要看日志**。

### 4.3 全量耗时不线性于轨迹数——我按 205 倍外推错了一个数量级

300 轨迹单臂 220 s，我据此按 component 的 205 倍外推出「12.5 小时/臂」并据此打断了
用户的决策流程，**实测只有 5831 s（1.6 小时）**，约 26 倍。原因：block-LM 按 sensor
分块，每个 sensor 只处理打到它的 hits，sensor 数不随轨迹数增长。
**下次排期先跑一个中间尺度，不要从小样本线性外推 block-LM 的墙钟。**

### 4.4 单臂 RSS 峰值约 6.6 GB，不是 3 GB

上一份交接记的「单臂峰值约 3 GB」偏低。全量实测 RSS 6.6 GB，本机可用约 8 GB——
**两臂并行必然 OOM，必须串行**。

### 4.5 `--output-geom` 默认两臂共用一个路径

默认写 `data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_v17l8_v24l4_ml.json`。
A/B 时**必须分臂显式指定**，否则第二臂覆盖第一臂，并污染下游 reconstruction 的输入
几何。本轮全部写 scratchpad，已核对该文件 mtime/size 逐位未变。

### 4.6 换 gauge 就不能沿用阈值

见 §3.1。`1e-6` 只在 `F = 0.5·chi2` gauge 下有意义；SMT 目标是 mm² 的几何距离平方和。
移植阈值前先确认 gauge 相同，否则「非驻点比例」是个无意义的数。

### 4.7 「多跑几次 Adam」不是安全的修法

`seed_helices` 的 lr = 0.5 在约 1000 次迭代后**发散**（300 轨迹上 +1800 → 38.2，
+2800 → 102.9，且 `n_effective` 未变，是真发散不是样本变化）。
**不要调大 `JOINT_SMT_SEED_ITERS`**——那会把一个「未收敛但稳定」的 seed 换成
「可能发散」的 seed。若要改，应换求解器而非加预算。

### 4.8 小样本能定方向，不能定幅度

P2''' 与 P4a 在 300 轨迹与全量上结论均同向，但 P2''' 的改善幅度差很多
（17:8 ty corr：小样本 +0.525，全量 +0.99993）。`--max-tracks 300` 仍可做快速筛查，
**但任何幅度性陈述必须全量**。

### 4.9 沿用既有教训（未变）

- **任何让结果变好的改动，先查行数/样本量**。本轮两处都做了：P2''' 的核对块全部逐位
  相同后才读效果指标；P4a 的 SMT loss 除以 effective 分量数，helix 跑飞会缩小分母而
  伪造改善——实测 `n_effective` 全程恒为 709672，改善是真的。
- 归因错了，所有候选修法都会错——本轮反过来印证：**归因对了，一个求解器替换就够了**。
- 不救活秩亏行；gate 判据不为通过而放宽。
- **必须用 conda pytorch 3.10**，默认 `python` 3.9 会把干净仓库报成 35 个假失败。

---

## 5. 未完成任务

### P4a' — 替换 seed 求解器（**可选优化，非修复**）

P4a 已证明 seed 停在非驻点，但下游被 GN refit 兜住。若要做，判据**不应**是
「SMT loss 是否更低」（已确定会更低约三个数量级），而应是下游 `truth_metrics` 是否
还有可测量的改善。**在 recovery 已 ≈ 1.0 的情况下，必须先预注册「多大的变化才算真
改善」**，否则只会测到噪声。

参考实现已在 `scratchpad/p4a/measure_seed.py` 的 `gn_solve` 里（per-track LM，
30 次迭代可把 `dec/F` p50 压到 1e-8），但那是离线测量代码，未按产品标准写。

### P1' — `24:4|backward_extrapolation|LOC0`（未变）

唯一仍对 damping 真实敏感的组（敏感度 2.223 = 统计不确定度的 2.2 倍），
NLL 结论**不得引用**。它同时是唯一有秩亏行、Iteration 14/16 唯一回退的组。
本轮该组的 LOC0 数字虽同向改善，但**未进任何结论**。

### P3 — `min_extrapolation_fit_hits = 4` 过松（未变）

single 臂 4 hits 撑 5 个 gauge-fixed 参数（插值组是 8）。属另一次预注册变更。

### P4b — 其他（未变）

- 结论仍只在 20000 events 上，未在 100000 events 上复现。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动，**Gate 4 已确认 failed**
  （三次复现，插值组改善仅 0.3–1.8%）。

### P5 — 分支处置（未变）

`feat/material-aware-broken-helix` 未推送、未 merge。**需用户明确批准。**
另有未跟踪文件 `surface_alignment_logic.pptx`，历次提交均未纳入。

### 文档提交

本轮的 `docs/history/ml_optimization_log.md`（Iteration 20、21）与两份交接文档
**尚未 commit**，等用户确认。

---

## 6. 环境（本轮实测）

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib   # 否则 awkward_cpp 缺 CXXABI_1.3.15
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q    # 3.10.20 / torch 2.11.0+cu126
```

**337 passed / 3 skipped / 11 deselected**（与 Iteration 19 基线一致；本轮零代码改动，
测试未再跑第二次）。3 项 skip：AlignDerivLib pybind 未编译 ×1、
**ACTS bindings 不可用 ×2**——**要用 ACTS 时当场实测确认，勿凭这两条 skip 或凭
memory 下结论。**

本机无 `/usr/bin/time`，墙钟用 `date +%s` 差值记录。GPU 存在（RTX 3050）但全部运行均
**不加 `--gpu`**，保持 CPU float64 与 Iteration 19 §7 同配置可比。

**不要用 `make test` 的默认 `python`**（3.9），它会把干净仓库报成 35 个失败。

### 本轮产物（scratchpad，会话结束后可能清理）

```
.../scratchpad/p2ppp/
  summary_{300,full}_{adam,gauss-newton}.json   # P2''' 比较依据
  log_{300,full}_{adam,gauss-newton}.txt
  geom_{300,full}_{adam,gauss-newton}.json
  compare_summaries.py                          # 一次性比较脚本，未入库
  run_stage.sh                                  # 两臂串行驱动

.../scratchpad/p4a/
  measure_seed.py                               # P4a 离线驻点性测量，未入库
  report_{300,full}.json  log_full.txt
```

复跑命令：

```bash
# P2'''：两臂只差 --joint-refit-solver，其余全默认；--output-geom 必须分臂
python scripts/ml_alignment_autograd.py --track-mode free \
  --joint-refit-solver {adam|gauss-newton} \
  --output-geom <分臂路径> --output-summary <分臂路径> --no-plot

# P4a：纯测量，不改产品代码
python measure_seed.py --max-tracks {300|0} --budget 400,800 --gn-iters 30 --out report.json
```
