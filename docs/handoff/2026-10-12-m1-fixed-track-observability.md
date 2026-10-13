# 交接：roadmap M1——fixed-track 可观测性与参考层传递（Schur P0.3）

**日期**：2026-10-12　**分支**：`feat/full-event-calibration`（未提交）
**上游**：roadmap §M1；`plan.md` §P0.3、§P9；理论评审 §5.4；P7/P8 交接 `2026-10-11-schur-p7-p8-sparse-and-loop.md`
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 40（§A–§F 预注册，§G 结果）
**性质**：只测量，不改求解器。对象是生产 `block-lm`（fixed-track）。只在 floor 臂读了一次 geometry truth（diagnostic_only）。

---

## 结论

- **M1 主判据通过**：6 个主单元（target 层 × 模式秩）在全量与 4 个事件不相交子样本上，恢复斜率 CI 全部落在 [0.9, 1.1] 内。
- **floor 次判据通过，但为平凡通过**：floor 1e-3 / 1e-5 两臂与 control 逐位相同，floor 在本行集上从未生效。
- **参考层预算**：504 个组合中有 5 个「参考敏感」，全部是 volume 17 的内层（17:2/4/6）→ 17:8。
  最坏的是 17:6 ty 平移 10 μm → 17:8 ty 相干偏移 15.2 μm（σ 的 7.9 倍，传递系数 1.52）。两个非线性臂的线性响应核对已通过。
- **Level C 的问题已有答案**：真实 target 中**没有**只有 LOC0 的 sensor，`schur-lm` 不会因此永远报 `rank_deficient`。
- 闸门：身份 `passed`；control 与 P0.1 等价（2.3e-14 mm），2268/2268 `converged`。
- 测试：`make test` 839 passed / 3 skipped / 0 failed；`make test-shell` exit 0。

## 对 P9 必须带走的三点

1. **Jacobi 口径只测共线性，不测精度。** 全量最小 μ_0 = 0.35，所有模式都是 observable，但 24:4 ty 的 σ 中位数是 160 μm
   （17:8 的平移约 2 μm）。比较 S 与 C 的谱时，要同时比较绝对 σ（`sqrt(diag(S⁻¹))` 对 `sqrt(diag(C⁻¹))`），以及 `R_i = tr(S_ii)/tr(C_ii)`。
2. **线性化点要对齐。** 24:4 的 μ 在 `a = 0` 与 `a*` 之间最多差 0.64。P9 应在同一个几何点上比较 S 与 C（`spectrum_full.npz` 里两个点都有）。
3. **参考敏感是 Schur 最可能有差异的地方**（理论评审 §7 表第 8 行）。P9 应在同样 28 层 × 3 DOF × ±、10 μm / 0.1 mrad 的扰动下
   重算传递系数，与 `reference-linear/arms/*.npz` 逐 sensor 对比。fixed-track 在 17:8 上把参考误差放大到 1.5 倍、而它的 σ 只有 2 μm；
   如果 Schur 把径迹不确定度传播进几何信息，这几个组合的比值应当下降。这是可检验的预测，跑之前要另写预注册。

## 产物（P9 直接读）

目录 `data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/m1-fixed-track-observability-20261011/`：

| 文件 | 内容 |
|---|---|
| `spectrum/spectrum_full.npz` | 按 `surfaces`（target surface 下标）排序：`{zero,star}_{information,gradient,eigenvalues,eigenvectors,absent,observable,has_data}`、`sigma`、`block_lm_weak`、`n_loc0/1`；C、c 是 u 单位 |
| `spectrum/spectrum_sub{0..3}.npz` | 子样本，只有 `star_*` |
| `spectrum/per_sensor.csv` | 可读版 |
| `control/arms/full_control.npz` | `a*`（`deltas` 6 列，physical）与逐 sensor 状态 |
| `target/arms/*.npz` | 30 个注入臂 |
| `reference-linear/arms/{vol}_{lay}_{dof}_{plus,minus}.npz` | 线性传递 `transfer`（physical），168 个 |
| `reference-nonlinear/arms/*.npz` | 17:6 ty、17:2 ty 的完整 block-LM ± |
| `report/m1_result.json` | 全部判据、单元 CI、参考表（504 行）、非线性核对 |
| `prepare/cache/{batches,tracks}.pt` | 缓存的 TrackBatches 与 ref-only refit 径迹（约 110 MB；P9 可直接复用，省去 80 分钟的 seed） |

## 新代码

| 文件 | 内容 |
|---|---|
| `scripts/analysis/measure_alignment_observability.py` | 纯函数：`jacobi_spectrum`、`statistical_sigma`、`block_lm_weak`、`target_information`（只线性化 target hit）、`inject_rows`（掩码或下标）、`reference_layers`、`response_matrix`、`mode_slopes`、`linear_transfer`、`refit_tracks`、`event_partition`、`bootstrap_mean_ci`、`paired_rms_ratio_ci`、`true_corrections`；以及按阶段续跑的驱动（每阶段一个 manifest，已存在的臂跳过） |
| `tests/test_measure_alignment_observability.py` | 18 项（O1–O6；O3/O4 用真实 `run_block_lm_alignment` 和独立线性 oracle） |
| `docs/history/ml_optimization_log.md` | Iteration 40 |

没改：任何求解器、`joint_alignment_schur.py`、P2–P8 测试。

## 遇到的问题

1. **block-LM 的成本主要由「有数据的 sensor 数 × 迭代数」决定，与径迹数关系不大。** 200 条径迹的冒烟一臂也要约 200 s，
   因为稀疏 sensor 会跑满 60 次迭代。冒烟要用 `--max-tracks N --no-truth --subsamples 0`；全量行集上 `--subsamples` 锁为 4。
2. **一个真实 bug，由冒烟抓到**：`reference_layers` 返回 surface 下标，而 `inject_rows` 原先只接受布尔掩码。
   已先补红测试，再修复（`inject_rows` 现在两者都接受）。合成 O4 测试当时传的是掩码，所以没覆盖到。
3. 机器 6 核、10 GB；全量单进程约 2.7 GB。两路并行可行，但每个全量 block-LM 臂会从 28 分钟拖到约 55 分钟。
4. `pkill -f <pattern>` 会匹配到执行它的 shell 自身；要用 `^python -u ...` 锚定，或者直接按 pid 杀。
5. 冒烟目录 `m1-smoke-2000-20261011`（control 之后中止）和 `m1-smoke-200-20261011`（完整）仍保留，数字不进入任何判据，可以删除。

## 门状态（Schur 序列）

| 门 | 状态 |
|---|---|
| P0.1 / P0.2 / P1 | 完成（P1 = outcome 1） |
| **P0.3 M1** | **完成（Iteration 40）** |
| Q6 H1 判据（WP3） | 延后，最迟在 Level D 之前定下 |
| P2–P8 | 通过（合成 / 受控注入层） |
| Level C 及以后、P9–P11 | 未开始；P9 的前置条件已满足 |

## 下一步（用户已定的顺序与决定）

1. **P9**：Schur 谱、弱模、`R_i`、`ρ^S_ij`，在同一几何点上与 `spectrum_full.npz` 对比；参考层传递按上面第 3 点重算。需另写预注册。
   真实数据上的 `schur-lm` 运行还需要 plan §14 的其余字段（包括 WP3，以及 `validate_against_truth` 会被无条件调用这件事）。
2. **P10**：非线性 / 信赖域工作区间的幅度扫描。
3. **P11**：diagonal-only Schur 消融。**用户决定：只做函数和测试，不接 CLI**（`diagonal_only(system)` 在全局累加后清零 `S_ij`、rhs 保持完整；
   按块的秩判据与终态聚合按 plan §P11 执行；回归：无共享径迹且满秩时与 full Schur 逐元相同）。
