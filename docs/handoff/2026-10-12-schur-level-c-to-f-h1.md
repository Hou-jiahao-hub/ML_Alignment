# 交接：WP3 H1 判据 + Schur Level C–F + P9 / P10 / P11（H1 闭环完成）

**日期**：2026-10-12（提交于 2026-10-13）　**分支**：`feat/full-event-calibration`
**上游**：`2026-10-12-m1-fixed-track-observability.md`；`plan.md` §P9–§P11、§13、§14、§6.2
**预注册与数字**：`docs/history/ml_optimization_log.md` Iteration 41（H1 判据与 §14）、42（Level C/D、P9，§H 结果）、43（P11）、
44（P10）、45（Level E，§E 结果）、46（Level F 的实现口径与结果）
**性质**：truth 只在 Level F 读了一次。`schur-lm` 一律由独立 driver 调用，不经过 pipeline 的 `validate_against_truth`。

---

## 结论

- **WP3 已定**（用户选择「非劣 + 定向改善」，Iteration 41）：
  - H1-P（否决）：几何六格与径迹 truth 距离两项，paired CI 上界都 ≤ 1.01；
  - H1-D（改善声明）：只看 17:8 tx、17:8 ty、24:4 ty，CI 上界 < 0.90；
  - H1-R：参考层传递，与 truth 无关。
- **Level C/D**：全量 `schur-lm` 用 3 次外迭代 `converged`（160 s，4.2 GB）；λ_init 扫描的最大差为 0.0032σ。
  **Schur 候选**：`data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_p9-schur-observability-20261012_schur_lm.json`，
  SHA256 `3395fd1e19197f10ed4cf75532bcd3318a87dede32c3b6e9bc8309e8a7014738`。
- **P9 谱**：没有全局弱模（最小 μ = 0.358）。R_i 在 17:8 约 0.55、在 24:4 约 0.87；17:8 ty 的 σ 从 1.93 μm 增到 3.27 μm（1.69 倍），24:4 ty 不变。
- **P9 主判据：按预注册判为未通过，原因只是 `not_evaluable`**。`sub3_ty_plus` 用完了 60 步外层预算；所有可评估格的 CI 都在 [0.9, 1.1] 内。
- **H1-R**：5 个参考敏感组合中只有 1 个下降（17:4 ty，0.891）；「不依赖 nominal 参考」次判据通过。
- **P10**：s = 0.5–2 时首步 ρ ≥ 0.9988；s = 2 时为 `constraint_stalled`。真正的限制是盒约束，不是非线性。
- **P11**：函数与回归通过，没有接 CLI。迭代到收敛时，diagonal-only 与 full 的不动点相同，所以这个消融不能比较点估计。
- **Level E 通过**：score 从 `[4, 1002, 1544]` 改善到 `[0, 0.9694, 0]`，四个分量都通过；阈值 profile 与 Iteration 30 相同。
- **Level F（只运行一次）**：**H1-P 通过；H1-D 只有 17:8 ty 改善**（0.795 [0.771, 0.821]）；没有被证伪。
  几何 ratio：17:8 tx/ty/rz 为 0.940 / 0.795 / 0.911，24:4 tx/ty/rz 为 0.981 / 0.999 / 0.978。
  径迹 truth 距离：target hit 0.692，ref hit 0.837（C 臂的径迹用了 target hit，A 臂没有，这部分改善不能全部归因于 Schur）。

## 未完成任务（按建议顺序）

1. **独立模拟样本复现 H1-D**（roadmap §H1 的正式要求）。100k 样本的 truth 已经用掉，目前的改善只能声明为 `post_hoc_single_sample`。
   做法：用同一错位文件、新的事件种子重新模拟和重建（需要 ACTS，先实测环境）；用同一冻结流程重跑 fixed-track 与 `schur-lm` 两臂，
   不改任何常量；再用 `run_h1_truth_veto.py` 读一次 truth。运行前另写预注册，**事先指定**只复现 17:8 ty 一格，还是三格都复现。
2. **约束策略（P9 中预算用完的根因）**：每次撞界都把整步统一缩放，而且下一次迭代才冻结，结果是迭代数随撞界 DOF 数线性增长
   （子样本 ty 臂 43–61 次，s = 2 时 `constraint_stalled`）。建议的方案 B：在同一次线性化内，把阻挡 DOF 冻结后重解剩余步。
   需要新的 Iteration、合成回归（迭代数不随撞界 DOF 数增长，且与旧路径落在同一个 KKT 点），再用新的 run-id 补跑 P9 sub3，
   作为修订后的补充复现**单独报告**，不改写 Iteration 42 §H。用户尚未决定是否做。
3. **`run_candidate_acceptance.sh` 没有 shell 测试**：应在 `make test-shell` 中加一个 dry-run 契约测试（路径解析、SHA 参数必填）。
   dry-run 已经抓到过一次相对路径错误，说明值得加。
4. **P9 `_primary` 的标签**：`not_evaluable` 的格被标成了 `unobservable_or_prior_dominated`（沿用 M1 的二分标签）。复用前应改成三态，
   并补一个测试；本轮 report 文件不重写，日志 Iteration 42 §H 已说明。
5. **P11 在真实数据上的运行**：没有做。Iteration 43 §B 说明它不会改变点估计，只有在需要比较收敛行为时才有意义；做之前另写预注册。
6. **结论文档**：plan §16 的最后一项「Update optimization history and finalize manifests」已经完成。H1 闭环的对外总结（报告或论文段落）还没写，
   而且应当在第 1 项完成后再写。
7. **工作树里不属于本轮的未提交改动**（本次没有提交，留给对应的会话处理）：
   - 日志中的 Iteration 31/32 两段；
   - `docs/architecture.md`；
   - `docs/presentations/`（seed_refit、block_lm）；
   - `PROJECT_CONTEXT.md`；
   - V17L8 验证与 helix truth 相关的脚本和测试：`tracking_geometry_validation`、`validate_helix_truth`、`plot_helix_truth_comparison`；
   - `docs/handoff/2026-10-09-*.md`。

## 必须带走的问题

1. **外层预算 60 在「子样本 + 盒约束」下不够**（见未完成第 2 项），不能为了让 P9 通过而直接加大预算重跑。
2. **参考层传递**：参考几何误差对 17:8 的相干偏差（17:6 ty → 17:8 ty，传递系数约 1.5）在 Schur 下没有变化。
   如果真实探测器的参考层有 10 μm 量级的误差，两种求解器都会把它几乎原样传给 17:8。
3. **口径差异**：24:4 tx 的 fixed-track 误差 RMS，按 truth-veto 口径（`_local_delta`）为 7.75 μm，按 `compare_geometries` 为 6.98 μm。判定一律用前者。
4. `measure_schur_observability.py` 在 target 运行期间改过两次（新增 P10 阶段、修复 `transfer_ratio`），所以各阶段的源码快照不同。

## 产物（都在 `data/` 下，不入库）

| 目录 | 内容 |
|---|---|
| `.../p9-schur-observability-20261012/` | Level C、control、λ 扫描、σ_S、谱、30 个注入臂、84 个参考臂、P10 的 6 臂、`report/p9_result.json` |
| `.../level-e-schur-20261012/` | 两份 acceptance JSON、`level_e_verdict.json`、重建日志；重建 ROOT 在 `data/reconstruction/aligned/.../*_schur_lm.root` |
| `.../h1-truth-veto-20261012/` | `h1_truth_veto.json`（几何、径迹、联合目标、判定） |

（`...` = `data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4`）

## 新代码

| 文件 | 内容 |
|---|---|
| `scripts/joint_alignment_schur.py` | P11：`diagonal_only`、`diagonal_block_rank_deficient`、`solve_geometry_diagonal`、`global_stationarity_diagonal`、`run_diagonal_only_loop`（只新增，原有函数不变） |
| `scripts/analysis/measure_schur_observability.py` | P9/P10 的纯函数与 9 阶段 driver（复用 M1 的 Context 与缓存） |
| `scripts/run_candidate_acceptance.sh` | Level E：单个冻结候选的 ACTS 重建 + acceptance（先核对 SHA；不做求解、不 polish、不读 truth） |
| `scripts/analysis/run_h1_truth_veto.py` | Level F：Schur 对 fixed-track 的配对 truth-veto（只运行一次） |
| `tests/test_joint_schur_diagonal.py` / `test_measure_schur_observability.py` / `test_h1_truth_veto.py` | 5 / 17 / 11 项 |

测试（conda pytorch，Python 3.10）：`make test` 872 passed / 3 skipped / 0 failed（14 deselected）；`make test-shell` exit 0。

## 门状态

| 门 | 状态 |
|---|---|
| P0–P8、M1 | 通过（之前各轮） |
| WP3 / Q6 | 已定（Iteration 41） |
| Level C / D | 通过；候选已冻结 |
| P9 | 谱、D3、F1 通过；主判据未通过（not_evaluable，预算）；H1-R 1/5 下降，nominal 独立性通过 |
| P10 | 完成（描述性） |
| P11 | 函数与回归通过；真实数据运行未做 |
| Level E | 通过（Iteration 45） |
| Level F / H1 | H1-P 通过，H1-D 只有 17:8 ty 改善（单样本），没有被证伪（Iteration 46） |
| 独立样本复现 | **未开始**（未完成第 1 项） |
