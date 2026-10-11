# 交接：Track–Geometry 联立 Schur 改造的 Phase 0（实现勘察与 fixed-track 基线冻结）

**交接日期**：2026-10-08
**分支**：`feat/full-event-calibration`
**HEAD**：`2b8cd3e`（工作树 dirty）
**上游**：`plan.md`、`docs/handoff/2026-09-22-schur-joint-alignment-theory-review.md`、
`docs/handoff/2026-09-07-alignment-methodology-roadmap.md` §M1/§H1、
`docs/handoff/2026-09-08-block-lm-convergence-contract.md`
**性质**：只读勘察 + 一次 solver-only 基线运行。**未修改任何求解器数学，未实现 Schur 消元，
未引入 track–geometry 耦合。**
**状态**：plan.md 的 P0.1（基线冻结）完成；P0.2（L1 真实数据两臂）与 P0.3（M1 可观测性基线）
**未做**。P1 的两条规格错误已定位，其中一条已关闭。

---

## 执行摘要

**结果**：完成了 plan.md 要求的 14 类求解器职责到文件/函数的逐项映射，确认其中
**ρ（LM 增益比）在当前仓库完全不存在**、**`B = ∂r/∂θ` 的装配路径按构造拒绝 target hit**、
**有效 component 守恒只覆盖 target 侧**——这三项是 Schur 改造的真实新增工作量，而不是
「替换阈值」级别的改动。另外在 HEAD `2b8cd3e` 上跑了一次只含 solver 的全量 fixed-track
基线（run tag `schur-p0-fixedtrack-20260922`，manifest 已 `completed`），证明
`ff972ba`（helix_model/autograd 大重构）与 `2b8cd3e`（拒绝欠约束测量行）**没有改变求解器
数值**：输出几何 2268 面的平移差 max 1.13e-07 mm、12 项 truth 指标相对差 ≤ 2e-8。

**遇到的问题**：三条，详见 §7。(1) plan.md §4 Q1/P1 要求在 `--no-event-split` 行集上跑判别
实验，而该开关连同整个 event split 机制已于 2026-09-10 移除，仓库里只剩断言其不存在的回归；
(2) 不设 `LD_LIBRARY_PATH` 时有 6 个测试假失败，极易被误判成回归；(3) 首次写 manifest 时
`--command-json` 漏掉了全部 I/O 路径参数，记录下来的命令无法复现，已重写。

**待完成**：P0.2、P0.3 两项前置工作包未做，均需新建脚本，**这是本轮交付的边界**：本轮的
授权范围是只读勘察与基线冻结，不含编写新的分析脚本。P1 自身还有两条阻塞项（§6）。

---

## 0. 宣称边界（先读这条）

本轮产出的 `schur-p0-fixedtrack-20260922` 是 **solver-only** 基线：

```
跑了:   ROOT 加载 -> 径迹筛选 -> SMT seed -> 非 target measurement GN refit -> block-LM 几何解 -> 几何导出
没跑:   ACTS 重建回灌、reconstruction acceptance、acceptance profile 标定、truth-veto
```

因此：

- 本次 manifest **不能支撑任何 acceptance 结论**。`nominal_states`、`simulation_dataset`、
  `odd_digi_config`、`odd_material_map` 四项输入不在清单里，因为 solver 不读它们。
- manifest 的 `metrics.scope` 与 `metrics.truth_metrics_claim_scope` 已把这一点写死，
  后者为 `diagnostic_only; must not inform solver tuning`。
- 「与 Iteration 30 数值等价」只能宣称**重构未改变求解器**，不得宣称物理精度有任何变化。
- truth 指标在本文中只作为复现性证据出现。**truth 在真实物理场景中不存在**，求解器内部
  任何一处引用 truth 都是错的。

---

## 1. 实现映射：plan.md 需求 → 文件:函数

### 1.1 已存在且可直接复用

| plan.md 要求 | 位置 |
|---|---|
| 径迹 refit（ref-only GN） | `ml_alignment_autograd.py:714 _refit_helices_gauss_newton` → `helix_model.py:633 fit_helix_measurements_gauss_newton` → `helix_model.py:425 damped_gauss_newton` |
| 种子拟合（SMT 点） | `ml_alignment_autograd.py:543 seed_helices` → `helix_model.py:849 fit_helix_points_gauss_newton` |
| 残差构造 — target | `ml_alignment_autograd.py:1265 _sensor_residual_data` |
| 残差构造 — 全 hit（评估用） | `ml_alignment_autograd.py:850 _joint_forward` |
| 残差构造 — ref（refit 目标） | `ml_alignment_autograd.py:596 _measurement_refit_objective`；whitened 版 `helix_model.py:304` |
| 径迹参数 θ（物理 6 参） | `helix_model.py:37 helix_point`，顺序 `(x0,y0,z0,psi0,tanλ,rho)` |
| 5 参数 gauge 固定表示 | `helix_model.py:196 reference_gauge_frame` / `:214 pack_gauge_fixed_track_parameters` / `:223 unpack_gauge_fixed_track_parameters` |
| 几何参数 / active DOF | `AlignmentRunConfig.active_dof:283`，解析 `:1794 parse_active_dof`，默认 `(0,1,5)` |
| 局部几何组合 | `:2197 apply_alignment_torch`（`c=c0+R0·δt`, `R=R0·R(δr)`）；导出 `:2259 update_geometry_to_json` |
| 几何 Jacobian `A = ∂r/∂a` | `:1059 _block_lm_linearized_block`（`torch.autograd.functional.jacobian`, vectorize） |
| 正规方程 `C=JᵀWJ`, `c=Jᵀr` | `:1086-1087`，逐 sensor 3×3 |
| LM damping | `:879 regularize_fisher_matrix`；逐 sensor 状态 `damping_by_sensor:1398`；重试阶梯 `:915 _block_lm_retry_damping` |
| `joint_clamp_t/r`（硬边界） | 常量 `:95-96`（1.5 mm / 0.05 rad）；`bound_limit:1389`；clamp `:1506-1507`；active-set `:930 _block_lm_free_dof_mask` |
| `lm_max_step_t/r`（解后裁剪） | 常量 `:147-148`（0.1 mm / 5e-3 rad）；`per_dof_limit:1384`；裁剪 `:1500-1503` |
| rank handling | eigen floor `:879`；谱分解剔弱模 `:950 _block_lm_decrement`；`lstsq` 粘滞回退 `:1493`；refit 行秩闸门 `helix_model.py:390` |
| 收敛 / 终止 | `BLOCK_LM_DECREMENT_STATIONARY=1e-6` `:179`；判定 `:1455`；四类终态 `:1097 _block_lm_block_status` |
| short-strip LOC1 方差下限 | `STRIP_Y_FLOOR=1.44` `:80`，施加于 `:2073` |
| `predictive_covariance()`（P1 需要） | `validate_helix_model.py:2290`，`PRODUCTION_DAMPING=0.0` |

（未标文件名的行号均指 `scripts/ml_alignment_autograd.py`。）

### 1.2 存在但不能直接用于 Schur

| 项 | 现状 | 为什么不能直接用 |
|---|---|---|
| `B = ∂r/∂θ` | `helix_model.py:373 measurement_residual_jacobian` | 它的 `component_mask` 来自 `_measurement_fit_inputs_from_joint`，该函数经 `:532 _non_target_fit_mask` **硬性拒绝含 target 的 mask**（`:536` 显式 raise）。plan.md §3.3 要求 `Γ_t = Γ_ref + Γ_tgt`，必须让同一路径接受 target hit——须新建入口，**不得放宽现有 fail-closed 断言** |
| 5 参 gauge 帧 | 同 1.1 | 仅在 `fit_helix_measurements_gauss_newton` 内部逐块构造并消费，未暴露为跨迭代可复用的 API。plan.md §3.10 的 gauge 生命周期需要新的调用层 |
| 有效 component 守恒 | `:1421 fixed_mask`、`:904 _block_lm_fixed_chi2`、`:1523` 丢失即拒 | 选择集是 `hit_mask & is_target & sensor_idx==s`，**ref hit 从未进入接受比较**。plan.md §3.9 要求 ref+target 全集守恒，是新增守卫 |

### 1.3 完全不存在

| 项 | 说明 |
|---|---|
| **ρ（增益比）** | 全仓搜索无 `rho` / `predicted_reduction` / `gain_ratio`。block-LM 与 `damped_gauss_newton` 都只用「目标是否下降」判接受，damping 按接受/拒绝 ×0.5 / ×10。plan.md §3.8/§3.8.2 的整套 ρ 阈值与自适应规则**没有任何现有常量可继承** |
| `--joint-solver schur-lm` | 当前 `choices=("block-lm","adam")`（`:2923`） |
| `scripts/joint_alignment_schur.py` | 不存在 |
| `tests/test_joint_schur_*.py`（10 个） | 全部不存在 |
| M1 观测性脚本 | `scripts/analysis/measure_alignment_observability.py` 不存在 |

---

## 2. 当前求解器数据流

```
misaligned geometry JSON + trackstates ROOT
  └─ load_joint_geometry():488        全 nominal 几何 + 覆盖目标面
  └─ load_variable_information():1896 ROOT→DataFrame；1D strip LOC1 置权 0；
                                       STRIP_Y_FLOOR 只对 strip 2D 生效
  └─ build_track_batches():369        按 (event,track) 分组 padding
                                       保留判据：有 target hit ∧ ≥8 非 target hit ∧ 折线跨度 ≥300 mm
  └─ seed_helices():543               仅非 target SMT 全局点，最小化欧氏距离（GN）
  └─ fit_helices_to_measurements():752
        仅非 target 局部测量，固定几何，5 参 gauge 帧下 GN 解 ½χ²
        预注册闸门：non-stationary 行占比 ≤1e-3（JOINT_REFIT_MAX_NON_STATIONARY_ROW_FRACTION），否则抛异常
  └─ run_block_lm_alignment():1316    ← 固定径迹，等价于施加 δθ ≡ 0
        for iteration in range(lm_max_iters=60):
          for sensor in active_sensors:              # 每个 target sensor 一个 3-DOF 独立块
            fixed_mask = 当前 component 集（冻结，只增不减）
            r, C, c    = _block_lm_linearized_block()    # C=JᵀJ, c=Jᵀr，经 physical_scales 无量纲化
            free_mask  = _block_lm_free_dof_mask()       # 箱约束 active set
            λ²         = _block_lm_decrement()           # 弱于 Fisher floor 的方向剔除，标 lower_bound
            if λ²/n ≤ 1e-6 or 自由集为空: 该 sensor 永久退休
            for attempt in 0..8:                         # 重试只改 damping，复用同一 Jacobian
              (C+ρI) δa = -c
              δa ← clip(δa, ±lm_max_step)                # 解后裁剪，不进入 KKT 判定
              a  ← clamp(a+δa, ±joint_clamp)
              接受 iff 无 component 丢失 ∧ invalid 未变差 ∧ χ²(fixed_mask) 严格下降
              接受: λ←λ/2, break；拒绝: λ←max(10λ, 1e-3·10^k)
  └─ export_joint_geometry():525 → aligned geometry JSON
```

关键性质：`_sensor_residual_data` 只按 sensor id 选 hit 且径迹冻结，因此各 block **精确解耦**
（回归 `tests/test_ml_alignment_autograd.py:1271`）。dtype：CPU 路径 float64，`--gpu` 下 float32。

---

## 3. 测试基线

在 conda `pytorch`（Python 3.10.20）中执行，**必须**设 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q
```

| 时点 | 工作树 | 结果 |
|---|---|---|
| 2026-09-22（基线冻结时） | HEAD `2b8cd3e` + 2 个未跟踪求解器文件 | `596 passed, 3 skipped, 12 deselected`（134 s） |
| 2026-10-08（本文撰写时） | 同 HEAD，另有 `validate_helix_truth` / `tracking_geometry_validation` / `plot_helix_truth_comparison` 三组新脚本与测试 | `624 passed, 3 skipped, 14 deselected`（204 s） |

两次均 0 failed。+28 passed / +2 deselected 来自基线冻结后新增的三组脚本及其测试，
与 Phase 0 无关；skip 的 3 项与原因两次完全一致。

`make test-shell`：3/3 通过（`test_run_ml_iterative.sh`、`test_run_surface_helix_experiment.sh`、
`test_plot_residuals_static.sh`），2026-09-22 实测。

3 个 skip 均为环境缺件、非回归：AlignDerivLib pybind 扩展未为 Python 3.10 构建（1 项）、
ACTS Python 绑定不可用（2 项）。

### 3.1 环境陷阱（本轮新发现，务必记住）

**不设 `LD_LIBRARY_PATH` 时会出现 6 个假失败**，全部是通过 subprocess 调
`python script.py --help` 的用例：

```
test_ml_alignment.py::test_reference_comparison_cli_is_not_exposed
test_ml_alignment_autograd.py::test_production_module_import_excludes_experiment_validation_and_acts
test_ml_alignment_autograd.py::test_reference_comparison_cli_is_not_exposed
test_ml_alignment_autograd.py::test_jacobian_update_interval_cli_is_not_exposed
test_project_config.py::test_pure_python_clis_expose_project_config_override[ml_alignment.py]
test_project_config.py::test_pure_python_clis_expose_project_config_override[ml_alignment_autograd.py]
```

根因：子进程里 `import torch` 先把系统 `/lib64/libstdc++.so.6` 载入进程，随后 `awkward_cpp`
找不到 `CXXABI_1.3.15`。pytest 主进程本身不受影响（它在 import torch 之前已 import awkward），
所以**只有这 6 个子进程用例失败**，看起来很像 CLI 契约回归。
`Makefile` 的 `make test` 不导出该变量，因此直接 `make test` 仍会看到这 6 个假失败。

---

## 4. 数值基线：`schur-p0-fixedtrack-20260922`

### 4.1 运行身份

| 项 | 值 |
|---|---|
| run tag | `schur-p0-fixedtrack-20260922` |
| 运行时间 | 2026-09-22 17:35 → 19:29，约 1 小时 54 分（不含 ACTS） |
| Git | `2b8cd3e`，dirty=true，源码快照 `written`（190 文件） |
| manifest | schema 2，`status=completed`，`completed_at=2026-09-22T11:30:34Z` |
| 输入 | `geometry` / `states` / `summary` / `nominal_geometry` / `acts_setup` / `conda_setup`，SHA256 与 Iteration 30 **逐字节一致** |
| 产物 | `data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4/schur-p0-fixedtrack-20260922/{joint_free_summary.json, manifest.json, *_joint_free.log}`、`data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_schur-p0-fixedtrack-20260922_joint_free.json` |

复现命令完整记录在 manifest 的 `command` 字段（含全部 I/O 路径）。

### 4.2 关键指标

| 项 | 值 |
|---|---|
| input / retained / excluded tracks | 98,958 / 61,482 / 37,476 |
| hits / fit hits | 852,859 / 709,672 |
| hit breakdown | 1,021,508 个 2D + 287,029 个 1D measurement |
| SMT seed loss | 0.002821816895555533 |
| measurement refit χ²/component | 31.86 → 0.7635026065294721（1,154,761 components） |
| refit non-stationary 行占比 | 100% → 0.00% |
| block-LM `best_iteration` / `stop_reason` | 13 / `all_blocks_terminal` |
| block 终态 | 2268/2268 interior `converged`；budget_exhausted / rank_deficient / intersection_failed 全 0 |
| decrement/component p99 | 7.774614198031347e-07（阈值 1e-6） |
| `initial_total_loss` → `last_total_loss` | 915.744 → 2.2126464974 |
| `correction_max_abs` | 1.3078521715504123（未撞 `JOINT_CLAMP_T=1.5`） |
| **`clipped_step_fraction`** | **0.825297269044819** |
| `lstsq_fallbacks` / `floor_limited_blocks` | 0 / 0 |

### 4.3 与 Iteration 30（commit `dfb6090`）的等价性

**结论：`ff972ba` + `2b8cd3e` 两次重构没有改变求解器数值。**

输出几何逐传感器对比（2268 面，ID 集合一致）：

```
平移      |diff| max = 1.134e-07 mm,  rms = 2.330e-09 mm
旋转矩阵元 |diff| max = 1.677e-09   ,  rms = 2.276e-11
逐位相同  = False
```

truth 指标 12 项（两层 × tx/ty/rz × recovery/err_rms）相对差全部 ≤ 2e-8，最大值出现在
`17:8 ty err_rms`（1.98e-08）。已知弱项原样复现：`24:4 ty` recovery 1.01759、corr 0.954、
err_rms 0.1659 mm。

`block_lm_summary` 的 `status_counts` / `converged_detail` / `component_guard` / `damping` /
`trust_region` 五个小节**逐字段相同**；`rank` 与 `decrement_per_component` 仅末几位浮点不同。

唯一相对偏移较大的是 refit Newton decrement p90（1.2e-13 vs 5.7e-14），仍比 1e-6 阈值低
七个数量级，属驻点邻域噪声。

差异量级与 Iteration 30 自己记录的「joint-only 复现 warm-start 几何到 2.4e-7 mm / 8.8e-9 rad」
同一档次，是 BLAS 归约顺序造成的浮点噪声。**比较两次运行按舍入量级判（~1e-7 mm / 1e-9 rad），
不要要求逐位相同。**

### 4.4 一条必须带走的观测

`clipped_step_fraction = 0.8253` 在 HEAD 上**逐位复现**：15,306 次评估中 12,632 次的步被
`lm_max_step_t/r` 解后裁剪。这不是旧版本的偶然，而是当前生产路径的稳定行为。

plan.md §3.8.1 要求 `schur-lm` 关闭该裁剪（理由正确：裁剪后的步不再是 LM 步，ρ 的分子分母
会自相矛盾）。但在 fixed-track 基线里，裁剪承担了 82.5% 的步长控制，而
`tests/test_joint_alignment.py:387` 明确把「cap × accepted_steps 必须跨越注入错位」写成了
可达性契约。**因此关闭裁剪是实质性的数值行为改变，必须配先红后绿的回归，不能当作清理历史包袱。**

---

## 5. plan.md 与当前实现的不一致

### 5.1 阻塞级

1. **`--no-event-split` 不存在。** plan.md §4 Q1 与 P1 要求在 `--no-event-split` 行集上跑判别
   实验。实测：`--no-event-split`、`--skip-blind-validation`、`--event-split-seed`、
   `--event-partition` 在 `scripts/` 下 grep 不到任何一处，`scripts/event_split.py` 已删除，
   `write_experiment_summary.py` 不再写 `event_split` 字段。`docs/architecture.md` 记为
   「removed 2026-09-10」。仓库里只剩三条断言其不存在的回归：
   `tests/test_run_surface_helix_experiment.sh:44-45`、`tests/test_compare_helix_recovery.py:458`、
   `tests/test_write_experiment_summary.py:93`。
   **处置**：P1 的行集定义改写为「当前全量事件口径 + 生产筛选（≥8 非 target hit、≥300 mm 跨度、
   17:8/24:4、DOF 0,1,5）」，并在预注册里写明 event split 已移除。

2. ~~HEAD 没有对应的全量基线~~ —— **本轮已关闭**，见 §4。

3. **`lm_max_step` 裁剪是主导机制**，见 §4.4。

### 5.2 重要级

4. **ρ 完全不存在**（§1.3）。plan.md §3.8.2 要求预注册的 9 个求解器常量
   （`schur_lm_lambda_init`、两个 λ 因子、三个 ρ 阈值、`lambda_min/max`、`max_lm_retries`）
   全部是新增量，没有现有常量可继承。这与 plan 自己的要求一致，但意味着 P6 的工作量大于
   「替换阈值」。
5. **component 守恒只覆盖 target**（§1.2）。
6. **5 参 gauge 帧未暴露为可复用 API**（§1.2）。
7. **`B` 的装配路径按构造拒绝 target hit**（§1.2）——这是与现有 fail-closed 断言的直接冲突。
8. **逐 block 契约与 Schur 互斥**。`tests/test_ml_alignment_autograd.py:1271` 断言 sensor 间
   逐位解耦，且它是「`converged` 的 sensor 可永久退休」的**唯一依据**。plan.md §6.1 已要求
   退役此契约，但该测试当前是绿的——`schur-lm` 必须与 `block-lm` 共存而不让它变红。

### 5.3 次要级

9. `--joint-smt-seed-iters`：生产 shell 传 2000，Python 默认仍为 200
   （`JOINT_SMT_SEED_ITERS`）。单独调用 solver 时必须显式给 2000，否则会重现 trial-009 的
   seed 驻点门控失败（4.6355%）。
10. plan.md §17 的 `make test` / `make test-shell` 在本机需额外 `LD_LIBRARY_PATH`（§3.1）。

---

## 6. 未完成任务

### 6.1 plan.md Phase 0 自身的剩余项

| 项 | plan 位置 | 状态 | 为何本轮未做 |
|---|---|---|---|
| L1 真实数据两臂 | P0.2 | **未做** | `2026-09-08-block-lm-convergence-contract.md` §11 已明确列为未做。需要冻结输入后专门执行，且预注册要求「先写预注册、再跑数据」 |
| M1 fixed-track 可观测性基线 | P0.3 | **未做** | 需新建 `scripts/analysis/measure_alignment_observability.py`（roadmap §M1 指定路径）。本轮授权范围是只读勘察 + 基线冻结，不含写新分析脚本 |

M1 的内容按 roadmap §M1：复用生产 residual/Jacobian，输出**未正则化、无量纲**的 Fisher
特征值与特征向量；注入目标层正/负扰动与非目标参考层协同扰动；测量阶段保持生产默认 active DOF
与 Fisher floor 不变。预注册主判据：声明为可观测的模式，在正负注入与独立噪声重复中恢复斜率的
95% CI 完全位于 `[0.9, 1.1]`。

**M1 必须先于 Phase P9 执行**（理论评审 §5.4）：`S` 的谱同时受「消元」与「未收敛」影响，
没有 fixed-track 对照谱，P9 测到的弱模无法归因，`R_i = tr(S_ii)/tr(C_ii)` 也没有基准。

### 6.2 Phase P1 的阻塞项

| # | 阻塞 | 性质 | 处置 |
|---|---|---|---|
| F1 | P1 行集引用不存在的 `--no-event-split` | 规格错误 | 见 §5.1 第 1 条 |
| F2 | `predictive_covariance()` 的现成调用者是 holdout 路径，口径为 4 hit / 100 mm 放宽门槛，**不是生产口径** | 工具口径不符 | P1 需新脚本在生产 `TrackBatches` 上装配 `Γ_ref`/`B`，复用 `predictive_covariance()` 但不复用其行选择 |
| F3 | ~~HEAD 无对应基线~~ | — | **已关闭**（§4） |
| F4 | P1 须按 **floor active / inactive** 分层（plan §3.2、§P1） | 缺数据通路 | `raw_covariances`（未加 floor 的方差）在 `AlignmentInputs:218` 保留了，但 **`TrackBatches` 没有携带它**；需透传才能分层 |
| F5 | L1/M1 前置未完成 | 前置 | 按 plan §4 Q7 须在 H1 生产评估前完成；**不阻塞 P1**，P1 是决策诊断，可先做 |
| F6 | `LD_LIBRARY_PATH` 未设时 6 个假失败 | 工程 | 最小修法：Makefile 导出 `LD_LIBRARY_PATH=$(CONDA_PREFIX)/lib` |

---

## 7. 本轮踩的坑

1. **manifest 的 `--command-json` 漏了 I/O 路径。** 首次写入时只记了算法参数，没记
   `--input-geom` / `--states` / `--summary` / `--output-geom` / `--output-summary` /
   `--nominal-full-geom`，记录下来的命令无法复现。已删除重写（当时目录内只有 manifest 与
   provenance，无实验产物，重写无副作用）。**写 manifest 时 `command` 必须是能原样粘贴执行的完整命令。**

2. **用日志 `tail -f` 监控 solver 是无效的。** solver 的 stdout 重定向到文件时是块缓冲，
   整个 3703 B 日志在进程结束时才一次性刷出——Iteration 30 那份 log 的 mtime 同样正是结束时刻。
   监控应基于进程存活与 RSS，不要基于日志增长。**「日志不动」不等于「卡住」。**

3. **`LD_LIBRARY_PATH` 那 6 个假失败**（§3.1）。第一次跑测试时报
   `6 failed, 590 passed`，差点被当成 CLI 契约回归。

---

## 8. 建议接续顺序

```
1. 关闭 F1：把 P1 的行集定义改写为全量事件口径，写进预注册
2. 关闭 F4：把 raw_covariances 透传进 TrackBatches（纯数据通路，不改数学）
3. 执行 P1：生产口径下逐目标层逐分量输出 tr(BΓ_ref⁻¹Bᵀ)/tr(V)，
   并把预测误差按「统计项 vs 残余」分解；写 P1 决策笔记
4. 据 P1 结果决定是否修订 H1 的「truth RMS 改善 ≥10%」主判据，
   并在 ml_optimization_log.md 写预注册条目（必须在看到新 H1 结果之前）
5. 补齐 P0.2（L1 真实数据两臂）与 P0.3（M1 可观测性基线）
6. 之后才进入 plan.md P2 及以后
```

**不要**在 F1、plan.md §4 的 Q2（阻尼加在哪里）、Q3（ρ 的定义）关闭之前进入 Phase P5 及之后
（理论评审 §10 的原话）。

---

## 9. 本轮未改动的内容

未修改任何求解器数学；未实现 Schur 消元；未引入 track–geometry 耦合；未做无关重构；
未修改 `scripts/ml_alignment_autograd.py`、`scripts/helix_model.py` 或任何测试。
唯一的磁盘写入是 `schur-p0-fixedtrack-20260922` 运行目录、其输出几何 JSON，以及本文档。

工作树中的 `validate_helix_truth.py`、`tracking_geometry_validation.py`、
`plot_helix_truth_comparison.py` 及其测试、`docs/architecture.md` 的改动**不属于本轮工作**，
它们在基线冻结之后出现，与 Phase 0 无关。
