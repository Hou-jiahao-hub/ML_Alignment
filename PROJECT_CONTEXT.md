# ML_Alignment：外部 AI 助手项目上下文

> 核查日期：2026-10-19；分支 `refactor/schur-lm-package`。2026-10-19 的整理只保留了 Schur-LM 主线，并把它打包成 `src/mlalign`。被删除的代码（surface/block-LM 求解器、AlignDerivLib、Gate 4 留出验证、材料与 pT=100 专项、各 gate 分析脚本）可以从标签 `pre-schur-package`（`62f062e`）复现。路径均相对仓库根目录。

## 1. 项目目标

利用 ACTS/OpenDataDetector（ODD）产生的径迹测量恢复目标层传感器的局部刚体错位，再交回 ACTS 重建以检验效果。主线算法是径迹与几何的联立 Schur 消元 Levenberg–Marquardt（`schur-lm`）。默认目标层为 `17:8,24:4`，活动自由度为局部 `(tx,ty,rz)`。这不是全探测器六自由度的全局对齐。

## 2. 两层结构

| 层 | 位置 | 依赖 | 职责 |
|---|---|---|---|
| 算法包 | `src/mlalign/` | torch、numpy、scipy、pandas、awkward、uproot；不依赖 ACTS、TOML 和仓库路径 | 输入是几何 JSON 和 ACTS trackstates，输出是对齐后的几何 JSON |
| 工作流 | `scripts/` | TOML 配置、`data/` 目录布局、ACTS（仅适配器使用） | 仿真、误差注入、ACTS-Alignment 对照、重建、验收、出图、编排、manifest |

安装方式：在 conda `pytorch` 环境里执行 `python -m pip install -e '.[test]'`。入口有三种：`mlalign-align`、`python -m mlalign`、`mlalign.run_alignment(AlignmentConfig(...))`。

## 3. 算法包模块

| 模块 | 职责 |
|---|---|
| `config.py` | 所有默认常量（`DEFAULT_*`、`DTYPE=float64`）、`AlignmentConfig` 和 argparse 帮助函数 |
| `cli.py`、`__main__.py` | 命令行入口 |
| `pipeline.py` | `run_alignment()`：加载 → 选径迹 → seed → refit → Schur-LM → 真值诊断 → 导出 → 分层诊断 |
| `schur_lm.py` | 联立 LM：线性化、正规方程块、Schur 消元、稀疏 SuperLU 求解、阻尼、box 约束、全局 decrement 与秩检验、终止状态；`SCHUR_LM_*` 为预注册常量 |
| `geometry_io.py` | 在 nominal 全几何上覆盖输入的目标面，导出修正后的输入面 |
| `root_inputs.py` | 把 trackstates ROOT 读成逐 hit 张量（PyROOT 优先，uproot 兜底） |
| `track_batches.py` | 按径迹 padding 的批次和非目标拟合掩码 |
| `helix_init.py` | SMT 点 seed、非目标测量 Gauss–Newton refit 及其预注册驻点判据 |
| `joint_residuals.py` | 前向残差、分层 chi2、component 掩码合并 |
| `helix/` | 六参数 helix（`trajectory`、`gauge`）、逐行 LM（`gauss_newton`）、点拟合和测量拟合 |
| `box_constraints.py`、`conventions.py`、`rotations.py`、`diagnostics.py`、`reporting.py` | 修正量边界与活动集、DOF/层解析、欧拉角约定、残差与真值诊断、JSON 摘要 |

## 4. 数据流

```text
alignment_odd.py full (ACTS)  →  misalignment_applied_*.json + misaligned trackstates_kf_*.root
  └─ mlalign.run_alignment
     ├─ load_joint_geometry          nominal 全几何 + 输入目标面
     ├─ load_variable_information    ROOT → AlignmentInputs（不读 tracksummary）
     ├─ build_track_batches          ≥8 个非目标 hit，跨度 ≥300 mm，且至少含 1 个目标 hit
     ├─ seed_helices                 只用非目标 SMT 点，最多 2000 次迭代
     ├─ fit_helices_to_measurements  只用非目标测量的 GN refit，须通过驻点判据
     ├─ run_schur_lm_alignment       目标几何与全部径迹联立求解，外迭代上限 60
     └─ export_joint_geometry        → aligned JSON → ACTS 重建 → acceptance
```

几何约定：`center = center0 + R0·delta_t`，`R = R0·R(delta_rot)`。平移单位 mm，旋转单位 rad。一维 strip 的 LOC1 不参与拟合；short-strip 的 LOC1 方差下限为 `1.2² mm²`。

## 5. 工作流脚本

| 脚本 | 职责 |
|---|---|
| `alignment_odd.py`、`generate_odd_transform.py` | ACTS 适配器，只有这两个文件允许 `import acts`。子命令：`full`、`simulation`、`alignment`（ACTS-Alignment 对照）、`reconstruction` |
| `run_schur_experiment.sh` | 端到端编排：冻结 acceptance profile → 评估基线 → `mlalign-align` → ACTS 重建 → acceptance 守卫 → 出图 → manifest |
| `calibrate_acceptance.py`、`reconstruction_acceptance.py`、`event_identity.py` | 冻结 bootstrap 阈值、重建 chi2 验收、事件身份核验 |
| `compare_geometries.py`、`plot_residuals.py`、`plot_chi2ndf.py` | 几何真值对比（含 ACTS/Schur 对比模式）、残差与 chi2/NDF 图 |
| `project_config.py`、`alignment_artifact_layout.py` | TOML 配置、`data/` 路径布局、`--shell` 导出变量 |
| `experiment_manifest.py`、`write_experiment_summary.py` | manifest 生命周期、源码快照、实验汇总（schema 4，workflow `schur-lm_full-events`） |

运行产物写入 `data/experiments/schur_lm/<target>/<run-tag>/`。acceptance 用的是全事件同一样本，只能算 `diagnostic_only` 的内部守卫，不能作为独立验证。

## 6. 配置

- 算法默认值：`src/mlalign/config.py` 中的常量，取值等于经过验证的主线运行参数。需要特别注意 SMT seed 是 2000 次：旧 CLI 的默认值 200 会让 seed 停在非驻点。
- 工作流：`configs/default.toml`，加上可选的 `configs/local.toml`、`ML_ALIGNMENT_CONFIG` 或 `--config`；相对路径按仓库根目录解析。
- 两处的 `target_layers`/`active_dof` 必须一致，由 `tests/test_config.py` 检查。

## 7. 测试

```bash
make test          # pytest，默认排除 root_io
make test-shell    # run_schur_experiment.sh 的 dry-run 契约、plot_residuals 静态检查
make ci
make test-root-io
```

主要测试文件：
- `test_joint_schur_*.py`：Schur-LM 的预言机、收敛合同、稀疏求解、阻尼、约束。
- `test_helix_model.py`、`test_joint_alignment.py`：helix、seed、refit、批次、加载器。
- `test_refit_geometry_reporting.py`。
- `test_config.py`、`test_cli.py`：默认值以及包的自包含性。
- 工作流层的测试。

## 8. 证据边界与已知问题

1. 非目标层固定在 nominal 几何上，结果不能外推为全局对齐。
2. 在 300 条和 2000 条径迹的小切片上，Schur-LM 用满 60 次外迭代后以 `rank_deficient` 终止。这些切片只用于逐位等价检查，不代表主线结果；主线结论以优化日志里的全量运行为准。
3. ACTS 适配器的 `simulation`/`reconstruction` 子命令已于 2026-10-19 从 material study 接线中恢复，但当时 CVMFS 没有挂载，未能实测。
4. 历史失败路线（per-surface、IRLS、重新数字化等）见 `docs/history/ml_optimization_log.md`。重新尝试之前必须说明改变了哪条假设。

外部助手修改算法前必须阅读 `docs/architecture.md`、`docs/history/ml_optimization_log.md` 和 `src/mlalign/schur_lm.py` 的模块文档。算法改动要先写一个失败的回归测试；实验无论成败都必须写完 manifest。
