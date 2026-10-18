# 交接：Track–Geometry 单 DOF 验证（V17L8 tx / ty / rz）

**日期**：2026-10-09　**分支**：`feat/full-event-calibration`（未提交）

**归档**：Stage I — Frozen-track / target-holdout geometry-response validation。
已验证 geometry arm（δg → Δr_g）、seed 响应、refit 与 seed 无关、G_g/G_t 的 FD 检查；
**未**验证显著 G_t δθ 下的 joint response（closure 因 Δr_t ≈ 0 而退化），见日志 Iteration 32「归档」。

## 门状态

| 门 | V17L8 tx | V17L8 ty | V17L8 rz |
|---|---|---|---|
| P0 baseline + 复现 | pass | pass | pass |
| P1 geometry-only | pass | pass | pass |
| P2/P3 + closure | pass | pass | pass |
| G_g / G_t | pass | pass | pass |
| P5 recovery | pass ×2 | pass ×2 | pass ×2 |

结果目录：`data/experiments/helix_truth/Vol17_Lay8__Vol24_Lay4/tracking_geometry_validation/`；
数字见 `docs/history/ml_optimization_log.md` Iteration 32。

## 生效中的预注册常量

- target `["17:8"]`、active DOF `[0,1,5]`；cohort 前 20000 个事例，ACTS `--num-events 20223`。
- 幅度 tx/ty ±0.5 mm、rz ±0.02 rad，整层同幅度、确定值。
- P5 求解器参数 = `run_surface_helix_experiment.sh` joint_free 阶段（seed 2000 次迭代）。
  **求解器自带默认值（seed 200 次）会被其驻点守卫拒绝**，不要直接用默认 CLI。
- τ_abs 来自 a=0 对照：tx 26.5 μm、ty 31.6 μm、rz 1.39 mrad。

## 刻意未做

- V24L4（B1–B3）：需新配置 `target_layers = ["24:4"]`、单独 P0；驱动脚本的 `LAYER`、
  `CONFIG_TEXT`、默认根目录目前写死为 17:8，扩展时要参数化，而不是复制脚本。
- 单层三 DOF 同时注入、两层综合、逐 module 随机注入（M4/M5）。
- 在修正后几何上重跑 ACTS 的 post-correction 验证。
- 多档幅度扫描（目前每 DOF 一档）。

## 注意

- 结构性结论：refit 只用 non-target hit，目标函数与 P0 相同，refit 层 Δr_t ≈ 0；
  ACTS 响应只在 seed 层可见。fixed-track block-LM 的恢复误差因此不随注入幅度变化。
- `tests/test_helix_model.py::test_both_consumers_reexport_the_shared_math_implementation`
  在与 `test_validate_helix_truth.py`、`test_joint_alignment.py` 同跑时失败，单独跑通过；
  与本次改动无关（不含新测试时同样复现）。
