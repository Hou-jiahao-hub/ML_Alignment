# 2026-10-19 只保留 Schur-LM 主线，打包为 src/mlalign

分支 `refactor/schur-lm-package`，基于 `62f062e`（打了标签 `pre-schur-package`，被删除的代码都能从这里复现）。

## 提交
1. `efacf67`：删除 schur-lm 不用的独立脚本及其测试：
   - AlignDerivLib，及 `ml_alignment.py`、`tune_solver.py`、`AlignmentDerivBridge.hpp`
   - frozen-helix、Gate 4、helix truth、tracking-geometry、broken-helix/material、helix-recovery
   - `plot_hit_number`、`migrate_local_data`、`path_inventory`
   - `run_ml_iterative.sh`、`run_candidate_acceptance.sh`
   - `scripts/analysis/` 全部
2. `06b64cb`：把 `scripts/mlalign` 移到 `src/mlalign`：
   - `joint_alignment_schur.py` 改名为 `mlalign.schur_lm`，`alignment_diagnostics.py` 改名为 `mlalign.diagnostics`，`free_helix_pipeline` 改名为 `mlalign.pipeline.run_alignment`。
   - 删除 surface、block-LM、`run_config`，以及两个门面模块。
   - 新增 `config.py`（`DEFAULT_*` 常量、dataclass、`add_*_arguments`，风格仿照 ACTS `kfalignment_config`）、`cli.py`（`mlalign-align`）和 `__main__`。
   - 去掉 GPU：不再有 `--gpu` 和 device 参数，统一 CPU float64。
   - 改为 pyproject 的 src 布局；本地已在 conda pytorch 环境执行 `pip install -e`（`--no-deps --no-build-isolation`）。
3. `8e31453`：裁剪 `alignment_odd.py`，删掉 material study 和 pt100 study。从 `b734326^` 恢复了 `simulation` 和 `reconstruction` 的 handler，此前它们被硬接到 material study 上。ACTS-Alignment 保留，作为对照基线。
4. `482826e`：`run_surface_helix_experiment.sh` 改写为 `run_schur_experiment.sh`：
   - 流程：preflight → profile → baseline → `mlalign-align` → 重建 → acceptance 守卫 → 出图 → manifest。
   - 产物目录改为 `data/experiments/schur_lm`。
   - 实验摘要升到 schema 4，workflow 为 `schur-lm_full-events`。
5. 文档提交：更新 README、AGENTS、CLAUDE、PROJECT_CONTEXT、architecture、io-paths、data/README。过时文档（`ml_alignment_principles`、`fix`、`ml_alignment_revise`、`helix-recovery`、`presentations/block_lm`）用 git mv 移到了 `docs/history/`。

## 默认值变化（与旧 CLI 相比）
- `--joint-smt-seed-iters` 改为 2000（旧 CLI 默认 200，编排器一直传 2000）。包的默认值等于编排器参数。
- `--lm-max-iters` 现在真正传给 Schur-LM 作为 `max_iters`。旧代码忽略它、固定用 `SCHUR_LM_MAX_ITERS`；默认值同为 60，数值不变。
- 删除对 schur 无效的参数：`--lambda-l2`、`--summary`、`--track-mode`、`--joint-solver`、`--joint-seed-mode`、`--alignment-gain`、`--epochs`、`--gpu`。
- 包 CLI 的相对路径改为相对 cwd，所有路径必填，不读 TOML。

## 验证（conda pytorch 3.10）
- 测试（另外 `make test-root-io` 10 passed）：
  - 基线是 902 passed / 3 skipped。
  - 提交 1 之后 441 passed。
  - 提交 2 之后 407 passed / 2 skipped；提交 3 删除 test_material_study（7 个）后，最终 `make ci` 为 400 passed / 2 skipped。
  - 新增 `test_config.py` 和 `test_cli.py`，共 16 个。
  - `make test-shell` 通过。
- 逐位等价（与 `62f062e` 比较；HEAD 用 orchestrator 参数加 `--joint-solver schur-lm`，新包只给路径、target 和 DOF，其余全用默认值）：
  - 对象：`misalignment_applied_v17l8_v24l4.json` 和 misaligned `trackstates_kf_v17l8_v24l4.root`。
  - 比较内容：aligned JSON 的 sha256，summary（去掉 `output_geom`），predicted_deltas、track_params、残差、掩码，以及全部标量和 history。
  - stdout 在去掉路径行后完全一致。
  - 结果：t300 `f6699440…` 在提交 1、提交 2 和最终状态下各验证一次；t2000 `f70fbac8…` 在最终状态下验证，stdout 也一致。全部逐位一致。
- wheel 安装到仓库外的 target 目录后，能 import，`mlalign-align` 能打印用法。

## 未验证与遗留
- CVMFS 未挂载，`alignment_odd.py` 的改动没有在 ACTS 下跑过。需要在 ACTS shell 里检查：`python scripts/alignment_odd.py reconstruction -h`、10 事件的 simulation/reconstruction、`pytest tests/test_alignment_odd_local_rotation.py`。
- `run_schur_experiment.sh` 只验证了 dry-run，没有端到端运行。
- `src/mlalign/schur_lm.py` 没有拆分（1600 行）。
- 后续可以做：给 summary 加一个有效配置块（会改变 summary 字节）、`--timings`（参考已删除的 `run_pt100_schur_calibration.StageTimer`）。
- `tests/test_experiment_manifest.py` 里还有 `material_artifact.py` 等字符串，它们只是 stage-name 测试用的样例命令，没有改。
