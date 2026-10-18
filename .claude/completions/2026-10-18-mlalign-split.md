# 2026-10-18 mlalign 模块拆分与历史实验代码删除

分支 `refactor/split-alignment-modules`（基于 `97cd01c`）。

## 提交
1. `b14afc1` 删除历史 Adam / hit-weight 路径：`alignment_experiments.py`、`--joint-solver adam`、
   `--joint-refit-solver`、仅服务这些路径的 CLI 参数和 config 字段、`validate_helix_model --fit-solver`。
2. 拆分提交：`ml_alignment_autograd.py`、`helix_model.py` 的实现原样搬进 `scripts/mlalign/`，
   两个旧文件改为 facade（所有旧名字都 re-export，CLI 不变）；Schur 只改 import，消除了
   autograd ↔ Schur 的循环依赖。

## 验证（conda pytorch 3.10）
- 全量测试：基线 922 passed → 删除后 902（删掉了 20 个只测历史路径的测试）→ 拆分后 902，0 failed；`make test-shell` 通过。
- 真实数据切片（1500 条 entry，300 tracks）跑 surface / block-lm / schur-lm：几何 JSON、deltas、
  tracks、history、summary、stdout 与基线逐位一致（sha256）。
- 旧名字 facade 全部存在，且与定义模块是同一对象；`--help` 输出一字不差；每个模块都能在新解释器里单独 import；
  `import ml_alignment_autograd` 不会加载 scipy/acts/validate_helix_model。

## 事故
一次中间测试里补丁打在 facade 上没生效，把测试桩几何写到了
`data/alignment/Vol17_Lay8__Vol24_Lay4/aligned_transforms_v17l8_v24l4_ml.json`，覆盖了
2026-08-03 的原文件（2,986,012 B，sha256 5cdc7856…）。没有副本，无法恢复；该文件没有被任何
manifest 当作输入引用。测试已改成写 tmp_path。桩文件仍留在原处，等用户决定是否删除。

## 未做
- Schur 内部拆分（按用户要求）。
- `run_surface_alignment` 开头保留了 `ACTIVE_DOF = active_dof` 这类别名，以便函数体原样搬迁；
  `--min-hits` 是死参数，与历史路径无关，未动。
