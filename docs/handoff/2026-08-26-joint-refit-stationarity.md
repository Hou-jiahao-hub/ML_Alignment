# 交接：joint 路径 Adam refit 的驻点性测量（P2 第一步）

**会话日期**：2026-08-26（接续同日的 damping 等价性修复）
**上一份交接**：`docs/handoff/2026-08-26-damping-equivalence-v2.md`
**详细记录**：`docs/history/ml_optimization_log.md` 的 Iteration 18
**分支**：`feat/material-aware-broken-helix`（**未推送、未 merge**）
**工作树状态**：本轮改动**尚未 commit**

---

## 0. 一句话状态

P2 的第一步（只测量、不改行为）**已完成**：joint 校准初始化里的 Adam 测量 refit
被证明**远未到驻点**——200 次迭代后 Newton decrement p90 仍是 4.17e5，
阈值 1e-6，非驻点行占比 100%。求解器未换，下游产物逐位不变。

---

## 1. 上一份交接的环境记录是错的 ⚠️

§7 记的「`python` = `/usr/bin/python` 3.9.25，`make test` 293 passed / 35 failed」
**是解释器用错了**。`pyproject.toml` 要求 `>=3.10`；3.9 没有
`zip(..., strict=True)`，`fit_helices_to_measurements` 在它下面跑不完第一次迭代。

**正确的运行方式**：

```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pytorch/lib   # 否则 awkward_cpp 缺 CXXABI_1.3.15
~/miniconda3/envs/pytorch/bin/python -m pytest tests -q     # 3.10.20 / torch 2.11.0+cu126
```

| 解释器 | 全量测试 |
|---|---|
| `/usr/bin/python` 3.9.25 | 293 passed / 35 failed / 3 skipped |
| conda `pytorch` 3.10.20（本轮前） | **328 passed / 0 failed** / 3 skipped |
| conda `pytorch` 3.10.20（本轮后） | **333 passed / 0 failed** / 3 skipped |

那 35 项「既有失败」是环境不匹配，不是代码欠债。**下轮不要再用 `make test` 的默认
`python`**，它会把一个干净的仓库报成 35 个失败。

---

## 2. 本轮的测量结果

真实数据，`--track-mode free --max-tracks 300`：

| 量 | seed | 200 次 Adam 迭代后 |
|---|---|---|
| non-stationary 行占比 | 100.00% | **100.00%** |
| Newton decrement p90 | 5.36e5 | **4.17e5** |

阈值 1e-6，也就是终点距驻点还有 **11 个数量级**；200 次迭代只降了约 22%。
与 Iteration 15 在 Gate 4 路径上的发现同向，量级更极端。

---

## 3. 改了什么

| # | 改动 | 文件 |
|---|---|---|
| 1 | `_validation_batches_from_joint`：两模块 `TrackBatches` 的字段重标注 | `scripts/ml_alignment_autograd.py` |
| 2 | `measure_joint_refit_stationarity`：走 validation 的 `newton_decrement` | 同上 |
| 3 | `fit_helices_to_measurements` 的 stats 加 `stationarity`（seed + refit 各一次）| 同上 |
| 4 | `--no-joint-refit-stationarity`（默认开启）| 同上 |
| 5 | 新增 5 项测试 | `tests/test_ml_alignment_autograd.py` |

**行为不变已验证**：同一命令开/关 flag，`measurement chi2/component = 7992.76` 逐位相同，
导出的对齐几何 JSON **文件级完全一致**。

---

## 4. 两个先解决掉的坑（下轮不要重踩）

### 4.1 gauge 必须是 `F = 0.5·chi2`

`_measurement_refit_objective` 的 loss 除以了 `component_mask.sum()`（真实运行约 1e5）。
Newton decrement 随目标线性缩放，若在归一化 loss 上测，每个值会缩小同样倍数，
而 `NEWTON_DECREMENT_STATIONARY = 1e-6` 是在 `0.5·chi2` 下预注册的。
**所以测量整个走 validation 模块，只把参数传过去。**

### 4.2 两个目标函数确认是同一个（逐位）

validation 用冻结弧长 `hit_s`，autograd 每次重解 `intersect_plane`，看起来不同，
实测 `0.5·chi2` 相对差 **0.000e+00**。原因：`intersect_plane` 两边都用默认
`n_newton = 12`；`EPS_VARIANCE == EPS_COVARIANCE == 1e-8`；零节点时
`broken_helix_track_parameters` 退化为 reference 广播，`hit_s` 只用于选段。

这条等价性是 P2 下一步（换成 GN）能成立的前提，已写成回归测试。

---

## 5. 未完成任务

### P2' — 把 refit 换成 Gauss-Newton（下一步）

`fit_helices_gauss_newton` 的 docstring 自称「`fit_helices_to_measurements` 减去优化器」，
本轮已实测确认目标函数逐位等价，且它的解被判为**收敛**
（`non_stationary_row_fraction == 0.0`，`decrement_max < 1e-6`）。
但换它会改变 `refitted_parameters`，**属于新一次预注册变更**，需要先定判据再动。

### P1' — `24:4|backward_extrapolation|LOC0`（未变）

唯一仍对 damping 真实敏感的组，NLL 结论**不得引用**。

### P3 — `min_extrapolation_fit_hits = 4` 过松（未变）

single 臂 4 hits 撑 5 个 gauge-fixed 参数。改它属于另一次预注册变更。

### P4 — 其他（未变）

- 本轮测量只在 300 条轨迹上做过，全量 joint 运行未复现。
- `seed_helices` 的 Adam 预拟合仍未检验是否欠收敛。
- 结论仍只在 20000 events 上。
- 工作包 ④ joint 校准接入：Gate 5 规定 Gate 4 未通过不得启动，**Gate 4 已确认 failed**。

### P5 — 分支处置（未变）

`feat/material-aware-broken-helix` 未推送、未 merge，**且本轮改动尚未 commit**。
需用户明确批准。
