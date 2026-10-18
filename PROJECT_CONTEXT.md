# ML_Alignment：外部 AI 助手项目上下文

> 核查日期：2026-09-20；Git HEAD：`2b8cd3e`。依据当前工作树源码、配置、测试和实验记录整理；工作树已有未提交文档改动，因此 HEAD 不能单独代表全部内容。本文未重新运行算法实验或测试套件，历史结果均注明来源。下列路径均相对仓库根目录 `/home/houjiahao/ML_Alignment`。

## 1. 项目目标与主要功能

本仓库是粒子探测器几何对齐算法的开发与验证主仓库：利用 ACTS/OpenDataDetector（ODD）产生的径迹测量，恢复传感器局部刚体错位，再回灌 ACTS 检验重建质量。ACTS 是外部仿真/重建运行时，不是本仓库核心求解器的依赖。

- 仿真、错位注入、重建与 nominal 几何导出。
- 解析 Jacobian、自动微分 surface 对齐，以及当前主线的非目标层 helix 重拟合 + 固定径迹 block-LM 几何对齐。
- single/broken helix 的测量留出验证、nominal/misaligned 配对恢复诊断、材料节点导出。
- 重建验收、几何真值比较、残差/chi2/命中数绘图，以及可复现实验记录。

这里的“ML”主要体现为 PyTorch 自动微分和数值优化；历史 hit 权重网络不是生产主线。当前默认目标层为 `17:8,24:4`，活动自由度为局部 `(tx,ty,rz)`，不是全探测器六自由度全局对齐。

## 2. 技术栈和依赖

| 层级 | 实际依赖与依据 |
|---|---|
| Python 核心 | Python ≥3.10；torch ≥2.1、numpy ≥1.24、scipy ≥1.11、pandas ≥2.0、awkward ≥2.6、uproot ≥5.3、matplotlib ≥3.8；Python <3.11 使用 tomli。见 `pyproject.toml` |
| 数值计算 | PyTorch autograd、批量 Jacobian、Gauss–Newton/Levenberg–Marquardt、线性代数；部分路径可用 CUDA，measurement GN 驻点检查使用 float64 |
| C++ 可选核心 | C++17、Eigen3、CMake；`AlignDerivLib/CMakeLists.txt` 提供 header-only 库、pybind11 的 `align_deriv_pb` 和另一套 nanobind 的 `align_deriv` 绑定 |
| 外部运行时 | ACTS、ODD、ROOT/PyROOT、LCG/CVMFS；ACTS 默认位于相邻 `../alignTest`。PyROOT 不由 pip 项目依赖安装 |
| 测试/编排 | pytest、pytest-timeout、Bash、Make；GitHub Actions 使用 Python 3.11 执行 `make ci` |

ML/绘图/测试使用 conda `pytorch`；ACTS 在新 shell 中 `source setup_acts.sh`。不要先 source LCG 再运行 conda Python：LCG 的 `PYTHONHOME` 会污染解释器。项目目前是脚本集合，`pyproject.toml` 的 `py-modules=[]`，没有统一安装式 CLI 服务。

## 3. 目录结构

以下覆盖全部主要目录；同类文件合并表示，省略 Git 内部文件、缓存及大批生成产物。

```text
ML_Alignment/
├── README.md / AGENTS.md / CLAUDE.md     项目介绍与助手工作约束
├── PROJECT_CONTEXT.md                  本文
├── pyproject.toml / Makefile / setup_acts.sh
├── .github/workflows/python-tests.yml   Python + shell CI
├── .agents/ / .codex/ / .claude/         本地助手配置
├── configs/
│   ├── default.toml                    版本化默认值
│   └── local.toml.example              机器配置模板；local.toml 可选且被忽略
├── scripts/                            求解器、ACTS 适配器、编排、验证、绘图
│   └── analysis/                       seed/refit、选择效应、truth-veto 等专项分析
├── AlignDerivLib/
│   ├── CMakeLists.txt
│   ├── include/AlignDeriv/              Surface、PlaneSurface、Jacobian 等头文件
│   ├── bindings/                       pybind11 / nanobind 绑定
│   └── tests/                          C++ 与 Python 绑定测试
├── tests/                              Python 回归、shell dry-run、辅助检查脚本
├── docs/
│   ├── architecture.md / io-paths.md    架构与 I/O 契约
│   ├── ml_alignment_principles.md       早期解析 surface 算法说明
│   ├── alignment_math_theory.md 等      数学、技术说明与环境指南
│   ├── history/                        实验历史与产物清理记录
│   ├── handoff/                        专项交接、阻塞记录与路线图
│   ├── research/material_aware_tracks/ 材料模型研究、来源与决策
│   ├── superpowers/{specs,plans}/       设计和实施计划，不能视为全部已完成
│   └── presentations/                 推导文稿、演示材料与生成脚本
├── surface_alignment_logic.pptx        现有演示文件
├── data/                               本地数据，主体被 Git 忽略
│   ├── geometry/ / simulation/ / alignment/ / reconstruction/
│   ├── experiments/                    iterative_ml、surface_helix、helix_validation、
│   │                                   helix_recovery、material、ablation 等运行命名空间
│   └── archive/                        历史产物
└── build/                              本地 C++/绑定构建产物，被 Git 忽略
```

## 4. 核心模块及职责

| 真实路径 | 职责 |
|---|---|
| `scripts/alignment_odd.py`、`scripts/generate_odd_transform.py` | ACTS 仿真/对齐/重建/材料导出及 nominal surface 几何导出；只有这两个适配器允许导入 `acts` |
| `scripts/ml_alignment_autograd.py` | 生产入口；ROOT 加载、surface 自动微分、非目标 helix 初始化/重拟合、逐 sensor block-LM、几何输出 |
| `scripts/helix_model.py` | ACTS-independent 六参数 helix、最近点/平面求交、gauge 固定、点拟合与独立 measurement GN |
| `scripts/ml_alignment.py` | 解析 Jacobian surface 基线：调用 AlignDerivLib，周期性重算 Jacobian，Adam 优化 |
| `scripts/ml_alignment_helix.py` | 独立 frozen-helix holdout 对齐及 assess/align CLI；不是当前主编排的核心 solver |
| `scripts/alignment_experiments.py` | 历史 Adam refit、joint Adam、学习 hit 权重；由生产模块延迟访问以支持历史复现 |
| `scripts/broken_helix_model.py`、`scripts/material_artifact.py` | 实验性分段 helix/散射先验，以及 ACTS 材料 ROOT 到版本化 NPZ 的转换 |
| `scripts/validate_helix_model.py`、`scripts/compare_helix_recovery.py` | 留出/配对 Gate 4；nominal 与 misaligned 的严格同身份 seed→refit 恢复诊断 |
| `scripts/reconstruction_acceptance.py`、`scripts/calibrate_acceptance.py` | 重建 measurement/SMT chi2 指标、冻结 bootstrap 阈值、profile 身份核验和候选排序 |
| `scripts/alignment_diagnostics.py`、`scripts/compare_geometries.py`、`scripts/plot_*.py` | 残差、几何真值及重建表现诊断；真值不参与生产候选调参 |
| `scripts/project_config.py`、`scripts/alignment_artifact_layout.py`、`scripts/path_inventory.py` | 类型化配置、兼容路径 API、绝对路径清单 |
| `scripts/experiment_manifest.py`、`scripts/write_experiment_summary.py`、`scripts/event_identity.py` | 输入哈希、源码/环境身份、manifest 生命周期、实验汇总、事件身份 |
| `scripts/analysis/run_truth_veto_closure.py`、`scripts/analysis/truth_veto_gate.py` | 冻结最终候选后的单次几何 truth-veto 闭环 |
| `scripts/migrate_local_data.py`、`scripts/tune_solver.py` | 数据迁移工具、求解器试验工具；不是生产入口 |

## 5. 主程序入口

仓库没有单一 `main.py`。主要入口如下（均从仓库根目录调用）：

```bash
# ACTS shell：仿真/重建等子命令；full 走适配器自身的完整流程
python scripts/alignment_odd.py --help
# 子命令：full、simulation、alignment、reconstruction、material

# ML shell：当前端到端生产编排；先检查实际命令，不启动实验
bash scripts/run_surface_helix_experiment.sh --run-tag example --dry-run

# ML shell：单独运行主求解器的参数入口
python scripts/ml_alignment_autograd.py --help
# --track-mode surface（Python 默认）或 free；free 默认 --joint-solver block-lm

# 历史迭代重建驱动
bash scripts/run_ml_iterative.sh --dry-run --iters 1

# 独立研究/验证入口
python scripts/validate_helix_model.py --help
python scripts/compare_helix_recovery.py --help
```

生产 shell 自动隔离 ACTS 与 conda 阶段，要求唯一 `--run-tag`，目前明确限定 `17:8,24:4` 和 DOF `0,1,5`；通用配置支持其他层，不代表这个实验配方也支持。

## 6. 核心算法的数据流和调用关系

```text
ACTS simulation / reconstruction
  └─ trackstates ROOT + 可选 tracksummary ROOT + geometry JSON
       └─ ml_alignment_autograd.run_alignment_pipeline(track_mode="free")
          └─ run_free_alignment_pipeline
             ├─ load_joint_geometry：nominal 全几何上覆盖输入目标层错位面
             ├─ load_variable_information：ROOT → 清洗后的逐 hit AlignmentInputs
             ├─ build_track_batches：按 event/track 分组 → padding + component mask
             ├─ seed_helices → helix_model.fit_helix_points_gauss_newton
             │    只用非目标层 SMT 全局点做几何预拟合
             ├─ fit_helices_to_measurements → fit_helix_measurements_gauss_newton
             │    只用非目标层局部 measurement，以固定参考面做 GN refit
             ├─ run_block_lm_alignment：固定 refitted helices，求目标面修正
             └─ export_joint_geometry → update_geometry_to_json
                  └─ aligned geometry JSON → ACTS reconstruction → acceptance
```

关键数据契约：

- 几何 JSON 是 surface 列表，包含 `ID`、`Translation`、`RotationMatrix`，可含 `FullMatrix`；以 `(volume,layer,module)` 映射 sensor。平移单位 mm，旋转单位 rad。
- ROOT 中 measurement-error 数组与全部 track-state 数组不一定等长，必须映射到实际 measured states；轨迹以 `(event_nr,track_nr)` 识别。free 路径当前 `load_summary=False`。
- `TrackBatches` 的测量/方差为 `[tracks,max_hits,2]`，另有 hit、LOC1、target 和 fit mask。至少 8 个非目标 hit、非目标折线跨度至少 300 mm，且包含目标 hit，才进入默认主线。
- 一维 strip 的 LOC1 不参与目标；short-strip 的二维 LOC1 方差下限为 `1.2² mm²`。不能用填充值当成真实测量。
- Helix 参数为 `(x0,y0,z0,psi0,tan_lambda,rho)`，路径变量是横向弧长，`rho` 为带符号曲率；规范固定后优化 5 个独立径迹参数。点 seed 最小化欧氏距离，measurement refit 最小化半 chi2，二者目标函数不同。
- 几何按局部修正组合：`center = center0 + R0·delta_t`，`R = R0·R(delta_rot)`。不能混用全局平移/旋转约定。
- block-LM 每个目标 sensor 独立构造标准化残差 Jacobian、`JᵀJ` 和 `Jᵀr`，有独立 damping、重试、步长上限和参数边界。接受步必须在同一有效 component 集上降低 chi2，且不得丢失原有效 component；收敛由 Newton decrement/KKT 判定，不是“没有接受步”。

端到端编排先冻结 nominal acceptance profile 并评价 misaligned baseline，再运行 free/block-LM、ACTS 回灌和候选比较，必要时最多尝试两次 gain=0.2 的 surface polish；已移除前置 surface warm start。当前使用完整事件集，没有 train/calibration/validation 分区；最终 acceptance 也只能标记 `diagnostic_only`、`verdict_eligible=false`，不是独立泛化证明。

## 7. 关键类和函数

| 定义文件 | 优先理解的对象 |
|---|---|
| `scripts/ml_alignment_autograd.py` | `AlignmentRunConfig`：运行参数；`AlignmentInputs`：逐 hit 输入；`TrackBatches`：按径迹 padding 后输入；`JointGeometry`：全几何及目标映射；`AlignmentEngine`：surface 参数模型 |
| 同上 | `run_alignment_pipeline()` 分派 surface/free；`run_block_lm_alignment()` 是当前几何主求解器；`_sensor_residual_data()` 做 sensor hit 选择、求交和局部残差；`_block_lm_linearized_block()` 形成正规方程；`run_joint_alignment()` 是历史 Adam 分派 |
| `scripts/helix_model.py` | `MeasurementFitInputs`、`helix_point()`、`intersect_plane()`、`closest_helix_points()`、`damped_gauss_newton()`、两种 `fit_helix_*_gauss_newton()`、`measurement_stationarity_report()` |
| `scripts/ml_alignment.py` | `compute_jacobians_from_alignderivlib()`、`physics_informed_loss()`、`run_alignment_pipeline()`；这里也有同名配置/输入类，勿与 autograd 版本混淆 |
| `scripts/validate_helix_model.py` | `HelixValidationConfig`、`run_helix_validation()`、`run_paired_holdout()`、`predictive_covariance()`、`evaluate_broken_helix_gate()` |
| `scripts/reconstruction_acceptance.py` | `build_acceptance_baseline()`、`compute_reconstruction_acceptance()`、`acceptance_score()`、`polish_improves()` |
| `scripts/project_config.py`、`scripts/experiment_manifest.py` | `ProjectConfig`、`load_project_config()`；`write_experiment_manifest()`、`finalize_experiment_manifest()`、`restore_source_snapshot()` |

## 8. 配置与产物系统

实际加载逻辑见 `scripts/project_config.py`：默认 TOML → 可选 `configs/local.toml` → 一个选定的 override 文件。显式 `--config PATH`/函数参数优先于 `ML_ALIGNMENT_CONFIG`，两者不是两个依次叠加的文件。所有相对配置/CLI 路径按仓库根目录解析，支持环境变量与 `~` 展开。

`configs/default.toml` 定义 `project.data_root`、`project.align_derivlib_root`、`acts.root`、`runtime.conda_setup`、`alignment.target_layers`、`alignment.active_dof`。目标目录和后缀由层列表派生，如 `Vol17_Lay8__Vol24_Lay4`、`v17l8_v24l4`，不得在其他模块另行拼一套规则。

算法预算并未全部集中到 TOML：还分布在 dataclass、模块常量、CLI 和 shell 配方。特别是生产 shell 的 SMT seed 上限为 **2000**，Python 配置默认仍为 **200**；`--joint-iters` 控制历史 Adam，不控制默认 block-LM（应看 `--lm-max-iters`）；measurement refit 的 Adam 迭代/学习率参数不控制 GN 路径。

```bash
python scripts/path_inventory.py --json  # 查看本机最终配置及绝对 I/O 路径
```

每个算法实验（含失败/中断）必须写版本化 `manifest.json`，记录有效配置、命令、输入 SHA256、Git 身份、已有指标、输出和状态。当前 manifest schema 为 2，支持 dirty 工作树身份和可选源码快照；不能只拿 commit 当作精确复现依据。大 ROOT/JSON/图像放 `data/`；迁移原始数据必须先复制、核验、切换配置，删除源文件需另行明确批准。

## 9. 测试结构和命令

`tests/conftest.py` 将 `scripts/` 加入导入路径。测试大致分为：

- 数学/求解：`tests/test_helix_model.py`、`tests/test_joint_alignment.py`、`tests/test_ml_alignment*.py`、`tests/test_alignment_derivative_consistency.py`。
- 数据与配置：`tests/test_loader_measurement_errors.py`、`tests/test_component_mask_wiring.py`、`tests/test_project_config.py`、`tests/test_path_inventory.py`、`tests/test_material_artifact.py`。
- 验证/复现：`tests/test_validate_helix_model.py`、`tests/test_compare_helix_recovery.py`、`tests/test_reconstruction_acceptance.py`、`tests/test_experiment_manifest.py`、truth-veto 与绘图测试。
- Bash 契约：`tests/test_run_ml_iterative.sh`、`tests/test_run_surface_helix_experiment.sh`、`tests/test_plot_residuals_static.sh`；C++/绑定另在 `AlignDerivLib/tests/`。

在干净的 conda `pytorch` shell 中执行：

```bash
make test           # Python；默认排除 root_io
make test-shell     # shell 路径与 dry-run 契约
make ci             # 上述两组；GitHub CI 同样只覆盖这两组
make test-root-io   # 显式 ROOT I/O 集成，120 秒 timeout
make test-cpp       # Eigen/C++17 header-only 检查
make test-bindings  # pybind11 编译及绑定测试
```

pytest 声明 `acts`、`root_io`、`slow` 标记，但默认表达式只排除 `root_io`。本次只生成上下文文档，未将历史测试通过数当作当前测试结果。

## 10. 当前已知问题与证据边界

1. **参考层假设与遗漏的相关性**：主线固定非目标 nominal 几何，并固定 refitted tracks 后解目标面；未形成完整 track/geometry 联立正规方程，也未传播径迹不确定度造成的跨 sensor 相关性。结果不能直接外推为全探测器全局对齐。
2. **材料模型未获准进入主线**：`docs/history/ml_optimization_log.md` Iteration 22 的 broken-helix Gate 4 仍失败，插值 RMSE 改善仅 0.18%–3.18%，不足预注册 20%；`24:4|backward_extrapolation|LOC0` 的 NLL 阻尼敏感性仍未关闭，该组 NLL 结论不得引用。
3. **同样本验收限制**：当前全事件 acceptance 是内部守卫；nominal profile 的事件身份仅覆盖 ROOT 中实际存在的事件，不代表全部生成事件。几何真值改善也不等于独立验证通过。
4. **历史失败不能当作新方向**：局部 surface 残差会被 Kalman 拟合吸收，早期独立对齐/反复回灌曾发散；Huber/IRLS/残差驱动权重曾删除错位信号。重试必须说明改变了哪项假设。
5. **工程使用风险**：算法默认值分散且入口间预算不同；大型脚本和历史兼容 API 并存。部分旧文档仍引用不存在的 ACTS 内部旧路径，旧路线图还描述 event split，必须以现代码和较新记录为准。
6. **环境限制**：不兼容的 uproot/fsspec 环境可能使 ROOT I/O 卡住；通用 CI 不证明 ACTS、C++ 绑定和真实 ROOT 链路全部通过。

最新记录的正面基线：优化日志 Iteration 30（2026-09-10）记载 schema-3.1 闭环完成，保留 61,482 条径迹、2,268 个 block 均 interior converged、四个 acceptance 分量及六项一次性几何 truth-veto 通过；此前缺失 `profile_type/verdict_eligible` 的字段问题已修复。此为历史记录，不是本次重新验证；仍属于 `diagnostic_only` / `post_hoc_truth_veto_only`。

## 11. 性能瓶颈

以下前两项是源码可见的潜在热点，并非本次 profiler 测量：

- `scripts/ml_alignment_autograd.py::_sensor_residual_data()` 每次按 sensor 扫描整份 padded hit mask；外层 sensor/迭代/重试会重复选择、求交和数据转换。小块求解还包含多次 Python 循环与 CPU/GPU 同步。
- ROOT jagged 数据展开为 DataFrame，再转换 tensor、按最大 hit 数 padding，增加峰值内存；自动微分 Jacobian 与反复 helix 求交有计算成本。`scripts/helix_model.py` 已按默认 2048 行分块，不能把“增加分块”描述为尚未存在的功能。
- ACTS 回灌、材料导出、配对 bootstrap、输入文件哈希增加端到端耗时。历史日志 Iteration 22 的全量 Gate 4 约 28 分钟，Iteration 26 的 trial-010 全流程约 2 小时 43 分钟；输入、硬件和版本不同，不宜直接作为当前性能承诺。

## 12. 后续可能的优化方向

以 `docs/handoff/2026-09-07-alignment-methodology-roadmap.md` 为研究路线参考，并用较新的 Iteration 27–30 更新完成状态：block-LM 收敛契约和 acceptance 元数据已有后续实现，不应重复列为待修复。

1. **优先做等价性能优化**：预计算 sensor→`(track,hit)` 索引，缓存 device/dtype 转换，批处理独立 sensor 块或分块累加正规矩阵；在固定输入、行集、权重和 component mask 下比较时间/内存与几何差异。
2. **验证可观测性边界**：正负错位注入、非目标参考层误差注入、Fisher 谱与正则敏感性分析，区分数据约束、弱模与先验主导解。
3. **关闭协方差稳定性问题**：在冻结 Gate 4 行集、参数与判据下，以 QR/SVD 和参数重缩放对照现有 Cholesky；不靠放宽阈值或删除困难行宣告通过。
4. **有前置条件的全局求解**：确认数值、真值否决和可观测性契约后，再研究保留 track/geometry 交叉项的 Schur 消元，先做小型稠密 oracle 和独立模拟闭环。
5. **材料模型与维护性**：只有模型证据支持且 Gate 4 通过后才推进 broken-helix/GBL 几何联立；大模块拆包、重复 helper 整理是独立项目，需保留现有 CLI 和 ACTS-independent 边界。

外部助手修改算法前必须依次阅读 `docs/architecture.md`、`docs/ml_alignment_principles.md`、`docs/history/ml_optimization_log.md`；先读变量定义和上下文，再推断含义；算法改动先写失败回归测试，实验无论成败都完成 manifest。不要将设计计划、历史代码或训练 loss 下降误报为已经验证的物理改进。
