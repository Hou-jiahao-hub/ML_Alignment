# ACTS + ML 对准完整工作流

本目录（`job/`）包含完整的探测器对准流水线：ACTS 几何对准 → 残差绘图 → ML 校准 → ML 验证绘图 →（可选）多轮迭代。ACTS 只在数据提取和最终验证阶段使用；ML 校准脚本本身不依赖 ACTS。

---

## 环境说明

| 任务 | 环境 | 激活方式 |
|------|------|----------|
| ACTS 仿真 / 对准 / 重建 | LCG_109 | `source ~/alignTest/setup_acts.sh` |
| ML 训练 + 残差绘图 + 测试 | conda pytorch（含 ROOT 6.40） | `source ~/miniconda3/etc/profile.d/conda.sh && conda activate pytorch` |

> **关键约束**：严禁同时 source LCG 再调用 conda Python。LCG 设置的 `PYTHONHOME` 会使 conda Python 崩溃（`ModuleNotFoundError: No module named 'encodings'`）。两套环境必须分开运行（`run_ml_iterative.sh` 内部用子 shell 隔离两套环境，见该脚本的 `run_conda`/`run_acts`）。

---

## 目录结构总览

| 路径 | 作用 |
|---|---|
| `alignment_artifact_layout.py` | **命名/路径唯一真源**（见下节），被本目录几乎所有脚本导入 |
| `alignment_odd.py` | ACTS 仿真 / misalignment / 重建 CLI（`full` / `simulation` / `alignment` / `reconstruction` 子命令） |
| `ml_alignment.py` | ML 求解器 A：调用 `AlignDerivLib`（`align_deriv_pb`）解析 Jacobian，逐 sensor 优化 6-DOF |
| `ml_alignment_autograd.py` | ML 求解器 B：纯 PyTorch autograd 把径迹重新投影到几何面求残差，不依赖 `AlignDerivLib`/ACTS |
| `run_ml_iterative.sh` | 多轮迭代驱动脚本：ML 求解 → ACTS 重建 → 残差绘图，循环 N 次 |
| `tune_solver.py` | 调参/诊断脚本：一次性加载几何+ROOT+Jacobian，反复 sweep 求解器配置，看 solved delta 与真值的相关性 |
| `plot_residuals.py` / `plot_chi2ndf.py` / `plot_hit_number.py` | 残差 / chi2 / hit 数对比绘图 |
| `generate_odd_transform.py` | 生成/导出 `odd_transforms.json`（nominal 几何） |
| `docs/ml_alignment_principles.md` | `ml_alignment.py` 当前实现的完整数学推导（Jacobian、loss、hit-weight 网络、几何更新公式） |
| `archive/docs/ml_optimization_log.md` | 历史优化日志，逐轮记录每次改动、训练/重建指标 —— **改动这两个 ML 求解器前必读**，避免重踩已知坑 |
| `archive/docs/alignment_math_theory.md` | 对齐数学理论背景 |
| `tests/` | pytest + shell 测试（见「测试」一节） |
| `odd_transforms.json` | nominal（无错位）全几何，18824 个面 |
| `alignment_output/`, `reconstruction_output/`, `simulation_output/` | 各阶段产物（按 `TARGET_LAYER_DIR` 分子目录，见下） |

---

## 命名与路径唯一真源：`alignment_artifact_layout.py`

**这是本目录最重要的架构约定**：目标层配置、文件名后缀、输出子目录名只在 `alignment_artifact_layout.py` 里定义一次，其余所有脚本（`alignment_odd.py`、`ml_alignment.py`、`ml_alignment_autograd.py`、`plot_*.py`、`run_ml_iterative.sh`）都从这里导入，不再各处手改文件名。

```python
TARGET_LAYERS: tuple[tuple[int, int], ...] = ((17, 8), (24, 4))
TARGET_LAYER_SUFFIX = "v17l8_v24l4"          # 文件名后缀
TARGET_LAYER_DIR = "Vol17_Lay8__Vol24_Lay4"  # 输出子目录名
```

**切换目标层时只需修改这三个常量**（不再需要在 `alignment_odd.py` 里改 `target_volume` 字典或全文件搜索替换后缀）。`target_volume_map()`、`artifact_filename()`、`alignment_json_path()`、`reconstruction_root_path()`、`target_subdir()` 等辅助函数会据此自动生成一致的文件名和目录结构。

产物统一落在 `<base_dir>/<TARGET_LAYER_DIR>/...` 下，例如：
- `alignment_output/Vol17_Lay8__Vol24_Lay4/misalignment_applied_v17l8_v24l4.json`
- `reconstruction_output/misaligned/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root`

shell 脚本可通过 `eval "$(python alignment_artifact_layout.py --shell)"` 拿到 `TARGET_LAYER_DIR` / `TARGET_LAYER_SUFFIX` / `TARGET_LAYERS_SPEC` 等变量（`run_ml_iterative.sh` 就是这么做的）。

---

## 阶段 1：ACTS 完整工作流

```bash
cd ~/alignTest/Examples/Scripts/Python/job
source ~/alignTest/setup_acts.sh

python alignment_odd.py full --num-events 100000 > alignLog_v17l8_v24l4 2>&1
```

包含 Simulation → Alignment(misalignment 注入) → Reconstruction（nominal / misaligned / aligned）三大步骤，产物见「目录结构总览」中的 `*_output/` 说明（均按 `TARGET_LAYER_DIR` 分子目录）。

---

## 阶段 2：绘制 ACTS 对准残差对比图

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pytorch

OUT_DIR=reconstruction_output/residuals/Vol17_Lay8__Vol24_Lay4

python plot_residuals.py --range -1.0 1.0 --volume_id 17 --layer_id 8 \
  --recon-dir reconstruction_output --output-dir $OUT_DIR
python plot_residuals.py --residuals X --range -1.0 1.0 --volume_id 24 --layer_id 4 \
  --recon-dir reconstruction_output --output-dir $OUT_DIR
python plot_residuals.py --residuals Y --range -6.0 6.0 --volume_id 24 --layer_id 4 \
  --recon-dir reconstruction_output --output-dir $OUT_DIR
```

> volume 23/24/25（ShortStrip）Y 方向本征分辨率约 1.2 mm，range 用 ±6.0；X 方向用 ±1.0。

---

## 阶段 3：ML 校准 —— 两种求解器

两个脚本解决同一个问题（逐 sensor 优化 6-DOF `[δt_x, δt_y, δt_z, δr_x, δr_y, δr_z]`，局部坐标系，`local0=u, local1=v`），但用不同方式算残差对齐参数的导数：

| | `ml_alignment.py` | `ml_alignment_autograd.py` |
|---|---|---|
| Jacobian 来源 | `AlignDerivLib`（`align_deriv_pb`）解析 Jacobian，PyTorch 不对 Jacobian 求梯度 | 纯 PyTorch autograd：把径迹（全局位置+方向）重新投影到平移/旋转后的几何面上，端到端可微 |
| 外部依赖 | 需要 `AlignDerivLib/build_pybind11` 在 `sys.path` 上 | 无 ACTS/AlignDerivLib 依赖，只用 torch/numpy |
| 残差模型 | `r_i^new = r_i^(0) - J_i·δα_i`（一阶线性化，`--jacobian-update-interval` 周期性用当前修正重新线性化） | 每步都对当前 `δα` 做真实的面几何变换+径迹求交，非线性 |
| Hit 权重 | 内置可学习 hit-quality 小 MLP（9→32→16→1），按 `(volume,layer)` 归一化 | 无 |
| CLI | 见下 | 同款参数，缺 `--debug-dataflow`；`--jacobian-update-interval` 被接受但忽略 |

两者共享的关键 CLI 参数：

```bash
python ml_alignment.py \
  --input-geom alignment_output/Vol17_Lay8__Vol24_Lay4/misalignment_applied_v17l8_v24l4.json \
  --states reconstruction_output/misaligned/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root \
  --summary reconstruction_output/misaligned/Vol17_Lay8__Vol24_Lay4/tracksummary_kf_v17l8_v24l4.root \
  --output-geom alignment_output/Vol17_Lay8__Vol24_Lay4/aligned_transforms_ml_v17l8_v24l4.json \
  --target-layers 17:8,24:4 --active-dof 0,1,3,4,5 --epochs 300 --lambda-l2 1e-5
```

- `--active-dof`：**真实错位是 tx,ty 平移 + rx,ry 倾斜**（约 0.5mm + 0.015rad），倾斜是残差主导，**必须包含 rx,ry（索引 3,4）**，否则残差降不下来。
- `--nominal`：传入 `odd_transforms.json` 可在训练结束后打印 ML 修正 vs 真值的 corr/recovery 诊断（`validate_against_truth()`）。
- 详细数学推导（Jacobian 公式、chi2 loss、hit-weight 网络、L2 正则、几何更新公式）见 `docs/ml_alignment_principles.md`（该文档描述的是 `ml_alignment.py` 的实现）。

**改动这两个脚本前，先读 `archive/docs/ml_optimization_log.md`**：里面记录了 chi2 clamp 导致像素层零梯度、方差公式漏减平滑轨迹贡献、L2 强度、Huber/IRLS 鲁棒损失在此场景下反而变差等已踩过的坑，避免重复调试。

---

## 阶段 4：ML 对齐几何重建验证

```bash
source ~/alignTest/setup_acts.sh

python alignment_odd.py reconstruction --mode aligned \
  --aligned-file Vol17_Lay8__Vol24_Lay4/aligned_transforms_ml_v17l8_v24l4.json \
  --output-suffix ml --num-events 100000 > recon_aligned_ml_log.txt 2>&1
```

`--aligned-file` 是相对 `alignment_output/` 的路径（注意别把文件名误当参数名传，见「常见问题」）。

---

## 阶段 5：绘制 ML 残差对比图

同阶段 2，加 `--aligned-trackstates <ml重建的trackstates.root>` 和 `--label`，对 v17l8 用 ±1.0mm、v24l4 X 用 ±1.0mm、v24l4 Y 用 ±6.0mm。

---

## 阶段 6：多轮迭代驱动 `run_ml_iterative.sh`

单轮 ML 校准往往不够；`run_ml_iterative.sh` 把「ML 求解 → ACTS 重建 → 残差绘图」打包循环 N 次，每轮的输出几何/重建结果作为下一轮输入。conda 和 LCG 环境在脚本内部用独立子 shell 隔离（见 `run_conda`/`run_acts`），无需手动切换。

```bash
cd ~/alignTest/Examples/Scripts/Python/job
./run_ml_iterative.sh --iters 3 --events 100000
```

常用参数：`--iters`、`--events`、`--start-iter`（续跑用，配合 `--input-geom/--states/--summary` 指定起点）、`--active-dof`、`--epochs`、`--lambda-l2`、`--dry-run`（只打印不执行）、`--no-plot`。

- **求解器切换是硬编码的**：脚本顶部 `ALIGNMENT_SOLVER`（`cpp` → `ml_alignment.py` / `autograd` → `ml_alignment_autograd.py`）和对应 `RUN_TAG` 需要直接编辑脚本，不是命令行参数（`--suffix`/`--target-layers` 也被脚本强制校验必须等于 `alignment_artifact_layout.py` 里的当前值，防止和目录布局脱节）。
- 每轮迭代日志在 `iterative_ml_logs/Vol17_Lay8__Vol24_Lay4/`，几何/重建产物按 `{RUN_TAG}_ml_{SUFFIX}_iter{N}` 命名。

### 方法论上的重要限制（务必了解再解读结果）

`ml_alignment.py`/`ml_alignment_autograd.py` 都是**逐 sensor 独立最小二乘**（per-surface）。`archive/docs/ml_optimization_log.md` 的 Iteration 6 记录了这类方法的一个原理性问题：Kalman Filter 会把探测器错位部分吸收进轨迹参数本身，导致"最小化单面残差"和"逼近真值几何"不是同一个目标，逐轮迭代时预测修正与真值的符号关系会翻转，固定增益/多轮独立求解无法收敛到真值。

真正解决了这个问题的方法是 Iteration 7-8 中实现的 **联立轨迹+对齐最小化径迹 χ²（Millepede 思路）**，当时的实现脚本 `global_align.py` 已从本目录移除（只在 `archive/logs/global_align_full*.log` 留下运行记录），核心结果：`tx` corr 恢复到 0.99+，重建 `res_eLOC0` 核心从 misaligned 2.05/7.27mm 降到 0.031/0.043mm，逼近 ACTS 基线 0.007/0.016mm。若需要「无监督、不依赖 ACTS、可移植」的对齐方案，参考这个思路重新实现，而不是指望迭代 per-surface 求解器能收敛到同等精度。

---

## 调参/诊断：`tune_solver.py`

一次性加载指定 iteration 的几何 + ROOT + Jacobian（避免每次 sweep 都重新读 ROOT 文件），反复尝试不同求解器配置，比较 solved delta 与真值逐 sensor 的相关性。脚本内 `import ml_alignment as M` 依赖 cwd 在 `job/` 下（把 `ml_alignment.py` 当模块导入复用其数据加载/Jacobian 计算函数），运行前确认 `INPUT_GEOM`/`STATES`/`SUMMARY` 常量指向的 iteration 存在。

---

## 测试

conda pytorch 环境下运行（不要 source LCG）：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pytorch
cd ~/alignTest/Examples/Scripts/Python/job

pytest tests/                              # 全部 Python 单测（无 pytest.ini，靠 importlib 按文件路径加载模块）
pytest tests/test_ml_alignment.py -v       # 单个文件
bash tests/test_run_ml_iterative.sh        # 校验 run_ml_iterative.sh 的 --dry-run 输出（求解器/路径拼接是否正确）
bash tests/test_plot_residuals_static.sh   # 静态 grep 校验 plot_residuals.py 里的目标层硬编码
```

测试覆盖：`ml_alignment.py`/`ml_alignment_autograd.py` 的数据处理与几何更新公式一致性、`alignment_artifact_layout.py` 常量与命名规则、`plot_chi2ndf.py`/`plot_hit_number.py`、`alignment_artifact_layout` 与其余脚本路径拼接是否一致。改动任何路径/命名相关代码后，`tests/test_alignment_artifact_layout.py` 和 `tests/test_run_ml_iterative.sh` 是最先该跑的回归测试。

---

## 常见问题

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'encodings'` | LCG `PYTHONHOME` 污染 conda Python | 不 source setup_acts.sh，直接用 conda |
| `OSError: CXXABI_1.3.15 not found` | 系统 libstdc++ 太旧 | 用 conda pytorch 环境（已内置新版 libstdc++） |
| `ModuleNotFoundError: No module named 'acts'` | 未加载 LCG 环境 | 先 source setup_acts.sh |
| `ROOT is None` / `AttributeError: 'NoneType'` | conda 环境无 PyROOT | 安装：`conda install -c conda-forge "root=6.40.00=py310he617d7b_0"` |
| `--aligned-file` 参数不识别 | 将文件名误写为参数名 | 正确用法：`--aligned-file <filename>.json`（相对 `alignment_output/` 的路径） |
| 改了目标层但某个脚本还在用旧后缀 | 该脚本没有从 `alignment_artifact_layout.py` 导入常量 | 检查脚本顶部是否 `from alignment_artifact_layout import ...`，而不是本地硬编码后缀 |
