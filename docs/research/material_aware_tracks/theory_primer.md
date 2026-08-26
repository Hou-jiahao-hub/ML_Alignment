# 理论速读：从单 helix 到材料感知 broken-helix

本文把本目录的文献笔记（`notes/`）、方法比较与决策记录串成一条完整的推理链，
面向即将实现 broken-helix 工作包的读者。目标是**在动手之前建立正确心智模型**：
每个技术选择的物理动机是什么、哪些是不可协商的约束、哪些是首版刻意留下的边界。

- 结论与接口见 [decision_record.md](decision_record.md)
- 选型对比见 [method_comparison.md](method_comparison.md)
- 一手文献摘录见 [notes/](notes/)

标注为「项目观测」的内容来自本仓库实验，不是文献结论。

---

## 一句话总览

> 当前模型假设「粒子在磁场里走一条完美的螺旋线」，但真实粒子每穿过一层探测器材料
> 就被原子核**随机踢一脚**。本方案要做的是：**承认这些踢脚的存在，把它们变成模型里
> 有物理代价的自由参数**，并且用一整套统计门控证明这样做真的更好，而不是靠多加参数
> 把残差「拟合掉」。

---

## 1. 病症诊断：bias≈0 而 RMSE 大 1–2 个量级

Iteration 10 的 nominal holdout（项目观测，见 `docs/history/ml_optimization_log.md`）：

| 层 | 模式 | LOC0 bias / RMSE [mm] |
|---|---|---|
| 17:8 | interpolation | -0.004 / 0.083 |
| 17:8 | backward | -0.001 / 0.579 |
| 24:4 | backward | +0.025 / 0.903 |

**怎么读这两个数**：

- `bias`（偏差）= 预测误差的**平均值**。接近零 → 模型没有系统性地偏向某一边。
- `RMSE` = 预测误差的**散布宽度**。比 bias 大 1–2 个数量级 → 每条轨迹都偏，
  但偏的方向随机、正负抵消。

这是一个**教科书式的随机效应指纹**：如果是几何没对准（系统误差），bias 会跟着变大；
如果是数值求交失败，会有失败计数（实测 0%）。**只有「逐轨迹随机扩散」能同时解释
「均值对、宽度错」**。而探测器里唯一的随机扩散源就是——多重散射。

**为什么 backward 外推最差（0.579 vs 0.083）**：内插（interpolation）时目标点被两侧
的 hit 夹住，随机踢脚被「包围」住了；外推时目标点之外的散射是**未来发生的、拟合数据
里完全没有信息的随机量**，误差随外推距离累积。这条观察后面会决定验收标准的形状
（见 §9）。

---

## 2. 多重库仑散射与 Highland 公式

主要来源：[notes/particle_material.md](notes/particle_material.md)

### 物理图像

带电粒子穿过物质时，被无数原子核的库仑场**小角度**偏折。单次偏折微不足道，但成千
上万次累积起来，出射方向相对入射方向有一个随机偏角。中心极限定理保证这个偏角的
**核心部分近似高斯分布，均值为零**。

### 材料预算 x/X0

衡量「这块材料有多能散射」的量不是厚度（mm），而是**辐射长度归一化厚度** `x/X0`：

- `X0`（辐射长度）是材料常数——铅的 X0 很短（约 5.6 mm），硅较长（约 94 mm）。
- `x/X0` 无量纲，可以理解为「以散射能力为尺子量出来的厚度」。
- **必须沿实际入射路径算**：斜着穿过一块板，走的路程比板厚长，`x/X0` 更大。
  ACTS 已经把 `pathCorrection` 乘进 `mat_step_length`，因此
  `x/X0 = mat_step_length / mat_X0`，**不能再乘一次**——重复修正是这类工程里
  最典型的静默 bug。

### Highland 公式

```
θ₀ ≈ (13.6 MeV / (β·p·c)) · |z| · √(x/X0) · [1 + 0.038·ln(x/X0)]
```

通俗读法：

- **θ₀ 是散射角的标准差**，不是某次具体的散射角。
- `√(x/X0)`：材料翻 4 倍，散射宽度只翻 2 倍。这是随机游走的典型标度
  （方差可加 → 标准差按 √ 增长）。
- `1/(βpc)`：**动量越高，越难被推歪**。动量翻倍，散射宽度减半。
  这也是样本选 `pT ≥ 2 GeV` muon 的原因——高动量 + β≈1，公式最干净。
- `ln` 项是经验修正项，让公式在很薄和很厚的材料上都保持精度。

实现见 `scripts/helix_model.py::highland_kink_sigma`，与
ACTS `Core/src/Material/Interactions.cpp` 同约定（含对数内的 `q²/β²` 因子）。

### 一个必须记住的坑：散射方差在此公式下不可加

Highland 的 `ln(x/X0)` 里那个 `x/X0` 是**整块 slab 的**，不是逐段的。
后果：**把一块材料拆成两半，两半的 θ₀² 之和 ≠ 整块的 θ₀²**。

因此「把节点拆分/重新聚合后总 scattering variance 守恒」这个看似天经地义的
closure 断言，**在物理上本来就不成立**；把它写进测试等于固化一条错误的物理假设。
正确的两条可证伪检查是：

1. **解析因子对拍**：把一个节点等分成 `k` 份，断言数值 `Σθ0²` 等于由 Highland
   闭式导出的**解析比值**（相对容差 1e-12）。这检验实现，不检验守恒。
2. **聚合影响上界**：报告原始节点列表与聚合后列表的逐轨迹 `Σθ0²` 相对变化，
   要求 ≤ `AGGREGATION_THETA2_TOL`（建议 2%）。超限说明聚合实质改变了物理，
   必须重新设计聚合规则，而不是放宽阈值。

> 这是「理论素养直接改变工程决策」最清晰的一个例子。

### 为什么 Huber / IRLS 被否决

鲁棒损失的假设是「离群点是坏数据，应该降权」。但这里的大残差**是真实物理**——
它是散射尾巴，不是坏 hit。降权等于把物理信息扔掉，同时还会误伤真正的几何错位信号
（而那才是对齐要找的东西）。Iteration 6/8 已实证判死（项目观测）。

---

## 3. 六参数 helix：现有模型是什么

`scripts/helix_model.py` 的参数序 `(x0, y0, z0, psi0, tan_lambda, rho)`：

| 参数 | 含义 | 直觉 |
|---|---|---|
| `x0,y0,z0` | 轨迹上的参考点 | 螺旋线的「起点」 |
| `psi0` | 该点的横向方位角 | 起点朝哪儿飞 |
| `tan_lambda` | 纵向倾角的正切 | 螺旋「爬升」多陡；`= cot θ` |
| `rho` | 带符号曲率 (1/mm) | 弯多狠、往哪弯（正负 = 电荷符号） |

路径参数 `s` 是**横向弧长（mm）**——即投影到 xy 平面的弧长。选横向弧长而非三维弧长，
是因为均匀磁场中横向运动是标准圆周运动，公式最简洁：`z = z0 + tanλ·s`，
三维路程 = `s·√(1+tan²λ)`。

**这个模型的根本局限**：`rho` 恒定 → 它只能表达「平均磁场弯曲」，
**一条 helix 没有任何自由度可以表达「在第 3 层被踢了一脚」**。

---

## 4. Broken Lines / GBL：核心思想

主要来源：[notes/general_broken_lines.md](notes/general_broken_lines.md)

### 思想

不要为整条轨迹找一条完美曲线，而是：

```
一条参考轨迹（reference helix，6 参数）
  + 在每个材料节点插入一个「折角」kink（2 个参数）
```

轨迹变成一条**折线式的螺旋**：位置处处连续（粒子不会瞬移），但**方向在材料节点处
允许突变**。

### 关键：为什么这不是「多加参数作弊」

如果 kink 完全自由，那就是被否决的「经验分段」方案——参数一多，轨迹能扭成任何形状，
把测量残差全部吸收掉，**包括真正的几何错位信号**。

> GBL 的 "broken" 不是让每段完全自由：若没有 scatterer covariance，
> 局部方向变化会失去物理代价，轨迹可吸收任意 measurement 或几何偏移。

**约束来自 Highland 给出的先验**：每个 kink 的大小有一个物理确定的期望尺度 θ₀。
kink 想变大，就要付出统计代价。

### 目标函数

```
F(q, k) = ½ [ Σᵢ rᵢᵀ Vᵢ⁻¹ rᵢ  +  Σⱼ kⱼᵀ Qⱼ⁻¹ kⱼ ]
                 ↑ 测量项              ↑ 散射惩罚项
```

- `q` = 参考轨迹 6 参数，`k` = 所有 kink，`V` = 测量误差协方差，
  `Q_j = diag(θ0², θ0²)` = 该节点的散射先验协方差。
- **贝叶斯视角**：这是一个 MAP（最大后验）估计。第一项是似然（数据说了什么），
  第二项是先验（物理说 kink 应该多大）。整个模型是**用物理先验正则化的拟合**，
  而不是无约束的自由拟合。
- 材料越薄（θ₀ 越小）→ `Q⁻¹` 越大 → 该处的 kink 被压得越死。
  在真空节点 `x/X0 = 0` 处 θ₀ = 0，`Q⁻¹ = ∞`，kink 被强制为零——退化回单 helix。
  **这就是「zero-material 极限必须复现现有 helix」这条验收标准的数学来源。**

### 为什么不能让散射尺度可训练

假设给 θ₀ 加一个可训练放大因子 `α`：优化器会发现，只要把 `α` 调大，`Q⁻¹` 就变小，
散射惩罚项趋近于零，于是 kink 可以自由拟合任何东西，χ² 轻松降到很低。

**这是一个纯粹的数值逃逸方向，不是物理改善。**

原则上的修法是在目标里加 `log det Q` 项（完整的高斯负对数似然会自动惩罚方差膨胀）。
但即便如此，材料尺度仍与几何参数、测量误差高度相关，难以分辨。
**所以首版直接固定 PDG/ACTS 的物理尺度，一个可训练放大因子都不给。**
可证伪条件里因此列了一条：改善若依赖可训练 covariance 膨胀 → 不得宣称成功。

---

## 5. 分段递推的数学细节

`broken_helix_segments()`（`scripts/helix_model.py`）沿节点逐个推进：

```
段 0 = reference helix
节点 j 处：
  起点   p_j   = helix_point(seg_{j-1}, s_j - s_{j-1})    ← 位置连续
  到达方向 psi = psi_{j-1} + rho·(s_j - s_{j-1})
  折角   psi_j = psi + k_j[0]/sin(θ)                      ← 方向突变
         θ_j   = θ + k_j[1]
         tanλ_j = cot(θ_j)
  rho 段间不变                                            ← 首版边界：不建模能损
  padding 节点恒等传递
```

### 为什么 kink 的第一个分量要除以 sin θ

kink 的两个分量必须**张成与入射方向正交的平面**，这样对角先验 `Q = diag(θ0², θ0²)`
才是真正**各向同性**的（散射不挑方向）。

但 `psi`（方位角）不是正交坐标：粒子越接近沿 z 轴飞（θ 接近 0 或 π），同样大小的
空间偏折对应的 `Δψ` 越大——就像地球上越靠近极点，同样距离跨越的经度越多。
除以 `sin θ` 正是这个「经度换算」因子，与 ACTS
`PointwiseMaterialInteraction.cpp` 的 `sigmaPhi = theta0/sinθ`、`sigmaTheta = theta0`
约定一致。

### 为什么段分配和 node_s 要 detach

`segment_of_hit = (node_s < hit_s).sum()` 是一个**整数、阶跃函数**。
如果让它参与自动微分，当某个 kink 稍微一动、把一个 hit 推过节点边界时，段索引会
整数跳变 → 损失函数出现不连续断崖 → 梯度爆炸或优化器震荡。

detach 的物理含义就是 **broken lines 的「线性化点」**：在参考轨迹附近做展开，
展开点在一轮优化里保持固定。进一步的要求是：若参考轨迹偏离线性化点超过
`RELINEARIZE_TRIGGER_MM`，**必须走显式 relinearization checkpoint 并记录 hash，
禁止隐式漂移**——隐式漂移会让结果不可复现，也让梯度失去意义。

---

## 6. 坐标契约：本轮最高风险

这是整套方案里最「反直觉」、也最容易实现错的一点。

**问题**：helix 的弧长 `s` 是从 `(x0,y0,z0)` 起算的。而 `(x0,y0,z0)` 是**拟合出来的
参数**，不是物理不变量。

validation 的设计是：同一条物理轨迹会出现在**多行 batch** 里（每行留出不同的
holdout hit、不同的 fit window）。每一行会拟合出**各自的**参考 helix，于是同一个材料
节点在不同行里的 `s` 值**根本不一样**。

**直接后果**：如果拿导出端（ACTS 传播时算的）`path_length_mm` 去和行内的 `hit_s`
比较、做区间裁剪，就会整体偏移或把节点分到错误的段——**静默产生错误结果，
而且很难发现**。

**契约**：

1. artifact 里的 `path_length_mm` **只用于导出顺序和完整性诊断**，绝不参与 validation
   的任何计算；manifest 里明确标注「非全局不变量」。
2. artifact 必须保留 **`position_mm`**（三维全局坐标——这个才是不变量）。
3. 每行拟合出参考 helix 后**冻结它**，用**同一个冻结的 reference** 现场重算
   `node_s` 和 `hit_s`（都通过 `closest_helix_points` 投影）。两者同源同尺度 →
   裁剪和段分配自洽。
4. 回归测试：**同一条物理轨迹在不同 fit window 下 `node_s` 应该不同，
   但段分配与最终预测必须一致。**

> 一句话记忆：**位置是不变量，弧长不是。传数据传位置，弧长每次现算。**

---

## 7. Predictive covariance：Laplace 近似

主要来源：[notes/kalman_global_covariance.md](notes/kalman_global_covariance.md)

### 要回答什么问题

不只是「预测值是多少」，还要「**这个预测有多不确定**」。对 one-sided 外推来说，
未来的随机散射**在原理上无法由均值预测**——收益必须体现在「模型自己知道自己有多不准」。

### 数学

在最优解附近把目标函数 `F` 二阶展开（Laplace 近似），后验协方差就是 Hessian 的逆：

```
H = J_rᵀ V⁻¹ J_r  +  blkdiag(Q⁻¹)      参数 = [reference(6), kinks(2N)]
C = H⁻¹
pred_cov = J_y C J_yᵀ                   y = (LOC0, LOC1)
```

通俗理解：**Hessian 是「损失碗」的曲率**。碗越陡（H 大）→ 参数被数据钉得越死
→ 不确定度越小（C 小）。第二项 `Q⁻¹` 表示先验本身也在提供曲率——材料越薄，
先验越强，那个 kink 的不确定度越小。`J_y C J_yᵀ` 是标准的**误差传播**。

### factor-of-two 陷阱

目标写成 `F = ½χ²` 时，`H = ∂²F/∂θ²` 直接就是 `JᵀV⁻¹J + Q⁻¹`，`C = H⁻¹`，
**没有额外的 ½ 因子**；写成 `F = χ²` 则 `H = 2(JᵀV⁻¹J + Q⁻¹)`，需要补 2 倍。

**搞错就是方差差 2 倍、pull 差 √2 倍——所有校准指标全线错位，而结果看起来
「很像对的」。** 因此必须在碰真实数据之前，先用线性高斯 toy 的解析后验对拍，
相对容差 1e-10。

### 数值实现

```python
L = torch.linalg.cholesky(H + damping * I * H.diagonal().mean())
pred_cov = J_y @ torch.cholesky_solve(J_y.transpose(-1, -2), L)
```

**禁止显式 `inv(H)`**——显式求逆数值上更差且更慢，Cholesky 分解 + 三角回代是标准做法。
`damping` 是 Levenberg–Marquardt 式的对角加载，防止 H 病态时分解失败。

**阻尼必须预注册**：扫描完 damping 再挑一个「结果最好」的就是 p-hacking——
用一个数值旋钮制造出想要的结论。production damping 必须在契约冻结阶段确定，
扫描只作敏感性诊断；**若结论随 damping 翻转，直接判定不可信**。

---

## 8. 数据工程里的理论：三方 join 与 fail-closed

主要来源：[notes/acts_material.md](notes/acts_material.md)

这部分看似纯工程，其实每条都由一个「数据模型事实」驱动。

**核心困难**：ACTS 的材料 ROOT 文件**丢掉了 track 索引**，而且遍历的是
`unordered_map`（哈希表，**遍历顺序不保证**）→ **绝对不能按行序对齐**。

**解法**：引入 `propsummary` 作为**索引桥**——它同时有 `(event_nr, track_nr)`
和 5 个 bound 参数：

```
material_kf  ←(phi, eta 指纹)→  propsummary  ←(event_nr,track_nr)→  tracksummary
                Join B                              Join A
```

- **Join A** 是精确键匹配 + 5 参数逐一校验（同一批 double 的 float32 cast，
  容差取 float32 eps 量级）→ 硬门控。
- **Join B** 只能用方向指纹 `(phi, eta)`，因为材料侧的 `v_px/v_py/v_pz` 是
  **单位方向向量，不含动量信息**。所以要求**事件内 1-1 双射**——重复、缺失、歧义
  一律 **fail closed**（宁可停机，不可猜）。

**fail-closed 的方法论意义**：join 错了不会报错，只会**静默产出错误物理结论**。
在这种场景下，「宁可失败」比「尽力而为」正确得多。

### 语义相同的字段不一定数值相同

- `nMaterials` 只数 stepping summary 里有 surface material 的 step；
- 材料 ROOT 遍历完整 `materialInteractions`，**含 volume material**
  → 节点数必然 > `nMaterials`。

要求两者严格相等的门控**必然 100% 失败**。正确做法：权威节点数取 jagged 分支自身
长度（硬门控），`nMaterials` 只作软诊断。

### 聚合的三个陷阱

1. **volume material 节点的 `sur_id` 被写成 0** → 按 `sur_id` 聚合会把**所有
   volume 节点合并成一个**，物理上完全错误。聚合键必须是 `(sur_id, vol_id)`
   **且路径相邻**。
2. **顺序必须是「先聚合、后判剔除」**。反过来的话，10 个各自很小、加起来不小的
   节点会被全部丢掉。而且剔除判据应是**丢弃材料占总材料的比例**（物理量），
   不是节点计数（无物理意义的量）。
3. **masked limit 而非 clamp**：`x/X0 == 0` 时不能用 `log(clamp(x, eps))`——
   那会在阈值处凭空造出一个不连续的非零下限。正确做法是
   `torch.where(x > 0, highland(x), 0)`，真空返回**精确 0**。

---

## 9. 统计门控：怎样才算「证明了」

主要来源：[notes/cms_run2_alignment.md](notes/cms_run2_alignment.md)、
[decision_record.md](decision_record.md)

### paired 设计

**单进程双臂**：batch 构建只调用**一次**，single 和 broken 共享同一批数据、
同一种子、同一迭代预算、同一存活行集合。导出 `holdout_keys.csv` 并写 SHA256，
比较前强制断言两臂 key 清单逐行相同。

**为什么**：配对比较（同一条轨迹用两个模型各测一次）消除了轨迹间的差异，统计功效
远高于两组独立样本。而「排序后截断」不是稳定哈希 split，跨命令运行可能取到不同
子集——**那样的对比没有意义**。

### paired bootstrap 与「CI 上界 < 0」

重采样 1000 次（固定种子写入 manifest），得到 ΔRMSE 的 95% 置信区间。

**「严格改善」定义为 CI 上界 < 0，而不是点估计为负。**

点估计为负可能只是随机涨落。要求整个置信区间都在零以下，等价于「在 95% 置信水平上，
改善不是噪声」——把「看起来好一点」升级成「统计上确实更好」。

### 必需组集合：堵「真空通过」

一个隐蔽的作弊路径：设一堆门控条件，然后让大部分组因为「统计量不足」被跳过，
最后只剩一两个组通过 → 宣称「全部门控通过」。

对策是**预先声明必须具备统计量的组**：

```
{17:8, 24:4} × {interpolation}      × {LOC0}   # 必需
{17:8, 24:4} × {forward, backward}  × {LOC0}   # 必需
```

**任一必需组 N < 200 → 整体结论 `inconclusive`（等同未通过），不得记为 passed。**

### 条件化门槛：理论决定指标形状

| 适用范围 | 指标 | 门槛 |
|---|---|---|
| 全部组 | intersection failure rate | ≤ 5% |
| 全部组 | predictive χ²/component | ∈ [0.7, 1.5] |
| 全部组 | \|pull mean\| | ≤ 0.2 |
| 全部组 | 逐分量对角 held-out NLL | bootstrap 95% CI 上界 < 0 |
| **仅 interpolation × LOC0** | RMSE vs single | 改善 ≥ 20%，CI 上界 < -20% |
| **仅 forward / backward** | RMSE vs single | 回退 ≤ 5%，CI 上界 < +5% |

**为什么内插要求「大幅改善」、外推只要求「不退步」**：回到 §1 的物理——外推区间
之外的散射是**未来的随机量，均值不可预测**。要求外推 RMSE 也改善 20% 等于要求模型
预测随机数，**这是一个物理上不可能满足的门控**。外推的真实收益体现在 **pull 和
predictive χ² 的校准上**：模型应该说「我在这里不确定度是 ±0.6 mm」，而不是把
0.6 mm 的误差变成 0.1 mm。

**pull 的定义与意义**：`pull = (预测 - 真值) / √(measurement_var + pred_var)`。
如果不确定度估对了，pull 应服从**标准正态**：mean ≈ 0，std ≈ 1。

- pull std > 1 → 低估了不确定度（模型过于自信）
- pull std < 1 → 高估了不确定度（模型过于保守）
- χ²/component ∈ [0.7, 1.5] 是同一件事的另一种表述

**NLL 口径**：主门控用**逐分量对角 NLL**（与按 component 的门控结构一致），
完整 2×2 协方差的二维 NLL 只作附加诊断。混用口径会让「通过」含义不清。

### 双臂收敛诊断

broken 的参数规模（6 + 2N）远大于 single（6）。「相同迭代数」只保证**预算相同**，
不保证**都已收敛**。两臂各自记录初始/最佳/最终 loss、逐参数组梯度范数、
best_iteration、停止原因。

> **若 broken 未收敛，负结果只能归类为数值失败，不得据此否定物理模型。**

这句话是整套方法论的浓缩：**分清「物理错了」和「优化没跑好」**。

---

## 10. Gate 化执行：把「能否证伪」排在「能否成功」之前

```
Gate 0  轨迹 ID / join / closure（100–1000 events）  → 数据管道对不对
Gate 1  冻结契约与预注册常量                        → 防事后调参
Gate 2  broken-helix 数学 + 确定性 MC closure       → 数学实现对不对
Gate 3  posterior covariance closure               → 方差尺度对不对
Gate 4  全量导出 + paired holdout                   → 物理结论
Gate 5  通过后才规划联合对齐接入
```

设计原则：**每一关都在「最便宜的地方」检出「最致命的错误」**。

- join 错了 → 后面全是垃圾，所以 Gate 0 用 1000 events 就查完；
- 方差差 2 倍 → 所有校准指标错位，所以 Gate 3 用解析 toy 而不是真实数据来查；
- 常量在 Gate 1 冻结 → 保证 Gate 4 的结论不能被事后调参制造出来。

**Gate 2 的两条精髓测试**：

- **zero-material 极限**：`node_valid` 全 False 时，点、切向、**以及对 6 参数的梯度**
  必须与 `helix_point`/`helix_tangent` 逐元素一致。梯度也要对——因为它决定优化行为。
- **synthetic MC ensemble**：≥1000 条 toy 轨迹按已知 `Q_j` 抽 kink，检验覆盖率与
  pull。**「单次已知 kink 恢复」不算证明**——单次成功可能是运气，统计模型必须用
  统计方式验证。

**Gate 0 的硬规定**：任一硬门控不成立 → **停止，换方案，不得放宽容差硬凑。**

---

## 11. 一条端到端的逻辑主线

```
单 helix 拟合 → holdout 显示 bias≈0 但 RMSE 大 1–2 量级
    ↓ [只有随机效应能解释这个指纹]
诊断为未建模的多重散射
    ↓ [Huber/IRLS 已判死：把物理当离群点]
选 GBL 方案：reference helix + 材料节点 2D kink
    ↓ [自由 kink 会吸收几何信号]
kink 必须带 Highland 物理先验 Q_j = diag(θ0², θ0²)，且尺度不可训练
    ↓ [x/X0 无法从 ROOT 查表复现：pre/post split、入射角、volume material]
必须由 ACTS propagation 逐轨迹记录材料节点
    ↓ [材料 ROOT 丢 track key，遍历顺序不定]
用 propsummary 做索引桥，三方 join，fail-closed
    ↓ [helix 弧长 s 不是全局不变量]
artifact 传 position_mm；validation 每行用冻结 reference 现算 node_s/hit_s
    ↓ [段分配是阶跃函数]
node_s 与 segment_of_hit 全部 detach，重新线性化必须显式 checkpoint
    ↓ [目标 = ½χ²，含 measurement 项 + scattering 惩罚项]
Gauss-Newton Hessian → Laplace 近似 → predictive covariance
    ↓ [factor-of-two 会让所有校准指标静默错位]
先用解析 toy 对拍到 1e-10
    ↓ [多参数模型总能降 training loss]
paired 双臂 + bootstrap CI 上界 + 必需组集合 + 收敛诊断
    ↓
只有全部门控通过，才承认「材料感知模型确实更好」
```

---

## 12. 显式留下的开放前提

这些是方案**自己承认还没验证**的地方，实现时需要重点关注：

1. **perigee 位置约定**：`(v_x,v_y,v_z) == (-d0·sinφ, d0·cosφ, z0)` 是**假设**，
   Gate 0 先当诊断项检验，全通过才升格为 Join B 硬校验；不通过则说明 reference
   surface 不是原点 perigee 或符号约定不同。
2. **弦长近似弧长的误差**：`L_cum` 由节点位置**弦长求和**近似弧长，离散化误差随
   节点间距和曲率增大。1e-3 的容差是否够，要用 Gate 0 实测的节点间距分布确认。
3. **`energyLoss=False` 的自洽性**：因为模型不建模能损，记录 pass 也关掉能损，
   否则记录到的节点几何与待拟合模型不自洽。需用「`energyLoss=True` vs `False`
   的节点位置差 ≪ 传感器分辨率」量化确认。
4. **弱模相关性**：kink 与局部几何旋转在数学上相关——「传感器转了一点」和
   「粒子在那里被踢了一脚」可能产生几乎相同的残差。nominal 阶段不判；
   **「kink 是否吸收几何信号」留到联合对齐工作包用 truth-known toy closure 判定，
   不得据 nominal 通过外推**。
5. **性能**：GBL 文献的线性复杂度来自专用带状代数（bordered-band / Schur 消元）。
   **直接用 Adam 优化不会自动获得该复杂度**。若统计模型闭合但速度/内存不可接受，
   再评估可微 Kalman 或专用求解器——**不得用无材料先验的自由分段替代正式模型**。

---

## 首版明确不包含

electron bremsstrahlung / GSF、随机能损、段间可变曲率、非均匀磁场积分、
active DOF 重选。
