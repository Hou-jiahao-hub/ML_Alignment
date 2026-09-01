# 评审：Broken-Joint Geometry Identifiability Design

**评审日期**：2026-09-01
**被评审对象**：`docs/superpowers/specs/2026-09-01-broken-joint-geometry-identifiability-design.md`
**被评审对象状态**：Approved in chat; awaiting written-spec review
**评审性质**：理论性评审（未运行任何实验；结论基于设计文档本身、`scripts/helix_model.py`
的 broken-helix 实现，以及 `docs/handoff/2026-08-27-mainline-free-helix-trial-008.md`
已发布的主线真值指标）
**评审结论**：**不建议按当前形态启动 formal ensemble。** 需先完成 §4 中标注为
「阻断」的四项改造。

---

## 0. 一句话结论

这份方案能验证**求解器实现正确**，但**无法支撑「broken helix 不会用 kink 替代几何修正」
这一目标结论**：它把最危险的那条通路（未建模几何误差被逐轨 kink 吸收）从实验设计里
排除掉了，而剩下的判据在当前参数标定下大多恒真。

按现设计跑出 `passed`，**不能**用来回答「holdout RMSE 下降是否说明探测器对齐更正确」。

---

## 1. 评审所依据的事实

| 编号 | 事实 | 出处 |
|---|---|---|
| F1 | kink 只改方向（`psi`、`theta`），曲率跨段不变，不建模能量损失 | `scripts/helix_model.py:290-345` `broken_helix_segments` |
| F2 | 段间弧长取自参考轨迹（broken-lines 线性化），非精确重积分 | 同上 docstring |
| F3 | `node_arc_lengths` 返回 **detach** 弧长；`assign_hit_segments` 返回 detach 整数 | `scripts/helix_model.py:235-289` |
| F4 | Highland 宽度按 RPP2018 eq.33.15，vacuum 节点严格返回 0 | `scripts/helix_model.py:193-225` |
| F5 | 主线 free-helix（**无任何 kink 模型**）在真 ACTS 数据上已达 `corr = 0.9981~0.9996`、`recovery = 0.9917~1.0006` | `docs/handoff/2026-08-27-mainline-free-helix-trial-008.md` §3.6 |
| F6 | 现有求解器常数：`LM_MAX_ITERS = 20`、`LM_MAX_STEP_R = 1e-3`；历史上曾有 14 个 sensor 被 `LM_MAX_STEP_R` 卡死 | `scripts/ml_alignment_autograd.py:111-115`；joint-only 无 warm start 试验记录 |

---

## 2. 缺陷清单

分级定义：
**阻断** = 使目标结论在逻辑上无法成立，必须改；
**严重** = 使判据丧失鉴别力或结论无法迁移到主线；
**一般** = 影响可复现性、成本或工程稳健性。

| # | 缺陷 | 级别 |
|---|---|---|
| A | 配对设计使「直接吸收」判据在数学上恒等于零 | **阻断** |
| B | 注入永远落在可拟合子空间内，真正的吸收机制被排除 | **阻断** |
| E | 全部是「不许失败」型判据，无阳性对照、无功效分析 | **阻断** |
| F | 从未测量 holdout 与几何精度的关联——而这正是要回答的问题 | **阻断** |
| C | 判据阈值比可达精度松 1–2 个量级 | 严重 |
| D | 散射尺度使 kink 对 target sensor 的杠杆低于噪声，retention ≈ 1 是必然 | 严重 |
| G | 真值与拟合共用同一模型，模型误设为零 | 严重 |
| J | 参考面固定在精确真值、module 无界，可辨识性被人为放大 | 严重 |
| K | 单 helix 对照臂无散射模型，不可比 | 严重 |
| H | 多重性方向反了；零几何 pull gate 功效压在阈值上 | 一般 |
| I | 收敛判据零容忍，有把结论变成 `inconclusive` 的现实风险 | 一般 |
| L | `node_s` 与 segment 冻结策略未预注册 | 一般 |

---

## 3. 逐条论证

### A（阻断）配对设计使「直接吸收」判据在数学上恒等于零

**设计原文**：`Delta k = k_fit(nonzero) - k_fit(zero)`，配对时「reusing tracks, truth
kinks, nodes, and measurement noise」；判据为 `RMS <= 0.10`、`p95 <= 0.30`。

**机制**：测量由「真值 broken 轨迹 ∩ **标称**面」生成，标称面与注入无关，因此
**两个 realization 的测量向量逐位相同**。

**论证**：按 `apply_alignment_torch` 约定（平移在 sensor 系、旋转右乘），
且 `T_mis = apply(T_nom, inj)`：

- 旋转：`R_mis · Rz(rz) = R_nom · Rz(rz_inj) · Rz(rz) = R_nom · Rz(rz_inj + rz)`，
  当 `rz` 遍历实数轴时与 `R_nom · Rz(rz)` 张成**同一集合**；
- 平移：`c = c_mis + R_mis·t = c_nom + R_nom·t_inj + R_nom·Rz(rz_inj)·t`
  `= c_nom + R_nom·(t_inj + Rz(rz_inj)·t)`。
  `t` 限制在局部 xy 平面（只有 `tx, ty`），`Rz` 保持该平面，故
  `Rz(rz_inj)·t` 也遍历同一平面 → 可达平移集合**同一**。

因此 **zero 与 nonzero 是同一个优化问题在 `g` 上的一个精确重参数化**
（局部 xy 平面内的一次旋转 + 平移，加 `rz` 的一次平移），最优点物理上完全重合。

**后果**：

1. 在设计要求的收敛精度（Newton decrement ≤ 1e-6）下 `Δk ≡ 0` 到数值噪声，
   吸收判据**恒真**，通过它不构成任何证据。
2. zero / small / production 三档注入给出的真值恢复统计量是**同一份信息**，
   差别只在起点离最优点多远——即只在测收敛半径。`400 × 3` 中 2/3 是重复算力。
3. 「zero-geometry control」作为「假几何」对照同时失效，因为它和 production 是同一个解。

---

### B（阻断）注入永远落在可拟合子空间内，真正的吸收机制被排除

**现实中的吸收机制**：存在一个求解器**拟合不了**的几何误差分量，它在残差里无处可去，
于是被每径迹自由的 kink 逐轨吸收 → holdout RMSE 下降，几何却更错。

**设计现状**：8/10 站固定在**精确真值**上；注入只有 `tx/ty/rz`，恰好就是被拟合的三个
DOF；`tz/rx/ry` 明确排除在闭合之外，**且也不注入**。

**后果**：实验世界里根本不存在「无处可去」的几何分量。设计把待证伪的假设本身从实验中
移除了。这与 A 是同一枚硬币的两面：A 说明配对无信息，B 说明**注入维度**无信息。

---

### C（严重）判据阈值比可达精度松 1–2 个量级

**可达精度估算**（production 档、`tx`）：
77 条对齐径迹、每站一个 module、3 个 module → 每 sensor 约 `77 × 2 / 6 ≈ 26` 次穿越；
每次穿越由 8 个固定参考面（16 个分量、`sigma = 0.05 mm`）内插约束，
单次对 sensor 位置的约束约 `0.06 mm`（含散射贡献）→

```
sigma_tx  ~  0.06 / sqrt(26)  ~  0.012 mm
```

`rz` 的杠杆臂约 20 mm → `sigma_rz ~ 6e-4 rad`。

**真值尺度**：`|tx| ~ U(1,3)` → `RMS(truth) = sqrt((3^3 - 1^3)/6) = 2.08 mm`；
`RMS(truth rz) = 0.0208 rad`。

| 判据 | 折算后的实际阈值 | 可达精度 | 松弛倍数 |
|---|---:|---:|---:|
| `\|bias\| <= 0.10 * RMS(truth)` | 0.21 mm | 0.012 mm | **~17x** |
| `NRMSE <= 0.20` | 0.42 mm | 0.012 mm | **~35x** |
| `corr >= 0.95` | — | 实测将是 ~0.9999 | 饱和 |
| `slope ∈ [0.90, 1.10]` | — | 实测偏差 ~0.005 | 饱和 |

**后果**：一个 0.05 mm 的系统偏差（每 sensor 约 4σ 的灾难性系统误差）可以轻松通过全部
四个门。small 档只把松弛倍数从 ~17 降到 ~8，无质变。

**独立反证（F5）**：主线 trial-008 用**没有任何 kink 模型的 free-helix**，在真 ACTS
数据上 `corr = 0.9981~0.9996`、`recovery = 0.9917~1.0006`。这套阈值**连「单 helix vs
broken helix」都区分不了**——单 helix 已经全过了。

---

### D（严重）散射尺度使 kink 对 target sensor 的杠杆低于噪声，retention ≈ 1 是必然

代入 `highland_kink_sigma`（F4），`p ∈ [4,10] GeV`、`x/X0 ∈ [0.005, 0.020]`：

```
p=4  GeV, x/X0=0.020  ->  theta0 = 4.1e-4 rad
p=10 GeV, x/X0=0.005  ->  theta0 = 0.8e-4 rad
```

平面垂直于全局 x、径迹近轴（`psi0 ∈ ±0.02`、`tan_lambda ∈ ±0.10`）→ `s ≈ x`。
node 在 `s = (250, 550, 650)`，target 站在 `x = 400, 700`：

| node `s` | 到 target 400 | 到 target 700 | 位移（theta0 = 2e-4） | / `sigma_meas = 0.05 mm` |
|---:|---:|---:|---:|---:|
| 250 | 150 mm | (450 mm) | 0.030 mm | **0.6σ** |
| 550 | — | 150 mm | 0.030 mm | **0.6σ** |
| 650 | — | **50 mm** | 0.010 mm | **0.2σ** |

**node 位置恰好被放在 target 站的紧上游，kink 在承载几何自由度的那些面上的杠杆臂被压到
最小**。几何信号 1–3 mm，kink 在 target 上的效应 0.01–0.06 mm，信噪比 30–100 : 1。

**Schur 语言下的后果**：`B = H_gk`（条件化后）天生就很小，`D = H_kk` 被 Highland 先验
主导，于是 `S_g = A - B D^-1 B^T ≈ A`，**`retention >= 0.80` 是构造性成立的**，与
broken helix 是否有病理行为无关。

设计中 0.5x / 2x 的先验宽度扫描**没有触及真正的无量纲量**
`kappa = theta0 * L_eff / sigma_meas`——它只改 `D`，不改 `B` 的几何杠杆。

---

### E（阻断）全部是「不许失败」型判据，无阳性对照、无功效分析

设计中十余个 gate，没有一个能证明「如果真的发生了吸收，这套 gate 会抓到」。
结合 A–D，`passed` 这个结果是**不可解释的**：它既可能意味着 broken helix 健康，
也可能意味着实验灵敏度为零。

预注册实验缺阳性对照，等于放弃了可证伪性。

---

### F（阻断）从未测量 holdout 与几何精度之间的关联

设计分别 gate 了几何真值和 holdout，却没有任何一处刻画二者的**联合分布**。

**逻辑循环**：如果所有几何 gate 都过了，你已经直接知道几何是对的，holdout RMSE 是冗余
信息；这推不出「在真值未知的 ACTS 场景下 holdout 可以当代理」。

设计给出的结论边界——「holdout 改善只有在所有 geometry-truth / kink / Schur gate 也通过
时才是更准确对齐的证据」——在真值未知时**不可执行**，因为那时你根本跑不了这些 gate。

要回答的其实是一个**代理效度（surrogate validity）**问题：需要跨一族配置（有些发生吸收、
有些不发生）证明 Δholdout 与 Δ几何误差同向。单点 pass/fail 做不到这件事。

---

### G（严重）真值与拟合共用同一模型，模型误设为零

真值轨迹也走 `broken_helix_segments`（F1、F2、F3）：曲率跨段不变（无能量损失）、
段间弧长取自 broken-lines 线性化、`node_s` detach、segment 分配 detach。
**这些近似在真值里同样存在，因此完全不可见。**

设计中的「No fitted quantity is reused to generate truth」只防了**数值泄漏**，
没防**模型泄漏**。而现实中的 kink-吸收几乎全部由误设驱动。

---

### H（一般）统计构造的三处问题

1. **多重性方向反了。** 「所有 DOF 都必须通过」是 intersection-union 检验，每个 DOF 用
   `alpha = 0.05` 已经控制族错误率；对**等价性**判据再套 max-statistic 同时区间只会
   无谓加宽区间、增加假失败，与设计追求严谨的意图相反。

2. **零几何 pull gate 的功效恰好压在阈值上。** 若 pull 只按 DOF 聚合而不跨 sensor 聚合，
   replicate 级均值的 `SE ≈ 1/sqrt(400) = 0.05`，95% CI 半宽 ≈ 0.098，与 `±0.10` 的阈值
   几乎重合——**完美实现也有约一半概率失败**；再叠加 max-statistic 则必然失败。
   设计没有预注册聚合层级，而这个选择让 gate 功效相差约 2.5 倍
   （跨 6 个 sensor 聚合后 `SE ≈ 0.02`，`±0.10` 才是一个 5σ 的门）。

3. **kink pull 只 gate 全局均值。** 系统性吸收的特征是「按 sensor / 按 node / 沿几何误差
   方向」的分组偏移，全局平均会把它抵消掉。

---

### I（一般）收敛判据零容忍，有把结论变成 `inconclusive` 的风险

设计要求：30 次迭代上限 + 缩放步长 `<= 1e-8` + Newton decrement `<= 1e-6`，
且「stopping at the iteration limit is non-converged」、paired-drop `<= 1e-3` 零容忍。

**前科**（F6）：joint-only 无 warm start 试验中，加大迭代后三判据全过，但 14 个 sensor
被 `LM_MAX_STEP_R` 卡死。

另外 LM 更新「接受除 2 / 拒绝乘 10」不对称：一次拒绝需要三次接受才能恢复到原阻尼。

---

### J（严重）参考面固定在精确真值、module 无界，可辨识性被人为放大

1. **module 无界。** 「assigned to the nearest module center」定义的是无界 Voronoi 元胞：
   外侧 module 覆盖 `y ∈ [20, +inf)`。模块物理边界从未定义，`rz` 的杠杆臂因此被人为放大，
   得到的 `rz` 可辨识性优于任何真实探测器。

2. **8/10 站固定在精确真值。** 这注入了现实中不存在的完美外部约束，把几何的弱模式全部
   消灭了。真实对齐里所有面都自由、只固定最小 gauge——**那才是 kink 与几何真正简并的
   地方**。

**后果**：即使实验全过，结论也**迁移不到主线**（17:8 / 24:4 全 sensor 自由）。

---

### K（严重）单 helix 对照臂无散射模型，不可比

设计中 single 臂是「the identical path with zero nodes」——**完全没有散射模型，连膨胀
协方差都没有**。它的残差被散射相关化却按独立处理，`chi2` 与几何协方差都是误设的，
其报告不确定度与 broken 臂不可比。

现实工程基线是「带散射膨胀协方差的单 helix」。缺这一臂会系统性抬高 broken 的相对优势。

同时，`broken/single` 的 NRMSE 比值判据是**单侧**的（只要求 broken 不比 single 差 10%），
而结论想要的是 broken **更好**。当前 gate 集合可以被「broken ≡ single」的配置完全通过。

---

### L（一般）`node_s` 与 segment 冻结策略未预注册

`node_arc_lengths` 返回 detach 弧长（F3）。设计没有说明：是初始化时冻结一次，
还是每个外层迭代对当前 reference 重算？

**后者会让目标函数在迭代之间改变**，LM 的接受/拒绝比较失去意义，
`decrement <= 1e-6` 也不可达。

`assign_hit_segments` 同理——需要预注册冻结点，并检查没有 hit 落在 node 的 `epsilon`
邻域内（否则目标函数在几何步长下不连续）。

---

## 4. 优化建议

按对「回答目标问题」的贡献排序。级别对应它修复的缺陷级别。

### R1（阻断）重新定义配对：注入落在**可拟合子空间之外**的几何扰动

*修复 A、B。对应设计文档 §Injection levels、§Kinks and direct absorption。*

把 `Delta k` 的配对从「同一测量 / 不同起点」改成「**有未建模几何误差 / 无未建模几何误差**」，
两者共用径迹、truth kink、节点、噪声。未建模分量可选：

- `tz / rx / ry` 注入但**不拟合**；
- sensor bow / 二阶形变；
- 让 8 个 reference 面本身带上**不被拟合的**小错位（最贴近 ACTS 现实）。

判据同步改造：不 gate `Delta k` 的 RMS，改为 gate

- `Delta k` 在**几何误差方向上的投影**；
- `corr(k_fit - k_truth, 未建模几何误差)`。

这两个量直接测量「kink 有没有沿着几何误差的方向系统性移动」，才是「吸收」的定义。

### R2（阻断）加阳性对照，并**预注册它必须 `failed`**

*修复 E。对应设计文档 §Formal ensemble、§Metrics and pre-registered gates。*

至少三档递增的吸收强度，例如：

| 档 | 手段 |
|---|---|
| 弱 | reference 锚从 8 面减到 4 面 |
| 中 | `kappa` 放大到 ~5（见 R4） |
| 强 | 注入 0.5 mm 的不可拟合 `tz` |

报告 gate 的**检出功效曲线**（吸收强度 vs 失败概率）。
只有在阳性对照确实 `failed` 之后，主配置的 `passed` 才可解释。

**这一条是把整份实验从「仪式」变成「可证伪检验」的关键。**

### R3（阻断→严重）判据从「相对真值 RMS」改为「相对统计精度」

*修复 C。对应设计文档 §Nonzero geometry。*

把 bias / NRMSE 判据统一改写成 pull 形式：`|bias| / sigma_stat`，
`sigma_stat` 取自**无阻尼 GN 边缘协方差**（设计已要求用无阻尼系统，直接复用）。

建议阈值量级：

| 判据 | 现行 | 建议 |
|---|---|---|
| bias | `<= 0.10 * RMS(truth)` | `<= 0.2 * sigma_stat` |
| 分散度 | `NRMSE <= 0.20` | `pull std ∈ [0.95, 1.05]` |
| `corr` / `slope` | gate | **降级为诊断项**（高 SNR 下饱和） |

同时把 `broken/single` 的 NRMSE 比值判据补成**双侧**：需要有一个「single 在某配置下确实
有偏」的证明，否则 broken helix 的存在意义没被验证。

### R4（阻断）补上 holdout–几何的联合分析

*修复 F。对应设计文档 §Holdout and verdict。这是直接回答目标问题的一条。*

每个 replicate 记录二元组 `(Delta holdout_RMSE, Delta geometry_error)`（broken 减 single），
然后：

1. bootstrap 回归斜率 + 符号一致率及其 CI；
2. **在阳性对照配置（R2）上**画 ROC：以 holdout 改善作为「几何改善」的分类器，报告 AUC；
3. 预注册两个可发表的量：
   - `P(holdout 改善 ∧ 几何变坏)` —— 假阳性率；
   - `P(holdout 改善 | 几何改善)` —— 灵敏度。

**只有这个才真正回答「holdout RMSE 下降能否说明对齐更正确」。当前设计给不出这两个数。**

### R5（严重）扫描无量纲散射数而不是先验宽度

*修复 D。对应设计文档 §Geometry-kink Hessian analysis 的 sensitivity 段、§Geometry。*

定义

```
kappa = theta0 * L_eff / sigma_meas
```

其中 `L_eff` = node 到**最近 target sensor** 的杠杆臂。扫描 `kappa ∈ {0.5, 2, 5, 10}`，
并明确标出 ODD / 主线的真实 `kappa` 落在哪一档；主判据在真实档给出，其余给趋势。

同时**把 node 位置从 target 站紧上游挪开**（现在 650 → 700 只有 50 mm，等于人为归零了
耦合），或把 node 位置显式作为扫描维度。

### R6（严重）引入受控的模型误设

*修复 G。对应设计文档 §Tracks, material, kinks, and measurements。*

真值生成改为：

- 逐段**精确重积分**（段内弧长按 kink 后的实际轨迹算，而非 reference 线性化）；
- 曲率随段变化（能量损失）；
- node 真值位置带小扰动。

拟合仍用仓库现有的 broken-lines 近似。这样才存在「模型错了一点」的现实条件。

### R7（严重）加第三臂：single + 散射膨胀协方差

*修复 K。对应设计文档 §Paired single and broken solvers。*

作为工程现实基线。三臂对比才能说明 broken helix 的增益来自「正确建模散射」，
而不是「多了自由度」。

### R8（严重）增加「全部面自由 + 最小 gauge」配置

*修复 J。这是把结论迁移到主线的前提。对应设计文档 §Geometry、§Formal ensemble。*

至少一个配置里放开所有面、只固定最小 gauge，报告：

- `S_g` 的弱模式谱；
- gauge 投影后的真值恢复；
- kink 是否沿弱模式方向吸收。

同时定义 module 的**物理边界与 acceptance**，禁止无界 Voronoi；每个 target sensor 报告
3×3 几何块的条件数与 hit 质心偏移。

### R9（一般）修正统计构造

*修复 H。对应设计文档 §Metrics and pre-registered gates。*

- 等价性判据改用 IU：每个 DOF 独立在 `alpha = 0.05` 判定，**去掉 max-statistic 收紧**
  （或明确限定它只用于「存在性」类判据）；
- **显式预注册聚合层级**：建议 `sensor × DOF` 先在 replicate 内聚合成一个标量，
  再 bootstrap replicate；
- 预注册要求 `threshold >= 3 * SE`，否则调整阈值或 replicate 数；
- kink pull 增加**分组**均值 gate（按 sensor、按 node index、按注入符号）。

### R10（一般）放宽求解器的零容忍

*修复 I、L。对应设计文档 §Local elimination、§Code boundaries。*

- 迭代上限提到 100+；
- 收敛判据改用相对量（如 `max |Delta g_i| / sigma_g,i <= 1e-3`）而非绝对 `1e-8`；
- LM 改用 Nielsen 更新（对称、基于增益比）；
- 把「非收敛 replicate 比例」设成**有预算的 gate**（如 `<= 1%`）而非零容忍；
- `node_s` / segment 的冻结策略写进 spec 并配单测，断言无 hit 落在 node 的
  `epsilon` 邻域。

### R11（一般）重分配算力

*修复 A 的副作用。对应设计文档 §Formal ensemble。*

三档 injection 在当前设计下是同一份信息（缺陷 A-2），建议改为
`1 x 400`（production）+ 一个小规模的收敛半径扫描，
把省下的约 800 个 replicate 全部投给阳性对照（R2）和 `kappa` 扫描（R5）。

---

## 5. 建议保留、不要改动的部分

以下设计决策是正确的，后续修订不应触碰：

- 预注册 + 一次性 holdout，且 holdout 不选择任何 checkpoint / 阈值 / 设置；
- 双臂共享字段的**逐字节哈希校验**，不匹配则在成指标前失败；
- 阻尼历史与最终信息矩阵**严格分离**（协方差与信息量一律用无阻尼 GN 系统）；
- 秩亏 **fail-closed**，且明确禁止用 damping 补救；
- `passed` / `failed` / `inconclusive` 三态区分，以及「算法性 failed 仍是完成的进程」；
- 「Lower broken holdout RMSE cannot override `failed` or `inconclusive`」这条硬规则；
- 一行不能只在单臂被丢弃（paired-drop 对称移除并 gate）；
- `manifest` 的 running → finalize 生命周期与异常时保留部分指标。

**问题全部集中在实验的物理配置与判据标定上，不在工程契约上。**

---

## 6. 建议的下一步

1. 按 R1 / R2 / R4 改写设计文档的 §Injection levels、§Kinks and direct absorption、
   §Holdout and verdict、§Formal ensemble 四节，产出 design v2；
2. design v2 定稿后再按 R3 / R5–R8 调整判据与配置；
3. R9 / R10 可与实现同步进行（属于 TDD 覆盖范围，可直接补进 §Test strategy）；
4. **在 design v2 未定稿前不要启动 formal ensemble**——当前形态跑出的
   `passed` 无法引用。
