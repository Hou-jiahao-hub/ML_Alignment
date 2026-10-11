# 交接：Track–Geometry 联立 Schur 方案的理论评审

**交接日期**：2026-09-22

**分支**：`feat/full-event-calibration`

**评审对象**：
1. `ML_Alignment_Track_Geometry_Joint_Normal_Equations_Theory_InlineMath_Final.pdf`
   （整理日期 2026-09-20，正规方程 / Schur 消元 / 跨 sensor 相关性的数学推导）
2. `plan.md`（分 Phase 0–9 的联立 Schur 求解实施方案）

**上游**：`docs/handoff/2026-09-07-alignment-methodology-roadmap.md` §H1、
`docs/handoff/2026-09-08-block-lm-convergence-contract.md`、`PROJECT_CONTEXT.md`

**文档性质**：**纯理论评审**。本轮未修改任何代码，未运行任何算法实验，未产生 manifest，
不改变任何既有预注册判据。文中所有数字要么引自既有记录，要么明确标注为量级估计。

---

## 0. 宣称边界（先读这条）

本文只回答一个问题：**在理论层面，PDF 的推导是否正确，以及 plan.md 是否可以照做。**

本文**不**宣称：

- Schur 联立能改善几何 truth 恢复（这必须由 H1 的预注册实验判定）；
- 任何现有生产结果有误（fixed-track 基线的正确性不在本次评审范围内）；
- §3 给出的 `tr(BΓ_ref⁻¹Bᵀ)/tr(V)` 量级是测量值（它是从 Iteration 10 的 RMSE 表
  反推的估计，依赖对该表 σ 取值的假设，见该节的显式说明）。

---

## 1. 总体判定

| 层面 | 判定 |
|---|---|
| PDF 的代数正确性 | **通过**。逐条核验无误，可直接作为实施依据 |
| PDF 的统计解释 | **有条件成立**。条件之一在本项目已被 Iteration 10 部分证伪，见 §3 |
| plan.md 能否照当前形态开工 | **不建议**。两条 Blocker、五条 Major 需先关闭 |

**一句话**：数学没有问题，计划有问题，而且方案的收益上限很可能已经被仓库里已有的
Iteration 10 数据限定住了——§9 的第 1 题应当在写第一行代码之前用现有工具回答。

---

## 2. PDF 推导逐条核验

| 节 | 内容 | 核验 |
|---|---|---|
| §3 | `A_t = [0; A_target]`，`B_t = [B_ref; B_target]` | 正确。参考层几何固定 ⇒ ref 行对 `a` 导数恒零 |
| §4 | `F = ½(r+Jδx)ᵀW(r+Jδx)` → `JᵀWJ δx = −JᵀWr` | 正确，标准 GN |
| §5 | `C_t` 仅含 target；`Γ_t`、`β_t` 含 ref+target | 正确。`W = diag(W_ref, W_target)` 的前提与生产一致：实测 `var_eLOC0`/`var_eLOC1` 逐分量对角，加载器不产生非对角项 |
| §6 | `S_t = C_t − G_tΓ_t⁻¹G_tᵀ`，`b_t = c_t − G_tΓ_t⁻¹β_t` | 正确。“Schur 不是额外物理假设，只是代数重写”这句话是准确的 |
| §7 | `C_{ij}=0`；`S_{t,ij} = −G_{ti}Γ_t⁻¹G_{tj}ᵀ` | 正确 |
| §8 | 一维最小例子 `S_{ij} = −g_i g_j/γ` | 正确 |
| §9 | `Cov(θ_t) ≈ Γ_t⁻¹` | **符号二义**。此处 `Γ_t` 含 target，是“几何固定、用全部 hit 拟合”的条件协方差；§10 的 `C_θ` 是“仅用 ref 拟合”的协方差。两者数值差别很大，同一符号承担了两个含义 |
| §10 | `R = V + B C_θ Bᵀ`；与联立 Schur “同一统计机制” | 结论正确，但**缺一步推导**，见 §2.1 |
| §11 | fixed-track ⇔ 施加 `δθ_t = 0`，而非 `G_t = 0` | 正确，且是全文最有价值的澄清 |
| §12 | 一条 track 在其穿过的目标 sensor 间形成 clique；全局稀疏度由 track topology 诱导 | 正确 |
| §14–15 | 关系汇总与边界（double counting、damping ≠ covariance、information ≠ covariance） | 正确且必要 |

**未发现数学错误。** 唯一推导缺陷是 §9/§10 的符号混用。`plan.md` §18.5 已要求把这一点
显式记录，但只要求“记录”，没有给出正确关系。

### 2.1 应补进推导的一步：Woodbury 桥

设 target 残差 `r = Aδa + Bδθ + ε`，`ε ~ N(0,V)`；ref hit 对 `δθ` 提供信息 `Γ_ref`。
联立正规矩阵：

```
[[ AᵀV⁻¹A ,  AᵀV⁻¹B         ],
 [ BᵀV⁻¹A ,  BᵀV⁻¹B + Γ_ref ]]  =  [[C, G],[Gᵀ, Γ_tgt + Γ_ref]]
```

Schur 消元：

```
S = C − G(Γ_tgt+Γ_ref)⁻¹Gᵀ
  = AᵀV⁻¹A − AᵀV⁻¹B(Γ_ref + BᵀV⁻¹B)⁻¹BᵀV⁻¹A
  = Aᵀ (V + B Γ_ref⁻¹ Bᵀ)⁻¹ A          ← Woodbury
  = Aᵀ R⁻¹ A
```

当径迹处于 ref-only 驻点（`β_ref = 0`）时同样有 `b = Aᵀ R⁻¹ r`。

**三条直接推论：**

1. **`plan.md` §18.4「target hit 是否进入 `Γ_t`」不是可配置选项。** 要得到联立 MLE
   必须进。把 target 从 `Γ` 里拿掉、却把 target 残差留在目标函数里，得到的迭代不是
   任何目标函数的 GN 步，LM 接受判据会与线性化自相矛盾。plan 把它列为“须显式记录的
   设计选择”，措辞上等于承认了一个不该存在的自由度。

2. **多出一个独立 oracle。** plan Phase 3 的 dense-vs-Schur 相等只是代数重写，两边
   共享同一套 `A/B/W`，**不可能**抓到 Jacobian 装配或索引错误。而
   `S ?= Aᵀ(V + BΓ_ref⁻¹Bᵀ)⁻¹A` 走完全不同的装配路径（显式构造稠密 `R` 再求逆），
   能抓到 dense oracle 抓不到的错。建议加进 `tests/test_joint_schur_oracle.py`。

3. **Phase 4 的 MC 闭合必须钉死 `C_θ = Γ_ref⁻¹`。** plan 现在写的是
   “`C_θ ≈ Γ⁻¹` under the exact definition used by the experiment”——把最容易出错的
   一步留成了待定项。几何固定、ref-only refit、在 target 面取残差，预言就是
   `Cov(r_i,r_j) = B_iΓ_ref⁻¹B_jᵀ`（`i≠j`），对角再加 `V_i`。用含 target 的 `Γ`
   会系统性偏小。

---

## 3. Blocker 1：Schur 修正在本项目是 O(1) 效应，但它修正的可能不是主要误差源

**这一条不需要写任何代码就能判定，数据已经在 `docs/history/ml_optimization_log.md`
Iteration 10 里。**

那次 measurement holdout 测的正是 `B Γ_ref⁻¹ Bᵀ` 所属的量：
`validate_helix_model.predictive_covariance()` 算的就是 `J H⁻¹ Jᵀ`，`H` 来自
ref-only 拟合。原表列为 bias / RMSE / p95（由原文“24:4 LOC1 的 measurement-normalized
RMS 内推/forward 约 0.96/0.98”与 `1.158/1.2 = 0.965`、`1.172/1.2 = 0.977` 对上确认）。

| 层 | 模式 | LOC0 RMSE (mm) | 该方向 σ (mm) | 归一化 |
|---|---|---|---|---|
| 17:8 (pixel) | interpolation | 0.083 | ~0.05 | **1.7** |
| 24:4 (strip) | interpolation | 0.170 | ~0.10 | **1.7** |
| 24:4 (strip) | LOC1 interp | 1.158 | 1.2（含下限） | 0.96 |
| 17:8 | backward | 0.579 | ~0.05 | **11.6** |
| 24:4 | backward | 0.903 | ~0.10 | **9.0** |

> **估计的前提**：σ 取该 Iteration 自己写的绝对门槛（pixel 0.05 / strip LOC0 0.10）
> 作为分辨率代表；外推行用的是 4hit/100mm 的放宽门槛，**不是生产口径**（8hit/300mm）。
> 这是量级估计，不是测量值。

### 3.1 两种读法，都对 plan 不利

**读法一（统计误差）**：若超额全部来自 `BΓ_ref⁻¹Bᵀ`，则 `R ≈ (2.8~3)V`（内推）到
`~100V`（backward）。于是：

- 方案动机被**定量证实**：fixed-track 的几何协方差 `C⁻¹` 偏乐观约 3 倍（backward 组
  两个量级）。这与 roadmap §2.1「几何协方差也会偏乐观」一致；
- 但同时 `S ≈ C/3`，**有效几何信息掉三分之二**，几何解不确定度涨约 1.7 倍。

**读法二（模型误差）**：Iteration 10 自己的验收结论是——

> 「失败不是 surface intersection 数值问题，而与单螺旋忽略材料效应/多次散射、单侧
> 窗口 lever arm 和传播方向不对称一致」

即超额的主要成分**不是** `Γ_ref⁻¹` 这个统计项，而是单螺旋模型本身的系统误差。
而 Schur 只传播 `Γ_ref⁻¹`。

| 读法 | 后果 |
|---|---|
| 统计误差 | 方案有效，但 `S` 比 `C` 小约 3 倍。H1 预注册的「truth RMS 改善 ≥10%」需重新论证：参考几何精确时 fixed-track 是**无偏**的（只是低效），Schur 的增益来自 GLS 加权效率，不是去偏 |
| 模型误差（Iteration 10 的结论倾向这一支） | `R = V + BΓ_ref⁻¹Bᵀ` 仍低估真实残差协方差，`Cov(a) ≈ S⁻¹` 依旧偏乐观，pull 仍会 >1。**散射项只能由 GBL/broken-helix 提供，而 `plan.md` §1.2 明确把它排除在外** |

### 3.2 要求

1. 在 Phase 0 之后、Phase 5 之前插入一个零成本判别实验：用现成的
   `predictive_covariance()`，在**生产口径**（8 hit / 300 mm、`--no-event-split` 行集）
   上逐目标层逐分量输出 `tr(BΓ_ref⁻¹Bᵀ)/tr(V)`，并把预测误差按「统计项 vs 残余」分解。
   这一步决定后面 9 个 Phase 是否值得跑。

2. `plan.md` §13 成功判据必须补一条**允许的负结果**：「Schur 把 fixed-track 下看似
   可测量的一部分模式如实暴露为弱模，导致 truth RMS 不改善」是**方法正确**的表现，
   不是失败。否则 Phase 7 的弱模诊断一旦命中，就会产生调参压力去把它掩盖掉——而这
   正是 roadmap §2.4 反复警告的事。

---

## 4. Blocker 2：LM 大阻尼极限恰好退化为已判死的路线

`plan.md` §3.5 写 `(S + λD)δa = −b`，§3.6 写 `δθ_t = −Γ_t⁻¹(β_t + G_tᵀδa)`。
两条放在一起看 `λ → ∞`：

```
λ → ∞   ⇒   δa → 0   ⇒   δθ_t → −Γ_t⁻¹ β_t
```

而 `Γ_t` 与 `β_t` **都含 target hit**。所以 `−Γ_t⁻¹β_t` 正是「几何固定在错位值、把
目标层 hit 并入径迹拟合」的 Gauss–Newton 步——即仓库已实测会把 truth 距离拉偏
**9.9 倍**、且 roadmap §6.5 明确列为禁止路线的那条。

**后果**：LM 最标准的安全阀（拒步就加阻尼）在这个方案里会**主动把解推向已知失效模式**。
而且 χ² 在那个方向上是**下降**的（这正是 9.9 倍那次实验的教训：chi2 变好而 truth 变差），
所以 `plan.md` §3.7 的接受判据（“nonlinear joint chi-square decreases”）
**不会拒绝它**。

补充：即使 `λ=0`，第一步的 `δθ` 里也含 `−Γ_t⁻¹β_tgt` 项。保证它不变成“径迹吸收错位”的，
只有 `δa` 同时在动这一点。

### 4.1 必须补的设计

1. **信赖域必须同时作用在 `δθ` 上**，不能只约束 `δa`。等价做法是阻尼整个联立系统
   （`Γ + λD_θ`）后再消元：`S(λ) = C + λD_a − G(Γ+λD_θ)⁻¹Gᵀ`。此时 `λ→∞` 给出
   `δθ→0`，退化到“不动”而不是“坏解”。
2. 每次迭代必须记录**目标残差下降在 `δa` 与 `δθ` 之间的归属分解**。`plan.md` §6 只记了
   “track back-substitution norm”，不足以证伪。
3. 合成层必须加一条**先红后绿**回归：构造一个几何错位，验证求解器把它归给 `δa` 而不是
   `δθ`；并验证大阻尼路径被信赖域拦住。这条应作为 Phase 3 之后、Phase 5 之前的强制门。

---

## 5. Major 级发现

### 5.1 ρ 的定义二义，会让整个 Phase 9 不可解释

`plan.md` §3.7 把接受判据写成 “the nonlinear **joint/profiled** objective”，两处都用
斜杠并列。但这是两个不同的量：

- **联合目标** `χ²(a+δa, θ+δθ)`，`δθ` 来自回代；
- **profile 目标** `χ²_prof(a+δa) = min_θ χ²(a+δa, θ)`，需要内层重新拟合径迹。

只阻尼几何时，回代的 `δθ` 精确最小化**二次模型**中的 θ，因此预测下降对应 profile
二次模型。若实际下降用 profile（内层重拟合到收敛）计算，会系统性大于预测，
**ρ 系统性 >1**，`plan.md` §9 对 ρ 的全部解读失效。

**要求**：二选一并写死。考虑到 §4 的 Blocker，推荐「阻尼联立系统 + 用回代 `δθ` 评估
联合 χ²」——两边用同一个二次模型，`ρ ≤ 1` 才有意义。

### 5.2 摧毁了 2026-09-08 已冻结的 block 解耦契约，plan 全文未提

`docs/handoff/2026-09-08-block-lm-convergence-contract.md` §5 有一条**精确结论**（不是
近似）：block 之间完全解耦，因此 `converged` 的 sensor 可以永久退休，回归见
`test_block_lm_blocks_are_decoupled_across_target_sensors`。

Schur 的全部意义就是制造 `S_ij ≠ 0`。**这条契约必然失效**：移动 sensor `j` 会改变
sensor `i` 的梯度。连带失效的还有：

- 四类终态（`converged`/`budget_exhausted`/`rank_deficient`/`intersection_failed`）
  是**逐 block** 定义的，全局系统里没有“这个 block”；
- `BLOCK_LM_DECREMENT_STATIONARY = 1e-6`（每 component）要换成全局 Newton decrement，
  阈值**必须重新预注册**，不能沿用；
- 逐 sensor damping（`damping_by_sensor`）在耦合系统里没有定义；
- 箱约束（`joint_clamp_t/r`）的 KKT 判定从 2268 个独立小问题变成一个 6804 维耦合约束
  问题，难度不是一个量级。

`plan.md` Phase 6 的 “Preserve current production safety expectations” 把这些一笔带过。

**要求**：Phase 6 之前单列一节，逐条说明每一条现有契约在全局系统里的替代物，以及哪些
阈值需要重新预注册。

### 5.3 有效 component 守恒必须扩展到 ref hit

现有守卫只看 target hit（`_sensor_residual_data` 的选择集是
`hit_mask & is_target & sensor_idx==s`）。联立后 **ref 残差进了目标函数，而径迹在动**
——`intersect_plane` 在 ref 面上同样可能失效。`plan.md` §3.7 说
“no originally valid component is silently dropped”，但没说明这个集合现在包含 ref。
**这是「靠丢困难 hit 换 χ² 下降」这条已被堵死的路径的一个新入口。**

### 5.4 与 roadmap 前置条件冲突

roadmap §H1 前置条件：**L1、L3、M1 全部完成**。当前状态：

- **L1**：代码与合成回归完成，**真实数据两臂尚未运行**
  （`2026-09-08-block-lm-convergence-contract.md` §11 明确列为“尚未做”）；
- **L3**：truth-veto 闭环脚本已实现；
- **M1（可观测性与参考层误差注入）：未做。**

`plan.md` 的 Phase 7（弱模/可观测性）与 Phase 8（参考层鲁棒性）**就是 M1**，但被折叠进
H1 内部、放在 Phase 5–6 之后。roadmap 把 M1 前置的理由是「否则谱变化可能只是求解器
未收敛」。在 Schur 系统里这个理由更强：`S` 的谱同时受“消元”和“未收敛”影响，没有
fixed-track 基线的无正则谱做对照，Phase 7 测到的弱模无法归因，
`R_i = tr(S_ii)/tr(C_ii)` 也没有基准可比。

**要求**：Phase 7/8 提到 Phase 5 之前，先在 fixed-track 基线上跑一遍产出对照谱。

### 5.5 plan 缺预注册块与量化判据

roadmap §4 第 1 条：跑任何真实数据前必须写预注册块。`plan.md` §13 的 12 条成功判据
全是定性的，且**未采纳 roadmap H1 已冻结的数值判据**：

- 主判据：小规模闭合问题中 Schur 步与完整系统步的无量纲相对差 `< 1e-8`；
- 主判据：独立模拟中几何 truth RMS 相对 fixed-track 改善 ≥10%（paired 95% CI 上界判定）；
- 次判据：径迹 truth 距离劣化 ≤1%；
- 证伪：chi2 降低而 truth 距离显著增加，或改善只在精确 nominal 参考下成立。

**要求**：逐字采纳，或显式写明修改理由（§3 给出了修改 “≥10%” 的正当理由）。

---

## 6. Minor 级发现

1. **`Γ_t` 的 gauge 已解决，plan 应引用而非重新发明。**
   `scripts/helix_model.py:214` 的 `pack_gauge_fixed_track_parameters` 已给出 5 参数
   可观测表示，因此 `Γ_t` 在该表示下满秩、Cholesky 可用，不存在零模。`plan.md` Phase 3
   只在 oracle 里提了 “5 independent parameters”，Phase 5 的生产路径没说。
   **陷阱**：`reference_gauge_frame(initial)` 是逐径迹、按线性化点确定的，每次重线性化后
   `δθ` 活在不同参数化里——回代后必须立刻 unpack 回物理 helix 参数再前进。Phase 1C 的
   pull 校准同样要把 truth 投到**同一个** frame。

2. **`plan.md` §4.3 的耦合归一化不保证 ≤1。**
   `‖S_ij‖_F / sqrt(‖S_ii‖_F‖S_jj‖_F)` 对 PSD 分块矩阵不是有界量。用迹做分母
   `‖S_ij‖_F / sqrt(tr S_ii · tr S_jj)` 由 Frobenius 次可乘性保证 ≤1。
   （§4.4 的 `R_i = tr(S_ii)/tr(C_ii)` 则良定义：`C − S ⪰ 0` ⇒ 主子块 PSD ⇒ `R_i ∈ [0,1]`。）

3. **`W = V⁻¹` 在 short-strip LOC1 上不是真实逆协方差。**
   `STRIP_Y_FLOOR = 1.44 mm²`（`scripts/ml_alignment_autograd.py:80`）是人为膨胀。
   Phase 1C 的 pull 与 `Cov(a) ≈ S⁻¹` 在这些分量上**按构造**就是失配的。要么把 pull 按
   “下限是否生效”分层，要么显式声明这是刻意的协方差膨胀、相应分量的 `S⁻¹` 不作协方差解释。

4. **稠密全局矩阵是浪费，建议直接上按 track topology 的块稀疏。**
   2268 个目标面 × 3 DOF = **6804** 维；稠密 float64 约 370 MB/份，LM 重试还要复制。
   而每条 track 只连接它穿过的目标 sensor（通常 2–4 个），61,482 条 track 最多诱导
   ~6 万个非零 3×3 off-diagonal 块，远小于 `C(2268,2) ≈ 2.6M`。`plan.md` Phase 5
   “start with an explicit dense matrix if target dimension is still small enough”
   的前提不成立。

5. **好消息：不需要 Millepede 式 gauge 约束。**
   非目标几何固定为参考系，每个目标面通过 ref hit 锚定到该参考系，全局刚体零模
   **不存在**。plan 没说这一点，但它显著降低了 Phase 7 的复杂度——`S` 的弱模全部是
   数据弱模，不是 gauge 零模，可以直接做谱分析而不必先处理零空间。

6. **Phase 1A 对 `β_ref ≈ 0` 的定位需澄清。**
   它是**初值条件**（§2.1 中 `b = AᵀR⁻¹r` 的化简依赖它），用作 Phase 0/1 的初始化质量门
   是对的；但接受第一步之后径迹就离开 ref-only 驻点，`β_ref ≠ 0` 是**正常**的，不能在
   迭代中当作失败判据。plan 把它写成笼统的 “stationarity 诊断”，容易被后续实现误用为收敛门。

7. **数值精度。**
   联合 χ² 现在含 ref 残差：~61k 径迹 × ~10 hit × 2 分量 ≈ 120 万个分量，而一次几何步
   带来的下降可能只有几百。相对量级 ~1e-4，float32 会淹掉。接受判据与 χ² 累加必须
   float64（生产路径部分是 float32）。

---

## 7. `plan.md` 各 Phase 可行性分级

| Phase | 判定 | 说明 |
|---|---|---|
| 0 基线冻结 | 可行 | 应补 §3.2 的判别实验，它决定后续是否值得跑 |
| 1 驻点/条件数/协方差校准 | 需修正 | 1A 的 `β_ref` 定位（§6.6）；1C pull 按方差下限分层（§6.3）、truth 投到同一 gauge frame（§6.1） |
| 2 Jacobian 验证 | 可行 | 分层维度设计得当，无异议 |
| 3 Dense oracle | 需增强 | dense-vs-Schur 只是代数重写，抓不到装配错误；**必须加 Woodbury oracle**（§2.1） |
| 4 跨 sensor 协方差闭合 | 需钉死定义 | `C_θ` 必须是 `Γ_ref⁻¹`（§2.1）。另需预先声明：MC 闭合**只验证代数**，不验证模型；通过不代表真实数据 pull 正常（§3） |
| 5 生产 Schur 累加 | 需前置 | 依赖 §5.2（契约替代物）与 §6.4（稀疏结构）先解决 |
| 6 Profiled/joint LM | **Blocker** | §4（阻尼极限退化）与 §5.1（ρ 二义）必须先关闭 |
| 7 可观测性/弱模 | 顺序错误 | 应提到 Phase 5 之前，先在 fixed-track 基线上产出对照谱（§5.4） |
| 8 参考几何鲁棒性 | 顺序 + 定位错误 | 同上；且这**不是补充验证，而是本方案增益唯一可能显著出现的区域** |
| 9 非线性/信赖域 | 依赖 §5.1 | ρ 定义不定则本 Phase 无意义 |
| §8 ML 扩展 | 隔离得当 | 门控清晰，§8.4 对学习权重的警告与 roadmap §6.2 一致，无异议 |

---

## 8. 建议增加的实验臂：diagonal-only Schur

`S` 的两个效应可以分离：

- **对角效应**：`S_ii = A_iᵀR⁻¹A_i`，按每个 target hit 的径迹预测不确定度做
  **异方差重加权**；
- **非对角效应**：`S_ij`，共享 track 诱导的跨 sensor 耦合。

在当前配置（2 个目标层、每条 track 通常 1–2 个 target hit、每个 sensor pair 平均只被
极少数 track 连接）下，**对角重加权几乎肯定是主效应**，而 PDF 与 plan 的叙事重心全在
非对角耦合上。

建议在 `plan.md` §9.6 的消融阶梯里插入一臂：**diagonal-only Schur（算 `S_ii` 但丢弃
`S_ij`）**。它把“重加权”和“耦合”分开归因，成本几乎为零，且能直接回答“跨 sensor 耦合
到底值不值 6804 维全局求解”。

**需要预先说明**：这个重加权与 roadmap §6.2 禁止的 IRLS/Huber **本质不同**——权重来自
`Γ_ref⁻¹` 的物理传播，与残差大小无关，不会把携带错位信号的大残差 hit 当离群点删掉。
这条论证应写进 plan，否则很容易被误判为已否决路线的复活。

---

## 9. 开工前必须先回答的问题（按优先级）

1. **Iteration 10 测到的目标层预测超额，有多少来自 `BΓ_ref⁻¹Bᵀ`，多少来自单螺旋模型
   误差？** 用现成 `predictive_covariance()` 在生产口径上分解。这一题决定整个方案的
   价值上限。
2. **LM 阻尼加在哪里？** 只加几何（`λ→∞` 退化为 9.9 倍路线）还是加在联立系统
   （`λ→∞` 退化为不动）？必须在写 Phase 6 之前定死，并配先红后绿的回归。
3. **ρ 的分母是 profile 二次模型还是联合二次模型？分子的实际下降用回代 `δθ` 还是
   内层重拟合？** 四种组合里只有两种自洽。
4. **2026-09-08 的逐 block 收敛契约（四类终态、逐 sensor damping、永久退休、每
   component `1e-6` 驻点阈值）在全局耦合系统里的替代物是什么？** 哪些阈值需要重新预注册？
5. **`converged` 的定义、箱约束 KKT 在 6804 维耦合系统里如何判定？**
6. **是否接受 roadmap H1 的「truth RMS 改善 ≥10%」作为主判据？** 若接受，需先论证在
   参考几何精确的模拟里它可达（§3 给出了怀疑的理由）；若修改，需在跑数据前写明理由。
7. **M1 是否先于 H1 执行？** 若坚持 plan 的顺序，需说明如何在没有 fixed-track 对照谱的
   情况下对 `S` 的弱模归因。

---

## 10. 下一步

本文件不是算法实验，无需 manifest（依 roadmap §4 末段）。建议的接续顺序：

1. 回答 §9 第 1 题（零成本，用现有脚本，不消费 calibration/validation）；
2. 据其结果决定是否修订 H1 的预注册主判据，并在
   `docs/history/ml_optimization_log.md` 写预注册条目；
3. 补齐 roadmap 的 H1 前置条件（L1 真实数据两臂、M1）；
4. 按 §4.1、§5.1、§5.2 修订 `plan.md` 后再开工。

**不要**在 §9 第 1、2、3 题关闭之前进入 `plan.md` Phase 5 及之后。
