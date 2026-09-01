# Broken-Joint Geometry Identifiability and Holdout-Surrogate Design v2

**Date:** 2026-09-01

**Status:** Review-corrected design; implementation plan source

**Supersedes:** `2026-09-01-broken-joint-geometry-identifiability-design.md`

**Review basis:** `2026-09-01-broken-joint-geometry-identifiability-review.md`

## 1. Revised scientific question

The first design mixed two different claims:

1. in a correctly specified joint model, are fitted `tx`, `ty`, and `rz`
   identifiable after adding Highland-constrained kinks; and
2. when truth is unknown, is lower holdout RMSE a reliable proxy for more
   accurate geometry under realistic model mismatch?

The first is a solver/identifiability closure. The second is a surrogate-validity
experiment and requires configurations where kink absorption actually occurs.
Version 2 answers them separately.

No finite synthetic ensemble can prove that kinks *never* replace geometry. The
formal conclusion is instead:

> Over the pre-registered scattering-strength and model-mismatch domain, report
> where broken-joint geometry is unbiased, where kink absorption is detected,
> and whether a holdout improvement predicts a geometry improvement with bounded
> false-assurance probability.

Only a passed sensitivity control and a passed proxy-validity gate permit the
statement that holdout reduction supports more accurate alignment in that
domain.

## 2. Review disposition

| Item | Decision | Design consequence |
|---|---|---|
| A | Correct | Remove zero/nonzero start-point `Delta k`; pair different physical truth data instead. |
| B | Partly correct | Keep in-subspace injection for identifiability, and add unmodelled reference-geometry modes for absorption. |
| C | Correct in direction | Replace truth-scale gates with posterior-pull gates; keep slope/correlation diagnostic. |
| D | Correct | Scan dimensionless `kappa` and lever-arm layout, not prior width alone. |
| E | Correct | Add forced and physical positive controls with required detection power. |
| F | Correct | Gate the joint distribution of holdout and geometry changes. |
| G | Correct | Generate formal truth with an independent piecewise transport, not the fitted broken-lines function. |
| H | Correct | Use intersection-union equivalence tests, explicit grouping, and analytic power checks. |
| I | Correct | Use Nielsen LM, relative stationarity, and a non-convergence budget. |
| J | Partly correct | Add finite module acceptance. Fixed references remain the mainline-matched primary configuration; all-free projected-gauge is a transfer stress test. |
| K | Correct in direction | Add a full correlated marginal-scattering single arm, not diagonal inflation. |
| L | Correct | Freeze `node_s`, `hit_s`, and segments once from the shared initializer and gate node-hit separation. |

## 3. Experiment families

### 3.1 Matched-model solver closure

A small ensemble uses the repository broken-lines model for both generation and
fit. Its only purpose is to verify implementation, posterior covariance, and
Schur algebra. It cannot support a real-world physics claim.

Required outcomes are calibrated geometry and kink pulls, agreement with dense
linear oracles, and exact equivalence between explicit-kink and analytically
marginalized formulations at one frozen linearization.

### 3.2 Independent-truth proxy experiment

Formal scientific evidence uses an independent event-driven piecewise-helix
transport:

- propagate exactly to physical material planes and measurement planes in a
  uniform `Bz=2 T` field;
- apply truth kinks in the incident transverse frame at the actual node;
- update segment curvature after a deterministic material energy-loss fraction;
- perturb recorded node positions relative to truth;
- intersect each downstream surface using the post-kink segment, rather than
  reference-trajectory arc increments.

The common generated track distribution is `x0=0 mm`,
`y0~Uniform(-45,45) mm`, `z0~Uniform(-50,50) mm`,
`psi0~Uniform(-0.02,0.02) rad`, `tan_lambda~Uniform(-0.10,0.10)`, equiprobable
charge sign, and `p~Uniform(4,10) GeV`. Curvature is
`0.3*Bz*charge/(p_GeV*1000) mm^-1`. Every measured component has independent
base noise `sigma_LOC0=sigma_LOC1=0.05 mm`.

Before kappa scaling, material uses `x/X0~LogUniform(0.005,0.020)`. At each node
truth momentum changes as
`p_after=p_before*(1-0.002*(x/X0)/0.020)`. The fitter receives a node position
whose global `y` and `z` coordinates have independent `N(0,0.25 mm)` errors;
the node's material-plane `x` coordinate remains exact. These constants are
fixed scientific inputs, not tuned to closure output. The zero-kink,
zero-energy-loss, zero-node-error limit must reduce to the ordinary helix oracle.

The fitter continues to use the repository's frozen-node broken-lines
approximation and constant curvature. The mismatch is therefore controlled and
observable, not shared by generator and fitter.

### 3.3 Geometry configurations

The compact telescope has planes at global `x=(0,100,...,900) mm`, with local
axes `x/y/z` along global `y/z/x`. Target stations at `x=400,700 mm` each have
three modules centered at local `x=(-40,0,40) mm`. Every module has finite active
bounds `abs(LOC0)<=20 mm`, `abs(LOC1)<=60 mm`; truth crossings outside bounds are
rejected before row identities are frozen.

Two geometry configurations are reported:

- **anchored primary:** six target modules fit `tx/ty/rz`; eight non-target
  stations are fixed. This matches the current mainline architecture.
- **all-free stress:** all fourteen surfaces fit `tx/ty/rz`. The exact nullspace
  of the undamped reduced geometry Hessian is identified at relative eigenvalue
  `1e-10`, projected out, and all truth/error/information metrics are evaluated
  only in the orthogonal quotient space. The nullspace dimension and basis
  stability are outputs; this configuration cannot silently borrow anchors.

The all-free result is a transfer diagnostic, not a reason to reject the
mainline-matched anchored result.

## 4. Physical pairing and absorption controls

Every replicate freezes one set of reference tracks, truth Highland draws,
measurement noise, target `tx/ty/rz` injection, and split keys. It then generates
different measurements for different physical truth conditions.

Each replicate retains 96 tracks after physical acceptance, then assigns 77
seed-permuted tracks to alignment and 19 to blind holdout. Holdout track
parameters use non-target measurements only; target measurements are predicted
once and never select a checkpoint or setting.

### 4.1 Fitted geometry injection

All scientific cells use the production injection distribution:

- `abs(tx),abs(ty) ~ Uniform(1,3) mm`, independent random signs;
- `abs(rz) ~ Uniform(0.01,0.03) rad`, independent random signs.

The exact correction is calculated by transform composition. Smaller scales
`0.25x,0.5x,1.5x` are a 25-replicate convergence-radius diagnostic only and do
not duplicate the formal ensemble.

### 4.2 Unmodelled reference-geometry mode

The anchored fitter always holds reference surfaces nominal. Truth reference
surfaces receive a coherent local-`ty` distortion

\[
\delta ty(x)=a\sin(2\pi x/900+\phi),
\]

with paired phase `phi~Uniform(0,2pi)` and amplitudes
`a=(0,0.05,0.25,0.50) mm`. These correspond to clean, weak, medium, and strong
unmodelled geometry. Target truth and measurement noise remain paired.

The physical absorption difference is now
`Delta k = k_fit(a)-k_fit(a=0)` on genuinely different measurement vectors.

### 4.3 Absorption direction

At the shared frozen linearization, profile the reference-track and fitted
geometry columns out of the unmodelled residual signature `r_u`. The remaining
kink score and precision define a predicted absorption template

\[
k_{template}=-D^{-1}J_k^T W r_{u,profiled}.
\]

For each track/node/component report:

- normalized projection
  `alpha_abs=(Delta k dot D k_template)/(k_template dot D k_template)`;
- `corr(Delta k, k_template)`;
- residual orthogonal fraction in the `D` metric;
- grouped kink-error means by target sensor, node index, injection sign, and
  unmodelled-mode phase quadrant.

Global ungrouped kink means are diagnostic only because cancellation can hide
absorption.

### 4.4 Positive controls

Two controls are required before a clean-cell pass is interpretable:

1. **Wiring control:** add a residual exactly in the profiled kink column space
   and disable its matching geometry column. The absorption detector must reject
   no-absorption in at least `99%` of 100 replicates.
2. **Physical controls:** medium and strong unmodelled-reference cells must be
   detected as absorption in at least `80%` and `95%` of replicates respectively;
   the clean-cell false-detection rate must have a 95% upper bound at most `5%`.

Failure of either sensitivity requirement makes the formal result
`inconclusive`, never `passed`.

## 5. Dimensionless scattering scan

For each kink-to-target pair define

\[
\kappa=\theta_0 L_{eff}/\sigma_{measurement},
\]

where `L_eff` is transverse path length to the first downstream target hit. A
pre-experiment calibration reads the existing material artifact, nominal states,
and nominal geometry and writes the empirical ODD `kappa` distribution with
input hashes. The frozen inputs are:

- material artifact SHA256
  `9d9b42e81aa0103f2b4fa21aa18af386fa58c1761dfb68571ee3bed7fdf50aab`;
- nominal geometry SHA256
  `b409bccfbec3008a37e9aae57a66dad7c2cb4c79ab2b31071aff6a8f760c8680`;
- nominal trackstates SHA256
  `317257a2c753d46aea8b9c623cc6d8d868e3c01e7514e89e8cb097990175acf1`.

The synthetic scan targets median `kappa=(0.5,2,5,10)`. Each target is generated
for two node layouts:

- near: material planes `x=(300,600,650) mm`;
- far: material planes `x=(50,150,450) mm`.

A deterministic bisection scales `x/X0` until generated median kappa is within
5% of target. This changes the prior scale while the near/far pair independently
changes the geometry-kink cross block. `kappa_calibration.json` identifies which
cell brackets the empirical ODD median and tails; that cell is called
`odd_matched`, not selected by closure results.

## 6. Three model arms

All arms share geometry parameters, gauge-fixed reference helices, row order,
initialization, frozen arc coordinates, optimizer, and convergence gates.

1. **single-iid:** zero kink nodes and measurement covariance `V`.
2. **single-marginal:** zero explicit kink parameters and frozen full correlated
   covariance `V_eff = V + J_k Q J_k^T`, including off-diagonal hit correlations.
3. **broken-map:** explicit kink parameters with fixed Highland precision `Q^-1`.

At a fixed linearization, `single-marginal` and `broken-map` must give the same
profiled geometry normal equation to numerical tolerance. This establishes that
any nonlinear difference comes from relinearization/model behavior, not merely
from adding parameters. Diagonal-only covariance inflation is forbidden.

Primary proxy comparisons use `broken-map - single-marginal`. `single-iid`
quantifies the cost of ignoring scattering and is never presented as the fair
uncertainty baseline.

## 7. Frozen discrete state

All arms first perform the same nodeless non-target fit. From that shared frozen
reference, they compute `node_s`, `hit_s`, node validity, and segment indices
once. Those tensors remain unchanged through every LM iteration and objective
comparison.

Every active hit must satisfy `min(abs(hit_s-node_s)) >= 5 mm`. A violation drops
the row symmetrically from all arms before fitting and is counted. Paired drops
above `1%` make the cell inconclusive.

## 8. Solver and information algebra

Per-track local variables are `[q_t]` or `[q_t,k_t]`; global variables are the
active geometry quotient coordinates. Block normal equations eliminate local
variables with Cholesky solves and back-substitution. Explicit inverses remain
forbidden.

Normal equations are built in fixed dimensionless coordinates. Geometry scales
are `(1 mm,1 mm,0.01 rad)` for `(tx,ty,rz)`. The gauge-fixed helix order
`(u,v,psi,tan_lambda,rho_scaled)` uses scales `(1 mm,1 mm,0.01 rad,0.10,1)`,
where `rho_scaled=rho/1e-4`. Broken fits optimize `eta=k/sigma_Highland`, so the
kink prior precision is the identity. These scales are part of the manifest.

For a normalized undamped Hessian `H`, relative damping is
`H_lambda=H+lambda*diag(d)`, where
`d_i=max(H_ii,1e-12*median_positive_diagonal(H))`. The damped matrix is used only
for a trial step; the undamped matrix supplies stationarity and information.

LM uses the Nielsen gain-ratio update:

- initialize relative damping `lambda=1e-3`, `nu=2`;
- accept only finite steps with gain ratio `rho>0` and valid intersections;
- after acceptance set
  `lambda *= max(1/3, 1-(2*rho-1)^3)` and reset `nu=2`;
- after rejection set `lambda *= nu` and `nu *= 2`.

The maximum budget is 120 iterations. Convergence requires undamped Newton
decrement at most `1e-6` and maximum posterior-standardized full parameter step
at most `1e-3`. A cell may contain at most `1%` non-converged replicates; above
that its result is inconclusive. Damping is excluded from all final covariance,
rank, and information calculations.

The Hessian order is `[g,q,k]`. Eliminate `q` first, then compute

\[
A=\bar H_{gg},\quad B=\bar H_{gk},\quad D=\bar H_{kk},\quad
S_g=A-BD^{-1}B^T.
\]

Report whitened cross-block singular values, retained-information generalized
eigenvalues, posterior geometry/kink covariance, condition numbers, and
per-sensor `3x3` geometry-block condition. Active rank uses relative threshold
`1e-10`; deficient primary replicates above `0.1%` make the cell inconclusive.

## 9. Geometry and kink closure metrics

For each sensor and DOF define geometry pull
`z_g=(g_fit-g_truth)/sigma_g` from the undamped marginal covariance. Within a
replicate, aggregate sensor-level statistics by DOF; bootstrap only independent
replicates.

Clean matched-model closure uses two-one-sided equivalence tests at `alpha=0.05`
independently for each DOF. Therefore equivalence decisions use 90% CIs; 95% CIs
are still reported for estimation and for non-equivalence metrics:

- geometry pull mean 90% CI inside `[-0.20,0.20]`;
- geometry pull standard-deviation 90% CI inside `[0.90,1.10]`;
- kink pull mean 90% CI inside `[-0.20,0.20]` for every node-index group;
- kink pull standard-deviation 90% CI inside `[0.90,1.10]` for every node group;
- empirical 68% and 95% coverage within binomial 95% intervals of nominal.

Recovery slope, truth-normalized RMSE, and correlation remain diagnostics because
they saturate at high signal-to-noise ratio. Before the formal run, analytic
normal/binomial standard errors must show every equivalence margin is at least
three expected standard errors; otherwise the formal configuration is invalid,
not silently resized after seeing closure results.

## 10. Holdout-surrogate validity

Use fixed engineering scales `s=(0.05 mm,0.05 mm,5e-4 rad)` to define a common,
arm-independent geometry error

\[
E_g^2=mean_{sensor,dof}((g_fit-g_truth)/s_{dof})^2.
\]

For each replicate compare `broken-map` with `single-marginal`:

- `Delta H = holdout_RMSE_broken - holdout_RMSE_marginal`;
- `Delta G = E_g_broken - E_g_marginal`.

Negative values mean improvement. Across clean and physical-control cells,
report bootstrap intervals for regression slope and sign concordance, and ROC
AUC using `-Delta H` to classify `Delta G<0`. Also report:

- joint false assurance `P(Delta H<0 and Delta G>0)`;
- conditional false assurance `P(Delta G>0 | Delta H<0)`;
- sensitivity `P(Delta H<0 | Delta G<0)`.

The proxy-validity gate requires:

- regression slope 95% lower bound `>0`;
- sign-concordance 95% lower bound `>=0.80`;
- ROC AUC 95% lower bound `>=0.80`;
- joint false-assurance 95% upper bound `<=0.05`;
- conditional false-assurance 95% upper bound `<=0.10`;
- sensitivity 95% lower bound `>=0.80`.

Undefined conditional probabilities due to empty denominators are
`inconclusive`, not zero. Results are reported by mismatch amplitude and kappa;
the aggregate gate is not allowed to hide a failing subgroup.

## 11. Pre-registered allocation

| Family | Cells | Replicates per cell | Total |
|---|---:|---:|---:|
| matched-model clean closure | 1 | 600 | 600 |
| physical mismatch at `odd_matched` kappa | 4 amplitudes | 200 | 800 |
| kappa x lever-layout scan at medium mismatch | 8 | 100 | 800 |
| all-free gauge stress, clean/medium | 2 | 100 | 200 |
| forced-absorption wiring control | 1 | 100 | 100 |
| convergence-radius diagnostic | 3 | 25 | 75 |

Bootstrap uses 10,000 replicate-level resamples and master seed `20260901`.
Reduced CLI settings are `diagnostic_only` and cannot emit a scientific verdict.

## 12. Verdicts

Each cell and the aggregate report one of:

- `passed`: closure/proxy gates pass and positive-control power is demonstrated;
- `failed`: identifiable, adequately powered data show a gate violation;
- `inconclusive`: sensitivity, convergence, pairing, finite-data, or rank budgets
  prevent the claim.

The top-level scientific statement is one of:

- `supports_holdout_proxy_in_preregistered_domain`;
- `does_not_support_holdout_proxy`;
- `inconclusive`.

Lower holdout RMSE can never override geometry, absorption, sensitivity, or rank
failure. A domain with mixed cell results must be reported as a bounded validity
region, not collapsed to a global pass.

## 13. Outputs and code boundaries

ACTS-free physics and algebra remain separate from orchestration:

- `scripts/joint_closure_types.py`: immutable schemas and constants;
- `scripts/joint_closure_truth.py`: matched and independent truth generators;
- `scripts/joint_closure_models.py`: three arm residual/covariance models and
  frozen discrete-state construction;
- `scripts/joint_closure_schur.py`: block normal equations, Nielsen LM,
  covariance, quotient gauge, and Hessian diagnostics;
- `scripts/joint_closure_metrics.py`: grouped pulls, absorption, power,
  bootstrap, ROC, and verdicts;
- `scripts/calibrate_joint_closure_kappa.py`: read-only empirical ODD kappa
  calibration;
- `scripts/run_joint_alignment_closure.py`: CLI, orchestration, atomic outputs,
  and manifest lifecycle.

Every run writes `manifest.json`, `preregistration.json`, `replicates.jsonl`,
`geometry_recovery.csv`, `kink_pulls.csv`, `hessian_scan.npz`,
`surrogate_validity.json`, and `summary.json`. The manifest starts as `running`
and is finalized on completion or exception with available partial metrics.
JSON rejects non-finite values; large generated artifacts remain under `data/`.

## 14. Required regression tests

Tests must prove, before formal execution:

1. old zero/nonzero start-point pairing produces invariant physical optima and is
   not used as an absorption metric;
2. physical mismatch pairing changes measurement bytes while preserving all
   intended latent draws;
3. finite module acceptance rejects out-of-bounds hits;
4. independent transport differs from fitted broken-lines under energy loss and
   node perturbation, but has an exact zero-mismatch limit;
5. `single-marginal` uses off-diagonal covariance and matches explicit-kink
   profiling at fixed linearization;
6. frozen node/hit arc lengths never change through LM, and node-hit separation
   is symmetrically gated;
7. Nielsen updates, relative convergence, non-convergence budget, and undamped
   rank failure follow their exact contracts;
8. dense linear oracles match Schur solves, marginal covariances, cross-block
   spectra, and quotient-gauge results;
9. forced absorption is detected while clean false detection remains bounded;
10. grouped kink means catch cancellation hidden by the global mean;
11. bootstrap resamples replicates, IU equivalence is per DOF, power checks run
    before results, and empty conditional metrics are inconclusive;
12. holdout-surrogate metrics distinguish all four sign combinations of
    `(Delta H,Delta G)` and compute ROC/AUC without label inversion;
13. reduced runs suppress formal verdicts and failures finalize manifests with
    partial metrics.

After focused tests, run `make test`. Formal ML execution uses only conda
`pytorch`, never an ACTS/LCG shell. Every algorithm experiment, including failed
or aborted runs, must finish with a schema-versioned manifest.
