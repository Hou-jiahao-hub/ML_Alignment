# Broken-Joint Geometry Identifiability Design

**Date:** 2026-09-01

**Status:** Approved in chat; awaiting written-spec review

**Scope:** ACTS-independent truth-known closure for local `tx`, `ty`, and `rz`

## Purpose and claim boundary

The existing paired holdout gate establishes whether a material-aware broken
helix predicts held-out measurements better than a single helix. Lower holdout
RMSE does not by itself establish that fitted detector geometry is closer to the
truth. The missing test is a joint-alignment closure in which material nodes,
scattering kinks, detector misalignment, and measurement noise are all known.

The experiment must establish together that:

1. injected `tx`, `ty`, and `rz` corrections are recovered without bias;
2. a zero-geometry control remains consistent with zero correction;
3. fitted kink pulls remain calibrated and do not shift with geometry injection;
4. eliminating kink variables does not destroy the identifiable geometry
   information in the Hessian.

The allowed conclusion is conditional:

> Within the pre-registered synthetic track, material, noise, and injection
> distributions, a holdout improvement from the broken arm is evidence for a
> more accurate alignment only when every geometry-truth, kink, and Schur gate
> also passes.

This does not make holdout RMSE universally sufficient, and it does not claim
closure for `tz`, `rx`, or `ry`.

## Selected approach

The formal experiment is a nonlinear, ACTS-free, ODD-like closure that reuses
the repository's helix, broken-helix, local-transform, and Highland conventions.
It exercises nonlinear intersections, local rotations, segment assignment, and
joint nuisance elimination while retaining every latent truth.

A small linear Gaussian problem independently checks the Hessian partitions and
Schur algebra against closed-form answers. It is an implementation oracle, not
the physics result.

A full ACTS ensemble is deferred. ACTS does not currently expose realized
per-node scattering truth through the material artifact, and it is too expensive
for the pre-registered ensemble. It may later provide external validation but is
not part of this verdict.

## Generated experiment

### Geometry

Ten planes normal to global `x` are centered at
`x=(0,100,...,900) mm`. Their local `x/y/z` axes point along global `y/z/x`,
respectively. The stations at global `x=400 mm` and `x=700 mm` each contain
three adjacent modules centered at global `y=(-40,0,40) mm`; the other eight
stations are fixed reference surfaces centered at global `y=0`. At a target
station, the truth crossing is assigned to the nearest module center. Each
replicate has 96 tracks; every retained track crosses one module at each target
station and all reference stations. Truth freezes sensor assignment before
fitting, so geometry motion cannot change row identity.

Surface transforms follow `apply_alignment_torch`: translations are in the
sensor frame and rotations are right-multiplied. Correction truth is computed
from the full nominal and misaligned transforms,

\[
\delta t_{true}=R_{mis}^{T}(c_{nom}-c_{mis}),\qquad
R_{\delta,true}=R_{mis}^{T}R_{nom},
\]

not by assuming a component-wise negative injection.

### Tracks, material, kinks, and measurements

Each replicate generates 96 six-parameter helices inside `S_MAX`. Independent
truth draws use `x0=0 mm`, `y0~Uniform(-45,45) mm`,
`z0~Uniform(-50,50) mm`, `psi0~Uniform(-0.02,0.02) rad`, and
`tan_lambda~Uniform(-0.10,0.10)`. Charge is equiprobable `-1/+1`, momentum is
`Uniform(4,10) GeV`, and the uniform field is `Bz=2 T`. The signed curvature is
`rho = 0.3 * Bz * charge / (p_GeV * 1000) mm^-1`, and Highland receives
`q/p = charge/p_GeV` in `GeV^-1`. Candidates that do not validly intersect all
ten truth stations are rejected; failure to retain 96 tracks within 960 proposals
fails the replicate.

Every track has three ordered nodes at reference transverse arc lengths
`(250,550,650) mm`. Node positions are the corresponding reference-helix global
positions. Independent material thicknesses use
`x/X0~LogUniform(0.005,0.020)`. Truth records global node position, reference arc
length, `x/X0`, Highland width, and two kink components.
For node `j` on track `t`,

\[
k_{tj}\sim N(0,\sigma_{tj}^{2}I_2),
\]

where the existing `highland_kink_sigma` computes `sigma_tj` from truth `q/p`
and `x/X0`. Vacuum padding is structural and enters neither residual nor prior.

The truth broken trajectory intersects nominal surfaces to produce local
measurements. Every sensor supplies LOC0 and LOC1 with independent Gaussian
noise `sigma_LOC0=sigma_LOC1=0.05 mm`. No fitted quantity is reused to generate
truth.

### Injection levels

Each target sensor gets independent signs and magnitudes:

| Level | `abs(tx)`, `abs(ty)` | `abs(rz)` | Purpose |
|---|---:|---:|---|
| zero | `0 mm` | `0 rad` | false-geometry control |
| small | uniform `[0.25,0.75] mm` | uniform `[0.0025,0.0075] rad` | local-linearity check |
| production | uniform `[1.0,3.0] mm` | uniform `[0.01,0.03] rad` | current injection scale |

Production matches the asymmetric magnitudes generated by
`setupMisalignment` with `shift_mag_mm=2.0` and `rotation_mag_rad=0.02`; small is
one quarter of that range.

### Train and holdout

The replicate seed permutes tracks and assigns the first 77 to alignment and the
remaining 19 to blind holdout.
Geometry uses alignment tracks only. For a holdout track, local track variables
are fitted from non-target measurements, with all target measurements excluded;
the target measurements are predicted once. Both arms share split keys and
component masks. No holdout result selects a checkpoint, threshold, or setting.

## Paired single and broken solvers

Both arms optimize the same global geometry vector `g`: `tx`, `ty`, and `rz` for
six target sensors. Fixed non-target surfaces anchor geometry. Each track has a
five-parameter gauge-fixed reference helix `q_t`. Broken adds two components for
every active material kink `k_t`; single uses the identical path with zero nodes.

The objective is

\[
F(g,l)=\tfrac12\sum_i r_i(g,l)^T V_i^{-1}r_i(g,l)
+\tfrac12\sum_{t,j} k_{tj}^TQ_{tj}^{-1}k_{tj}.
\]

The prior term is an empty sum for single. Node positions and Highland widths
are fixed inputs. Segment indices are detached integers.

Both arms initialize `q_t` with the same gauge-fixed nodeless Gauss-Newton fit to
that track's non-target measurements, and initialize every kink to zero. Truth
parameters are forbidden as solver initial values.

Before running, the orchestrator hashes ordered row identities, measurements,
variances, masks, initial helices, and geometry. Shared single/broken fields must
match byte-for-byte. Both use the same Gauss-Newton linearization, block-LM
acceptance, convergence threshold, iteration limit, and invalid-intersection
policy. A mismatch fails before metrics are formed.

### Local elimination

For each track, linearization produces `H_gg,t`, `H_gl,t`, `H_ll,t`, `b_g,t`, and
`b_l,t`, with `l_t=[q_t,k_t]` for broken and `l_t=[q_t]` for single. The solver
accumulates

\[
S_g=\sum_t(H_{gg,t}-H_{gl,t}H_{ll,t}^{-1}H_{lg,t}),
\]

solves the small global geometry step, then back-substitutes local steps.
Cholesky and triangular solves are required; explicit inverses are forbidden.

Block-LM uses at most 30 iterations, starts at relative damping `1e-3`, divides
damping by 2 after an accepted step, and multiplies it by 10 after rejection.
It stops successfully only when the undamped Newton decrement is at most `1e-6`
and the maximum scaled parameter step is at most `1e-8`; stopping at the iteration
limit is non-converged. An arm's invalid-intersection or paired-drop fraction may
not exceed `1e-3`.

LM damping may find the optimum, but final covariance and information metrics
use the undamped Gauss-Newton system. Rank deficiency is reported, never rescued
by damping.

## Geometry-kink Hessian analysis

The full order is `[g,q,k]`. Reference parameters are eliminated first so the
kink diagnostic measures their incremental effect beyond ordinary track freedom:

\[
\bar H_{[g,k]}=H_{[g,k],[g,k]}
-H_{[g,k],q}H_{qq}^{-1}H_{q,[g,k]}.
\]

With `A=bar(H)_gg`, `B=bar(H)_gk`, and `D=bar(H)_kk`, eliminating kinks gives

\[
S_g=A-BD^{-1}B^T,
\qquad C=A^{-1/2}BD^{-1/2}.
\]

The implementation need not materialize wide `C`. Its squared singular values
are generalized eigenvalues of `BD^-1 B^T` relative to `A`; generalized
eigenvalues of `S_g` relative to `A` are retained geometry-information fractions.

The formal result uses physical `1x` Highland widths. Sensitivity runs scale
prior widths to `0.5x` and `2x`, 100 replicates each. They record the cross-block
spectrum and retention but do not change the primary verdict.

## Metrics and pre-registered gates

Metrics are first summarized within each independent replicate. Bootstrap
resamples replicates, never correlated tracks, sensors, or kinks. Ten thousand
resamples use seed `20260901`. Families over `tx/ty/rz` use a max-statistic
construction for simultaneous 95% intervals.

### Nonzero geometry

For each active DOF and small/production injection, report recovery slope,
`fit-truth` bias, truth-normalized RMSE, and correlation. Broken passes only if:

| Metric | Requirement |
|---|---:|
| recovery slope | interval inside `[0.90,1.10]` |
| absolute bias | upper bound `<= 0.10 * RMS(truth)` |
| truth-normalized RMSE | upper bound `<= 0.20` |
| correlation | lower bound `>= 0.95` |

For nonzero injections, the broken/single geometry-NRMSE ratio's simultaneous
95% upper bound must be at most `1.10` for every active DOF.

### Zero geometry

The marginal geometry covariance from the undamped reduced system standardizes
zero-truth corrections. Geometry-pull mean intervals must lie inside
`[-0.10,0.10]`, and standard-deviation intervals inside `[0.90,1.10]`.

### Kinks and direct absorption

Marginal kink covariance includes local conditional covariance and propagated
global-geometry uncertainty. Kink pulls `(k_fit-k_truth)/sigma_posterior` must
have mean intervals inside `[-0.10,0.10]` and standard-deviation intervals inside
`[0.90,1.10]`.

Every realization pairs zero and nonzero geometry while reusing tracks, truth
kinks, nodes, and measurement noise. For corresponding fitted kinks, define
`Delta k = k_fit(nonzero)-k_fit(zero)` and divide by generating Highland width.
Its RMS 95% upper bound must be at most `0.10`; its p95 95% upper bound at most
`0.30`.

### Schur information and rank

Report whitened cross-block singular values, generalized retention eigenvalues,
and maximum implied variance inflation. The 95% lower confidence bound on the
5th percentile of minimum retention must be at least `0.80`, limiting maximum
geometry-variance inflation to `1.25x`.

A replicate is deficient if required undamped `H_qq`, conditioned kink block,
`A`, or `S_g` has a minimum eigenvalue no greater than `1e-10` times its maximum
eigenvalue, or has a non-positive maximum eigenvalue. Such replicates receive no
regularized metric. More than `0.1%` rank-deficient primary replicates makes the
verdict `inconclusive`.

### Holdout and verdict

Holdout reports RMSE, bias, predictive NLL, pull mean, and pull width for both
arms. It is explanatory, not a geometry-truth gate. Overall verdict is:

- `passed`: every geometry, zero-control, kink, absorption, Schur, pairing,
  convergence, and rank requirement passes;
- `failed`: adequate identifiable data complete, but at least one truth or
  information gate fails;
- `inconclusive`: pairing, convergence, finite-data, or rank limits prevent
  evaluation on the pre-registered population.

Lower broken holdout RMSE cannot override `failed` or `inconclusive`.

## Formal ensemble

Only this exact configuration may emit a formal verdict:

| Setting | Value |
|---|---:|
| primary replicates per injection level | `400` |
| tracks per replicate | `96` |
| target sensors | `6` |
| nodes per track | `3` |
| alignment/holdout tracks | `77 / 19` |
| stress replicates per scale | `100` |
| stress prior-width scales | `0.5,2.0` |
| bootstrap resamples | `10000` |
| master seed | `20260901` |

CLI overrides may create smoke runs. Any override marks the run
`diagnostic_only` and suppresses a formal verdict.

## Code boundaries

### `scripts/joint_alignment_closure.py`

- immutable schema-versioned geometry, track, material, measurement, and arm
  data structures;
- truth generator, local misalignment injection, and exact correction truth;
- shared single/broken residual and prior objective;
- train/holdout split and row-identity hashing.

It imports no ACTS package and performs no experiment-directory I/O.

### `scripts/joint_alignment_schur.py`

- per-track Jacobians and normal blocks;
- shared block-LM with a legal zero-kink block;
- back-substitution and stationarity diagnostics;
- undamped marginal covariance, cross-block, Schur retention, and rank checks.

It imports no ACTS package and performs no output I/O.

### `scripts/run_joint_alignment_closure.py`

- CLI validation and replicate orchestration;
- paired arm and paired injection invariant checks;
- simultaneous bootstrap, gates, and verdict;
- atomic output and manifest lifecycle.

The runner starts a `running` manifest before computation and finalizes it after
success or exception. Algorithmic `failed`/`inconclusive` is still a completed
process; an exception records available partial metrics and process failure.

## Output contract

Each unique run directory below `data/experiments/` writes:

| Artifact | Contents |
|---|---|
| `manifest.json` | effective configuration, command, inputs, Git commit, environment, outputs, status, verdict |
| `preregistration.json` | semantic version, distributions, thresholds, seed, formal/diagnostic class |
| `replicates.jsonl` | pairing hashes, convergence, truth and holdout metrics per realization |
| `hessian_scan.npz` | spectra, retention, condition, rank, and stress arrays |
| `geometry_recovery.csv` | sensor-level truth, fit, uncertainty, arm, level, replicate key |
| `kink_pulls.csv` | node truth, fit, width, pull, paired shift, replicate key |
| `summary.json` | intervals, gates, verdict, and exact reasons |

JSON summaries reject NaN. Large arrays remain under `data/` and are not
committed.

## Failure handling

- Invalid intersections, non-finite values, pairing mismatches, and rank failures
  fail closed and are counted.
- An arm cannot silently drop a row. A row unusable in either arm is removed from
  both, with paired-drop fraction reported and gated.
- Damping history is recorded but excluded from final information matrices.
- Source ROOT/JSON artifacts are never modified or deleted.
- Aborted formal runs retain finalized manifests with metrics available at exit.

## Test strategy

Implementation follows TDD. Tests must cover:

1. ordered nodes, exact Highland widths, and seeded kink/noise reproducibility;
2. composition of injected and truth-correction transforms for `tx/ty/rz`;
3. byte-identical shared arm fields and exact single `N=0` behavior;
4. noiseless recovery of identifiable geometry and kinks;
5. linear-oracle agreement between block and dense solves;
6. zero cross block giving retention one;
7. normalized coupling `rho` giving retention `1-rho^2`;
8. undamped rank failure despite a solvable damped step;
9. marginal kink covariance agreement with a dense small-problem oracle;
10. replicate-level simultaneous bootstrap over all active DOFs;
11. diagnostic CLI artifact writing and formal-verdict suppression;
12. manifest finalization with partial metrics on injected failure.

After focused tests, run `make test`. The formal ensemble runs only in the conda
`pytorch` environment, never an ACTS/LCG shell. Completion always requires a
schema-versioned manifest, regardless of scientific verdict.

## Completion criteria

Implementation is complete when focused tests and `make test` pass, the exact
formal ensemble finishes with every output, its manifest records effective
configuration/input hashes/Git commit/metrics/status, and `summary.json` reports
`passed`, `failed`, or `inconclusive` without manual reinterpretation. The final
report must state whether lower holdout RMSE coincided with demonstrably more
accurate geometry rather than treating RMSE reduction alone as proof.
