# Broken-Joint Geometry Identifiability v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a truth-known, sensitivity-validated joint-alignment experiment that measures kink absorption and determines whether lower holdout RMSE predicts more accurate `tx/ty/rz` geometry over a pre-registered domain.

**Architecture:** Keep truth generation, arm models, Schur algebra, statistics, and orchestration in separate ACTS-free modules. Use three paired arms (`single-iid`, full-covariance `single-marginal`, and explicit-kink `broken-map`), an independent event-driven truth transport, positive absorption controls, replicate-level bootstrap, and schema-versioned experiment outputs.

**Tech Stack:** Python 3.10, PyTorch float64/autograd, NumPy, SciPy-free linear algebra where practical, pytest, existing project configuration and manifest helpers.

**Spec:** `docs/superpowers/specs/2026-09-01-broken-joint-geometry-identifiability-v2-design.md`

## Global Constraints

- Active geometry DOFs are exactly `(0,1,5) = (tx,ty,rz)`.
- Core modules must import without ACTS; formal execution uses conda `pytorch`, never an ACTS/LCG shell.
- Truth transport must not call `broken_helix_segments`; it may reuse fundamental single-segment helix equations and plane intersection.
- `single-marginal` must use the full correlated covariance `V + J_k Q J_k^T`; diagonal inflation is forbidden.
- Single and broken arms share row order, masks, initialization, optimizer, convergence rules, and frozen discrete state.
- `node_s`, `hit_s`, validity, and segment indices are computed once from the shared initializer and never updated during LM.
- Solver damping is excluded from final covariance, rank, cross-block, and Schur information metrics.
- Production algebra uses Cholesky/triangular solves; explicit inverses are allowed only inside a tiny dense test oracle.
- Bootstrap resamples independent replicates, not tracks, sensors, nodes, or components.
- TOST equivalence uses 90% intervals at `alpha=0.05` independently per DOF; estimation and proxy metrics report 95% intervals.
- Any reduced ensemble is `diagnostic_only` and cannot emit a formal scientific verdict.
- Every algorithm run, including failed or aborted runs, finalizes a schema-versioned `manifest.json`.
- Generated arrays stay below `data/` and must not be committed.

## File Map

| File | Responsibility |
|---|---|
| `scripts/joint_closure_types.py` | Frozen schemas, formal constants, validation, canonical serialization |
| `scripts/joint_closure_truth.py` | Telescope, accepted track sampling, matched truth, independent event-driven truth, physical mismatch |
| `scripts/joint_closure_models.py` | Shared initialization, frozen arc/segment state, three arm likelihoods, pairing hashes |
| `scripts/joint_closure_schur.py` | Normal blocks, quotient gauge, Schur reduction, Nielsen LM, covariance and information diagnostics |
| `scripts/joint_closure_metrics.py` | Pull groups, absorption templates, power, bootstrap, ROC, surrogate verdicts |
| `scripts/calibrate_joint_closure_kappa.py` | Read-only empirical ODD kappa calibration and manifest |
| `scripts/run_joint_alignment_closure.py` | CLI, cell orchestration, atomic tabular/NPZ/JSON output, manifest lifecycle |
| `tests/test_joint_closure_types.py` | Configuration and schema contracts |
| `tests/test_joint_closure_truth.py` | Geometry acceptance, reproducibility, independent truth and mismatch |
| `tests/test_joint_closure_models.py` | Frozen state, pairing, full covariance, arm equivalence |
| `tests/test_joint_closure_schur.py` | Dense oracles, rank, quotient gauge, Nielsen solver |
| `tests/test_joint_closure_metrics.py` | Grouped pulls, sensitivity, IU/TOST, ROC and proxy probabilities |
| `tests/test_run_joint_alignment_closure.py` | CLI classification, outputs, manifest success/failure |

---

### Task 1: Freeze schemas and pre-registration constants

**Files:**
- Create: `scripts/joint_closure_types.py`
- Create: `tests/test_joint_closure_types.py`

**Interfaces:**
- Produces: `ClosureConfig`, `CellSpec`, `SurfaceSet`, `GeneratedReplicate`, `FrozenProblem`, `FitResult`, `Interval`, `canonical_json_bytes()`, `formal_fingerprint()`.
- Consumes: only standard library, NumPy, and PyTorch.

- [ ] **Step 1: Write failing configuration and serialization tests**

```python
def test_formal_config_has_exact_v2_fingerprint():
    cfg = ClosureConfig.formal()
    assert cfg.active_dof == (0, 1, 5)
    assert cfg.tracks_per_replicate == 96
    assert cfg.train_tracks == 77
    assert cfg.holdout_tracks == 19
    assert cfg.bootstrap_resamples == 10_000
    assert cfg.master_seed == 20260901
    assert cfg.is_formal()


def test_any_scientific_override_is_diagnostic_only():
    cfg = dataclasses.replace(ClosureConfig.formal(), tracks_per_replicate=12)
    assert not cfg.is_formal()
    assert cfg.run_class == "diagnostic_only"


def test_canonical_json_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"bad": float("nan")})
```

- [ ] **Step 2: Run the tests and verify missing-module failure**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_types.py`

Expected: collection fails because `joint_closure_types` does not exist.

- [ ] **Step 3: Implement immutable schemas and exact constants**

```python
ACTIVE_DOF = (0, 1, 5)
ACTIVE_DOF_NAMES = ("tx", "ty", "rz")
FORMAL_SCHEMA_VERSION = 2
MASTER_SEED = 20260901
GEOMETRY_SCALES = (1.0, 1.0, 0.01)
REFERENCE_SCALES = (1.0, 1.0, 0.01, 0.10, 1.0)


@dataclass(frozen=True)
class ClosureConfig:
    schema_version: int = FORMAL_SCHEMA_VERSION
    active_dof: tuple[int, ...] = ACTIVE_DOF
    tracks_per_replicate: int = 96
    train_tracks: int = 77
    holdout_tracks: int = 19
    bootstrap_resamples: int = 10_000
    master_seed: int = MASTER_SEED
    max_lm_iters: int = 120
    max_nonconverged_fraction: float = 0.01
    max_rank_deficient_fraction: float = 0.001
    max_paired_drop_fraction: float = 0.01
    rank_relative_tolerance: float = 1.0e-10
    node_hit_separation_mm: float = 5.0
    measurement_sigma_mm: float = 0.05

    @classmethod
    def formal(cls) -> "ClosureConfig":
        return cls()

    def is_formal(self) -> bool:
        return formal_fingerprint(self) == formal_fingerprint(type(self).formal())

    @property
    def run_class(self) -> str:
        return "formal" if self.is_formal() else "diagnostic_only"
```

Implement tensor-bearing dataclasses with shape comments and validation methods.
`GeneratedReplicate` must contain paired latent draws separately from generated
measurements so tests can prove which fields remain equal across mismatch cells.

- [ ] **Step 4: Implement canonical finite JSON and fingerprinting**

```python
def canonical_json_bytes(value: object) -> bytes:
    def clean(item: object) -> object:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("canonical JSON contains a non-finite float")
        if dataclasses.is_dataclass(item):
            return clean(dataclasses.asdict(item))
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        return item
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()


def formal_fingerprint(config: ClosureConfig) -> str:
    payload = dataclasses.asdict(config)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
```

Validate positive counts, exact train/holdout sum, active DOF uniqueness, and
probability bounds in `__post_init__`. `run_class` remains a derived property so
constructing `ClosureConfig.formal()` cannot recurse through mutation logic.

- [ ] **Step 5: Run focused tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_types.py`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add scripts/joint_closure_types.py tests/test_joint_closure_types.py
git commit -m "feat: freeze joint closure v2 schemas"
```

---

### Task 2: Build finite telescope geometry and reproducible accepted tracks

**Files:**
- Create: `scripts/joint_closure_truth.py`
- Create: `tests/test_joint_closure_truth.py`

**Interfaces:**
- Consumes: `ClosureConfig`, `SurfaceSet`, `GeneratedReplicate`; `helix_point`, `helix_tangent`, `intersect_plane`, `highland_kink_sigma` from `scripts/helix_model.py`; local transform convention from `scripts/ml_alignment_autograd.py:1962-1990`.
- Produces: `make_telescope()`, `sample_accepted_tracks()`, `inject_target_geometry()`, `compute_truth_correction()`.

- [ ] **Step 1: Write failing telescope acceptance and transform tests**

```python
def test_target_modules_have_finite_nonoverlapping_acceptance():
    surfaces = make_telescope(ClosureConfig.formal())
    target = surfaces.target_mask
    assert target.sum().item() == 6
    assert torch.all(surfaces.half_size[target, 0] == 20.0)
    assert torch.all(surfaces.half_size[target, 1] == 60.0)
    assert assign_target_module(torch.tensor([-61.0, -39.0, 1.0, 59.0])).tolist() == [-1, 0, 1, 2]


def test_truth_correction_composes_misalignment_back_to_nominal():
    nominal = make_telescope(ClosureConfig.formal())
    misaligned, injected = inject_target_geometry(nominal, torch.Generator().manual_seed(7))
    correction = compute_truth_correction(nominal, misaligned)
    recovered = apply_active_correction(misaligned, correction)
    assert torch.allclose(recovered.centers, nominal.centers, atol=1e-11)
    assert torch.allclose(recovered.rotations, nominal.rotations, atol=1e-11)
    assert torch.count_nonzero(injected[:, [2, 3, 4]]) == 0
```

- [ ] **Step 2: Run tests to verify RED**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_truth.py -k 'telescope or correction'`

Expected: import failure for missing truth module.

- [ ] **Step 3: Implement the fourteen finite surfaces**

```python
def make_telescope(config: ClosureConfig) -> SurfaceSet:
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    centers, rotations, half_size, target, station = [], [], [], [], []
    for x in range(0, 1000, 100):
        ys = (-40.0, 0.0, 40.0) if x in (400, 700) else (0.0,)
        for y in ys:
            centers.append((float(x), y, 0.0))
            rotations.append(rotation)
            half_size.append((20.0, 60.0) if x in (400, 700) else (80.0, 80.0))
            target.append(x in (400, 700))
            station.append(x // 100)
    return SurfaceSet.from_arrays(centers, rotations, half_size, target, station)
```

Implement `assign_target_module()` with explicit closed intervals
`[-60,-20)`, `[-20,20)`, `[20,60]`; outside returns `-1`.

- [ ] **Step 4: Implement track sampling with exact distributions and rejection**

```python
def sample_reference_track(generator: torch.Generator) -> tuple[torch.Tensor, float]:
    charge = -1.0 if torch.rand((), generator=generator) < 0.5 else 1.0
    momentum = 4.0 + 6.0 * torch.rand((), generator=generator)
    y0 = -45.0 + 90.0 * torch.rand((), generator=generator)
    z0 = -50.0 + 100.0 * torch.rand((), generator=generator)
    psi0 = -0.02 + 0.04 * torch.rand((), generator=generator)
    tanl = -0.10 + 0.20 * torch.rand((), generator=generator)
    rho = 0.3 * 2.0 * charge / (momentum * 1000.0)
    return torch.stack((torch.zeros(()), y0, z0, psi0, tanl, rho)).double(), float(charge / momentum)
```

`sample_accepted_tracks()` proposes at most `10*n_tracks`, intersects all station
planes, applies finite target acceptance, freezes module indices, and raises
`RuntimeError("accepted 96 tracks requires more than 960 proposals")` on failure.

- [ ] **Step 5: Implement exact active correction composition**

Reuse `torch_euler_to_rotation_matrix` and express translation in the current
local frame. Compute truth with `R_mis.T @ (c_nom-c_mis)` and Euler-decompose
`R_mis.T @ R_nom`; assert inactive components are below `1e-12`.

- [ ] **Step 6: Run focused truth tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_truth.py -k 'telescope or accepted or correction'`

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add scripts/joint_closure_truth.py tests/test_joint_closure_truth.py
git commit -m "feat: generate finite joint closure telescope"
```

---

### Task 3: Add independent transport, mismatch cells, and physical pairing

**Files:**
- Modify: `scripts/joint_closure_truth.py`
- Modify: `tests/test_joint_closure_truth.py`

**Interfaces:**
- Produces: `propagate_event_driven()`, `generate_replicate()`, `generate_matched_replicate()`, `reference_distortion()`.
- Consumes: Task 2 telescope and accepted tracks; `highland_kink_sigma`.

- [ ] **Step 1: Write failing independent-truth and pairing tests**

```python
def test_physical_pair_changes_measurements_but_preserves_latent_draws():
    clean = generate_replicate(ClosureConfig.formal(), replicate=3, mismatch_mm=0.0)
    medium = generate_replicate(ClosureConfig.formal(), replicate=3, mismatch_mm=0.25)
    assert torch.equal(clean.truth_kinks, medium.truth_kinks)
    assert torch.equal(clean.measurement_noise, medium.measurement_noise)
    assert torch.equal(clean.split_keys, medium.split_keys)
    assert not torch.equal(clean.measured_local, medium.measured_local)


def test_independent_truth_is_not_the_fitted_broken_lines_model():
    data = generate_replicate(ClosureConfig.formal(), replicate=9, mismatch_mm=0.0)
    fitted_model_truth = generate_matched_replicate(ClosureConfig.formal(), replicate=9)
    assert torch.max(torch.abs(data.truth_points - fitted_model_truth.truth_points)) > 1e-8


def test_zero_kink_loss_and_node_error_reduce_to_single_helix():
    state = deterministic_truth_state()
    propagated = propagate_event_driven(state, zero_mismatch_material(), measurement_x=(400.0, 700.0))
    expected = ordinary_helix_intersections(state, (400.0, 700.0))
    assert torch.allclose(propagated.points, expected, atol=1e-10)
```

- [ ] **Step 2: Run tests and confirm missing behavior**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_truth.py -k 'physical_pair or independent_truth or zero_kink'`

Expected: failures because the independent transport API is absent.

- [ ] **Step 3: Implement event-driven segment propagation**

```python
def propagate_segment_to_x(state: TransportState, x_mm: float) -> TransportState:
    track = torch.stack((
        state.position[0], state.position[1], state.position[2],
        state.psi, state.tan_lambda, state.rho,
    ))
    center = torch.tensor((x_mm, 0.0, 0.0), dtype=torch.float64)
    normal = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
    path, point, valid = intersect_plane(track, center, normal, n_newton=20)
    if not bool(valid):
        raise RuntimeError(f"truth segment does not intersect x={x_mm} mm")
    direction = helix_tangent(track, path)
    return dataclasses.replace(
        state,
        position=point,
        psi=torch.atan2(direction[1], direction[0]),
        tan_lambda=direction[2] / torch.linalg.vector_norm(direction[:2]),
    )
```

Process a sorted event list of material and measurement planes. At material,
rotate direction by the two truth kink components in the incident transverse
basis, update `p_after=p_before*(1-0.002*(x/X0)/0.020)`, recompute signed rho,
and start a new segment at the actual arrival point. Do not call
`broken_helix_segments`.

Implement `generate_matched_replicate()` as a separately named solver-closure
generator that does call the production broken-lines primitives. Tag its output
`truth_model="matched_broken_lines"`; the runner must refuse to include those
replicates in proxy-validity metrics.

- [ ] **Step 4: Implement node recording error and reference mismatch**

```python
def reference_distortion(x_mm: torch.Tensor, amplitude_mm: float, phase: float) -> torch.Tensor:
    return amplitude_mm * torch.sin(2.0 * torch.pi * x_mm / 900.0 + phase)


def recorded_node_position(actual: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    error_yz = 0.25 * torch.randn((2,), generator=generator, dtype=torch.float64)
    return actual + torch.stack((torch.zeros((), dtype=torch.float64), error_yz[0], error_yz[1]))
```

Generate clean/weak/medium/strong cells with amplitudes
`(0.0,0.05,0.25,0.50) mm` from one latent draw bundle. Measurements use distorted
truth references; the fitter receives nominal reference surfaces.

- [ ] **Step 5: Implement kappa targeting and both lever layouts**

Compute each node's first downstream target path lever and median kappa. Bisection
uses bounds `[2^-12,2^12]` on the base `x/X0` scale, 60 iterations, and succeeds
only when relative median error is at most `0.05`. Store achieved median and scale
in `GeneratedReplicate`.

```python
def kappa(theta0: torch.Tensor, lever_mm: torch.Tensor, sigma_mm: float) -> torch.Tensor:
    return theta0 * lever_mm / sigma_mm
```

- [ ] **Step 6: Run all truth tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_truth.py`

Expected: all tests pass, including byte-level latent pairing.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/joint_closure_truth.py tests/test_joint_closure_truth.py
git commit -m "feat: add independent truth and absorption controls"
```

---

### Task 4: Freeze discrete state and implement the three comparable arms

**Files:**
- Create: `scripts/joint_closure_models.py`
- Create: `tests/test_joint_closure_models.py`

**Interfaces:**
- Consumes: generated replicate; `reference_gauge_frame`, `node_arc_lengths`, `assign_hit_segments`, and broken prediction primitives.
- Produces: `initialize_shared_reference()`, `freeze_problem()`, `arm_residual()`, `marginal_measurement_covariance()`, `pairing_digest()`.

- [ ] **Step 1: Write failing pairing, freeze, and covariance tests**

```python
def test_all_arms_share_byte_identical_common_fields():
    problem = small_frozen_problem(seed=4)
    digests = {arm: pairing_digest(problem.for_arm(arm)) for arm in ArmKind}
    assert len(set(digests.values())) == 1


def test_arc_and_segment_state_is_frozen_across_parameter_updates():
    problem = small_frozen_problem(seed=5)
    before = problem.discrete_digest
    _ = arm_residual(problem, ArmKind.BROKEN_MAP, problem.geometry + 0.01, problem.local)
    assert problem.discrete_digest == before


def test_single_marginal_covariance_has_scattering_correlations():
    problem = small_frozen_problem(seed=6)
    covariance = marginal_measurement_covariance(problem.track(0))
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    assert torch.max(torch.abs(off_diagonal)) > 0
    torch.linalg.cholesky(covariance)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_models.py`

Expected: collection fails because `joint_closure_models` is absent.

- [ ] **Step 3: Implement shared initialization and one-time freezing**

Use the existing gauge-fixed Gauss-Newton nodeless refit on non-target
measurements. Then compute node/hit arc coordinates once, assign segments once,
and store cloned detached tensors plus a SHA256 digest.

```python
def freeze_problem(data: GeneratedReplicate, config: ClosureConfig) -> FrozenProblem:
    reference = initialize_shared_reference(data, config).detach()
    node_s, node_ok = node_arc_lengths(reference, data.recorded_node_positions, data.node_valid)
    hit_s = project_hit_arc_lengths(reference, data.truth_surface_points).detach()
    segments = assign_hit_segments(node_s, hit_s, node_ok).detach()
    separation = torch.where(
        node_ok[:, :, None],
        torch.abs(node_s[:, :, None] - hit_s[:, None, :]),
        torch.full_like(node_s[:, :, None] - hit_s[:, None, :], float("inf")),
    )
    keep = separation.amin(dim=(1, 2)) >= config.node_hit_separation_mm
    return FrozenProblem.from_data(data, reference, node_s, hit_s, segments, keep)
```

Symmetrically remove rejected tracks from every arm and record the paired-drop
fraction.

- [ ] **Step 4: Implement full marginal scattering covariance**

Flatten active measurement components per track, compute the prediction
Jacobian `J_k` at the frozen state, and form:

```python
def marginal_measurement_covariance(track: FrozenTrack) -> torch.Tensor:
    measurement = torch.diag(track.measurement_variance)
    kink_covariance = torch.diag(track.kink_sigma.repeat_interleave(2).square())
    return measurement + track.kink_jacobian @ kink_covariance @ track.kink_jacobian.T
```

Factor with Cholesky and whiten residual/Jacobian by triangular solve. Never
replace this matrix with its diagonal.

- [ ] **Step 5: Implement arm residual/prior contracts**

`single-iid` uses zero-width kink tensors and diagonal `V`; `single-marginal`
uses zero explicit kinks and full frozen covariance; `broken-map` predicts with
explicit kinks and appends standardized prior residuals `k/sigma`. All return one
`ArmEvaluation(residual_vector, jacobian_masks, valid)` shape contract.

- [ ] **Step 6: Run focused model tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_models.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/joint_closure_models.py tests/test_joint_closure_models.py
git commit -m "feat: add paired joint closure model arms"
```

---

### Task 5: Implement Schur algebra, quotient gauge, and dense oracles

**Files:**
- Create: `scripts/joint_closure_schur.py`
- Create: `tests/test_joint_closure_schur.py`

**Interfaces:**
- Consumes: frozen arm evaluations.
- Produces: `NormalBlocks`, `schur_reduce()`, `back_substitute()`, `profile_q_then_k()`, `quotient_geometry_basis()`, `marginal_covariance_blocks()`, `information_diagnostics()`.

- [ ] **Step 1: Write failing closed-form Schur tests**

```python
def test_zero_cross_block_retains_all_geometry_information():
    a = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    b = torch.zeros((2, 1), dtype=torch.float64)
    d = torch.ones((1, 1), dtype=torch.float64)
    result = information_diagnostics(a, b, d, rank_rtol=1e-10)
    assert torch.equal(result.retention, torch.ones(2, dtype=torch.float64))


def test_scalar_normalized_coupling_has_one_minus_rho_squared_retention():
    rho = 0.6
    result = information_diagnostics(
        torch.ones((1, 1), dtype=torch.float64),
        torch.tensor([[rho]], dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        rank_rtol=1e-10,
    )
    assert result.retention.item() == pytest.approx(1.0 - rho**2)


def test_block_solution_matches_dense_oracle():
    blocks = deterministic_normal_blocks()
    reduced = schur_reduce(blocks)
    delta_g = torch.linalg.solve(reduced.hessian, -reduced.gradient)
    delta_l = back_substitute(blocks, delta_g)
    dense = torch.linalg.solve(blocks.dense_hessian(), -blocks.dense_gradient())
    assert torch.allclose(torch.cat((delta_g, *delta_l)), dense, atol=1e-10)


def test_explicit_kink_profile_matches_full_covariance_marginal_arm():
    explicit, marginal = fixed_linearization_arm_blocks()
    explicit_geometry = schur_reduce(explicit)
    marginal_geometry = schur_reduce(marginal)
    assert torch.allclose(explicit_geometry.hessian, marginal_geometry.hessian, rtol=1e-9, atol=1e-11)
    assert torch.allclose(explicit_geometry.gradient, marginal_geometry.gradient, rtol=1e-9, atol=1e-11)


def test_quotient_error_is_invariant_to_nullspace_shift():
    hessian, truth, estimate, null_shift = gauge_oracle()
    quotient = quotient_geometry_basis(hessian, rank_rtol=1e-10)
    baseline = quotient.basis.T @ (estimate - truth)
    shifted = quotient.basis.T @ (estimate + null_shift - truth)
    assert torch.allclose(baseline, shifted, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_schur.py -k 'retains or coupling or dense'`

Expected: import failure for missing Schur module.

- [ ] **Step 3: Implement solve-only Schur reduction and back-substitution**

```python
def solve_spd(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    factor, info = torch.linalg.cholesky_ex(0.5 * (matrix + matrix.T))
    if int(info) != 0:
        raise RankDeficientError("undamped active block is not positive definite")
    return torch.cholesky_solve(rhs, factor)


def schur_reduce(blocks: NormalBlocks) -> ReducedSystem:
    hessian = blocks.h_gg.clone()
    gradient = blocks.b_g.clone()
    for h_gl, h_ll, b_l in zip(blocks.h_gl, blocks.h_ll, blocks.b_l):
        solved_h = solve_spd(h_ll, h_gl.T)
        solved_b = solve_spd(h_ll, b_l[:, None])[:, 0]
        hessian -= h_gl @ solved_h
        gradient -= h_gl @ solved_b
    return ReducedSystem(symmetrize(hessian), gradient)
```

Implement back-substitution as
`delta_l=-solve(H_ll,b_l+H_lg delta_g)`.

- [ ] **Step 4: Implement rank and information diagnostics without inverse**

Whiten `A`, solve for `B D^-1 B.T`, and use `eigvalsh` on the symmetric whitened
lost-information matrix. Retention is `1-lost_eigenvalue`. Reject a block when
`lambda_max<=0` or `lambda_min<=rank_rtol*lambda_max`.

- [ ] **Step 5: Implement quotient-gauge basis**

```python
def quotient_geometry_basis(hessian: torch.Tensor, rank_rtol: float) -> QuotientBasis:
    values, vectors = torch.linalg.eigh(symmetrize(hessian))
    maximum = values[-1]
    active = values > rank_rtol * maximum
    if maximum <= 0 or not bool(active.any()):
        raise RankDeficientError("geometry has no identifiable quotient")
    return QuotientBasis(
        basis=vectors[:, active],
        nullspace=vectors[:, ~active],
        eigenvalues=values,
    )
```

All-free truth and estimates are projected with `basis.T`; report nullspace
dimension and principal angles between replicate bases.

- [ ] **Step 6: Implement marginal geometry and kink covariance blocks**

Use block inverse identities expressed entirely as solves. Test the returned
small-problem blocks against `torch.linalg.inv(dense_hessian)` only in the test
oracle, including the propagated geometry term in each kink marginal.

- [ ] **Step 7: Add source guard against production inverse calls**

```python
def test_production_schur_module_never_calls_inverse():
    source = inspect.getsource(joint_closure_schur)
    assert "torch.linalg.inv" not in source
    assert ".inverse(" not in source
```

- [ ] **Step 8: Run Schur tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_schur.py`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 5**

```bash
git add scripts/joint_closure_schur.py tests/test_joint_closure_schur.py
git commit -m "feat: add joint closure Schur diagnostics"
```

---

### Task 6: Add Nielsen block-LM and stationarity budgets

**Files:**
- Modify: `scripts/joint_closure_schur.py`
- Modify: `tests/test_joint_closure_schur.py`

**Interfaces:**
- Produces: `solve_joint(problem, arm, config) -> FitResult`, `fit_holdout_tracks()`, `evaluate_holdout()`, `nielsen_update()`, `stationarity_report()`.
- Consumes: Task 4 arm evaluation and Task 5 Schur helpers.

- [ ] **Step 1: Write failing gain-ratio, convergence, and rank tests**

```python
def test_nielsen_accept_and_reject_updates_are_exact():
    accepted = nielsen_update(damping=1e-3, nu=2.0, gain_ratio=0.75)
    assert accepted.accepted
    assert accepted.damping == pytest.approx(1e-3 * max(1/3, 1-(2*0.75-1)**3))
    assert accepted.nu == 2.0
    rejected = nielsen_update(damping=1e-3, nu=2.0, gain_ratio=-0.1)
    assert not rejected.accepted
    assert rejected.damping == pytest.approx(2e-3)
    assert rejected.nu == 4.0


def test_damped_step_cannot_rescue_undamped_rank_failure():
    problem = singular_geometry_problem()
    with pytest.raises(RankDeficientError, match="undamped"):
        solve_joint(problem, ArmKind.BROKEN_MAP, ClosureConfig.formal())


def test_relative_step_not_absolute_step_controls_convergence():
    report = stationarity_report(
        decrement=5e-7,
        step=torch.tensor([1e-6]),
        posterior_sigma=torch.tensor([1e-2]),
    )
    assert report.converged
    assert report.max_standardized_step == pytest.approx(1e-4)


def test_holdout_refit_excludes_every_target_component():
    problem = small_frozen_problem(seed=21)
    fit = solve_joint(problem.train, ArmKind.BROKEN_MAP, diagnostic_config())
    result = evaluate_holdout(problem.holdout, fit.geometry, ArmKind.BROKEN_MAP, diagnostic_config())
    assert not result.fit_component_mask[problem.holdout.is_target].any()
    assert result.predicted_component_mask[problem.holdout.is_target].any()
```

- [ ] **Step 2: Run selected tests to verify RED**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_schur.py -k 'nielsen or damped_step or relative_step'`

Expected: failures because solver functions are absent.

- [ ] **Step 3: Implement Nielsen update and predicted reduction**

```python
def nielsen_update(damping: float, nu: float, gain_ratio: float) -> LMUpdate:
    if math.isfinite(gain_ratio) and gain_ratio > 0.0:
        factor = max(1.0 / 3.0, 1.0 - (2.0 * gain_ratio - 1.0) ** 3)
        return LMUpdate(True, damping * factor, 2.0)
    return LMUpdate(False, damping * nu, nu * 2.0)
```

Build normal equations in fixed dimensionless coordinates using geometry scales
`(1,1,0.01)`, reference scales `(1,1,0.01,0.10,1)`, and standardized kink
parameters `eta=k/sigma_Highland`. For normalized `H`, construct the trial matrix
with:

```python
def relative_damping_diagonal(hessian: torch.Tensor) -> torch.Tensor:
    diagonal = torch.diagonal(hessian)
    positive = diagonal[diagonal > 0]
    if positive.numel() == 0:
        raise RankDeficientError("undamped Hessian has no positive diagonal")
    floor = 1e-12 * torch.median(positive)
    return torch.clamp(diagonal, min=floor)
```

Use `H_trial=H+lambda*diag(relative_damping_diagonal(H))` only for the candidate
step.

Compute gain ratio as actual objective decrease divided by the quadratic model's
predicted decrease. Reject non-finite predictions, non-positive prediction, or
invalid intersections.

- [ ] **Step 4: Implement the shared solve loop**

For each of at most 120 iterations: evaluate residual/Jacobians, assemble blocks,
form a damped copy for the trial step, solve/back-substitute, evaluate the exact
candidate objective on the unchanged discrete state, update damping, and retain
the best accepted state. Before declaring convergence, rebuild the undamped
system and compute decrement and posterior-standardized full step.

- [ ] **Step 5: Implement convergence and cell-budget summaries**

`stationarity_report()` requires decrement `<=1e-6` and maximum standardized
step `<=1e-3`. `summarize_solver_population()` returns `inconclusive` when
non-converged fraction exceeds `0.01`, invalid/paired-drop exceeds `0.01`, or
rank-deficient fraction exceeds `0.001`.

- [ ] **Step 6: Implement blind holdout refits and predictions**

With geometry fixed at the training result, fit each holdout track's local
parameters using non-target components only and the selected arm likelihood.
Evaluate target measurements once after convergence. Return paired component
keys, residuals, predictive covariance, RMSE, bias, NLL, and pull values. Refuse
any overlap between fit and predicted target masks.

- [ ] **Step 7: Add a nonlinear noiseless recovery test**

Use 24 accepted tracks, one target station, no mismatch, and known nonzero
`tx/ty/rz`. Assert geometry error is below `1e-7` in physical units, objective
decreases, undamped stationarity passes, and the single `N=0` path uses the same
solver function.

- [ ] **Step 8: Run Schur and solver tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_schur.py`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 6**

```bash
git add scripts/joint_closure_schur.py tests/test_joint_closure_schur.py
git commit -m "feat: solve joint closure with Nielsen LM"
```

---

### Task 7: Calibrate empirical ODD kappa without changing source artifacts

**Files:**
- Create: `scripts/calibrate_joint_closure_kappa.py`
- Create: `tests/test_calibrate_joint_closure_kappa.py`

**Interfaces:**
- Consumes: schema-1.1 material NPZ via `validate_helix_model.load_material_artifact`, nominal trackstates/geometry, target layers from project config.
- Produces: `calibrate_kappa() -> dict`, CLI output `kappa_calibration.json`, independent `manifest.json`.

- [ ] **Step 1: Write failing fixture and hash tests**

Create a two-track NPZ/measurement fixture with analytically known
`theta0`, downstream lever, component sigma, and expected kappa.

```python
def test_calibration_matches_known_kappa_and_records_hashes(tmp_path):
    paths = write_kappa_fixture(tmp_path)
    result = calibrate_kappa(paths.material, paths.states, paths.geometry, ((17, 8),))
    np.testing.assert_allclose(result["kappa"], paths.expected_kappa, rtol=1e-10)
    assert result["inputs"]["material_artifact"]["sha256"] == sha256(paths.material)
```

- [ ] **Step 2: Run test and verify RED**

Run: `conda run -n pytorch pytest -q tests/test_calibrate_joint_closure_kappa.py`

Expected: missing calibration module.

- [ ] **Step 3: Implement track-key join and first-downstream-target lever**

Reuse the existing material loader and ROOT measurement loader. Recompute node
and target hit arc coordinates from the same frozen nodeless reference, clip
nodes to each fit window, then pair each node with its first downstream target
hit. Do not reuse exporter `path_length_mm` as helix arc length.

- [ ] **Step 4: Implement finite summaries and frozen-hash enforcement**

Write count, p01/p05/p50/p95/p99/max for theta0, lever, component sigma, and
kappa. Formal calibration refuses input hashes different from the three hashes
in the v2 spec; `--allow-diagnostic-inputs` marks output diagnostic.

- [ ] **Step 5: Implement calibration manifest lifecycle**

Call `write_experiment_manifest()` before ROOT/NPZ loading and
`finalize_experiment_manifest()` on success or exception. Output lives in a
dedicated calibration run directory and never overwrites source artifacts.

- [ ] **Step 6: Run calibration tests**

Run: `conda run -n pytorch pytest -q tests/test_calibrate_joint_closure_kappa.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 7**

```bash
git add scripts/calibrate_joint_closure_kappa.py tests/test_calibrate_joint_closure_kappa.py
git commit -m "feat: calibrate joint closure scattering scale"
```

---

### Task 8: Implement grouped absorption and holdout-surrogate statistics

**Files:**
- Create: `scripts/joint_closure_metrics.py`
- Create: `tests/test_joint_closure_metrics.py`

**Interfaces:**
- Consumes: per-replicate arm fits, covariances, holdout residuals, mismatch signatures.
- Produces: `make_forced_absorption_problem()`, `absorption_template()`, `grouped_pull_metrics()`, `bootstrap_interval()`, `tost_equivalent()`, `proxy_validity_metrics()`, `evaluate_scientific_verdict()`.

- [ ] **Step 1: Write failing absorption and cancellation tests**

```python
def test_grouped_kink_metric_detects_absorption_hidden_by_global_mean():
    delta = np.array([1.0, 1.0, -1.0, -1.0])
    groups = np.array([0, 0, 1, 1])
    summary = grouped_pull_metrics(delta, groups)
    assert summary["global_mean"] == 0.0
    assert summary["groups"]["0"]["mean"] == 1.0
    assert summary["groups"]["1"]["mean"] == -1.0


def test_absorption_projection_recovers_known_template_scale():
    precision = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    template = torch.tensor([0.5, -0.25], dtype=torch.float64)
    assert absorption_projection(0.8 * template, template, precision) == pytest.approx(0.8)


def test_forced_absorption_problem_places_residual_in_profiled_kink_space():
    problem = small_frozen_problem(seed=31)
    forced = make_forced_absorption_problem(problem, amplitude=1.0)
    projected = forced.profiled_kink_jacobian @ forced.forced_kink
    assert torch.allclose(forced.added_residual, projected, atol=1e-12)
    assert not forced.enabled_geometry_columns[forced.forced_geometry_column]
```

- [ ] **Step 2: Write failing bootstrap, TOST, and proxy tests**

```python
def test_bootstrap_resamples_replicates_as_indivisible_blocks():
    records = synthetic_clustered_records()
    interval = bootstrap_interval(records, statistic=replicate_mean, seed=11, n_resamples=2000, level=0.95)
    assert interval.unit == "replicate"
    assert interval.n_units == len(records)


def test_tost_uses_ninety_percent_interval_at_alpha_point_zero_five():
    interval = Interval(low=-0.1, point=0.0, high=0.1, level=0.90, unit="replicate", n_units=600)
    assert tost_equivalent(interval, lower=-0.2, upper=0.2, alpha=0.05)


@pytest.mark.parametrize(
    "delta_h,delta_g,holdout_better,geometry_better",
    [(-1,-1,True,True),(-1,1,True,False),(1,-1,False,True),(1,1,False,False)],
)
def test_proxy_sign_labels(delta_h, delta_g, holdout_better, geometry_better):
    labels = classify_proxy_pair(delta_h, delta_g)
    assert labels.holdout_better is holdout_better
    assert labels.geometry_better is geometry_better
```

- [ ] **Step 3: Run metrics tests and verify RED**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_metrics.py`

Expected: missing metrics module.

- [ ] **Step 4: Implement profiled absorption template and grouped metrics**

Profile q and g with Task 5 solves, compute kink score from the residual signature,
solve `D*k_template=-score`, and evaluate projection/correlation/orthogonal
fraction. Group rows by target sensor, node index, injection sign, and phase
quadrant before replicate aggregation.

`make_forced_absorption_problem()` first profiles the selected geometry column
out of the enabled set, then sets `added_residual=J_k_profiled @ forced_kink` with
`forced_kink` alternating by node so a global mean cancels. It returns the exact
template and disabled-column index, allowing the wiring-control detector to be
tested independently of truth transport.

- [ ] **Step 5: Implement deterministic replicate bootstrap and IU/TOST**

Use NumPy `Generator(PCG64(seed))`. Resample integer replicate indices; every
statistic receives complete replicate records. Implement percentile intervals
for 90%/95%, binomial Wilson intervals, and analytic pre-run SE checks. Do not
apply a max-statistic correction to the per-DOF intersection-union decision.

- [ ] **Step 6: Implement ROC AUC and proxy probabilities without sklearn**

Use average ranks/Mann-Whitney for AUC, bootstrap whole replicates for its CI,
and return `None` plus an inconclusive reason when labels or conditional
denominators are empty.

```python
def geometry_error(fit: np.ndarray, truth: np.ndarray) -> float:
    scale = np.array([0.05, 0.05, 5e-4], dtype=np.float64)
    return float(np.sqrt(np.mean(((fit - truth) / scale) ** 2)))
```

Compute joint false assurance, conditional false assurance, sensitivity,
regression slope, and sign concordance exactly as defined in the v2 spec.

- [ ] **Step 7: Implement sensitivity and scientific verdict gates**

Require forced-control detection lower bound `>=0.99`, medium physical detection
`>=0.80`, strong `>=0.95`, and clean false-detection upper bound `<=0.05`.
Then apply every per-cell proxy threshold; a failing subgroup cannot be hidden by
the aggregate. Missing power or undefined metrics yields `inconclusive`.

- [ ] **Step 8: Run metrics tests**

Run: `conda run -n pytorch pytest -q tests/test_joint_closure_metrics.py`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 8**

```bash
git add scripts/joint_closure_metrics.py tests/test_joint_closure_metrics.py
git commit -m "feat: evaluate kink absorption and holdout proxy"
```

---

### Task 9: Build the experiment runner and auditable output contract

**Files:**
- Create: `scripts/run_joint_alignment_closure.py`
- Create: `tests/test_run_joint_alignment_closure.py`
- Modify: `scripts/path_inventory.py`
- Modify: `tests/test_path_inventory.py`

**Interfaces:**
- Consumes: all prior modules, `write_experiment_manifest`, `finalize_experiment_manifest`, project config.
- Produces: formal/diagnostic CLI, all v2 output artifacts, terminal process and scientific verdicts.

- [ ] **Step 1: Write failing CLI classification and artifact tests**

```python
def test_reduced_run_is_diagnostic_and_writes_every_artifact(tmp_path):
    result = run_cli(tmp_path, "--replicates", "2", "--tracks", "12", "--bootstrap", "100")
    assert result.returncode == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["run_class"] == "diagnostic_only"
    assert summary["scientific_verdict"] is None
    for name in REQUIRED_OUTPUTS:
        assert (tmp_path / name).exists(), name


def test_exception_finalizes_manifest_with_partial_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_cell", raising_cell)
    with pytest.raises(RuntimeError, match="injected runner failure"):
        runner.run(diagnostic_config(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["completed_cells"] == 1
```

- [ ] **Step 2: Run runner tests and verify RED**

Run: `conda run -n pytorch pytest -q tests/test_run_joint_alignment_closure.py`

Expected: missing runner module.

- [ ] **Step 3: Implement immutable cell matrix and formal-override refusal**

Build exact `CellSpec` entries for matched closure, four mismatch amplitudes,
eight kappa/layout cells, two all-free cells, forced control, and three convergence
diagnostics. `--formal` rejects every count/threshold override. Without
`--formal`, any override is accepted only as `diagnostic_only`.

- [ ] **Step 4: Implement deterministic replicate execution and row outputs**

Derive each cell/replicate seed with SHA256 of
`(master_seed,cell_id,replicate_index)` rather than Python hash. Run all three
arms from one frozen problem, write one JSONL record atomically per completed
replicate, and accumulate CSV/NPZ arrays only from paired finite rows.

- [ ] **Step 5: Implement atomic outputs and finite JSON**

Write temporary files in the run directory and rename on completion.
`summary.json` and `surrogate_validity.json` use `allow_nan=False`; represent
undefined metrics as `null` plus explicit reasons. NPZ stores spectra/conditions,
CSV stores sensor/node rows, and `preregistration.json` contains the formal
fingerprint and all constants.

- [ ] **Step 6: Implement manifest lifecycle**

Start the manifest before loading kappa calibration. Inputs include the v2 spec,
kappa calibration, and its three frozen source inputs. On normal scientific
failure, finalize process status `completed` with scientific verdict `failed`.
On exception, finalize process status `failed`, partial outputs, completed-cell
count, and exception class/message, then re-raise.

- [ ] **Step 7: Add path inventory entries**

Expose the default joint-closure experiment root and kappa-calibration path from
project configuration without hard-coding target layer suffixes outside the
artifact layout helpers. Extend JSON inventory assertions.

- [ ] **Step 8: Run runner and inventory tests**

Run: `conda run -n pytorch pytest -q tests/test_run_joint_alignment_closure.py tests/test_path_inventory.py`

Expected: all tests pass.

- [ ] **Step 9: Commit Task 9**

```bash
git add scripts/run_joint_alignment_closure.py tests/test_run_joint_alignment_closure.py scripts/path_inventory.py tests/test_path_inventory.py
git commit -m "feat: orchestrate joint identifiability experiment"
```

---

### Task 10: Verify the implementation and execute retained experiments

**Files:**
- Modify: `docs/history/ml_optimization_log.md`
- Create under ignored data: `data/experiments/joint_identifiability/Vol17_Lay8__Vol24_Lay4/formal-v2-20260901/`

**Interfaces:**
- Consumes: completed implementation and frozen v2 spec.
- Produces: kappa calibration manifest, diagnostic smoke manifest, formal ensemble manifest, results recorded in optimization history.

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
conda run -n pytorch pytest -q \
  tests/test_joint_closure_types.py \
  tests/test_joint_closure_truth.py \
  tests/test_joint_closure_models.py \
  tests/test_joint_closure_schur.py \
  tests/test_calibrate_joint_closure_kappa.py \
  tests/test_joint_closure_metrics.py \
  tests/test_run_joint_alignment_closure.py
```

Expected: all focused tests pass with no warnings or skips introduced by these
modules.

- [ ] **Step 2: Run repository verification**

Run: `make test`

Expected: quick Python suite passes. If shell path contracts changed, also run
`make test-shell` before proceeding.

- [ ] **Step 3: Run empirical kappa calibration**

Run in a fresh conda shell:

```bash
source /home/houjiahao/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
python scripts/calibrate_joint_closure_kappa.py \
  --material-artifact data/experiments/material/Vol17_Lay8__Vol24_Lay4/gate4-20260825/material_nodes.npz \
  --trackstates data/reconstruction/nominal/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root \
  --geometry data/geometry/odd_transforms.json \
  --output-dir data/experiments/joint_identifiability/Vol17_Lay8__Vol24_Lay4/kappa-v2-20260901
```

Expected: completed manifest and finite `kappa_calibration.json` with all three
expected hashes.

- [ ] **Step 4: Run a diagnostic smoke ensemble**

```bash
python scripts/run_joint_alignment_closure.py \
  --kappa-calibration data/experiments/joint_identifiability/Vol17_Lay8__Vol24_Lay4/kappa-v2-20260901/kappa_calibration.json \
  --output-dir data/experiments/joint_identifiability/Vol17_Lay8__Vol24_Lay4/smoke-v2-20260901 \
  --replicates 2 --tracks 12 --bootstrap 100
```

Expected: process status completed, `run_class=diagnostic_only`, no scientific
verdict, and every declared artifact present.

- [ ] **Step 5: Run the exact formal ensemble**

```bash
python scripts/run_joint_alignment_closure.py \
  --formal \
  --kappa-calibration data/experiments/joint_identifiability/Vol17_Lay8__Vol24_Lay4/kappa-v2-20260901/kappa_calibration.json \
  --output-dir data/experiments/joint_identifiability/Vol17_Lay8__Vol24_Lay4/formal-v2-20260901
```

Expected: all pre-registered cells complete or the manifest records the exact
failed/inconclusive reason. Do not alter thresholds or rerun with tuned constants
after reading results.

- [ ] **Step 6: Validate retained artifacts**

Check that manifest input hashes match, single/broken pairing digests agree,
positive controls have reported detection power, no subgroup is missing, JSON
contains no NaN, and the scientific verdict is consistent with individual gates.
Run an independent summary rebuild from `replicates.jsonl` and require byte-equal
canonical `surrogate_validity.json`.

- [ ] **Step 7: Record the experiment in optimization history**

Append one schema-aware iteration entry containing command, Git commit, input
hashes, cell counts, convergence/rank/drop fractions, geometry and kink pull
gates, positive-control power, kappa scan, Schur retention, proxy metrics,
verdict, output paths, and explicit limitations. Record failed or inconclusive
results without changing their thresholds.

- [ ] **Step 8: Run final verification-before-completion checks**

Run: `make ci`

Expected: Python and shell contract suites pass. Environment-sensitive ROOT I/O
remains separately reported if not run.

- [ ] **Step 9: Commit Task 10 documentation only**

```bash
git add docs/history/ml_optimization_log.md
git commit -m "docs: record broken joint proxy closure"
```

Do not add any file below `data/`.
