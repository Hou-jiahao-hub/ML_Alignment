"""ACTS-independent paired diagnostic stages for helix recovery.

This module deliberately owns no ROOT I/O, CLI, files, or statistics.  It
turns already-loaded per-hit values into one frozen paired cohort and runs the
same helix seed/refit implementation for the nominal and misaligned branches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

try:
    from .validate_helix_model import (
        HelixEvaluation,
        NominalGeometry,
        TrackBatches,
        evaluate_helix_on_surfaces,
        fit_helices_gauss_newton,
        helix_tangent,
        seed_helices,
    )
except ImportError:  # Direct script execution and importlib-based tests.
    from validate_helix_model import (
        HelixEvaluation,
        NominalGeometry,
        TrackBatches,
        evaluate_helix_on_surfaces,
        fit_helices_gauss_newton,
        helix_tangent,
        seed_helices,
    )


@dataclass(frozen=True, order=True)
class HitIdentity:
    """One measured state, including its stable repeated-surface occurrence."""

    event_nr: int
    track_nr: int
    volume_id: int
    layer_id: int
    module_id: int
    occurrence: int


@dataclass
class RecoveryHitInputs:
    """One branch's already-loaded measured and ACTS-SMT hit values.

    ``component_mask[:, 0]`` and ``component_mask[:, 1]`` state whether LOC0
    and LOC1 are available respectively.  ``target_mask`` identifies all
    target-layer hits; each retained target becomes one probe row.
    """

    identities: tuple[HitIdentity, ...]
    measured_local: torch.Tensor
    measurement_variance: torch.Tensor
    component_mask: torch.Tensor
    smt_local: torch.Tensor
    global_positions: torch.Tensor
    directions: torch.Tensor
    sensor_idx: torch.Tensor
    target_mask: torch.Tensor
    path_length: torch.Tensor


@dataclass
class PairedRecoveryCohort:
    """Canonical paired rows, frozen before either branch is fitted."""

    nominal: RecoveryHitInputs
    misaligned: RecoveryHitInputs
    row_indices: torch.Tensor
    hit_mask: torch.Tensor
    probe_mask: torch.Tensor
    protected_fit_mask: torch.Tensor
    exposed_fit_mask: torch.Tensor
    non_target_span_mm: torch.Tensor
    probe_mode: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return int(self.row_indices.shape[0])


@dataclass
class ACTSSMTStage:
    """The supplied ACTS smoothed local states for one frozen batch."""

    local_position: torch.Tensor
    component_mask: torch.Tensor
    hit_mask: torch.Tensor


@dataclass
class FittedStage:
    """A fitted helix stage evaluated on the nominal geometry."""

    parameters: torch.Tensor
    evaluation: HelixEvaluation
    diagnostics: dict

    @property
    def intersection_valid(self) -> torch.Tensor:
        return self.evaluation.intersection_valid


@dataclass
class RecoveryArm:
    """The three diagnostic stages for one branch/exposure combination."""

    fit_mask: torch.Tensor
    acts_smt: ACTSSMTStage
    lm_seed: FittedStage
    gauss_newton: FittedStage


@dataclass
class FourArmRecovery:
    cohort: PairedRecoveryCohort
    arms: dict[tuple[str, str], RecoveryArm]


# These statistics are deliberately local to the recovery comparison rather
# than reusing the material-model Gate 4 constants.  They describe a different
# paired question and are consumed by Task 3's presentation layer.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 6006
MIN_VALID_COMPONENTS = 200
MAX_NON_STATIONARY_ROW_FRACTION = 0.001
PRACTICAL_EQUIVALENCE_RATIO_LIMIT = 1.10
BIAS_REFERENCE_FRACTION = 0.10
ABSOLUTE_DETECTOR_THRESHOLDS = {
    "pixel_local_mm": 0.05,
    "strip_LOC0_mm": 0.10,
    "strip_LOC1_mm": 1.50,
    "direction_rad": 0.005,
}
RECOVERY_PREDICTION_MODES = (
    "interpolation",
    "forward_extrapolation",
    "backward_extrapolation",
)


def build_hit_identities(
    event_nr: torch.Tensor,
    track_nr: torch.Tensor,
    volume_id: torch.Tensor,
    layer_id: torch.Tensor,
    module_id: torch.Tensor,
    occurrence: torch.Tensor | None = None,
) -> tuple[HitIdentity, ...]:
    """Build identities without inventing a branch-local duplicate ordering.

    A repeated ``(event, track, volume, layer, module)`` requires an explicit
    occurrence from a shared state ordering established before the two branches
    are split.  Source order is not such an ordering: ROOT branches may present
    repeated states in a different order, which would silently swap a pair.
    """
    values = (event_nr, track_nr, volume_id, layer_id, module_id)
    if any(value.ndim != 1 for value in values):
        raise ValueError("Hit identity inputs must be one-dimensional tensors.")
    if len({value.shape[0] for value in values}) != 1:
        raise ValueError("Hit identity inputs must have the same length.")
    keys = [
        tuple(int(value) for value in fields)
        for fields in zip(*(value.detach().cpu().tolist() for value in values))
    ]
    if occurrence is None:
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Duplicate physical states require a shared stable occurrence; "
                "source-order occurrence assignment is forbidden."
            )
        return tuple(HitIdentity(*key, 0) for key in keys)
    if occurrence.ndim != 1 or occurrence.shape[0] != len(keys):
        raise ValueError("occurrence must be one-dimensional and match hit count.")
    if occurrence.dtype.is_floating_point or occurrence.dtype.is_complex:
        raise ValueError("occurrence must use an integral tensor dtype.")
    values = occurrence.detach().cpu().tolist()
    identities = tuple(HitIdentity(*key, int(value)) for key, value in zip(keys, values))
    if len(set(identities)) != len(identities):
        raise ValueError("Shared stable occurrences must make hit identities unique.")
    return identities


def _validate_inputs(inputs: RecoveryHitInputs) -> None:
    n_hits = len(inputs.identities)
    expected = {
        "measured_local": (n_hits, 2),
        "measurement_variance": (n_hits, 2),
        "component_mask": (n_hits, 2),
        "smt_local": (n_hits, 2),
        "global_positions": (n_hits, 3),
        "directions": (n_hits, 3),
        "sensor_idx": (n_hits,),
        "target_mask": (n_hits,),
        "path_length": (n_hits,),
    }
    for name, shape in expected.items():
        value = getattr(inputs, name)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}.")
    if inputs.component_mask.dtype is not torch.bool:
        raise ValueError("component_mask must be boolean.")
    if inputs.target_mask.dtype is not torch.bool:
        raise ValueError("target_mask must be boolean.")
    if len(set(inputs.identities)) != n_hits:
        raise ValueError("Hit identities must be unique after occurrence assignment.")
    if not torch.isfinite(inputs.global_positions).all() or not torch.isfinite(inputs.directions).all():
        raise ValueError("Global positions and directions must be finite.")
    if not torch.isfinite(inputs.path_length).all():
        raise ValueError("path_length must be finite for cohort freezing.")
    available = inputs.component_mask
    if (
        not torch.isfinite(inputs.measured_local[available]).all()
        or not torch.isfinite(inputs.measurement_variance[available]).all()
    ):
        raise ValueError("Non-finite available measurement value or variance.")
    if (inputs.measurement_variance[available] <= 0).any():
        raise ValueError("Available measurement components need positive variance.")
    if (inputs.target_mask & ~inputs.component_mask[:, 0]).any():
        raise ValueError("Every target probe requires an available LOC0 component.")


def _canonicalize(inputs: RecoveryHitInputs) -> RecoveryHitInputs:
    _validate_inputs(inputs)
    order = sorted(range(len(inputs.identities)), key=inputs.identities.__getitem__)
    index = torch.tensor(order, dtype=torch.long, device=inputs.sensor_idx.device)
    return replace(
        inputs,
        identities=tuple(inputs.identities[item] for item in order),
        measured_local=inputs.measured_local[index],
        measurement_variance=inputs.measurement_variance[index],
        component_mask=inputs.component_mask[index],
        smt_local=inputs.smt_local[index],
        global_positions=inputs.global_positions[index],
        directions=inputs.directions[index],
        sensor_idx=inputs.sensor_idx[index],
        target_mask=inputs.target_mask[index],
        path_length=inputs.path_length[index],
    )


def _require_paired_contract(
    nominal: RecoveryHitInputs, misaligned: RecoveryHitInputs
) -> None:
    if nominal.identities != misaligned.identities:
        raise ValueError("paired inputs have different hit identities; refusing to shrink an arm.")
    if not torch.equal(nominal.target_mask, misaligned.target_mask):
        raise ValueError("paired inputs have different target masks; refusing to shrink an arm.")
    if not torch.equal(nominal.component_mask, misaligned.component_mask):
        raise ValueError("paired inputs have different component availability; refusing to shrink an arm.")


def _probe_mode(probe_path: float, non_target_paths: torch.Tensor) -> str:
    lower = float(non_target_paths.min())
    upper = float(non_target_paths.max())
    if probe_path < lower:
        return "backward_extrapolation"
    if probe_path > upper:
        return "forward_extrapolation"
    return "interpolation"


def freeze_paired_cohort(
    nominal: RecoveryHitInputs,
    misaligned: RecoveryHitInputs,
    *,
    min_non_target_hits: int = 8,
    min_non_target_span_mm: float = 300.0,
) -> PairedRecoveryCohort:
    """Freeze target probes before fitting either branch.

    A row includes every hit from its physical track, but the protected fit mask
    uses only non-target LOC0 measurements.  The exposed mask adds exactly that
    row's probe, so changing exposure never changes the selected cohort.
    """
    if min_non_target_hits < 1:
        raise ValueError("min_non_target_hits must be at least 1.")
    if min_non_target_span_mm < 0:
        raise ValueError("min_non_target_span_mm must be non-negative.")
    nominal = _canonicalize(nominal)
    misaligned = _canonicalize(misaligned)
    _require_paired_contract(nominal, misaligned)

    track_rows: dict[tuple[int, int], list[int]] = {}
    for index, identity in enumerate(nominal.identities):
        track_rows.setdefault((identity.event_nr, identity.track_nr), []).append(index)
    selected: list[tuple[list[int], int, float, str]] = []
    for key in sorted(track_rows):
        indices = track_rows[key]
        indices.sort(key=lambda item: (float(nominal.path_length[item]), nominal.identities[item]))
        non_target = [
            item
            for item in indices
            if not nominal.target_mask[item] and nominal.component_mask[item, 0]
        ]
        if len(non_target) < min_non_target_hits:
            continue
        non_target_paths = nominal.path_length[torch.tensor(non_target, dtype=torch.long)]
        span = float(non_target_paths.max() - non_target_paths.min())
        if span < min_non_target_span_mm:
            continue
        for probe in indices:
            if not nominal.target_mask[probe] or not nominal.component_mask[probe, 0]:
                continue
            selected.append(
                (
                    indices,
                    probe,
                    span,
                    _probe_mode(float(nominal.path_length[probe]), non_target_paths),
                )
            )
    if not selected:
        raise ValueError("No paired target probes satisfy the frozen cohort requirements.")

    max_hits = max(len(indices) for indices, _probe, _span, _mode in selected)
    n_rows = len(selected)
    row_indices = torch.zeros((n_rows, max_hits), dtype=torch.long)
    hit_mask = torch.zeros((n_rows, max_hits), dtype=torch.bool)
    probe_mask = torch.zeros((n_rows, max_hits), dtype=torch.bool)
    protected = torch.zeros((n_rows, max_hits), dtype=torch.bool)
    exposed = torch.zeros((n_rows, max_hits), dtype=torch.bool)
    span = torch.empty(n_rows, dtype=nominal.path_length.dtype)
    modes = []
    for row, (indices, probe, value, mode) in enumerate(selected):
        count = len(indices)
        index = torch.tensor(indices, dtype=torch.long)
        row_indices[row, :count] = index
        hit_mask[row, :count] = True
        probe_column = indices.index(probe)
        probe_mask[row, probe_column] = True
        available_loc0 = nominal.component_mask[index, 0]
        is_target = nominal.target_mask[index]
        protected[row, :count] = available_loc0 & ~is_target
        exposed[row, :count] = protected[row, :count]
        exposed[row, probe_column] = True
        span[row] = value
        modes.append(mode)
    return PairedRecoveryCohort(
        nominal=nominal,
        misaligned=misaligned,
        row_indices=row_indices,
        hit_mask=hit_mask,
        probe_mask=probe_mask,
        protected_fit_mask=protected,
        exposed_fit_mask=exposed,
        non_target_span_mm=span,
        probe_mode=tuple(modes),
    )


def _batches_for(
    cohort: PairedRecoveryCohort,
    branch: str,
    exposure: str,
    *,
    sensor_lookup_branch: str = "nominal",
) -> TrackBatches:
    if branch not in {"nominal", "misaligned"}:
        raise ValueError("branch must be 'nominal' or 'misaligned'.")
    if exposure not in {"protected", "exposed"}:
        raise ValueError("exposure must be 'protected' or 'exposed'.")
    if sensor_lookup_branch not in {"nominal", "misaligned"}:
        raise ValueError("sensor_lookup_branch must be 'nominal' or 'misaligned'.")
    inputs = getattr(cohort, branch)
    index = cohort.row_indices
    active = cohort.hit_mask
    safe_index = index.clamp_min(0)
    component = inputs.component_mask[safe_index]
    measured = inputs.measured_local[safe_index]
    variance = inputs.measurement_variance[safe_index]
    smt_local = inputs.smt_local[safe_index]
    g_smt = inputs.global_positions[safe_index]
    p_smt = inputs.directions[safe_index]
    sensor_idx = getattr(cohort, sensor_lookup_branch).sensor_idx[safe_index]
    loc1_valid = component[..., 1] & active
    component = component & active.unsqueeze(-1)
    measured = torch.where(component, measured, torch.zeros_like(measured))
    variance = torch.where(component, variance, torch.ones_like(variance))
    identities = cohort.nominal.identities
    event_track = torch.empty((cohort.n_rows, 2), dtype=torch.long)
    volume = torch.zeros_like(index)
    layer = torch.zeros_like(index)
    module = torch.zeros_like(index)
    for row in range(cohort.n_rows):
        for column in torch.nonzero(active[row], as_tuple=False).flatten().tolist():
            identity = identities[int(index[row, column])]
            volume[row, column] = identity.volume_id
            layer[row, column] = identity.layer_id
            module[row, column] = identity.module_id
            event_track[row] = torch.tensor((identity.event_nr, identity.track_nr))
    fit_mask = cohort.protected_fit_mask if exposure == "protected" else cohort.exposed_fit_mask
    return TrackBatches(
        measured_local=measured,
        measurement_variance=variance,
        loc1_valid=loc1_valid,
        smt_local=smt_local,
        sensor_idx=sensor_idx,
        hit_mask=active,
        fit_mask=fit_mask,
        holdout_mask=cohort.probe_mask,
        g_smt=g_smt,
        p_smt=p_smt,
        volume_id=volume,
        layer_id=layer,
        module_id=module,
        event_track_ids=event_track,
        fit_span=cohort.non_target_span_mm,
        prediction_mode=cohort.probe_mode,
        context_hits_before=torch.zeros(cohort.n_rows, dtype=torch.long),
        context_hits_after=torch.zeros(cohort.n_rows, dtype=torch.long),
        nearest_context_distance=torch.zeros_like(cohort.non_target_span_mm),
        farthest_context_distance=torch.zeros_like(cohort.non_target_span_mm),
    )


def evaluate_fitted_stage_on_nominal_surfaces(
    parameters: torch.Tensor,
    cohort: PairedRecoveryCohort,
    nominal_geometry: NominalGeometry,
    *,
    branch: str = "nominal",
    exposure: str = "protected",
) -> HelixEvaluation:
    """Evaluate every requested hit on nominal surfaces without filtering invalids."""
    return evaluate_helix_on_surfaces(
        parameters,
        _batches_for(
            cohort, branch, exposure, sensor_lookup_branch="nominal"
        ),
        nominal_geometry,
    )


def run_four_arm_recovery(
    cohort: PairedRecoveryCohort,
    *,
    nominal_geometry: NominalGeometry,
    misaligned_geometry: NominalGeometry,
    seed_iterations: int = 200,
    refit_iterations: int = 60,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> FourArmRecovery:
    """Run ACTS-SMT, the shared LM seed, and shared GN refit in four arms."""
    if seed_iterations < 1 or refit_iterations < 1:
        raise ValueError("seed_iterations and refit_iterations must be at least 1.")
    arms: dict[tuple[str, str], RecoveryArm] = {}
    for branch, fit_geometry in (
        ("nominal", nominal_geometry),
        ("misaligned", misaligned_geometry),
    ):
        for exposure in ("protected", "exposed"):
            batches = _batches_for(
                cohort, branch, exposure, sensor_lookup_branch=branch
            )
            evaluation_batches = _batches_for(
                cohort, branch, exposure, sensor_lookup_branch="nominal"
            )
            source = getattr(cohort, branch)
            component_mask = source.component_mask[
                cohort.row_indices.clamp_min(0)
            ] & cohort.hit_mask.unsqueeze(-1)
            acts_smt = ACTSSMTStage(
                local_position=batches.smt_local.clone(),
                component_mask=component_mask,
                hit_mask=batches.hit_mask.clone(),
            )
            seeded, seed_diagnostics = seed_helices(
                batches,
                n_iter=seed_iterations,
                device=device,
                dtype=dtype,
                return_diagnostics=True,
            )
            seed_evaluation = evaluate_helix_on_surfaces(
                seeded, evaluation_batches, nominal_geometry
            )
            fitted, refit_diagnostics = fit_helices_gauss_newton(
                batches,
                fit_geometry,
                seeded,
                max_iters=refit_iterations,
                device=device,
                dtype=dtype,
            )
            fitted_evaluation = evaluate_helix_on_surfaces(
                fitted, evaluation_batches, nominal_geometry
            )
            arms[(branch, exposure)] = RecoveryArm(
                fit_mask=batches.fit_mask.clone(),
                acts_smt=acts_smt,
                lm_seed=FittedStage(seeded, seed_evaluation, seed_diagnostics),
                gauss_newton=FittedStage(fitted, fitted_evaluation, refit_diagnostics),
            )
    return FourArmRecovery(cohort=cohort, arms=arms)


def _json_number(value: float | int | np.number | None) -> float | int | None:
    """Return a strict-JSON scalar, preserving undefined values as ``None``."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def _ratio(numerator: float | np.ndarray, denominator: float | np.ndarray):
    """Divide only by finite positive denominators (scalar or vectorized)."""
    numerator_array, denominator_array = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator_array.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator_array)
        & np.isfinite(denominator_array)
        & (denominator_array > 0.0)
    )
    np.divide(numerator_array, denominator_array, out=result, where=valid)
    return float(result) if result.ndim == 0 else result


def _cluster_bootstrap_ci(
    first: np.ndarray,
    second: np.ndarray,
    event_ids: np.ndarray,
    statistic,
    rng: np.random.Generator,
) -> dict[str, float | None]:
    """Percentile CI from whole-event resampling, never global NumPy state.

    ``first`` and ``second`` are aggregated per event before each draw.  This
    is algebraically identical to repeating every selected probe row from a
    sampled event, while avoiding a large object array for real cohorts.
    """
    if not (first.ndim == second.ndim == event_ids.ndim == 1):
        raise ValueError("Bootstrap inputs must be one-dimensional.")
    if not (len(first) == len(second) == len(event_ids)):
        raise ValueError("Bootstrap inputs must have matching lengths.")
    if not len(first):
        return {
            "low": None,
            "high": None,
            "valid": False,
            "n_requested_resamples": BOOTSTRAP_REPLICATES,
            "n_finite_resamples": 0,
        }
    _events, inverse = np.unique(event_ids, return_inverse=True)
    n_events = int(_events.size)
    count = np.bincount(inverse, minlength=n_events).astype(np.float64)
    first_by_event = np.bincount(inverse, weights=first, minlength=n_events)
    second_by_event = np.bincount(inverse, weights=second, minlength=n_events)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Chunking bounds temporary allocation when a group spans many events.
    for start in range(0, BOOTSTRAP_REPLICATES, 64):
        stop = min(start + 64, BOOTSTRAP_REPLICATES)
        draws = rng.integers(0, n_events, size=(stop - start, n_events))
        with np.errstate(all="ignore"):
            samples[start:stop] = statistic(
                first_by_event[draws].sum(axis=1),
                second_by_event[draws].sum(axis=1),
                count[draws].sum(axis=1),
            )
    finite = samples[np.isfinite(samples)]
    valid = finite.size == BOOTSTRAP_REPLICATES
    return {
        "low": _json_number(np.percentile(finite, 2.5)) if valid else None,
        "high": _json_number(np.percentile(finite, 97.5)) if valid else None,
        "valid": bool(valid),
        "n_requested_resamples": BOOTSTRAP_REPLICATES,
        "n_finite_resamples": int(finite.size),
    }


def _stationarity_valid(diagnostics: dict, *, stage: str) -> dict[str, object]:
    """Read the seed/refit diagnostic shapes emitted by the shared solvers."""
    if not isinstance(diagnostics, dict):
        return {"valid": False, "reason": "missing stationarity observability"}
    report = diagnostics if stage == "seed" else diagnostics.get("stationarity")
    # Accept the direct shape too: it is useful for a stage adapter but not an
    # excuse to accept missing fields from Task 1's actual nested refit report.
    if stage == "refit" and report is None:
        report = diagnostics if "non_stationary_row_fraction" in diagnostics else None
    if not isinstance(report, dict):
        return {"valid": False, "reason": "missing stationarity observability"}
    fraction = report.get("non_stationary_row_fraction")
    non_finite = report.get("n_non_finite")
    if not isinstance(fraction, (float, int)) or not isinstance(non_finite, (float, int)):
        return {"valid": False, "reason": "missing stationarity observability"}
    if not np.isfinite(fraction) or not np.isfinite(non_finite):
        return {"valid": False, "reason": "non-finite stationarity observability"}
    valid = fraction <= MAX_NON_STATIONARY_ROW_FRACTION and non_finite == 0
    return {
        "valid": bool(valid),
        "reason": None if valid else "stationarity gate failed",
        "non_stationary_row_fraction": float(fraction),
        "n_non_finite": int(non_finite),
    }


def _probe_columns(cohort: PairedRecoveryCohort) -> torch.Tensor:
    if cohort.probe_mask.shape != cohort.hit_mask.shape or not torch.all(
        cohort.probe_mask.sum(dim=1) == 1
    ):
        raise ValueError("Every recovery row must contain exactly one frozen probe.")
    return cohort.probe_mask.to(torch.int64).argmax(dim=1)


def _probe_observables(
    report: FourArmRecovery, exposure: str) -> dict[str, np.ndarray]:
    """Extract every value needed by every metric before a shared mask is made."""
    cohort = report.cohort
    columns = _probe_columns(cohort)
    rows = torch.arange(cohort.n_rows, device=columns.device)
    values: dict[str, np.ndarray] = {}
    for branch in ("nominal", "misaligned"):
        arm = report.arms.get((branch, exposure))
        if arm is None:
            raise ValueError(f"Missing {branch}|{exposure} recovery arm.")
        for stage_name, stage in (("seed", arm.lm_seed), ("refit", arm.gauss_newton)):
            evaluation = stage.evaluation
            if tuple(evaluation.local_position.shape[:2]) != tuple(cohort.probe_mask.shape):
                raise ValueError("Fitted-stage evaluation shape does not match frozen cohort rows.")
            if tuple(evaluation.intersection_valid.shape) != tuple(cohort.probe_mask.shape):
                raise ValueError("Fitted-stage intersection mask does not match frozen cohort rows.")
            if stage.parameters.ndim != 2 or stage.parameters.shape != (cohort.n_rows, 6):
                raise ValueError("Fitted-stage parameters must have shape [rows, 6].")
            values[f"{branch}_{stage_name}_local"] = (
                evaluation.local_position[rows, columns].detach().cpu().numpy()
            )
            values[f"{branch}_{stage_name}_valid"] = (
                evaluation.intersection_valid[rows, columns].detach().cpu().numpy()
            )
            values[f"{branch}_{stage_name}_angle"] = (
                evaluation.tangent_angle[rows, columns].detach().cpu().numpy()
            )
            tracks = stage.parameters[:, None, :].expand(
                -1, evaluation.path_length.shape[1], -1
            )
            tangent = helix_tangent(tracks, evaluation.path_length)
            values[f"{branch}_{stage_name}_tangent"] = (
                tangent[rows, columns].detach().cpu().numpy()
            )
            values[f"{branch}_{stage_name}_rho"] = (
                stage.parameters[:, 5].detach().cpu().numpy()
            )
        smt = arm.acts_smt
        if tuple(smt.local_position.shape[:2]) != tuple(cohort.probe_mask.shape):
            raise ValueError("ACTS-SMT stage shape does not match frozen cohort rows.")
        values[f"{branch}_acts_local"] = smt.local_position[rows, columns].detach().cpu().numpy()
    return values


def _shared_component_mask(
    component_available: np.ndarray, values: dict[str, np.ndarray], component: int
) -> np.ndarray:
    """The one mask shared by every seed/refit statistic in one component group."""
    mask = component_available.copy()
    for branch in ("nominal", "misaligned"):
        for stage in ("seed", "refit"):
            mask &= values[f"{branch}_{stage}_valid"].astype(bool)
            mask &= np.isfinite(values[f"{branch}_{stage}_local"][:, component])
            mask &= np.isfinite(values[f"{branch}_{stage}_angle"])
            mask &= np.isfinite(values[f"{branch}_{stage}_tangent"]).all(axis=1)
            mask &= np.linalg.norm(values[f"{branch}_{stage}_tangent"], axis=1) > 0.0
            mask &= np.isfinite(values[f"{branch}_{stage}_rho"])
        mask &= np.isfinite(values[f"{branch}_acts_local"][:, component])
    return mask


def _intersection_failures(
    requested: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, dict[str, float | int]]:
    result = {}
    count = int(requested.sum())
    for branch in ("nominal", "misaligned"):
        for stage in ("seed", "refit"):
            valid = values[f"{branch}_{stage}_valid"].astype(bool)
            failures = int((requested & ~valid).sum())
            result[f"{branch}_{stage}"] = {
                "count": failures,
                "rate": float(failures / count) if count else None,
            }
    return result


def _tangent_angles(values: dict[str, np.ndarray], selection: np.ndarray) -> np.ndarray:
    """Geometric angle between the two fitted helices on nominal surfaces."""
    nominal = values["nominal_refit_tangent"][selection]
    misaligned = values["misaligned_refit_tangent"][selection]
    nominal_unit = nominal / np.linalg.norm(nominal, axis=1, keepdims=True)
    misaligned_unit = misaligned / np.linalg.norm(misaligned, axis=1, keepdims=True)
    cosine = np.clip(np.sum(nominal_unit * misaligned_unit, axis=1), -1.0, 1.0)
    return np.arccos(cosine)


def _group_statistics(
    requested: np.ndarray,
    shared: np.ndarray,
    values: dict[str, np.ndarray],
    component: int,
    event_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    selection = np.flatnonzero(shared)
    entry: dict[str, object] = {
        "requested_count": int(requested.sum()),
        "valid_count": int(selection.size),
        "intersection_failures": _intersection_failures(requested, values),
    }
    if not selection.size:
        entry.update(
            {
                "seed_branch_difference_rmse_mm": None,
                "refit_branch_difference_rmse_mm": None,
                "recovery_ratio_q": None,
                "recovery_fraction": None,
                "refit_branch_difference_bias_mm": None,
                "nominal_refit_to_nominal_ACTS_rmse_mm": None,
                "misaligned_refit_to_nominal_ACTS_rmse_mm": None,
                "practical_equivalence_ratio": None,
                "direction_angle_branch_difference_rms_rad": None,
                "direction_angle_branch_difference_bias_rad": None,
                "rho_branch_difference_rms_per_mm": None,
                "rho_branch_difference_bias_per_mm": None,
                "recovery_ratio_q_ci95": _cluster_bootstrap_ci(
                    np.array([]), np.array([]), np.array([], dtype=np.int64), lambda *_: float("nan"), rng
                ),
                "practical_equivalence_ratio_ci95": _cluster_bootstrap_ci(
                    np.array([]), np.array([]), np.array([], dtype=np.int64), lambda *_: float("nan"), rng
                ),
                "refit_branch_difference_bias_mm_ci95": _cluster_bootstrap_ci(
                    np.array([]), np.array([]), np.array([], dtype=np.int64), lambda *_: float("nan"), rng
                ),
                "verdict": "inconclusive",
                "gates": {"restoration": False, "practical_equivalence": False, "bias": False},
            }
        )
        return entry
    nominal_seed = values["nominal_seed_local"][selection, component]
    misaligned_seed = values["misaligned_seed_local"][selection, component]
    nominal_refit = values["nominal_refit_local"][selection, component]
    misaligned_refit = values["misaligned_refit_local"][selection, component]
    acts = values["nominal_acts_local"][selection, component]
    seed_difference = misaligned_seed - nominal_seed
    refit_difference = misaligned_refit - nominal_refit
    nominal_reference = nominal_refit - acts
    misaligned_reference = misaligned_refit - acts
    seed_rmse = _rmse(seed_difference)
    refit_rmse = _rmse(refit_difference)
    nominal_reference_rmse = _rmse(nominal_reference)
    misaligned_reference_rmse = _rmse(misaligned_reference)
    q = _ratio(refit_rmse, seed_rmse)
    practical = _ratio(misaligned_reference_rmse, nominal_reference_rmse)
    selected_events = event_ids[selection]

    def ratio_stat(first_sum, second_sum, count):
        first_rmse = np.sqrt(first_sum / count)
        second_rmse = np.sqrt(second_sum / count)
        return _ratio(second_rmse, first_rmse)

    q_ci = _cluster_bootstrap_ci(
        np.square(seed_difference), np.square(refit_difference), selected_events, ratio_stat, rng
    )
    practical_ci = _cluster_bootstrap_ci(
        np.square(nominal_reference), np.square(misaligned_reference), selected_events, ratio_stat, rng
    )
    bias_ci = _cluster_bootstrap_ci(
        refit_difference, np.zeros_like(refit_difference), selected_events,
        lambda total, _unused, count: np.divide(
            total,
            count,
            out=np.full(np.broadcast(total, count).shape, np.nan, dtype=np.float64),
            where=np.asarray(count) > 0,
        ),
        rng,
    )
    entry.update(
        {
            "seed_branch_difference_rmse_mm": _json_number(seed_rmse),
            "refit_branch_difference_rmse_mm": _json_number(refit_rmse),
            "recovery_ratio_q": _json_number(q),
            "recovery_fraction": _json_number(1.0 - q) if np.isfinite(q) else None,
            "refit_branch_difference_bias_mm": _json_number(np.mean(refit_difference)),
            "nominal_refit_to_nominal_ACTS_rmse_mm": _json_number(nominal_reference_rmse),
            "misaligned_refit_to_nominal_ACTS_rmse_mm": _json_number(misaligned_reference_rmse),
            "practical_equivalence_ratio": _json_number(practical),
            "direction_angle_branch_difference_rms_rad": _json_number(
                _rmse(_tangent_angles(values, selection))
            ),
            "direction_angle_branch_difference_bias_rad": _json_number(
                np.mean(_tangent_angles(values, selection))
            ),
            "rho_branch_difference_rms_per_mm": _json_number(
                _rmse(values["misaligned_refit_rho"][selection] - values["nominal_refit_rho"][selection])
            ),
            "rho_branch_difference_bias_per_mm": _json_number(
                np.mean(values["misaligned_refit_rho"][selection] - values["nominal_refit_rho"][selection])
            ),
            "recovery_ratio_q_ci95": q_ci,
            "practical_equivalence_ratio_ci95": practical_ci,
            "refit_branch_difference_bias_mm_ci95": bias_ci,
        }
    )
    restoration = q_ci["valid"] and q_ci["high"] is not None and q_ci["high"] < 1.0
    equivalence = (
        practical_ci["valid"]
        and practical_ci["high"] is not None
        and practical_ci["high"] <= PRACTICAL_EQUIVALENCE_RATIO_LIMIT
    )
    bias_limit = BIAS_REFERENCE_FRACTION * nominal_reference_rmse
    bias = (
        np.isfinite(bias_limit)
        and bias_ci["valid"]
        and bias_ci["low"] is not None
        and bias_ci["high"] is not None
        and bias_ci["low"] >= -bias_limit
        and bias_ci["high"] <= bias_limit
    )
    gates = {
        "restoration": bool(restoration),
        "practical_equivalence": bool(equivalence),
        "bias": bool(bias),
    }
    entry["gates"] = gates
    entry["verdict"] = (
        "inconclusive" if selection.size < MIN_VALID_COMPONENTS else "passed" if all(gates.values()) else "failed"
    )
    return entry


def _per_row_table(report: FourArmRecovery, values_by_exposure: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    cohort = report.cohort
    columns = _probe_columns(cohort).detach().cpu().tolist()
    rows: list[dict] = []
    for row, column in enumerate(columns):
        identity = cohort.nominal.identities[int(cohort.row_indices[row, column])]
        record: dict[str, object] = {
            "event_nr": identity.event_nr,
            "track_nr": identity.track_nr,
            "volume_id": identity.volume_id,
            "layer_id": identity.layer_id,
            "module_id": identity.module_id,
            "occurrence": identity.occurrence,
            "prediction_mode": cohort.probe_mode[row],
        }
        for component, name in enumerate(("LOC0", "LOC1")):
            available = bool(cohort.nominal.component_mask[cohort.row_indices[row, column], component])
            record[f"requested_{name}"] = available
            for exposure, values in values_by_exposure.items():
                shared = _shared_component_mask(
                    np.array([available]),
                    {key: value[row : row + 1] for key, value in values.items()},
                    component,
                )[0]
                record[f"shared_valid_{exposure}_{name}"] = bool(shared)
                if exposure == "protected":
                    record[f"shared_valid_{name}"] = bool(shared)
                # Protected is the canonical per-row view; exposed is retained
                # separately because its fit mask legitimately changes estimates.
                for branch in ("nominal", "misaligned"):
                    for stage in ("seed", "refit"):
                        record[f"{exposure}_{branch}_{stage}_{name}_mm"] = _json_number(
                            values[f"{branch}_{stage}_local"][row, component]
                        )
        rows.append(record)
    return rows


def summarize_four_arm_recovery(report: FourArmRecovery) -> dict[str, object]:
    """Compute paired recovery metrics, event-cluster CIs, verdicts, and rows.

    This function intentionally has no filesystem side effects.  Task 3 owns
    JSON/CSV/plot emission and manifest lifecycle; keeping this boundary pure
    lets the statistical contract be exercised without ACTS or ROOT.
    """
    cohort = report.cohort
    columns = _probe_columns(cohort)
    row_indices = cohort.row_indices[torch.arange(cohort.n_rows), columns]
    identities = [cohort.nominal.identities[int(index)] for index in row_indices.tolist()]
    event_ids = np.asarray([identity.event_nr for identity in identities], dtype=np.int64)
    layer_ids = np.asarray([(identity.volume_id, identity.layer_id) for identity in identities])
    modes = np.asarray(cohort.probe_mode, dtype=object)
    availability = cohort.nominal.component_mask[row_indices].detach().cpu().numpy()
    numerical: dict[str, dict[str, object]] = {}
    for exposure in ("protected", "exposed"):
        for branch in ("nominal", "misaligned"):
            arm = report.arms.get((branch, exposure))
            if arm is None:
                raise ValueError(f"Missing {branch}|{exposure} recovery arm.")
            numerical[f"{branch}|{exposure}|seed"] = _stationarity_valid(
                arm.lm_seed.diagnostics, stage="seed"
            )
            numerical[f"{branch}|{exposure}|refit"] = _stationarity_valid(
                arm.gauss_newton.diagnostics, stage="refit"
            )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    exposure_values = {exposure: _probe_observables(report, exposure) for exposure in ("protected", "exposed")}
    exposure_report: dict[str, dict[str, object]] = {}
    target_layers = {
        (identity.volume_id, identity.layer_id)
        for identity, is_target in zip(
            cohort.nominal.identities, cohort.nominal.target_mask.detach().cpu().tolist()
        )
        if is_target
    }
    if not target_layers:
        raise ValueError("Frozen recovery cohort has no target-layer identities.")
    for exposure in ("protected", "exposed"):
        values = exposure_values[exposure]
        groups: dict[str, dict[str, object]] = {}
        for volume, layer in sorted(target_layers):
            for mode in RECOVERY_PREDICTION_MODES:
                in_group = (layer_ids[:, 0] == volume) & (layer_ids[:, 1] == layer) & (modes == mode)
                for component, name in enumerate(("LOC0", "LOC1")):
                    requested = in_group & availability[:, component]
                    shared = in_group & _shared_component_mask(availability[:, component], values, component)
                    groups[f"{volume}:{layer}|{mode}|{name}"] = _group_statistics(
                        requested, shared, values, component, event_ids, rng
                    )
        numerical_ok = all(
            bool(numerical[f"{branch}|{exposure}|{stage}"]["valid"])
            for branch in ("nominal", "misaligned")
            for stage in ("seed", "refit")
        )
        group_verdicts = [entry["verdict"] for entry in groups.values()]
        verdict = (
            "failed"
            if not numerical_ok or "failed" in group_verdicts
            else "inconclusive"
            if "inconclusive" in group_verdicts or not group_verdicts
            else "passed"
        )
        exposure_report[exposure] = {"verdict": verdict, "groups": groups}
    return {
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "minimum_valid_components": MIN_VALID_COMPONENTS,
        "absolute_detector_thresholds_diagnostic_only": ABSOLUTE_DETECTOR_THRESHOLDS,
        "numerical_validity": numerical,
        "exposures": exposure_report,
        "per_row": _per_row_table(report, exposure_values),
    }
