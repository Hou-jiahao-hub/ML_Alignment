"""ACTS-independent paired diagnostic stages for helix recovery.

This module deliberately owns no ROOT I/O, CLI, files, or statistics.  It
turns already-loaded per-hit values into one frozen paired cohort and runs the
same helix seed/refit implementation for the nominal and misaligned branches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

try:
    from .validate_helix_model import (
        HelixEvaluation,
        NominalGeometry,
        TrackBatches,
        evaluate_helix_on_surfaces,
        fit_helices_gauss_newton,
        seed_helices,
    )
except ImportError:  # Direct script execution and importlib-based tests.
    from validate_helix_model import (
        HelixEvaluation,
        NominalGeometry,
        TrackBatches,
        evaluate_helix_on_surfaces,
        fit_helices_gauss_newton,
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
