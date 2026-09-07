"""ACTS-independent paired diagnostic stages for helix recovery."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

try:
    from .validate_helix_model import (
        HelixEvaluation,
        NominalGeometry,
        TrackBatches,
        evaluate_helix_on_surfaces,
        fit_helices_gauss_newton,
        helix_tangent,
        load_nominal_geometry,
        load_working_geometry,
        seed_helices,
        expand_measurement_errors,
    )
except ImportError:  # Direct script execution and importlib-based tests.
    from validate_helix_model import (
        HelixEvaluation,
        NominalGeometry,
        TrackBatches,
        evaluate_helix_on_surfaces,
        fit_helices_gauss_newton,
        helix_tangent,
        load_nominal_geometry,
        load_working_geometry,
        seed_helices,
        expand_measurement_errors,
    )

try:
    from .experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from .project_config import ProjectConfig, absolute_project_path, load_project_config
except ImportError:  # Direct script execution and importlib-based tests.
    from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from project_config import ProjectConfig, absolute_project_path, load_project_config


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
    # None preserves programmatic callers that infer targets from their hits.
    # ROOT callers retain the explicit request even when selection removes a layer.
    requested_target_layers: tuple[tuple[int, int], ...] | None = None
    available_unique_events: int | None = None
    selected_unique_events: int | None = None


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

_PAIRED_ROOT_PER_STATE_BRANCHES = (
    "volume_id",
    "layer_id",
    "module_id",
    "pathLength",
    "l_x_hit",
    "l_y_hit",
    "eLOC0_smt",
    "eLOC1_smt",
    "g_x_smt",
    "g_y_smt",
    "g_z_smt",
    "px_smt",
    "py_smt",
    "pz_smt",
)


def _flatten_jagged(array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(array, axis=None))


def _load_recovery_branch(
    path: str | Path,
    geometry: NominalGeometry,
    target_layers: tuple[tuple[int, int], ...],
) -> RecoveryHitInputs:
    """Load one branch while retaining each measurement's source state ordinal."""
    path = Path(path)
    with uproot.open(path) as root_file:
        if "trackstates" not in root_file:
            raise ValueError(f"ROOT file {path} has no 'trackstates' tree.")
        tree = root_file["trackstates"]
        required = (*_PAIRED_ROOT_PER_STATE_BRANCHES, "err_x_hit", "err_y_hit", "event_nr", "track_nr")
        missing = [name for name in required if name not in tree]
        if missing:
            raise ValueError(f"ROOT trackstates is missing required branches: {missing}")
        arrays = tree.arrays(_PAIRED_ROOT_PER_STATE_BRANCHES, library="ak")
        errors_x = tree["err_x_hit"].array(library="ak")
        errors_y = tree["err_y_hit"].array(library="ak")
        events = ak.to_numpy(tree["event_nr"].array(library="ak"))
        tracks = ak.to_numpy(tree["track_nr"].array(library="ak"))

    counts = ak.to_numpy(ak.num(arrays["volume_id"], axis=1))
    if len(events) != len(counts) or len(tracks) != len(counts):
        raise ValueError("event_nr/track_nr entries do not align with jagged state entries.")
    for name in _PAIRED_ROOT_PER_STATE_BRANCHES:
        branch_counts = ak.to_numpy(ak.num(arrays[name], axis=1))
        if not np.array_equal(branch_counts, counts):
            raise ValueError(f"ROOT branch {name} does not align with per-track state counts.")

    flat = {name: _flatten_jagged(arrays[name]) for name in _PAIRED_ROOT_PER_STATE_BRANCHES}
    flat["err_x_hit"], flat["err_y_hit"] = expand_measurement_errors(
        arrays["l_x_hit"], errors_x, errors_y
    )
    flat["event_nr"] = np.repeat(events, counts)
    flat["track_nr"] = np.repeat(tracks, counts)
    flat["occurrence"] = np.concatenate(
        [np.arange(int(count), dtype=np.int64) for count in counts]
    ) if len(counts) else np.asarray([], dtype=np.int64)

    measured = np.isfinite(flat["l_x_hit"])
    if not measured.any():
        raise ValueError("No measured states with finite l_x_hit are available.")
    module = flat["module_id"].astype(np.int64)
    if (measured & (module <= 0)).any():
        raise ValueError("Measured states must have positive module IDs.")
    invalid_x = measured & (~np.isfinite(flat["err_x_hit"]) | (flat["err_x_hit"] <= 0))
    if invalid_x.any():
        raise ValueError("Measured states have invalid err_x_hit values.")
    required_finite = (
        "eLOC0_smt", "eLOC1_smt", "g_x_smt", "g_y_smt", "g_z_smt",
        "px_smt", "py_smt", "pz_smt", "pathLength",
    )
    if any((measured & ~np.isfinite(flat[name])).any() for name in required_finite):
        raise ValueError("Measured states contain non-finite SMT or path values.")

    keep = np.flatnonzero(measured)
    loc1 = (
        np.isfinite(flat["l_y_hit"][keep])
        & np.isfinite(flat["err_y_hit"][keep])
        & (flat["err_y_hit"][keep] > 0)
    )
    keys = [
        (int(flat["volume_id"][index]), int(flat["layer_id"][index]), int(module[index]))
        for index in keep
    ]
    missing_surfaces = sorted({key for key in keys if key not in geometry.id_to_idx})
    if missing_surfaces:
        raise ValueError(
            "Measured states reference surfaces absent from branch geometry; "
            f"examples: {missing_surfaces[:5]}."
        )

    def matrix(names: tuple[str, ...]) -> torch.Tensor:
        return torch.tensor(
            np.column_stack([flat[name][keep] for name in names]), dtype=torch.float64
        )

    event_tensor = torch.tensor(flat["event_nr"][keep], dtype=torch.long)
    track_tensor = torch.tensor(flat["track_nr"][keep], dtype=torch.long)
    volume_tensor = torch.tensor(flat["volume_id"][keep], dtype=torch.long)
    layer_tensor = torch.tensor(flat["layer_id"][keep], dtype=torch.long)
    module_tensor = torch.tensor(module[keep], dtype=torch.long)
    occurrence_tensor = torch.tensor(flat["occurrence"][keep], dtype=torch.long)
    measured_y = np.where(loc1, flat["l_y_hit"][keep], 0.0)
    variance_y = np.where(loc1, np.square(flat["err_y_hit"][keep]), 1.0)
    return RecoveryHitInputs(
        identities=build_hit_identities(
            event_tensor,
            track_tensor,
            volume_tensor,
            layer_tensor,
            module_tensor,
            occurrence_tensor,
        ),
        measured_local=torch.tensor(
            np.column_stack((flat["l_x_hit"][keep], measured_y)), dtype=torch.float64
        ),
        measurement_variance=torch.tensor(
            np.column_stack((np.square(flat["err_x_hit"][keep]), variance_y)),
            dtype=torch.float64,
        ),
        component_mask=torch.tensor(
            np.column_stack((np.ones(len(keep), dtype=bool), loc1)), dtype=torch.bool
        ),
        smt_local=matrix(("eLOC0_smt", "eLOC1_smt")),
        global_positions=matrix(("g_x_smt", "g_y_smt", "g_z_smt")),
        directions=matrix(("px_smt", "py_smt", "pz_smt")),
        sensor_idx=torch.tensor([geometry.id_to_idx[key] for key in keys], dtype=torch.long),
        target_mask=torch.tensor(
            [(key[0], key[1]) in target_layers for key in keys], dtype=torch.bool
        ),
        path_length=torch.tensor(flat["pathLength"][keep], dtype=torch.float64),
        requested_target_layers=target_layers,
    )


def _select_recovery_hits(inputs: RecoveryHitInputs, selected: torch.Tensor) -> RecoveryHitInputs:
    indices = selected.nonzero(as_tuple=False).flatten()
    return replace(
        inputs,
        identities=tuple(inputs.identities[index] for index in indices.tolist()),
        measured_local=inputs.measured_local[indices],
        measurement_variance=inputs.measurement_variance[indices],
        component_mask=inputs.component_mask[indices],
        smt_local=inputs.smt_local[indices],
        global_positions=inputs.global_positions[indices],
        directions=inputs.directions[indices],
        sensor_idx=inputs.sensor_idx[indices],
        target_mask=inputs.target_mask[indices],
        path_length=inputs.path_length[indices],
    )


def load_paired_recovery_inputs(
    nominal_states: str | Path,
    misaligned_states: str | Path,
    nominal_geometry: str | Path,
    misaligned_geometry: str | Path,
    *,
    target_layers: tuple[tuple[int, int], ...],
    event_partition: str = "all",
    max_events: int = 0,
    max_tracks: int = 0,
) -> tuple[RecoveryHitInputs, RecoveryHitInputs]:
    """Load, validate, and deterministically cap a paired ROOT cohort by event."""
    if event_partition != "all":
        raise ValueError(
            "hash partition selection is disabled; use event_partition='all' "
            "with max_events instead."
        )
    if max_events < 0 or max_tracks < 0:
        raise ValueError("max_events and max_tracks must be non-negative.")
    target_layers = tuple((int(volume), int(layer)) for volume, layer in target_layers)
    if not target_layers or len(set(target_layers)) != len(target_layers):
        raise ValueError("target_layers must be non-empty and unique.")
    nominal_geometry_data = load_nominal_geometry(nominal_geometry)
    misaligned_geometry_data = load_nominal_geometry(misaligned_geometry)
    nominal_keys = set(nominal_geometry_data.id_to_idx)
    misaligned_keys = set(misaligned_geometry_data.id_to_idx)
    if misaligned_keys != nominal_keys:
        if not misaligned_keys.issubset(nominal_keys):
            extras = sorted(misaligned_keys - nominal_keys)[:5]
            raise ValueError(f"Misaligned geometry contains surfaces absent from nominal: {extras}.")
        misaligned_geometry_data = load_working_geometry(misaligned_geometry, nominal_geometry)
    nominal = _canonicalize(
        _load_recovery_branch(nominal_states, nominal_geometry_data, target_layers)
    )
    misaligned = _canonicalize(
        _load_recovery_branch(
            misaligned_states, misaligned_geometry_data, target_layers
        )
    )
    _require_paired_contract(nominal, misaligned)

    events = np.asarray([identity.event_nr for identity in nominal.identities], dtype=np.int64)
    available_events = sorted(set(events.tolist()))
    retained_events = available_events[:max_events] if max_events else available_events
    selected_event_set = set(retained_events)
    selected = np.asarray([int(event) in selected_event_set for event in events], dtype=bool)
    tracks = sorted(
        {(identity.event_nr, identity.track_nr) for identity, use in zip(nominal.identities, selected) if use}
    )
    if max_tracks:
        tracks = tracks[:max_tracks]
    selected_tracks = set(tracks)
    selected &= np.asarray(
        [(identity.event_nr, identity.track_nr) in selected_tracks for identity in nominal.identities]
    )
    mask = torch.tensor(selected, dtype=torch.bool)
    selected_event_count = len(
        {identity.event_nr for identity, use in zip(nominal.identities, selected) if use}
    )
    metadata = {
        "available_unique_events": len(available_events),
        "selected_unique_events": selected_event_count,
    }
    return (
        replace(_select_recovery_hits(nominal, mask), **metadata),
        replace(_select_recovery_hits(misaligned, mask), **metadata),
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
    if nominal.requested_target_layers != misaligned.requested_target_layers:
        raise ValueError(
            "paired inputs have different requested target layers; refusing to pair states."
        )
    if nominal.identities != misaligned.identities:
        raise ValueError("paired inputs have different hit identities; refusing to shrink an arm.")
    if not torch.equal(nominal.target_mask, misaligned.target_mask):
        raise ValueError("paired inputs have different target masks; refusing to shrink an arm.")
    if not torch.equal(nominal.component_mask, misaligned.component_mask):
        raise ValueError("paired inputs have different component availability; refusing to shrink an arm.")
    if not torch.equal(nominal.measured_local, misaligned.measured_local):
        raise ValueError("paired inputs have different measurement coordinates; refusing to pair states.")
    if not torch.equal(nominal.measurement_variance, misaligned.measurement_variance):
        raise ValueError("paired inputs have different measurement uncertainties; refusing to pair states.")


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
    if not selected and nominal.requested_target_layers is None:
        raise ValueError("No paired target probes satisfy the frozen cohort requirements.")

    max_hits = max((len(indices) for indices, _probe, _span, _mode in selected), default=0)
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
    exposures: tuple[str, ...] = ("protected", "exposed"),
) -> FourArmRecovery:
    """Run requested exposure arms for both branches with the shared solvers."""
    if seed_iterations < 1 or refit_iterations < 1:
        raise ValueError("seed_iterations and refit_iterations must be at least 1.")
    if not exposures or len(set(exposures)) != len(exposures) or any(
        exposure not in {"protected", "exposed"} for exposure in exposures
    ):
        raise ValueError("exposures must be a non-empty unique selection of protected/exposed.")
    arms: dict[tuple[str, str], RecoveryArm] = {}
    if not cohort.n_rows:
        return FourArmRecovery(cohort=cohort, arms=arms)
    for branch, fit_geometry in (
        ("nominal", nominal_geometry),
        ("misaligned", misaligned_geometry),
    ):
        for exposure in exposures:
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
    n_replicates: int = BOOTSTRAP_REPLICATES,
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
            "n_requested_resamples": n_replicates,
            "n_finite_resamples": 0,
        }
    _events, inverse = np.unique(event_ids, return_inverse=True)
    n_events = int(_events.size)
    count = np.bincount(inverse, minlength=n_events).astype(np.float64)
    first_by_event = np.bincount(inverse, weights=first, minlength=n_events)
    second_by_event = np.bincount(inverse, weights=second, minlength=n_events)
    samples = np.empty(n_replicates, dtype=np.float64)
    # Chunking bounds temporary allocation when a group spans many events.
    for start in range(0, n_replicates, 64):
        stop = min(start + 64, n_replicates)
        draws = rng.integers(0, n_events, size=(stop - start, n_events))
        with np.errstate(all="ignore"):
            samples[start:stop] = statistic(
                first_by_event[draws].sum(axis=1),
                second_by_event[draws].sum(axis=1),
                count[draws].sum(axis=1),
            )
    finite = samples[np.isfinite(samples)]
    valid = finite.size == n_replicates
    return {
        "low": _json_number(np.percentile(finite, 2.5)) if valid else None,
        "high": _json_number(np.percentile(finite, 97.5)) if valid else None,
        "valid": bool(valid),
        "n_requested_resamples": n_replicates,
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
            failures = (
                int((requested & ~values[f"{branch}_{stage}_valid"].astype(bool)).sum())
                if count else 0
            )
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
    bootstrap_replicates: int,
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
                    np.array([]), np.array([]), np.array([], dtype=np.int64), lambda *_: float("nan"), rng,
                    bootstrap_replicates,
                ),
                "practical_equivalence_ratio_ci95": _cluster_bootstrap_ci(
                    np.array([]), np.array([]), np.array([], dtype=np.int64), lambda *_: float("nan"), rng,
                    bootstrap_replicates,
                ),
                "refit_branch_difference_bias_mm_ci95": _cluster_bootstrap_ci(
                    np.array([]), np.array([]), np.array([], dtype=np.int64), lambda *_: float("nan"), rng,
                    bootstrap_replicates,
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
        np.square(seed_difference), np.square(refit_difference), selected_events, ratio_stat, rng,
        bootstrap_replicates,
    )
    practical_ci = _cluster_bootstrap_ci(
        np.square(nominal_reference), np.square(misaligned_reference), selected_events, ratio_stat, rng,
        bootstrap_replicates,
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
        bootstrap_replicates,
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


def summarize_four_arm_recovery(
    report: FourArmRecovery,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    exposures: tuple[str, ...] = ("protected", "exposed"),
) -> dict[str, object]:
    """Compute paired recovery metrics, event-cluster CIs, verdicts, and rows.

    This function intentionally has no filesystem side effects.  Task 3 owns
    JSON/CSV/plot emission and manifest lifecycle; keeping this boundary pure
    lets the statistical contract be exercised without ACTS or ROOT.
    """
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")
    if not exposures or len(set(exposures)) != len(exposures) or any(
        exposure not in {"protected", "exposed"} for exposure in exposures
    ):
        raise ValueError("exposures must be a non-empty unique selection of protected/exposed.")
    cohort = report.cohort
    target_layers = cohort.nominal.requested_target_layers
    if target_layers is None:
        target_layers = {
            (identity.volume_id, identity.layer_id)
            for identity, is_target in zip(
                cohort.nominal.identities, cohort.nominal.target_mask.detach().cpu().tolist()
            )
            if is_target
        }
    if not target_layers:
        raise ValueError("Frozen recovery cohort has no requested target layers.")
    result = {
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "minimum_valid_components": MIN_VALID_COMPONENTS,
        "absolute_detector_thresholds_diagnostic_only": ABSOLUTE_DETECTOR_THRESHOLDS,
    }
    if not cohort.n_rows:
        empty = np.array([], dtype=bool)
        result["numerical_validity"] = {
            f"{branch}|{exposure}|{stage}": {
                "valid": None,
                "reason": "no frozen probe rows; stage not run",
            }
            for exposure in exposures
            for branch in ("nominal", "misaligned")
            for stage in ("seed", "refit")
        }
        result["exposures"] = {
            exposure: {
                "verdict": "inconclusive",
                "groups": {
                    f"{volume}:{layer}|{mode}|{name}": _group_statistics(
                        empty, empty, {}, component, np.array([], dtype=np.int64),
                        np.random.default_rng(bootstrap_seed), bootstrap_replicates,
                    )
                    for volume, layer in sorted(target_layers)
                    for mode in RECOVERY_PREDICTION_MODES
                    for component, name in enumerate(("LOC0", "LOC1"))
                },
            }
            for exposure in exposures
        }
        result["per_row"] = []
        return result
    columns = _probe_columns(cohort)
    row_indices = cohort.row_indices[torch.arange(cohort.n_rows), columns]
    identities = [cohort.nominal.identities[int(index)] for index in row_indices.tolist()]
    event_ids = np.asarray([identity.event_nr for identity in identities], dtype=np.int64)
    layer_ids = np.asarray([(identity.volume_id, identity.layer_id) for identity in identities])
    modes = np.asarray(cohort.probe_mode, dtype=object)
    availability = cohort.nominal.component_mask[row_indices].detach().cpu().numpy()
    numerical: dict[str, dict[str, object]] = {}
    for exposure in exposures:
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
    exposure_values = {exposure: _probe_observables(report, exposure) for exposure in exposures}
    exposure_report: dict[str, dict[str, object]] = {}
    for exposure in exposures:
        # Separate named streams make an exposure bitwise invariant to whether
        # the other exposure was requested or evaluated first.
        exposure_code = 0 if exposure == "protected" else 1
        rng = np.random.default_rng(
            np.random.SeedSequence([bootstrap_seed, exposure_code])
        )
        values = exposure_values[exposure]
        groups: dict[str, dict[str, object]] = {}
        for volume, layer in sorted(target_layers):
            for mode in RECOVERY_PREDICTION_MODES:
                in_group = (layer_ids[:, 0] == volume) & (layer_ids[:, 1] == layer) & (modes == mode)
                for component, name in enumerate(("LOC0", "LOC1")):
                    requested = in_group & availability[:, component]
                    shared = in_group & _shared_component_mask(availability[:, component], values, component)
                    groups[f"{volume}:{layer}|{mode}|{name}"] = _group_statistics(
                        requested, shared, values, component, event_ids, rng,
                        bootstrap_replicates,
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
        **result,
        "numerical_validity": numerical,
        "exposures": exposure_report,
        "per_row": _per_row_table(report, exposure_values),
    }


def _parse_cli_layers(value: str) -> tuple[tuple[int, int], ...]:
    layers: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            volume_text, layer_text = item.strip().split(":", maxsplit=1)
            layer = (int(volume_text), int(layer_text))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "target layers must use comma-separated positive volume:layer values"
            ) from error
        if layer[0] <= 0 or layer[1] <= 0:
            raise argparse.ArgumentTypeError("target layers must be positive")
        layers.append(layer)
    if not layers or len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("target layers must be non-empty and unique")
    return tuple(layers)


def _default_cli_paths(config: ProjectConfig) -> dict[str, str]:
    suffix = config.target_layer_suffix
    target = config.target_layer_dir
    return {
        "nominal_states": str(
            config.data_root / "reconstruction" / "nominal" / target / f"trackstates_kf_{suffix}.root"
        ),
        "misaligned_states": str(
            config.data_root / "reconstruction" / "misaligned" / target / f"trackstates_kf_{suffix}.root"
        ),
        "nominal_geometry": str(config.data_root / "geometry" / "odd_transforms.json"),
        "misaligned_geometry": str(
            config.data_root / "alignment" / target / f"misalignment_applied_{suffix}.json"
        ),
        "output_dir": str(config.data_root / "experiments" / "helix_recovery" / target),
    }


def build_argument_parser(config: ProjectConfig | None = None) -> argparse.ArgumentParser:
    config = config or load_project_config()
    defaults = _default_cli_paths(config)
    parser = argparse.ArgumentParser(
        description="Compare nominal and misaligned helix recovery on one strictly paired cohort."
    )
    parser.add_argument("--nominal-states", type=absolute_project_path, default=defaults["nominal_states"])
    parser.add_argument("--misaligned-states", type=absolute_project_path, default=defaults["misaligned_states"])
    parser.add_argument("--nominal-geometry", type=absolute_project_path, default=defaults["nominal_geometry"])
    parser.add_argument("--misaligned-geometry", type=absolute_project_path, default=defaults["misaligned_geometry"])
    parser.add_argument("--target-layers", type=_parse_cli_layers, default=config.target_layers)
    parser.add_argument(
        "--event-partition", choices=("all",), default="all",
        help="Legacy compatibility flag; hash partitions are disabled.",
    )
    parser.add_argument(
        "--max-events", type=int, default=20000,
        help="Keep the lowest N unique event_nr values after full pairing; 0 keeps all.",
    )
    parser.add_argument("--exposure", choices=("both", "protected", "exposed"), default="both")
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--output-dir", type=absolute_project_path, default=defaults["output_dir"])
    parser.add_argument("--seed-iters", type=int, default=200)
    parser.add_argument("--refit-iters", type=int, default=60)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--config", help="Path to a TOML project configuration override.")
    return parser


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = tuple(rows[0]) if rows else ()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _plot_paths(output_dir: Path, exposures: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for exposure in exposures:
        result[f"seed_vs_refit_paired_difference_{exposure}"] = (
            output_dir / f"seed_vs_refit_paired_difference_{exposure}.png"
        )
        result[f"refit_difference_distribution_{exposure}"] = (
            output_dir / f"refit_difference_distribution_{exposure}.png"
        )
        result[f"grouped_recovery_ratios_{exposure}"] = (
            output_dir / f"grouped_recovery_ratios_{exposure}.png"
        )
    return result


def _save_figure_atomic(figure, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".png", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        figure.savefig(temporary, dpi=140, bbox_inches="tight")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_diagnostic_plots(
    output_dir: Path,
    statistics: dict[str, object],
    exposures: tuple[str, ...],
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths = _plot_paths(output_dir, exposures)
    rows = statistics["per_row"]
    figures = []
    try:
        for exposure in exposures:
            seed_values: list[float] = []
            refit_values: list[float] = []
            for row in rows:
                for component in ("LOC0", "LOC1"):
                    if not row.get(f"shared_valid_{exposure}_{component}", False):
                        continue
                    seed_values.append(
                        row[f"{exposure}_misaligned_seed_{component}_mm"]
                        - row[f"{exposure}_nominal_seed_{component}_mm"]
                    )
                    refit_values.append(
                        row[f"{exposure}_misaligned_refit_{component}_mm"]
                        - row[f"{exposure}_nominal_refit_{component}_mm"]
                    )

            figure, axis = plt.subplots(figsize=(6.4, 4.8))
            figures.append(figure)
            axis.scatter(seed_values, refit_values, s=10, alpha=0.55)
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.set(xlabel="seed branch difference [mm]", ylabel="refit branch difference [mm]", title=f"{exposure}: seed vs refit")
            _save_figure_atomic(figure, paths[f"seed_vs_refit_paired_difference_{exposure}"])
            plt.close(figure)
            figures.remove(figure)

            figure, axis = plt.subplots(figsize=(6.4, 4.8))
            figures.append(figure)
            axis.hist(refit_values, bins=min(40, max(5, len(refit_values))), alpha=0.75)
            axis.set(xlabel="refit branch difference [mm]", ylabel="components", title=f"{exposure}: refit differences")
            _save_figure_atomic(figure, paths[f"refit_difference_distribution_{exposure}"])
            plt.close(figure)
            figures.remove(figure)

            groups = statistics["exposures"][exposure]["groups"]
            labels = [name for name, group in groups.items() if group["recovery_ratio_q"] is not None]
            ratios = [groups[name]["recovery_ratio_q"] for name in labels]
            figure, axis = plt.subplots(figsize=(max(7.0, len(labels) * 0.45), 4.8))
            figures.append(figure)
            axis.bar(range(len(labels)), ratios)
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.set_xticks(range(len(labels)), labels, rotation=75, ha="right")
            axis.set(ylabel="recovery ratio Q", title=f"{exposure}: grouped recovery")
            _save_figure_atomic(figure, paths[f"grouped_recovery_ratios_{exposure}"])
            plt.close(figure)
            figures.remove(figure)
    finally:
        for figure in figures:
            plt.close(figure)
    return paths


def _selected_exposures(value: str) -> tuple[str, ...]:
    return ("protected", "exposed") if value == "both" else (value,)


def _effective_experiment_config(args, *, device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    return {
        "nominal_states": args.nominal_states,
        "misaligned_states": args.misaligned_states,
        "nominal_geometry": args.nominal_geometry,
        "misaligned_geometry": args.misaligned_geometry,
        "target_layers": [f"{volume}:{layer}" for volume, layer in args.target_layers],
        "event_selection": "lowest_unique_event_nr",
        "max_events": args.max_events,
        "exposures": list(_selected_exposures(args.exposure)),
        "max_tracks": args.max_tracks,
        "seed_iterations": args.seed_iters,
        "refit_iterations": args.refit_iters,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "gpu_requested": bool(args.gpu),
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
    }


def _cohort_counts(nominal: RecoveryHitInputs, cohort: PairedRecoveryCohort) -> dict[str, object]:
    tracks = {(identity.event_nr, identity.track_nr) for identity in nominal.identities}
    return {
        "paired_measured_hits": len(nominal.identities),
        "available_unique_events": nominal.available_unique_events,
        "selected_unique_events": nominal.selected_unique_events,
        "paired_unique_tracks": len(tracks),
        "paired_target_hits": int(nominal.target_mask.sum()),
        "frozen_probe_rows": cohort.n_rows,
        "prediction_modes": dict(sorted(Counter(cohort.probe_mode).items())),
    }


def _output_paths(output_dir: Path, exposures: tuple[str, ...]) -> dict[str, Path]:
    return {
        "paired_track_keys": output_dir / "paired_track_keys.csv",
        "per_track_stage_metrics": output_dir / "per_track_stage_metrics.csv",
        "summary": output_dir / "helix_recovery_summary.json",
        **_plot_paths(output_dir, exposures),
    }


def _failure_reason(error: BaseException) -> str:
    detail = str(error).strip()
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _finalize_started_failure(
    manifest_path: Path,
    *,
    command: list[str],
    metrics: dict[str, object],
    error: BaseException,
) -> bool:
    """Finalize only the running manifest started by this exact invocation."""
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") != "running" or current.get("command") != command:
            return False
        finalize_experiment_manifest(
            manifest_path,
            status="failed",
            metrics={**metrics, "failure_reason": _failure_reason(error)},
        )
        return True
    except Exception as manifest_error:
        print(f"Could not finalize failed manifest: {manifest_error}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config")
    requested_config, _unknown = config_probe.parse_known_args(raw_arguments)
    config = load_project_config(requested_config.config)
    args = build_argument_parser(config).parse_args(raw_arguments)
    if args.max_events < 0 or args.max_tracks < 0:
        raise ValueError("--max-events and --max-tracks must be non-negative")
    if args.seed_iters < 1 or args.refit_iters < 1 or args.bootstrap_replicates < 1:
        raise ValueError("iteration and bootstrap budgets must be positive")

    exposures = _selected_exposures(args.exposure)
    output_dir = Path(args.output_dir)
    outputs = _output_paths(output_dir, exposures)
    inputs = {
        "nominal_trackstates": args.nominal_states,
        "misaligned_trackstates": args.misaligned_states,
        "nominal_geometry": args.nominal_geometry,
        "misaligned_geometry": args.misaligned_geometry,
    }
    manifest_path = output_dir / "manifest.json"
    resolved_outputs = {
        path.resolve(): name for name, path in {**outputs, "manifest": manifest_path}.items()
    }
    source_paths = {
        **inputs,
        **{
            f"configuration[{index}]": path
            for index, path in enumerate(config.source_paths)
        },
    }
    for name, path in source_paths.items():
        resolved = Path(path).resolve()
        if resolved in resolved_outputs:
            print(
                f"Helix recovery input/output path collision: {name} and "
                f"{resolved_outputs[resolved]} both resolve to {resolved}",
                file=sys.stderr,
            )
            return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = bool(args.gpu and torch.cuda.is_available())
    if args.gpu and not use_cuda:
        print("[warn] --gpu requested but CUDA is unavailable; using CPU float64.", file=sys.stderr)
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    effective = _effective_experiment_config(args, device=device, dtype=dtype)
    command = [sys.executable, str(Path(__file__).resolve()), *raw_arguments]
    available_metrics: dict[str, object] = {"effective_experiment_config": effective}
    try:
        manifest_path = write_experiment_manifest(
            output_dir,
            run_id=output_dir.name or "helix-recovery",
            command=command,
            config=config,
            inputs=inputs,
            outputs=outputs,
            metrics=available_metrics,
            status="running",
            allow_missing_inputs=True,
        )
    except (Exception, KeyboardInterrupt) as error:
        _finalize_started_failure(
            manifest_path,
            command=command,
            metrics=available_metrics,
            error=error,
        )
        print(f"Helix recovery failed during manifest preflight: {_failure_reason(error)}", file=sys.stderr)
        return 1

    try:
        nominal, misaligned = load_paired_recovery_inputs(
            args.nominal_states,
            args.misaligned_states,
            args.nominal_geometry,
            args.misaligned_geometry,
            target_layers=args.target_layers,
            event_partition=args.event_partition,
            max_events=args.max_events,
            max_tracks=args.max_tracks,
        )
        cohort = freeze_paired_cohort(nominal, misaligned)
        counts = _cohort_counts(nominal, cohort)
        available_metrics["cohort_counts"] = counts
        nominal_geometry = load_nominal_geometry(args.nominal_geometry)
        misaligned_geometry = load_nominal_geometry(args.misaligned_geometry)
        if set(misaligned_geometry.id_to_idx) != set(nominal_geometry.id_to_idx):
            misaligned_geometry = load_working_geometry(
                args.misaligned_geometry, args.nominal_geometry
            )
        report = run_four_arm_recovery(
            cohort,
            nominal_geometry=nominal_geometry,
            misaligned_geometry=misaligned_geometry,
            seed_iterations=args.seed_iters,
            refit_iterations=args.refit_iters,
            device=device,
            dtype=dtype,
            exposures=exposures,
        )
        statistics = summarize_four_arm_recovery(
            report,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            exposures=exposures,
        )
        verdicts = {
            exposure: statistics["exposures"][exposure]["verdict"]
            for exposure in exposures
        }
        available_metrics.update(
            {
                "cohort_counts": counts,
                "bootstrap": statistics["bootstrap"],
                "verdicts": verdicts,
                "numerical_validity": statistics["numerical_validity"],
            }
        )
        summary = {
            "schema_version": 1,
            "effective_experiment_config": effective,
            "cohort_counts": counts,
            "bootstrap": statistics["bootstrap"],
            "minimum_valid_components": statistics["minimum_valid_components"],
            "absolute_detector_thresholds_diagnostic_only": statistics[
                "absolute_detector_thresholds_diagnostic_only"
            ],
            "numerical_validity": statistics["numerical_validity"],
            "exposures": statistics["exposures"],
            "verdicts": verdicts,
        }
        keys = [
            {
                name: row[name]
                for name in (
                    "event_nr", "track_nr", "volume_id", "layer_id", "module_id",
                    "occurrence", "prediction_mode",
                )
            }
            for row in statistics["per_row"]
        ]
        _atomic_write_csv(
            outputs["paired_track_keys"],
            keys,
            ("event_nr", "track_nr", "volume_id", "layer_id", "module_id", "occurrence", "prediction_mode"),
        )
        _atomic_write_csv(outputs["per_track_stage_metrics"], statistics["per_row"])
        _atomic_write_json(outputs["summary"], summary)
        plot_outputs = write_diagnostic_plots(output_dir, statistics, exposures)
        finalize_experiment_manifest(
            manifest_path,
            status="completed",
            outputs={**outputs, **plot_outputs},
            metrics=available_metrics,
        )
    except (Exception, KeyboardInterrupt) as error:
        _finalize_started_failure(
            Path(manifest_path),
            command=command,
            metrics=available_metrics,
            error=error,
        )
        print(f"Helix recovery failed: {_failure_reason(error)}", file=sys.stderr)
        return 1

    print(f"Helix recovery outputs: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
