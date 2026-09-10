"""Build the ACTS-free per-track material-node artifact for broken-helix fits.

The tracking core must stay importable without ACTS, so this module consumes
plain ROOT output written by ``alignment_odd.py material`` and emits a
schema-versioned NPZ plus a manifest.  Everything here is fail-closed: a
silently mis-associated material track would corrupt every scattering prior
downstream without any visible error.

Three ROOT tables are joined:

``material_kf``      per-track material interactions, keyed only by ``event_id``
                     because ``RootMaterialTrackWriter.cpp:87`` drops the track
                     index and iterates an ``unordered_map``.
``propsummary_kf``   the missing index bridge: it carries ``(event_nr, track_nr)``
                     together with the five start bound parameters that the same
                     propagation fed to the material recorder.
``tracksummary_kf``  the reconstruction-side track identity used everywhere else
                     in this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:  # Direct script execution and package import must both work.
    from .helix_model import highland_kink_sigma
    from .experiment_manifest import (
        finalize_experiment_manifest,
        write_experiment_manifest,
    )
    from .project_config import absolute_project_path, add_config_argument, load_project_config
except ImportError:  # pragma: no cover - exercised by direct execution
    from helix_model import highland_kink_sigma
    from experiment_manifest import (
        finalize_experiment_manifest,
        write_experiment_manifest,
    )
    from project_config import absolute_project_path, add_config_argument, load_project_config


# ---------------------------------------------------------------------------
# Gate 1 pre-registered contract.  Changing any value requires a schema bump.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.1"

#: Aggregated nodes thinner than this carry no usable Highland scale.
X_OVER_X0_MIN = 1e-4
#: Volume-material nodes farther apart than this are never merged.
VOLUME_MERGE_WINDOW_MM = 5.0
#: Per-track ceiling on the material budget discarded as thin nodes.
DROPPED_MATERIAL_FRACTION_MAX = 0.01
#: Per-track ceiling on how much aggregation may move sum(theta0^2).
AGGREGATION_THETA2_TOL = 0.02
#: sum(mat_step_length / mat_X0) against the ROOT ``t_X0`` branch.
T_X0_CLOSURE_RTOL = 1e-4
#: Chord-sum arc length against (s - s[0]) * sqrt(1 + tan^2 lambda).
PATH_CLOSURE_RTOL = 1e-3

DIRECTION_NORM_ATOL = 1e-5
#: float32 round-trip slack for the Join A bound-parameter verification.
BOUND_PARAMETER_RTOL = 1e-5
BOUND_PARAMETER_ATOL = 1e-6
#: Join B matches on (phi, eta); tracks within an event differ by far more.
DIRECTION_JOIN_ATOL = 1e-4
PERIGEE_HYPOTHESIS_ATOL_MM = 1e-3
#: Gate 0 (2026-08-25, 1000 events) measured a pass rate of exactly 1.0, so the
#: origin-perigee convention is promoted from diagnostic to hard check.
PERIGEE_HYPOTHESIS_REQUIRED = True

ARTIFACT_FIELDS = (
    "event_nr",
    "track_nr",
    "q_over_p_gev_inv",
    "node_offsets",
    "path_length_mm",
    "position_mm",
    "direction",
    "x_over_x0",
    "surface_geometry_id",
    "volume_geometry_id",
)

MATERIAL_TREE = "material-tracks"
PROPSUMMARY_TREE = "propagation_summary"
TRACKSUMMARY_TREE = "tracksummary"

BOUND_PARAMETER_PAIRS = (
    ("d0", "eLOC0_fit"),
    ("z0", "eLOC1_fit"),
    ("phi", "ePHI_fit"),
    ("theta", "eTHETA_fit"),
    ("qOverP", "eQOP_fit"),
)


@dataclass
class NodeArrays:
    """Material nodes of a single track, ordered along the trajectory."""

    surface_id: np.ndarray
    volume_id: np.ndarray
    x_over_x0: np.ndarray
    position: np.ndarray
    direction: np.ndarray
    path_mm: np.ndarray

    def __len__(self) -> int:
        return int(self.x_over_x0.size)


# ---------------------------------------------------------------------------
# Aggregation and thin-node removal
# ---------------------------------------------------------------------------


def _aggregation_group_ids(
    surface_id: np.ndarray,
    volume_id: np.ndarray,
    path_mm: np.ndarray,
    volume_merge_window_mm: float,
) -> np.ndarray:
    """Assign a group id to each node; only adjacent same-key nodes may merge.

    Surface material is keyed by ``sur_id``.  Volume material is written with
    ``sur_id == 0`` for every node (``RootMaterialTrackIo.cpp:245``), so it is
    keyed by ``vol_id`` and additionally split whenever consecutive nodes are
    farther apart than the merge window.  Without that split, one group would
    swallow every volume interaction along the track.
    """
    n = surface_id.size
    groups = np.zeros(n, dtype=np.int64)
    for index in range(1, n):
        is_volume = surface_id[index] == 0
        same_key = (
            volume_id[index] == volume_id[index - 1]
            if is_volume
            else surface_id[index] == surface_id[index - 1]
        )
        # A surface node and a volume node never share a key.
        same_kind = (surface_id[index] == 0) == (surface_id[index - 1] == 0)
        adjacent = True
        if is_volume:
            adjacent = (
                float(path_mm[index] - path_mm[index - 1]) <= volume_merge_window_mm
            )
        groups[index] = groups[index - 1] + (
            0 if (same_kind and same_key and adjacent) else 1
        )
    return groups


def aggregate_nodes(
    surface_id: np.ndarray,
    volume_id: np.ndarray,
    x_over_x0: np.ndarray,
    position: np.ndarray,
    direction: np.ndarray,
    path_mm: np.ndarray,
    volume_merge_window_mm: float = VOLUME_MERGE_WINDOW_MM,
) -> NodeArrays:
    """Merge adjacent nodes of the same scatterer into one effective node.

    Aggregation runs BEFORE thin-node removal so that many individually
    negligible slabs of one surface still add up to a real scatterer.
    """
    surface_id = np.asarray(surface_id)
    volume_id = np.asarray(volume_id)
    x_over_x0 = np.asarray(x_over_x0, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    path_mm = np.asarray(path_mm, dtype=np.float64)
    if x_over_x0.size == 0:
        return NodeArrays(surface_id, volume_id, x_over_x0, position, direction, path_mm)

    groups = _aggregation_group_ids(
        surface_id, volume_id, path_mm, volume_merge_window_mm
    )
    n_groups = int(groups[-1]) + 1
    weights = np.maximum(x_over_x0, 0.0)
    weight_sum = np.bincount(groups, weights=weights, minlength=n_groups)
    # An all-vacuum group would divide by zero; fall back to an unweighted mean.
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    degenerate = weight_sum <= 0.0
    effective = np.where(degenerate[groups], 1.0, weights)
    effective_sum = np.where(degenerate, counts, weight_sum)

    def _weighted(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            total = np.bincount(groups, weights=effective * values, minlength=n_groups)
            return total / effective_sum
        columns = [
            np.bincount(groups, weights=effective * values[:, axis], minlength=n_groups)
            / effective_sum
            for axis in range(values.shape[1])
        ]
        return np.stack(columns, axis=1)

    merged_direction = _weighted(direction)
    norms = np.linalg.norm(merged_direction, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("aggregated direction collapsed to a zero vector")
    merged_direction = merged_direction / norms

    first_of_group = np.zeros(n_groups, dtype=np.int64)
    first_of_group[groups[::-1]] = np.arange(groups.size)[::-1]
    return NodeArrays(
        surface_id=surface_id[first_of_group],
        volume_id=volume_id[first_of_group],
        x_over_x0=weight_sum,
        position=_weighted(position),
        direction=merged_direction,
        path_mm=_weighted(path_mm),
    )


def drop_thin_nodes(
    nodes: NodeArrays,
    x_over_x0_min: float = X_OVER_X0_MIN,
    dropped_fraction_max: float = DROPPED_MATERIAL_FRACTION_MAX,
) -> tuple[NodeArrays, float]:
    """Remove negligible nodes, gating on discarded material rather than count.

    A track may legitimately lose many nodes as long as they carry almost no
    budget; losing few nodes that carry real material is the actual failure.
    """
    total = float(np.sum(nodes.x_over_x0))
    keep = nodes.x_over_x0 >= x_over_x0_min
    dropped = float(np.sum(nodes.x_over_x0[~keep]))
    fraction = 0.0 if total <= 0.0 else dropped / total
    if fraction > dropped_fraction_max:
        raise ValueError(
            "dropped material fraction exceeds the pre-registered limit: "
            f"{fraction:.4%} > {dropped_fraction_max:.4%}"
        )
    kept = NodeArrays(
        surface_id=nodes.surface_id[keep],
        volume_id=nodes.volume_id[keep],
        x_over_x0=nodes.x_over_x0[keep],
        position=nodes.position[keep],
        direction=nodes.direction[keep],
        path_mm=nodes.path_mm[keep],
    )
    return kept, fraction


def _sum_theta2(x_over_x0: np.ndarray, q_over_p: float) -> float:
    sigma = highland_kink_sigma(
        torch.as_tensor(np.asarray(x_over_x0, dtype=np.float64))[None, :],
        torch.as_tensor(np.asarray([float(q_over_p)], dtype=np.float64)),
    )
    return float(sigma.square().sum())


def aggregation_theta2_change(
    raw: NodeArrays, aggregated: NodeArrays, q_over_p: float
) -> float:
    """Relative change of sum(theta0^2) caused by aggregation.

    This is NOT expected to be zero.  ACTS evaluates the Highland logarithm with
    the per-slab x/X0 (``Interactions.cpp:516``), so merging slabs shifts the
    total by a closed-form log factor.  The quantity is reported per track and
    bounded, which catches aggregation bugs without asserting a conservation law
    that the physics does not obey.
    """
    before = _sum_theta2(raw.x_over_x0, q_over_p)
    after = _sum_theta2(aggregated.x_over_x0, q_over_p)
    if before <= 0.0:
        return 0.0 if after <= 0.0 else float("inf")
    return abs(after - before) / before


def check_aggregation_theta2(
    raw: NodeArrays,
    aggregated: NodeArrays,
    q_over_p: float,
    tolerance: float = AGGREGATION_THETA2_TOL,
) -> float:
    change = aggregation_theta2_change(raw, aggregated, q_over_p)
    if change > tolerance:
        raise ValueError(
            "aggregation changed sum(theta0^2) beyond the pre-registered bound: "
            f"{change:.4%} > {tolerance:.4%}"
        )
    return change


# ---------------------------------------------------------------------------
# Closures and structural validation
# ---------------------------------------------------------------------------


def check_t_x0_closure(
    step_length: np.ndarray,
    mat_x0: np.ndarray,
    t_x0: float,
    rtol: float = T_X0_CLOSURE_RTOL,
) -> float:
    """Verify the per-track material budget against the ROOT ``t_X0`` branch."""
    recomputed = float(np.sum(np.asarray(step_length) / np.asarray(mat_x0)))
    reference = float(t_x0)
    scale = max(abs(reference), abs(recomputed), 1e-30)
    deviation = abs(recomputed - reference) / scale
    if deviation > rtol:
        raise ValueError(
            "t_X0 closure failed: "
            f"sum(step_length / X0)={recomputed:.9g} vs t_X0={reference:.9g} "
            f"(relative {deviation:.3g} > {rtol:.3g})"
        )
    return deviation


def check_path_closure(
    position: np.ndarray,
    transverse_s: np.ndarray,
    tan_lambda: float,
    rtol: float = PATH_CLOSURE_RTOL,
) -> float:
    """Compare chord-sum arc length with transverse arc length, as increments.

    The helix ``s`` origin is set by ``(x0, y0, z0)`` of whichever fit produced
    it, so absolute values are not comparable across trajectories.  Only the
    increments are.  ``L_cum`` approximates arc length by summing chords, whose
    discretisation error grows with node spacing and curvature; ``rtol`` must
    cover it.
    """
    position = np.asarray(position, dtype=np.float64)
    transverse_s = np.asarray(transverse_s, dtype=np.float64)
    if position.shape[0] < 2:
        return 0.0
    chords = np.linalg.norm(np.diff(position, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(chords)))
    expected = (transverse_s - transverse_s[0]) * np.sqrt(1.0 + float(tan_lambda) ** 2)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    deviation = float(np.max(np.abs(cumulative - expected))) / scale
    if deviation > rtol:
        raise ValueError(
            "path closure failed: chord-sum arc length disagrees with "
            f"(s - s[0]) * sqrt(1 + tan^2 lambda) by {deviation:.3g} > {rtol:.3g}"
        )
    return deviation


def validate_material_arrays(
    step_length: np.ndarray,
    mat_x0: np.ndarray,
    direction: np.ndarray,
    path_mm: np.ndarray,
) -> None:
    """Structural fail-closed checks on one track's jagged material branches."""
    lengths = {
        "step_length": np.asarray(step_length).shape[0],
        "mat_x0": np.asarray(mat_x0).shape[0],
        "direction": np.asarray(direction).shape[0],
        "path_mm": np.asarray(path_mm).shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"material branch lengths disagree: {lengths}")
    norms = np.linalg.norm(np.asarray(direction, dtype=np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=DIRECTION_NORM_ATOL):
        raise ValueError(
            "material direction vectors are not unit normalised: "
            f"max deviation {float(np.max(np.abs(norms - 1.0))):.3g}"
        )
    if not np.all(np.asarray(mat_x0, dtype=np.float64) > 0.0):
        raise ValueError("non-positive radiation length in mat_X0")
    path = np.asarray(path_mm, dtype=np.float64)
    if path.size > 1 and not np.all(np.diff(path) > 0.0):
        raise ValueError("material path length is not strictly monotonic")


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def join_prop_to_summary(
    prop: Mapping[str, np.ndarray], summary: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Join A: propagation summary to track summary on ``(event_nr, track_nr)``.

    Both sides are float32 casts of the same fitted bound parameters, so the
    five-parameter verification is a genuine identity check rather than a
    tolerance-tuned heuristic.  Returns the summary row index per prop row.
    """
    lookup = {
        (int(event), int(track)): index
        for index, (event, track) in enumerate(
            zip(summary["event_nr"], summary["track_nr"])
        )
    }
    order = np.empty(len(prop["event_nr"]), dtype=np.int64)
    unmatched = []
    for index, (event, track) in enumerate(zip(prop["event_nr"], prop["track_nr"])):
        key = (int(event), int(track))
        if key not in lookup:
            unmatched.append(key)
            continue
        order[index] = lookup[key]
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} propagation-summary rows are unmatched in the "
            f"track summary; first: {unmatched[0]}"
        )
    for prop_name, summary_name in BOUND_PARAMETER_PAIRS:
        left = np.asarray(prop[prop_name], dtype=np.float64)
        right = np.asarray(summary[summary_name], dtype=np.float64)[order]
        if not np.allclose(
            left, right, rtol=BOUND_PARAMETER_RTOL, atol=BOUND_PARAMETER_ATOL
        ):
            worst = int(np.argmax(np.abs(left - right)))
            raise ValueError(
                f"bound parameter mismatch in Join A for '{prop_name}' vs "
                f"'{summary_name}': row {worst} {left[worst]:.9g} != {right[worst]:.9g}"
            )
    return order


def _theta_to_eta(theta: np.ndarray) -> np.ndarray:
    return -np.log(np.tan(np.asarray(theta, dtype=np.float64) / 2.0))


def join_material_to_prop(
    material: Mapping[str, np.ndarray],
    prop: Mapping[str, np.ndarray],
    atol: float = DIRECTION_JOIN_ATOL,
) -> np.ndarray:
    """Join B: material tracks to propagation summary within each event.

    ``RootMaterialTrackWriter`` drops the track index and iterates an unordered
    map, so row order carries no information.  Both sides expose the same
    ``startParameters.direction()``; matching on ``(phi, eta)`` and demanding a
    bijection is what makes the association verifiable.  Returns the prop row
    index per material row.
    """
    material_events = np.asarray(material["event_id"], dtype=np.int64)
    prop_events = np.asarray(prop["event_nr"], dtype=np.int64)
    if material_events.size != prop_events.size:
        raise ValueError(
            "material and propagation-summary row count differ: "
            f"{material_events.size} vs {prop_events.size}"
        )
    material_phi = np.asarray(material["v_phi"], dtype=np.float64)
    material_eta = np.asarray(material["v_eta"], dtype=np.float64)
    prop_phi = np.asarray(prop["phi"], dtype=np.float64)
    prop_eta = _theta_to_eta(prop["theta"])

    order = np.full(material_events.size, -1, dtype=np.int64)
    for event in np.unique(material_events):
        left = np.flatnonzero(material_events == event)
        right = np.flatnonzero(prop_events == event)
        if left.size != right.size:
            raise ValueError(
                f"event {int(event)} row count differ between material "
                f"({left.size}) and propagation summary ({right.size})"
            )
        taken: set[int] = set()
        for index in left:
            close = right[
                (np.abs(prop_phi[right] - material_phi[index]) <= atol)
                & (np.abs(prop_eta[right] - material_eta[index]) <= atol)
            ]
            if close.size == 0:
                raise ValueError(
                    f"unmatched material row {int(index)} in event {int(event)}"
                )
            if close.size > 1:
                raise ValueError(
                    f"ambiguous material row {int(index)} in event {int(event)}: "
                    f"{close.size} propagation-summary candidates within {atol:g}"
                )
            candidate = int(close[0])
            if candidate in taken:
                raise ValueError(
                    f"ambiguous join in event {int(event)}: propagation-summary "
                    f"row {candidate} claimed twice"
                )
            taken.add(candidate)
            order[index] = candidate
    if np.any(order < 0):
        raise ValueError("Join B left material rows unassigned")
    return order


def check_propagation_coverage(
    propagated_per_event: Mapping[int, int],
    reconstructed_per_event: Mapping[int, int],
) -> int:
    """Fail closed when a selected track produced no propagation.

    Join A only proves that every propagation row maps onto a reconstructed
    track.  The opposite direction is what detects a silently dropped
    propagation (``PropagationAlgorithm.cpp:63`` continues past a failure), and
    a missing material track would otherwise just shrink the artifact.
    """
    failures = 0
    for event, reconstructed in reconstructed_per_event.items():
        propagated = int(propagated_per_event.get(int(event), 0))
        failures += max(0, int(reconstructed) - propagated)
    if failures:
        raise ValueError(
            f"{failures} propagation failures: reconstructed tracks without a "
            "recorded material track"
        )
    return failures


def check_perigee_hypothesis(
    position: np.ndarray,
    d0: np.ndarray,
    z0: np.ndarray,
    phi: np.ndarray,
    atol: float = PERIGEE_HYPOTHESIS_ATOL_MM,
) -> float:
    rate = perigee_hypothesis_pass_rate(position, d0, z0, phi, atol=atol)
    if PERIGEE_HYPOTHESIS_REQUIRED and rate < 1.0:
        raise ValueError(
            "perigee hypothesis violated: only "
            f"{rate:.4%} of material tracks start at (-d0 sin(phi), d0 cos(phi), z0)"
        )
    return rate


def perigee_hypothesis_pass_rate(
    position: np.ndarray,
    d0: np.ndarray,
    z0: np.ndarray,
    phi: np.ndarray,
    atol: float = PERIGEE_HYPOTHESIS_ATOL_MM,
) -> float:
    """Fraction of tracks whose start position matches an origin perigee.

    Reported by Gate 0 as a diagnostic.  It is only promoted to a hard Join B
    check once the pass rate is measured to be 1.0; assuming the convention up
    front would bake an unverified surface definition into the contract.
    """
    position = np.asarray(position, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    expected = np.stack(
        (
            -np.asarray(d0, dtype=np.float64) * np.sin(phi),
            np.asarray(d0, dtype=np.float64) * np.cos(phi),
            np.asarray(z0, dtype=np.float64),
        ),
        axis=1,
    )
    matches = np.all(np.abs(position - expected) <= atol, axis=1)
    return float(np.mean(matches)) if matches.size else 0.0


# ---------------------------------------------------------------------------
# ROOT ingestion and artifact assembly
# ---------------------------------------------------------------------------

MATERIAL_BRANCHES = (
    "event_id",
    "v_x",
    "v_y",
    "v_z",
    "v_phi",
    "v_eta",
    "t_X0",
    "mat_x",
    "mat_y",
    "mat_z",
    "mat_dx",
    "mat_dy",
    "mat_dz",
    "mat_step_length",
    "mat_X0",
    "sur_id",
    "vol_id",
)
PROPSUMMARY_BRANCHES = ("event_nr", "track_nr", "d0", "z0", "phi", "theta", "qOverP", "nMaterials")
TRACKSUMMARY_BRANCHES = (
    "event_nr",
    "track_nr",
    "eLOC0_fit",
    "eLOC1_fit",
    "ePHI_fit",
    "eTHETA_fit",
    "eQOP_fit",
)


def _read_tree(path: str | Path, tree: str, branches) -> dict[str, Any]:
    import uproot

    with uproot.open(f"{path}:{tree}") as handle:
        missing = [name for name in branches if name not in handle]
        if missing:
            raise ValueError(f"'{path}:{tree}' is missing branches: {missing}")
        return handle.arrays(list(branches), library="np")


def _flatten_scalar(values) -> np.ndarray:
    """Track summary trees may be jagged per event or already flat per track."""
    array = np.asarray(values, dtype=object) if values.dtype == object else np.asarray(values)
    if array.dtype != object:
        return array
    return np.concatenate([np.atleast_1d(np.asarray(entry)) for entry in array])


def _normalize_track_summary(summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    track_entries = summary["track_nr"]
    jagged = track_entries.dtype == object
    if not jagged:
        return {name: np.asarray(values) for name, values in summary.items()}
    counts = np.array([len(np.atleast_1d(np.asarray(e))) for e in track_entries], dtype=np.int64)
    output = {"event_nr": np.repeat(np.asarray(summary["event_nr"]), counts)}
    for name, values in summary.items():
        if name == "event_nr":
            continue
        output[name] = _flatten_scalar(values)
    return output


def transverse_chord_path(position: np.ndarray) -> np.ndarray:
    """Export-side path coordinate: cumulative transverse chord length.

    Deliberately NOT a helix projection.  The export coordinate only has to
    order nodes and support the completeness closures; deriving it from a
    reconstructed reference helix would drag in the field value and a curvature
    sign convention for no benefit, and its origin would still be arbitrary.
    Validation recomputes its own ``node_s`` per fit window from
    ``position_mm`` (see the plan, section 1.5).
    """
    position = np.asarray(position, dtype=np.float64)
    if position.shape[0] < 2:
        return np.zeros(position.shape[0], dtype=np.float64)
    steps = np.hypot(np.diff(position[:, 0]), np.diff(position[:, 1]))
    return np.concatenate(([0.0], np.cumsum(steps)))


@dataclass
class ArtifactDiagnostics:
    n_tracks: int = 0
    n_nodes_raw: int = 0
    n_nodes_final: int = 0
    max_t_x0_deviation: float = 0.0
    max_path_deviation: float = 0.0
    max_dropped_material_fraction: float = 0.0
    max_aggregation_theta2_change: float = 0.0
    perigee_hypothesis_pass_rate: float = 0.0
    n_materials_vs_surface_nodes_max_diff: int = 0
    propagation_failures: int = 0
    #: 1 means Join B never had to disambiguate between candidates on real data.
    max_tracks_per_event: int = 0
    unmatched: int = 0
    duplicated: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}


def build_artifact(
    material_path: str | Path,
    propsummary_path: str | Path,
    tracksummary_path: str | Path,
) -> tuple[dict[str, np.ndarray], ArtifactDiagnostics]:
    """Join the three tables and emit schema 1.1 arrays plus diagnostics."""
    material = _read_tree(material_path, MATERIAL_TREE, MATERIAL_BRANCHES)
    prop = _read_tree(propsummary_path, PROPSUMMARY_TREE, PROPSUMMARY_BRANCHES)
    summary = _normalize_track_summary(
        _read_tree(tracksummary_path, TRACKSUMMARY_TREE, TRACKSUMMARY_BRANCHES)
    )

    summary_order = join_prop_to_summary(prop, summary)          # Join A (hard gate)
    material_order = join_material_to_prop(material, prop)       # Join B (bijective)

    summary_events, summary_counts = np.unique(
        np.asarray(summary["event_nr"], dtype=np.int64), return_counts=True
    )
    prop_events, prop_counts = np.unique(
        np.asarray(prop["event_nr"], dtype=np.int64), return_counts=True
    )
    diagnostics = ArtifactDiagnostics()
    diagnostics.propagation_failures = check_propagation_coverage(
        dict(zip(prop_events.tolist(), prop_counts.tolist())),
        dict(zip(summary_events.tolist(), summary_counts.tolist())),
    )
    diagnostics.max_tracks_per_event = int(prop_counts.max()) if prop_counts.size else 0
    diagnostics.perigee_hypothesis_pass_rate = check_perigee_hypothesis(
        np.stack(
            (
                np.asarray(material["v_x"], dtype=np.float64),
                np.asarray(material["v_y"], dtype=np.float64),
                np.asarray(material["v_z"], dtype=np.float64),
            ),
            axis=1,
        ),
        np.asarray(prop["d0"], dtype=np.float64)[material_order],
        np.asarray(prop["z0"], dtype=np.float64)[material_order],
        np.asarray(prop["phi"], dtype=np.float64)[material_order],
    )

    q_over_p_all = np.asarray(summary["eQOP_fit"], dtype=np.float64)
    theta_all = np.asarray(prop["theta"], dtype=np.float64)

    event_nr, track_nr, q_over_p = [], [], []
    offsets, path, position, direction, x_over_x0, sur_out, vol_out = [0], [], [], [], [], [], []

    for material_row in range(len(material["event_id"])):
        prop_row = int(material_order[material_row])
        summary_row = int(summary_order[prop_row])

        step_length = np.asarray(material["mat_step_length"][material_row], dtype=np.float64)
        mat_x0 = np.asarray(material["mat_X0"][material_row], dtype=np.float64)
        node_position = np.stack(
            (
                np.asarray(material["mat_x"][material_row], dtype=np.float64),
                np.asarray(material["mat_y"][material_row], dtype=np.float64),
                np.asarray(material["mat_z"][material_row], dtype=np.float64),
            ),
            axis=1,
        )
        node_direction = np.stack(
            (
                np.asarray(material["mat_dx"][material_row], dtype=np.float64),
                np.asarray(material["mat_dy"][material_row], dtype=np.float64),
                np.asarray(material["mat_dz"][material_row], dtype=np.float64),
            ),
            axis=1,
        )
        surface_id = np.asarray(material["sur_id"][material_row], dtype=np.uint64)
        volume_id = np.asarray(material["vol_id"][material_row], dtype=np.uint64)
        export_path = transverse_chord_path(node_position)

        validate_material_arrays(step_length, mat_x0, node_direction, export_path)
        if surface_id.size != step_length.size or volume_id.size != step_length.size:
            raise ValueError(
                f"surface/volume branch lengths disagree on material row {material_row}"
            )

        diagnostics.n_nodes_raw += int(step_length.size)
        diagnostics.max_t_x0_deviation = max(
            diagnostics.max_t_x0_deviation,
            check_t_x0_closure(step_length, mat_x0, float(material["t_X0"][material_row])),
        )
        tan_lambda = 1.0 / np.tan(theta_all[prop_row])
        diagnostics.max_path_deviation = max(
            diagnostics.max_path_deviation,
            check_path_closure(node_position, export_path, float(tan_lambda)),
        )
        surface_nodes = int(np.count_nonzero(surface_id != 0))
        diagnostics.n_materials_vs_surface_nodes_max_diff = max(
            diagnostics.n_materials_vs_surface_nodes_max_diff,
            abs(int(prop["nMaterials"][prop_row]) - surface_nodes),
        )

        raw = NodeArrays(
            surface_id=surface_id,
            volume_id=volume_id,
            x_over_x0=step_length / mat_x0,
            position=node_position,
            direction=node_direction,
            path_mm=export_path,
        )
        merged = aggregate_nodes(
            surface_id=raw.surface_id,
            volume_id=raw.volume_id,
            x_over_x0=raw.x_over_x0,
            position=raw.position,
            direction=raw.direction,
            path_mm=raw.path_mm,
        )
        track_q_over_p = float(q_over_p_all[summary_row])
        diagnostics.max_aggregation_theta2_change = max(
            diagnostics.max_aggregation_theta2_change,
            check_aggregation_theta2(raw, merged, track_q_over_p),
        )
        kept, dropped_fraction = drop_thin_nodes(merged)
        diagnostics.max_dropped_material_fraction = max(
            diagnostics.max_dropped_material_fraction, dropped_fraction
        )

        event_nr.append(int(summary["event_nr"][summary_row]))
        track_nr.append(int(summary["track_nr"][summary_row]))
        q_over_p.append(track_q_over_p)
        path.append(kept.path_mm)
        position.append(kept.position)
        direction.append(kept.direction)
        x_over_x0.append(kept.x_over_x0)
        sur_out.append(kept.surface_id)
        vol_out.append(kept.volume_id)
        offsets.append(offsets[-1] + len(kept))

    diagnostics.n_tracks = len(event_nr)
    diagnostics.n_nodes_final = offsets[-1]
    empty3 = np.zeros((0, 3), dtype=np.float64)
    artifact = {
        "event_nr": np.asarray(event_nr, dtype=np.int64),
        "track_nr": np.asarray(track_nr, dtype=np.int64),
        "q_over_p_gev_inv": np.asarray(q_over_p, dtype=np.float64),
        "node_offsets": np.asarray(offsets, dtype=np.int64),
        "path_length_mm": np.concatenate(path) if path else np.zeros(0),
        "position_mm": np.concatenate(position) if position else empty3,
        "direction": np.concatenate(direction) if direction else empty3,
        "x_over_x0": np.concatenate(x_over_x0) if x_over_x0 else np.zeros(0),
        "surface_geometry_id": (
            np.concatenate(sur_out) if sur_out else np.zeros(0, dtype=np.uint64)
        ),
        "volume_geometry_id": (
            np.concatenate(vol_out) if vol_out else np.zeros(0, dtype=np.uint64)
        ),
    }
    if set(artifact) != set(ARTIFACT_FIELDS):
        raise ValueError("artifact fields drifted from the declared schema")
    return artifact, diagnostics


def artifact_metadata() -> dict[str, Any]:
    """Units and pre-registered constants recorded alongside every artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {
            "path_length_mm": "export_transverse_chord_length_mm",
            "position_mm": "mm",
            "direction": "unit_vector",
            "x_over_x0": "radiation_lengths_including_acts_path_correction",
            "q_over_p_gev_inv": "1/GeV, signed, constant along the track",
        },
        "path_length_note": (
            "Cumulative transverse chord length from the first material node. "
            "NOT a track-invariant helix coordinate: consumers must recompute "
            "node arc length from position_mm against their own frozen "
            "reference helix before clipping or segment assignment."
        ),
        "constants": {
            "X_OVER_X0_MIN": X_OVER_X0_MIN,
            "VOLUME_MERGE_WINDOW_MM": VOLUME_MERGE_WINDOW_MM,
            "DROPPED_MATERIAL_FRACTION_MAX": DROPPED_MATERIAL_FRACTION_MAX,
            "AGGREGATION_THETA2_TOL": AGGREGATION_THETA2_TOL,
            "T_X0_CLOSURE_RTOL": T_X0_CLOSURE_RTOL,
            "PATH_CLOSURE_RTOL": PATH_CLOSURE_RTOL,
            "BOUND_PARAMETER_RTOL": BOUND_PARAMETER_RTOL,
            "DIRECTION_JOIN_ATOL": DIRECTION_JOIN_ATOL,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ACTS material-track ROOT output into a broken-helix artifact"
    )
    add_config_argument(parser)
    parser.add_argument("--material", type=absolute_project_path, required=True)
    parser.add_argument("--propsummary", type=absolute_project_path, required=True)
    parser.add_argument("--summary", type=absolute_project_path, required=True)
    parser.add_argument("--out", type=absolute_project_path, required=True)
    parser.add_argument("--run-tag", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--capture-source-snapshot",
        action="store_true",
        help="Archive the exact tracked+untracked source tree into the run's "
        "manifest so it can be replayed independent of git commit/dirty state.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_project_config()
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing artifact: {output}")
    # The artifact stays at --out; its provenance goes in a stage-owned
    # directory so it can never overwrite the export manifest.
    run_dir = output.parent / "convert"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *sys.argv]
    inputs = {
        "material_root": args.material,
        "propsummary_root": args.propsummary,
        "tracksummary_root": args.summary,
    }
    manifest = write_experiment_manifest(
        run_dir,
        run_id=args.run_tag,
        command=command,
        config=config,
        inputs=inputs,
        outputs={"material_artifact": output},
        metrics=artifact_metadata(),
        status="running",
        capture_source_snapshot=args.capture_source_snapshot,
    )
    try:
        artifact, diagnostics = build_artifact(args.material, args.propsummary, args.summary)
    except Exception as error:  # fail closed with the counts retained
        finalize_experiment_manifest(
            manifest,
            status="failed",
            metrics={**artifact_metadata(), "error": f"{type(error).__name__}: {error}"},
        )
        raise
    np.savez_compressed(output, **artifact)
    (run_dir / "material_artifact_metadata.json").write_text(
        json.dumps(
            {**artifact_metadata(), "diagnostics": diagnostics.as_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    finalize_experiment_manifest(
        manifest,
        status="completed",
        metrics={**artifact_metadata(), "diagnostics": diagnostics.as_dict()},
    )
    print(
        f"Material artifact written: {output} "
        f"({diagnostics.n_tracks} tracks, {diagnostics.n_nodes_final} nodes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
