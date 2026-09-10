"""Does an all-hit helix refit restore a misaligned-geometry track to nominal?

Four arms share one track selection, one seed procedure and one solver, so the
only thing that varies is which hits are allowed to bend the helix and which
geometry the surfaces sit on:

    nominal_all         nominal states    + nominal geometry    + every hit
    nominal_holdout     nominal states    + nominal geometry    + non-target hits
    misaligned_all      misaligned states + misaligned geometry + every hit
    misaligned_holdout  misaligned states + misaligned geometry + non-target hits

``nominal_holdout`` isolates the ACTS smoothing pass itself: the KF smoother
that produced ``g_smt``/``p_smt`` used every hit, target included, so even
under untouched nominal geometry a target-hit-holdout refit inherits whatever
the smoother already leaked into non-target states.  Comparing it against
``misaligned_holdout`` (both target-holdout, so the columns match) isolates
the effect of misalignment alone, instead of confounding it with whatever the
smoothing pass already contributes when the fit is compared against
``nominal_all`` (which does not hold target hits out of its own refit).

The chi2 the fit itself minimises cannot settle the question.  Six free helix
parameters absorb a rigid surface displacement, so the in-sample chi2 of the
all-hit arm is pulled *down* by exactly the failure mode under test: the more
completely the helix follows the misaligned hits, the better its own chi2 looks.
The primary criterion is therefore geometry-independent.  Both reconstructions
run over the same simulated events and carry the same per-state truth position
(``t_x, t_y, t_z``, verified bit-for-bit here), so

    d_truth = | closest point on the fitted helix - truth position |

measures in millimetres how far each arm's trajectory sits from where the
particle actually was.  ``nominal_all`` is the achievable floor; recovery means
``misaligned_all`` reaches it.  Every hit is also paired across arms, so the
per-hit displacement between arms is reported as well -- a paired statistic that
does not depend on the width of the underlying track sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from helix_model import (  # noqa: E402
    closest_helix_points,
    fit_helix_points_gauss_newton,
    intersect_plane,
)
from ml_alignment_autograd import (  # noqa: E402
    EPS_COVARIANCE,
    EPS_WEIGHT,
    STRIP_VOLUMES,
    STRIP_Y_FLOOR,
    _surface_key,
    parse_target_layers,
)
import validate_helix_model as validation  # noqa: E402

#: Mainline joint-alignment track selection (``ml_alignment_autograd`` lines
#: 81-82).  Reused verbatim so the rows measured here are the rows the mainline
#: actually fits, rather than a looser sample chosen for this diagnostic.
MIN_FIT_HITS = 8
MIN_FIT_SPAN = 300.0

PER_STATE_COLUMNS = (
    "volume_id",
    "layer_id",
    "module_id",
    "l_x_hit",
    "l_y_hit",
    "t_x",
    "t_y",
    "t_z",
    "g_x_smt",
    "g_y_smt",
    "g_z_smt",
    "px_smt",
    "py_smt",
    "pz_smt",
)
HIT_KEY = ("event_nr", "track_nr", "volume_id", "layer_id", "module_id")


def load_states(path: Path, entry_stop: int | None) -> pd.DataFrame:
    """Flatten measurement states, expanding per-measurement errors onto them.

    ``err_x_hit`` / ``err_y_hit`` are stored per measurement while everything
    else is per state, and hole/material states carry a NaN ``l_x_hit``.  The
    NaN pattern is the only key between the two lengths, which is how
    ``ml_alignment_autograd.load_variable_information`` does it too.
    """
    with uproot.open(str(path)) as handle:
        tree = handle["trackstates"]
        arrays = tree.arrays(
            list(PER_STATE_COLUMNS) + ["err_x_hit", "err_y_hit", "event_nr", "track_nr"],
            entry_stop=entry_stop,
            library="ak",
        )
    counts = ak.to_numpy(ak.num(arrays["volume_id"]))
    frame = pd.DataFrame(
        {name: ak.to_numpy(ak.flatten(arrays[name])) for name in PER_STATE_COLUMNS}
    )
    frame["event_nr"] = np.repeat(ak.to_numpy(arrays["event_nr"]), counts)
    frame["track_nr"] = np.repeat(ak.to_numpy(arrays["track_nr"]), counts)

    lx_jagged = arrays["l_x_hit"]
    err_x = np.full(len(frame), np.nan)
    err_y = np.full(len(frame), np.nan)
    offset = 0
    for index in range(len(counts)):
        row = ak.to_numpy(lx_jagged[index])
        measured = ~np.isnan(row)
        n_measured = int(measured.sum())
        values_x = ak.to_numpy(arrays["err_x_hit"][index])
        values_y = ak.to_numpy(arrays["err_y_hit"][index])
        for name, values in (("err_x_hit", values_x), ("err_y_hit", values_y)):
            if len(values) != n_measured:
                raise ValueError(
                    f"{name} length mismatch at entry {index}: "
                    f"got {len(values)}, expected {n_measured}"
                )
        err_x[offset : offset + len(row)][measured] = values_x
        err_y[offset : offset + len(row)][measured] = values_y
        offset += len(row)
    frame["err_x_hit"] = err_x
    frame["err_y_hit"] = err_y

    frame = frame[frame["l_x_hit"].notna()].copy()
    frame = frame[frame["module_id"] > 0].copy()
    frame = frame.dropna(subset=["g_x_smt", "g_y_smt", "g_z_smt", "err_x_hit"])
    return frame.reset_index(drop=True)


def require_paired(nominal: pd.DataFrame, misaligned: pd.DataFrame) -> None:
    """Refuse to continue unless the two arms describe the same measurements.

    The whole design rests on this: the measurement and the truth position are
    properties of the simulated event, not of the geometry the reconstruction
    assumed, so they must come out bit-for-bit identical.  Anything else means
    the two files are not two views of one sample and no paired statistic below
    is meaningful.
    """
    keys = list(HIT_KEY)
    if len(nominal) != len(misaligned):
        raise ValueError(
            f"Arms hold different measurement counts: {len(nominal)} vs {len(misaligned)}."
        )
    for frame, label in ((nominal, "nominal"), (misaligned, "misaligned")):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} states contain duplicate hit keys.")
    left = nominal.set_index(keys).sort_index()
    right = misaligned.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Arms do not cover the same set of hits.")
    for column in ("l_x_hit", "l_y_hit", "t_x", "t_y", "t_z", "err_x_hit", "err_y_hit"):
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError(
                f"{column} differs between arms; it must be geometry independent."
            )


@dataclass
class ArmBatches:
    measured_local: torch.Tensor
    variance: torch.Tensor
    loc1_valid: torch.Tensor
    sensor_idx: torch.Tensor
    hit_mask: torch.Tensor
    is_target: torch.Tensor
    g_smt: torch.Tensor
    p_smt: torch.Tensor
    truth_global: torch.Tensor
    event_track_ids: torch.Tensor


def build_batches(
    frame: pd.DataFrame,
    id_to_idx: dict[tuple[int, int, int], int],
    target_layers: set[tuple[int, int]],
    row_order: list[tuple[tuple[int, int], list[int]]],
) -> ArmBatches:
    """Pad the selected tracks into ``[tracks, hits, ...]`` tensors."""
    n_tracks = len(row_order)
    max_hits = max(len(indices) for _key, indices in row_order)

    measured = torch.zeros((n_tracks, max_hits, 2), dtype=torch.float64)
    variance = torch.ones((n_tracks, max_hits, 2), dtype=torch.float64)
    loc1_valid = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    sensor_idx = torch.zeros((n_tracks, max_hits), dtype=torch.long)
    hit_mask = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    is_target = torch.zeros((n_tracks, max_hits), dtype=torch.bool)
    g_smt = torch.zeros((n_tracks, max_hits, 3), dtype=torch.float64)
    p_smt = torch.zeros((n_tracks, max_hits, 3), dtype=torch.float64)
    truth = torch.zeros((n_tracks, max_hits, 3), dtype=torch.float64)
    event_track_ids = torch.zeros((n_tracks, 2), dtype=torch.long)

    var0 = np.clip(frame["err_x_hit"].to_numpy() ** 2, EPS_WEIGHT, None)
    raw_var1 = frame["err_y_hit"].to_numpy() ** 2
    has_loc1 = np.isfinite(raw_var1)
    var1 = np.where(has_loc1, np.clip(raw_var1, EPS_WEIGHT, None), 1.0)
    strip = frame["volume_id"].isin(STRIP_VOLUMES).to_numpy() & has_loc1
    var1 = np.where(strip, np.maximum(var1, STRIP_Y_FLOOR), var1)
    local = np.stack(
        [frame["l_x_hit"].to_numpy(), np.nan_to_num(frame["l_y_hit"].to_numpy())], axis=1
    )
    sensor = np.array(
        [
            id_to_idx[(int(v), int(l), int(m))]
            for v, l, m in zip(frame["volume_id"], frame["layer_id"], frame["module_id"])
        ],
        dtype=np.int64,
    )
    positions = frame[["g_x_smt", "g_y_smt", "g_z_smt"]].to_numpy()
    directions = frame[["px_smt", "py_smt", "pz_smt"]].to_numpy()
    truth_positions = frame[["t_x", "t_y", "t_z"]].to_numpy()
    target_flag = np.array(
        [
            (int(v), int(l)) in target_layers
            for v, l in zip(frame["volume_id"], frame["layer_id"])
        ]
    )

    for row, (key, indices) in enumerate(row_order):
        count = len(indices)
        take = np.asarray(indices, dtype=np.int64)
        measured[row, :count] = torch.from_numpy(local[take])
        variance[row, :count, 0] = torch.from_numpy(var0[take])
        variance[row, :count, 1] = torch.from_numpy(var1[take])
        loc1_valid[row, :count] = torch.from_numpy(has_loc1[take])
        sensor_idx[row, :count] = torch.from_numpy(sensor[take])
        hit_mask[row, :count] = True
        is_target[row, :count] = torch.from_numpy(target_flag[take])
        g_smt[row, :count] = torch.from_numpy(positions[take])
        p_smt[row, :count] = torch.from_numpy(directions[take])
        truth[row, :count] = torch.from_numpy(truth_positions[take])
        event_track_ids[row] = torch.tensor(key, dtype=torch.long)

    return ArmBatches(
        measured_local=measured,
        variance=variance,
        loc1_valid=loc1_valid,
        sensor_idx=sensor_idx,
        hit_mask=hit_mask,
        is_target=is_target,
        g_smt=g_smt,
        p_smt=p_smt,
        truth_global=truth,
        event_track_ids=event_track_ids,
    )


def select_tracks(
    frame: pd.DataFrame,
    target_layers: set[tuple[int, int]],
    max_tracks: int,
) -> list[tuple[tuple[int, int], list[int]]]:
    """Apply the mainline selection and fix one hit ordering for every arm.

    The ordering is built from the *nominal* frame only and reused by both arms
    so that hit ``k`` of track ``i`` denotes the same measurement everywhere.
    Ordering by the reconstructed radius, as the mainline does, would order the
    two arms differently wherever the misalignment moved a hit across another.
    """
    radius = np.hypot(frame["t_x"].to_numpy(), frame["t_y"].to_numpy())
    order = np.lexsort(
        (
            frame["module_id"].to_numpy(),
            frame["layer_id"].to_numpy(),
            frame["volume_id"].to_numpy(),
            np.abs(frame["t_z"].to_numpy()),
            radius,
            frame["track_nr"].to_numpy(),
            frame["event_nr"].to_numpy(),
        )
    )
    events = frame["event_nr"].to_numpy()[order]
    tracks = frame["track_nr"].to_numpy()[order]
    boundaries = np.flatnonzero((events[1:] != events[:-1]) | (tracks[1:] != tracks[:-1])) + 1

    volumes = frame["volume_id"].to_numpy()
    layers = frame["layer_id"].to_numpy()
    truth_xyz = frame[["t_x", "t_y", "t_z"]].to_numpy()
    event_values = frame["event_nr"].to_numpy()
    track_values = frame["track_nr"].to_numpy()
    is_target = np.array(
        [(int(v), int(l)) in target_layers for v, l in zip(volumes, layers)]
    )

    selected: list[tuple[tuple[int, int], list[int]]] = []
    n_groups = 0
    for indices in np.split(order, boundaries):
        n_groups += 1
        flags = is_target[indices]
        fit_indices = indices[~flags]
        if len(fit_indices) < MIN_FIT_HITS or not flags.any():
            continue
        span = float(np.linalg.norm(np.diff(truth_xyz[fit_indices], axis=0), axis=1).sum())
        if span < MIN_FIT_SPAN:
            continue
        key = (int(event_values[indices[0]]), int(track_values[indices[0]]))
        selected.append((key, indices.tolist()))
    if max_tracks > 0:
        selected = selected[:max_tracks]
    if not selected:
        raise ValueError("No track satisfies the mainline joint-alignment selection.")
    print(f"  selection: {len(selected)} of {n_groups} tracks retained")
    return selected


def as_validation_batches(batches: ArmBatches, fit_mask: torch.Tensor):
    """Relabel as the validation module's ``TrackBatches`` for its solver.

    Only ``fit_mask`` steers the fit; the ``holdout_*`` fields are inert for the
    nodeless arm but must still be well formed, so the target hits stand in.
    """
    n_rows = fit_mask.shape[0]
    zeros_long = torch.zeros(n_rows, dtype=torch.long)
    zeros_real = torch.zeros(n_rows, dtype=torch.float64)
    return validation.TrackBatches(
        measured_local=batches.measured_local,
        measurement_variance=batches.variance,
        loc1_valid=batches.loc1_valid,
        smt_local=torch.zeros_like(batches.measured_local),
        sensor_idx=batches.sensor_idx,
        hit_mask=batches.hit_mask,
        fit_mask=fit_mask,
        holdout_mask=batches.is_target,
        g_smt=batches.g_smt,
        p_smt=batches.p_smt,
        volume_id=torch.zeros_like(batches.sensor_idx),
        layer_id=torch.zeros_like(batches.sensor_idx),
        module_id=torch.zeros_like(batches.sensor_idx),
        event_track_ids=batches.event_track_ids,
        fit_span=zeros_real,
        prediction_mode=("all-hit-refit-diagnostic",) * n_rows,
        context_hits_before=zeros_long,
        context_hits_after=zeros_long,
        nearest_context_distance=zeros_real,
        farthest_context_distance=zeros_real,
    )


def seed_from_points(batches: ArmBatches, fit_mask: torch.Tensor, max_iters: int):
    """The mainline seed: a Gauss-Newton point fit to the KF smoothed positions."""
    points = batches.g_smt
    rows = torch.arange(points.shape[0])
    first = fit_mask.to(torch.int64).argmax(dim=1)
    origin = points[rows, first]
    direction = batches.p_smt[rows, first]
    transverse = torch.linalg.norm(direction[:, :2], dim=1).clamp(min=1e-6)
    initial = torch.cat(
        (
            origin,
            torch.atan2(direction[:, 1], direction[:, 0])[:, None],
            (direction[:, 2] / transverse)[:, None],
            torch.zeros((points.shape[0], 1), dtype=points.dtype),
        ),
        dim=1,
    )
    return fit_helix_points_gauss_newton(initial, points, fit_mask, max_iters=max_iters)


def local_residuals(parameters: torch.Tensor, batches: ArmBatches, geometry):
    """Measured-minus-predicted local coordinates on the arm's own geometry."""
    n_hits = batches.measured_local.shape[1]
    tracks = parameters[:, None, :].expand(-1, n_hits, -1)
    centers = geometry.centers[batches.sensor_idx]
    rotations = geometry.rotations[batches.sensor_idx]
    _path, intersections, valid = intersect_plane(tracks, centers, rotations[..., :, 2])
    predicted = torch.matmul(
        rotations.transpose(-1, -2), (intersections - centers).unsqueeze(-1)
    ).squeeze(-1)[..., :2]
    return batches.measured_local - predicted, valid


def truth_distance(parameters: torch.Tensor, batches: ArmBatches):
    """Distance from each truth position to the fitted helix, in millimetres."""
    n_hits = batches.truth_global.shape[1]
    tracks = parameters[:, None, :].expand(-1, n_hits, -1)
    closest, _path, valid = closest_helix_points(tracks, batches.truth_global)
    return torch.linalg.norm(closest - batches.truth_global, dim=-1), valid


def chi2_block(residuals, variance, mask) -> dict:
    """Total chi2, component count and both normalisations for one hit subset."""
    weighted = residuals.square() / variance.clamp_min(EPS_COVARIANCE)
    total = float(weighted[mask].sum())
    count = int(mask.sum())
    return {
        "chi2_total": total,
        "n_components": count,
        "chi2_per_component": total / count if count else None,
    }


def distance_block(distance, mask) -> dict:
    values = distance[mask].numpy()
    return {
        "n_hits": int(values.size),
        "mean_mm": float(values.mean()),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "median_mm": float(np.median(values)),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(values.max()),
    }


def run_arm(name: str, batches: ArmBatches, geometry, fit_mask: torch.Tensor, args) -> dict:
    print(f"[{name}] seeding {batches.hit_mask.shape[0]} tracks ...")
    seed, seed_stats = seed_from_points(batches, fit_mask, args.seed_iters)
    print(f"[{name}] refitting ...")
    fitted, solver_stats = validation.fit_helices_gauss_newton(
        as_validation_batches(batches, fit_mask),
        geometry,
        seed,
        max_iters=args.refit_iters,
        dtype=torch.float64,
    )

    residuals, intersection_valid = local_residuals(fitted, batches, geometry)
    loc0 = batches.hit_mask & intersection_valid
    loc1 = loc0 & batches.loc1_valid
    component = torch.stack((loc0, loc1), dim=-1)
    target = batches.is_target.unsqueeze(-1).expand_as(component)

    distance, distance_valid = truth_distance(fitted, batches)
    hit_valid = batches.hit_mask & distance_valid

    # ``ndf`` subtracts the six helix parameters this arm actually spent on the
    # rows it fitted; only the fitted components paid for them, which is why the
    # held-out arm's target block is reported per component alone.
    fitted_components = component & fit_mask.unsqueeze(-1)
    n_fitted = int(fitted_components.sum())
    n_parameters = 6 * batches.hit_mask.shape[0]

    result = {
        "arm": name,
        "n_tracks": int(batches.hit_mask.shape[0]),
        "n_hits": int(batches.hit_mask.sum()),
        "n_fitted_hits": int(fit_mask.sum()),
        "intersection_failure_fraction": float(
            1.0 - intersection_valid[batches.hit_mask].double().mean()
        ),
        "chi2": {
            "fitted": chi2_block(residuals, batches.variance, fitted_components),
            "all_hits": chi2_block(residuals, batches.variance, component),
            "non_target": chi2_block(residuals, batches.variance, component & ~target),
            "target": chi2_block(residuals, batches.variance, component & target),
        },
        "chi2_per_ndf_fitted": (
            float(
                (
                    residuals.square() / batches.variance.clamp_min(EPS_COVARIANCE)
                )[fitted_components].sum()
            )
            / (n_fitted - n_parameters)
            if n_fitted > n_parameters
            else None
        ),
        "truth_distance": {
            "all_hits": distance_block(distance, hit_valid),
            "non_target": distance_block(distance, hit_valid & ~batches.is_target),
            "target": distance_block(distance, hit_valid & batches.is_target),
        },
        "seed": {
            "converged": bool(seed_stats["converged"]),
            "non_stationary_row_fraction": float(seed_stats["non_stationary_row_fraction"]),
        },
        "refit": {
            "non_stationary_row_fraction": float(
                solver_stats["stationarity"]["non_stationary_row_fraction"]
            ),
            "newton_decrement_p90": float(
                solver_stats["stationarity"]["newton_decrement_p90"]
            ),
        },
    }
    return result, fitted, distance, hit_valid


def paired_block(distance_a, distance_b, mask) -> dict:
    """Per-hit trajectory displacement between two arms over the same hits."""
    delta = (distance_a - distance_b)[mask].numpy()
    return {
        "n_hits": int(delta.size),
        "mean_mm": float(delta.mean()),
        "rms_mm": float(np.sqrt(np.mean(delta**2))),
        "p95_abs_mm": float(np.percentile(np.abs(delta), 95)),
    }


def trajectory_separation(params_a, params_b, batches: ArmBatches, mask) -> dict:
    """Distance between the two arms' helices at each shared truth point."""
    n_hits = batches.truth_global.shape[1]
    points = batches.truth_global
    a, _sa, va = closest_helix_points(params_a[:, None, :].expand(-1, n_hits, -1), points)
    b, _sb, vb = closest_helix_points(params_b[:, None, :].expand(-1, n_hits, -1), points)
    separation = torch.linalg.norm(a - b, dim=-1)
    valid = mask & va & vb
    values = separation[valid].numpy()
    return {
        "n_hits": int(values.size),
        "mean_mm": float(values.mean()),
        "rms_mm": float(np.sqrt(np.mean(values**2))),
        "p95_mm": float(np.percentile(values, 95)),
        "max_mm": float(values.max()),
    }


def build_geometry(transforms: list[dict]):
    id_to_idx = {_surface_key(item): index for index, item in enumerate(transforms)}
    return validation.NominalGeometry(
        transforms=transforms,
        centers=torch.tensor([item["Translation"] for item in transforms], dtype=torch.float64),
        rotations=torch.tensor([item["RotationMatrix"] for item in transforms], dtype=torch.float64),
        id_to_idx=id_to_idx,
    )


def load_geometries(nominal_path: Path, misaligned_path: Path, target_layers):
    with open(nominal_path, "r", encoding="utf-8") as stream:
        nominal = json.load(stream)
    with open(misaligned_path, "r", encoding="utf-8") as stream:
        overlay = json.load(stream)
    working = json.loads(json.dumps(nominal))
    index = {_surface_key(item): position for position, item in enumerate(working)}
    for item in overlay:
        key = _surface_key(item)
        if key not in index:
            raise KeyError(f"Input surface {key} is absent from the nominal geometry.")
        if key[:2] not in target_layers:
            raise ValueError(f"Input surface {key} lies outside the target layers.")
        working[index[key]] = json.loads(json.dumps(item))
    return build_geometry(nominal), build_geometry(working)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-states", required=True)
    parser.add_argument("--misaligned-states", required=True)
    parser.add_argument("--nominal-geometry", required=True)
    parser.add_argument("--misaligned-geometry", required=True)
    parser.add_argument("--target-layers", default="17:8,24:4")
    parser.add_argument("--max-entries", type=int, default=20000)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--seed-iters", type=int, default=200)
    parser.add_argument("--refit-iters", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    target_layers = set(parse_target_layers(args.target_layers))
    nominal_geometry, misaligned_geometry = load_geometries(
        Path(args.nominal_geometry), Path(args.misaligned_geometry), target_layers
    )

    entry_stop = args.max_entries if args.max_entries > 0 else None
    print("loading nominal states ...")
    nominal_frame = load_states(Path(args.nominal_states), entry_stop)
    print("loading misaligned states ...")
    misaligned_frame = load_states(Path(args.misaligned_states), entry_stop)
    require_paired(nominal_frame, misaligned_frame)
    print(f"paired measurements: {len(nominal_frame)}")

    keys = list(HIT_KEY)
    misaligned_frame = (
        misaligned_frame.set_index(keys)
        .reindex(pd.MultiIndex.from_frame(nominal_frame[keys]))
        .reset_index()
    )

    row_order = select_tracks(nominal_frame, target_layers, args.max_tracks)
    nominal_batches = build_batches(
        nominal_frame, nominal_geometry.id_to_idx, target_layers, row_order
    )
    misaligned_batches = build_batches(
        misaligned_frame, misaligned_geometry.id_to_idx, target_layers, row_order
    )

    all_hits = nominal_batches.hit_mask
    non_target = nominal_batches.hit_mask & ~nominal_batches.is_target

    arms = {}
    fitted = {}
    distances = {}
    valids = {}
    for name, batches, geometry, mask in (
        ("nominal_all", nominal_batches, nominal_geometry, all_hits),
        ("nominal_holdout", nominal_batches, nominal_geometry, non_target),
        ("misaligned_all", misaligned_batches, misaligned_geometry, all_hits),
        ("misaligned_holdout", misaligned_batches, misaligned_geometry, non_target),
    ):
        summary, parameters, distance, valid = run_arm(name, batches, geometry, mask, args)
        arms[name] = summary
        fitted[name] = parameters
        distances[name] = distance
        valids[name] = valid

    shared = valids["nominal_all"] & valids["misaligned_all"] & valids["misaligned_holdout"]
    shared_holdout = valids["nominal_holdout"] & valids["misaligned_holdout"]
    paired = {
        "misaligned_all_vs_nominal_all": {
            "truth_distance_delta": {
                "all_hits": paired_block(distances["misaligned_all"], distances["nominal_all"], shared),
                "non_target": paired_block(
                    distances["misaligned_all"], distances["nominal_all"], shared & ~nominal_batches.is_target
                ),
                "target": paired_block(
                    distances["misaligned_all"], distances["nominal_all"], shared & nominal_batches.is_target
                ),
            },
            "trajectory_separation": trajectory_separation(
                fitted["misaligned_all"], fitted["nominal_all"], nominal_batches, shared
            ),
        },
        "misaligned_holdout_vs_nominal_all": {
            "truth_distance_delta": {
                "all_hits": paired_block(
                    distances["misaligned_holdout"], distances["nominal_all"], shared
                ),
                "non_target": paired_block(
                    distances["misaligned_holdout"],
                    distances["nominal_all"],
                    shared & ~nominal_batches.is_target,
                ),
                "target": paired_block(
                    distances["misaligned_holdout"],
                    distances["nominal_all"],
                    shared & nominal_batches.is_target,
                ),
            },
            "trajectory_separation": trajectory_separation(
                fitted["misaligned_holdout"], fitted["nominal_all"], nominal_batches, shared
            ),
        },
        "misaligned_holdout_vs_nominal_holdout": {
            "truth_distance_delta": {
                "all_hits": paired_block(
                    distances["misaligned_holdout"], distances["nominal_holdout"], shared_holdout
                ),
                "non_target": paired_block(
                    distances["misaligned_holdout"],
                    distances["nominal_holdout"],
                    shared_holdout & ~nominal_batches.is_target,
                ),
                "target": paired_block(
                    distances["misaligned_holdout"],
                    distances["nominal_holdout"],
                    shared_holdout & nominal_batches.is_target,
                ),
            },
            "trajectory_separation": trajectory_separation(
                fitted["misaligned_holdout"], fitted["nominal_holdout"], nominal_batches, shared_holdout
            ),
        },
    }

    report = {
        "inputs": {
            "nominal_states": str(args.nominal_states),
            "misaligned_states": str(args.misaligned_states),
            "nominal_geometry": str(args.nominal_geometry),
            "misaligned_geometry": str(args.misaligned_geometry),
            "target_layers": args.target_layers,
            "max_entries": args.max_entries,
            "min_fit_hits": MIN_FIT_HITS,
            "min_fit_span": MIN_FIT_SPAN,
        },
        "arms": arms,
        "paired": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {output}")


if __name__ == "__main__":
    main()
