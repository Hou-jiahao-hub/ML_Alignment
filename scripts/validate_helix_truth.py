"""Non-target helix refit from nominal/misaligned reconstruction with MC truth diagnostics.

Truth is RootTrackStatesWriter's t_eLOC0/1: averageSimHits transforms each
associated sim hit to the reference surface and averages local positions.
These are not measurements or smoothed states. Multiple contributors are
averaged; writer-side missing associations / failed transforms cannot be
identified from these two branches alone (the writer can emit zero).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

# Legacy shared CLI modules use absolute sibling imports. Keep this new entry
# importable as scripts.validate_helix_truth without requiring caller PYTHONPATH.
_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.append(_SCRIPT_DIRECTORY)

try:
    from . import helix_model as hm
    from .project_config import absolute_project_path, load_project_config
    from .experiment_manifest import (write_experiment_manifest,
                                      finalize_experiment_manifest,
                                      update_running_experiment_manifest)
    from .validate_helix_model import load_nominal_geometry, load_working_geometry, expand_measurement_errors, predictive_covariance
except ImportError:  # Direct CLI execution.
    import helix_model as hm
    from project_config import absolute_project_path, load_project_config
    from experiment_manifest import (write_experiment_manifest,
                                     finalize_experiment_manifest,
                                     update_running_experiment_manifest)
    from validate_helix_model import load_nominal_geometry, load_working_geometry, expand_measurement_errors, predictive_covariance


STATE_BRANCHES = (
    "volume_id", "layer_id", "module_id", "pathLength", "l_x_hit", "l_y_hit",
    "g_x_smt", "g_y_smt", "g_z_smt", "px_smt", "py_smt", "pz_smt",
    "t_eLOC0", "t_eLOC1",
)
TRUTH_SEMANTICS = "ACTS averageSimHits local position on nominal reference surface"
GROUPS = ("non_target", "target", "all")
PARAMETERS = ("x0", "y0", "z0", "psi0", "tan_lambda", "rho")


def new_timings():
    return {"schema_version": 1, "seed_seconds": 0., "refit_seconds": 0.,
            "stationarity_seconds": 0., "intersection_seconds": 0.,
            "budget_seconds": 0., "helix_wall_seconds": 0.,
            "seed_calls": 0, "refit_calls": 0, "retry_tracks": 0}


@contextmanager
def timed(timings, stage):
    start = time.perf_counter()
    if timings is not None and f"{stage}_calls" in timings:
        timings[f"{stage}_calls"] += 1
    try:
        yield
    finally:
        if timings is not None:
            timings[f"{stage}_seconds"] += time.perf_counter() - start


@dataclass
class TruthTrack:
    event_nr: int
    track_nr: int
    identities: list[tuple[int, int, int, int]]
    path_length: torch.Tensor
    centers: torch.Tensor
    rotations: torch.Tensor
    smt_points: torch.Tensor
    smt_directions: torch.Tensor
    measured: torch.Tensor
    variance: torch.Tensor
    truth: torch.Tensor
    target: torch.Tensor

    @property
    def component_mask(self):
        return (torch.isfinite(self.measured) & torch.isfinite(self.variance)
                & (self.variance > 0))

    @property
    def fit_mask(self):
        return ~self.target & self.component_mask.any(dim=1)


@dataclass
class FitResult:
    parameters: torch.Tensor | None
    seed_parameters: torch.Tensor | None
    seed_stationary: bool
    refit_stationary: bool
    seed_relative_decrement: float | None = None
    refit_decrement: float | None = None
    error: str | None = None


def require_branches(names):
    required = set(STATE_BRANCHES) | {"event_nr", "track_nr", "err_x_hit", "err_y_hit"}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"Missing required trackstates branches: {missing}")


def surface_identities(volumes, layers, modules):
    """Occurrence is per surface within one track, before path sorting."""
    seen = Counter()
    result = []
    for volume, layer, sensor in zip(volumes, layers, modules):
        key = (int(volume), int(layer), int(sensor))
        if min(key) <= 0:
            raise ValueError(f"Invalid measured surface ID: {key}")
        result.append((*key, seen[key]))
        seen[key] += 1
    return result


def load_tracks(states, geometry_path, target_layers, *, max_events=0, max_tracks=0,
                retain_ineligible=False):
    """Freeze lowest event/track keys before any Truth-dependent evaluation.

    ROOT branches belong to the same state record. Validate jagged shape, then
    attach one explicit event/track/surface/occurrence identity to that record;
    never join independent files by row number or by nearest position.
    """
    geometry = load_nominal_geometry(geometry_path)
    if not torch.isfinite(geometry.centers).all() or not torch.isfinite(geometry.rotations).all():
        raise ValueError("Non-finite nominal geometry")
    identity = torch.eye(3, dtype=torch.float64).expand_as(geometry.rotations)
    if not torch.allclose(geometry.rotations.transpose(-1, -2) @ geometry.rotations,
                          identity, atol=1.e-6, rtol=0):
        raise ValueError("Nominal rotations must be orthonormal")
    tracks, excluded = [], []
    with uproot.open(states, handler=uproot.MemmapSource) as root:
        tree = root["trackstates"]
        require_branches(tree.keys())
        events = tree["event_nr"].array(library="np")
        numbers = tree["track_nr"].array(library="np")
        keys = list(zip(events.tolist(), numbers.tolist()))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate event_nr/track_nr ROOT entries")
        unique_events = np.unique(events)
        selected_events = unique_events[:max_events] if max_events else unique_events
        entries = np.flatnonzero(np.isin(events, selected_events))
        entries = entries[np.lexsort((numbers[entries], events[entries]))]
        if max_tracks:
            entries = entries[:max_tracks]
        selected_entries = set(entries.tolist())
        counts = {"available_events": len(unique_events), "available_tracks": len(events),
                  "selected_events": len(np.unique(events[entries])), "selected_tracks": len(entries)}
        # Read bounded entry blocks, avoiding full 100k-event jagged materialization.
        for start in sorted({int(i) // 1024 * 1024 for i in entries}):
            stop = min(start + 1024, tree.num_entries)
            arrays = tree.arrays([*STATE_BRANCHES, "err_x_hit", "err_y_hit"],
                                 entry_start=start, entry_stop=stop, library="ak")
            for entry in range(start, stop):
                if entry not in selected_entries:
                    continue
                index = entry - start
                values = {name: ak.to_numpy(arrays[name][index]) for name in STATE_BRANCHES}
                n_states = len(values["volume_id"])
                if any(len(value) != n_states for value in values.values()):
                    raise ValueError(f"Misaligned per-state branches at entry {entry}")
                errors = expand_measurement_errors(
                    arrays["l_x_hit"][index:index + 1],
                    arrays["err_x_hit"][index:index + 1],
                    arrays["err_y_hit"][index:index + 1])
                # Same measured-state convention as the existing validators.
                measured = np.isfinite(values["l_x_hit"])
                identities = surface_identities(*(values[name][measured] for name in
                                                 ("volume_id", "layer_id", "module_id")))
                order = np.argsort(values["pathLength"][measured], kind="stable")
                identities = [identities[i] for i in order]
                try:
                    sensor = [geometry.id_to_idx[key[:3]] for key in identities]
                except KeyError as error:
                    raise ValueError(f"Measured surface absent from nominal geometry: {error}") from error

                def tensor(names):
                    return torch.tensor(np.column_stack([values[n][measured][order] for n in names]),
                                        dtype=torch.float64)

                track = TruthTrack(
                    event_nr=int(events[entry]), track_nr=int(numbers[entry]), identities=identities,
                    path_length=tensor(("pathLength",)).flatten(),
                    centers=geometry.centers[sensor], rotations=geometry.rotations[sensor],
                    smt_points=tensor(("g_x_smt", "g_y_smt", "g_z_smt")),
                    smt_directions=tensor(("px_smt", "py_smt", "pz_smt")),
                    measured=tensor(("l_x_hit", "l_y_hit")),
                    variance=torch.tensor(np.column_stack([e[measured][order] ** 2 for e in errors]),
                                          dtype=torch.float64),
                    truth=tensor(("t_eLOC0", "t_eLOC1")),
                    target=torch.tensor([key[:2] in target_layers for key in identities], dtype=torch.bool),
                )
                # Negative errors are invalid, not positive variances after squaring.
                for component, error in enumerate(errors):
                    track.variance[torch.tensor(error[measured][order] <= 0), component] = float("nan")
                reason = eligibility_reason(track)
                if reason:
                    excluded.append({"event_nr": track.event_nr, "track_nr": track.track_nr, "reason": reason})
                if not reason or retain_ineligible:
                    tracks.append(track)
    tracks.sort(key=lambda t: (t.event_nr, t.track_nr))
    counts.update(retained_tracks=len(tracks), excluded_tracks=len(excluded),
                  exclusion_reasons=dict(Counter(row["reason"] for row in excluded)))
    return tracks, excluded, counts


def merge_misaligned_track(nominal, misaligned):
    """Use branch-local SMT only; all scoring and measurement data stay nominal."""
    if ((nominal.event_nr, nominal.track_nr) != (misaligned.event_nr, misaligned.track_nr)
            or set(nominal.identities) != set(misaligned.identities)
            or len(nominal.identities) != len(misaligned.identities)):
        raise ValueError("Nominal/misaligned track or surface identities differ")
    lookup = {key: i for i, key in enumerate(misaligned.identities)}
    index = torch.tensor([lookup[key] for key in nominal.identities], dtype=torch.long)
    for name in ("measured", "variance"):
        if not torch.allclose(getattr(nominal, name), getattr(misaligned, name)[index],
                              rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"Nominal/misaligned measurement {name} differs")
    if (not torch.equal(nominal.component_mask, misaligned.component_mask[index])
            or not torch.equal(nominal.target, misaligned.target[index])):
        raise ValueError("Nominal/misaligned component or target masks differ")
    return replace(nominal, smt_points=misaligned.smt_points[index],
                   smt_directions=misaligned.smt_directions[index])


def load_misaligned_tracks(args, config, manifest):
    """Verify the historical cohort and hashes before replacing its SMT seed input."""
    import pandas as pd
    try:
        from . import plot_helix_truth_comparison as comparison
    except ImportError:
        import plot_helix_truth_comparison as comparison
    run = Path(args.nominal_run_dir)
    parent = json.loads((run / "manifest.json").read_text())
    frozen = pd.read_csv(run / "per_track.csv")
    surfaces = pd.read_csv(run / "per_surface.csv")
    if parent["status"] != "completed":
        raise ValueError("Nominal source run must be completed")
    comparison.verify_cohort_counts(parent, len(frozen), len(surfaces))
    comparison.require_unique(frozen, comparison.TRACK_KEYS)
    comparison.require_unique(surfaces, comparison.KEYS)
    actual_inputs = json.loads(manifest.read_text())["inputs"]
    for key in ("nominal_states", "nominal_geometry"):
        if actual_inputs[key]["sha256"] != parent["inputs"][key]["sha256"]:
            raise ValueError(f"Original input hash mismatch: {key}")
    if config.as_dict()["target_layers"] != parent["configuration"]["target_layers"]:
        raise ValueError("Target layers differ from nominal source run")
    reference = load_nominal_geometry(args.nominal_geometry)
    working = load_working_geometry(args.misaligned_geometry, args.nominal_geometry)
    non_target = torch.tensor([key[:2] not in config.target_layers for key in reference.id_to_idx])
    if (not torch.allclose(reference.centers[non_target], working.centers[non_target], rtol=0, atol=1.e-12)
            or not torch.allclose(reference.rotations[non_target], working.rotations[non_target],
                                  rtol=0, atol=1.e-12)):
        raise ValueError("Non-target misaligned geometry differs from nominal reference")
    # Both loads retain excluded records for an exact identity contract. The
    # nominal source alone freezes eligibility; branch-specific SMT failures
    # are reported during fitting, never removed from this cohort.
    nominal, excluded, counts = load_tracks(args.nominal_states, args.nominal_geometry,
        config.target_layers, max_events=args.max_events, max_tracks=args.max_tracks,
        retain_ineligible=True)
    branch, _, _ = load_tracks(args.misaligned_states, args.nominal_geometry,
        config.target_layers, max_events=args.max_events, max_tracks=args.max_tracks,
        retain_ineligible=True)
    nominal_keys = {(t.event_nr, t.track_nr) for t in nominal}
    branch_by_key = {(t.event_nr, t.track_nr): t for t in branch}
    if nominal_keys != set(branch_by_key):
        raise ValueError("Nominal/misaligned selected track identities differ")
    frozen_keys = set(map(tuple, frozen[comparison.TRACK_KEYS].to_numpy()))
    excluded_keys = {(row["event_nr"], row["track_nr"]) for row in excluded}
    retained_keys = nominal_keys - excluded_keys
    if retained_keys != frozen_keys & nominal_keys:
        raise ValueError("Selected cohort differs from frozen nominal source run")
    by_track = {key: frame for key, frame in surfaces.groupby(comparison.TRACK_KEYS, sort=False)}
    paired = []
    for track in nominal:
        key = (track.event_nr, track.track_nr)
        merged = merge_misaligned_track(track, branch_by_key[key])
        if key not in retained_keys:
            continue
        saved = by_track[key]
        identities = list(map(tuple, saved[comparison.KEYS[2:]].to_numpy()))
        if identities != track.identities:
            raise ValueError("Frozen nominal surface order or identities differ from ROOT")
        for field, prefix, suffix in (("truth", "truth", "mm"), ("measured", "measurement", "mm"),
                                      ("variance", "variance", "mm2")):
            saved_values = saved[[f"{prefix}_LOC{a}_{suffix}" for a in range(2)]].to_numpy()
            if not np.allclose(saved_values, getattr(track, field).numpy(), rtol=1.e-12,
                               atol=1.e-12, equal_nan=True):
                raise ValueError(f"Frozen nominal {field} differs from ROOT")
        if (not np.array_equal(saved.target.to_numpy(), track.target.numpy())
                or not np.array_equal(saved[[f"weighted_LOC{a}" for a in range(2)]].to_numpy(),
                                      track.component_mask.numpy())):
            raise ValueError("Frozen nominal target or component mask differs from ROOT")
        paired.append(merged)
    counts.update(retained_tracks=len(paired), excluded_tracks=len(excluded),
                  frozen_retained_tracks=len(frozen_keys),
                  frozen_surfaces=len(surfaces),
                  cohort_scope="full_frozen_cohort" if retained_keys == frozen_keys else "diagnostic_subset")
    return paired, excluded, counts


def paired_chi2_comparison(data, tracks, *, replicates=2000, seed=6006, frozen_tracks=None):
    """Full-cohort verdict, with event-cluster bootstrap and explicit partial diagnostics."""
    records = tracks[["event_nr", "track_nr", "seed_stationary", "refit_stationary"]].copy()
    full_cohort = frozen_tracks is None or len(tracks) == frozen_tracks
    result = {"schema_version": 1, "preregistration_version": 1,
              "primary_group": "all", "reference": "nominal ACTS SMT",
              "bootstrap": {"unit": "event", "replicates": replicates, "seed": seed},
              "cohort_scope": "full_frozen_cohort" if full_cohort else "diagnostic_subset",
              "frozen_tracks": frozen_tracks if frozen_tracks is not None else len(tracks),
              "groups": {}, "verdict": "inconclusive"}
    stationary = tracks.refit_stationary.to_numpy(dtype=bool)
    events = tracks.event_nr.to_numpy()
    for group in ("all", "non_target", "target"):
        group_seed = seed + {"all": 0, "non_target": 1, "target": 2}[group]
        rng = np.random.default_rng(group_seed)
        raw = data["track_raw"][group]
        helix, acts = raw["Helix"], raw["ACTS"]
        valid = helix["complete"] & acts["complete"] & stationary
        expected = helix["N"] > 0
        complete = bool(full_cohort and expected.any() and valid[expected].all())
        h = np.where(valid, helix["Q"], np.nan)
        a = np.where(valid, acts["Q"], np.nan)
        delta = h - a
        records[f"{group}_complete"] = valid
        records[f"{group}_n_components"] = helix["N"]
        records[f"{group}_helix_chi2"] = h
        records[f"{group}_nominal_acts_chi2"] = a
        records[f"{group}_delta_chi2"] = delta
        records[f"{group}_helix_chi2_per_component"] = np.divide(
            h, helix["N"], out=np.full(len(h), np.nan), where=expected)
        records[f"{group}_nominal_acts_chi2_per_component"] = np.divide(
            a, helix["N"], out=np.full(len(a), np.nan), where=expected)
        ci = None
        if valid.any():
            unique, inverse = np.unique(events[valid], return_inverse=True)
            sums = np.bincount(inverse, weights=delta[valid])
            sizes = np.bincount(inverse)
            estimates = np.empty(replicates)
            for i in range(replicates):
                sampled = rng.integers(0, len(unique), size=len(unique))
                estimates[i] = sums[sampled].sum() / sizes[sampled].sum()
            ci = np.quantile(estimates, [.025, .975]).tolist()
        subset = {"tracks": int(valid.sum()), "helix_sum_chi2": number(np.nansum(h)) if valid.any() else None,
                  "nominal_acts_sum_chi2": number(np.nansum(a)) if valid.any() else None,
                  "helix_mean_chi2": number(h[valid].mean()) if valid.any() else None,
                  "nominal_acts_mean_chi2": number(a[valid].mean()) if valid.any() else None,
                  "mean_delta_chi2": number(delta[valid].mean()) if valid.any() else None,
                  "mean_delta_ci95": ci}
        result["groups"][group] = {"expected_tracks": int(expected.sum()),
            "incomplete_tracks": int((expected & ~valid).sum()), "full_cohort_complete": complete,
            **{key: value if complete else None for key, value in subset.items() if key != "tracks"},
            "bootstrap_seed": group_seed, "valid_subset": subset}
    primary = result["groups"]["all"]
    if primary["full_cohort_complete"]:
        result["verdict"] = "improved" if primary["mean_delta_ci95"][1] < 0 else "not_demonstrated"
    return records, result


def write_paired_comparison(output, counts):
    import pandas as pd
    try:
        from . import plot_helix_truth_comparison as comparison
    except ImportError:
        import plot_helix_truth_comparison as comparison
    manifest = json.loads((output / "manifest.json").read_text())
    surfaces = pd.read_csv(output / "per_surface.csv")
    frame = pd.read_csv(output / "per_track.csv")
    paired = comparison.pair_surfaces(surfaces, comparison.load_acts(
        manifest["inputs"]["nominal_states"]["path"], frame))
    data, _ = comparison.analyze(paired, frame)
    comparison.verify_saved_helix(data, frame)
    rows, summary = paired_chi2_comparison(data, frame,
                                         frozen_tracks=counts["frozen_retained_tracks"])
    rows.to_csv(output / "paired_comparison.csv", index=False)
    write_json(output / "paired_summary.json", summary)
    return summary


def eligibility_reason(track):
    fit = track.fit_mask
    if int(fit.sum()) < 8:
        return "fewer_than_8_non_target_hits"
    if not torch.isfinite(track.path_length).all():
        return "invalid_path_order"
    paths = track.path_length[fit]
    if float(paths.max() - paths.min()) < 300.:
        return "non_target_span_below_300_mm"
    if (not torch.isfinite(track.smt_points[fit]).all()
            or not torch.isfinite(track.smt_directions[fit]).all()
            or (track.smt_directions[fit, :2].norm(dim=1) <= 1.e-6).any()):
        return "invalid_non_target_smt"
    return None


def _point_stationarity(parameters, initial, points, mask):
    """Use the shared point residual/Jacobian and its exact fixed gauge.

    The public point fitter reports only batch aggregates. Recompute per-row
    decrements with the same residual and criterion; do not run another fit.
    """
    frame = hm.reference_gauge_frame(initial)
    displacement = parameters[:, :3] - frame[0]
    theta = hm.pack_gauge_fixed_track_parameters(parameters)
    theta[:, 0] = (displacement * frame[1]).sum(dim=1)
    theta[:, 1] = (displacement * frame[2]).sum(dim=1)
    residual = torch.func.vmap(hm._single_point_fit_residual)(theta, *frame, points, mask)
    jacobian = hm._BATCHED_POINT_FIT_JACOBIAN(theta, *frame, points, mask)
    hessian = jacobian.transpose(1, 2) @ jacobian
    gradient = jacobian.transpose(1, 2) @ residual.unsqueeze(-1)
    scale = hessian.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1.e-30)
    hessian = hessian + hm.GAUSS_NEWTON_MIN_DAMPING * scale[:, None, None] * torch.eye(5, dtype=theta.dtype)
    decrement = (gradient.transpose(1, 2) @ torch.linalg.solve(hessian, gradient)).flatten()
    relative = decrement / (.5 * residual.square().sum(dim=1)).clamp_min(
        hm.GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE)
    stationary = torch.isfinite(decrement) & torch.isfinite(relative) & (
        (decrement <= hm.GAUSS_NEWTON_ABSOLUTE_DECREMENT_TOLERANCE)
        | (relative <= hm.GAUSS_NEWTON_TOLERANCE))
    return stationary, relative


def fit_tracks(tracks, *, seed_iters=600, refit_iters=60, timings=None):
    """Fit each track once, batching independent rows; target/Truth never enter."""
    if not tracks:
        return []
    selected = [t.fit_mask.nonzero().flatten() for t in tracks]
    n_hits = max(len(indices) for indices in selected)
    n_tracks = len(tracks)
    points = torch.zeros((n_tracks, n_hits, 3), dtype=torch.float64)
    centers = torch.zeros_like(points)
    rotations = torch.eye(3, dtype=torch.float64).expand(n_tracks, n_hits, 3, 3).clone()
    measured = torch.zeros((n_tracks, n_hits, 2), dtype=torch.float64)
    variance = torch.ones_like(measured)
    components = torch.zeros_like(measured, dtype=torch.bool)
    mask = torch.zeros((n_tracks, n_hits), dtype=torch.bool)
    initial = torch.zeros((n_tracks, 6), dtype=torch.float64)
    for row, (track, indices) in enumerate(zip(tracks, selected)):
        if not len(indices):
            raise ValueError("Every fit needs non-target measurements")
        count = len(indices)
        points[row] = track.smt_points[indices[0]]
        centers[row] = track.centers[indices[0]]
        rotations[row] = track.rotations[indices[0]]
        points[row, :count] = track.smt_points[indices]
        centers[row, :count] = track.centers[indices]
        rotations[row, :count] = track.rotations[indices]
        active = track.component_mask[indices]
        measured[row, :count] = torch.where(active, track.measured[indices], 0.)
        variance[row, :count] = torch.where(active, track.variance[indices], 1.)
        components[row, :count] = active
        mask[row, :count] = True
        direction = track.smt_directions[indices[0]]
        initial[row, :3] = points[row, 0]
        initial[row, 3] = torch.atan2(direction[1], direction[0])
        initial[row, 4] = direction[2] / direction[:2].norm()
    inputs = hm.MeasurementFitInputs(measured, variance, components, centers, rotations)
    with timed(timings, "seed"):
        seeded, _ = hm.fit_helix_points_gauss_newton(initial, points, mask, max_iters=seed_iters)
    with timed(timings, "stationarity"):
        seed_ok, seed_relative = _point_stationarity(seeded, initial, points, mask)
    with timed(timings, "refit"):
        fitted, _ = hm.fit_helix_measurements_gauss_newton(seeded, inputs, max_iters=refit_iters)
    with timed(timings, "stationarity"):
        decrement = hm.measurement_newton_decrement(fitted, inputs)
    refit_ok = torch.isfinite(decrement) & (decrement <= hm.MEASUREMENT_NEWTON_DECREMENT_STATIONARY)
    return [FitResult(fitted[i], seeded[i], bool(seed_ok[i]), bool(refit_ok[i]),
                      number(seed_relative[i]), number(decrement[i])) for i in range(n_tracks)]


def ordered_predictions(tracks, results):
    """Continue each frozen helix through ordered planes, without hit coordinates.

    Re-anchoring is an exact representation change, not a refit. Branches that
    go backwards or repeat a crossing at a later recorded path are invalidated.
    This deliberately does not guess another winding using Truth or SMT.
    """
    count = max(len(t.identities) for t in tracks)
    parameters = torch.stack([r.parameters if r.parameters is not None else
                              torch.zeros(6, dtype=torch.float64) for r in results])
    current = parameters.clone()
    offset = torch.zeros(len(tracks), dtype=torch.float64)
    previous = torch.full_like(offset, float("nan"))
    outputs = [[] for _ in tracks]
    for column in range(count):
        centers = torch.stack([t.centers[min(column, len(t.identities) - 1)] for t in tracks])
        rotations = torch.stack([t.rotations[min(column, len(t.identities) - 1)] for t in tracks])
        step, position, valid = hm.intersect_plane(current, centers, rotations[..., :, 2])
        path = offset + step
        plane = ((position - centers) * rotations[..., :, 2]).sum(dim=-1).abs()
        local = (rotations.transpose(-1, -2) @ (position - centers).unsqueeze(-1)).squeeze(-1)
        for row, (track, result) in enumerate(zip(tracks, results)):
            if column >= len(track.identities):
                continue
            ok = bool(valid[row]) and result.parameters is not None
            if torch.isfinite(previous[row]):
                ok = ok and float(path[row] - previous[row]) >= -1.e-6
                if column and float(track.path_length[column] - track.path_length[column - 1]) > 1.e-6:
                    ok = ok and float(path[row] - previous[row]) > 1.e-6
            ok = ok and abs(float(path[row])) <= hm.S_MAX
            outputs[row].append((local[row, :2].clone(), float(path[row]), float(plane[row]), ok))
            if ok:
                previous[row] = path[row]
                offset[row] = path[row]
                current[row, :3] = position[row]
                current[row, 3] = parameters[row, 3] + parameters[row, 5] * path[row]
    return outputs


def target_prediction_budget(track, result, predictions):
    """Statistical target P in mm² at the frozen ref-only measurement optimum.

    Whitened target Jacobians give P/V; undo whitening once. Target values and
    truth never enter Gamma_ref. Match the saved ordered crossing before using P.
    """
    budget = torch.full_like(track.variance, float("nan"))
    if result.parameters is None or not result.refit_stationary:
        return budget
    active = track.component_mask
    variance = track.variance.clamp_min(hm.MEASUREMENT_VARIANCE_FLOOR)
    inputs = hm.MeasurementFitInputs(
        torch.where(active, track.measured, 0.)[None],
        torch.where(active, variance, 1.)[None], active[None],
        track.centers[None], track.rotations[None])
    parameters = result.parameters[None]
    local, valid = hm.measurement_local_prediction(parameters, inputs)
    if not bool(valid[0, track.fit_mask].all()):
        return budget
    saved = torch.stack([p[0] for p in predictions])
    consistent = torch.tensor([p[3] for p in predictions]) & valid[0]
    consistent &= (local[0] - saved).abs().amax(dim=-1) <= 1.e-9
    frame = hm.reference_gauge_frame(parameters)
    theta = hm.pack_gauge_fixed_track_parameters(parameters)
    jacobian = hm.measurement_residual_jacobian(theta, frame, inputs)
    refs = (active & track.fit_mask[:, None]).flatten()
    ref_jacobian = jacobian[:, refs]
    gamma = ref_jacobian.transpose(-1, -2) @ ref_jacobian
    eigenvalues = torch.linalg.eigvalsh(gamma)[0]
    if not torch.isfinite(eigenvalues).all() or float(eigenvalues[0]) <= 0:
        return budget
    targets = (active & track.target[:, None] & consistent[:, None]).flatten()
    if not bool(targets.any()):
        return budget
    try:
        covariance, diagnostics = predictive_covariance(gamma, jacobian[:, targets], damping=0.)
    except (ValueError, torch.linalg.LinAlgError):
        return budget
    if diagnostics.get("n_rank_deficient_rows", 0):
        return budget
    values = covariance[0].diagonal() * variance.flatten()[targets]
    budget.flatten()[targets] = torch.where(
        torch.isfinite(values) & (values >= 0), values, float("nan"))
    return budget


def number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def chi2_term(residual, variance):
    """Overflow is a missing diagnostic component, never a complete score."""
    if residual is None:
        return None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return number(np.square(np.float64(residual)) / variance)


def evaluate_track(track, result, predictions=None, budget=None):
    if predictions is None:
        predictions = ordered_predictions([track], [result])[0]
    rows = []
    for i, (prediction, path, plane, valid) in enumerate(predictions):
        volume, layer, sensor, occurrence = track.identities[i]
        row = {"event_nr": track.event_nr, "track_nr": track.track_nr,
               "volume_id": volume, "layer_id": layer, "module_id": sensor,
               "occurrence": occurrence, "target": bool(track.target[i]),
               "fit_hit": bool(track.fit_mask[i]), "intersection_valid": valid,
               "path_mm": number(path) if valid else None,
               "plane_residual_mm": number(plane) if valid else None}
        before = bool(track.fit_mask[:i].any())
        after = bool(track.fit_mask[i + 1:].any())
        row["prediction_regime"] = ("interpolation" if before and after else
                                    "forward_extrapolation" if before else "backward_extrapolation") if track.target[i] else "fit"
        row["refit_eta"] = number(torch.asinh(result.parameters[4])) if result.parameters is not None else None
        for a in range(2):
            label = f"LOC{a}"
            truth = number(track.truth[i, a])
            measured = number(track.measured[i, a])
            variance = number(track.variance[i, a])
            active = bool(track.component_mask[i, a])
            residual = number(prediction[a] - track.truth[i, a]) if valid and truth is not None else None
            row.update({f"prediction_{label}_mm": number(prediction[a]) if valid else None,
                        f"truth_{label}_mm": truth, f"measurement_{label}_mm": measured,
                        f"residual_{label}_mm": residual, f"variance_{label}_mm2": variance,
                        f"weighted_{label}": active,
                        f"truth_chi2_{label}": chi2_term(residual, variance) if active else None,
                        f"measurement_chi2_{label}": chi2_term(float(prediction[a] - measured), variance)
                        if active and valid else None})
            if budget is not None:
                p = number(budget[i, a])
                row.update({f"P_{label}_mm2": p,
                            f"V_used_{label}_mm2": max(variance, hm.MEASUREMENT_VARIANCE_FLOOR) if active else None,
                            f"measurement_residual_{label}_mm": number(prediction[a] - track.measured[i, a]) if active and valid else None,
                            f"budget_valid_{label}": bool(track.target[i]) and active and valid and p is not None})
        rows.append(row)
    status = "failed" if result.error else (
        "stationary" if result.seed_stationary and result.refit_stationary else "non_stationary")
    summary = {"event_nr": track.event_nr, "track_nr": track.track_nr, "fit_status": status,
               "fit_error": result.error, "seed_stationary": result.seed_stationary,
               "refit_stationary": result.refit_stationary,
               "seed_relative_decrement": result.seed_relative_decrement,
               "refit_decrement": result.refit_decrement,
               "n_surfaces": len(rows), "intersection_failures": sum(not r["intersection_valid"] for r in rows)}
    for stage, parameters in (("seed", result.seed_parameters), ("refit", result.parameters)):
        for name, value in zip(PARAMETERS, parameters if parameters is not None else [float("nan")] * 6):
            summary[f"{stage}_{name}"] = number(value)
    for group in GROUPS:
        selected = [r for r in rows if group == "all" or r["target"] == (group == "target")]
        values = [r[f"truth_chi2_LOC{a}"] for r in selected for a in range(2)
                  if r[f"weighted_LOC{a}"]]
        finite = [v for v in values if v is not None]
        total = number(sum(finite))
        complete = (bool(values) and len(finite) == len(values) and total is not None
                    and all(r["intersection_valid"] for r in selected))
        summary.update({f"{group}_complete": complete, f"{group}_n_expected_components": len(values),
                        f"{group}_n_components": len(finite),
                        f"{group}_truth_chi2": total if complete or not selected else None,
                        f"{group}_truth_chi2_per_component": total / len(values) if complete else None,
                        f"{group}_valid_subset_truth_chi2": total if finite else None})
    measurement = [r[f"measurement_chi2_LOC{a}"] for r in rows if not r["target"]
                   for a in range(2) if r[f"weighted_LOC{a}"]]
    valid_measurement = [v for v in measurement if v is not None]
    summary["non_target_measurement_n_components"] = len(valid_measurement)
    summary["non_target_measurement_chi2"] = number(sum(valid_measurement)) if len(valid_measurement) == len(measurement) and measurement else None
    return rows, summary


def summarize_prediction_budget(surfaces):
    cells = defaultdict(list)
    for row in surfaces:
        if not row["target"]:
            continue
        for a in range(2):
            label = f"LOC{a}"
            if row[f"weighted_{label}"]:
                cells[f'{row["volume_id"]}:{row["layer_id"]}|{label}'].append((row, label))
    result = {}
    for key, entries in sorted(cells.items()):
        complete = all(r[f"budget_valid_{label}"] and r[f"truth_chi2_{label}"] is not None
                       and r[f"measurement_chi2_{label}"] is not None for r, label in entries)
        cell = {"n_expected_components": len(entries), "complete": complete}
        if complete:
            v = np.array([r[f"V_used_{a}_mm2"] for r, a in entries])
            p = np.array([r[f"P_{a}_mm2"] for r, a in entries])
            residual = np.array([r[f"measurement_residual_{a}_mm"] for r, a in entries])
            truth = np.array([r[f"residual_{a}_mm"] for r, a in entries])
            cell.update(V=float(v.mean()), P=float(p.mean()), M=float(np.square(residual).mean()),
                        truth_M=float(np.square(truth).mean()), U=float((residual**2 - v - p).mean()),
                        pull2=float((residual**2 / (v + p)).mean()))
            if not all(np.isfinite(value) for name, value in cell.items() if name not in ("complete",)):
                cell = {"n_expected_components": len(entries), "complete": False}
        result[key] = cell
    return {"schema_version": 1, "evaluation_geometry": "nominal", "cells": result,
            "scope": "target-only statistical prediction budget; no material process covariance"}


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    scale = float(np.max(np.abs(values))) if len(values) else 0.
    scaled = values / scale if scale else values
    return {"count": len(values), "bias": float(scaled.mean() * scale) if len(values) else None,
            "rmse": float(np.sqrt(np.mean(scaled ** 2)) * scale) if len(values) else None,
            "p95_abs": float(np.quantile(np.abs(scaled), .95) * scale) if len(values) else None}


class Statistics:
    def __init__(self):
        self.residuals = defaultdict(list)
        self.track_rows = []
        self.surfaces = 0
        self.failed_intersections = 0
        self.missing_truth_components = 0
        self.unweighted_components = 0

    def add(self, surfaces, track):
        self.track_rows.append(track)
        self.surfaces += len(surfaces)
        self.failed_intersections += sum(not r["intersection_valid"] for r in surfaces)
        for row in surfaces:
            for a in range(2):
                key = f'{row["volume_id"]}:{row["layer_id"]}|LOC{a}|{"target" if row["target"] else "non_target"}'
                values = self.residuals[key]  # Keep empty groups in the summary.
                value = row[f"residual_LOC{a}_mm"]
                if value is not None:
                    values.append(value)
                self.missing_truth_components += row[f"truth_LOC{a}_mm"] is None
                self.unweighted_components += not row[f"weighted_LOC{a}"]

    def summary(self):
        tracks = len(self.track_rows)
        return {"schema_version": 1, "truth_semantics": TRUTH_SEMANTICS,
                "statistic": "measurement-resolution-weighted truth residual diagnostic; not a calibrated chi-square",
                "strict_target_holdout": False,
                "tracks_evaluated": tracks, "surfaces_evaluated": self.surfaces,
                "fit_status_counts": dict(Counter(t["fit_status"] for t in self.track_rows)),
                "seed_non_stationary_tracks": sum(not t["seed_stationary"] for t in self.track_rows),
                "refit_non_stationary_tracks": sum(not t["refit_stationary"] for t in self.track_rows),
                "fit_failure_rate": sum(t["fit_status"] != "stationary" for t in self.track_rows) / tracks if tracks else None,
                "fit_failure_rate_semantics": "legacy combined seed/refit stationarity diagnostic",
                "refit_failure_rate": sum(not t["refit_stationary"] for t in self.track_rows) / tracks if tracks else None,
                "intersection_failure_rate": self.failed_intersections / self.surfaces if self.surfaces else None,
                "missing_truth_components": self.missing_truth_components,
                "unweighted_components": self.unweighted_components,
                "layer_residuals_mm": {k: distribution(v) for k, v in sorted(self.residuals.items())},
                "track_chi2": {g: {"complete_tracks": sum(t[f"{g}_complete"] for t in self.track_rows),
                                    "chi2": distribution([t[f"{g}_truth_chi2"] for t in self.track_rows if t[f"{g}_complete"]]),
                                    "chi2_per_component": distribution([t[f"{g}_truth_chi2_per_component"] for t in self.track_rows if t[f"{g}_complete"]])}
                               for g in GROUPS}}

    def plot(self, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for target in (False, True):
            groups = [(k, v) for k, v in sorted(self.residuals.items())
                      if k.endswith("|target" if target else "|non_target")]
            fig, axes = plt.subplots(max(len(groups), 1), 1, figsize=(8, max(3, 2.3 * len(groups))), squeeze=False)
            for axis, (key, values) in zip(axes.flat, groups):
                axis.hist(values, bins=60)
                axis.set(title=key, xlabel="Prediction - MC truth [mm]", ylabel="Surfaces")
            fig.tight_layout()
            fig.savefig(output / ("target_residuals.png" if target else "non_target_residuals.png"))
            plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(11, 10))
        for row, group in enumerate(GROUPS):
            for col, suffix in enumerate(("truth_chi2", "truth_chi2_per_component")):
                values = [t[f"{group}_{suffix}"] for t in self.track_rows if t[f"{group}_complete"]]
                axes[row, col].hist(values, bins=60)
                axes[row, col].set(title=group, xlabel=suffix, ylabel="Complete tracks")
        fig.tight_layout()
        fig.savefig(output / "track_chi2.png")
        plt.close(fig)


def build_parser(config):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--geometry", choices=("nominal", "misaligned"), default="nominal",
                        help="Reconstruction branch; Truth scoring always uses nominal surfaces")
    parser.add_argument("--nominal-states", type=absolute_project_path,
                        default=str(config.data_root / "reconstruction" / "nominal" / config.target_layer_dir /
                                    f"trackstates_kf_{config.target_layer_suffix}.root"))
    parser.add_argument("--nominal-geometry", type=absolute_project_path,
                        default=str(config.data_root / "geometry" / "odd_transforms.json"))
    parser.add_argument("--misaligned-states", type=absolute_project_path,
                        default=str(config.data_root / "reconstruction" / "misaligned" /
                                    config.target_layer_dir / f"trackstates_kf_{config.target_layer_suffix}.root"))
    parser.add_argument("--misaligned-geometry", type=absolute_project_path,
                        default=str(config.data_root / "alignment" / config.target_layer_dir /
                                    f"misalignment_applied_{config.target_layer_suffix}.json"))
    parser.add_argument("--nominal-run-dir", type=absolute_project_path,
                        help="Completed nominal experiment whose cohort and Truth are frozen")
    parser.add_argument("--output-dir", type=absolute_project_path)
    parser.add_argument("--run-tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--seed-iters", type=int, default=600)
    parser.add_argument("--refit-iters", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--prediction-budget", action="store_true",
                        help="Compute target statistical prediction variance from the same refit")
    return parser


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    request, _ = probe.parse_known_args(raw)
    config = load_project_config(request.config)
    parser = build_parser(config)
    args = parser.parse_args(raw)
    if args.geometry == "misaligned" and not args.nominal_run_dir:
        parser.error("--geometry misaligned requires --nominal-run-dir")
    if min(args.max_events, args.max_tracks) < 0 or min(args.seed_iters, args.refit_iters, args.batch_size, args.threads) < 1:
        parser.error("Limits must be non-negative; iteration, batch and thread counts must be positive")
    if Path(args.run_tag).name != args.run_tag or args.run_tag in (".", ".."):
        parser.error("--run-tag must be a single directory name")
    output = Path(args.output_dir) if args.output_dir else config.data_root / "experiments" / "helix_truth" / config.target_layer_dir / args.run_tag
    inputs = {"nominal_states": args.nominal_states, "nominal_geometry": args.nominal_geometry}
    if args.geometry == "misaligned":
        inputs.update(misaligned_states=args.misaligned_states, misaligned_geometry=args.misaligned_geometry)
        for name in ("manifest.json", "summary.json", "per_track.csv", "per_surface.csv", "excluded_tracks.csv"):
            inputs[f"nominal_run_{name}"] = Path(args.nominal_run_dir) / name
    for source in (*inputs.values(), *config.source_paths):
        if Path(source).resolve().is_relative_to(output.resolve()):
            parser.error("Input/output path collision: source is inside the output directory")
    # Fresh directory prevents overwriting source artifacts, previous runs or provenance.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be new or empty")
    outputs = {name: output / name for name in (
        "per_surface.csv", "per_track.csv", "excluded_tracks.csv", "summary.json",
        "target_residuals.png", "non_target_residuals.png", "track_chi2.png")}
    outputs.update({name: output / name for name in ("timings.json", "helix.log")})
    if args.prediction_budget:
        outputs["error_budget.json"] = output / "error_budget.json"
    if args.geometry == "misaligned":
        outputs.update({name: output / name for name in ("paired_comparison.csv", "paired_summary.json")})
        for name in ("acts_vs_helix_residuals.png", "acts_vs_helix_residuals.pdf",
                     "acts_vs_helix_chi2.png", "acts_vs_helix_chi2.pdf",
                     "comparison_summary.json", "manifest.json"):
            outputs[f"acts_comparison/{name}"] = output / "acts_comparison" / name
    manifest = output / "manifest.json"
    effective = {**vars(args), "target_layers": config.target_layers, "dtype": "float64",
                 "min_non_target_hits": 8, "min_non_target_span_mm": 300.,
                 "truth_semantics": TRUTH_SEMANTICS, "strict_target_holdout": False,
                 "fit_geometry": "non-target nominal surfaces (verified equal to working geometry)",
                 "evaluation_geometry": "nominal", "reference_track": "nominal ACTS SMT",
                 "bootstrap_replicates": 2000, "bootstrap_seed": 6006,
                 "preregistration_version": 1}
    metrics = {"effective_experiment_config": effective}
    stats = Statistics()
    timings, budget_surfaces = new_timings(), []
    helix_start = time.perf_counter()
    def log(message, error=False):
        print(message, file=sys.stderr if error else sys.stdout, flush=True)
        if output.is_dir():
            with outputs["helix.log"].open("a") as stream:
                stream.write(message + "\n")
    try:
        print(f"Starting helix Truth diagnostic: {output}", flush=True)
        write_experiment_manifest(output, run_id=output.name,
                                  command=[sys.executable, str(Path(__file__).resolve()), *raw],
                                  config=config, inputs=inputs, outputs=outputs, metrics=metrics,
                                  allow_missing_inputs=True, capture_source_snapshot=True)
        torch.set_num_threads(args.threads)
        if args.geometry == "misaligned":
            log("Verifying frozen nominal cohort and pairing misaligned SMT")
            tracks, excluded, counts = load_misaligned_tracks(args, config, manifest)
        else:
            tracks, excluded, counts = load_tracks(args.nominal_states, args.nominal_geometry,
                                                  config.target_layers, max_events=args.max_events,
                                                  max_tracks=args.max_tracks)
        metrics["cohort"] = counts
        update_running_experiment_manifest(manifest, metrics=metrics)
        with outputs["excluded_tracks.csv"].open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["event_nr", "track_nr", "reason"])
            writer.writeheader()
            writer.writerows(excluded)
        with outputs["per_surface.csv"].open("w", newline="") as surface_file, outputs["per_track.csv"].open("w", newline="") as track_file:
            surface_writer = track_writer = None
            for start in range(0, len(tracks), args.batch_size):
                batch = tracks[start:start + args.batch_size]
                try:
                    fitted = fit_tracks(batch, seed_iters=args.seed_iters, refit_iters=args.refit_iters, timings=timings)
                except (ValueError, FloatingPointError, torch.linalg.LinAlgError):
                    # Isolate numerical failures; retain every frozen track in the output.
                    fitted = []
                    for track in batch:
                        timings["retry_tracks"] += 1
                        try:
                            fitted.extend(fit_tracks([track], seed_iters=args.seed_iters, refit_iters=args.refit_iters, timings=timings))
                        except (ValueError, FloatingPointError, torch.linalg.LinAlgError) as error:
                            fitted.append(FitResult(None, None, False, False, error=str(error)))
                with timed(timings, "intersection"):
                    predictions = ordered_predictions(batch, fitted)
                for track, fit, prediction in zip(batch, fitted, predictions):
                    budget = None
                    if args.prediction_budget:
                        with timed(timings, "budget"):
                            budget = target_prediction_budget(track, fit, prediction)
                    surfaces, row = evaluate_track(track, fit, prediction, budget=budget)
                    if args.prediction_budget:
                        budget_surfaces.extend(r for r in surfaces if r["target"])
                    if surface_writer is None:
                        surface_writer = csv.DictWriter(surface_file, fieldnames=list(surfaces[0]))
                        track_writer = csv.DictWriter(track_file, fieldnames=list(row))
                        surface_writer.writeheader()
                        track_writer.writeheader()
                    surface_writer.writerows(surfaces)
                    track_writer.writerow(row)
                    stats.add(surfaces, row)
                surface_file.flush()
                track_file.flush()
                metrics["progress"] = {"tracks_evaluated": len(stats.track_rows),
                                       "surfaces_evaluated": stats.surfaces,
                                       "failed_intersections": stats.failed_intersections}
                metrics["timings"] = dict(timings)
                write_json(outputs["timings.json"], timings)
                update_running_experiment_manifest(manifest, metrics=metrics)
                log(f"Evaluated {len(stats.track_rows)}/{len(tracks)} tracks")
        summary = stats.summary()
        summary["cohort"] = counts
        summary["effective_experiment_config"] = effective
        summary["geometry"] = args.geometry
        if args.geometry == "misaligned":
            log("Computing paired Truth chi2 against nominal ACTS SMT")
            summary["paired_comparison"] = write_paired_comparison(output, counts)
        if args.prediction_budget:
            summary["prediction_budget"] = summarize_prediction_budget(budget_surfaces)
            write_json(outputs["error_budget.json"], summary["prediction_budget"])
        stats.plot(output)
        timings["helix_wall_seconds"] = time.perf_counter() - helix_start
        summary["timings"] = dict(timings)
        write_json(outputs["timings.json"], timings)
        write_json(outputs["summary.json"], summary)
        metrics.update(summary=summary, timings=dict(timings))
        finalize_experiment_manifest(manifest, status="completed", metrics=metrics, outputs=outputs)
        if args.geometry == "misaligned":
            try:
                from . import plot_helix_truth_comparison as comparison
            except ImportError:
                import plot_helix_truth_comparison as comparison
            if comparison.main(["--run-dir", str(output)]) != 0:
                raise RuntimeError("ACTS comparison plotting failed; see acts_comparison/manifest.json")
            # The child hashes this completed parent manifest. Its outputs were
            # registered before finalization, so success must not rewrite it.
        return 0
    except (Exception, KeyboardInterrupt) as error:
        timings["helix_wall_seconds"] = time.perf_counter() - helix_start
        if output.is_dir():
            write_json(outputs["timings.json"], timings)
            if args.prediction_budget:
                write_json(outputs["error_budget.json"], summarize_prediction_budget(budget_surfaces))
        metrics.update(error=f"{type(error).__name__}: {error}", available_summary=stats.summary())
        metrics["timings"] = dict(timings)
        if manifest.exists():
            finalize_experiment_manifest(manifest, status="failed", metrics=metrics, outputs=outputs)
        log(metrics["error"], error=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
