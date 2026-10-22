"""Target prediction budget for the three-arm material study on mlalign.

Re-implements the 2026-10-14 study (``material-helix-20261015-10a9266``) on the
current package: ref-only helix refit, then per target component

    M = (prediction - measurement)^2,   truth_M = (prediction - truth)^2,
    P = diag(B_t Gamma_ref^-1 B_t^T) * V,   U = M - V - P,   truth_excess = truth_M - P.

The statistics helpers are ported from 97cd01c
``scripts/analysis/compare_helix_material_effects.py`` with unchanged semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot

from mlalign.geometry_io import load_joint_geometry
from mlalign.helix.gauge import pack_gauge_fixed_track_parameters, reference_gauge_frame
from mlalign.helix.measurement_fit import (
    MEASUREMENT_NEWTON_DECREMENT_STATIONARY,
    MEASUREMENT_VARIANCE_FLOOR,
    MeasurementFitInputs,
    measurement_local_prediction,
    measurement_newton_decrement,
    measurement_residual_jacobian,
)
from mlalign.helix_init import fit_helices_to_measurements, seed_helices
from mlalign.root_inputs import STRIP_VOLUMES, STRIP_Y_FLOOR, load_variable_information
from mlalign.track_batches import build_track_batches

TARGET_LAYERS = ((17, 8), (24, 4))
GROUPS = ("g1_pt10_material_on", "g2_pt10_material_off", "g3_pt100_material_on")
ARM_CONFIG = {
    "g1_pt10_material_on": (10.0, True),
    "g2_pt10_material_off": (10.0, False),
    "g3_pt100_material_on": (100.0, True),
}
METRICS = {"M": "measurement2", "truth_M": "truth2", "V": "V", "P": "P",
           "U": "U", "truth_excess": "truth_excess"}
COMPONENTS = ("LOC0", "LOC1")
#: Material switch: relative |p| spread along a track; Fatras without
#: interactions keeps |p| to float rounding (measured max 1.9e-7).
MATERIAL_OFF_MAX_P_SPREAD = 1.0e-5
#: Same rank floor as ``validate_helix_model.predictive_covariance``.
RANK_FLOOR_FACTOR = 5


# --------------------------------------------------------------------------- inputs

def read_particles(path):
    """One generated particle per event, sorted by event id."""
    columns = ["event_id", "vx", "vy", "vz", "px", "py", "pz", "q", "eta", "phi"]
    arrays = uproot.open(path)["particles"].arrays(columns, library="ak")
    counts = ak.to_numpy(ak.num(arrays["px"]))
    if not np.all(counts == 1):
        raise ValueError(f"study requires exactly one generated particle per event: {path}")
    frame = pd.DataFrame({c: ak.to_numpy(arrays[c]) if c == "event_id"
                          else ak.to_numpy(ak.flatten(arrays[c])) for c in columns})
    frame = frame.sort_values("event_id", kind="mergesort").reset_index(drop=True)
    if not frame.event_id.is_unique:
        raise ValueError(f"duplicate generated event id: {path}")
    return frame


def momentum_spread(hits_path, entry_stop=200_000):
    """Median and max relative |p| spread of the true momentum along each track."""
    arrays = uproot.open(hits_path)["hits"].arrays(
        ["event_id", "tpx", "tpy", "tpz"], entry_stop=entry_stop, library="np")
    frame = pd.DataFrame(arrays)
    frame["p"] = np.sqrt(frame.tpx**2 + frame.tpy**2 + frame.tpz**2)
    spread = frame.groupby("event_id").p.agg(lambda p: (p.max() - p.min()) / p.max())
    return float(spread.median()), float(spread.max())


def verify_arms(particles, spreads):
    """Paired initial conditions and the declared pT / material switch per arm.

    ``particles`` and ``spreads`` are dicts keyed by group name.
    """
    first = particles[GROUPS[0]]
    for group in GROUPS:
        frame = particles[group]
        pt_gev, material = ARM_CONFIG[group]
        if not np.array_equal(frame.event_id.to_numpy(), first.event_id.to_numpy()):
            raise ValueError(f"generated event ids differ: {group}")
        for column in ("vx", "vy", "vz", "q", "eta", "phi"):
            if not np.allclose(frame[column], first[column], rtol=1e-6, atol=1e-7):
                raise ValueError(f"initial conditions differ: {group}, {column}")
        scale = pt_gev / ARM_CONFIG[GROUPS[0]][0]
        for column in ("px", "py", "pz"):
            if not np.allclose(frame[column], scale * first[column], rtol=2e-6, atol=1e-7):
                raise ValueError(f"momentum is not {scale:g} x arm 1: {group}, {column}")
        if not np.allclose(np.hypot(frame.px, frame.py), pt_gev, rtol=2e-6, atol=1e-7):
            raise ValueError(f"generated pT differs from {pt_gev} GeV: {group}")
        median, maximum = spreads[group]
        if material != (maximum > MATERIAL_OFF_MAX_P_SPREAD):
            raise ValueError(f"material switch does not match {group}: max |p| spread {maximum:.3g}")
    return {"passed": True, "n_generated_events": int(len(first)),
            "momentum_spread": {g: {"median": spreads[g][0], "max": spreads[g][1]} for g in GROUPS},
            "scope": "same event ids, vertices, charge and direction; arm 3 momentum x10"}


def nominal_geometry(nominal_path, workdir):
    """Full nominal working geometry; the target overlay is empty."""
    overlay = Path(workdir) / "nominal_target_overlay.json"
    overlay.write_text("[]\n", encoding="utf-8")
    return load_joint_geometry(nominal_path, overlay, TARGET_LAYERS)


def load_arm_batches(states, summary, geometry, max_track_entries=None):
    """Production loader and selection; variances switched to the pre-floor values.

    The 10-14 study weighted with V = err^2 (only the 1e-8 floor); the
    production STRIP_Y_FLOOR is not part of that protocol.
    """
    inputs = load_variable_information(
        summary, states, geometry.id_to_idx,
        max_track_entries=max_track_entries, load_summary=False)
    batches = build_track_batches(inputs, TARGET_LAYERS)
    floored = batches.covariances[..., 1] > batches.raw_covariances[..., 1]
    active = batches.hit_mask & batches.loc1_mask
    strip = torch.zeros_like(active)
    for volume in STRIP_VOLUMES:
        strip |= batches.volume_id == volume
    floor_stats = {
        "strip_y_floor_mm2": STRIP_Y_FLOOR,
        "loc1_floor_would_apply_fraction": float((floored & active).sum() / active.sum().clamp_min(1)),
        "target_loc1_floor_would_apply": int((floored & active & batches.is_target).sum()),
        "strip_loc1_components": int((active & strip).sum()),
    }
    batches.covariances = batches.raw_covariances.clone()
    del inputs
    return batches, floor_stats


def read_target_truth(states, target_layers=TARGET_LAYERS, max_entries=None):
    """Measured target states with their truth local position, one row per state."""
    columns = ["event_nr", "track_nr", "volume_id", "layer_id", "module_id",
               "l_x_hit", "t_eLOC0", "t_eLOC1"]
    tree = uproot.open(states)["trackstates"]
    parts = []
    for arrays in tree.iterate(columns, step_size=20_000, entry_stop=max_entries, library="ak"):
        counts = ak.to_numpy(ak.num(arrays["volume_id"]))
        frame = pd.DataFrame({
            "event_nr": np.repeat(ak.to_numpy(arrays["event_nr"]), counts),
            "track_nr": np.repeat(ak.to_numpy(arrays["track_nr"]), counts),
            **{c: ak.to_numpy(ak.flatten(arrays[c])) for c in columns[2:]},
        })
        on_target = np.zeros(len(frame), dtype=bool)
        for volume, layer in target_layers:
            on_target |= (frame.volume_id.to_numpy() == volume) & (frame.layer_id.to_numpy() == layer)
        parts.append(frame.loc[frame.l_x_hit.notna().to_numpy() & on_target].drop(columns="l_x_hit"))
    return pd.concat(parts, ignore_index=True).astype(
        {"event_nr": np.int64, "track_nr": np.int64, "volume_id": np.int64,
         "layer_id": np.int64, "module_id": np.int64})


def unique_truth(truth):
    """Drop every state whose (event, track, surface) key is not unique."""
    keys = ["event_nr", "track_nr", "volume_id", "layer_id", "module_id"]
    duplicated = truth.duplicated(keys, keep=False)
    return truth.loc[~duplicated].reset_index(drop=True), int(duplicated.sum())


# --------------------------------------------------------------------------- fit + budget

def fit_arm(batches, geometry, seed_iters=2000):
    """Non-target SMT seed, then the production ref-only Gauss-Newton refit."""
    seeded, seed_stats = seed_helices(
        batches, n_iter=seed_iters, dtype=torch.float64, return_diagnostics=True)
    refitted, refit_stats = fit_helices_to_measurements(
        batches, geometry, seeded, dtype=torch.float64, stationarity=False)
    return refitted, seed_stats, refit_stats


def budget_inputs(batches, geometry, dtype=torch.float64):
    """All measured components (ref and target) of every selected track."""
    centers = geometry.centers.to(dtype)[batches.sensor_idx]
    rotations = geometry.rotations.to(dtype)[batches.sensor_idx]
    components = torch.stack((batches.hit_mask, batches.hit_mask & batches.loc1_mask), dim=-1)
    return MeasurementFitInputs(
        measured=batches.measured_local.to(dtype),
        variance=batches.covariances.to(dtype),
        component_mask=components,
        centers=centers,
        rotations=rotations,
    )


def target_budget(parameters, inputs, fit_mask, target_mask, chunk_rows=1024):
    """Per component: prediction, intersection validity, P [mm^2] and its validity.

    ``fit_mask`` / ``target_mask`` are [tracks, hits]; Gamma_ref is built from
    fit rows only, so target measurements and variances cannot enter it.
    """
    n_tracks, n_hits = fit_mask.shape
    prediction = torch.empty((n_tracks, n_hits, 2), dtype=parameters.dtype)
    valid = torch.empty((n_tracks, n_hits), dtype=torch.bool)
    budget = torch.full((n_tracks, n_hits, 2), float("nan"), dtype=parameters.dtype)
    rank_deficient = torch.zeros(n_tracks, dtype=torch.bool)
    for start in range(0, n_tracks, chunk_rows):
        stop = min(start + chunk_rows, n_tracks)
        rows = inputs.slice(start, stop)
        params = parameters[start:stop]
        local, ok = measurement_local_prediction(params, rows)
        prediction[start:stop], valid[start:stop] = local, ok
        frame = reference_gauge_frame(params)
        theta = pack_gauge_fixed_track_parameters(params)
        jacobian = measurement_residual_jacobian(theta, frame, rows)  # [T, 2H, 5], whitened
        ref_rows = (rows.component_mask & fit_mask[start:stop, :, None]).flatten(1)
        ref_jacobian = jacobian * ref_rows.unsqueeze(-1).to(jacobian.dtype)
        gamma = ref_jacobian.transpose(-1, -2) @ ref_jacobian
        gamma = 0.5 * (gamma + gamma.transpose(-1, -2))
        eigenvalues = torch.linalg.eigvalsh(gamma)
        floor = eigenvalues[:, -1] * torch.finfo(gamma.dtype).eps * RANK_FLOOR_FACTOR
        degenerate = ~(eigenvalues[:, 0] > floor) | ~torch.isfinite(eigenvalues).all(dim=1)
        safe = torch.where(degenerate[:, None, None],
                           torch.eye(5, dtype=gamma.dtype).expand_as(gamma), gamma)
        factor, info = torch.linalg.cholesky_ex(safe)
        degenerate |= info != 0
        solved = torch.cholesky_solve(jacobian.transpose(-1, -2), factor)  # [T, 5, 2H]
        whitened = (jacobian * solved.transpose(-1, -2)).sum(-1).reshape(stop - start, n_hits, 2)
        variance = rows.variance.clamp_min(MEASUREMENT_VARIANCE_FLOOR)
        values = whitened * variance
        target_rows = rows.component_mask & target_mask[start:stop, :, None] & ok[..., None]
        good = target_rows & ~degenerate[:, None, None] & torch.isfinite(values) & (values >= 0)
        budget[start:stop] = torch.where(good, values, budget[start:stop])
        rank_deficient[start:stop] = degenerate
    return prediction, valid, budget, rank_deficient


def row_stationary(parameters, batches, geometry):
    """Per-row Newton decrement of the ref-only objective against 1e-6."""
    centers = geometry.centers.to(parameters.dtype)[batches.sensor_idx]
    rotations = geometry.rotations.to(parameters.dtype)[batches.sensor_idx]
    fit = batches.fit_mask
    inputs = MeasurementFitInputs(
        measured=batches.measured_local.to(parameters.dtype),
        variance=batches.covariances.to(parameters.dtype),
        component_mask=torch.stack((fit, fit & batches.loc1_mask), dim=-1),
        centers=centers, rotations=rotations)
    decrement = measurement_newton_decrement(parameters, inputs)
    return torch.isfinite(decrement) & (decrement <= MEASUREMENT_NEWTON_DECREMENT_STATIONARY)


def component_table(batches, geometry, prediction, valid, budget, stationary, truth):
    """One row per measured target component, joined to truth by surface identity."""
    idx_to_key = {index: key for key, index in geometry.id_to_idx.items()}
    track, hit = torch.nonzero(batches.is_target & batches.hit_mask, as_tuple=True)
    sensors = batches.sensor_idx[track, hit].tolist()
    keys = np.array([idx_to_key[s] for s in sensors], dtype=np.int64).reshape(-1, 3)
    event_track = batches.event_track_ids[track].numpy()
    slots = pd.DataFrame({
        "event_nr": event_track[:, 0], "track_nr": event_track[:, 1],
        "volume_id": keys[:, 0], "layer_id": keys[:, 1], "module_id": keys[:, 2],
        "track_index": track.numpy(), "hit_index": hit.numpy(),
    })
    surface_keys = ["event_nr", "track_nr", "volume_id", "layer_id", "module_id"]
    repeated = slots.duplicated(surface_keys, keep=False)
    slots = slots.loc[~repeated]
    merged = slots.merge(truth, on=surface_keys, how="left", validate="one_to_one")
    counts = {"target_slots": int(len(repeated)), "repeated_surface_slots": int(repeated.sum()),
              "missing_truth": int(merged.t_eLOC0.isna().sum())}
    merged = merged.loc[merged.t_eLOC0.notna()]
    t, h = merged.track_index.to_numpy(), merged.hit_index.to_numpy()
    parts = []
    for c, component in enumerate(COMPONENTS):
        mask = batches.hit_mask[t, h] if c == 0 else batches.loc1_mask[t, h]
        mask = mask.numpy()
        data = merged.loc[mask].copy()
        tt, hh = t[mask], h[mask]
        data["component"] = component
        data["layer"] = data.volume_id.astype(str) + ":" + data.layer_id.astype(str)
        data["prediction"] = prediction[tt, hh, c].numpy()
        data["measurement"] = batches.measured_local[tt, hh, c].double().numpy()
        data["truth"] = data[f"t_eLOC{c}"]
        data["V"] = batches.covariances[tt, hh, c].double().clamp_min(MEASUREMENT_VARIANCE_FLOOR).numpy()
        data["P"] = budget[tt, hh, c].numpy()
        data["measurement_residual"] = data.prediction - data.measurement
        data["truth_residual"] = data.prediction - data.truth
        data["measurement2"] = data.measurement_residual**2
        data["truth2"] = data.truth_residual**2
        data["U"] = data.measurement2 - data.V - data.P
        data["truth_excess"] = data.truth2 - data.P
        data["intersection_valid"] = valid[tt, hh].numpy()
        data["refit_stationary"] = stationary[tt].numpy()
        data["complete"] = (data.intersection_valid & data.refit_stationary
                            & np.isfinite(data.P) & np.isfinite(data.measurement2))
        parts.append(data.drop(columns=["t_eLOC0", "t_eLOC1"]))
    return pd.concat(parts, ignore_index=True), counts


# --------------------------------------------------------------------------- statistics
# Ported from 97cd01c scripts/analysis/compare_helix_material_effects.py.

def summarize_cell(frame):
    complete = (len(frame) > 0 and bool(frame["complete"].all())
                and np.isfinite(frame[list(METRICS.values())].to_numpy()).all())
    result = {"n_components": int(len(frame)), "n_events": int(frame.event_nr.nunique()),
              "complete": bool(complete)}
    for metric, column in METRICS.items():
        value = float(frame[column].mean()) if complete else None
        result[metric] = value if value is None or np.isfinite(value) else None
    result["complete"] &= all(result[m] is not None for m in METRICS)
    if not result["complete"]:
        result.update({m: None for m in METRICS})
    if "measurement_residual" in frame and result["complete"]:
        for name in ("measurement", "truth"):
            r = frame[f"{name}_residual"].to_numpy()
            result.update({f"{name}_mean": float(r.mean()), f"{name}_std": float(r.std()),
                           f"{name}_rmse": float(np.sqrt(np.mean(r * r)))})
        result["pull2"] = float((frame.measurement2 / (frame.V + frame.P)).mean())
    return result


def bootstrap_metrics(frames, event_ids, *, replicates=2000, seed=6006):
    """Resample the same generated event pool for every arm; retain empty events.

    Each statistic is a component-weighted mean. Missing reconstructed events
    have zero numerator and denominator, never an invented zero residual.
    """
    events = pd.Index(event_ids)
    if len(events) < 2 or not events.is_unique:
        raise ValueError("bootstrap needs at least two unique generated events")
    result, totals = {}, {}
    columns = list(METRICS.values())
    for group, frame in frames.items():
        if not set(frame.event_nr).issubset(set(event_ids)):
            raise ValueError("reconstruction contains an event absent from generated particles")
        cell = summarize_cell(frame)
        sums = frame.groupby("event_nr")[columns].sum().reindex(events, fill_value=0).to_numpy()
        counts = frame.groupby("event_nr").size().reindex(events, fill_value=0).to_numpy()
        totals[group] = (sums, counts, cell["complete"])
        result[group] = {m: (cell[m], np.full(replicates, np.nan)) for m in METRICS}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, 32):
        size = min(32, replicates - start)
        indices = rng.integers(0, len(events), size=(size, len(events)))
        for group, (sums, counts, complete) in totals.items():
            if not complete:
                continue
            denominator = counts[indices].sum(axis=1)
            numerator = sums[indices].sum(axis=1)
            means = np.divide(numerator, denominator[:, None],
                              out=np.full_like(numerator, np.nan, dtype=float),
                              where=denominator[:, None] > 0)
            for j, metric in enumerate(METRICS):
                result[group][metric][1][start:start + size] = means[:, j]
    return result


def interval(values):
    # Do not narrow the event pool when a rare stratum vanishes in resampling.
    return np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None


def interval_verdict(estimate, ci, *, complete):
    if not complete or estimate is None or ci is None:
        return "not_evaluable"
    return "decreased" if ci[1] < 0 else "increased" if ci[0] > 0 else "insufficient_evidence"


def closure_verdict(estimate, ci, *, complete):
    if not complete or estimate is None or ci is None:
        return "not_evaluable"
    return "statistically_compatible_with_zero" if ci[0] <= 0 <= ci[1] else "not_closed"


CONTRASTS = (
    ("material_effect", ("g1_pt10_material_on", "g2_pt10_material_off"), ("M", "U", "truth_M")),
    ("high_momentum", ("g3_pt100_material_on", "g1_pt10_material_on"), ("M", "U", "truth_M")),
    ("no_material_closure", ("g2_pt10_material_off", None), ("U", "truth_excess")),
)


def contrasts(boot):
    """Rows of (question, metric, estimate, 95% CI, verdict) for one cell."""
    rows = []
    for question, (left, right), metrics in CONTRASTS:
        for metric in metrics:
            estimate, samples = boot[left][metric]
            if right is not None:
                other, other_samples = boot[right][metric]
                estimate = None if estimate is None or other is None else estimate - other
                samples = samples - other_samples
            ci = interval(samples) if estimate is not None else None
            judge = closure_verdict if right is None else interval_verdict
            rows.append({"question": question, "metric": metric, "difference_mm2": estimate,
                         "ci_low": None if ci is None else ci[0],
                         "ci_high": None if ci is None else ci[1],
                         "verdict": judge(estimate, ci, complete=estimate is not None)})
    return rows


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
