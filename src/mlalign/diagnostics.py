"""Residual reporting and truth-geometry diagnostics for alignment runs."""

from __future__ import annotations

import json
import os
import re

import numpy as np

from .config import DEFAULT_TARGET_LAYERS as TARGET_LAYERS


def _parse_target_layers(target_layers):
    if target_layers is None or target_layers == "":
        return TARGET_LAYERS
    if not isinstance(target_layers, str):
        return tuple((int(volume), int(layer)) for volume, layer in target_layers)
    parsed = []
    for item in target_layers.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Target layer {item!r} must have format volume:layer.")
        parsed.append((int(parts[0]), int(parts[1])))
    if not parsed:
        raise ValueError("At least one target layer must be selected.")
    return tuple(parsed)


def _rotation_matrix_to_euler(rotation):
    rotation = np.asarray(rotation, dtype=float)
    ry = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    rx = np.arctan2(rotation[2, 1], rotation[2, 2])
    rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    return float(rx), float(ry), float(rz)


def _as_numpy(values):
    """Detach torch tensors without forcing a dtype promotion.

    Callers mix torch tensors and ndarrays; converting only some of them would
    leave `ndarray & Tensor` mixed operations whose semantics depend on the
    conversion rules.  No `dtype=` is passed: numpy's `float` is float64 and the
    surface residuals are float32, so promoting here would change the reported
    statistics.
    """
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


def _stat_pair(values):
    """(mean, RMS) of a 1-D array, or (nan, nan) when it is empty.

    RMS is sqrt(mean(x^2)) — not np.std.  A detector misalignment shows up as a
    non-zero residual *mean*, which np.std subtracts away.
    """
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.sqrt(np.mean(values ** 2)))


def _fmt_mm(value):
    return "       N/A" if not np.isfinite(value) else f"{value:>9.4f}mm"


def print_target_residual_stats(
    initial_residuals, hit_volume_ids, hit_layer_ids, target_layers, loc1_valid=None
):
    res_np = _as_numpy(initial_residuals)
    vols_np = _as_numpy(hit_volume_ids)
    lays_np = _as_numpy(hit_layer_ids)
    if loc1_valid is None:
        loc1_np = np.ones(len(res_np), dtype=bool)
    else:
        loc1_np = _as_numpy(loc1_valid).astype(bool)

    print("\n=== Target-layer ACTS KF-smoothed residuals before solve ===")
    print(f"  {'Layer':<20}  {'N0':>8}  {'N1':>8}  {'mean LOC0':>12}  {'RMS LOC0':>10}"
          f"  {'mean LOC1':>12}  {'RMS LOC1':>10}")
    for vol, lay in _parse_target_layers(target_layers):
        mask = (vols_np == vol) & (lays_np == lay)
        if mask.sum() == 0:
            print(f"  vol={vol} lay={lay:<6}  no valid hits")
            continue
        mask1 = mask & loc1_np
        mean0, rms0 = _stat_pair(res_np[mask, 0])
        mean1, rms1 = _stat_pair(res_np[mask1, 1])
        label = f"vol={vol} lay={lay}"
        cell = lambda v, w: f"{'N/A':>{w}}" if not np.isfinite(v) else f"{v:>{w}.5f}"
        print(f"  {label:<20}  {int(mask.sum()):>8}  {int(mask1.sum()):>8}  "
              f"{cell(mean0, 12)}  {cell(rms0, 10)}  {cell(mean1, 12)}  {cell(rms1, 10)}")


def print_per_layer_residual_diagnostics(
    init_res_np, final_res, vols_np, lays_np, target_layers,
    return_metrics=False,
    component_mask=None,
):
    """Per-layer before/after residual tables.

    ``component_mask`` is an ``(N, 2)`` boolean array marking which residual
    components are real measurements.  Without it the zeros written into LOC1
    for 1-D strip hits are averaged in and dilute the Y statistics.
    New parameters are appended so existing positional calls keep working.
    """
    init_res_np = _as_numpy(init_res_np)
    final_res = _as_numpy(final_res)
    vols_np = _as_numpy(vols_np)
    lays_np = _as_numpy(lays_np)

    if init_res_np.shape != final_res.shape:
        raise ValueError(
            f"init_res_np shape {init_res_np.shape} != final_res shape {final_res.shape}"
        )
    if init_res_np.ndim != 2 or init_res_np.shape[1] != 2:
        raise ValueError(f"residuals must have shape (N, 2), got {init_res_np.shape}")
    for name, array in (("init_res_np", init_res_np), ("final_res", final_res)):
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    n_hits = init_res_np.shape[0]
    if len(vols_np) != n_hits or len(lays_np) != n_hits:
        raise ValueError(
            f"vols_np ({len(vols_np)}) and lays_np ({len(lays_np)}) must both have "
            f"length {n_hits}"
        )

    if component_mask is None:
        component_mask = np.ones(init_res_np.shape, dtype=bool)
    else:
        component_mask = _as_numpy(component_mask).astype(bool)
    if component_mask.shape != init_res_np.shape:
        raise ValueError(
            f"component_mask shape {component_mask.shape} != residual shape "
            f"{init_res_np.shape}"
        )

    header = (f"  {'Layer':<20}  {'X_before':>10}  {'X_after':>10}"
              f"  {'Y_before':>10}  {'Y_after':>10}")
    metrics = {}
    layer_masks = {}
    for vol, lay in target_layers:
        layer = (vols_np == vol) & (lays_np == lay)
        layer_masks[(vol, lay)] = (layer & component_mask[:, 0], layer & component_mask[:, 1])

    print("\n=== Per-layer residual mean (training data) ===")
    print(header)
    for vol, lay in target_layers:
        mask0, mask1 = layer_masks[(vol, lay)]
        n0, n1 = int(mask0.sum()), int(mask1.sum())
        x_bef, rms_x_bef = _stat_pair(init_res_np[mask0, 0])
        x_aft, rms_x_aft = _stat_pair(final_res[mask0, 0])
        y_bef, rms_y_bef = _stat_pair(init_res_np[mask1, 1])
        y_aft, rms_y_aft = _stat_pair(final_res[mask1, 1])
        label = f"vol={vol} lay={lay} (n0={n0}, n1={n1})"
        print(f"  {label:<28}  {_fmt_mm(x_bef)}  {_fmt_mm(x_aft)}"
              f"  {_fmt_mm(y_bef)}  {_fmt_mm(y_aft)}")
        metrics[(vol, lay)] = {
            "n_hits": n0,          # retained alias for existing callers
            "n_hits_loc0": n0,
            "n_hits_loc1": n1,
            "mean_before_loc0": x_bef, "mean_after_loc0": x_aft,
            "mean_before_loc1": y_bef, "mean_after_loc1": y_aft,
            "rms_before_loc0": rms_x_bef, "rms_after_loc0": rms_x_aft,
            "rms_before_loc1": rms_y_bef, "rms_after_loc1": rms_y_aft,
        }

    print("\n=== Per-layer residual RMS (training data) ===")
    print(header)
    for vol, lay in target_layers:
        entry = metrics[(vol, lay)]
        label = f"vol={vol} lay={lay} (n0={entry['n_hits_loc0']}, n1={entry['n_hits_loc1']})"
        print(f"  {label:<28}  {_fmt_mm(entry['rms_before_loc0'])}"
              f"  {_fmt_mm(entry['rms_after_loc0'])}"
              f"  {_fmt_mm(entry['rms_before_loc1'])}"
              f"  {_fmt_mm(entry['rms_after_loc1'])}")
    print()
    return metrics if return_metrics else None

def validate_against_truth(transforms_list, predicted_deltas, nominal_json,
                           target_layers):
    """Compare ML correction against the TRUE misalignment per sensor (Iteration 4).

    The real ground truth: applying δα to the misaligned surface must recover the
    nominal one.  In ml convention (new_c = c_mis + R_mis·δt, new_R = R_mis·Rδ):
        δt_true  = R_misᵀ (c_nom - c_mis)        (local tx, ty, tz)
        Rδ_true  = R_misᵀ R_nom                  → rz = atan2(Rδ[1,0], Rδ[0,0])
    Reports per-DOF correlation + recovery fraction over each target layer; this is
    the metric that actually decides whether the local signal is recoverable.
    """
    if not os.path.exists(nominal_json):
        print(f"  [truth] nominal {nominal_json} not found — skipping.")
        return {}
    nom = {}
    with open(nominal_json, "r", encoding="utf-8") as f:
        nominal_data = json.load(f)
    for it in nominal_data:
        m = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
        if m:
            nom[tuple(map(int, m.groups()))] = (np.array(it["Translation"], float),
                                                np.array(it["RotationMatrix"], float))
    print("\n=== ML vs TRUE misalignment (local frame, per sensor) ===")
    metrics = {}
    for vol, lay in target_layers:
        true_v, pred_v = [], []
        for i, it in enumerate(transforms_list):
            m = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
            if not m:
                continue
            key = tuple(map(int, m.groups()))
            if key[0] != vol or key[1] != lay or key not in nom:
                continue
            c_mis = np.array(it["Translation"], float); R_mis = np.array(it["RotationMatrix"], float)
            c_nom, R_nom = nom[key]
            dt_true = R_mis.T @ (c_nom - c_mis)
            Rd = R_mis.T @ R_nom
            rx_true, ry_true, rz_true = _rotation_matrix_to_euler(Rd)
            true_v.append([*dt_true, rx_true, ry_true, rz_true])
            p = predicted_deltas[i]
            pred_v.append(p[:6])
        if not true_v:
            continue
        true_v = np.array(true_v); pred_v = np.array(pred_v)
        names = ["tx", "ty", "tz", "rx", "ry", "rz"]
        print(f"  vol={vol} lay={lay} (n={len(true_v)})")
        layer_metrics = {}
        for j, nm in enumerate(names):
            t, p = true_v[:, j], pred_v[:, j]
            corr = np.corrcoef(t, p)[0, 1] if np.std(t) > 0 and np.std(p) > 0 else float("nan")
            recov = (p @ t) / (t @ t) if (t @ t) > 0 else float("nan")  # projection / recovery fraction
            unit = "mm" if j < 3 else "rad"
            layer_metrics[nm] = {
                "corr": float(corr),
                "recovery": float(recov),
                "true_rms": float(np.sqrt((t ** 2).mean())),
                "pred_rms": float(np.sqrt((p ** 2).mean())),
                # RMS of the residual truth error, the quantity a candidate arm
                # has to avoid making worse.  ``true_rms`` and ``pred_rms`` are
                # magnitudes of the injection and of the correction; neither of
                # them gets smaller when the answer gets better.  Point estimate
                # only -- the interval comes from a paired bootstrap over
                # sensors in scripts/analysis/compare_truth_rms.py.
                "err_rms": float(np.sqrt(((p - t) ** 2).mean())),
                "n_sensors": int(len(t)),
                "unit": unit,
            }
            print(f"    {nm}: corr={corr:+.3f}  recovery={recov:+.3f}  "
                  f"|true|rms={np.sqrt((t**2).mean()):.4f}{unit}  |pred|rms={np.sqrt((p**2).mean()):.4f}{unit}  "
                  f"|err|rms={np.sqrt(((p-t)**2).mean()):.4f}{unit}")
        metrics[(vol, lay)] = layer_metrics
    return metrics
