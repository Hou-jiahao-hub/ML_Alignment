"""ACTS ``trackstates``/``tracksummary`` ROOT loading into flat hit tensors."""

from __future__ import annotations

from dataclasses import dataclass, field

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot

EPS_WEIGHT = 1e-12
STRIP_VOLUMES = {23, 24, 25}
STRIP_Y_FLOOR = 1.44  # (1.2 mm)^2


@dataclass
class AlignmentInputs:
    hit_sensor_ids: torch.Tensor
    initial_residuals: torch.Tensor
    covariances: torch.Tensor
    # Not read by any solver here; kept as part of the pinned ROOT loader contract.
    track_params: torch.Tensor
    track_chi2_ndf: torch.Tensor
    hit_global_positions: torch.Tensor
    hit_directions: torch.Tensor
    hit_qoverp: torch.Tensor
    hit_volume_ids: torch.Tensor
    hit_layer_ids: torch.Tensor
    hit_module_ids: torch.Tensor
    hit_event_ids: torch.Tensor
    hit_track_ids: torch.Tensor
    measured_local: torch.Tensor | None = None
    loc1_valid: torch.Tensor | None = None
    # Measurement variance BEFORE the STRIP_Y_FLOOR clip is applied; kept as an
    # observable diagnostic and pinned by the ROOT loader contract test.
    raw_covariances: torch.Tensor | None = None
    merged_df: pd.DataFrame | None = field(default=None, repr=False)


def _read_root_columns(path, tree_name, columns, entry_stop=None):
    """Read ROOT columns with PyROOT first; uproot remains a fixture fallback."""
    try:
        import ROOT
    except ImportError:
        ROOT = None
    if ROOT is not None:
        frame = ROOT.RDataFrame(tree_name, str(path))
        requested = list(columns)
        available = {str(name) for name in frame.GetColumnNames()}
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                f"ROOT tree {tree_name!r} is missing columns: {', '.join(missing)}; "
                f"requested columns: {', '.join(requested)}"
            )
        if entry_stop is not None:
            frame = frame.Range(int(entry_stop))
        try:
            return frame.AsNumpy(requested), "pyroot"
        except Exception as error:
            raise ValueError(
                f"PyROOT could not read {tree_name} columns from {path}: {error}"
            ) from error
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise ValueError(f"ROOT tree {tree_name!r} is missing from {path}")
        tree = root_file[tree_name]
        missing = sorted(set(columns) - set(tree.keys()))
        if missing:
            raise ValueError(
                f"ROOT tree {tree_name!r} is missing columns: {', '.join(missing)}"
            )
        return tree.arrays(
            list(columns), entry_stop=entry_stop, library="ak", how=dict
        ), "uproot"


def _root_column_entries(values):
    if isinstance(values, ak.Array):
        return ak.to_list(values)
    return list(values)


def _flatten_root_entries(values):
    entries = _root_column_entries(values)
    if not entries:
        return np.asarray([], dtype=float)
    return np.concatenate([np.asarray(entry) for entry in entries])


def load_variable_information(
    tracksummary_filepath,
    trackstates_filepath,
    id_to_idx_map,
    max_track_entries=None,
    load_summary=True,
):
    # 1. Read tracksummary
    sum_df = None
    if load_summary:
        sum_keys = ["event_nr", "track_nr", "chi2Sum", "NDF"]
        sum_arrays, summary_backend = _read_root_columns(
            tracksummary_filepath,
            "tracksummary",
            sum_keys,
            entry_stop=max_track_entries,
        )
        track_entries = _root_column_entries(sum_arrays["track_nr"])
        summary_is_jagged = bool(
            track_entries and np.asarray(track_entries[0]).ndim > 0
        )
        if summary_is_jagged:
            summary_counts = np.asarray(
                [len(np.asarray(entry)) for entry in track_entries], dtype=np.int64
            )
            event_values = np.asarray(sum_arrays["event_nr"])
            summary_data = {
                "event_nr": np.repeat(event_values, summary_counts),
                "track_nr": _flatten_root_entries(sum_arrays["track_nr"]),
                "chi2Sum": _flatten_root_entries(sum_arrays["chi2Sum"]),
                "NDF": _flatten_root_entries(sum_arrays["NDF"]),
            }
        else:
            summary_data = {
                name: np.asarray(sum_arrays[name]) for name in sum_keys
            }
        sum_df = pd.DataFrame(summary_data).reset_index(drop=True)
        sum_df["event_nr"] = sum_df["event_nr"].astype(np.int64)
        sum_df["track_nr"] = sum_df["track_nr"].astype(np.int64)
        sum_df = sum_df.rename(columns={"chi2Sum":"chi2", "NDF": "ndf"})
        sum_df["chi2_ndf"] = sum_df["chi2"] / sum_df["ndf"]
        print(f"Tracks loaded: {len(sum_df)} trajectories ({summary_backend})")

    # 2. Load per-state columns only.
    # Background: the ROOT trackstates tree has two array-length classes per event:
    #   - per-state  (N_states / event): volume_id, l_x_hit, eLOC0_smt, g_x_smt, ...
    #   - per-measurement (N_meas / event, N_meas <= N_states): dim_hit, err_x_hit, err_y_hit
    #   - per-event  (scalar): event_nr, track_nr
    # Loading all classes together with ak.to_dataframe() truncates to N_meas rows and
    # fails to broadcast event_nr. Load per-state columns here, then expand true
    # measurement errors onto the measured states.
    per_state_keys = [
        "volume_id", "layer_id", "module_id",
        "l_x_hit", "l_y_hit",             # NaN for hole/material states
        "eLOC0_smt", "eLOC1_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
        "px_smt", "py_smt", "pz_smt",
        "ePHI_smt", "eTHETA_smt", "eQOP_smt",
    ]
    state_keys = per_state_keys + [
        "err_x_hit", "err_y_hit", "event_nr", "track_nr"
    ]
    state_columns, states_backend = _read_root_columns(
        trackstates_filepath,
        "trackstates",
        state_keys,
        entry_stop=max_track_entries,
    )
    states_df = pd.DataFrame(
        {name: _flatten_root_entries(state_columns[name]) for name in per_state_keys}
    )

    # Load true measurement errors separately: err_x_hit / err_y_hit are per-measurement
    # (N_meas ≤ N_states per track).  Expand them to per-state length by inserting NaN
    # at hole states, keyed by the per-state l_x_hit NaN pattern.
    # Both branches are required: the variance columns below consume them
    # unconditionally, so there is no posterior-variance fallback to degrade to.
    # A length mismatch used to leave the whole entry as NaN, after which the
    # affected hits were dropped silently by the dropna() further down.
    try:
        lx_jagged = _root_column_entries(state_columns["l_x_hit"])
        errx_jagged = _root_column_entries(state_columns["err_x_hit"])
        erry_jagged = _root_column_entries(state_columns["err_y_hit"])
        errx_flat = []
        erry_flat = []
        for i in range(len(lx_jagged)):
            lx_tr = np.asarray(lx_jagged[i])
            is_meas = ~np.isnan(lx_tr)
            n_st = len(lx_tr)
            n_meas = int(is_meas.sum())
            ex = np.full(n_st, np.nan)
            ey = np.full(n_st, np.nan)
            em_x = np.asarray(errx_jagged[i])
            em_y = np.asarray(erry_jagged[i])
            for name, values in (("err_x_hit", em_x), ("err_y_hit", em_y)):
                if len(values) != n_meas:
                    raise ValueError(
                        f"{name} length mismatch at entry {i}: "
                        f"got {len(values)}, expected {n_meas}"
                    )
            ex[is_meas] = em_x
            ey[is_meas] = em_y
            errx_flat.extend(ex.tolist())
            erry_flat.extend(ey.tolist())
        states_df["err_x_hit"] = errx_flat
        states_df["err_y_hit"] = erry_flat
        print(
            "  Loaded true measurement errors (err_x_hit / err_y_hit) "
            f"via per-state expansion ({states_backend})."
        )
    except (KeyError, ValueError) as exc:
        # Interpolate the cause so the entry/length detail stays visible at the
        # top level; `raise ... from exc` alone would hide it behind __cause__.
        raise ValueError(
            f"Required err_x_hit/err_y_hit measurement errors could not be loaded: {exc}"
        ) from exc

    # 3. Broadcast per-event event_nr / track_nr to per-state level.
    volume_entries = _root_column_entries(state_columns["volume_id"])
    counts = np.asarray([len(entry) for entry in volume_entries], dtype=np.int64)
    ev_flat = np.asarray(state_columns["event_nr"])
    tr_flat = np.asarray(state_columns["track_nr"])
    states_df["event_nr"] = np.repeat(ev_flat, counts)
    states_df["track_nr"] = np.repeat(tr_flat, counts)
    states_df["event_nr"] = states_df["event_nr"].astype(np.int64)
    states_df["track_nr"] = states_df["track_nr"].astype(np.int64)

    # 4. Inner merge: map track-level chi2/ndf onto each hit state.
    if sum_df is None:
        merged_df = states_df
        merged_df["chi2_ndf"] = 0.0
    else:
        merged_df = pd.merge(states_df, sum_df, on=["event_nr", "track_nr"], how="inner")

    # 5. Remove hole/material states: l_x_hit is NaN for states without a measurement.
    merged_df = merged_df[merged_df["l_x_hit"].notna()].copy()

    # 6. Map sensor ID.
    sensor_index = pd.Series(id_to_idx_map)
    sensor_index.index = pd.MultiIndex.from_tuples(
        sensor_index.index, names=["volume_id", "layer_id", "module_id"]
    )
    hit_index = pd.MultiIndex.from_frame(
        merged_df[["volume_id", "layer_id", "module_id"]]
    )
    merged_df["sensor_idx"] = hit_index.map(sensor_index)

    # 7. ACTS-like alignment residuals.
    # AlignmentAlgorithm builds residuals from the measurement and the smoothed state.
    merged_df["res_eLOC0"] = merged_df["l_x_hit"] - merged_df["eLOC0_smt"]
    merged_df["res_eLOC1"] = merged_df["l_y_hit"] - merged_df["eLOC1_smt"]

    meas_var0 = (merged_df["err_x_hit"] ** 2).clip(lower=EPS_WEIGHT)
    meas_var1 = (merged_df["err_y_hit"] ** 2).clip(lower=EPS_WEIGHT)
    merged_df["loc1_valid"] = merged_df["err_y_hit"].notna()
    merged_df["measured_loc1"] = merged_df["l_y_hit"].fillna(0.0)
    merged_df["raw_var_eLOC0"] = meas_var0
    merged_df["raw_var_eLOC1"] = meas_var1.fillna(1.0)
    merged_df["var_eLOC0"] = meas_var0
    merged_df["var_eLOC1"] = meas_var1
    print("  Using ACTS-like residuals: l_x/y_hit - eLOC0/1_smt .")

    # 8. Handle 1D strip hits: l_y_hit is NaN for 1D measurements.
    # Set LOC1 residual = 0 and weight = 0 so it contributes nothing to the normal equation.
    is_1d = merged_df["err_y_hit"].isna()
    merged_df.loc[is_1d, "res_eLOC1"] = 0.0
    merged_df.loc[is_1d, "var_eLOC1"] = 1.0
    n_1d = int(is_1d.sum())
    n_2d = int((~is_1d).sum())
    print(f"  Hit breakdown: {n_2d} 2D measurement, {n_1d} 1D measurement")

    # Physics floor for strip detector Y direction.
    # Vol 23/24/25 (ShortStrip barrel/endcap): intrinsic Y resolution ~1.2mm.
    # the optimizer to over-correct Y.  Clip var_eLOC1 to at least (1.2mm)^2.
    is_strip_vol = merged_df["volume_id"].isin(STRIP_VOLUMES)
    before_floor = merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].mean() if is_strip_vol.any() else 0.0
    strip_2d = is_strip_vol & ~is_1d
    merged_df.loc[strip_2d, "var_eLOC1"] = merged_df.loc[strip_2d, "var_eLOC1"].clip(lower=STRIP_Y_FLOOR)
    if is_strip_vol.any():
        after_floor = merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].mean()
        print(f"  Strip Y variance floor applied: mean var_LOC1 {before_floor:.4f} -> {after_floor:.4f} mm^2")

    # 9. Drop remaining invalids (sensor not in alignment set, NaN smoothed params).
    required_cols = [
        "sensor_idx", "res_eLOC0", "res_eLOC1",
        "var_eLOC0", "var_eLOC1",  "eLOC0_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
    ]
    merged_df = merged_df.dropna(subset=required_cols)
    merged_df = merged_df[merged_df["module_id"] > 0].copy()
    merged_df["r_xy"] = np.hypot(merged_df["g_x_smt"], merged_df["g_y_smt"])
    merged_df["abs_g_z"] = merged_df["g_z_smt"].abs()
    merged_df = merged_df.sort_values(
        [
            "event_nr", "track_nr", "r_xy", "abs_g_z",
            "volume_id", "layer_id", "module_id",
            "g_x_smt", "g_y_smt", "g_z_smt",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    num_hits = len(merged_df)
    print(f"Feature alignment complete! A total of {num_hits} valid tensor data points have been constructed")

    # 10. Build PyTorch Tensors.
    hit_sensor_ids = torch.tensor(merged_df["sensor_idx"].values, dtype=torch.long)

    # Alignment residuals: r = m_k - H * x_smt
    initial_residuals = torch.tensor(
        merged_df[["res_eLOC0", "res_eLOC1"]].values, dtype=torch.float32)

    # Measurement variances: V = err_x/y_hit^2 (diagonal LOC0/LOC1 components)
    covariances = torch.tensor(
        merged_df[["var_eLOC0", "var_eLOC1"]].values, dtype=torch.float32)
    raw_covariances = torch.tensor(
        merged_df[["raw_var_eLOC0", "raw_var_eLOC1"]].values, dtype=torch.float32)
    measured_local = torch.tensor(
        merged_df[["l_x_hit", "measured_loc1"]].values, dtype=torch.float32)
    loc1_valid = torch.tensor(merged_df["loc1_valid"].values, dtype=torch.bool)

    # Smoothed track parameters as neural-net input features.
    # Jacobian is also evaluated at the smoothed position/direction (consistent
    # with AlignmentEngine.hpp:229-233 using freeSmoothed).
    track_params = torch.tensor(
        merged_df[["eLOC0_smt", "eLOC1_smt", "eTHETA_smt", "ePHI_smt", "eQOP_smt"]].values,
        dtype=torch.float32)

    track_chi2_ndf = torch.tensor(merged_df[["chi2_ndf"]].values, dtype=torch.float32)

    hit_global_positions = torch.tensor(
        merged_df[["g_x_smt", "g_y_smt", "g_z_smt"]].values, dtype=torch.float32)
    hit_directions = torch.tensor(
        merged_df[["px_smt", "py_smt", "pz_smt"]].values, dtype=torch.float32)
    hit_qoverp = torch.tensor(merged_df[["eQOP_smt"]].values, dtype=torch.float32)

    hit_volume_ids = torch.tensor(merged_df["volume_id"].values, dtype=torch.long)
    hit_layer_ids  = torch.tensor(merged_df["layer_id"].values,  dtype=torch.long)
    hit_module_ids = torch.tensor(merged_df["module_id"].values, dtype=torch.long)
    hit_event_ids = torch.tensor(merged_df["event_nr"].values, dtype=torch.long)
    hit_track_ids = torch.tensor(merged_df["track_nr"].values, dtype=torch.long)

    inputs = AlignmentInputs(
        hit_sensor_ids=hit_sensor_ids,
        initial_residuals=initial_residuals,
        covariances=covariances,
        track_params=track_params,
        track_chi2_ndf=track_chi2_ndf,
        hit_global_positions=hit_global_positions,
        hit_directions=hit_directions,
        hit_qoverp=hit_qoverp,
        hit_volume_ids=hit_volume_ids,
        hit_layer_ids=hit_layer_ids,
        hit_module_ids=hit_module_ids,
        hit_event_ids=hit_event_ids,
        hit_track_ids=hit_track_ids,
        measured_local=measured_local,
        loc1_valid=loc1_valid,
        raw_covariances=raw_covariances,
        merged_df=merged_df,
    )
    return inputs
