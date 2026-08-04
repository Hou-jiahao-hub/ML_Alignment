from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
import uproot
import pandas as pd
import awkward as ak
import importlib
import importlib.machinery
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from alignment_artifact_layout import (
    TARGET_LAYERS,
    alignment_json_path,
    reconstruction_root_path,
    target_layers_spec,
)

EPS_COVARIANCE = 1e-8
EPS_WEIGHT = 1e-12
MIN_HIT_ERROR = 1e-4
EPS_DIRECTION_NORM = 1e-9
FULL_MATRIX_TOLERANCE = 1e-4
STRIP_VOLUMES = {23, 24, 25}
STRIP_Y_FLOOR = 1.44  # (1.2 mm)^2
DEFAULT_TARGET_LAYERS = TARGET_LAYERS
DEFAULT_ACTIVE_DOF = (0, 1, 3, 4, 5)
DEFAULT_EPOCHS = 300
DEFAULT_JACOBIAN_UPDATE_INTERVAL = 20
DEFAULT_LAMBDA_L2 = 1e-5
DEFAULT_ADAM_LR = 1e-2
DEFAULT_ADAM_ETA_MIN = 1e-5
ADAM_CLAMP_T = 1.5
ADAM_CLAMP_R = 0.05
HIT_WEIGHT_FLOOR = 0.3
HIT_QUALITY_INPUT_SIZE = 9
HIT_QUALITY_HIDDEN_ONE = 32
HIT_QUALITY_HIDDEN_TWO = 16
HIT_QUALITY_OUTPUT_SIZE = 1

# matplotlib is imported lazily in run_alignment_pipeline only when an Adam
# training curve is actually written.


@dataclass
class AlignmentInputs:
    hit_sensor_ids: torch.Tensor
    initial_residuals: torch.Tensor
    covariances: torch.Tensor
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
    merged_df: pd.DataFrame | None = field(default=None, repr=False)


@dataclass
class AlignmentRunConfig:
    input_geom: str = str(alignment_json_path("alignment_output", "misalignment_applied"))
    states_root: str = str(
        reconstruction_root_path(
            "reconstruction_output",
            "misaligned",
            "trackstates_kf",
        )
    )
    summary_root: str = str(
        reconstruction_root_path(
            "reconstruction_output",
            "misaligned",
            "tracksummary_kf",
        )
    )
    output_geom: str = str(
        alignment_json_path(
            "alignment_output",
            "aligned_transforms",
            stem_prefix="ml",
        )
    )
    nominal_file: str = "odd_transforms.json"
    iter_label: str | None = None
    make_plot: bool = True
    active_dof: tuple[int, ...] | str = DEFAULT_ACTIVE_DOF
    target_layers: tuple[tuple[int, int], ...] | str = DEFAULT_TARGET_LAYERS
    fix_reference: bool = True
    epochs: int = DEFAULT_EPOCHS
    jacobian_update_interval: int = DEFAULT_JACOBIAN_UPDATE_INTERVAL
    lambda_l2: float = DEFAULT_LAMBDA_L2
    debug_dataflow: bool = False


def _debug_prefix(name):
    return f"[debug:dataflow] {name}"


def _to_numpy_for_debug(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy(), str(values.dtype)
    arr = np.asarray(values)
    return arr, str(arr.dtype)


def debug_tensor_summary(name, values, enabled=False, sample_rows=5):
    if not enabled:
        return

    arr, dtype_name = _to_numpy_for_debug(values)
    flat = arr.reshape(-1) if arr.size else arr
    finite_mask = np.isfinite(flat) if np.issubdtype(flat.dtype, np.number) else np.zeros(flat.shape, dtype=bool)
    finite = flat[finite_mask]
    nan_count = int(np.isnan(flat).sum()) if np.issubdtype(flat.dtype, np.number) else 0
    inf_count = int(np.isinf(flat).sum()) if np.issubdtype(flat.dtype, np.number) else 0

    stats = [
        f"shape={arr.shape}",
        f"dtype={dtype_name}",
        f"nan={nan_count}",
        f"inf={inf_count}",
    ]
    if finite.size:
        stats.extend([
            f"finite_min={float(np.min(finite)):.6f}",
            f"finite_max={float(np.max(finite)):.6f}",
            f"finite_mean={float(np.mean(finite)):.6f}",
            f"finite_std={float(np.std(finite)):.6f}",
        ])
    else:
        stats.append("finite_values=0")
    print(f"{_debug_prefix(name)} " + " | ".join(stats))

    if arr.ndim > 0 and arr.shape[0] > 0 and sample_rows > 0:
        rows = arr[:sample_rows]
        print(f"{_debug_prefix(name)} sample_first_{min(sample_rows, arr.shape[0])}: {rows}")


def debug_dataframe_summary(name, df, enabled=False, sample_rows=5):
    if not enabled:
        return
    print(f"{_debug_prefix(name)} rows={len(df)} columns={len(df.columns)}")
    if len(df.columns):
        print(f"{_debug_prefix(name)} columns={list(df.columns)}")
    if len(df) and sample_rows > 0:
        print(f"{_debug_prefix(name)} sample_first_{min(sample_rows, len(df))}:")
        print(df.head(sample_rows).to_string(index=False))


def debug_layer_distribution(name, hit_volume_ids, hit_layer_ids, enabled=False):
    if not enabled:
        return
    vols, _ = _to_numpy_for_debug(hit_volume_ids)
    lays, _ = _to_numpy_for_debug(hit_layer_ids)
    print(f"{_debug_prefix(name)} layer distribution")
    if len(vols) == 0:
        print(f"{_debug_prefix(name)} no hits")
        return
    keys, counts = np.unique(np.stack((vols, lays), axis=1), axis=0, return_counts=True)
    for (vol, lay), count in zip(keys, counts):
        print(f"  vol={int(vol)} lay={int(lay)} count={int(count)}")


def debug_geometry_summary(transforms_list, initial_positions, id_to_idx_map, enabled=False):
    if not enabled:
        return
    debug_tensor_summary("geometry.initial_positions", initial_positions, enabled=True)
    sample_ids = list(id_to_idx_map.items())[:5]
    print(f"{_debug_prefix('geometry.id_to_idx')} entries={len(id_to_idx_map)} sample={sample_ids}")
    sample_transforms = [
        {
            "idx": idx,
            "ID": item.get("ID"),
            "Translation": item.get("Translation"),
        }
        for idx, item in enumerate(transforms_list[:5])
    ]
    print(f"{_debug_prefix('geometry.transforms')} sample={sample_transforms}")


def debug_alignment_inputs(name, inputs, enabled=False):
    if not enabled:
        return
    debug_tensor_summary(f"{name}.hit_sensor_ids", inputs.hit_sensor_ids, enabled=True)
    debug_tensor_summary(f"{name}.initial_residuals", inputs.initial_residuals, enabled=True)
    debug_tensor_summary(f"{name}.covariances", inputs.covariances, enabled=True)
    debug_tensor_summary(f"{name}.track_params", inputs.track_params, enabled=True)
    debug_tensor_summary(f"{name}.track_chi2_ndf", inputs.track_chi2_ndf, enabled=True)
    debug_tensor_summary(f"{name}.hit_global_positions", inputs.hit_global_positions, enabled=True)
    debug_tensor_summary(f"{name}.hit_directions", inputs.hit_directions, enabled=True)
    debug_tensor_summary(f"{name}.hit_qoverp", inputs.hit_qoverp, enabled=True)
    debug_layer_distribution(f"{name}", inputs.hit_volume_ids, inputs.hit_layer_ids, enabled=True)

# =====================================================================
# Milestone 1: PyTorch class
# =====================================================================

class AlignmentEngine(nn.Module):
    def __init__(self, num_sensors, use_hit_weights=False):
        super().__init__()
        self.num_sensors = num_sensors
        self.use_hit_weights = use_hit_weights

        self.alignment_params = nn.Parameter(torch.zeros(num_sensors, 6))

        if self.use_hit_weights:
            self.hit_quality_net = nn.Sequential(
                nn.Linear(HIT_QUALITY_INPUT_SIZE, HIT_QUALITY_HIDDEN_ONE),
                nn.GELU(),
                nn.Linear(HIT_QUALITY_HIDDEN_ONE, HIT_QUALITY_HIDDEN_TWO),
                nn.GELU(),
                nn.Linear(HIT_QUALITY_HIDDEN_TWO, HIT_QUALITY_OUTPUT_SIZE),
            )

    def forward(self, hit_sensor_ids, track_params, sensor_center_positions, track_chi2_ndf, H_geom_batch):
        current_params = self.alignment_params[hit_sensor_ids]
        residual_shift = torch.bmm(H_geom_batch, current_params.unsqueeze(-1)).squeeze(-1)

        if self.use_hit_weights:
            hit_sensor_center_positions = sensor_center_positions[hit_sensor_ids]
            advanced_features = torch.cat([track_params, hit_sensor_center_positions, track_chi2_ndf], dim=-1)
            raw_logits = self.hit_quality_net(advanced_features)
            hit_weights = HIT_WEIGHT_FLOOR + (1.0 - HIT_WEIGHT_FLOOR) * torch.sigmoid(raw_logits)
        else:
            hit_weights = torch.ones(len(hit_sensor_ids), 1, device=residual_shift.device)

        return residual_shift, hit_weights

# =====================================================================
# Milestone 1.2: Physics unsupervised Loss (Introducing ML dynamic weights)
# =====================================================================

def normalize_hit_weights(hit_weights, hit_volume_ids=None, hit_layer_ids=None):
    if hit_volume_ids is None and hit_layer_ids is None:
        return hit_weights / (torch.mean(hit_weights) + EPS_WEIGHT)
    if hit_volume_ids is None or hit_layer_ids is None:
        raise ValueError("hit_volume_ids and hit_layer_ids must be provided together.")
    if len(hit_weights) != len(hit_volume_ids) or len(hit_weights) != len(hit_layer_ids):
        raise ValueError("Hit weight and layer identifier tensors must have matching lengths.")

    volume_ids = hit_volume_ids.to(device=hit_weights.device)
    layer_ids = hit_layer_ids.to(device=hit_weights.device)
    layer_keys = torch.stack((volume_ids, layer_ids), dim=1)
    unique_layers = torch.unique(layer_keys, dim=0)

    normalized = torch.zeros_like(hit_weights)
    for vol, lay in unique_layers:
        layer_mask = (volume_ids == vol) & (layer_ids == lay)
        layer_scale = torch.mean(hit_weights[layer_mask]) + EPS_WEIGHT
        layer_weight = layer_mask.to(hit_weights.dtype).unsqueeze(-1)
        normalized = normalized + layer_weight * (hit_weights / layer_scale)
    return normalized


def physics_informed_loss(
    initial_residuals,
    residual_shift,
    covariances,
    hit_weights,
    hit_mask=None,
    hit_volume_ids=None,
    hit_layer_ids=None,
):
    new_residuals = initial_residuals + residual_shift
    raw_chi2 = (new_residuals ** 2) / torch.clamp(covariances, min=EPS_COVARIANCE)

    if hit_mask is not None:
        raw_chi2 = raw_chi2[hit_mask]
        hit_weights = hit_weights[hit_mask]
        if hit_volume_ids is not None:
            hit_volume_ids = hit_volume_ids[hit_mask]
        if hit_layer_ids is not None:
            hit_layer_ids = hit_layer_ids[hit_mask]

    w_normalized = normalize_hit_weights(hit_weights, hit_volume_ids, hit_layer_ids)
    chi2_loss = torch.mean(w_normalized * raw_chi2)
    return chi2_loss

# =====================================================================
# Milestone 2: Simulate pybind11 Data
# =====================================================================
def load_geometry_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data or "Translation" not in data[0]:
        raise ValueError(f"The format of {filepath} is incorrect. Please check the JSON structure.")

    transforms_list = data
    num_sensors = len(transforms_list)

    positions = []
    id_to_idx = {}
    for i, item in enumerate(transforms_list):
        positions.append(item["Translation"])
        match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", item.get("ID", ""))
        if match:
            vol, lay, sen = map(int, match.groups())
            id_to_idx[(vol, lay, sen)] = i

    return transforms_list, num_sensors, torch.tensor(positions, dtype=torch.float32), id_to_idx

def parse_active_dof(active_dof):
    """Parse active alignment DOF from CLI text or an iterable of integers."""
    names = {
        "tx": 0, "ty": 1, "tz": 2,
        "rx": 3, "alpha": 3,
        "ry": 4, "beta": 4,
        "rz": 5, "gamma": 5,
    }
    if active_dof is None or active_dof == "":
        return DEFAULT_ACTIVE_DOF
    if isinstance(active_dof, str):
        raw_items = [item.strip().lower() for item in active_dof.split(",") if item.strip()]
    else:
        raw_items = list(active_dof)

    parsed = []
    for item in raw_items:
        if isinstance(item, str) and item in names:
            value = names[item]
        else:
            value = int(item)
        if value < 0 or value > 5:
            raise ValueError(f"Active DOF {value} is outside the valid range [0, 5].")
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("At least one active DOF must be selected.")
    return tuple(parsed)


def parse_target_layers(target_layers):
    """Parse target layer spec like '16:6,23:4' into ((16, 6), (23, 4))."""
    if target_layers is None or target_layers == "":
        return DEFAULT_TARGET_LAYERS
    if not isinstance(target_layers, str):
        return tuple((int(vol), int(lay)) for vol, lay in target_layers)

    parsed = []
    for item in target_layers.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Target layer '{item}' must have format volume:layer.")
        parsed.append((int(parts[0]), int(parts[1])))
    if not parsed:
        raise ValueError("At least one target layer must be selected.")
    return tuple(parsed)


ALIGN_DERIVLIB_MODULE = "align_deriv_pb"
ALIGN_DERIVLIB_BUILD_DIR = "build_pybind11"


def align_derivlib_root():
    return (Path(__file__).resolve().parent / "../../../../AlignDerivLib").resolve()


def align_derivlib_build_dir(align_deriv_dir=None):
    return Path(align_deriv_dir or align_derivlib_root()).resolve() / ALIGN_DERIVLIB_BUILD_DIR


def find_compatible_align_derivlib_build_dir(align_deriv_dir=None):
    root = Path(align_deriv_dir or align_derivlib_root()).resolve()
    preferred = root / ALIGN_DERIVLIB_BUILD_DIR
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)

    candidate_dirs = []
    if preferred.is_dir():
        candidate_dirs.append(preferred)
    if root.is_dir():
        candidate_dirs.extend(path for path in sorted(root.iterdir()) if path.is_dir() and path != preferred)

    for candidate in candidate_dirs:
        for suffix in suffixes:
            if (candidate / f"{ALIGN_DERIVLIB_MODULE}{suffix}").exists():
                return candidate
    return preferred if preferred.is_dir() else None


def add_align_derivlib_to_path(align_deriv_dir=None):
    """Expose the AlignDerivLib pybind11 extension built for the pytorch Python 3.10 env."""
    build_dir = find_compatible_align_derivlib_build_dir(align_deriv_dir)
    if build_dir is None or not build_dir.is_dir():
        return None
    path_str = str(build_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return build_dir


def import_align_derivlib():
    add_align_derivlib_to_path()
    try:
        return importlib.import_module(ALIGN_DERIVLIB_MODULE)
    except ModuleNotFoundError as exc:
        build_dir = align_derivlib_build_dir()
        found = sorted(str(path) for path in build_dir.glob(f"{ALIGN_DERIVLIB_MODULE}*"))
        suffixes = ", ".join(importlib.machinery.EXTENSION_SUFFIXES)
        expected = ", ".join(
            f"{ALIGN_DERIVLIB_MODULE}{suffix}"
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )
        message = (
            f"Could not import {ALIGN_DERIVLIB_MODULE} for the active Python interpreter "
            f"({sys.version.split()[0]}). Expected {expected} under {build_dir}. "
            f"This interpreter accepts extension suffixes: {suffixes}. "
            f"Found binaries: {found or 'none'}. Run ml_alignment.py in the conda pytorch "
            "Python 3.10 environment, or rebuild AlignDerivLib/build_pybind11 for this Python."
        )
        raise ModuleNotFoundError(message) from exc


def load_variable_information(tracksummary_filepath, trackstates_filepath, id_to_idx_map, debug_dataflow=False):
    # 1. Read tracksummary
    summary_file = uproot.open(tracksummary_filepath)
    summary_tree = summary_file["tracksummary"]

    sum_keys = ["event_nr", "track_nr", "chi2Sum", "NDF"]
    sum_arrays = summary_tree.arrays(sum_keys)
    sum_df = ak.to_dataframe(sum_arrays).reset_index(drop=True)

    sum_df = sum_df.rename(columns={"chi2Sum":"chi2", "NDF": "ndf"})
    sum_df["chi2_ndf"] = sum_df["chi2"] / sum_df["ndf"] 
    print(f"Tracks loaded: {len(sum_df)} trajectories")
    debug_dataframe_summary("tracksummary", sum_df, enabled=debug_dataflow)

    # 2. Load per-state columns only.
    # Background: the ROOT trackstates tree has two array-length classes per event:
    #   - per-state  (N_states / event): volume_id, l_x_hit, eLOC0_smt, g_x_smt, ...
    #   - per-measurement (N_meas / event, N_meas <= N_states): dim_hit, err_x_hit, err_y_hit
    #   - per-event  (scalar): event_nr, track_nr
    # Loading all classes together with ak.to_dataframe() truncates to N_meas rows and
    # fails to broadcast event_nr. Load per-state columns here, then expand true
    # measurement errors onto the measured states.
    states_file = uproot.open(trackstates_filepath)
    states_tree = states_file["trackstates"]

    per_state_keys = [
        "volume_id", "layer_id", "module_id",
        "l_x_hit", "l_y_hit",             # NaN for hole/material states
        "eLOC0_smt", "eLOC1_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
        "px_smt", "py_smt", "pz_smt",
        "ePHI_smt", "eTHETA_smt", "eQOP_smt",
    ]
    states_array = states_tree.arrays(per_state_keys)
    states_df = ak.to_dataframe(states_array).reset_index(drop=True)
    debug_dataframe_summary("trackstates.per_state_raw", states_df, enabled=debug_dataflow)

    # Load true measurement errors separately: err_x_hit / err_y_hit are per-measurement
    # (N_meas ≤ N_states per track).  Expand them to per-state length by inserting NaN
    # at hole states, keyed by the per-state l_x_hit NaN pattern.
    try:
        lx_jagged   = states_tree["l_x_hit"].array(library="ak")
        errx_jagged = states_tree["err_x_hit"].array(library="ak")
        erry_jagged = states_tree["err_y_hit"].array(library="ak")
        errx_flat = []
        erry_flat = []
        for i in range(len(lx_jagged)):
            lx_tr  = ak.to_numpy(lx_jagged[i])
            is_meas = ~np.isnan(lx_tr)
            n_st = len(lx_tr)
            ex = np.full(n_st, np.nan)
            ey = np.full(n_st, np.nan)
            em_x = ak.to_numpy(errx_jagged[i])
            em_y = ak.to_numpy(erry_jagged[i])
            if len(em_x) == int(is_meas.sum()):
                ex[is_meas] = em_x
                ey[is_meas] = em_y
            errx_flat.extend(ex.tolist())
            erry_flat.extend(ey.tolist())
        states_df["err_x_hit"] = errx_flat
        states_df["err_y_hit"] = erry_flat
        print("  Loaded true measurement errors (err_x_hit / err_y_hit) via per-state expansion.")
    except Exception as _e:
        print(f"  Could not load err_x/y_hit ({_e}); will use smoothed posterior variance.")

    # 3. Broadcast per-event event_nr / track_nr to per-state level.
    vol_jagged = states_tree["volume_id"].array(library="ak")
    counts = ak.to_numpy(ak.num(vol_jagged, axis=1))
    ev_flat = ak.to_numpy(states_tree["event_nr"].array(library="ak"))
    tr_flat = ak.to_numpy(states_tree["track_nr"].array(library="ak"))
    states_df["event_nr"] = np.repeat(ev_flat, counts)
    states_df["track_nr"] = np.repeat(tr_flat, counts)
    debug_dataframe_summary("trackstates.with_event_track", states_df, enabled=debug_dataflow)

    # 4. Inner merge: map track-level chi2/ndf onto each hit state.
    merged_df = pd.merge(states_df, sum_df, on=["event_nr", "track_nr"], how="inner")
    debug_dataframe_summary("merged.all_states", merged_df, enabled=debug_dataflow)

    # 5. Remove hole/material states: l_x_hit is NaN for states without a measurement.
    merged_df = merged_df[merged_df["l_x_hit"].notna()].copy()
    debug_dataframe_summary("merged.measurement_states", merged_df, enabled=debug_dataflow)

    # 6. Map sensor ID.
    sensor_index = pd.Series(id_to_idx_map)
    sensor_index.index = pd.MultiIndex.from_tuples(
        sensor_index.index, names=["volume_id", "layer_id", "module_id"]
    )
    hit_index = pd.MultiIndex.from_frame(
        merged_df[["volume_id", "layer_id", "module_id"]]
    )
    merged_df["sensor_idx"] = hit_index.map(sensor_index)
    if debug_dataflow:
        missing_sensor = int(merged_df["sensor_idx"].isna().sum())
        print(f"{_debug_prefix('sensor_mapping')} missing_sensor_idx={missing_sensor} mapped={len(merged_df) - missing_sensor}")

    # 7. ACTS-like alignment residuals.
    # AlignmentAlgorithm builds residuals from the measurement and the smoothed state.
    merged_df["res_eLOC0"] = merged_df["l_x_hit"] - merged_df["eLOC0_smt"]
    merged_df["res_eLOC1"] = merged_df["l_y_hit"] - merged_df["eLOC1_smt"]
    if debug_dataflow:
        debug_tensor_summary("residuals.raw", merged_df[["res_eLOC0", "res_eLOC1"]].values, enabled=True)

    meas_var0 = (merged_df["err_x_hit"] ** 2).clip(lower=EPS_WEIGHT)
    meas_var1 = (merged_df["err_y_hit"] ** 2).clip(lower=EPS_WEIGHT)
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
    merged_df = merged_df[merged_df["module_id"] > 0].reset_index(drop=True)
    debug_dataframe_summary("merged.valid_hits", merged_df, enabled=debug_dataflow)
    if debug_dataflow and len(merged_df):
        debug_layer_distribution(
            "valid_hits",
            merged_df["volume_id"].values,
            merged_df["layer_id"].values,
            enabled=True,
        )

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
    )
    debug_alignment_inputs("load_variable_information.output", inputs, enabled=debug_dataflow)
    return inputs

def euler_to_rotation_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def compute_jacobians_from_alignderivlib(
    hit_sensor_ids,
    hit_global_positions,
    hit_directions,
    hit_qoverp,
    transforms_list,
    current_deltas=None,
):
    """Compute H_geom_batch [N,2,6] (∂residual/∂α) per hit via AlignDerivLib C++ bindings.

    current_deltas: numpy ndarray [num_sensors, 6] with current alignment corrections, or None.
    Returns a detached float32 tensor — does not participate in autograd.
    """
    ad = import_align_derivlib()

    num_hits = len(hit_sensor_ids)
    H_batch = np.zeros((num_hits, 2, 6), dtype=np.float32)

    ids_np = hit_sensor_ids.numpy()
    pos_np = hit_global_positions.numpy()
    dir_np = hit_directions.numpy()
    qop_np = hit_qoverp.numpy().ravel()
    surface_cache = {}

    for i in range(num_hits):
        sensor_idx = int(ids_np[i])
        if sensor_idx not in surface_cache:
            surf_data = transforms_list[sensor_idx]

            center = np.array(surf_data["Translation"], dtype=np.float64)
            rotation = np.array(surf_data["RotationMatrix"], dtype=np.float64)  # (3,3)

            if current_deltas is not None:
                delta_t = current_deltas[sensor_idx, :3].astype(np.float64)
                alpha, beta, gamma = current_deltas[sensor_idx, 3:].astype(np.float64)
                center = center + rotation @ delta_t
                rotation = rotation @ euler_to_rotation_matrix(alpha, beta, gamma)

            surface_cache[sensor_idx] = ad.PlaneSurface(center, rotation)
        surface = surface_cache[sensor_idx]

        pos = pos_np[i].astype(np.float64)
        direction = dir_np[i].astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm > EPS_DIRECTION_NORM:
            direction = direction / norm

        free_track = ad.FreeTrackParameters(pos, 0.0, direction, float(qop_np[i]))
        J = surface.alignmentToResidualDerivative(free_track, pos, 0.0)
        H_batch[i] = J.astype(np.float32)

    return torch.tensor(H_batch, dtype=torch.float32)


def update_geometry_to_json(transforms_list, predicted_deltas, output_filepath):
    for i, item in enumerate(transforms_list):
        #The delta transform
        delta_t_local = predicted_deltas[i, :3]
        delta_euler = predicted_deltas[i, 3:]

        # The original transform
        old_t = np.array(item["Translation"])
        old_R = np.array(item["RotationMatrix"])  # (3,3)
        #The new transform
        new_t = old_t + old_R @ delta_t_local
        item["Translation"] = new_t.tolist()

        alpha, beta, gamma = delta_euler
        R_delta = euler_to_rotation_matrix(alpha, beta, gamma)

        new_R = old_R @ R_delta
        item["RotationMatrix"] = new_R.tolist()

        fm = np.eye(4)
        fm[:3, :3] = new_R
        fm[:3, 3] = new_t

        if "FullMatrix" in item:
            old_fm = item["FullMatrix"]
            if len(old_fm) == 16:
                if abs(old_fm[12] - old_t[0]) < FULL_MATRIX_TOLERANCE:
                    item["FullMatrix"] = fm.flatten('F').tolist()
                else:
                    item["FullMatrix"] = fm.flatten("C").tolist()
        else:
            item["FullMatrix"] = fm.tolist()

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as f_out:
        json.dump(transforms_list, f_out, indent=4)


Update_geometry_to_json = update_geometry_to_json


def build_target_mask(num_sensors, id_to_idx_map, target_layers):
    target_layers = set(parse_target_layers(target_layers))
    target_mask = torch.zeros(num_sensors, dtype=torch.bool)
    for (vol, lay, _sen), idx in id_to_idx_map.items():
        if (vol, lay) in target_layers:
            target_mask[idx] = True
    return target_mask


def print_target_residual_stats(initial_residuals, hit_volume_ids, hit_layer_ids, target_layers):
    res_np = initial_residuals.detach().cpu().numpy()
    vols_np = hit_volume_ids.detach().cpu().numpy()
    lays_np = hit_layer_ids.detach().cpu().numpy()
    print("\n=== Target-layer ACTS-like residuals before solve ===")
    print(f"  {'Layer':<20}  {'N':>8}  {'mean LOC0':>12}  {'RMS LOC0':>10}  {'mean LOC1':>12}  {'RMS LOC1':>10}")
    for vol, lay in parse_target_layers(target_layers):
        mask = (vols_np == vol) & (lays_np == lay)
        if mask.sum() == 0:
            print(f"  vol={vol} lay={lay:<6}  no valid hits")
            continue
        vals = res_np[mask]
        mean0 = float(np.mean(vals[:, 0]))
        rms0 = float(np.sqrt(np.mean(vals[:, 0] ** 2)))
        mean1 = float(np.mean(vals[:, 1]))
        rms1 = float(np.sqrt(np.mean(vals[:, 1] ** 2)))
        label = f"vol={vol} lay={lay}"
        print(f"  {label:<20}  {int(mask.sum()):>8}  {mean0:>12.5f}  {rms0:>10.5f}  {mean1:>12.5f}  {rms1:>10.5f}")


def print_per_layer_residual_diagnostics(init_res_np, final_res, vols_np, lays_np, target_layers):
    print("\n=== Per-layer residual mean (training data) ===")
    print(f"  {'Layer':<20}  {'X_before':>10}  {'X_after':>10}  {'Y_before':>10}  {'Y_after':>10}")
    for vol, lay in target_layers:
        mask = (vols_np == vol) & (lays_np == lay)
        if mask.sum() == 0:
            print(f"  vol={vol} lay={lay}  — no hits in training set")
            continue
        x_bef = float(np.mean(init_res_np[mask, 0]))
        x_aft = float(np.mean(final_res[mask, 0]))
        y_bef = float(np.mean(init_res_np[mask, 1]))
        y_aft = float(np.mean(final_res[mask, 1]))
        label = f"vol={vol} lay={lay} (n={mask.sum()})"
        print(f"  {label:<28}  {x_bef:>9.4f}mm  {x_aft:>9.4f}mm  {y_bef:>9.4f}mm  {y_aft:>9.4f}mm")

    print("\n=== Per-layer residual RMS (training data) ===")
    print(f"  {'Layer':<20}  {'X_before':>10}  {'X_after':>10}  {'Y_before':>10}  {'Y_after':>10}")
    for vol, lay in target_layers:
        mask = (vols_np == vol) & (lays_np == lay)
        if mask.sum() == 0:
            print(f"  vol={vol} lay={lay}  — no hits in training set")
            continue
        x_bef = float(np.std(init_res_np[mask, 0]))
        x_aft = float(np.std(final_res[mask, 0]))
        y_bef = float(np.std(init_res_np[mask, 1]))
        y_aft = float(np.std(final_res[mask, 1]))
        label = f"vol={vol} lay={lay} (n={mask.sum()})"
        print(f"  {label:<28}  {x_bef:>9.4f}mm  {x_aft:>9.4f}mm  {y_bef:>9.4f}mm  {y_aft:>9.4f}mm")
    print()

def validate_against_truth(transforms_list, predicted_deltas, nominal_json,
                           target_layers, id_to_idx_map):
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
        return
    nom = {}
    with open(nominal_json, "r", encoding="utf-8") as f:
        nominal_data = json.load(f)
    for it in nominal_data:
        m = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
        if m:
            nom[tuple(map(int, m.groups()))] = (np.array(it["Translation"], float),
                                                np.array(it["RotationMatrix"], float))
    print("\n=== ML vs TRUE misalignment (local frame, per sensor) ===")
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
            rz_true = np.arctan2(Rd[1, 0], Rd[0, 0])
            true_v.append([dt_true[0], dt_true[1], rz_true])
            p = predicted_deltas[i]
            pred_v.append([p[0], p[1], p[5]])
        if not true_v:
            continue
        true_v = np.array(true_v); pred_v = np.array(pred_v)
        names = ["tx", "ty", "rz"]
        print(f"  vol={vol} lay={lay} (n={len(true_v)})")
        for j, nm in enumerate(names):
            t, p = true_v[:, j], pred_v[:, j]
            corr = np.corrcoef(t, p)[0, 1] if np.std(t) > 0 and np.std(p) > 0 else float("nan")
            recov = (p @ t) / (t @ t) if (t @ t) > 0 else float("nan")  # projection / recovery fraction
            unit = "mm" if j < 2 else "rad"
            print(f"    {nm}: corr={corr:+.3f}  recovery={recov:+.3f}  "
                  f"|true|rms={np.sqrt((t**2).mean()):.4f}{unit}  |pred|rms={np.sqrt((p**2).mean()):.4f}{unit}")


# =====================================================================
# Milestone 3: Core logic
# =====================================================================

def run_alignment_pipeline(config=None, **legacy_kwargs):
    if config is None:
        config = AlignmentRunConfig(**legacy_kwargs)
    elif legacy_kwargs:
        raise ValueError("Pass either AlignmentRunConfig or keyword arguments, not both.")

    ACTIVE_DOF = parse_active_dof(config.active_dof)
    DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]
    print(f"Active alignment DOF: {[DOF_NAMES[d] for d in ACTIVE_DOF]}")
    TARGET_LAYERS = parse_target_layers(config.target_layers)
    IDEAL_JSON_FILE = config.input_geom
    ALIGNED_JSON_FILE = config.output_geom
    SUMMARY_ROOT_FILE = config.summary_root
    STATES_ROOT_FILE = config.states_root

    print("======================================================")
    print(" Activate the ML-Deriven detector closed alignment engine ")
    if config.iter_label is not None:
        print(f" Route-A closed loop — iteration {config.iter_label}")
    print("======================================================")

    if not os.path.exists(IDEAL_JSON_FILE):
        raise FileNotFoundError(f"ACTS geometry file '{IDEAL_JSON_FILE}' not found.")

    print(f"Reading detector misaligned geometry: {IDEAL_JSON_FILE}")
    transforms_list, NUM_SENSORS, initial_positions, id_to_idx_map= load_geometry_from_json(IDEAL_JSON_FILE)
    print(f"Geometry file loaded successfully! {NUM_SENSORS} detector modules resolved.")
    debug_geometry_summary(
        transforms_list, initial_positions, id_to_idx_map, enabled=config.debug_dataflow
    )

    if not os.path.exists(SUMMARY_ROOT_FILE) or not os.path.exists(STATES_ROOT_FILE):
        missing = [path for path in (SUMMARY_ROOT_FILE, STATES_ROOT_FILE) if not os.path.exists(path)]
        raise FileNotFoundError(f"ROOT file not found: {', '.join(missing)}")

    inputs = load_variable_information(
        SUMMARY_ROOT_FILE,
        STATES_ROOT_FILE,
        id_to_idx_map,
        config.debug_dataflow,
    )
    debug_alignment_inputs("inputs", inputs, enabled=config.debug_dataflow)

    print_target_residual_stats(
        inputs.initial_residuals, inputs.hit_volume_ids, inputs.hit_layer_ids, TARGET_LAYERS
    )

    training_hit_mask = torch.zeros(len(inputs.hit_sensor_ids), dtype=torch.bool)
    for vol, lay in TARGET_LAYERS:
        training_hit_mask |= (inputs.hit_volume_ids == vol) & (inputs.hit_layer_ids == lay)
    training_hit_count = int(training_hit_mask.sum())
    print(f"Training hit mask: {training_hit_count} target-layer hits "
          f"out of {len(training_hit_mask)} total")
    if training_hit_count == 0:
        raise ValueError(f"No hits found for target layers {TARGET_LAYERS}.")

    train_hit_sensor_ids = inputs.hit_sensor_ids[training_hit_mask]
    train_track_params = inputs.track_params[training_hit_mask]
    train_track_chi2_ndf = inputs.track_chi2_ndf[training_hit_mask]
    train_initial_residuals = inputs.initial_residuals[training_hit_mask]
    train_covariances = inputs.covariances[training_hit_mask]
    train_hit_global_positions = inputs.hit_global_positions[training_hit_mask]
    train_hit_directions = inputs.hit_directions[training_hit_mask]
    train_hit_qoverp = inputs.hit_qoverp[training_hit_mask]
    train_hit_volume_ids = inputs.hit_volume_ids[training_hit_mask]
    train_hit_layer_ids = inputs.hit_layer_ids[training_hit_mask]
    if config.debug_dataflow:
        debug_tensor_summary("training.hit_sensor_ids", train_hit_sensor_ids, enabled=True)
        debug_tensor_summary("training.initial_residuals", train_initial_residuals, enabled=True)
        debug_tensor_summary("training.covariances", train_covariances, enabled=True)
        debug_tensor_summary("training.track_params", train_track_params, enabled=True)
        debug_tensor_summary("training.track_chi2_ndf", train_track_chi2_ndf, enabled=True)
        debug_tensor_summary("training.hit_global_positions", train_hit_global_positions, enabled=True)
        debug_tensor_summary("training.hit_directions", train_hit_directions, enabled=True)
        debug_tensor_summary("training.hit_qoverp", train_hit_qoverp, enabled=True)
        debug_layer_distribution(
            "training", train_hit_volume_ids, train_hit_layer_ids, enabled=True
        )

    print("Pre-computing alignment Jacobians via AlignDerivLib...")
    H_geom_batch = compute_jacobians_from_alignderivlib(
        train_hit_sensor_ids, train_hit_global_positions, train_hit_directions, train_hit_qoverp,
        transforms_list, current_deltas=None,
    )
    print(f"Jacobian tensor shape: {H_geom_batch.shape}")
    debug_tensor_summary("jacobian.initial", H_geom_batch, enabled=config.debug_dataflow)

    history_epochs = []
    history_chi2_loss = []

    # Build the fixed-reference mask: True = target surface (calibrated),
    # False = reference surface (held fixed at δ=0).
    target_mask = None
    if config.fix_reference:
        target_mask = build_target_mask(NUM_SENSORS, id_to_idx_map, TARGET_LAYERS)
        print(f"Fixed-reference mode: {int(target_mask.sum())} target surfaces "
              f"({TARGET_LAYERS}) calibrated, {NUM_SENSORS - int(target_mask.sum())} held as reference.")

    engine = AlignmentEngine(NUM_SENSORS)
    dof_mask = torch.zeros(6, dtype=torch.bool)
    for d in ACTIVE_DOF:
        dof_mask[d] = True
    optimizer = optim.Adam(engine.parameters(), lr=DEFAULT_ADAM_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=DEFAULT_ADAM_ETA_MIN
    )
    for epoch in range(1, config.epochs + 1):
        # Recompute Jacobians every configured interval using updated geometry.
        if epoch > 1 and (epoch - 1) % config.jacobian_update_interval == 0:
            current_deltas_np = engine.alignment_params.detach().cpu().numpy()
            H_geom_batch = compute_jacobians_from_alignderivlib(
                train_hit_sensor_ids, train_hit_global_positions,
                train_hit_directions, train_hit_qoverp,
                transforms_list, current_deltas=current_deltas_np,
            )
            debug_tensor_summary(
                f"jacobian.epoch_{epoch}", H_geom_batch, enabled=config.debug_dataflow
            )

        optimizer.zero_grad()

        residual_shift, hit_weights = engine(
            train_hit_sensor_ids, train_track_params, initial_positions,
            train_track_chi2_ndf, H_geom_batch
        )

        chi2_loss = physics_informed_loss(
            train_initial_residuals, residual_shift, train_covariances,
            hit_weights,
            hit_volume_ids=train_hit_volume_ids,
            hit_layer_ids=train_hit_layer_ids,
        )

        l2_reg = config.lambda_l2 * torch.sum(engine.alignment_params ** 2)
        loss = chi2_loss + l2_reg
        loss.backward()
        if engine.alignment_params.grad is not None:
            engine.alignment_params.grad[:, ~dof_mask] = 0.0
            if target_mask is not None:
                engine.alignment_params.grad[~target_mask] = 0.0
        torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        history_epochs.append(epoch)
        history_chi2_loss.append(chi2_loss.item())

        with torch.no_grad():
            engine.alignment_params[:, ~dof_mask] = 0.0
            engine.alignment_params[:, :3].clamp_(-ADAM_CLAMP_T, ADAM_CLAMP_T)
            engine.alignment_params[:, 3:].clamp_(-ADAM_CLAMP_R, ADAM_CLAMP_R)

            if target_mask is not None:
                engine.alignment_params[~target_mask] = 0.0

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                cur_residual_shift = torch.bmm(
                    H_geom_batch,
                    engine.alignment_params[train_hit_sensor_ids].unsqueeze(-1)
                ).squeeze(-1)
                cur_res = train_initial_residuals + cur_residual_shift
                cur_chi2_per_hit = (cur_res ** 2) / torch.clamp(
                    train_covariances, min=EPS_COVARIANCE
                )
                if config.debug_dataflow and (
                    epoch == 1 or (epoch - 1) % config.jacobian_update_interval == 0
                ):
                    debug_tensor_summary(f"epoch_{epoch}.residual_shift", cur_residual_shift, enabled=True)
                    debug_tensor_summary(f"epoch_{epoch}.residuals", cur_res, enabled=True)
                    debug_tensor_summary(f"epoch_{epoch}.hit_weights", hit_weights, enabled=True)
                layer_info = []
                for vl, ll in TARGET_LAYERS:
                    vmask = (train_hit_volume_ids == vl) & (train_hit_layer_ids == ll)
                    if vmask.any():
                        lchi2 = cur_chi2_per_hit[vmask].mean().item()
                        layer_info.append(f"v{vl}l{ll}={lchi2:.2f}")
                tw = hit_weights.detach()
                twn = normalize_hit_weights(
                    hit_weights, train_hit_volume_ids, train_hit_layer_ids
                ).detach()
                layer_weight_info = []
                for vl, ll in TARGET_LAYERS:
                    vmask = (train_hit_volume_ids == vl) & (train_hit_layer_ids == ll)
                    if vmask.any():
                        raw_mean = tw[vmask].mean()
                        norm_mean = twn[vmask].mean()
                        layer_weight_info.append(
                            f"Wv{vl}l{ll}: raw={raw_mean:.3f} norm={norm_mean:.3f}"
                        )
                w_info = f"W: mean={tw.mean():.3f} std={tw.std():.3f} min={tw.min():.3f}"
                print(f"Epoch [{epoch:03d}/{config.epochs}] | LR: {current_lr:.6f} | "
                      f"Chi2: {chi2_loss.item():.4f} | L2: {l2_reg.item():.6f} | "
                      f"Total: {loss.item():.4f} | {w_info} | "
                      + " | ".join(layer_info + layer_weight_info))

    predicted_deltas = engine.alignment_params.detach().cpu().numpy()
    debug_tensor_summary("predicted_deltas", predicted_deltas, enabled=config.debug_dataflow)

    predicted_delta_tensor = torch.tensor(predicted_deltas, dtype=torch.float32)

    # ── Per-layer residual diagnostics ──────────────────────────────────────
    with torch.no_grad():
        final_residual_shift = torch.bmm(
            H_geom_batch,
            predicted_delta_tensor[train_hit_sensor_ids].unsqueeze(-1)
        ).squeeze(-1)
        final_res = (train_initial_residuals + final_residual_shift).numpy()
    debug_tensor_summary("final_residual_shift", final_residual_shift, enabled=config.debug_dataflow)
    debug_tensor_summary("final_residuals", final_res, enabled=config.debug_dataflow)
    init_res_np = train_initial_residuals.numpy()
    vols_np  = train_hit_volume_ids.numpy()
    lays_np  = train_hit_layer_ids.numpy()

    print_per_layer_residual_diagnostics(init_res_np, final_res, vols_np, lays_np, TARGET_LAYERS)

    # The decisive check vs TRUE misalignment.
    validate_against_truth(
        transforms_list, predicted_deltas, config.nominal_file, TARGET_LAYERS, id_to_idx_map
    )

    Path(ALIGNED_JSON_FILE).parent.mkdir(parents=True, exist_ok=True)
    update_geometry_to_json(transforms_list, predicted_deltas, ALIGNED_JSON_FILE)

    print(f"Export successful! The aligned, complete geometry file has been saved to: {ALIGNED_JSON_FILE}")

    # Draw and save loss line chart.
    if config.make_plot and history_epochs:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(history_epochs, history_chi2_loss, label='Chi2 loss', color='firebrick', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss(Log Scale)', fontsize=12)
        plt.title("ML Alignment Engine Training Curve", fontsize=14, fontweight='bold')
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plot_filepath = str(Path(ALIGNED_JSON_FILE).parent / f"loss_curve_{Path(ALIGNED_JSON_FILE).stem}.png")
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        print(f"Training convergence plot saved: '{plot_filepath}'")

    target_metrics = {}
    for vol, lay in TARGET_LAYERS:
        mask = (vols_np == vol) & (lays_np == lay)
        if mask.sum() == 0:
            continue
        target_metrics[(vol, lay)] = {
            "n_hits": int(mask.sum()),
            "mean_before_loc0": float(np.mean(init_res_np[mask, 0])),
            "mean_after_loc0": float(np.mean(final_res[mask, 0])),
            "mean_before_loc1": float(np.mean(init_res_np[mask, 1])),
            "mean_after_loc1": float(np.mean(final_res[mask, 1])),
            "rms_before_loc0": float(np.sqrt(np.mean(init_res_np[mask, 0] ** 2))),
            "rms_after_loc0": float(np.sqrt(np.mean(final_res[mask, 0] ** 2))),
            "rms_before_loc1": float(np.sqrt(np.mean(init_res_np[mask, 1] ** 2))),
            "rms_after_loc1": float(np.sqrt(np.mean(final_res[mask, 1] ** 2))),
        }
    return {
        "output_geom": ALIGNED_JSON_FILE,
        "target_metrics": target_metrics,
        "predicted_deltas": predicted_deltas,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="ML alignment solver (Adam neural-network optimizer). "
                    "Reads current geometry + reconstruction residuals → writes updated geometry."
    )
    p.add_argument("--input-geom", default=str(alignment_json_path("alignment_output", "misalignment_applied")),
                   help="Current geometry G_k (correction base; delta is composed onto it).")
    p.add_argument("--states", dest="states_root",
                   default=str(reconstruction_root_path("reconstruction_output", "misaligned", "trackstates_kf")),
                   help="trackstates ROOT from reconstruction performed with G_k.")
    p.add_argument("--summary", dest="summary_root",
                   default=str(reconstruction_root_path("reconstruction_output", "misaligned", "tracksummary_kf")),
                   help="tracksummary ROOT from reconstruction performed with G_k.")
    p.add_argument("--output-geom", default=str(alignment_json_path("alignment_output", "aligned_transforms", stem_prefix="ml")),
                   help="Updated geometry G_{k+1} = G_k compose delta.")
    p.add_argument("--nominal", dest="nominal_file", default="odd_transforms.json",
                   help="Nominal/truth geometry for the convergence metric (|true|rms).")
    p.add_argument("--iter", dest="iter_label", default=None,
                   help="Iteration label for logging.")
    p.add_argument("--active-dof", default="0,1,3,4,5",
                   help="Comma-separated active DOF indices or names. Default: 0,1,3,4,5 (tx,ty,rx,ry,rz).")
    p.add_argument("--target-layers", default=target_layers_spec(),
                   help=f"Comma-separated target layers as volume:layer. Default: {target_layers_spec()}.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help=f"Number of Adam training epochs. Default: {DEFAULT_EPOCHS}.")
    p.add_argument("--jacobian-update-interval", type=int, default=DEFAULT_JACOBIAN_UPDATE_INTERVAL,
                   help=f"Re-compute Jacobians every N epochs. Default: {DEFAULT_JACOBIAN_UPDATE_INTERVAL}.")
    p.add_argument("--lambda-l2", type=float, default=DEFAULT_LAMBDA_L2,
                   help=f"L2 regularisation weight. Default: {DEFAULT_LAMBDA_L2}.")
    p.add_argument("--debug-dataflow", action="store_true",
                   help="Print compact data-flow diagnostics for geometry, ROOT-derived tensors, Jacobians, and deltas.")
    p.add_argument("--no-plot", action="store_true", help="Skip the loss-curve PNG.")
    a = p.parse_args()

    run_alignment_pipeline(AlignmentRunConfig(
        input_geom=a.input_geom,
        states_root=a.states_root,
        summary_root=a.summary_root,
        output_geom=a.output_geom,
        nominal_file=a.nominal_file,
        iter_label=a.iter_label,
        make_plot=not a.no_plot,
        active_dof=a.active_dof,
        target_layers=a.target_layers,
        epochs=a.epochs,
        jacobian_update_interval=a.jacobian_update_interval,
        lambda_l2=a.lambda_l2,
        debug_dataflow=a.debug_dataflow,
    ))
