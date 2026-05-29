import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
import uproot
import pandas as pd
import awkward as ak
import re
import matplotlib.pyplot as plt

# =====================================================================
# Milestone 1: PyTorch class
# =====================================================================

class AlignmentEngine(nn.Module):
    def __init__(self, num_sensors):
        super().__init__()
        self.num_sensors = num_sensors
        
        # 1. Detector geometry parameters：[tx, ty, tz, alpha, beta, gamma]
        self.alignment_params = nn.Parameter(torch.zeros(num_sensors, 6))
        
        # 2. Hit Quality Attention
        # Input Dimensions: 5 (track parameters) + 3 (detector global coordinates XYZ) + 1 (chi2/ndf) = 9 dim
        # Output Dimensions: 1 (The confidence weight of the hit point in global alignment，during 0~1)
        hidden_size_one = 32
        hidden_size_two = 16
        output_size = 1

        self.hit_quality_net = nn.Sequential(
            nn.Linear(9, hidden_size_one),
            nn.GELU(),
            nn.Linear(hidden_size_one, hidden_size_two),
            nn.GELU(),
            nn.Linear(hidden_size_two, output_size),
            nn.Sigmoid()  # Compress the confidence level to between 0 and 1.
        )

    def forward(self, hit_sensor_ids, track_params, sensor_center_positions, track_chi2_ndf, H_geom_batch):
        """
        Forward propagation: Calculate geometric offset and use machine learning to evaluate data quality.
        sensor_center_positions: [NUM_SENSORS, 3] global XYZ translation of each sensor centre (from JSON).
        H_geom_batch: pre-computed (N,2,6) Jacobian ∂residual/∂α from AlignDerivLib, detached from autograd.
        """
        # ============ B. Geometric projection calculation  ============
        current_params = self.alignment_params[hit_sensor_ids]
        shift_effect = torch.bmm(H_geom_batch, current_params.unsqueeze(-1)).squeeze(-1)

        # ============ C. ML weight calculation based on high-level physical features ============
        # 1. Look up the global centre position of the sensor that recorded each hit. [Num_Hits, 3]
        hit_sensor_center_positions = sensor_center_positions[hit_sensor_ids]

        # 2. Assemble all high-level features from ACTS.
        # shape combination: [Num_Hits, 5] + [Num_Hits, 3] + [Num_Hits, 1] -> [Num_Hits, 9]
        advanced_features = torch.cat([track_params, hit_sensor_center_positions, track_chi2_ndf], dim=-1)
        
        # 3. Each hit point is scored using a neural network.
        # hit_weights shape: [Num_Hits, 1]
        hit_weights = self.hit_quality_net(advanced_features)
        
        return shift_effect, hit_weights

# =====================================================================
# Milestone 1.2: Physics unsupervised Loss (Introducing ML dynamic weights)
# =====================================================================

def physics_informed_loss(initial_residuals, shift_effect, covariances, hit_weights):
    new_residuals = initial_residuals - shift_effect
    raw_chi2 = (new_residuals ** 2) / torch.clamp(covariances, min=1e-5)
    clamped_chi2 = torch.clamp(raw_chi2, max=50.0)
    chi2_loss = torch.mean(hit_weights * clamped_chi2)
    return chi2_loss

# =====================================================================
# Milestone 2: Simulate pybind11 Data 
# =====================================================================
def load_geometry_from_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    transforms_list = None

    if isinstance(data, list) and len(data) > 0 and "Translation" in data[0]:
            transforms_list = data
    else:
        print(f"The format of {filepath} is incorrect")
        return

    if transforms_list is None:
        raise ValueError(f"Unable to find translation quota in {filepath}! Please chech the JSON structure.")

    if isinstance(data, dict):
        num_sensors = data.get("Count", data.get("Count", len(transforms_list)))
    else:
        num_sensors = len(transforms_list)

    positions = []
    id_to_idx = {}
    for i, item in enumerate(transforms_list):
        positions.append(item["Translation"])
        match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", item.get("ID", ""))
        if match:
            vol, lay, sen = map(int, match.groups())
            id_to_idx[(vol, lay, sen)] = i

    return data, transforms_list, num_sensors, torch.tensor(positions, dtype=torch.float32), id_to_idx

def load_variable_information(tracksummary_filepath, trackstates_filepath, id_to_idx_map):
    # 1. Read tracksummary and filter high-quality tracks
    summary_file = uproot.open(tracksummary_filepath)
    summary_tree = summary_file["tracksummary"]

    sum_keys = ["event_nr", "track_nr", "chi2Sum", "NDF"]
    sum_arrays = summary_tree.arrays(sum_keys)
    sum_df = ak.to_dataframe(sum_arrays).reset_index(drop=True)

    sum_df = sum_df.rename(columns={"chi2Sum":"chi2", "NDF": "ndf"})
    sum_df["chi2_ndf"] = sum_df["chi2"] / sum_df["ndf"]
    sum_df = sum_df[sum_df["chi2_ndf"] < 50.0]
    print(f"The remaining high-quality trajectories after filtering: {len(sum_df)} trajectories")

    # 2. Load per-state columns only.
    # Background: the ROOT trackstates tree has two array-length classes per event:
    #   - per-state  (N_states / event): volume_id, l_x_hit, eLOC0_smt, g_x_smt, …
    #   - per-measurement (N_meas / event, N_meas ≤ N_states): dim_hit, err_x_hit, err_y_hit
    #   - per-event  (scalar): event_nr, track_nr
    # Loading all classes together with ak.to_dataframe() truncates to N_meas rows and
    # fails to broadcast event_nr, producing only ~64 valid hits instead of ~2000.
    # Fix: load only per-state columns here; add event_nr via np.repeat; detect
    # 1D/2D from l_y_hit NaN; use err_eLOC0_smt^2 as variance proxy.
    states_file = uproot.open(trackstates_filepath)
    states_tree = states_file["trackstates"]

    per_state_keys = [
        "volume_id", "layer_id", "module_id",
        "l_x_hit", "l_y_hit",             # NaN for hole/material states
        "eLOC0_smt", "eLOC1_smt",
        "err_eLOC0_smt", "err_eLOC1_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
        "px_smt", "py_smt", "pz_smt",
        "ePHI_smt", "eTHETA_smt", "eQOP_smt",
    ]
    states_array = states_tree.arrays(per_state_keys)
    states_df = ak.to_dataframe(states_array).reset_index(drop=True)

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

    # 4. Inner merge: map track-level chi2/ndf onto each hit state.
    merged_df = pd.merge(states_df, sum_df, on=["event_nr", "track_nr"], how="inner")

    # 5. Remove hole/material states: l_x_hit is NaN for states without a measurement.
    merged_df = merged_df[merged_df["l_x_hit"].notna()].copy()

    # 6. Map sensor ID.
    merged_df["sensor_tuple"] = list(zip(merged_df.volume_id, merged_df.layer_id, merged_df.module_id))
    merged_df["sensor_idx"] = merged_df["sensor_tuple"].map(id_to_idx_map)

    # 7. Construct alignment residuals: r = m_k - H * x_smt
    merged_df["res_eLOC0"] = merged_df["l_x_hit"] - merged_df["eLOC0_smt"]
    merged_df["res_eLOC1"] = merged_df["l_y_hit"] - merged_df["eLOC1_smt"]

    # Variance: prefer true measurement errors (err_x_hit/err_y_hit) when available.
    # These are stored per-state with NaN for non-measurement (hole/material) states.
    # Fall back to smoothed posterior variance if true errors are missing.
    has_true_errors = (
        "err_x_hit" in merged_df.columns
        and merged_df["err_x_hit"].notna().any()
    )
    if has_true_errors:
        merged_df["var_eLOC0"] = np.where(
            merged_df["err_x_hit"].notna() & (merged_df["err_x_hit"] > 0),
            (merged_df["err_x_hit"] ** 2).clip(lower=1e-8),
            (merged_df["err_eLOC0_smt"] ** 2).clip(lower=1e-8),
        )
        merged_df["var_eLOC1"] = np.where(
            merged_df["err_y_hit"].notna() & (merged_df["err_y_hit"] > 0),
            (merged_df["err_y_hit"] ** 2).clip(lower=1e-8),
            (merged_df["err_eLOC1_smt"] ** 2).clip(lower=1e-8),
        )
        print("  Using true measurement errors (err_x_hit/err_y_hit) as variance.")
    else:
        merged_df["var_eLOC0"] = (merged_df["err_eLOC0_smt"] ** 2).clip(lower=1e-8)
        merged_df["var_eLOC1"] = (merged_df["err_eLOC1_smt"] ** 2).clip(lower=1e-8)
        print("  Fallback: using smoothed posterior variance (err_eLOC0/1_smt^2).")

    # 8. Handle 1D strip hits: l_y_hit is NaN for 1D measurements.
    # Set LOC1 residual = 0 and variance = 1 so their chi2 contribution is 0²/1 = 0.
    is_1d = merged_df["l_y_hit"].isna()
    merged_df.loc[is_1d, "res_eLOC1"] = 0.0
    merged_df.loc[is_1d, "var_eLOC1"] = 1.0
    n_1d = int(is_1d.sum())
    n_2d = int((~is_1d).sum())
    print(f"  Hit breakdown: {n_2d} 2D (pixel), {n_1d} 1D (strip)")

    # Physics floor for strip detector Y direction.
    # Vol 23/24/25 (ShortStrip barrel/endcap): intrinsic Y resolution ~1.2mm.
    # Using err_eLOC1_smt^2 alone underestimates V_y, inflating chi2 and causing
    # the optimizer to over-correct Y.  Clip var_eLOC1 to at least (1.2mm)^2.
    STRIP_VOLUMES = {23, 24, 25}
    STRIP_Y_FLOOR = 1.44  # (1.2 mm)^2
    is_strip_vol = merged_df["volume_id"].isin(STRIP_VOLUMES)
    before_floor = merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].mean() if is_strip_vol.any() else 0.0
    merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"] = (
        merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].clip(lower=STRIP_Y_FLOOR)
    )
    if is_strip_vol.any():
        after_floor = merged_df.loc[is_strip_vol & ~is_1d, "var_eLOC1"].mean()
        print(f"  Strip Y variance floor applied: mean var_LOC1 {before_floor:.4f} → {after_floor:.4f} mm²")

    # 9. Drop remaining invalids (sensor not in alignment set, NaN smoothed params).
    required_cols = [
        "sensor_idx", "res_eLOC0", "res_eLOC1",
        "var_eLOC0", "var_eLOC1", "eLOC0_smt",
        "g_x_smt", "g_y_smt", "g_z_smt",
    ]
    merged_df = merged_df.dropna(subset=required_cols)
    merged_df = merged_df[merged_df["module_id"] > 0].reset_index(drop=True)

    num_hits = len(merged_df)
    print(f"Feature alignment complete! A total of {num_hits} valid tensor data points have been constructed")

    # 7. Build PyTorch Tensors.
    hit_sensor_ids = torch.tensor(merged_df["sensor_idx"].values, dtype=torch.long)

    # Alignment residuals: r = m_k - H * x_smt
    initial_residuals = torch.tensor(
        merged_df[["res_eLOC0", "res_eLOC1"]].values, dtype=torch.float32)

    # Residual variances: V - H*P_smt*H^T  (diagonal LOC0/LOC1 components)
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

    return (hit_sensor_ids, initial_residuals, covariances, track_params, track_chi2_ndf,
            hit_global_positions, hit_directions, hit_qoverp,
            hit_volume_ids, hit_layer_ids)

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
    import align_deriv_pb as ad

    num_hits = len(hit_sensor_ids)
    H_batch = np.zeros((num_hits, 2, 6), dtype=np.float32)

    ids_np = hit_sensor_ids.numpy()
    pos_np = hit_global_positions.numpy()
    dir_np = hit_directions.numpy()
    qop_np = hit_qoverp.numpy().ravel()

    for i in range(num_hits):
        sensor_idx = int(ids_np[i])
        surf_data = transforms_list[sensor_idx]

        center = np.array(surf_data["Translation"], dtype=np.float64)
        rotation = np.array(surf_data["RotationMatrix"], dtype=np.float64)  # (3,3)

        if current_deltas is not None:
            delta_t = current_deltas[sensor_idx, :3].astype(np.float64)
            alpha, beta, gamma = current_deltas[sensor_idx, 3:].astype(np.float64)
            center = center + rotation @ delta_t
            ca, sa = np.cos(alpha), np.sin(alpha)
            cb, sb = np.cos(beta),  np.sin(beta)
            cg, sg = np.cos(gamma), np.sin(gamma)
            Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
            Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
            rotation = rotation @ (Rz @ Ry @ Rx)

        surface = ad.PlaneSurface(center, rotation)

        pos = pos_np[i].astype(np.float64)
        direction = dir_np[i].astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction = direction / norm

        free_track = ad.FreeTrackParameters(pos, 0.0, direction, float(qop_np[i]))
        J = surface.alignmentToResidualDerivative(free_track, pos, 0.0)
        H_batch[i] = J.astype(np.float32)

    return torch.tensor(H_batch, dtype=torch.float32)


def Update_geometry_to_json(original_data, transforms_list, predicted_deltas, output_filepath):
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
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta),  np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        R_delta = Rz @ Ry @ Rx

        new_R = old_R @ R_delta
        item["RotationMatrix"] = new_R.tolist()

        fm = np.eye(4)
        fm[:3, :3] = new_R
        fm[:3, 3] = new_t
        
        if "FullMatrix" in item:
            old_fm = item["FullMatrix"]
            if len(old_fm) == 16:
                if abs(old_fm[12] - old_t[0]) < 1e-4:
                    item["FullMatrix"] = fm.flatten('F').tolist()
                else:
                    item["FullMatrix"] = fm.flatten("C").tolist()
        else:
            item["FullMatrix"] = fm.tolist()

    with open(output_filepath, "w", encoding="utf-8") as f_out:
        json.dump(original_data, f_out, indent=4)

# =====================================================================
# Milestone 3: Core logic
# =====================================================================

def run_alignment_pipeline():
    EPOCHS = 500
    JACOBIAN_UPDATE_INTERVAL = 20
    LAMBDA_L2 = 1e-7   # L2 regularisation on alignment params to prevent runaway corrections
    # Base geometry = misaligned (the perturbed starting model).
    # ML learns corrections δα: misaligned + δα → nominal (ground truth).
    IDEAL_JSON_FILE = "alignment_output/misalignment_applied.json"
    ALIGNED_JSON_FILE = "alignment_output/aligned_transforms_v17l8_v24l4_ml.json"

    # Training signal: misaligned reconstruction (nominal hits + misaligned geometry).
    # Nominal hits interpreted under the wrong (misaligned) geometry produce large
    # systematic residuals (≈ misalignment offset), giving a strong gradient signal.
    SUMMARY_ROOT_FILE = "reconstruction_output/misaligned/tracksummary_kf_v17l8_v24l4.root"
    STATES_ROOT_FILE = "reconstruction_output/misaligned/trackstates_kf_v17l8_v24l4.root"

    # Import AlignDerivLib pybind11 module
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../../../../AlignDerivLib/build_pybind11"))
    import align_deriv_pb  # noqa: F401  — verifies the .so is importable before training

    print("======================================================")
    print(" Activate the ML-Deriven detector closed alignment engine ")
    print("======================================================")
    
    if not os.path.exists(IDEAL_JSON_FILE):
        print(f"Error: ACTS geometry file '{IDEAL_JSON_FILE}' not found!")
        return 

    print(f"Reading detector misaligned geometry: {IDEAL_JSON_FILE}")
    original_data, transforms_list, NUM_SENSORS, initial_positions, id_to_idx_map= load_geometry_from_json(IDEAL_JSON_FILE)
    print(f"Geometry file loaded successfully! {NUM_SENSORS} detector modules resolved.")
    
    if not os.path.exists(SUMMARY_ROOT_FILE) or not os.path.exists(STATES_ROOT_FILE):
        print(f"ERROR: ROOT file not found")
        return

    (hit_sensor_ids, initial_residuals, covariances, track_params, track_chi2_ndf,
     hit_global_positions, hit_directions, hit_qoverp,
     hit_volume_ids, hit_layer_ids) = \
        load_variable_information(SUMMARY_ROOT_FILE, STATES_ROOT_FILE, id_to_idx_map)
        
    engine = AlignmentEngine(NUM_SENSORS)

    optimizer = optim.Adam(engine.parameters(), lr=1e-2)
    # Cos annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print("Pre-computing alignment Jacobians via AlignDerivLib...")
    H_geom_batch = compute_jacobians_from_alignderivlib(
        hit_sensor_ids, hit_global_positions, hit_directions, hit_qoverp,
        transforms_list, current_deltas=None,
    )
    print(f"Jacobian tensor shape: {H_geom_batch.shape}")

    history_epochs = []
    history_chi2_loss = []

    for epoch in range(1, EPOCHS + 1):
        # Recompute Jacobians every JACOBIAN_UPDATE_INTERVAL epochs using updated geometry
        if epoch > 1 and (epoch - 1) % JACOBIAN_UPDATE_INTERVAL == 0:
            current_deltas_np = engine.alignment_params.detach().cpu().numpy()
            H_geom_batch = compute_jacobians_from_alignderivlib(
                hit_sensor_ids, hit_global_positions, hit_directions, hit_qoverp,
                transforms_list, current_deltas=current_deltas_np,
            )

        optimizer.zero_grad()

        # Features read from ACTS are fed into the model for forward propagation.
        shift_effect, hit_weights = engine(
            hit_sensor_ids, track_params, initial_positions, track_chi2_ndf, H_geom_batch
        )

        # Fix hit_weights = 1: force chi2 to be driven purely by geometric residuals.
        # The quality network cannot "cheat" by zeroing weights to trivially minimize chi2.
        unit_weights = torch.ones_like(hit_weights)
        chi2_loss = physics_informed_loss(
            initial_residuals, shift_effect, covariances, unit_weights
        )

        # L2 regularisation: penalise large alignment corrections to prevent overfitting.
        l2_reg = LAMBDA_L2 * torch.sum(engine.alignment_params ** 2)

        loss = chi2_loss + l2_reg

        loss.backward()

        #Gradient clipping: if the total gradient norm exceeds 1.0,scale it down proportionally.
        torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=10.0)

        optimizer.step()

        scheduler.step()

        history_epochs.append(epoch)
        history_chi2_loss.append(chi2_loss.item())

        with torch.no_grad():
            # Clamp translation to ±1.5mm (actual misalignment is ~0.5mm)
            engine.alignment_params[:, :3].clamp_(-1.5, 1.5)
            engine.alignment_params[:, 3:].clamp_(-0.05, 0.05)

            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch:03d}/{EPOCHS}] | LR: {current_lr:.6f} | "
                      f"Chi2: {chi2_loss.item():.8f} | L2: {l2_reg.item():.8f} | "
                      f"Total: {loss.item():.8f}")

    predicted_deltas = engine.alignment_params.detach().cpu().numpy()

    # ── Per-layer residual diagnostics ──────────────────────────────────────
    with torch.no_grad():
        final_shift = torch.bmm(
            H_geom_batch,
            engine.alignment_params[hit_sensor_ids].unsqueeze(-1)
        ).squeeze(-1)
        final_res = (initial_residuals - final_shift).numpy()
    init_res_np = initial_residuals.numpy()
    vols_np  = hit_volume_ids.numpy()
    lays_np  = hit_layer_ids.numpy()

    TARGET_LAYERS = [(17, 8), (24, 4)]
    print("\n=== Per-layer residual RMS (training data) ===")
    print(f"  {'Layer':<20}  {'X_before':>10}  {'X_after':>10}  {'Y_before':>10}  {'Y_after':>10}")
    for vol, lay in TARGET_LAYERS:
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

    Update_geometry_to_json(original_data, transforms_list, predicted_deltas, ALIGNED_JSON_FILE)

    print(f"Export successful! The aligned, complete geometry file has been saved to: {ALIGNED_JSON_FILE}")
            
    # Draw and save loss line chart
    plt.figure(figsize=(10, 6))

    plt.plot(history_epochs, history_chi2_loss, label='Chi2 loss', color='firebrick', linewidth='2')

    plt.yscale('log')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss(Log Scale)', fontsize=12)
    plt.title("ML Alignment Engine Training Curve", fontsize=14, fontweight='bold')
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plot_filepath = "alignment_loss_curve.png"
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    print(f"THe training convergence line plot has been generated and saved in the current directory:'{plot_filepath}'")

if __name__ == "__main__":
    run_alignment_pipeline()
