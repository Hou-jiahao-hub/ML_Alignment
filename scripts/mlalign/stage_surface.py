"""Stage A -- surface mode: frozen KF-smoothed tracks, Adam on target surfaces."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from alignment_diagnostics import (
    print_per_layer_residual_diagnostics,
    print_target_residual_stats,
    validate_against_truth,
)
from .conventions import EPS_COVARIANCE
from .geometry_io import (
    build_target_mask,
    load_geometry_from_json,
    update_geometry_to_json,
)
from .root_inputs import load_variable_information
from .rotations import apply_alignment_torch

EPS_DIRECTION_NORM = 1e-9
DEFAULT_EPOCHS = 200
DEFAULT_LAMBDA_L2 = 1e-5
DEFAULT_ADAM_LR = 1e-2
DEFAULT_ADAM_ETA_MIN = 1e-5
ADAM_CLAMP_T = 1.5
ADAM_CLAMP_R = 0.05


def clamp_parameter_and_reset_adam(parameter, lower, upper, optimizer):
    """Clamp a parameter and clear Adam moments for entries that hit a bound."""
    with torch.no_grad():
        saturated = (parameter < lower) | (parameter > upper)
        parameter.clamp_(min=lower, max=upper)
        state = optimizer.state.get(parameter, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None:
                moment.masked_fill_(saturated, 0.0)
    return saturated


class AlignmentEngine(nn.Module):
    """Production surface-alignment parameters with unweighted residuals."""

    def __init__(self, num_sensors):
        super().__init__()
        self.num_sensors = num_sensors
        self.alignment_params = nn.Parameter(torch.zeros(num_sensors, 6))

    def forward(
        self,
        hit_sensor_ids,
        sensor_center_positions,
        sensor_rotations,
        measured_local,
        hit_global_positions,
        hit_directions,
    ):
        return compute_autograd_residuals(
            measured_local,
            hit_global_positions,
            hit_directions,
            hit_sensor_ids,
            sensor_center_positions,
            sensor_rotations,
            self.alignment_params,
        )


def physics_informed_residual_loss(residuals, covariances, component_mask=None):
    """Mean chi2 over the valid measurement components."""
    raw_chi2 = residuals.square() / torch.clamp(
        covariances, min=EPS_COVARIANCE
    )
    if component_mask is None:
        return torch.mean(raw_chi2)
    return torch.sum(
        torch.where(component_mask, raw_chi2, torch.zeros_like(raw_chi2))
    ) / component_mask.sum().clamp(min=1)


def project_track_to_aligned_surface(
    hit_global_positions,
    hit_directions,
    center,
    rotation,
):
    direction_norm = torch.linalg.norm(hit_directions, dim=-1, keepdim=True)
    direction = hit_directions / torch.clamp(direction_norm, min=EPS_DIRECTION_NORM)
    normal = rotation[..., :, 2]
    numerator = torch.sum((center - hit_global_positions) * normal, dim=-1)
    denominator = torch.sum(direction * normal, dim=-1)
    eps = torch.full_like(denominator, EPS_DIRECTION_NORM)
    denominator_safe = torch.where(
        denominator.abs() > EPS_DIRECTION_NORM,
        denominator,
        torch.where(denominator >= 0, eps, -eps),
    )
    path_length = numerator / denominator_safe
    return hit_global_positions + path_length.unsqueeze(-1) * direction


def compute_autograd_residuals(
    measured_local,
    hit_global_positions,
    hit_directions,
    hit_sensor_ids,
    sensor_center_positions,
    sensor_rotations,
    alignment_params,
):
    current_params = alignment_params[hit_sensor_ids]
    center0 = sensor_center_positions[hit_sensor_ids].to(
        device=current_params.device, dtype=current_params.dtype
    )
    rotation0 = sensor_rotations[hit_sensor_ids].to(
        device=current_params.device, dtype=current_params.dtype
    )
    hit_global_positions = hit_global_positions.to(
        device=current_params.device, dtype=current_params.dtype
    )
    hit_directions = hit_directions.to(device=current_params.device, dtype=current_params.dtype)
    measured_local = measured_local.to(device=current_params.device, dtype=current_params.dtype)

    center, rotation = apply_alignment_torch(center0, rotation0, current_params)
    intersection = project_track_to_aligned_surface(
        hit_global_positions, hit_directions, center, rotation
    )
    local3 = torch.matmul(
        rotation.transpose(-1, -2), (intersection - center).unsqueeze(-1)
    ).squeeze(-1)
    predicted_local = local3[..., :2]
    return measured_local - predicted_local


def run_surface_alignment(config, target_layers, active_dof):
    """Surface mode: Adam on the target surfaces against frozen KF-smoothed tracks."""
    ACTIVE_DOF = active_dof
    TARGET_LAYERS = target_layers
    IDEAL_JSON_FILE = config.input_geom
    ALIGNED_JSON_FILE = config.output_geom
    SUMMARY_ROOT_FILE = config.summary_root
    STATES_ROOT_FILE = config.states_root

    print("======================================================")
    print(" Activate the ML-Driven detector alignment engine ")
    if config.iter_label is not None:
        print(f" Iteration {config.iter_label}")
    print("======================================================")

    if not os.path.exists(IDEAL_JSON_FILE):
        raise FileNotFoundError(f"ACTS geometry file '{IDEAL_JSON_FILE}' not found.")

    print(f"Reading detector misaligned geometry: {IDEAL_JSON_FILE}")
    (
        transforms_list,
        NUM_SENSORS,
        initial_positions,
        initial_rotations,
        id_to_idx_map,
    ) = load_geometry_from_json(IDEAL_JSON_FILE)
    print(f"Geometry file loaded successfully! {NUM_SENSORS} detector modules resolved.")

    if not os.path.exists(SUMMARY_ROOT_FILE) or not os.path.exists(STATES_ROOT_FILE):
        missing = [path for path in (SUMMARY_ROOT_FILE, STATES_ROOT_FILE) if not os.path.exists(path)]
        raise FileNotFoundError(f"ROOT file not found: {', '.join(missing)}")

    inputs = load_variable_information(
        SUMMARY_ROOT_FILE,
        STATES_ROOT_FILE,
        id_to_idx_map,
    )
    if inputs.measured_local is None or inputs.loc1_valid is None:
        raise ValueError("Surface mode requires measured_local and loc1_valid.")

    print_target_residual_stats(
        inputs.initial_residuals, inputs.hit_volume_ids, inputs.hit_layer_ids, TARGET_LAYERS,
        loc1_valid=inputs.loc1_valid,
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
    train_measured_local = inputs.measured_local[training_hit_mask]
    train_loc1_valid = inputs.loc1_valid[training_hit_mask]
    train_component_mask = torch.stack(
        (torch.ones_like(train_loc1_valid), train_loc1_valid), dim=1
    )
    train_covariances = inputs.covariances[training_hit_mask]
    train_hit_global_positions = inputs.hit_global_positions[training_hit_mask]
    train_hit_directions = inputs.hit_directions[training_hit_mask]
    train_hit_volume_ids = inputs.hit_volume_ids[training_hit_mask]
    train_hit_layer_ids = inputs.hit_layer_ids[training_hit_mask]

    print("Using PyTorch autograd residual projection; no C++ alignment Jacobian is imported.")

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
        optimizer.zero_grad()

        cur_res = engine(
            train_hit_sensor_ids,
            initial_positions,
            initial_rotations,
            train_measured_local,
            train_hit_global_positions,
            train_hit_directions,
        )

        chi2_loss = physics_informed_residual_loss(
            cur_res, train_covariances, component_mask=train_component_mask
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
            if target_mask is not None:
                engine.alignment_params[~target_mask] = 0.0
        alignment_lower = engine.alignment_params.new_tensor(
            [-ADAM_CLAMP_T] * 3 + [-ADAM_CLAMP_R] * 3
        )
        alignment_upper = -alignment_lower
        clamp_parameter_and_reset_adam(
            engine.alignment_params, alignment_lower, alignment_upper, optimizer
        )

        with torch.no_grad():
            if epoch % 10 == 0 or epoch == 1:
                current_lr = scheduler.get_last_lr()[0]
                cur_res = compute_autograd_residuals(
                    train_measured_local,
                    train_hit_global_positions,
                    train_hit_directions,
                    train_hit_sensor_ids,
                    initial_positions,
                    initial_rotations,
                    engine.alignment_params,
                )
                cur_chi2_per_hit = (cur_res ** 2) / torch.clamp(
                    train_covariances, min=EPS_COVARIANCE
                )
                cur_chi2_per_hit = torch.where(
                    train_component_mask,
                    cur_chi2_per_hit,
                    torch.zeros_like(cur_chi2_per_hit),
                )
                layer_info = []
                for vl, ll in TARGET_LAYERS:
                    vmask = (train_hit_volume_ids == vl) & (train_hit_layer_ids == ll)
                    if vmask.any():
                        lchi2 = cur_chi2_per_hit[vmask].mean().item()
                        layer_info.append(f"v{vl}l{ll}={lchi2:.2f}")
                # The Epoch line itself is never conditional: the invariant
                # extractor and the acceptance criteria both depend on it.
                print(f"Epoch [{epoch:03d}/{config.epochs}] | LR: {current_lr:.6f} | "
                      f"Chi2: {chi2_loss.item():.4f} | L2: {l2_reg.item():.6f} | "
                      f"Total: {loss.item():.4f} | "
                      + " | ".join(layer_info))

    predicted_deltas = engine.alignment_params.detach().cpu().numpy()

    predicted_delta_tensor = torch.tensor(predicted_deltas, dtype=torch.float32)

    # ── Per-layer residual diagnostics ──────────────────────────────────────
    # Before and after must share one residual definition.  The ROOT/KF residual
    # (l_x/y_hit - eLOC0/1_smt) and the geometric projection only agree
    # approximately at delta=0, so using the former as the baseline would fold
    # that discrepancy into the reported improvement.  Evaluate the same
    # projection at delta=0 instead.
    def _masked_residuals(deltas):
        residuals = compute_autograd_residuals(
            train_measured_local,
            train_hit_global_positions,
            train_hit_directions,
            train_hit_sensor_ids,
            initial_positions,
            initial_rotations,
            deltas,
        )
        return torch.where(
            train_component_mask, residuals, torch.zeros_like(residuals)
        ).numpy()

    with torch.no_grad():
        init_res_np = _masked_residuals(torch.zeros_like(predicted_delta_tensor))
        final_res = _masked_residuals(predicted_delta_tensor)
    vols_np  = train_hit_volume_ids.numpy()
    lays_np  = train_hit_layer_ids.numpy()

    target_metrics = print_per_layer_residual_diagnostics(
        init_res_np, final_res, vols_np, lays_np, TARGET_LAYERS,
        return_metrics=True, component_mask=train_component_mask,
    )

    # The decisive check vs TRUE misalignment.
    truth_metrics = validate_against_truth(
        transforms_list, predicted_deltas, config.nominal_file, TARGET_LAYERS
    )

    Path(ALIGNED_JSON_FILE).parent.mkdir(parents=True, exist_ok=True)
    update_geometry_to_json(
        transforms_list,
        predicted_deltas,
        ALIGNED_JSON_FILE,
        alignment_gain=config.alignment_gain,
    )

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

    return {
        "output_geom": ALIGNED_JSON_FILE,
        "target_metrics": target_metrics,
        "truth_metrics": truth_metrics,
        "predicted_deltas": predicted_deltas,
    }
