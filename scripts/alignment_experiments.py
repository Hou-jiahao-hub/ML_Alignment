"""Historical alignment experiments kept outside the production solver path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


HIT_WEIGHT_FLOOR = 0.3
HIT_QUALITY_INPUT_SIZE = 9
HIT_QUALITY_HIDDEN_ONE = 32
HIT_QUALITY_HIDDEN_TWO = 16
HIT_QUALITY_OUTPUT_SIZE = 1
EPS_WEIGHT = 1.0e-12


class JointBestCheckpoint:
    """Track and restore the lowest evaluated historical Adam objective."""

    def __init__(self, parameters, patience=0, min_delta=0.0):
        self.parameters = tuple(parameters)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.best_iteration = -1
        self._state = None

    def observe(self, loss, iteration):
        loss = float(loss)
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_iteration = int(iteration)
            self._state = tuple(parameter.detach().clone() for parameter in self.parameters)
        return (
            self.patience > 0
            and self._state is not None
            and int(iteration) - self.best_iteration >= self.patience
        )

    def restore(self):
        if self._state is None:
            raise RuntimeError("No joint checkpoint has been observed.")
        with torch.no_grad():
            for parameter, value in zip(self.parameters, self._state):
                parameter.copy_(value)


def joint_checkpoint_score(total_loss, invalid_fraction, max_invalid_fraction):
    """Reject checkpoints that reduce loss by dropping intersections."""
    if float(invalid_fraction) > float(max_invalid_fraction):
        return float("inf")
    return float(total_loss)


def refit_helices_adam(
    core,
    batches,
    geometry,
    initial,
    n_iter,
    max_invalid_fraction,
    lr_position,
    lr_direction,
    lr_rho_scaled,
    initial_loss,
):
    """Historical first-order measurement refit for run reproduction."""
    linear_initial, rho_initial = core.split_joint_track_parameters(initial)
    position = nn.Parameter(linear_initial[:, :3].clone())
    direction = nn.Parameter(linear_initial[:, 3:].clone())
    rho_scaled = nn.Parameter(rho_initial.clone())
    optimizer = optim.Adam(
        (
            {"params": [position], "lr": lr_position},
            {"params": [direction], "lr": lr_direction},
            {"params": [rho_scaled], "lr": lr_rho_scaled},
        )
    )
    checkpoint = JointBestCheckpoint((position, direction, rho_scaled))
    checkpoint.observe(initial_loss, iteration=-1)
    first_step = None
    for iteration in range(n_iter):
        optimizer.zero_grad()
        linear = torch.cat((position, direction), dim=1)
        parameters = core.assemble_joint_track_parameters(linear, rho_scaled)
        loss, invalid_fraction, component_mask = core._measurement_refit_objective(
            parameters, batches, geometry
        )
        if not component_mask.any():
            raise ValueError(
                "No valid non-target measurement components remain for helix refit."
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite non-target measurement helix loss.")
        loss.backward()
        before = (
            position.detach().clone(),
            direction.detach().clone(),
            rho_scaled.detach().clone(),
        )
        gradient_norms = {
            "position": float(position.grad.norm().detach().cpu()),
            "direction": float(direction.grad.norm().detach().cpu()),
            "rho_scaled": float(rho_scaled.grad.norm().detach().cpu()),
        }
        optimizer.step()
        with torch.no_grad():
            evaluated = core.assemble_joint_track_parameters(
                torch.cat((position, direction), dim=1), rho_scaled
            )
            evaluated_loss, evaluated_invalid, _evaluated_mask = (
                core._measurement_refit_objective(evaluated, batches, geometry)
            )
        evaluated_value = joint_checkpoint_score(
            float(evaluated_loss.cpu()), evaluated_invalid, max_invalid_fraction
        )
        if iteration == 0:
            updates = (
                position.detach() - before[0],
                direction.detach() - before[1],
                rho_scaled.detach() - before[2],
            )
            names = ("position", "direction", "rho_scaled")
            first_step = {
                "loss_before": float(loss.detach().cpu()),
                "loss_after": float(evaluated_loss.cpu()),
                "invalid_fraction_after": evaluated_invalid,
                "gradient_norms": gradient_norms,
                "update_rms": {
                    name: float(torch.sqrt(torch.mean(update.square())).cpu())
                    for name, update in zip(names, updates, strict=True)
                },
                "update_max": {
                    name: float(update.abs().max().cpu())
                    for name, update in zip(names, updates, strict=True)
                },
            }
        checkpoint.observe(evaluated_value, iteration=iteration)
    checkpoint.restore()
    fitted = core.assemble_joint_track_parameters(
        torch.cat((position, direction), dim=1), rho_scaled
    ).detach()
    return fitted, {
        "best_iteration": checkpoint.best_iteration,
        "first_step": first_step,
    }


class HitQualityAlignmentEngine(nn.Module):
    """Historical learned hit-weight surface alignment experiment."""

    def __init__(self, core, num_sensors):
        super().__init__()
        self.core = core
        self.num_sensors = num_sensors
        self.use_hit_weights = True
        self.alignment_params = nn.Parameter(torch.zeros(num_sensors, 6))
        self.hit_quality_net = nn.Sequential(
            nn.Linear(HIT_QUALITY_INPUT_SIZE, HIT_QUALITY_HIDDEN_ONE),
            nn.GELU(),
            nn.Linear(HIT_QUALITY_HIDDEN_ONE, HIT_QUALITY_HIDDEN_TWO),
            nn.GELU(),
            nn.Linear(HIT_QUALITY_HIDDEN_TWO, HIT_QUALITY_OUTPUT_SIZE),
        )

    def forward(
        self,
        hit_sensor_ids,
        track_params,
        sensor_center_positions,
        sensor_rotations,
        track_chi2_ndf,
        measured_local,
        hit_global_positions,
        hit_directions,
    ):
        residuals = self.core.compute_autograd_residuals(
            measured_local,
            hit_global_positions,
            hit_directions,
            hit_sensor_ids,
            sensor_center_positions,
            sensor_rotations,
            self.alignment_params,
        )
        hit_centers = sensor_center_positions[hit_sensor_ids]
        features = torch.cat((track_params, hit_centers, track_chi2_ndf), dim=-1)
        raw_logits = self.hit_quality_net(features)
        weights = HIT_WEIGHT_FLOOR + (1.0 - HIT_WEIGHT_FLOOR) * torch.sigmoid(
            raw_logits
        )
        return residuals, weights


def normalize_hit_weights(
    hit_weights, hit_volume_ids=None, hit_layer_ids=None
):
    if hit_volume_ids is None and hit_layer_ids is None:
        return hit_weights / (torch.mean(hit_weights) + EPS_WEIGHT)
    if hit_volume_ids is None or hit_layer_ids is None:
        raise ValueError("hit_volume_ids and hit_layer_ids must be provided together.")
    if len(hit_weights) != len(hit_volume_ids) or len(hit_weights) != len(hit_layer_ids):
        raise ValueError(
            "Hit weight and layer identifier tensors must have matching lengths."
        )
    volume_ids = hit_volume_ids.to(device=hit_weights.device)
    layer_ids = hit_layer_ids.to(device=hit_weights.device)
    unique_layers = torch.unique(torch.stack((volume_ids, layer_ids), dim=1), dim=0)
    normalized = torch.zeros_like(hit_weights)
    for volume, layer in unique_layers:
        layer_mask = (volume_ids == volume) & (layer_ids == layer)
        scale = torch.mean(hit_weights[layer_mask]) + EPS_WEIGHT
        normalized += layer_mask.to(hit_weights.dtype).unsqueeze(-1) * (
            hit_weights / scale
        )
    return normalized


def weighted_residual_loss(
    residuals,
    covariances,
    hit_weights,
    covariance_floor,
    component_mask=None,
    hit_volume_ids=None,
    hit_layer_ids=None,
):
    raw_chi2 = residuals.square() / torch.clamp(
        covariances, min=covariance_floor
    )
    normalized = normalize_hit_weights(
        hit_weights, hit_volume_ids, hit_layer_ids
    )
    weighted = normalized * raw_chi2
    if component_mask is None:
        return torch.mean(weighted)
    return torch.sum(
        torch.where(component_mask, weighted, torch.zeros_like(weighted))
    ) / component_mask.sum().clamp(min=1)


def run_joint_alignment(
    core,
    batches,
    geometry,
    config,
    active_dof,
    initial_track_params=None,
):
    """Historical joint Adam track/geometry alignment experiment."""
    seed_helices = core.seed_helices
    split_joint_track_parameters = core.split_joint_track_parameters
    assemble_joint_track_parameters = core.assemble_joint_track_parameters
    joint_forward = core._joint_forward
    clamp_parameter_and_reset_adam = core.clamp_parameter_and_reset_adam
    covariance_floor = core.EPS_COVARIANCE
    use_cuda = bool(config.gpu and torch.cuda.is_available())
    if config.gpu and not use_cuda:
        print("  [warn] --gpu requested but CUDA is unavailable; using CPU float64.")
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32 if use_cuda else torch.float64
    torch.manual_seed(config.rng_seed)
    np.random.seed(config.rng_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.rng_seed)

    if initial_track_params is None:
        initial_track_params = seed_helices(batches, device=device, dtype=dtype)
    initial_track_params = initial_track_params.to(device=device, dtype=dtype)
    initial_linear, initial_rho_scaled = split_joint_track_parameters(initial_track_params)
    track_linear = nn.Parameter(initial_linear.clone())
    track_rho_scaled = nn.Parameter(initial_rho_scaled.clone())
    n_surfaces = len(geometry.centers)
    translation = nn.Parameter(torch.zeros((n_surfaces, 3), device=device, dtype=dtype))
    rotation = nn.Parameter(torch.zeros((n_surfaces, 3), device=device, dtype=dtype))
    dof_mask = torch.zeros(6, device=device, dtype=dtype)
    dof_mask[list(active_dof)] = 1.0
    target_mask = geometry.target_mask.to(device=device, dtype=dtype).unsqueeze(-1)
    parameter_groups = [
        {"params": [translation], "lr": config.joint_lr_trans},
        {"params": [rotation], "lr": config.joint_lr_trans / config.rot_lr_scale},
    ]
    if not config.freeze_tracks:
        parameter_groups[0:0] = [
            {"params": [track_linear], "lr": config.joint_lr_track},
            {"params": [track_rho_scaled], "lr": config.joint_lr_track},
        ]
    optimizer = optim.Adam(parameter_groups)
    checkpoint = JointBestCheckpoint(
        (track_linear, track_rho_scaled, translation, rotation),
        patience=config.joint_patience,
        min_delta=config.joint_min_delta,
    )
    history = []
    checkpoint_history = []
    invalid_history = []
    n_tracks = len(track_linear)
    batch_size = (
        min(config.track_batch_size, n_tracks)
        if config.track_batch_size > 0
        else n_tracks
    )
    with torch.no_grad():
        initial_tracks = assemble_joint_track_parameters(track_linear, track_rho_scaled)
        initial_alignment = torch.zeros((n_surfaces, 6), device=device, dtype=dtype)
        initial_residuals, _initial_covariance, initial_component_mask, _initial_valid = (
            joint_forward(initial_tracks, initial_alignment, batches, geometry)
        )
        initial_weighted = initial_residuals.square() / torch.clamp(
            _initial_covariance, min=covariance_floor
        )
        initial_total_loss = float((torch.sum(torch.where(
            initial_component_mask, initial_weighted, torch.zeros_like(initial_weighted)
        )) / initial_component_mask.sum().clamp(min=1)).cpu())
        initial_invalid_fraction = 1.0 - float(
            _initial_valid[batches.hit_mask.to(device)].float().mean().item()
        )
    initial_checkpoint_loss = joint_checkpoint_score(
        initial_total_loss,
        initial_invalid_fraction,
        config.joint_max_invalid_fraction,
    )
    if not np.isfinite(initial_checkpoint_loss):
        raise ValueError(
            "Initial helix seed has too many invalid intersections for a safe checkpoint: "
            f"{initial_invalid_fraction:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    checkpoint.observe(initial_checkpoint_loss, iteration=-1)
    checkpoint_history.append((-1, initial_total_loss))

    if config.joint_iters < 1:
        raise ValueError("joint_iters must be at least 1.")
    checkpoint_interval = max(1, int(config.joint_checkpoint_interval))
    last_total_loss = float("inf")
    stopped_early = False
    for iteration in range(config.joint_iters):
        start = (iteration * batch_size) % n_tracks
        track_indices = torch.arange(start, min(start + batch_size, n_tracks), device=device)
        if len(track_indices) < batch_size:
            track_indices = torch.cat((
                track_indices,
                torch.arange(0, batch_size - len(track_indices), device=device),
            ))
        optimizer.zero_grad()
        track_parameters = assemble_joint_track_parameters(track_linear, track_rho_scaled)
        alignment = torch.cat((translation, rotation), dim=1)
        alignment = alignment * target_mask * dof_mask
        residuals, covariance, component_mask, intersection_valid = joint_forward(
            track_parameters, alignment, batches, geometry, track_indices
        )
        weighted = residuals.square() / torch.clamp(covariance, min=covariance_floor)
        data_loss = torch.sum(torch.where(component_mask, weighted, torch.zeros_like(weighted)))
        data_loss = data_loss / component_mask.sum().clamp(min=1)
        l2 = config.lambda_l2 * alignment.square().sum()
        loss = data_loss + l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite joint loss at iteration {iteration}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([track_linear, track_rho_scaled, translation, rotation], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            translation.mul_(target_mask * dof_mask[:3])
            rotation.mul_(target_mask * dof_mask[3:])
        clamp_parameter_and_reset_adam(
            translation, -config.joint_clamp_t, config.joint_clamp_t, optimizer
        )
        clamp_parameter_and_reset_adam(
            rotation, -config.joint_clamp_r, config.joint_clamp_r, optimizer
        )
        invalid_fraction = 1.0 - float(
            intersection_valid[batches.hit_mask.to(device)[track_indices]].float().mean().item()
        )
        history.append(float(data_loss.detach().cpu()))
        invalid_history.append(invalid_fraction)
        evaluate_checkpoint = (
            iteration == 0
            or (iteration + 1) % checkpoint_interval == 0
            or iteration == config.joint_iters - 1
        )
        if evaluate_checkpoint:
            with torch.no_grad():
                evaluated_tracks = assemble_joint_track_parameters(track_linear, track_rho_scaled)
                evaluated_alignment = torch.cat((translation, rotation), dim=1)
                evaluated_alignment = evaluated_alignment * target_mask * dof_mask
                evaluated_residuals, evaluated_covariance, evaluated_mask, evaluated_valid = joint_forward(
                    evaluated_tracks, evaluated_alignment, batches, geometry
                )
                evaluated_weighted = evaluated_residuals.square() / torch.clamp(
                    evaluated_covariance, min=covariance_floor
                )
                evaluated_data_loss = torch.sum(torch.where(
                    evaluated_mask, evaluated_weighted, torch.zeros_like(evaluated_weighted)
                )) / evaluated_mask.sum().clamp(min=1)
                evaluated_total_loss = (
                    evaluated_data_loss + config.lambda_l2 * evaluated_alignment.square().sum()
                )
                last_total_loss = float(evaluated_total_loss.cpu())
                evaluated_invalid_fraction = 1.0 - float(
                    evaluated_valid[batches.hit_mask.to(device)].float().mean().item()
                )
            checkpoint_loss = joint_checkpoint_score(
                last_total_loss,
                evaluated_invalid_fraction,
                config.joint_max_invalid_fraction,
            )
            checkpoint_history.append((iteration, last_total_loss))
            if checkpoint.observe(checkpoint_loss, iteration):
                stopped_early = True
                print(
                    f"Joint early stop at iteration {iteration}; "
                    f"best={checkpoint.best_loss:.4f} at iteration {checkpoint.best_iteration}."
                )
                break
        if iteration % 100 == 0 or iteration == config.joint_iters - 1:
            warning = " WARNING: >5%" if invalid_fraction > 0.05 else ""
            print(
                f"Joint [{iteration:04d}/{config.joint_iters}] chi2/component={history[-1]:.4f} "
                f"invalid={invalid_fraction:.2%}{warning}"
            )

    checkpoint.restore()
    with torch.no_grad():
        track_parameters = assemble_joint_track_parameters(track_linear, track_rho_scaled)
        predicted = torch.cat((translation, rotation), dim=1)
        predicted = predicted * target_mask * dof_mask
        final_residuals, _covariance, final_component_mask, _valid = joint_forward(
            track_parameters, predicted, batches, geometry
        )
        final_invalid_fraction = 1.0 - float(
            _valid[batches.hit_mask.to(device)].float().mean().cpu()
        )
    if final_invalid_fraction > config.joint_max_invalid_fraction:
        raise ValueError(
            "Best joint checkpoint exceeds the invalid-intersection limit: "
            f"{final_invalid_fraction:.2%} > {config.joint_max_invalid_fraction:.2%}."
        )
    return {
        "predicted_deltas": predicted.detach().cpu().double().numpy(),
        "track_params": track_parameters.detach().cpu().double().numpy(),
        "final_residuals": final_residuals.detach().cpu(),
        "final_component_mask": final_component_mask.detach().cpu(),
        "initial_residuals": initial_residuals.detach().cpu(),
        "initial_component_mask": initial_component_mask.detach().cpu(),
        "history": history,
        "checkpoint_history": checkpoint_history,
        "best_total_loss": checkpoint.best_loss,
        "last_total_loss": last_total_loss,
        "best_iteration": checkpoint.best_iteration,
        "stopped_early": stopped_early,
        "initial_total_loss": initial_total_loss,
        "initial_invalid_fraction": initial_invalid_fraction,
        "invalid_fraction": final_invalid_fraction,
    }
