from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import json
import awkward as ak
import numpy as np
import pytest
import uproot


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "ml_alignment_autograd.py"
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("ml_alignment_autograd_joint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_helix_point_and_tangent_have_stable_straight_line_limit():
    m = load_module()
    track = torch.tensor([[1.0, 2.0, 3.0, 0.3, 0.4, 0.0]], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([7.0], dtype=torch.float64)

    point = m.helix_point(track, s)
    tangent = m.helix_tangent(track, s)
    expected = torch.tensor(
        [[1.0 + 7.0 * torch.cos(torch.tensor(0.3)),
          2.0 + 7.0 * torch.sin(torch.tensor(0.3)),
          3.0 + 7.0 * 0.4]],
        dtype=torch.float64,
    )

    assert torch.allclose(point, expected, atol=1e-7)
    assert torch.allclose(
        tangent,
        torch.tensor([[torch.cos(torch.tensor(0.3)), torch.sin(torch.tensor(0.3)), 0.4]], dtype=torch.float64),
        atol=1e-7,
    )
    point.sum().backward()
    assert torch.isfinite(track.grad).all()


def test_helix_point_follows_signed_circle():
    m = load_module()
    track = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=torch.float64)
    s = torch.tensor([5.0 * torch.pi], dtype=torch.float64)

    point = m.helix_point(track, s)
    tangent = m.helix_tangent(track, s)

    assert torch.allclose(point, torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float64), atol=1e-7)
    assert torch.allclose(tangent, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-7)


def test_intersect_plane_matches_closed_form_and_masks_parallel_track():
    m = load_module()
    tracks = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    centers = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64)
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    s, points, valid = m.intersect_plane(tracks, centers, normals)

    assert valid.tolist() == [True, False]
    assert torch.allclose(s[0], torch.tensor(4.0, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(points[0], centers[0], atol=1e-8)
    assert torch.isfinite(points).all()


def test_intersect_plane_keeps_odd_outer_radius_but_rejects_runaway_path():
    m = load_module()
    tracks = torch.zeros((2, 6), dtype=torch.float64)
    centers = torch.tensor([[1000.0, 0.0, 0.0], [1300.0, 0.0, 0.0]], dtype=torch.float64)
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

    _s, _points, valid = m.intersect_plane(tracks, centers, normals)

    assert valid.tolist() == [True, False]


def test_newton_invalid_candidate_keeps_last_feasible_path_and_stays_invalid():
    m = load_module()
    previous = torch.tensor([12.0, 8.0])
    candidate = torch.tensor([13.0, float("inf")])
    valid = torch.tensor([True, True])
    candidate_ok = torch.tensor([True, False])

    path, updated_valid = m.accept_newton_candidate(
        previous, candidate, valid, candidate_ok
    )

    assert torch.equal(path, torch.tensor([13.0, 8.0]))
    assert updated_valid.tolist() == [True, False]


def _make_inputs(m, frame: pd.DataFrame):
    n = len(frame)
    return m.AlignmentInputs(
        hit_sensor_ids=torch.tensor(frame.sensor_idx.to_numpy(), dtype=torch.long),
        initial_residuals=torch.zeros((n, 2)),
        covariances=torch.ones((n, 2)),
        track_params=torch.zeros((n, 5)),
        track_chi2_ndf=torch.ones((n, 1)),
        hit_global_positions=torch.tensor(frame[["g_x_smt", "g_y_smt", "g_z_smt"]].to_numpy(), dtype=torch.float32),
        hit_directions=torch.ones((n, 3)),
        hit_qoverp=torch.ones((n, 1)),
        hit_volume_ids=torch.tensor(frame.volume_id.to_numpy(), dtype=torch.long),
        hit_layer_ids=torch.tensor(frame.layer_id.to_numpy(), dtype=torch.long),
        hit_module_ids=torch.tensor(frame.module_id.to_numpy(), dtype=torch.long),
        hit_event_ids=torch.tensor(frame.event_nr.to_numpy(), dtype=torch.long),
        hit_track_ids=torch.tensor(frame.track_nr.to_numpy(), dtype=torch.long),
        measured_local=torch.tensor(frame[["l_x_hit", "l_y_filled"]].to_numpy(), dtype=torch.float32),
        loc1_valid=torch.tensor(frame.loc1_valid.to_numpy(), dtype=torch.bool),
        raw_covariances=torch.tensor(frame[["var0", "var1"]].to_numpy(), dtype=torch.float32),
        merged_df=frame,
    )


def test_build_track_batches_is_deterministic_and_preserves_1d_mask():
    m = load_module()
    rows = []
    for track_nr in (11, 12):
        for j, radius in enumerate((30.0, 10.0, 20.0)):
            rows.append(
                dict(
                    event_nr=1,
                    track_nr=track_nr,
                    sensor_idx=j,
                    volume_id=17 if j == 1 else 10,
                    layer_id=8 if j == 1 else j,
                    module_id=j + 1,
                    g_x_smt=radius,
                    g_y_smt=0.0,
                    g_z_smt=float(j),
                    l_x_hit=float(j),
                    l_y_filled=0.0,
                    loc1_valid=j != 2,
                    var0=0.01,
                    var1=0.04,
                )
            )
    frame = pd.DataFrame(rows)
    a = m.build_track_batches(
        _make_inputs(m, frame), ((17, 8),), min_hits=2, min_span=0.0
    )
    b = m.build_track_batches(
        _make_inputs(m, frame.sample(frac=1.0, random_state=7).reset_index(drop=True)),
        ((17, 8),),
        min_hits=2,
        min_span=0.0,
    )

    assert a.measured_local.shape == (2, 3, 2)
    assert torch.equal(a.sensor_idx, b.sensor_idx)
    assert torch.equal(a.hit_mask, b.hit_mask)
    assert torch.equal(a.loc1_mask, b.loc1_mask)
    assert a.sensor_idx[0].tolist() == [1, 2, 0]
    assert a.loc1_mask[0].tolist() == [True, False, True]
    assert torch.allclose(a.covariances[a.hit_mask][:, 1], torch.ones(6))


def test_build_track_batches_carries_unfloored_variance_in_covariance_order():
    m = load_module()
    rows = []
    for track_nr, n_hits in ((11, 3), (12, 2)):
        for j in range(n_hits):
            rows.append(
                dict(
                    event_nr=1,
                    track_nr=track_nr,
                    sensor_idx=j,
                    volume_id=24 if j == 0 else 10,
                    layer_id=4 if j == 0 else j,
                    module_id=j + 1,
                    g_x_smt=10.0 * (3 - j),
                    g_y_smt=0.0,
                    g_z_smt=float(j),
                    l_x_hit=0.0,
                    l_y_filled=0.0,
                    loc1_valid=True,
                    var0=0.01 * (j + 1),
                    var1=0.04 * (j + 1),
                )
            )
    frame = pd.DataFrame(rows)
    inputs = _make_inputs(m, frame)
    inputs.covariances = inputs.raw_covariances.clone()
    floored = torch.tensor(frame.volume_id.to_numpy() == 24)
    inputs.covariances[floored, 1] = m.STRIP_Y_FLOOR

    batches = m.build_track_batches(inputs, ((24, 4),), min_hits=1, min_span=0.0)

    assert batches.raw_covariances.shape == batches.covariances.shape
    assert torch.equal(batches.raw_covariances[~batches.hit_mask], torch.ones(1, 2))
    assert torch.equal(batches.covariances[~batches.hit_mask], torch.ones(1, 2))
    floor_active = (batches.raw_covariances < batches.covariances) & batches.hit_mask[..., None]
    assert torch.equal(
        floor_active[..., 1], batches.is_target & batches.hit_mask
    )
    assert not floor_active[..., 0].any()
    unchanged = ~floor_active & batches.hit_mask[..., None]
    assert torch.equal(batches.raw_covariances[unchanged], batches.covariances[unchanged])


def test_build_track_batches_leaves_raw_variance_absent_when_input_has_none():
    m = load_module()
    frame = pd.DataFrame(
        [
            dict(
                event_nr=1, track_nr=1, sensor_idx=j,
                volume_id=17 if j == 0 else 10, layer_id=8 if j == 0 else j,
                module_id=j + 1, g_x_smt=10.0 * (j + 1), g_y_smt=0.0, g_z_smt=0.0,
                l_x_hit=0.0, l_y_filled=0.0, loc1_valid=True, var0=0.01, var1=0.04,
            )
            for j in range(3)
        ]
    )
    inputs = _make_inputs(m, frame)
    inputs.raw_covariances = None

    batches = m.build_track_batches(inputs, ((17, 8),), min_hits=1, min_span=0.0)

    assert batches.raw_covariances is None


def _initialization_frame(target_x=200.0, target_measurement=0.0):
    rows = []
    for index, x in enumerate(np.linspace(0.0, 400.0, 9)):
        rows.append(
            dict(
                event_nr=1,
                track_nr=7,
                sensor_idx=index,
                volume_id=10,
                layer_id=index + 1,
                module_id=index + 1,
                g_x_smt=x,
                g_y_smt=0.0,
                g_z_smt=0.0,
                l_x_hit=0.0,
                l_y_filled=0.0,
                loc1_valid=True,
                var0=0.01,
                var1=0.04,
            )
        )
    rows.append(
        dict(
            event_nr=1,
            track_nr=7,
            sensor_idx=9,
            volume_id=17,
            layer_id=8,
            module_id=10,
            g_x_smt=target_x,
            g_y_smt=1000.0,
            g_z_smt=0.0,
            l_x_hit=target_measurement,
            l_y_filled=target_measurement,
            loc1_valid=True,
            var0=0.01,
            var1=0.04,
        )
    )
    return pd.DataFrame(rows)


def test_joint_initialization_mask_excludes_targets_and_enforces_span():
    m = load_module()
    batches = m.build_track_batches(
        _make_inputs(m, _initialization_frame()),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )

    assert batches.fit_mask.sum().item() == 9
    assert not (batches.fit_mask & batches.is_target).any()
    assert batches.fit_span.tolist() == pytest.approx([400.0])
    assert batches.selection_stats == {
        "input_tracks": 1,
        "retained_tracks": 1,
        "excluded_tracks": 0,
    }

    short = _initialization_frame()
    short.loc[short.volume_id == 10, "g_x_smt"] *= 0.5
    with pytest.raises(ValueError, match="No tracks satisfy"):
        m.build_track_batches(
            _make_inputs(m, short), ((17, 8),), min_hits=8, min_span=300.0
        )


def test_smt_seed_ignores_target_layer_points():
    m = load_module()
    original = m.build_track_batches(
        _make_inputs(m, _initialization_frame()),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    shifted_frame = _initialization_frame()
    shifted_frame.loc[shifted_frame.volume_id == 17, ["g_y_smt", "g_z_smt"]] = [
        -5000.0,
        9000.0,
    ]
    shifted = m.build_track_batches(
        _make_inputs(m, shifted_frame),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )

    seed_a = m.seed_helices(original, n_iter=5, dtype=torch.float64)
    seed_b = m.seed_helices(shifted, n_iter=5, dtype=torch.float64)

    torch.testing.assert_close(seed_a, seed_b, rtol=0.0, atol=0.0)


def test_joint_initializers_fail_closed_without_non_target_fit_mask():
    m = load_module()
    batches = m.build_track_batches(
        _make_inputs(m, _initialization_frame()),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    batches.fit_mask = None

    with pytest.raises(ValueError, match="fit_mask"):
        m.seed_helices(batches, n_iter=1, dtype=torch.float64)
    with pytest.raises(ValueError, match="fit_mask"):
        m._measurement_refit_objective(
            torch.zeros((1, 6), dtype=torch.float64),
            batches,
            None,
        )


def test_measurement_refit_ignores_target_measurements_and_reports_chi2():
    m = load_module()
    batches_a = m.build_track_batches(
        _make_inputs(m, _initialization_frame(target_measurement=0.0)),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    batches_b = m.build_track_batches(
        _make_inputs(m, _initialization_frame(target_measurement=5000.0)),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    x_positions = torch.cat(
        (torch.linspace(0.0, 400.0, 9, dtype=torch.float64), torch.tensor([200.0]))
    )
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    geometry = m.JointGeometry(
        transforms=[],
        input_transforms=[],
        centers=torch.stack(
            (x_positions, torch.zeros_like(x_positions), torch.zeros_like(x_positions)),
            dim=1,
        ),
        rotations=rotation.expand(10, -1, -1).clone(),
        target_mask=torch.tensor([False] * 9 + [True]),
        id_to_idx={},
        input_indices=[9],
    )
    initial = torch.zeros((1, 6), dtype=torch.float64)

    fitted_a, stats_a = m.fit_helices_to_measurements(
        batches_a, geometry, initial, n_iter=3, dtype=torch.float64, solver="adam"
    )
    fitted_b, stats_b = m.fit_helices_to_measurements(
        batches_b, geometry, initial, n_iter=3, dtype=torch.float64, solver="adam"
    )

    torch.testing.assert_close(fitted_a, fitted_b, rtol=0.0, atol=0.0)
    assert stats_a["chi2_per_component"] == pytest.approx(0.0)
    assert stats_a["initial_chi2_per_component"] == pytest.approx(0.0)
    assert stats_a["best_iteration"] == -1
    assert stats_a["improved"] is False
    assert stats_a["parameter_updates"]["position_rms"] == pytest.approx(0.0)
    assert stats_a["parameter_updates"]["direction_rms"] == pytest.approx(0.0)
    assert stats_a["parameter_updates"]["rho_rms"] == pytest.approx(0.0)
    assert stats_a["first_step"]["loss_before"] == pytest.approx(0.0)
    assert stats_a["first_step"]["loss_after"] == pytest.approx(0.0)
    assert stats_a["first_step"]["gradient_norms"] == pytest.approx(
        {"position": 0.0, "direction": 0.0, "rho_scaled": 0.0}
    )
    assert stats_a["first_step"]["update_max"] == pytest.approx(
        {"position": 0.0, "direction": 0.0, "rho_scaled": 0.0}
    )
    assert stats_a["invalid_fraction"] == pytest.approx(0.0)
    assert stats_b == stats_a


def test_joint_seed_cli_defaults_to_non_target_measurement_refit():
    m = load_module()
    args = m.build_argument_parser().parse_args(["--track-mode", "free"])
    config = m.alignment_config_from_args(args)

    assert config.joint_seed_mode == "non-target-measurement"
    assert config.joint_seed_min_hits == 8
    assert config.joint_seed_min_span == 300.0
    assert config.joint_smt_seed_iters == 200
    assert config.joint_measurement_refit_iters == 200
    assert config.measurement_refit_lr_position == pytest.approx(1e-3)
    assert config.measurement_refit_lr_direction == pytest.approx(1e-5)
    assert config.measurement_refit_lr_rho_scaled == pytest.approx(1e-3)


def test_joint_config_defaults_keep_surface_mode_compatible():
    m = load_module()
    config = m.AlignmentRunConfig()

    assert config.track_mode == "surface"
    assert config.joint_iters == 1000
    assert config.min_hits == 8
    assert config.track_batch_size == 0
    assert config.rng_seed == 42
    assert config.nominal_full_geom.endswith("data/geometry/odd_transforms.json")


def test_block_lm_is_the_default_free_geometry_solver():
    m = load_module()
    args = m.build_argument_parser().parse_args(["--track-mode", "free"])
    config = m.alignment_config_from_args(args)

    assert config.joint_solver == "block-lm"
    assert config.lm_max_iters == 60
    assert config.fisher_eigen_floor == pytest.approx(1e-4)


def test_block_lm_step_caps_span_the_raw_misalignment_without_a_warm_start():
    """The step caps must let block-LM traverse a raw injected misalignment.

    ``alignment_odd.py`` injects up to 1.5x its nominal magnitude, i.e. about
    0.77 mm and 0.030 rad.  A per-iteration cap only lets a sensor travel
    ``cap * accepted_steps``, so a cap that needs more accepted steps than
    ``lm_max_iters`` allows leaves sensors stranded at their injected value.
    Measured on the full 20260901 joint-only run: at 1e-3 rad, 14 of 2268
    sensors stalled after 1-6 capped steps and never moved again, even with
    60 iterations available.
    """
    m = load_module()
    args = m.build_argument_parser().parse_args(["--track-mode", "free"])
    config = m.alignment_config_from_args(args)

    max_injected_shift_mm = 2.0 * 1.5 * 0.256  # asymmetric_random upper bound
    max_injected_rotation_rad = 0.02 * 1.5

    assert config.lm_max_step_t == pytest.approx(0.1)
    assert config.lm_max_step_r == pytest.approx(5e-3)
    assert max_injected_shift_mm / config.lm_max_step_t <= config.lm_max_iters
    assert max_injected_rotation_rad / config.lm_max_step_r <= config.lm_max_iters


def test_fisher_regularization_floors_weak_normalized_mode():
    m = load_module()
    fisher = torch.diag(torch.tensor([100.0, 1e-8], dtype=torch.float64))

    regularized, stats = m.regularize_fisher_matrix(
        fisher, damping=0.0, eigen_floor_ratio=1e-4
    )

    eigenvalues = torch.linalg.eigvalsh(regularized)
    assert eigenvalues[0].item() == pytest.approx(0.01, rel=1e-10)
    assert stats["condition_number"] == pytest.approx(1e10)
    assert stats["regularization"] == pytest.approx(0.01 - 1e-8)


def _block_lm_translation_problem(m, non_target_measurement=0.0):
    dtype = torch.float64
    n_tracks = 12
    plane_x = torch.tensor([5.0, 10.0], dtype=dtype)
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    rotations = rotation.expand(2, -1, -1).clone()
    centers = torch.stack((plane_x, torch.tensor([0.0, 0.4]), torch.zeros(2)), dim=1)
    target_mask = torch.tensor([False, True])
    geometry = m.JointGeometry(
        transforms=[_surface(10, 1, 1, 5.0), _surface(17, 8, 2, 10.0)],
        input_transforms=[_surface(17, 8, 2, 10.0)],
        centers=centers,
        rotations=rotations,
        target_mask=target_mask,
        id_to_idx={},
        input_indices=[1],
    )
    y0 = torch.linspace(-1.0, 1.0, n_tracks, dtype=dtype)
    z0 = torch.linspace(0.5, -0.5, n_tracks, dtype=dtype)
    tracks = torch.stack(
        (torch.zeros_like(y0), y0, z0, torch.zeros_like(y0), torch.zeros_like(y0), torch.zeros_like(y0)),
        dim=1,
    )
    measured = torch.stack((y0, z0), dim=1)[:, None, :].expand(-1, 2, -1).clone()
    measured[:, 0, 0] = non_target_measurement
    batches = m.TrackBatches(
        measured_local=measured,
        covariances=torch.full((n_tracks, 2, 2), 0.01, dtype=dtype),
        sensor_idx=torch.tensor([0, 1]).expand(n_tracks, -1),
        hit_mask=torch.ones((n_tracks, 2), dtype=torch.bool),
        loc1_mask=torch.ones((n_tracks, 2), dtype=torch.bool),
        g_smt=torch.zeros((n_tracks, 2, 3), dtype=dtype),
        p_smt=torch.zeros((n_tracks, 2, 3), dtype=dtype),
        volume_id=torch.tensor([10, 17]).expand(n_tracks, -1),
        layer_id=torch.tensor([1, 8]).expand(n_tracks, -1),
        is_target=target_mask.expand(n_tracks, -1),
        event_track_ids=torch.stack((torch.zeros(n_tracks, dtype=torch.long), torch.arange(n_tracks)), dim=1),
        fit_mask=(~target_mask).expand(n_tracks, -1),
    )
    return batches, geometry, tracks


def test_block_lm_recovers_target_translation_without_changing_tracks():
    m = load_module()
    batches, geometry, tracks = _block_lm_translation_problem(m)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=8,
        lambda_l2=0.0, make_plot=False,
    )

    result = m.run_block_lm_alignment(
        batches, geometry, config, active_dof=(0,), initial_track_params=tracks
    )

    assert result["solver"] == "block-lm"
    assert result["best_iteration"] >= 0
    assert result["best_total_loss"] < result["initial_total_loss"]
    # The solver stops at the pre-registered stationary point, not at an
    # arbitrary parameter precision, so the tolerance is that threshold read
    # back through this problem's closed form rather than a hand-picked number.
    # The loc0 residual is exactly ``0.4 + tx`` with unit derivative over m = 12
    # rows at sigma = 0.1, giving H = m / sigma^2 and lambda2 = m e^2 / sigma^2
    # across n = 24 components:
    #     m e^2 / sigma^2 / n <= threshold  =>  |e| <= sqrt(n * threshold * sigma^2 / m)
    stationary_bound = math.sqrt(24 * m.BLOCK_LM_DECREMENT_STATIONARY * 0.01 / 12)
    assert stationary_bound == pytest.approx(1.4142e-4, rel=1e-4)
    assert abs(result["predicted_deltas"][1, 0] + 0.4) <= stationary_bound
    assert result["block_lm_summary"]["status_counts"]["converged"] == 1
    assert result["block_lm_summary"]["converged_detail"]["interior"] == 1
    np.testing.assert_array_equal(result["track_params"], tracks.numpy())
    assert result["fisher_diagnostics"]


def test_block_lm_geometry_is_independent_of_non_target_measurements():
    m = load_module()
    a_batches, geometry, tracks = _block_lm_translation_problem(m, 0.0)
    b_batches, _, _ = _block_lm_translation_problem(m, 1.0e6)
    config = m.AlignmentRunConfig(
        track_mode="free", joint_solver="block-lm", lm_max_iters=4,
        lambda_l2=0.0, make_plot=False,
    )

    a = m.run_block_lm_alignment(a_batches, geometry, config, (0,), tracks)
    b = m.run_block_lm_alignment(b_batches, geometry, config, (0,), tracks)

    np.testing.assert_array_equal(a["predicted_deltas"], b["predicted_deltas"])


def test_cli_freeze_tracks_reaches_alignment_config():
    m = load_module()
    parser = m.build_argument_parser()

    args = parser.parse_args(["--track-mode", "free", "--freeze-tracks"])
    config = m.alignment_config_from_args(args)

    assert config.track_mode == "free"
    assert config.freeze_tracks is True


def test_both_track_modes_share_one_default_dof():
    """free used to default to (0,1,3,4), which cannot express rz at all."""
    m = load_module()

    surface = m.resolve_active_dof(m.AlignmentRunConfig(track_mode="surface"))
    free = m.resolve_active_dof(m.AlignmentRunConfig(track_mode="free"))

    assert surface == m.DEFAULT_ACTIVE_DOF
    assert free == m.DEFAULT_ACTIVE_DOF


def test_joint_default_active_dof_constant_is_gone():
    m = load_module()

    assert not hasattr(m, "JOINT_DEFAULT_ACTIVE_DOF")


def test_default_dof_includes_rz():
    """Canary: rz is the only rotation the current injection actually produces.

    Measured over misalignment_applied vs odd_transforms (186 surfaces):
    tx/ty ~0.52 mm, rz ~0.021 rad, and tz/rx/ry exactly zero -- alignment_odd.py
    hardcodes rot_local_x/y to 0 despite its rx/ry flags being set.
    """
    m = load_module()

    assert 5 in m.DEFAULT_ACTIVE_DOF


def test_run_joint_alignment_signature_default_matches():
    m = load_module()

    default = inspect.signature(m.run_joint_alignment).parameters["active_dof"].default

    assert default == m.DEFAULT_ACTIVE_DOF


def test_empty_active_dof_string_does_not_bypass_mode_default():
    """parse_active_dof('') used to hand free mode the surface default silently."""
    m = load_module()

    resolved = m.resolve_active_dof(
        m.AlignmentRunConfig(track_mode="free", active_dof="")
    )

    assert resolved == m.DEFAULT_ACTIVE_DOF


def test_joint_checkpoint_restores_best_parameters_and_stops_after_patience():
    m = load_module()
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    checkpoint = m.JointBestCheckpoint((parameter,), patience=2, min_delta=0.0)

    assert checkpoint.observe(5.0, iteration=0) is False
    with torch.no_grad():
        parameter.fill_(0.5)
    assert checkpoint.observe(4.0, iteration=1) is False
    with torch.no_grad():
        parameter.fill_(9.0)
    assert checkpoint.observe(6.0, iteration=2) is False
    with torch.no_grad():
        parameter.fill_(10.0)
    assert checkpoint.observe(7.0, iteration=3) is True

    checkpoint.restore()
    assert parameter.item() == 0.5
    assert checkpoint.best_loss == 4.0
    assert checkpoint.best_iteration == 1


def test_joint_checkpoint_score_rejects_excess_invalid_intersections():
    m = load_module()

    accepted = m.joint_checkpoint_score(12.0, invalid_fraction=0.01, max_invalid_fraction=0.05)
    rejected = m.joint_checkpoint_score(0.0, invalid_fraction=0.06, max_invalid_fraction=0.05)

    assert accepted == 12.0
    assert rejected == float("inf")


def test_joint_layer_chi2_counts_only_valid_components():
    m = load_module()
    residuals = torch.tensor([[[1.0, 2.0], [3.0, 999.0], [4.0, 5.0]]])
    covariance = torch.tensor([[[1.0, 4.0], [9.0, 1.0], [4.0, 25.0]]])
    component_mask = torch.tensor([[[True, True], [True, False], [True, True]]])
    volume = torch.tensor([[17, 17, 24]])
    layer = torch.tensor([[8, 8, 4]])

    metrics = m.summarize_joint_layer_chi2(
        residuals,
        covariance,
        component_mask,
        volume,
        layer,
        ((17, 8), (24, 4)),
    )

    assert metrics["17:8"]["LOC0"] == {
        "chi2_per_component": pytest.approx(1.0),
        "n_components": 2,
    }
    assert metrics["17:8"]["LOC1"] == {
        "chi2_per_component": pytest.approx(1.0),
        "n_components": 1,
    }
    assert metrics["24:4"]["LOC0"]["chi2_per_component"] == pytest.approx(4.0)


def test_joint_checkpoint_patience_counts_from_initial_baseline():
    m = load_module()
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    checkpoint = m.JointBestCheckpoint((parameter,), patience=2)

    assert checkpoint.observe(3.0, iteration=-1) is False
    assert checkpoint.observe(4.0, iteration=0) is False
    assert checkpoint.observe(5.0, iteration=1) is True


def test_clamp_resets_adam_momentum_only_for_saturated_entries():
    m = load_module()
    parameter = torch.nn.Parameter(torch.tensor([-2.0, 0.2, 3.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["exp_avg"].fill_(4.0)
    optimizer.state[parameter]["exp_avg_sq"].fill_(9.0)

    saturated = m.clamp_parameter_and_reset_adam(parameter, -1.0, 1.0, optimizer)

    assert saturated.tolist() == [True, False, True]
    assert torch.allclose(parameter.detach(), torch.tensor([-1.0, 0.1, 1.0]))
    assert torch.equal(optimizer.state[parameter]["exp_avg"], torch.tensor([0.0, 4.0, 0.0]))
    assert torch.equal(optimizer.state[parameter]["exp_avg_sq"], torch.tensor([0.0, 9.0, 0.0]))


def test_surface_residual_loss_gives_invalid_loc1_zero_weight():
    m = load_module()
    residuals = torch.tensor([[2.0, 1.0e6], [4.0, 3.0]], dtype=torch.float64)
    covariances = torch.ones_like(residuals)
    hit_weights = torch.ones((2, 1), dtype=torch.float64)
    component_mask = torch.tensor([[True, False], [True, True]])

    loss = m.physics_informed_residual_loss(
        residuals,
        covariances,
        hit_weights,
        component_mask=component_mask,
    )

    assert torch.allclose(loss, torch.tensor((4.0 + 16.0 + 9.0) / 3.0, dtype=torch.float64))


def _surface(vol, lay, sen, x):
    return {
        "ID": f"vol={vol}|lay={lay}|sen={sen}",
        "Translation": [x, 0.0, 0.0],
        "RotationMatrix": torch.eye(3).tolist(),
        "FullMatrix": torch.eye(4).flatten().tolist(),
    }


def test_joint_geometry_overlays_targets_and_exports_only_input_surfaces(tmp_path):
    m = load_module()
    nominal = [_surface(10, 1, 1, 1.0), _surface(17, 8, 2, 2.0)]
    misaligned = [_surface(17, 8, 2, 2.5)]
    nominal_path = tmp_path / "nominal.json"
    misaligned_path = tmp_path / "misaligned.json"
    output_path = tmp_path / "aligned.json"
    nominal_path.write_text(json.dumps(nominal), encoding="utf-8")
    misaligned_path.write_text(json.dumps(misaligned), encoding="utf-8")

    geometry = m.load_joint_geometry(nominal_path, misaligned_path, ((17, 8),))

    assert geometry.centers.shape == (2, 3)
    assert geometry.centers[0, 0].item() == 1.0
    assert geometry.centers[1, 0].item() == 2.5
    assert geometry.target_mask.tolist() == [False, True]

    correction = torch.zeros((2, 6), dtype=torch.float64)
    correction[1, 0] = -0.5
    m.export_joint_geometry(geometry, correction.numpy(), output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(exported) == 1
    assert exported[0]["ID"] == "vol=17|lay=8|sen=2"
    assert abs(exported[0]["Translation"][0] - 2.0) < 1e-12


def test_seed_helices_fits_smoothed_points():
    m = load_module()
    truth = torch.tensor([[2.0, -1.0, 0.5, 0.4, 0.2, 0.002]], dtype=torch.float64)
    s = torch.linspace(0.0, 70.0, 10, dtype=torch.float64)
    points = m.helix_point(truth.expand(10, -1), s).unsqueeze(0)
    batches = m.TrackBatches(
        measured_local=torch.zeros((1, 10, 2), dtype=torch.float64),
        covariances=torch.ones((1, 10, 2), dtype=torch.float64),
        sensor_idx=torch.zeros((1, 10), dtype=torch.long),
        hit_mask=torch.ones((1, 10), dtype=torch.bool),
        loc1_mask=torch.ones((1, 10), dtype=torch.bool),
        g_smt=points,
        p_smt=m.helix_tangent(truth.expand(10, -1), s).unsqueeze(0),
        volume_id=torch.full((1, 10), 10, dtype=torch.long),
        layer_id=torch.full((1, 10), 2, dtype=torch.long),
        is_target=torch.zeros((1, 10), dtype=torch.bool),
        event_track_ids=torch.tensor([[1, 1]], dtype=torch.long),
        fit_mask=torch.ones((1, 10), dtype=torch.bool),
    )

    seed, diagnostics = m.seed_helices(
        batches, n_iter=30, dtype=torch.float64, return_diagnostics=True
    )
    fitted, _points, valid = m.closest_helix_points(seed[:, None, :].expand(-1, 10, -1), points)

    assert valid.all()
    assert diagnostics["solver"] == "gauss-newton"
    assert diagnostics["final_objective"] < diagnostics["initial_objective"] * 1.0e-8
    assert torch.sqrt(torch.mean((fitted - points) ** 2)).item() < 1.0e-3


def test_joint_track_parameterization_scales_physical_curvature():
    m = load_module()
    seed = torch.tensor([[1.0, 2.0, 3.0, 0.4, 0.2, 4e-5]], dtype=torch.float64)

    linear, rho_scaled = m.split_joint_track_parameters(seed)
    rho_scaled = rho_scaled + 1e-3
    assembled = m.assemble_joint_track_parameters(linear, rho_scaled)

    assert torch.allclose(assembled[:, :5], seed[:, :5])
    assert abs((assembled[0, 5] - seed[0, 5]).item() - 1e-7) < 1e-14


def test_free_joint_fit_recovers_hidden_shift_tilt_and_masks_strip_component():
    m = load_module()
    dtype = torch.float64
    n_tracks, n_planes = 24, 6
    x_planes = torch.linspace(0.0, 40.0, n_planes, dtype=dtype)
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    rotations = rotation.expand(n_planes, -1, -1).clone()
    nominal_rotations = rotations.clone()
    nominal_centers = torch.stack((x_planes, torch.zeros_like(x_planes), torch.zeros_like(x_planes)), dim=1)
    centers = nominal_centers.clone()
    centers[2, 1] = 0.4
    centers[2, 2] = -0.2
    rotations[2] = torch.tensor(m.euler_to_rotation_matrix(0.012, -0.018, 0.0), dtype=dtype)
    target_mask = torch.tensor([False, False, True, False, False, False])
    geometry = m.JointGeometry(
        transforms=[_surface(17 if i == 2 else 10, 8 if i == 2 else i, i + 1, float(x)) for i, x in enumerate(x_planes)],
        input_transforms=[_surface(17, 8, 3, float(x_planes[2]))],
        centers=centers,
        rotations=rotations,
        target_mask=target_mask,
        id_to_idx={},
        input_indices=[2],
    )

    y0 = torch.linspace(-1.0, 1.0, n_tracks, dtype=dtype)
    z0 = torch.linspace(0.5, -0.5, n_tracks, dtype=dtype)
    psi = torch.linspace(-0.02, 0.02, n_tracks, dtype=dtype)
    tanl = torch.linspace(-0.03, 0.03, n_tracks, dtype=dtype)
    tracks = torch.stack((torch.zeros_like(y0), y0, z0, psi, tanl, torch.zeros_like(y0)), dim=1)
    expanded_tracks = tracks[:, None, :].expand(-1, n_planes, -1)
    expanded_centers = nominal_centers[None, :, :].expand(n_tracks, -1, -1)
    expanded_normals = nominal_rotations[:, :, 2][None, :, :].expand(n_tracks, -1, -1)
    _s, points, valid = m.intersect_plane(expanded_tracks, expanded_centers, expanded_normals)
    local = torch.matmul(
        nominal_rotations[None, :, :, :].transpose(-1, -2),
        (points - expanded_centers).unsqueeze(-1),
    ).squeeze(-1)[..., :2]
    assert valid.all()
    batches = m.TrackBatches(
        measured_local=local,
        covariances=torch.full((n_tracks, n_planes, 2), 0.01, dtype=dtype),
        sensor_idx=torch.arange(n_planes).expand(n_tracks, -1),
        hit_mask=torch.ones((n_tracks, n_planes), dtype=torch.bool),
        loc1_mask=torch.tensor([True, True, True, True, True, False]).expand(n_tracks, -1),
        g_smt=points,
        p_smt=m.helix_tangent(expanded_tracks, _s),
        volume_id=torch.tensor([10, 10, 17, 10, 10, 10]).expand(n_tracks, -1),
        layer_id=torch.tensor([0, 1, 8, 3, 4, 5]).expand(n_tracks, -1),
        is_target=target_mask.expand(n_tracks, -1),
        event_track_ids=torch.stack((torch.zeros(n_tracks, dtype=torch.long), torch.arange(n_tracks)), dim=1),
    )
    config = m.AlignmentRunConfig(
        track_mode="free",
        joint_iters=1000,
        joint_lr_trans=0.01,
        joint_lr_track=1e-4,
        lambda_l2=0.0,
        make_plot=False,
    )

    sys.modules.pop("alignment_experiments", None)
    result = m.run_joint_alignment(
        batches,
        geometry,
        config,
        active_dof=(0, 1, 3, 4),
        initial_track_params=tracks,
    )
    assert "alignment_experiments" in sys.modules

    correction = result["predicted_deltas"]
    assert result["best_total_loss"] <= result["last_total_loss"]
    assert result["best_total_loss"] <= result["initial_total_loss"]
    assert -1 <= result["best_iteration"] < config.joint_iters
    assert np.isfinite(correction[2, :]).all()
    assert np.linalg.norm(correction[2, (0, 1)]) > 1e-3
    assert np.linalg.norm(correction[2, (3, 4)]) > 1e-4
    assert not result["final_component_mask"][:, -1, 1].any()

    frozen = m.run_joint_alignment(
        batches, geometry,
        m.AlignmentRunConfig(track_mode="free", joint_iters=20, joint_lr_trans=0.01,
                             joint_lr_track=0.01, lambda_l2=0.0, make_plot=False,
                             freeze_tracks=True),
        active_dof=(0, 1, 3, 4), initial_track_params=tracks,
    )
    assert np.allclose(frozen["track_params"], tracks.numpy())
    assert torch.count_nonzero(torch.tensor(correction)[~target_mask]).item() == 0


def test_frozen_helix_fit_recovers_rz_only_injection():
    m = load_module()
    dtype = torch.float64
    n_tracks, n_planes = 16, 5
    x_planes = torch.linspace(0.0, 32.0, n_planes, dtype=dtype)
    nominal_rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype
    )
    nominal_rotations = nominal_rotation.expand(n_planes, -1, -1).clone()
    rotations = nominal_rotations.clone()
    injected_rz = 0.02
    rotations[2] = rotations[2] @ torch.tensor(
        m.euler_to_rotation_matrix(0.0, 0.0, injected_rz), dtype=dtype
    )
    centers = torch.stack((x_planes, torch.zeros_like(x_planes), torch.zeros_like(x_planes)), dim=1)
    target_mask = torch.tensor([False, False, True, False, False])
    geometry = m.JointGeometry(
        transforms=[_surface(17 if i == 2 else 10, 8 if i == 2 else i, i + 1, float(x))
                    for i, x in enumerate(x_planes)],
        input_transforms=[_surface(17, 8, 3, float(x_planes[2]))],
        centers=centers,
        rotations=rotations,
        target_mask=target_mask,
        id_to_idx={},
        input_indices=[2],
    )
    y0 = torch.linspace(-4.0, 4.0, n_tracks, dtype=dtype)
    z0 = torch.linspace(3.0, -3.0, n_tracks, dtype=dtype)
    tracks = torch.stack((
        torch.zeros_like(y0), y0, z0,
        torch.linspace(-0.02, 0.02, n_tracks, dtype=dtype),
        torch.linspace(-0.03, 0.03, n_tracks, dtype=dtype),
        torch.zeros_like(y0),
    ), dim=1)
    expanded_tracks = tracks[:, None, :].expand(-1, n_planes, -1)
    expanded_centers = centers[None, :, :].expand(n_tracks, -1, -1)
    expanded_normals = nominal_rotations[:, :, 2][None, :, :].expand(n_tracks, -1, -1)
    path, points, valid = m.intersect_plane(expanded_tracks, expanded_centers, expanded_normals)
    measured = torch.matmul(
        nominal_rotations[None].transpose(-1, -2),
        (points - expanded_centers).unsqueeze(-1),
    ).squeeze(-1)[..., :2]
    assert valid.all()
    batches = m.TrackBatches(
        measured_local=measured,
        covariances=torch.full((n_tracks, n_planes, 2), 0.01, dtype=dtype),
        sensor_idx=torch.arange(n_planes).expand(n_tracks, -1),
        hit_mask=torch.ones((n_tracks, n_planes), dtype=torch.bool),
        loc1_mask=torch.ones((n_tracks, n_planes), dtype=torch.bool),
        g_smt=points,
        p_smt=m.helix_tangent(expanded_tracks, path),
        volume_id=torch.tensor([10, 10, 17, 10, 10]).expand(n_tracks, -1),
        layer_id=torch.tensor([0, 1, 8, 3, 4]).expand(n_tracks, -1),
        is_target=target_mask.expand(n_tracks, -1),
        event_track_ids=torch.stack((torch.zeros(n_tracks, dtype=torch.long), torch.arange(n_tracks)), dim=1),
    )
    config = m.AlignmentRunConfig(
        track_mode="free", freeze_tracks=True, joint_iters=300,
        joint_lr_trans=0.02, rot_lr_scale=1.0, lambda_l2=0.0,
        joint_patience=0, make_plot=False,
    )

    result = m.run_joint_alignment(
        batches, geometry, config, active_dof=(5,), initial_track_params=tracks
    )

    correction = result["predicted_deltas"][2]
    assert abs(correction[5] + injected_rz) < 2e-3
    assert np.count_nonzero(correction[:5]) == 0


def test_rotation_decomposition_matches_rz_ry_rx_convention():
    m = load_module()
    expected = (0.012, -0.021, 0.033)
    matrix = m.euler_to_rotation_matrix(*expected)

    actual = m.rotation_matrix_to_euler(matrix)

    assert torch.allclose(torch.tensor(actual), torch.tensor(expected), atol=1e-10)


@pytest.mark.root_io
def test_root_loader_exposes_raw_joint_measurements_and_stable_order(tmp_path):
    m = load_module()
    summary_path = tmp_path / "summary.root"
    states_path = tmp_path / "states.root"
    with uproot.recreate(summary_path) as root_file:
        root_file["tracksummary"] = {
            "event_nr": np.array([1], dtype=np.int64),
            "track_nr": np.array([7], dtype=np.int64),
            "chi2Sum": np.array([2.0]),
            "NDF": np.array([1.0]),
        }
    state_values = {
        "volume_id": [17, 10, 10],
        "layer_id": [8, 2, 1],
        "module_id": [3, 2, 1],
        "l_x_hit": [0.3, 0.2, 0.1],
        "l_y_hit": [0.4, np.nan, 0.2],
        "eLOC0_smt": [0.0, 0.0, 0.0],
        "eLOC1_smt": [0.0, 0.0, 0.0],
        "g_x_smt": [30.0, 20.0, 10.0],
        "g_y_smt": [0.0, 0.0, 0.0],
        "g_z_smt": [0.0, 0.0, 0.0],
        "px_smt": [1.0, 1.0, 1.0],
        "py_smt": [0.0, 0.0, 0.0],
        "pz_smt": [0.0, 0.0, 0.0],
        "ePHI_smt": [0.0, 0.0, 0.0],
        "eTHETA_smt": [1.0, 1.0, 1.0],
        "eQOP_smt": [0.001, 0.001, 0.001],
    }
    with uproot.recreate(states_path) as root_file:
        tree = {name: ak.Array([values]) for name, values in state_values.items()}
        tree.update({
            "err_x_hit": ak.Array([[0.1, 0.1, 0.1]]),
            "err_y_hit": ak.Array([[0.2, np.nan, 0.2]]),
            "event_nr": np.array([1], dtype=np.int64),
            "track_nr": np.array([7], dtype=np.int64),
        })
        root_file["trackstates"] = tree

    original_uproot_open = m.uproot.open

    def reject_uproot(*_args, **_kwargs):
        raise AssertionError("PyROOT-capable loader unexpectedly used uproot")

    m.uproot.open = reject_uproot

    try:
        inputs = m.load_variable_information(
            summary_path,
            states_path,
            {(17, 8, 3): 2, (10, 2, 2): 1, (10, 1, 1): 0},
        )
    finally:
        m.uproot.open = original_uproot_open

    assert inputs.hit_sensor_ids.tolist() == [0, 1, 2]
    assert torch.allclose(inputs.measured_local[:, 1], torch.tensor([0.2, 0.0, 0.4]))
    assert inputs.loc1_valid.tolist() == [True, False, True]
    assert torch.allclose(inputs.raw_covariances[:, 1], torch.tensor([0.04, 1.0, 0.04]))
