from __future__ import annotations

import pandas as pd
import torch
import json
import awkward as ak
import numpy as np
import pytest
import uproot

from mlalign.config import DEFAULT_ACTIVE_DOF
from mlalign.conventions import parse_active_dof
from mlalign.helix.gauge import assemble_track_parameters, split_track_parameters
from mlalign.geometry_io import (
    JointGeometry,
    export_joint_geometry,
    load_joint_geometry,
)
from mlalign.helix.trajectory import (
    accept_newton_candidate,
    closest_helix_points,
    helix_point,
    helix_tangent,
    intersect_plane,
)
from mlalign.helix_init import (
    _measurement_refit_objective,
    fit_helices_to_measurements,
    seed_helices,
)
from mlalign.joint_residuals import summarize_joint_layer_chi2
from mlalign.root_inputs import (
    AlignmentInputs,
    STRIP_Y_FLOOR,
    load_variable_information,
)
from mlalign.rotations import euler_to_rotation_matrix, rotation_matrix_to_euler
from mlalign.track_batches import TrackBatches, build_track_batches




def test_helix_point_and_tangent_have_stable_straight_line_limit():
    track = torch.tensor([[1.0, 2.0, 3.0, 0.3, 0.4, 0.0]], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([7.0], dtype=torch.float64)

    point = helix_point(track, s)
    tangent = helix_tangent(track, s)
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
    track = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=torch.float64)
    s = torch.tensor([5.0 * torch.pi], dtype=torch.float64)

    point = helix_point(track, s)
    tangent = helix_tangent(track, s)

    assert torch.allclose(point, torch.tensor([[10.0, 10.0, 0.0]], dtype=torch.float64), atol=1e-7)
    assert torch.allclose(tangent, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-7)


def test_intersect_plane_matches_closed_form_and_masks_parallel_track():
    tracks = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    centers = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=torch.float64)
    normals = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)

    s, points, valid = intersect_plane(tracks, centers, normals)

    assert valid.tolist() == [True, False]
    assert torch.allclose(s[0], torch.tensor(4.0, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(points[0], centers[0], atol=1e-8)
    assert torch.isfinite(points).all()


def test_intersect_plane_keeps_odd_outer_radius_but_rejects_runaway_path():
    tracks = torch.zeros((2, 6), dtype=torch.float64)
    centers = torch.tensor([[1000.0, 0.0, 0.0], [1300.0, 0.0, 0.0]], dtype=torch.float64)
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)

    _s, _points, valid = intersect_plane(tracks, centers, normals)

    assert valid.tolist() == [True, False]


def test_newton_invalid_candidate_keeps_last_feasible_path_and_stays_invalid():
    previous = torch.tensor([12.0, 8.0])
    candidate = torch.tensor([13.0, float("inf")])
    valid = torch.tensor([True, True])
    candidate_ok = torch.tensor([True, False])

    path, updated_valid = accept_newton_candidate(
        previous, candidate, valid, candidate_ok
    )

    assert torch.equal(path, torch.tensor([13.0, 8.0]))
    assert updated_valid.tolist() == [True, False]


def _make_inputs(frame: pd.DataFrame):
    n = len(frame)
    return AlignmentInputs(
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
    a = build_track_batches(
        _make_inputs(frame), ((17, 8),), min_hits=2, min_span=0.0
    )
    b = build_track_batches(
        _make_inputs(frame.sample(frac=1.0, random_state=7).reset_index(drop=True)),
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
    inputs = _make_inputs(frame)
    inputs.covariances = inputs.raw_covariances.clone()
    floored = torch.tensor(frame.volume_id.to_numpy() == 24)
    inputs.covariances[floored, 1] = STRIP_Y_FLOOR

    batches = build_track_batches(inputs, ((24, 4),), min_hits=1, min_span=0.0)

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
    inputs = _make_inputs(frame)
    inputs.raw_covariances = None

    batches = build_track_batches(inputs, ((17, 8),), min_hits=1, min_span=0.0)

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
    batches = build_track_batches(
        _make_inputs(_initialization_frame()),
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
        build_track_batches(
            _make_inputs(short), ((17, 8),), min_hits=8, min_span=300.0
        )


def test_smt_seed_ignores_target_layer_points():
    original = build_track_batches(
        _make_inputs(_initialization_frame()),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    shifted_frame = _initialization_frame()
    shifted_frame.loc[shifted_frame.volume_id == 17, ["g_y_smt", "g_z_smt"]] = [
        -5000.0,
        9000.0,
    ]
    shifted = build_track_batches(
        _make_inputs(shifted_frame),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )

    seed_a = seed_helices(original, n_iter=5, dtype=torch.float64)
    seed_b = seed_helices(shifted, n_iter=5, dtype=torch.float64)

    torch.testing.assert_close(seed_a, seed_b, rtol=0.0, atol=0.0)


def test_joint_initializers_fail_closed_without_non_target_fit_mask():
    batches = build_track_batches(
        _make_inputs(_initialization_frame()),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    batches.fit_mask = None

    with pytest.raises(ValueError, match="fit_mask"):
        seed_helices(batches, n_iter=1, dtype=torch.float64)
    with pytest.raises(ValueError, match="fit_mask"):
        _measurement_refit_objective(
            torch.zeros((1, 6), dtype=torch.float64),
            batches,
            None,
        )


def test_measurement_refit_ignores_target_measurements_and_reports_chi2():
    batches_a = build_track_batches(
        _make_inputs(_initialization_frame(target_measurement=0.0)),
        ((17, 8),),
        min_hits=8,
        min_span=300.0,
    )
    batches_b = build_track_batches(
        _make_inputs(_initialization_frame(target_measurement=5000.0)),
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
    geometry = JointGeometry(
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

    fitted_a, stats_a = fit_helices_to_measurements(
        batches_a, geometry, initial, dtype=torch.float64
    )
    fitted_b, stats_b = fit_helices_to_measurements(
        batches_b, geometry, initial, dtype=torch.float64
    )

    torch.testing.assert_close(fitted_a, fitted_b, rtol=0.0, atol=0.0)
    assert stats_a["chi2_per_component"] == pytest.approx(0.0)
    assert stats_a["initial_chi2_per_component"] == pytest.approx(0.0)
    assert stats_a["parameter_updates"]["position_rms"] == pytest.approx(0.0)
    assert stats_a["parameter_updates"]["direction_rms"] == pytest.approx(0.0)
    assert stats_a["parameter_updates"]["rho_rms"] == pytest.approx(0.0)
    assert stats_a["invalid_fraction"] == pytest.approx(0.0)
    assert stats_b == stats_a


def test_default_dof_includes_rz():
    """Canary: rz is the only rotation the current injection actually produces.

    Measured over misalignment_applied vs odd_transforms (186 surfaces):
    tx/ty ~0.52 mm, rz ~0.021 rad, and tz/rx/ry exactly zero -- alignment_odd.py
    hardcodes rot_local_x/y to 0 despite its rx/ry flags being set.
    """

    assert 5 in DEFAULT_ACTIVE_DOF


def test_empty_active_dof_string_does_not_bypass_mode_default():
    """parse_active_dof('') used to hand free mode the surface default silently."""

    resolved = parse_active_dof("")

    assert resolved == DEFAULT_ACTIVE_DOF


def test_joint_layer_chi2_counts_only_valid_components():
    residuals = torch.tensor([[[1.0, 2.0], [3.0, 999.0], [4.0, 5.0]]])
    covariance = torch.tensor([[[1.0, 4.0], [9.0, 1.0], [4.0, 25.0]]])
    component_mask = torch.tensor([[[True, True], [True, False], [True, True]]])
    volume = torch.tensor([[17, 17, 24]])
    layer = torch.tensor([[8, 8, 4]])

    metrics = summarize_joint_layer_chi2(
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


def _surface(vol, lay, sen, x):
    return {
        "ID": f"vol={vol}|lay={lay}|sen={sen}",
        "Translation": [x, 0.0, 0.0],
        "RotationMatrix": torch.eye(3).tolist(),
        "FullMatrix": torch.eye(4).flatten().tolist(),
    }


def test_joint_geometry_overlays_targets_and_exports_only_input_surfaces(tmp_path):
    nominal = [_surface(10, 1, 1, 1.0), _surface(17, 8, 2, 2.0)]
    misaligned = [_surface(17, 8, 2, 2.5)]
    nominal_path = tmp_path / "nominal.json"
    misaligned_path = tmp_path / "misaligned.json"
    output_path = tmp_path / "aligned.json"
    nominal_path.write_text(json.dumps(nominal), encoding="utf-8")
    misaligned_path.write_text(json.dumps(misaligned), encoding="utf-8")

    geometry = load_joint_geometry(nominal_path, misaligned_path, ((17, 8),))

    assert geometry.centers.shape == (2, 3)
    assert geometry.centers[0, 0].item() == 1.0
    assert geometry.centers[1, 0].item() == 2.5
    assert geometry.target_mask.tolist() == [False, True]

    correction = torch.zeros((2, 6), dtype=torch.float64)
    correction[1, 0] = -0.5
    export_joint_geometry(geometry, correction.numpy(), output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(exported) == 1
    assert exported[0]["ID"] == "vol=17|lay=8|sen=2"
    assert abs(exported[0]["Translation"][0] - 2.0) < 1e-12


def test_seed_helices_fits_smoothed_points():
    truth = torch.tensor([[2.0, -1.0, 0.5, 0.4, 0.2, 0.002]], dtype=torch.float64)
    s = torch.linspace(0.0, 70.0, 10, dtype=torch.float64)
    points = helix_point(truth.expand(10, -1), s).unsqueeze(0)
    batches = TrackBatches(
        measured_local=torch.zeros((1, 10, 2), dtype=torch.float64),
        covariances=torch.ones((1, 10, 2), dtype=torch.float64),
        sensor_idx=torch.zeros((1, 10), dtype=torch.long),
        hit_mask=torch.ones((1, 10), dtype=torch.bool),
        loc1_mask=torch.ones((1, 10), dtype=torch.bool),
        g_smt=points,
        p_smt=helix_tangent(truth.expand(10, -1), s).unsqueeze(0),
        volume_id=torch.full((1, 10), 10, dtype=torch.long),
        layer_id=torch.full((1, 10), 2, dtype=torch.long),
        is_target=torch.zeros((1, 10), dtype=torch.bool),
        event_track_ids=torch.tensor([[1, 1]], dtype=torch.long),
        fit_mask=torch.ones((1, 10), dtype=torch.bool),
    )

    seed, diagnostics = seed_helices(
        batches, n_iter=30, dtype=torch.float64, return_diagnostics=True
    )
    fitted, _points, valid = closest_helix_points(seed[:, None, :].expand(-1, 10, -1), points)

    assert valid.all()
    assert diagnostics["solver"] == "gauss-newton"
    assert diagnostics["final_objective"] < diagnostics["initial_objective"] * 1.0e-8
    assert torch.sqrt(torch.mean((fitted - points) ** 2)).item() < 1.0e-3


def test_joint_track_parameterization_scales_physical_curvature():
    seed = torch.tensor([[1.0, 2.0, 3.0, 0.4, 0.2, 4e-5]], dtype=torch.float64)

    linear, rho_scaled = split_track_parameters(seed)
    rho_scaled = rho_scaled + 1e-3
    assembled = assemble_track_parameters(linear, rho_scaled)

    assert torch.allclose(assembled[:, :5], seed[:, :5])
    assert abs((assembled[0, 5] - seed[0, 5]).item() - 1e-7) < 1e-14


def test_rotation_decomposition_matches_rz_ry_rx_convention():
    expected = (0.012, -0.021, 0.033)
    matrix = euler_to_rotation_matrix(*expected)

    actual = rotation_matrix_to_euler(matrix)

    assert torch.allclose(torch.tensor(actual), torch.tensor(expected), atol=1e-10)


@pytest.mark.root_io
def test_root_loader_exposes_raw_joint_measurements_and_stable_order(tmp_path):
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

    original_uproot_open = uproot.open

    def reject_uproot(*_args, **_kwargs):
        raise AssertionError("PyROOT-capable loader unexpectedly used uproot")

    uproot.open = reject_uproot

    try:
        inputs = load_variable_information(
            summary_path,
            states_path,
            {(17, 8, 3): 2, (10, 2, 2): 1, (10, 1, 1): 0},
        )
    finally:
        uproot.open = original_uproot_open

    assert inputs.hit_sensor_ids.tolist() == [0, 1, 2]
    assert torch.allclose(inputs.measured_local[:, 1], torch.tensor([0.2, 0.0, 0.4]))
    assert inputs.loc1_valid.tolist() == [True, False, True]
    assert torch.allclose(inputs.raw_covariances[:, 1], torch.tensor([0.04, 1.0, 0.04]))
