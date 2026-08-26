from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pytest
import torch


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_helix_model.py"
    assert module_path.exists(), "validate_helix_model.py has not been implemented"
    spec = importlib.util.spec_from_file_location("validate_helix_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validation_module_does_not_import_plotting_eagerly(monkeypatch):
    real_import = builtins.__import__

    def reject_plotting_import(name, *args, **kwargs):
        if name == "helix_validation_plots":
            raise ImportError("synthetic matplotlib import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_plotting_import)

    module = load_module()

    assert callable(module.run_helix_validation)


def test_validation_cli_paths_are_anchored_to_repository(monkeypatch, tmp_path):
    module = load_module()
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    args = module.build_argument_parser().parse_args(
        [
            "--states",
            "inputs/states.root",
            "--geometry",
            "inputs/current.json",
            "--output-dir",
            "outputs/validation",
        ]
    )

    assert args.states == str(repository_root / "inputs" / "states.root")
    assert args.geometry == str(repository_root / "inputs" / "current.json")
    assert args.output_dir == str(repository_root / "outputs" / "validation")


def load_plot_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "helix_validation_plots.py"
    assert module_path.exists(), "helix_validation_plots.py has not been implemented"
    spec = importlib.util.spec_from_file_location("helix_validation_plots", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _comparison_frame():
    return pd.DataFrame(
        {
            "split": ["fit", "fit", "holdout", "holdout"],
            "prediction_mode": [
                "interpolation",
                "interpolation",
                "interpolation",
                "forward_extrapolation",
            ],
            "volume_id": [10, 10, 17, 17],
            "layer_id": [2, 2, 8, 8],
            "fitted_intersection_valid": [True, False, True, True],
            "fitted_dLOC0": [1.0, 999.0, -2.0, np.nan],
            "fitted_dLOC1": [2.0, 999.0, 3.0, 4.0],
            "fitted_measurement_residual_LOC0": [0.5, 999.0, -1.5, np.nan],
            "fitted_measurement_residual_LOC1": [1.0, 999.0, 2.5, 3.5],
            "fitted_dg_x": [3.0, 999.0, 0.0, 5.0],
            "fitted_dg_y": [4.0, 999.0, 0.0, 12.0],
            "fitted_dg_z": [0.0, 999.0, 2.0, 0.0],
        }
    )


def test_fitted_residual_components_filter_invalid_values_and_compute_norm():
    plots = load_plot_module()

    components = plots.fitted_residual_components(_comparison_frame())

    assert components["local0"].tolist() == [0.5, -1.5]
    assert components["local1"].tolist() == [1.0, 2.5, 3.5]
    assert components["global_norm"].tolist() == [5.0, 2.0, 13.0]


def test_write_residual_plots_creates_fit_holdout_and_layer_pngs(tmp_path):
    plots = load_plot_module()

    paths = plots.write_residual_plots(
        _comparison_frame(), tmp_path, holdout_layers=((17, 8),)
    )

    assert {path.name for path in paths} == {
        "fit_local_residuals.png",
        "fit_global_residuals.png",
        "holdout_overall_local_residuals.png",
        "holdout_overall_global_residuals.png",
        "holdout_vol17_lay8_local_residuals.png",
        "holdout_vol17_lay8_global_residuals.png",
        "holdout_interpolation_local_residuals.png",
        "holdout_interpolation_global_residuals.png",
        "holdout_forward_extrapolation_local_residuals.png",
        "holdout_forward_extrapolation_global_residuals.png",
        "holdout_backward_extrapolation_local_residuals.png",
        "holdout_backward_extrapolation_global_residuals.png",
        "holdout_vol17_lay8_interpolation_local_residuals.png",
        "holdout_vol17_lay8_interpolation_global_residuals.png",
        "holdout_vol17_lay8_forward_extrapolation_local_residuals.png",
        "holdout_vol17_lay8_forward_extrapolation_global_residuals.png",
        "holdout_vol17_lay8_backward_extrapolation_local_residuals.png",
        "holdout_vol17_lay8_backward_extrapolation_global_residuals.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_write_residual_plots_creates_placeholder_for_no_valid_intersections(tmp_path):
    plots = load_plot_module()
    frame = _comparison_frame()
    frame["fitted_intersection_valid"] = False

    paths = plots.write_residual_plots(
        frame, tmp_path, holdout_layers=((17, 8),)
    )

    assert len(paths) == 18
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_empty_histogram_message_describes_missing_finite_residuals():
    plots = load_plot_module()

    class FakeAxis:
        def __init__(self):
            self.messages = []
            self.transAxes = object()

        def text(self, _x, _y, message, **_kwargs):
            self.messages.append(message)

        def set_title(self, _value):
            pass

        def set_xlabel(self, _value):
            pass

        def set_ylabel(self, _value):
            pass

        def grid(self, **_kwargs):
            pass

    axis = FakeAxis()

    plots._draw_histogram(axis, np.array([]), "title", "xlabel")

    assert axis.messages[0] == "No finite fitted residuals"


@pytest.mark.parametrize(
    ("writer_name", "axes"),
    [
        ("_write_local_plot", [object(), object()]),
        ("_write_global_plot", np.array([[object(), object()], [object(), object()]])),
    ],
)
def test_plot_closes_figure_when_save_fails(
    monkeypatch, tmp_path, writer_name, axes
):
    plots = load_plot_module()
    closed = []

    class FailingFigure:
        def suptitle(self, _title):
            pass

        def savefig(self, _path, **_kwargs):
            raise OSError("synthetic write failure")

    figure = FailingFigure()
    monkeypatch.setattr(plots.plt, "subplots", lambda *_args, **_kwargs: (figure, axes))
    monkeypatch.setattr(plots, "_draw_histogram", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plots.plt, "close", lambda value: closed.append(value))

    with pytest.raises(OSError, match="synthetic write failure"):
        getattr(plots, writer_name)(
            _comparison_frame(), tmp_path / "plot.png", "scope"
        )

    assert closed == [figure]


def test_helix_point_and_tangent_match_straight_line_limit():
    m = load_module()
    track = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.5, 0.0]], dtype=torch.float64
    )
    path = torch.tensor([4.0], dtype=torch.float64)

    assert torch.allclose(
        m.helix_point(track, path),
        torch.tensor([[5.0, 2.0, 5.0]], dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        m.helix_tangent(track, path),
        torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64),
        atol=1e-12,
    )


def _straight_inputs(m, holdout_shift=0.0, holdout_measurement_shift=0.0):
    x = torch.linspace(0.0, 450.0, 10, dtype=torch.float64)
    points = torch.stack((x, torch.zeros_like(x), 0.1 * x), dim=1)
    holdout = torch.tensor([False, False, False, False, True, False, False, True, False, False])
    points[holdout, 1] += holdout_shift
    volume = torch.where(holdout, torch.tensor(17), torch.tensor(10)).long()
    layer = torch.where(holdout, torch.tensor(8), torch.arange(10)).long()
    measured = torch.zeros((10, 2), dtype=torch.float64)
    measured[holdout, 0] = holdout_measurement_shift
    return m.HelixInputs(
        hit_sensor_ids=torch.arange(10),
        measured_local=measured,
        measurement_variance=torch.ones((10, 2), dtype=torch.float64),
        loc1_valid=torch.ones(10, dtype=torch.bool),
        smt_local=torch.zeros((10, 2), dtype=torch.float64),
        hit_global_positions=points,
        hit_directions=torch.tensor([[1.0, 0.0, 0.1]], dtype=torch.float64).expand(10, -1).clone(),
        hit_volume_ids=volume,
        hit_layer_ids=layer,
        hit_module_ids=torch.arange(1, 11),
        hit_event_ids=torch.zeros(10, dtype=torch.long),
        hit_track_ids=torch.ones(10, dtype=torch.long),
    )


def test_holdout_layers_do_not_influence_initialization_or_fit():
    m = load_module()
    original = m.build_track_batches(
        _straight_inputs(m), ((17, 8),), min_fit_hits=8, min_fit_span=300.0
    )
    shifted = m.build_track_batches(
        _straight_inputs(m, holdout_shift=1000.0),
        ((17, 8),), min_fit_hits=8, min_fit_span=300.0
    )

    assert original.fit_mask.sum().item() == 16
    assert original.holdout_mask.sum().item() == 2
    assert torch.allclose(m.initialize_helices(original), m.initialize_helices(shifted))

    fit_original = m.seed_helices(original, n_iter=3, dtype=torch.float64)
    fit_shifted = m.seed_helices(shifted, n_iter=3, dtype=torch.float64)
    assert torch.allclose(fit_original, fit_shifted, atol=0.0, rtol=0.0)


def test_holdout_measurements_do_not_influence_measurement_refit():
    m = load_module()
    original = m.build_track_batches(
        _straight_inputs(m), ((17, 8),), min_fit_hits=8, min_fit_span=300.0
    )
    shifted = m.build_track_batches(
        _straight_inputs(m, holdout_measurement_shift=1000.0),
        ((17, 8),),
        min_fit_hits=8,
        min_fit_span=300.0,
    )
    x = torch.linspace(0.0, 450.0, 10, dtype=torch.float64)
    centers = torch.stack((x, torch.zeros_like(x), 0.1 * x), dim=1)
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    geometry = m.NominalGeometry(
        transforms=[],
        centers=centers,
        rotations=rotation.expand(10, -1, -1).clone(),
        id_to_idx={},
    )
    initial = torch.tensor(
        [[0.0, 0.5, 0.0, 0.01, 0.08, 0.0]], dtype=torch.float64
    )

    fit_original = m.fit_helices_to_measurements(
        original, geometry, initial, n_iter=20, dtype=torch.float64
    )
    fit_shifted = m.fit_helices_to_measurements(
        shifted, geometry, initial, n_iter=20, dtype=torch.float64
    )

    assert torch.allclose(fit_original, fit_shifted, atol=0.0, rtol=0.0)


def test_track_selection_enforces_fit_hit_count_and_span():
    m = load_module()

    with pytest.raises(ValueError, match="No tracks satisfy"):
        m.build_track_batches(
            _straight_inputs(m), ((17, 8),), min_fit_hits=9, min_fit_span=300.0
        )
    with pytest.raises(ValueError, match="No tracks satisfy"):
        m.build_track_batches(
            _straight_inputs(m), ((17, 8),), min_fit_hits=8, min_fit_span=451.0
        )


def test_track_selection_uses_nearest_hits_before_and_after_holdout():
    m = load_module()
    inputs = _straight_inputs(m)
    inputs.hit_path_length = torch.arange(10, dtype=torch.float64)
    inputs.hit_volume_ids[:] = 10
    inputs.hit_layer_ids[:] = torch.arange(10)
    inputs.hit_volume_ids[5] = 17
    inputs.hit_layer_ids[5] = 8

    batches = m.build_track_batches(
        inputs,
        ((17, 8),),
        min_fit_hits=4,
        min_fit_span=3.0,
        context_hits_before=2,
        context_hits_after=2,
    )

    interpolation_row = batches.prediction_mode.index("interpolation")
    selected_modules = batches.module_id[
        interpolation_row, batches.hit_mask[interpolation_row]
    ].tolist()
    assert selected_modules == [4, 5, 6, 7, 8]
    assert batches.fit_mask[interpolation_row].sum().item() == 4
    assert batches.holdout_mask[interpolation_row].sum().item() == 1


def test_track_selection_builds_interpolation_and_both_extrapolation_directions():
    m = load_module()
    inputs = _straight_inputs(m)
    extra = _straight_inputs(m)
    for name in (
        "hit_sensor_ids",
        "measured_local",
        "measurement_variance",
        "loc1_valid",
        "smt_local",
        "hit_global_positions",
        "hit_directions",
        "hit_volume_ids",
        "hit_layer_ids",
        "hit_module_ids",
        "hit_event_ids",
        "hit_track_ids",
    ):
        setattr(inputs, name, torch.cat((getattr(inputs, name), getattr(extra, name)[1:4])))
    inputs.hit_sensor_ids = torch.arange(13)
    inputs.hit_module_ids = torch.arange(1, 14)
    inputs.hit_global_positions[:, 0] = torch.arange(13, dtype=torch.float64)
    inputs.hit_global_positions[:, 2] = 0.1 * torch.arange(13, dtype=torch.float64)
    inputs.hit_volume_ids[:] = 10
    inputs.hit_layer_ids[:] = torch.arange(13)
    inputs.hit_volume_ids[6] = 17
    inputs.hit_layer_ids[6] = 8
    inputs.hit_path_length = torch.arange(13, dtype=torch.float64)

    batches = m.build_track_batches(
        inputs,
        ((17, 8),),
        min_fit_hits=4,
        min_fit_span=3.0,
        context_hits_before=2,
        context_hits_after=2,
    )

    assert batches.prediction_mode == (
        "interpolation",
        "forward_extrapolation",
        "backward_extrapolation",
    )
    assert batches.context_hits_before.tolist() == [2, 4, 0]
    assert batches.context_hits_after.tolist() == [2, 0, 4]
    assert batches.nearest_context_distance.tolist() == [1.0, 1.0, 1.0]
    assert batches.farthest_context_distance.tolist() == [2.0, 4.0, 4.0]
    for row in range(3):
        target_column = batches.holdout_mask[row].nonzero().item()
        assert batches.measured_local[row, target_column, 0].item() == 0.0
        assert batches.fit_mask[row, target_column].item() is False


def test_measurement_residual_uses_prediction_minus_measurement_and_masks_loc1():
    m = load_module()
    predicted = torch.tensor(
        [[[1.5, 4.0], [-2.0, 8.0]]], dtype=torch.float64
    )
    measured = torch.tensor(
        [[[1.0, 2.5], [-1.5, 999.0]]], dtype=torch.float64
    )
    variance = torch.tensor(
        [[[0.25, 2.25], [0.04, 1.0]]], dtype=torch.float64
    )
    loc1_valid = torch.tensor([[True, False]])

    residual, normalized = m.measurement_residuals(
        predicted, measured, variance, loc1_valid
    )

    assert torch.allclose(
        residual[0, 0], torch.tensor([0.5, 1.5], dtype=torch.float64)
    )
    assert residual[0, 1, 0].item() == pytest.approx(-0.5)
    assert torch.isnan(residual[0, 1, 1])
    assert torch.allclose(
        normalized[0, 0], torch.tensor([1.0, 1.0], dtype=torch.float64)
    )
    assert normalized[0, 1, 0].item() == pytest.approx(-2.5)
    assert torch.isnan(normalized[0, 1, 1])


def test_measurement_metrics_report_signed_accuracy_and_loc1_count():
    m = load_module()
    inputs = _straight_inputs(m)
    inputs.hit_path_length = torch.arange(10, dtype=torch.float64)
    inputs.hit_volume_ids[:] = 10
    inputs.hit_layer_ids[:] = torch.arange(10)
    inputs.hit_volume_ids[5] = 17
    inputs.hit_layer_ids[5] = 8
    batches = m.build_track_batches(
        inputs,
        ((17, 8),),
        min_fit_hits=4,
        min_fit_span=3.0,
        context_hits_before=2,
        context_hits_after=2,
    )
    row = batches.prediction_mode.index("interpolation")
    target = batches.holdout_mask[row].nonzero().item()
    batches.measured_local[row, target] = torch.tensor([1.0, 8.0])
    batches.measurement_variance[row, target] = torch.tensor([0.25, 4.0])
    batches.loc1_valid[row, target] = False
    shape = batches.hit_mask.shape
    predicted = batches.measured_local.clone()
    predicted[row, target] = torch.tensor([1.5, 999.0])
    evaluation = m.HelixEvaluation(
        path_length=torch.zeros(shape),
        global_position=torch.zeros((*shape, 3)),
        local_position=predicted,
        global_difference=torch.zeros((*shape, 3)),
        local_difference=torch.zeros((*shape, 2)),
        tangent_angle=torch.full(shape, 0.002),
        intersection_valid=batches.hit_mask.clone(),
    )

    metrics = m.summarize_measurement_evaluation(
        evaluation, batches, batches.holdout_mask & (batches.volume_id == 17)
    )

    assert metrics["local0_count"] == 3
    assert metrics["local0_bias_mm"] == pytest.approx(1.0 / 6.0)
    assert metrics["local0_rmse_mm"] == pytest.approx(np.sqrt(0.25 / 3.0))
    assert metrics["local0_normalized_rms"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["local1_count"] == 2
    assert metrics["direction_angle_rms_rad"] == pytest.approx(0.002)


def test_measurement_gate_checks_every_layer_and_prediction_mode():
    m = load_module()

    def comparison(loc0=0.01, loc1=0.02, angle=0.001, count=10):
        return {
            "requested_count": count,
            "common_valid_count": count,
            "fitted_intersection_failure_rate": 0.0,
            "fitted": {
                "local0_count": count,
                "local1_count": count,
                "local0_rmse_mm": loc0,
                "local1_rmse_mm": loc1,
                "direction_angle_rms_rad": angle,
            },
        }

    scopes = {
        "17:8": {mode: comparison() for mode in m.PREDICTION_MODES},
        "24:4": {mode: comparison(loc1=1.0) for mode in m.PREDICTION_MODES},
    }
    scopes["17:8"]["backward_extrapolation"] = comparison(loc0=0.08)

    passed, reasons = m._measurement_validation_pass(
        scopes,
        max_failure_rate=0.05,
        layer_local_limits={"17:8": (0.05, 0.05), "24:4": (0.10, 1.50)},
        max_direction_angle_rms=0.005,
    )

    assert passed is False
    assert reasons == [
        "holdout layer 17:8 backward_extrapolation LOC0 RMSE 0.08 mm exceeds 0.05 mm"
    ]


def test_geometry_contract_and_surface_intersection_use_the_same_nominal_plane():
    m = load_module()
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    geometry = m.NominalGeometry(
        transforms=[],
        centers=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64),
        rotations=rotation.unsqueeze(0),
        id_to_idx={(10, 2, 1): 0},
    )
    inputs = m.HelixInputs(
        hit_sensor_ids=torch.tensor([0]),
        measured_local=torch.tensor([[2.0, 3.0]], dtype=torch.float64),
        measurement_variance=torch.ones((1, 2), dtype=torch.float64),
        loc1_valid=torch.ones(1, dtype=torch.bool),
        smt_local=torch.tensor([[2.0, 3.0]], dtype=torch.float64),
        hit_global_positions=torch.tensor([[10.0, 2.0, 3.0]], dtype=torch.float64),
        hit_directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        hit_volume_ids=torch.tensor([10]),
        hit_layer_ids=torch.tensor([2]),
        hit_module_ids=torch.tensor([1]),
        hit_event_ids=torch.tensor([0]),
        hit_track_ids=torch.tensor([0]),
    )

    report = m.validate_geometry_consistency(inputs, geometry, tolerance=1e-3)
    assert report["max_local_error_mm"] == pytest.approx(0.0, abs=1e-12)
    assert report["max_plane_distance_mm"] == pytest.approx(0.0, abs=1e-12)

    track = torch.tensor([[0.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    path, point, valid = m.intersect_plane(
        track, geometry.centers, geometry.rotations[:, :, 2]
    )
    assert valid.tolist() == [True]
    assert path.item() == pytest.approx(10.0)
    assert torch.allclose(point, inputs.hit_global_positions)

    shifted = m.NominalGeometry(
        transforms=[],
        centers=torch.tensor([[10.0, 0.1, 0.0]], dtype=torch.float64),
        rotations=geometry.rotations,
        id_to_idx=geometry.id_to_idx,
    )
    with pytest.raises(ValueError, match="geometry is inconsistent"):
        m.validate_geometry_consistency(inputs, shifted, tolerance=1e-3)


def test_working_geometry_overlays_partial_current_geometry(tmp_path):
    m = load_module()
    nominal = [
        {
            "ID": "vol=10|lay=2|sen=1",
            "Translation": [1.0, 0.0, 0.0],
            "RotationMatrix": np.eye(3).tolist(),
        },
        {
            "ID": "vol=17|lay=8|sen=2",
            "Translation": [2.0, 0.0, 0.0],
            "RotationMatrix": np.eye(3).tolist(),
        },
    ]
    current = [
        {
            "ID": "vol=17|lay=8|sen=2",
            "Translation": [2.5, 0.0, 0.0],
            "RotationMatrix": np.eye(3).tolist(),
        }
    ]
    nominal_path = tmp_path / "nominal.json"
    current_path = tmp_path / "current.json"
    nominal_path.write_text(json.dumps(nominal))
    current_path.write_text(json.dumps(current))

    geometry = m.load_working_geometry(current_path, nominal_path)

    assert len(geometry.transforms) == 2
    assert torch.equal(
        geometry.centers,
        torch.tensor([[1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64),
    )


def test_measured_state_missing_from_geometry_is_rejected():
    m = load_module()
    flat = {
        "volume_id": np.array([10]),
        "layer_id": np.array([2]),
        "module_id": np.array([99]),
        "l_x_hit": np.array([0.0]),
        "eLOC0_smt": np.array([0.0]),
        "eLOC1_smt": np.array([0.0]),
        "g_x_smt": np.array([1.0]),
        "g_y_smt": np.array([2.0]),
        "g_z_smt": np.array([3.0]),
        "px_smt": np.array([1.0]),
        "py_smt": np.array([0.0]),
        "pz_smt": np.array([0.0]),
        "event_nr": np.array([1]),
        "track_nr": np.array([7]),
    }

    with pytest.raises(ValueError, match="absent from nominal geometry"):
        m._helix_inputs_from_flat(flat, {(10, 2, 1): 0})


def test_flat_loader_exposes_measurements_variances_and_component_mask():
    m = load_module()
    flat = {
        "volume_id": np.array([10, 10]),
        "layer_id": np.array([2, 3]),
        "module_id": np.array([1, 2]),
        "l_x_hit": np.array([0.25, -0.50]),
        "l_y_hit": np.array([0.75, np.nan]),
        "err_x_hit": np.array([0.10, 0.20]),
        "err_y_hit": np.array([0.30, np.nan]),
        "eLOC0_smt": np.array([0.20, -0.45]),
        "eLOC1_smt": np.array([0.70, 0.00]),
        "g_x_smt": np.array([1.0, 2.0]),
        "g_y_smt": np.array([2.0, 3.0]),
        "g_z_smt": np.array([3.0, 4.0]),
        "px_smt": np.array([1.0, 1.0]),
        "py_smt": np.array([0.0, 0.0]),
        "pz_smt": np.array([0.0, 0.0]),
        "event_nr": np.array([1, 1]),
        "track_nr": np.array([7, 7]),
    }

    inputs = m._helix_inputs_from_flat(
        flat,
        {(10, 2, 1): 0, (10, 3, 2): 1},
    )

    assert torch.equal(
        inputs.measured_local,
        torch.tensor([[0.25, 0.75], [-0.50, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        inputs.measurement_variance,
        torch.tensor([[0.01, 0.09], [0.04, 1.0]], dtype=torch.float64),
    )
    assert inputs.loc1_valid.tolist() == [True, False]


def test_measurement_error_expansion_aligns_compact_arrays_to_states():
    m = load_module()
    l_x = ak.Array([[0.1, None, 0.2], [None, -0.3]])
    err_x = ak.Array([[0.01, 0.02], [0.03]])
    err_y = ak.Array([[0.04, 0.05], [0.06]])

    expanded_x, expanded_y = m.expand_measurement_errors(l_x, err_x, err_y)

    assert np.allclose(
        expanded_x,
        np.array([0.01, np.nan, 0.02, np.nan, 0.03]),
        equal_nan=True,
    )
    assert np.allclose(
        expanded_y,
        np.array([0.04, np.nan, 0.05, np.nan, 0.06]),
        equal_nan=True,
    )


def test_fitted_helix_improves_all_holdout_metrics_on_curved_track():
    m = load_module()
    dtype = torch.float64
    path = torch.linspace(0.0, 500.0, 10, dtype=dtype)
    rho = 0.002
    points = torch.stack(
        (
            torch.sin(rho * path) / rho,
            (1.0 - torch.cos(rho * path)) / rho,
            0.1 * path,
        ),
        dim=1,
    )
    tangents = torch.stack(
        (torch.cos(rho * path), torch.sin(rho * path), torch.full_like(path, 0.1)),
        dim=1,
    )
    normals = torch.nn.functional.normalize(tangents, dim=1)
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand_as(normals)
    local0 = torch.nn.functional.normalize(torch.cross(reference, normals, dim=1), dim=1)
    local1 = torch.cross(normals, local0, dim=1)
    rotations = torch.stack((local0, local1, normals), dim=2)
    holdout = torch.tensor([False, False, False, False, True, False, False, True, False, False])
    volume = torch.where(holdout, torch.tensor(17), torch.tensor(10)).long()
    layer = torch.where(holdout, torch.tensor(8), torch.arange(10)).long()
    geometry = m.NominalGeometry(
        transforms=[],
        centers=points,
        rotations=rotations,
        id_to_idx={(int(volume[i]), int(layer[i]), i + 1): i for i in range(10)},
    )
    inputs = m.HelixInputs(
        hit_sensor_ids=torch.arange(10),
        measured_local=torch.zeros((10, 2), dtype=dtype),
        measurement_variance=torch.ones((10, 2), dtype=dtype),
        loc1_valid=torch.ones(10, dtype=torch.bool),
        smt_local=torch.zeros((10, 2), dtype=dtype),
        hit_global_positions=points,
        hit_directions=tangents,
        hit_volume_ids=volume,
        hit_layer_ids=layer,
        hit_module_ids=torch.arange(1, 11),
        hit_event_ids=torch.zeros(10, dtype=torch.long),
        hit_track_ids=torch.ones(10, dtype=torch.long),
    )
    batches = m.build_track_batches(
        inputs, ((17, 8),), min_fit_hits=8, min_fit_span=300.0
    )
    initial = m.initialize_helices(batches)
    fitted = m.seed_helices(batches, n_iter=250, dtype=dtype)

    initial_metrics = m.summarize_evaluation(
        m.evaluate_helix_on_surfaces(initial, batches, geometry),
        batches.holdout_mask,
    )
    fitted_metrics = m.summarize_evaluation(
        m.evaluate_helix_on_surfaces(fitted, batches, geometry),
        batches.holdout_mask,
    )

    assert fitted_metrics["global_rmse_mm"] < initial_metrics["global_rmse_mm"]
    assert fitted_metrics["local_rmse_mm"] < initial_metrics["local_rmse_mm"]
    assert fitted_metrics["direction_angle_rms_rad"] < initial_metrics["direction_angle_rms_rad"]
    assert fitted_metrics["intersection_failure_rate"] == 0.0


def test_zero_common_intersections_become_validation_failure_not_exception():
    m = load_module()
    evaluation = m.HelixEvaluation(
        path_length=torch.zeros((1, 1)),
        global_position=torch.zeros((1, 1, 3)),
        local_position=torch.zeros((1, 1, 2)),
        global_difference=torch.zeros((1, 1, 3)),
        local_difference=torch.zeros((1, 1, 2)),
        tangent_angle=torch.zeros((1, 1)),
        intersection_valid=torch.zeros((1, 1), dtype=torch.bool),
    )

    comparison = m._comparison_metrics(
        evaluation, evaluation, torch.ones((1, 1), dtype=torch.bool)
    )
    passed, reasons = m._validation_pass(
        comparison, {"17:8": comparison}, max_failure_rate=0.05
    )

    assert comparison["common_valid_count"] == 0
    assert comparison["fitted"]["global_rmse_mm"] is None
    assert passed is False
    assert any("no common valid intersection" in reason for reason in reasons)


def test_validation_rejects_holdout_residual_above_layer_limit():
    m = load_module()
    metrics = {
        "requested_count": 10,
        "common_valid_count": 10,
        "initial_intersection_failure_rate": 0.0,
        "fitted_intersection_failure_rate": 0.0,
        "initial": {
            "global_rmse_mm": 1.0,
            "local_rmse_mm": 1.0,
            "local0_rmse_mm": 1.0,
            "local1_rmse_mm": 1.0,
            "direction_angle_rms_rad": 0.01,
        },
        "fitted": {
            "global_rmse_mm": 0.2,
            "local_rmse_mm": 0.2,
            "local0_rmse_mm": 0.2,
            "local1_rmse_mm": 0.01,
            "direction_angle_rms_rad": 0.001,
        },
    }

    passed, reasons = m._validation_pass(
        metrics,
        {"17:8": metrics},
        max_failure_rate=0.05,
        layer_local_limits={"17:8": (0.05, 0.05)},
        max_direction_angle_rms=0.005,
    )

    assert passed is False
    assert any("LOC0 RMSE" in reason for reason in reasons)


def test_cli_runs_holdout_validation_and_writes_outputs(tmp_path, monkeypatch):
    m = load_module()
    path = np.linspace(0.0, 500.0, 10)
    rho = 0.002
    points = np.column_stack(
        (
            np.sin(rho * path) / rho,
            (1.0 - np.cos(rho * path)) / rho,
            0.1 * path,
        )
    )
    tangents = np.column_stack(
        (np.cos(rho * path), np.sin(rho * path), np.full_like(path, 0.1))
    )
    normals = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    reference = np.repeat([[0.0, 0.0, 1.0]], 10, axis=0)
    local0 = np.cross(reference, normals)
    local0 /= np.linalg.norm(local0, axis=1, keepdims=True)
    local1 = np.cross(normals, local0)
    rotations = np.stack((local0, local1, normals), axis=2)
    holdout = np.array([False, False, False, False, True, False, False, True, False, False])
    volume = np.where(holdout, 17, 10)
    layer = np.where(holdout, 8, np.arange(10))

    geometry = m.NominalGeometry(
        transforms=[],
        centers=torch.tensor(points, dtype=torch.float64),
        rotations=torch.tensor(rotations, dtype=torch.float64),
        id_to_idx={(int(volume[i]), int(layer[i]), i + 1): i for i in range(10)},
    )
    inputs = m.HelixInputs(
        hit_sensor_ids=torch.arange(10),
        measured_local=torch.zeros((10, 2), dtype=torch.float64),
        measurement_variance=torch.ones((10, 2), dtype=torch.float64),
        loc1_valid=torch.ones(10, dtype=torch.bool),
        smt_local=torch.zeros((10, 2), dtype=torch.float64),
        hit_global_positions=torch.tensor(points, dtype=torch.float64),
        hit_directions=torch.tensor(tangents, dtype=torch.float64),
        hit_volume_ids=torch.tensor(volume),
        hit_layer_ids=torch.tensor(layer),
        hit_module_ids=torch.arange(1, 11),
        hit_event_ids=torch.ones(10, dtype=torch.long),
        hit_track_ids=torch.full((10,), 7, dtype=torch.long),
    )
    geometry_path = tmp_path / "nominal.json"
    states_path = tmp_path / "trackstates.root"
    geometry_path.touch()
    states_path.touch()
    monkeypatch.setattr(m, "load_nominal_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        m, "load_nominal_states", lambda _path, _mapping, max_track_entries=None: inputs
    )

    output_dir = tmp_path / "output"
    exit_code = m.main(
        [
            "--states",
            str(states_path),
            "--geometry",
            str(geometry_path),
            "--nominal-full-geometry",
            str(geometry_path),
            "--holdout-layers",
            "17:8",
            "--min-fit-hits",
            "4",
            "--min-fit-span",
            "100",
            "--max-tracks",
            "1",
            "--fit-iters",
            "250",
            "--max-pixel-local-rmse",
            "10.0",
            "--max-direction-angle-rms",
            "0.2",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "helix_validation_summary.json").read_text())
    comparison = pd.read_csv(output_dir / "helix_smt_comparison.csv")
    measurement_comparison = pd.read_csv(
        output_dir / "helix_measurement_comparison.csv"
    )
    assert summary["passed"] is True
    assert summary["fit_source"] == "measurement"
    assert summary["validation_reference"] == "measurement"
    assert summary["comparison_baseline"] == "unrefined_smt_initialization"
    assert set(comparison["split"]) == {"fit", "holdout"}
    assert set(measurement_comparison["prediction_mode"]) == set(m.PREDICTION_MODES)
    assert "fitted_measurement_residual_LOC0" in measurement_comparison
    assert set(summary["holdout_by_layer"]) == {"17:8"}
    assert set(summary["holdout_by_mode"]) == set(m.PREDICTION_MODES)
    assert set(summary["holdout_by_layer_and_mode"]["17:8"]) == set(
        m.PREDICTION_MODES
    )
    plot_dir = output_dir / "plots"
    assert (
        plot_dir / "holdout_forward_extrapolation_local_residuals.png"
    ).is_file()
    assert (
        plot_dir
        / "holdout_vol17_lay8_backward_extrapolation_local_residuals.png"
    ).is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["validation_summary"]["passed"] is True
    assert manifest["inputs"]["nominal_trackstates"]["sha256"]


def test_cli_writes_failure_artifacts_and_returns_one(tmp_path, monkeypatch):
    m = load_module()
    inputs = _straight_inputs(m, holdout_measurement_shift=1.0)
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    geometry = m.NominalGeometry(
        transforms=[],
        centers=inputs.hit_global_positions.clone(),
        rotations=rotation.expand(10, -1, -1).clone(),
        id_to_idx={
            (
                int(inputs.hit_volume_ids[i]),
                int(inputs.hit_layer_ids[i]),
                int(inputs.hit_module_ids[i]),
            ): i
            for i in range(10)
        },
    )
    geometry_path = tmp_path / "nominal.json"
    states_path = tmp_path / "trackstates.root"
    geometry_path.touch()
    states_path.touch()
    monkeypatch.setattr(m, "load_nominal_geometry", lambda _path: geometry)
    monkeypatch.setattr(
        m, "load_nominal_states", lambda _path, _mapping, max_track_entries=None: inputs
    )
    output_dir = tmp_path / "failure-output"

    exit_code = m.main(
        [
            "--states",
            str(states_path),
            "--geometry",
            str(geometry_path),
            "--nominal-full-geometry",
            str(geometry_path),
            "--holdout-layers",
            "17:8",
            "--min-fit-span",
            "300",
            "--max-tracks",
            "1",
            "--fit-iters",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 1
    assert (output_dir / "helix_smt_comparison.csv").exists()
    assert (output_dir / "helix_measurement_comparison.csv").exists()
    assert len(list((output_dir / "plots").glob("*.png"))) == 18
    summary = json.loads((output_dir / "helix_validation_summary.json").read_text())
    assert summary["passed"] is False
    assert summary["failure_reasons"]
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["validation_summary"]["passed"] is False


def test_cli_records_failed_manifest_for_runtime_error(tmp_path, monkeypatch):
    m = load_module()
    geometry_path = tmp_path / "nominal.json"
    states_path = tmp_path / "trackstates.root"
    geometry_path.write_text(
        json.dumps(
            [
                {
                    "ID": "vol=17|lay=8|sen=1",
                    "Translation": [0.0, 0.0, 0.0],
                    "RotationMatrix": np.eye(3).tolist(),
                }
            ]
        )
    )
    states_path.touch()
    monkeypatch.setattr(
        m,
        "load_nominal_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic loader failure")
        ),
    )
    output_dir = tmp_path / "runtime-failure"

    exit_code = m.main(
        [
            "--states",
            str(states_path),
            "--geometry",
            str(geometry_path),
            "--nominal-full-geometry",
            str(geometry_path),
            "--holdout-layers",
            "17:8",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "synthetic loader failure" in manifest["metrics"]["error"]


def test_cli_defaults_follow_nominal_artifact_layout():
    m = load_module()
    config = m.config_from_args(m.build_argument_parser().parse_args([]))

    assert "/nominal/" in config.states
    assert config.states.endswith("trackstates_kf_v17l8_v24l4.root")
    assert config.holdout_layers == "17:8,24:4"
    assert config.min_fit_hits == 8
    assert config.min_fit_span == 300.0
    assert config.min_extrapolation_fit_hits == 4
    assert config.min_extrapolation_fit_span == 100.0
    assert config.output_dir.endswith("data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4")


# ---------------------------------------------------------------------------
# Gate 3: posterior / predictive covariance
# ---------------------------------------------------------------------------


def _linear_gaussian_toy(dtype=torch.float64):
    """A small linear-Gaussian problem with a closed-form posterior."""
    torch.manual_seed(20260825)
    n_parameters, n_measurements = 4, 11
    design = torch.randn(n_measurements, n_parameters, dtype=dtype)
    variance = torch.rand(n_measurements, dtype=dtype) * 0.5 + 0.25
    prior_precision = torch.diag(
        torch.tensor([0.0, 4.0, 9.0, 16.0], dtype=dtype)
    )
    prediction = torch.randn(2, n_parameters, dtype=dtype)
    return design, variance, prior_precision, prediction


def test_gauss_newton_hessian_uses_the_half_chi2_convention():
    """F = 1/2 chi^2 gives H = J^T V^-1 J; a chi^2 Hessian would be twice this."""
    module = load_module()
    design, variance, prior_precision, _prediction = _linear_gaussian_toy()

    hessian = module.gauss_newton_hessian(design, variance, prior_precision)
    expected = design.T @ torch.diag(1.0 / variance) @ design + prior_precision
    torch.testing.assert_close(hessian, expected, rtol=0, atol=1e-12)

    # Differentiating 1/2 * chi2 twice must reproduce it exactly.
    parameters = torch.zeros(design.shape[1], dtype=design.dtype, requires_grad=True)

    def objective(theta):
        residual = design @ theta
        return 0.5 * (
            (residual.square() / variance).sum() + theta @ prior_precision @ theta
        )

    autograd_hessian = torch.autograd.functional.hessian(objective, parameters)
    torch.testing.assert_close(hessian, autograd_hessian, rtol=0, atol=1e-9)
    del parameters


def test_predictive_covariance_matches_the_closed_form_posterior():
    module = load_module()
    design, variance, prior_precision, prediction = _linear_gaussian_toy()

    hessian = module.gauss_newton_hessian(design, variance, prior_precision)
    covariance, diagnostics = module.predictive_covariance(
        hessian, prediction, damping=0.0
    )
    analytic = prediction @ torch.linalg.inv(hessian) @ prediction.T
    torch.testing.assert_close(covariance, analytic, rtol=1e-10, atol=1e-12)

    assert diagnostics["cholesky_ok"] is True
    assert diagnostics["symmetry_residual"] < 1e-12
    assert diagnostics["min_eigenvalue"] > 0.0
    assert diagnostics["condition_number"] > 1.0


def test_predictive_covariance_scale_is_confirmed_by_monte_carlo():
    """Independent check that no factor of two crept into the convention.

    With no prior the MAP estimator's sampling covariance IS the posterior
    covariance, so an empirical covariance over noise realisations validates the
    absolute scale without reusing the analytic formula under test.
    """
    module = load_module()
    dtype = torch.float64
    torch.manual_seed(4242)
    n_parameters, n_measurements, n_trials = 3, 40, 40000
    design = torch.randn(n_measurements, n_parameters, dtype=dtype)
    variance = torch.rand(n_measurements, dtype=dtype) * 0.4 + 0.3
    truth = torch.tensor([0.7, -1.3, 2.1], dtype=dtype)

    hessian = module.gauss_newton_hessian(design, variance, None)
    predicted, _diagnostics = module.predictive_covariance(
        hessian, torch.eye(n_parameters, dtype=dtype), damping=0.0
    )

    weighted = design.T @ torch.diag(1.0 / variance)
    noise = torch.randn(n_trials, n_measurements, dtype=dtype) * variance.sqrt()
    observations = (design @ truth).unsqueeze(0) + noise
    estimates = torch.linalg.solve(hessian, (weighted @ observations.T)).T
    empirical = torch.cov(estimates.T)

    # Off-diagonal elements sit near zero, so judge the match on the matrix
    # norm rather than element-wise relative error.
    mismatch = float((empirical - predicted).norm() / predicted.norm())
    assert mismatch < 0.05, f"empirical posterior deviates by {mismatch:.3%}"
    # The diagonal is what a factor-of-two error would blow up by 100%.
    ratio = empirical.diagonal() / predicted.diagonal()
    assert float(ratio.max()) < 1.05
    assert float(ratio.min()) > 0.95


def test_predictive_covariance_never_forms_an_explicit_inverse():
    module = load_module()
    import inspect

    source = inspect.getsource(module.predictive_covariance)
    assert "linalg.inv" not in source
    assert "cholesky_solve" in source


def test_predictive_covariance_reports_and_survives_a_singular_hessian():
    module = load_module()
    dtype = torch.float64
    # A rank-deficient direction: without damping the Cholesky must fail.
    design = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=dtype)
    variance = torch.ones(2, dtype=dtype)
    hessian = module.gauss_newton_hessian(design, variance, None)
    prediction = torch.eye(2, dtype=dtype)

    with pytest.raises(torch.linalg.LinAlgError):
        module.predictive_covariance(hessian, prediction, damping=0.0)

    covariance, diagnostics = module.predictive_covariance(
        hessian, prediction, damping=1e-6
    )
    assert torch.isfinite(covariance).all()
    assert diagnostics["min_eigenvalue"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["damping"] == pytest.approx(1e-6)


def test_damping_grid_is_pre_registered_and_brackets_production():
    module = load_module()
    assert module.PRODUCTION_DAMPING in module.DAMPING_SCAN
    assert module.DAMPING_SCAN[0] == 0.0
    assert list(module.DAMPING_SCAN) == sorted(module.DAMPING_SCAN)
    # Damping may only shrink the reported variance, never inflate it.
    design, variance, prior_precision, prediction = _linear_gaussian_toy()
    hessian = module.gauss_newton_hessian(design, variance, prior_precision)
    previous = None
    for damping in module.DAMPING_SCAN:
        covariance, _diagnostics = module.predictive_covariance(
            hessian, prediction, damping=damping
        )
        trace = float(covariance.diagonal().sum())
        if previous is not None:
            assert trace <= previous + 1e-12
        previous = trace


# ---------------------------------------------------------------------------
# Gate 2: broken-helix MAP fit and the synthetic Monte-Carlo closure
# ---------------------------------------------------------------------------

MC_ENSEMBLE_SIZE = 1200
MC_SEED = 20260825
MC_PULL_MEAN_TOL = 0.06
MC_PULL_STD_BAND = (0.93, 1.07)
MC_COVERAGE_TOL = 0.03


def _toy_planes(module, reference, path_values, dtype=torch.float64):
    """Barrel-like planes centred on the trajectory at the requested arc lengths.

    The normal is radial, as a silicon barrel sensor's is -- deliberately NOT
    the track tangent.  A plane normal to the track cannot see the reference
    helix slide along its own trajectory at all, which turns a merely soft
    direction into an exact null space and makes the fit look divergent for a
    reason no real detector has.
    """
    helix_model = importlib.import_module("scripts.helix_model")
    n_hits = path_values.shape[1]
    expanded = reference[:, None, :].expand(-1, n_hits, -1)
    centers = helix_model.helix_point(expanded, path_values)
    radial = torch.stack(
        (centers[..., 0], centers[..., 1], torch.zeros_like(centers[..., 2])), dim=-1
    )
    normal = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Complete an orthonormal frame; column 2 is the plane normal by convention.
    helper = torch.zeros_like(normal)
    helper[..., 2] = 1.0
    axis0 = torch.cross(helper, normal, dim=-1)
    axis0 = axis0 / axis0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis1 = torch.cross(normal, axis0, dim=-1)
    rotations = torch.stack((axis0, axis1, normal), dim=-1)
    del module, dtype
    return centers, rotations


def test_broken_helix_prediction_matches_the_single_helix_without_kinks():
    module = load_module()
    helix_model = importlib.import_module("scripts.helix_model")
    reference = torch.tensor([[3.0, -1.0, 0.5, 0.2, 0.4, 2.0e-4]], dtype=torch.float64)
    path_values = torch.tensor([[60.0, 180.0, 320.0, 480.0]], dtype=torch.float64)
    centers, rotations = _toy_planes(module, reference, path_values)
    node_s = torch.tensor([[120.0, 400.0]], dtype=torch.float64)
    node_valid = torch.ones((1, 2), dtype=torch.bool)
    kinks = torch.zeros((1, 2, 2), dtype=torch.float64)

    local, valid = module.broken_helix_local_prediction(
        reference, kinks, node_s, node_valid, path_values, centers, rotations
    )
    assert bool(valid.all())
    # Planes are centred on the trajectory, so the local prediction is the origin.
    torch.testing.assert_close(local, torch.zeros_like(local), atol=1e-9, rtol=0)
    del helix_model


def test_gauss_newton_recovers_injected_kinks_on_a_noiseless_toy():
    module = load_module()
    torch.manual_seed(7)
    reference = torch.tensor([[3.0, -1.0, 0.5, 0.2, 0.4, 2.0e-4]], dtype=torch.float64)
    path_values = torch.tensor(
        [[40.0, 110.0, 200.0, 300.0, 410.0, 520.0, 640.0]], dtype=torch.float64
    )
    centers, rotations = _toy_planes(module, reference, path_values)
    node_s = torch.tensor([[150.0, 360.0]], dtype=torch.float64)
    node_valid = torch.ones((1, 2), dtype=torch.bool)
    kink_sigma = torch.full((1, 2), 5.0e-4, dtype=torch.float64)
    truth = torch.tensor([[[6.0e-4, -4.0e-4], [-3.0e-4, 7.0e-4]]], dtype=torch.float64)

    measured, valid = module.broken_helix_local_prediction(
        reference, truth, node_s, node_valid, path_values, centers, rotations
    )
    assert bool(valid.all())
    variance = torch.full_like(measured, 1.0e-6)
    mask = torch.ones_like(measured, dtype=torch.bool)

    fitted_reference, fitted_kinks, stats = module.fit_broken_helix(
        reference,
        torch.zeros_like(truth),
        node_s,
        node_valid,
        kink_sigma,
        measured,
        variance,
        mask,
        path_values,
        centers,
        rotations,
    )
    assert stats["converged"], stats
    assert stats["final_objective"] < stats["initial_objective"]
    torch.testing.assert_close(fitted_kinks, truth, atol=2e-6, rtol=0)

    # The reference helix carries a gauge freedom: sliding its origin along the
    # trajectory and rotating psi0 by rho*ds describes the same curve, so the
    # fitted (x0, y0, z0, psi0) need not equal the injected ones.  Assert the
    # gauge-invariant thing instead -- the trajectory the model predicts.
    refit_local, refit_valid = module.broken_helix_local_prediction(
        fitted_reference, fitted_kinks, node_s, node_valid,
        path_values, centers, rotations,
    )
    assert bool(refit_valid.all())
    # 3.7e-5 mm is the floor set by the near-degenerate slide direction, which
    # is orders below any sensor resolution in this detector.
    torch.testing.assert_close(refit_local, measured, atol=1e-4, rtol=0)


def test_monte_carlo_ensemble_produces_calibrated_pulls():
    """Gate 2 closure: sample kinks from the prior, refit, and check calibration.

    A single recovered kink proves nothing about the prior's scale.  This draws a
    fixed-seed ensemble, predicts a held-out plane that took no part in the fit,
    and checks that the pull built from the Gate 3 predictive covariance is
    centred and unit-width, plus that the 68%/95% intervals cover as advertised.
    """
    module = load_module()
    helix_model = importlib.import_module("scripts.helix_model")
    dtype = torch.float64
    generator = torch.Generator().manual_seed(MC_SEED)
    ensemble = MC_ENSEMBLE_SIZE

    reference = torch.tensor(
        [[3.0, -1.0, 0.5, 0.2, 0.4, 2.0e-4]], dtype=dtype
    ).repeat(ensemble, 1)
    fit_path = torch.tensor(
        [[40.0, 110.0, 200.0, 300.0, 410.0, 520.0]], dtype=dtype
    ).repeat(ensemble, 1)
    holdout_path = torch.tensor([[700.0]], dtype=dtype).repeat(ensemble, 1)
    node_s = torch.tensor([[150.0, 360.0, 600.0]], dtype=dtype).repeat(ensemble, 1)
    node_valid = torch.ones((ensemble, 3), dtype=torch.bool)

    x_over_x0 = torch.full((ensemble, 3), 0.02, dtype=dtype)
    q_over_p = torch.full((ensemble,), 0.2, dtype=dtype)
    kink_sigma = helix_model.highland_kink_sigma(x_over_x0, q_over_p)

    truth_kinks = torch.randn((ensemble, 3, 2), dtype=dtype, generator=generator)
    truth_kinks = truth_kinks * kink_sigma.unsqueeze(-1)

    fit_centers, fit_rotations = _toy_planes(module, reference, fit_path)
    holdout_centers, holdout_rotations = _toy_planes(module, reference, holdout_path)

    clean, valid = module.broken_helix_local_prediction(
        reference, truth_kinks, node_s, node_valid, fit_path, fit_centers, fit_rotations
    )
    truth_holdout, holdout_valid = module.broken_helix_local_prediction(
        reference,
        truth_kinks,
        node_s,
        node_valid,
        holdout_path,
        holdout_centers,
        holdout_rotations,
    )
    assert bool(valid.all()) and bool(holdout_valid.all())

    sigma_measurement = 0.05
    variance = torch.full_like(clean, sigma_measurement**2)
    measured = clean + torch.randn(clean.shape, dtype=dtype, generator=generator) * sigma_measurement
    mask = torch.ones_like(measured, dtype=torch.bool)

    fitted_reference, fitted_kinks, stats = module.fit_broken_helix(
        reference,
        torch.zeros_like(truth_kinks),
        node_s,
        node_valid,
        kink_sigma,
        measured,
        variance,
        mask,
        fit_path,
        fit_centers,
        fit_rotations,
    )
    assert stats["converged"], stats

    predicted, covariance = module.broken_helix_predictive_distribution(
        fitted_reference,
        fitted_kinks,
        node_s,
        node_valid,
        kink_sigma,
        variance,
        mask,
        fit_path,
        fit_centers,
        fit_rotations,
        holdout_path,
        holdout_centers,
        holdout_rotations,
    )
    residual = (predicted - truth_holdout)[:, 0, :]
    denominator = covariance[:, 0, :].sqrt()
    pull = residual / denominator

    for component in range(2):
        values = pull[:, component]
        mean = float(values.mean())
        std = float(values.std(correction=1))
        assert abs(mean) < MC_PULL_MEAN_TOL, f"component {component} pull mean {mean:.4f}"
        assert MC_PULL_STD_BAND[0] < std < MC_PULL_STD_BAND[1], (
            f"component {component} pull std {std:.4f}"
        )
        for quantile, target in ((1.0, 0.6827), (1.96, 0.9500)):
            coverage = float((values.abs() < quantile).to(dtype).mean())
            assert abs(coverage - target) < MC_COVERAGE_TOL, (
                f"component {component} coverage at {quantile} sigma: {coverage:.4f}"
            )


# ---------------------------------------------------------------------------
# Gate 4: paired single/broken held-out comparison
# ---------------------------------------------------------------------------


def _gate4_fixture(
    module,
    n_tracks=6,
    n_hits=9,
    holdout_index=4,
    modes=None,
    kink_scale=8.0e-4,
    noise=0.0,
    seed=11,
    node_paths=(150.0, 330.0),
    layer=(17, 8),
):
    """Synthetic batches, geometry and material for one paired-holdout run."""
    helix_model = importlib.import_module("scripts.helix_model")
    generator = torch.Generator().manual_seed(seed)
    reference = torch.zeros((n_tracks, 6), dtype=torch.float64)
    reference[:, 3] = 0.2 + 0.05 * torch.arange(n_tracks, dtype=torch.float64)
    reference[:, 4] = 0.4
    reference[:, 5] = 2.0e-4
    path_values = torch.linspace(40.0, 640.0, n_hits, dtype=torch.float64)
    path_values = path_values[None, :].expand(n_tracks, -1).contiguous()
    centers, rotations = _toy_planes(module, reference, path_values)

    node_s = torch.tensor(node_paths, dtype=torch.float64)[None, :].expand(
        n_tracks, -1
    ).contiguous()
    node_valid = torch.ones_like(node_s, dtype=torch.bool)
    kinks = kink_scale * torch.randn(
        (n_tracks, node_s.shape[1], 2), generator=generator, dtype=torch.float64
    )
    measured, valid = module.broken_helix_local_prediction(
        reference, kinks, node_s, node_valid, path_values, centers, rotations
    )
    assert bool(valid.all())
    if noise:
        measured = measured + noise * torch.randn(
            measured.shape, generator=generator, dtype=torch.float64
        )

    sensor_idx = torch.arange(n_tracks * n_hits, dtype=torch.long).reshape(
        n_tracks, n_hits
    )
    geometry = module.NominalGeometry(
        transforms=[],
        centers=centers.reshape(-1, 3),
        rotations=rotations.reshape(-1, 3, 3),
        id_to_idx={},
    )
    holdout_mask = torch.zeros((n_tracks, n_hits), dtype=torch.bool)
    holdout_mask[:, holdout_index] = True
    volume_id = torch.zeros((n_tracks, n_hits), dtype=torch.long)
    layer_id = torch.zeros((n_tracks, n_hits), dtype=torch.long)
    volume_id[:, holdout_index] = layer[0]
    layer_id[:, holdout_index] = layer[1]
    if modes is None:
        modes = tuple("interpolation" for _ in range(n_tracks))
    batches = module.TrackBatches(
        measured_local=measured,
        measurement_variance=torch.full_like(measured, 1.0e-4),
        loc1_valid=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        smt_local=torch.zeros_like(measured),
        sensor_idx=sensor_idx,
        hit_mask=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        fit_mask=~holdout_mask,
        holdout_mask=holdout_mask,
        g_smt=helix_model.helix_point(
            reference[:, None, :].expand(-1, n_hits, -1), path_values
        ),
        p_smt=helix_model.helix_tangent(
            reference[:, None, :].expand(-1, n_hits, -1), path_values
        ),
        volume_id=volume_id,
        layer_id=layer_id,
        module_id=torch.zeros((n_tracks, n_hits), dtype=torch.long),
        event_track_ids=torch.stack(
            (torch.arange(n_tracks), torch.zeros(n_tracks, dtype=torch.long)), dim=1
        ),
        fit_span=torch.full((n_tracks,), 600.0, dtype=torch.float64),
        prediction_mode=tuple(modes),
        context_hits_before=torch.zeros(n_tracks, dtype=torch.long),
        context_hits_after=torch.zeros(n_tracks, dtype=torch.long),
        nearest_context_distance=torch.zeros(n_tracks, dtype=torch.float64),
        farthest_context_distance=torch.zeros(n_tracks, dtype=torch.float64),
    )
    node_position = helix_model.helix_point(
        reference[:, None, :].expand(-1, node_s.shape[1], -1), node_s
    )
    return {
        "batches": batches,
        "geometry": geometry,
        "reference": reference,
        "node_position": node_position,
        "node_s": node_s,
        "path_values": path_values,
        "kinks": kinks,
    }


def _write_material_npz(path, fixture, x_over_x0=2.0e-2, keys=None):
    n_tracks, n_nodes, _ = fixture["node_position"].shape
    keys = keys or [(index, 0) for index in range(n_tracks)]
    np.savez_compressed(
        path,
        event_nr=np.array([key[0] for key in keys], dtype=np.int64),
        track_nr=np.array([key[1] for key in keys], dtype=np.int64),
        q_over_p_gev_inv=np.full(n_tracks, 1.0 / 5.0),
        node_offsets=np.arange(n_tracks + 1, dtype=np.int64) * n_nodes,
        path_length_mm=np.tile(
            fixture["node_s"][0].numpy(), n_tracks
        ),
        position_mm=fixture["node_position"].reshape(-1, 3).numpy(),
        direction=np.zeros((n_tracks * n_nodes, 3)),
        x_over_x0=np.full(n_tracks * n_nodes, x_over_x0),
        surface_geometry_id=np.zeros(n_tracks * n_nodes, dtype=np.uint64),
        volume_geometry_id=np.zeros(n_tracks * n_nodes, dtype=np.uint64),
    )
    return path


def test_material_artifact_rejects_duplicate_track_keys(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=2)
    path = _write_material_npz(
        tmp_path / "dup.npz", fixture, keys=[(0, 0), (0, 0)]
    )
    with pytest.raises(ValueError, match="duplicate track key"):
        module.load_material_artifact(path)


def test_material_artifact_rejects_a_missing_schema_field(tmp_path):
    module = load_module()
    path = tmp_path / "partial.npz"
    np.savez_compressed(path, event_nr=np.array([0]), track_nr=np.array([0]))
    with pytest.raises(KeyError, match="missing fields"):
        module.load_material_artifact(path)


def test_paired_holdout_fails_closed_when_a_row_has_no_material(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=3)
    path = _write_material_npz(
        tmp_path / "partial.npz", fixture, keys=[(0, 0), (1, 0), (99, 0)]
    )
    material = module.load_material_artifact(path)
    with pytest.raises(ValueError, match="no material nodes"):
        module.run_paired_holdout(
            fixture["batches"],
            fixture["geometry"],
            fixture["reference"],
            material,
            ((17, 8),),
            tmp_path,
        )


def test_paired_holdout_drop_mode_respects_the_pre_registered_limit(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=3)
    path = _write_material_npz(
        tmp_path / "partial.npz", fixture, keys=[(0, 0), (1, 0), (99, 0)]
    )
    material = module.load_material_artifact(path)
    # One row of three is 33%, far above MAX_MISSING_MATERIAL_FRACTION.
    with pytest.raises(ValueError, match="missing material fraction"):
        module.run_paired_holdout(
            fixture["batches"],
            fixture["geometry"],
            fixture["reference"],
            material,
            ((17, 8),),
            tmp_path,
            on_missing_material="drop",
        )


def test_prepare_broken_inputs_clips_nodes_to_the_row_fit_window():
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=2, node_paths=(150.0, 330.0, 5000.0))
    inputs = module.prepare_broken_inputs(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        fixture["node_position"],
        torch.full(fixture["node_s"].shape, 2.0e-2, dtype=torch.float64),
        torch.ones_like(fixture["node_s"], dtype=torch.bool),
        torch.full((2,), 0.2, dtype=torch.float64),
    )
    # The hits span s in [40, 640]; the node at 5000 mm is outside every window.
    assert inputs.n_nodes_in_window.tolist() == [2, 2]


def test_prepare_broken_inputs_sorts_nodes_along_the_frozen_reference():
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=2, node_paths=(330.0, 150.0, 500.0))
    inputs = module.prepare_broken_inputs(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        fixture["node_position"],
        torch.full(fixture["node_s"].shape, 2.0e-2, dtype=torch.float64),
        torch.ones_like(fixture["node_s"], dtype=torch.bool),
        torch.full((2,), 0.2, dtype=torch.float64),
    )
    # The segment recursion walks nodes in index order, so unsorted arc lengths
    # would silently build the wrong trajectory rather than raise.
    ordered = inputs.node_s[:, :3]
    assert bool((ordered[:, 1:] >= ordered[:, :-1]).all())
    torch.testing.assert_close(
        ordered[0], torch.tensor([150.0, 330.0, 500.0], dtype=torch.float64),
        atol=1e-6, rtol=0,
    )


def test_single_arm_runs_the_same_solver_with_no_material_nodes():
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=2)
    inputs = module.prepare_broken_inputs(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        None,
        None,
        None,
        None,
    )
    assert inputs.node_s.shape[1] == 0
    arm = module.run_broken_arm(inputs, fixture["reference"])
    # Zero nodes means zero kinks: the arm IS the single helix, fitted by the
    # same Gauss-Newton solver so the paired comparison has no optimiser confound.
    assert arm["kinks"].shape[1] == 0
    assert arm["convergence"]["converged"]
    assert torch.isfinite(arm["predictive_variance"]).all()


def test_paired_holdout_is_inconclusive_when_a_required_group_is_small(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
    )
    assert report["status"] == "inconclusive"
    assert any("N=" in reason for reason in report["reasons"])
    # Never silently reported as a pass just because no threshold was violated.
    assert report["status"] != "passed"


def test_paired_holdout_shares_one_row_set_and_hashes_it(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
    )
    keys_path = tmp_path / "holdout_keys.csv"
    assert keys_path.exists()
    assert not (tmp_path / "holdout_keys_single_arm.csv").exists()
    digest = hashlib.sha256(keys_path.read_bytes()).hexdigest()
    assert report["holdout_keys_sha256"] == digest
    rows = keys_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split(",")[:3] == ["event_nr", "track_nr", "mode"]
    assert len(rows) == 1 + report["n_rows_used"]


def test_broken_arm_recovers_material_kinks_the_single_arm_cannot(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=8, noise=2.0e-3)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
    )
    entry = report["groups"]["17:8"]["interpolation"]["LOC0"]
    assert entry["n"] == 8
    # Not a gate -- the fixture is far too small for one -- but the mechanism
    # has to be visible or the wiring is wrong.
    assert entry["broken"]["rmse_mm"] < entry["single"]["rmse_mm"]
    assert report["convergence"]["single"]["converged"]
    assert report["convergence"]["broken"]["converged"]


def test_damping_sensitivity_covers_the_pre_registered_grid(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4, noise=1.0e-3)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
    )
    scanned = report["damping_sensitivity"]["by_damping"]
    assert set(scanned) == {str(value) for value in module.DAMPING_SCAN}
    assert isinstance(report["damping_sensitivity"]["stable"], bool)


def test_energy_loss_reverification_uses_the_gate0_limits(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4, noise=1.0e-3)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    # A node set displaced far enough to move a segment boundary must reopen the
    # item, never quietly close it.
    displaced = module.load_material_artifact(
        _write_material_npz(
            tmp_path / "eloss.npz",
            _gate4_fixture(module, n_tracks=4, node_paths=(150.0, 620.0)),
        )
    )
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
        energy_loss_material=displaced,
    )
    check = report["energy_loss_reverification"]
    assert check["status"] == "open"
    assert check["segment_flip_limit"] == module.ENERGY_LOSS_MAX_SEGMENT_FLIP_RATE
    assert (
        check["prediction_shift_limit_p99"]
        == module.ENERGY_LOSS_MAX_PREDICTION_SIGMA_P99
    )


def test_energy_loss_item_stays_open_without_a_second_artifact(tmp_path):
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
    )
    assert report["energy_loss_reverification"]["status"] == "open"


def test_gate4_thresholds_are_pre_registered_constants():
    module = load_module()
    assert module.BOOTSTRAP_RESAMPLES == 1000
    assert module.BOOTSTRAP_SEED == 20260825
    assert module.MIN_GROUP_COMPONENTS == 200
    assert module.INTERPOLATION_RMSE_IMPROVEMENT == 0.20
    assert module.EXTRAPOLATION_RMSE_REGRESSION == 0.05
    assert module.PREDICTIVE_CHI2_BAND == (0.7, 1.5)
    assert module.MAX_ABS_PULL_MEAN == 0.2
    assert module.ENERGY_LOSS_MAX_SEGMENT_FLIP_RATE == 0.0
    assert module.ENERGY_LOSS_MAX_PREDICTION_SIGMA_P99 == 0.10


def test_gate4_improvement_requires_the_ci_bound_not_the_point_estimate():
    module = load_module()
    groups = {
        "17:8": {
            mode: {
                "LOC0": {
                    "n": 500,
                    "single": {"rmse_mm": 1.0},
                    "broken": {
                        "rmse_mm": 0.5,
                        "pull_mean": 0.0,
                        "pull_std": 1.0,
                        "predictive_chi2_per_component": 1.0,
                        "intersection_failure_rate": 0.0,
                    },
                    # Point estimate clears the bar; the CI does not.
                    "relative_rmse_change": {"point": -0.5, "low": -0.9, "high": -0.05},
                    "delta_mean_nll": {"point": -0.4, "low": -0.9, "high": -0.1},
                }
            }
            for mode in module.PREDICTION_MODES
        }
    }
    status, reasons = module.evaluate_broken_helix_gate(groups, ((17, 8),))
    assert status == "failed"
    assert any("relative RMSE CI upper bound" in reason for reason in reasons)


def test_gate4_nll_gate_rejects_a_ci_straddling_zero():
    module = load_module()
    groups = {
        "17:8": {
            mode: {
                "LOC0": {
                    "n": 500,
                    "single": {"rmse_mm": 1.0},
                    "broken": {
                        "rmse_mm": 0.5,
                        "pull_mean": 0.0,
                        "pull_std": 1.0,
                        "predictive_chi2_per_component": 1.0,
                        "intersection_failure_rate": 0.0,
                    },
                    "relative_rmse_change": {"point": -0.5, "low": -0.9, "high": -0.4},
                    "delta_mean_nll": {"point": -0.4, "low": -0.9, "high": 0.2},
                }
            }
            for mode in module.PREDICTION_MODES
        }
    }
    status, reasons = module.evaluate_broken_helix_gate(groups, ((17, 8),))
    assert status == "failed"
    assert any("NLL CI upper bound" in reason for reason in reasons)


def test_broken_helix_gate_is_skipped_without_a_material_artifact(tmp_path, monkeypatch):
    module = load_module()
    config = module.HelixValidationConfig(model="broken", material_artifact=None)
    with pytest.raises(ValueError, match="requires --material-artifact"):
        module.run_helix_validation(config)
    del tmp_path, monkeypatch


def test_row_chunking_reproduces_the_unchunked_arm_exactly():
    """Chunking must be an execution detail, not a numerical one."""
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=7, noise=1.0e-3)
    inputs = module.prepare_broken_inputs(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        fixture["node_position"],
        torch.full(fixture["node_s"].shape, 2.0e-2, dtype=torch.float64),
        torch.ones_like(fixture["node_s"], dtype=torch.bool),
        torch.full((7,), 0.2, dtype=torch.float64),
    )
    whole = module.run_broken_arm(inputs, fixture["reference"], chunk_rows=0)
    chunked = module.run_broken_arm(inputs, fixture["reference"], chunk_rows=3)

    assert chunked["convergence"]["n_chunks"] == 3
    for name in ("predicted", "predictive_variance", "kinks", "segment_of_hit"):
        torch.testing.assert_close(chunked[name], whole[name], atol=0, rtol=0)
    assert chunked["covariance_diagnostics"][
        str(module.PRODUCTION_DAMPING)
    ]["batch_size"] == 7


def test_newton_decrement_is_near_zero_at_the_optimum_and_large_away_from_it():
    """The scale-free convergence statement, since the gradient is whitened."""
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4, noise=1.0e-3)
    inputs = module.prepare_broken_inputs(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        fixture["node_position"],
        torch.full(fixture["node_s"].shape, 2.0e-2, dtype=torch.float64),
        torch.ones_like(fixture["node_s"], dtype=torch.bool),
        torch.full((4,), 0.2, dtype=torch.float64),
    )
    arm = module.run_broken_arm(inputs, fixture["reference"])
    at_optimum = arm["newton_decrement"]
    assert bool((at_optimum < module.NEWTON_DECREMENT_STATIONARY).all()), at_optimum

    displaced = arm["reference"].clone()
    displaced[:, 3] += 1.0e-3
    away = module.newton_decrement(
        displaced,
        arm["kinks"],
        inputs.node_s,
        inputs.node_valid,
        inputs.kink_sigma,
        inputs.measured,
        inputs.variance,
        inputs.mask,
        inputs.hit_s,
        inputs.centers,
        inputs.rotations,
    )
    assert bool((away > at_optimum).all())


def test_non_stationary_rows_make_the_verdict_inconclusive(tmp_path, monkeypatch):
    """A stalled fit is a numerical failure and must not read as a physics result."""
    module = load_module()
    fixture = _gate4_fixture(module, n_tracks=4, noise=1.0e-3)
    material = module.load_material_artifact(
        _write_material_npz(tmp_path / "mat.npz", fixture)
    )
    monkeypatch.setattr(module, "NEWTON_DECREMENT_STATIONARY", -1.0)
    report = module.run_paired_holdout(
        fixture["batches"],
        fixture["geometry"],
        fixture["reference"],
        material,
        ((17, 8),),
        tmp_path,
    )
    assert report["status"] == "inconclusive"
    assert any("non-stationary" in reason for reason in report["reasons"])
    assert report["stationarity"]["broken"]["non_stationary_row_fraction"] == 1.0


def _measurement_fit_fixture(m, n_hits=12, seed=7, dtype=torch.float64):
    """A curved track with noisy measurements: enough to separate two solvers."""
    path = torch.linspace(0.0, 500.0, n_hits, dtype=dtype)
    rho = 0.002
    points = torch.stack(
        (
            torch.sin(rho * path) / rho,
            (1.0 - torch.cos(rho * path)) / rho,
            0.1 * path,
        ),
        dim=1,
    )
    tangents = torch.stack(
        (torch.cos(rho * path), torch.sin(rho * path), torch.full_like(path, 0.1)),
        dim=1,
    )
    normals = torch.nn.functional.normalize(tangents, dim=1)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand_as(normals)
    local0 = torch.nn.functional.normalize(torch.cross(up, normals, dim=1), dim=1)
    local1 = torch.cross(normals, local0, dim=1)
    rotations = torch.stack((local0, local1, normals), dim=2)
    holdout = torch.zeros(n_hits, dtype=torch.bool)
    holdout[5] = True
    holdout[9] = True
    volume = torch.where(holdout, torch.tensor(17), torch.tensor(10)).long()
    layer = torch.where(holdout, torch.tensor(8), torch.arange(n_hits)).long()
    geometry = m.NominalGeometry(
        transforms=[],
        centers=points,
        rotations=rotations,
        id_to_idx={
            (int(volume[i]), int(layer[i]), i + 1): i for i in range(n_hits)
        },
    )
    generator = torch.Generator().manual_seed(seed)
    measured = 0.02 * torch.randn((n_hits, 2), generator=generator, dtype=dtype)
    inputs = m.HelixInputs(
        hit_sensor_ids=torch.arange(n_hits),
        measured_local=measured,
        measurement_variance=torch.full((n_hits, 2), 4.0e-4, dtype=dtype),
        loc1_valid=torch.ones(n_hits, dtype=torch.bool),
        smt_local=torch.zeros((n_hits, 2), dtype=dtype),
        hit_global_positions=points,
        hit_directions=tangents,
        hit_volume_ids=volume,
        hit_layer_ids=layer,
        hit_module_ids=torch.arange(1, n_hits + 1),
        hit_event_ids=torch.zeros(n_hits, dtype=torch.long),
        hit_track_ids=torch.ones(n_hits, dtype=torch.long),
    )
    batches = m.build_track_batches(
        inputs, ((17, 8),), min_fit_hits=8, min_fit_span=300.0
    )
    seeded = m.seed_helices(batches, n_iter=200, dtype=dtype)
    return batches, geometry, seeded


def _measurement_objective(m, batches, geometry, parameters, dtype=torch.float64):
    """The half-chi2 the two solvers are supposed to be minimising, per row."""
    inputs = m.prepare_broken_inputs(
        batches, geometry, parameters, None, None, None, None, dtype=dtype
    )
    kinks = inputs.node_s.new_zeros((parameters.shape[0], 0, 2))
    residual = m._standardized_residual(
        m._pack_parameters(parameters, kinks),
        0,
        m.reference_gauge_frame(parameters),
        inputs.node_s,
        inputs.node_valid,
        inputs.kink_sigma,
        inputs.measured,
        inputs.variance,
        inputs.mask,
        inputs.hit_s,
        inputs.centers,
        inputs.rotations,
    )
    return 0.5 * residual.square().sum(dim=1)


def test_gauss_newton_measurement_fit_beats_adam_on_the_same_rows():
    m = load_module()
    batches, geometry, seeded = _measurement_fit_fixture(m)

    adam = m.fit_helices_to_measurements(
        batches, geometry, seeded, n_iter=m.DEFAULT_FIT_ITERS
    )
    gauss_newton, _diagnostics = m.fit_helices_gauss_newton(
        batches, geometry, seeded
    )

    adam_objective = _measurement_objective(m, batches, geometry, adam)
    gn_objective = _measurement_objective(m, batches, geometry, gauss_newton)
    assert torch.all(gn_objective < adam_objective)
    # The Iteration 14 finding was an order-of-magnitude gap, not a rounding
    # difference; a test that only checks "<" would pass on a broken solver.
    assert float(gn_objective.max()) < 0.01 * float(adam_objective.min())


def test_gauss_newton_measurement_fit_reaches_a_stationary_point():
    m = load_module()
    batches, geometry, seeded = _measurement_fit_fixture(m)

    _fitted, diagnostics = m.fit_helices_gauss_newton(batches, geometry, seeded)

    stationarity = diagnostics["stationarity"]
    assert stationarity["non_stationary_row_fraction"] == 0.0
    assert stationarity["newton_decrement_max"] < m.NEWTON_DECREMENT_STATIONARY
    assert diagnostics["convergence"]["converged"] is True


def test_adam_measurement_fit_is_not_stationary_which_is_the_defect():
    m = load_module()
    batches, geometry, seeded = _measurement_fit_fixture(m)

    adam = m.fit_helices_to_measurements(
        batches, geometry, seeded, n_iter=m.DEFAULT_FIT_ITERS
    )
    decrement = m.measurement_newton_decrement(batches, geometry, adam)

    # This is the defect Iteration 14 mis-diagnosed as multiple scattering: the
    # Adam refit stops on budget, far from any stationary point.
    assert float(decrement.min()) > m.NEWTON_DECREMENT_STATIONARY


def test_gauss_newton_measurement_fit_matches_the_gate4_single_arm():
    m = load_module()
    batches, geometry, seeded = _measurement_fit_fixture(m)

    fitted, _diagnostics = m.fit_helices_gauss_newton(batches, geometry, seeded)
    inputs = m.prepare_broken_inputs(
        batches, geometry, seeded, None, None, None, None
    )
    arm = m.run_broken_arm(inputs, seeded)

    # Same solver, same objective, same rows: the main path must not be a
    # second, subtly different fit sitting next to the gate's single arm.
    assert torch.allclose(fitted, arm["reference"], atol=1e-9, rtol=0.0)


def test_gauss_newton_measurement_fit_chunking_is_exact():
    m = load_module()
    batches, geometry, seeded = _measurement_fit_fixture(m)

    whole, _ = m.fit_helices_gauss_newton(batches, geometry, seeded, chunk_rows=0)
    chunked, _ = m.fit_helices_gauss_newton(batches, geometry, seeded, chunk_rows=1)

    assert torch.equal(whole, chunked)


def test_fit_solver_option_rejects_an_unknown_name():
    m = load_module()
    config = m.HelixValidationConfig(states="states.root", fit_solver="lbfgs")
    with pytest.raises(ValueError, match="fit_solver"):
        m.run_helix_validation(config)


def test_fit_solver_defaults_to_gauss_newton():
    m = load_module()
    assert m.HelixValidationConfig(states="states.root").fit_solver == "gauss-newton"


def test_measurement_newton_decrement_chunking_is_exact():
    m = load_module()
    batches, geometry, seeded = _measurement_fit_fixture(m)

    whole = m.measurement_newton_decrement(batches, geometry, seeded, chunk_rows=0)
    chunked = m.measurement_newton_decrement(batches, geometry, seeded, chunk_rows=1)

    assert torch.equal(whole, chunked)
