from __future__ import annotations

import builtins
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
            "volume_id": [10, 10, 17, 17],
            "layer_id": [2, 2, 8, 8],
            "fitted_intersection_valid": [True, False, True, True],
            "fitted_dLOC0": [1.0, 999.0, -2.0, np.nan],
            "fitted_dLOC1": [2.0, 999.0, 3.0, 4.0],
            "fitted_dg_x": [3.0, 999.0, 0.0, 5.0],
            "fitted_dg_y": [4.0, 999.0, 0.0, 12.0],
            "fitted_dg_z": [0.0, 999.0, 2.0, 0.0],
        }
    )


def test_fitted_residual_components_filter_invalid_values_and_compute_norm():
    plots = load_plot_module()

    components = plots.fitted_residual_components(_comparison_frame())

    assert components["local0"].tolist() == [1.0, -2.0]
    assert components["local1"].tolist() == [2.0, 3.0, 4.0]
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
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_write_residual_plots_creates_placeholder_for_no_valid_intersections(tmp_path):
    plots = load_plot_module()
    frame = _comparison_frame()
    frame["fitted_intersection_valid"] = False

    paths = plots.write_residual_plots(
        frame, tmp_path, holdout_layers=((17, 8),)
    )

    assert len(paths) == 6
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

    selected_modules = batches.module_id[0, batches.hit_mask[0]].tolist()
    assert selected_modules == [4, 5, 6, 7, 8]
    assert batches.fit_mask.sum().item() == 4
    assert batches.holdout_mask.sum().item() == 1


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
            "8",
            "--min-fit-span",
            "300",
            "--max-tracks",
            "1",
            "--fit-iters",
            "250",
            "--max-pixel-local-rmse",
            "2.0",
            "--max-direction-angle-rms",
            "0.02",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "helix_validation_summary.json").read_text())
    comparison = pd.read_csv(output_dir / "helix_smt_comparison.csv")
    assert summary["passed"] is True
    assert summary["fit_source"] == "measurement"
    assert summary["comparison_baseline"] == "unrefined_smt_initialization"
    assert set(comparison["split"]) == {"fit", "holdout"}
    assert set(summary["holdout_by_layer"]) == {"17:8"}
    plot_dir = output_dir / "plots"
    assert {path.name for path in plot_dir.glob("*.png")} == {
        "fit_local_residuals.png",
        "fit_global_residuals.png",
        "holdout_overall_local_residuals.png",
        "holdout_overall_global_residuals.png",
        "holdout_vol17_lay8_local_residuals.png",
        "holdout_vol17_lay8_global_residuals.png",
    }


def test_cli_writes_failure_artifacts_and_returns_one(tmp_path, monkeypatch):
    m = load_module()
    inputs = _straight_inputs(m)
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
    assert len(list((output_dir / "plots").glob("*.png"))) == 6
    summary = json.loads((output_dir / "helix_validation_summary.json").read_text())
    assert summary["passed"] is False
    assert summary["failure_reasons"]


def test_cli_defaults_follow_nominal_artifact_layout():
    m = load_module()
    config = m.config_from_args(m.build_argument_parser().parse_args([]))

    assert "/nominal/" in config.states
    assert config.states.endswith("trackstates_kf_v17l8_v24l4.root")
    assert config.holdout_layers == "17:8,24:4"
    assert config.output_dir.endswith("data/experiments/helix_validation/Vol17_Lay8__Vol24_Lay4")
