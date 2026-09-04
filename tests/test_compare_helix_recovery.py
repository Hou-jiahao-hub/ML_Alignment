from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import torch
import uproot


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_helix_recovery.py"
    assert path.exists(), "compare_helix_recovery.py has not been implemented"
    spec = importlib.util.spec_from_file_location("compare_helix_recovery", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_geometry(path, surfaces):
    import json

    transforms = [
        {
            "ID": f"vol={volume}|lay={layer}|sen={module}",
            "Translation": [float(index * 100), 0.0, 0.0],
            "RotationMatrix": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
        for index, (volume, layer, module) in enumerate(surfaces)
    ]
    path.write_text(json.dumps(transforms), encoding="utf-8")


def _write_trackstates(path, tracks, *, local_x_override=None):
    per_state = {
        name: []
        for name in (
            "volume_id",
            "layer_id",
            "module_id",
            "pathLength",
            "l_x_hit",
            "l_y_hit",
            "eLOC0_smt",
            "eLOC1_smt",
            "g_x_smt",
            "g_y_smt",
            "g_z_smt",
            "px_smt",
            "py_smt",
            "pz_smt",
            "err_x_hit",
            "err_y_hit",
        )
    }
    for track_index, (_event, _track, states) in enumerate(tracks):
        for name in per_state:
            if name == "err_x_hit":
                values = [0.1 for state in states if np.isfinite(state[4])]
            elif name == "err_y_hit":
                values = [0.2 for state in states if np.isfinite(state[4])]
            else:
                field = {
                    "volume_id": 0,
                    "layer_id": 1,
                    "module_id": 2,
                    "pathLength": 3,
                    "l_x_hit": 4,
                    "l_y_hit": 5,
                    "eLOC0_smt": 6,
                    "eLOC1_smt": 7,
                    "g_x_smt": 8,
                    "g_y_smt": 9,
                    "g_z_smt": 10,
                    "px_smt": 11,
                    "py_smt": 12,
                    "pz_smt": 13,
                }[name]
                values = [state[field] for state in states]
            per_state[name].append(values)
    if local_x_override is not None:
        entry, state, value = local_x_override
        per_state["l_x_hit"][entry][state] = value
    with uproot.recreate(path) as root_file:
        root_file["trackstates"] = {
            "event_nr": np.asarray([track[0] for track in tracks], dtype=np.int64),
            "track_nr": np.asarray([track[1] for track in tracks], dtype=np.int64),
            **{name: ak.Array(values) for name, values in per_state.items()},
        }


def _loader_fixture(tmp_path):
    surfaces = ((10, 1, 1), (17, 8, 2))
    geometry_nominal = tmp_path / "nominal.json"
    geometry_misaligned = tmp_path / "misaligned.json"
    _write_geometry(geometry_nominal, surfaces)
    _write_geometry(geometry_misaligned, tuple(reversed(surfaces)))

    def state(volume, layer, module, s, x, y=0.5):
        return (
            volume,
            layer,
            module,
            s,
            x,
            y,
            x + 0.01,
            y + 0.01,
            s,
            1.0,
            2.0,
            1.0,
            0.0,
            0.0,
        )

    tracks = [
        (5, 20, [state(10, 1, 1, 0.0, 1.0), state(10, 1, 1, 1.0, 2.0)]),
        (5, 10, [state(10, 1, 1, 0.0, 3.0), state(17, 8, 2, 1.0, 4.0)]),
        (2, 30, [state(10, 1, 1, 0.0, 5.0), state(17, 8, 2, 1.0, 6.0)]),
    ]
    nominal_states = tmp_path / "nominal.root"
    misaligned_states = tmp_path / "misaligned.root"
    _write_trackstates(nominal_states, tracks)
    _write_trackstates(misaligned_states, tracks)
    return nominal_states, misaligned_states, geometry_nominal, geometry_misaligned, tracks


def _cli_fixture(tmp_path):
    surfaces = [(10, index + 1, index + 1) for index in range(11)]
    surfaces[5] = (17, 8, 6)
    nominal_geometry = tmp_path / "nominal-geometry.json"
    misaligned_geometry = tmp_path / "misaligned-geometry.json"
    _write_geometry(nominal_geometry, surfaces)
    _write_geometry(misaligned_geometry, surfaces)
    states = []
    for index, (volume, layer, module) in enumerate(surfaces):
        position = float(index * 100)
        states.append(
            (
                volume,
                layer,
                module,
                position,
                0.0,
                0.0,
                0.0,
                0.0,
                position,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )
        )
    tracks = [(5, 10, states)]
    nominal_states = tmp_path / "nominal-states.root"
    misaligned_states = tmp_path / "misaligned-states.root"
    _write_trackstates(nominal_states, tracks)
    _write_trackstates(misaligned_states, tracks)
    return nominal_states, misaligned_states, nominal_geometry, misaligned_geometry


def test_paired_root_loader_preserves_state_ordinal_then_partitions_and_caps_tracks(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, _tracks = (
        _loader_fixture(tmp_path)
    )

    nominal, misaligned = m.load_paired_recovery_inputs(
        nominal_states,
        misaligned_states,
        nominal_geometry,
        misaligned_geometry,
        target_layers=((17, 8),),
        event_partition="validation",
        event_split_seed=6006,
        max_tracks=1,
    )

    assert [(item.event_nr, item.track_nr) for item in nominal.identities] == [(5, 10), (5, 10)]
    assert [item.occurrence for item in nominal.identities] == [0, 1]
    assert nominal.identities == misaligned.identities
    assert nominal.target_mask.tolist() == [False, True]
    assert nominal.sensor_idx.tolist() == [0, 1]
    assert misaligned.sensor_idx.tolist() == [1, 0]


def test_paired_root_loader_refuses_measurement_mismatch_before_partition_filtering(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, tracks = (
        _loader_fixture(tmp_path)
    )
    # The mismatch is in calibration event 2 while validation is requested.
    _write_trackstates(misaligned_states, tracks, local_x_override=(2, 1, 999.0))

    with pytest.raises(ValueError, match="measurement coordinates"):
        m.load_paired_recovery_inputs(
            nominal_states,
            misaligned_states,
            nominal_geometry,
            misaligned_geometry,
            target_layers=((17, 8),),
            event_partition="validation",
        )


def test_paired_root_loader_refuses_swapped_duplicate_surface_measurements(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, tracks = (
        _loader_fixture(tmp_path)
    )
    swapped = list(tracks)
    first = list(swapped[0][2])
    first[0], first[1] = first[1], first[0]
    swapped[0] = (swapped[0][0], swapped[0][1], first)
    _write_trackstates(misaligned_states, swapped)

    with pytest.raises(ValueError, match="measurement coordinates"):
        m.load_paired_recovery_inputs(
            nominal_states,
            misaligned_states,
            nominal_geometry,
            misaligned_geometry,
            target_layers=((17, 8),),
            event_partition="all",
        )


def test_cli_defaults_and_help_follow_project_layout():
    m = load_module()
    config = m.load_project_config()

    parser = m.build_argument_parser(config)
    args = parser.parse_args([])
    help_text = parser.format_help()

    assert args.target_layers == config.target_layers
    assert args.event_partition == "validation"
    assert args.event_split_seed == 6006
    assert args.exposure == "both"
    assert args.max_tracks == 0
    assert args.seed_iters == 200
    assert args.refit_iters == 60
    assert args.bootstrap_replicates == 2000
    assert args.bootstrap_seed == 6006
    assert Path(args.nominal_states).name == f"trackstates_kf_{config.target_layer_suffix}.root"
    assert "/reconstruction/nominal/" in args.nominal_states
    assert "/reconstruction/misaligned/" in args.misaligned_states
    assert Path(args.nominal_geometry) == config.data_root / "geometry" / "odd_transforms.json"
    assert Path(args.misaligned_geometry).name == (
        f"misalignment_applied_{config.target_layer_suffix}.json"
    )
    assert Path(args.output_dir) == (
        config.data_root / "experiments" / "helix_recovery" / config.target_layer_dir
    )
    for option in (
        "--nominal-states",
        "--misaligned-states",
        "--nominal-geometry",
        "--misaligned-geometry",
        "--target-layers",
        "--event-partition",
        "--event-split-seed",
        "--exposure",
        "--max-tracks",
        "--output-dir",
        "--seed-iters",
        "--refit-iters",
        "--bootstrap-replicates",
        "--bootstrap-seed",
        "--gpu",
        "--config",
    ):
        assert option in help_text


def test_cli_writes_selected_exposure_artifacts_and_completed_manifest(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    output_dir = tmp_path / "run"

    result = m.main(
        [
            "--nominal-states", str(nominal_states),
            "--misaligned-states", str(misaligned_states),
            "--nominal-geometry", str(nominal_geometry),
            "--misaligned-geometry", str(misaligned_geometry),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--exposure", "protected",
            "--max-tracks", "1",
            "--output-dir", str(output_dir),
            "--seed-iters", "8",
            "--refit-iters", "8",
            "--bootstrap-replicates", "7",
            "--bootstrap-seed", "23",
        ]
    )

    assert result == 0
    assert {path.name for path in output_dir.iterdir()} == {
        "paired_track_keys.csv",
        "per_track_stage_metrics.csv",
        "helix_recovery_summary.json",
        "seed_vs_refit_paired_difference_protected.png",
        "refit_difference_distribution_protected.png",
        "grouped_recovery_ratios_protected.png",
        "manifest.json",
    }
    summary = json.loads((output_dir / "helix_recovery_summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["bootstrap"] == {"replicates": 7, "seed": 23}
    assert set(summary["exposures"]) == {"protected"}
    assert set(summary["verdicts"]) == {"protected"}
    assert summary["effective_experiment_config"]["max_tracks"] == 1
    assert summary["cohort_counts"]["frozen_probe_rows"] == 1
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert set(manifest["inputs"]) == {
        "nominal_trackstates",
        "misaligned_trackstates",
        "nominal_geometry",
        "misaligned_geometry",
    }
    assert all(record["sha256"] for record in manifest["inputs"].values())
    assert manifest["metrics"]["bootstrap"] == {"replicates": 7, "seed": 23}
    assert set(manifest["metrics"]["verdicts"]) == {"protected"}


def test_cli_finalizes_failed_manifest_when_root_loading_raises(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    nominal_states.write_bytes(b"not a ROOT file")
    output_dir = tmp_path / "failed-run"

    result = m.main(
        [
            "--nominal-states", str(nominal_states),
            "--misaligned-states", str(misaligned_states),
            "--nominal-geometry", str(nominal_geometry),
            "--misaligned-geometry", str(misaligned_geometry),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
            "--seed-iters", "1",
            "--refit-iters", "1",
            "--bootstrap-replicates", "1",
        ]
    )

    assert result == 1
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["failure_reason"]
    assert manifest["inputs"]["nominal_trackstates"]["sha256"]
    assert not (output_dir / "helix_recovery_summary.json").exists()


def _inputs(module, *, reverse=False, target_at=5):
    path = torch.arange(11, dtype=torch.float64) * 100.0
    order = torch.arange(11)
    if reverse:
        order = torch.flip(order, dims=(0,))
    values = path[order]
    identities = tuple(
        module.HitIdentity(7, 11, 10 if int(i) != target_at else 17, int(i), int(i) + 1, 0)
        for i in order.tolist()
    )
    target = torch.tensor([int(i) == target_at for i in order.tolist()])
    return module.RecoveryHitInputs(
        identities=identities,
        measured_local=torch.zeros((11, 2), dtype=torch.float64),
        measurement_variance=torch.ones((11, 2), dtype=torch.float64),
        component_mask=torch.ones((11, 2), dtype=torch.bool),
        smt_local=torch.zeros((11, 2), dtype=torch.float64),
        global_positions=torch.stack((values, torch.zeros_like(values), torch.zeros_like(values)), dim=1),
        directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(11, -1).clone(),
        sensor_idx=order.clone(),
        target_mask=target,
        path_length=values,
    )


def _geometry(module):
    x = torch.arange(11, dtype=torch.float64) * 100.0
    rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    return module.NominalGeometry(
        transforms=[],
        centers=torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1),
        rotations=rotation.expand(11, -1, -1).clone(),
        id_to_idx={},
    )


def _misaligned_geometry_with_different_index_order(module):
    geometry = _geometry(module)
    centers = torch.full((111, 3), 10_000.0, dtype=torch.float64)
    centers[100:111] = geometry.centers
    return module.NominalGeometry(
        transforms=[],
        centers=centers,
        rotations=geometry.rotations[0].expand(111, -1, -1).clone(),
        id_to_idx={},
    )


def test_freeze_paired_cohort_canonicalizes_order_and_keeps_protected_and_exposed_masks():
    m = load_module()

    cohort = m.freeze_paired_cohort(
        _inputs(m), _inputs(m, reverse=True), min_non_target_hits=8, min_non_target_span_mm=300.0
    )

    assert cohort.n_rows == 1
    assert cohort.probe_mode == ("interpolation",)
    assert cohort.protected_fit_mask.sum().item() == 10
    assert cohort.exposed_fit_mask.sum().item() == 11
    probe = cohort.probe_mask.nonzero().tolist()
    assert probe == [[0, 5]]
    assert cohort.protected_fit_mask[0, 5].item() is False
    assert cohort.exposed_fit_mask[0, 5].item() is True
    assert cohort.non_target_span_mm.tolist() == [1000.0]


@pytest.mark.parametrize(
    ("target_at", "mode"),
    ((0, "backward_extrapolation"), (5, "interpolation"), (10, "forward_extrapolation")),
)
def test_probe_classification_uses_the_protected_non_target_window(target_at, mode):
    m = load_module()

    cohort = m.freeze_paired_cohort(_inputs(m, target_at=target_at), _inputs(m, target_at=target_at))

    assert cohort.probe_mode == (mode,)


def test_build_hit_identities_refuses_duplicate_surfaces_without_shared_occurrences():
    m = load_module()
    event = torch.tensor([7, 7])
    track = torch.tensor([11, 11])
    volume = torch.tensor([10, 10])
    layer = torch.tensor([2, 2])
    module = torch.tensor([4, 4])

    with pytest.raises(ValueError, match="stable occurrence"):
        m.build_hit_identities(event, track, volume, layer, module)

    # Reversing the same two states must not create a second, independently
    # ordered occurrence assignment that the paired cohort could accept.
    with pytest.raises(ValueError, match="stable occurrence"):
        m.build_hit_identities(
            event.flip(0), track.flip(0), volume.flip(0), layer.flip(0), module.flip(0)
        )


def test_build_hit_identities_accepts_explicit_shared_duplicate_occurrences():
    m = load_module()
    event = torch.tensor([7, 7])
    track = torch.tensor([11, 11])
    volume = torch.tensor([10, 10])
    layer = torch.tensor([2, 2])
    module = torch.tensor([4, 4])

    nominal = m.build_hit_identities(event, track, volume, layer, module, torch.tensor([4, 9]))
    misaligned = m.build_hit_identities(
        event.flip(0), track.flip(0), volume.flip(0), layer.flip(0), module.flip(0), torch.tensor([9, 4])
    )

    assert sorted(nominal) == sorted(misaligned)


@pytest.mark.parametrize("field", ("measured_local", "measurement_variance"))
def test_freeze_paired_cohort_rejects_nonfinite_available_components(field):
    m = load_module()
    nominal = _inputs(m)
    values = getattr(nominal, field).clone()
    values[0, 0] = torch.nan if field == "measured_local" else torch.inf
    nominal = m.replace(nominal, **{field: values})

    with pytest.raises(ValueError, match="finite available"):
        m.freeze_paired_cohort(nominal, _inputs(m))


@pytest.mark.parametrize("kind", ("identity", "target_mask", "component_mask"))
def test_freeze_paired_cohort_fails_closed_on_unpaired_contract(kind):
    m = load_module()
    nominal = _inputs(m)
    misaligned = _inputs(m)
    if kind == "identity":
        identities = list(misaligned.identities)
        identities[0] = m.HitIdentity(99, 11, 10, 0, 1, 0)
        misaligned = m.replace(misaligned, identities=tuple(identities))
    elif kind == "target_mask":
        mask = misaligned.target_mask.clone()
        mask[0] = True
        misaligned = m.replace(misaligned, target_mask=mask)
    else:
        mask = misaligned.component_mask.clone()
        mask[0, 1] = False
        misaligned = m.replace(misaligned, component_mask=mask)

    with pytest.raises(ValueError, match="paired"):
        m.freeze_paired_cohort(nominal, misaligned)


def test_four_arm_runner_reuses_shared_stages_and_evaluates_on_nominal_surfaces():
    m = load_module()
    nominal = _inputs(m)
    misaligned = _inputs(m, reverse=True)
    geometry = _geometry(m)
    cohort = m.freeze_paired_cohort(nominal, misaligned)

    report = m.run_four_arm_recovery(
        cohort,
        nominal_geometry=geometry,
        misaligned_geometry=geometry,
        seed_iterations=8,
        refit_iterations=8,
    )

    assert set(report.arms) == {
        ("nominal", "protected"),
        ("nominal", "exposed"),
        ("misaligned", "protected"),
        ("misaligned", "exposed"),
    }
    protected = report.arms[("nominal", "protected")]
    exposed = report.arms[("nominal", "exposed")]
    assert protected.fit_mask[0, 5].item() is False
    assert exposed.fit_mask[0, 5].item() is True
    for arm in report.arms.values():
        assert arm.acts_smt.local_position.shape == (1, 11, 2)
        assert arm.lm_seed.intersection_valid.shape == (1, 11)
        assert arm.gauss_newton.intersection_valid.shape == (1, 11)
        assert arm.lm_seed.intersection_valid.all()
        assert arm.gauss_newton.intersection_valid.all()


def test_nominal_surface_evaluation_retains_invalid_intersections():
    m = load_module()
    cohort = m.freeze_paired_cohort(_inputs(m), _inputs(m))
    parameters = torch.tensor([[0.0, 0.0, 0.0, torch.pi / 2, 0.0, 0.0]], dtype=torch.float64)

    evaluation = m.evaluate_fitted_stage_on_nominal_surfaces(parameters, cohort, _geometry(m))

    assert evaluation.intersection_valid.shape == (1, 11)
    assert not evaluation.intersection_valid.any()


def test_misaligned_refit_uses_its_own_sensor_lookup_then_evaluates_nominal_surfaces():
    m = load_module()
    nominal = _inputs(m)
    misaligned = _inputs(m, reverse=True)
    misaligned = m.replace(misaligned, sensor_idx=misaligned.sensor_idx + 100)
    cohort = m.freeze_paired_cohort(nominal, misaligned)

    report = m.run_four_arm_recovery(
        cohort,
        nominal_geometry=_geometry(m),
        misaligned_geometry=_misaligned_geometry_with_different_index_order(m),
        seed_iterations=4,
        refit_iterations=4,
    )

    arm = report.arms[("misaligned", "protected")]
    assert arm.gauss_newton.intersection_valid.all()


def _four_arm_statistics_fixture(
    module,
    *,
    n_rows=201,
    invalidate_seed_row=False,
    modes=("interpolation", "forward_extrapolation", "backward_extrapolation"),
    unretained_target_layer=None,
):
    """Hand-checked paired predictions: Q=.25, practical ratio=1.05, bias=.05."""
    n_rows *= len(modes)
    identities = tuple(
        module.HitIdentity(row // 3, row, 17, 8, row + 1, 0)
        for row in range(n_rows)
    )
    inputs = module.RecoveryHitInputs(
        identities=identities,
        measured_local=torch.zeros((n_rows, 2), dtype=torch.float64),
        measurement_variance=torch.ones((n_rows, 2), dtype=torch.float64),
        component_mask=torch.ones((n_rows, 2), dtype=torch.bool),
        smt_local=torch.zeros((n_rows, 2), dtype=torch.float64),
        global_positions=torch.zeros((n_rows, 3), dtype=torch.float64),
        directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(n_rows, -1).clone(),
        sensor_idx=torch.arange(n_rows),
        target_mask=torch.ones(n_rows, dtype=torch.bool),
        path_length=torch.arange(n_rows, dtype=torch.float64),
    )
    source_inputs = inputs
    if unretained_target_layer is not None:
        volume, layer = unretained_target_layer
        source_inputs = module.replace(
            inputs,
            identities=inputs.identities
            + (module.HitIdentity(999, 999, volume, layer, 999, 0),),
            measured_local=torch.cat((inputs.measured_local, torch.zeros((1, 2), dtype=torch.float64))),
            measurement_variance=torch.cat((inputs.measurement_variance, torch.ones((1, 2), dtype=torch.float64))),
            component_mask=torch.cat((inputs.component_mask, torch.ones((1, 2), dtype=torch.bool))),
            smt_local=torch.cat((inputs.smt_local, torch.zeros((1, 2), dtype=torch.float64))),
            global_positions=torch.cat((inputs.global_positions, torch.zeros((1, 3), dtype=torch.float64))),
            directions=torch.cat((inputs.directions, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))),
            sensor_idx=torch.cat((inputs.sensor_idx, torch.tensor([999], dtype=torch.long))),
            target_mask=torch.cat((inputs.target_mask, torch.tensor([True]))),
            path_length=torch.cat((inputs.path_length, torch.tensor([999.0], dtype=torch.float64))),
        )
    cohort = module.PairedRecoveryCohort(
        nominal=source_inputs,
        misaligned=source_inputs,
        row_indices=torch.arange(n_rows, dtype=torch.long).reshape(n_rows, 1),
        hit_mask=torch.ones((n_rows, 1), dtype=torch.bool),
        probe_mask=torch.ones((n_rows, 1), dtype=torch.bool),
        protected_fit_mask=torch.zeros((n_rows, 1), dtype=torch.bool),
        exposed_fit_mask=torch.ones((n_rows, 1), dtype=torch.bool),
        non_target_span_mm=torch.full((n_rows,), 300.0, dtype=torch.float64),
        probe_mode=tuple(mode for mode in modes for _ in range(n_rows // len(modes))),
    )

    def stage(value, *, valid=True, rho=0.0, angle=0.0, diagnostics=None):
        local = torch.full((n_rows, 1, 2), value, dtype=torch.float64)
        intersection = torch.full((n_rows, 1), valid, dtype=torch.bool)
        return module.FittedStage(
            parameters=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, rho], dtype=torch.float64).repeat(n_rows, 1),
            evaluation=module.HelixEvaluation(
                path_length=torch.zeros((n_rows, 1), dtype=torch.float64),
                global_position=torch.zeros((n_rows, 1, 3), dtype=torch.float64),
                local_position=local,
                global_difference=torch.zeros((n_rows, 1, 3), dtype=torch.float64),
                local_difference=torch.zeros((n_rows, 1, 2), dtype=torch.float64),
                tangent_angle=torch.full((n_rows, 1), angle, dtype=torch.float64),
                intersection_valid=intersection,
            ),
            diagnostics=diagnostics
            or {
                "non_stationary_row_fraction": 0.0,
                "n_non_finite": 0,
                "stationarity": {
                    "non_stationary_row_fraction": 0.0,
                    "n_non_finite": 0,
                },
            },
        )

    nominal_seed = stage(1.0)
    if invalidate_seed_row:
        invalid = nominal_seed.evaluation.intersection_valid.clone()
        invalid[0, 0] = False
        nominal_seed = module.replace(
            nominal_seed,
            evaluation=module.replace(nominal_seed.evaluation, intersection_valid=invalid),
        )
    arms = {}
    for exposure in ("protected", "exposed"):
        arms[("nominal", exposure)] = module.RecoveryArm(
            fit_mask=cohort.protected_fit_mask,
            acts_smt=module.ACTSSMTStage(
                inputs.smt_local[:, None], inputs.component_mask[:, None], cohort.hit_mask
            ),
            lm_seed=nominal_seed,
            gauss_newton=stage(1.0, rho=0.001, angle=0.01),
        )
        arms[("misaligned", exposure)] = module.RecoveryArm(
            fit_mask=cohort.protected_fit_mask,
            acts_smt=module.ACTSSMTStage(
                inputs.smt_local[:, None], inputs.component_mask[:, None], cohort.hit_mask
            ),
            lm_seed=stage(3.0),
            gauss_newton=stage(1.05, rho=0.002, angle=0.02),
        )
    return module.FourArmRecovery(cohort=cohort, arms=arms)


def test_recovery_statistics_use_one_shared_mask_and_clustered_bootstrap_without_global_rng_mutation():
    m = load_module()
    report = _four_arm_statistics_fixture(m, invalidate_seed_row=True)
    before = np.random.get_state()

    result = m.summarize_four_arm_recovery(report)

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    group = result["exposures"]["protected"]["groups"]["17:8|interpolation|LOC0"]
    assert group["requested_count"] == 201
    assert group["valid_count"] == 200
    assert group["seed_branch_difference_rmse_mm"] == pytest.approx(2.0)
    assert group["refit_branch_difference_rmse_mm"] == pytest.approx(0.05)
    assert group["recovery_ratio_q"] == pytest.approx(0.025)
    assert group["recovery_ratio_q_ci95"]["low"] == pytest.approx(0.025)
    assert group["recovery_ratio_q_ci95"]["high"] == pytest.approx(0.025)
    assert group["recovery_ratio_q_ci95"]["valid"] is True
    assert group["practical_equivalence_ratio"] == pytest.approx(1.05)
    assert group["practical_equivalence_ratio_ci95"]["low"] == pytest.approx(1.05)
    assert group["practical_equivalence_ratio_ci95"]["high"] == pytest.approx(1.05)
    assert group["refit_branch_difference_bias_mm_ci95"]["low"] == pytest.approx(0.05)
    assert group["refit_branch_difference_bias_mm_ci95"]["high"] == pytest.approx(0.05)
    assert group["direction_angle_branch_difference_rms_rad"] == pytest.approx(0.0)
    assert group["rho_branch_difference_rms_per_mm"] == pytest.approx(0.001)
    assert group["intersection_failures"]["nominal_seed"]["count"] == 1
    assert group["verdict"] == "passed"
    assert result["exposures"]["protected"]["verdict"] == "passed"
    assert len(result["per_row"]) == 603
    assert result["per_row"][0]["shared_valid_LOC0"] is False


def test_recovery_statistics_are_inconclusive_when_a_group_has_fewer_than_200_shared_components():
    m = load_module()

    result = m.summarize_four_arm_recovery(_four_arm_statistics_fixture(m, n_rows=199))

    group = result["exposures"]["exposed"]["groups"]["17:8|interpolation|LOC1"]
    assert group["valid_count"] == 199
    assert group["verdict"] == "inconclusive"
    assert result["exposures"]["exposed"]["verdict"] == "inconclusive"


def test_recovery_statistics_fail_closed_when_nested_refit_stationarity_is_missing():
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    arm = report.arms[("nominal", "protected")]
    report.arms[("nominal", "protected")] = m.replace(
        arm,
        gauss_newton=m.replace(arm.gauss_newton, diagnostics={}),
    )

    result = m.summarize_four_arm_recovery(report)

    numerical = result["numerical_validity"]["nominal|protected|refit"]
    assert numerical["valid"] is False
    assert numerical["reason"] == "missing stationarity observability"
    assert result["exposures"]["protected"]["verdict"] == "failed"


def test_recovery_statistics_report_an_absent_prediction_mode_as_an_underpowered_group():
    m = load_module()

    result = m.summarize_four_arm_recovery(
        _four_arm_statistics_fixture(m, modes=("interpolation",))
    )

    group = result["exposures"]["protected"]["groups"]["17:8|forward_extrapolation|LOC0"]
    assert group["requested_count"] == 0
    assert group["valid_count"] == 0
    assert group["verdict"] == "inconclusive"
    assert result["exposures"]["protected"]["verdict"] == "inconclusive"


def test_recovery_statistics_leave_zero_denominator_ratios_undefined():
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    for exposure in ("protected", "exposed"):
        arm = report.arms[("misaligned", exposure)]
        report.arms[("misaligned", exposure)] = m.replace(
            arm,
            lm_seed=m.replace(
                arm.lm_seed,
                evaluation=m.replace(
                    arm.lm_seed.evaluation,
                    local_position=torch.ones_like(arm.lm_seed.evaluation.local_position),
                ),
            ),
        )

    result = m.summarize_four_arm_recovery(report)

    group = result["exposures"]["protected"]["groups"]["17:8|interpolation|LOC0"]
    assert group["seed_branch_difference_rmse_mm"] == 0.0
    assert group["recovery_ratio_q"] is None
    assert group["recovery_ratio_q_ci95"]["low"] is None
    assert group["recovery_ratio_q_ci95"]["high"] is None
    assert group["recovery_ratio_q_ci95"]["valid"] is False
    assert group["verdict"] == "failed"


def test_recovery_statistics_keep_a_target_layer_with_no_retained_probe_inconclusive():
    m = load_module()

    result = m.summarize_four_arm_recovery(
        _four_arm_statistics_fixture(m, unretained_target_layer=(24, 4))
    )

    group = result["exposures"]["protected"]["groups"]["24:4|interpolation|LOC0"]
    assert group["requested_count"] == 0
    assert group["valid_count"] == 0
    assert group["intersection_failures"]["nominal_seed"]["rate"] is None
    assert group["verdict"] == "inconclusive"
    assert result["exposures"]["protected"]["verdict"] == "inconclusive"


def test_direction_difference_uses_fitted_tangents_not_subtracted_smt_error_angles():
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    for exposure in ("protected", "exposed"):
        nominal = report.arms[("nominal", exposure)]
        misaligned = report.arms[("misaligned", exposure)]
        nominal_parameters = nominal.gauss_newton.parameters.clone()
        misaligned_parameters = misaligned.gauss_newton.parameters.clone()
        nominal_parameters[:, 3] = 0.0
        misaligned_parameters[:, 3] = torch.pi
        report.arms[("nominal", exposure)] = m.replace(
            nominal, gauss_newton=m.replace(nominal.gauss_newton, parameters=nominal_parameters)
        )
        report.arms[("misaligned", exposure)] = m.replace(
            misaligned, gauss_newton=m.replace(misaligned.gauss_newton, parameters=misaligned_parameters)
        )

    result = m.summarize_four_arm_recovery(report)

    group = result["exposures"]["protected"]["groups"]["17:8|interpolation|LOC0"]
    assert group["direction_angle_branch_difference_rms_rad"] == pytest.approx(float(torch.pi))


def test_cluster_bootstrap_resamples_whole_nonconstant_multirow_events_not_individual_rows():
    m = load_module()
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    events = np.array([10, 10, 10, 10, 10, 11])

    cluster = m._cluster_bootstrap_ci(
        values,
        np.zeros_like(values),
        events,
        lambda total, _unused, count: total / count,
        np.random.default_rng(6006),
    )
    row_rng = np.random.default_rng(6006)
    draws = row_rng.integers(0, len(values), size=(m.BOOTSTRAP_REPLICATES, len(values)))
    row_high = float(np.percentile(values[draws].mean(axis=1), 97.5))

    assert cluster["valid"] is True
    assert cluster["high"] == pytest.approx(1.0)
    assert cluster["high"] > row_high


def test_bootstrap_refuses_a_gate_when_some_event_resamples_have_zero_denominator():
    m = load_module()
    values = np.array([0.0, 0.0, 2.0])
    events = np.array([10, 10, 11])

    ci = m._cluster_bootstrap_ci(
        np.square(values),
        np.square(values),
        events,
        lambda first, second, count: np.sqrt(second / count) / np.sqrt(first / count),
        np.random.default_rng(6006),
    )

    assert ci["valid"] is False
    assert ci["n_requested_resamples"] == 2000
    assert 0 < ci["n_finite_resamples"] < 2000
    assert ci["low"] is None
    assert ci["high"] is None


def test_recovery_statistics_honor_bootstrap_overrides_and_emit_only_requested_exposure():
    m = load_module()

    result = m.summarize_four_arm_recovery(
        _four_arm_statistics_fixture(m),
        bootstrap_replicates=17,
        bootstrap_seed=23,
        exposures=("protected",),
    )

    assert result["bootstrap"] == {"replicates": 17, "seed": 23}
    assert set(result["exposures"]) == {"protected"}
    assert all("|protected|" in key for key in result["numerical_validity"])
    interval = result["exposures"]["protected"]["groups"][
        "17:8|interpolation|LOC0"
    ]["recovery_ratio_q_ci95"]
    assert interval["n_requested_resamples"] == 17
    assert "shared_valid_exposed_LOC0" not in result["per_row"][0]


def test_recovery_statistics_reject_duplicate_or_unknown_exposure_selection():
    m = load_module()
    report = _four_arm_statistics_fixture(m)

    with pytest.raises(ValueError, match="exposures"):
        m.summarize_four_arm_recovery(report, exposures=("protected", "protected"))
    with pytest.raises(ValueError, match="exposures"):
        m.summarize_four_arm_recovery(report, exposures=("hidden",))


@pytest.mark.parametrize(
    ("misaligned_refit", "expected_practical"),
    (
        (1.10, True),
        (np.nextafter(1.10, np.inf), False),
    ),
)
def test_recovery_statistics_practical_gate_uses_exact_inclusive_boundary(
    misaligned_refit, expected_practical
):
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    for exposure in ("protected", "exposed"):
        arm = report.arms[("misaligned", exposure)]
        report.arms[("misaligned", exposure)] = m.replace(
            arm,
            gauss_newton=m.replace(
                arm.gauss_newton,
                evaluation=m.replace(
                    arm.gauss_newton.evaluation,
                    local_position=torch.full_like(
                        arm.gauss_newton.evaluation.local_position, misaligned_refit
                    ),
                ),
            ),
        )

    result = m.summarize_four_arm_recovery(report)

    group = result["exposures"]["protected"]["groups"]["17:8|interpolation|LOC0"]
    assert group["gates"]["practical_equivalence"] is expected_practical


@pytest.mark.parametrize(
    ("misaligned_refit", "expected_bias"),
    (
        (1.375, True),
        (np.nextafter(1.375, np.inf), False),
    ),
)
def test_recovery_statistics_bias_gate_uses_exact_inclusive_margin(
    misaligned_refit, expected_bias
):
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    for exposure in ("protected", "exposed"):
        arm = report.arms[("nominal", exposure)]
        report.arms[("nominal", exposure)] = m.replace(
            arm,
            gauss_newton=m.replace(
                arm.gauss_newton,
                evaluation=m.replace(
                    arm.gauss_newton.evaluation,
                    local_position=torch.full_like(
                        arm.gauss_newton.evaluation.local_position, 1.25
                    ),
                ),
            ),
        )
        arm = report.arms[("misaligned", exposure)]
        report.arms[("misaligned", exposure)] = m.replace(
            arm,
            gauss_newton=m.replace(
                arm.gauss_newton,
                evaluation=m.replace(
                    arm.gauss_newton.evaluation,
                    local_position=torch.full_like(
                        arm.gauss_newton.evaluation.local_position, misaligned_refit
                    ),
                ),
            ),
        )

    result = m.summarize_four_arm_recovery(report)

    group = result["exposures"]["protected"]["groups"]["17:8|interpolation|LOC0"]
    assert group["gates"]["bias"] is expected_bias


def test_recovery_statistics_restoration_requires_q_strictly_below_one():
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    for exposure in ("protected", "exposed"):
        arm = report.arms[("misaligned", exposure)]
        report.arms[("misaligned", exposure)] = m.replace(
            arm,
            gauss_newton=m.replace(
                arm.gauss_newton,
                evaluation=m.replace(
                    arm.gauss_newton.evaluation,
                    local_position=torch.full_like(arm.gauss_newton.evaluation.local_position, 3.0),
                ),
            ),
        )

    result = m.summarize_four_arm_recovery(report)

    group = result["exposures"]["protected"]["groups"]["17:8|interpolation|LOC0"]
    assert group["recovery_ratio_q_ci95"]["high"] == pytest.approx(1.0)
    assert group["gates"]["restoration"] is False
