from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from dataclasses import replace
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


def _write_trackstates(path, tracks, *, local_x_override=None, err_x_override=None):
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
    if err_x_override is not None:
        entry, measurement, value = err_x_override
        per_state["err_x_hit"][entry][measurement] = value
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
        (
            5,
            20,
            [
                state(10, 1, 1, 0.0, 1.0),
                state(10, 1, 1, 0.5, np.nan, np.nan),
                state(10, 1, 1, 1.0, 2.0),
            ],
        ),
        (5, 10, [state(10, 1, 1, 0.0, 3.0), state(17, 8, 2, 1.0, 4.0)]),
        (2, 30, [state(10, 1, 1, 0.0, 5.0), state(17, 8, 2, 1.0, 6.0)]),
    ]
    nominal_states = tmp_path / "nominal.root"
    misaligned_states = tmp_path / "misaligned.root"
    _write_trackstates(nominal_states, tracks)
    _write_trackstates(misaligned_states, tracks)
    return nominal_states, misaligned_states, geometry_nominal, geometry_misaligned, tracks


def _cli_fixture(tmp_path, *, additional_layer_event=None):
    surfaces = [(10, index + 1, index + 1) for index in range(11)]
    surfaces[5] = (17, 8, 6)
    if additional_layer_event is not None:
        surfaces.append((24, 4, 12))
    nominal_geometry = tmp_path / "nominal-geometry.json"
    misaligned_geometry = tmp_path / "misaligned-geometry.json"
    _write_geometry(nominal_geometry, surfaces)
    _write_geometry(misaligned_geometry, surfaces)
    states = []
    for index, (volume, layer, module) in enumerate(surfaces[:11]):
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
    if additional_layer_event is not None:
        additional_states = list(states)
        additional_states[5] = (24, 4, 12, *states[5][3:])
        tracks.append((additional_layer_event, 10, additional_states))
    nominal_states = tmp_path / "nominal-states.root"
    misaligned_states = tmp_path / "misaligned-states.root"
    _write_trackstates(nominal_states, tracks)
    _write_trackstates(misaligned_states, tracks)
    return nominal_states, misaligned_states, nominal_geometry, misaligned_geometry


def test_recovery_root_opener_uses_memmap_for_existing_local_file(tmp_path, monkeypatch):
    m = load_module()
    local_path = tmp_path / "states.root"
    local_path.touch()
    sentinel = object()
    calls = []

    def open_root(path, **options):
        calls.append((path, options))
        return sentinel

    monkeypatch.setattr(m.uproot, "open", open_root)

    assert m._open_recovery_root(local_path) is sentinel
    assert calls == [(local_path, {"handler": m.uproot.MemmapSource})]


@pytest.mark.parametrize("source", ("https://example.invalid/states.root", io.BytesIO(b"root")))
def test_recovery_root_opener_preserves_default_source_for_nonlocal_inputs(source, monkeypatch):
    m = load_module()
    sentinel = object()
    calls = []

    def open_root(path, **options):
        calls.append((path, options))
        return sentinel

    monkeypatch.setattr(m.uproot, "open", open_root)

    assert m._open_recovery_root(source) is sentinel
    assert len(calls) == 1
    assert calls[0][0] is source
    assert calls[0][1] == {}


def test_paired_root_loader_opens_both_branches_through_shared_helper(tmp_path, monkeypatch):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, _tracks = (
        _loader_fixture(tmp_path)
    )
    original_open = m.uproot.open
    opened = []

    def open_recovery_root(path):
        opened.append(path)
        return original_open(path, handler=m.uproot.MemmapSource)

    monkeypatch.setattr(m, "_open_recovery_root", open_recovery_root, raising=False)

    m.load_paired_recovery_inputs(
        nominal_states,
        misaligned_states,
        nominal_geometry,
        misaligned_geometry,
        target_layers=((17, 8),),
    )

    assert opened == [nominal_states, misaligned_states]


def test_paired_root_loader_preserves_state_ordinal_then_caps_tracks(tmp_path):
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
        event_partition="all",
        max_events=2,
        max_tracks=2,
    )

    assert [(item.event_nr, item.track_nr) for item in nominal.identities] == [
        (2, 30), (2, 30), (5, 10), (5, 10)
    ]
    assert [item.occurrence for item in nominal.identities] == [0, 1, 0, 1]
    assert nominal.identities == misaligned.identities
    assert nominal.target_mask.tolist() == [False, True, False, True]
    assert nominal.sensor_idx.tolist() == [0, 1, 0, 1]
    assert misaligned.sensor_idx.tolist() == [1, 0, 1, 0]


def test_paired_root_loader_selects_complete_lowest_unique_events_without_hashing(tmp_path):
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
        event_partition="all",
        max_events=1,
    )

    assert {identity.event_nr for identity in nominal.identities} == {2}
    assert nominal.identities == misaligned.identities
    assert nominal.available_unique_events == 2
    assert nominal.selected_unique_events == 1
    assert len(nominal.identities) == 2


def test_paired_root_loader_rejects_hash_partitions(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, _tracks = (
        _loader_fixture(tmp_path)
    )

    with pytest.raises(ValueError, match="hash partition"):
        m.load_paired_recovery_inputs(
            nominal_states,
            misaligned_states,
            nominal_geometry,
            misaligned_geometry,
            target_layers=((17, 8),),
            event_partition="validation",
            max_events=1,
        )


def test_paired_root_loader_occurrence_is_measured_state_ordinal(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, _tracks = (
        _loader_fixture(tmp_path)
    )

    nominal, _misaligned = m.load_paired_recovery_inputs(
        nominal_states,
        misaligned_states,
        nominal_geometry,
        misaligned_geometry,
        target_layers=((17, 8),),
        event_partition="all",
    )

    duplicate_states = [
        identity for identity in nominal.identities
        if (identity.event_nr, identity.track_nr) == (5, 20)
    ]
    assert [identity.occurrence for identity in duplicate_states] == [0, 1]


def test_paired_root_loader_ignores_branch_local_nonmeasurement_state_positions(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, tracks = (
        _loader_fixture(tmp_path)
    )
    shifted = list(tracks)
    event, track, states = shifted[0]
    shifted[0] = (event, track, [states[1], states[0], states[2]])
    _write_trackstates(misaligned_states, shifted)

    nominal, misaligned = m.load_paired_recovery_inputs(
        nominal_states,
        misaligned_states,
        nominal_geometry,
        misaligned_geometry,
        target_layers=((17, 8),),
        event_partition="all",
    )

    assert nominal.identities == misaligned.identities
    shifted_track = [
        identity for identity in nominal.identities
        if (identity.event_nr, identity.track_nr) == (5, 20)
    ]
    assert [identity.occurrence for identity in shifted_track] == [0, 1]


def test_paired_root_loader_refuses_measurement_mismatch_before_partition_filtering(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, tracks = (
        _loader_fixture(tmp_path)
    )
    # The mismatch is in event 5, outside the one-event selection window.
    _write_trackstates(misaligned_states, tracks, local_x_override=(0, 0, 999.0))

    with pytest.raises(ValueError, match="measurement coordinates"):
        m.load_paired_recovery_inputs(
            nominal_states,
            misaligned_states,
            nominal_geometry,
            misaligned_geometry,
            target_layers=((17, 8),),
            event_partition="all",
            max_events=1,
        )


def test_paired_root_loader_refuses_swapped_duplicate_surface_measurements(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, tracks = (
        _loader_fixture(tmp_path)
    )
    swapped = list(tracks)
    first = list(swapped[0][2])
    first[0], first[2] = first[2], first[0]
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


def test_paired_root_loader_refuses_cross_branch_variance_mismatch(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry, tracks = (
        _loader_fixture(tmp_path)
    )
    _write_trackstates(misaligned_states, tracks, err_x_override=(1, 0, 0.123))

    with pytest.raises(ValueError, match="measurement uncertainties"):
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
    assert args.event_partition == "all"
    assert args.max_events == 20000
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
        "--max-events",
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
    def reject_nonfinite(value):
        raise AssertionError(f"non-finite JSON token: {value}")

    summary = json.loads(
        (output_dir / "helix_recovery_summary.json").read_text(),
        parse_constant=reject_nonfinite,
    )
    assert summary["schema_version"] == 1
    assert summary["bootstrap"] == {"replicates": 7, "seed": 23}
    assert set(summary["exposures"]) == {"protected"}
    assert set(summary["verdicts"]) == {"protected"}
    assert summary["effective_experiment_config"]["max_tracks"] == 1
    assert summary["cohort_counts"]["frozen_probe_rows"] == 1
    with (output_dir / "paired_track_keys.csv").open(newline="", encoding="utf-8") as stream:
        key_rows = list(csv.DictReader(stream))
    assert len(key_rows) == 1
    assert key_rows[0]["prediction_mode"] == "interpolation"
    assert key_rows[0]["occurrence"] == "5"
    with (output_dir / "per_track_stage_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        metric_rows = list(csv.DictReader(stream))
    assert len(metric_rows) == 1
    assert "protected_nominal_seed_LOC0_mm" in metric_rows[0]
    for png_path in output_dir.glob("*.png"):
        content = png_path.read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(content) > 100
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


@pytest.mark.parametrize(
    ("additional_layer_event", "max_events", "max_tracks"),
    ((None, 0, 0), (6, 1, 0), (6, 0, 1)),
    ids=("absent-from-root", "removed-by-event-cap", "removed-by-track-cap"),
)
def test_cli_keeps_requested_layers_missing_after_input_selection_inconclusive(
    tmp_path, additional_layer_event, max_events, max_tracks
):
    m = load_module()
    paths = _cli_fixture(tmp_path, additional_layer_event=additional_layer_event)
    output_dir = tmp_path / "missing-layer-run"
    result = m.main(
        [
            "--nominal-states", str(paths[0]),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--target-layers", "17:8,24:4",
            "--event-partition", "all",
            "--max-events", str(max_events),
            "--max-tracks", str(max_tracks),
            "--output-dir", str(output_dir),
            "--seed-iters", "8",
            "--refit-iters", "8",
            "--bootstrap-replicates", "7",
        ]
    )

    assert result == 0
    summary = json.loads((output_dir / "helix_recovery_summary.json").read_text())
    assert summary["cohort_counts"]["frozen_probe_rows"] == 1
    for exposure in ("protected", "exposed"):
        groups = summary["exposures"][exposure]["groups"]
        for mode in ("interpolation", "forward_extrapolation", "backward_extrapolation"):
            for component in ("LOC0", "LOC1"):
                key = f"24:4|{mode}|{component}"
                assert key in groups
                assert groups[key]["requested_count"] == groups[key]["valid_count"] == 0
                assert groups[key]["verdict"] == "inconclusive"
        assert summary["verdicts"][exposure] == "inconclusive"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["verdicts"] == summary["verdicts"]


@pytest.mark.parametrize(
    "input_index", range(4),
    ids=("nominal-root", "misaligned-root", "nominal-geometry", "misaligned-geometry"),
)
@pytest.mark.parametrize(
    "artifact_name",
    (
        "manifest.json",
        "paired_track_keys.csv",
        "per_track_stage_metrics.csv",
        "helix_recovery_summary.json",
        "seed_vs_refit_paired_difference_protected.png",
        "refit_difference_distribution_protected.png",
        "grouped_recovery_ratios_protected.png",
        "seed_vs_refit_paired_difference_exposed.png",
        "refit_difference_distribution_exposed.png",
        "grouped_recovery_ratios_exposed.png",
    ),
)
def test_cli_rejects_input_output_collisions_before_any_artifact_write(
    tmp_path, capsys, input_index, artifact_name
):
    m = load_module()
    paths = list(_cli_fixture(tmp_path))
    output_dir = tmp_path / "collision-run"
    output_dir.mkdir()
    source = output_dir / artifact_name
    source_bytes = paths[input_index].read_bytes()
    source.write_bytes(source_bytes)
    paths[input_index] = source

    result = m.main(
        [
            "--nominal-states", str(paths[0]),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
            "--seed-iters", "8",
            "--refit-iters", "8",
            "--bootstrap-replicates", "7",
        ]
    )

    assert source.read_bytes() == source_bytes
    assert {path.name for path in output_dir.iterdir()} == {artifact_name}
    assert result == 1
    assert "collision" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    ("artifact_name", "alias_kind"),
    (
        ("helix_recovery_summary.json", "input"),
        ("manifest.json", "output"),
        ("grouped_recovery_ratios_protected.png", "output"),
    ),
)
def test_cli_rejects_resolved_symlink_collisions_without_changing_sources(
    tmp_path, capsys, artifact_name, alias_kind
):
    m = load_module()
    paths = list(_cli_fixture(tmp_path))
    output_dir = tmp_path / "alias-run"
    output_dir.mkdir()
    source_bytes = paths[2].read_bytes()
    artifact = output_dir / artifact_name
    if alias_kind == "input":
        artifact.write_bytes(source_bytes)
        alias = tmp_path / "nominal-alias.json"
        alias.symlink_to(artifact)
        paths[2] = alias
    else:
        artifact.symlink_to(paths[2])

    result = m.main(
        [
            "--nominal-states", str(paths[0]),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
            "--seed-iters", "8",
            "--refit-iters", "8",
            "--bootstrap-replicates", "7",
        ]
    )

    assert paths[2].read_bytes() == artifact.read_bytes() == source_bytes
    assert {path.name for path in output_dir.iterdir()} == {artifact_name}
    assert (paths[2] if alias_kind == "input" else artifact).is_symlink()
    assert result == 1
    assert "collision" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("artifact_name", ("manifest.json", "helix_recovery_summary.json"))
def test_cli_rejects_config_output_collisions_before_writing(tmp_path, capsys, artifact_name):
    m = load_module()
    paths = _cli_fixture(tmp_path)
    output_dir = tmp_path / "config-collision-run"
    output_dir.mkdir()
    config_path = output_dir / artifact_name
    source_bytes = b'[alignment]\ntarget_layers = ["17:8"]\n'
    config_path.write_bytes(source_bytes)

    result = m.main(
        [
            "--config", str(config_path),
            "--nominal-states", str(paths[0]),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--event-partition", "all",
            "--output-dir", str(output_dir),
            "--seed-iters", "8",
            "--refit-iters", "8",
            "--bootstrap-replicates", "7",
        ]
    )

    assert config_path.read_bytes() == source_bytes
    assert {path.name for path in output_dir.iterdir()} == {artifact_name}
    assert result == 1
    assert "collision" in capsys.readouterr().err.lower()


def test_cli_checks_every_configuration_source_for_output_collisions(
    tmp_path, capsys, monkeypatch
):
    m = load_module()
    paths = _cli_fixture(tmp_path)
    output_dir = tmp_path / "layered-config-collision-run"
    output_dir.mkdir()
    hidden_source = output_dir / "manifest.json"
    source_bytes = b'[alignment]\ntarget_layers = ["17:8"]\n'
    hidden_source.write_bytes(source_bytes)
    load_config = m.load_project_config

    def layered_config(path=None):
        config = load_config(path)
        return replace(
            config,
            source_paths=(config.source_path, hidden_source),
        )

    monkeypatch.setattr(m, "load_project_config", layered_config)
    result = m.main(
        [
            "--nominal-states", str(paths[0]),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
        ]
    )

    assert result == 1
    assert hidden_source.read_bytes() == source_bytes
    assert {path.name for path in output_dir.iterdir()} == {"manifest.json"}
    assert "collision" in capsys.readouterr().err.lower()


def test_cli_rejects_missing_input_colliding_with_future_manifest(tmp_path, capsys):
    m = load_module()
    paths = _cli_fixture(tmp_path)
    output_dir = tmp_path / "missing-collision-run"

    result = m.main(
        [
            "--nominal-states", str(output_dir / "manifest.json"),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
        ]
    )

    assert not output_dir.exists()
    assert result == 1
    assert "collision" in capsys.readouterr().err.lower()


def test_cli_reports_empty_requested_cohort_as_inconclusive_without_fitting(
    tmp_path, monkeypatch
):
    m = load_module()
    paths = _cli_fixture(tmp_path)
    output_dir = tmp_path / "empty-run"

    def refuse_empty_fit(*_args, **_kwargs):
        pytest.fail("An empty cohort must not invoke a fit solver")

    monkeypatch.setattr(m, "seed_helices", refuse_empty_fit)
    monkeypatch.setattr(m, "fit_helices_gauss_newton", refuse_empty_fit)
    result = m.main(
        [
            "--nominal-states", str(paths[0]),
            "--misaligned-states", str(paths[1]),
            "--nominal-geometry", str(paths[2]),
            "--misaligned-geometry", str(paths[3]),
            "--target-layers", "24:4",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
            "--bootstrap-replicates", "7",
        ]
    )

    assert result == 0
    summary = json.loads((output_dir / "helix_recovery_summary.json").read_text())
    assert summary["cohort_counts"]["frozen_probe_rows"] == 0
    for exposure in ("protected", "exposed"):
        groups = summary["exposures"][exposure]["groups"]
        assert len(groups) == 6
        for mode in ("interpolation", "forward_extrapolation", "backward_extrapolation"):
            for component in ("LOC0", "LOC1"):
                group = groups[f"24:4|{mode}|{component}"]
                assert group["requested_count"] == group["valid_count"] == 0
                assert group["recovery_ratio_q"] is None
                assert group["verdict"] == "inconclusive"
        assert summary["verdicts"][exposure] == "inconclusive"
    assert all(
        stage["valid"] is None and "not run" in stage["reason"]
        for stage in summary["numerical_validity"].values()
    )
    assert json.loads((output_dir / "manifest.json").read_text())["status"] == "completed"
    with (output_dir / "paired_track_keys.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []


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


def test_cli_replaces_stale_completed_manifest_and_records_missing_input_failure(tmp_path):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    nominal_states.unlink()
    output_dir = tmp_path / "missing-input-run"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "command": [sys.executable, str(Path(m.__file__).resolve())],
            }
        ),
        encoding="utf-8",
    )

    result = m.main(
        [
            "--nominal-states", str(nominal_states),
            "--misaligned-states", str(misaligned_states),
            "--nominal-geometry", str(nominal_geometry),
            "--misaligned-geometry", str(misaligned_geometry),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
        ]
    )

    assert result != 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["inputs"]["nominal_trackstates"]["exists"] is False
    assert manifest["metrics"]["failure_reason"]


def test_cli_starts_manifest_before_input_hash_and_finalizes_unreadable_failure(
    tmp_path, monkeypatch
):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    output_dir = tmp_path / "unreadable-input-run"
    manifest_module = sys.modules[m.write_experiment_manifest.__module__]

    def fail_hash(_path):
        running = json.loads((output_dir / "manifest.json").read_text())
        assert running["status"] == "running"
        raise PermissionError("input is unreadable")

    monkeypatch.setattr(manifest_module, "_sha256", fail_hash)

    result = m.main(
        [
            "--nominal-states", str(nominal_states),
            "--misaligned-states", str(misaligned_states),
            "--nominal-geometry", str(nominal_geometry),
            "--misaligned-geometry", str(misaligned_geometry),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
        ]
    )

    assert result != 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["failure_reason"] == (
        "PermissionError: input is unreadable"
    )


def test_cli_finalizes_failed_manifest_on_keyboard_interrupt_after_start(tmp_path, monkeypatch):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    output_dir = tmp_path / "interrupted-run"

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(m, "load_paired_recovery_inputs", interrupt)

    result = m.main(
        [
            "--nominal-states", str(nominal_states),
            "--misaligned-states", str(misaligned_states),
            "--nominal-geometry", str(nominal_geometry),
            "--misaligned-geometry", str(misaligned_geometry),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
        ]
    )

    assert result != 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "KeyboardInterrupt" in manifest["metrics"]["failure_reason"]


def test_cli_late_failure_preserves_available_scientific_metrics(tmp_path, monkeypatch):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    output_dir = tmp_path / "late-failure-run"

    def fail_plots(*_args, **_kwargs):
        raise RuntimeError("plot write failed")

    monkeypatch.setattr(m, "write_diagnostic_plots", fail_plots)

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
            "--seed-iters", "4",
            "--refit-iters", "4",
            "--bootstrap-replicates", "7",
            "--bootstrap-seed", "23",
        ]
    )

    assert result != 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["failure_reason"] == "RuntimeError: plot write failed"
    assert manifest["metrics"]["cohort_counts"]["frozen_probe_rows"] == 1
    assert manifest["metrics"]["bootstrap"] == {"replicates": 7, "seed": 23}
    assert set(manifest["metrics"]["verdicts"]) == {"protected"}
    assert set(manifest["metrics"]["numerical_validity"]) == {
        "nominal|protected|seed",
        "nominal|protected|refit",
        "misaligned|protected|seed",
        "misaligned|protected|refit",
    }


def test_cli_fit_failure_preserves_already_available_cohort_counts(tmp_path, monkeypatch):
    m = load_module()
    nominal_states, misaligned_states, nominal_geometry, misaligned_geometry = _cli_fixture(tmp_path)
    output_dir = tmp_path / "fit-failure-run"

    def fail_fit(*_args, **_kwargs):
        raise RuntimeError("fit failed")

    monkeypatch.setattr(m, "run_four_arm_recovery", fail_fit)

    result = m.main(
        [
            "--nominal-states", str(nominal_states),
            "--misaligned-states", str(misaligned_states),
            "--nominal-geometry", str(nominal_geometry),
            "--misaligned-geometry", str(misaligned_geometry),
            "--target-layers", "17:8",
            "--event-partition", "all",
            "--output-dir", str(output_dir),
        ]
    )

    assert result != 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["failure_reason"] == "RuntimeError: fit failed"
    assert manifest["metrics"]["cohort_counts"] == {
        "available_unique_events": 1,
        "selected_unique_events": 1,
        "paired_measured_hits": 11,
        "paired_unique_tracks": 1,
        "paired_target_hits": 1,
        "frozen_probe_rows": 1,
        "prediction_modes": {"interpolation": 1},
    }


def test_main_activates_explicit_config_before_building_path_defaults(tmp_path, monkeypatch):
    m = load_module()
    data_root = tmp_path / "configured-data"
    config_path = tmp_path / "override.toml"
    config_path.write_text(
        "\n".join(
            (
                "[project]",
                f'data_root = "{data_root}"',
                "[alignment]",
                'target_layers = ["16:6", "23:4"]',
            )
        ),
        encoding="utf-8",
    )
    captured = {}

    def inspect_manifest(run_dir, **kwargs):
        captured["run_dir"] = Path(run_dir)
        captured["inputs"] = kwargs["inputs"]
        captured["config"] = kwargs["config"]
        raise RuntimeError("stop after config inspection")

    monkeypatch.setattr(m, "write_experiment_manifest", inspect_manifest)

    assert m.main(["--config", str(config_path)]) == 1
    assert captured["config"].source_path == config_path
    assert captured["run_dir"] == (
        data_root / "experiments" / "helix_recovery" / "Vol16_Lay6__Vol23_Lay4"
    )
    assert Path(captured["inputs"]["nominal_trackstates"]).name == (
        "trackstates_kf_v16l6_v23l4.root"
    )
    assert Path(captured["inputs"]["misaligned_geometry"]).name == (
        "misalignment_applied_v16l6_v23l4.json"
    )


def test_atomic_json_writer_removes_temporary_file_on_serialization_failure(tmp_path):
    m = load_module()
    destination = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="Out of range float"):
        m._atomic_write_json(destination, {"bad": float("nan")})

    assert list(tmp_path.iterdir()) == []


def test_atomic_csv_writer_removes_temporary_file_on_write_failure(tmp_path):
    m = load_module()
    destination = tmp_path / "metrics.csv"

    with pytest.raises(ValueError, match="fields not in fieldnames"):
        m._atomic_write_csv(destination, [{"expected": 1, "unexpected": 2}], ("expected",))

    assert list(tmp_path.iterdir()) == []


def test_diagnostic_plot_writer_closes_figures_when_save_fails(tmp_path, monkeypatch):
    m = load_module()
    statistics = m.summarize_four_arm_recovery(
        _four_arm_statistics_fixture(m), bootstrap_replicates=3,
        exposures=("protected",),
    )

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(m, "_save_figure_atomic", fail_save)
    with pytest.raises(RuntimeError, match="save failed"):
        m.write_diagnostic_plots(tmp_path, statistics, ("protected",))

    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


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


def test_freeze_paired_cohort_refuses_different_requested_target_layers():
    m = load_module()
    nominal = _inputs(m)
    misaligned = _inputs(m)
    nominal.requested_target_layers = ((17, 5),)
    misaligned.requested_target_layers = ((17, 5), (24, 4))

    with pytest.raises(ValueError, match="requested target layers"):
        m.freeze_paired_cohort(nominal, misaligned)


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


def test_four_arm_runner_does_not_execute_an_unrequested_broken_exposure(monkeypatch):
    m = load_module()
    cohort = m.freeze_paired_cohort(_inputs(m), _inputs(m))
    geometry = _geometry(m)
    real_seed = m.seed_helices

    def fail_if_exposed(batches, *args, **kwargs):
        if batches.fit_mask[0, 5]:
            raise RuntimeError("exposed arm is broken")
        return real_seed(batches, *args, **kwargs)

    monkeypatch.setattr(m, "seed_helices", fail_if_exposed)

    report = m.run_four_arm_recovery(
        cohort,
        nominal_geometry=geometry,
        misaligned_geometry=geometry,
        seed_iterations=4,
        refit_iterations=4,
        exposures=("protected",),
    )

    assert set(report.arms) == {("nominal", "protected"), ("misaligned", "protected")}


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


def test_recovery_statistics_cannot_pass_when_a_requested_layer_has_no_input_hits():
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    # These inputs share all observed hits but explicitly request another layer.
    report.cohort.nominal.requested_target_layers = ((17, 8), (24, 4))
    report.cohort.misaligned.requested_target_layers = ((17, 8), (24, 4))

    result = m.summarize_four_arm_recovery(report, bootstrap_replicates=7)

    for exposure in ("protected", "exposed"):
        groups = result["exposures"][exposure]["groups"]
        for mode in ("interpolation", "forward_extrapolation", "backward_extrapolation"):
            for component in ("LOC0", "LOC1"):
                assert groups[f"17:8|{mode}|{component}"]["verdict"] == "passed"
                key = f"24:4|{mode}|{component}"
                assert key in groups
                assert groups[key]["requested_count"] == groups[key]["valid_count"] == 0
                assert groups[key]["verdict"] == "inconclusive"
        assert result["exposures"][exposure]["verdict"] == "inconclusive"


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


def test_exposed_bootstrap_stream_is_invariant_to_whether_protected_is_requested():
    m = load_module()
    report = _four_arm_statistics_fixture(m)
    values = torch.linspace(1.01, 1.50, report.cohort.n_rows, dtype=torch.float64)
    for exposure in ("protected", "exposed"):
        arm = report.arms[("misaligned", exposure)]
        local = values[:, None, None].expand(-1, 1, 2).clone()
        report.arms[("misaligned", exposure)] = m.replace(
            arm,
            gauss_newton=m.replace(
                arm.gauss_newton,
                evaluation=m.replace(arm.gauss_newton.evaluation, local_position=local),
            ),
        )

    both = m.summarize_four_arm_recovery(
        report, bootstrap_replicates=101, bootstrap_seed=23,
        exposures=("protected", "exposed"),
    )
    exposed_only = m.summarize_four_arm_recovery(
        report, bootstrap_replicates=101, bootstrap_seed=23,
        exposures=("exposed",),
    )
    protected_only = m.summarize_four_arm_recovery(
        report, bootstrap_replicates=101, bootstrap_seed=23,
        exposures=("protected",),
    )

    assert both["exposures"]["exposed"] == exposed_only["exposures"]["exposed"]
    assert both["exposures"]["protected"] == protected_only["exposures"]["protected"]


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
