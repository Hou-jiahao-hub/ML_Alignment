import importlib

import numpy as np
import pandas as pd
import pytest
import awkward as ak
import uproot


def _frame():
    return pd.DataFrame(
        {
            "event_nr": [0, 1, 2, 5],
            "volume_id": [17, 17, 24, 24],
            "layer_id": [8, 8, 4, 4],
            "l_x_hit": [1.0, -1.0, 1.0, -1.0],
            "l_y_hit": [1.0, -1.0, 1.2, -1.2],
            "eLOC0_smt": [0.0] * 4,
            "eLOC1_smt": [0.0] * 4,
            "err_x_hit": [1.0] * 4,
            "err_y_hit": [1.0, 1.0, 0.1, 0.1],
        }
    )


def test_per_component_thresholds_drive_normalized_acceptance_and_score():
    m = importlib.import_module("scripts.reconstruction_acceptance")
    frame = _frame()
    frame.loc[0, "l_x_hit"] = 2.0
    thresholds = {
        "17:8": {"LOC0": 2.5, "LOC1": 1.0},
        "24:4": {"LOC0": 1.0, "LOC1": 1.0},
    }

    result = m.compute_reconstruction_acceptance(
        frame, ((17, 8), (24, 4)), thresholds=thresholds
    )

    metric = result["layers"]["17:8"]["LOC0"]
    assert metric["chi2_per_component"] == pytest.approx(2.5)
    assert metric["threshold"] == pytest.approx(2.5)
    assert metric["normalized_chi2"] == pytest.approx(1.0)
    assert result["passed"] is True
    assert m.acceptance_score(result) == pytest.approx((0, 1.0, 0.0))


def test_acceptance_can_filter_the_blind_validation_partition():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    result = m.compute_reconstruction_acceptance(
        _frame(),
        ((17, 8), (24, 4)),
        event_partition="validation",
        split_seed=6006,
    )

    assert result["event_partition"] == "validation"
    assert result["layers"]["17:8"]["LOC0"]["n_components"] == 0
    assert result["layers"]["24:4"]["LOC0"]["n_components"] == 1
    assert result["passed"] is False


def test_nominal_calibration_builds_frozen_predictive_thresholds():
    m = importlib.import_module("scripts.reconstruction_acceptance")
    rows = []
    for event_nr in range(100):
        rows.append(
            {
                "event_nr": event_nr,
                "volume_id": 17,
                "layer_id": 8,
                "l_x_hit": 2.0,
                "l_y_hit": 1.0,
                "eLOC0_smt": 0.0,
                "eLOC1_smt": 0.0,
                "err_x_hit": 1.0,
                "err_y_hit": 1.0,
            }
        )

    baseline = m.build_acceptance_baseline(
        pd.DataFrame(rows),
        ((17, 8),),
        split_seed=6006,
        bootstrap_seed=17,
        n_bootstrap=20,
    )

    assert baseline["schema_version"] == "2.0"
    assert baseline["event_split"]["calibration_events"] == 10
    assert baseline["event_split"]["validation_events"] == 10
    assert baseline["layers"]["17:8"]["LOC0"]["threshold"] == pytest.approx(4.0)
    assert baseline["layers"]["17:8"]["LOC0"]["predictive_sample_events"] == 10
    assert baseline["layers"]["17:8"]["LOC1"]["threshold"] == pytest.approx(1.0)


def test_cli_uses_threshold_profile_and_selected_partition(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    profile = tmp_path / "baseline.json"
    profile.write_text(
        __import__("json").dumps(
            {
                "schema_version": "2.0",
                "event_split": {"seed": 6006},
                "layers": {
                    "17:8": {"LOC0": {"threshold": 2.5}, "LOC1": {"threshold": 1.0}},
                    "24:4": {"LOC0": {"threshold": 1.0}, "LOC1": {"threshold": 1.0}},
                },
            }
        )
    )
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(
        m, "load_reconstruction_frame", lambda _path, **_kwargs: _frame()
    )

    status = m.main(
        [
            "--states",
            str(tmp_path / "states.root"),
            "--output",
            str(output),
            "--target-layers",
            "17:8,24:4",
            "--threshold-profile",
            str(profile),
            "--event-partition",
            "validation",
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert status == 1  # Vol17 is deliberately absent from the validation fixture.
    assert result["threshold_profile"] == str(profile)
    assert result["event_partition"] == "validation"
    assert result["layers"]["24:4"]["LOC1"]["normalized_chi2"] == pytest.approx(1.0)


def test_track_level_event_numbers_expand_to_each_state():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    expanded = m._expand_event_numbers([[17, 0, 24], [17]], [5, 9])

    assert expanded.tolist() == [5, 5, 5, 9]


def test_acceptance_threshold_is_inclusive_and_strip_loc1_uses_floor():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    result = m.compute_reconstruction_acceptance(_frame(), ((17, 8), (24, 4)))

    assert result["passed"] is True
    assert result["failed_layers"] == []
    assert result["failed_components"] == []
    assert result["layers"]["17:8"]["LOC0"]["chi2_per_component"] == pytest.approx(1.0)
    assert result["layers"]["24:4"]["LOC1"]["chi2_per_component"] == pytest.approx(1.0)
    assert result["layers"]["24:4"]["LOC1"]["n_components"] == 2
    assert m.acceptance_score(result) == pytest.approx((0, 1.0, 0.0))


def test_any_component_above_threshold_fails_its_layer():
    m = importlib.import_module("scripts.reconstruction_acceptance")
    frame = _frame()
    frame.loc[0, "l_x_hit"] = 2.0

    result = m.compute_reconstruction_acceptance(frame, ((17, 8), (24, 4)))

    assert result["passed"] is False
    assert result["failed_layers"] == ["17:8"]
    assert result["failed_components"] == ["17:8:LOC0"]
    assert m.acceptance_score(result)[0] == 1


@pytest.mark.parametrize("failure", ["missing_layer", "zero_component", "nonfinite"])
def test_missing_zero_or_nonfinite_target_metrics_fail(failure):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    frame = _frame()
    if failure == "missing_layer":
        frame = frame[frame.volume_id == 17].copy()
    elif failure == "zero_component":
        frame.loc[frame.volume_id == 24, "l_y_hit"] = np.nan
    else:
        frame.loc[frame.volume_id == 17, "eLOC0_smt"] = np.nan

    result = m.compute_reconstruction_acceptance(frame, ((17, 8), (24, 4)))

    assert result["passed"] is False
    assert result["failed_components"]
    assert m.acceptance_score(result)[0] >= 1


def test_incomplete_root_fields_are_not_silently_skipped():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    with pytest.raises(ValueError, match="missing required fields"):
        m.compute_reconstruction_acceptance(
            _frame().drop(columns=["err_x_hit"]), ((17, 8), (24, 4))
        )


def test_cli_root_read_failure_marks_every_target_component_failed(
    tmp_path, monkeypatch
):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(
        m,
        "load_reconstruction_frame",
        lambda _path: (_ for _ in ()).throw(ValueError("broken ROOT fields")),
    )

    status = m.main(
        [
            "--states",
            str(tmp_path / "broken.root"),
            "--output",
            str(output),
            "--target-layers",
            "17:8,24:4",
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert status == 1
    assert result["failed_components"] == [
        "17:8:LOC0",
        "17:8:LOC1",
        "24:4:LOC0",
        "24:4:LOC1",
    ]
    assert result["acceptance_score"] == [4, None, None]


def test_polish_score_compares_failed_count_then_max_then_excess():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    assert m.polish_improves((1, 2.0, 1.0), (2, 1.1, 0.1))
    assert m.polish_improves((1, 1.5, 0.7), (1, 2.0, 0.1))
    assert not m.polish_improves((1, 2.0, 0.5), (1, 2.0, 0.5))


@pytest.mark.root_io
def test_root_loader_expands_compact_measurement_errors(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    path = tmp_path / "states.root"
    with uproot.recreate(path) as root_file:
        root_file["trackstates"] = {
            "volume_id": ak.Array([[17, 0, 24]]),
            "layer_id": ak.Array([[8, 0, 4]]),
            "l_x_hit": ak.Array([[1.0, np.nan, 1.0]]),
            "l_y_hit": ak.Array([[1.0, np.nan, 1.2]]),
            "eLOC0_smt": ak.Array([[0.0, 0.0, 0.0]]),
            "eLOC1_smt": ak.Array([[0.0, 0.0, 0.0]]),
            "err_x_hit": ak.Array([[1.0, 1.0]]),
            "err_y_hit": ak.Array([[1.0, 0.1]]),
        }

    monkeypatch.setattr(
        m.uproot,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PyROOT-capable environments must not use uproot")
        ),
    )

    frame = m.load_reconstruction_frame(path)
    result = m.compute_reconstruction_acceptance(frame, ((17, 8), (24, 4)))

    assert len(frame) == 3
    assert np.isnan(frame.loc[1, "err_x_hit"])
    assert result["passed"] is True
