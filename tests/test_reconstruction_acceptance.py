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


def test_nominal_calibration_builds_frozen_empirical_thresholds():
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
        bootstrap_seed=17,
        n_bootstrap=20,
    )

    assert baseline["schema_version"] == "3.1"
    assert baseline["selection_version"] == "full_event_bootstrap_v1"
    assert baseline["method"] == "event_bootstrap_empirical_upper_99"
    assert baseline["event_identity"]["nominal_events"] == 100
    assert len(baseline["event_identity"]["event_key_sha256"]) == 64
    assert baseline["layers"]["17:8"]["LOC0"]["threshold"] == pytest.approx(4.0)
    assert baseline["layers"]["17:8"]["LOC0"]["bootstrap_sample_events"] == 100
    assert baseline["layers"]["17:8"]["LOC0"]["bootstrap_upper_99"] == pytest.approx(4.0)
    assert baseline["layers"]["17:8"]["LOC1"]["threshold"] == pytest.approx(1.0)


def test_generated_profile_is_accepted_without_reshaping_its_event_identity():
    """The producer's real JSON shape must be consumable by the verifier."""
    m = importlib.import_module("scripts.reconstruction_acceptance")
    rows = []
    for event_nr in range(8):
        for volume, layer in ((17, 8), (24, 4)):
            rows.append(
                {
                    "event_nr": event_nr,
                    "volume_id": volume,
                    "layer_id": layer,
                    "l_x_hit": 1.0,
                    "l_y_hit": 1.0,
                    "eLOC0_smt": 0.0,
                    "eLOC1_smt": 0.0,
                    "err_x_hit": 1.0,
                    "err_y_hit": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    profile = m.build_acceptance_baseline(frame, n_bootstrap=4)

    result = m.compute_reconstruction_acceptance(frame, profile=profile)

    assert result["profile_identity_verified"] is True


def test_bootstrap_population_includes_events_without_target_components():
    """A full-event bootstrap must not collapse to contributing events only."""
    m = importlib.import_module("scripts.reconstruction_acceptance")
    frame = pd.DataFrame(
        [
            {
                "event_nr": 0,
                "volume_id": 17,
                "layer_id": 8,
                "l_x_hit": 1.0,
                "l_y_hit": 1.0,
                "eLOC0_smt": 0.0,
                "eLOC1_smt": 0.0,
                "err_x_hit": 1.0,
                "err_y_hit": 1.0,
            },
            {
                "event_nr": 1,
                "volume_id": 18,
                "layer_id": 2,
                "l_x_hit": 1.0,
                "l_y_hit": 1.0,
                "eLOC0_smt": 0.0,
                "eLOC1_smt": 0.0,
                "err_x_hit": 1.0,
                "err_y_hit": 1.0,
            },
        ]
    )

    profile = m.build_acceptance_baseline(
        frame, ((17, 8),), bootstrap_seed=6, n_bootstrap=1
    )

    assert profile["event_identity"]["nominal_events"] == 2
    assert profile["layers"]["17:8"]["LOC0"]["bootstrap_sample_events"] == 2


def test_bootstrap_fails_closed_when_a_replicate_has_no_components():
    """An undefined bootstrap ratio must never become NaN JSON or a threshold."""
    m = importlib.import_module("scripts.reconstruction_acceptance")
    frame = pd.DataFrame(
        [
            {
                "event_nr": 0,
                "volume_id": 17,
                "layer_id": 8,
                "l_x_hit": 1.0,
                "l_y_hit": 1.0,
                "eLOC0_smt": 0.0,
                "eLOC1_smt": 0.0,
                "err_x_hit": 1.0,
                "err_y_hit": 1.0,
            },
            {
                "event_nr": 1,
                "volume_id": 18,
                "layer_id": 2,
                "l_x_hit": 1.0,
                "l_y_hit": 1.0,
                "eLOC0_smt": 0.0,
                "eLOC1_smt": 0.0,
                "err_x_hit": 1.0,
                "err_y_hit": 1.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="zero components"):
        m.build_acceptance_baseline(
            frame, ((17, 8),), bootstrap_seed=0, n_bootstrap=1
        )


def _write_profile(tmp_path, m, frame, event_key_sha256=None):
    import hashlib
    import json as _json

    if event_key_sha256 is None:
        event_key_sha256 = m.event_key_hash(frame["event_nr"].to_numpy())
    profile = tmp_path / "baseline.json"
    profile.write_text(
        _json.dumps(
            {
                "schema_version": "3.1",
                "selection_version": m.SELECTION_VERSION,
                "event_identity": {
                    "nominal_events": int(frame["event_nr"].nunique()),
                    "event_key_sha256": event_key_sha256,
                },
                "layers": {
                    "17:8": {"LOC0": {"threshold": 2.5}, "LOC1": {"threshold": 1.0}},
                    "24:4": {"LOC0": {"threshold": 1.0}, "LOC1": {"threshold": 1.0}},
                },
            }
        )
    )
    profile_sha256 = hashlib.sha256(profile.read_bytes()).hexdigest()
    return profile, profile_sha256


def test_cli_uses_threshold_profile(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")

    frame = _frame()
    profile, profile_sha256 = _write_profile(tmp_path, m, frame)
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(
        m, "load_reconstruction_frame", lambda _path, **_kwargs: frame
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
            "--threshold-profile-sha256",
            profile_sha256,
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert status == 0
    assert result["threshold_profile"] == str(profile)
    assert result["threshold_profile_sha256"] == profile_sha256
    assert result["profile_identity_verified"] is True
    assert "event_partition" not in result
    assert result["profile_type"] == "diagnostic_only"
    assert result["verdict_eligible"] is False
    assert result["layers"]["24:4"]["LOC1"]["normalized_chi2"] == pytest.approx(1.0)


def test_cli_omits_profile_hash_when_no_threshold_profile_given(tmp_path, monkeypatch):
    """A run's frozen profile hash must be discoverable from its own acceptance
    result rather than only living in the shell variable that produced it."""
    m = importlib.import_module("scripts.reconstruction_acceptance")

    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(m, "load_reconstruction_frame", lambda _path, **_kwargs: _frame())

    m.main(
        [
            "--states",
            str(tmp_path / "states.root"),
            "--output",
            str(output),
            "--target-layers",
            "17:8,24:4",
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert result["threshold_profile_sha256"] is None


def test_cli_requires_frozen_profile_hash_whenever_profile_is_given(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")

    frame = _frame()
    profile, _profile_sha256 = _write_profile(tmp_path, m, frame)
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(m, "load_reconstruction_frame", lambda _path, **_kwargs: frame)

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
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert status == 1
    assert "threshold-profile-sha256" in result["error"]


def test_cli_fails_closed_on_tampered_profile_file_hash(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")

    frame = _frame()
    profile, _profile_sha256 = _write_profile(tmp_path, m, frame)
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(m, "load_reconstruction_frame", lambda _path, **_kwargs: frame)

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
            "--threshold-profile-sha256",
            "0" * 64,
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert status == 1
    assert "hash mismatch" in result["error"]


def test_cli_fails_closed_on_mismatched_profile_event_key(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    import hashlib

    frame = _frame()
    profile = tmp_path / "baseline.json"
    profile.write_text(
        __import__("json").dumps(
            {
                "schema_version": "3.1",
                "selection_version": m.SELECTION_VERSION,
                "event_identity": {
                    "nominal_events": int(frame["event_nr"].nunique()),
                    "event_key_sha256": "deadbeef",
                },
                "layers": {
                    "17:8": {"LOC0": {"threshold": 2.5}, "LOC1": {"threshold": 1.0}},
                    "24:4": {"LOC0": {"threshold": 1.0}, "LOC1": {"threshold": 1.0}},
                },
            }
        )
    )
    profile_sha256 = hashlib.sha256(profile.read_bytes()).hexdigest()
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(m, "load_reconstruction_frame", lambda _path, **_kwargs: frame)

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
            "--threshold-profile-sha256",
            profile_sha256,
        ]
    )

    result = __import__("json").loads(output.read_text())
    assert status == 1
    assert result["passed"] is False
    assert "event key does not match" in result["error"]


def test_compute_reconstruction_acceptance_verifies_identity_against_full_event_set():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    frame = _frame()
    profile = {
        "selection_version": m.SELECTION_VERSION,
        "event_identity": {
            "nominal_events": int(frame["event_nr"].nunique()),
            "event_key_sha256": m.event_key_hash(frame["event_nr"].to_numpy()),
        },
    }

    result = m.compute_reconstruction_acceptance(
        frame, ((17, 8), (24, 4)), profile=profile
    )
    assert result["profile_identity_verified"] is True

    with pytest.raises(m.AcceptanceProfileIdentityError, match="event key does not match"):
        m.compute_reconstruction_acceptance(
            frame,
            ((17, 8), (24, 4)),
            profile={
                "selection_version": m.SELECTION_VERSION,
                "event_identity": {"nominal_events": 4, "event_key_sha256": "deadbeef"},
            },
        )

    with pytest.raises(m.AcceptanceProfileIdentityError, match="selection_version"):
        m.compute_reconstruction_acceptance(
            frame,
            ((17, 8), (24, 4)),
            profile={
                "selection_version": "stale",
                "event_identity": {"nominal_events": 4, "event_key_sha256": "deadbeef"},
            },
        )


def test_load_threshold_profile_rejects_v2_schema(tmp_path):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    profile = tmp_path / "legacy.json"
    profile.write_text('{"schema_version":"2.0"}')

    with pytest.raises(ValueError, match="Unsupported threshold profile schema"):
        m.load_threshold_profile(profile, ((17, 8),))


def test_track_level_event_numbers_expand_to_each_state():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    expanded = m._expand_event_numbers([[17, 0, 24], [17]], [5, 9])

    assert expanded.tolist() == [5, 5, 5, 9]


def test_compute_reconstruction_acceptance_records_the_pinned_profile_hash():
    m = importlib.import_module("scripts.reconstruction_acceptance")

    result = m.compute_reconstruction_acceptance(
        _frame(), ((17, 8), (24, 4)), threshold_profile_sha256="deadbeef" * 8
    )

    assert result["threshold_profile_sha256"] == "deadbeef" * 8


def test_cli_root_read_failure_records_no_profile_hash(tmp_path, monkeypatch):
    m = importlib.import_module("scripts.reconstruction_acceptance")
    output = tmp_path / "acceptance.json"
    monkeypatch.setattr(
        m,
        "load_reconstruction_frame",
        lambda _path, **_kwargs: (_ for _ in ()).throw(ValueError("broken ROOT fields")),
    )

    m.main(
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
    assert result["threshold_profile_sha256"] is None


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
