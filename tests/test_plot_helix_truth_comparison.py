import importlib

import numpy as np
import pandas as pd
import pytest


def model():
    return importlib.import_module("scripts.plot_helix_truth_comparison")


def example():
    rows = []
    for event, sensor, occurrence, target in [(1, 3, 0, False), (1, 3, 1, True), (2, 4, 0, False)]:
        row = dict(event_nr=event, track_nr=0, volume_id=17, layer_id=2,
                   module_id=sensor, occurrence=occurrence, target=target,
                   intersection_valid=True)
        for a in range(2):
            row.update({f"prediction_LOC{a}_mm": 2., f"truth_LOC{a}_mm": 0.,
                        f"measurement_LOC{a}_mm": 1., f"variance_LOC{a}_mm2": 4.,
                        f"weighted_LOC{a}": True, f"acts_LOC{a}_mm": 1.})
        rows.append(row)
    paired = pd.DataFrame(rows)
    acts = paired.drop(columns=["target", "intersection_valid", "prediction_LOC0_mm", "prediction_LOC1_mm"])
    helix = paired.drop(columns=["acts_LOC0_mm", "acts_LOC1_mm"])
    tracks = paired[["event_nr", "track_nr"]].drop_duplicates().reset_index(drop=True)
    return helix, acts, tracks


def test_exact_pairing_is_order_independent_and_preserves_occurrences():
    m = model()
    helix, acts, _ = example()
    result = m.pair_surfaces(helix, acts.iloc[::-1])
    assert len(result) == 3
    assert result.occurrence.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="identity"):
        m.pair_surfaces(helix, acts.iloc[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        m.pair_surfaces(helix, pd.concat([acts, acts.iloc[:1]]))
    acts.loc[0, "truth_LOC0_mm"] = 1.
    with pytest.raises(ValueError, match="truth_LOC0"):
        m.pair_surfaces(helix, acts)


def test_hand_computed_chi2_and_empty_target_are_correct():
    m = model()
    helix, acts, tracks = example()
    data, summary = m.analyze(m.pair_surfaces(helix, acts), tracks)
    np.testing.assert_allclose(data["chi2"]["all"]["ACTS"]["Q"], [1., .5])
    np.testing.assert_allclose(data["chi2"]["all"]["Helix"]["Q"], [4., 2.])
    np.testing.assert_allclose(data["chi2"]["all"]["ACTS"]["Q_per_component"], [.25, .25])
    assert summary["chi2"]["target"]["paired_tracks"] == 1
    assert summary["residuals"]["all"]["LOC0"]["ACTS"]["bias_mm"] == 1.
    assert summary["residuals"]["all"]["LOC1"]["Helix"]["rmse_mm"] == 2.


def test_missing_acts_does_not_shrink_complete_track_sum():
    m = model()
    helix, acts, tracks = example()
    acts.loc[0, "acts_LOC0_mm"] = np.nan
    data, summary = m.analyze(m.pair_surfaces(helix, acts), tracks)
    assert summary["chi2"]["all"]["paired_tracks"] == 1
    assert summary["chi2"]["all"]["excluded_incomplete_tracks"] == 1
    assert summary["chi2"]["target"]["paired_tracks"] == 1
    assert len(data["residuals"]["all"]["LOC0"]["ACTS"]) == 2


def test_one_dimensional_component_keeps_residual_but_has_no_weight():
    m = model()
    helix, acts, tracks = example()
    for frame in (helix, acts):
        frame.loc[0, "weighted_LOC1"] = False
        frame.loc[0, "measurement_LOC1_mm"] = np.nan
    data, _ = m.analyze(m.pair_surfaces(helix, acts), tracks)
    np.testing.assert_allclose(data["chi2"]["all"]["Helix"]["Q"], [3., 2.])
    assert len(data["residuals"]["all"]["LOC1"]["Helix"]) == 3


def test_identical_truth_gives_zero_and_density_uses_full_sample():
    m = model()
    helix, acts, tracks = example()
    for a in range(2):
        helix[f"prediction_LOC{a}_mm"] = 0.
        acts[f"acts_LOC{a}_mm"] = 0.
    data, _ = m.analyze(m.pair_surfaces(helix, acts), tracks)
    assert np.all(data["chi2"]["all"]["Helix"]["Q"] == 0.)
    density = m.full_sample_density(np.array([0., 0., 10.]), np.array([-1., 0., 1.]))
    assert density.sum() == pytest.approx(2 / 3)


def test_plot_annotations_are_exactly_the_requested_statistics(tmp_path):
    m = model()
    helix, acts, tracks = example()
    data, summary = m.analyze(m.pair_surfaces(helix, acts), tracks)
    figures = m.make_figures(data, summary)
    residual, chi2 = figures
    assert len(residual.axes) == len(chi2.axes) == 6
    for ax in residual.axes:
        text = " ".join(t.get_text() for t in ax.texts)
        assert "N=" in text and "bias=" in text and "RMSE=" in text
        assert "P95" not in text and "mean=" not in text
    for ax in chi2.axes:
        text = " ".join(t.get_text() for t in ax.texts)
        assert "tracks=" in text and "mean=" in text
        assert "median" not in text and "P95" not in text
    import matplotlib.pyplot as plt
    for fig in figures:
        plt.close(fig)


def test_saved_helix_scores_are_verified():
    m = model()
    helix, acts, tracks = example()
    data, _ = m.analyze(m.pair_surfaces(helix, acts), tracks)
    for group, raw in data["helix_raw"].items():
        tracks[f"{group}_complete"] = raw["complete"]
        tracks[f"{group}_n_components"] = raw["valid_N"]
        tracks[f"{group}_truth_chi2"] = raw["Q"]
    m.verify_saved_helix(data, tracks)
    tracks.loc[0, "all_truth_chi2"] += 1.
    with pytest.raises(ValueError, match="chi2 mismatch"):
        m.verify_saved_helix(data, tracks)


def test_truncated_cohort_is_rejected():
    m = model()
    parent = {"metrics": {"summary": {"tracks_evaluated": 2, "surfaces_evaluated": 3}}}
    m.verify_cohort_counts(parent, 2, 3)
    with pytest.raises(ValueError, match="cohort"):
        m.verify_cohort_counts(parent, 1, 2)


@pytest.mark.root_io
def test_root_loading_keeps_repeated_occurrences_and_compact_errors(tmp_path):
    m = model()
    import awkward as ak
    import uproot
    root_path = tmp_path / "states.root"
    payload = {"event_nr": np.array([1]), "track_nr": np.array([0])}
    state = {
        "volume_id": [17, 0, 17], "layer_id": [2, 0, 2], "module_id": [3, 0, 3],
        "l_x_hit": [1., np.nan, 1.], "l_y_hit": [1., np.nan, np.nan],
        "t_eLOC0": [0., np.nan, 0.], "t_eLOC1": [0., np.nan, 0.],
        "eLOC0_smt": [1., np.nan, 2.], "eLOC1_smt": [1., np.nan, 2.],
        "err_x_hit": [2., 2.], "err_y_hit": [2., 0.],
    }
    payload.update({k: ak.Array([v]) for k, v in state.items()})
    with uproot.recreate(root_path) as root:
        root.mktree("trackstates", payload)
    tracks = pd.DataFrame({"event_nr": [1], "track_nr": [0]})
    loaded = m.load_acts(root_path, tracks)
    assert loaded.occurrence.tolist() == [0, 1]
    assert loaded.acts_LOC0_mm.tolist() == [1., 2.]
    assert loaded.variance_LOC0_mm2.tolist() == [4., 4.]
    assert loaded.weighted_LOC1.tolist() == [True, False]


def test_hash_failure_finalizes_manifest_and_preserves_source(tmp_path):
    m = model()
    import json
    run = tmp_path / "source"
    run.mkdir()
    root = run / "states.root"
    geometry = run / "geometry.json"
    root.write_bytes(b"input")
    geometry.write_text("[]")
    parent = {"status": "completed", "configuration": {}, "inputs": {
        "nominal_states": {"path": str(root), "sha256": "wrong"},
        "nominal_geometry": {"path": str(geometry), "sha256": "wrong"}}}
    (run / "manifest.json").write_text(json.dumps(parent))
    output = tmp_path / "plot"
    assert m.main(["--run-dir", str(run), "--output-dir", str(output)]) == 1
    child = json.loads((output / "manifest.json").read_text())
    assert child["status"] == "failed"
    assert "hash mismatch" in child["metrics"]["error"]
    assert root.read_bytes() == b"input"
    assert json.loads((run / "manifest.json").read_text()) == parent
