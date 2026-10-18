import importlib

import numpy as np
import pandas as pd
import pytest


def module():
    return importlib.import_module("scripts.analysis.compare_helix_material_effects")


def test_comparison_cli_imports_without_acts_or_pythonpath():
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/analysis/compare_helix_material_effects.py"), "--help"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, cwd="/tmp")
    assert result.returncode == 0, result.stderr
    assert "--run-dirs" in result.stdout


def components(values, *, valid=True):
    x = np.array(values, dtype=float)
    return pd.DataFrame(dict(event_nr=np.arange(len(x)), measurement2=x, truth2=x / 2,
                             V=np.full(len(x), .5), P=np.full(len(x), .25),
                             U=x - .75, truth_excess=x / 2 - .25,
                             complete=np.full(len(x), valid)))


def test_bootstrap_uses_common_generated_events_without_survivor_intersection():
    m = module()
    first = components([1, 3, 5])
    second = components([.5, 2, 4]).iloc[:2]
    result = m.bootstrap_metrics({"on": first, "off": second}, np.arange(4), replicates=100, seed=5)
    assert result["on"]["M"][0] == pytest.approx(3)
    assert result["off"]["M"][0] == pytest.approx(1.25)
    assert len(result["on"]["M"][1]) == 100
    assert np.isfinite(result["on"]["M"][1]).any()


def test_incomplete_component_cannot_produce_valid_improvement():
    m = module()
    row = m.summarize_cell(components([1, 2, np.inf]))
    assert not row["complete"]
    assert row["M"] is None
    assert m.interval_verdict(-2., [-3., -1.], complete=False) == "not_evaluable"
    assert m.interval_verdict(-2., [-3., -1.], complete=True) == "decreased"
    assert m.interval_verdict(.1, [-.2, .3], complete=True) == "insufficient_evidence"


def test_initial_particle_matching_rejects_changed_geometry_distribution():
    m = module()
    first = pd.DataFrame({"event_id": [0, 1], "particle_hash": [1, 2],
                          "vx": [0., 1.], "vy": [0., 0.], "vz": [0., 0.], "vt": [0., 0.],
                          "px": [10., 0.], "py": [0., 10.], "pz": [1., 2.],
                          "m": [.105, .105], "q": [1., -1.], "eta": [.1, .2], "phi": [0., 1.]})
    high = first.copy()
    high[["px", "py", "pz"]] *= 10
    assert m.verify_initial_conditions([first, first.copy(), high])["passed"]
    high.loc[1, "vz"] += .1
    with pytest.raises(ValueError, match="initial"):
        m.verify_initial_conditions([first, first.copy(), high])


def test_reduced_surface_cohort_and_changed_saved_prediction_are_rejected():
    from test_validate_helix_truth import synthetic_track, module as validator
    m, v = module(), validator()
    track = synthetic_track()
    fit = v.fit_tracks([track], seed_iters=100)[0]
    prediction = v.ordered_predictions([track], [fit])[0]
    budget = v.target_prediction_budget(track, fit, prediction)
    surfaces, summary = v.evaluate_track(track, fit, prediction, budget)
    table, tracks = pd.DataFrame(surfaces), pd.DataFrame([summary])
    assert len(m.component_table(table, tracks)) == 24
    with pytest.raises(ValueError, match="cohort"):
        m.component_table(table.iloc[:-1], tracks)
    table.loc[0, "prediction_LOC0_mm"] += .1
    with pytest.raises(ValueError, match="chi2 mismatch"):
        m.component_table(table, tracks)


def test_changed_arm_configuration_is_rejected():
    m = module()
    records = [{"pt_gev": pt, "fatras_interactions": interactions, "geometry": "nominal",
                "num_events": 20000, "random_seed": 42}
               for pt, interactions in ((10., True), (10., False), (100., True))]
    m.verify_arm_configs(records)
    records[1]["fatras_interactions"] = True
    with pytest.raises(ValueError, match="arm configuration"):
        m.verify_arm_configs(records)


def test_truncated_cohort_or_unmatched_generated_event_is_rejected():
    m = module()
    effective = {"max_events": 0, "max_tracks": 0}
    counts = {"available_tracks": 3, "selected_tracks": 3, "available_events": 3, "selected_events": 3}
    tracks = pd.DataFrame({"event_nr": [0, 1], "track_nr": [0, 0]})
    excluded = pd.DataFrame({"event_nr": [2], "track_nr": [0], "reason": ["few hits"]})
    m.verify_full_cohort(effective, counts, tracks, excluded, np.arange(3))
    with pytest.raises(ValueError, match="full cohort"):
        m.verify_full_cohort({**effective, "max_tracks": 1}, counts, tracks, excluded, np.arange(3))
    with pytest.raises(ValueError, match="generated"):
        m.verify_full_cohort(effective, counts, tracks, excluded, np.arange(2))


@pytest.mark.root_io
def test_three_arm_root_to_budget_to_comparison_pipeline(tmp_path):
    """Synthetic I/O evidence only; it is not a Fatras material experiment."""
    import json
    import awkward as ak
    import uproot
    from PIL import Image
    from test_validate_helix_truth import synthetic_track, module as validator
    from scripts.experiment_manifest import write_experiment_manifest, finalize_experiment_manifest, _sha256
    from scripts.project_config import load_project_config

    m, v = module(), validator()
    track = synthetic_track()
    config_path = tmp_path / "config.toml"
    config_path.write_text('[alignment]\ntarget_layers = ["17:8"]\n')
    config = load_project_config(config_path)
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps([
        {"ID": f"vol={k[0]}|lay={k[1]}|sen={k[2]}", "Translation": track.centers[i].tolist(),
         "RotationMatrix": track.rotations[i].tolist()} for i, k in enumerate(track.identities)]))
    runs = []
    for index, group in enumerate(m.GROUPS):
        simulation, reconstruction, run = (tmp_path / name / group for name in ("simulation", "reconstruction", "helix"))
        simulation.mkdir(parents=True)
        values = {"event_id": np.arange(4)}
        for column in m.PARTICLE_COLUMNS:
            default = 10. if column == "px" else .105 if column == "m" else 1. if column in ("q", "particle_hash") else 0.
            if index == 2 and column in ("px", "py", "pz"):
                default *= 10
            values[column] = ak.Array([[default] for _ in range(4)])
        with uproot.recreate(simulation / "particles.root") as root:
            root.mktree("particles", values)
        (simulation / "timings.json").write_text(json.dumps({"wall_seconds": 1., "setup_seconds": .25, "run_seconds": .75}))
        sim_manifest = write_experiment_manifest(simulation, run_id=group, command=["synthetic-fixture"], config=config,
            inputs={"nominal_geometry": geometry}, metrics={"study_config": dict(
                pt_gev=100. if index == 2 else 10., fatras_interactions=index != 1,
                geometry="nominal", num_events=4, random_seed=42)})
        finalize_experiment_manifest(sim_manifest, status="completed", metrics={"output_sha256": {
            name: _sha256(simulation / name) for name in ("particles.root", "timings.json")}})
        reconstruction.mkdir(parents=True)
        payload = {"event_nr": np.arange(4), "track_nr": np.zeros(4, dtype=int),
            "volume_id": ak.Array([[k[0] for k in track.identities]] * 4),
            "layer_id": ak.Array([[k[1] for k in track.identities]] * 4),
            "module_id": ak.Array([[k[2] for k in track.identities]] * 4),
            "pathLength": ak.Array([track.path_length.tolist()] * 4)}
        for names, data in ((("g_x_smt", "g_y_smt", "g_z_smt"), track.smt_points),
                            (("px_smt", "py_smt", "pz_smt"), track.smt_directions),
                            (("l_x_hit", "l_y_hit"), track.measured),
                            (("t_eLOC0", "t_eLOC1"), track.truth),
                            (("err_x_hit", "err_y_hit"), track.variance.sqrt())):
            for a, name in enumerate(names):
                payload[name] = ak.Array([data[:, a].tolist()] * 4)
        states = reconstruction / "trackstates.root"
        with uproot.recreate(states) as root:
            root.mktree("trackstates", payload)
        (reconstruction / "timings.json").write_text(json.dumps({"wall_seconds": 2., "setup_seconds": .5, "run_seconds": 1.5}))
        manifest = write_experiment_manifest(reconstruction, run_id=group, command=["synthetic-fixture"],
            config=config, inputs={"simulation_dir": simulation, "nominal_geometry": geometry})
        finalize_experiment_manifest(manifest, status="completed", metrics={"output_sha256": {
            name: _sha256(reconstruction / name) for name in ("trackstates.root", "timings.json")}})
        assert v.main(["--config", str(config_path), "--nominal-states", str(states),
                       "--nominal-geometry", str(geometry), "--output-dir", str(run),
                       "--prediction-budget", "--max-events", "0", "--seed-iters", "100"]) == 0
        summary = json.loads((run / "summary.json").read_text())
        assert summary["prediction_budget"]["cells"]["17:8|LOC0"]["complete"]
        assert summary["timings"]["seed_calls"] == summary["timings"]["refit_calls"] == 1
        runs.append(str(run))
    output = tmp_path / "comparison"
    assert m.main(["--config", str(config_path), "--run-dirs", *runs, "--output-dir", str(output)]) == 0
    summary = json.loads((output / "comparison_summary.json").read_text())
    assert summary["initial_matching"]["n_generated_events"] == 4
    assert len(summary["main_contrasts"]) == 16
    assert (output / "timings.csv").exists()
    for name in ("measurement_residuals_six_panel.png", "truth_residuals_six_panel.png", "target_error_budget.png"):
        with Image.open(output / name) as picture:
            picture.verify()
