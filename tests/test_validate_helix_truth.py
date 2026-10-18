"""Truth diagnostics must not feed held-out values back into the fit."""
import importlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts import helix_model as hm


def module():
    return importlib.import_module("scripts.validate_helix_truth")


def synthetic_track():
    m = module()
    p = torch.tensor([2., -1., .5, .2, .1, 2.e-4], dtype=torch.float64)
    path = torch.linspace(0., 600., 12, dtype=torch.float64)
    points = hm.helix_point(p.expand(12, -1), path)
    tangent = hm.helix_tangent(p.expand(12, -1), path)
    normal = torch.nn.functional.normalize(tangent, dim=-1)
    up = torch.tensor([0., 0., 1.], dtype=torch.float64).expand_as(normal)
    axis0 = torch.nn.functional.normalize(torch.cross(up, normal, dim=-1), dim=-1)
    rotations = torch.stack((axis0, torch.cross(normal, axis0, dim=-1), normal), dim=-1)
    target = torch.zeros(12, dtype=torch.bool)
    target[[0, 5, 11]] = True
    return m.TruthTrack(
        event_nr=3, track_nr=7,
        identities=[(17, 8 if target[i] else 2, i + 1, 0) for i in range(12)],
        path_length=path, centers=points.clone(), rotations=rotations,
        smt_points=points, smt_directions=tangent,
        measured=torch.zeros((12, 2), dtype=torch.float64),
        variance=torch.full((12, 2), .01, dtype=torch.float64),
        truth=torch.zeros((12, 2), dtype=torch.float64), target=target,
    )


def test_noiseless_helix_and_direct_target_truth_noninterference():
    m = module()
    track = synthetic_track()
    changed = replace(track, measured=track.measured.clone(),
                      smt_points=track.smt_points.clone(),
                      smt_directions=track.smt_directions.clone(),
                      truth=track.truth + 2.)
    changed.measured[track.target] = 999.
    changed.smt_points[track.target] = float("nan")
    changed.smt_directions[track.target] = float("nan")
    first, second = m.fit_tracks([track, changed], seed_iters=100, refit_iters=60)
    torch.testing.assert_close(first.parameters, second.parameters, atol=0, rtol=0)
    torch.testing.assert_close(first.seed_parameters, second.seed_parameters, atol=0, rtol=0)
    assert first.seed_stationary and first.refit_stationary
    surfaces, row = m.evaluate_track(track, first)
    assert row["all_complete"]
    assert row["all_truth_chi2"] < 1.e-8
    assert max(abs(s["residual_LOC0_mm"]) for s in surfaces) < 1.e-5
    _, changed_row = m.evaluate_track(changed, second)
    assert changed_row["all_truth_chi2"] > 1000


def test_chi2_masks_missing_truth_and_partial_sums():
    m = module()
    track = synthetic_track()
    fit = m.fit_tracks([track], seed_iters=100, refit_iters=60)[0]
    track.truth[:] = torch.tensor([.2, .3])
    track.measured[2, 1] = float("nan")  # strip-like LOC1: residual but no weight
    track.variance[3, 1] = 0.
    surfaces, row = m.evaluate_track(track, fit)
    assert row["all_n_components"] == 22
    assert row["all_truth_chi2"] == pytest.approx(12 * 4 + 10 * 9, abs=1.e-3)
    assert row["all_truth_chi2"] == pytest.approx(
        row["target_truth_chi2"] + row["non_target_truth_chi2"])
    assert surfaces[2]["residual_LOC1_mm"] == pytest.approx(-.3, abs=1.e-5)
    assert surfaces[2]["truth_chi2_LOC1"] is None
    track.truth[0, 0] = float("nan")
    _, partial = m.evaluate_track(track, fit)
    assert not partial["all_complete"]
    assert partial["all_truth_chi2"] is None
    assert partial["all_valid_subset_truth_chi2"] > 0
    assert partial["all_n_components"] == 21


def test_intersection_failure_and_nonstationarity_are_not_hidden():
    m = module()
    track = synthetic_track()
    fit = m.fit_tracks([track], seed_iters=1, refit_iters=1)[0]
    assert not (fit.seed_stationary and fit.refit_stationary)
    # A z-parallel plane is unreachable for this straight helix.
    straight = torch.tensor([0., 0., 0., 0., 0., 0.], dtype=torch.float64)
    track.rotations[0] = torch.eye(3, dtype=torch.float64)
    track.centers[0] = torch.tensor([0., 0., 10.])
    fit = replace(fit, parameters=straight)
    surfaces, row = m.evaluate_track(track, fit)
    assert not surfaces[0]["intersection_valid"]
    assert not row["all_complete"]
    assert row["all_truth_chi2"] is None
    assert row["fit_status"] == "non_stationary"


def test_missing_truth_fields_fail_before_fitting():
    m = module()
    with pytest.raises(ValueError, match="t_eLOC"):
        m.require_branches(set(m.STATE_BRANCHES) - {"t_eLOC0"})


def test_prediction_budget_matches_independent_finite_difference_oracle():
    m = module()
    track = synthetic_track()
    fit = m.fit_tracks([track], seed_iters=100, refit_iters=60)[0]
    predictions = m.ordered_predictions([track], [fit])[0]
    budget = m.target_prediction_budget(track, fit, predictions)
    frame = hm.reference_gauge_frame(fit.parameters[None])
    theta = hm.pack_gauge_fixed_track_parameters(fit.parameters[None])
    inputs = hm.MeasurementFitInputs(track.measured[None], track.variance[None],
                                    track.component_mask[None], track.centers[None], track.rotations[None])
    columns = []
    for k in range(5):
        delta = torch.zeros_like(theta)
        delta[0, k] = 1.e-5
        plus = hm.unpack_gauge_fixed_track_parameters(theta + delta, frame)
        minus = hm.unpack_gauge_fixed_track_parameters(theta - delta, frame)
        yp, _ = hm.measurement_local_prediction(plus, inputs)
        ym, _ = hm.measurement_local_prediction(minus, inputs)
        columns.append(((yp - ym) / 2.e-5).flatten().numpy())
    jacobian = np.stack(columns, axis=1)
    refs = np.repeat(track.fit_mask.numpy(), 2)
    targets = np.repeat(track.target.numpy(), 2)
    variance = track.variance.numpy().flatten()
    information = jacobian[refs].T @ (jacobian[refs] / variance[refs, None])
    expected = np.einsum("ij,jk,ik->i", jacobian[targets], np.linalg.inv(information), jacobian[targets])
    np.testing.assert_allclose(budget[track.target].numpy().flatten(), expected, rtol=2.e-5, atol=1.e-9)
    changed = replace(track, measured=track.measured.clone(), truth=track.truth + 100)
    changed.measured[track.target] = 500
    torch.testing.assert_close(m.target_prediction_budget(changed, fit, predictions), budget, equal_nan=True)


def test_seed_refit_timing_survives_failure(monkeypatch):
    m = module()
    timing = m.new_timings()
    def fail(*args, **kwargs):
        raise FloatingPointError("seed failure")
    monkeypatch.setattr(hm, "fit_helix_points_gauss_newton", fail)
    with pytest.raises(FloatingPointError):
        m.fit_tracks([synthetic_track()], timings=timing)
    assert timing["seed_calls"] == 1
    assert timing["seed_seconds"] >= 0
    assert timing["refit_calls"] == 0


def test_repeated_surfaces_get_occurrences_and_tracks_are_not_merged():
    m = module()
    assert m.surface_identities([17, 17, 17], [2, 4, 2], [8, 9, 8]) == [
        (17, 2, 8, 0), (17, 4, 9, 0), (17, 2, 8, 1)]


def test_cli_failure_still_finalizes_manifest(tmp_path):
    m = module()
    output = tmp_path / "failed"
    assert m.main(["--nominal-states", str(tmp_path / "missing.root"),
                   "--output-dir", str(output)]) == 1
    import json
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["schema_version"] == 2
    assert manifest["metrics"]["error"]


def test_input_output_collision_is_rejected_without_writing(tmp_path):
    m = module()
    output = tmp_path / "collision"
    with pytest.raises(SystemExit):
        m.main(["--nominal-states", str(output / "manifest.json"),
                "--output-dir", str(output)])
    assert not output.exists()


def test_predictions_are_invariant_to_helix_origin_shift():
    m = module()
    track = synthetic_track()
    fit = m.fit_tracks([track], seed_iters=100, refit_iters=60)[0]
    shifted = fit.parameters.clone()
    shift = torch.tensor(120., dtype=torch.float64)
    shifted[:3] = hm.helix_point(fit.parameters, shift)
    shifted[3] += shifted[5] * shift
    first, _ = m.evaluate_track(track, fit)
    second, _ = m.evaluate_track(track, replace(fit, parameters=shifted))
    for a, b in zip(first, second):
        assert a["intersection_valid"] and b["intersection_valid"]
        assert a["prediction_LOC0_mm"] == pytest.approx(b["prediction_LOC0_mm"], abs=1.e-9)
        assert a["path_mm"] == pytest.approx(b["path_mm"] + 120., abs=1.e-9)


def test_empty_target_sum_preserves_additivity_without_claiming_coverage():
    m = module()
    track = synthetic_track()
    track.target[:] = False
    fit = m.fit_tracks([track], seed_iters=100, refit_iters=60)[0]
    _, row = m.evaluate_track(track, fit)
    assert row["target_truth_chi2"] == 0.
    assert row["target_n_components"] == 0
    assert row["target_truth_chi2_per_component"] is None
    assert not row["target_complete"]
    assert row["all_truth_chi2"] == row["non_target_truth_chi2"]


def test_invalid_truth_score_overflow_cannot_be_complete():
    m = module()
    track = synthetic_track()
    fit = m.fit_tracks([track], seed_iters=100, refit_iters=60)[0]
    track.truth[0, 0] = 1.e200
    _, row = m.evaluate_track(track, fit)
    assert row["all_truth_chi2"] is None
    assert not row["all_complete"]


def test_package_import_without_acts_or_pythonpath():
    import os
    import subprocess
    import sys
    from pathlib import Path
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'acts' or name.startswith('acts.'):
        raise AssertionError('ACTS dependency in diagnostic')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import scripts.validate_helix_truth
"""
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, cwd=Path(__file__).resolve().parents[1])
    assert completed.returncode == 0, completed.stderr


@pytest.mark.root_io
def test_root_contract_with_compact_errors_and_repeated_surface(tmp_path):
    m = module()
    import awkward as ak
    import uproot
    import json
    track = synthetic_track()
    payload = {
        "event_nr": np.array([track.event_nr]), "track_nr": np.array([track.track_nr]),
        "volume_id": ak.Array([[k[0] for k in track.identities]]),
        "layer_id": ak.Array([[k[1] for k in track.identities]]),
        "module_id": ak.Array([[k[2] for k in track.identities]]),
        "pathLength": ak.Array([track.path_length.tolist()]),
    }
    for names, values in [(("g_x_smt", "g_y_smt", "g_z_smt"), track.smt_points),
                          (("px_smt", "py_smt", "pz_smt"), track.smt_directions),
                          (("l_x_hit", "l_y_hit"), track.measured),
                          (("t_eLOC0", "t_eLOC1"), track.truth),
                          (("err_x_hit", "err_y_hit"), track.variance.sqrt())]:
        for j, name in enumerate(names):
            payload[name] = ak.Array([values[:, j].tolist()])
    # Last state revisits the first surface. Add a nonmeasurement hole while
    # retaining compact errors (12 errors for 13 state records).
    payload["module_id"] = ak.Array([[*payload["module_id"][0][:-1].to_list(), 1]])
    for name in m.STATE_BRANCHES:
        values = payload[name][0].to_list()
        values.insert(3, 0 if name in ("volume_id", "layer_id", "module_id") else float("nan"))
        payload[name] = ak.Array([values])
    root = tmp_path / "states.root"
    with uproot.recreate(root) as output:
        output.mktree("trackstates", payload)
    geo = tmp_path / "geometry.json"
    geo.write_text(json.dumps([
        {"ID": f"vol={k[0]}|lay={k[1]}|sen={k[2]}",
         "Translation": track.centers[i].tolist(), "RotationMatrix": track.rotations[i].tolist()}
        for i, k in enumerate(track.identities)]))
    loaded, excluded, counts = m.load_tracks(root, geo, ((17, 8),), max_events=25)
    assert len(loaded) == 1 and not excluded
    torch.testing.assert_close(loaded[0].truth, track.truth)
    assert loaded[0].identities[-1] == (17, 8, 1, 1)
    assert counts["selected_events"] == 1


def test_geometry_cli_preserves_nominal_default_and_requires_reference(tmp_path):
    m = module()
    parser = m.build_parser(m.load_project_config())
    assert parser.parse_args([]).geometry == "nominal"
    assert parser.parse_args(["--geometry", "misaligned"]).geometry == "misaligned"
    with pytest.raises(SystemExit):
        m.main(["--geometry", "misaligned", "--output-dir", str(tmp_path / "unused")])
    assert not (tmp_path / "unused").exists()


def test_misaligned_pair_uses_only_branch_smt_and_nominal_reference():
    from dataclasses import replace
    m = module()
    nominal = synthetic_track()
    shifted = replace(nominal, smt_points=nominal.smt_points + .1,
                      truth=nominal.truth + 999., centers=nominal.centers + 30.)
    paired = m.merge_misaligned_track(nominal, shifted)
    torch.testing.assert_close(paired.smt_points, shifted.smt_points)
    torch.testing.assert_close(paired.centers, nominal.centers)
    torch.testing.assert_close(paired.truth, nominal.truth)
    changed = replace(shifted, measured=shifted.measured.clone())
    changed.measured[0, 0] += 1.
    with pytest.raises(ValueError, match="measurement"):
        m.merge_misaligned_track(nominal, changed)
    changed = replace(shifted, identities=shifted.identities[:-1])
    with pytest.raises(ValueError, match="identit"):
        m.merge_misaligned_track(nominal, changed)


def test_misaligned_target_smt_and_truth_cannot_affect_fitted_parameters():
    from dataclasses import replace
    m = module()
    nominal = synthetic_track()
    changed = replace(nominal, smt_points=nominal.smt_points.clone(),
                      smt_directions=nominal.smt_directions.clone(), truth=nominal.truth + 100.)
    changed.smt_points[nominal.target] = float("nan")
    changed.smt_directions[nominal.target] = float("nan")
    first, second = m.fit_tracks([m.merge_misaligned_track(nominal, nominal),
                                m.merge_misaligned_track(nominal, changed)],
                               seed_iters=100, refit_iters=60)
    torch.testing.assert_close(first.parameters, second.parameters, atol=0, rtol=0)


def test_paired_chi2_hand_calculation_and_fail_closed_stationarity():
    import pandas as pd
    from scripts import plot_helix_truth_comparison as comparison
    m = module()
    rows = []
    for event in (1, 2):
        for target in (False, True):
            row = dict(event_nr=event, track_nr=0, target=target, intersection_valid=True)
            for a in range(2):
                row.update({f"truth_LOC{a}_mm": 0., f"prediction_LOC{a}_mm": 1.,
                            f"acts_LOC{a}_mm": 2., f"weighted_LOC{a}": True,
                            f"variance_LOC{a}_mm2": 4.})
            rows.append(row)
    tracks = pd.DataFrame({"event_nr": [1, 2], "track_nr": [0, 0],
                           "seed_stationary": [False, True], "refit_stationary": [True, True]})
    data, _ = comparison.analyze(pd.DataFrame(rows), tracks)
    paired, summary = m.paired_chi2_comparison(data, tracks, replicates=100, seed=6006)
    assert paired.all_helix_chi2.tolist() == [1., 1.]
    assert paired.all_nominal_acts_chi2.tolist() == [4., 4.]
    assert paired.all_delta_chi2.tolist() == [-3., -3.]
    assert summary["verdict"] == "improved"
    assert summary["groups"]["all"]["mean_delta_ci95"] == [-3., -3.]
    assert summary["groups"]["all"]["helix_sum_chi2"] == 2.
    assert (paired.all_helix_chi2 == paired.target_helix_chi2 + paired.non_target_helix_chi2).all()
    tracks.loc[0, "refit_stationary"] = False
    _, summary = m.paired_chi2_comparison(data, tracks, replicates=100, seed=6006)
    assert summary["verdict"] == "inconclusive"
    assert summary["groups"]["all"]["valid_subset"]["tracks"] == 1
    assert summary["groups"]["all"]["helix_sum_chi2"] is None

    tracks.loc[0, "refit_stationary"] = True
    _, summary = m.paired_chi2_comparison(data, tracks, replicates=100, seed=6006,
                                         frozen_tracks=3)
    assert summary["verdict"] == "inconclusive"
    assert summary["cohort_scope"] == "diagnostic_subset"
    assert summary["groups"]["all"]["helix_sum_chi2"] is None


def test_misaligned_missing_reference_finalizes_manifest(tmp_path):
    import json
    m = module()
    output = tmp_path / "failed"
    assert m.main(["--geometry", "misaligned", "--nominal-run-dir", str(tmp_path / "missing"),
                   "--output-dir", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["metrics"]["effective_experiment_config"]["geometry"] == "misaligned"


def test_bootstrap_resamples_whole_events_and_missing_component_is_inconclusive():
    import pandas as pd
    m = module()
    tracks = pd.DataFrame({"event_nr": [1, 1], "track_nr": [0, 1],
                           "seed_stationary": [True, True], "refit_stationary": [True, True]})
    raw = {"Helix": {"Q": np.array([1., 5.]), "N": np.array([2, 2]),
                     "complete": np.array([True, True])},
           "ACTS": {"Q": np.array([4., 4.]), "N": np.array([2, 2]),
                    "complete": np.array([True, True])}}
    data = {"track_raw": {g: raw for g in m.GROUPS}}
    _, summary = m.paired_chi2_comparison(data, tracks, replicates=100)
    assert summary["groups"]["all"]["mean_delta_ci95"] == [-1., -1.]
    raw["ACTS"]["complete"][0] = False
    rows, summary = m.paired_chi2_comparison(data, tracks, replicates=100)
    assert len(rows) == 2 and not rows.all_complete.iloc[0]
    assert summary["verdict"] == "inconclusive"


@pytest.mark.root_io
def test_misaligned_cli_end_to_end_uses_nominal_truth_and_acts(tmp_path):
    import json
    import pandas as pd
    import awkward as ak
    import uproot
    m = module()
    track = synthetic_track()
    nominal_root, branch_root = tmp_path / "nominal.root", tmp_path / "misaligned.root"
    payload = {"event_nr": np.array([3]), "track_nr": np.array([7]),
               "volume_id": ak.Array([[k[0] for k in track.identities]]),
               "layer_id": ak.Array([[k[1] for k in track.identities]]),
               "module_id": ak.Array([[k[2] for k in track.identities]]),
               "pathLength": ak.Array([track.path_length.tolist()])}
    for names, values in [(("g_x_smt", "g_y_smt", "g_z_smt"), track.smt_points),
                          (("px_smt", "py_smt", "pz_smt"), track.smt_directions),
                          (("l_x_hit", "l_y_hit"), track.measured),
                          (("t_eLOC0", "t_eLOC1"), track.truth),
                          (("err_x_hit", "err_y_hit"), track.variance.sqrt()),
                          (("eLOC0_smt", "eLOC1_smt"), torch.full((12, 2), .2))]:
        for a, name in enumerate(names):
            payload[name] = ak.Array([values[:, a].tolist()])
    with uproot.recreate(nominal_root) as root:
        root.mktree("trackstates", payload)
    for a, name in enumerate(("g_x_smt", "g_y_smt", "g_z_smt")):
        values = track.smt_points[:, a] + .1
        values[track.target] = float("nan")
        payload[name] = ak.Array([values.tolist()])
    for name in ("t_eLOC0", "t_eLOC1"):
        payload[name] = ak.Array([[999.] * 12])
    with uproot.recreate(branch_root) as root:
        root.mktree("trackstates", payload)
    geometry = [{"ID": f"vol={k[0]}|lay={k[1]}|sen={k[2]}",
                 "Translation": track.centers[i].tolist(),
                 "RotationMatrix": track.rotations[i].tolist()}
                for i, k in enumerate(track.identities)]
    nominal_geometry, misaligned_geometry = tmp_path / "nominal.json", tmp_path / "misaligned.json"
    nominal_geometry.write_text(json.dumps(geometry))
    misaligned_geometry.write_text(json.dumps(geometry))
    nominal_run, output = tmp_path / "nominal", tmp_path / "misaligned"
    common = ["--nominal-states", str(nominal_root), "--nominal-geometry", str(nominal_geometry),
              "--seed-iters", "100", "--refit-iters", "60"]
    assert m.main([*common, "--output-dir", str(nominal_run)]) == 0
    args = [*common, "--geometry", "misaligned", "--nominal-run-dir", str(nominal_run),
            "--misaligned-states", str(branch_root), "--misaligned-geometry", str(misaligned_geometry)]
    assert m.main([*args, "--output-dir", str(output)]) == 0
    scores = pd.read_csv(output / "paired_comparison.csv")
    assert scores.all_nominal_acts_chi2.iloc[0] == pytest.approx(96., rel=1.e-6)
    assert scores.all_helix_chi2.iloc[0] < 1.e-6
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["metrics"]["summary"]["paired_comparison"]["verdict"] == "improved"
    assert (output / "acts_comparison" / "acts_vs_helix_chi2.pdf").is_file()
    from scripts.plot_helix_truth_comparison import sha256
    child = json.loads((output / "acts_comparison" / "manifest.json").read_text())
    assert child["inputs"]["source_manifest"]["sha256"] == sha256(output / "manifest.json")
    geometry[1]["Translation"][0] += .01  # A non-target reference changed.
    misaligned_geometry.write_text(json.dumps(geometry))
    failed = tmp_path / "bad_geometry"
    assert m.main([*args, "--output-dir", str(failed)]) == 1
    failure = json.loads((failed / "manifest.json").read_text())
    assert "Non-target" in failure["metrics"]["error"]
