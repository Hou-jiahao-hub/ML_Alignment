from __future__ import annotations

import importlib

import pytest
import torch

import joint_schur_synthetic as syn


def _module():
    return importlib.import_module("scripts.analysis.measure_track_block_conditioning")


def _problem():
    truth = syn.make_tracks(10, seed=5)
    topology = [[0, 1], [0, 1, 2], [1, 2], [2, 3], [0, 3], [0, 1, 2, 3], [1], [3], [0, 2], [1, 3]]
    return syn.make_problem(
        truth,
        targets=((450.0, 0.0), (550.0, 15.0), (650.0, 75.0), (850.0, 0.0)),
        topology=topology,
        loc1_targets=(3,),
        injection=[[0.02, -0.01, 0.0, 0.0, 0.0, 2.0e-4]] * 4,
        noise_seed=7,
    )


def test_closure_passes_and_covers_every_stratum_on_synthetic_tracks():
    module = _module()
    problem = _problem()
    parameters = syn.ref_only_fit(problem)

    closure = module.jacobian_closure(problem.batches, problem.geometry, parameters)

    assert closure["passed"]
    for name in ("ref_LOC0", "ref_LOC1", "target_near_parallel_LOC0", "target_strip_1d_LOC0"):
        assert closure["strata"][name]["n"] > 0
        assert closure["strata"][name]["max_tolerance_ratio"] <= 1.0
    # 1D strips have no LOC1 component at all.
    assert closure["strata"]["target_strip_1d_LOC1"]["n"] == 0


def test_closure_reports_a_wrong_jacobian(monkeypatch):
    module = _module()
    problem = _problem()
    parameters = syn.ref_only_fit(problem)
    real = module.J.linearize_tracks

    def broken(*args, **kwargs):
        lin = real(*args, **kwargs)
        jacobian = lin.track_jacobian.clone()
        jacobian[..., 4] *= 1.0 + 1.0e-4
        return type(lin)(**{**vars(lin), "track_jacobian": jacobian})

    monkeypatch.setattr(module.J, "linearize_tracks", broken)
    closure = module.jacobian_closure(problem.batches, problem.geometry, parameters)
    assert not closure["passed"]
    assert closure["n_failed"] > 0


def test_track_block_summary_counts_components_and_blocks():
    module = _module()
    problem = _problem()
    parameters = syn.ref_only_fit(problem)

    table, summary = module.track_blocks(problem.batches, problem.geometry, parameters)

    lin_mask = problem.batches.hit_mask
    assert summary["n_tracks"] == len(table) == 10
    assert summary["n_ref_components"] == int(2 * (lin_mask & ~problem.batches.is_target).sum())
    strip_hits = int((problem.batches.is_target & ~problem.batches.loc1_mask).sum())
    assert summary["n_target_components"] == int(2 * problem.batches.is_target.sum()) - strip_hits
    for name in ("gamma_ref", "gamma"):
        assert summary[name]["n_cholesky_failed"] == 0
        assert summary[name]["n_non_finite"] == 0
        assert summary[name]["rank_counts"] == {"5": 10}
    assert summary["n_ref_non_stationary"] == 0


def test_select_rows_keeps_every_tensor_field_aligned():
    module = _module()
    batches = _problem().batches
    index = torch.tensor([1, 4, 7])
    subset = module.select_rows(batches, index)
    assert torch.equal(subset.measured_local, batches.measured_local[index])
    assert torch.equal(subset.raw_covariances, batches.raw_covariances[index])
    assert torch.equal(subset.event_track_ids, batches.event_track_ids[index])


def test_inputs_must_match_the_baseline_manifest():
    module = _module()
    own = {"inputs": {key: {"sha256": "a"} for key in module.BASELINE_INPUT_KEYS}}
    baseline = {"inputs": {key: {"sha256": "a"} for key in module.BASELINE_INPUT_KEYS}}
    module.check_baseline_inputs(own, baseline)
    baseline["inputs"]["states"]["sha256"] = "b"
    with pytest.raises(ValueError, match="states"):
        module.check_baseline_inputs(own, baseline)


def test_full_row_set_is_refused():
    module = _module()
    args = module.build_parser().parse_args(
        [
            "--input-geom", "g", "--nominal-full-geom", "n", "--states", "s", "--summary", "m",
            "--baseline-manifest", "b", "--output-dir", "o", "--run-id", "r", "--max-tracks", "0",
        ]
    )
    with pytest.raises(ValueError, match="subset"):
        module.run(args)
