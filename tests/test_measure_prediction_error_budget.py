from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


REF_X = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0)
# Local frame (u, v, n) = (y, z, x): planes normal to x, measuring y and z.
ROTATION = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)


def _module():
    return importlib.import_module("scripts.analysis.measure_prediction_error_budget")


def _geometry(budget, target_x, target_shift_u=0.0, ref_shift_u=0.0):
    """Ref planes at REF_X, one target plane per entry of ``target_x``."""
    m = budget.M
    xs = list(REF_X) + list(target_x)
    centers = torch.zeros((len(xs), 3), dtype=torch.float64)
    centers[:, 0] = torch.tensor(xs, dtype=torch.float64)
    target_mask = torch.zeros(len(xs), dtype=torch.bool)
    target_mask[len(REF_X):] = True
    centers[target_mask, 1] += target_shift_u
    centers[~target_mask, 1] += ref_shift_u
    return m.JointGeometry(
        transforms=[],
        input_transforms=[],
        centers=centers,
        rotations=ROTATION.expand(len(xs), 3, 3).clone(),
        target_mask=target_mask,
        id_to_idx={},
        input_indices=[],
    )


def _batches(budget, measured, variance, raw_variance, target_sensor, loc1_target=None):
    """One row per track: the 7 ref hits in REF_X order plus one target hit."""
    m = budget.M
    n_tracks = measured.shape[0]
    n_ref = len(REF_X)
    target_x = [float(x) for x in target_sensor]
    order = []
    for track in range(n_tracks):
        slots = list(range(n_ref))
        position = int(np.searchsorted(REF_X, target_x[track]))
        slots.insert(position, n_ref)
        order.append(slots)
    order = torch.tensor(order)
    sensor = torch.where(
        order == n_ref,
        torch.tensor([n_ref + i for i in range(n_tracks)])[:, None],
        order,
    )
    is_target = order == n_ref
    loc1 = torch.ones_like(is_target)
    if loc1_target is not None:
        loc1[is_target] = torch.as_tensor(loc1_target)
    gather = order.unsqueeze(-1).expand(-1, -1, 2)
    volume = torch.where(is_target, torch.tensor(17), torch.tensor(10))
    layer = torch.where(is_target, torch.tensor(8), order + 1)
    return m.TrackBatches(
        measured_local=torch.gather(measured, 1, gather),
        covariances=torch.gather(variance, 1, gather),
        sensor_idx=sensor,
        hit_mask=torch.ones_like(is_target),
        loc1_mask=loc1,
        g_smt=torch.zeros((n_tracks, n_ref + 1, 3), dtype=torch.float64),
        p_smt=torch.zeros((n_tracks, n_ref + 1, 3), dtype=torch.float64),
        volume_id=volume,
        layer_id=layer,
        is_target=is_target,
        event_track_ids=torch.stack((torch.zeros(n_tracks, dtype=torch.long), torch.arange(n_tracks)), 1),
        fit_mask=~is_target,
        raw_covariances=torch.gather(raw_variance, 1, gather),
    )


def _weighted_prediction_variance(design_columns, variance, target_row):
    design = np.stack(design_columns, axis=1)
    information = design.T @ (design / variance[:, None])
    return float(target_row @ np.linalg.solve(information, target_row))


def test_prediction_regime_codes_interpolation_and_both_extrapolations():
    budget = _module()
    fit = torch.tensor([[True, False, True, False], [False, True, True, True]])
    target = torch.tensor([[False, True, False, True], [True, False, False, False]])

    regime = budget.prediction_regime(fit, target)

    assert regime.tolist() == [[-1, 0, -1, 1], [2, -1, -1, -1]]
    with pytest.raises(ValueError, match="no fit hit"):
        budget.prediction_regime(torch.zeros((1, 2), dtype=torch.bool), torch.ones((1, 2), dtype=torch.bool))


def test_propagated_variance_matches_independent_weighted_least_squares():
    budget = _module()
    target_x = (450.0, 950.0)
    n_hits = len(REF_X) + 1
    rng = np.random.default_rng(3)
    variance_hits = rng.uniform(0.005, 0.05, size=(2, n_hits, 2))
    # Hit slot k < 7 is ref plane k; slot 7 is the target, matching _batches' gather.
    variance = torch.tensor(variance_hits)
    raw = variance.clone()
    raw[0, len(REF_X), 1] *= 0.5
    measured = torch.zeros((2, n_hits, 2), dtype=torch.float64)
    batches = _batches(budget, measured, variance, raw, target_x)
    geometry = _geometry(budget, target_x)

    table, diagnostics = budget.component_budget(
        torch.zeros((2, 6), dtype=torch.float64), batches, geometry, geometry
    )

    assert diagnostics["n_rank_deficient_rows"] == 0
    assert len(table) == 4
    ref = np.asarray(REF_X)
    for track, x_t in enumerate(target_x):
        v_ref = variance_hits[track, : len(REF_X)]
        expected_u = _weighted_prediction_variance(
            [np.ones_like(ref), ref, ref**2], v_ref[:, 0], np.array([1.0, x_t, x_t**2])
        )
        expected_v = _weighted_prediction_variance(
            [np.ones_like(ref), ref], v_ref[:, 1], np.array([1.0, x_t])
        )
        rows = table[table.track == track].set_index("component")
        assert rows.loc["LOC0", "P"] == pytest.approx(expected_u, rel=1e-7)
        assert rows.loc["LOC1", "P"] == pytest.approx(expected_v, rel=1e-7)
        assert rows.loc["LOC0", "V"] == pytest.approx(variance_hits[track, -1, 0])
    regimes = table.groupby("track")["regime"].first().tolist()
    assert regimes == ["interpolation", "forward_extrapolation"]
    assert table.floor_active.tolist() == [False, True, False, False]


def test_one_dimensional_target_contributes_only_loc0():
    budget = _module()
    measured = torch.zeros((1, len(REF_X) + 1, 2), dtype=torch.float64)
    variance = torch.full_like(measured, 0.01)
    batches = _batches(budget, measured, variance, variance.clone(), (450.0,), loc1_target=[False])
    geometry = _geometry(budget, (450.0,))

    table, _ = budget.component_budget(torch.zeros((1, 6), dtype=torch.float64), batches, geometry, geometry)

    assert table.component.tolist() == ["LOC0"]


def test_decomposition_closes_when_the_single_helix_model_is_true():
    budget = _module()
    n_tracks = 1500
    generator = torch.Generator().manual_seed(11)
    truth = torch.zeros((n_tracks, 6), dtype=torch.float64)
    truth[:, 1] = torch.randn(n_tracks, generator=generator, dtype=torch.float64)
    truth[:, 2] = torch.randn(n_tracks, generator=generator, dtype=torch.float64)
    truth[:, 3] = 0.01 * torch.randn(n_tracks, generator=generator, dtype=torch.float64)
    truth[:, 4] = 0.01 * torch.randn(n_tracks, generator=generator, dtype=torch.float64)
    truth[:, 5] = 1.0e-5 * torch.randn(n_tracks, generator=generator, dtype=torch.float64)
    target_x = tuple(float(x) for x in np.where(np.arange(n_tracks) % 2, 450.0, 950.0))
    geometry = _geometry(budget, target_x)
    n_hits = len(REF_X) + 1
    variance = torch.full((n_tracks, n_hits, 2), 0.01, dtype=torch.float64)
    variance[..., 1] = 0.04
    placeholder = torch.zeros((n_tracks, n_hits, 2), dtype=torch.float64)
    batches = _batches(budget, placeholder, variance, variance.clone(), target_x)
    inputs = budget._fit_inputs(batches, geometry, slice(0, n_tracks))
    exact, valid = budget.measurement_local_prediction(truth, inputs)
    assert bool(valid.all())
    noise = torch.randn(exact.shape, generator=generator, dtype=torch.float64)
    batches.measured_local = exact + noise * batches.covariances.sqrt()

    refitted, stats = budget.M.fit_helices_to_measurements(
        batches, geometry, truth, dtype=torch.float64, stationarity=False
    )
    assert stats["stationarity_gate"]["non_stationary_row_fraction"] == 0.0
    shifted = _geometry(budget, target_x, target_shift_u=0.2)
    table, _ = budget.component_budget(refitted, batches, geometry, shifted)
    cells, strata = budget.aggregate(table)

    assert sum(cell["n_components"] for cell in cells) == len(table) == 2 * n_tracks
    assert sum(stratum["n_components"] for stratum in strata) == len(table)
    extrapolation = [s for s in strata if s["regime"] == "forward_extrapolation"]
    for stratum in strata:
        # Correct model: E[r^2] = V + P, so the pull with P is unit width.
        assert stratum["pull2_with_P"] == pytest.approx(1.0, abs=0.12)
        assert abs(stratum["U"]) < 0.15 * (stratum["V_mean"] + stratum["P_mean"])
        if stratum["component"] == "LOC0":
            assert stratum["M_comparison"] > stratum["M_budget"] + 0.03
        else:
            assert stratum["M_comparison"] == pytest.approx(stratum["M_budget"], rel=1e-9)
    for stratum in extrapolation:
        assert stratum["R_stat"] > 0.5
        assert stratum["pull2_without_P"] > 1.4


def test_geometries_must_share_non_target_surfaces():
    budget = _module()
    measured = torch.zeros((1, len(REF_X) + 1, 2), dtype=torch.float64)
    variance = torch.full_like(measured, 0.01)
    batches = _batches(budget, measured, variance, variance.clone(), (450.0,))

    with pytest.raises(ValueError, match="non-target"):
        budget.component_budget(
            torch.zeros((1, 6), dtype=torch.float64),
            batches,
            _geometry(budget, (450.0,)),
            _geometry(budget, (450.0,), ref_shift_u=1e-6),
        )


def test_component_budget_requires_raw_variance():
    budget = _module()
    measured = torch.zeros((1, len(REF_X) + 1, 2), dtype=torch.float64)
    variance = torch.full_like(measured, 0.01)
    batches = _batches(budget, measured, variance, variance.clone(), (450.0,))
    batches.raw_covariances = None
    geometry = _geometry(budget, (450.0,))

    with pytest.raises(ValueError, match="raw_covariances"):
        budget.component_budget(torch.zeros((1, 6), dtype=torch.float64), batches, geometry, geometry)


def _cell(r_stat, u, p=1.0, v=10.0):
    return {"layer": "17:8", "component": "LOC0", "R_stat": r_stat, "U": u, "P_mean": p, "V_mean": v}


@pytest.mark.parametrize(
    ("cells", "outcome"),
    [
        ([_cell(0.10, 0.5)], "outcome_1_statistical_important"),
        ([_cell(0.0999, 0.5)], "outcome_3_both_small"),
        ([_cell(0.5, 2.0)], "outcome_2_model_error_dominates"),
        # U exceeds P but stays below 10% of V: not model dominated.
        ([_cell(0.5, 0.9, p=0.5, v=10.0)], "outcome_1_statistical_important"),
        ([_cell(0.01, 2.0), _cell(0.3, 0.1)], "outcome_1_statistical_important"),
        ([_cell(0.01, 2.0), _cell(0.01, 0.1)], "outcome_2_model_error_dominates"),
    ],
)
def test_classify_applies_the_preregistered_rule(cells, outcome):
    assert _module().classify(cells)["outcome"] == outcome


def test_identity_gate_fails_closed_on_full_runs_and_flags_subsets():
    budget = _module()
    same = {key: {"sha256": "a"} for key in budget.BASELINE_INPUT_KEYS}
    other = dict(same, states={"sha256": "b"})

    assert budget.identity_gate(10, 20, full_row_set=False) == "not_applicable_subset"
    assert (
        budget.identity_gate(
            budget.EXPECTED_RETAINED_TRACKS,
            budget.EXPECTED_REFIT_COMPONENTS,
            full_row_set=True,
            manifest_inputs=same,
            baseline_inputs=same,
        )
        == "passed"
    )
    with pytest.raises(ValueError, match="Retained tracks"):
        budget.identity_gate(1, budget.EXPECTED_REFIT_COMPONENTS, full_row_set=True)
    with pytest.raises(ValueError, match="Refit components"):
        budget.identity_gate(budget.EXPECTED_RETAINED_TRACKS, 1, full_row_set=True)
    with pytest.raises(ValueError, match="states"):
        budget.identity_gate(
            10, 20, full_row_set=False, manifest_inputs=other, baseline_inputs=same
        )
