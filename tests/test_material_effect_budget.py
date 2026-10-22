import numpy as np
import pandas as pd
import pytest
import torch

from material_effect import material_budget as mb
from mlalign.helix.measurement_fit import (
    MeasurementFitInputs,
    fit_helix_measurements_gauss_newton,
    measurement_local_prediction,
)
from mlalign.helix.trajectory import helix_point

TRUE_HELIX = torch.tensor([[0.0, 0.0, 0.0, 0.3, 0.2, 1.0e-3]], dtype=torch.float64)
REF_S = (40.0, 90.0, 140.0, 200.0, 260.0, 330.0, 400.0, 480.0, 560.0)
SIGMA = (0.01, 0.05)


def _planes(path_lengths):
    """Planes through the true helix with radial normals and a z-aligned LOC1 axis."""
    s = torch.tensor(path_lengths, dtype=torch.float64)
    centers = helix_point(TRUE_HELIX.expand(len(s), -1), s)
    normal = torch.nn.functional.normalize(
        torch.stack((centers[:, 0], centers[:, 1], torch.zeros_like(s)), dim=-1), dim=-1)
    axis1 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand_as(normal)
    axis0 = torch.cross(axis1, normal, dim=-1)
    return centers, torch.stack((axis0, axis1, normal), dim=-1)


def _inputs(target_s, n_rows, generator, target_variance=None):
    centers, rotations = _planes(REF_S + target_s)
    n_hits = len(REF_S) + len(target_s)
    centers = centers.expand(n_rows, -1, -1).clone()
    rotations = rotations.expand(n_rows, -1, -1, -1).clone()
    truth = torch.zeros((n_rows, n_hits, 2), dtype=torch.float64)
    variance = torch.tensor(SIGMA, dtype=torch.float64).square().expand(n_rows, n_hits, 2).clone()
    if target_variance is not None:
        variance[:, len(REF_S):] = target_variance
    noise = torch.randn((n_rows, n_hits, 2), generator=generator, dtype=torch.float64)
    measured = truth + noise * variance.sqrt()
    fit = torch.zeros((n_rows, n_hits), dtype=torch.bool)
    fit[:, :len(REF_S)] = True
    target = ~fit
    all_components = torch.ones((n_rows, n_hits, 2), dtype=torch.bool)
    full = MeasurementFitInputs(measured, variance, all_components, centers, rotations)
    ref_only = MeasurementFitInputs(measured, variance, fit[..., None].expand(-1, -1, 2).clone(),
                                    centers, rotations)
    return full, ref_only, fit, target


def _refit(ref_only):
    initial = TRUE_HELIX.expand(ref_only.measured.shape[0], -1).clone()
    initial[:, 3] += 1e-4
    fitted, _stats = fit_helix_measurements_gauss_newton(initial, ref_only, dtype=torch.float64)
    return fitted


@pytest.mark.parametrize("target_s", [(300.0,), (800.0,)])
def test_budget_matches_monte_carlo_variance_of_the_target_prediction(target_s):
    generator = torch.Generator().manual_seed(7)
    full, ref_only, fit, target = _inputs(target_s, 4000, generator)
    fitted = _refit(ref_only)
    prediction, valid, budget, rank_deficient = mb.target_budget(fitted, full, fit, target)
    assert bool(valid.all()) and not bool(rank_deficient.any())
    empirical = prediction[:, -1].var(dim=0)
    predicted = budget[:, -1].mean(dim=0)
    # 4000 draws: the sample variance has ~2.2% relative standard error.
    assert torch.allclose(empirical, predicted, rtol=0.08)


def test_budget_ignores_target_measurements_and_variances():
    generator = torch.Generator().manual_seed(3)
    full, ref_only, fit, target = _inputs((300.0, 800.0), 8, generator)
    fitted = _refit(ref_only)
    _p, _v, budget, _r = mb.target_budget(fitted, full, fit, target)
    changed = MeasurementFitInputs(
        full.measured + target[..., None] * 0.3, full.variance * (1 + 9 * target[..., None]),
        full.component_mask, full.centers, full.rotations)
    _p, _v, budget_changed, _r = mb.target_budget(fitted, changed, fit, target)
    assert torch.allclose(budget[target], budget_changed[target], rtol=1e-9)
    assert torch.isnan(budget[fit]).all()


def test_extrapolation_budget_exceeds_interpolation():
    generator = torch.Generator().manual_seed(5)
    full, ref_only, fit, target = _inputs((300.0, 800.0), 4, generator)
    _p, _v, budget, _r = mb.target_budget(_refit(ref_only), full, fit, target)
    interpolation, extrapolation = budget[:, len(REF_S)], budget[:, len(REF_S) + 1]
    assert bool((extrapolation > 5 * interpolation).all())
    assert bool((interpolation < torch.tensor(SIGMA).square()).all())


def test_rank_deficient_reference_rows_give_no_budget():
    generator = torch.Generator().manual_seed(1)
    full, ref_only, fit, target = _inputs((300.0,), 2, generator)
    fitted = _refit(ref_only)
    fit[:, 2:] = False  # two reference hits cannot fix five parameters
    _p, _v, budget, rank_deficient = mb.target_budget(fitted, full, fit, target)
    assert bool(rank_deficient.all())
    assert torch.isnan(budget).all()


def test_prediction_is_the_local_intersection():
    generator = torch.Generator().manual_seed(2)
    full, ref_only, fit, target = _inputs((300.0,), 3, generator)
    fitted = _refit(ref_only)
    prediction, _v, _b, _r = mb.target_budget(fitted, full, fit, target, chunk_rows=2)
    expected, _valid = measurement_local_prediction(fitted, full)
    assert torch.equal(prediction, expected)


def _frame(event_ids, offset=0.0):
    rows = len(event_ids)
    return pd.DataFrame({
        "event_nr": event_ids, "complete": True,
        "measurement2": np.linspace(1, 2, rows) + offset, "truth2": np.linspace(0.5, 1, rows),
        "V": 0.5, "P": 0.25, "U": np.linspace(1, 2, rows) + offset - 0.75,
        "truth_excess": np.linspace(0.5, 1, rows) - 0.25,
    })


def test_bootstrap_resamples_the_same_events_for_every_arm():
    events = np.arange(50)
    frames = {"a": _frame(np.repeat(events, 2)), "b": _frame(np.repeat(events, 2), offset=0.1)}
    boot = mb.bootstrap_metrics(frames, events, replicates=64, seed=11)
    difference = boot["b"]["M"][1] - boot["a"]["M"][1]
    assert np.allclose(difference, 0.1)
    again = mb.bootstrap_metrics(frames, events, replicates=64, seed=11)
    assert np.array_equal(again["a"]["M"][1], boot["a"]["M"][1])


def test_bootstrap_keeps_empty_generated_events_out_of_the_means():
    events = np.arange(40)
    frame = _frame(np.arange(20))
    boot = mb.bootstrap_metrics({"a": frame}, events, replicates=16, seed=1)
    assert boot["a"]["M"][0] == pytest.approx(frame.measurement2.mean())
    assert np.isfinite(boot["a"]["M"][1]).all()
    with pytest.raises(ValueError):
        mb.bootstrap_metrics({"a": _frame(np.array([0, 99]))}, events, replicates=4)


def test_incomplete_cells_are_not_evaluable():
    frame = _frame(np.arange(5))
    frame.loc[0, "complete"] = False
    cell = mb.summarize_cell(frame)
    assert cell["complete"] is False and cell["M"] is None
    assert mb.interval_verdict(None, None, complete=False) == "not_evaluable"


def test_verdict_rules():
    assert mb.interval_verdict(1.0, [0.5, 1.5], complete=True) == "increased"
    assert mb.interval_verdict(-1.0, [-1.5, -0.5], complete=True) == "decreased"
    assert mb.interval_verdict(0.1, [-0.5, 0.7], complete=True) == "insufficient_evidence"
    assert mb.closure_verdict(0.1, [-0.5, 0.7], complete=True) == "statistically_compatible_with_zero"
    assert mb.closure_verdict(-1.0, [-1.5, -0.5], complete=True) == "not_closed"


def test_contrasts_cover_the_three_preregistered_questions():
    events = np.arange(30)
    frames = {g: _frame(np.repeat(events, 2), offset=0.1 * i) for i, g in enumerate(mb.GROUPS)}
    rows = mb.contrasts(mb.bootstrap_metrics(frames, events, replicates=32, seed=2))
    keys = {(r["question"], r["metric"]) for r in rows}
    assert keys == {("material_effect", "M"), ("material_effect", "U"), ("material_effect", "truth_M"),
                    ("high_momentum", "M"), ("high_momentum", "U"), ("high_momentum", "truth_M"),
                    ("no_material_closure", "U"), ("no_material_closure", "truth_excess")}
    material = next(r for r in rows if r["question"] == "material_effect" and r["metric"] == "M")
    assert material["difference_mm2"] == pytest.approx(-0.1)


def test_truth_states_with_repeated_surface_keys_are_dropped():
    truth = pd.DataFrame({
        "event_nr": [1, 1, 1], "track_nr": [0, 0, 0], "volume_id": [17, 17, 24],
        "layer_id": [8, 8, 4], "module_id": [5, 5, 9], "t_eLOC0": [0.1, 0.2, 0.3],
        "t_eLOC1": [1.0, 2.0, 3.0]})
    unique, repeated = mb.unique_truth(truth)
    assert repeated == 2
    assert unique.module_id.tolist() == [9]


def test_verify_arms_rejects_wrong_material_switch_and_momentum():
    base = pd.DataFrame({"event_id": [0, 1], "vx": 0.0, "vy": 0.0, "vz": [1.0, 2.0], "q": [1.0, -1.0],
                         "eta": [0.1, -0.2], "phi": [0.3, 1.0], "px": [10.0, 0.0], "py": [0.0, 10.0],
                         "pz": [1.0, -2.0]})
    pt100 = base.assign(px=base.px * 10, py=base.py * 10, pz=base.pz * 10)
    particles = dict(zip(mb.GROUPS, (base, base.copy(), pt100)))
    spreads = dict(zip(mb.GROUPS, ((1e-3, 5e-2), (0.0, 2e-7), (1e-4, 5e-3))))
    assert mb.verify_arms(particles, spreads)["passed"]
    with pytest.raises(ValueError, match="material switch"):
        mb.verify_arms(particles, {**spreads, mb.GROUPS[1]: (1e-3, 5e-2)})
    with pytest.raises(ValueError, match="momentum"):
        mb.verify_arms({**particles, mb.GROUPS[2]: base.copy()}, spreads)
