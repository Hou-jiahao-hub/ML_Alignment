from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_helix_recovery.py"
    assert path.exists(), "compare_helix_recovery.py has not been implemented"
    spec = importlib.util.spec_from_file_location("compare_helix_recovery", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
