from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
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
    cohort = module.PairedRecoveryCohort(
        nominal=inputs,
        misaligned=inputs,
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
            acts_smt=module.ACTSSMTStage(inputs.smt_local[:, None], inputs.component_mask[:, None], cohort.hit_mask),
            lm_seed=nominal_seed,
            gauss_newton=stage(1.0, rho=0.001, angle=0.01),
        )
        arms[("misaligned", exposure)] = module.RecoveryArm(
            fit_mask=cohort.protected_fit_mask,
            acts_smt=module.ACTSSMTStage(inputs.smt_local[:, None], inputs.component_mask[:, None], cohort.hit_mask),
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
    assert group["recovery_ratio_q_ci95"] == pytest.approx({"low": 0.025, "high": 0.025})
    assert group["practical_equivalence_ratio"] == pytest.approx(1.05)
    assert group["practical_equivalence_ratio_ci95"] == pytest.approx({"low": 1.05, "high": 1.05})
    assert group["refit_branch_difference_bias_mm_ci95"] == pytest.approx({"low": 0.05, "high": 0.05})
    assert group["direction_angle_branch_difference_rms_rad"] == pytest.approx(0.01)
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
    assert group["recovery_ratio_q_ci95"] == {"low": None, "high": None}
    assert group["verdict"] == "failed"
