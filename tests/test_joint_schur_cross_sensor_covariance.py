"""P5: Monte-Carlo closure of the cross-sensor target residual covariance
(Iteration 36).

With geometry at the truth and tracks fitted to ref hits only, the whitened
target residuals of one track are predicted to have covariance
``Sigma = I + B_tgt Gamma_ref^-1 B_tgt^T``; its off-diagonal sensor blocks are
the coupling the Schur ``S_ij`` is built from.  ``Sigma`` is assembled here
from the module's rows, independently of its elimination path.  The closure
checks algebra and linearisation, not the single-helix model.
"""

from __future__ import annotations

import functools
import itertools
import math

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn
from helix_model import (
    MEASUREMENT_NEWTON_DECREMENT_STATIONARY,
    MeasurementFitInputs,
    measurement_newton_decrement,
)

N_REPLICAS = 16000
SEED = 20261012
REF_SPAN = (100.0, 800.0)
GN_DECREMENT = 1.0e-12
GN_MAX_ITERATIONS = 20
N_PRODUCTION_CHECK = 200
# Every threshold is 4.5 Monte-Carlo standard errors at N_REPLICAS.
Z_LIMIT = 4.5
MEAN_LIMIT = Z_LIMIT / math.sqrt(N_REPLICAS)
STD_LIMIT = Z_LIMIT / math.sqrt(2 * N_REPLICAS)
CORR_LIMIT = Z_LIMIT / math.sqrt(N_REPLICAS)
# (targets as (x, yaw), 1D-strip targets); ref planes span 100-800 mm.
LAYOUTS = {
    "interpolation": (((350.0, 0.0), (450.0, 0.0), (550.0, 15.0)), ()),
    "extrapolation": (((850.0, 0.0), (950.0, 15.0), (1000.0, 0.0)), ()),
    "mixed": (((450.0, 0.0), (550.0, 15.0), (950.0, 0.0)), (1,)),
}
CELLS = list(itertools.product((8, 16), (0.2, 1.5), LAYOUTS))
# Cells whose noise-free separation is >= 2 * Z_LIMIT (Iteration 36 §A).
POWER_CELLS = {
    "drop_cross_sensor": set(range(len(CELLS))),
    "target_inclusive_gamma": set(range(len(CELLS))) - {6, 9},
}


def _problem(cell: int, n: int, noisy: bool):
    n_ref, tanl, layout = CELLS[cell]
    targets, strips = LAYOUTS[layout]
    truth = syn.make_tracks(1, seed=SEED + cell, tanl=tanl).expand(n, 6).clone()
    return syn.make_problem(
        truth,
        targets=targets,
        n_ref=n_ref,
        ref_span=REF_SPAN,
        loc1_targets=strips,
        noise_seed=SEED + cell if noisy else None,
    )


def _ref_only_gauss_newton(problem):
    """Ref-only GN on the module's ref rows from the truth; tracks never see target hits."""
    parameters = problem.truth
    for _ in range(GN_MAX_ITERATIONS):
        lin = J.linearize_tracks(parameters, problem.batches, problem.geometry)
        blocks = J.normal_blocks(lin)
        step = -torch.linalg.solve(blocks.gamma_ref, blocks.beta_ref)
        decrement = -(blocks.beta_ref * step).sum(1)
        if float(decrement.max()) <= GN_DECREMENT:
            return lin, blocks, decrement
        parameters = J.commit_track_step(parameters, step)
    raise AssertionError(f"ref-only GN not stationary: max decrement {float(decrement.max()):.3e}")


def _propagated(B: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    factor = torch.linalg.cholesky(gamma)
    return B @ torch.cholesky_solve(B.transpose(-1, -2), factor)


@functools.lru_cache(maxsize=None)
def _closure(cell: int, n: int = N_REPLICAS, noisy: bool = True):
    problem = _problem(cell, n, noisy)
    lin, blocks, decrement = _ref_only_gauss_newton(problem)
    selected = lin.component_mask & lin.is_target.unsqueeze(-1)
    slots = selected[0]
    assert bool((selected == slots).all()), "replicas must share one component set"
    B = lin.track_jacobian[:, slots]
    eye = torch.eye(B.shape[1], dtype=torch.float64)
    return {
        "problem": problem,
        "parameters": lin.parameters,
        "decrement": decrement,
        "residual": lin.residual[:, slots],
        "sigma": eye + _propagated(B, blocks.gamma_ref),
        "sigma_target_inclusive": eye + _propagated(B, blocks.gamma),
        "sensor": lin.block_index[0].unsqueeze(-1).expand(-1, 2)[slots],
        "loc": torch.arange(2).expand_as(slots)[slots],
    }


def _correlation(covariance: torch.Tensor) -> torch.Tensor:
    scale = covariance.diagonal().sqrt()
    return covariance / scale[:, None] / scale[None, :]


def _cross_sensor_pairs(sensor: torch.Tensor) -> torch.Tensor:
    upper = torch.ones((len(sensor), len(sensor)), dtype=torch.bool).triu(1)
    return upper & (sensor[:, None] != sensor[None, :])


def _fisher_z(residual: torch.Tensor, predicted: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """``|atanh rho_mc - atanh rho_pred| sqrt(N - 3)`` on ``pairs``."""
    observed = torch.corrcoef(residual.T)[pairs]
    return (torch.atanh(observed) - torch.atanh(predicted[pairs])).abs() * math.sqrt(len(residual) - 3)


def _drop_cross_sensor(covariance: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
    return covariance * (sensor[:, None] == sensor[None, :])


def _production_decrement(data, sample: torch.Tensor) -> torch.Tensor:
    batches = data["problem"].batches
    geometry = data["problem"].geometry
    sensor = batches.sensor_idx[sample]
    mask = torch.stack((batches.hit_mask, batches.hit_mask & batches.loc1_mask), -1)[sample]
    inputs = MeasurementFitInputs(
        measured=batches.measured_local[sample],
        variance=batches.covariances[sample],
        component_mask=mask & ~batches.is_target[sample].unsqueeze(-1),
        centers=geometry.centers[sensor],
        rotations=geometry.rotations[sensor],
    )
    return measurement_newton_decrement(data["parameters"][sample], inputs)


def _label(cell: int) -> str:
    n_ref, tanl, layout = CELLS[cell]
    return f"cell {cell} n_ref={n_ref} tanl={tanl} {layout}"


@pytest.mark.parametrize("cell", [0, 8])
def test_noise_free_residuals_vanish_and_the_strip_loc1_is_absent(cell):
    data = _closure(cell, n=4, noisy=False)
    assert float(data["residual"].abs().max()) <= 1.0e-10
    expected = 5 if CELLS[cell][2] == "mixed" else 6
    assert data["residual"].shape[1] == expected
    assert sorted(set(data["sensor"].tolist())) == [0, 1, 2]
    if CELLS[cell][2] == "mixed":
        strip_components = data["loc"][data["sensor"] == 1].tolist()
        assert strip_components == [0]


@pytest.mark.parametrize("cell", [0, 1, 8])
def test_predicted_covariance_is_spd_and_couples_sensors(cell):
    data = _closure(cell, n=4, noisy=False)
    sigma = data["sigma"]
    assert torch.allclose(sigma, sigma.transpose(-1, -2), rtol=0.0, atol=1.0e-12)
    assert bool((torch.linalg.cholesky_ex(sigma)[1] == 0).all())
    pairs = _cross_sensor_pairs(data["sensor"])
    assert bool((sigma[:, pairs].abs() > 1.0e-3).any(dim=1).all())
    # Adding target information to Gamma can only shrink the prediction.
    gap = torch.linalg.eigvalsh(sigma - data["sigma_target_inclusive"])[:, 0]
    assert bool((gap >= -1.0e-12).all())


@pytest.mark.slow
@pytest.mark.parametrize("cell", range(len(CELLS)))
def test_cross_sensor_residual_covariance_closes(cell):
    data = _closure(cell)
    assert float(data["decrement"].max()) <= GN_DECREMENT
    generator = torch.Generator().manual_seed(SEED)
    sample = torch.randperm(N_REPLICAS, generator=generator)[:N_PRODUCTION_CHECK]
    production = _production_decrement(data, sample)
    print(
        f"{_label(cell)}: module decrement max {float(data['decrement'].max()):.2e}, "
        f"production decrement max {float(production.max()):.2e}"
    )
    assert float(production.max()) <= MEASUREMENT_NEWTON_DECREMENT_STATIONARY

    residual, sigma, sensor = data["residual"], data["sigma"], data["sensor"]
    assert residual.shape == (N_REPLICAS, 5 if CELLS[cell][2] == "mixed" else 6)

    whitened = torch.linalg.solve_triangular(
        torch.linalg.cholesky(sigma), residual.unsqueeze(-1), upper=False
    ).squeeze(-1)
    mean, std = whitened.mean(0), whitened.std(0)
    corr = torch.corrcoef(whitened.T)
    off = ~torch.eye(len(sensor), dtype=torch.bool)
    pairs = _cross_sensor_pairs(sensor)
    z = _fisher_z(residual, _correlation(sigma.mean(0)), pairs)
    print(
        f"{_label(cell)}: |mean| max {float(mean.abs().max()):.4f}, "
        f"|std-1| max {float((std - 1).abs().max()):.4f}, |corr| max {float(corr[off].abs().max()):.4f}, "
        f"cross-sensor z max {float(z.max()):.2f}"
    )
    print("  rho_pred " + " ".join(f"{v:+.3f}" for v in _correlation(sigma.mean(0))[pairs].tolist()))
    print("  rho_mc   " + " ".join(f"{v:+.3f}" for v in torch.corrcoef(residual.T)[pairs].tolist()))
    assert float(mean.abs().max()) <= MEAN_LIMIT
    assert float((std - 1).abs().max()) <= STD_LIMIT
    assert float(corr[off].abs().max()) <= CORR_LIMIT
    assert float(z.max()) <= Z_LIMIT


@pytest.mark.slow
@pytest.mark.parametrize("cell", range(len(CELLS)))
def test_negative_controls_are_caught_in_power_cells(cell):
    data = _closure(cell)
    residual, sensor = data["residual"], data["sensor"]
    pairs = _cross_sensor_pairs(sensor)
    controls = {
        "drop_cross_sensor": _drop_cross_sensor(data["sigma"].mean(0), sensor),
        "target_inclusive_gamma": data["sigma_target_inclusive"].mean(0),
    }
    for name, predicted in controls.items():
        z = float(_fisher_z(residual, _correlation(predicted), pairs).max())
        powered = cell in POWER_CELLS[name]
        print(f"{_label(cell)} {name}: z max {z:.2f} ({'judged' if powered else 'reported'})")
        if powered:
            assert z > Z_LIMIT, (name, z)
