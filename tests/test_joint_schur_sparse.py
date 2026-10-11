"""P7: block-sparse production path of ``schur-lm`` (Iteration 38 §C, S1-S5).

The reduced system is assembled as CSC from the existing 3x3 coordinate
blocks and solved with SuperLU; the global decrement and rank test run per
connected component of the sensor coupling graph.  Every quantity is checked
against the dense oracles of P4/P6, which share no assembly with this path.
"""

from __future__ import annotations

import dataclasses
import time

import pytest
import torch

import joint_alignment_schur as J
import joint_schur_synthetic as syn

DTYPE = torch.float64
ALGEBRA_RTOL = 1.0e-10
LEVEL_A_TOPOLOGY = [[0, 1], [0, 1], [1, 0], [2, 3], [3, 2], [2, 3], [4], [4], [0], [1], [2], [3], [4]]
# Components {0, 1}, {2, 3, 4}, {5}: three independent coupled systems.
MULTI_COMPONENT_TOPOLOGY = [[0, 1], [1, 0], [0], [1], [2, 3], [3, 4], [4, 2], [2], [3], [4], [5], [5], [5]]


def _relative(a, b):
    a, b = torch.as_tensor(a, dtype=DTYPE), torch.as_tensor(b, dtype=DTYPE)
    return float((a - b).norm() / b.norm().clamp_min(1.0e-300))


def _level_a(seed=0):
    return syn.random_linearization(LEVEL_A_TOPOLOGY, 5, seed=seed)


def _multi_component(seed=1):
    return syn.random_linearization(MULTI_COMPONENT_TOPOLOGY, 6, seed=seed)


def _level_b():
    problem = syn.shared_topology_problem(injection=[[0.05, 0.05, 0, 0, 0, 2e-4]] * 3)
    return J.linearize_tracks(syn.ref_only_fit(problem), problem.batches, problem.geometry)


def _rank_deficient():
    lin = _level_a(seed=2)
    jacobian = lin.geometry_jacobian.clone()
    on_block = (lin.block_index == 4).unsqueeze(-1)
    jacobian[..., 2] = torch.where(on_block, jacobian[..., 0], jacobian[..., 2])
    return dataclasses.replace(lin, geometry_jacobian=jacobian)


FROZEN = torch.tensor(
    [[True, False, True], [True, True, True], [False, False, False], [True, True, False], [True, True, True]]
)


def _systems():
    a = J.normal_blocks(_level_a())
    m = J.normal_blocks(_multi_component())
    b = J.normal_blocks(_level_b())
    return {
        "level_a_undamped": J.schur_reduce(a, 0.0),
        "level_a_damped": J.schur_reduce(a, 1.0e-3),
        "level_a_restricted": J.restrict_to_free(J.schur_reduce(a, 1.0e-3), FROZEN),
        "multi_component": J.schur_reduce(m, 1.0e-3),
        "level_b": J.schur_reduce(b, 1.0e-3),
    }


# S1 -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["level_a_undamped", "level_a_damped", "level_a_restricted", "multi_component", "level_b"])
def test_sparse_assembly_equals_the_dense_reduced_matrix(name):
    system = _systems()[name]
    dense = system.to_dense()
    sparse = torch.from_numpy(J.to_sparse(system).toarray())

    assert sparse.dtype == DTYPE
    assert _relative(sparse, dense) <= 1.0e-14
    assert float((dense - dense.T).norm() / dense.norm()) <= 1.0e-14


def test_sparse_matrix_stores_only_existing_blocks():
    system = _systems()["multi_component"]
    sparse = J.to_sparse(system)
    k = system.rhs.shape[1]
    assert sparse.format == "csc"
    assert sparse.nnz == len(system.rows) * k * k
    assert (1, 2) not in system.block_keys() and (0, 5) not in system.block_keys()


# S2 -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["level_a_damped", "level_a_restricted", "multi_component", "level_b"])
def test_sparse_solve_matches_the_dense_solve(name):
    system = _systems()[name]
    assert _relative(J.solve_geometry_sparse(system), J.solve_geometry_dense(system)) <= ALGEBRA_RTOL


def test_feasible_step_with_the_sparse_solve_matches_dense_through_a_resolve():
    """The coupled step pushes a bound DOF outwards (alpha = 0), forcing a resolve."""
    S = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=DTYPE)
    b = torch.tensor([0.1, 1.0], dtype=DTYPE)
    system = J.SchurSystem(
        rows=torch.tensor([0]),
        cols=torch.tensor([0]),
        values=S[None],
        rhs=b[None],
        lam=0.0,
        track_step=torch.zeros((1, 5), dtype=DTYPE),
        track_gain=torch.zeros((1, 5, 2), dtype=DTYPE),
        pair_track=torch.tensor([0]),
        pair_block=torch.tensor([0]),
    )
    bounds = torch.tensor([1.5, 1.5], dtype=DTYPE)
    values = torch.tensor([[1.5, 0.0]], dtype=DTYPE)
    free = J.free_geometry_mask(values, b[None], bounds)

    dense = J.feasible_joint_step(system, values, free, bounds, (0, 1))
    sparse = J.feasible_joint_step(system, values, free, bounds, (0, 1), solve=J.solve_geometry_sparse)

    assert sparse.n_resolves == dense.n_resolves == 1
    assert sparse.free.tolist() == dense.free.tolist()
    assert sparse.alpha == dense.alpha
    assert torch.allclose(sparse.delta_u, dense.delta_u, rtol=ALGEBRA_RTOL, atol=1e-14)


def test_feasible_step_with_the_sparse_solve_matches_dense_on_level_a():
    blocks = J.normal_blocks(_level_a(seed=5))
    undamped = J.schur_reduce(blocks, 0.0)
    unconstrained = J.solve_geometry_dense(undamped)
    bounds = unconstrained.abs().max(dim=0).values * 0.4
    values = torch.zeros_like(unconstrained)
    free = J.free_geometry_mask(values, undamped.rhs, bounds)
    system = J.schur_reduce(blocks, 1.0e-3)

    dense = J.feasible_joint_step(system, values, free, bounds, (0, 1, 2))
    sparse = J.feasible_joint_step(system, values, free, bounds, (0, 1, 2), solve=J.solve_geometry_sparse)

    assert dense.alpha < 1.0
    assert sparse.alpha == pytest.approx(dense.alpha, rel=ALGEBRA_RTOL)
    assert _relative(sparse.delta_u, dense.delta_u) <= ALGEBRA_RTOL
    assert _relative(sparse.delta_theta, dense.delta_theta) <= ALGEBRA_RTOL


# S3 -------------------------------------------------------------------------


def _stationarity_cases():
    cases = {
        "level_a": (_level_a(seed=4), None),
        "multi_component": (_multi_component(), None),
        "frozen": (_level_a(seed=4), FROZEN),
        "rank_deficient": (_rank_deficient(), None),
        "multi_component_frozen": (_multi_component(), torch.tensor([[True] * 3, [False] * 3] + [[True] * 3] * 4)),
    }
    return cases


@pytest.mark.parametrize("case", ["level_a", "multi_component", "frozen", "rank_deficient", "multi_component_frozen"])
def test_component_stationarity_matches_the_dense_oracle(case):
    lin, free = _stationarity_cases()[case]
    blocks = J.normal_blocks(lin)
    system = J.schur_reduce(blocks, 0.0)

    dense = J.global_stationarity(blocks, system, free)
    sparse = J.global_stationarity_sparse(blocks, system, free)

    assert sparse.lambda2_geo == pytest.approx(dense.lambda2_geo, rel=ALGEBRA_RTOL, abs=1e-300)
    assert torch.allclose(sparse.lambda2_trk, dense.lambda2_trk, rtol=ALGEBRA_RTOL, atol=0.0)
    assert sparse.n_free == dense.n_free
    assert sparse.rank_deficient == dense.rank_deficient
    assert sparse.lambda2_is_lower_bound == dense.lambda2_is_lower_bound
    assert sparse.min_scaled_eigenvalue == pytest.approx(dense.min_scaled_eigenvalue, rel=1e-8, abs=1e-14)
    assert sparse.null_gradient_max == pytest.approx(dense.null_gradient_max, rel=1e-8, abs=1e-14)
    assert sparse.stationary == dense.stationary


def test_rank_deficient_case_is_flagged_by_the_component_path():
    blocks = J.normal_blocks(_rank_deficient())
    result = J.global_stationarity_sparse(blocks, J.schur_reduce(blocks, 0.0))
    assert result.rank_deficient and result.lambda2_is_lower_bound
    assert result.min_scaled_eigenvalue <= J.SCHUR_LM_RANK_TOL


def test_largest_component_is_reported():
    blocks = J.normal_blocks(_multi_component())
    result = J.global_stationarity_sparse(blocks, J.schur_reduce(blocks, 0.0))
    assert result.largest_component_dof == 9
    frozen = torch.ones((6, 3), dtype=torch.bool)
    frozen[2] = False
    result = J.global_stationarity_sparse(blocks, J.schur_reduce(blocks, 0.0), frozen)
    assert result.largest_component_dof == 6


def test_component_stationarity_with_nothing_free_is_a_kkt_point():
    blocks = J.normal_blocks(_level_a())
    result = J.global_stationarity_sparse(
        blocks, J.schur_reduce(blocks, 0.0), torch.zeros((5, 3), dtype=torch.bool)
    )
    assert result.n_free == 0 and result.lambda2_geo == 0.0 and not result.rank_deficient


def test_component_stationarity_requires_the_undamped_system():
    blocks = J.normal_blocks(_level_a())
    with pytest.raises(ValueError):
        J.global_stationarity_sparse(blocks, J.schur_reduce(blocks, 1.0))


# S4 -------------------------------------------------------------------------


def _permute_tracks(lin, order):
    return dataclasses.replace(
        lin,
        parameters=lin.parameters[order],
        residual=lin.residual[order],
        geometry_jacobian=lin.geometry_jacobian[order],
        track_jacobian=lin.track_jacobian[order],
        component_mask=lin.component_mask[order],
        is_target=lin.is_target[order],
        is_ref=lin.is_ref[order],
        block_index=lin.block_index[order],
    )


def _sparse_pipeline(lin):
    blocks = J.normal_blocks(lin)
    system = J.schur_reduce(blocks, 1.0e-3)
    du = J.solve_geometry_sparse(system)
    stationarity = J.global_stationarity_sparse(blocks, J.schur_reduce(blocks, 0.0))
    return J.to_sparse(system), du, system.back_substitute(du), stationarity


def test_sparse_path_is_bitwise_deterministic():
    first = _sparse_pipeline(_multi_component())
    second = _sparse_pipeline(_multi_component())

    assert (first[0] != second[0]).nnz == 0
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert first[3].lambda2_geo == second[3].lambda2_geo
    assert first[3].min_scaled_eigenvalue == second[3].min_scaled_eigenvalue


def test_track_order_does_not_change_the_sparse_solution():
    lin = _multi_component()
    order = torch.randperm(lin.residual.shape[0], generator=torch.Generator().manual_seed(11))
    _sparse, du, dtheta, stationarity = _sparse_pipeline(lin)
    _sparse_p, du_p, dtheta_p, stationarity_p = _sparse_pipeline(_permute_tracks(lin, order))

    assert _relative(du_p, du) <= 1.0e-12
    assert _relative(dtheta_p, dtheta[order]) <= 1.0e-12
    assert stationarity_p.lambda2_geo == pytest.approx(stationarity.lambda2_geo, rel=1.0e-12)


# S5 -------------------------------------------------------------------------


def _production_topology(n_blocks=2268, n_tracks=61_482, seed=0):
    """Two layers of ``n_blocks / 2`` sensors; each track crosses 1-2 of each (2-4 targets)."""
    generator = torch.Generator().manual_seed(seed)
    half = n_blocks // 2
    counts = torch.randint(1, 3, (n_tracks, 2), generator=generator)
    first = torch.randint(0, half, (n_tracks, 2), generator=generator)
    second = torch.randint(half, n_blocks, (n_tracks, 2), generator=generator)
    topology = []
    for track in range(n_tracks):
        inner = first[track, : counts[track, 0]].tolist()
        outer = second[track, : counts[track, 1]].tolist()
        topology.append(sorted(set(inner)) + sorted(set(outer)))
    return topology


@pytest.mark.slow
def test_production_scale_sparse_path_never_densifies(monkeypatch):
    def forbidden(self):
        raise AssertionError("to_dense called on the production path")

    topology = _production_topology()
    lin = syn.random_linearization(topology, 2268, seed=3)
    monkeypatch.setattr(J.SchurSystem, "to_dense", forbidden)

    start = time.perf_counter()
    blocks = J.normal_blocks(lin)
    undamped = J.schur_reduce(blocks, 0.0)
    damped = J.schur_reduce(blocks, 1.0e-3)
    reduced = time.perf_counter()
    du = J.solve_geometry_sparse(damped)
    solved = time.perf_counter()
    stationarity = J.global_stationarity_sparse(blocks, undamped)
    finished = time.perf_counter()

    n_offdiagonal = int((damped.rows != damped.cols).sum())
    print(
        f"\nS5: blocks=2268 tracks={len(topology)} offdiagonal_blocks={n_offdiagonal} "
        f"reduce={reduced - start:.1f}s splu={solved - reduced:.1f}s "
        f"stationarity={finished - solved:.1f}s largest_component_dof={stationarity.largest_component_dof}"
    )
    assert du.shape == (2268, 3) and bool(torch.isfinite(du).all())
    assert stationarity.n_free == 2268 * 3
    assert stationarity.largest_component_dof > 0
