"""Contract tests for the ACTS-free material-node artifact builder.

Every check here is fail-closed by design: the downstream broken-helix prior is
only as trustworthy as the material budget and the track association feeding it.
"""

import importlib

import numpy as np
import pytest


def _module():
    return importlib.import_module("scripts.material_artifact")


# ---------------------------------------------------------------------------
# Pre-registered constants (Gate 1)
# ---------------------------------------------------------------------------


def test_gate1_constants_are_frozen_and_schema_is_versioned():
    material_artifact = _module()
    assert material_artifact.SCHEMA_VERSION == "1.1"
    assert material_artifact.X_OVER_X0_MIN == pytest.approx(1e-4)
    assert material_artifact.VOLUME_MERGE_WINDOW_MM == pytest.approx(5.0)
    assert material_artifact.DROPPED_MATERIAL_FRACTION_MAX == pytest.approx(0.01)
    assert material_artifact.AGGREGATION_THETA2_TOL == pytest.approx(0.02)
    assert material_artifact.T_X0_CLOSURE_RTOL == pytest.approx(1e-4)
    assert material_artifact.PATH_CLOSURE_RTOL == pytest.approx(1e-3)
    assert set(material_artifact.ARTIFACT_FIELDS) == {
        "event_nr",
        "track_nr",
        "q_over_p_gev_inv",
        "node_offsets",
        "path_length_mm",
        "position_mm",
        "direction",
        "x_over_x0",
        "surface_geometry_id",
        "volume_geometry_id",
    }


def test_module_imports_without_acts():
    """Repository boundary: only the adapter may import acts."""
    import inspect

    source = inspect.getsource(_module())
    assert "import acts" not in source


# ---------------------------------------------------------------------------
# Aggregation (plan section 1.4)
# ---------------------------------------------------------------------------


def test_aggregation_merges_adjacent_pre_post_split_of_one_surface():
    material_artifact = _module()
    result = material_artifact.aggregate_nodes(
        surface_id=np.array([7, 7, 9], dtype=np.uint64),
        volume_id=np.array([100, 100, 100], dtype=np.uint64),
        x_over_x0=np.array([0.004, 0.006, 0.02]),
        position=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (3, 1)),
        path_mm=np.array([0.0, 10.0, 50.0]),
    )
    assert result.x_over_x0.tolist() == pytest.approx([0.01, 0.02])
    # x/X0-weighted mean of 0 and 10 with weights 0.004 / 0.006 is 6.0 mm.
    assert result.position[0, 0] == pytest.approx(6.0)
    assert result.surface_id.tolist() == [7, 9]


def test_aggregation_never_merges_volume_nodes_across_volume_id():
    """sur_id is 0 for all volume material (RootMaterialTrackIo.cpp:245)."""
    material_artifact = _module()
    result = material_artifact.aggregate_nodes(
        surface_id=np.array([0, 0], dtype=np.uint64),
        volume_id=np.array([100, 200], dtype=np.uint64),
        x_over_x0=np.array([0.01, 0.02]),
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (2, 1)),
        path_mm=np.array([0.0, 1.0]),
    )
    assert result.x_over_x0.tolist() == pytest.approx([0.01, 0.02])
    assert result.volume_id.tolist() == [100, 200]


def test_aggregation_never_merges_volume_nodes_across_the_path_window():
    material_artifact = _module()
    window = material_artifact.VOLUME_MERGE_WINDOW_MM
    result = material_artifact.aggregate_nodes(
        surface_id=np.zeros(3, dtype=np.uint64),
        volume_id=np.full(3, 100, dtype=np.uint64),
        x_over_x0=np.array([0.01, 0.01, 0.01]),
        position=np.array(
            [[0.0, 0.0, 0.0], [0.5 * window, 0.0, 0.0], [10.0 * window, 0.0, 0.0]]
        ),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (3, 1)),
        path_mm=np.array([0.0, 0.5 * window, 10.0 * window]),
    )
    assert result.x_over_x0.tolist() == pytest.approx([0.02, 0.01])


def test_aggregation_renormalises_the_weighted_mean_direction():
    material_artifact = _module()
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([0.0, 1.0, 0.0])
    result = material_artifact.aggregate_nodes(
        surface_id=np.array([7, 7], dtype=np.uint64),
        volume_id=np.array([100, 100], dtype=np.uint64),
        x_over_x0=np.array([0.01, 0.01]),
        position=np.zeros((2, 3)),
        direction=np.stack((first, second)),
        path_mm=np.array([0.0, 1.0]),
    )
    norm = float(np.linalg.norm(result.direction[0]))
    assert norm == pytest.approx(1.0, abs=1e-12)


def test_aggregation_runs_before_thin_node_removal():
    """Many individually thin nodes must survive as one non-negligible node."""
    material_artifact = _module()
    n = 20
    each = material_artifact.X_OVER_X0_MIN / 2.0
    aggregated = material_artifact.aggregate_nodes(
        surface_id=np.full(n, 7, dtype=np.uint64),
        volume_id=np.full(n, 100, dtype=np.uint64),
        x_over_x0=np.full(n, each),
        position=np.zeros((n, 3)),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (n, 1)),
        path_mm=np.arange(n, dtype=float),
    )
    kept, dropped_fraction = material_artifact.drop_thin_nodes(aggregated)
    assert kept.x_over_x0.size == 1
    assert kept.x_over_x0[0] == pytest.approx(n * each)
    assert dropped_fraction == pytest.approx(0.0)


def test_drop_thin_nodes_gates_on_material_fraction_not_node_count():
    material_artifact = _module()
    thin = material_artifact.X_OVER_X0_MIN / 10.0
    aggregated = material_artifact.NodeArrays(
        surface_id=np.array([1, 2, 3, 4], dtype=np.uint64),
        volume_id=np.full(4, 100, dtype=np.uint64),
        x_over_x0=np.array([thin, thin, thin, 0.05]),
        position=np.zeros((4, 3)),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (4, 1)),
        path_mm=np.arange(4, dtype=float) * 100.0,
    )
    kept, dropped_fraction = material_artifact.drop_thin_nodes(aggregated)
    # Three of four nodes are dropped but they carry a negligible budget.
    assert kept.x_over_x0.size == 1
    assert dropped_fraction < material_artifact.DROPPED_MATERIAL_FRACTION_MAX


def test_drop_thin_nodes_fails_closed_when_too_much_material_is_discarded():
    material_artifact = _module()
    thin = material_artifact.X_OVER_X0_MIN * 0.9
    n = 200
    aggregated = material_artifact.NodeArrays(
        surface_id=np.arange(n + 1, dtype=np.uint64),
        volume_id=np.full(n + 1, 100, dtype=np.uint64),
        x_over_x0=np.concatenate((np.full(n, thin), [thin * 10.0])),
        position=np.zeros((n + 1, 3)),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (n + 1, 1)),
        path_mm=np.arange(n + 1, dtype=float) * 100.0,
    )
    with pytest.raises(ValueError, match="dropped material fraction"):
        material_artifact.drop_thin_nodes(aggregated)


def test_aggregation_theta2_change_matches_the_closed_form_and_fails_closed():
    """Merging slabs shifts sum(theta0^2) by the Highland log factor.

    ACTS evaluates the log with the per-slab x/X0 (Interactions.cpp:516), so the
    change is deterministic and non-zero.  Assert the closed form, then assert
    that exceeding the pre-registered bound is fail-closed.
    """
    material_artifact = _module()
    q_over_p, each = 0.2, 0.01
    raw = material_artifact.NodeArrays(
        surface_id=np.array([7, 7], dtype=np.uint64),
        volume_id=np.full(2, 100, dtype=np.uint64),
        x_over_x0=np.array([each, each]),
        position=np.zeros((2, 3)),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (2, 1)),
        path_mm=np.array([0.0, 1.0]),
    )
    merged = material_artifact.aggregate_nodes(
        surface_id=raw.surface_id,
        volume_id=raw.volume_id,
        x_over_x0=raw.x_over_x0,
        position=raw.position,
        direction=raw.direction,
        path_mm=raw.path_mm,
    )
    assert merged.x_over_x0.tolist() == pytest.approx([2 * each])

    helix_model = importlib.import_module("scripts.helix_model")
    q2b2 = 1.0 + (helix_model.MUON_MASS_GEV * q_over_p) ** 2
    log_factor = (1.0 + 0.038 * np.log(2 * each * q2b2)) / (
        1.0 + 0.038 * np.log(each * q2b2)
    )
    expected = abs(log_factor**2 - 1.0)

    change = material_artifact.aggregation_theta2_change(raw, merged, q_over_p=q_over_p)
    assert change == pytest.approx(expected, rel=1e-10)
    assert change > material_artifact.AGGREGATION_THETA2_TOL

    with pytest.raises(ValueError, match="aggregation changed"):
        material_artifact.check_aggregation_theta2(raw, merged, q_over_p=q_over_p)


def test_check_aggregation_theta2_accepts_a_no_op_merge():
    material_artifact = _module()
    nodes = material_artifact.NodeArrays(
        surface_id=np.array([7, 9], dtype=np.uint64),
        volume_id=np.full(2, 100, dtype=np.uint64),
        x_over_x0=np.array([0.01, 0.02]),
        position=np.zeros((2, 3)),
        direction=np.tile(np.array([1.0, 0.0, 0.0]), (2, 1)),
        path_mm=np.array([0.0, 50.0]),
    )
    merged = material_artifact.aggregate_nodes(
        surface_id=nodes.surface_id,
        volume_id=nodes.volume_id,
        x_over_x0=nodes.x_over_x0,
        position=nodes.position,
        direction=nodes.direction,
        path_mm=nodes.path_mm,
    )
    assert merged.x_over_x0.size == 2
    assert material_artifact.aggregation_theta2_change(
        nodes, merged, q_over_p=0.2
    ) == pytest.approx(0.0, abs=1e-15)
    material_artifact.check_aggregation_theta2(nodes, merged, q_over_p=0.2)


# ---------------------------------------------------------------------------
# Closures (plan sections 1.4 and 1.5)
# ---------------------------------------------------------------------------


def test_t_x0_closure_accepts_consistent_input_and_rejects_drift():
    material_artifact = _module()
    step_length = np.array([0.35, 0.70, 1.05])
    mat_x0 = np.array([35.0, 35.0, 35.0])
    total = float(np.sum(step_length / mat_x0))
    material_artifact.check_t_x0_closure(step_length, mat_x0, total)
    with pytest.raises(ValueError, match="t_X0 closure"):
        material_artifact.check_t_x0_closure(step_length, mat_x0, total * 1.01)


def test_path_closure_compares_relative_increments_not_absolute_values():
    """The helix s origin is arbitrary, so only increments are comparable."""
    material_artifact = _module()
    tan_lambda = 0.75
    scale = np.sqrt(1.0 + tan_lambda**2)
    transverse = np.array([0.0, 100.0, 250.0])
    position = np.stack(
        (transverse, np.zeros_like(transverse), transverse * tan_lambda), axis=1
    )
    # A large constant offset in s must not break the closure.
    material_artifact.check_path_closure(position, transverse + 12345.0, tan_lambda)

    stretched = transverse * 1.02
    with pytest.raises(ValueError, match="path closure"):
        material_artifact.check_path_closure(position, stretched + 12345.0, tan_lambda)
    del scale


def test_validate_material_arrays_rejects_each_malformed_input():
    material_artifact = _module()
    good = dict(
        step_length=np.array([0.35, 0.70]),
        mat_x0=np.array([35.0, 35.0]),
        direction=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        path_mm=np.array([0.0, 10.0]),
    )
    material_artifact.validate_material_arrays(**good)

    with pytest.raises(ValueError, match="branch lengths"):
        material_artifact.validate_material_arrays(
            **{**good, "mat_x0": np.array([35.0])}
        )
    with pytest.raises(ValueError, match="direction"):
        material_artifact.validate_material_arrays(
            **{**good, "direction": np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])}
        )
    with pytest.raises(ValueError, match="radiation length"):
        material_artifact.validate_material_arrays(
            **{**good, "mat_x0": np.array([35.0, 0.0])}
        )
    with pytest.raises(ValueError, match="monotonic"):
        material_artifact.validate_material_arrays(
            **{**good, "path_mm": np.array([10.0, 0.0])}
        )


# ---------------------------------------------------------------------------
# Joins (plan section 1.2)
# ---------------------------------------------------------------------------


def _prop_table(event, track, d0, z0, phi, theta, qop):
    return {
        "event_nr": np.asarray(event, dtype=np.int64),
        "track_nr": np.asarray(track, dtype=np.int64),
        "d0": np.asarray(d0, dtype=np.float32),
        "z0": np.asarray(z0, dtype=np.float32),
        "phi": np.asarray(phi, dtype=np.float32),
        "theta": np.asarray(theta, dtype=np.float32),
        "qOverP": np.asarray(qop, dtype=np.float32),
    }


def _summary_table(event, track, loc0, loc1, phi, theta, qop):
    return {
        "event_nr": np.asarray(event, dtype=np.int64),
        "track_nr": np.asarray(track, dtype=np.int64),
        "eLOC0_fit": np.asarray(loc0, dtype=np.float32),
        "eLOC1_fit": np.asarray(loc1, dtype=np.float32),
        "ePHI_fit": np.asarray(phi, dtype=np.float32),
        "eTHETA_fit": np.asarray(theta, dtype=np.float32),
        "eQOP_fit": np.asarray(qop, dtype=np.float32),
    }


def test_join_a_matches_on_keys_and_verifies_five_bound_parameters():
    material_artifact = _module()
    prop = _prop_table([0, 0], [0, 1], [0.1, -0.2], [1.0, 2.0], [0.3, 1.1], [1.2, 0.9], [0.4, -0.25])
    summary = _summary_table(
        [0, 0], [0, 1], [0.1, -0.2], [1.0, 2.0], [0.3, 1.1], [1.2, 0.9], [0.4, -0.25]
    )
    order = material_artifact.join_prop_to_summary(prop, summary)
    assert order.tolist() == [0, 1]

    corrupted = dict(summary)
    corrupted["ePHI_fit"] = np.asarray([0.3, 1.4], dtype=np.float32)
    with pytest.raises(ValueError, match="bound parameter"):
        material_artifact.join_prop_to_summary(prop, corrupted)


def test_join_a_fails_closed_on_a_missing_track():
    material_artifact = _module()
    prop = _prop_table([0, 0], [0, 1], [0.1, -0.2], [1.0, 2.0], [0.3, 1.1], [1.2, 0.9], [0.4, -0.25])
    summary = _summary_table([0], [0], [0.1], [1.0], [0.3], [1.2], [0.4])
    with pytest.raises(ValueError, match="unmatched"):
        material_artifact.join_prop_to_summary(prop, summary)


def _material_table(event, phi, eta):
    return {
        "event_id": np.asarray(event, dtype=np.int64),
        "v_phi": np.asarray(phi, dtype=np.float32),
        "v_eta": np.asarray(eta, dtype=np.float32),
    }


def _eta(theta):
    return -np.log(np.tan(np.asarray(theta, dtype=np.float64) / 2.0))


def test_join_b_is_bijective_within_an_event_and_survives_row_shuffling():
    material_artifact = _module()
    theta = np.array([1.2, 0.9, 2.0])
    prop = _prop_table([0, 0, 0], [0, 1, 2], [0, 0, 0], [0, 0, 0], [0.3, 1.1, -2.0], theta, [0.4, -0.25, 0.1])
    # The ROOT writer iterates an unordered_map, so rows arrive permuted.
    material = _material_table([0, 0, 0], [-2.0, 0.3, 1.1], _eta(theta[[2, 0, 1]]))
    order = material_artifact.join_material_to_prop(material, prop)
    assert order.tolist() == [2, 0, 1]


def test_join_b_fails_closed_on_ambiguous_and_on_missing_rows():
    material_artifact = _module()
    theta = np.array([1.2, 1.2])
    prop = _prop_table([0, 0], [0, 1], [0, 0], [0, 0], [0.3, 0.3], theta, [0.4, 0.4])
    material = _material_table([0, 0], [0.3, 0.3], _eta(theta))
    with pytest.raises(ValueError, match="ambiguous"):
        material_artifact.join_material_to_prop(material, prop)

    short = _material_table([0], [0.3], _eta(theta[:1]))
    with pytest.raises(ValueError, match="row count"):
        material_artifact.join_material_to_prop(short, prop)


def test_perigee_hypothesis_is_reported_not_enforced():
    """Gate 0 measures the pass rate; it only becomes a hard check afterwards."""
    material_artifact = _module()
    phi = np.array([0.3, 1.1])
    d0 = np.array([0.1, -0.2])
    z0 = np.array([1.0, 2.0])
    consistent = np.stack((-d0 * np.sin(phi), d0 * np.cos(phi), z0), axis=1)
    rate = material_artifact.perigee_hypothesis_pass_rate(consistent, d0, z0, phi)
    assert rate == pytest.approx(1.0)

    broken = consistent.copy()
    broken[1, 0] += 5.0
    assert material_artifact.perigee_hypothesis_pass_rate(
        broken, d0, z0, phi
    ) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Export path coordinate and end-to-end assembly
# ---------------------------------------------------------------------------


def test_transverse_chord_path_is_monotonic_and_ignores_z():
    material_artifact = _module()
    position = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 100.0], [3.0, 8.0, 200.0]])
    path = material_artifact.transverse_chord_path(position)
    assert path.tolist() == pytest.approx([0.0, 5.0, 9.0])
    assert material_artifact.transverse_chord_path(position[:1]).tolist() == [0.0]


@pytest.mark.root_io
def test_build_artifact_round_trips_a_synthetic_three_table_run(tmp_path):
    """End-to-end: the permuted material rows must land on the right tracks."""
    import awkward as ak
    import uproot

    material_artifact = _module()
    theta = np.array([1.2, 0.9], dtype=np.float32)
    phi = np.array([0.30, -1.10], dtype=np.float32)
    d0 = np.array([0.10, -0.20], dtype=np.float32)
    z0 = np.array([1.0, 2.0], dtype=np.float32)
    qop = np.array([0.40, -0.25], dtype=np.float32)

    # Two nodes per track: one surface node and one volume node far apart.
    def nodes(offset):
        x = np.array([10.0, 400.0]) + offset
        return dict(
            mat_x=x,
            mat_y=np.zeros(2),
            mat_z=x / np.tan(np.float64(theta[0 if offset == 0 else 1])),
            mat_dx=np.ones(2),
            mat_dy=np.zeros(2),
            mat_dz=np.zeros(2),
            mat_step_length=np.array([0.35, 0.70]),
            mat_X0=np.array([35.0, 35.0]),
            sur_id=np.array([11, 0], dtype=np.uint64),
            vol_id=np.array([100, 100], dtype=np.uint64),
        )

    # tan(lambda) = 1 / tan(theta); build z so the path closure holds exactly.
    per_track = []
    for index in range(2):
        node = nodes(0 if index == 0 else 5.0)
        transverse = node["mat_x"] - node["mat_x"][0]
        node["mat_z"] = transverse / np.tan(np.float64(theta[index]))
        per_track.append(node)

    # RootMaterialTrackWriter iterates an unordered_map: write the rows reversed.
    order = [1, 0]
    material = {
        "event_id": np.array([0, 0], dtype=np.int64),
        "v_x": (-d0 * np.sin(phi)).astype(np.float32)[order],
        "v_y": (d0 * np.cos(phi)).astype(np.float32)[order],
        "v_z": z0[order],
        "v_phi": phi[order],
        "v_eta": (-np.log(np.tan(theta.astype(np.float64) / 2.0)))[order].astype(np.float32),
        "t_X0": np.array(
            [float(np.sum(per_track[i]["mat_step_length"] / per_track[i]["mat_X0"])) for i in order],
            dtype=np.float32,
        ),
    }
    for branch in (
        "mat_x", "mat_y", "mat_z", "mat_dx", "mat_dy", "mat_dz",
        "mat_step_length", "mat_X0", "sur_id", "vol_id",
    ):
        material[branch] = ak.Array([per_track[i][branch] for i in order])

    with uproot.recreate(tmp_path / "material.root") as handle:
        handle[material_artifact.MATERIAL_TREE] = material
    with uproot.recreate(tmp_path / "propsummary.root") as handle:
        handle[material_artifact.PROPSUMMARY_TREE] = {
            "event_nr": np.array([0, 0], dtype=np.int32),
            "track_nr": np.array([0, 1], dtype=np.int32),
            "d0": d0, "z0": z0, "phi": phi, "theta": theta, "qOverP": qop,
            "nMaterials": np.array([1, 1], dtype=np.int32),
        }
    with uproot.recreate(tmp_path / "summary.root") as handle:
        handle[material_artifact.TRACKSUMMARY_TREE] = {
            "event_nr": np.array([0, 0], dtype=np.int32),
            "track_nr": np.array([0, 1], dtype=np.int32),
            "eLOC0_fit": d0, "eLOC1_fit": z0, "ePHI_fit": phi,
            "eTHETA_fit": theta, "eQOP_fit": qop,
        }

    artifact, diagnostics = material_artifact.build_artifact(
        tmp_path / "material.root",
        tmp_path / "propsummary.root",
        tmp_path / "summary.root",
    )
    assert set(artifact) == set(material_artifact.ARTIFACT_FIELDS)
    assert diagnostics.n_tracks == 2
    assert diagnostics.unmatched == 0
    # Rows were written reversed; the artifact must be keyed by the real track.
    assert artifact["event_nr"].tolist() == [0, 0]
    assert sorted(artifact["track_nr"].tolist()) == [0, 1]
    for row, track in enumerate(artifact["track_nr"]):
        assert artifact["q_over_p_gev_inv"][row] == pytest.approx(
            float(qop[int(track)]), rel=1e-6
        )
    assert artifact["node_offsets"].tolist() == [0, 2, 4]
    assert diagnostics.max_t_x0_deviation < material_artifact.T_X0_CLOSURE_RTOL
    assert diagnostics.perigee_hypothesis_pass_rate == pytest.approx(1.0)


def test_propagation_failure_check_is_bidirectional():
    """Join A alone only proves prop -> summary; a dropped propagation is silent."""
    material_artifact = _module()
    summary_counts = {0: 2, 1: 1}
    material_artifact.check_propagation_coverage({0: 2, 1: 1}, summary_counts)
    with pytest.raises(ValueError, match="propagation failures"):
        material_artifact.check_propagation_coverage({0: 1, 1: 1}, summary_counts)
    # An event with no reconstructed track at all is not a propagation failure.
    material_artifact.check_propagation_coverage({0: 2}, {0: 2, 2: 0})


def test_perigee_hypothesis_is_enforced_once_gate0_measured_a_unit_pass_rate():
    material_artifact = _module()
    assert material_artifact.PERIGEE_HYPOTHESIS_REQUIRED is True
    phi = np.array([0.3, 1.1])
    d0 = np.array([0.1, -0.2])
    z0 = np.array([1.0, 2.0])
    position = np.stack((-d0 * np.sin(phi), d0 * np.cos(phi), z0), axis=1)
    material_artifact.check_perigee_hypothesis(position, d0, z0, phi)
    position[1, 0] += 5.0
    with pytest.raises(ValueError, match="perigee"):
        material_artifact.check_perigee_hypothesis(position, d0, z0, phi)


def test_capture_source_snapshot_flag_defaults_off_and_threads_through(tmp_path, monkeypatch):
    """The opt-in source-snapshot switch must reach write_experiment_manifest
    unchanged; default False keeps every existing manifest call unaffected."""
    material_artifact = _module()

    default_args = material_artifact.build_argument_parser().parse_args(
        [
            "--material", "m.root",
            "--propsummary", "p.root",
            "--summary", "s.root",
            "--out", "artifact.npz",
            "--run-tag", "test-run",
        ]
    )
    assert default_args.capture_source_snapshot is False

    captured = {}

    def fake_write_experiment_manifest(*_args, **kwargs):
        captured.update(kwargs)
        raise SystemExit("stop-after-manifest")

    monkeypatch.setattr(
        material_artifact, "write_experiment_manifest", fake_write_experiment_manifest
    )
    material = tmp_path / "material.root"
    propsummary = tmp_path / "propsummary.root"
    summary = tmp_path / "summary.root"
    for path in (material, propsummary, summary):
        path.touch()

    with pytest.raises(SystemExit):
        material_artifact.main(
            [
                "--material", str(material),
                "--propsummary", str(propsummary),
                "--summary", str(summary),
                "--out", str(tmp_path / "artifact.npz"),
                "--run-tag", "test-run",
                "--capture-source-snapshot",
            ]
        )

    assert captured["capture_source_snapshot"] is True
