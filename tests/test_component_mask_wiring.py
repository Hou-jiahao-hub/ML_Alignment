"""A2: before/after component-mask intersection, and its wiring into the free pipeline."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import torch

from mlalign.config import AlignmentConfig
from mlalign.geometry_io import JointGeometry
from mlalign.joint_residuals import combine_component_masks
from mlalign.track_batches import TrackBatches




def test_combine_component_masks_intersects_and_counts_exclusions():
    """Exercises the production helper, not a re-implementation of its logic."""
    T = True
    F = False
    # One track, four hits; the last one is padding.
    hit_mask = torch.tensor([[T, T, T, F]])
    initial = torch.tensor([[[T, T], [T, F], [F, F], [T, T]]])
    final = torch.tensor([[[T, T], [T, T], [T, F], [T, T]]])

    hit_valid, comp, initial_only, final_only = combine_component_masks(
        hit_mask, initial, final
    )

    assert hit_valid.tolist() == [[T, T, F, F]]
    assert int(hit_valid.sum()) == 2                      # n_hits_loc0
    assert comp.tolist() == [[T, T], [T, F]]
    assert int(comp[:, 1].sum()) == 1                     # n_hits_loc1
    # hit0/hit1 stay valid on both sides; hit3 is excluded by hit_mask.
    assert initial_only == 0
    # hit2: invalid before, valid after, inside hit_mask.
    assert final_only == 1


def test_combine_component_masks_counts_initial_only_exclusions():
    T, F = True, False
    hit_mask = torch.tensor([[T, T]])
    initial = torch.tensor([[[T, T], [T, T]]])
    final = torch.tensor([[[T, T], [F, F]]])

    hit_valid, comp, initial_only, final_only = combine_component_masks(
        hit_mask, initial, final
    )

    assert hit_valid.tolist() == [[T, F]]
    assert comp.tolist() == [[T, T]]
    assert initial_only == 1
    assert final_only == 0


def test_free_pipeline_wires_combine_component_masks(monkeypatch, tmp_path, capsys):
    """The helper must actually be called by run_alignment.

    Every heavy dependency is stubbed, and the helper is replaced by a recorder
    returning sentinel counts, so a passing assertion can only come from real
    wiring rather than from a coincidental substring in the output.
    """
    pipeline = importlib.import_module("mlalign.pipeline")

    n_tracks, n_hits = 1, 2
    batches = TrackBatches(
        measured_local=torch.zeros((n_tracks, n_hits, 2)),
        covariances=torch.ones((n_tracks, n_hits, 2)),
        sensor_idx=torch.zeros((n_tracks, n_hits), dtype=torch.long),
        hit_mask=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        loc1_mask=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        g_smt=torch.zeros((n_tracks, n_hits, 3)),
        p_smt=torch.zeros((n_tracks, n_hits, 3)),
        volume_id=torch.full((n_tracks, n_hits), 16, dtype=torch.long),
        layer_id=torch.full((n_tracks, n_hits), 6, dtype=torch.long),
        is_target=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        event_track_ids=torch.zeros((n_tracks, 2), dtype=torch.long),
        fit_mask=torch.ones((n_tracks, n_hits), dtype=torch.bool),
        fit_span=torch.tensor([300.0]),
        selection_stats={"input_tracks": 1, "retained_tracks": 1, "excluded_tracks": 0},
    )
    geometry = JointGeometry(
        transforms=[], input_transforms=[],
        centers=torch.zeros((1, 3), dtype=torch.float64),
        rotations=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        target_mask=torch.ones(1, dtype=torch.bool),
        id_to_idx={(16, 6, 0): 0},
        input_indices=[0],
    )
    component_mask = torch.ones((n_tracks, n_hits, 2), dtype=torch.bool)
    joint_result = {
        "predicted_deltas": np.zeros((1, 6)),
        "final_residuals": torch.zeros((n_tracks, n_hits, 2)),
        "final_component_mask": component_mask,
        "initial_residuals": torch.zeros((n_tracks, n_hits, 2)),
        "initial_component_mask": component_mask,
        "history": [],
    }

    monkeypatch.setattr(pipeline.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(pipeline, "load_joint_geometry", lambda *_a, **_k: geometry)
    # Arguments are evaluated before the stubbed callee runs, so the fake inputs
    # object still needs the attributes the call sites read.
    inputs_stub = SimpleNamespace(
        initial_residuals=torch.zeros((n_hits, 2)),
        hit_volume_ids=torch.full((n_hits,), 16, dtype=torch.long),
        hit_layer_ids=torch.full((n_hits,), 6, dtype=torch.long),
        loc1_valid=torch.ones(n_hits, dtype=torch.bool),
    )
    monkeypatch.setattr(pipeline, "load_variable_information", lambda *_a, **_k: inputs_stub)
    monkeypatch.setattr(pipeline, "print_target_residual_stats", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "build_track_batches", lambda *_a, **_k: batches)
    seeded = torch.zeros((n_tracks, 6), dtype=torch.float64)
    monkeypatch.setattr(
        pipeline,
        "seed_helices",
        lambda *_a, **_k: (
            seeded,
            {
                "solver": "gauss-newton",
                "converged": True,
                "iterations": 1,
                "initial_objective": 0.0,
                "final_objective": 0.0,
                "max_relative_decrease": 0.0,
                "max_gradient_norm": 0.0,
                "relative_newton_decrement_max": 0.0,
                "non_stationary_row_fraction": 0.0,
                "n_non_finite": 0,
                "n_chunks": 1,
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "smt_seed_statistics",
        lambda *_a, **_k: {"loss": 0.0, "n_points": 2, "invalid_fraction": 0.0},
    )
    monkeypatch.setattr(
        pipeline,
        "fit_helices_to_measurements",
        lambda *_a, **_k: (
            seeded,
            {
                "solver": "gauss-newton",
                "chi2_per_component": 0.0,
                "n_components": 4,
                "invalid_fraction": 0.0,
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_joint_forward",
        lambda *_a, **_k: (
            torch.zeros((n_tracks, n_hits, 2)),
            torch.ones((n_tracks, n_hits, 2)),
            component_mask,
            torch.ones((n_tracks, n_hits), dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(pipeline, "summarize_joint_layer_chi2", lambda *_a, **_k: {})
    monkeypatch.setattr(pipeline, "run_schur_lm_alignment", lambda *_a, **_k: joint_result)
    monkeypatch.setattr(pipeline, "validate_against_truth", lambda *_a, **_k: {})
    monkeypatch.setattr(pipeline, "export_joint_geometry", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "print_per_layer_residual_diagnostics", lambda *_a, **_k: {})

    calls = []

    def recorder(hit_mask, initial_component_mask, final_component_mask):
        calls.append((hit_mask, initial_component_mask, final_component_mask))
        keep = hit_mask & initial_component_mask[..., 0] & final_component_mask[..., 0]
        # Sentinel counts: 17/23 cannot arise from this 1x2 fixture.
        return keep, (initial_component_mask & final_component_mask)[keep], 17, 23

    monkeypatch.setattr(pipeline, "combine_component_masks", recorder)

    config = AlignmentConfig(
        input_geom=str(tmp_path / "in.json"),
        states_root=str(tmp_path / "states.root"),
        output_geom=str(tmp_path / "out.json"),
        nominal_full_geom=str(tmp_path / "nominal.json"),
        make_plot=False,
        target_layers=((16, 6),),
    )
    pipeline.run_alignment(config)

    assert len(calls) == 1, "combine_component_masks was not called exactly once"
    output = capsys.readouterr().out
    assert "initial_only=17" in output
    assert "final_only=23" in output
