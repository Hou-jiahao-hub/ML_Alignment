import importlib
import math

import pytest
import torch


def test_shared_helix_point_and_tangent_match_hand_checked_circle():
    model = importlib.import_module("scripts.helix_model")
    track = torch.tensor(
        [[1.0, -2.0, 3.0, 0.0, 0.5, 0.01]], dtype=torch.float64
    )
    path = torch.tensor([100.0], dtype=torch.float64)

    point = model.helix_point(track, path)
    tangent = model.helix_tangent(track, path)

    assert point[0, 0].item() == pytest.approx(1.0 + 100.0 * math.sin(1.0))
    assert point[0, 1].item() == pytest.approx(-2.0 + 100.0 * (1.0 - math.cos(1.0)))
    assert point[0, 2].item() == pytest.approx(53.0)
    assert tangent[0].tolist() == pytest.approx(
        [math.cos(1.0), math.sin(1.0), 0.5]
    )


def test_shared_closest_point_and_plane_intersection_report_invalid_paths():
    model = importlib.import_module("scripts.helix_model")
    track = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    point = torch.tensor([[25.0, 3.0, 0.0]], dtype=torch.float64)
    fitted, path, closest_valid = model.closest_helix_points(track, point)
    assert closest_valid.tolist() == [True]
    assert path.tolist() == pytest.approx([25.0])
    torch.testing.assert_close(
        fitted, torch.tensor([[25.0, 0.0, 0.0]], dtype=torch.float64)
    )

    _path, fallback, intersection_valid = model.intersect_plane(
        track,
        center=torch.tensor([[0.0, 5.0, 0.0]], dtype=torch.float64),
        normal=torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
    )
    assert intersection_valid.tolist() == [False]
    torch.testing.assert_close(fallback, torch.zeros_like(fallback))


def test_curvature_optimizer_scaling_round_trips_physical_parameters():
    model = importlib.import_module("scripts.helix_model")
    tracks = torch.tensor(
        [[1.0, 2.0, 3.0, 0.4, -0.2, 2.5e-4]], dtype=torch.float64
    )
    linear, scaled = model.split_track_parameters(tracks)
    torch.testing.assert_close(
        scaled, torch.tensor([[2.5]], dtype=torch.float64)
    )
    assert torch.equal(model.assemble_track_parameters(linear, scaled), tracks)


def test_both_consumers_reexport_the_shared_math_implementation():
    shared = importlib.import_module("scripts.helix_model")
    alignment = importlib.import_module("scripts.ml_alignment_autograd")
    validation = importlib.import_module("scripts.validate_helix_model")

    for name in (
        "helix_point",
        "helix_tangent",
        "closest_helix_points",
        "intersect_plane",
    ):
        assert getattr(alignment, name) is getattr(shared, name)
        assert getattr(validation, name) is getattr(shared, name)
