"""Contracts for the single-DOF track-geometry validation driver."""
import importlib
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts import helix_model as hm


def module():
    return importlib.import_module("scripts.tracking_geometry_validation")


def truth_module():
    return importlib.import_module("scripts.validate_helix_truth")


def nominal_items():
    tilt = .6
    rotation = [[1., 0., 0.],
                [0., math.cos(tilt), -math.sin(tilt)],
                [0., math.sin(tilt), math.cos(tilt)]]
    return [
        {"ID": "vol=17|lay=8|sen=1", "Translation": [10., 20., 30.], "RotationMatrix": rotation},
        {"ID": "vol=17|lay=8|sen=2", "Translation": [-5., 2., 1.], "RotationMatrix": np.eye(3).tolist()},
        {"ID": "vol=24|lay=4|sen=1|ext=3", "Translation": [1., 2., 3.], "RotationMatrix": np.eye(3).tolist()},
    ]


@pytest.mark.parametrize("dof,amplitude", [("tx", .5), ("ty", -.5), ("rz", .02)])
def test_injection_matches_solver_parametrisation(dof, amplitude):
    m = module()
    solver = importlib.import_module("scripts.ml_alignment_autograd")
    items = nominal_items()
    injected = m.perturb_layer(items, (17, 8), dof, amplitude)
    assert [item["ID"] for item in injected] == ["vol=17|lay=8|sen=1", "vol=17|lay=8|sen=2"]
    delta = torch.zeros(6, dtype=torch.float64)
    delta[m.DOF_INDEX[dof]] = amplitude
    for item, original in zip(injected, items):
        center, rotation = solver.apply_alignment_torch(
            torch.tensor(original["Translation"], dtype=torch.float64),
            torch.tensor(original["RotationMatrix"], dtype=torch.float64), delta)
        np.testing.assert_allclose(item["Translation"], center.numpy(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(item["RotationMatrix"], rotation.numpy(), rtol=0, atol=1e-12)


def test_injection_uses_one_amplitude_and_leaves_nominal_untouched():
    m = module()
    items = nominal_items()
    before = [dict(item) for item in items]
    injected = m.perturb_layer(items, (17, 8), "tx", .5)
    assert items == before
    shifts = [np.linalg.norm(np.subtract(a["Translation"], b["Translation"]))
              for a, b in zip(injected, items)]
    np.testing.assert_allclose(shifts, [.5, .5], rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        m.perturb_layer(items, (17, 8), "tz", .5)


def synthetic_track():
    """Twelve planes along a helix; target planes belong to 17:8, others to 24:4."""
    t = truth_module()
    p = torch.tensor([2., -1., .5, .2, .1, 2.e-4], dtype=torch.float64)
    path = torch.linspace(0., 600., 12, dtype=torch.float64)
    points = hm.helix_point(p.expand(12, -1), path)
    tangent = hm.helix_tangent(p.expand(12, -1), path)
    normal = torch.nn.functional.normalize(tangent, dim=-1)
    up = torch.tensor([0., 0., 1.], dtype=torch.float64).expand_as(normal)
    axis0 = torch.nn.functional.normalize(torch.cross(up, normal, dim=-1), dim=-1)
    rotations = torch.stack((axis0, torch.cross(normal, axis0, dim=-1), normal), dim=-1)
    target = torch.zeros(12, dtype=torch.bool)
    target[[0, 5, 11]] = True
    # Offset each plane so the crossing has non-zero local (u, v).
    offsets = torch.stack((torch.full((12,), 3., dtype=torch.float64),
                           torch.full((12,), -7., dtype=torch.float64)), dim=1)
    centers = points - (rotations[..., :2] @ offsets.unsqueeze(-1)).squeeze(-1)
    identities = [(17, 8, i + 1, 0) if target[i] else (24, 4, i + 1, 0) for i in range(12)]
    track = t.TruthTrack(
        event_nr=3, track_nr=0, identities=identities, path_length=path,
        centers=centers, rotations=rotations, smt_points=points, smt_directions=tangent,
        measured=offsets.clone(), variance=torch.full((12, 2), .01, dtype=torch.float64),
        truth=offsets.clone(), target=target)
    return track, p


def test_target_is_only_the_tested_layer_and_other_layer_is_fitted():
    track, _ = synthetic_track()
    assert track.fit_mask.tolist() == (~track.target).tolist()
    assert all(track.identities[i][:2] == (24, 4) for i in track.fit_mask.nonzero().flatten())


def perturbed_copy(m, track, dof, amplitude):
    delta = np.zeros(6)
    delta[m.DOF_INDEX[dof]] = amplitude
    centers, rotations = track.centers.clone(), track.rotations.clone()
    for i in track.target.nonzero().flatten().tolist():
        c, r = m.apply_local_delta(centers[i].numpy(), rotations[i].numpy(), delta)
        centers[i] = torch.tensor(c)
        rotations[i] = torch.tensor(r)
    return replace(track, centers=centers, rotations=rotations)


@pytest.mark.parametrize("dof,amplitude", [("tx", .5), ("ty", -.5)])
def test_geometry_only_translation_moves_target_residual_by_minus_amplitude(dof, amplitude):
    m = module()
    track, p = synthetic_track()
    r0 = m.residuals([track], [p])[0]
    rg = m.residuals([perturbed_copy(m, track, dof, amplitude)], [p])[0]
    delta = rg - r0
    axis = m.DOF_INDEX[dof]
    expected = torch.zeros_like(delta)
    expected[track.target, axis] = -amplitude
    torch.testing.assert_close(delta, expected, rtol=0, atol=1e-9)


def test_geometry_only_rotation_matches_exact_in_plane_rotation():
    m = module()
    track, p = synthetic_track()
    phi = .02
    r0 = m.residuals([track], [p])[0]
    rg = m.residuals([perturbed_copy(m, track, "rz", phi)], [p])[0]
    local = r0 + track.measured
    u, v = local[:, 0], local[:, 1]
    expected = torch.stack((math.cos(phi) * u + math.sin(phi) * v - u,
                            -math.sin(phi) * u + math.cos(phi) * v - v), dim=1)
    expected[~track.target] = 0.
    torch.testing.assert_close(rg - r0, expected, rtol=0, atol=1e-9)
    # First-order form used by the analytic Jacobian.
    first_order = torch.stack((phi * v, -phi * u), dim=1)
    assert float((expected[track.target] - first_order[track.target]).abs().max()) < phi ** 2 * 10


def test_zero_amplitude_reproduces_baseline_predictions_bitwise():
    m = module()
    track, p = synthetic_track()
    r0 = m.residuals([track], [p])[0]
    again = m.residuals([perturbed_copy(m, track, "tx", 0.)], [p])[0]
    assert torch.equal(r0, again)


def test_pairing_records_lost_tracks_with_reason():
    m = module()
    track, _ = synthetic_track()
    other = replace(track, event_nr=4)
    changed = replace(track, event_nr=5, measured=track.measured + 1.)
    branch_changed = replace(changed, measured=track.measured.clone())
    paired, lost = m.pair_branch([track, other, changed], [track, branch_changed])
    assert [t.event_nr for t in paired] == [3]
    reasons = {row["event_nr"]: row["reason"] for row in lost}
    assert reasons[4] == "absent_in_perturbed_reconstruction"
    assert reasons[5].startswith("pairing_failed")


def test_angle_difference_wraps_into_half_open_interval():
    m = module()
    assert m.wrap_angle(math.pi - .1 - (-math.pi + .1)) == pytest.approx(-.2)
    assert m.wrap_angle(-math.pi) == pytest.approx(math.pi)
    assert m.wrap_angle(.3) == pytest.approx(.3)


def test_closure_metrics_report_absolute_and_relative_scales():
    m = module()
    full = np.array([1., -1., 2.])
    geometry = np.array([.5, -.5, 1.])
    track = np.array([.5, -.5, .9])
    metrics = m.closure_summary(geometry, track, full)
    assert metrics["count"] == 3
    assert metrics["closure_rmse"] == pytest.approx(math.sqrt(.01 / 3))
    assert metrics["eta_closure"] == pytest.approx(.1 / math.sqrt(6))
    assert metrics["eta_aux"] == pytest.approx(.1 / (math.sqrt(1.5) + math.sqrt(1.31)))


def test_amplitude_labels_are_unique_and_signed():
    m = module()
    assert m.amplitude_label("tx", .5) == "p0p5mm"
    assert m.amplitude_label("ty", -.5) == "m0p5mm"
    assert m.amplitude_label("rz", -.02) == "m0p02rad"
