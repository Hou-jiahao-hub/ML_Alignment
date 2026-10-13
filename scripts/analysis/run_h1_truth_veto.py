"""Level F: the one-shot H1 truth veto, Schur candidate against the fixed-track arm.

POST-HOC ONLY -- NEVER A SELECTION INPUT.  Pre-registered in
``ml_optimization_log.md`` Iteration 41 §B (criteria) and Iteration 46
(implementation); run once after the Level E acceptance (Iteration 45).

* Geometry: per target layer and active DOF, ``RMS(e_C) / RMS(e_A)`` with
  ``e = a_hat - delta_true`` in the misaligned local frame, sensor-paired
  bootstrap (2000 resamples, seed 6006, two-sided 95%).
* Tracks: distance from every hit's truth position to the fitted helix, Schur
  tracks against the ref-only refit, target and reference hits separately,
  event-block bootstrap (``truth_veto_gate.paired_truth_distance_ratio_ci``).
* Falsification: the joint objective of C below that of A while a truth
  ratio's lower bound exceeds one; or an H1-D improvement while the P9
  nominal-reference criterion failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_manifest import (  # noqa: E402
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from project_config import load_project_config  # noqa: E402
from truth_veto_gate import paired_truth_distance_ratio_ci  # noqa: E402

PREREGISTRATION = "ml_optimization_log.md Iteration 41 §B / Iteration 46"
BOOTSTRAP_SEED = 6006
N_BOOTSTRAP = 2000
CI_QUANTILE = 0.975
NON_INFERIORITY_MAX = 1.01
IMPROVEMENT_MAX = 0.90
DIRECTED_CELLS = ("17:8|tx", "17:8|ty", "24:4|ty")
TARGET_LAYERS = ((17, 8), (24, 4))
ACTIVE_DOF = (0, 1, 5)
DOF_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
HIT_KEY = ["event_nr", "track_nr", "volume_id", "layer_id", "module_id"]
DTYPE = torch.float64


def paired_ratio(candidate, baseline, *, seed=BOOTSTRAP_SEED, n_bootstrap=N_BOOTSTRAP, block_ids=None,
                 candidate_valid=None, baseline_valid=None) -> dict:
    """``RMS(candidate) / RMS(baseline)`` with a paired block bootstrap and both CI ends.

    The upper end is ``truth_veto_gate``'s; the lower end reuses the same draws
    with the arms swapped: each resample of ``A/C`` is ``1/(C/A)``, so the
    0.975 "higher" quantile of ``A/C`` is the reciprocal of the 0.025 "lower"
    quantile of ``C/A``.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if block_ids is None:
        block_ids = np.arange(candidate.size).reshape(candidate.shape)
    valid_c = np.ones(candidate.shape, bool) if candidate_valid is None else candidate_valid
    valid_b = np.ones(baseline.shape, bool) if baseline_valid is None else baseline_valid
    kwargs = dict(bootstrap_seed=seed, n_bootstrap=n_bootstrap, ci_quantile=CI_QUANTILE)
    forward = paired_truth_distance_ratio_ci(candidate, baseline, block_ids, valid_c, valid_b, **kwargs)
    backward = paired_truth_distance_ratio_ci(baseline, candidate, block_ids, valid_b, valid_c, **kwargs)
    return {
        "nominal_ratio": forward["nominal_ratio"],
        "ci_lower": 1.0 / backward["ci_upper"],
        "ci_upper": forward["ci_upper"],
        "n_blocks": forward["n_events"],
        "n_values": forward["n_hits"],
        "bootstrap": forward["bootstrap"],
    }


def attach_truth(frame: pd.DataFrame, surface_keys, event_track_ids, sensor_idx, hit_mask, measured):
    """Truth positions [T, H, 3] for the batch hits, matched on ``HIT_KEY``.

    Fails closed if a batch hit has no unique state or if the matched state's
    ``l_x_hit`` differs from the batch measurement.
    """
    rows, slots = torch.nonzero(hit_mask, as_tuple=True)
    keys = np.array([surface_keys[int(s)] for s in sensor_idx[rows, slots].tolist()], dtype=np.int64)
    hits = pd.DataFrame({
        "event_nr": event_track_ids[rows, 0].numpy(), "track_nr": event_track_ids[rows, 1].numpy(),
        "volume_id": keys[:, 0], "layer_id": keys[:, 1], "module_id": keys[:, 2],
        "_measured": measured[rows, slots, 0].numpy(),
    })
    states = frame[HIT_KEY + ["l_x_hit", "t_x", "t_y", "t_z"]].astype({k: np.int64 for k in HIT_KEY})
    if states.duplicated(HIT_KEY).any():
        raise ValueError("Trackstates contain duplicate hit keys.")
    merged = hits.merge(states, on=HIT_KEY, how="left", validate="one_to_one")
    if merged["t_x"].isna().any():
        raise ValueError(f"{int(merged['t_x'].isna().sum())} batch hits are missing from the trackstates.")
    if not np.array_equal(merged["l_x_hit"].to_numpy(np.float64), merged["_measured"].to_numpy(np.float64)):
        raise ValueError("l_x_hit of a matched state differs from the batch measurement.")
    truth = torch.zeros((*hit_mask.shape, 3), dtype=DTYPE)
    truth[rows, slots] = torch.from_numpy(merged[["t_x", "t_y", "t_z"]].to_numpy(np.float64))
    return truth


def h1_verdict(geometry: dict, tracks: dict, objective_c: float, objective_a: float,
               nominal_reference_independent: bool) -> dict:
    """Iteration 41 §B: H1-P veto, H1-D directed improvement and the two falsifications."""
    metrics = {f"geometry:{name}": value for name, value in geometry.items()}
    metrics.update({f"track:{name}": value for name, value in tracks.items()})
    failed = [name for name, value in metrics.items() if not value["ci_upper"] <= NON_INFERIORITY_MAX]
    improved = [name for name in DIRECTED_CELLS if geometry[name]["ci_upper"] < IMPROVEMENT_MAX]
    worse = [name for name, value in metrics.items() if value["ci_lower"] > 1.0]
    chi2_truth = objective_c < objective_a and bool(worse)
    nominal_only = bool(improved) and not nominal_reference_independent
    return {
        "h1_p_passed": not failed,
        "h1_p_failed": failed,
        "h1_d_improved": improved,
        "h1_d_claim_scope": "post_hoc_single_sample",
        "significantly_worse": worse,
        "falsified_chi2_better_truth_worse": chi2_truth,
        "falsified_nominal_reference_only": nominal_only,
        "falsified": chi2_truth or nominal_only,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _repo(*parts) -> Path:
    return SCRIPT_DIR.parents[1].joinpath(*parts)


EXPERIMENTS = _repo("data/experiments/surface_helix/Vol17_Lay8__Vol24_Lay4")
ALIGNMENT = _repo("data/alignment/Vol17_Lay8__Vol24_Lay4")
DEFAULTS = {
    "nominal_geom": _repo("data/geometry/odd_transforms.json"),
    "misaligned_geom": ALIGNMENT / "misalignment_applied_v17l8_v24l4.json",
    "baseline_geom": ALIGNMENT / "aligned_transforms_schur-p0-fixedtrack-20260922_joint_free.json",
    "candidate_geom": ALIGNMENT / "aligned_transforms_p9-schur-observability-20261012_schur_lm.json",
    "states": _repo("data/reconstruction/misaligned/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root"),
    "cache_dir": EXPERIMENTS / "m1-fixed-track-observability-20261011" / "prepare" / "cache",
    "baseline_control": EXPERIMENTS / "m1-fixed-track-observability-20261011" / "control" / "arms" / "full_control.npz",
    "candidate_control": EXPERIMENTS / "p9-schur-observability-20261012" / "control" / "arms" / "full_control.npz",
    "p9_report": EXPERIMENTS / "p9-schur-observability-20261012" / "report" / "p9_result.json",
}


def geometry_metrics(args) -> dict:
    from compare_geometries import load_geometry
    from run_truth_veto_closure import _local_delta

    nominal = load_geometry(args.nominal_geom, label="nominal")
    misaligned = load_geometry(args.misaligned_geom, label="misaligned")
    arms = {"A": load_geometry(args.baseline_geom, label="baseline"),
            "C": load_geometry(args.candidate_geom, label="candidate")}
    for label, geometry in arms.items():
        if set(geometry) != set(misaligned):
            raise ValueError(f"Arm {label} covers a different sensor set from the misaligned overlay.")
    result = {}
    for volume, layer in TARGET_LAYERS:
        keys = sorted(key for key in misaligned if key[:2] == (volume, layer))
        truth = np.stack([_local_delta(misaligned[k], nominal[k]) for k in keys])
        errors = {label: np.stack([_local_delta(misaligned[k], geometry[k]) for k in keys]) - truth
                  for label, geometry in arms.items()}
        for dof in ACTIVE_DOF:
            name = f"{volume}:{layer}|{DOF_NAMES[dof]}"
            result[name] = paired_ratio(errors["C"][:, dof], errors["A"][:, dof])
            result[name]["rms_c"] = float(np.sqrt(np.mean(errors["C"][:, dof] ** 2)))
            result[name]["rms_a"] = float(np.sqrt(np.mean(errors["A"][:, dof] ** 2)))
    return result


def track_and_objective_metrics(args) -> tuple[dict, dict]:
    import joint_alignment_schur as J
    import ml_alignment_autograd as M
    from helix_model import closest_helix_points
    from measure_all_hit_refit import load_states

    batches = M.TrackBatches(**torch.load(Path(args.cache_dir) / "batches.pt", weights_only=False))
    tracks_a = torch.load(Path(args.cache_dir) / "tracks.pt", weights_only=False).to(DTYPE)
    control_c = np.load(args.candidate_control)
    tracks_c = torch.as_tensor(control_c["tracks"], dtype=DTYPE)
    geometry = M.load_joint_geometry(str(args.nominal_geom), str(args.misaligned_geom), M.parse_target_layers("17:8,24:4"))
    surface_keys = {index: key for key, index in geometry.id_to_idx.items()}

    frame = load_states(Path(args.states), None)
    truth = attach_truth(frame, surface_keys, batches.event_track_ids, batches.sensor_idx,
                         batches.hit_mask, batches.measured_local.to(DTYPE))
    events = batches.event_track_ids[:, 0, None].expand_as(batches.hit_mask).numpy()
    distances = {}
    for label, parameters in (("A", tracks_a), ("C", tracks_c)):
        expanded = parameters[:, None, :].expand(-1, truth.shape[1], -1)
        closest, _path, valid = closest_helix_points(expanded, truth)
        distances[label] = (torch.linalg.norm(closest - truth, dim=-1).numpy(), valid.numpy())
    hit_mask, is_target = batches.hit_mask.numpy(), (batches.is_target & batches.hit_mask).numpy()
    track_result = {}
    for name, subset in (("target", is_target), ("ref", hit_mask & ~is_target)):
        track_result[name] = paired_ratio(
            np.where(subset, distances["C"][0], 0.0), np.where(subset, distances["A"][0], 0.0),
            block_ids=events, candidate_valid=subset & distances["C"][1], baseline_valid=subset & distances["A"][1],
        )
        shared = subset & distances["C"][1] & distances["A"][1]
        track_result[name]["rms_c_mm"] = float(np.sqrt(np.mean(distances["C"][0][shared] ** 2)))
        track_result[name]["rms_a_mm"] = float(np.sqrt(np.mean(distances["A"][0][shared] ** 2)))

    objective = {}
    alignments = {}
    for label, path in (("A", args.baseline_control), ("C", args.candidate_control)):
        arm = np.load(path)
        alignment = torch.zeros((len(geometry.centers), 6), dtype=DTYPE)
        alignment[torch.as_tensor(arm["surfaces"])] = torch.as_tensor(arm["deltas"], dtype=DTYPE)
        alignments[label] = alignment
    residuals = {label: J.whitened_residuals(params, batches, geometry, alignments[label])
                 for label, params in (("A", tracks_a), ("C", tracks_c))}
    shared = residuals["A"][1] & residuals["C"][1]
    for label, (residual, _valid) in residuals.items():
        objective[label] = 0.5 * float(residual[shared].square().sum())
    objective["n_components"] = int(shared.sum())
    return track_result, objective


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(EXPERIMENTS))
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=str(value))
    parser.add_argument("--capture-source-snapshot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root) / args.run_id
    if (root / "manifest.json").exists():
        raise SystemExit(f"{root} exists: the H1 truth veto runs once.")
    metrics = {"preregistration": PREREGISTRATION, "claim_scope": "post_hoc_truth_veto_only"}
    inputs = {name: getattr(args, name) for name in DEFAULTS if name != "cache_dir"}
    inputs.update(cache_batches=Path(args.cache_dir) / "batches.pt", cache_tracks=Path(args.cache_dir) / "tracks.pt")
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    manifest = write_experiment_manifest(root, run_id=args.run_id, command=command, config=load_project_config(None), inputs=inputs,
                                         metrics=metrics, capture_source_snapshot=args.capture_source_snapshot)
    started = time.perf_counter()
    try:
        # Tracks first: the hit matching is what can fail on engineering
        # grounds, and it must do so before any geometry ratio exists.
        tracks, objective = track_and_objective_metrics(args)
        geometry = geometry_metrics(args)
        independent = json.loads(Path(args.p9_report).read_text())["h1_r"]["nominal_reference_independent"]
        verdict = h1_verdict(geometry, tracks, objective["C"], objective["A"], independent)
        result = {**metrics, "geometry": geometry, "tracks": tracks, "joint_objective": objective,
                  "nominal_reference_independent": independent, "verdict": verdict}
        path = root / "h1_truth_veto.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        metrics.update(verdict=verdict, duration_s=time.perf_counter() - started)
        finalize_experiment_manifest(manifest, status="completed", metrics=metrics, outputs={"result": path})
        print(json.dumps(verdict, indent=2))
        return 0
    except BaseException as error:
        metrics.update(duration_s=time.perf_counter() - started, error_type=type(error).__name__,
                       error_message=str(error))
        finalize_experiment_manifest(manifest, status="failed", metrics=metrics)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
