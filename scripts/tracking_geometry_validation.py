"""Track-geometry validation for known single-DOF local misalignments (P0-P5).

Each unit injects one deterministic local translation or rotation into every
sensor of the tested layer, starting from nominal geometry. The stages separate
the response of the helix Truth diagnostic into a geometry-only part (frozen
nominal helices on perturbed planes), a track-response part (ACTS rerun on the
perturbed geometry, new SMT seed and non-target refit, scored on nominal
planes) and the full response, then check closure, finite-difference
Jacobians and block-LM recovery of the injected parameter.

Residuals are prediction minus measurement, ``r = p_local - m_local``. The
joint solver internally uses ``m - p``; its Jacobian is sign-flipped before
comparison. ACTS is only ever run in a fresh LCG shell through
``alignment_odd.py``; this module never imports ``acts``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.append(_SCRIPT_DIRECTORY)

try:
    from . import helix_model as hm
    from . import validate_helix_truth as vht
    from .experiment_manifest import (finalize_experiment_manifest,
                                      write_experiment_manifest)
    from .project_config import absolute_project_path, load_project_config
    from .validate_helix_model import load_nominal_geometry, load_working_geometry
except ImportError:  # Direct CLI execution.
    import helix_model as hm
    import validate_helix_truth as vht
    from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from project_config import absolute_project_path, load_project_config
    from validate_helix_model import load_nominal_geometry, load_working_geometry


SCHEMA_VERSION = 1
LAYER = (17, 8)
LAYER_TAG = "17_8"
DOF_INDEX = {"tx": 0, "ty": 1, "rz": 5}
DOF_UNIT = {"tx": "mm", "ty": "mm", "rz": "rad"}
AMPLITUDES = {"tx": (-0.5, 0.5), "ty": (-0.5, 0.5), "rz": (-0.02, 0.02)}
GEOMETRY_FD_STEPS = {"tx": 1.e-3, "ty": 1.e-3, "rz": 1.e-4}
PHYSICAL_TRACK_FD_STEPS = (1.e-4, 1.e-4, 1.e-4, 1.e-7, 1.e-7, 1.e-10)
GAUGE_TRACK_FD_STEPS = (1.e-4, 1.e-4, 1.e-7, 1.e-7, 1.e-6)
GAUGE_JACOBIAN_SAMPLE_TRACKS = 512
MAX_EVENTS = 20000
TOLERANCES = {
    "p1_exact_mm": 1.e-9,
    "closure_eta_max": 0.10,
    "geometry_jacobian_relative_max": 1.e-6,
    "track_jacobian_relative_max": 1.e-5,
    "recovery_tau_rel": 0.05,
    "recovery_tau_abs_rule": "3 * RMS over converged sensors of the a=0 control correction, per DOF",
    "recovery_sensor_fraction_min": 0.95,
    "recovery_mean_rule": "|mean(e)| <= tau_rel*|a| + 3*RMS(e)/sqrt(N)",
    "crosstalk_rule": ">= 95% of converged sensors have |correction_j| <= tau_abs_j for non-injected DOF j",
}
# Same solver call as run_surface_helix_experiment.sh (joint_free stage); the
# solver's own defaults (e.g. 200 SMT seed iterations) fail its stationarity gate.
SOLVER_OPTIONS = ("--track-mode", "free", "--active-dof", "tx,ty,rz",
                  "--joint-seed-mode", "non-target-measurement", "--joint-seed-min-hits", "8",
                  "--joint-seed-min-span", "300", "--joint-smt-seed-iters", "2000",
                  "--joint-measurement-refit-iters", "200", "--joint-solver", "block-lm",
                  "--lm-max-iters", "60", "--lm-max-step-r", "5e-3", "--joint-iters", "1000",
                  "--max-tracks", "0", "--lambda-l2", "1e-5")
CONFIG_TEXT = ('[alignment]\ntarget_layers = ["17:8"]\nactive_dof = [0, 1, 5]\n')
DEFAULT_ROOT = "data/experiments/helix_truth/Vol17_Lay8__Vol24_Lay4/tracking_geometry_validation"
NOMINAL_STATES = "data/reconstruction/nominal/Vol17_Lay8__Vol24_Lay4/trackstates_kf_v17l8_v24l4.root"
NOMINAL_GEOMETRY = "data/geometry/odd_transforms.json"
SIMULATION_DIR = "data/simulation"
KEY_FIELDS = ("event_nr", "track_nr", "volume_id", "layer_id", "module_id", "occurrence")


# ----------------------------------------------------------------------------
# Geometry injection
# ----------------------------------------------------------------------------

def _solver():
    try:
        from . import ml_alignment_autograd as solver
    except ImportError:
        import ml_alignment_autograd as solver
    return solver


def surface_key(identifier):
    match = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", identifier)
    if match is None:
        raise ValueError(f"Surface ID has no vol/lay/sen fields: {identifier!r}")
    return tuple(map(int, match.groups()))


def apply_local_delta(center, rotation, delta):
    """``c + R dt`` and ``R @ (Rz Ry Rx)``, the solver's local convention."""
    center = np.asarray(center, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    delta = np.asarray(delta, dtype=float)
    return (center + rotation @ delta[:3],
            rotation @ _solver().euler_to_rotation_matrix(*delta[3:]))


def delta_vector(dof, amplitude):
    if dof not in DOF_INDEX:
        raise ValueError(f"Unsupported DOF {dof!r}; expected one of {sorted(DOF_INDEX)}")
    delta = np.zeros(6)
    delta[DOF_INDEX[dof]] = float(amplitude)
    return delta


def perturb_layer(items, layer, dof, amplitude):
    """Return only the tested layer's sensors, each shifted by the same amount."""
    delta = delta_vector(dof, amplitude)
    output = []
    for item in items:
        if surface_key(item["ID"])[:2] != tuple(layer):
            continue
        center, rotation = apply_local_delta(item["Translation"], item["RotationMatrix"], delta)
        output.append({"ID": item["ID"], "Translation": center.tolist(),
                       "RotationMatrix": rotation.tolist()})
    return output


def amplitude_label(dof, amplitude):
    sign = "p" if amplitude > 0 else "m" if amplitude < 0 else "z"
    return f"{sign}{format(abs(amplitude), 'g').replace('.', 'p')}{DOF_UNIT[dof]}"


@dataclass(frozen=True)
class Unit:
    name: str
    dof: str | None
    amplitude: float
    directory: Path

    @property
    def control(self):
        return self.dof is None


def all_units(root, dofs=tuple(DOF_INDEX)):
    units = [Unit("a0", None, 0., root / "controls" / "a0")]
    for dof in dofs:
        for amplitude in AMPLITUDES[dof]:
            label = amplitude_label(dof, amplitude)
            units.append(Unit(f"{dof}_{label}", dof, amplitude,
                              root / f"{LAYER_TAG}_{dof}" / f"amp_{label}"))
    return units


# ----------------------------------------------------------------------------
# Residuals and pairing
# ----------------------------------------------------------------------------

def with_geometry(track, centers, rotations, id_to_idx):
    index = [id_to_idx[key[:3]] for key in track.identities]
    return replace(track, centers=centers[index], rotations=rotations[index])


def predict(tracks, parameters, chunk=1024):
    """Frozen helices through ordered planes: local prediction and validity."""
    locals_, valid = [], []
    for start in range(0, len(tracks), chunk):
        batch = tracks[start:start + chunk]
        results = [vht.FitResult(p, None, True, True) for p in parameters[start:start + chunk]]
        for rows in vht.ordered_predictions(batch, results):
            locals_.append(torch.stack([row[0] for row in rows]))
            valid.append(torch.tensor([row[3] for row in rows], dtype=torch.bool))
    return locals_, valid


def residuals(tracks, parameters, chunk=1024):
    locals_, valid = predict(tracks, parameters, chunk)
    output = []
    for track, local, ok in zip(tracks, locals_, valid):
        value = local - track.measured
        value[~ok] = float("nan")
        output.append(value)
    return output


def pair_branch(nominal, branch):
    """Pair perturbed-reconstruction SMT onto frozen nominal tracks.

    Identity is event/track number plus bitwise-equal measurements, variances
    and masks (``merge_misaligned_track``); anything else is reported lost.
    """
    by_key = {(t.event_nr, t.track_nr): t for t in branch}
    paired, lost = [], []
    for track in nominal:
        other = by_key.get((track.event_nr, track.track_nr))
        if other is None:
            lost.append({"event_nr": track.event_nr, "track_nr": track.track_nr,
                         "reason": "absent_in_perturbed_reconstruction"})
            continue
        try:
            paired.append(vht.merge_misaligned_track(track, other))
        except ValueError as error:
            lost.append({"event_nr": track.event_nr, "track_nr": track.track_nr,
                         "reason": f"pairing_failed: {error}"})
    return paired, lost


def wrap_angle(value):
    wrapped = np.mod(np.asarray(value, dtype=float) + math.pi, 2 * math.pi) - math.pi
    wrapped = np.where(wrapped == -math.pi, math.pi, wrapped)
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped


def parameter_difference(after, before):
    difference = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    difference[..., 3] = wrap_angle(difference[..., 3])
    return difference


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def _num(value):
    value = float(value)
    return value if math.isfinite(value) else None


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "rms": None, "p95_abs": None, "max_abs": None}
    return {"count": int(len(values)), "mean": _num(values.mean()),
            "rms": _num(np.sqrt(np.mean(values ** 2))),
            "p95_abs": _num(np.quantile(np.abs(values), .95)),
            "max_abs": _num(np.abs(values).max())}


def closure_summary(geometry, track, full):
    geometry, track, full = (np.asarray(v, dtype=float).ravel() for v in (geometry, track, full))
    finite = np.isfinite(geometry) & np.isfinite(track) & np.isfinite(full)
    g, t, f = geometry[finite], track[finite], full[finite]
    e = f - g - t
    norm = {name: float(np.linalg.norm(v)) for name, v in (("full", f), ("geometry", g),
                                                          ("track", t), ("closure", e))}
    return {"count": int(finite.sum()), "excluded_non_finite": int((~finite).sum()),
            "geometry": stats(g), "track": stats(t), "full": stats(f),
            "closure_mean": _num(e.mean()) if len(e) else None,
            "closure_rmse": _num(np.sqrt(np.mean(e ** 2))) if len(e) else None,
            "closure_p95_abs": _num(np.quantile(np.abs(e), .95)) if len(e) else None,
            "closure_max_abs": _num(np.abs(e).max()) if len(e) else None,
            "norms": norm,
            "eta_closure": norm["closure"] / norm["full"] if norm["full"] > 0 else None,
            "eta_aux": (norm["closure"] / (norm["geometry"] + norm["track"])
                        if norm["geometry"] + norm["track"] > 0 else None)}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    vht.write_json(path, payload)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float(text):
    return float(text) if text not in ("", None) else float("nan")


# ----------------------------------------------------------------------------
# Stage bookkeeping
# ----------------------------------------------------------------------------

class Context:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root)
        self.config_path = self.root / "config_v17l8.toml"
        if self.config_path.exists():
            # alignment_artifact_layout freezes target layers at import time.
            os.environ["ML_ALIGNMENT_CONFIG"] = str(self.config_path)
        self.config = load_project_config(self.config_path if self.config_path.exists() else None)
        self.repo = self.config.repository_root
        self.nominal_states = Path(absolute_project_path(args.nominal_states))
        self.nominal_geometry = Path(absolute_project_path(args.nominal_geometry))
        self.simulation_dir = Path(absolute_project_path(args.simulation_dir))
        self.threads = args.threads
        self.batch_size = args.batch_size
        self.max_events = args.max_events

    def require_layer_config(self):
        if not self.config_path.exists():
            raise RuntimeError("Run the prepare stage first")
        os.environ["ML_ALIGNMENT_CONFIG"] = str(self.config_path)
        self.config = load_project_config(self.config_path)
        if list(self.config.target_layers) != [LAYER]:
            raise RuntimeError(f"Target layers must be exactly {LAYER}: {self.config.target_layers}")

    @property
    def prereg(self):
        return json.loads((self.root / "preregistration.json").read_text())


def manifest_status(directory):
    path = Path(directory) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("status")


def run_stage(ctx, directory, name, inputs, body):
    """Write a running manifest, execute ``body(metrics)``, then finalize it."""
    directory = Path(directory)
    if manifest_status(directory) == "completed":
        print(f"[skip] {name}: {directory} already completed", flush=True)
        return json.loads((directory / "manifest.json").read_text())["metrics"]
    metrics = {"stage": name, "schema_version": SCHEMA_VERSION}
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    manifest = write_experiment_manifest(
        directory, run_id=f"{name}-{directory.name}", command=command, config=ctx.config,
        inputs=inputs, metrics=metrics, capture_source_snapshot=True)
    started = time.time()
    print(f"[run ] {name}: {directory}", flush=True)
    try:
        outputs = body(metrics) or {}
        metrics["duration_s"] = time.time() - started
        finalize_experiment_manifest(manifest, status="completed", metrics=metrics, outputs=outputs)
        return metrics
    except (Exception, KeyboardInterrupt) as error:
        metrics.update(duration_s=time.time() - started, error=f"{type(error).__name__}: {error}")
        finalize_experiment_manifest(manifest, status="failed", metrics=metrics)
        raise


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listing_digest(directory):
    """Names and sizes only: the simulation tree has ~6e5 files."""
    digest, count, total = hashlib.sha256(), 0, 0
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if entry.is_file():
            size = entry.stat().st_size
            digest.update(f"{entry.name}\0{size}\n".encode())
            count += 1
            total += size
    return {"path": str(directory), "files": count, "bytes": total,
            "names_and_sizes_sha256": digest.hexdigest()}


# ----------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------

def stage_prepare(ctx):
    import uproot

    ctx.root.mkdir(parents=True, exist_ok=True)
    if ctx.config_path.exists():
        if ctx.config_path.read_text() != CONFIG_TEXT:
            raise RuntimeError(f"{ctx.config_path} differs from the frozen configuration")
    else:
        ctx.config_path.write_text(CONFIG_TEXT)
    ctx.require_layer_config()

    def body(metrics):
        with uproot.open(ctx.nominal_states) as root:
            events = np.unique(root["trackstates"]["event_nr"].array(library="np"))
        acts_events = int(events[:ctx.max_events].max()) + 1
        items = json.loads(ctx.nominal_geometry.read_text())
        layer_sensors = sum(surface_key(i["ID"])[:2] == LAYER for i in items)
        prereg = {
            "schema_version": SCHEMA_VERSION,
            "plan": "tracking_geometry_validation_plan.md (2026-10-08), first batch A1-A3",
            "layer": f"{LAYER[0]}:{LAYER[1]}", "layer_sensors": layer_sensors,
            "target_layers": ["17:8"], "non_target_note": "24:4 hits enter the SMT seed and refit",
            "dofs": list(DOF_INDEX), "amplitudes": {k: list(v) for k, v in AMPLITUDES.items()},
            "units": DOF_UNIT, "injection": "deterministic, identical for every sensor of the layer",
            "convention": {"translation": "c' = c + R[:, axis] * a (local)",
                           "rotation": "R' = R @ Rz(phi), about the sensor centre (local)",
                           "residual": "r = p_local - m_local (solver uses m - p)",
                           "pairing": "event_nr, track_nr + bitwise-equal measurement/variance/mask",
                           "psi0_difference": "wrapped into (-pi, pi]"},
            "control": "a=0: ACTS rerun through the alignment decorator with nominal 17:8 transforms",
            "max_events": ctx.max_events, "acts_num_events": acts_events,
            "geometry_fd_steps": GEOMETRY_FD_STEPS, "geometry_fd_multipliers": [.5, 1., 2.],
            "physical_track_fd_steps": PHYSICAL_TRACK_FD_STEPS,
            "gauge_track_fd_steps": GAUGE_TRACK_FD_STEPS,
            "gauge_jacobian_sample_tracks": GAUGE_JACOBIAN_SAMPLE_TRACKS,
            "solver_options": list(SOLVER_OPTIONS), "tolerances": TOLERANCES,
            "inputs": {"nominal_states_sha256": sha256(ctx.nominal_states),
                       "nominal_geometry_sha256": sha256(ctx.nominal_geometry),
                       "simulation": listing_digest(ctx.simulation_dir)},
        }
        path = ctx.root / "preregistration.json"
        if path.exists():
            frozen = json.loads(path.read_text())
            frozen.pop("created_at", None)
            if frozen != json.loads(json.dumps(prereg)):
                raise RuntimeError("preregistration.json is frozen and differs from this run")
        else:
            write_json(path, {**prereg, "created_at": datetime.now(timezone.utc).isoformat()})
        for unit in all_units(ctx.root):
            injected = perturb_layer(items, LAYER, unit.dof or "tx", unit.amplitude)
            target = unit.directory / "injected_geometry.json"
            if target.exists() and json.loads(target.read_text()) != injected:
                raise RuntimeError(f"{target} differs from the frozen injection")
            write_json(target, injected)
        metrics.update(acts_num_events=acts_events, layer_sensors=layer_sensors,
                       units=[u.name for u in all_units(ctx.root)])
        return {"preregistration": path}

    run_stage(ctx, ctx.root / "prepare", "prepare",
              {"nominal_states": ctx.nominal_states, "nominal_geometry": ctx.nominal_geometry}, body)


# ----------------------------------------------------------------------------
# P0 baseline
# ----------------------------------------------------------------------------

def _truth_argv(ctx, output, max_events):
    return ["--config", str(ctx.config_path), "--geometry", "nominal",
            "--nominal-states", str(ctx.nominal_states),
            "--nominal-geometry", str(ctx.nominal_geometry),
            "--max-events", str(max_events), "--batch-size", str(ctx.batch_size),
            "--threads", str(ctx.threads), "--output-dir", str(output), "--run-tag", output.name]


def stage_baseline(ctx):
    ctx.require_layer_config()
    baseline = ctx.root / "baseline"
    if manifest_status(baseline) != "completed":
        if vht.main(_truth_argv(ctx, baseline, ctx.max_events)) != 0:
            raise RuntimeError("P0 baseline failed; see baseline/manifest.json")
    repro = ctx.root / "baseline_repro"
    if not (repro / "reproducibility.json").exists():
        runs = [repro / "run1", repro / "run2"]
        repro_events = min(500, ctx.max_events)
        for run in runs:
            if manifest_status(run) != "completed" and vht.main(_truth_argv(ctx, run, repro_events)) != 0:
                raise RuntimeError(f"Reproducibility run failed: {run}")
        tables = [{(int(r["event_nr"]), int(r["track_nr"])): r for r in read_csv(run / "per_track.csv")}
                  for run in (*runs, baseline)]
        names = [f"refit_{p}" for p in vht.PARAMETERS] + [f"seed_{p}" for p in vht.PARAMETERS]

        def max_difference(a, b):
            keys = sorted(set(a) & set(b))
            values = [abs(_float(a[k][n]) - _float(b[k][n])) for k in keys for n in names]
            finite = [v for v in values if math.isfinite(v)]
            return {"tracks": len(keys), "max_abs_parameter_difference": max(finite) if finite else None,
                    "identical": all(a[k][n] == b[k][n] for k in keys for n in names)}

        write_json(repro / "reproducibility.json", {
            "schema_version": SCHEMA_VERSION, "max_events": repro_events,
            "run1_vs_run2": max_difference(tables[0], tables[1]),
            "run1_vs_baseline_subset": max_difference(tables[0], tables[2]),
            "note": "subset comparison mixes batch composition; run1_vs_run2 is the repeatability check"})


def load_cohort(ctx):
    """Frozen P0 cohort with nominal geometry and the saved baseline helices."""
    tracks, _excluded, counts = vht.load_tracks(ctx.nominal_states, ctx.nominal_geometry,
                                                ctx.config.target_layers, max_events=ctx.max_events)
    saved = {(int(r["event_nr"]), int(r["track_nr"])): r
             for r in read_csv(ctx.root / "baseline" / "per_track.csv")}
    if {(t.event_nr, t.track_nr) for t in tracks} != set(saved):
        raise RuntimeError("Reloaded cohort differs from the frozen P0 per_track.csv")

    def params(prefix):
        output = []
        for track in tracks:
            row = saved[(track.event_nr, track.track_nr)]
            values = [_float(row[f"{prefix}_{p}"]) for p in vht.PARAMETERS]
            output.append(torch.tensor(values, dtype=torch.float64)
                          if all(map(math.isfinite, values)) else None)
        return output

    return tracks, params("refit"), params("seed"), counts


def check_baseline_predictions(ctx, tracks, theta0):
    """Recomputed nominal predictions must equal the saved P0 predictions."""
    saved = {}
    for row in read_csv(ctx.root / "baseline" / "per_surface.csv"):
        key = tuple(int(row[f]) for f in KEY_FIELDS)
        saved[key] = (_float(row["prediction_LOC0_mm"]), _float(row["prediction_LOC1_mm"]))
    locals_, valid = predict(tracks, theta0)
    worst, compared, mismatched_validity = 0., 0, 0
    for track, local, ok in zip(tracks, locals_, valid):
        for i, identity in enumerate(track.identities):
            reference = saved[(track.event_nr, track.track_nr, *identity)]
            if bool(ok[i]) != all(map(math.isfinite, reference)):
                mismatched_validity += 1
                continue
            if ok[i]:
                worst = max(worst, float(np.max(np.abs(local[i].numpy() - np.array(reference)))))
                compared += 1
    return {"surfaces_compared": compared, "max_abs_difference_mm": worst,
            "validity_mismatches": mismatched_validity}


# ----------------------------------------------------------------------------
# ACTS reconstruction (fresh LCG shell)
# ----------------------------------------------------------------------------

def stage_acts(ctx, unit):
    ctx.require_layer_config()
    out = unit.directory / "acts"
    suffix = f"tgv_{unit.name}"
    injected = unit.directory / "injected_geometry.json"
    events = ctx.prereg["acts_num_events"]

    def body(metrics):
        inner = " && ".join([
            f"source {shlex.quote(str(ctx.config.acts_setup))}",
            f"cd {shlex.quote(str(ctx.repo / 'scripts'))}",
            "exec python alignment_odd.py --config {} reconstruction --mode aligned --num-events {} "
            "--input-dir {} --output-dir {} --aligned-file {} --output-suffix {}".format(
                shlex.quote(str(ctx.config_path)), events, shlex.quote(str(ctx.simulation_dir)),
                shlex.quote(str(out)), shlex.quote(str(injected)), suffix)])
        env = {"HOME": os.environ.get("HOME", ""), "USER": os.environ.get("USER", ""),
               "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "ML_ALIGNMENT_CONFIG": str(ctx.config_path)}
        metrics.update(shell_command=inner, num_events=events, suffix=suffix)
        log = out / "reconstruction.log"
        with log.open("w") as stream:
            result = subprocess.run(["bash", "--noprofile", "--norc", "-c", inner], env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        metrics["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"ACTS reconstruction failed ({result.returncode}); see {log}")
        outputs = {"log": log}
        for stem in ("trackstates_kf", "tracksummary_kf", "performance_kf"):
            found = sorted(out.rglob(f"{stem}_{suffix}.root"))
            if len(found) != 1:
                raise RuntimeError(f"Expected one {stem}_{suffix}.root below {out}, found {found}")
            outputs[stem] = found[0]
            metrics[f"{stem}_sha256"] = sha256(found[0])
        import uproot
        with uproot.open(outputs["trackstates_kf"]) as root:
            numbers = root["trackstates"]["event_nr"].array(library="np")
        metrics.update(tracks_written=int(len(numbers)), events_with_tracks=int(len(np.unique(numbers))))
        return outputs

    out.mkdir(parents=True, exist_ok=True)
    run_stage(ctx, out, "acts", {"injected_geometry": injected, "nominal_geometry": ctx.nominal_geometry}, body)


def acts_output(unit, stem):
    found = sorted((unit.directory / "acts").rglob(f"{stem}_tgv_{unit.name}.root"))
    if len(found) != 1 or manifest_status(unit.directory / "acts") != "completed":
        raise RuntimeError(f"ACTS output {stem} for {unit.name} is not available")
    return found[0]


# ----------------------------------------------------------------------------
# P1 / P2 / P3 and closure
# ----------------------------------------------------------------------------

def fit_all(tracks, batch_size, seed_iters=600, refit_iters=60):
    fits = []
    for start in range(0, len(tracks), batch_size):
        batch = tracks[start:start + batch_size]
        try:
            fits.extend(vht.fit_tracks(batch, seed_iters=seed_iters, refit_iters=refit_iters))
        except (ValueError, FloatingPointError, torch.linalg.LinAlgError):
            for track in batch:
                try:
                    fits.extend(vht.fit_tracks([track], seed_iters=seed_iters, refit_iters=refit_iters))
                except (ValueError, FloatingPointError, torch.linalg.LinAlgError) as error:
                    fits.append(vht.FitResult(None, None, False, False, error=str(error)))
        if (start // batch_size) % 20 == 0:
            print(f"  fitted {min(start + batch_size, len(tracks))}/{len(tracks)}", flush=True)
    return fits


def layer_geometry(ctx, injected_path):
    nominal = load_nominal_geometry(ctx.nominal_geometry)
    working = load_working_geometry(injected_path, ctx.nominal_geometry)
    if working.id_to_idx != nominal.id_to_idx:
        raise RuntimeError("Working geometry ordering differs from nominal")
    outside = torch.tensor([key[:2] != LAYER for key in nominal.id_to_idx])
    if (not torch.equal(working.centers[outside], nominal.centers[outside])
            or not torch.equal(working.rotations[outside], nominal.rotations[outside])):
        raise RuntimeError("Injected geometry changes surfaces outside the tested layer")
    return nominal, working


def in_memory_geometry(nominal, dof, amplitude):
    centers, rotations = nominal.centers.clone(), nominal.rotations.clone()
    delta = delta_vector(dof, amplitude)
    for key, index in nominal.id_to_idx.items():
        if key[:2] == LAYER:
            c, r = apply_local_delta(centers[index].numpy(), rotations[index].numpy(), delta)
            centers[index] = torch.from_numpy(c)
            rotations[index] = torch.from_numpy(r)
    return centers, rotations


def p1_expected(local0, dof, amplitude):
    """Exact in-plane response of a frozen crossing point."""
    expected = np.zeros_like(local0)
    if dof in ("tx", "ty"):
        expected[:, DOF_INDEX[dof]] = -amplitude
    elif dof == "rz":
        u, v = local0[:, 0], local0[:, 1]
        c, s = math.cos(amplitude), math.sin(amplitude)
        expected[:, 0] = c * u + s * v - u
        expected[:, 1] = -s * u + c * v - v
    return expected


def flatten(tracks, values):
    return np.concatenate([v.numpy() for v in values]) if values else np.zeros((0, 2))


def key_rows(tracks):
    return [(t.event_nr, t.track_nr, *identity, bool(t.target[i]), t.measured[i].tolist(),
             t.truth[i].tolist()) for t in tracks for i, identity in enumerate(t.identities)]


def group_masks(keys):
    layer = np.array([f"{k[2]}:{k[3]}" for k in keys])
    target = np.array([k[6] for k in keys], dtype=bool)
    masks = {"all": np.ones(len(keys), dtype=bool), "target": target, "non_target": ~target}
    for name in sorted(set(layer)):
        masks[f"layer_{name}"] = layer == name
    return masks


def binned_profile(coordinate, values, edges):
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (coordinate >= low) & (coordinate < high) & np.isfinite(values)
        rows.append({"low": float(low), "high": float(high), "count": int(selected.sum()),
                     "mean": _num(values[selected].mean()) if selected.any() else None})
    return rows


def slope(coordinate, values):
    selected = np.isfinite(coordinate) & np.isfinite(values)
    if selected.sum() < 2:
        return None
    x, y = coordinate[selected], values[selected]
    return _num((x * y).sum() / (x * x).sum())


def stage_response(ctx, unit):
    ctx.require_layer_config()
    if manifest_status(ctx.root / "baseline") != "completed":
        raise RuntimeError("P0 baseline is not completed")
    injected = unit.directory / "injected_geometry.json"
    states = acts_output(unit, "trackstates_kf")

    def body(metrics):
        torch.set_num_threads(ctx.threads)
        tracks, theta0, seed0, counts = load_cohort(ctx)
        metrics["cohort"] = counts
        metrics["baseline_reproduction"] = check_baseline_predictions(ctx, tracks, theta0)
        if metrics["baseline_reproduction"]["max_abs_difference_mm"] > 1.e-12 or \
                metrics["baseline_reproduction"]["validity_mismatches"]:
            raise RuntimeError(f"P0 predictions not reproduced: {metrics['baseline_reproduction']}")
        nominal, working = layer_geometry(ctx, injected)
        memory_centers, memory_rotations = in_memory_geometry(nominal, unit.dof or "tx", unit.amplitude)
        metrics["json_vs_memory_geometry_max_abs"] = max(
            float((memory_centers - working.centers).abs().max()),
            float((memory_rotations - working.rotations).abs().max()))
        perturbed = [with_geometry(t, working.centers, working.rotations, working.id_to_idx) for t in tracks]

        # P1: frozen nominal helices on perturbed planes.
        r0_all = residuals(tracks, theta0)
        rg_all = residuals(perturbed, theta0)
        keys = key_rows(tracks)
        r0, rg = flatten(tracks, r0_all), flatten(tracks, rg_all)
        target = np.array([k[6] for k in keys], dtype=bool)
        local0 = r0 + np.array([k[7] for k in keys])
        expected = p1_expected(local0, unit.dof, unit.amplitude)
        p1_error = (rg - r0) - np.where(target[:, None], expected, 0.)
        finite = np.isfinite(p1_error)
        p1 = {"target_max_abs_error_mm": _num(np.abs(p1_error[finite & target[:, None]]).max()),
              "non_target_max_abs_response_mm": _num(np.abs(p1_error[finite & ~target[:, None]]).max()),
              "validity_changes": int((np.isfinite(r0) != np.isfinite(rg)).any(axis=1).sum()),
              "tracks_without_baseline_fit": sum(p is None for p in theta0)}
        p1["pass"] = bool(p1["target_max_abs_error_mm"] <= TOLERANCES["p1_exact_mm"]
                          and p1["non_target_max_abs_response_mm"] <= TOLERANCES["p1_exact_mm"])
        metrics["p1"] = p1

        # P2: perturbed-ACTS SMT seed + non-target refit, scored on nominal planes.
        branch, _, branch_counts = vht.load_tracks(states, ctx.nominal_geometry, ctx.config.target_layers,
                                                   max_events=0, retain_ineligible=True)
        paired, lost = pair_branch(tracks, branch)
        metrics["pairing"] = {"cohort_tracks": len(tracks), "paired_tracks": len(paired),
                              "retention": len(paired) / len(tracks),
                              "lost_reasons": dict(_count(row["reason"].split(":")[0] for row in lost)),
                              "branch_counts": branch_counts}
        write_csv(unit.directory / "track_response" / "lost_tracks.csv", lost,
                  ["event_nr", "track_nr", "reason"])
        print(f"  P2 refit of {len(paired)} paired tracks", flush=True)
        fits = fit_all(paired, ctx.batch_size)
        index = {(t.event_nr, t.track_nr): i for i, t in enumerate(tracks)}
        rows_track = []
        theta_delta, seed_delta, parameter_diffs, seed_diffs = [], [], [], []
        for track, fit in zip(paired, fits):
            i = index[(track.event_nr, track.track_nr)]
            theta_delta.append(fit.parameters)
            seed_delta.append(fit.seed_parameters)
            row = {"event_nr": track.event_nr, "track_nr": track.track_nr,
                   "fit_status": "failed" if fit.error else (
                       "stationary" if fit.seed_stationary and fit.refit_stationary else "non_stationary"),
                   "fit_error": fit.error, "seed_stationary": fit.seed_stationary,
                   "refit_stationary": fit.refit_stationary}
            for stage, now, before, store in (("seed", fit.seed_parameters, seed0[i], seed_diffs),
                                              ("refit", fit.parameters, theta0[i], parameter_diffs)):
                values = now.numpy() if now is not None else np.full(6, np.nan)
                reference = before.numpy() if before is not None else np.full(6, np.nan)
                delta = parameter_difference(values, reference)
                store.append(delta)
                for name, value, change in zip(vht.PARAMETERS, values, delta):
                    row[f"{stage}_{name}"] = _num(value)
                    row[f"delta_{stage}_{name}"] = _num(change)
            rows_track.append(row)
        write_csv(unit.directory / "track_response" / "per_track.csv", rows_track)
        parameter_diffs, seed_diffs = np.array(parameter_diffs), np.array(seed_diffs)
        metrics["p2_parameters"] = {
            stage: {name: stats(values[:, k]) for k, name in enumerate(vht.PARAMETERS)}
            for stage, values in (("seed", seed_diffs), ("refit", parameter_diffs))}
        metrics["p2_fit_status"] = dict(_count(r["fit_status"] for r in rows_track))

        # P2 / P3 residuals on the paired subset; P0/P1 reused by key.
        paired_nominal = [tracks[index[(t.event_nr, t.track_nr)]] for t in paired]
        paired_perturbed = [perturbed[index[(t.event_nr, t.track_nr)]] for t in paired]
        rt = flatten(paired, residuals(paired_nominal, theta_delta))
        rf = flatten(paired, residuals(paired_perturbed, theta_delta))
        seed_t = flatten(paired, residuals(paired_nominal, seed_delta))
        seed_0 = flatten(paired, residuals(paired_nominal, [seed0[index[(t.event_nr, t.track_nr)]]
                                                            for t in paired]))
        offsets = np.cumsum([0] + [len(t.identities) for t in tracks])
        take = np.concatenate([np.arange(offsets[index[(t.event_nr, t.track_nr)]],
                                         offsets[index[(t.event_nr, t.track_nr)] + 1]) for t in paired]) \
            if paired else np.zeros(0, dtype=int)
        pk = [keys[j] for j in take]
        p_r0, p_rg = r0[take], rg[take]
        d_g, d_t, d_f = p_rg - p_r0, rt - p_r0, rf - p_r0
        closure = d_f - d_g - d_t

        # Per-surface outputs.
        def base(k):
            return dict(zip(KEY_FIELDS, k[:6]), target=k[6],
                        measurement_LOC0_mm=_num(k[7][0]), measurement_LOC1_mm=_num(k[7][1]))

        geometry_rows = []
        for j, k in enumerate(keys):
            row = base(k)
            row.update(truth_LOC0_mm=_num(k[8][0]), truth_LOC1_mm=_num(k[8][1]))
            for a in range(2):
                row.update({f"r0_LOC{a}_mm": _num(r0[j, a]), f"rg_LOC{a}_mm": _num(rg[j, a]),
                            f"delta_rg_LOC{a}_mm": _num(rg[j, a] - r0[j, a]),
                            f"expected_delta_rg_LOC{a}_mm": _num(expected[j, a]) if k[6] else 0.})
            geometry_rows.append(row)
        write_csv(unit.directory / "geometry_only" / "per_surface.csv", geometry_rows)
        track_rows, full_rows = [], []
        for j, k in enumerate(pk):
            row_t, row_f = base(k), base(k)
            for a in range(2):
                row_t.update({f"r0_LOC{a}_mm": _num(p_r0[j, a]), f"rt_LOC{a}_mm": _num(rt[j, a]),
                              f"delta_rt_LOC{a}_mm": _num(d_t[j, a]),
                              f"delta_rt_seed_LOC{a}_mm": _num(seed_t[j, a] - seed_0[j, a])})
                row_f.update({f"r0_LOC{a}_mm": _num(p_r0[j, a]), f"rfull_LOC{a}_mm": _num(rf[j, a]),
                              f"delta_rg_LOC{a}_mm": _num(d_g[j, a]),
                              f"delta_rt_LOC{a}_mm": _num(d_t[j, a]),
                              f"delta_rfull_LOC{a}_mm": _num(d_f[j, a]),
                              f"closure_LOC{a}_mm": _num(closure[j, a])})
            track_rows.append(row_t)
            full_rows.append(row_f)
        write_csv(unit.directory / "track_response" / "per_surface.csv", track_rows)
        write_csv(unit.directory / "full_response" / "per_surface.csv", full_rows)

        # Closure metrics (P4 part 1).
        masks = group_masks(pk)
        groups = {}
        for name, mask in masks.items():
            groups[name] = {"LOC0": closure_summary(d_g[mask, 0], d_t[mask, 0], d_f[mask, 0]),
                            "LOC1": closure_summary(d_g[mask, 1], d_t[mask, 1], d_f[mask, 1]),
                            "both": closure_summary(d_g[mask], d_t[mask], d_f[mask])}
        seed_response = {name: {f"LOC{a}": stats((seed_t - seed_0)[mask, a]) for a in range(2)}
                         for name, mask in masks.items()}
        target_mask = masks["target"]
        local_paired = p_r0 + np.array([k[7] for k in pk]) if pk else np.zeros((0, 2))
        profiles = {}
        for label, values in (("geometry", d_g), ("track", d_t), ("full", d_f)):
            profiles[label] = {
                "LOC0_vs_LOC1": binned_profile(local_paired[target_mask, 1], values[target_mask, 0],
                                               np.linspace(-36, 36, 13)),
                "LOC1_vs_LOC0": binned_profile(local_paired[target_mask, 0], values[target_mask, 1],
                                               np.linspace(-9, 9, 13)),
                "slope_LOC0_per_LOC1": slope(local_paired[target_mask, 1], values[target_mask, 0]),
                "slope_LOC1_per_LOC0": slope(local_paired[target_mask, 0], values[target_mask, 1])}
        any_invalid = ~(np.isfinite(d_g) & np.isfinite(d_t) & np.isfinite(d_f)).all(axis=1)
        eta = groups["target"]["both"]["eta_closure"]
        p4 = {"eta_target": eta, "target_invalid_rows": int((any_invalid & target_mask).sum()),
              "closure_rmse_target_mm": groups["target"]["both"]["closure_rmse"],
              "pass": None if unit.control else bool(eta is not None
                                                     and eta <= TOLERANCES["closure_eta_max"])}
        metrics["p4_closure"] = p4
        write_json(unit.directory / "closure_metrics.json", {
            "schema_version": SCHEMA_VERSION, "unit": unit.name, "dof": unit.dof,
            "amplitude": unit.amplitude, "unit_of_amplitude": DOF_UNIT.get(unit.dof),
            "p1": p1, "pairing": metrics["pairing"], "p2_parameters": metrics["p2_parameters"],
            "p2_fit_status": metrics["p2_fit_status"], "p2_seed_level_track_response": seed_response,
            "groups": groups, "target_profiles": profiles, "p4": p4,
            "note": ("tx/ty/rz are in-plane: P1 is exact; the refit objective is identical to P0 "
                     "(same non-target measurements and fit geometry), so refit-level track "
                     "response measures seed dependence only")})
        return {name: unit.directory / name for name in (
            "closure_metrics.json", "geometry_only/per_surface.csv",
            "track_response/per_surface.csv", "track_response/per_track.csv",
            "track_response/lost_tracks.csv", "full_response/per_surface.csv")}

    run_stage(ctx, unit.directory, "response",
              {"injected_geometry": injected, "perturbed_trackstates": states,
               "baseline_per_track": ctx.root / "baseline" / "per_track.csv",
               "nominal_states": ctx.nominal_states, "nominal_geometry": ctx.nominal_geometry}, body)


def _count(values):
    output = {}
    for value in values:
        output[value] = output.get(value, 0) + 1
    return output


# ----------------------------------------------------------------------------
# P4 Jacobians
# ----------------------------------------------------------------------------

def solver_geometry_jacobian(tracks, theta0, nominal, dof):
    """d r / d g_dof from the block-LM residual path, r = p - m."""
    solver = _solver()
    usable = [i for i, p in enumerate(theta0) if p is not None]
    hits = [[j for j, key in enumerate(tracks[i].identities) if key[:2] == LAYER] for i in usable]
    width = max((len(h) for h in hits), default=0)
    if not width:
        return {}
    shape = (len(usable), width)
    hit_mask = torch.zeros(shape, dtype=torch.bool)
    sensor = torch.zeros(shape, dtype=torch.long)
    measured = torch.zeros((*shape, 2), dtype=torch.float64)
    covariance = torch.ones((*shape, 2), dtype=torch.float64)
    loc1 = torch.zeros(shape, dtype=torch.bool)
    for row, (i, columns) in enumerate(zip(usable, hits)):
        track = tracks[i]
        for column, j in enumerate(columns):
            hit_mask[row, column] = True
            sensor[row, column] = nominal.id_to_idx[track.identities[j][:3]]
            measured[row, column] = torch.nan_to_num(track.measured[j])
            covariance[row, column] = torch.nan_to_num(track.variance[j], nan=1.)
            loc1[row, column] = bool(track.component_mask[j, 1])
    batches = SimpleNamespace(hit_mask=hit_mask, is_target=hit_mask.clone(), sensor_idx=sensor,
                              measured_local=measured, covariances=covariance, loc1_mask=loc1)
    geometry = SimpleNamespace(centers=nominal.centers, rotations=nominal.rotations)
    parameters = torch.stack([theta0[i] for i in usable])
    column_index = DOF_INDEX[dof]
    output = {}
    for sensor_index in torch.unique(sensor[hit_mask]).tolist():
        def residual(row, sensor_index=sensor_index):
            return solver._sensor_residual_data(parameters, row, sensor_index, batches, geometry)[0]

        zero = torch.zeros(6, dtype=torch.float64)
        data = solver._sensor_residual_data(parameters, zero, sensor_index, batches, geometry)
        jacobian = torch.autograd.functional.jacobian(residual, zero)
        selection = hit_mask & (sensor == sensor_index)
        rows, columns = torch.nonzero(selection, as_tuple=True)
        local = measured[rows, columns] - data[0]
        for n, (row, column) in enumerate(zip(rows.tolist(), columns.tolist())):
            i = usable[row]
            output[(i, hits[row][column])] = (-jacobian[n, :, column_index].numpy(),
                                              local[n].numpy(), bool(data[3][n]))
    return output


def stage_jacobian(ctx, dof):
    ctx.require_layer_config()
    directory = ctx.root / f"{LAYER_TAG}_{dof}" / "jacobian"

    def body(metrics):
        torch.set_num_threads(ctx.threads)
        tracks, theta0, _seed0, _counts = load_cohort(ctx)
        nominal = load_nominal_geometry(ctx.nominal_geometry)
        base_local, base_valid = predict(tracks, theta0)
        rows = [(i, j) for i, t in enumerate(tracks) for j in range(len(t.identities))]
        target = np.array([bool(tracks[i].target[j]) for i, j in rows])
        local0 = np.concatenate([l.numpy() for l in base_local])
        valid0 = np.concatenate([v.numpy() for v in base_valid])

        # Geometry FD through the ordered-plane path with in-memory perturbations.
        fd = {}
        for multiplier in (.5, 1., 2.):
            step = GEOMETRY_FD_STEPS[dof] * multiplier
            sides = []
            for sign in (1., -1.):
                centers, rotations = in_memory_geometry(nominal, dof, sign * step)
                moved = [with_geometry(t, centers, rotations, nominal.id_to_idx) for t in tracks]
                sides.append(np.concatenate([r.numpy() for r in residuals(moved, theta0)]))
            fd[multiplier] = (sides[0] - sides[1]) / (2 * step)
        analytic = np.zeros_like(local0)
        if dof in ("tx", "ty"):
            analytic[:, DOF_INDEX[dof]] = -1.
        else:
            analytic[:, 0], analytic[:, 1] = local0[:, 1], -local0[:, 0]
        analytic[~target] = 0.

        print("  solver autograd geometry Jacobian", flush=True)
        solver_rows = solver_geometry_jacobian(tracks, theta0, nominal, dof)
        solver_g = np.full_like(local0, np.nan)
        branch_mismatch = 0
        for n, (i, j) in enumerate(rows):
            item = solver_rows.get((i, j))
            if item is None or not item[2] or not valid0[n]:
                continue
            if np.abs(item[1] - local0[n]).max() > 1.e-6:
                branch_mismatch += 1
                continue
            solver_g[n] = item[0]

        def relative(a, b, mask):
            mask = mask & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            norm = np.linalg.norm(b[mask])
            return {"rows": int(mask.sum()),
                    "relative": _num(np.linalg.norm(a[mask] - b[mask]) / norm) if norm else None,
                    "max_abs": _num(np.abs(a[mask] - b[mask]).max()) if mask.any() else None}

        comparison = {}
        for multiplier, values in fd.items():
            comparison[f"eps_x{multiplier:g}"] = {
                "step": GEOMETRY_FD_STEPS[dof] * multiplier,
                "solver_vs_fd_target": relative(solver_g, values, target),
                "analytic_vs_fd_target": relative(analytic, values, target),
                "fd_non_target_max_abs": _num(np.nanmax(np.abs(values[~target]))),
            }
        solver_vs_analytic = relative(solver_g, analytic, target)
        limit = TOLERANCES["geometry_jacobian_relative_max"]
        def within(value):
            return value is not None and value <= limit

        geometry_pass = all(
            within(c["solver_vs_fd_target"]["relative"]) and within(c["analytic_vs_fd_target"]["relative"])
            and within(c["fd_non_target_max_abs"]) for c in comparison.values())
        metrics["geometry"] = {"comparisons": comparison, "solver_vs_analytic": solver_vs_analytic,
                               "branch_mismatch_rows": branch_mismatch, "pass": geometry_pass}

        # Implemented track Jacobian: autograd vs FD in the gauge-fixed measurement fit.
        sample = [i for i, p in enumerate(theta0) if p is not None][:GAUGE_JACOBIAN_SAMPLE_TRACKS]
        metrics["track_gauge"] = gauge_track_jacobian([tracks[i] for i in sample],
                                                      torch.stack([theta0[i] for i in sample]))

        # First-order track response G_t * delta_theta vs P2, per amplitude.
        first_order = {}
        units = [u for u in all_units(ctx.root, (dof,)) if not u.control]
        ready = [u for u in units if manifest_status(u.directory) == "completed"]
        if ready:
            print("  physical track Jacobian for first-order prediction", flush=True)
            columns = physical_track_jacobian(tracks, theta0)
            offsets = np.cumsum([0] + [len(t.identities) for t in tracks])
            position = {(t.event_nr, t.track_nr): n for n, t in enumerate(tracks)}
            for unit in ready:
                actual = {}
                for row in read_csv(unit.directory / "track_response" / "per_surface.csv"):
                    actual[tuple(int(row[f]) for f in KEY_FIELDS)] = (
                        _float(row["delta_rt_LOC0_mm"]), _float(row["delta_rt_LOC1_mm"]))
                predicted, observed, is_target = [], [], []
                for row in read_csv(unit.directory / "track_response" / "per_track.csv"):
                    n = position[(int(row["event_nr"]), int(row["track_nr"]))]
                    delta = np.array([_float(row[f"delta_refit_{p}"]) for p in vht.PARAMETERS])
                    if not np.isfinite(delta).all():
                        continue
                    for j, identity in enumerate(tracks[n].identities):
                        g = columns[offsets[n] + j]
                        predicted.append(g @ delta)
                        observed.append(actual[(tracks[n].event_nr, tracks[n].track_nr, *identity)])
                        is_target.append(bool(tracks[n].target[j]))
                predicted, observed = np.array(predicted), np.array(observed)
                is_target = np.array(is_target, dtype=bool)
                first_order[unit.name] = {
                    group: {**relative(predicted, observed, mask),
                            "observed": stats(observed[mask].ravel()),
                            "residual": stats((predicted - observed)[mask].ravel())}
                    for group, mask in (("all", np.ones(len(is_target), dtype=bool)),
                                        ("target", is_target), ("non_target", ~is_target))}
        metrics["track_first_order"] = first_order
        metrics["pass"] = bool(geometry_pass and metrics["track_gauge"]["pass"])
        write_json(directory.parent / "jacobian_metrics.json", {
            "schema_version": SCHEMA_VERSION, "dof": dof, **{k: metrics[k] for k in (
                "geometry", "track_gauge", "track_first_order", "pass")}})
        return {"jacobian_metrics": directory.parent / "jacobian_metrics.json"}

    run_stage(ctx, directory, "jacobian",
              {"baseline_per_track": ctx.root / "baseline" / "per_track.csv",
               "nominal_geometry": ctx.nominal_geometry}, body)


def fit_inputs(tracks):
    selected = [t.fit_mask.nonzero().flatten() for t in tracks]
    width = max(len(s) for s in selected)
    n = len(tracks)
    measured = torch.zeros((n, width, 2), dtype=torch.float64)
    variance = torch.ones_like(measured)
    components = torch.zeros_like(measured, dtype=torch.bool)
    centers = torch.zeros((n, width, 3), dtype=torch.float64)
    rotations = torch.eye(3, dtype=torch.float64).expand(n, width, 3, 3).clone()
    for row, (track, indices) in enumerate(zip(tracks, selected)):
        count = len(indices)
        centers[row] = track.centers[indices[0]]
        rotations[row] = track.rotations[indices[0]]
        centers[row, :count] = track.centers[indices]
        rotations[row, :count] = track.rotations[indices]
        active = track.component_mask[indices]
        measured[row, :count] = torch.where(active, track.measured[indices], 0.)
        variance[row, :count] = torch.where(active, track.variance[indices], 1.)
        components[row, :count] = active
    return hm.MeasurementFitInputs(measured, variance, components, centers, rotations)


def gauge_track_jacobian(tracks, theta0):
    inputs = fit_inputs(tracks)
    frame = hm.reference_gauge_frame(theta0)
    theta = hm.pack_gauge_fixed_track_parameters(theta0)
    autograd = hm.measurement_residual_jacobian(theta, frame, inputs)
    columns = {}
    for k, base_step in enumerate(GAUGE_TRACK_FD_STEPS):
        per_step = {}
        for multiplier in (.5, 1., 2.):
            step = base_step * multiplier
            shift = torch.zeros_like(theta)
            shift[:, k] = step
            plus, minus = theta + shift, theta - shift
            valid = torch.ones(inputs.measured.shape[:2], dtype=torch.bool)
            for value in (plus, minus):
                _, ok = hm.measurement_local_prediction(hm.unpack_gauge_fixed_track_parameters(value, frame), inputs)
                valid &= ok
            fd = (hm.measurement_standardized_residual(plus, frame, inputs)
                  - hm.measurement_standardized_residual(minus, frame, inputs)) / (2 * step)
            mask = valid.repeat_interleave(2, dim=1) & inputs.component_mask.flatten(1)
            a, b = autograd[:, :, k][mask], fd[mask]
            norm = float(b.norm())
            per_step[f"h_x{multiplier:g}"] = {"step": step, "components": int(mask.sum()),
                                             "relative": float((a - b).norm()) / norm if norm else None}
        columns[("offset0", "offset1", "psi0", "tan_lambda", "rho_scaled")[k]] = per_step
    limit = TOLERANCES["track_jacobian_relative_max"]
    passed = all(v["relative"] is not None and v["relative"] <= limit
                 for column in columns.values() for v in column.values())
    return {"tracks": len(tracks), "columns": columns, "pass": passed,
            "function": "helix_model.measurement_residual_jacobian (non-target fit residuals)"}


def physical_track_jacobian(tracks, theta0):
    """Central FD of r w.r.t. the six physical helix parameters, all surfaces."""
    columns = None
    for k, step in enumerate(PHYSICAL_TRACK_FD_STEPS):
        sides = []
        for sign in (1., -1.):
            moved = []
            for p in theta0:
                if p is None:
                    moved.append(None)
                    continue
                q = p.clone()
                q[k] += sign * step
                moved.append(q)
            sides.append(np.concatenate([r.numpy() for r in residuals(tracks, moved)]))
        column = (sides[0] - sides[1]) / (2 * step)
        if columns is None:
            columns = np.zeros((len(column), 2, 6))
        columns[:, :, k] = column
    return columns


# ----------------------------------------------------------------------------
# P5 recovery
# ----------------------------------------------------------------------------

def local_difference(reference, current):
    """Local delta taking ``reference`` to ``current`` in the solver convention."""
    solver = _solver()
    r0 = np.asarray(reference["RotationMatrix"])
    dt = r0.T @ (np.asarray(current["Translation"]) - np.asarray(reference["Translation"]))
    rx, ry, rz = solver.rotation_matrix_to_euler(r0.T @ np.asarray(current["RotationMatrix"]))
    return np.array([*dt, rx, ry, rz])


def stage_recover(ctx, unit):
    ctx.require_layer_config()
    out = unit.directory / "recovery"
    injected = unit.directory / "injected_geometry.json"
    states = acts_output(unit, "trackstates_kf")
    summary = acts_output(unit, "tracksummary_kf")
    control_metrics = None
    if not unit.control:
        control_path = ctx.root / "controls" / "a0" / "recovery" / "recovery_metrics.json"
        if not control_path.exists():
            raise RuntimeError("a=0 control recovery must complete first (it fixes tau_abs)")
        control_metrics = json.loads(control_path.read_text())

    def body(metrics):
        torch.set_num_threads(ctx.threads)
        solver = _solver()
        aligned = out / "aligned_transforms.json"
        argv = ["--config", str(ctx.config_path), "--input-geom", str(injected),
                "--states", str(states), "--summary", str(summary), "--output-geom", str(aligned),
                "--nominal", str(ctx.nominal_geometry), "--nominal-full-geom", str(ctx.nominal_geometry),
                "--target-layers", "17:8", "--output-summary", str(out / "solver_summary.json"),
                "--no-plot", *SOLVER_OPTIONS]
        metrics["solver_argv"] = argv
        args = solver.build_argument_parser().parse_args(argv)
        result = solver.run_alignment_pipeline(solver.alignment_config_from_args(args))
        solver.write_alignment_summary(result, args.output_summary)
        nominal_items = json.loads(ctx.nominal_geometry.read_text())
        index_to_id = [item["ID"] for item in nominal_items]
        blocks = {}
        block_rows = []
        for record in result.get("block_lm_blocks") or []:
            identifier = index_to_id[record["sensor_index"]]
            blocks[identifier] = record
            block_rows.append({"ID": identifier, **{k: record.get(k) for k in (
                "sensor_index", "status", "accepted_steps", "evaluations", "boundary",
                "rank_deficient", "n_components_start", "n_components_end", "clipped_steps")}})
        write_csv(out / "blocks.csv", block_rows)
        if not blocks:
            raise RuntimeError("Solver returned no per-sensor block records")

        nominal_by_id = {item["ID"]: item for item in nominal_items}
        injected_by_id = {item["ID"]: item for item in json.loads(injected.read_text())}
        aligned_items = json.loads(aligned.read_text())
        rows = []
        for item in aligned_items:
            identifier = item["ID"]
            correction = local_difference(injected_by_id[identifier], item)
            remaining = local_difference(nominal_by_id[identifier], item)
            injection = delta_vector(unit.dof, unit.amplitude) if unit.dof else np.zeros(6)
            row = {"ID": identifier, "status": blocks.get(identifier, {}).get("status", "no_record")}
            for name, k in DOF_INDEX.items():
                row[f"correction_{name}"] = float(correction[k])
                row[f"error_{name}"] = float(correction[k] + injection[k])
                row[f"remaining_{name}"] = float(remaining[k])
            rows.append(row)
        write_csv(out / "per_sensor.csv", rows)
        converged = [r for r in rows if r["status"] == "converged"]
        metrics["status_counts"] = dict(_count(r["status"] for r in rows))
        metrics["sensors"] = len(rows)
        metrics["converged_sensors"] = len(converged)
        distributions = {kind: {name: stats([r[f"{kind}_{name}"] for r in converged]) for name in DOF_INDEX}
                         for kind in ("correction", "error", "remaining")}
        metrics["distributions"] = distributions
        if unit.control:
            metrics["tau_abs"] = {name: 3 * (distributions["correction"][name]["rms"] or 0.)
                                  for name in DOF_INDEX}
        else:
            tau_abs = control_metrics["tau_abs"]
            tau_rel = TOLERANCES["recovery_tau_rel"]
            fraction = TOLERANCES["recovery_sensor_fraction_min"]
            a = abs(unit.amplitude)
            errors = np.array([r[f"error_{unit.dof}"] for r in converged])
            limit = tau_abs[unit.dof] + tau_rel * a
            within = float(np.mean(np.abs(errors) <= limit)) if len(errors) else 0.
            mean_bound = tau_rel * a + 3 * float(np.sqrt(np.mean(errors ** 2))) / math.sqrt(max(len(errors), 1))
            crosstalk = {}
            for other in DOF_INDEX:
                if other == unit.dof:
                    continue
                values = np.array([r[f"correction_{other}"] for r in converged])
                share = float(np.mean(np.abs(values) <= tau_abs[other])) if len(values) else 0.
                crosstalk[other] = {"tau_abs": tau_abs[other], "fraction_within": share,
                                    "pass": share >= fraction}
            criteria = {"tau_abs": tau_abs[unit.dof], "tau_rel": tau_rel, "per_sensor_limit": limit,
                        "fraction_within": within, "per_sensor_pass": within >= fraction,
                        "mean_error": _num(errors.mean()) if len(errors) else None,
                        "mean_bound": mean_bound,
                        "mean_pass": bool(len(errors) and abs(errors.mean()) <= mean_bound),
                        "crosstalk": crosstalk}
            criteria["pass"] = bool(criteria["per_sensor_pass"] and criteria["mean_pass"]
                                    and all(c["pass"] for c in crosstalk.values()))
            metrics["criteria"] = criteria

        # Post-correction residual with frozen P0 helices (no ACTS rerun).
        tracks, theta0, _seed0, _counts = load_cohort(ctx)
        working = load_working_geometry(aligned, ctx.nominal_geometry)
        corrected = [with_geometry(t, working.centers, working.rotations, working.id_to_idx) for t in tracks]
        r0 = np.concatenate([r.numpy() for r in residuals(tracks, theta0)])
        rc = np.concatenate([r.numpy() for r in residuals(corrected, theta0)])
        target = np.concatenate([t.target.numpy() for t in tracks])
        metrics["post_correction_residual_shift"] = {
            f"LOC{a}": {"target": stats((rc - r0)[target, a]), "non_target": stats((rc - r0)[~target, a])}
            for a in range(2)}
        for key in ("best_iteration", "stop_reason", "n_tracks", "truth_metrics", "target_metrics"):
            if key in result:
                metrics[f"solver_{key}"] = json.loads(json.dumps(vht_json_ready(result[key])))
        write_json(out / "recovery_metrics.json", {"schema_version": SCHEMA_VERSION, "unit": unit.name,
                                                   "dof": unit.dof, "amplitude": unit.amplitude,
                                                   **{k: v for k, v in metrics.items() if k != "solver_argv"}})
        return {name: out / name for name in ("aligned_transforms.json", "solver_summary.json",
                                              "blocks.csv", "per_sensor.csv", "recovery_metrics.json")}

    out.mkdir(parents=True, exist_ok=True)
    run_stage(ctx, out, "recover", {"injected_geometry": injected, "trackstates": states,
                                     "tracksummary": summary, "nominal_geometry": ctx.nominal_geometry},
              body)


def vht_json_ready(value):
    return _solver()._json_ready(value)


# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------

def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def antisymmetry(units):
    """||d(+a) + d(-a)|| / ||d(+a) - d(-a)|| on rows paired in both amplitudes."""
    if len(units) != 2 or any(manifest_status(u.directory) != "completed" for u in units):
        return None
    tables = []
    for unit in units:
        table = {}
        for row in read_csv(unit.directory / "full_response" / "per_surface.csv"):
            if row["target"] != "True":
                continue
            key = tuple(int(row[f]) for f in KEY_FIELDS)
            table[key] = np.array([[_float(row[f"delta_{kind}_LOC{a}_mm"]) for a in range(2)]
                                   for kind in ("rg", "rt", "rfull")])
        tables.append(table)
    common = sorted(set(tables[0]) & set(tables[1]))
    if not common:
        return None
    minus, plus = (np.stack([t[k] for k in common]) for t in tables)
    output = {"rows": len(common)}
    for k, kind in enumerate(("geometry", "track", "full")):
        s, d = plus[:, k] + minus[:, k], plus[:, k] - minus[:, k]
        finite = np.isfinite(s).all(axis=1) & np.isfinite(d).all(axis=1)
        denominator = np.linalg.norm(d[finite])
        output[kind] = _num(np.linalg.norm(s[finite]) / denominator) if denominator else None
    return output


def plot_dof(ctx, dof, units):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case = ctx.root / f"{LAYER_TAG}_{dof}"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for unit in units:
        metrics = _load(unit.directory / "closure_metrics.json")
        if not metrics:
            continue
        for axis, kind in zip(axes, ("geometry", "track", "full")):
            profile = metrics["target_profiles"][kind]["LOC0_vs_LOC1"]
            x = [(b["low"] + b["high"]) / 2 for b in profile if b["mean"] is not None]
            y = [b["mean"] * 1e3 for b in profile if b["mean"] is not None]
            axis.plot(x, y, marker="o", label=f"a = {unit.amplitude:g} {DOF_UNIT[dof]}")
            axis.set(title=f"17:8 {kind} response", xlabel="LOC1 at crossing [mm]",
                     ylabel="mean ΔLOC0 residual [μm]")
            axis.axhline(0, color="0.6", lw=.8)
            axis.legend()
    fig.tight_layout()
    fig.savefig(case / "response_profiles.png", dpi=120)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4))
    plotted = False
    for unit in units:
        path = unit.directory / "recovery" / "per_sensor.csv"
        if not path.exists():
            continue
        values = [_float(r[f"error_{dof}"]) for r in read_csv(path) if r["status"] == "converged"]
        scale = 1e3
        axis.hist(np.array(values) * scale, bins=60, histtype="step",
                  label=f"a = {unit.amplitude:g} {DOF_UNIT[dof]}")
        plotted = True
    if plotted:
        axis.set(xlabel=f"recovery error ĝ + a [{'μm' if DOF_UNIT[dof] == 'mm' else 'mrad'}]",
                 ylabel="converged sensors", title=f"17:8 {dof} recovery")
        axis.legend()
        fig.tight_layout()
        fig.savefig(case / "recovery_errors.png", dpi=120)
    plt.close(fig)


def stage_report(ctx, dofs):
    lines = ["# Track–Geometry 验证汇总：V17L8 tx / ty / rz", "",
             f"生成时间：{datetime.now(timezone.utc).isoformat()}", "",
             "预注册常量见 `preregistration.json`；未执行或未通过的项目标为 pending / FAIL。", ""]
    base = _load(ctx.root / "baseline" / "summary.json")
    repro = _load(ctx.root / "baseline_repro" / "reproducibility.json")
    lines += ["## P0 baseline", "",
              f"- 状态：{manifest_status(ctx.root / 'baseline') or 'pending'}",
              f"- 轨迹：{base['tracks_evaluated'] if base else '—'}，surface：{base['surfaces_evaluated'] if base else '—'}",
              f"- 复现（500 事例两次）：{json.dumps(repro['run1_vs_run2']) if repro else 'pending'}", ""]
    control = all_units(ctx.root, ())[0]
    control_closure = _load(control.directory / "closure_metrics.json")
    control_recovery = _load(control.directory / "recovery" / "recovery_metrics.json")
    lines += ["## a=0 对照", ""]
    if control_closure:
        refit = control_closure["p2_parameters"]["refit"]
        lines.append(f"- 配对保留率 {control_closure['pairing']['retention']:.4f}；refit Δθ rms："
                     + "，".join(f"{k} {_fmt(v['rms'])}" for k, v in refit.items()))
        lines.append(f"- target Δr_t rms：LOC0 {_fmt(control_closure['groups']['target']['LOC0']['track']['rms'])} mm，"
                     f"LOC1 {_fmt(control_closure['groups']['target']['LOC1']['track']['rms'])} mm")
    else:
        lines.append("- pending")
    if control_recovery:
        lines.append(f"- τ_abs（3×RMS 修正量）：{json.dumps(control_recovery.get('tau_abs'))}")
    lines.append("")

    for dof in dofs:
        units = [u for u in all_units(ctx.root, (dof,)) if not u.control]
        case = ctx.root / f"{LAYER_TAG}_{dof}"
        jac = _load(case / "jacobian_metrics.json")
        rows = ["| 幅度 | P1 | 配对保留 | P2 refit Δθ 状态 | η_closure(target) | 判 P4 | P5 恢复误差均值 | P5 per-sensor | 串扰 | 判 P5 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for unit in units:
            closure = _load(unit.directory / "closure_metrics.json")
            recovery = _load(unit.directory / "recovery" / "recovery_metrics.json")
            criteria = (recovery or {}).get("criteria", {})
            rows.append("| {} {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                unit.amplitude, DOF_UNIT[dof],
                _fmt(closure["p1"]["pass"]) if closure else "pending",
                _fmt(closure["pairing"]["retention"]) if closure else "pending",
                json.dumps(closure["p2_fit_status"]) if closure else "pending",
                _fmt(closure["p4"]["eta_target"]) if closure else "pending",
                _fmt(closure["p4"]["pass"]) if closure else "pending",
                _fmt(criteria.get("mean_error")) if recovery else "pending",
                _fmt(criteria.get("fraction_within")) if recovery else "pending",
                ", ".join(f"{k}:{_fmt(v['fraction_within'])}" for k, v in criteria.get("crosstalk", {}).items())
                if recovery else "pending",
                _fmt(criteria.get("pass")) if recovery else "pending"))
        anti = antisymmetry(units)
        scan = {"dof": dof, "amplitudes": AMPLITUDES[dof], "antisymmetry_target": anti, "units": {}}
        for unit in units:
            closure = _load(unit.directory / "closure_metrics.json")
            if closure:
                axis = "LOC0" if dof in ("tx", "rz") else "LOC1"
                group = closure["groups"]["target"][axis]
                scan["units"][unit.name] = {
                    "mean_delta_rg": group["geometry"]["mean"], "mean_delta_rt": group["track"]["mean"],
                    "mean_delta_rfull": group["full"]["mean"],
                    "slope_geometry_per_unit": (group["geometry"]["mean"] / unit.amplitude
                                                if group["geometry"]["mean"] is not None else None),
                    "rz_slope_LOC0_per_LOC1": closure["target_profiles"]["geometry"]["slope_LOC0_per_LOC1"]}
        write_json(case / "amplitude_scan.json", scan)
        try:
            plot_dof(ctx, dof, units)
        except Exception as error:  # Plots are auxiliary; the report must still be written.
            print(f"plot failed for {dof}: {error}", file=sys.stderr)
        jac_line = "pending" if not jac else (
            f"G_g {_fmt(jac['geometry']['pass'])}（solver vs FD 相对误差 "
            + "，".join(f"{k}: {_fmt(v['solver_vs_fd_target']['relative'])}"
                       for k, v in jac["geometry"]["comparisons"].items())
            + f"；branch mismatch {jac['geometry']['branch_mismatch_rows']}），"
            f"G_t {_fmt(jac['track_gauge']['pass'])}")
        dof_lines = [f"## {dof}", "", *rows, "", f"- Jacobian：{jac_line}",
                     f"- 正负反对称度（target）：{json.dumps(anti) if anti else 'pending'}",
                     "- 幅度只有一档，线性区间结论仅基于该幅度。", ""]
        (case / "summary.md").write_text("\n".join([f"# V17L8 {dof} 验证", "", *dof_lines[2:]]) + "\n")
        lines += dof_lines
    report = ctx.root / "reports" / "validation_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print(f"report: {report}", flush=True)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def gate_for_recovery(ctx, unit):
    """P5 only after P1 sanity, closure and the geometry Jacobian pass."""
    if unit.control:
        return True, "control"
    closure = _load(unit.directory / "closure_metrics.json")
    jac = _load(ctx.root / f"{LAYER_TAG}_{unit.dof}" / "jacobian_metrics.json")
    if not closure or not jac:
        return False, "missing P4 metrics"
    if not closure["p1"]["pass"]:
        return False, "P1 sanity failed"
    if not closure["p4"]["pass"]:
        return False, "closure failed"
    if not jac["geometry"]["pass"]:
        return False, "geometry Jacobian failed"
    return True, "P1/P4 passed"


def run_all(ctx, dofs):
    stage_prepare(ctx)
    stage_baseline(ctx)
    units = all_units(ctx.root, dofs)
    failures = {}

    def attempt(name, function, *arguments):
        try:
            function(*arguments)
            return True
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[fail] {name}: {failures[name]}", file=sys.stderr, flush=True)
            return False

    for unit in units:
        if attempt(f"acts:{unit.name}", stage_acts, ctx, unit):
            attempt(f"response:{unit.name}", stage_response, ctx, unit)
    for dof in dofs:
        attempt(f"jacobian:{dof}", stage_jacobian, ctx, dof)
    for unit in units:
        allowed, reason = gate_for_recovery(ctx, unit)
        if not allowed:
            failures[f"recover:{unit.name}"] = f"blocked: {reason}"
            continue
        attempt(f"recover:{unit.name}", stage_recover, ctx, unit)
    stage_report(ctx, dofs)
    write_json(ctx.root / "reports" / "run_all_status.json",
               {"finished_at": datetime.now(timezone.utc).isoformat(), "failures": failures})
    return 1 if failures else 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Ignored; the frozen root/config_v17l8.toml is used")
    parser.add_argument("--root", type=absolute_project_path, default=DEFAULT_ROOT)
    parser.add_argument("--nominal-states", default=NOMINAL_STATES)
    parser.add_argument("--nominal-geometry", default=NOMINAL_GEOMETRY)
    parser.add_argument("--simulation-dir", default=SIMULATION_DIR)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=MAX_EVENTS,
                        help="Smoke tests only; the frozen run uses %(default)s")
    parser.add_argument("--dofs", default="tx,ty,rz")
    parser.add_argument("stage", choices=("prepare", "baseline", "acts", "response", "jacobian",
                                          "recover", "report", "run-all"))
    parser.add_argument("--unit", help="Unit name, e.g. a0, tx_p0p5mm")
    parser.add_argument("--dof", choices=tuple(DOF_INDEX))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    ctx = Context(args)
    dofs = tuple(d for d in args.dofs.split(",") if d)
    if args.stage == "prepare":
        stage_prepare(ctx)
    elif args.stage == "baseline":
        stage_baseline(ctx)
    elif args.stage in ("acts", "response", "recover"):
        units = {u.name: u for u in all_units(ctx.root)}
        if args.unit not in units:
            raise SystemExit(f"--unit must be one of {sorted(units)}")
        {"acts": stage_acts, "response": stage_response, "recover": stage_recover}[args.stage](
            ctx, units[args.unit])
    elif args.stage == "jacobian":
        stage_jacobian(ctx, args.dof)
    elif args.stage == "report":
        stage_report(ctx, dofs)
    else:
        return run_all(ctx, dofs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
