#!/usr/bin/env python
"""Tabulate ACTS-Alignment vs Schur-LM on the pinned pT=100 GeV study.

Reuses compare_geometries.py, plot_chi2ndf.py and plot_residuals.py exactly as
the 2026-10-09 ACTS baseline did, then writes the figure-2 tables plus a
calibration-time table to report.md / tables.json. Never runs a solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

METHODS = ("schur", "acts")
METHOD_LABELS = {"nominal": "nominal", "misaligned": "misaligned",
                 "schur": "Schur-LM", "acts": "ACTS-Alignment"}
DOF_UNITS = (("tx", "μm"), ("ty", "μm"), ("rz", "mrad"))
AXES = (("LOC0", "X", "l_x_hit", "t_eLOC0"), ("LOC1", "Y", "l_y_hit", "t_eLOC1"))
_LOG_LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\s+(\S+)\s+\S+\s+(.*)$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_rows(summaries, target_layers):
    """Residual RMS per layer/DOF in μm / mrad; ``summaries`` maps method -> summary JSON."""
    rows = []
    for volume, layer in target_layers:
        key = f"vol={volume}|lay={layer}"
        for dof, unit in DOF_UNITS:
            row = {"surface": f"{volume}:{layer}", "dof": dof, "unit": unit}
            injected = set()
            for method, summary in summaries.items():
                stats = summary["by_layer"][key]["dof"][dof]
                row[method] = 1e3 * stats["aligned_difference"]["rms"]
                injected.add(round(1e3 * stats["misaligned_difference"]["rms"], 9))
            if len(injected) != 1:
                raise ValueError(f"methods disagree on injected misalignment for {key} {dof}")
            row["injected"] = injected.pop()
            rows.append(row)
    return rows


def distribution_summary(values):
    """Count, mean, std (population) and RMS of a finite 1-D sample."""
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("distribution needs a non-empty finite sample")
    n = len(values)
    mean = sum(values) / n
    rms = math.sqrt(sum(v * v for v in values) / n)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"count": n, "mean": mean, "std": std, "rms": rms}


def geometry_max_difference(records_a, records_b):
    """Max |Δ| of translation (mm) and rotation-matrix entries over shared IDs."""
    b = {row["ID"]: row for row in records_b}
    shared = [row for row in records_a if row["ID"] in b]
    if not shared or len(shared) != len(records_a) or len(shared) != len(b):
        raise ValueError("geometries cover different surfaces")
    translation = max(abs(x - y) for row in shared
                      for x, y in zip(row["Translation"], b[row["ID"]]["Translation"]))
    rotation = max(abs(x - y) for row in shared
                   for ra, rb in zip(row["RotationMatrix"], b[row["ID"]]["RotationMatrix"])
                   for x, y in zip(ra, rb))
    return {"surfaces": len(shared), "max_translation_mm": translation,
            "max_rotation_entry": rotation}


def thread_consistency(difference, geometry):
    """Multi-thread arm agrees if its deviation is <1% of Schur's residual RMS (μm / mrad)."""
    translation_rms = min(r["schur"] for r in geometry if r["dof"] in ("tx", "ty"))
    rotation_rms = min(r["schur"] for r in geometry if r["dof"] == "rz")
    return {**difference,
            "translation_limit_um": 0.01 * translation_rms,
            "rotation_limit_mrad": 0.01 * rotation_rms,
            "consistent": (1e3 * difference["max_translation_mm"] < 0.01 * translation_rms
                           and 1e3 * difference["max_rotation_entry"] < 0.01 * rotation_rms)}


def parse_acts_alignment_log(text):
    """Split the ACTS run into event loop and finalize (iterations) from log stamps.

    Stamps are HH:MM:SS (1 s resolution); midnight rollovers are unwrapped.
    """
    events, offset, previous = [], 0, None
    for line in text.splitlines():
        match = _LOG_LINE.match(line)
        if not match:
            continue
        seconds = int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) + offset
        if previous is not None and seconds < previous:
            offset += 86400
            seconds += 86400
        previous = seconds
        events.append((seconds, match[4], match[5]))

    def first(predicate):
        return next((t for t, source, message in events if predicate(source, message)), None)

    loop = first(lambda s, m: s == "Sequencer" and m.startswith("Starting event loop"))
    finalize = first(lambda s, m: m.startswith("Finalizing alignment with"))
    iterations = [(t, float(re.search(r"average chi2/ndf = (\S+)", m)[1]))
                  for t, s, m in events if m.startswith("iIter = ")]
    converged = [m for _, _, m in events if "Alignment has converged" in m]
    tracks = re.search(r"with (\d+) collected tracks", next(
        (m for _, _, m in events if m.startswith("Finalizing alignment with")), ""))
    end = events[-1][0] if events else None
    return {
        "event_loop_seconds": None if loop is None or finalize is None else finalize - loop,
        "finalize_seconds": None if finalize is None or end is None else end - finalize,
        "iterations": len(iterations),
        "iteration_end_seconds": [t - finalize for t, _ in iterations] if finalize else [],
        "average_chi2_ndf": [value for _, value in iterations],
        "converged": bool(converged),
        "converged_message": converged[0] if converged else None,
        "collected_tracks": int(tracks[1]) if tracks else None,
        "resolution_seconds": 1,
    }


_PROCESSED = re.compile(r"Processed \d+ events in (\S+) s \(wall clock\)")


def full_workflow_section(text, step):
    """Lines of the single ``STEP <n>:`` section of an ``alignment_odd.py full`` log."""
    lines, inside, seen = [], False, 0
    for line in text.splitlines():
        if line.startswith("STEP "):
            inside = line.startswith(f"STEP {step}:")
            seen += inside
            continue
        if inside:
            lines.append(line)
    if seen != 1:
        raise ValueError(f"full-workflow log has {seen} STEP {step} sections, expected 1")
    return "\n".join(lines)


def full_workflow_records(text):
    """ACTS-side timings of a full run from the Sequencer wall clock of STEP 2 and STEP 4.

    The full workflow writes one log and no per-stage timings.json, so process
    CPU time is unavailable and setup outside the Sequencer is excluded.
    """
    timings = {}
    for name, step in (("alignment", 2), ("misaligned", 4)):
        found = _PROCESSED.findall(full_workflow_section(text, step))
        if len(found) != 1:
            raise ValueError(f"STEP {step} must report one Sequencer wall clock, got {found}")
        timings[name] = {"wall_seconds": float(found[0]), "process_cpu_seconds": None,
                         "source": f"full-workflow log STEP {step} Sequencer wall clock"}
    return timings, parse_acts_alignment_log(full_workflow_section(text, 2))


def is_full_workflow(manifest):
    return "workflow_config" in manifest.get("metrics", {})


def full_workflow_output(manifest, key):
    recorded = manifest.get("outputs", {}).get(key)
    if recorded is None:
        raise ValueError(f"full-workflow manifest records no {key}")
    return Path(recorded)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value, digits):
    return "—" if value is None else f"{value:.{digits}f}"


def run_logged(command, log):
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=SCRIPTS.parent, stdout=stream,
                                stderr=subprocess.STDOUT, env=os.environ.copy())
    if result.returncode:
        raise RuntimeError(f"{command[1]} failed ({result.returncode}); see {log}")


def load_manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"stage not completed: {directory}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import plot_chi2ndf
    import plot_residuals
    import pt100_alignment_layout as pt100
    from alignment_artifact_layout import TARGET_LAYER_DIR, TARGET_LAYERS, artifact_filename
    from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from project_config import load_project_config

    config = load_project_config()
    run_id = args.run_id or pt100.PT100_ALIGNMENT_RUN
    paths = pt100.pt100_alignment_paths(config.data_root, TARGET_LAYER_DIR, run_id)
    out = paths["comparison"]
    if out.exists():
        raise FileExistsError(f"refusing to reuse comparison output: {out}")

    alignment_manifest = load_manifest(paths["alignment"])
    stage_keys = (("aligned_schur", "aligned-schur"), ("schur_single_thread", "schur_single_thread"))
    full_log = None
    if is_full_workflow(alignment_manifest):
        # `alignment_odd.py full`: its one manifest and log cover the ACTS alignment
        # and the misaligned / ACTS-aligned reconstructions it wrote itself.
        if full_workflow_output(alignment_manifest, "reconstruction_misaligned").resolve() \
                != paths["misaligned"].resolve():
            raise ValueError(f"full-workflow manifest does not cover {paths['misaligned']}")
        full_log = paths["alignment"].parent.parent / f"{run_id}.log"
    else:
        stage_keys = (("misaligned", "misaligned"), ("aligned_acts", "aligned-acts"), *stage_keys)
    manifests = {"acts_alignment": alignment_manifest,
                 **{name: load_manifest(paths[key]) for name, key in stage_keys}}
    multi_thread = paths["schur_multi_thread"]
    if (multi_thread / "manifest.json").exists():
        manifests["schur_multi_thread"] = load_manifest(multi_thread)

    misalignment_sha = sha256(paths["misalignment"])
    for name, manifest in manifests.items():
        recorded = manifest.get("inputs", {}).get("misalignment")
        if recorded is not None and recorded["sha256"] != misalignment_sha:
            raise ValueError(f"{name} used a different misalignment draw")

    suffix = artifact_filename("trackstates_kf", ".root").removeprefix("trackstates_kf_")
    suffix = suffix.removesuffix(".root")
    trackstates = {
        "nominal": paths["nominal"] / f"trackstates_kf_{suffix}.root",
        "misaligned": paths["misaligned"] / f"trackstates_kf_{suffix}.root",
        "acts": paths["aligned-acts"] / f"trackstates_kf_{suffix}_acts.root",
        "schur": paths["aligned-schur"] / f"trackstates_kf_{suffix}_schur_lm.root",
    }
    if full_log is not None:
        trackstates["acts"] = (full_workflow_output(alignment_manifest, "reconstruction_aligned")
                               / f"trackstates_kf_{suffix}.root")
    tracksummary = {k: Path(str(v).replace("trackstates_kf_", "tracksummary_kf_"))
                    for k, v in trackstates.items()}
    geometries = {"acts": paths["acts_geometry"], "schur": paths["schur_geometry"]}
    nominal_geometry = config.data_root / "geometry" / "odd_transforms.json"
    inputs = {"misalignment": paths["misalignment"], "nominal_geometry": nominal_geometry,
              **{f"geometry_{k}": v for k, v in geometries.items()},
              **{f"trackstates_{k}": v for k, v in trackstates.items()},
              **{f"tracksummary_{k}": v for k, v in tracksummary.items()}}
    timed_stages = ("aligned-schur", "schur_single_thread")
    if full_log is None:
        timed_stages = ("alignment", "misaligned", "aligned-acts", *timed_stages)
    else:
        inputs["full_workflow_log"] = full_log
    for name in timed_stages:
        inputs[f"timings_{name}"] = paths[name] / "timings.json"
    out.mkdir(parents=True)
    commands = []
    for method in METHODS:
        commands += [
            [sys.executable, str(SCRIPTS / "compare_geometries.py"), "--nominal",
             str(nominal_geometry), "--misaligned", str(paths["misalignment"]),
             "--aligned", str(geometries[method]), "--output-dir", str(out / f"geometry_{method}")],
            [sys.executable, str(SCRIPTS / "plot_chi2ndf.py"), "--recon-dir",
             str(paths["reconstruction"]), "--aligned-tracksummary", str(tracksummary[method]),
             "--output", str(out / "chi2" / f"{method}_chi2ndf_{suffix}.png")],
            [sys.executable, str(SCRIPTS / "plot_residuals.py"), "--recon-dir",
             str(paths["reconstruction"]), "--aligned-trackstates", str(trackstates[method]),
             "--label", method, "--output-dir", str(out / "residuals")],
        ]
    (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    manifest = write_experiment_manifest(
        out, run_id=f"{run_id}/comparison", command=[sys.executable, *sys.argv],
        config=config, inputs=inputs, metrics={"commands": commands})
    try:
        for index, command in enumerate(commands):
            run_logged(command, out / f"step{index:02d}_{Path(command[1]).stem}.log")

        geometry = geometry_rows(
            {m: json.loads((out / f"geometry_{m}" / "geometry_comparison_summary.json").read_text())
             for m in METHODS}, TARGET_LAYERS)
        chi2 = {mode: distribution_summary(plot_chi2ndf.load_chi2_ndf(path, label=mode))
                for mode, path in tracksummary.items()}

        residuals = {}
        for volume, layer in TARGET_LAYERS:
            for loc, axis, meas, pred in AXES:
                key = f"{volume}:{layer} {loc}"
                residuals[key] = {}
                for mode, path in trackstates.items():
                    data = plot_residuals.get_branch_data(path, "trackstates", meas, pred,
                                                          vol_ids=volume, lay_id=layer)
                    row = distribution_summary(data)
                    residuals[key][mode] = row
                # Figure-2 convention: TH1 std inside the plot range (no under/overflow).
                import ROOT
                for method in METHODS:
                    histogram_file = out / "residuals" / f"{method}_res_{axis}_Vol{volume}_Lay{layer}_comparison.root"
                    handle = ROOT.TFile.Open(str(histogram_file), "READ")
                    for mode, name in (("nominal", "hist_nominal"), ("misaligned", "hist_misaligned"),
                                       (method, "hist_aligned")):
                        h = handle.Get(name)
                        residuals[key][mode][f"hist_std_{method}"] = float(h.GetStdDev())
                        residuals[key][mode][f"hist_out_of_range_{method}"] = int(
                            h.GetBinContent(0) + h.GetBinContent(h.GetNbinsX() + 1))
                    handle.Close()

        counts_tracks = {mode: row["count"] for mode, row in chi2.items()}
        counts_hits = {key: {mode: row["count"] for mode, row in value.items()}
                       for key, value in residuals.items()}
        controlled = (counts_tracks["acts"] == counts_tracks["schur"]
                      and all(v["acts"] == v["schur"] for v in counts_hits.values()))

        timings = {"schur_single_thread": json.loads(
            (paths["schur_single_thread"] / "timings.json").read_text())}
        if full_log is None:
            timings.update({name: json.loads((paths[name] / "timings.json").read_text())
                            for name in ("alignment", "misaligned")})
            acts_log = parse_acts_alignment_log(
                (paths["alignment"] / "acts-alignment.log").read_text())
            acts_label = "ACTS-Alignment (Sequencer 1 thread)"
        else:
            full_timings, acts_log = full_workflow_records(full_log.read_text())
            timings.update(full_timings)
            acts_label = "ACTS-Alignment (Sequencer all threads; finalize serial)"
        if "schur_multi_thread" in manifests:
            timings["schur_multi_thread"] = json.loads((multi_thread / "timings.json").read_text())
        threads_check = None
        if "schur_multi_thread" in manifests:
            multi_geometry = next(multi_thread.glob("aligned_transforms_*_multi_thread.json"))
            threads_check = thread_consistency(geometry_max_difference(
                json.loads(paths["schur_geometry"].read_text()),
                json.loads(multi_geometry.read_text())), geometry)

        tables = {"geometry": geometry, "chi2_ndf": chi2, "residuals": residuals,
                  "timings": timings, "acts_log": acts_log, "acts_label": acts_label,
                  "thread_consistency": threads_check,
                  "track_counts": counts_tracks,
                  "hit_counts": counts_hits, "controlled_same_tracks": controlled,
                  "misalignment_sha256": misalignment_sha}
        (out / "tables.json").write_text(json.dumps(tables, indent=2, allow_nan=False) + "\n")
        (out / "report.md").write_text(render_report(run_id, tables, TARGET_LAYERS))
        outputs = {str(p.relative_to(out)): p for p in out.rglob("*")
                   if p.is_file() and p.name != "manifest.json"}
        finalize_experiment_manifest(manifest, status="completed", outputs=outputs,
                                     metrics={"controlled_same_tracks": controlled,
                                              "thread_consistency": threads_check,
                                              "track_counts": counts_tracks})
    except BaseException as error:
        finalize_experiment_manifest(manifest, status="failed", metrics={"error": str(error)})
        raise
    print((out / "report.md").read_text())


def render_report(run_id, tables, target_layers):
    columns = ("nominal", "misaligned", "schur", "acts")
    lines = [f"# pT=100 GeV material-on: ACTS-Alignment vs Schur-LM ({run_id})", "",
             f"Targets: {', '.join(f'{v}:{l}' for v, l in target_layers)}; "
             f"misalignment SHA-256 `{tables['misalignment_sha256'][:12]}…` shared by both methods.", "",
             "## Geometry residual RMS (aligned − nominal, nominal local frame)", ""]
    lines.append(markdown_table(
        ["surface", "injected", "Schur-LM", "ACTS-Alignment"],
        [[f"{r['surface']} {r['dof']} ({r['unit']})", fmt(r["injected"], 1),
          fmt(r["schur"], 3), fmt(r["acts"], 3)] for r in tables["geometry"]]))
    lines += ["", "## Reconstruction χ²/ndf per track (tracksummary chi2Sum/NDF)", ""]
    chi2 = tables["chi2_ndf"]
    lines.append(markdown_table(
        ["χ²/ndf", *(METHOD_LABELS[c] for c in columns)],
        [[stat, *(fmt(chi2[c][stat], 3) for c in columns)] for stat in ("mean", "std")]
        + [["tracks", *(chi2[c]["count"] for c in columns)]]))
    for title, field in (("figure-2 convention: TH1 std inside plot range", "hist_std"),
                         ("all hits: √mean(r²)", "rms")):
        lines += ["", f"## Measurement − truth on target layers, μm ({title})", ""]
        rows = []
        for key, value in tables["residuals"].items():
            cells = []
            for c in columns:
                if field == "rms":
                    cells.append(fmt(1e3 * value[c]["rms"], 2))
                else:
                    method = c if c in METHODS else "acts"
                    cells.append(fmt(1e3 * value[c][f"hist_std_{method}"], 2))
            rows.append([f"{key}", *cells])
        lines.append(markdown_table(["surface", *(METHOD_LABELS[c] for c in columns)], rows))
    t = tables["timings"]
    log = tables["acts_log"]
    schur = t["schur_single_thread"]
    lines += ["", "## Calibration time (primary: Schur single-threaded)", ""]
    rows = [[tables["acts_label"], fmt(t["alignment"]["wall_seconds"], 1),
             fmt(t["alignment"].get("process_cpu_seconds"), 1),
             f"event loop {fmt(log['event_loop_seconds'], 0)} s; finalize {fmt(log['finalize_seconds'], 0)} s; "
             f"{log['iterations']} iterations; converged={log['converged']}"]]
    stages = schur["stages"]
    detail = "; ".join(f"{k} {v['wall_seconds']:.1f} s" for k, v in stages.items())
    rows.append([f"Schur-LM (torch threads {schur['threads']['torch_num_threads']})",
                 fmt(schur["pipeline_total"]["wall_seconds"], 1),
                 fmt(schur["pipeline_total"]["cpu_seconds"], 1), detail])
    rows.append(["misaligned KF reconstruction (Schur input only)",
                 fmt(t["misaligned"]["wall_seconds"], 1),
                 fmt(t["misaligned"].get("process_cpu_seconds"), 1), "not needed by ACTS-Alignment"])
    rows.append(["Schur-LM incl. misaligned reconstruction",
                 fmt(schur["pipeline_total"]["wall_seconds"] + t["misaligned"]["wall_seconds"], 1),
                 "—", "end-to-end from the same simulation"])
    if "schur_multi_thread" in t:
        multi = t["schur_multi_thread"]
        rows.append([f"(secondary) Schur-LM, torch threads {multi['threads']['torch_num_threads']}",
                     fmt(multi["pipeline_total"]["wall_seconds"], 1),
                     fmt(multi["pipeline_total"]["cpu_seconds"], 1),
                     "available parallelism only; not an algorithmic comparison; geometry vs 1-thread "
                     f"consistent={tables['thread_consistency']['consistent']}"])
    lines.append(markdown_table(["method", "wall s", "CPU s", "breakdown"], rows))
    lines += ["", f"Same tracks for both methods: {tables['controlled_same_tracks']} "
              f"(tracks {tables['track_counts']}).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
