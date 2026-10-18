"""Compare three frozen nominal helix material arms; never fit or import ACTS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.append(str(_ROOT / "scripts"))
from scripts.experiment_manifest import (
    write_experiment_manifest, finalize_experiment_manifest, _sha256, _directory_record,
)
from scripts.project_config import load_project_config, absolute_project_path
from scripts.validate_helix_truth import write_json

GROUPS = ("g1_pt10_material_on", "g2_pt10_material_off", "g3_pt100_material_on")
METRICS = {"M": "measurement2", "truth_M": "truth2", "V": "V", "P": "P",
           "U": "U", "truth_excess": "truth_excess"}
PARTICLE_COLUMNS = ("particle_hash", "vx", "vy", "vz", "vt", "px", "py", "pz",
                    "m", "q", "eta", "phi")


def verify_arm_configs(records):
    common = records[0]
    for record, (pt, interactions) in zip(records, ((10., True), (10., False), (100., True))):
        if (record["pt_gev"] != pt or record["fatras_interactions"] != interactions
                or record["geometry"] != "nominal" or record["random_seed"] != 42
                or record["num_events"] != common["num_events"]):
            raise ValueError("unexpected arm configuration")


def verify_full_cohort(effective, counts, tracks, excluded, generated_events, original_keys=None):
    if (effective["max_events"] != 0 or effective["max_tracks"] != 0
            or counts["selected_tracks"] != counts["available_tracks"]
            or counts["selected_events"] != counts["available_events"]):
        raise ValueError("comparison requires the full cohort, without event/track caps")
    combined = pd.concat([tracks[["event_nr", "track_nr"]], excluded[["event_nr", "track_nr"]]])
    if len(combined) != counts["available_tracks"] or combined.duplicated().any():
        raise ValueError("evaluated/excluded records do not partition the full cohort")
    if not set(combined.event_nr).issubset(set(generated_events)):
        raise ValueError("reconstructed event absent from generated events")
    if original_keys is not None and set(map(tuple, combined.to_numpy())) != set(map(tuple, original_keys.to_numpy())):
        raise ValueError("evaluated/excluded keys differ from the original full cohort")


def read_particles(path):
    # Local memory-mapped reads avoid the asynchronous fsspec basket queue,
    # matching the existing trackstates loader's ROOT I/O contract.
    with uproot.open(path, handler=uproot.MemmapSource) as root:
        tree = root["particles"]
        arrays = tree.arrays(["event_id", *PARTICLE_COLUMNS], library="ak")
    counts = ak.to_numpy(ak.num(arrays["particle_hash"]))
    if not np.all(counts == 1):
        raise ValueError("study requires exactly one generated particle per event")
    return pd.DataFrame({"event_id": ak.to_numpy(arrays["event_id"]), **{
        name: ak.to_numpy(ak.flatten(arrays[name])) for name in PARTICLE_COLUMNS}})


def verify_initial_conditions(frames):
    first = frames[0].sort_values(["event_id", "particle_hash"]).reset_index(drop=True)
    keys = ["event_id", "particle_hash"]
    if first.duplicated(keys).any():
        raise ValueError("duplicate generated initial particle identity")
    for i, candidate in enumerate(frames):
        candidate = candidate.sort_values(keys).reset_index(drop=True)
        if not first[keys].equals(candidate[keys]):
            raise ValueError("initial particle identities differ")
        for column in PARTICLE_COLUMNS[1:]:
            scale = 10. if i == 2 and column in ("px", "py", "pz") else 1.
            if not np.allclose(candidate[column], first[column] * scale, rtol=2.e-6, atol=1.e-7):
                raise ValueError(f"initial conditions differ: arm {i + 1}, {column}")
    return {"passed": True, "n_generated_events": int(len(first)),
            "scope": "same initial identities, vertices, charge and direction; third momentum scaled by 10"}


def summarize_cell(frame):
    complete = (len(frame) > 0 and bool(frame["complete"].all())
                and np.isfinite(frame[list(METRICS.values())].to_numpy()).all())
    result = {"n_components": int(len(frame)), "n_events": int(frame.event_nr.nunique()),
              "complete": bool(complete)}
    for metric, column in METRICS.items():
        value = float(frame[column].mean()) if complete else None
        result[metric] = value if value is None or np.isfinite(value) else None
    result["complete"] &= all(result[m] is not None for m in METRICS)
    if not result["complete"]:
        result.update({m: None for m in METRICS})
    if "measurement_residual" in frame and result["complete"]:
        for name in ("measurement", "truth"):
            r = frame[f"{name}_residual"].to_numpy()
            result.update({f"{name}_mean": float(r.mean()), f"{name}_std": float(r.std()),
                           f"{name}_rmse": float(np.sqrt(np.mean(r*r)))})
        result["pull2"] = float((frame.measurement2 / (frame.V + frame.P)).mean())
    return result


def bootstrap_metrics(frames, event_ids, *, replicates=2000, seed=6006):
    """Resample the same generated event pool for every arm; retain empty events.

    Each statistic is a component-weighted mean. Missing reconstructed events
    have zero numerator and denominator, never an invented zero residual.
    """
    events = pd.Index(event_ids)
    if len(events) < 2 or not events.is_unique:
        raise ValueError("bootstrap needs at least two unique generated events")
    result, totals = {}, {}
    columns = list(METRICS.values())
    for group, frame in frames.items():
        if not set(frame.event_nr).issubset(set(event_ids)):
            raise ValueError("reconstruction contains an event absent from generated particles")
        cell = summarize_cell(frame)
        sums = frame.groupby("event_nr")[columns].sum().reindex(events, fill_value=0).to_numpy()
        counts = frame.groupby("event_nr").size().reindex(events, fill_value=0).to_numpy()
        totals[group] = (sums, counts, cell["complete"])
        result[group] = {m: (cell[m], np.full(replicates, np.nan)) for m in METRICS}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, 32):
        size = min(32, replicates - start)
        indices = rng.integers(0, len(events), size=(size, len(events)))
        for group, (sums, counts, complete) in totals.items():
            if not complete:
                continue
            denominator = counts[indices].sum(axis=1)
            numerator = sums[indices].sum(axis=1)
            means = np.divide(numerator, denominator[:, None],
                              out=np.full_like(numerator, np.nan), where=denominator[:, None] > 0)
            for j, metric in enumerate(METRICS):
                result[group][metric][1][start:start + size] = means[:, j]
    return result


def interval(values):
    # Do not narrow the event pool when a rare stratum vanishes in resampling.
    return np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None


def interval_verdict(estimate, ci, *, complete):
    if not complete or estimate is None or ci is None:
        return "not_evaluable"
    return "decreased" if ci[1] < 0 else "increased" if ci[0] > 0 else "insufficient_evidence"


def verify_inputs(manifest):
    for name, record in manifest["inputs"].items():
        path = Path(record["path"])
        expected = record.get("sha256")
        if expected is None:
            raise ValueError(f"unhashed input: {name}")
        actual = _directory_record(path)["sha256"] if path.is_dir() else _sha256(path)
        if actual != expected:
            raise ValueError(f"input hash mismatch: {path}")


def component_table(surfaces, tracks):
    keys = ["event_nr", "track_nr"]
    if tracks.duplicated(keys).any() or surfaces.duplicated(keys + ["volume_id", "layer_id", "module_id", "occurrence"]).any():
        raise ValueError("duplicate saved track/surface identity")
    if set(map(tuple, surfaces[keys].to_numpy())) != set(map(tuple, tracks[keys].to_numpy())):
        raise ValueError("surface cohort contains missing/unknown track identities")
    sizes = surfaces.groupby(keys).size().reindex(pd.MultiIndex.from_frame(tracks[keys]), fill_value=0)
    if not np.array_equal(sizes.to_numpy(), tracks.n_surfaces.to_numpy()):
        raise ValueError("saved surface cohort is incomplete")
    merged = surfaces.merge(tracks[keys + ["refit_stationary"]], on=keys, validate="many_to_one", how="left")
    parts = []
    for component in ("LOC0", "LOC1"):
        data = merged.loc[merged[f"weighted_{component}"]].copy()
        data["component"] = component
        data["layer"] = data.volume_id.astype(str) + ":" + data.layer_id.astype(str)
        data["measurement_residual"] = data[f"prediction_{component}_mm"] - data[f"measurement_{component}_mm"]
        data["truth_residual"] = data[f"prediction_{component}_mm"] - data[f"truth_{component}_mm"]
        with np.errstate(over="ignore", invalid="ignore"):
            data["measurement2"] = data.measurement_residual**2
            data["truth2"] = data.truth_residual**2
        # Compare to saved scores, rather than trusting summary totals.
        for prefix, values in (("truth", data.truth2), ("measurement", data.measurement2)):
            expected = values / data[f"variance_{component}_mm2"]
            if not np.allclose(data[f"{prefix}_chi2_{component}"], expected, rtol=1.e-9, atol=1.e-9, equal_nan=True):
                raise ValueError(f"saved {prefix} chi2 mismatch")
        data["V"] = data[f"V_used_{component}_mm2"]
        data["P"] = data[f"P_{component}_mm2"]
        data["U"] = data.measurement2 - data.V - data.P
        data["truth_excess"] = data.truth2 - data.P
        data["complete"] = data.refit_stationary & data.intersection_valid & data[f"budget_valid_{component}"]
        parts.append(data)
    return pd.concat(parts, ignore_index=True)


def load_arm(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"helix run is not completed: {run}")
    verify_inputs(manifest)
    effective = manifest["metrics"]["effective_experiment_config"]
    if effective["geometry"] != "nominal" or effective["evaluation_geometry"] != "nominal" or not effective.get("prediction_budget"):
        raise ValueError("comparison requires nominal evaluation and prediction budget")
    reconstruction = Path(effective["nominal_states"]).parent
    reco_manifest = json.loads((reconstruction / "manifest.json").read_text())
    verify_inputs(reco_manifest)
    simulation = Path(reco_manifest["inputs"]["simulation_dir"]["path"])
    sim_manifest = json.loads((simulation / "manifest.json").read_text())
    verify_inputs(sim_manifest)
    for directory, record in ((simulation, sim_manifest), (reconstruction, reco_manifest)):
        if record["status"] != "completed":
            raise ValueError(f"upstream stage failed: {directory}")
        for name, digest in record["metrics"]["output_sha256"].items():
            if _sha256(directory / name) != digest:
                raise ValueError(f"upstream output hash mismatch: {name}")
    particles = read_particles(simulation / "particles.root")
    study = sim_manifest["metrics"]["study_config"]
    if len(particles) != study["num_events"] or not np.allclose(
            np.hypot(particles.px, particles.py), study["pt_gev"], rtol=2.e-6, atol=1.e-7):
        raise ValueError("generated particles do not match declared event count/pt")
    surfaces, tracks = pd.read_csv(run / "per_surface.csv"), pd.read_csv(run / "per_track.csv")
    summary = json.loads((run / "summary.json").read_text())
    excluded = pd.read_csv(run / "excluded_tracks.csv")
    with uproot.open(Path(effective["nominal_states"]), handler=uproot.MemmapSource) as root:
        original_keys = pd.DataFrame(root["trackstates"].arrays(["event_nr", "track_nr"], library="np"))
    verify_full_cohort(effective, summary["cohort"], tracks, excluded, particles.event_id.to_numpy(), original_keys)
    if len(tracks) != summary["tracks_evaluated"] or len(surfaces) != summary["surfaces_evaluated"]:
        raise ValueError("saved cohort differs from summary")
    components = component_table(surfaces, tracks)
    components = components.merge(particles[["event_id", "eta"]], left_on="event_nr", right_on="event_id", validate="many_to_one", how="left")
    if components.eta.isna().any():
        raise ValueError("unmatched generated event metadata")
    components["eta_bin"] = pd.cut(components.eta, [-3, -2, -1, 0, 1, 2, 3], include_lowest=True).astype(str)
    if (not bool(tracks.refit_stationary.all()) or int(tracks.intersection_failures.sum()) > 0):
        components["complete"] = False
    timings = {"simulation": json.loads((simulation / "timings.json").read_text()),
               "reconstruction": json.loads((reconstruction / "timings.json").read_text()),
               "helix": json.loads((run / "timings.json").read_text())}
    for stage in ("simulation", "reconstruction"):
        t = timings[stage]
        if not np.isclose(t["wall_seconds"], t["setup_seconds"] + t["run_seconds"], atol=1.e-9, rtol=0):
            raise ValueError("inconsistent stage timings")
    for t in timings.values():
        if any(not np.isfinite(value) or value < 0 for key, value in t.items() if key.endswith("seconds")):
            raise ValueError("invalid timing value")
    return components, particles, timings, {"tracks": len(tracks), "surfaces": len(surfaces),
        "reconstructed_events": int(tracks.event_nr.nunique()),
        "generated_events": len(particles), "excluded_tracks": len(excluded),
        "simulation": str(simulation), "reconstruction": str(reconstruction),
        "study_config": study, "fit_config": {k: effective[k] for k in (
            "seed_iters", "refit_iters", "batch_size", "threads", "target_layers")},
        "geometry_sha256": manifest["inputs"]["nominal_geometry"]["sha256"],
        "digi_sha256": sim_manifest["inputs"].get("digi_config", {}).get("sha256")}


def make_figures(arms, output, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in ("measurement", "truth"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 11))
        for row, scope in enumerate(("all", "non_target", "target")):
            for column, component in enumerate(("LOC0", "LOC1")):
                arrays = {}
                for name, table in arms.items():
                    mask = table.component == component
                    if scope != "all":
                        mask &= table.target == (scope == "target")
                    arrays[name] = table.loc[mask, f"{kind}_residual"].to_numpy()
                finite = np.concatenate([v[np.isfinite(v)] for v in arrays.values()])
                edges = np.histogram_bin_edges(finite, bins=100) if len(finite) else np.linspace(-1, 1, 101)
                for name, values in arrays.items():
                    axes[row, column].hist(values[np.isfinite(values)], bins=edges, histtype="step", label=name)
                axes[row, column].set(title=f"{scope} {component}", xlabel=f"Prediction − {kind} [mm]", ylabel="Surface entries per bin")
                axes[row, column].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / f"{kind}_residuals_six_panel.png")
        plt.close(fig)
    main = metrics.loc[metrics.stratum == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ("M", "P", "U")):
        for name in GROUPS:
            cell = main.loc[main.group == name]
            ax.plot(cell.cell, cell[metric], marker="o", label=name)
        ax.set(title=metric, ylabel="mm²")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "target_error_budget.png")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--run-dirs", type=absolute_project_path, nargs=3, required=True,
                        help="g1, g2, g3 completed helix run directories, in that order")
    parser.add_argument("--output-dir", type=absolute_project_path, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    args.run_dirs = [Path(path) for path in args.run_dirs]
    if output.exists():
        parser.error("Output directory must be new")
    config = load_project_config(args.config)
    inputs = {f"{group}/{name}": run / name for group, run in zip(GROUPS, args.run_dirs)
              for name in ("manifest.json", "per_surface.csv", "per_track.csv", "summary.json", "timings.json", "excluded_tracks.csv")}
    if any(path.resolve().is_relative_to(output.resolve()) for path in inputs.values()):
        parser.error("Input/output collision")
    manifest = write_experiment_manifest(output, run_id=output.name,
        command=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        config=config, inputs=inputs, allow_missing_inputs=True, capture_source_snapshot=True)
    try:
        loaded = [load_arm(run) for run in args.run_dirs]
        verify_arm_configs([x[3]["study_config"] for x in loaded])
        for item in loaded[1:]:
            for key in ("fit_config", "geometry_sha256", "digi_sha256"):
                if item[3][key] != loaded[0][3][key]:
                    raise ValueError(f"arms differ in {key}")
        matching = verify_initial_conditions([x[1] for x in loaded])
        events = loaded[0][1].event_id.to_numpy()
        arms = {name: item[0] for name, item in zip(GROUPS, loaded)}
        for left, right in ((GROUPS[0], GROUPS[1]), (GROUPS[0], GROUPS[2])):
            a, b = arms[left], arms[right]
            keys = ["event_nr", "track_nr", "volume_id", "layer_id", "module_id", "occurrence", "component"]
            aset, bset = set(map(tuple, a[keys].to_numpy())), set(map(tuple, b[keys].to_numpy()))
            matching[f"{left}_vs_{right}"] = {"matched_components": len(aset & bset),
                "left_components": len(aset), "right_components": len(bset),
                "matched_fraction_of_union": len(aset & bset) / len(aset | bset) if aset | bset else None,
                "semantics": "record identity overlap only; never used to reduce scoring cohorts"}
        metrics, contrasts, timing_rows = [], [], []
        for volume, layer in config.target_layers:
            for component in ("LOC0", "LOC1"):
                cell = f"{volume}:{layer}|{component}"
                tables = {name: frame.loc[frame.target & (frame.layer == f"{volume}:{layer}") &
                                         (frame.component == component)] for name, frame in arms.items()}
                strata = [("all", tables)]
                for column in ("eta_bin", "prediction_regime"):
                    values = sorted(set().union(*(set(f[column]) for f in tables.values())))
                    strata.extend((f"{column}={value}", {g: f.loc[f[column] == value] for g, f in tables.items()}) for value in values)
                for stratum, frames in strata:
                    boots = bootstrap_metrics(frames, events)
                    for name, frame in frames.items():
                        metrics.append({"group": name, "cell": cell, "stratum": stratum, **summarize_cell(frame)})
                    for left, right, question in ((GROUPS[0], GROUPS[1], "material_effect"),
                                                   (GROUPS[2], GROUPS[0], "high_momentum")):
                        for metric in ("M", "U", "truth_M"):
                            a, b = boots[left][metric][0], boots[right][metric][0]
                            estimate = a - b if a is not None and b is not None else None
                            ci = interval(boots[left][metric][1] - boots[right][metric][1])
                            contrasts.append(dict(question=question, cell=cell, stratum=stratum, metric=metric,
                                estimate=estimate, ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None,
                                verdict=interval_verdict(estimate, ci, complete=estimate is not None)))
                    for metric in ("U", "truth_excess"):
                        estimate, samples = boots[GROUPS[1]][metric]
                        ci = interval(samples)
                        verdict = "not_evaluable" if estimate is None or ci is None else (
                            "statistically_compatible_with_zero" if ci[0] <= 0 <= ci[1] else "not_closed")
                        contrasts.append(dict(question="no_material_closure", cell=cell, stratum=stratum,
                            metric=metric, estimate=estimate, ci_low=ci[0] if ci else None,
                            ci_high=ci[1] if ci else None, verdict=verdict))
        for name, item in zip(GROUPS, loaded):
            for stage, timings in item[2].items():
                for key, value in timings.items():
                    if key.endswith("seconds"):
                        timing_rows.append(dict(group=name, stage=stage, measurement=key, seconds=value))
        metric_frame, contrast_frame = pd.DataFrame(metrics), pd.DataFrame(contrasts)
        metric_frame.to_csv(output / "group_metrics.csv", index=False)
        contrast_frame.to_csv(output / "contrasts.csv", index=False)
        pd.DataFrame(timing_rows).to_csv(output / "timings.csv", index=False)
        make_figures(arms, output, metric_frame)
        summary = {"schema_version": 1, "scope": "diagnostic_only", "evaluation_geometry": "nominal",
                   "bootstrap_replicates": 2000, "bootstrap_seed": 6006, "initial_matching": matching,
                   "coverage": {g: item[3] for g, item in zip(GROUPS, loaded)},
                   "timings": {g: item[2] for g, item in zip(GROUPS, loaded)},
                   "main_contrasts": [r for r in contrasts if r["stratum"] == "all"],
                   "limitations": ["material switch combines scattering and energy loss",
                       "no material-off 100 GeV arm; momentum contrast also changes curvature and acceptance",
                       "statistical compatibility is not an equivalence proof",
                       "target-only direct holdout; upstream ACTS TruthSmeared seed and smoothing are retained"]}
        write_json(output / "comparison_summary.json", summary)
        lines = ["# 三组材料效应比较", "", "评分几何：nominal；主结论只使用 target。", "",
                 "|问题|单元|指标|差值 mm²|95% CI|判断|", "|---|---|---|---|---|---|"]
        for row in summary["main_contrasts"]:
            lines.append(f'|{row["question"]}|{row["cell"]}|{row["metric"]}|{row["estimate"]}|[{row["ci_low"]}, {row["ci_high"]}]|{row["verdict"]}|')
        lines.extend(["", "|分组|模拟 s|重建 s|seed s|refit s|", "|---|---:|---:|---:|---:|"])
        for group, item in zip(GROUPS, loaded):
            t = item[2]
            lines.append(f'|{group}|{t["simulation"]["wall_seconds"]:.6f}|{t["reconstruction"]["wall_seconds"]:.6f}|{t["helix"]["seed_seconds"]:.6f}|{t["helix"]["refit_seconds"]:.6f}|')
        lines.extend(["", "完整耗时见 timings.csv；覆盖、匹配与重试次数见 comparison_summary.json。", "",
                      "材料开关衡量整体作用；高动量差异不能全部归因于材料。统计相容不等于证明纯数值误差为零。"])
        (output / "report.md").write_text("\n".join(lines) + "\n")
        finalize_experiment_manifest(manifest, status="completed", metrics=summary,
            outputs={p.name: p for p in output.iterdir() if p.name != "manifest.json" and p.is_file()})
        return 0
    except (Exception, KeyboardInterrupt) as error:
        finalize_experiment_manifest(manifest, status="failed", metrics={"error": f"{type(error).__name__}: {error}"})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
