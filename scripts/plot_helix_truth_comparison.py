"""Paired ACTS SMT / frozen helix residual and Truth-score comparison.

Reads retained results only; no fit, selection by residual, or solver tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.append(_SCRIPTS)
from project_config import absolute_project_path, load_project_config
from experiment_manifest import write_experiment_manifest, finalize_experiment_manifest
from validate_helix_model import expand_measurement_errors

KEYS = ["event_nr", "track_nr", "volume_id", "layer_id", "module_id", "occurrence"]
TRACK_KEYS = KEYS[:2]
GROUPS = ("all", "non_target", "target")
ARMS = ("ACTS", "Helix")
COLORS = {"ACTS": "#2878B5", "Helix": "#E07B24"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_unique(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Duplicate or missing identity: {keys}")


def verify_cohort_counts(parent, tracks, surfaces):
    frozen = parent["metrics"]["summary"]
    if (tracks != frozen["tracks_evaluated"] or surfaces != frozen["surfaces_evaluated"]):
        raise ValueError("CSV cohort counts differ from the frozen source summary")


def pair_surfaces(helix, acts):
    require_unique(helix, KEYS)
    require_unique(acts, KEYS)
    merged = helix.merge(acts, on=KEYS, how="outer", suffixes=("", "_root"),
                         indicator=True, validate="one_to_one", sort=False)
    if not merged["_merge"].eq("both").all():
        raise ValueError("ROOT/CSV surface identity sets differ")
    for a in range(2):
        for base in ("truth", "measurement", "variance"):
            col = f"{base}_LOC{a}_{'mm2' if base == 'variance' else 'mm'}"
            if not np.allclose(merged[col], merged[col + "_root"], rtol=1.e-9,
                               atol=1.e-10, equal_nan=True):
                raise ValueError(f"ROOT/CSV mismatch: {col}")
        col = f"weighted_LOC{a}"
        if not np.array_equal(merged[col], merged[col + "_root"]):
            raise ValueError(f"ROOT/CSV mismatch: {col}")
    return merged.drop(columns=[c for c in merged if c.endswith("_root")] + ["_merge"])


def load_acts(root_path, track_keys):
    wanted = set(map(tuple, track_keys[TRACK_KEYS].to_numpy()))
    records = []
    state_names = ["volume_id", "layer_id", "module_id", "l_x_hit", "l_y_hit",
                   "t_eLOC0", "t_eLOC1", "eLOC0_smt", "eLOC1_smt"]
    with uproot.open(root_path, handler=uproot.MemmapSource) as root:
        tree = root["trackstates"]
        required = [*TRACK_KEYS, *state_names, "err_x_hit", "err_y_hit"]
        missing = set(required) - set(tree.keys())
        if missing:
            raise ValueError(f"Missing ROOT branches: {sorted(missing)}")
        events = tree["event_nr"].array(library="np")
        tracks = tree["track_nr"].array(library="np")
        selected = [i for i, key in enumerate(zip(events, tracks)) if key in wanted]
        keys = [(int(events[i]), int(tracks[i])) for i in selected]
        if set(keys) != wanted or len(set(keys)) != len(keys):
            raise ValueError("ROOT track identities are missing or duplicated")
        selected = set(selected)
        for start in sorted({i // 1024 * 1024 for i in selected}):
            stop = min(start + 1024, tree.num_entries)
            arrays = tree.arrays([*state_names, "err_x_hit", "err_y_hit"],
                                 entry_start=start, entry_stop=stop, library="ak")
            for i in range(start, stop):
                if i not in selected:
                    continue
                j = i - start
                values = {name: ak.to_numpy(arrays[name][j]) for name in state_names}
                count = len(values["volume_id"])
                if any(len(v) != count for v in values.values()):
                    raise ValueError(f"ROOT state branch length mismatch at entry {i}")
                errors = expand_measurement_errors(arrays["l_x_hit"][j:j+1],
                                                   arrays["err_x_hit"][j:j+1],
                                                   arrays["err_y_hit"][j:j+1])
                seen = Counter()
                for hit in np.flatnonzero(np.isfinite(values["l_x_hit"])):
                    surface = tuple(int(values[k][hit]) for k in KEYS[2:5])
                    record = dict(zip(KEYS, (int(events[i]), int(tracks[i]), *surface, seen[surface])))
                    seen[surface] += 1
                    for a, measurement_name in enumerate(("l_x_hit", "l_y_hit")):
                        error = errors[a][hit]
                        variance = error ** 2 if error > 0 else np.nan
                        measurement = values[measurement_name][hit]
                        record.update({f"acts_LOC{a}_mm": values[f"eLOC{a}_smt"][hit],
                                       f"truth_LOC{a}_mm": values[f"t_eLOC{a}"][hit],
                                       f"measurement_LOC{a}_mm": measurement,
                                       f"variance_LOC{a}_mm2": variance,
                                       f"weighted_LOC{a}": bool(np.isfinite(measurement) and
                                                                np.isfinite(variance) and variance > 0)})
                    records.append(record)
    if not records:
        raise ValueError("No retained ROOT measurement states")
    return pd.DataFrame.from_records(records)


def residual_stats(values):
    if not len(values):
        return {"N": 0, "bias_mm": None, "rmse_mm": None}
    scale = np.max(np.abs(values)) or 1.
    return {"N": len(values), "bias_mm": float(np.mean(values / scale) * scale),
            "rmse_mm": float(np.sqrt(np.mean((values / scale) ** 2)) * scale)}


def full_sample_density(values, edges):
    """Legacy density helper: out-of-view samples remain in the denominator."""
    counts, _ = np.histogram(values, edges)
    return counts / (len(values) * np.diff(edges)) if len(values) else np.zeros(len(edges) - 1)


def analyze(paired, tracks):
    require_unique(tracks, TRACK_KEYS)
    lookup = {tuple(key): i for i, key in enumerate(tracks[TRACK_KEYS].to_numpy())}
    indices = np.array([lookup[tuple(key)] for key in paired[TRACK_KEYS].to_numpy()])
    if set(indices) != set(range(len(tracks))):
        raise ValueError("Retained track and surface identity sets differ")
    target = paired["target"].to_numpy(dtype=bool)
    intersection = paired["intersection_valid"].to_numpy(dtype=bool)
    active = paired[[f"weighted_LOC{a}" for a in range(2)]].to_numpy(dtype=bool)
    variance = paired[[f"variance_LOC{a}_mm2" for a in range(2)]].to_numpy(dtype=float)
    truth = paired[[f"truth_LOC{a}_mm" for a in range(2)]].to_numpy(dtype=float)
    residuals = {
        arm: paired[[f"{prefix}_LOC{a}_mm" for a in range(2)]].to_numpy(dtype=float) - truth
        for arm, prefix in (("ACTS", "acts"), ("Helix", "prediction"))}
    valid = {arm: np.isfinite(r) & intersection[:, None] for arm, r in residuals.items()}
    common = valid["ACTS"] & valid["Helix"]
    data = {"residuals": {}, "chi2": {}, "helix_raw": {}, "track_raw": {}}
    summary = {"schema_version": 1, "paired_surfaces": len(paired), "retained_tracks": len(tracks),
               "residuals": {}, "chi2": {}}

    def total(values):
        return np.bincount(indices, weights=values, minlength=len(tracks))

    for group in GROUPS:
        selected = np.ones(len(paired), dtype=bool) if group == "all" else (target if group == "target" else ~target)
        data["residuals"][group] = {}
        summary["residuals"][group] = {}
        for a in range(2):
            mask = selected & common[:, a]
            values = {arm: r[mask, a] for arm, r in residuals.items()}
            union = np.concatenate(list(values.values()))
            limit = max(float(np.quantile(np.abs(union), .995)), 1.e-6) if len(union) else 1.
            data["residuals"][group][f"LOC{a}"] = values
            summary["residuals"][group][f"LOC{a}"] = {
                **{arm: {**residual_stats(v),   "underflow": int((v < -limit).sum()),
                "overflow": int((v > limit).sum()), "outside_view": int((np.abs(v) > limit).sum()),}
                   for arm, v in values.items()},
                "core_half_width_mm": limit, "requested_surfaces": int(selected.sum()),
                "excluded_unpaired_surfaces": int((selected & ~common[:, a]).sum())}
        expected_mask = active & selected[:, None]
        expected = total(expected_mask.sum(axis=1)).astype(int)
        count_surfaces = total(selected)
        bad_intersections = total(selected & ~intersection)
        raw = {}
        for arm in ARMS:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                terms = residuals[arm] ** 2 / variance
            term_valid = valid[arm] & np.isfinite(terms)
            n_valid = total((expected_mask & term_valid).sum(axis=1)).astype(int)
            q = total(np.where(expected_mask & term_valid, terms, 0.).sum(axis=1))
            complete = (expected > 0) & (n_valid == expected) & (bad_intersections == 0) & np.isfinite(q)
            raw[arm] = {"Q": q, "N": expected, "valid_N": n_valid, "complete": complete,
                        "surface_N": count_surfaces}
        data["helix_raw"][group] = raw["Helix"]
        data["track_raw"][group] = raw
        both = raw["ACTS"]["complete"] & raw["Helix"]["complete"]
        data["chi2"][group] = {arm: {"Q": r["Q"][both], "Q_per_component": r["Q"][both] / expected[both]}
                                for arm, r in raw.items()}
        summary["chi2"][group] = {
            "paired_tracks": int(both.sum()), "no_weighted_components_tracks": int((expected == 0).sum()),
            "excluded_incomplete_tracks": int(((expected > 0) & ~both).sum()),
            **{arm: {key: {"tracks": len(v), "mean": float(v.mean()) if len(v) else None}
                     for key, v in data["chi2"][group][arm].items()} for arm in ARMS}}
    return data, summary


def verify_saved_helix(data, tracks):
    for group, raw in data["helix_raw"].items():
        expected_q = np.where(raw["complete"] | (raw["surface_N"] == 0), raw["Q"], np.nan)
        if not np.array_equal(raw["complete"], tracks[f"{group}_complete"].to_numpy(dtype=bool)):
            raise ValueError(f"Saved helix completeness mismatch: {group}")
        if not np.array_equal(raw["valid_N"], tracks[f"{group}_n_components"]):
            raise ValueError(f"Saved helix component count mismatch: {group}")
        if not np.allclose(expected_q, tracks[f"{group}_truth_chi2"], atol=1.e-9, rtol=1.e-9, equal_nan=True):
            raise ValueError(f"Saved helix chi2 mismatch: {group}")


def make_figures(data, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = summary.get("arm_labels", {arm: arm for arm in ARMS})
    residual_fig, residual_axes = plt.subplots(3, 2, figsize=(12, 12))
    chi_fig, chi_axes = plt.subplots(3, 2, figsize=(12, 12))

    def fmt(value):
        return "n/a" if value is None else f"{value:.4g}"
    for row, group in enumerate(GROUPS):
        title = {
            "all": "All surfaces",
            "non_target": "Non-target surfaces",
            "target": "Target surfaces"
            }[group]

        for a in range(2):
            ax = residual_axes[row, a]

            info = summary["residuals"][group][f"LOC{a}"]
            limit = info["core_half_width_mm"]

            # 80 displayed bins in total:
            # 1 underflow + 78 core bins + 1 overflow.
            core_edges = np.linspace(-limit, limit, 79)
            bin_width = core_edges[1] - core_edges[0]

            # Artificial plotting edges for underflow/overflow bins.
            plot_edges = np.concatenate((
                [core_edges[0] - bin_width],
                core_edges,
                [core_edges[-1] + bin_width],
            ))

            info["histogram"] = {
                "normalization": "counts",
                "binning": "core_plus_overflow",
                "core_bin_edges_mm": core_edges.tolist(),
                "plot_bin_edges_mm": plot_edges.tolist(),
                "core_half_width_mm": limit,
            }

            for index, arm in enumerate(ARMS):
                values = data["residuals"][group][f"LOC{a}"][arm]

                underflow = int(np.count_nonzero(values < -limit))
                overflow = int(np.count_nonzero(values > limit))

                core_mask = (
                    (values >= -limit) &
                    (values <= limit)
                )

                core_counts, _ = np.histogram(
                    values[core_mask],
                    bins=core_edges,
                )

                counts = np.concatenate((
                    [underflow],
                    core_counts,
                    [overflow],
                ))

                # Every residual must be represented exactly once.
                if int(counts.sum()) != len(values):
                    raise RuntimeError(
                        f"Residual histogram count mismatch for "
                        f"{group}/LOC{a}/{arm}: "
                        f"{counts.sum()} != {len(values)}"
                    )

                ax.stairs(
                    counts,
                    plot_edges,
                    color=COLORS[arm],
                    linewidth=1.7,
                    label=labels[arm],
                )

                stat = info[arm]

                ax.text(
                    .98,
                    .96 - index * .085,
                    (
                        f"{labels[arm]}: N={stat['N']:,}, "
                        f"bias={fmt(stat['bias_mm'])}, "
                        f"RMSE={fmt(stat['rmse_mm'])}"
                    ),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    color=COLORS[arm],
                    fontsize=9,
                )

            # Mark boundaries of the central 99.5%-defined region.
            ax.axvline(
                -limit,
                linestyle="--",
                linewidth=.8,
                alpha=.35,
            )
            ax.axvline(
                limit,
                linestyle="--",
                linewidth=.8,
                alpha=.35,
            )

            ax.set(
                title=f"{title} · LOC{a}",
                xlabel="Prediction − MC Truth [mm]",
                ylabel="Surface",
                xlim=(
                    -limit - bin_width,
                    limit + bin_width,
                ),
            )

            ax.set_ylim(
                0,
                max(ax.get_ylim()[1] * 1.28, 1.),
            )

            ax.grid(axis="y", alpha=.18)
        for col, key in enumerate(("Q", "Q_per_component")):
            ax = chi_axes[row, col]
            values = data["chi2"][group]
            union = np.concatenate([values[arm][key] for arm in ARMS])
            positive = union[union > 0]
            maximum = float(positive.max()) if len(positive) else 1.
            threshold = min(max(float(np.quantile(positive, .005)), 1.e-8), maximum / 10.) if len(positive) else .1
            edges = np.concatenate(([0.], np.geomspace(threshold, maximum, 80)))
            summary["chi2"][group].setdefault("histogram", {})[key] = {
                "bin_edges": edges.tolist(), "x_scale": "symlog", "linear_threshold": threshold}
            for index, arm in enumerate(ARMS):
                ax.hist(values[arm][key], bins=edges, histtype="step", color=COLORS[arm], linewidth=1.7, label=labels[arm])
                stat = summary["chi2"][group][arm][key]
                ax.text(.98, .96 - index * .085, f"{labels[arm]}: tracks={stat['tracks']:,}, mean={fmt(stat['mean'])}",
                        transform=ax.transAxes, ha="right", va="top", color=COLORS[arm], fontsize=9)
            ax.set_yscale("log")
            ax.set_xscale("symlog", linthresh=threshold, linscale=.5)
            # Keep the zero label clear of the first logarithmic decade.
            powers = np.arange(np.ceil(np.log10(threshold * 3.)), np.floor(np.log10(maximum)) + 1)
            ax.set_xticks(np.concatenate(([0.], 10. ** powers)))
            ax.set(title=title, xlabel=r"$\chi^2$" if col == 0 else r"$\chi^2/N$",
                   ylabel="Tracks", xlim=(edges[0], edges[-1]))
            ax.set_ylim(bottom=.7, top=max(3., ax.get_ylim()[1] * 3.))
            ax.grid(axis="y", alpha=.18)
    for fig, title in ((residual_fig, "ACTS SMT vs helix refit — local Truth residuals"),
                       (chi_fig, "ACTS SMT vs helix refit — track Truth scores")):
        fig.suptitle(summary.get("comparison_title", title), fontsize=15)
        fig.tight_layout(rect=(0, .065, 1, .965), h_pad=2.2)
    return residual_fig, chi_fig



def make_chi2_per_component_figure(data, summary):
    """Three comparable panels with common bins, scales and full sample counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    key = "Q_per_component"
    labels = summary.get("arm_labels", {arm: arm for arm in ARMS})
    pooled = np.concatenate([data["chi2"][g][arm][key] for g in GROUPS for arm in ARMS])
    positive = pooled[pooled > 0]
    maximum = float(positive.max()) if len(positive) else 1.
    threshold = min(max(float(np.quantile(positive, .005)), 1.e-8), maximum / 10.) if len(positive) else .1
    edges = np.r_[0., np.geomspace(threshold, maximum, 80)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
    peak = 1
    for ax, group, title in zip(axes, GROUPS, ("All surfaces", "Non-target surfaces", "Target surfaces")):
        summary["chi2"][group]["histogram"] = {key: {
            "bin_edges": edges.tolist(), "x_scale": "symlog", "linear_threshold": threshold,
            "shared_across_groups": True}}
        for index, arm in enumerate(ARMS):
            values = data["chi2"][group][arm][key]
            counts, _ = np.histogram(values, edges)
            peak = max(peak, int(counts.max()))
            ax.stairs(counts, edges, baseline=None, color=COLORS[arm], linewidth=1.8)
            stat = summary["chi2"][group][arm][key]
            mean = "n/a" if stat["mean"] is None else f"{stat['mean']:.4g}"
            ax.text(.02, 1.16 - index * .075, f"{labels[arm]}: tracks={stat['tracks']:,}, mean={mean}",
                    transform=ax.transAxes, color=COLORS[arm], fontsize=10, va="top")
        ax.set_title(title, fontsize=13, y=1.22)
        ax.set_xscale("symlog", linthresh=threshold, linscale=.6)
        ax.set_yscale("log")
        powers = np.arange(np.ceil(np.log10(threshold * 5.)), np.floor(np.log10(maximum)) + 1, 2)
        ax.set_xticks(np.r_[0., 10. ** powers])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0, maximum * 1.08)
        ax.set_xlabel(r" $\chi^2/N$", fontsize=11)
        ax.grid(axis="y", which="major", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10, labelleft=True)
    axes[0].set_ylim(.7, peak * 1.5)
    axes[0].set_ylabel("Tracks", fontsize=11)
    fig.suptitle(summary.get("comparison_title", "ACTS SMT vs helix refit"), fontsize=16, y=.98)
    fig.subplots_adjust(left=.065, right=.985, bottom=.22, top=.70, wspace=.13)
    return fig


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=absolute_project_path, required=True)
    parser.add_argument("--output-dir", type=absolute_project_path)
    parser.add_argument("--chi2-only", action="store_true", help="Only draw a horizontal chi2/N comparison PNG (no PDFs)")
    args = parser.parse_args(raw)
    run = Path(args.run_dir)
    parent_path = run / "manifest.json"
    parent = json.loads(parent_path.read_text())
    if parent["status"] != "completed":
        parser.error("Source run must be completed")
    output = Path(args.output_dir) if args.output_dir else run / "acts_comparison"
    inputs = {"source_manifest": parent_path, "per_surface": run / "per_surface.csv", "per_track": run / "per_track.csv"}
    for key in ("nominal_states", "nominal_geometry"):
        inputs[key] = Path(parent["inputs"][key]["path"])
    if any(Path(path).resolve().is_relative_to(output.resolve()) for path in inputs.values()):
        parser.error("Input/output path collision")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be new or empty")
    outputs = {name: output / name for name in (
        "acts_vs_helix_residuals.png","acts_vs_helix_residuals.pdf",
        "acts_vs_helix_chi2.png", "acts_vs_helix_chi2.pdf", "comparison_summary.json")}
    if args.chi2_only:
        outputs = {name: output / name for name in ("acts_vs_helix_chi2.png", "comparison_summary.json")}
    metrics = {"source_run": str(run), "source_configuration": parent["configuration"],
               "bins": 80, "residual_view_quantile": .995}
    metrics["historical_csv_hashes"] = "unavailable in source manifest; current hashes captured here; frozen counts and helix scores verified"
    manifest = output / "manifest.json"
    try:
        print("Recording provenance and verifying original ROOT/geometry hashes", flush=True)
        write_experiment_manifest(output, run_id=output.name,
                                  command=[sys.executable, str(Path(__file__).resolve()), *raw],
                                  config=load_project_config(), inputs=inputs, outputs=outputs, metrics=metrics,
                                  capture_source_snapshot=True, allow_missing_inputs=True)
        for key in ("nominal_states", "nominal_geometry"):
            if sha256(inputs[key]) != parent["inputs"][key]["sha256"]:
                raise ValueError(f"Original input hash mismatch: {key}")
        tracks = pd.read_csv(inputs["per_track"])
        helix = pd.read_csv(inputs["per_surface"])
        verify_cohort_counts(parent, len(tracks), len(helix))
        for frame in (tracks, helix):
            for col in frame:
                if col in ("target", "intersection_valid") or col.startswith("weighted_") or col.endswith("_complete"):
                    if frame[col].isna().any() or not frame[col].isin([True, False]).all():
                        raise ValueError(f"Invalid Boolean column: {col}")
        print(f"Matching {len(helix):,} surfaces on {len(tracks):,} retained tracks", flush=True)
        paired = pair_surfaces(helix, load_acts(inputs["nominal_states"], tracks))
        layers = {tuple(map(int, item.split(":"))) for item in parent["configuration"]["target_layers"]}
        expected_target = [tuple(k) in layers for k in paired[["volume_id", "layer_id"]].to_numpy()]
        if not np.array_equal(expected_target, paired["target"]):
            raise ValueError("Target membership differs from source configuration")
        data, summary = analyze(paired, tracks)
        geometry = parent["metrics"]["summary"].get("geometry", "nominal")
        summary["geometry"] = geometry
        summary["evaluation_geometry"] = "nominal"
        summary["arm_labels"] = {"ACTS": "nominal ACTS SMT", "Helix": f"{geometry} input helix"}
        summary["comparison_title"] = f"{geometry} input helix vs nominal ACTS SMT — nominal MC Truth"
        verify_saved_helix(data, tracks)
        summary["saved_helix_chi2_verified"] = True
        metrics["summary"] = summary
        print("Writing chi2 comparison figure", flush=True)
        if args.chi2_only:
            figures = {
                "acts_vs_helix_chi2":
                    make_chi2_per_component_figure(data, summary)
        }
        else:
            residual_fig, chi2_fig = make_figures(data, summary)

            figures = {
            "acts_vs_helix_residuals": residual_fig,
            "acts_vs_helix_chi2": chi2_fig,
            }

        outputs["comparison_summary.json"].write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n")

        import matplotlib.pyplot as plt

        for stem, fig in figures.items():
            extensions = ("png",) if args.chi2_only else ("png", "pdf")

            for extension in extensions:
                fig.savefig(outputs[f"{stem}.{extension}"],dpi=180,)

            plt.close(fig)
        metrics["output_sha256"] = {name: sha256(path) for name, path in outputs.items()}
        finalize_experiment_manifest(manifest, status="completed", metrics=metrics, outputs=outputs)
        print(f"Completed: {output}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        if manifest.exists():
            finalize_experiment_manifest(manifest, status="failed", metrics=metrics, outputs=outputs)
        print(metrics["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
