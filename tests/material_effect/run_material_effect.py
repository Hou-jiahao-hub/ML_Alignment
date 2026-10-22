"""Three-arm material study on the current mlalign package.

    python tests/material_effect/run_material_effect.py verify  --run-root R --arm G=STATES,PARTICLES,HITS ...
    python tests/material_effect/run_material_effect.py arm     --run-root R --group G --states S --summary S --particles P
    python tests/material_effect/run_material_effect.py compare --run-root R [--reference-contrasts CSV]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest  # noqa: E402
from project_config import load_project_config  # noqa: E402

from material_effect import material_budget as mb  # noqa: E402

NOMINAL_GEOMETRY = REPO / "data" / "geometry" / "odd_transforms.json"


def _arm_paths(values):
    arms = {}
    for value in values:
        group, paths = value.split("=", 1)
        states, particles, hits = paths.split(",")
        arms[group] = (Path(states), Path(particles), Path(hits))
    if tuple(arms) != mb.GROUPS:
        raise ValueError(f"--arm must list exactly {mb.GROUPS} in order")
    return arms


def run_verify(args):
    arms = _arm_paths(args.arm)
    particles = {g: mb.read_particles(p) for g, (_s, p, _h) in arms.items()}
    spreads = {g: mb.momentum_spread(h) for g, (_s, _p, h) in arms.items()}
    result = mb.verify_arms(particles, spreads)
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    mb.write_json(root / "arm_verification.json", result)
    print(result)


def run_arm(args):
    torch.set_num_threads(args.threads)
    output = Path(args.run_root) / args.group
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = load_project_config()
    inputs = {"states": args.states, "summary": args.summary, "particles": args.particles,
              "nominal_geometry": args.nominal_geometry}
    manifest = write_experiment_manifest(
        output, run_id=f"{Path(args.run_root).name}-{args.group}", command=[sys.executable, *sys.argv],
        config=config, inputs=inputs, capture_source_snapshot=True,
        metrics={"group": args.group, "max_entries": args.max_entries})
    timings, clock = {}, time.perf_counter
    try:
        start = clock()
        geometry = mb.nominal_geometry(args.nominal_geometry, output)
        batches, floor_stats = mb.load_arm_batches(
            args.states, args.summary, geometry, max_track_entries=args.max_entries)
        truth, repeated_truth = mb.unique_truth(mb.read_target_truth(args.states, max_entries=args.max_entries))
        timings["load_seconds"] = clock() - start

        start = clock()
        parameters, seed_stats, refit_stats = mb.fit_arm(batches, geometry, seed_iters=args.seed_iters)
        timings["seed_refit_seconds"] = clock() - start

        start = clock()
        stationary = mb.row_stationary(parameters, batches, geometry)
        timings["stationarity_seconds"] = clock() - start

        start = clock()
        prediction, valid, budget, rank_deficient = mb.target_budget(
            parameters, mb.budget_inputs(batches, geometry), batches.fit_mask,
            batches.is_target & batches.hit_mask, chunk_rows=args.chunk_rows)
        timings["budget_seconds"] = clock() - start

        components, join_counts = mb.component_table(
            batches, geometry, prediction, valid, budget, stationary, truth)
        components.to_csv(output / "components.csv.gz", index=False)
        target = batches.is_target & batches.hit_mask
        summary = {
            "group": args.group,
            "particles": str(Path(args.particles).resolve()),
            "selection": batches.selection_stats,
            "n_tracks": int(parameters.shape[0]),
            "selected_event_ids": (sorted(set(batches.event_track_ids[:, 0].tolist()))
                                   if args.max_entries else None),
            "variance_protocol": "V = err^2 (pre STRIP_Y_FLOOR, floor 1e-8)",
            "floor": floor_stats,
            "truth_join": {**join_counts, "repeated_truth_states": repeated_truth},
            "refit": {k: refit_stats[k] for k in ("chi2_per_component", "n_components", "invalid_fraction")},
            "refit_stationarity_gate": refit_stats.get("stationarity_gate"),
            "row_non_stationary": int((~stationary).sum()),
            "target_intersection_failures": int((target & ~valid).sum()),
            "rank_deficient_tracks": int(rank_deficient.sum()),
            "components": int(len(components)),
            "incomplete_components": int((~components.complete).sum()),
        }
        timings["wall_seconds"] = sum(timings.values())
        mb.write_json(output / "summary.json", summary)
        mb.write_json(output / "timings.json", timings)
        finalize_experiment_manifest(manifest, status="completed", metrics={"summary": summary, "timings": timings},
            outputs={name: output / name for name in ("components.csv.gz", "summary.json", "timings.json")})
        print(summary)
    except BaseException as error:
        finalize_experiment_manifest(manifest, status="failed", metrics={"error": f"{type(error).__name__}: {error}"})
        raise


def _load_components(run_root, group):
    frame = pd.read_csv(Path(run_root) / group / "components.csv.gz")
    frame["cell"] = frame.layer + "|" + frame.component
    return frame


def run_compare(args):
    root = Path(args.run_root)
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    summaries = {g: pd.read_json(root / g / "summary.json", typ="series") for g in mb.GROUPS}
    frames = {g: _load_components(root, g) for g in mb.GROUPS}
    particles = mb.read_particles(summaries[mb.GROUPS[0]]["particles"])
    partial = [s["selected_event_ids"] for s in summaries.values() if s["selected_event_ids"] is not None]
    events = particles.event_id.to_numpy()
    if partial:
        events = np.array(sorted(set.intersection(*map(set, partial))))
        frames = {g: f[f.event_nr.isin(events)] for g, f in frames.items()}
    eta = particles.set_index("event_id").eta
    cells = sorted(set.union(*(set(f.cell) for f in frames.values())))

    metric_rows, contrast_rows, eta_rows = [], [], []
    for cell in cells:
        by_group = {g: f[f.cell == cell] for g, f in frames.items()}
        for group, frame in by_group.items():
            metric_rows.append({"group": group, "cell": cell, "stratum": "all", **mb.summarize_cell(frame)})
            bins = pd.cut(frame.event_nr.map(eta), [-3, -2, -1, 0, 1, 2, 3], include_lowest=True).astype(str)
            for label, part in frame.groupby(bins):
                eta_rows.append({"group": group, "cell": cell, "stratum": f"eta {label}", **mb.summarize_cell(part)})
        boot = mb.bootstrap_metrics(by_group, events, replicates=args.replicates, seed=args.seed)
        contrast_rows += [{"cell": cell, "stratum": "all", **row} for row in mb.contrasts(boot)]
    metrics = pd.DataFrame(metric_rows + eta_rows)
    table = pd.DataFrame(contrast_rows)
    metrics.to_csv(output / "group_metrics.csv", index=False)
    table.to_csv(output / "contrasts.csv", index=False)

    lines = ["# 三组材料效应比较（100k，mlalign 重写）", "",
             f"评分几何：nominal；bootstrap {args.replicates} 次，seed {args.seed}，"
             f"事件池 {len(events)} 个 generated event。", "",
             "|问题|单元|指标|差值 mm²|95% CI|判断|", "|---|---|---|---:|---|---|"]
    for row in table.itertuples():
        ci = "—" if pd.isna(row.ci_low) else f"[{row.ci_low:.3e}, {row.ci_high:.3e}]"
        value = "—" if pd.isna(row.difference_mm2) else f"{row.difference_mm2:.3e}"
        lines.append(f"|{row.question}|{row.cell.replace('|', ' ')}|{row.metric}|{value}|{ci}|{row.verdict}|")
    lines += ["", "|分组|单元|n|M|V|P|U|truth_M|pull²|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics[metrics.stratum == "all"].itertuples():
        fmt = lambda v: "—" if v is None or pd.isna(v) else f"{v:.3e}"  # noqa: E731
        lines.append(f"|{row.group}|{row.cell.replace('|', ' ')}|{row.n_components}|{fmt(row.M)}|{fmt(row.V)}|"
                     f"{fmt(row.P)}|{fmt(row.U)}|{fmt(row.truth_M)}|{fmt(getattr(row, 'pull2', None))}|")

    comparison = None
    if args.reference_contrasts:
        reference = pd.read_csv(args.reference_contrasts)
        reference = reference[reference.stratum == "all"].rename(columns={
            "estimate": "ref_estimate", "ci_low": "ref_ci_low", "ci_high": "ref_ci_high", "verdict": "ref_verdict"})
        comparison = table.merge(reference, on=["question", "cell", "stratum", "metric"], how="left")
        comparison["inside_reference_ci"] = ((comparison.difference_mm2 >= comparison.ref_ci_low)
                                             & (comparison.difference_mm2 <= comparison.ref_ci_high))
        comparison["ci_width_ratio"] = ((comparison.ci_high - comparison.ci_low)
                                        / (comparison.ref_ci_high - comparison.ref_ci_low))
        comparison["verdict_unchanged"] = comparison.verdict == comparison.ref_verdict
        comparison.to_csv(output / "reference_comparison.csv", index=False)
        lines += ["", "## 对照 20k 旧结果", "",
                  "|问题|单元|指标|100k 估计|20k 估计|落在 20k CI 内|CI 宽度比|判读 100k / 20k|",
                  "|---|---|---|---:|---:|---|---:|---|"]
        for row in comparison.itertuples():
            lines.append(f"|{row.question}|{row.cell.replace('|', ' ')}|{row.metric}|{row.difference_mm2:.3e}|"
                         f"{row.ref_estimate:.3e}|{row.inside_reference_ci}|{row.ci_width_ratio:.2f}|"
                         f"{row.verdict} / {row.ref_verdict}|")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    mb.write_json(output / "comparison_summary.json", {
        "events_in_pool": int(len(events)), "partial_arms": bool(partial),
        "replicates": args.replicates, "seed": args.seed,
        "arms": {g: {k: summaries[g][k] for k in ("n_tracks", "components", "incomplete_components",
                                                   "row_non_stationary", "target_intersection_failures",
                                                   "rank_deficient_tracks")} for g in mb.GROUPS},
        "verdicts": table[["question", "cell", "metric", "verdict"]].to_dict("records"),
    })
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--arm", action="append", required=True, help="GROUP=STATES,PARTICLES,HITS")
    arm = sub.add_parser("arm")
    arm.add_argument("--run-root", required=True)
    arm.add_argument("--group", required=True, choices=mb.GROUPS)
    arm.add_argument("--states", required=True)
    arm.add_argument("--summary", required=True)
    arm.add_argument("--particles", required=True)
    arm.add_argument("--nominal-geometry", default=str(NOMINAL_GEOMETRY))
    arm.add_argument("--max-entries", type=int, default=None)
    arm.add_argument("--seed-iters", type=int, default=2000)
    arm.add_argument("--chunk-rows", type=int, default=1024)
    arm.add_argument("--threads", type=int, default=6)
    compare = sub.add_parser("compare")
    compare.add_argument("--run-root", required=True)
    compare.add_argument("--reference-contrasts")
    compare.add_argument("--replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=6006)
    args = parser.parse_args(argv)
    {"verify": run_verify, "arm": run_arm, "compare": run_compare}[args.command](args)


if __name__ == "__main__":
    main()
