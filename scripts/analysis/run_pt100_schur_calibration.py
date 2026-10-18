#!/usr/bin/env python
"""Time Schur-LM calibration on the pinned pT=100 GeV ACTS-vs-Schur study inputs.

Runs the production free-mode pipeline (``run_free_alignment_pipeline``) with the
orchestrator's joint_free arguments and wraps its stage functions with wall/CPU
timers. ``--threads 1`` is the primary arm: ACTS-Alignment is serial, so the
timing comparison is made at equal compute resources.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Copied from run_surface_helix_experiment.sh joint_free; CLI defaults are unvalidated.
ORCHESTRATOR_ARGS = (
    "--track-mode", "free", "--joint-solver", "schur-lm",
    "--joint-smt-seed-iters", "2000",
    "--joint-seed-min-hits", "8", "--joint-seed-min-span", "300",
    "--lambda-l2", "1e-5", "--lm-max-iters", "60",
    "--no-plot",
)
TIMED_STAGES = {
    "load": "load_variable_information",
    "track_selection": "build_track_batches",
    "seed": "seed_helices",
    "refit": "fit_helices_to_measurements",
    "truth_validation": "validate_against_truth",
    "export": "export_joint_geometry",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", choices=("1", "default"), default="1",
                        help="1 = primary equal-resource arm; default = torch's own count.")
    parser.add_argument("--run-id", default=None, help="Defaults to the pinned study run.")
    parser.add_argument("--max-tracks", type=int, default=0, help="Smoke runs only.")
    return parser.parse_args(argv)


def require_completed_inputs(paths):
    """ACTS alignment and the misaligned reconstruction must both be completed.

    The pt100 study gives the misaligned reconstruction its own manifest; the
    `alignment_odd.py full` workflow records it in the alignment manifest instead.
    """
    import json

    alignment = json.loads((Path(paths["alignment"]) / "manifest.json").read_text())
    if alignment["status"] != "completed":
        raise ValueError(f"ACTS alignment is not completed: {paths['alignment']}")
    stage_manifest = Path(paths["misaligned"]) / "manifest.json"
    if stage_manifest.exists():
        completed = json.loads(stage_manifest.read_text())["status"] == "completed"
    else:
        recorded = alignment.get("outputs", {}).get("reconstruction_misaligned")
        completed = (recorded is not None
                     and Path(recorded).resolve() == Path(paths["misaligned"]).resolve())
    if not completed:
        raise ValueError(f"misaligned reconstruction is not completed: {paths['misaligned']}")


class StageTimer:
    def __init__(self, clock_wall, clock_cpu):
        self._wall, self._cpu = clock_wall, clock_cpu
        self.stages: dict[str, dict[str, float]] = {}

    def add(self, name, wall, cpu):
        row = self.stages.setdefault(name, {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0})
        row["wall_seconds"] += wall
        row["cpu_seconds"] += cpu
        row["calls"] += 1

    def wrap(self, name, function, sink=None):
        def timed(*args, **kwargs):
            wall, cpu = self._wall(), self._cpu()
            try:
                result = function(*args, **kwargs)
            finally:
                self.add(name, self._wall() - wall, self._cpu() - cpu)
            if sink is not None:
                sink[name] = result
            return result
        return timed


def main(argv=None):
    args = parse_args(argv)
    if args.threads == "1":
        for name in THREAD_ENV:
            os.environ[name] = "1"
    sys.path.insert(0, str(SCRIPTS))
    import json
    import resource
    import time

    import torch

    if args.threads == "1":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    import ml_alignment_autograd as M
    import mlalign.free_helix_pipeline as pipeline
    import pt100_alignment_layout as pt100
    from alignment_artifact_layout import TARGET_LAYER_DIR, artifact_filename
    from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from project_config import load_project_config

    config = load_project_config()
    run_id = args.run_id or pt100.PT100_ALIGNMENT_RUN
    paths = pt100.pt100_alignment_paths(config.data_root, TARGET_LAYER_DIR, run_id)
    arm = "schur_single_thread" if args.threads == "1" else "schur_multi_thread"
    output = paths[arm]
    if output.exists():
        raise FileExistsError(f"refusing to reuse Schur arm output: {output}")
    require_completed_inputs(paths)
    output.mkdir(parents=True)
    geometry = (paths["schur_geometry"] if arm == "schur_single_thread"
                else output / f"aligned_transforms_{run_id}_schur_lm_multi_thread.json")
    states = paths["misaligned"] / artifact_filename("trackstates_kf", ".root")
    summary = paths["misaligned"] / artifact_filename("tracksummary_kf", ".root")
    cli = [*ORCHESTRATOR_ARGS, "--states", str(states), "--summary", str(summary),
           "--input-geom", str(paths["misalignment"]), "--output-geom", str(geometry),
           "--max-tracks", str(args.max_tracks)]
    run_config = M.alignment_config_from_args(M.build_argument_parser().parse_args(cli))
    threads = {"torch_num_threads": torch.get_num_threads(),
               "torch_num_interop_threads": torch.get_num_interop_threads(),
               **{name: os.environ.get(name) for name in THREAD_ENV}}
    if args.threads == "1" and (threads["torch_num_threads"] != 1
                                or threads["torch_num_interop_threads"] != 1):
        raise RuntimeError(f"single-thread arm is not single-threaded: {threads}")
    inputs = {"states": states, "summary": summary, "misalignment": paths["misalignment"],
              "nominal_geometry": run_config.nominal_full_geom}
    timings_path, summary_path = output / "timings.json", output / "alignment_summary.json"
    manifest = write_experiment_manifest(
        output, run_id=f"{run_id}/{arm}", command=[sys.executable, *sys.argv],
        config=config, inputs=inputs,
        metrics={"arm": arm, "solver_cli": cli, "threads": threads},
        capture_source_snapshot=True)

    timer = StageTimer(time.perf_counter, time.process_time)
    captured: dict = {}
    # Wrap the names where run_free_alignment_pipeline looks them up; rebinding
    # them on the M facade would not reach the pipeline.
    for stage, attribute in TIMED_STAGES.items():
        setattr(pipeline, attribute, timer.wrap(stage, getattr(pipeline, attribute)))
    select_solver = pipeline.select_joint_solver
    pipeline.select_joint_solver = lambda cfg: timer.wrap("schur_solve", select_solver(cfg), captured)

    status, error, result = "failed", None, None
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        result = M.run_alignment_pipeline(run_config)
        status = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        total = {"wall_seconds": time.perf_counter() - wall,
                 "cpu_seconds": time.process_time() - cpu}
        timings = {"schema_version": 1, "arm": arm, "threads": threads,
                   "stages": timer.stages, "pipeline_total": total,
                   "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                   "excludes": "interpreter start, imports and manifest hashing"}
        timings_path.write_text(json.dumps(timings, indent=2, allow_nan=False) + "\n")
        metrics = {"timings": timings, "error": error}
        outputs = {"timings": timings_path}
        if result is not None:
            M.write_alignment_summary(result, summary_path)
            outputs.update({"alignment_summary": summary_path, "geometry": geometry})
            metrics["schur_lm_summary"] = M._json_ready(result.get("schur_lm_summary"))
            metrics["truth_metrics"] = M._json_ready(result.get("truth_metrics"))
        finalize_experiment_manifest(manifest, status=status, outputs=outputs, metrics=metrics)
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
