"""One-shot post-selection geometry-truth closure for simulated alignment.

The production pipeline must freeze exactly one candidate without consulting
truth. This standalone diagnostic then compares that candidate with nominal
geometry, using the original misaligned geometry as the no-correction baseline.
It must never be imported by a solver or candidate-selection path.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from compare_geometries import (  # noqa: E402
    SurfaceTransform,
    load_geometry,
    rotation_matrix_to_euler,
)
from experiment_manifest import (  # noqa: E402
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from project_config import (  # noqa: E402
    absolute_project_path,
    add_config_argument,
    load_project_config,
)
from truth_veto_gate import (  # noqa: E402
    evaluate_truth_veto,
    paired_truth_distance_ratio_ci,
    primary_metric,
)

DOF_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
DEFAULT_BOOTSTRAP_SEED = 6006
DEFAULT_N_BOOTSTRAP = 2000


class ExperimentInterrupted(KeyboardInterrupt):
    """Raised by the CLI signal handler so the manifest can be finalized."""


def _parse_target_layers(value: str) -> tuple[tuple[int, int], ...]:
    layers: list[tuple[int, int]] = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                f"invalid target layer {item!r}; expected VOLUME:LAYER"
            )
        try:
            target_layer = (int(fields[0]), int(fields[1]))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid target layer {item!r}; expected integer IDs"
            ) from error
        if target_layer in layers:
            raise argparse.ArgumentTypeError(f"duplicate target layer {item!r}")
        layers.append(target_layer)
    if not layers:
        raise argparse.ArgumentTypeError("at least one target layer is required")
    return tuple(layers)


def _parse_active_dof(value: str) -> tuple[int, ...]:
    try:
        active = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "active DOF must be comma-separated integer indices"
        ) from error
    if not active:
        raise argparse.ArgumentTypeError("at least one active DOF is required")
    if len(active) != len(set(active)) or any(
        index < 0 or index >= len(DOF_NAMES) for index in active
    ):
        raise argparse.ArgumentTypeError(
            "active DOF indices must be unique values in [0, 5]"
        )
    return active


def _local_delta(reference: SurfaceTransform, other: SurfaceTransform) -> np.ndarray:
    translation = reference.rotation.T @ (
        other.translation - reference.translation
    )
    rotation = rotation_matrix_to_euler(reference.rotation.T @ other.rotation)
    return np.concatenate((translation, np.asarray(rotation, dtype=np.float64)))


def _require_geometry_parity(
    nominal: dict[tuple[int, int, int], SurfaceTransform],
    misaligned: dict[tuple[int, int, int], SurfaceTransform],
    candidate: dict[tuple[int, int, int], SurfaceTransform],
    target_layers: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    target = set(target_layers)
    outside_misaligned = sorted(key for key in misaligned if key[:2] not in target)
    outside_candidate = sorted(key for key in candidate if key[:2] not in target)
    if outside_misaligned or outside_candidate:
        raise ValueError(
            "geometry overlays contain surfaces outside target layers: "
            f"misaligned={outside_misaligned[:5]}, "
            f"candidate={outside_candidate[:5]}"
        )
    if set(candidate) != set(misaligned):
        raise ValueError(
            "candidate and misaligned sensor sets differ: "
            f"missing_candidate={sorted(set(misaligned) - set(candidate))[:5]}, "
            f"extra_candidate={sorted(set(candidate) - set(misaligned))[:5]}"
        )
    missing_nominal = sorted(set(misaligned) - set(nominal))
    if missing_nominal:
        raise ValueError(
            f"target sensors are absent from nominal geometry: {missing_nominal[:5]}"
        )

    grouped = {
        target_layer: sorted(
            key for key in misaligned if key[:2] == target_layer
        )
        for target_layer in target_layers
    }
    empty = [
        f"{volume}:{layer}"
        for (volume, layer), keys in grouped.items()
        if not keys
    ]
    if empty:
        raise ValueError(f"target layers contain no comparable sensors: {empty}")
    return grouped


def compute_geometry_truth_veto(
    *,
    misaligned_geometry_path: Path,
    nominal_geometry_path: Path,
    candidate_geometry_path: Path,
    target_layers: tuple[tuple[int, int], ...],
    active_dof: tuple[int, ...],
    bootstrap_seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    nominal = load_geometry(nominal_geometry_path, label="nominal")
    misaligned = load_geometry(misaligned_geometry_path, label="misaligned")
    candidate = load_geometry(candidate_geometry_path, label="candidate")
    grouped = _require_geometry_parity(
        nominal, misaligned, candidate, target_layers
    )

    metric_results: dict[str, dict[str, Any]] = {}
    metric_specs = []
    layer_counts: dict[str, int] = {}
    for volume, layer in target_layers:
        keys = grouped[(volume, layer)]
        layer_counts[f"{volume}:{layer}"] = len(keys)
        truth = np.stack(
            [_local_delta(misaligned[key], nominal[key]) for key in keys]
        )
        prediction = np.stack(
            [_local_delta(misaligned[key], candidate[key]) for key in keys]
        )
        baseline_error = -truth
        candidate_error = prediction - truth
        sensor_ids = np.arange(len(keys), dtype=np.int64)
        valid = np.ones(len(keys), dtype=bool)

        for dof_index in active_dof:
            name = f"{volume}_{layer}_{DOF_NAMES[dof_index]}"
            raw = paired_truth_distance_ratio_ci(
                candidate_error[:, dof_index],
                baseline_error[:, dof_index],
                sensor_ids,
                valid,
                valid,
                bootstrap_seed=bootstrap_seed,
                n_bootstrap=n_bootstrap,
            )
            metric_results[name] = {
                "n_blocks": raw["n_events"],
                "n_values": raw["n_hits"],
                "nominal_ratio": raw["nominal_ratio"],
                "ci_upper": raw["ci_upper"],
                "ci_quantile": raw["ci_quantile"],
                "n_finite_resamples": raw["n_finite_resamples"],
                "bootstrap": raw["bootstrap"],
            }
            metric_specs.append(primary_metric(name))

    verdict = evaluate_truth_veto(
        metric_results, metric_specs, after_candidate_selection=True
    )
    return {
        "schema_version": 1,
        "claim_scope": "post_hoc_geometry_truth_veto_only",
        "target_layers": [
            f"{volume}:{layer}" for volume, layer in target_layers
        ],
        "active_dof": [DOF_NAMES[index] for index in active_dof],
        "layer_sensor_counts": layer_counts,
        "metrics": metric_results,
        "verdict": verdict,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--misaligned-geometry", type=absolute_project_path, required=True
    )
    parser.add_argument(
        "--nominal-geometry", type=absolute_project_path, required=True
    )
    parser.add_argument(
        "--candidate-geometry", type=absolute_project_path, required=True
    )
    parser.add_argument(
        "--target-layers",
        type=_parse_target_layers,
        default=((17, 8), (24, 4)),
    )
    parser.add_argument(
        "--active-dof", type=_parse_active_dof, default=(0, 1, 5)
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP
    )
    parser.add_argument(
        "--run-dir", type=absolute_project_path, required=True
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=absolute_project_path, required=True)
    parser.add_argument("--capture-source-snapshot", action="store_true")
    return parser


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.bootstrap_seed < 0:
        raise ValueError("--bootstrap-seed must be non-negative")
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")

    run_dir = Path(args.run_dir)
    output = Path(args.output)
    if run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing run directory: {run_dir}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    run_id = args.run_id or f"truth-veto-closure-{run_dir.name}"
    command_args = list(sys.argv[1:] if argv is None else argv)
    manifest_path = run_dir / "manifest.json"

    def interrupt(signum, _frame):
        raise ExperimentInterrupted(f"received signal {signum}")

    old_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    for signum in old_handlers:
        signal.signal(signum, interrupt)

    try:
        write_experiment_manifest(
            run_dir,
            run_id=run_id,
            command=[
                sys.executable,
                str(Path(__file__).resolve()),
                *command_args,
            ],
            config=load_project_config(args.config),
            inputs={
                "misaligned_geometry": args.misaligned_geometry,
                "nominal_geometry": args.nominal_geometry,
                "candidate_geometry": args.candidate_geometry,
            },
            capture_source_snapshot=args.capture_source_snapshot,
        )
        result = compute_geometry_truth_veto(
            misaligned_geometry_path=Path(args.misaligned_geometry),
            nominal_geometry_path=Path(args.nominal_geometry),
            candidate_geometry_path=Path(args.candidate_geometry),
            target_layers=args.target_layers,
            active_dof=args.active_dof,
            bootstrap_seed=args.bootstrap_seed,
            n_bootstrap=args.n_bootstrap,
        )
        _atomic_write_json(output, result)
        finalize_experiment_manifest(
            manifest_path,
            status="completed",
            outputs={"truth_veto_result": output},
            metrics={"verdict": result["verdict"]},
        )
        print(json.dumps(result["verdict"], indent=2))
        return 0 if result["verdict"]["passed"] else 1
    except BaseException as error:
        if manifest_path.is_file():
            outputs = (
                {"truth_veto_result": output} if output.is_file() else None
            )
            finalize_experiment_manifest(
                manifest_path,
                status="failed",
                outputs=outputs,
                metrics={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
