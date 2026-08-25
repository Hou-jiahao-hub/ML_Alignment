"""Evaluate ACTS reconstruction residual chi2 for aligned target layers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

try:
    from .alignment_artifact_layout import TARGET_LAYERS, target_layers_spec
    from .event_split import DEFAULT_SPLIT_SEED, PARTITIONS, event_partition_mask
    from .project_config import absolute_project_path, add_config_argument
except ImportError:  # Direct script execution.
    from alignment_artifact_layout import TARGET_LAYERS, target_layers_spec
    from event_split import DEFAULT_SPLIT_SEED, PARTITIONS, event_partition_mask
    from project_config import absolute_project_path, add_config_argument


SCHEMA_VERSION = "2.0"
DEFAULT_THRESHOLD = 1.0
STRIP_VOLUMES = {23, 24, 25}
STRIP_LOC1_VARIANCE_FLOOR = 1.2**2
REQUIRED_FIELDS = {
    "volume_id",
    "layer_id",
    "l_x_hit",
    "l_y_hit",
    "eLOC0_smt",
    "eLOC1_smt",
    "err_x_hit",
    "err_y_hit",
}


def parse_layers(value) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, str):
        return tuple((int(volume), int(layer)) for volume, layer in value)
    parsed = []
    for item in value.split(","):
        volume, layer = item.strip().split(":", 1)
        parsed.append((int(volume), int(layer)))
    if not parsed:
        raise ValueError("At least one target layer is required.")
    return tuple(parsed)


def _invalid_component() -> dict[str, object]:
    return {
        "chi2_per_component": None,
        "n_components": 0,
        "valid": False,
        "passed": False,
    }


def _component_metric(
    residual: np.ndarray,
    variance: np.ndarray,
    source_mask: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    count = int(source_mask.sum())
    if count == 0:
        return _invalid_component()
    selected_residual = residual[source_mask]
    selected_variance = variance[source_mask]
    valid = bool(
        np.isfinite(selected_residual).all()
        and np.isfinite(selected_variance).all()
        and (selected_variance > 0).all()
    )
    if not valid:
        metric = _invalid_component()
        metric["n_components"] = count
        return metric
    chi2 = float(np.mean(selected_residual**2 / selected_variance))
    valid = math.isfinite(chi2)
    return {
        "chi2_per_component": chi2 if valid else None,
        "threshold": threshold,
        "normalized_chi2": chi2 / threshold if valid and threshold > 0 else None,
        "n_components": count,
        "valid": valid,
        "passed": bool(valid and chi2 <= threshold),
    }


def acceptance_score(result: dict) -> tuple[int, float, float]:
    """Return the lexicographic polish comparison score (smaller is better)."""
    metrics = [
        component
        for layer in result.get("layers", {}).values()
        for component in layer.values()
    ]
    failed_count = sum(not bool(metric.get("passed")) for metric in metrics)
    values = []
    for metric in metrics:
        normalized = metric.get("normalized_chi2")
        if normalized is None:
            value = metric.get("chi2_per_component")
            threshold = metric.get("threshold", result.get("threshold", 1.0))
            normalized = None if value is None or not threshold else float(value) / float(threshold)
        values.append(normalized)
    if not values or any(value is None or not math.isfinite(float(value)) for value in values):
        return failed_count, float("inf"), float("inf")
    finite_values = [float(value) for value in values]
    return (
        failed_count,
        max(finite_values),
        sum(max(value - 1.0, 0.0) for value in finite_values),
    )


def polish_improves(candidate_score, previous_score) -> bool:
    return tuple(candidate_score) < tuple(previous_score)


def compute_reconstruction_acceptance(
    frame: pd.DataFrame,
    target_layers=TARGET_LAYERS,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    thresholds: dict | None = None,
    event_partition: str | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict:
    """Compute measurement-minus-smoothed-state chi2 for LOC0 and LOC1."""
    missing = sorted(REQUIRED_FIELDS - set(frame.columns))
    if missing:
        raise ValueError(f"trackstates are missing required fields: {', '.join(missing)}")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative value")
    if event_partition is not None:
        if "event_nr" not in frame:
            raise ValueError("trackstates are missing required event partition field: event_nr")
        frame = frame[event_partition_mask(frame["event_nr"].to_numpy(), event_partition, split_seed)]
    targets = parse_layers(target_layers)
    layers: dict[str, dict[str, dict[str, object]]] = {}
    failed_layers = []
    failed_components = []
    for volume, layer in targets:
        key = f"{volume}:{layer}"
        component_thresholds = (thresholds or {}).get(key, {})
        loc0_threshold = float(component_thresholds.get("LOC0", threshold))
        loc1_threshold = float(component_thresholds.get("LOC1", threshold))
        if not all(math.isfinite(value) and value > 0 for value in (loc0_threshold, loc1_threshold)):
            raise ValueError(f"Thresholds for {key} must be finite and positive")
        rows = frame[(frame["volume_id"] == volume) & (frame["layer_id"] == layer)]
        if rows.empty:
            layer_metrics = {"LOC0": _invalid_component(), "LOC1": _invalid_component()}
        else:
            loc0_source = np.isfinite(rows["l_x_hit"].to_numpy(dtype=float))
            loc1_source = np.isfinite(rows["l_y_hit"].to_numpy(dtype=float))
            residual0 = (
                rows["l_x_hit"].to_numpy(dtype=float)
                - rows["eLOC0_smt"].to_numpy(dtype=float)
            )
            residual1 = (
                rows["l_y_hit"].to_numpy(dtype=float)
                - rows["eLOC1_smt"].to_numpy(dtype=float)
            )
            variance0 = rows["err_x_hit"].to_numpy(dtype=float) ** 2
            variance1 = rows["err_y_hit"].to_numpy(dtype=float) ** 2
            if volume in STRIP_VOLUMES:
                variance1 = np.maximum(variance1, STRIP_LOC1_VARIANCE_FLOOR)
            layer_metrics = {
                "LOC0": _component_metric(
                    residual0, variance0, loc0_source, loc0_threshold
                ),
                "LOC1": _component_metric(
                    residual1, variance1, loc1_source, loc1_threshold
                ),
            }
        layers[key] = layer_metrics
        failed = False
        for component_name, metric in layer_metrics.items():
            if not metric["passed"]:
                failed = True
                failed_components.append(f"{key}:{component_name}")
        if failed:
            failed_layers.append(key)
    result = {
        "schema_version": SCHEMA_VERSION,
        "threshold": threshold,
        "threshold_mode": "per_component" if thresholds is not None else "scalar",
        "event_partition": event_partition,
        "split_seed": split_seed if event_partition is not None else None,
        "layers": layers,
        "valid": not any(
            not metric["valid"]
            for layer_metrics in layers.values()
            for metric in layer_metrics.values()
        ),
        "passed": not failed_components,
        "failed_layers": failed_layers,
        "failed_components": failed_components,
    }
    score = acceptance_score(result)
    result["acceptance_score"] = [
        score[0],
        score[1] if math.isfinite(score[1]) else None,
        score[2] if math.isfinite(score[2]) else None,
    ]
    return result


def _component_contributions(rows: pd.DataFrame, volume: int, component: str):
    if component == "LOC0":
        measured, predicted, errors = "l_x_hit", "eLOC0_smt", "err_x_hit"
    else:
        measured, predicted, errors = "l_y_hit", "eLOC1_smt", "err_y_hit"
    source = np.isfinite(rows[measured].to_numpy(dtype=float))
    selected = rows.loc[source, ["event_nr", measured, predicted, errors]].copy()
    residual = selected[measured].to_numpy(dtype=float) - selected[predicted].to_numpy(dtype=float)
    variance = selected[errors].to_numpy(dtype=float) ** 2
    if component == "LOC1" and volume in STRIP_VOLUMES:
        variance = np.maximum(variance, STRIP_LOC1_VARIANCE_FLOOR)
    if len(selected) == 0 or not np.isfinite(residual).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError(f"Invalid nominal calibration component {volume}:{component}")
    selected["weighted"] = residual**2 / variance
    grouped = selected.groupby("event_nr", sort=True)["weighted"].agg(["sum", "count"])
    return grouped["sum"].to_numpy(dtype=float), grouped["count"].to_numpy(dtype=np.int64)


def build_acceptance_baseline(
    frame: pd.DataFrame,
    target_layers=TARGET_LAYERS,
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    bootstrap_seed: int = 6006,
    n_bootstrap: int = 2000,
) -> dict:
    """Build frozen per-component predictive limits from nominal calibration events."""
    missing = sorted((REQUIRED_FIELDS | {"event_nr"}) - set(frame.columns))
    if missing:
        raise ValueError(f"trackstates are missing required fields: {', '.join(missing)}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    event_ids = frame["event_nr"].to_numpy(dtype=np.int64)
    calibration = frame[event_partition_mask(event_ids, "calibration", split_seed)]
    validation = frame[event_partition_mask(event_ids, "validation", split_seed)]
    calibration_events = int(calibration["event_nr"].nunique())
    validation_events = int(validation["event_nr"].nunique())
    if calibration_events == 0 or validation_events == 0:
        raise ValueError("Calibration and validation partitions must both contain events")
    rng = np.random.default_rng(bootstrap_seed)
    layers = {}
    for volume, layer in parse_layers(target_layers):
        key = f"{volume}:{layer}"
        rows = calibration[(calibration["volume_id"] == volume) & (calibration["layer_id"] == layer)]
        validation_rows = validation[
            (validation["volume_id"] == volume) & (validation["layer_id"] == layer)
        ]
        if rows.empty:
            raise ValueError(f"Nominal calibration layer {key} is missing")
        if validation_rows.empty:
            raise ValueError(f"Nominal validation layer {key} is missing")
        layer_result = {}
        for component in ("LOC0", "LOC1"):
            sums, counts = _component_contributions(rows, volume, component)
            validation_sums, _validation_counts = _component_contributions(
                validation_rows, volume, component
            )
            predictive_sample_events = len(validation_sums)
            draws = rng.integers(
                0, len(sums), size=(n_bootstrap, predictive_sample_events)
            )
            sampled_sum = sums[draws].sum(axis=1)
            sampled_count = counts[draws].sum(axis=1)
            predictive = sampled_sum / sampled_count
            upper = float(np.quantile(predictive, 0.99, method="higher"))
            layer_result[component] = {
                "nominal_chi2": float(sums.sum() / counts.sum()),
                "predictive_upper_99": upper,
                "threshold": max(DEFAULT_THRESHOLD, upper),
                "n_components": int(counts.sum()),
                "predictive_sample_events": predictive_sample_events,
            }
        layers[key] = layer_result
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "event_bootstrap_predictive_upper_99",
        "event_split": {
            "seed": split_seed,
            "calibration_events": calibration_events,
            "validation_events": validation_events,
        },
        "bootstrap": {"seed": bootstrap_seed, "replicates": n_bootstrap},
        "layers": layers,
    }


def load_threshold_profile(path: str | Path, target_layers=TARGET_LAYERS):
    profile_path = Path(path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported threshold profile schema {profile.get('schema_version')!r}"
        )
    thresholds = {}
    for volume, layer in parse_layers(target_layers):
        key = f"{volume}:{layer}"
        try:
            thresholds[key] = {
                component: float(profile["layers"][key][component]["threshold"])
                for component in ("LOC0", "LOC1")
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Threshold profile is incomplete for {key}") from error
    return profile, thresholds


def _column_entries(values):
    if isinstance(values, ak.Array):
        return ak.to_list(values)
    return list(values)


def _flatten_entries(values):
    entries = _column_entries(values)
    if not entries:
        return np.asarray([], dtype=float)
    return np.concatenate([np.asarray(entry) for entry in entries])


def _expand_errors(local_x, errors, field_name):
    expanded = []
    for track_local, track_errors in zip(
        _column_entries(local_x), _column_entries(errors)
    ):
        local = np.asarray(track_local, dtype=float)
        values = np.asarray(track_errors, dtype=float)
        measurement_mask = np.isfinite(local)
        if len(values) == len(local):
            expanded.append(values)
        elif len(values) == int(measurement_mask.sum()):
            aligned = np.full(len(local), np.nan)
            aligned[measurement_mask] = values
            expanded.append(aligned)
        else:
            raise ValueError(
                f"{field_name} has {len(values)} entries for "
                f"{int(measurement_mask.sum())} measurements and {len(local)} states"
            )
    return np.concatenate(expanded) if expanded else np.asarray([], dtype=float)


def _expand_event_numbers(state_reference, event_numbers):
    counts = np.asarray([len(entry) for entry in _column_entries(state_reference)])
    events = np.asarray(_column_entries(event_numbers))
    if events.ndim == 1 and len(events) == len(counts):
        return np.repeat(events.astype(np.int64), counts)
    flattened = _flatten_entries(event_numbers).astype(np.int64)
    if len(flattened) != int(counts.sum()):
        raise ValueError("event_nr does not match track-level or state-level lengths")
    return flattened


def load_reconstruction_frame(path: str | Path, *, require_event_nr: bool = False) -> pd.DataFrame:
    path = Path(path)
    required = REQUIRED_FIELDS | ({"event_nr"} if require_event_nr else set())
    try:
        import ROOT
    except ImportError:
        ROOT = None
    if ROOT is not None:
        frame = ROOT.RDataFrame("trackstates", str(path))
        available = {str(name) for name in frame.GetColumnNames()}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"trackstates are missing required fields: {', '.join(missing)}"
            )
        try:
            columns = sorted(REQUIRED_FIELDS | ({"event_nr"} if "event_nr" in available else set()))
            arrays = frame.AsNumpy(columns)
        except Exception as error:
            raise ValueError(f"PyROOT could not read trackstates from {path}: {error}") from error
    else:
        with uproot.open(path) as root_file:
            if "trackstates" not in root_file:
                raise ValueError(f"ROOT tree 'trackstates' is missing from {path}")
            tree = root_file["trackstates"]
            available = set(tree.keys())
            missing = sorted(required - available)
            if missing:
                raise ValueError(
                    f"trackstates are missing required fields: {', '.join(missing)}"
                )
            columns = sorted(REQUIRED_FIELDS | ({"event_nr"} if "event_nr" in available else set()))
            arrays = tree.arrays(columns, library="ak", how=dict)
    local_x = arrays["l_x_hit"]
    data = {}
    for name in REQUIRED_FIELDS - {"err_x_hit", "err_y_hit"}:
        data[name] = _flatten_entries(arrays[name])
    data["err_x_hit"] = _expand_errors(local_x, arrays["err_x_hit"], "err_x_hit")
    data["err_y_hit"] = _expand_errors(local_x, arrays["err_y_hit"], "err_y_hit")
    if "event_nr" in arrays:
        data["event_nr"] = _expand_event_numbers(arrays["volume_id"], arrays["event_nr"])
    lengths = {name: len(values) for name, values in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Flattened trackstate field lengths differ: {lengths}")
    return pd.DataFrame(data)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--states", type=absolute_project_path, required=True)
    parser.add_argument("--output", type=absolute_project_path, required=True)
    parser.add_argument("--target-layers", default=target_layers_spec())
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--threshold-profile", type=absolute_project_path)
    parser.add_argument("--event-partition", choices=PARTITIONS)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        require_events = bool(args.event_partition or args.threshold_profile)
        frame = (
            load_reconstruction_frame(args.states, require_event_nr=True)
            if require_events
            else load_reconstruction_frame(args.states)
        )
        thresholds = None
        profile = None
        split_seed = args.split_seed
        if args.threshold_profile:
            profile, thresholds = load_threshold_profile(
                args.threshold_profile, parse_layers(args.target_layers)
            )
            split_seed = int(profile.get("event_split", {}).get("seed", split_seed))
        result = compute_reconstruction_acceptance(
            frame,
            parse_layers(args.target_layers),
            args.threshold,
            thresholds=thresholds,
            event_partition=args.event_partition,
            split_seed=split_seed,
        )
        result["threshold_profile"] = str(args.threshold_profile) if profile else None
    except Exception as error:
        targets = parse_layers(args.target_layers)
        failed_components = [
            f"{volume}:{layer}:{component}"
            for volume, layer in targets
            for component in ("LOC0", "LOC1")
        ]
        result = {
            "schema_version": SCHEMA_VERSION,
            "valid": False,
            "passed": False,
            "failed_layers": [f"{v}:{l}" for v, l in targets],
            "failed_components": failed_components,
            "acceptance_score": [len(failed_components), None, None],
            "error": str(error),
        }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
