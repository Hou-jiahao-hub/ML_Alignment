#!/usr/bin/env python
"""Post-hoc: is the ACTS vs Schur geometry RMS gap carried by low-hit sensors?

Reads the per-sensor CSVs written by compare_pt100_alignment.py and the target
hit count per sensor from the misaligned KF trackstates. Not preregistered.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
DOFS = ("tx", "ty", "rz")


def tail_rows(per_sensor, hits, low_quantile=0.1):
    """``per_sensor[(vol, lay, sen, dof)] = (acts_abs, schur_abs)`` in mm/rad -> μm/mrad stats."""
    rows = []
    layers = sorted({key[:2] for key in per_sensor})
    for layer in layers:
        sensors = sorted({key[2] for key in per_sensor if key[:2] == layer})
        counts = sorted(hits.get((*layer, s), 0) for s in sensors)
        cut = counts[max(int(math.floor(low_quantile * len(counts))) - 1, 0)]
        for dof in DOFS:
            pairs = [(1e3 * per_sensor[(*layer, s, dof)][0], 1e3 * per_sensor[(*layer, s, dof)][1],
                      hits.get((*layer, s), 0)) for s in sensors]
            kept = [(a, b) for a, b, n in pairs if n > cut]

            def rms(values):
                return math.sqrt(sum(v * v for v in values) / len(values))

            def median(values):
                values = sorted(values)
                mid = len(values) // 2
                return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])

            rows.append({
                "surface": f"{layer[0]}:{layer[1]}", "dof": dof,
                "sensors": len(pairs), "hits_median": median([n for *_, n in pairs]),
                "hits_cut": cut, "sensors_kept": len(kept),
                "acts_rms": rms([a for a, _, _ in pairs]), "schur_rms": rms([b for _, b, _ in pairs]),
                "acts_median": median([a for a, _, _ in pairs]),
                "schur_median": median([b for _, b, _ in pairs]),
                "acts_max": max(a for a, _, _ in pairs), "schur_max": max(b for _, b, _ in pairs),
                "acts_rms_kept": rms([a for a, _ in kept]), "schur_rms_kept": rms([b for _, b in kept]),
                "schur_better_fraction": sum(b < a for a, b, _ in pairs) / len(pairs),
            })
    return rows


def load_per_sensor(acts_csv, schur_csv):
    def read(path):
        with Path(path).open() as stream:
            return {(int(r["volume_id"]), int(r["layer_id"]), int(r["sensor_id"]), r["dof"]):
                    float(r["aligned_abs_difference"]) for r in csv.DictReader(stream)
                    if r["dof"] in DOFS}
    acts, schur = read(acts_csv), read(schur_csv)
    if acts.keys() != schur.keys():
        raise ValueError("ACTS and Schur CSVs cover different sensors")
    return {key: (acts[key], schur[key]) for key in acts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    import awkward as ak
    import uproot

    import pt100_alignment_layout as pt100
    from alignment_artifact_layout import TARGET_LAYER_DIR, TARGET_LAYERS, artifact_filename
    from experiment_manifest import finalize_experiment_manifest, write_experiment_manifest
    from project_config import load_project_config

    config = load_project_config()
    run_id = args.run_id or pt100.PT100_ALIGNMENT_RUN
    paths = pt100.pt100_alignment_paths(config.data_root, TARGET_LAYER_DIR, run_id)
    out = paths["experiment"] / "sensor_tail"
    if out.exists():
        raise FileExistsError(f"refusing to reuse output: {out}")
    inputs = {m: paths["comparison"] / f"geometry_{m}" / "geometry_comparison.csv"
              for m in ("acts", "schur")}
    inputs["trackstates"] = paths["misaligned"] / artifact_filename("trackstates_kf", ".root")
    out.mkdir(parents=True)
    manifest = write_experiment_manifest(
        out, run_id=f"{run_id}/sensor_tail", command=[sys.executable, *sys.argv],
        config=config, inputs=inputs, metrics={"preregistered": False})
    try:
        tree = uproot.open(inputs["trackstates"])["trackstates"]
        arrays = tree.arrays(["volume_id", "layer_id", "module_id"])
        flat = [ak.flatten(arrays[name]).to_numpy().tolist()
                for name in ("volume_id", "layer_id", "module_id")]
        targets = {tuple(layer) for layer in TARGET_LAYERS}
        hits = collections.Counter(key for key in zip(*flat) if key[:2] in targets)
        rows = tail_rows(load_per_sensor(inputs["acts"], inputs["schur"]), hits)
        (out / "sensor_tail.json").write_text(json.dumps(
            {"hit_source": "misaligned KF trackstates, all target states", "rows": rows},
            indent=2) + "\n")
        finalize_experiment_manifest(manifest, status="completed",
                                     outputs={"sensor_tail": out / "sensor_tail.json"},
                                     metrics={"preregistered": False, "rows": rows})
    except BaseException as error:
        finalize_experiment_manifest(manifest, status="failed", metrics={"error": str(error)})
        raise
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
