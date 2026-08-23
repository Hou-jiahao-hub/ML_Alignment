import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import uproot

try:
    import ROOT

    ROOT.gROOT.SetBatch(True)
except ImportError:
    ROOT = None

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from alignment_artifact_layout import TARGET_LAYER_DIR, TARGET_LAYER_SUFFIX, artifact_filename, data_path
from project_config import absolute_project_path, add_config_argument


GEOMETRY_ORDER = ("Nominal", "Misaligned", "Aligned")
GEOMETRY_COLORS = {
    "Nominal": "blue",
    "Misaligned": "red",
    "Aligned": "green",
}
DEFAULT_TRACKSUMMARY = artifact_filename("tracksummary_kf", ".root")


def _flatten_to_numpy(array):
    import awkward as ak

    try:
        values = ak.to_numpy(ak.flatten(array, axis=None))
    except Exception:
        values = ak.to_numpy(array)
    return np.asarray(values)


def hit_number_values(n_measurements, label=""):
    values = _flatten_to_numpy(n_measurements).astype(np.float64)
    finite_mask = np.isfinite(values)
    n_nonfinite = len(values) - int(finite_mask.sum())
    if n_nonfinite:
        prefix = f"{label}: " if label else ""
        print(f"info: {prefix}dropped {n_nonfinite}/{len(values)} non-finite hit number values")

    values = values[finite_mask]
    integer_values = values.astype(np.int64)
    return integer_values


def _load_hit_numbers_pyroot(input_path: Path, label=""):
    root_file = ROOT.TFile.Open(str(input_path))
    if not root_file or root_file.IsZombie():
        sys.exit(f"error: cannot open ROOT file {input_path}")

    try:
        tree = root_file.Get("tracksummary")
        if not tree:
            sys.exit(f"error: tree 'tracksummary' not found in {input_path}")
        if not tree.GetBranch("nMeasurements"):
            sys.exit(f"error: branch 'nMeasurements' not found in {input_path}")

        values = []
        for entry in tree:
            values.extend(int(value) for value in entry.nMeasurements)
        return hit_number_values(np.asarray(values, dtype=np.int64), label=label)
    finally:
        root_file.Close()


def _load_hit_numbers_uproot(input_path: Path, label=""):
    with uproot.open(input_path) as root_file:
        if "tracksummary" not in root_file:
            sys.exit(f"error: tree 'tracksummary' not found in {input_path}")
        tree = root_file["tracksummary"]
        if "nMeasurements" not in tree:
            sys.exit(f"error: branch 'nMeasurements' not found in {input_path}")
        n_measurements = tree["nMeasurements"].array(library="ak")

    return hit_number_values(n_measurements, label=label)


def load_hit_numbers(input_path: Path, label=""):
    if ROOT is not None:
        return _load_hit_numbers_pyroot(input_path, label=label)
    return _load_hit_numbers_uproot(input_path, label=label)


def vol_lay_directory_name(file_name: str | Path):
    pairs = re.findall(r"v(\d+)_?l(\d+)", Path(file_name).name)
    if not pairs:
        raise ValueError(f"Cannot infer vol/lay directory from file name: {file_name}")
    return "__".join(f"Vol{vol}_Lay{lay}" for vol, lay in pairs)


def _tracksummary_path(recon_dir: Path, mode: str, file_name: str | Path):
    path = Path(file_name)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return recon_dir / mode / path
    return recon_dir / mode / vol_lay_directory_name(path.name) / path


def comparison_files(recon_dir: Path, tracksummary_name: str, aligned_tracksummary=None):
    aligned_name = aligned_tracksummary or tracksummary_name
    return {
        "Nominal": _tracksummary_path(recon_dir, "nominal", tracksummary_name),
        "Misaligned": _tracksummary_path(recon_dir, "misaligned", tracksummary_name),
        "Aligned": _tracksummary_path(recon_dir, "aligned", aligned_name),
    }


def load_comparison_data(files):
    data = {}
    for label in GEOMETRY_ORDER:
        path = files[label]
        if not path.exists():
            print(f"warning: {label} file not found: {path}")
            continue
        values = load_hit_numbers(path, label=label)
        if len(values) == 0:
            print(f"warning: {label} has no valid hit number entries")
            continue
        data[label] = values
        print(
            f"{label}: loaded {len(values)} tracks"
            f"  |  mean hits = {np.mean(values):.3f}"
            f"  |  median hits = {np.median(values):.3f}"
        )
    return data


def _auto_range(data_dict):
    all_values = np.concatenate(
        [np.asarray(values, dtype=np.float64) for values in data_dict.values() if len(values) > 0]
    )
    if len(all_values) == 0:
        sys.exit("error: no hit number values to plot")
    low = int(np.floor(np.min(all_values))) - 1
    high = int(np.ceil(np.max(all_values))) + 1
    if low == high:
        high = low + 1
    return (low, high)


def plot_comparison(data_dict, output_path: Path, bins: int = 16, x_range=(6, 22)):
    data_dict = {
        label: np.asarray(values, dtype=np.int64)
        for label, values in data_dict.items()
        if values is not None and len(values) > 0
    }
    if not data_dict:
        sys.exit("error: no valid hit number values to plot")

    if x_range is None:
        x_range = _auto_range(data_dict)

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_labels = []
    for label in GEOMETRY_ORDER:
        if label not in data_dict:
            continue
        plotted_labels.append(label)
        ax.hist(
            data_dict[label],
            bins=bins,
            range=x_range,
            histtype="step",
            fill=False,
            linewidth=1.8,
            color=GEOMETRY_COLORS.get(label, "black"),
            label=label,
        )

    ax.set_xlabel(r"$N_{\mathrm{hits}}$", fontsize=13)
    ax.set_ylabel("Entries", fontsize=13)
    ax.set_xlim(*x_range)
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both", top=True, right=True)
    legend_handles = [
        Line2D([0], [0], color=GEOMETRY_COLORS.get(label, "black"), linewidth=1.8, label=label)
        for label in plotted_labels
    ]
    ax.legend(
        handles=legend_handles,
        title="ACTS OpenDataDetector",
        frameon=False,
        fontsize=10,
        title_fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot reconstructed hit number distributions for nominal, misaligned, and aligned geometries."
    )
    add_config_argument(parser)
    parser.add_argument(
        "-o",
        "--output",
        type=absolute_project_path,
        default=str(data_path("reconstruction_output/hit_number") / TARGET_LAYER_DIR / f"hit_number_{TARGET_LAYER_SUFFIX}.png"),
        help="Path to output image file",
    )
    parser.add_argument("--bins", type=int, default=16, help="Number of histogram bins")
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=(6, 22),
        metavar=("MIN", "MAX"),
        help="Hit-number plot range (default: 6 22). Use --auto-range for data-driven range.",
    )
    parser.add_argument(
        "--auto-range",
        action="store_true",
        help="Use the finite data range instead of --range",
    )
    parser.add_argument(
        "--recon-dir",
        type=absolute_project_path,
        default=str(data_path("reconstruction_output")),
        help="Reconstruction output base directory",
    )
    parser.add_argument(
        "--tracksummary-name",
        type=str,
        default=DEFAULT_TRACKSUMMARY,
        help=f"Tracksummary file name for nominal/misaligned comparison (default: {DEFAULT_TRACKSUMMARY})",
    )
    parser.add_argument(
        "--aligned-tracksummary",
        type=absolute_project_path,
        default=None,
        help="Path to the aligned tracksummary ROOT file",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.bins <= 0:
        sys.exit(f"error: invalid bins value: {args.bins}")
    if args.range[0] >= args.range[1]:
        sys.exit(f"error: invalid plot range: {args.range} (min >= max)")

    try:
        files = comparison_files(
            Path(args.recon_dir),
            args.tracksummary_name,
            aligned_tracksummary=args.aligned_tracksummary,
        )
    except ValueError as err:
        sys.exit(f"error: {err}")

    data = load_comparison_data(files)
    x_range = None if args.auto_range else tuple(args.range)
    plot_comparison(data, Path(args.output), bins=args.bins, x_range=x_range)


if __name__ == "__main__":
    main()
