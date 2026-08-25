import argparse
import os
import sys
from pathlib import Path

import numpy as np
import uproot
try:
    import ROOT
except ImportError:
    ROOT = None
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment_artifact_layout import artifact_filename, data_path
from project_config import absolute_project_path, add_config_argument


GEOMETRY_ORDER = ("Nominal", "Aligned", "Misaligned")
GEOMETRY_COLORS = {
    "Nominal": "tab:blue",
    "Aligned": "tab:green",
    "Misaligned": "tab:red",
}
DEFAULT_TRACKSUMMARY = artifact_filename("tracksummary_kf", ".root")
DEFAULT_DISPLAY_XMAX = None


def _flatten_to_numpy(array):
    import awkward as ak

    try:
        return ak.to_numpy(ak.flatten(array)).astype(np.float64)
    except Exception:
        return ak.to_numpy(array).astype(np.float64)


def chi2_ndf_values(chi2, ndf, label=""):
    chi2 = np.asarray(chi2, dtype=np.float64)
    ndf = np.asarray(ndf, dtype=np.float64)
    if chi2.shape != ndf.shape:
        raise ValueError(f"chi2 and NDF shapes differ: {chi2.shape} vs {ndf.shape}")

    mask = ndf > 0
    n_total = len(chi2)
    n_valid = int(mask.sum())
    n_dropped = n_total - n_valid
    if n_dropped:
        prefix = f"{label}: " if label else ""
        print(f"info: {prefix}dropped {n_dropped}/{n_total} tracks with NDF <= 0")

    values = chi2[mask] / ndf[mask]
    finite_mask = np.isfinite(values)
    n_nonfinite = len(values) - int(finite_mask.sum())
    if n_nonfinite:
        prefix = f"{label}: " if label else ""
        print(f"info: {prefix}dropped {n_nonfinite}/{len(values)} non-finite chi2/NDF values")

    return values[finite_mask]


def load_chi2_ndf(input_path: Path, label=""):
    if ROOT is not None:
        frame = ROOT.RDataFrame("tracksummary", str(input_path))
        available = {str(name) for name in frame.GetColumnNames()}
        missing = {"chi2Sum", "NDF"} - available
        if missing:
            sys.exit(
                f"error: branches {sorted(missing)} not found in {input_path}"
            )
        arrays = frame.AsNumpy(["chi2Sum", "NDF"])

        def flatten(values):
            entries = list(values)
            return np.concatenate([np.asarray(entry) for entry in entries])

        return chi2_ndf_values(
            flatten(arrays["chi2Sum"]), flatten(arrays["NDF"]), label=label
        )

    with uproot.open(input_path) as f:
        if "tracksummary" not in f:
            sys.exit(f"error: tree 'tracksummary' not found in {input_path}")
        tree = f["tracksummary"]
        if "chi2Sum" not in tree or "NDF" not in tree:
            sys.exit(f"error: branch 'chi2Sum' or 'NDF' not found in {input_path}")
        chi2 = tree["chi2Sum"].array(library="ak")
        ndf  = tree["NDF"].array(library="ak")

    chi2 = _flatten_to_numpy(chi2)
    ndf  = _flatten_to_numpy(ndf)
    return chi2_ndf_values(chi2, ndf, label=label)


def vol_lay_directory_name(file_name: str | Path):
    import re

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
        "Aligned": _tracksummary_path(recon_dir, "aligned", aligned_name),
        "Misaligned": _tracksummary_path(recon_dir, "misaligned", tracksummary_name),
    }


def load_comparison_data(files):
    data = {}
    for label in GEOMETRY_ORDER:
        path = files[label]
        if not path.exists():
            print(f"warning: {label} file not found: {path}")
            continue
        values = load_chi2_ndf(path, label=label)
        if len(values) == 0:
            print(f"warning: {label} has no valid chi2/NDF entries")
            continue
        data[label] = values
        print(
            f"{label}: loaded {len(values)} valid tracks"
            f"  |  mean = {np.mean(values):.3f}"
            f"  |  median = {np.median(values):.3f}"
        )
    return data


def _stats_line(label, values, n_nonpositive=0):
    mean = float(np.mean(values))
    std = float(np.std(values))
    median = float(np.median(values))
    line = (
        f"{label:<10} N={len(values):>6}  "
        f"Mean={mean:>9.3f}  Std={std:>9.3f}  "
        f"Median={median:>8.3f}  "
    )
    if n_nonpositive:
        line += f"N(<=0)={n_nonpositive:>6}  "
    return line


def plot_comparison(data_dict, output_path: Path, bins: int = 100, display_xmax: float = DEFAULT_DISPLAY_XMAX):
    data_dict = {
        label: np.asarray(values, dtype=np.float64)
        for label, values in data_dict.items()
        if values is not None and len(values) > 0
    }
    if not data_dict:
        sys.exit("error: no valid chi2/NDF values to plot")

    all_values = np.concatenate(list(data_dict.values()))
    finite_values = all_values[np.isfinite(all_values)]
    if len(finite_values) == 0:
        sys.exit("error: no finite chi2/NDF values to plot")

    # A log x-axis cannot represent chi2/NDF <= 0; count them explicitly per
    # geometry instead of dropping them silently.
    nonpositive_counts = {}
    for label, values in data_dict.items():
        n_nonpositive = int(np.sum(values <= 0))
        nonpositive_counts[label] = n_nonpositive
        if n_nonpositive:
            print(
                f"info: {label}: {n_nonpositive}/{len(values)} entries with "
                f"chi2/NDF <= 0 fall outside the log x-axis"
            )

    positive_values = finite_values[finite_values > 0]
    if len(positive_values) == 0:
        sys.exit("error: no positive chi2/NDF values to plot on a log x-axis")

    x_min = float(np.min(positive_values))
    if display_xmax is None:
        x_max = float(np.max(positive_values))
        if x_max <= x_min:
            # Degenerate data (all values identical): widen by one decade so
            # logspace still produces usable bin edges.
            x_min, x_max = x_min / np.sqrt(10.0), x_max * np.sqrt(10.0)
    else:
        if display_xmax <= 0:
            sys.exit(f"error: invalid display_xmax value: {display_xmax}")
        x_max = float(display_xmax)
        if x_max <= x_min:
            sys.exit(
                f"error: display_xmax {x_max} must exceed the smallest positive "
                f"chi2/NDF value {x_min}"
            )
        for label, values in data_dict.items():
            n_above = int(np.sum(values > x_max))
            if n_above:
                print(
                    f"info: {label}: {n_above}/{len(values)} entries above the "
                    f"requested display-xmax={x_max:g} are outside the plot"
                )

    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)
    # logspace round-trips through log10/power, so its endpoints need not
    # reproduce x_min/x_max exactly; pin them so the extreme values always land
    # inside a bin instead of being silently dropped.
    bin_edges[0] = x_min
    bin_edges[-1] = x_max

    fig, ax = plt.subplots(figsize=(10, 7))
    for label in GEOMETRY_ORDER:
        if label not in data_dict:
            continue
        values = data_dict[label]
        ax.hist(
            values,
            bins=bin_edges,
            histtype="step",
            fill=False,
            linewidth=2.0,
            color=GEOMETRY_COLORS.get(label, "black"),
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0.5)
    ax.set_xlabel(r"$\chi^2/\mathrm{ndf}$", fontsize=13)
    ax.set_ylabel("Entries", fontsize=13)
    ax.set_title(r"Track $\chi^2 / \mathrm{NDF}$ comparison", fontsize=14)
    ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(0.98, 0.92))

    stats_text = "\n".join(
        _stats_line(label, data_dict[label], nonpositive_counts.get(label, 0))
        for label in GEOMETRY_ORDER
        if label in data_dict
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot chi2/NDF comparison from nominal, aligned, and misaligned tracksummary ROOT files.")
    add_config_argument(parser)
    parser.add_argument("-o", "--output", type=absolute_project_path, required=True,
                        help="Path to output image file (e.g. chi2ndf.png)")
    parser.add_argument("--bins",  type=int,   default=100,
                        help="Number of logarithmic histogram bins (default: 100)")
    parser.add_argument("--display-xmax", type=float, default=DEFAULT_DISPLAY_XMAX,
                        help="Optional override for the log x-axis upper bound "
                             "(default: largest positive value in the data)")
    parser.add_argument("--recon-dir", type=absolute_project_path, default=str(data_path("reconstruction_output")),
                        help="Reconstruction output base directory for comparison mode")
    parser.add_argument("--tracksummary-name", type=str, default=DEFAULT_TRACKSUMMARY,
                        help=f"Tracksummary file name for comparison mode (default: {DEFAULT_TRACKSUMMARY})")
    parser.add_argument("--aligned-tracksummary", type=absolute_project_path, default=None,
                        help="Path to the aligned tracksummary ROOT file")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    
    output_path = Path(args.output)

    if args.bins <= 0:
        sys.exit(f"error: invalid bins value: {args.bins}")
    if args.display_xmax is not None and args.display_xmax <= 0:
        sys.exit(f"error: invalid display-xmax value: {args.display_xmax}")

    try:
        files = comparison_files(
            Path(args.recon_dir),
            args.tracksummary_name,
            aligned_tracksummary=args.aligned_tracksummary,
        )
    except ValueError as err:
        sys.exit(f"error: {err}")
    data = load_comparison_data(files)
    plot_comparison(data, output_path, bins=args.bins, display_xmax=args.display_xmax)


if __name__ == "__main__":
    main()
