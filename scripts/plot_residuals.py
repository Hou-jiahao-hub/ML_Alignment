import uproot
import awkward as ak
import numpy as np
import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment_artifact_layout import TARGET_LAYERS, artifact_filename, data_path
from project_config import absolute_project_path, add_config_argument

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.StatOverflows(True)
except ImportError:
    ROOT = None

ROOT_COLORS = {"Nominal": ROOT.kBlue, "Aligned": ROOT.kGreen+2, "Misaligned": ROOT.kRed} if ROOT else {}
PLOT_COLORS = {"Nominal": "blue", "Aligned": "green", "Misaligned": "red"}
GEOMETRY_ORDER = ("Nominal", "Aligned", "Misaligned")

DEFAULT_TRACKSTATES = artifact_filename("trackstates_kf", ".root")

# ODD subsystem definitions: volume IDs, intrinsic resolutions [mm], and default plot ranges [mm]
SUBSYSTEMS = {
    "Pixel": {
        "volumes": [16, 17, 18],
        "res_x": 0.015, "res_y": 0.015,
        "range_x": (-0.15, 0.15), "range_y": (-0.15, 0.15),
    },
    "ShortStrip": {
        "volumes": [23, 24, 25],
        "res_x": 0.043, "res_y": 1.200,
        "range_x": (-0.30, 0.30), "range_y": (-5.0, 5.0),
    },
    "LongStrip": {
        "volumes": [28, 29, 30],
        "res_x": 0.072, "res_y": None,
        "range_x": (-0.50, 0.50), "range_y": (-5.0, 5.0),
    },
}


def get_subsystem_config(volume_id):
    for sub_cfg in SUBSYSTEMS.values():
        if volume_id in sub_cfg["volumes"]:
            return sub_cfg
    raise ValueError(f"No subsystem configuration found for volume {volume_id}")


def vol_lay_directory_name(file_name: str):
    pairs = re.findall(r"v(\d+)_?l(\d+)", Path(file_name).name)
    if not pairs:
        raise ValueError(f"Cannot infer vol/lay directory from file name: {file_name}")
    return "__".join(f"Vol{vol}_Lay{lay}" for vol, lay in pairs)


def _trackstates_path(recon_dir: Path, mode: str, file_name: str):
    path = Path(file_name)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return recon_dir / mode / path
    return recon_dir / mode / vol_lay_directory_name(file_name) / file_name


def comparison_files(recon_dir: Path, trackstates_name: str, aligned_trackstates=None):
    aligned_name = aligned_trackstates or trackstates_name
    return {
        "Nominal": _trackstates_path(recon_dir, "nominal", trackstates_name),
        "Aligned": _trackstates_path(recon_dir, "aligned", aligned_name),
        "Misaligned": _trackstates_path(recon_dir, "misaligned", trackstates_name),
    }


def get_branch_data(file_path: Path, tree_name: str, branch_meas: str, branch_pred: str, vol_ids=None, lay_id=None):
    """Read residual data from a ROOT tree, optionally filtered by one or more volume IDs."""
    if not file_path.exists():
        print(f"warning: No file found at {file_path}")
        return None

    try:
        if ROOT is not None:
            branches = [branch_meas, branch_pred]
            if vol_ids is not None:
                branches.append("volume_id")
            if lay_id is not None:
                branches.append("layer_id")
            frame = ROOT.RDataFrame(tree_name, str(file_path))
            available = {str(name) for name in frame.GetColumnNames()}
            missing = [name for name in branches if name not in available]
            if missing:
                print(
                    f"error: Branches {missing} not found in tree "
                    f"'{tree_name}'"
                )
                return None
            arrays = frame.AsNumpy(branches)

            def flatten(values):
                entries = list(values)
                return np.concatenate([np.asarray(entry) for entry in entries])

            data = flatten(arrays[branch_meas]) - flatten(arrays[branch_pred])
            mask = np.ones(len(data), dtype=bool)
            if vol_ids is not None:
                volume = flatten(arrays["volume_id"])
                volume_ids = [vol_ids] if isinstance(vol_ids, int) else list(vol_ids)
                mask &= np.isin(volume, volume_ids)
            if lay_id is not None:
                mask &= flatten(arrays["layer_id"]) == lay_id
            selected = data[mask]
            return selected[np.isfinite(selected)]

        with uproot.open(file_path) as file:
            if tree_name not in file:
                print(f"error: Tree '{tree_name}' not found in {file_path}")
                return None

            tree = file[tree_name]
            if branch_meas not in tree or branch_pred not in tree:
                print(f"error: Branch '{branch_meas}' or '{branch_pred}' not found in tree '{tree_name}'")
                return None

            data_meas = tree[branch_meas].array(library="ak")
            data_pred = tree[branch_pred].array(library="ak")
            data = data_meas - data_pred

            mask = None
            if vol_ids is not None:
                vol_ids_list = [vol_ids] if isinstance(vol_ids, int) else list(vol_ids)
                if "volume_id" in tree:
                    v_arr = tree["volume_id"].array(library="ak")
                    vol_mask = (v_arr == vol_ids_list[0])
                    for vid in vol_ids_list[1:]:
                        vol_mask = vol_mask | (v_arr == vid)
                    mask = vol_mask
                else:
                    print("Warning: 'volume_id' branch not found. Ignoring volume filter.")

            if lay_id is not None:
                if "layer_id" in tree:
                    l_arr = tree["layer_id"].array(library="ak")
                    mask = (l_arr == lay_id) if mask is None else mask & (l_arr == lay_id)
                else:
                    print("Warning: 'layer_id' branch not found. Ignoring layer filter.")

            if mask is not None:
                data = data[mask]

            try:
                flat_data = ak.flatten(data).to_numpy()
            except Exception:
                flat_data = ak.to_numpy(data)

            return flat_data[np.isfinite(flat_data)]

    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None


def write_residual_png(
    data_dict,
    output_png: Path,
    var_name,
    layer_tag="",
    units="mm",
    plot_range=(-1.0, 1.0),
    bins=100,
    odd_resolution=None,
    output_label="",
):
    plotted = {}
    for label in GEOMETRY_ORDER:
        data = data_dict.get(label)
        if data is None:
            continue
        values = np.asarray(data, dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        counts, bin_edges = np.histogram(values, bins=bins, range=plot_range)
        if np.all(counts == 0):
            continue
        plotted[label] = (values, counts, bin_edges)

    if not plotted:
        print(f"Warning: No valid residual data for PNG output: {output_png.name}")
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for label in GEOMETRY_ORDER:
            if label not in plotted:
                continue
            values, counts, bin_edges = plotted[label]
            legend_label = (
                f"{label}\nMean: {np.mean(values):.4f} {units}, "
                f"Std: {np.std(values):.4f} {units}"
            )
            ax.stairs(
                counts,
                bin_edges,
                color=PLOT_COLORS.get(label, "black"),
                linewidth=2.0,
                label=legend_label,
            )

        if odd_resolution is not None:
            reference_label = f"± ODD σ = {odd_resolution * 1e3:.1f} μm"
            ax.axvline(
                odd_resolution,
                color="violet",
                linestyle="--",
                linewidth=1.8,
                label=reference_label,
            )
            ax.axvline(
                -odd_resolution,
                color="violet",
                linestyle="--",
                linewidth=1.8,
            )

        label_suffix = f"-{output_label.upper()}" if output_label else ""
        layer_suffix = f"_{layer_tag}" if layer_tag else ""
        ax.set_title(f"({var_name}{layer_suffix}) Comparison{label_suffix}")
        ax.set_xlabel(f"Residual [{units}]")
        ax.set_ylabel("Number of hits")
        ax.set_xlim(*plot_range)
        ax.minorticks_on()
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=9, loc="upper right")

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    print(f" saved PNG: {output_png}")
    return True


def histograms(data_dict, base_dir, var_name, layer_tag="", units="mm", plot_range=(-1.0, 1.0), bins=100, odd_resolution=None, output_label=""):
    base_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = f"{layer_tag}" if layer_tag else ""
    label_prefix = f"{output_label}_" if output_label else ""
    output_root = base_dir / f"{label_prefix}{var_name}_{file_suffix}_comparison.root"
    output_png = output_root.with_suffix(".png")
    print(f"\n Generating ROOT and PNG files: {output_root.stem}")

    try:
        f = ROOT.TFile(str(output_root), "RECREATE")
        c = ROOT.TCanvas(f"canvas_{var_name}", f"Overlay of {var_name}", 800, 600)
        c.cd()

        label_suffix = f"-{output_label.upper()}" if output_label else ""
        stack_title = f"({var_name}_{file_suffix}) Comparison{label_suffix};Residual [{units}];Number of hits"
        hs = ROOT.THStack(f"hs_{var_name}", stack_title)
        hists = {}
        # fitted_funcs = {}

        for label, data in data_dict.items():
            if data is None or len(data) == 0:
                continue
            mean = np.mean(data)
            std  = np.std(data)
            counts, bin_edges = np.histogram(data, bins=bins, range=plot_range)

            if np.all(counts == 0):
                print(f"Warning: {label} histogram all 0, check plot range!")
                continue

            hname  = f"hist_{label.lower()}"
            htitle = f"{label} (Mean: {mean:.4f} {units}, Std: {std:.4f} {units})"

            h = ROOT.TH1D(hname, htitle, bins, plot_range[0], plot_range[1])
            data_f  = np.ascontiguousarray(data, dtype=np.float64)
            weights = np.ones(len(data_f), dtype=np.float64)
            h.FillN(len(data_f), data_f, weights)
            h.SetLineColor(ROOT_COLORS.get(label, ROOT.kBlack))
            h.SetLineWidth(3)

            # # Gaussian fit via ROOT TF1
            # # "QSN": Q=quiet, S=store result, N=do NOT auto-draw (we draw explicitly later)
            # gaus_func = ROOT.TF1(f"gaus_fit_{hname}", "gaus", plot_range[0], plot_range[1])
            # try:
            #     fit_ptr = h.Fit(gaus_func, "QSN")
            #     if fit_ptr.IsValid():
            #         mu  = gaus_func.GetParameter(1)
            #         sig = abs(gaus_func.GetParameter(2))
            #         print(f"    Gauss fit [{label}]: μ={mu*1e3:.3f} μm, σ={sig*1e3:.3f} μm")
            #         gaus_func.SetLineColor(ROOT_COLORS.get(label, ROOT.kBlack))
            #         gaus_func.SetLineWidth(2)
            #         gaus_func.SetLineStyle(1)
            #         fitted_funcs[label] = gaus_func
            #         gaus_func.Write()
            # except Exception as fit_err:
            #     print(f"    Gauss fit failed for {label}: {fit_err}")

            hists[label] = h
            h.Write()

        if not hists:
            print("Error: No valid histograms to draw!")
            f.Close()
            return

        # ── Draw histograms (step style) ──────────────────────────────────
        leg = ROOT.TLegend(0.55, 0.62, 0.90, 0.90)
        leg.SetBorderSize(1)
        leg.SetTextSize(0.028)

        for label in ["Nominal", "Aligned", "Misaligned"]:
            if label in hists:
                hs.Add(hists[label])
                leg.AddEntry(hists[label], label, "l")

        hs.Draw("HIST NOSTACK")

        # # ── Overlay Gaussian fit curves (explicit Draw after THStack) ─────
        # for label in ["Nominal", "Aligned", "Misaligned"]:
        #     if label in fitted_funcs:
        #         func = fitted_funcs[label]
        #         mu   = func.GetParameter(1)
        #         sig  = abs(func.GetParameter(2))
        #         func.Draw("SAME")
        #         leg.AddEntry(func,
        #                      f"Gauss {label}: #mu={mu*1e3:.2f} #mum, #sigma={sig*1e3:.2f} #mum",
        #                      "l")

        # ── ODD reference lines ───────────────────────────────────────────
        ref_lines = []
        if odd_resolution is not None:
            y_max = hs.GetMaximum() * 1.05
            for x_val in (odd_resolution, -odd_resolution):
                line = ROOT.TLine(x_val, 0, x_val, y_max)
                line.SetLineColor(ROOT.kViolet)
                line.SetLineStyle(9)
                line.SetLineWidth(2)
                line.Draw("SAME")
                ref_lines.append(line)
            leg.AddEntry(ref_lines[0], f"#pm ODD #sigma = {odd_resolution*1e3:.1f} #mum", "l")

        leg.Draw("SAME")
        c.Modified()
        c.Update()
        c.Write()
        hs.Write()
        f.Close()

    except (ImportError, AttributeError):
        print("  -> Warning: PyROOT not found. Falling back to basic uproot writing.")
        with uproot.recreate(output_root) as root_file:
            for label, data in data_dict.items():
                if data is None or len(data) == 0:
                    continue
                counts, bin_edges = np.histogram(data, bins=bins, range=plot_range)
                root_file[f"hist_{label.lower()}"] = (counts, bin_edges, f"{label};Residual [{units}];Hits")

    write_residual_png(
        data_dict,
        output_png,
        var_name=var_name,
        layer_tag=layer_tag,
        units=units,
        plot_range=plot_range,
        bins=bins,
        odd_resolution=odd_resolution,
        output_label=output_label,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save track residual distributions to ROOT and PNG files for the configured ODD target layers.")
    add_config_argument(parser)
    parser.add_argument("--recon-dir", type=absolute_project_path, default=str(data_path("reconstruction_output")),
                        help="Reconstruction output base directory")
    parser.add_argument("--residuals", nargs="+", default=["X", "Y"],
                        help="Residual axes to plot (X, Y, or both)")
    parser.add_argument("--range", type=float, nargs=2, default=None,
                        help="Override plot range [mm]: min max. Default: per-target-layer subsystem config.")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--volume_id", type=int, default=None,
                        help="Single Volume ID (manual mode, bypasses default target-layer loop)")
    parser.add_argument("--layer_id", type=int, default=None, help="Layer ID filter")
    parser.add_argument("--odd-res-x", type=float, default=None,
                        help="ODD resolution in local X [mm] (manual mode only)")
    parser.add_argument("--odd-res-y", type=float, default=None,
                        help="ODD resolution in local Y [mm] (manual mode only)")
    parser.add_argument("--aligned-trackstates", type=absolute_project_path, default=None,
                        help="Path to ML-aligned trackstates ROOT file (overrides default Aligned entry)")
    parser.add_argument("--label", type=str, default="",
                        help="Label appended to output file name and title (e.g. 'ml' → ml_res_X_..._comparison.root, title Comparison-ML)")
    parser.add_argument("--output-dir", type=absolute_project_path, default=None,
                        help="Directory to save output ROOT and PNG files (default: same as --recon-dir)")

    return parser


def main():
    parser = build_argument_parser()

    args = parser.parse_args()

    if args.bins <= 0:
        raise ValueError(f"Invalid bins value: {args.bins}")
    if args.range is not None and args.range[0] >= args.range[1]:
        raise ValueError(f"Invalid plot range: {args.range} (min >= max)")

    base_dir   = Path(args.recon_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir
    tree_name  = "trackstates"

    files = comparison_files(
        base_dir,
        DEFAULT_TRACKSTATES,
        aligned_trackstates=args.aligned_trackstates,
    )

    def _run_axis(axis, vol_ids, lay_id, layer_tag, plot_range, odd_res):
        axis = axis.upper()
        if axis == "X":
            b_meas, b_pred, var_name = "l_x_hit", "t_eLOC0", "res_X"
        elif axis == "Y":
            b_meas, b_pred, var_name = "l_y_hit", "t_eLOC1", "res_Y"
        else:
            print(f"Warning: Unknown axis '{axis}'. Skipping.")
            return

        ref_str = f"{odd_res*1e3:.0f} μm" if odd_res is not None else "N/A"
        print(f"\n  [{axis}]  ODD σ_ref = {ref_str},  range = {plot_range}")

        data_dict = {}
        for label, path in files.items():
            data = get_branch_data(path, tree_name, b_meas, b_pred,
                                   vol_ids=vol_ids, lay_id=lay_id)
            data_dict[label] = data
            if data is not None:
                print(f"   - {label}: {len(data)} valid hits")

        histograms(data_dict, base_dir=output_dir, var_name=var_name,
                   layer_tag=layer_tag, plot_range=plot_range,
                   bins=args.bins, odd_resolution=odd_res, output_label=args.label)

    if args.volume_id is not None:
        # ── Manual single-volume mode ─────────────────────────────────────
        lay_suffix = f"_Lay{args.layer_id}" if args.layer_id is not None else ""
        layer_tag = f"Vol{args.volume_id}{lay_suffix}"
        print(f"==> Manual mode: Volume {args.volume_id}" +
              (f", Layer {args.layer_id}" if args.layer_id else ""))
        for axis in args.residuals:
            plot_range = tuple(args.range) if args.range else (-0.5, 0.5)
            odd_res = args.odd_res_x if axis.upper() == "X" else args.odd_res_y
            _run_axis(axis, args.volume_id, args.layer_id, layer_tag, plot_range, odd_res)

    else:
        # ── Default target-layer mode ────────────────────────────────────
        for vol_id, lay_id in TARGET_LAYERS:
            sub_cfg = get_subsystem_config(vol_id)
            print(f"\n{'='*62}")
            print(f"  Target layer: Volume {vol_id}, Layer {lay_id}")
            print(f"{'='*62}")
            layer_tag = f"Vol{vol_id}_Lay{lay_id}"
            for axis in args.residuals:
                if axis.upper() == "X":
                    odd_res    = sub_cfg["res_x"]
                    plot_range = tuple(args.range) if args.range else sub_cfg["range_x"]
                else:
                    odd_res    = sub_cfg["res_y"]
                    plot_range = tuple(args.range) if args.range else sub_cfg["range_y"]
                _run_axis(axis, vol_id, lay_id, layer_tag, plot_range, odd_res)

    print(f"\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
