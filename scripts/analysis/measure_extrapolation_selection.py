"""Iteration 22 (P3): what the extrapolation floor actually selects.

归档说明：一次性分析脚本，未按产品标准编写，无测试覆盖，不被任何产品路径导入。
保留在库内只为让 Iteration 22 的证伪与选值可以被重新核对；改动产品代码时无需同步
维护它。产物在 data/experiments/joint_downstream/Vol17_Lay8__Vol24_Lay4/p3-20260827/。

三个 mode，只跑选择阶段，从不拟合、从不读效果指标：

  hits      各 min_extrapolation_fit_hits 阈值下每组的分量数。证伪抬高 hits 阈值：
            floor=6 让 17:8 forward 从 5034 掉到 42，击穿 MIN_GROUP_COMPONENTS。
  residuals 当前 floor 下每行窗口内的有效残差分量数 sum(1 + loc1_valid)。
            全部 47809 个外推行里只有 3 行欠定，全在 24:4 backward。
  verify    生产残差下限相对判据关闭时的增量，即 Iteration 22 §1.3 的核对块。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validate_helix_model as V

HIT_FLOORS = (4, 5, 6, 7, 8)
N_PARAMETERS = 5  # gauge-fixed reference helix, single arm


def load(states):
    config = V.HelixValidationConfig(states=states) if states else V.HelixValidationConfig()
    holdout_layers = V.parse_layers(config.holdout_layers)
    geometry = V.load_working_geometry(config.geometry, config.nominal_full_geometry)
    print(f"loading all nominal SMT states from {config.states} ...", flush=True)
    inputs = V.load_nominal_states(
        config.states, geometry.id_to_idx, max_track_entries=None
    )
    print(f"loaded {len(inputs.hit_sensor_ids)} hits", flush=True)
    return config, holdout_layers, inputs


def batches_for(config, holdout_layers, inputs, *, hit_floor, residual_floor):
    return V.build_track_batches(
        inputs,
        holdout_layers,
        min_fit_hits=config.min_fit_hits,
        min_fit_span=config.min_fit_span,
        max_tracks=0,
        context_hits_before=config.context_hits_before,
        context_hits_after=config.context_hits_after,
        min_extrapolation_fit_hits=hit_floor,
        min_extrapolation_fit_span=config.min_extrapolation_fit_span,
        min_extrapolation_fit_residuals=residual_floor,
    )


def group_labels(batches):
    holdout = batches.holdout_mask
    row = torch.arange(holdout.shape[0])
    col = holdout.float().argmax(dim=1)
    volume = batches.volume_id[row, col].numpy()
    layer = batches.layer_id[row, col].numpy()
    modes = np.array(batches.prediction_mode)
    return np.array(
        [f"{v}:{l}|{m}" for v, l, m in zip(volume.tolist(), layer.tolist(), modes.tolist())]
    )


def counts_by_group(batches):
    labels = group_labels(batches)
    return {label: int((labels == label).sum()) for label in sorted(set(labels))}


def mode_hits(config, holdout_layers, inputs):
    results = {
        floor: counts_by_group(
            batches_for(config, holdout_layers, inputs, hit_floor=floor, residual_floor=1)
        )
        for floor in HIT_FLOORS
    }
    print()
    print("=" * 78)
    print(f"LOC0 components per group; MIN_GROUP_COMPONENTS = {V.MIN_GROUP_COMPONENTS}")
    print("=" * 78)
    print(f"{'group':<42}" + "".join(f"{floor:>7}" for floor in HIT_FLOORS))
    for label in sorted(results[HIT_FLOORS[0]]):
        cells = "".join(f"{results[floor].get(label, 0):>7}" for floor in HIT_FLOORS)
        print(f"{label:<42}{cells}")


def mode_residuals(config, holdout_layers, inputs):
    batches = batches_for(
        config, holdout_layers, inputs,
        hit_floor=config.min_extrapolation_fit_hits, residual_floor=1,
    )
    fit = batches.fit_mask
    residuals = (fit.long() + (fit & batches.loc1_valid).long()).sum(dim=1).numpy()
    labels = group_labels(batches)

    print()
    print("=" * 78)
    print(f"residual components per fit window at a hit floor of "
          f"{config.min_extrapolation_fit_hits}; {N_PARAMETERS} gauge-fixed parameters")
    print("=" * 78)
    print(f"{'group':<42}{'rows':>8}{'p50':>6}{'min':>6}{'<5':>7}{'<6':>7}")
    for label in sorted(set(labels)):
        values = residuals[labels == label]
        print(
            f"{label:<42}{len(values):>8}{int(np.median(values)):>6}{values.min():>6}"
            f"{int((values < N_PARAMETERS).sum()):>7}"
            f"{int((values < N_PARAMETERS + 1).sum()):>7}"
        )

    print()
    print("distribution across every extrapolation row:")
    extrapolation = np.array(["interpolation" not in label for label in labels])
    values, counts = np.unique(residuals[extrapolation], return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        flag = "  <-- underdetermined" if value < N_PARAMETERS else ""
        print(f"  {value} residuals: {count:>7} rows{flag}")


def mode_verify(config, holdout_layers, inputs):
    disabled = counts_by_group(
        batches_for(config, holdout_layers, inputs,
                    hit_floor=config.min_extrapolation_fit_hits, residual_floor=1)
    )
    production = counts_by_group(
        batches_for(config, holdout_layers, inputs,
                    hit_floor=config.min_extrapolation_fit_hits,
                    residual_floor=V.DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS)
    )
    print()
    print("=" * 78)
    print(f"residual floor disabled vs {V.DEFAULT_MIN_EXTRAPOLATION_FIT_RESIDUALS} (production)")
    print("=" * 78)
    print(f"{'group':<42}{'before':>9}{'after':>9}{'delta':>8}")
    for label in sorted(disabled):
        delta = production[label] - disabled[label]
        flag = "  <200 !!" if production[label] < V.MIN_GROUP_COMPONENTS else ""
        print(f"{label:<42}{disabled[label]:>9}{production[label]:>9}{delta:>+8}{flag}")
    print(f"{'TOTAL':<42}{sum(disabled.values()):>9}{sum(production.values()):>9}"
          f"{sum(production.values()) - sum(disabled.values()):>+8}")
    print()
    unchanged = all(
        disabled[label] == production[label]
        for label in disabled if "interpolation" in label
    )
    print(f"interpolation groups unchanged: {unchanged}")
    print(f"every group >= {V.MIN_GROUP_COMPONENTS}: "
          f"{all(value >= V.MIN_GROUP_COMPONENTS for value in production.values())}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hits", "residuals", "verify"))
    parser.add_argument(
        "--states",
        default="data/reconstruction/nominal/Vol17_Lay8__Vol24_Lay4/trackstates_kf_mat1.root",
        help="Gate 4 的输入；留空则用 HelixValidationConfig 的默认值",
    )
    args = parser.parse_args()
    config, holdout_layers, inputs = load(args.states)
    {"hits": mode_hits, "residuals": mode_residuals, "verify": mode_verify}[args.mode](
        config, holdout_layers, inputs
    )


if __name__ == "__main__":
    main()
