"""P2''' 两臂 summary 比较：只打印，不判定。按预注册核对顺序输出。

归档说明：一次性分析脚本，未按产品标准编写，无测试覆盖，不被任何产品路径导入。
保留在库内只为让 Iteration 20（P2'''） 的结论可以被重新核对；改动产品代码时无需同步维护它。
产物在 data/experiments/joint_downstream/Vol17_Lay8__Vol24_Lay4/p2ppp-20260826/。
"""
import argparse
import json

P1_PRIME = ("(24, 4)", "loc0")   # 24:4 backward_extrapolation LOC0，结论不得引用


def load(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def same(a, b):
    return "same" if a == b else "DIFFER"


def fmt(value, width=14):
    if value is None:
        return "n/a".rjust(width)
    if isinstance(value, float):
        return f"{value:+.6g}".rjust(width)
    return str(value).rjust(width)


def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return fmt(float(b) - float(a))
    return "n/a".rjust(14)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parity_block(left, right, name_l, name_r):
    section("1. 核对块（样本量先于效果；任一 DIFFER 则本次比较作废）")
    rows = [
        ("n_tracks", left.get("n_tracks"), right.get("n_tracks")),
        ("n_hits", left.get("n_hits"), right.get("n_hits")),
    ]
    init_l = left.get("initialization_stats", {})
    init_r = right.get("initialization_stats", {})
    for key in ("input_tracks", "retained_tracks", "excluded_tracks", "fit_hits"):
        rows.append((f"initialization_stats.{key}", init_l.get(key), init_r.get(key)))
    rows.append(("invalid_fraction", left.get("invalid_fraction"), right.get("invalid_fraction")))
    rows.append(("initial_invalid_fraction",
                 left.get("initial_invalid_fraction"), right.get("initial_invalid_fraction")))

    print(f"  {'量':<34}{name_l:>14}{name_r:>14}   verdict")
    violations = []
    for label, a, b in rows:
        verdict = same(a, b)
        if verdict == "DIFFER":
            violations.append(label)
        print(f"  {label:<34}{fmt(a)}{fmt(b)}   {verdict}")
    return violations


def truth_block(left, right, name_l, name_r):
    section("2. 主指标 truth_metrics（per-DOF correlation / recovery fraction）")
    tl, tr = left.get("truth_metrics") or {}, right.get("truth_metrics") or {}
    if not tl and not tr:
        print("  两臂均无 truth_metrics（nominal 几何缺失？）")
        return
    for layer in sorted(set(tl) | set(tr)):
        print(f"\n  --- layer {layer} ---")
        dl, dr = tl.get(layer, {}), tr.get(layer, {})
        print(f"  {'dof/量':<20}{name_l:>14}{name_r:>14}{'delta':>14}")
        for dof in ("tx", "ty", "tz", "rx", "ry", "rz"):
            el, er = dl.get(dof, {}), dr.get(dof, {})
            for metric in ("corr", "recovery", "true_rms", "pred_rms"):
                a, b = el.get(metric), er.get(metric)
                if a is None and b is None:
                    continue
                print(f"  {dof + '.' + metric:<20}{fmt(a)}{fmt(b)}{delta(a, b)}")


def target_block(left, right, name_l, name_r):
    section("3. 次指标 target_metrics（残差 mean/RMS；可被轨迹自由度吸收而虚假变好）")
    tl, tr = left.get("target_metrics") or {}, right.get("target_metrics") or {}
    for layer in sorted(set(tl) | set(tr)):
        print(f"\n  --- layer {layer} ---")
        dl, dr = tl.get(layer, {}), tr.get(layer, {})
        print(f"  {'量':<26}{name_l:>14}{name_r:>14}{'delta':>14}")
        for metric in ("n_hits_loc0", "n_hits_loc1",
                       "mean_before_loc0", "mean_after_loc0",
                       "mean_before_loc1", "mean_after_loc1",
                       "rms_before_loc0", "rms_after_loc0",
                       "rms_before_loc1", "rms_after_loc1"):
            a, b = dl.get(metric), dr.get(metric)
            if a is None and b is None:
                continue
            tag = "  [P1' 不可引用]" if (layer == P1_PRIME[0] and P1_PRIME[1] in metric) else ""
            print(f"  {metric:<26}{fmt(a)}{fmt(b)}{delta(a, b)}{tag}")


def convergence_block(left, right, name_l, name_r):
    section("4. 收敛块与验收留痕")
    print(f"  {'量':<40}{name_l:>16}{name_r:>16}")
    for key in ("initial_total_loss", "best_total_loss", "last_total_loss",
                "best_iteration", "stopped_early", "solver"):
        print(f"  {key:<40}{fmt(left.get(key), 16)}{fmt(right.get(key), 16)}")

    print("\n  -- initialization_stats.measurement_refit --")
    ml = (left.get("initialization_stats") or {}).get("measurement_refit") or {}
    mr = (right.get("initialization_stats") or {}).get("measurement_refit") or {}
    for key in sorted(set(ml) | set(mr)):
        vl, vr = ml.get(key), mr.get(key)
        if isinstance(vl, (dict, list)) or isinstance(vr, (dict, list)):
            print(f"  {key:<40}{str(vl)[:16]:>16}{str(vr)[:16]:>16}")
        else:
            print(f"  {key:<40}{fmt(vl, 16)}{fmt(vr, 16)}")

    print("\n  -- stage_layer_chi2（smt_seed / measurement_refit / joint）--")
    sl = left.get("stage_layer_chi2") or {}
    sr = right.get("stage_layer_chi2") or {}
    for stage in ("smt_seed", "measurement_refit", "joint"):
        layers_l, layers_r = sl.get(stage) or {}, sr.get(stage) or {}
        for layer in sorted(set(layers_l) | set(layers_r)):
            a, b = layers_l.get(layer), layers_r.get(layer)
            if isinstance(a, dict) or isinstance(b, dict):
                keys = sorted(set(a or {}) | set(b or {}))
                for key in keys:
                    print(f"  {stage}.{layer}.{key:<24}"
                          f"{fmt((a or {}).get(key), 16)}{fmt((b or {}).get(key), 16)}")
            else:
                print(f"  {stage}.{layer:<30}{fmt(a, 16)}{fmt(b, 16)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="基线臂 summary JSON（adam）")
    parser.add_argument("right", help="对照臂 summary JSON（gauss-newton）")
    parser.add_argument("--name-left", default="adam")
    parser.add_argument("--name-right", default="gauss-newton")
    args = parser.parse_args()

    left, right = load(args.left), load(args.right)
    print(f"left  ({args.name_left}) : {args.left}")
    print(f"right ({args.name_right}) : {args.right}")

    violations = parity_block(left, right, args.name_left, args.name_right)
    if violations:
        print("\n  *** 核对未通过：" + ", ".join(violations))
        print("  *** 样本量在两臂间发生了变化，效果指标不可比 —— 先查原因。")

    truth_block(left, right, args.name_left, args.name_right)
    target_block(left, right, args.name_left, args.name_right)
    convergence_block(left, right, args.name_left, args.name_right)

    print("\n" + "=" * 78)
    if violations:
        print("结论：核对块 DIFFER，本次比较作废。")
    else:
        print("核对块全部相同；效果指标可比。判读见 Iteration 20 预注册 §1.3。")
    print("提醒：24:4 backward_extrapolation LOC0 的数字不得进入结论（P1'）。")


if __name__ == "__main__":
    main()
