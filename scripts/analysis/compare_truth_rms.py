"""L1 次判据：两臂几何 truth 误差 RMS 的配对非劣性检验。

真实物理场景里没有 truth。它只在带标注的实验里存在，只能用于**校准完成之后**核对
模型是否正确，绝不进入求解器的接受或停止判据 —— 求解器里任何一处读 truth 都是错的。
本脚本是仓库里唯一可以谈 truth 的地方，它的输出**也不得反过来指导求解器调参**：
判定跑完就固定，按预注册规则读，不据其继续调。

判据：逐 target layer、逐 DOF，误差 RMS = RMS over sensors of (pred - true)，
统计量 ratio = err_rms(candidate) / err_rms(baseline)，按 sensor 配对 bootstrap 取
95% 百分位 CI，**上界 <= 1.01** 为非劣性通过。

输入是几何 JSON，不是 summary JSON。per-sensor 的 pred/true 向量不落进 summary
（2268 sensor x 6 DOF x 2 臂约 300 KB，会把 summary 撑大一个数量级），而是从磁盘上
已有的三份几何精确重建：update_geometry_to_json 写的是 t_new = t + R @ dt、
R_new = R @ Rd，严格可逆。

归档说明：分析脚本，不被任何产品路径导入。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import ml_alignment_autograd as autograd  # noqa: E402
import validate_helix_model as validation  # noqa: E402

#: 预注册的非劣性上界：候选臂的 truth 误差 RMS 不得比基线差过 1%。
NON_INFERIORITY_RATIO = 1.01
DOF_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
#: 24:8 backward_extrapolation LOC0 的结论不得引用（P1'）。
P1_PRIME_NOTE = "提醒：24:4 backward_extrapolation LOC0 的数字不得进入结论（P1'）。"

_ID = re.compile(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)")


def load_geometry(path):
    """按 vol/lay/sen 键读一份几何，返回 {key: (translation, rotation)}。"""
    with open(path, "r", encoding="utf-8") as stream:
        items = json.load(stream)
    table = {}
    for item in items:
        match = _ID.search(item.get("ID", ""))
        if not match:
            continue
        table[tuple(map(int, match.groups()))] = (
            np.asarray(item["Translation"], dtype=float),
            np.asarray(item["RotationMatrix"], dtype=float),
        )
    return table


def local_delta(misaligned, other):
    """把一份几何相对错位几何的差表达成 ml 约定下的局部 6 分量。

    与 ml_alignment_autograd.validate_against_truth 用的是同一个公式：
        dt = R_mis^T (c - c_mis)
        Rd = R_mis^T R          -> (rx, ry, rz)
    对 truth 传入 nominal，对某一臂传入该臂的输出几何 —— 后者是
    update_geometry_to_json 的严格逆运算，因此拿回的就是实际施加的修正量
    （alignment_gain != 1 时也一样，那本来就是该跟 truth 比的量）。
    """
    center_mis, rotation_mis = misaligned
    center, rotation = other
    translation = rotation_mis.T @ (center - center_mis)
    angles = autograd.rotation_matrix_to_euler(rotation_mis.T @ rotation)
    return np.concatenate((translation, np.asarray(angles, dtype=float)))


def collect(misaligned, nominal, arms, volume, layer):
    """该层上可比较的 sensor 的 truth 与各臂预测，按同一顺序返回。

    ``expected`` 是错位几何与标称几何都认得的那一批 —— 也就是本该出现在两臂里的
    全集。**不**在这里跟各臂取交集：交集会把某一臂丢掉的 sensor 静默抹掉，让样本量
    悄悄变小而指标变好，正是 damping-v2 §4.1 记的那个失效模式。差异交给 parity()
    去发现并宣告比较作废。
    """
    expected = sorted(
        key for key in misaligned
        if key[0] == volume and key[1] == layer and key in nominal
    )
    keys = [key for key in expected if all(key in arm for arm in arms.values())]
    truth = np.array([local_delta(misaligned[key], nominal[key]) for key in keys])
    predictions = {
        name: np.array([local_delta(misaligned[key], arm[key]) for key in keys])
        for name, arm in arms.items()
    }
    return expected, keys, truth, predictions


def _rms(values):
    finite = values[np.isfinite(values)]
    return float(np.sqrt((finite ** 2).mean())) if finite.size else float("nan")


def ratio_statistic(column):
    """paired_bootstrap_ci 的 statistic：两臂逐 sensor 误差,同一批行被重采样。"""

    def statistic(baseline_errors, candidate_errors):
        base = _rms(baseline_errors[:, column])
        if not np.isfinite(base) or base <= 0.0:
            return float("nan")
        return _rms(candidate_errors[:, column]) / base

    return statistic


def parity(expected, arms):
    """先核对样本身份，不同就宣告比较作废 —— 效果指标一眼都不看。

    damping-v2 §4.1 的教训：秩亏判据的一处 bug 让 50 行健康数据被丢掉，RMSE 从
    0.8427 “改善”到 0.4921（42%）。发现它靠核对各组的 N，不是靠看 RMSE。任何让
    结果变好的改动，先查行数。
    """
    problems = []
    if not expected:
        problems.append("该层在错位/标称几何里没有可解析的 sensor")
    for name, arm in sorted(arms.items()):
        missing = [key for key in expected if key not in arm]
        if missing:
            problems.append(
                f"{name} 缺少 {len(missing)}/{len(expected)} 个 sensor："
                + ", ".join(f"{v}:{l}:{s}" for v, l, s in missing[:5])
                + (" ..." if len(missing) > 5 else "")
            )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--misaligned", required=True, help="注入错位后的输入几何 JSON")
    parser.add_argument("--nominal", required=True, help="标称全几何 JSON（truth 来源）")
    parser.add_argument("--baseline", required=True, help="基线臂输出几何 JSON")
    parser.add_argument("--candidate", required=True, help="候选臂输出几何 JSON")
    parser.add_argument("--target-layers", required=True, help='例如 "17:8,24:4"')
    parser.add_argument("--json", help="把判定结果写到这个路径")
    parser.add_argument("--resamples", type=int, default=validation.BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()

    misaligned = load_geometry(args.misaligned)
    nominal = load_geometry(args.nominal)
    arms = {
        "baseline": load_geometry(args.baseline),
        "candidate": load_geometry(args.candidate),
    }
    layers = autograd.parse_target_layers(args.target_layers)
    rng = np.random.default_rng(validation.BOOTSTRAP_SEED)

    print("=" * 78)
    print("L1 次判据：几何 truth 误差 RMS 的配对非劣性检验")
    print(f"  错位几何 : {args.misaligned}")
    print(f"  标称几何 : {args.nominal}")
    print(f"  baseline : {args.baseline}")
    print(f"  candidate: {args.candidate}")
    print(f"  bootstrap: resamples={args.resamples} seed={validation.BOOTSTRAP_SEED}"
          f"  非劣性上界={NON_INFERIORITY_RATIO}")

    report = {
        "criterion": "paired_bootstrap_truth_error_rms_ratio",
        "non_inferiority_ratio": NON_INFERIORITY_RATIO,
        "resamples": args.resamples,
        "bootstrap_seed": validation.BOOTSTRAP_SEED,
        "claim_scope": "verification-only",
        "layers": {},
    }
    invalidated = False
    failures = []
    undefined = []

    for volume, layer in layers:
        name = f"{volume}:{layer}"
        expected, keys, truth, predictions = collect(
            misaligned, nominal, arms, volume, layer
        )
        problems = parity(expected, arms)
        print(f"\n--- layer {name} ---")
        print(f"  核对：expected={len(expected)} comparable={len(keys)}  "
              + ("OK" if not problems else "DIFFER"))
        if problems:
            invalidated = True
            for problem in problems:
                print(f"    *** {problem}")
            report["layers"][name] = {
                "n_expected": len(expected),
                "n_comparable": len(keys),
                "parity": problems,
            }
            continue

        errors = {name_: predictions[name_] - truth for name_ in arms}
        entry = {
            "n_expected": len(expected),
            "n_comparable": len(keys),
            "parity": [],
            "dof": {},
        }
        header = f"  {'dof':<6}{'err_rms(base)':>16}{'err_rms(cand)':>16}{'ratio':>10}{'95% CI':>22}{'':>8}"
        print(header)
        for column, dof in enumerate(DOF_NAMES):
            finite_inputs = bool(
                np.isfinite(errors["baseline"][:, column]).all()
                and np.isfinite(errors["candidate"][:, column]).all()
            )
            interval = (
                validation.paired_bootstrap_ci(
                    errors["baseline"], errors["candidate"],
                    ratio_statistic(column), rng, resamples=args.resamples,
                )
                if finite_inputs
                else {
                    "point": float("nan"),
                    "low": float("nan"),
                    "high": float("nan"),
                    "n_finite_resamples": 0,
                }
            )
            base = _rms(errors["baseline"][:, column])
            cand = _rms(errors["candidate"][:, column])
            high = interval["high"]
            metric_undefined = bool(
                not finite_inputs
                or not all(
                    np.isfinite(interval[key]) for key in ("point", "low", "high")
                )
                or interval.get("n_finite_resamples") != args.resamples
            )
            passed = bool(not metric_undefined and high <= NON_INFERIORITY_RATIO)
            verdict = "n/a" if metric_undefined else ("PASS" if passed else "FAIL")
            if metric_undefined:
                undefined.append(f"{name}.{dof}")
            elif not passed:
                failures.append(f"{name}.{dof} (CI 上界 {high:.4f})")
            confidence = f"[{interval['low']:.4f}, {high:.4f}]"
            print(f"  {dof:<6}{base:>16.6g}{cand:>16.6g}{interval['point']:>10.4f}"
                  f"{confidence:>22}{verdict:>8}")
            entry["dof"][dof] = {
                "err_rms_baseline": base,
                "err_rms_candidate": cand,
                "ratio": interval["point"],
                "ci_low": interval["low"],
                "ci_high": high,
                "n_finite_resamples": interval.get("n_finite_resamples"),
                "verdict": verdict,
            }
        report["layers"][name] = entry

    print("\n" + "=" * 78)
    if invalidated:
        report["verdict"] = "invalidated"
        print("结论：核对块 DIFFER，本次比较作废。两臂的 sensor 集合必须逐位相同。")
    elif undefined:
        report["verdict"] = "invalidated"
        print("结论：存在未定义或非有限的非劣性指标，本次比较作废 —— " + "; ".join(undefined))
    elif failures:
        report["verdict"] = "failed"
        print("结论：非劣性未通过 —— " + "; ".join(failures))
    else:
        report["verdict"] = "passed"
        print(f"结论：所有 DOF 的比值 95% CI 上界 <= {NON_INFERIORITY_RATIO}，非劣性通过。")
    report["failures"] = failures
    report["undefined"] = undefined
    print("宣称边界：两臂几何等价时只能宣称收敛契约得到验证，不得宣称物理精度提升；")
    print("          本判据不解释也不修正已知的 9.9 倍 truth 距离偏差。")
    print(P1_PRIME_NOTE)

    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(autograd._json_ready(report), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"判定结果已写入 {destination}")
    return 0 if report["verdict"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
