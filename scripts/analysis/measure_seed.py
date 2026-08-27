"""P4a: seed_helices 的 Adam 预拟合驻点性测量。纯观测，不改产品代码。

三个判据，前两个无 gauge 争议：
  1. 预算外延：继续跑 Adam，SMT loss 还能降多少（截断 vs 收敛的直接证据）
  2. 相对 Newton decrement: (g^T H^-1 g) / F —— 无量纲
  3. GN 求解同一目标，看真正的最小值在哪

归档说明：一次性分析脚本，未按产品标准编写，无测试覆盖，不被任何产品路径导入。
保留在库内只为让 Iteration 21（P4a） 的结论可以被重新核对；改动产品代码时无需同步维护它。
产物在 data/experiments/joint_downstream/Vol17_Lay8__Vol24_Lay4/p4a-20260826/。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import ml_alignment_autograd as M


def build(max_tracks):
    cfg = M.AlignmentRunConfig(
        input_geom=str(M.alignment_json_path("alignment_output", "misalignment_applied")),
        states_root=str(M.reconstruction_root_path("reconstruction_output", "misaligned", "trackstates_kf")),
        summary_root=str(M.reconstruction_root_path("reconstruction_output", "misaligned", "tracksummary_kf")),
        output_geom="/dev/null",
        max_tracks=max_tracks,
    )
    target_layers = M.parse_target_layers(M.target_layers_spec())
    geometry = M.load_joint_geometry(cfg.nominal_full_geom, cfg.input_geom, target_layers)
    inputs = M.load_variable_information(
        cfg.summary_root, cfg.states_root, geometry.id_to_idx,
        max_track_entries=(cfg.max_tracks * 4 if cfg.max_tracks > 0 else None),
        load_summary=False,
    )
    batches = M.build_track_batches(
        inputs, target_layers,
        min_hits=cfg.joint_seed_min_hits, min_span=cfg.joint_seed_min_span,
        max_tracks=cfg.max_tracks,
    )
    return batches, geometry


def smt_residuals(theta, points, mask, dtype):
    """固定最近点参数 s（包络定理：梯度与 min over s 一致），返回 per-track 残差。"""
    n_hits = points.shape[1]
    expanded = theta[:, None, :].expand(-1, n_hits, -1)
    with torch.no_grad():
        _fit, path_length, valid = M.closest_helix_points(expanded.detach(), points)
    predicted = M.helix_point(expanded, path_length)
    effective = mask & valid
    diff = predicted - points
    diff = torch.where(effective[..., None], diff, torch.zeros_like(diff))
    return diff, effective


def smt_loss_mean(theta, points, mask):
    """返回 (loss, n_effective)。分母会变——loss 变好必须同时看 n_effective。"""
    diff, effective = smt_residuals(theta, points, mask, theta.dtype)
    sq = torch.sum(diff * diff, dim=-1)
    n_eff = int(effective.sum())
    return float(sq.sum() / max(n_eff, 1)), n_eff


def per_track_F(theta, points, mask):
    """F = 0.5 * sum_hits ||r||^2，per-track，不做 count 归一（与 Iteration 18 同精神）。"""
    diff, effective = smt_residuals(theta, points, mask, theta.dtype)
    return 0.5 * torch.sum(diff * diff, dim=(1, 2)), effective


def gn_step_and_decrement(theta, points, mask, damping=1e-9):
    """damping 可为标量或 per-track 张量。"""
    """per-track Gauss-Newton: 返回 (decrement, step, F)。"""
    n_tracks, n_hits, _ = points.shape
    theta = theta.detach().clone().requires_grad_(True)
    diff, effective = smt_residuals(theta, points, mask, theta.dtype)
    resid = diff.reshape(n_tracks, -1)                      # (T, H*3)
    rows = resid.shape[1]
    jac = torch.zeros((n_tracks, rows, 6), dtype=theta.dtype)
    for k in range(rows):
        grad = torch.autograd.grad(resid[:, k].sum(), theta, retain_graph=(k < rows - 1))[0]
        jac[:, k, :] = grad
    resid = resid.detach()
    hess = jac.transpose(1, 2) @ jac
    grad = (jac.transpose(1, 2) @ resid.unsqueeze(-1)).squeeze(-1)
    scale = hess.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    eye = torch.eye(6, dtype=theta.dtype)
    lam = damping if torch.is_tensor(damping) else torch.full_like(scale, damping)
    step = torch.linalg.solve(hess + lam[:, None, None] * scale[:, None, None] * eye,
                              grad.unsqueeze(-1)).squeeze(-1)
    decrement = (grad * step).sum(dim=1)
    F = 0.5 * (resid * resid).sum(dim=1)
    return decrement.detach(), step.detach(), F.detach()


def continue_adam(theta, points, mask, n_iter, lr=0.5):
    linear = theta[:, :5].detach().clone().requires_grad_(True)
    rho = theta[:, 5:].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [linear], "lr": lr}, {"params": [rho], "lr": lr}])
    n_hits = points.shape[1]
    for _ in range(n_iter):
        opt.zero_grad()
        tp = M.assemble_joint_track_parameters(linear, rho)
        expanded = tp[:, None, :].expand(-1, n_hits, -1)
        with torch.no_grad():
            _f, s, valid = M.closest_helix_points(expanded.detach(), points)
        pred = M.helix_point(expanded, s)
        eff = mask & valid
        sq = torch.sum((pred - points) ** 2, dim=-1)
        loss = torch.sum(torch.where(eff, sq, torch.zeros_like(sq))) / eff.sum().clamp(min=1)
        loss.backward()
        opt.step()
    return M.assemble_joint_track_parameters(linear, rho).detach()


def gn_solve(theta, points, mask, iters=30):
    """LM 求解 SMT 目标，带步长接受。"""
    cur = theta.detach().clone()
    damping = torch.full((cur.shape[0],), 1e-6, dtype=cur.dtype)
    F_cur, _ = per_track_F(cur, points, mask)
    for _ in range(iters):
        dec, step, _F = gn_step_and_decrement(cur, points, mask, damping=damping)
        cand = cur - step
        F_cand, _ = per_track_F(cand, points, mask)
        better = F_cand < F_cur
        cur = torch.where(better[:, None], cand, cur)
        F_cur = torch.where(better, F_cand, F_cur)
        damping = torch.where(better, damping * 0.3, damping * 10.0).clamp(1e-12, 1e3)
    return cur


def q(t, p):
    return float(t.quantile(p)) if t.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tracks", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", default="200,400,800,1800,2800",
                    help="继续跑的 Adam 迭代数，逗号分隔")
    ap.add_argument("--gn-iters", type=int, default=30)
    args = ap.parse_args()

    torch.manual_seed(42)
    batches, _geom = build(args.max_tracks)
    points = batches.g_smt.to(dtype=torch.float64)
    mask = M._non_target_fit_mask(batches)
    print(f"tracks={points.shape[0]} padded_hits={points.shape[1]} fit_hits={int(mask.sum())}")

    report = {"n_tracks": int(points.shape[0]), "fit_hits": int(mask.sum())}

    # 基线：产品代码的 seed（200 iters）
    seed = M.seed_helices(batches, n_iter=200, dtype=torch.float64)
    base_loss, base_eff = smt_loss_mean(seed, points, mask)
    report["seed_200_loss"] = base_loss
    report["seed_200_n_effective"] = base_eff
    print(f"[baseline] seed_helices(200) SMT loss = {base_loss:.8g}  n_effective={base_eff}")

    # 判据 1：预算外延
    for extra in [int(x) for x in args.budget.split(',') if x]:
        more = continue_adam(seed, points, mask, extra)
        loss, eff = smt_loss_mean(more, points, mask)
        report[f"adam_plus_{extra}"] = {"loss": loss, "n_effective": eff}
        flag = ""
        if eff < base_eff:
            flag = f"  <-- 分母缩小 {base_eff - eff} ({100.0*(base_eff-eff)/base_eff:.1f}%)"
        print(f"[budget]   +{extra:5d} Adam iters -> {loss:.8g}  n_eff={eff}  "
              f"({100.0 * (base_loss - loss) / base_loss:+.3f}%){flag}")

    # 判据 2：相对 Newton decrement
    dec, _step, F = gn_step_and_decrement(seed, points, mask)
    rel = dec / F.clamp_min(1e-30)
    report["decrement"] = {
        "p50": q(dec, 0.5), "p90": q(dec, 0.9), "p99": q(dec, 0.99),
        "max": float(dec.max()),
        "n_non_finite": int((~torch.isfinite(dec)).sum()),
    }
    report["relative_decrement"] = {
        "p50": q(rel, 0.5), "p90": q(rel, 0.9), "p99": q(rel, 0.99),
        "max": float(rel.max()),
        "frac_above_1pct": float((rel > 0.01).double().mean()),
        "frac_above_10pct": float((rel > 0.10).double().mean()),
    }
    print(f"[decrement] absolute (mm^2, SMT gauge): p50={q(dec,0.5):.4g} "
          f"p90={q(dec,0.9):.4g} p99={q(dec,0.99):.4g} max={float(dec.max()):.4g}")
    print(f"[decrement] relative dec/F: p50={q(rel,0.5):.4g} p90={q(rel,0.9):.4g} "
          f"p99={q(rel,0.99):.4g} | >1%: {100*report['relative_decrement']['frac_above_1pct']:.2f}% "
          f" >10%: {100*report['relative_decrement']['frac_above_10pct']:.2f}%")

    # 判据 3：GN 求解同一目标
    solved = gn_solve(seed, points, mask, iters=args.gn_iters)
    gn_loss, gn_eff = smt_loss_mean(solved, points, mask)
    report["gn_solved_loss"] = gn_loss
    report["gn_solved_n_effective"] = gn_eff
    dec2, _s2, F2 = gn_step_and_decrement(solved, points, mask)
    rel2 = dec2 / F2.clamp_min(1e-30)
    report["gn_solved_relative_decrement"] = {
        "p50": q(rel2, 0.5), "p90": q(rel2, 0.9), "max": float(rel2.max()),
    }
    print(f"[gn]       GN-solved SMT loss = {gn_loss:.8g}  n_eff={gn_eff}  "
          f"({100.0 * (base_loss - gn_loss) / base_loss:+.3f}% vs 200-iter seed)"
          + ("" if gn_eff == base_eff else f"  <-- 分母变化 {gn_eff - base_eff}"))
    print(f"[gn]       post-GN relative decrement: p50={q(rel2,0.5):.4g} p90={q(rel2,0.9):.4g}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
