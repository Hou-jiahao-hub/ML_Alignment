"""Solver tuning harness: load iteration-0 data ONCE, then sweep solver configs and
report, per target layer, the correlation of the SOLVED delta with the true per-sensor
misalignment.  Goal: lift corr from ~0.003 back toward the raw-residual level (~0.19)
by not rejecting genuine-signal hits as outliers."""
import re, json
import numpy as np
import torch

from alignment_artifact_layout import alignment_json_path, data_path, reconstruction_root_path

import ml_alignment as M

M.add_align_derivlib_to_path()

INPUT_GEOM = str(alignment_json_path("alignment_output", "loop_geom_0"))
STATES = str(reconstruction_root_path("reconstruction_output", "misaligned", "trackstates_kf"))
SUMMARY = str(reconstruction_root_path("reconstruction_output", "misaligned", "tracksummary_kf"))
NOMINAL = str(data_path("odd_transforms.json"))
TARGET = [(17, 8), (24, 4)]

# ---- load geometry + data + jacobians ONCE ----
print("Loading geometry / ROOT / jacobians (once) ...")
orig, transforms_list, NUM_SENSORS, init_pos, id_to_idx = M.load_geometry_from_json(INPUT_GEOM)
inputs = M.load_variable_information(SUMMARY, STATES, id_to_idx)
H_geom_batch = M.compute_jacobians_from_alignderivlib(
    inputs.hit_sensor_ids, inputs.hit_global_positions, inputs.hit_directions, inputs.hit_qoverp,
    transforms_list, current_deltas=None)
print(f"Data ready: {H_geom_batch.shape[0]} hits, {NUM_SENSORS} sensors.\n")

# ---- truth table ----
nom = {}
with open(NOMINAL, "r", encoding="utf-8") as f:
    nominal_data = json.load(f)
for it in nominal_data:
    g = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
    if g:
        nom[tuple(map(int, g.groups()))] = (np.array(it["Translation"], float),
                                            np.array(it["RotationMatrix"], float))

def eval_corr(pred):
    out = {}
    for vol, lay in TARGET:
        tv, pv = [], []
        for i, it in enumerate(transforms_list):
            g = re.search(r"vol=(\d+)\|lay=(\d+)\|sen=(\d+)", it.get("ID", ""))
            if not g:
                continue
            key = tuple(map(int, g.groups()))
            if key[0] != vol or key[1] != lay or key not in nom:
                continue
            c_mis = np.array(it["Translation"], float); R_mis = np.array(it["RotationMatrix"], float)
            c_nom, R_nom = nom[key]
            dt = R_mis.T @ (c_nom - c_mis)
            Rd = R_mis.T @ R_nom
            tv.append([dt[0], dt[1], np.arctan2(Rd[1, 0], Rd[0, 0])])
            p = pred[i]; pv.append([p[0], p[1], p[5]])
        tv = np.array(tv); pv = np.array(pv)
        res = {}
        for j, nm in enumerate(["tx", "ty", "rz"]):
            t, p = tv[:, j], pv[:, j]
            c = np.corrcoef(t, p)[0, 1] if np.std(t) > 0 and np.std(p) > 0 else float("nan")
            rec = (p @ t) / (t @ t) if (t @ t) > 0 else float("nan")
            res[nm] = (c, rec, np.sqrt((p**2).mean()), np.sqrt((t**2).mean()))
        out[(vol, lay)] = res
    return out

CONFIGS = [
    ("baseline pull=5  irls=4",  dict(pull_cut=5.0,   n_irls=4)),
    ("pull=50 irls=4",           dict(pull_cut=50.0,  n_irls=4)),
    ("pull=100 irls=4",          dict(pull_cut=100.0, n_irls=4)),
    ("no-reject (irls=1)",       dict(pull_cut=1e9,   n_irls=1)),
    ("no-reject low-reg",        dict(pull_cut=1e9,   n_irls=1, lambda_rel=1e-5, lambda_abs=1e-9)),
    ("floor=0.1 no-reject",      dict(sigma_floor=0.1, pull_cut=1e9, n_irls=1)),
    ("floor=0.2 no-reject",      dict(sigma_floor=0.2, pull_cut=1e9, n_irls=1)),
    ("floor=0.5 no-reject",      dict(sigma_floor=0.5, pull_cut=1e9, n_irls=1)),
    ("floor=0.2 pull=20 irls=3", dict(sigma_floor=0.2, pull_cut=20.0, n_irls=3)),
    ("floor=0.2 no-reject reg1e-4", dict(sigma_floor=0.2, pull_cut=1e9, n_irls=1, lambda_rel=1e-4)),
]

for name, kw in CONFIGS:
    base = dict(active_dof=(0, 1, 5), lambda_rel=1e-3, lambda_abs=1e-6,
                clamp_t=5.0, pull_cut=5.0, n_irls=4)
    base.update(kw)
    pred = M.solve_alignment_weighted_lsq(
        inputs.hit_sensor_ids, inputs.initial_residuals, inputs.covariances,
        H_geom_batch, NUM_SENSORS, **base)
    res = eval_corr(pred)
    print(f"\n===== {name} =====")
    for (vol, lay), r in res.items():
        s = "  ".join(f"{nm}: corr={r[nm][0]:+.3f} recov={r[nm][1]:+.3f} |pred|={r[nm][2]:.4f}"
                      for nm in ["tx", "ty", "rz"])
        print(f"  vol={vol} lay={lay}  {s}")
