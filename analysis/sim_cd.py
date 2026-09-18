"""Sim-domain reconstruction quality against the YCB meshes.

Pose: from the dump when logged (obj_pos / obj_quat_wxyz, 2026-09-16 on), else reconstructed
from the runner protocol -- solo spawn at (solo_radius, 0, .) on a 0.25 m pedestal, yaw from
random.Random(scene_seed) (trial 0) or random.Random((seed+1)*100003 + trial) (trial >= 1),
orientation = quat_mul(yaw_quat, UP_AXIS_FIX[up_axis]), z such that the rotated mesh's lowest
point rests on the pedestal. Reconstruction is checked per cloud: the OBSERVED points must lie
on the GT surface (obs->GT median), which also catches a wrong yaw on asymmetric objects.

Metrics (mm): CD-L1 (mean of both one-sided mean NN distances) between the generated points
and 8192 GT samples; precision = gen->GT (hallucination off the surface); recall = GT->gen
(coverage / extent recovery). Per arm and object.

    python canon_frame/analysis/sim_cd.py --tag runLat04 --arms gate_off complete_reg
"""
import argparse, glob, math, os, random, re, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(REPO)
sys.path.insert(0, "icra/isaac_graspgen/source")
from sim.config import YCB_CONFIG                                   # noqa: E402
from sim.ycb import UP_AXIS_FIX, quat_mul, yaw_to_quat             # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="runLat04"); ap.add_argument("--arms", nargs="*", default=["gate_off", "complete_reg"])
ap.add_argument("--as-module", action="store_true", help=argparse.SUPPRESS)
ap.add_argument("--solo-radius", type=float, default=0.50); ap.add_argument("--pedestal", type=float, default=0.25)
ap.add_argument("--max-per-cell", type=int, default=50); ap.add_argument("--n-gt", type=int, default=8192)
ap.add_argument("--up-override", nargs="*", default=[], help="name=axis for runs made before a config fix, e.g. 022_windex_bottle=-y")
a = ap.parse_args([]) if __name__ != "__main__" else ap.parse_args()
O = "icra/isaac_graspgen/output/graspgen"; GT_DIR = "canon_frame/assets/ycb_gt"
UP = {o["name"]: o.get("up_axis", "z") for o in YCB_CONFIG["objects"]}
for kv in a.up_override:
    k, v = kv.split("="); UP[k] = v


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def nn(a_, b_, chunk=512):
    """nearest-neighbour distances a_ -> b_ (KD-tree; exact)"""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(b_).query(a_, workers=4)[0]
    except ImportError:
        out = np.empty(len(a_))
        for i in range(0, len(a_), chunk):
            out[i:i + chunk] = np.sqrt(((a_[i:i + chunk, None, :] - b_[None]) ** 2).sum(-1).min(1))
        return out


def pose(d, name, seed, trial):
    if "obj_pos" in d.files:
        return d["obj_pos"].astype(float), quat_to_R(d["obj_quat_wxyz"].astype(float)), "logged"
    yaw = random.Random(seed).uniform(-math.pi, math.pi) if trial == 0 else random.Random((seed + 1) * 100003 + trial).uniform(-math.pi, math.pi)
    q = quat_mul(yaw_to_quat(yaw), UP_AXIS_FIX[UP[name]]); R = quat_to_R(q)
    return None, R, "reconstructed"


rng = np.random.default_rng(0); gt_cache = {}
def gt_for(name):
    if name not in gt_cache:
        g = np.load(f"{GT_DIR}/{name}.npy").astype(float); gt_cache[name] = g[rng.choice(len(g), a.n_gt, replace=False)], g
    return gt_cache[name]


if __name__ != "__main__":
    sys.exit  # noqa (imported: definitions only)
print(f"{a.tag}: {'arm':>14} {'object':>22} {'n':>3} {'pose':>13} {'obs->GT med':>11} {'CD-L1':>7} {'prec':>6} {'recall':>7}   (mm)")
summary = {}
for arm in (a.arms if __name__ == "__main__" else []):
    per = {}
    for f in sorted(glob.glob(f"{O}/{a.tag}_dump/s*_{arm}_*.npz")):
        m = re.search(rf"s(\d+)_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
        if not m: continue
        seed, name, trial = int(m.group(1)), m.group(2), int(m.group(3))
        if len(per.get(name, [])) >= a.max_per_cell: continue
        d = np.load(f)
        if "completed" not in d.files: continue
        comp = d["completed"].astype(float); gen, obs = comp[:-2048], comp[-2048:]
        gts, gfull = gt_for(name); t, R, how = pose(d, name, seed, trial)
        if t is None:
            t = np.array([a.solo_radius, 0.0, 0.0]); t[2] = a.pedestal - (gfull @ R.T)[:, 2].min()
        gw = gts @ R.T + t
        obs_err = np.median(nn(obs[rng.choice(len(obs), 512, replace=False)], gw)) * 1000
        p = nn(gen, gw) * 1000; r = nn(gw, gen) * 1000
        per.setdefault(name, []).append((obs_err, 0.5 * (p.mean() + r.mean()), p.mean(), r.mean(), how))
    for name, L in sorted(per.items()):
        A = np.array([x[:4] for x in L])
        print(f"{'':>10} {arm:>14} {name:>22} {len(L):>3} {L[0][4]:>13} {np.median(A[:, 0]):>11.1f} {A[:, 1].mean():>7.1f} {A[:, 2].mean():>6.1f} {A[:, 3].mean():>7.1f}")
        summary[(arm, name)] = A.mean(0)
    A = np.array([x[:4] for L in per.values() for x in L])
    if len(A): print(f"{'':>10} {arm:>14} {'ALL':>22} {len(A):>3} {'':>13} {np.median(A[:, 0]):>11.1f} {A[:, 1].mean():>7.1f} {A[:, 2].mean():>6.1f} {A[:, 3].mean():>7.1f}")
