"""Sim-domain check of the regressor WITHOUT ground truth: under view truncation, does the
completion recover the object's true extent? Truth proxy = bbox extent of the FULL observed
cloud of the same (object, trial) pose from runCanon (observed tail of its dumps, 2048 FPS
points of the untruncated view). Compare generated-cloud extents of complete_only and
complete_reg under truncation (runOccF035) with (a) the truncated observation and (b) the
full observation.

    python canon_frame/analysis/occlusion_extent.py --occ runOccF035 --full runCanon
"""
import argparse, glob, os, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--occ", default="runOccF035"); ap.add_argument("--full", default="runCanon")
a = ap.parse_args(); O = "icra/isaac_graspgen/output/graspgen"


def load(tag, arm):
    out = {}
    for f in glob.glob(f"{O}/{tag}_dump/s0_{arm}_*.npz"):
        m = re.search(rf"s0_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
        if not m: continue
        d = np.load(f)
        if "completed" not in d.files: continue
        comp = d["completed"].astype(float); n_obs = 2048
        out[(m.group(1), int(m.group(2)))] = (comp[:-n_obs], comp[-n_obs:])
    return out


def ext(p): return p.max(0) - p.min(0)


full = load(a.full, "gate_off")
print(f"{'object':>22} {'n':>3} {'obs_trunc/full':>14} {'gen: only/full':>14} {'gen: reg/full':>13}   (bbox extent ratios, mean of x,y,z)")
for arm_pair in [("gate_off", "complete_reg")]:
    A = load(a.occ, arm_pair[0]); B = load(a.occ, arm_pair[1])
    keys = sorted(set(full) & set(A) & set(B)); objs = sorted(set(k[0] for k in keys))
    rows = []
    for k in keys:
        gen_o, obs_t = A[k]; gen_r, _ = B[k]; _, obs_f = full[k]
        ef = ext(obs_f); rows.append((k[0], (ext(obs_t) / ef).mean(), (ext(gen_o) / ef).mean(), (ext(gen_r) / ef).mean()))
    for o in objs:
        r = np.array([x[1:] for x in rows if x[0] == o])
        print(f"{o:>22} {len(r):>3} {r[:, 0].mean():>14.2f} {r[:, 1].mean():>14.2f} {r[:, 2].mean():>13.2f}")
    r = np.array([x[1:] for x in rows])
    if len(r): print(f"{'ALL':>22} {len(r):>3} {r[:, 0].mean():>14.2f} {r[:, 1].mean():>14.2f} {r[:, 2].mean():>13.2f}")
