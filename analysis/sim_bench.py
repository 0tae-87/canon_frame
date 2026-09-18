"""Offline sim-domain completion benchmark (no Isaac): take the observed clouds stored in
trial dumps (2048-point FPS tails = exactly what the server saw, world metres), send them
to one or more completion servers, and score the returned completion against the YCB mesh
at the trial's pose (logged or reconstructed, see sim_cd.py). Lets regressor variants be
compared in seconds on the same observations that produced the grasp results.

    python canon_frame/analysis/sim_bench.py --tag runLat04 --arm gate_off \
        --servers base=172.17.0.2:5557 reg=127.0.0.1:5559 --per-cell 12
"""
import argparse, glob, math, os, random, re, sys
import numpy as np, zmq, msgpack, msgpack_numpy
msgpack_numpy.patch()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(REPO)
sys.path.insert(0, "icra/isaac_graspgen/source"); sys.path.insert(0, "canon_frame/analysis")
from sim_cd import nn, pose, gt_for, UP                              # noqa: E402  (module-level arg parsing is guarded below)
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="runLat04"); ap.add_argument("--arm", default="gate_off")
ap.add_argument("--servers", nargs="+", default=["base=172.17.0.2:5557", "reg=127.0.0.1:5559"])
ap.add_argument("--per-cell", type=int, default=12); ap.add_argument("--solo-radius", type=float, default=0.50); ap.add_argument("--pedestal", type=float, default=0.25)
ap.add_argument("--up-override", nargs="*", default=["022_windex_bottle=-y"])
a = ap.parse_args()
for kv in a.up_override:
    k, v = kv.split("="); UP[k] = v
O = "icra/isaac_graspgen/output/graspgen"
ctx = zmq.Context(); socks = {}
for spec in a.servers:
    lab, hp = spec.split("="); s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 60000); s.connect(f"tcp://{hp}"); socks[lab] = s


def complete(lab, pts):
    socks[lab].send(msgpack.packb({"action": "complete", "point_cloud": pts.astype(np.float32)}, use_bin_type=True))
    r = msgpack.unpackb(socks[lab].recv(), raw=False)
    if "error" in r: raise RuntimeError(r["error"])
    return r


rng = np.random.default_rng(0); per = {}
for f in sorted(glob.glob(f"{O}/{a.tag}_dump/s*_{a.arm}_*.npz")):
    m = re.search(rf"s(\d+)_{a.arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
    if not m: continue
    seed, name, trial = int(m.group(1)), m.group(2), int(m.group(3))
    if len(per.get(name, [])) >= a.per_cell: continue
    d = np.load(f)
    obs = d["observed"].astype(float) if "observed" in d.files else (d["completed"][-2048:].astype(float) if "completed" in d.files else None)
    if obs is None: continue
    gts, gfull = gt_for(name); t, R, how = pose(d, name, seed, trial)
    if t is None:
        t = np.array([a.solo_radius, 0.0, 0.0]); t[2] = a.pedestal - (gfull @ R.T)[:, 2].min()
    gw = gts @ R.T + t; gc = (gw.min(0) + gw.max(0)) / 2; gr = np.linalg.norm(gw - gc, axis=1).max()
    row = {}
    for lab in socks:
        r = complete(lab, obs); n_obs = int(r["n_input_used"]); gen = r["completed"][:-n_obs].astype(float)
        p = nn(gen, gw) * 1000; rc = nn(gw, gen) * 1000
        cc = (gen.min(0) + gen.max(0)) / 2; cr = np.linalg.norm(gen - cc, axis=1).max()
        row[lab] = (0.5 * (p.mean() + rc.mean()), p.mean(), rc.mean(), np.linalg.norm(cc - gc) * 1000, cr / gr,
                    abs(cc[0] - gc[0]) * 1000, abs(cc[1] - gc[1]) * 1000, abs(cc[2] - gc[2]) * 1000)   # |dx| (camera depth axis), |dy| lateral, |dz| up
    per.setdefault(name, []).append(row)
labs = list(socks)
print(f"{a.tag}/{a.arm} observations -> " + " | ".join(labs) + "     CD-L1 / precision / recall (mm), bbox-centre error (mm), radius ratio")
print(f"{'object':>22} {'n':>3} " + " ".join(f"{lab:>34}" for lab in labs))
allrows = []
for name, L in sorted(per.items()):
    print(f"{name:>22} {len(L):>3} " + " ".join(f"{np.mean([r[lab][0] for r in L]):>6.1f} {np.mean([r[lab][1] for r in L]):>5.1f} {np.mean([r[lab][2] for r in L]):>6.1f}  c{np.mean([r[lab][3] for r in L]):>5.1f}  r{np.mean([r[lab][4] for r in L]):>5.2f}" for lab in labs))
    allrows += L
print(f"{'ALL':>22} {len(allrows):>3} " + " ".join(f"{np.mean([r[lab][0] for r in allrows]):>6.1f} {np.mean([r[lab][1] for r in allrows]):>5.1f} {np.mean([r[lab][2] for r in allrows]):>6.1f}  c{np.mean([r[lab][3] for r in allrows]):>5.1f}  r{np.mean([r[lab][4] for r in allrows]):>5.2f}" for lab in labs))
print("bbox-centre error by axis, mean |dx| (depth/camera axis) |dy| (lateral) |dz| (up), mm: " + " | ".join(f"{lab}: {np.mean([r[lab][5] for r in allrows]):.1f} {np.mean([r[lab][6] for r in allrows]):.1f} {np.mean([r[lab][7] for r in allrows]):.1f}" for lab in labs))
