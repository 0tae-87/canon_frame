"""Paired choice analysis between a control arm and a gated arm on the SAME (object,
trial) poses: did the gate change the executed grasp, and how did the outcome move?

For every trial present in both arms (dump + CSV verdict), compare the executed grasp
poses (same if tool-centre within 5 mm and approach axis within 10 deg). Then a 2x2 of
outcomes for changed vs unchanged choices, per object and pooled, plus d_obs of the
executed grasp in both arms (support_d_obs_mm column if logged, else recomputed from
the dump's observed tail).

    python canon_frame/analysis/paired_choice.py --ctrl runCanon:gate_off --arm runLat04:complete_reg
"""
import argparse, csv, glob, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(REPO)
sys.path.insert(0, "icra/isaac_graspgen/source")
from graspgen import config as gc                                  # noqa: E402
from graspgen.uncertainty_gate import observed_support             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ctrl", default="runCanon:gate_off"); ap.add_argument("--arm", default="runLat04:complete_reg")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
O = "icra/isaac_graspgen/output/graspgen"


def load(tag, arm):
    verdict = {}
    for p in glob.glob(f"{O}/{tag}_grasp_log*.csv"):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["executed"] == "1" and r["lift_success"] in ("0", "1"):
                    verdict[(r["trial"], int(r["grasp_id"]))] = (int(r["lift_success"]), r.get("support_d_obs_mm", ""))
    out = {}
    for f in glob.glob(f"{O}/{tag}_dump/s{a.seed}_{arm}_*.npz"):
        m = re.search(rf"s{a.seed}_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
        if not m: continue
        d = np.load(f); ch = int(np.ravel(d["chosen"])[0]); trial = os.path.splitext(os.path.basename(f))[0]
        if ch < 0 or (trial, ch) not in verdict or "completed" not in d.files: continue
        ok, dcol = verdict[(trial, ch)]
        T = d["grasps"][ch].astype(float); comp = d["completed"].astype(float)
        if dcol not in ("", None):
            dob = float(dcol)
        elif "d_obs" in d.files:
            dob = float(d["d_obs"][ch]) * 1000
        else:
            dob = float(observed_support(T[None], comp, comp[-2048:], depth=gc.PAD_CENTER_DEPTH,
                                         region_extents=gc.GRIPPER_CLOSING_REGION)[0]) * 1000
        out[(m.group(1), int(m.group(2)))] = dict(ok=ok, T=T, d_obs=dob, conf=float(d["confidences"][ch]))
    return out


tc, ac = a.ctrl.split(":"), a.arm.split(":")
C, A = load(*tc), load(*ac)
keys = sorted(set(C) & set(A))
print(f"{a.ctrl} vs {a.arm}: {len(keys)} paired trials (ctrl {len(C)}, arm {len(A)})")


def same(T1, T2):
    tc1 = T1[:3, 3] + 0.195 * T1[:3, 2]; tc2 = T2[:3, 3] + 0.195 * T2[:3, 2]
    ang = np.degrees(np.arccos(np.clip(T1[:3, 2] @ T2[:3, 2], -1, 1)))
    return np.linalg.norm(tc1 - tc2) < 0.005 and ang < 10


rows = [(k[0], same(C[k]["T"], A[k]["T"]), C[k]["ok"], A[k]["ok"], C[k]["d_obs"], A[k]["d_obs"]) for k in keys]
R = np.array([(r[1], r[2], r[3], r[4], r[5]) for r in rows], dtype=float)
objs = sorted(set(r[0] for r in rows))
print(f"\n{'object':>22} {'n':>4} {'same':>5} {'ctrl%':>6} {'arm%':>6} | changed: {'n':>3} {'ctrl->arm':>10} {'win':>4} {'loss':>4} | d_obs ctrl/arm (changed)")
def block(mask, name):
    n = int(mask.sum()); s = R[mask]
    if n == 0: return
    ch = s[s[:, 0] == 0]; win = int(((ch[:, 1] == 0) & (ch[:, 2] == 1)).sum()); loss = int(((ch[:, 1] == 1) & (ch[:, 2] == 0)).sum())
    dd = f"{np.nanmean(ch[:, 3]):5.1f}/{np.nanmean(ch[:, 4]):<5.1f}" if len(ch) else "    -"
    print(f"{name:>22} {n:>4} {int(s[:, 0].sum()):>5} {100 * s[:, 1].mean():>6.1f} {100 * s[:, 2].mean():>6.1f} | "
          f"{len(ch):>12} {100 * ch[:, 1].mean() if len(ch) else 0:>4.0f}->{100 * ch[:, 2].mean() if len(ch) else 0:<4.0f} {win:>4} {loss:>4} | {dd}")
for o in objs:
    block(np.array([r[0] == o for r in rows]), o)
block(np.ones(len(rows), bool), "ALL")
ch = R[R[:, 0] == 0]
if len(ch):
    print(f"\nchanged choices: {len(ch)} ({100 * len(ch) / len(R):.0f} %); ctrl success {100 * ch[:, 1].mean():.1f} % -> arm {100 * ch[:, 2].mean():.1f} %; "
          f"net {int(((ch[:, 1] == 0) & (ch[:, 2] == 1)).sum()) - int(((ch[:, 1] == 1) & (ch[:, 2] == 0)).sum()):+d} trials")
    un = R[R[:, 0] == 1]
    if len(un): print(f"unchanged choices: {len(un)}; ctrl {100 * un[:, 1].mean():.1f} % -> arm {100 * un[:, 2].mean():.1f} % (execution noise floor)")
