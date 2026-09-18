"""Is distance-to-observed-surface CAUSAL for failure, or a marker of a hard pose?
Pair the trials of a control arm and a gated arm on identical poses, stratify by the
CONTROL's executed d_obs, and compare success inside each stratum. If the gated arm's
replacement grasps (closer to the observed surface, lower confidence rank) do not
succeed more often in the far strata, the distance was a marker, not a cause.

    python canon_frame/analysis/paired_strata.py --ctrl runCanon:gate_off --arm runLat04:complete_reg
"""
import argparse, csv, glob, os, re, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(REPO)
sys.path.insert(0, "icra/isaac_graspgen/source")
from graspgen import config as gc                                  # noqa: E402
from graspgen.uncertainty_gate import observed_support             # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--ctrl", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); O = "icra/isaac_graspgen/output/graspgen"


def load(tag, arm, seed):
    verdict = {}
    for p in glob.glob(f"{O}/{tag}_grasp_log*.csv"):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["executed"] == "1" and r["lift_success"] in ("0", "1"):
                    verdict[(r["trial"], int(r["grasp_id"]))] = (int(r["lift_success"]), r.get("support_d_obs_mm", ""))
    out = {}
    for f in glob.glob(f"{O}/{tag}_dump/s{seed}_{arm}_*.npz"):
        m = re.search(rf"s{seed}_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f); d = np.load(f)
        ch = int(np.ravel(d["chosen"])[0]); trial = os.path.splitext(os.path.basename(f))[0]
        if not m or ch < 0 or (trial, ch) not in verdict or "completed" not in d.files: continue
        ok, dc = verdict[(trial, ch)]; comp = d["completed"].astype(float)
        if dc: dob = float(dc)
        elif "d_obs" in d.files: dob = float(d["d_obs"][ch]) * 1000
        else: dob = float(observed_support(d["grasps"][ch:ch + 1].astype(float), comp, comp[-2048:], depth=gc.PAD_CENTER_DEPTH, region_extents=gc.GRIPPER_CLOSING_REGION)[0]) * 1000
        ex = d["executable"].astype(bool); conf = d["confidences"].astype(float)
        out[(m.group(1), int(m.group(2)))] = (ok, dob, int((conf[ex] > conf[ch]).sum()))
    return out


C = load(*a.ctrl.split(":"), a.seed); A = load(*a.arm.split(":"), a.seed); keys = sorted(set(C) & set(A))
c = np.array([C[k] for k in keys]); r = np.array([A[k] for k in keys])
print(f"{a.ctrl} vs {a.arm}: {len(keys)} paired trials; stratum = CONTROL's executed d_obs")
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 1e9)):
    m = (c[:, 1] >= lo) & (c[:, 1] < hi)
    if m.sum():
        print(f"  {lo:>3.0f}-{min(hi, 999):>3.0f} mm: n={int(m.sum()):3d}  ctrl {100 * c[m, 0].mean():5.1f}%  arm {100 * r[m, 0].mean():5.1f}%  "
              f"(arm executed d_obs {r[m, 1].mean():5.1f} mm, conf-rank {r[m, 2].mean():4.1f})")
print(f"  ALL: ctrl {100 * c[:, 0].mean():.1f}%  arm {100 * r[:, 0].mean():.1f}%")
