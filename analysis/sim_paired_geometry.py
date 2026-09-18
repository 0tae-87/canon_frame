"""Why does a completion with lower CD grasp worse? Paired, per-object diagnosis on
the Isaac dumps of one tag: for every (object, trial) present in both the control arm
and a test arm, compare the generated cloud's extent, the number of executable
candidates, and where the EXECUTED grasp sits relative to the OBSERVED surface
(tool centre = base + 0.195 z; nearest observed / generated point). Also the per-object
lift-success table from the CSVs.

    python canon_frame/analysis/sim_paired_geometry.py --tag runCanon --arms complete_canon complete_reg
"""
import argparse, collections, csv, glob, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="runCanon"); ap.add_argument("--control", default="gate_off")
ap.add_argument("--arms", nargs="*", default=["complete_canon", "complete_reg"])
ap.add_argument("--all-arms", nargs="*", default=["gate_off", "partial_only", "complete_canon", "complete_reg"])
a = ap.parse_args()
O = "icra/isaac_graspgen/output/graspgen"; D = f"{O}/{a.tag}_dump"

# ---- success table -------------------------------------------------------------
rows = []
for f in glob.glob(f"{O}/{a.tag}_grasp_log*.csv"):
    with open(f) as fh:
        rows += [r for r in csv.DictReader(fh) if r["executed"] == "1"]
succ = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    m = re.match(r"s\d+_(.+?)_(\d{3}_[a-z_]+?)(?:_t\d+)?$", r["trial"])
    if m:
        succ[m.group(2)][m.group(1)].append(int(r["lift_success"] or 0))
print(f"lift success per object ({a.tag})\n{'object':>22} " + " ".join(f"{x:>17}" for x in a.all_arms))
for o in sorted(succ):
    print(f"{o:>22} " + " ".join((f"{np.mean(succ[o][x]) * 100:6.0f}% (n={len(succ[o][x]):2d})" if succ[o][x] else f"{'-':>17}") for x in a.all_arms))
tot = {x: sum(sum(succ[o][x]) for o in succ) for x in a.all_arms}; n = {x: sum(len(succ[o][x]) for o in succ) for x in a.all_arms}
print(f"{'ALL':>22} " + " ".join((f"{tot[x] / n[x] * 100:6.1f}% (n={n[x]:3d})" if n[x] else f"{'-':>17}") for x in a.all_arms))


# ---- paired geometry -----------------------------------------------------------
def load(arm):
    out = {}
    for f in glob.glob(f"{D}/s*_{arm}_*.npz"):
        m = re.search(rf"s(\d+)_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
        if m:
            out[(int(m.group(1)), m.group(2), int(m.group(3)))] = f
    return out


def stats(f, n_obs=2048):
    d = np.load(f); comp = d["completed"]; gen, obs = comp[:-n_obs], comp[-n_obs:]
    res = {"ext": gen.max(0) - gen.min(0), "exec": int(d["executable"].sum()) if "executable" in d else np.nan}
    g = d["grasps"]; ch = int(np.ravel(d["chosen"])[0]) if d["chosen"].size else -1
    if 0 <= ch < len(g):
        T = g[ch]; tc = T[:3, 3] + 0.195 * T[:3, 2]
        res["d_obs"] = np.linalg.norm(obs - tc, axis=1).min() * 1000
        res["d_gen"] = np.linalg.norm(gen - tc, axis=1).min() * 1000
    return res


A = load(a.control)
for arm in a.arms:
    B = load(arm); keys = sorted(set(A) & set(B))
    if not keys:
        print(f"\n{arm}: no paired dumps yet"); continue
    per = collections.defaultdict(list)
    for k in keys:
        per[k[1]].append((stats(A[k]), stats(B[k])))
    print(f"\n{arm} vs {a.control}: {len(keys)} paired trials")
    print(f"{'object':>22} {'gen extent ratio x/y/z':>24} {'executable ctrl/arm':>20} {'exec grasp -> nearest OBSERVED pt (mm) ctrl/arm':>48} {'-> nearest GEN pt':>18}")
    agg = []
    for o, L in sorted(per.items()):
        r = np.array([b["ext"] / c["ext"] for c, b in L]).mean(0)
        ex = (np.nanmean([c["exec"] for c, b in L]), np.nanmean([b["exec"] for c, b in L]))
        do = (np.nanmean([c.get("d_obs", np.nan) for c, b in L]), np.nanmean([b.get("d_obs", np.nan) for c, b in L]))
        dg = (np.nanmean([c.get("d_gen", np.nan) for c, b in L]), np.nanmean([b.get("d_gen", np.nan) for c, b in L]))
        agg.append((r, ex, do, dg))
        print(f"{o:>22} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f}   {ex[0]:>8.1f}/{ex[1]:<8.1f}   {do[0]:>21.1f}/{do[1]:<21.1f}   {dg[0]:>7.1f}/{dg[1]:<7.1f}")
    r = np.mean([x[0] for x in agg], 0); ex = np.mean([x[1] for x in agg], 0); do = np.mean([x[2] for x in agg], 0); dg = np.mean([x[3] for x in agg], 0)
    print(f"{'MEAN over objects':>22} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f}   {ex[0]:>8.1f}/{ex[1]:<8.1f}   {do[0]:>21.1f}/{do[1]:<21.1f}   {dg[0]:>7.1f}/{dg[1]:<7.1f}")
