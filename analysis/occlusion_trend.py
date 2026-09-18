"""Occlusion trend: lift success of plain completion / partial / v2 at lateral occlusion
0 (full view), 0.25 and 0.4, 16 graspable objects, seed 0, 50 trials per cell.
Sources: 0 -> runCanon (+runCanon_v2) for the original 9, runNewFull for the 7 new;
0.25 -> runOccL025; 0.4 -> runLat04 / runNewLat04 / runLat04_v2.  Windex: upright only
(runNewFull / runNewLat04 / runLat04_v2 / runOccL025).

    python canon_frame/analysis/occlusion_trend.py
"""
import argparse, csv, glob, os, re
import numpy as np
_ap = argparse.ArgumentParser(); _ap.add_argument("--exclude", nargs="*", default=[]); _args = _ap.parse_args()
from math import lgamma, exp
O = "icra/isaac_graspgen/output/graspgen"; CACHE = "canon_frame/results/exec_rows"
ORIG = ["005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can", "011_banana", "021_bleach_cleanser", "025_mug", "035_power_drill", "051_large_clamp", "061_foam_brick"]
NEW = ["003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box", "019_pitcher_base", "052_extra_large_clamp", "022_windex_bottle"]
ALL = [o for o in ORIG + NEW if o not in _args.exclude]; ORIG = [o for o in ORIG if o in ALL]; NEW = [o for o in NEW if o in ALL]


def load(tag, arm, objs, seed=0):
    out = {}
    src = [f"{CACHE}/{tag}.csv"] if os.path.exists(f"{CACHE}/{tag}.csv") else glob.glob(f"{O}/{tag}_grasp_log*.csv")
    for f in src:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["executed"] != "1": continue
                m = re.match(rf"s{seed}_(.+?)_(\d{{3}}_[a-z_]+?)_t(\d+)$", r["trial"])
                if m and m.group(1) == arm and m.group(2) in objs: out[(m.group(2), int(m.group(3)))] = int(r["lift_success"] or 0)
    return out


def mcnemar(A, B):
    ks = set(A) & set(B); b = sum(1 for k in ks if A[k] == 1 and B[k] == 0); c = sum(1 for k in ks if A[k] == 0 and B[k] == 1); n = b + c
    return b, c, (min(1.0, 2 * sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * np.log(2)) for i in range(0, min(b, c) + 1))) if n else 1.0)


levels = {
    "0 (full view)": {"plain": {**load("runCanon", "gate_off", ORIG), **load("runNewFull", "gate_off", NEW)},
                      "partial": {**load("runCanon", "partial_only", ORIG), **load("runNewFull", "partial_only", NEW)},
                      "v2": {**load("runCanon_v2", "complete_v2", ORIG), **load("runNewFull", "complete_v2", NEW)}},
    "0.25": {"plain": load("runOccL025", "gate_off", ALL), "partial": load("runOccL025", "partial_only", ALL), "v2": load("runOccL025", "complete_v2", ALL)},
    "0.4": {"plain": {**load("runLat04", "gate_off", ORIG), **load("runNewLat04", "gate_off", NEW)},
            "partial": {**load("runLat04", "partial_only", ORIG), **load("runNewLat04", "partial_only", NEW)},
            "v2": load("runLat04_v2", "complete_v2", ALL)},
}
N = 50 * len(ALL)
print("| lateral occlusion | plain completion | partial | v2 | v2 − plain | v2 − partial |\n|---|---|---|---|---|---|")
for lv, arms in levels.items():
    S = {a: sum(v.values()) for a, v in arms.items()}; n = {a: len(v) for a, v in arms.items()}
    if min(n.values()) < N * 0.9: print(f"| {lv} | (incomplete: {n}) |"); continue
    bp, cp, pp = mcnemar(arms["plain"], arms["v2"]); bq, cq, pq = mcnemar(arms["partial"], arms["v2"])
    print(f"| {lv} | {100 * S['plain'] / N:.1f} | {100 * S['partial'] / N:.1f} | {100 * S['v2'] / N:.1f} | {100 * (S['v2'] - S['plain']) / N:+.1f} (p {pp:.2g}) | {100 * (S['v2'] - S['partial']) / N:+.1f} (p {pq:.2g}) |")
print(f"\n({len(ALL)} objects, seed 0, n = {N} per arm and level; McNemar on identical poses)")
print("\nper object (plain / partial / v2 at 0 | 0.25 | 0.4):")
for o in ALL:
    cells = []
    for lv, arms in levels.items():
        cells.append("/".join(f"{100 * sum(v for k, v in arms[a].items() if k[0] == o) / 50:.0f}" for a in ("plain", "partial", "v2")))
    print(f"  {o:>22}  " + "  |  ".join(cells))
