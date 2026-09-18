"""Assemble the lift-success table across all tags: view x arm x object, strict counting
(50 trials per cell, no-candidate = failure; done markers define which cells exist).
Pairwise McNemar / Fisher for the pooled rows. Prints markdown.

    python canon_frame/analysis/results_table.py
"""
import csv, glob, os, re, collections
import numpy as np
from math import lgamma, exp
O = "icra/isaac_graspgen/output/graspgen"
# (tag, arm label in CSV/dump names) -> (view, display arm)
SOURCES = {
    ("runCanon", "gate_off"): ("full", "complete_only"), ("runCanon", "partial_only"): ("full", "partial_only"),
    ("runCanon", "complete_reg"): ("full", "reg_v1"), ("runCanon_v2", "complete_v2"): ("full", "reg_v2"),
    ("runNewFull", "gate_off"): ("full", "complete_only"), ("runNewFull", "partial_only"): ("full", "partial_only"),
    ("runNewFull", "complete_v2"): ("full", "reg_v2"),
    ("runLat04", "gate_off"): ("lat40", "complete_only"), ("runLat04", "partial_only"): ("lat40", "partial_only"),
    ("runLat04", "complete_reg"): ("lat40", "reg_v1"), ("runLat04_v2", "complete_v2"): ("lat40", "reg_v2"),
    ("runNewLat04", "gate_off"): ("lat40", "complete_only"), ("runNewLat04", "partial_only"): ("lat40", "partial_only"),
    ("runThinLat04", "gate_off"): ("lat40", "complete_only"), ("runThinLat04", "partial_only"): ("lat40", "partial_only"),
    ("runLat04_s1", "gate_off"): ("lat40_s1", "complete_only"), ("runLat04_s1", "partial_only"): ("lat40_s1", "partial_only"),
    ("runLat04_s1", "complete_reg"): ("lat40_s1", "reg_v1"), ("runLat04_v2_s1", "complete_v2"): ("lat40_s1", "reg_v2"),
}
ARMS = ["complete_only", "partial_only", "reg_v1", "reg_v2"]
# windex before the up-axis fix (tags run before 2026-09-16 11:00) was lying on its side; label it
LYING = {"runCanon", "runLat04", "runLat04_s1", "runLat04_v2"}   # runLat04_v2 ran after the fix? it launched 12:01 -> after fix (11:xx) -> upright
LYING.discard("runLat04_v2")

cells = collections.defaultdict(dict)      # (view, arm) -> {(obj, trial): success}
present = collections.defaultdict(set)     # (view, arm) -> objects with a done marker
for (tag, arm), (view, disp) in SOURCES.items():
    if not os.path.isdir(f"{O}/{tag}_done"): continue
    for d in os.listdir(f"{O}/{tag}_done"):
        m = re.match(rf"s\d+_{arm}_(\d{{3}}_[a-z_]+)$", d)
        if m:
            name = m.group(1) + ("*" if (name_ := m.group(1)) == "022_windex_bottle" and tag in LYING else "")
            present[(view, disp)].add(name)
    cache = f"canon_frame/results/exec_rows/{tag}.csv"
    for f in ([cache] if os.path.exists(cache) else glob.glob(f"{O}/{tag}_grasp_log*.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["executed"] != "1": continue
                m = re.match(rf"s(\d+)_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)$", r["trial"])
                if not m: continue
                name = m.group(2) + ("*" if m.group(2) == "022_windex_bottle" and tag in LYING else "")
                cells[(view, disp)][(name, int(m.group(3)))] = int(r["lift_success"] or 0)


def fisher(a, b, c, d):
    def logp(x): return lgamma(a + b + 1) - lgamma(x + 1) - lgamma(a + b - x + 1) + lgamma(c + d + 1) - lgamma(a + c - x + 1) - lgamma(c + d - (a + c - x) + 1) - (lgamma(a + b + c + d + 1) - lgamma(a + c + 1) - lgamma(b + d + 1))
    p0 = logp(a); return min(1.0, sum(exp(logp(x)) for x in range(max(0, a - d), min(a + b, a + c) + 1) if logp(x) <= p0 + 1e-9))


def mcnemar(A, B):
    ks = set(A) & set(B); b = sum(1 for k in ks if A[k] == 1 and B[k] == 0); c = sum(1 for k in ks if A[k] == 0 and B[k] == 1); n = b + c
    return b, c, (min(1.0, 2 * sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * np.log(2)) for i in range(0, min(b, c) + 1))) if n else 1.0)


for view in ("full", "lat40", "lat40_s1"):
    objs = sorted(set().union(*[present[(view, a)] for a in ARMS]))
    if not objs: continue
    print(f"\n### view = {view}   (success %, 50 trials per cell; '-' = not run; windex* = lying on its side, pre-fix)\n")
    print("| object | " + " | ".join(ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for o in objs:
        row = []
        for a in ARMS:
            if o not in present[(view, a)]: row.append("-"); continue
            s = sum(v for k, v in cells[(view, a)].items() if k[0] == o); row.append(f"{100 * s / 50:.0f}")
        print(f"| {o} | " + " | ".join(row) + " |")
    common = [o for o in objs if all(o in present[(view, a)] for a in ARMS)]
    if common:
        S = {a: sum(v for k, v in cells[(view, a)].items() if k[0] in common) for a in ARMS}; N = 50 * len(common)
        print(f"| **ALL ({len(common)} common objects, n={N})** | " + " | ".join(f"**{100 * S[a] / N:.1f}**" for a in ARMS) + " |")
        for a in ("complete_only", "partial_only", "reg_v1"):
            A = {k: v for k, v in cells[(view, a)].items() if k[0] in common}; B = {k: v for k, v in cells[(view, "reg_v2")].items() if k[0] in common}
            b, c, p = mcnemar(A, B)
            print(f"  reg_v2 vs {a}: {100 * (S['reg_v2'] - S[a]) / N:+.1f} pp, Fisher p={fisher(S['reg_v2'], N - S['reg_v2'], S[a], N - S[a]):.2g}, McNemar {b}/{c} p={p:.2g}")
