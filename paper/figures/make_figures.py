"""Figures for the IPIU paper (canon_frame study). Everything is computed from canon_frame/results (executed-trial rows,
bench log, diagnosis json, training log); numbers reproduce results/*.md.

    docker run --rm -v /home/wim/Desktop/yt_ws:/workspace -w /workspace/PoinTr pointr_blackwell:gpufix \
        python canon_frame/paper/figures/make_figures.py
"""
import csv, json, os, re
from math import lgamma, exp
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results"); ROWS = os.path.join(DATA, "exec_rows")   # canon_frame/results
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C = {"plain": "#7f7f7f", "partial": "#1f77b4", "v2": "#d62728", "v1": "#ff7f0e"}
LBL = {"plain": "completion (plain frame)", "partial": "partial only", "v2": "completion + frame regressor (v2)", "v1": "regressor v1"}
ORIG = ["005_tomato_soup_can", "006_mustard_bottle", "010_potted_meat_can", "011_banana", "021_bleach_cleanser", "025_mug", "035_power_drill", "051_large_clamp", "061_foam_brick"]
NEW = ["003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box", "019_pitcher_base", "052_extra_large_clamp", "022_windex_bottle"]
CLAMPS = ["051_large_clamp", "052_extra_large_clamp"]
SHORT = lambda o: o.split("_", 1)[1].replace("_", " ")


def load(tag, arm, objs, seed=0):
    out = {}
    with open(os.path.join(ROWS, f"{tag}.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["executed"] != "1": continue
            m = re.match(rf"s{seed}_(.+?)_(\d{{3}}_[a-z_]+?)_t(\d+)$", r["trial"])
            if m and m.group(1) == arm and m.group(2) in objs: out[(m.group(2), int(m.group(3)))] = int(r["lift_success"] or 0)
    return out


def mcnemar(A, B):
    ks = set(A) & set(B); b = sum(1 for k in ks if A[k] == 1 and B[k] == 0); c = sum(1 for k in ks if A[k] == 0 and B[k] == 1); n = b + c
    p = min(1.0, 2 * sum(exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * np.log(2)) for i in range(0, min(b, c) + 1))) if n else 1.0
    return b, c, p


def levels(exclude=()):
    orig = [o for o in ORIG if o not in exclude]; new = [o for o in NEW if o not in exclude]; allo = orig + new
    return {"0": {"plain": {**load("runCanon", "gate_off", orig), **load("runNewFull", "gate_off", new)},
                  "partial": {**load("runCanon", "partial_only", orig), **load("runNewFull", "partial_only", new)},
                  "v2": {**load("runCanon_v2", "complete_v2", orig), **load("runNewFull", "complete_v2", new)}},
            "0.25": {a: load("runOccL025", t, allo) for a, t in (("plain", "gate_off"), ("partial", "partial_only"), ("v2", "complete_v2"))},
            "0.4": {"plain": {**load("runLat04", "gate_off", orig), **load("runNewLat04", "gate_off", new)},
                    "partial": {**load("runLat04", "partial_only", orig), **load("runNewLat04", "partial_only", new)},
                    "v2": load("runLat04_v2", "complete_v2", allo)}}


def n_of(d): return 50 * len({k[0] for k in d})                       # 50 trials per object; a trial with no executed grasp counts as a failure
def rate(d): return 100 * sum(d.values()) / n_of(d) if d else np.nan
def wilson(d):
    n = n_of(d); p = sum(d.values()) / n; z = 1.96; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- Fig 1: occlusion trend (16 objects | main 14)
def fig_trend():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    for ax, (title, ex) in zip(axes, (("all 16 graspable objects", ()), ("main set: 14 objects (clamps excluded)", tuple(CLAMPS)))):
        L = levels(ex); xs = [0, 25, 40]
        for arm in ("plain", "partial", "v2"):
            ys = [rate(L[k][arm]) for k in ("0", "0.25", "0.4")]; ci = np.array([wilson(L[k][arm]) for k in ("0", "0.25", "0.4")])
            ax.errorbar(xs, ys, yerr=[np.array(ys) - ci[:, 0], ci[:, 1] - np.array(ys)], marker="o", ms=4, lw=1.8, capsize=2, color=C[arm], label=LBL[arm])
        for k, x in zip(("0", "0.25", "0.4"), xs):
            _, _, p1 = mcnemar(L[k]["v2"], L[k]["plain"]); _, _, p2 = mcnemar(L[k]["v2"], L[k]["partial"])
            ax.annotate(f"p {p1:.2g} | {p2:.2g}", (x, rate(L[k]["v2"])), textcoords="offset points", xytext=(6, 8) if x else (6, -12), ha="left", fontsize=6.5, color=C["v2"])
        ax.set_xticks(xs); ax.set_xticklabels(["0 (full view)", "25 %", "40 %"]); ax.set_xlim(-6, 52); ax.set_title(title, fontsize=9)
        n = n_of(L["0.4"]["v2"]); ax.text(0.02, 0.03, f"n = {n} trials per arm and level, seed 0\nWilson 95 % CI; McNemar on identical poses", transform=ax.transAxes, fontsize=6.5, color="#555")
    axes[0].set_ylabel("lift success (%)"); axes[0].set_ylim(30, 82); axes[0].legend(fontsize=7, loc="upper right", frameon=False)
    fig.supxlabel("lateral occlusion (share of observed points hidden);  p = McNemar of v2 vs plain | vs partial", fontsize=8)
    save(fig, "fig1_occlusion_trend")


# ---------------------------------------------------------------- Fig 2: per-object at 40 % occlusion (main 14)
def fig_per_object():
    L = levels(tuple(CLAMPS))["0.4"]; objs = sorted({k[0] for k in L["v2"]}, key=lambda o: rate({k: v for k, v in L["v2"].items() if k[0] == o}) - rate({k: v for k, v in L["plain"].items() if k[0] == o}), reverse=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.7)); w = 0.27; x = np.arange(len(objs))
    for i, arm in enumerate(("plain", "partial", "v2")):
        ax.bar(x + (i - 1) * w, [rate({k: v for k, v in L[arm].items() if k[0] == o}) for o in objs], w, color=C[arm], label=LBL[arm])
    ax.set_xticks(x); ax.set_xticklabels([SHORT(o) for o in objs], rotation=35, ha="right", fontsize=7.5); ax.set_ylabel("lift success (%)"); ax.set_ylim(0, 100)
    ax.set_title("40 % lateral occlusion, per object (50 trials per cell, seed 0), sorted by v2 − plain", fontsize=9); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    save(fig, "fig2_per_object_lat40")


# ---------------------------------------------------------------- Fig 3: mechanism — mesh CD and centre error on the sim bench
def fig_bench():
    txt = open(os.path.join(DATA, "sim_bench_v2.log")).read().split("=== ")
    blocks = {b.split("\n", 1)[0].strip(): b for b in txt if b.strip()}
    res = {}
    for name, key in (("runCanon", "full view"), ("runLat04", "40 % occlusion")):
        allrow = [l for l in blocks[name].splitlines() if l.strip().startswith("ALL")][0].split()
        # columns: ALL n | base: cd mean min max c centre r ratio | v1: ... | v2: ...
        vals = [float(v) for v in allrow[2:] if re.match(r"^-?\d+(\.\d+)?$", v)]
        # per arm 5 numbers: cd_mean, cd_min, cd_max, centre_err, scale_ratio
        arms = {"plain": vals[0:5], "v1": vals[5:10], "v2": vals[10:15]}
        ax_line = [l for l in blocks[name].splitlines() if l.startswith("bbox-centre error by axis")][0]
        axes_err = {a: [float(v) for v in re.search(rf"{t}: ([\d. ]+)", ax_line).group(1).split()] for a, t in (("plain", "base"), ("v1", "v1"), ("v2", "v2"))}
        res[key] = (arms, axes_err)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), constrained_layout=True)
    keys = list(res); x = np.arange(len(keys)); w = 0.26
    for i, arm in enumerate(("plain", "v1", "v2")):
        axes[0].bar(x + (i - 1) * w, [res[k][0][arm][0] for k in keys], w, color=C[arm], label=LBL[arm])
        axes[1].bar(x + (i - 1) * w, [res[k][0][arm][3] for k in keys], w, color=C[arm])
    axes[0].set_xticks(x); axes[0].set_xticklabels(keys); axes[0].set_ylabel("completion-to-mesh CD (mm)"); axes[0].set_title("completion accuracy", fontsize=9); axes[0].set_ylim(0, 16); axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    axes[1].set_xticks(x); axes[1].set_xticklabels(keys); axes[1].set_ylabel("bbox-centre error (mm)"); axes[1].set_title("frame (centre) error", fontsize=9)
    ax = axes[2]; lab = ["depth", "lateral", "up"]; xx = np.arange(3)
    for i, arm in enumerate(("plain", "v1", "v2")):
        ax.bar(xx + (i - 1) * w, res["40 % occlusion"][1][arm], w, color=C[arm])
    ax.set_xticks(xx); ax.set_xticklabels(lab); ax.set_ylabel("|centre error| (mm)"); ax.set_title("40 % occlusion, by axis", fontsize=9)
    fig.suptitle("sim bench: stored Isaac observations re-completed through each server; CD against the YCB mesh surface (10 objects × 15 observations)", fontsize=7, color="#555", y=-0.02)
    save(fig, "fig3_bench_mechanism")


# ---------------------------------------------------------------- Fig 4: diagnosis — yaw vs centring
def fig_diagnosis():
    A = json.load(open(os.path.join(DATA, "yaw_sensitivity_A_summary.json"))); N = json.load(open(os.path.join(DATA, "norm_yaw_offline.json")))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), constrained_layout=True)
    ax = axes[0]; yaw = A["yaw_deg"]; cd = np.array(A["cd_by_yaw"]); epi = np.array(A["epi_by_yaw"])
    ax.plot(yaw, 100 * (cd / cd[0] - 1), "o-", ms=3, color="#333", label="CD-L1 increase (%)"); ax.set_xlabel("input yaw offset (deg)"); ax.set_ylabel("CD-L1 vs canonical yaw (%)")
    ax2 = ax.twinx(); ax2.plot(yaw, epi / epi[0], "s--", ms=3, color=C["v2"], label="mean epistemic (×)"); ax2.set_ylabel("epistemic (× value at 0°)", color=C["v2"], fontsize=8); ax2.spines["right"].set_visible(True); ax2.tick_params(colors=C["v2"], labelsize=7)
    ax.set_title("(a) yaw offset: CD and epistemic", fontsize=8.5); ax.set_xticks([0, 90, 180, 270]); r = np.corrcoef(cd, epi)[0, 1]; ax.text(0.03, 0.9, f"r = {r:.2f}\n{A['n']} held-out clouds, 3 seeds", transform=ax.transAxes, fontsize=6.5)
    ax = axes[1]; S = N["summary"]; names = ["GT\nframe", "bbox\ncentre", "centroid"]; vals = [S["gt"]["base"], S["bbox"]["base"], S["centroid"]["base"]]
    bars = ax.bar(names, [v * 1e3 for v in vals], color=["#2ca02c", C["plain"], "#9467bd"]); ax.set_ylabel("CD-L1 (×10⁻³, unit sphere)"); ax.set_title("(b) centring cost (100 clouds × 8 yaws)", fontsize=8.5); ax.tick_params(axis="x", labelsize=7.5)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v * 1e3 + 1, f"{100 * (v / vals[0] - 1):+.0f} %", ha="center", fontsize=7)
    ax = axes[2]; off = np.array(N["bbox_offset"]); ax.hist(off, bins=20, color=C["plain"], edgecolor="white"); ax.axvline(off.mean(), color=C["v2"], lw=1.5); ax.text(off.mean() + 0.01, ax.get_ylim()[1] * 0.9, f"mean {off.mean():.2f} r", color=C["v2"], fontsize=7)
    ax.set_xlabel("bbox-centre error of the partial (r)"); ax.set_ylabel("clouds"); ax.set_title("(c) frame error is shape-dependent", fontsize=8.5)
    save(fig, "fig4_diagnosis_yaw_vs_centre")


# ---------------------------------------------------------------- Fig 5: seed replication at 40 % occlusion (8 original objects, clamp excluded)
def fig_replication():
    objs = [o for o in ORIG if o not in CLAMPS]
    runs = {"seed 0": {"plain": load("runLat04", "gate_off", objs, 0), "partial": load("runLat04", "partial_only", objs, 0), "v2": load("runLat04_v2", "complete_v2", objs, 0)},
            "seed 1": {"plain": load("runLat04_s1", "gate_off", objs, 1), "partial": load("runLat04_s1", "partial_only", objs, 1), "v2": load("runLat04_v2_s1", "complete_v2", objs, 1)},
            "seed 1, 2nd draw": {"v2": load("runFinalLat04_s1", "complete_v2", objs, 1)}}
    fig, ax = plt.subplots(figsize=(4.2, 2.6)); x = np.arange(len(runs)); w = 0.26
    for i, arm in enumerate(("plain", "partial", "v2")):
        ys = [rate(runs[k][arm]) if arm in runs[k] else np.nan for k in runs]; ax.bar(x + (i - 1) * w, ys, w, color=C[arm], label=LBL[arm])
        for xi, y in zip(x + (i - 1) * w, ys):
            if not np.isnan(y): ax.text(xi, y + 1, f"{y:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(list(runs)); ax.set_ylabel("lift success (%)"); ax.set_ylim(0, 88); ax.set_title("40 % occlusion, 8 objects × 50 trials, independent draws", fontsize=9); ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=1)
    save(fig, "fig5_seed_replication")


# ---------------------------------------------------------------- Fig 6: regressor training curve
def fig_training():
    ep, pred, bbox = [], [], []
    for l in open(os.path.join(DATA, "center_reg_v2_train.log")):
        m = re.match(r"ep (\d+) loss ([\d.]+)\s+view-partial centre err pred ([\d.]+)\s+bbox ([\d.]+)", l)
        if m: ep.append(int(m.group(1))); pred.append(float(m.group(3))); bbox.append(float(m.group(4)))
    fig, ax = plt.subplots(figsize=(3.6, 2.4)); ax.plot(ep, pred, "o-", ms=3, color=C["v2"], label="regressor v2"); ax.plot(ep, bbox, "--", color=C["plain"], label="bbox centre (no regressor)")
    ax.set_xlabel("epoch"); ax.set_ylabel("held-out centre error (object radii)"); ax.set_ylim(0, 0.2); ax.legend(fontsize=7, frameon=False); ax.set_title("single-view ShapeNet-55 partials", fontsize=9)
    save(fig, "fig6_regressor_training")


# ---------------------------------------------------------------- Fig 0: pipeline schematic
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.6, 2.0)); ax.axis("off")
    boxes = [("single-view\ndepth capture\n(2048 pts)", "#dddddd"), ("bbox normalise\n(centre, radius)", "#f7c6c6"),
             ("frame regressor\n1.4 M PointNet\nΔc, log s", "#ffd9b3"), ("re-normalise\n(predicted frame)", "#ffe8cc"), ("PoinTr + EDL\n(frozen)", "#cfe2f3"), ("completion\n+ epistemic", "#cfe2f3"), ("GraspGen\ngate → Panda", "#d9ead3")]
    n = len(boxes); xs = np.linspace(0.01, 0.99, n + 1)[:-1]; w = 0.98 / n - 0.014
    for i, (t, c) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((xs[i], 0.28), w, 0.46, boxstyle="round,pad=0.01", fc=c, ec="#555", lw=0.8)); ax.text(xs[i] + w / 2, 0.51, t, ha="center", va="center", fontsize=6.6)
        if i < n - 1: ax.add_patch(FancyArrowPatch((xs[i] + w, 0.51), (xs[i + 1], 0.51), arrowstyle="-|>", mutation_scale=8, color="#555", lw=0.8))
    ax.text((xs[2] + xs[3] + w) / 2, 0.1, "added step (≈1 ms): trained on hidden-point-removal single views of ShapeNet-55 + lateral-occluder augmentation", ha="center", fontsize=6.3, color="#a0522d")
    ax.text(xs[1] + w / 2, 0.88, "wrong frame when one side is hidden\n(centre error ≈ 0.2 r, +64 % CD)", ha="center", fontsize=6.3, color="#b22222")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); save(fig, "fig0_pipeline")


if __name__ == "__main__":
    fig_pipeline(); fig_trend(); fig_per_object(); fig_bench(); fig_diagnosis(); fig_replication(); fig_training()
    # numbers behind fig 1 for RESULTS.md cross-check
    for title, ex in (("16", ()), ("14", tuple(CLAMPS))):
        L = levels(ex)
        for k in ("0", "0.25", "0.4"):
            print(f"[{title} objects] occl {k}: plain {rate(L[k]['plain']):.1f} partial {rate(L[k]['partial']):.1f} v2 {rate(L[k]['v2']):.1f} | v2-plain p {mcnemar(L[k]['v2'], L[k]['plain'])[2]:.2g} | v2-partial p {mcnemar(L[k]['v2'], L[k]['partial'])[2]:.2g} | n {len(L[k]['v2'])}")
