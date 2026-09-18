"""When should the (regressor-centred) completion be trusted over the partial?

Data: runLat04 + runLat04_s1 — on identical poses both arms were executed, so for each
pose we know y_c (completion arm lift) and y_p (partial arm lift). Features are computed
from the COMPLETION only (available before any grasp is planned): epistemic statistics,
how far the completion extends beyond the observation, how much of it is unsupported.
Model: predict d = y_c - y_p, choose the completion when d_hat > 0.
Evaluation: leave-one-object-out (does the rule generalise beyond object identity?) and
leave-one-seed-out (does it generalise across poses of the same objects?).

    python canon_frame/analysis/selector_study.py
"""
import csv, glob, os, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(REPO)
O = "icra/isaac_graspgen/output/graspgen"
TAGS = {"runLat04": 0, "runLat04_s1": 1}


def verdicts(tag):
    v = {}
    for p in glob.glob(f"{O}/{tag}_grasp_log*.csv"):
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["executed"] == "1" and r["lift_success"] in ("0", "1"):
                    v[(r["trial"], int(r["grasp_id"]))] = int(r["lift_success"])
    return v


def outcome(tag, arm, seed, v):
    out = {}
    for f in glob.glob(f"{O}/{tag}_dump/s{seed}_{arm}_*.npz"):
        m = re.search(rf"s{seed}_{arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
        if not m: continue
        d = np.load(f); ch = int(np.ravel(d["chosen"])[0]); trial = os.path.splitext(os.path.basename(f))[0]
        if ch < 0 or (trial, ch) not in v: continue
        out[(seed, m.group(1), int(m.group(2)))] = (v[(trial, ch)], f)
    return out


def nn_min(a, b, chunk=512):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        d = ((a[i:i + chunk, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out


def features(f):
    d = np.load(f); comp = d["completed"].astype(float); epi = d["epi"].astype(float)
    gen, obs = comp[:-2048], comp[-2048:]; e = epi[:len(gen)]; R = float(d["obj_radius"])
    eg, eo = gen.max(0) - gen.min(0), obs.max(0) - obs.min(0)
    dn = nn_min(gen, obs) * 1000                                    # generated -> nearest observed (mm)
    en = e / R ** 2
    med = np.median(en)
    return dict(
        epi_mean=en.mean(), epi_p90=np.percentile(en, 90), epi_cv=en.std() / (en.mean() + 1e-12),
        epi_tail=(en > 4 * med).mean(),                              # concentrated hot spots
        ext_ratio=(eg / eo).mean(), ext_ratio_max=(eg / eo).max(),  # how much the completion extends the observation
        unsup20=(dn > 20).mean(), unsup40=(dn > 40).mean(),          # unsupported share of the completion
        dn_mean=dn.mean(),
        epi_unsup=en[dn > 20].mean() if (dn > 20).any() else en.mean(),   # epi where it is unsupported
        n_obs_ratio=len(obs) / 2048.0, radius=R,
    )


rows = []
for tag, seed in TAGS.items():
    v = verdicts(tag); C = outcome(tag, "complete_reg", seed, v); P = outcome(tag, "partial_only", seed, v)
    for k in sorted(set(C) & set(P)):
        rows.append((k[0], k[1], k[2], C[k][0], P[k][0], features(C[k][1])))
names = list(rows[0][5].keys())
X = np.array([[r[5][n] for n in names] for r in rows]); yc = np.array([r[3] for r in rows]); yp = np.array([r[4] for r in rows])
seed = np.array([r[0] for r in rows]); obj = np.array([r[1] for r in rows]); d = yc - yp
objs = sorted(set(obj))
print(f"{len(rows)} paired poses; always-partial {100 * yp.mean():.1f}%  always-completion {100 * yc.mean():.1f}%  "
      f"object oracle {100 * np.mean([max(yc[obj == o].mean(), yp[obj == o].mean()) for o in objs]):.1f}%  pose oracle {100 * np.maximum(yc, yp).mean():.1f}% (noise-inflated)")


def auc(score, target):
    pos, neg = score[target > 0], score[target < 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    dd = pos[:, None] - neg[None, :]; return float((dd > 0).mean() + 0.5 * (dd == 0).mean())


print("\nunivariate: AUC of feature for 'completion wins' (d=+1) vs 'partial wins' (d=-1); pooled and within-object mean")
for i, n in enumerate(names):
    a_pool = auc(X[:, i], d); a_w = np.nanmean([auc(X[obj == o, i], d[obj == o]) for o in objs])
    print(f"  {n:>13}: pooled {a_pool:.3f}  within-object {a_w:.3f}")


def evaluate(model_fn, split_groups, label):
    chosen = np.zeros(len(rows)); 
    for g in sorted(set(split_groups)):
        tr, te = split_groups != g, split_groups == g
        m = model_fn(); m.fit(X[tr], d[tr]); pred = m.predict(X[te])
        chosen[te] = np.where(pred > 0, yc[te], yp[te])
    # per-object choice frequency
    frac_c = np.mean([np.mean([1 for _ in range(1)]) for _ in [0]])
    return chosen.mean()


models = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "gbr": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
}
print(f"\n{'model':>8} {'LOO-object':>11} {'LOO-seed':>9}   (achieved lift success of the selected arm, %)")
for nm, fn in models.items():
    lo = evaluate(fn, obj, nm); ls = evaluate(fn, seed, nm)
    print(f"{nm:>8} {100 * lo:>10.1f}% {100 * ls:>8.1f}%")
# single-feature rules with LOO-object threshold
print("\nsingle-feature rules (choose completion when feature < / > threshold fitted on the other objects):")
for i, n in enumerate(names):
    best = None
    for sign in (+1, -1):
        got = np.zeros(len(rows))
        for o in objs:
            tr, te = obj != o, obj == o
            cands = np.quantile(X[tr, i], np.linspace(0.1, 0.9, 17)); bt = None
            for t in cands:
                pick = (sign * X[tr, i] > sign * t); s = np.where(pick, yc[tr], yp[tr]).mean()
                if bt is None or s > bt[1]: bt = (t, s)
            got[te] = np.where(sign * X[te, i] > sign * bt[0], yc[te], yp[te])
        if best is None or got.mean() > best[1]: best = (sign, got.mean())
    print(f"  {n:>13}: LOO-object {100 * best[1]:.1f}%  (completion when feature {'>' if best[0] > 0 else '<'} thr)")
