"""Does the canonical-frame regressor (center_reg_v2, frozen) transfer to a different completion network without retraining?
Offline mesh benchmark with the official SeedFormer ShapeNet-55 checkpoint (github.com/hrzhou2/seedformer).

Observations: the same stored Isaac observations as sim_bench_v2 (10 objects x 15 per level; runCanon = full view,
runLat04 = 40 % lateral occluder), i.e. the 2048-point observed tail of each dump = exactly what the server saw, world metres.
Three input frames per observation, same 2048 points, same network, weights frozen:
  bbox : partial bbox centre + max radius, world up -> ShapeNet up      (what the completion server does)
  v2   : bbox frame, then the regressor's centre shift / log-scale       (the method under test)
  gt   : full-mesh mean centre + max radius at the trial pose            (diagnostic reference only; needs the mesh)
SeedFormer's ShapeNet-55 training frame is pc_norm = mean centroid + max radius of the full cloud (identical to PoinTr's),
which is the frame the regressor was trained to recover, so the module is used unchanged. Output = final upsampling stage
(8192 points, up_factors [1,4,4]); all points are network outputs (P0 = FPS(seeds U partial) is refined by every layer, no
verbatim observed tail is appended) -- verified per cloud by the min output-to-input distance recorded in the CSV.
Completion is mapped back to world metres through the same frame it went in by; CD-L1 (mm) = mean of both one-sided
mean NN distances vs 8192 mesh samples (sim_cd.py, same rng), precision = gen->mesh, recall = mesh->gen.

    docker run --rm --gpus all -v /home/wim/Desktop/yt_ws:/workspace -w /workspace/PoinTr pointr_blackwell:gpufix \
        python canon_frame/analysis/sim_bench_seedformer.py --per-cell 15
"""
import argparse, csv, glob, json, os, re, sys, time
import numpy as np, torch

REPO = "/workspace/PoinTr" if os.path.isdir("/workspace/PoinTr") else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WS = os.path.dirname(REPO); SEED_ROOT = os.path.join(WS, "seedformer", "codes")
SF_CKPT = os.path.join(WS, "seedformer", "pretrained", "models", "shapenet55", "ShapeNet-55", "ckpt-best.pth")
os.chdir(REPO)

# --- SeedFormer first (its package names `models`/`utils` clash with PoinTr's; import, keep the class, purge the names)
sys.path.insert(0, SEED_ROOT)
import model as _sf                                                  # noqa: E402
SeedFormerNet = _sf.seedformer_dim128
for k in [m for m in sys.modules if m == "models" or m.startswith("models.") or m == "model"]:
    del sys.modules[k]
sys.path.remove(SEED_ROOT)
# --- PoinTr side
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "canon_frame", "canon")); sys.path.insert(0, os.path.join(REPO, "canon_frame", "analysis"))
from train_center_reg import CenterNet                              # noqa: E402
import sim_cd                                                        # noqa: E402  (arg parsing guarded; defines nn/pose/gt_for/UP; chdir REPO)
from sim_cd import nn, pose, gt_for, UP                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-cell", type=int, default=15); ap.add_argument("--arm", default="gate_off")
ap.add_argument("--levels", nargs="+", default=["runCanon=0", "runLat04=0.4"])
ap.add_argument("--reg", default="canon_frame/ckpts/center_reg_v2.pth"); ap.add_argument("--sf-ckpt", default=SF_CKPT)
ap.add_argument("--out", default="canon_frame/results/seedformer_bench"); ap.add_argument("--solo-radius", type=float, default=0.50); ap.add_argument("--pedestal", type=float, default=0.25)
ap.add_argument("--up-override", nargs="*", default=["022_windex_bottle=-y"], help="same as sim_bench.py for these pre-fix dumps")
a = ap.parse_args()
for kv in a.up_override:
    k, v = kv.split("="); UP[k] = v
O = "icra/isaac_graspgen/output/graspgen"; dev = "cuda"; os.makedirs(a.out, exist_ok=True)


def align_rotation(src, dst):
    s = np.asarray(src, float); d = np.asarray(dst, float); s /= np.linalg.norm(s); d /= np.linalg.norm(d)
    v = np.cross(s, d); c = float(np.dot(s, d))
    if np.linalg.norm(v) < 1e-9: return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


R_ALIGN = align_rotation((0, 0, 1), (0, 1, 0)).astype(np.float32)      # world z-up -> ShapeNet y-up (completion_server config)

# --- models
sf = SeedFormerNet(up_factors=[1, 4, 4]).to(dev).eval()
ck = torch.load(a.sf_ckpt, map_location="cpu"); sd = {k.replace("module.", "", 1): v for k, v in ck["model"].items()}
missing, unexpected = sf.load_state_dict(sd, strict=True), None
rk = torch.load(a.reg, map_location="cpu"); reg = CenterNet(rk["width"]).to(dev).eval(); reg.load_state_dict(rk["model"])
meta = {"seedformer_ckpt": os.path.relpath(a.sf_ckpt, WS), "seedformer_epoch_index": int(ck["epoch_index"]), "seedformer_best_metric": float(ck["best_metrics"]),
        "seedformer_arch": "seedformer_dim128 up_factors [1,4,4] (ShapeNet-55 config), 8192 output points",
        "regressor_ckpt": a.reg, "regressor_epoch": int(rk["epoch"]), "regressor_heldout_centre_err_r": float(rk["err_pred"]),
        "input_points": 2048, "gt_samples": 8192, "levels": a.levels, "per_cell": a.per_cell, "arm": a.arm, "observations": []}
print(f"[sf] SeedFormer ShapeNet-55 ckpt epoch {ck['epoch_index']} (best metric {ck['best_metrics']:.3f}); regressor v2 epoch {rk['epoch']}", flush=True)


@torch.no_grad()
def run_sf(x_unit):
    """x_unit (2048,3) numpy in the network frame -> (8192,3) numpy, ms"""
    torch.manual_seed(0)
    t = torch.from_numpy(np.ascontiguousarray(x_unit, dtype=np.float32)).unsqueeze(0).to(dev)
    torch.cuda.synchronize(); t0 = time.perf_counter(); out = sf(t)[-1][0]; torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.float64), (time.perf_counter() - t0) * 1000


@torch.no_grad()
def run_reg(u):
    t = torch.from_numpy(np.ascontiguousarray(u, dtype=np.float32)).unsqueeze(0).to(dev)
    torch.cuda.synchronize(); t0 = time.perf_counter(); o = reg(t)[0]; torch.cuda.synchronize()
    return o[:3].cpu().numpy().astype(np.float64), float(torch.exp(o[3])), (time.perf_counter() - t0) * 1000


# warm-up (CUDA context / cuDNN autotune) so the recorded ms are steady-state
with torch.no_grad():
    _w = torch.zeros(1, 2048, 3, device=dev); sf(_w); reg(_w); torch.cuda.synchronize()

rows = []; FIELDS = ["level", "tag", "seed", "object", "trial", "cond", "cd_l1_mm", "prec_mm", "recall_mm", "bbox_centre_err_mm", "radius_ratio",
                     "frame_centre_err_mm", "frame_scale_ratio", "net_ms", "reg_ms", "n_out", "min_out_to_in_mm", "pose"]
for spec in a.levels:
    tag, lvl = spec.split("="); per = {}
    for f in sorted(glob.glob(f"{O}/{tag}_dump/s*_{a.arm}_*.npz")):
        m = re.search(rf"s(\d+)_{a.arm}_(\d{{3}}_[a-z_]+)_t(\d+)\.npz$", f)
        if not m: continue
        seed, name, trial = int(m.group(1)), m.group(2), int(m.group(3))
        if per.get(name, 0) >= a.per_cell: continue
        d = np.load(f)
        obs = d["observed"].astype(np.float64) if "observed" in d.files else (d["completed"][-2048:].astype(np.float64) if "completed" in d.files else None)
        if obs is None or len(obs) != 2048: continue
        per[name] = per.get(name, 0) + 1; meta["observations"].append(os.path.relpath(f, REPO))
        gts, gfull = gt_for(name); t_, R, how = pose(d, name, seed, trial)
        if t_ is None:
            t_ = np.array([a.solo_radius, 0.0, 0.0]); t_[2] = a.pedestal - (gfull @ R.T)[:, 2].min()
        gw = gts @ R.T + t_; gc = (gw.min(0) + gw.max(0)) / 2; gr = np.linalg.norm(gw - gc, axis=1).max()
        c_gt = gw.mean(0); s_gt = np.linalg.norm(gw - c_gt, axis=1).max()          # SeedFormer / PoinTr training frame (pc_norm)
        # server frame: bbox centre, max radius, up-axis alignment
        cb = (obs.min(0) + obs.max(0)) / 2; cen = obs - cb; sb = float(np.sqrt((cen ** 2).sum(1)).max()); unit = (cen / sb) @ R_ALIGN.T
        frames = {}
        frames["bbox"] = (unit, cb, sb, np.zeros(3), 1.0, 0.0)
        c_r, s_r, reg_ms = run_reg(unit); frames["v2"] = ((unit - c_r) / s_r, cb, sb, c_r, s_r, reg_ms)
        frames["gt"] = (((obs - c_gt) / s_gt) @ R_ALIGN.T, c_gt, s_gt, np.zeros(3), 1.0, 0.0)
        for cond, (x, c0, s0, c_r_, s_r_, rms) in frames.items():
            gen_u, net_ms = run_sf(x)
            gen_b = gen_u * s_r_ + c_r_                                   # back to the frame the network saw (bbox or gt frame)
            gen = (gen_b @ R_ALIGN) * s0 + c0                             # world metres
            c_used = c0 + (c_r_ @ R_ALIGN) * s0; s_used = s0 * s_r_        # world-frame centre / radius implied by the condition
            p = nn(gen, gw) * 1000; rc = nn(gw, gen) * 1000
            cc = (gen.min(0) + gen.max(0)) / 2; cr = np.linalg.norm(gen - cc, axis=1).max()
            d_in = nn(gen, obs).min() * 1000
            rows.append(dict(level=lvl, tag=tag, seed=seed, object=name, trial=trial, cond=cond, cd_l1_mm=0.5 * (p.mean() + rc.mean()), prec_mm=p.mean(), recall_mm=rc.mean(),
                             bbox_centre_err_mm=np.linalg.norm(cc - gc) * 1000, radius_ratio=cr / gr, frame_centre_err_mm=np.linalg.norm(c_used - c_gt) * 1000,
                             frame_scale_ratio=s_used / s_gt, net_ms=net_ms, reg_ms=rms, n_out=len(gen), min_out_to_in_mm=d_in, pose=how))
        if sum(per.values()) % 10 == 0: print(f"  {tag}: {sum(per.values())} observations done", flush=True)
    print(f"[{tag}] {sum(per.values())} observations over {len(per)} objects", flush=True)

with open(os.path.join(a.out, "rows.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); [w.writerow(r) for r in rows]
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)

# --- summary
def agg(rs, key): return float(np.mean([r[key] for r in rs])) if rs else float("nan")
lines = ["# SeedFormer (official ShapeNet-55 ckpt, frozen) + canonical-frame regressor v2 (frozen): offline mesh benchmark", "",
         f"SeedFormer ckpt `{meta['seedformer_ckpt']}` (epoch {meta['seedformer_epoch_index']}), regressor `{a.reg}` (epoch {rk['epoch']}); "
         f"{len(meta['observations'])} observations, 2048 input points each, 8192 output points, CD vs 8192 mesh samples (mm). "
         "Conditions: bbox = server frame; v2 = bbox + regressor; gt = full-mesh mean centre / max radius (diagnostic reference).", ""]
for spec in a.levels:
    tag, lvl = spec.split("="); R_ = [r for r in rows if r["tag"] == tag]
    lines += [f"## lateral occlusion {lvl} ({tag}, n = {len(R_) // 3} observations)", "",
              "| frame | CD-L1 | precision (gen→mesh) | recall (mesh→gen) | frame centre err (mm) | frame scale ratio | bbox-centre err of output (mm) | net ms | extra reg ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in ("bbox", "v2", "gt"):
        rs = [r for r in R_ if r["cond"] == cond]
        lines.append(f"| {cond} | {agg(rs, 'cd_l1_mm'):.2f} | {agg(rs, 'prec_mm'):.2f} | {agg(rs, 'recall_mm'):.2f} | {agg(rs, 'frame_centre_err_mm'):.1f} | {agg(rs, 'frame_scale_ratio'):.3f} | {agg(rs, 'bbox_centre_err_mm'):.1f} | {agg(rs, 'net_ms'):.1f} | {agg(rs, 'reg_ms'):.2f} |")
    b = agg([r for r in R_ if r["cond"] == "bbox"], "cd_l1_mm"); v = agg([r for r in R_ if r["cond"] == "v2"], "cd_l1_mm"); g = agg([r for r in R_ if r["cond"] == "gt"], "cd_l1_mm")
    wins = sum(1 for o in sorted({r["object"] for r in R_}) for tr in sorted({r["trial"] for r in R_ if r["object"] == o})
               if next(r["cd_l1_mm"] for r in R_ if r["object"] == o and r["trial"] == tr and r["cond"] == "v2") < next(r["cd_l1_mm"] for r in R_ if r["object"] == o and r["trial"] == tr and r["cond"] == "bbox"))
    lines += ["", f"v2 − bbox: **{v - b:+.2f} mm** ({100 * (v - b) / b:+.1f} %); gt − bbox: {g - b:+.2f} mm; v2 closes {100 * (b - v) / (b - g) if b != g else float('nan'):.0f} % of the bbox→gt gap; "
              f"v2 better than bbox on {wins}/{len(R_) // 3} observations.", "", "| object | bbox | v2 | gt | v2 − bbox | frame centre err bbox → v2 (mm) |", "|---|---:|---:|---:|---:|---:|"]
    for o in sorted({r["object"] for r in R_}):
        ro = {c: [r for r in R_ if r["object"] == o and r["cond"] == c] for c in ("bbox", "v2", "gt")}
        lines.append(f"| {o} | {agg(ro['bbox'], 'cd_l1_mm'):.1f} | {agg(ro['v2'], 'cd_l1_mm'):.1f} | {agg(ro['gt'], 'cd_l1_mm'):.1f} | {agg(ro['v2'], 'cd_l1_mm') - agg(ro['bbox'], 'cd_l1_mm'):+.1f} | {agg(ro['bbox'], 'frame_centre_err_mm'):.0f} → {agg(ro['v2'], 'frame_centre_err_mm'):.0f} |")
    lines.append("")
mn = min(r["min_out_to_in_mm"] for r in rows)
lines += [f"Verbatim-copy check: min distance from any output point to the nearest input point over all clouds = {mn:.3f} mm (> 0 ⇒ no observed points are appended to SeedFormer's output; all 8192 points are generated).", ""]
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines)); print("\n".join(lines))
