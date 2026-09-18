"""Why the epi-argmin yaw canonicalisation works in the GT frame but not
through the server: the server has to re-normalise the PARTIAL (bbox centre,
own max radius) while the network was trained on partials that sit in the
GT's frame. This isolates centre vs scale, and tests a joint argmin-epi search
over yaw x scale x centre-shift as a no-retraining remedy.

    python canon_frame/analysis/norm_yaw_offline.py --n 100
"""
import argparse, json, os, sys, time
import numpy as np, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "icra", "completion_server"))
from easydict import EasyDict                                       # noqa: E402
from utils.config import cfg_from_yaml_file                         # noqa: E402
from datasets.ShapeNet55Dataset import ShapeNet                     # noqa: E402
from new_uncertainty.scripts.train_edl_v11 import make_partial      # noqa: E402
from new_uncertainty.models.evidential_loss_v9 import nig_activate, compute_uncertainty  # noqa: E402
from engine import CompletionEngine                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100); ap.add_argument("--start", type=int, default=3200)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--yaws", type=float, nargs="*", default=[0, 45, 90, 135, 180, 225, 270, 315])
ap.add_argument("--scales", type=float, nargs="*", default=[1.0, 1.15, 1.3], help="multiply partial radius")
ap.add_argument("--shifts", type=float, nargs="*", default=[0.0, 0.1, 0.2], help="centre shift away from camera, x radius")
ap.add_argument("--reg", default="", help="centre regressor ckpt (canon/train_center_reg.py); adds variants reg / reg_c")
ap.add_argument("--no-joint", action="store_true")
ap.add_argument("--out", default="canon_frame/results/norm_yaw_offline")
a = ap.parse_args(); os.chdir(REPO); dev = "cuda"
eng = CompletionEngine(); model = eng.model
K = a.k; yk = np.arange(K) * 2 * np.pi / K
reg = None
if a.reg:
    sys.path.insert(0, os.path.join(REPO, "canon_frame", "canon"))
    from train_center_reg import CenterNet, bbox_norm                # noqa: E402
    ck = torch.load(a.reg, map_location="cpu"); reg = CenterNet(ck["width"]).to(dev).eval(); reg.load_state_dict(ck["model"])
    print(f"regressor: epoch {ck['epoch']} centre err {ck['err_pred']:.4f} (bbox {ck['err_bbox']:.4f})")


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=dev)


RK = torch.stack([rot_y(t) for t in yk])                               # (K,3,3)


@torch.no_grad()
def fwd(x, chunk=36):
    """x (B,N,3) -> gen (B,M,3), score (B,) mean epistemic over generated points"""
    gens, scs = [], []
    for i in range(0, len(x), chunk):
        out = model(x[i:i + chunk]); gens.append(out["gen_points"])
        _, nu, al, be = nig_activate(out["nig_raw"], beta_min=1e-6)
        _, epi = compute_uncertainty(nu, al, be); scs.append(epi.sum(-1).mean(1))
    return torch.cat(gens), torch.cat(scs)


@torch.no_grad()
def cd_l1(p, g, chunk=8):                                              # p (B,M,3), g (M',3)
    outv = []
    for i in range(0, len(p), chunk):
        d = torch.cdist(p[i:i + chunk], g[None].expand(min(chunk, len(p) - i), -1, -1), compute_mode='donot_use_mm_for_euclid_dist')
        outv.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)))
    return torch.cat(outv)


def norm_params(part, center, scale_by):
    if center == "gt": c = torch.zeros(3, device=dev)
    elif center == "bbox": c = (part.min(0).values + part.max(0).values) / 2
    else: c = part.mean(0)
    s = torch.tensor(1.0, device=dev) if scale_by == "gt" else (part - c).norm(dim=1).max()
    return c, s


VARIANTS = [("gt", "gt", "gt"), ("bbox", "bbox", "part"), ("centroid", "centroid", "part"),
            ("bbox_gtscale", "bbox", "gt"), ("gtcenter_pscale", "gt", "part")]


@torch.no_grad()
def variants(part):
    """list of (name, centre, scale) for this partial"""
    out = [(n, *norm_params(part, ce, sc)) for n, ce, sc in VARIANTS]
    if reg is not None:
        u, cb, r = bbox_norm(part[None]); o = reg(u)[0]
        c_reg = cb[0] + r[0] * o[:3]; s_reg = r[0] * torch.exp(o[3])
        out += [("reg", c_reg, s_reg), ("reg_c", c_reg, r[0])]
    return out


NAMES = [n for n, _, _ in VARIANTS] + (["reg", "reg_c"] if a.reg else [])
ds_cfg = cfg_from_yaml_file("cfgs/dataset_configs/ShapeNet-55.yaml")
v = EasyDict(dict(ds_cfg)); v.subset = "test"; ds = ShapeNet(v)
res = {n: {"base": [], "canon": [], "pick_ok": []} for n in NAMES}
if not a.no_joint:
    res["joint"] = {"base": [], "canon": [], "oracle": [], "pick_ok": [], "pick_scale": [], "pick_shift": []}
cerr = {n: [] for n in NAMES}
ratio, boff = [], []
t0 = time.time()
for i in range(a.start, a.start + a.n):
    _, _, gt_np = ds[i]
    gt = torch.as_tensor(np.asarray(gt_np), dtype=torch.float32, device=dev)[None]
    torch.manual_seed(1234 + i); part0 = make_partial(gt, fixed=True, sigma=0.0)[0]
    c_b, s_b = norm_params(part0, "bbox", "part"); ratio.append(float(s_b)); boff.append(float(c_b.norm()))
    for yd in a.yaws:
        R = rot_y(np.radians(yd)); g = gt[0] @ R.T; part = part0 @ R.T
        # --- five normalisations x K yaws in one batch -----------------------
        xs, meta = [], []
        for name, c, s in variants(part):
            u = (part - c) / s; cerr[name].append(float(c.norm()))     # GT centre is the origin
            xs.append(torch.einsum("kij,nj->kni", RK, u)); meta.append((name, c, s))
        gen, score = fwd(torch.cat(xs))
        for vi, (name, c, s) in enumerate(meta):
            gk, sk = gen[vi * K:(vi + 1) * K], score[vi * K:(vi + 1) * K]
            back = torch.einsum("kmi,kij->kmj", gk, RK) * s + c        # undo yaw k, un-normalise
            kb = int(sk.argmin()); cds = cd_l1(back[[0, kb]], g)
            res[name]["base"].append(float(cds[0])); res[name]["canon"].append(float(cds[1]))
            want = (-yd) % 360; got = np.degrees(yk[kb]) % 360
            res[name]["pick_ok"].append(float(min(abs(got - want), 360 - abs(got - want)) <= 180.0 / K + 1e-6))
        if a.no_joint:
            continue
        # --- joint search: yaw x scale x shift (bbox centre, shift away from camera)
        c_b, s_b = norm_params(part, "bbox", "part")
        vdir = part.mean(0) - c_b; vdir = vdir / (vdir.norm() + 1e-9)   # centroid is dragged toward camera
        xs, meta = [], []
        for sf in a.scales:
            for sh in a.shifts:
                c = c_b - sh * s_b * vdir; s = s_b * sf; u = (part - c) / s
                xs.append(torch.einsum("kij,nj->kni", RK, u)); meta.append((sf, sh, c, s))
        gen, score = fwd(torch.cat(xs))
        allcd = []
        for mi, (sf, sh, c, s) in enumerate(meta):
            gk = gen[mi * K:(mi + 1) * K]
            allcd.append(cd_l1(torch.einsum("kmi,kij->kmj", gk, RK) * s + c, g))
        allcd = torch.stack(allcd)                                       # (S*H, K)
        jb = int(score.argmin()); mi, kb = divmod(jb, K)
        res["joint"]["base"].append(float(allcd[0, 0])); res["joint"]["canon"].append(float(allcd[mi, kb]))
        res["joint"]["oracle"].append(float(allcd.min()))
        res["joint"]["pick_scale"].append(meta[mi][0]); res["joint"]["pick_shift"].append(meta[mi][1])
        want = (-yd) % 360; got = np.degrees(yk[kb]) % 360
        res["joint"]["pick_ok"].append(float(min(abs(got - want), 360 - abs(got - want)) <= 180.0 / K + 1e-6))
    if (i - a.start) % 10 == 9:
        print(f"{i - a.start + 1}/{a.n}  {time.time() - t0:.0f}s", flush=True)

print(f"\npartial radius / GT radius: {np.mean(ratio):.3f} +- {np.std(ratio):.3f}   bbox-centre offset: {np.mean(boff):.3f} +- {np.std(boff):.3f} (unit frame)")
print(f"{'normalisation':>16} {'CD base':>8} {'CD canon':>9} {'gain':>7} {'pick ok':>8} {'centre err':>10}")
summ = {}
for name in res:
    b, c = np.mean(res[name]["base"]), np.mean(res[name]["canon"]); ok = np.mean(res[name]["pick_ok"])
    extra = f"   oracle {np.mean(res[name]['oracle']):.4f}" if name == "joint" else ""
    ce = f"{np.mean(cerr[name]):>10.3f}" if name in cerr else f"{'':>10}"
    print(f"{name:>16} {b:>8.4f} {c:>9.4f} {100 * (b - c) / b:>+6.1f}% {100 * ok:>7.0f}% {ce}{extra}")
    summ[name] = {"base": b, "canon": c, "pick_ok": ok, "centre_err": float(np.mean(cerr[name])) if name in cerr else None}
if not a.no_joint and res["joint"]["pick_scale"]:
    print("joint picked scale:", {s: int(np.sum(np.isclose(res['joint']['pick_scale'], s))) for s in a.scales})
    print("joint picked shift:", {s: int(np.sum(np.isclose(res['joint']['pick_shift'], s))) for s in a.shifts})
    summ["joint"]["oracle"] = float(np.mean(res["joint"]["oracle"]))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"args": vars(a), "summary": summ, "ratio": ratio, "bbox_offset": boff}, open(a.out + ".json", "w"), indent=1)
