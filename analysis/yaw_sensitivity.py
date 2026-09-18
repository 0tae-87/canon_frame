"""Yaw sensitivity of the frozen PoinTr + EDL completion on held-out ShapeNet-55.

PoinTr was trained on yaw-aligned ShapeNet; the completion server canonicalises
centre, scale and the up axis but not yaw. This measures what that costs, and
tests the epistemic-argmin canonicalisation idea in the same pass.

Protocol A (pure pose change): crop the partial ONCE in the canonical frame,
then rotate partial and GT together about the up axis (y) by each yaw. The
visible surface is identical across yaws; only the input orientation changes.

Per cloud x yaw: CD-L1 (symmetric NN mean, completion vs rotated GT), AUSE of
the epistemic score, mean epi and mean ale over generated points.

    python canon_frame/analysis/yaw_sensitivity.py --n 500 --seeds 0 1 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from easydict import EasyDict                                       # noqa: E402
from utils.config import cfg_from_yaml_file                         # noqa: E402
from datasets.ShapeNet55Dataset import ShapeNet                     # noqa: E402
from models.PoinTr import PoinTr                                    # noqa: E402
from new_uncertainty.models.PoinTr_EDL_v10 import PoinTrEDL, load_pointr_checkpoint  # noqa: E402
from new_uncertainty.models.evidential_loss_v9 import nig_activate, compute_uncertainty  # noqa: E402
from new_uncertainty.scripts.train_edl_v11 import make_partial, ause  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500, help="held-out clouds, test.txt[3200:3200+n]")
ap.add_argument("--start", type=int, default=3200)
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--yaws", type=int, default=12, help="steps over 360 deg")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--protocol", choices=["A", "B"], default="A",
                help="A: crop once, rotate partial+GT. B: rotate GT, then crop "
                     "(pose AND visible surface change, as with a fixed camera)")
ap.add_argument("--head", default="experiments/PoinTr_EDL_v11/geomix0.8_seed{s}/head_best.pth")
ap.add_argument("--backbone", default="ckpts/pointr_training_from_scratch_c55_best.pth")
ap.add_argument("--out", default="canon_frame/results/yaw_sensitivity")
a = ap.parse_args()
os.chdir(REPO)
dev = "cuda"
EVAL_SEED = 1234


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=dev)


@torch.no_grad()
def cd_l1(pred, gt):
    """symmetric chamfer-L1 per cloud: mean NN distance both ways, averaged"""
    d = torch.cdist(pred, gt, compute_mode='donot_use_mm_for_euclid_dist')  # (B, Np, Ng); mm path is TF32-corrupted
    return 0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1))


@torch.no_grad()
def nn_resid(pred, gt):
    d = torch.cdist(pred, gt, compute_mode='donot_use_mm_for_euclid_dist')
    return d.min(2).values                        # (B, Np)


# ---- data: one partial per cloud, fixed, in the canonical frame -----------
ds_cfg = cfg_from_yaml_file("cfgs/dataset_configs/ShapeNet-55.yaml")
v = EasyDict(dict(ds_cfg)); v.subset = "test"
ds = ShapeNet(v)
gts, parts, names = [], [], []
for i in range(a.start, a.start + a.n):
    tax, mid, gt_np = ds[i]
    gt = torch.as_tensor(np.asarray(gt_np), dtype=torch.float32, device=dev)[None]
    torch.manual_seed(EVAL_SEED + i); torch.cuda.manual_seed_all(EVAL_SEED + i)
    parts.append(make_partial(gt, fixed=True, sigma=0.0)[0])       # (2048, 3)
    gts.append(gt[0]); names.append(f"{tax}-{mid}")
GT = torch.stack(gts); PART = torch.stack(parts)
print(f"{len(GT)} clouds  gt {tuple(GT.shape[1:])}  partial {tuple(PART.shape[1:])}", flush=True)

# ---- models ------------------------------------------------------------------
cfg = cfg_from_yaml_file("cfgs/ShapeNet55_models/PoinTr.yaml").model
yaw_deg = np.arange(a.yaws) * (360.0 / a.yaws)
S, N, Y = len(a.seeds), len(GT), len(yaw_deg)
CD = np.zeros((S, N, Y), np.float32); AUSE = np.zeros_like(CD)
EPI = np.zeros_like(CD); ALE = np.zeros_like(CD)
t0 = time.time()
for si, s in enumerate(a.seeds):
    base = PoinTr(cfg); load_pointr_checkpoint(base, a.backbone)
    hc = torch.load(a.head.format(s=s), map_location="cpu")
    model = PoinTrEDL(base, hidden_dim=hc.get("args", {}).get("hidden_dim", 128)).to(dev).eval()
    model.uq_head.load_state_dict(hc["uq_head"])
    for yi, deg in enumerate(yaw_deg):
        R = rot_y(np.radians(deg))
        for b0 in range(0, N, a.batch):
            sl = slice(b0, min(N, b0 + a.batch))
            if a.protocol == "A":
                gt_r = GT[sl] @ R.T; part_r = PART[sl] @ R.T
            else:
                gt_r = GT[sl] @ R.T
                torch.manual_seed(EVAL_SEED); part_r = make_partial(gt_r, fixed=True, sigma=0.0)
            with torch.no_grad():
                out = model(part_r)
                gen = out["gen_points"]
                _, nu, alpha, beta = nig_activate(out["nig_raw"], beta_min=1e-6)
                ale, epi = compute_uncertainty(nu, alpha, beta)
                e_pt, a_pt = epi.sum(-1), ale.sum(-1)                # (B, Ngen)
                CD[si, sl, yi] = cd_l1(gen, gt_r).cpu().numpy()
                err = nn_resid(gen, gt_r)
                for k in range(gen.shape[0]):
                    r_ = ause(err[k], e_pt[k]); AUSE[si, b0 + k, yi] = float(r_[0] if isinstance(r_, (tuple, list)) else r_)
                EPI[si, sl, yi] = e_pt.mean(1).cpu().numpy(); ALE[si, sl, yi] = a_pt.mean(1).cpu().numpy()
        print(f"seed {s} yaw {deg:5.1f}: CD {CD[si, :, yi].mean():.4f}  AUSE {AUSE[si, :, yi].mean():.4f}  "
              f"epi {EPI[si, :, yi].mean():.3e}  ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez_compressed(a.out + ".npz", cd=CD, ause=AUSE, epi=EPI, ale=ALE, yaw_deg=yaw_deg,
                    seeds=np.array(a.seeds), names=np.array(names), protocol=a.protocol)

# ---- summary ------------------------------------------------------------------
am = EPI.argmin(2)                                      # (S, N) yaw index with min epi
cd_am = np.take_along_axis(CD, am[..., None], 2)[..., 0]
summ = {
    "protocol": a.protocol, "n": N, "seeds": a.seeds, "yaw_deg": yaw_deg.tolist(),
    "cd_by_yaw": CD.mean((0, 1)).tolist(), "ause_by_yaw": AUSE.mean((0, 1)).tolist(),
    "epi_by_yaw": EPI.mean((0, 1)).tolist(), "ale_by_yaw": ALE.mean((0, 1)).tolist(),
    "cd_at_0": float(CD[:, :, 0].mean()), "cd_worst_yaw": float(CD.mean((0, 1)).max()),
    "epi_argmin_is_0_frac": float((am == 0).mean()),
    "epi_argmin_within_30_frac": float(((am == 0) | (am == 1) | (am == Y - 1)).mean()),
    "cd_at_epi_argmin": float(cd_am.mean()), "cd_mean_over_yaw": float(CD.mean()),
}
json.dump(summ, open(a.out + "_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
