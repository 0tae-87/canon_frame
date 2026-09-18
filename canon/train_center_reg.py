"""Centre/scale regressor for serve-time canonicalisation.

PoinTr (+EDL head) was trained on partials that sit in the GT's normalised
frame (centroid 0, max radius 1). At serve time only the partial exists, so the
server centres on the partial's bbox and scales by its max radius. Measured on
held-out ShapeNet-55 (canon_frame/analysis/norm_yaw_offline.py): that centring
error alone (0.20 r) costs +64 % CD-L1, more than the yaw ambiguity (+20 %),
and it also flattens the epistemic landscape so the yaw argmin stops working.

This trains a small PointNet to predict, from the bbox-normalised partial,
  t = (c_gt - c_b) / r   (3)   and   log(1 / r)   (1)
i.e. where the GT centre is and how much bigger the full object is.
Augmentation: random yaw about the canonical up axis (y), random crop ratio
in [1/4, 3/4] like PoinTr training.

    python canon_frame/canon/train_center_reg.py --epochs 15
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from easydict import EasyDict                                       # noqa: E402
from utils.config import cfg_from_yaml_file                         # noqa: E402
from datasets.ShapeNet55Dataset import ShapeNet                     # noqa: E402
from new_uncertainty.scripts.train_edl_v11 import make_partial      # noqa: E402
from utils import misc                                              # noqa: E402


def make_partial_fast(gt, n_partial=2048, lo=0.25, hi=0.75):
    """Batched version of misc.seprate_point_cloud's random crop: one random unit
    vector per sample, keep the points FARTHEST from it, fps to n_partial. The crop
    ratio is drawn once per batch (per-sample in the original); same marginal."""
    B, N, _ = gt.shape
    num_crop = int(N * (lo + (hi - lo) * float(torch.rand(()))))
    d = F.normalize(torch.randn(B, 1, 3, device=gt.device), dim=-1)
    idx = torch.argsort((gt - d).norm(dim=-1), dim=1)[:, num_crop:]          # (B, N-num_crop)
    kept = torch.gather(gt, 1, idx[..., None].expand(-1, -1, 3)).contiguous()
    return misc.fps(kept, n_partial)


class CenterNet(nn.Module):
    """PointNet with one round of global-context concatenation (seg-style), then pooled regression."""
    def __init__(self, w=256):
        super().__init__()
        self.m1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                nn.Conv1d(128, w, 1), nn.BatchNorm1d(w), nn.ReLU())
        self.m2 = nn.Sequential(nn.Conv1d(w + 2 * w, w, 1), nn.BatchNorm1d(w), nn.ReLU(),
                                nn.Conv1d(w, w, 1), nn.BatchNorm1d(w), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * w, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, u):                                            # u (B,N,3), bbox-normalised
        f = self.m1(u.transpose(1, 2))                               # (B,w,N)
        g = torch.cat([f.max(2).values, f.mean(2)], 1)               # (B,2w)
        f2 = self.m2(torch.cat([f, g[:, :, None].expand(-1, -1, f.shape[2])], 1))
        g2 = torch.cat([f2.max(2).values, f2.mean(2)], 1)
        return self.head(g2)                                         # (B,4): t_xyz, log(1/r)


def bbox_norm(p):
    """p (B,N,3) -> u (B,N,3), c_b (B,3), r (B,)"""
    c = (p.min(1).values + p.max(1).values) / 2
    r = (p - c[:, None]).norm(dim=2).max(1).values
    return (p - c[:, None]) / r[:, None, None], c, r


def rot_y_batch(theta):
    c, s = torch.cos(theta), torch.sin(theta); z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, z, s], 1), torch.stack([z, o, z], 1), torch.stack([-s, z, c], 1)], 1)


def targets(gt, part):
    """GT frame: centre 0, radius 1. Return u, t, logs, c_b, r"""
    u, c, r = bbox_norm(part)
    t = -c / r[:, None]; logs = torch.log(1.0 / r)
    return u, t, logs, c, r


@torch.no_grad()
def evaluate(model, ds, dev, start=3200, n=300, yaw_deg=0.0):
    model.eval(); errs, errs0, serr = [], [], []
    R = rot_y_batch(torch.tensor([np.radians(yaw_deg)], device=dev, dtype=torch.float32))[0]
    for i in range(start, start + n):
        _, _, g = ds[i]; gt = torch.as_tensor(np.asarray(g), dtype=torch.float32, device=dev)[None]
        torch.manual_seed(1234 + i); part = make_partial(gt, fixed=True, sigma=0.0) @ R.T
        u, t, logs, c, r = targets(gt, part)
        o = model(u)
        errs.append(float((o[0, :3] - t[0]).norm() * r[0])); errs0.append(float(t[0].norm() * r[0]))
        serr.append(float((o[0, 3] - logs[0]).abs()))
    model.train()
    return float(np.mean(errs)), float(np.mean(errs0)), float(np.mean(serr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="canon_frame/ckpts/center_reg.pth")
    a = ap.parse_args(); os.chdir(REPO); dev = "cuda"; torch.manual_seed(0)
    cfg = cfg_from_yaml_file("cfgs/dataset_configs/ShapeNet-55.yaml")
    vt = EasyDict(dict(cfg)); vt.subset = "train"; ds_tr = ShapeNet(vt)
    vv = EasyDict(dict(cfg)); vv.subset = "test"; ds_te = ShapeNet(vv)
    dl = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    model = CenterNet(a.width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl), eta_min=1e-5)
    e_pred, e_bbox, e_s = evaluate(model, ds_te, dev)
    print(f"init: centre err pred {e_pred:.4f}  bbox {e_bbox:.4f}  |dlogs| {e_s:.3f}", flush=True)
    best, hist, t0 = 1e9, [], time.time()
    for ep in range(a.epochs):
        tot, n_it = 0.0, 0
        for _, _, gt in dl:
            gt = gt.to(dev, non_blocking=True).float()                       # (B,8192,3), GT frame
            R = rot_y_batch(torch.rand(gt.shape[0], device=dev) * 2 * np.pi)
            gt = torch.bmm(gt, R.transpose(1, 2))                            # random yaw (about origin)
            part = make_partial_fast(gt)                                      # random crop 1/4..3/4
            u, t, logs, c, r = targets(gt, part)
            o = model(u)
            loss = F.smooth_l1_loss(o[:, :3], t, beta=0.05) + 0.5 * F.smooth_l1_loss(o[:, 3], logs, beta=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += float(loss); n_it += 1
        e_pred, e_bbox, e_s = evaluate(model, ds_te, dev)
        e_pred90, _, _ = evaluate(model, ds_te, dev, yaw_deg=90.0)
        hist.append({"epoch": ep, "loss": tot / n_it, "err_pred": e_pred, "err_pred_yaw90": e_pred90, "err_bbox": e_bbox, "dlogs": e_s})
        print(f"ep {ep:2d} loss {tot / n_it:.4f}  centre err pred {e_pred:.4f} (yaw90 {e_pred90:.4f})  bbox {e_bbox:.4f}  |dlogs| {e_s:.3f}  {time.time() - t0:.0f}s", flush=True)
        if e_pred < best:
            best = e_pred
            torch.save({"model": model.state_dict(), "width": a.width, "epoch": ep, "err_pred": e_pred, "err_bbox": e_bbox}, a.out)
    json.dump(hist, open(a.out.replace(".pth", "_hist.json"), "w"), indent=1)
    print(f"best centre err {best:.4f} (bbox {e_bbox:.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
