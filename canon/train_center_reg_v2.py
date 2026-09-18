"""Centre/scale regressor v2: trained on DEPTH-CAMERA-LIKE partials.

v1 (train_center_reg.py) learned on PoinTr's random half-space crops and, on real sim
observations, made the centre WORSE (sim bench, full view: bbox 10 mm -> regressor 31 mm;
lateral view: 31 -> 37 mm) because a depth camera sees a front surface, not a half-space.
v2 trains on hidden-point-removal views of ShapeNet-55 (precompute_visibility.py), with
  - a lateral occluder (remove the fraction u ~ U(0, 0.5) of visible points beyond a quantile
    along a random horizontal direction) with probability 0.5,
  - random yaw about the up axis, and random subsampling to 2048 points,
  - 25 % of batches still use the old random crops so nothing is forgotten.
Same model / targets / bbox normalisation as v1; the checkpoint format is identical so
engine_canon.py serves it with CANON_REG=<path>.

    python canon_frame/canon/train_center_reg_v2.py --epochs 40 --width 512
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "canon_frame", "canon"))
from easydict import EasyDict                                       # noqa: E402
from utils.config import cfg_from_yaml_file                         # noqa: E402
from datasets.ShapeNet55Dataset import ShapeNet                     # noqa: E402
from train_center_reg import CenterNet, bbox_norm, rot_y_batch, targets, make_partial_fast  # noqa: E402
from utils import misc                                              # noqa: E402


class ViewDS(Dataset):
    def __init__(self, subset, vis_path):
        cfg = cfg_from_yaml_file("cfgs/dataset_configs/ShapeNet-55.yaml"); v = EasyDict(dict(cfg)); v.subset = subset
        self.ds = ShapeNet(v); z = np.load(vis_path); self.masks = z["masks"]; self.views = int(z["views"])
        assert len(self.masks) == len(self.ds), (len(self.masks), len(self.ds))
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        gt = np.asarray(self.ds[i][2], dtype=np.float32)
        m = np.unpackbits(self.masks[i], axis=1)[:, :len(gt)].astype(bool)      # (views, N)
        return torch.from_numpy(gt), torch.from_numpy(m)


def view_partial(gt, masks, rng, n_out=2048, p_lateral=0.5, max_lateral=0.5, dev="cuda"):
    """gt (B,N,3) canonical y-up; masks (B,V,N) bool -> (B, n_out, 3) view partials with optional occluder"""
    B, N, _ = gt.shape; out = []
    for b in range(B):
        v = int(rng.integers(masks.shape[1])); pts = gt[b][masks[b, v]]
        if len(pts) < 64: pts = gt[b]
        if rng.random() < p_lateral:
            u = rng.uniform(0.05, max_lateral); th = rng.uniform(0, 2 * np.pi)
            dvec = torch.tensor([np.cos(th), 0.0, np.sin(th)], device=pts.device, dtype=pts.dtype)
            s = pts @ dvec; thr = torch.quantile(s, 1.0 - u); pts = pts[s <= thr]
            if len(pts) < 64: pts = gt[b][masks[b, v]]
        idx = torch.randint(len(pts), (n_out,), device=pts.device) if len(pts) < n_out else torch.randperm(len(pts), device=pts.device)[:n_out]
        out.append(pts[idx])
    return torch.stack(out)


@torch.no_grad()
def evaluate_views(model, ds, dev, n=300, start=3200, seed=7):
    model.eval(); rng = np.random.default_rng(seed); errs, errs0 = [], []
    for i in range(start, start + n):
        gt, m = ds[i]; gt = gt.to(dev)[None]; m = m.to(dev)[None]
        part = view_partial(gt, m, rng, p_lateral=0.5); u, t, logs, c, r = targets(gt, part); o = model(u)
        errs.append(float((o[0, :3] - t[0]).norm() * r[0])); errs0.append(float(t[0].norm() * r[0]))
    model.train(); return float(np.mean(errs)), float(np.mean(errs0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=48); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=512); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--p-crop", type=float, default=0.25)
    ap.add_argument("--init", default="", help="warm start from a v1 checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="canon_frame/ckpts/center_reg_v2.pth")
    a = ap.parse_args(); os.chdir(REPO); dev = "cuda"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr = ViewDS("train", "canon_frame/data/visibility_train.npz"); te = ViewDS("test", "canon_frame/data/visibility_test.npz")
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    model = CenterNet(a.width).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location="cpu"); model.load_state_dict(ck["model"]); print(f"warm start from {a.init}")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl), eta_min=1e-5)
    e, e0 = evaluate_views(model, te, dev); print(f"init: view-partial centre err pred {e:.4f}  bbox {e0:.4f}", flush=True)
    best, hist, t0 = 1e9, [], time.time()
    for ep in range(a.epochs):
        tot, n_it = 0.0, 0
        for gt, m in dl:
            gt = gt.to(dev, non_blocking=True); m = m.to(dev, non_blocking=True)
            R = rot_y_batch(torch.rand(gt.shape[0], device=dev) * 2 * np.pi); gt = torch.bmm(gt, R.transpose(1, 2))
            # masks are per point, so they survive the rotation unchanged
            part = make_partial_fast(gt) if rng.random() < a.p_crop else view_partial(gt, m, rng)
            u, t, logs, c, r = targets(gt, part); o = model(u)
            loss = F.smooth_l1_loss(o[:, :3], t, beta=0.05) + 0.5 * F.smooth_l1_loss(o[:, 3], logs, beta=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += float(loss); n_it += 1
        e, e0 = evaluate_views(model, te, dev)
        hist.append({"epoch": ep, "loss": tot / n_it, "err_pred": e, "err_bbox": e0})
        print(f"ep {ep:2d} loss {tot / n_it:.4f}  view-partial centre err pred {e:.4f}  bbox {e0:.4f}  {time.time() - t0:.0f}s", flush=True)
        if e < best:
            best = e; torch.save({"model": model.state_dict(), "width": a.width, "epoch": ep, "err_pred": e, "err_bbox": e0, "variant": "v2-views"}, a.out)
    json.dump(hist, open(a.out.replace(".pth", "_hist.json"), "w"), indent=1)
    print(f"best view-partial centre err {best:.4f} (bbox {e0:.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
