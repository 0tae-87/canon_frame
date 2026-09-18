"""How predictable is the GT-centre offset of a partial from the partial plus the
view direction? GT centre is 0 in the GT frame, so the bbox-centre error is |c_b|.
seprate_point_cloud keeps the points FARTHEST from a random unit vector `center`,
so the missing side lies toward +center (camera at -center). Fit alpha in
c_hat = c_b + alpha * r * center (shift away from the camera)."""
import os, sys, numpy as np, torch
import torch.nn.functional as F
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO); os.chdir(REPO)
from easydict import EasyDict
from utils.config import cfg_from_yaml_file
from datasets.ShapeNet55Dataset import ShapeNet
from new_uncertainty.scripts.train_edl_v11 import make_partial
v = EasyDict(dict(cfg_from_yaml_file("cfgs/dataset_configs/ShapeNet-55.yaml"))); v.subset = "test"; ds = ShapeNet(v)
for crop in (0.5, None):
    C, D, Rr, E = [], [], [], []
    for i in range(3200, 3500):
        _, _, g = ds[i]; gt = torch.as_tensor(np.asarray(g), dtype=torch.float32).cuda()[None]
        torch.manual_seed(1234 + i); d = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1)[0, 0].numpy()   # same draw as seprate_point_cloud
        torch.manual_seed(1234 + i); p = make_partial(gt, fixed=crop is not None, crop_ratio=crop, sigma=0.0)[0]
        cb = (p.min(0).values + p.max(0).values) / 2; r = (p - cb).norm(dim=1).max()
        ext = (p.max(0).values - p.min(0).values).cpu().numpy()
        C.append(cb.cpu().numpy()); D.append(d); Rr.append(float(r)); E.append(ext)
    C, D, Rr, E = np.array(C), np.array(D), np.array(Rr), np.array(E)
    err0 = np.linalg.norm(C, axis=1)
    cosang = -(C * D).sum(1) / (err0 + 1e-9)               # +1 => bbox centre displaced toward the camera (-d)
    proj = -(C * D).sum(1) / Rr
    best = min(((al, np.linalg.norm(C + al * Rr[:, None] * D, axis=1).mean()) for al in np.linspace(0, 0.6, 61)), key=lambda t: t[1])
    # shift proportional to the bbox extent along d instead of the radius
    extd = np.abs(E * D).sum(1)
    best2 = min(((al, np.linalg.norm(C + al * extd[:, None] * D, axis=1).mean()) for al in np.linspace(0, 1.0, 101)), key=lambda t: t[1])
    print(f"crop={crop}: |c_b| {err0.mean():.3f}+-{err0.std():.3f}  cos(-c_b,d) {cosang.mean():.2f}+-{cosang.std():.2f}  proj/r {proj.mean():.3f}+-{proj.std():.3f}  "
          f"alpha*r: {best[0]:.2f} -> {best[1]:.3f}   alpha*extent_d: {best2[0]:.2f} -> {best2[1]:.3f}")
