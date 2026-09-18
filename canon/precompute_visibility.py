"""Depth-camera-like single-view partials for ShapeNet-55: hidden-point removal (Katz et al.
spherical flip + convex hull) from random viewpoints. The sim's observation is a front SURFACE
seen by a camera, not the half-space crop PoinTr trains on (which keeps back-facing surface
inside the half-space, e.g. both walls of a bottle) -- the regressor v1 trained on crops
mis-centres real depth views (sim bench: centre error 10 -> 31 mm at full view). Writes one
packed visibility mask per (cloud, view).

    python canon_frame/canon/precompute_visibility.py --subset train --views 3 --workers 12
"""
import argparse, os, sys, time
import numpy as np
from multiprocessing import Pool
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO); os.chdir(REPO)
ap = argparse.ArgumentParser()
ap.add_argument("--subset", default="train"); ap.add_argument("--views", type=int, default=3)
ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="canon_frame/data/visibility_{subset}.npz")
ap.add_argument("--elev", type=float, nargs=2, default=[5, 50], help="camera elevation range (deg)")
ap.add_argument("--dist", type=float, nargs=2, default=[2.5, 2.5], help="camera distance range (object radius = 1)")
a = ap.parse_args()
from easydict import EasyDict                                       # noqa: E402
from utils.config import cfg_from_yaml_file                         # noqa: E402
from datasets.ShapeNet55Dataset import ShapeNet                     # noqa: E402
from scipy.spatial import ConvexHull                                # noqa: E402


def hpr_visible(pts, cam, gamma=3.0):
    """Katz 2007: indices visible from `cam`."""
    p = pts - cam; nrm = np.linalg.norm(p, axis=1); R = nrm.max() * 10 ** gamma
    flipped = p + 2 * (R - nrm)[:, None] * p / nrm[:, None]
    hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
    vis = np.unique(hull.vertices); return vis[vis < len(pts)]


def work(args):
    i, pts, seed = args
    rng = np.random.default_rng(seed); masks = []
    for v in range(a.views):
        # camera on a sphere of radius 2.5 (objects are unit-radius, y up), elevation 5-50 deg
        az = rng.uniform(0, 2 * np.pi); el = np.radians(rng.uniform(*a.elev)); r = rng.uniform(*a.dist)
        cam = np.array([r * np.cos(el) * np.cos(az), r * np.sin(el), r * np.cos(el) * np.sin(az)])
        vis = hpr_visible(pts, cam); m = np.zeros(len(pts), bool); m[vis] = True; masks.append(np.packbits(m))
    return i, np.stack(masks)


if __name__ == "__main__":
    cfg = cfg_from_yaml_file("cfgs/dataset_configs/ShapeNet-55.yaml"); v = EasyDict(dict(cfg)); v.subset = a.subset; ds = ShapeNet(v)
    n = len(ds) if not a.limit else a.limit
    def gen():
        for i in range(n):
            yield i, np.asarray(ds[i][2], dtype=np.float64), 1000 + i
    t0 = time.time(); out = np.zeros((n, a.views, 1024), np.uint8); vis_frac = []
    with Pool(a.workers) as pool:
        for k, (i, m) in enumerate(pool.imap_unordered(work, gen(), chunksize=16)):
            out[i] = m
            if k % 5000 == 0:
                print(f"{k}/{n}  {time.time() - t0:.0f}s", flush=True)
    frac = np.unpackbits(out, axis=2)[:, :, :8192].mean()
    os.makedirs(os.path.dirname(a.out.format(subset=a.subset)), exist_ok=True)
    np.savez_compressed(a.out.format(subset=a.subset), masks=out, n_points=8192, views=a.views)
    print(f"done: {n} clouds x {a.views} views, mean visible fraction {frac:.3f}, {time.time() - t0:.0f}s -> {a.out.format(subset=a.subset)}")
