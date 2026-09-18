"""Ground-truth surface clouds for the YCB assets used in the sim, in each asset's ROOT
frame (the frame of the referenced prim), so that GT_world = R(quat) p + pos with the pose
that indy7.py now logs into every trial dump. Also lists every asset available in the
Isaac YCB Axis_Aligned pack (candidates for the object-set expansion).

    ~/isaacsim/python.sh canon_frame/tools/ycb_gt_clouds.py [--n 30000]
"""
import argparse, os, sys, json
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30000); ap.add_argument("--all", action="store_true", help="also extract every asset in the pack")
args = ap.parse_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from pxr import Usd, UsdGeom, Gf
import omni.client
from isaacsim.storage.native import get_assets_root_path
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "icra", "isaac_graspgen", "source"))
from sim.config import YCB_CONFIG
OUT = os.path.join(REPO, "canon_frame", "assets", "ycb_gt"); os.makedirs(OUT, exist_ok=True)
root = get_assets_root_path(); pack = root + "/Isaac/Props/YCB/Axis_Aligned/"
res, entries = omni.client.list(pack)
avail = sorted(e.relative_path for e in entries if e.relative_path.endswith(".usd"))
print(f"assets root: {root}\n{len(avail)} assets in the pack:\n  " + "\n  ".join(avail), flush=True)
json.dump(avail, open(os.path.join(OUT, "_pack_listing.json"), "w"), indent=1)


def sample_stage(usd_path, n):
    stage = Usd.Stage.Open(usd_path); xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    tris, areas = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh): continue
        m = UsdGeom.Mesh(prim); pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
        if pts.size == 0: continue
        M = np.array(xc.GetLocalToWorldTransform(prim)).T                # row-vector convention -> transpose
        pts = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
        fvc = np.array(m.GetFaceVertexCountsAttr().Get()); fvi = np.array(m.GetFaceVertexIndicesAttr().Get()); k = 0
        for c in fvc:
            idx = fvi[k:k + c]; k += c
            for j in range(1, c - 1):                                   # fan triangulation
                a, b, cc = pts[idx[0]], pts[idx[j]], pts[idx[j + 1]]
                tris.append((a, b, cc)); areas.append(0.5 * np.linalg.norm(np.cross(b - a, cc - a)))
    tris = np.array(tris); areas = np.array(areas)
    rng = np.random.default_rng(0); pick = rng.choice(len(tris), n, p=areas / areas.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    a, b, c = tris[pick, 0], tris[pick, 1], tris[pick, 2]
    return (1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b + (r1 * r2)[:, None] * c, len(tris), float(areas.sum())


names = [o["usd"] for o in YCB_CONFIG["objects"]]
if args.all: names += ["/Isaac/Props/YCB/Axis_Aligned/" + a for a in avail if a not in [os.path.basename(u) for u in names]]
for usd in names:
    path = (root + usd) if usd.startswith("/Isaac/") else os.path.join(REPO, "icra", "isaac_graspgen", usd)
    name = os.path.splitext(os.path.basename(usd))[0]
    try:
        pts, ntri, area = sample_stage(path, args.n)
        ext = pts.max(0) - pts.min(0)
        np.save(os.path.join(OUT, f"{name}.npy"), pts.astype(np.float32))
        print(f"{name:>28}: {ntri:6d} tris, area {area * 1e4:7.1f} cm^2, extent (cm) {np.round(ext * 100, 1)}", flush=True)
    except Exception as e:                                                # noqa: BLE001
        print(f"{name:>28}: FAILED {type(e).__name__}: {e}", flush=True)
app.close()
