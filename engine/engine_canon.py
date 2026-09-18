"""Completion with a learned canonical frame (regressor v2).

The frozen PoinTr + EDL completion was trained on ShapeNet partials that sit in the full
object's frame (centroid 0, max radius 1). The serving engine (icra/completion_server) has
only the partial, so it centres on the partial's bounding box and scales by its max radius.
Under occlusion that frame is wrong (centre error ~0.2 r) and the completion lands in the
wrong place -- worse for grasping than no completion at all (Isaac, 40 % lateral occluder:
plain completion 41 % vs partial 47 % lift success over 16 objects).

This engine inserts one step: a small PointNet (canon/train_center_reg_v2.py, 1.4 M params,
trained on hidden-point-removal single views of ShapeNet-55) predicts the object's centre
shift and scale factor from the bbox-normalised partial, the partial is re-normalised with
that prediction, and only then completed. Everything else (bbox normalisation, up-axis
alignment, observed tail, units) matches CompletionEngine, so the Isaac side needs no change
beyond pointing at this server's port. Result (same 16 objects, 40 % occluder): 52 %.

Env: CANON_REG = checkpoint path (default canon_frame/ckpts/center_reg_v2.pth).
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "icra", "completion_server"))
sys.path.insert(0, os.path.join(REPO, "canon_frame", "canon"))
sys.path.insert(0, REPO)
import config as C                                                   # noqa: E402
from engine import CompletionEngine                                  # noqa: E402
from utils import misc                                               # noqa: E402
from new_uncertainty.models.evidential_loss_v9 import nig_activate, compute_uncertainty  # noqa: E402
from train_center_reg import CenterNet                               # noqa: E402

REG_CKPT = os.environ.get("CANON_REG", os.path.join(REPO, "canon_frame", "ckpts", "center_reg_v2.pth"))


class CanonCompletionEngine(CompletionEngine):
    def __init__(self, device="cuda"):
        super().__init__(device)
        ck = torch.load(REG_CKPT, map_location="cpu")
        self.reg = CenterNet(ck["width"]).to(device).eval()
        self.reg.load_state_dict(ck["model"])
        self.reg_info = {"ckpt": os.path.relpath(REG_CKPT, REPO), "epoch": int(ck["epoch"]),
                         "heldout_centre_err_r": float(ck["err_pred"]), "bbox_centre_err_r": float(ck["err_bbox"])}
        print(f"[canon] centre/scale regressor loaded: {self.reg_info}", flush=True)

    def metadata(self):
        m = super().metadata()
        m["model"] = m["model"] + " + learned canonical frame (centre/scale regressor v2)"
        m["canon"] = {"rule": "bbox-normalise the partial, re-centre/re-scale with the regressor, then complete",
                      "regressor": self.reg_info}
        return m

    @torch.no_grad()
    def complete(self, points, use_reg=True):
        t0 = time.perf_counter()
        xyz = np.ascontiguousarray(np.asarray(points, dtype=np.float32)[:, :3])
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if len(xyz) < 64:
            raise ValueError(f"need >= 64 finite points, got {len(xyz)}")
        # --- same normalisation as CompletionEngine (bbox centre, max radius, up-axis) ---
        centroid = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
        centered = xyz - centroid
        scale = float(np.sqrt((centered ** 2).sum(axis=1)).max())
        if scale <= 0:
            raise ValueError("degenerate input cloud (zero radius)")
        unit = (centered / scale) @ self.R_align.T                   # canonical up = y
        t = torch.from_numpy(unit).float().unsqueeze(0).to(self.device)
        if t.shape[1] >= C.N_INPUT:
            t = misc.fps(t, C.N_INPUT)
        else:
            idx = torch.randint(t.shape[1], (C.N_INPUT - t.shape[1],), device=t.device)
            t = torch.cat([t, t[:, idx]], dim=1)
        # --- the one added step: learned frame correction (bbox frame: centre 0, radius 1) ---
        c_r = torch.zeros(3, device=self.device); s_r = torch.ones((), device=self.device)
        if use_reg:
            o = self.reg(t)[0]
            c_r, s_r = o[:3], torch.exp(o[3])
        x = (t - c_r) / s_r
        # --- completion + uncertainty, mapped back through both frames ---
        out = self.model(x)
        gen = out["gen_points"][0]                                    # (M, 3)
        _, nu, alpha, beta = nig_activate(out["nig_raw"], beta_min=1e-6)
        ale, epi = compute_uncertainty(nu, alpha, beta)
        epi_pt, ale_pt = epi.sum(-1)[0], ale.sum(-1)[0]
        gen_b = gen * s_r + c_r                                       # back to the bbox frame
        obs_b = t[0]
        torch.cuda.synchronize()
        completed = (gen_b.cpu().numpy() @ self.R_align) * scale + centroid
        obs = (obs_b.cpu().numpy() @ self.R_align) * scale + centroid
        s2 = scale * scale * float(s_r) ** 2
        zeros = np.zeros(len(obs), dtype=np.float32)
        return {
            "completed": np.ascontiguousarray(np.concatenate([completed, obs]), dtype=np.float32),
            "epi": np.ascontiguousarray(np.concatenate([epi_pt.cpu().numpy() * s2, zeros]), dtype=np.float32),
            "ale": np.ascontiguousarray(np.concatenate([ale_pt.cpu().numpy() * s2, zeros]), dtype=np.float32),
            "centroid": centroid.astype(np.float32),
            "scale": np.float32(scale),
            "n_input_used": int(C.N_INPUT),
            "canon_reg_used": bool(use_reg),
            "canon_reg_centre_shift": float(c_r.norm()),
            "canon_reg_scale": float(s_r),
            "infer_ms": float((time.perf_counter() - t0) * 1000.0),
        }
