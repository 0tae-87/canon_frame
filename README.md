# canon_frame — completion is only as good as its frame

Next-paper workspace (started 2026-09-15). `icra/` stays the frozen evidence base of the ICRA
paper; everything here builds on its Isaac pipeline (Franka Panda, GraspGen, YCB) and its
frozen PoinTr + EDL completion server. Cleaned up 2026-09-18: only the final method, the
analyses behind the numbers, and the data those numbers come from are kept.

## Claim (one paragraph)

The completion server normalises the observed partial by its own bounding box and max radius,
but the network was trained on partials sitting in the full object's frame. With one side of
the object hidden that frame is wrong (centre error ~0.2 object radii), the completion lands in
the wrong place, and completing becomes worse for grasping than not completing (40 % lateral
occluder: 41 % vs 47 % lift success). A 1.4 M-parameter PointNet trained on camera-like
single views of ShapeNet-55 predicts the correct centre and scale from the partial alone;
re-normalising with it before completion restores the completion to above the partial
(52 %), with the margin growing with occlusion, and costs nothing at full view.

## Setup

This folder is a standalone repo (github.com/0tae-87/canon_frame) but runs inside the PoinTr
workspace: clone it to `PoinTr/canon_frame` next to `icra/` (the Isaac pipeline, completion
server and batch runner it builds on) and the PoinTr models/checkpoints. All paths in the
scripts are relative to the PoinTr root. GPU jobs run in the `pointr_blackwell:gpufix`
container with the workspace mounted at `/workspace/PoinTr`.

## Final method: regressor v2 (`ckpts/center_reg_v2.pth`, served on port 5560)

- `canon/train_center_reg_v2.py` — PointNet (`canon/train_center_reg.py::CenterNet`, width 512),
  inputs the bbox-normalised 2048-point partial, outputs centre shift (3) and log scale (1).
  Training partials = hidden-point-removal single views of ShapeNet-55 train
  (`canon/precompute_visibility.py`, 3 viewpoints per shape, elevation 5-50 deg) with a
  lateral occluder augmentation (p 0.5, 5-50 % of the visible points removed), random yaw,
  25 % PoinTr-style crops. SmoothL1, Adam 1e-3 cosine, 40 epochs, ~20 min. Held-out centre
  error 0.161 r (bbox) -> 0.078 r.
- `engine/engine_canon.py` + `engine/server_canon.py` — CompletionEngine with the one extra
  step; same wire contract as `icra/completion_server`. `complete_base` = same process
  without the correction.
- Isaac arm: `python3 icra/run_batch_m4.py ... --canon-arms complete_v2:5560`
  (`run_isaac.sh <ckpt> <arm> <port>` runs a checkpoint on the 16 objects at both views).

## Results (Isaac, 50 trials per object, identical pose sequence per seed, McNemar on pairs)

Lift success over 16 graspable YCB objects, seed 0 (`results/occlusion_trend.md`):

| lateral occlusion | plain completion | partial only | **v2** | v2 − plain | v2 − partial |
|---|---|---|---|---|---|
| 0 (full view) | 62.6 | 55.9 | 59.1 | −3.5 (p 0.19) | +3.2 (p 0.05) |
| 25 % hidden | 51.4 | 53.5 | 57.5 | +6.1 (p 0.002) | +4.0 (p 0.06) |
| 40 % hidden | 41.1 | 46.6 | 52.0 | +10.9 (p 7e-8) | +5.4 (p 0.035) |

Main paper set = 14 objects (the two clamps excluded, see Not-to-do 7): v2 − partial
+5.4 / +7.1 / +9.1 pp (all p < 0.003); v2 − plain +1.1 (n.s.) / +8.4 / +12.9
(`results/per_object_main14.md`, per object). Replication at seed 1 (original objects, 40 %):
v2 vs partial +8.4 / +6.7 / +6.7 pp over three independent draws; full view v2 vs plain
−2.2 / −2.2 (n.s.). Every cell: `results/results_table.md`.

Mechanism (`analysis/sim_bench.py`, mesh CD in mm on stored observations): the plain frame
gives CD 7.2 (full) / 12.6 (40 %); v2 6.9 / 9.6; the earlier crop-trained regressor v1 made
the full view worse (8.4, centre error 10 -> 31 mm) because it had never seen a front surface.
The v1 -> v2 step is the only change in the whole project that moved lift success.

## How to reproduce

```
# GT clouds for the sim bench (once):  ~/isaacsim/python.sh canon_frame/tools/ycb_gt_clouds.py --all
# visibility masks (once, 8 min):      python canon_frame/canon/precompute_visibility.py --subset train --views 3
#                                      python canon_frame/canon/precompute_visibility.py --subset test  --views 2
# train (GPU container, 20 min):       python canon_frame/canon/train_center_reg_v2.py --epochs 40 --width 512
# serve:                               see engine/server_canon.py docstring (port 5560)
# bench vs the plain server:           python canon_frame/analysis/sim_bench.py --tag runLat04 --arm gate_off \
#                                        --servers base=172.17.0.2:5557 v2=127.0.0.1:5560
# Isaac, both views, 16 objects:       SEED=0 STAGE=both bash canon_frame/run_isaac.sh canon_frame/ckpts/center_reg_v2.pth complete_v2 5560
# tables:                              bash canon_frame/tools/morning_wrapup.sh
```

## Layout

- `analysis/` — `sim_cd.py` / `sim_bench.py` (mesh CD, pose from dumps or reconstructed),
  `occlusion_trend.py`, `results_table.py`, `sim_paired_geometry.py`, `occlusion_extent.py`
  (why the plain completion fails), `paired_choice.py` / `paired_strata.py` (paired-pose
  analysis), `yaw_sensitivity.py`, `norm_yaw_offline.py`, `center_offset_stats.py` (offline
  diagnosis), `selector_study.py` (Not-to-do 3).
- `canon/` — regressor model and trainers; `ckpts/` the deployed checkpoint; `data/` the
  visibility masks; `assets/ycb_gt/` mesh surface samples of every YCB asset.
- `paper/` — IPIU paper package: protocol, results with provenance, outline, figures (`paper/figures/make_figures.py`
  recomputes every number from `results/`).
- `results/` — the three tables above, `exec_rows/` (executed-trial rows of every Isaac tag,
  11 MB — the 17 GB grasp logs are not needed for any table), bench/diagnosis logs.
- Isaac data (`icra/isaac_graspgen/output/graspgen/`): tags runCanon, runCanon_v2, runNewFull,
  runLat04, runLat04_v2, runNewLat04, runLat04_s1, runLat04_v2_s1, runFinalFull_s1,
  runFinalLat04_s1, runOccL025, runThinLat04 (+ runX_confirm from `icra/` as the seed-1 full
  view control). Dumps of those tags hold observed cloud, completion, epi, candidates, pose.

## Not-to-do (tried, measured, closed — do not repeat)

1. **Per-candidate uncertainty gates** (patch epi at 0.30 R, footprint epi, observed-support
   distance, global epi): none moves lift success at full view or under occlusion
   (5,000 + 3,500 + 1,500 + 2,000 trials; within-object AUROC ~0.5). A far-from-observed grasp
   marks a hard pose, not a bad grasp — replacing it does not rescue the pose.
2. **Yaw canonicalisation** (argmin of epistemic over K yaw hypotheses): recovers 91-98 % of the
   yaw loss offline in the GT frame, 0 % through the server — centring dominates yaw 3:1.
3. **Learned partial-vs-completion selector**: no pose-level signal (within-object AUC 0.5);
   object-level only (+1-2 pp leave-one-object-out without the soup can).
4. **Depth-shell truncation as the occlusion model** (nearest 35 % of points): every arm
   collapses to ~28 %; neither the network nor the regressor extrapolates a thin shell.
5. **Regressor training tricks once the data is right**: yaw TTA, anisotropic squash
   augmentation (any axis), DGCNN backbone, heteroscedastic head, completion-in-the-loop
   Chamfer loss (from scratch: diverges; warm-started: bench CD −7 to −10 %) — all within
   noise of v2 on the bench or, where better (B', Y: 6.2 / 8.9 mm), identical in Isaac
   (60.8 / 52.2 vs 59.6 / 52.0). After the frame is right, completion CD no longer limits grasps.
6. **Gating the correction on its own uncertainty** (3-seed disagreement, heteroscedastic
   std): both flag the out-of-distribution clamp on the bench and fix its CD, but in Isaac
   they lose elsewhere (full view 57.2 / 58.9 vs v2 59.1; 40 % occluder 53.4 / 49.5 vs 52.0).
7. **Thin flat tools (large / extra-large clamp)**: ShapeNet-55 has no such class; v2's
   correction is confidently wrong on them (full view 66 -> 14 %). Not fixable by
   augmentation or architecture (item 5); excluded from the main set, numbers kept in
   `results/per_object_main14.md`.
8. **Objects wider than the 77 mm aperture** (master chef can, tuna can, bowl, wood block) and
   flat objects with no top-down candidates (scissors) cannot be used in any arm comparison.
9. **Offline ShapeNet CD as the metric**: it did not predict the v1 failure; the sim mesh
   bench (`sim_bench.py`) did. Only the sim bench is used from here on.

## Future work (open, evidence-backed)

- **Epistemic-driven re-observation.** Under occlusion the residual loss is a genuinely
  unknown side; the EDL head's epistemic field says where. Second capture from the
  high-epistemic direction, merge, complete — the first place the uncertainty head can act on
  grasping; control = random second view. Testable in Isaac (movable front camera).
- **Shape-class coverage of the regressor.** The clamp failure is a data gap, not capacity:
  train on a broader shape set (tools, flat objects) and re-check item 7.
- **Real-robot check** of the lateral-occluder regime on the Panda (3-4 objects x 10 trials,
  plain / partial / v2).

## Operational notes (traps that cost hours)

- `torch.cdist` runs its matmul path under TF32 in the NGC image: at world offsets (~0.4 m)
  CDs came out 2-50x too small. Use `compute_mode='donot_use_mm_for_euclid_dist'` or re-centre.
- Never train on the GPU while an Isaac batch runs: GraspGen requests time out and cells abort
  (12 cells lost on 09-17). One training job at a time; DGCNN needs 57 GB at batch 48.
- Runner: `OBJECTS` now lists 22 entries — always pass `--objects`. Repeated runs of the same
  seed share the executed grasp in only 11 % of trials (GraspGen resampling): re-runs are
  independent draws on identical poses, not replicas.
- `pgrep -f` / `pkill -f` with a literal pattern matches the shell that runs it (exit 144).
- Windex bottle was spawning on its side before 09-16 (config bug, fixed: up_axis "z");
  rows marked `022_windex_bottle*` in `results_table.md` are the lying bottle.
