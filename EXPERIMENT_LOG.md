# canon_frame experiment log (condensed 2026-09-18)

Chronological record of what was measured and decided. Negative results are kept as one line
each with their key number; the full reasoning behind them lives in README.md "Not-to-do".
Every number below has its data on disk (`results/exec_rows/<tag>.csv` for Isaac tags,
`results/*.json|md|log` for offline analyses).

## 2026-09-15 — diagnosis

- **Yaw** (`analysis/yaw_sensitivity.py`, 500 held-out ShapeNet clouds x 12 yaws, 3 seeds):
  a random yaw costs +27 % CD-L1 (worst +40 %); mean epistemic rises x8.6 and tracks CD
  (r 0.96); argmin of epistemic over K = 12 yaws recovers 91-98 % of the loss in the GT frame.
  Through the server (bbox-normalised partial) the gain vanishes (−0.6 %).
- **Centre** (`analysis/norm_yaw_offline.py`, 100 clouds x 8 yaws): centring on the partial's
  bbox (error 0.20 r ± 0.13) costs +64 % CD-L1 — three times the yaw loss — and flattens the
  epistemic-vs-yaw landscape; scale error is harmless; epistemic cannot select the centre.
  The bbox offset is shape-dependent (fixed view-direction shift recovers nothing:
  `analysis/center_offset_stats.py`) -> must be learned.
- Two measurement bugs fixed on the way: a mirrored world transform in the sanity script,
  and `torch.cdist`'s TF32 matmul path (2-50x too small CDs at world offsets).
- **Regressor v1** (`canon/train_center_reg.py`, PoinTr-style random crops, 60 epochs):
  held-out centre error 0.20 -> 0.085 r; offline CD 0.0752 -> 0.0508 (−32 %). Isaac
  (`runCanon`, 10 objects x 50, full view): complete_only 61.8 / partial 58.2 / **v1 55.0 %**
  (−6.8 pp, p 0.03). Better offline CD, worse grasps: the regressor extended the completion
  along the camera axis and GraspGen selected grasps on the hallucinated far side
  (`analysis/sim_paired_geometry.py`: executed grasp 15 -> 33 mm from the nearest observed
  point on the bleach bottle).

## 2026-09-15/16 night — selection gates and occlusion models (all closed)

- Observed-support gate (reject grasps > 20 mm from any observed point), on both completions:
  no gain (support_only 59.2 vs 61.8; reg_support 57.2 vs 55.0). Stratified pairing: a far
  executed grasp marks a hard pose; the replacement fails too (`analysis/paired_strata.py`).
  The ICRA footprint-epi gate on the extended completion: 56.2 vs 55.0, null.
- Depth-shell truncation (`--obs-nearest-frac 0.35`): all arms ~28 %; regressor extent
  0.53 -> 0.58 only. Wrong occlusion model.
- **Lateral occluder** (`--obs-lateral-crop 0.4`, new indy7 flag; the half-space crop both
  networks were trained on), `runLat04`, 10 objects x 50: plain completion **37.6** /
  partial **48.4** / v1 **51.2 %** (+13.6 pp over plain, p < 0.001; extent 0.82 -> 0.90).
  Seed 1 (`runLat04_s1`): 38.6 / 50.2 / 49.6. Pooled n = 900: v1 vs plain +12.3 (p 4e-8),
  vs partial +1.1 (n.s.). Under occlusion the frame is the whole story.

## 2026-09-16 — sim-domain benchmark and regressor v2 (the result)

- Tooling: YCB mesh surface samples from the Isaac USDs (`tools/ycb_gt_clouds.py`), object
  pose + full observed cloud logged into every trial dump, poses of older dumps reconstructed
  exactly from (scene_seed, trial) (observed points sit on the mesh at ~1 mm),
  `analysis/sim_cd.py` / `analysis/sim_bench.py` (re-complete stored observations through any
  server, CD vs mesh, 5 s per 150 clouds).
- **v1 mis-centres real depth views**: full view centre error 10 -> 31 mm (CD 7.2 -> 8.4),
  lateral 31 -> 37 mm. Cause: trained on half-space crops (both walls of a bottle inside),
  while a depth camera returns only the front surface.
- **v2** (`canon/precompute_visibility.py` + `canon/train_center_reg_v2.py`: hidden-point-removal
  single views, lateral-occluder augmentation): held-out 0.161 -> 0.078 r; bench full
  **6.9** / lateral **9.6** mm (base 7.2 / 12.6; v1 8.4 / 11.6).
- Isaac `runLat04_v2` (10 original objects, 40 % occluder): **v2 58.2 %** vs partial 48.4
  (p 0.002), v1 51.2, plain 37.6. Sim CD of the executed completions 9.7 mm (plain 12.6,
  v1 11.7) — the bench predicted the ordering and the grasps followed.
- Object-set expansion: 22 YCB assets checked; 17 graspable (master chef, tuna, bowl, wood
  block wider than the 77 mm aperture; scissors no candidates). Windex was spawning on its
  side in every earlier run (config bug, fixed). Marker: plain 62 / partial 14 / v2 52 %.
- New objects (`runNewLat04` / `runNewFull`): boxes tie across arms (flat faces), clamps
  prefer the partial. 16-object pooled, 40 %: plain 39.1 / partial 45.5 / v2 52.0
  (v2 vs partial +6.5, p 0.011). Full view: plain 55.7 / partial 47.1 / v2 50.6 on the new
  objects; on the original objects plain 68.0 / v2 65.8 (n.s.) with the large clamp 66 -> 14.
- Ensemble gate (3 seeds, correction only when they agree): bench full 6.5 / lateral 9.6
  with the clamp protected — Isaac `runEnsFull` / `runEnsLat04`: 57.2 / 53.4 vs v2 59.1 / 52.0.
  Closed.

## 2026-09-17 — replications, trend, object set, training sweep

- Seed-1 replication of v2 (`runLat04_v2_s1`, `runFinalFull_s1`, `runFinalLat04_s1`): 40 %
  occluder v2 vs partial +6.7 pp (p 0.015; third independent draw +6.7); full view v2 vs
  plain −2.2 / −2.2 (n.s.). Two-seed pooled (9 objects, n = 900): plain 41.1 / partial 53.0 /
  v1 54.6 / **v2 60.6** (vs partial +7.6, p 0.0009).
- **Occlusion trend** (`runOccL025` + the above; `results/occlusion_trend.md`), 16 objects:
  plain / partial / v2 = 62.6 / 55.9 / 59.1 (full), 51.4 / 53.5 / 57.5 (25 %),
  41.1 / 46.6 / 52.0 (40 %). v2's margin grows with occlusion.
- Main paper set fixed at 14 (clamps excluded; user decision): v2 − partial +5.4 / +7.1 /
  +9.1 pp, all p < 0.003 (`results/per_object_main14.md`).
- Repeated runs of the same seed share the executed grasp in 11 % of trials — re-runs are
  independent draws on identical poses.
- Training-strategy sweep (`canon/train_center_reg_v3.py`, removed after the sweep): yaw TTA,
  squash augmentation (any axis), DGCNN, heteroscedastic head, completion-in-the-loop
  Chamfer loss (from scratch diverges; warm-started B' 6.4 / 9.3, Y 6.2 / 8.9 mm on the bench).
  Heteroscedastic std flags the clamp (0.18 r vs 0.04-0.12) and gating on it wins the bench
  (6.45 / 9.41) but not Isaac (58.9 / 49.5). B' and Y in Isaac: 60.2 / 52.5 and 60.8 / 52.2
  vs v2 59.6 / 52.0 — identical. DGCNN alone: 3 min/epoch, bench 5.8 / 9.9.
- Operational: 12 Isaac cells lost to GraspGen timeouts while trainings shared the GPU;
  rule = no GPU training during an Isaac batch.

## 2026-09-18 — closing verdict and cleanup

Once the frame is right, further completion-CD gains (up to −10 %) do not change lift
success; the v1 -> v2 step (matching the training partials to the sensor) is the only change
that moved grasps. **Final method: always-on v2.** Workspace cleaned: negative-result
checkpoints, scripts, bulky Isaac logs/dumps of closed arms removed (executed rows cached in
`results/exec_rows/`); engine reduced to the v2 path and re-verified on the bench (6.9 / 9.6).
Open directions in README.md "Future work".
