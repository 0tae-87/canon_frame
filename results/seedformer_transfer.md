# Does the canonical-frame regressor transfer to another completion network? SeedFormer, no retraining (2026-09-30)

**Question.** `center_reg_v2.pth` was trained once (ShapeNet-55 single views) and used with the frozen PoinTr + EDL
server. Does the same frozen module improve a different completion network, without retraining either?

**Model.** Official SeedFormer (github.com/hrzhou2/seedformer), ShapeNet-55 checkpoint from the authors' Google-Drive
folder `1waTq7npTO068qyOAkK7HkSW0z69JVwAE` → `../seedformer/pretrained/models/shapenet55/ShapeNet-55/ckpt-best.pth`
(13.2 MB, `epoch_index` 255, `best_metrics` 12.684; architecture `seedformer_dim128`, `UPSAMPLE_FACTORS [1,4,4]` from
`train_shapenet55.py`, 2048 input → 8192 output points). Checkpoint conventions verified in the authors' code:
training data = ShapeNet-55 `shapenet_pc/*.npy` normalised by `pc_norm` (mean centroid, max radius = 1) of the full cloud,
partial = 2048-point crop of that cloud (`utils/data_loaders.py:449`, `CONST.N_INPUT_POINTS 2048`) — the same frame and
protocol as PoinTr's ShapeNet-55 loader, and exactly the frame the regressor was trained to recover
(`canon/train_center_reg.py::targets`: GT frame = centre 0 / radius 1 of the full cloud). Up-axis: ShapeNet y-up, world
z-up, rotated with the same `R_align` as the completion server. Both networks frozen; nothing tuned on the benchmark.

**Protocol** (`analysis/sim_bench_seedformer.py`). Same observations as `sim_bench_v2.log`: 10 objects × 15 stored Isaac
observations per level (`runCanon` = full view, `runLat04` = 40 % lateral occluder), the 2048-point observed tail of each
dump (what the server saw, world metres; the 300 file paths are in `results/seedformer_bench/meta.json`). Per observation
the **same 2048 points** go through SeedFormer in three frames: `bbox` (partial bbox centre + max radius, the server rule),
`v2` (bbox frame, then the regressor's centre shift and log-scale, no second bbox normalisation), `gt` (full-mesh mean
centre + max radius at the trial pose; diagnostic reference, uses the mesh). The output is mapped back to world metres
through the frame it went in by. CD-L1 (mm) = mean of the two one-sided mean nearest-neighbour Euclidean distances between
the 8192 output points and 8192 mesh-surface samples (same sampler and rng as `sim_cd.py`); precision = output→mesh,
recall = mesh→output. `torch.manual_seed(0)` before every forward; FPS inside SeedFormer is deterministic for a given input.
Nothing is appended to SeedFormer's output by a server, so all 8192 points are scored; the network does re-emit observed
geometry itself (median min output→input distance 0.3 µm, ≈ 26 % of outputs within 0.1 mm of an input point, ≈ 2.6 %
within 1 µm — refined copies, not a verbatim tail; the same holds for PoinTr's generated points over observed regions).

## Result

| lateral occlusion | frame | CD-L1 | precision | recall | frame centre err | scale ratio | v2 − bbox (paired) |
|---|---|---:|---:|---:|---:|---:|---|
| 0 (n = 150) | bbox | 4.28 | 2.13 | 6.43 | 12.8 mm | 0.945 | |
| | **v2** | **3.74** | 2.34 | 5.14 | 11.7 mm | 1.044 | **−0.54 mm (−12.5 %)**, median −0.12, better on 87/150, Wilcoxon p = 3e-4 |
| | gt (ref.) | 2.98 | 1.95 | 4.02 | 0 | 1 | v2 closes 42 % of the bbox→gt gap |
| 40 % (n = 150) | bbox | 9.24 | 2.01 | 16.48 | 29.5 mm | 0.870 | |
| | **v2** | **6.60** | 2.28 | 10.91 | 19.6 mm | 0.962 | **−2.64 mm (−28.6 %)**, median −2.71, better on 133/150, Wilcoxon p = 1e-22 |
| | gt (ref.) | 3.55 | 2.21 | 4.89 | 0 | 1 | v2 closes 46 % of the bbox→gt gap |

Per object (CD-L1 mm, bbox / v2 / gt; Δ = v2 − bbox):

| object | full: bbox / v2 / gt | Δ | 40 %: bbox / v2 / gt | Δ |
|---|---|---:|---|---:|
| 005_tomato_soup_can | 1.6 / 1.7 / 1.6 | +0.1 | 4.5 / 1.8 / 1.7 | −2.7 |
| 006_mustard_bottle | 4.8 / 3.3 / 2.8 | −1.5 | 11.3 / 8.6 / 3.1 | −2.7 |
| 010_potted_meat_can | 2.2 / 2.3 / 2.4 | +0.2 | 6.1 / 3.3 / 3.1 | −2.8 |
| 011_banana | 5.1 / 3.3 / 2.3 | −1.8 | 9.0 / 7.7 / 4.0 | −1.3 |
| 021_bleach_cleanser | 6.2 / 4.0 / 3.5 | −2.2 | 12.6 / 8.0 / 3.8 | −4.6 |
| 022_windex_bottle | 6.7 / 5.5 / 5.8 | −1.2 | 17.0 / 14.9 / 6.4 | −2.0 |
| 025_mug | 2.6 / 2.2 / 2.3 | −0.4 | 6.5 / 3.2 / 2.6 | −3.2 |
| 035_power_drill | 9.2 / 8.4 / 4.7 | −0.8 | 12.1 / 7.5 / 4.8 | −4.6 |
| 051_large_clamp | 2.5 / 4.5 / 2.5 | **+2.0** | 8.1 / 7.8 / 3.5 | −0.2 |
| 061_foam_brick | 1.8 / 2.1 / 1.9 | +0.4 | 5.3 / 3.1 / 2.6 | −2.3 |

**Improvement.** The frozen regressor transfers: with SeedFormer it recovers 42 % (full view) and 46 % (40 % occluder) of
the gap between the server's bbox frame and the true frame, improving 9/10 objects under occlusion and 6/10 at full view
(the other four are within 0.4 mm of an already near-perfect frame, except the clamp). The pattern is the one seen with
PoinTr: the gain is a frame effect — recall (mesh→output, i.e. extent / placement) drops 16.5 → 10.9 mm while precision
stays at 2.0–2.3 mm — and it grows with occlusion. The remaining bbox→gt gap under occlusion (6.6 vs 3.6 mm) is centre
error the regressor does not remove (19.6 mm left of 29.5).

**Failure cases** (kept; no object excluded). (i) `051_large_clamp` at full view: +2.0 mm, frame centre error 5 → 34 mm —
the known shape-class gap (thin flat tool absent from ShapeNet-55; README Not-to-do 7), identical with SeedFormer; the worst
five full-view losses are four clamp observations (+2.6 … +5.8 mm) and `power_drill t1` (+2.8 on an 18.4 mm bbox case).
(ii) Under occlusion the regressor under-corrects bottles seen from the narrow side: `windex t15` +2.7, `mustard t1` +2.6,
`mustard t19` +1.8 (residual centre error 24 and 45 mm for mustard / windex). (iii) At full view the regressor slightly
*worsens* the frame on already-centred objects (windex 13 → 18 mm, clamp) — the same "costs nothing at full view" claim
holds only on average (−0.54 mm, p = 3e-4), not per object.

**Extra inference time.** Regressor forward 5.8–6.2 ms per cloud, SeedFormer 48 ms (both measured with `cuda.synchronize`
while a training job shared the GPU, so absolute times are inflated; the regressor is a 1.4 M-parameter PointNet and
measured ≈ 1 ms on an idle GPU in the PoinTr server). No second normalisation pass, no extra data movement.

**PoinTr reference (same 300 observations, `sim_bench_v2.log`; PoinTr + EDL server, 6144 generated points — output count
differs from SeedFormer's 8192, so read as a reference, not a paired comparison).** CD-L1 bbox / v2: full view 7.2 / 6.9
(−4 %), 40 % occluder 12.6 / 9.6 (−24 %). SeedFormer is the stronger completer on this benchmark in absolute terms
(4.3 / 9.2 mm plain) and the relative gain from the same frozen frame correction is the same or larger (−12.5 % / −28.6 %).

## Reproduce

```
# once: pip install gdown; clone + weights (≈ 26 MB)
git clone --depth 1 https://github.com/hrzhou2/seedformer.git ../seedformer
(cd ../seedformer && mkdir -p pretrained && cd pretrained && gdown --folder https://drive.google.com/drive/folders/1waTq7npTO068qyOAkK7HkSW0z69JVwAE)
# benchmark (container; pointnet2_ops present in pointr_blackwell:gpufix; ~3 min)
docker run --rm --gpus all -v /home/wim/Desktop/yt_ws:/workspace -w /workspace/PoinTr pointr_blackwell:gpufix \
    python canon_frame/analysis/sim_bench_seedformer.py --per-cell 15
# outputs: results/seedformer_bench/{rows.csv (900 rows), summary.md, meta.json (checkpoints + the 300 observation files)}
```
Scope: offline shape evaluation only; no Isaac grasp trials with SeedFormer.
