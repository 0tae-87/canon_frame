# SeedFormer (official ShapeNet-55 ckpt, frozen) + canonical-frame regressor v2 (frozen): offline mesh benchmark

SeedFormer ckpt `seedformer/pretrained/models/shapenet55/ShapeNet-55/ckpt-best.pth` (epoch 255), regressor `canon_frame/ckpts/center_reg_v2.pth` (epoch 39); 300 observations, 2048 input points each, 8192 output points, CD vs 8192 mesh samples (mm). Conditions: bbox = server frame; v2 = bbox + regressor; gt = full-mesh mean centre / max radius (diagnostic reference).

## lateral occlusion 0 (runCanon, n = 150 observations)

| frame | CD-L1 | precision (gen→mesh) | recall (mesh→gen) | frame centre err (mm) | frame scale ratio | bbox-centre err of output (mm) | net ms | extra reg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bbox | 4.28 | 2.13 | 6.43 | 12.8 | 0.945 | 8.7 | 49.6 | 0.00 |
| v2 | 3.74 | 2.34 | 5.14 | 11.7 | 1.044 | 14.2 | 49.9 | 6.17 |
| gt | 2.98 | 1.95 | 4.02 | 0.0 | 1.000 | 8.6 | 43.9 | 0.00 |

v2 − bbox: **-0.54 mm** (-12.5 %); gt − bbox: -1.29 mm; v2 closes 42 % of the bbox→gt gap; v2 better than bbox on 87/150 observations.

| object | bbox | v2 | gt | v2 − bbox | frame centre err bbox → v2 (mm) |
|---|---:|---:|---:|---:|---:|
| 005_tomato_soup_can | 1.6 | 1.7 | 1.6 | +0.1 | 2 → 1 |
| 006_mustard_bottle | 4.8 | 3.3 | 2.8 | -1.5 | 22 → 5 |
| 010_potted_meat_can | 2.2 | 2.3 | 2.4 | +0.2 | 3 → 2 |
| 011_banana | 5.1 | 3.3 | 2.3 | -1.8 | 23 → 19 |
| 021_bleach_cleanser | 6.2 | 4.0 | 3.5 | -2.2 | 27 → 6 |
| 022_windex_bottle | 6.7 | 5.5 | 5.8 | -1.2 | 13 → 18 |
| 025_mug | 2.6 | 2.2 | 2.3 | -0.4 | 8 → 4 |
| 035_power_drill | 9.2 | 8.4 | 4.7 | -0.8 | 25 → 25 |
| 051_large_clamp | 2.5 | 4.5 | 2.5 | +2.0 | 5 → 34 |
| 061_foam_brick | 1.8 | 2.1 | 1.9 | +0.4 | 2 → 4 |

## lateral occlusion 0.4 (runLat04, n = 150 observations)

| frame | CD-L1 | precision (gen→mesh) | recall (mesh→gen) | frame centre err (mm) | frame scale ratio | bbox-centre err of output (mm) | net ms | extra reg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bbox | 9.24 | 2.01 | 16.48 | 29.5 | 0.870 | 30.4 | 49.0 | 0.00 |
| v2 | 6.60 | 2.28 | 10.91 | 19.6 | 0.962 | 22.4 | 49.7 | 5.75 |
| gt | 3.55 | 2.21 | 4.89 | 0.0 | 1.000 | 11.1 | 43.9 | 0.00 |

v2 − bbox: **-2.64 mm** (-28.6 %); gt − bbox: -5.70 mm; v2 closes 46 % of the bbox→gt gap; v2 better than bbox on 133/150 observations.

| object | bbox | v2 | gt | v2 − bbox | frame centre err bbox → v2 (mm) |
|---|---:|---:|---:|---:|---:|
| 005_tomato_soup_can | 4.5 | 1.8 | 1.7 | -2.7 | 14 → 1 |
| 006_mustard_bottle | 11.3 | 8.6 | 3.1 | -2.7 | 34 → 24 |
| 010_potted_meat_can | 6.1 | 3.3 | 3.1 | -2.8 | 19 → 7 |
| 011_banana | 9.0 | 7.7 | 4.0 | -1.3 | 35 → 30 |
| 021_bleach_cleanser | 12.6 | 8.0 | 3.8 | -4.6 | 36 → 21 |
| 022_windex_bottle | 17.0 | 14.9 | 6.4 | -2.0 | 49 → 45 |
| 025_mug | 6.5 | 3.2 | 2.6 | -3.2 | 22 → 7 |
| 035_power_drill | 12.1 | 7.5 | 4.8 | -4.6 | 39 → 22 |
| 051_large_clamp | 8.1 | 7.8 | 3.5 | -0.2 | 30 → 33 |
| 061_foam_brick | 5.3 | 3.1 | 2.6 | -2.3 | 18 → 5 |

Verbatim-copy check: min distance from any output point to the nearest input point over all clouds = 0.000 mm (> 0 ⇒ no observed points are appended to SeedFormer's output; all 8192 points are generated).
