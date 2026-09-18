# Per-object lift success, main object set (14 objects, clamps excluded), seed 0, 50 trials/cell

Generated 2026-09-17 by analysis/occlusion_trend.py --exclude 051_large_clamp 052_extra_large_clamp

| lateral occlusion | plain completion | partial | v2 | v2 − plain | v2 − partial |
|---|---|---|---|---|---|
| 0 (full view) | 64.6 | 60.3 | 65.7 | +1.1 (p 0.24) | +5.4 (p 0.0024) |
| 0.25 | 51.9 | 53.1 | 60.3 | +8.4 (p 0.00011) | +7.1 (p 0.00095) |
| 0.4 | 41.3 | 45.0 | 54.1 | +12.9 (p 1.2e-09) | +9.1 (p 0.00022) |

(14 objects, seed 0, n = 700 per arm and level; McNemar on identical poses)

per object (plain / partial / v2 at 0 | 0.25 | 0.4):
     005_tomato_soup_can  84/94/90  |  32/60/84  |  12/36/92
      006_mustard_bottle  74/82/78  |  62/82/72  |  40/70/52
     010_potted_meat_can  82/34/66  |  36/12/28  |  20/8/46
              011_banana  42/80/68  |  72/88/74  |  70/82/70
     021_bleach_cleanser  64/56/68  |  28/52/58  |  28/44/44
                 025_mug  76/74/78  |  64/50/76  |  50/66/72
         035_power_drill  44/34/48  |  54/50/50  |  48/44/40
          061_foam_brick  80/70/82  |  70/64/78  |  46/56/78
         003_cracker_box  10/6/14  |  14/0/6  |  6/2/6
           004_sugar_box  74/64/78  |  54/50/64  |  36/18/30
         008_pudding_box  92/88/90  |  94/92/96  |  86/72/86
         009_gelatin_box  94/98/92  |  96/96/96  |  88/96/94
        019_pitcher_base  18/8/8  |  6/8/14  |  8/6/14
       022_windex_bottle  70/56/60  |  44/40/48  |  40/30/34

## Excluded objects (recorded, not in the main table)

         051_large_clamp  66/40/14  |  62/72/62  |  54/66/54
   052_extra_large_clamp  32/10/12  |  34/40/14  |  26/50/20

## Seed-1 replication, original objects, lateral 0.4 (clamps excluded -> 8 objects)

| object | plain s0 | partial s0 | v2 s0 | plain s1 | partial s1 | v2 s1 | v2 s1 (2nd run) |
|---|---|---|---|---|---|---|---|
| 005_tomato_soup_can | 12 | 36 | 92 | 2 | 36 | 80 | 88 |
| 006_mustard_bottle | 40 | 70 | 52 | 48 | 68 | 52 | 50 |
| 010_potted_meat_can | 20 | 8 | 46 | 24 | 8 | 44 | 56 |
| 011_banana | 70 | 82 | 70 | 72 | 88 | 72 | 74 |
| 021_bleach_cleanser | 28 | 44 | 44 | 28 | 40 | 48 | 36 |
| 025_mug | 50 | 66 | 72 | 54 | 58 | 68 | 62 |
| 035_power_drill | 48 | 44 | 40 | 46 | 50 | 48 | 40 |
| 061_foam_brick | 46 | 56 | 78 | 40 | 54 | 66 | 68 |
| **ALL (n=400)** | **39.2** | **50.8** | **61.8** | **39.2** | **50.2** | **59.8** | **59.2** |
