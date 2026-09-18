#!/bin/bash
# After the night chain: rebuild the executed-row cache, then the full table and the occlusion trend.
cd /home/wim/Desktop/yt_ws/PoinTr; PY=/home/wim/anaconda3/envs/complete/bin/python
bash canon_frame/tools/cache_exec_rows.sh > canon_frame/results/cache_exec_rows.log 2>&1
$PY canon_frame/analysis/results_table.py > canon_frame/results/results_table.md 2>&1
$PY canon_frame/analysis/occlusion_trend.py > canon_frame/results/occlusion_trend.md 2>&1
echo "wrapup done $(date +%H:%M)"
