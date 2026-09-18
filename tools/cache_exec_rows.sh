#!/bin/bash
# Extract only the EXECUTED candidate rows (1 per trial) from the 17 GB of grasp logs into small per-tag CSVs.
cd /home/wim/Desktop/yt_ws/PoinTr; O=icra/isaac_graspgen/output/graspgen; D=canon_frame/results/exec_rows; rm -f $D/*.csv   # rebuild from scratch (running tags would otherwise accumulate duplicates)
for f in $O/*_grasp_log*.csv; do
  tag=$(basename $f | sed -E 's/_grasp_log(_w[0-9]+)?\.csv$//'); out=$D/$tag.csv
  awk -F, -v out="$out" 'NR==1{for(i=1;i<=NF;i++) if($i=="executed") c=i; if(!hdr[out]++){ if(system("test -s " out)!=0) print > out }; next} $c=="1"{print >> out}' "$f"
done
ls -la $D | tail -n +2 | awk '{print $9, $5}' | head -40
