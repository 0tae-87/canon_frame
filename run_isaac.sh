#!/bin/bash
# Isaac run of one regressor checkpoint. env: SEED (0), STAGE = both | full | lat04 | lat025
#   usage: SEED=1 STAGE=both bash run_isaac_variant_seed.sh <ckpt> <arm> <port> [extra docker env]
set -u; cd /home/wim/Desktop/yt_ws/PoinTr; O=icra/isaac_graspgen/output/graspgen; export ISAAC_INDY7_CPU_THREADS=2
CKPT=$1; ARM=$2; PORT=${3:-5563}; EXTRA_ENV=${4:-}; SEED=${SEED:-0}; STAGE=${STAGE:-both}
OBJS="005_tomato_soup_can 006_mustard_bottle 010_potted_meat_can 011_banana 021_bleach_cleanser 022_windex_bottle 025_mug 035_power_drill 061_foam_brick 003_cracker_box 004_sugar_box 008_pudding_box 009_gelatin_box 019_pitcher_base 051_large_clamp 052_extra_large_clamp"
docker rm -f canon_$ARM >/dev/null 2>&1
docker run -d --name canon_$ARM --gpus all --network host --ipc=host -v /home/wim/Desktop/yt_ws:/workspace -w /workspace/PoinTr -e COMPLETION_PORT=$PORT -e CANON_YAW=0 -e CANON_REG=/workspace/PoinTr/$CKPT $EXTRA_ENV pointr_blackwell:gpufix python canon_frame/engine/server_canon.py >/dev/null
for i in $(seq 1 60); do sleep 3; docker logs canon_$ARM 2>&1 | grep -q listening && break; done
run_stage () {  # tag extra-flags
  local TAG=$1 EXTRA=$2; echo "$TAG launch $(date +%H:%M) $CKPT" >> $O/${TAG}_chain.log
  for w in 0 1 2 3 4 5; do
    nohup python3 icra/run_batch_m4.py --tag $TAG --seeds $SEED --trials 50 --dump --fracs --no-control $EXTRA --objects $OBJS --canon-arms $ARM:$PORT --shard $w --num-shards 6 > $O/${TAG}_w${w}_$(date +%H%M).log 2>&1 &
    sleep 4
  done
  sleep 120; while pgrep -f "run_batch_m4.py --tag $TAG " > /dev/null; do sleep 120; done
  # retry aborted cells once (done cells are skipped)
  if [ "$(ls $O/${TAG}_done | wc -l)" -lt 16 ]; then
    echo "$TAG retry $(date +%H:%M) ($(ls $O/${TAG}_done | wc -l)/16)" >> $O/${TAG}_chain.log
    for w in 0 1 2 3; do nohup python3 icra/run_batch_m4.py --tag $TAG --seeds $SEED --trials 50 --dump --fracs --no-control $EXTRA --objects $OBJS --canon-arms $ARM:$PORT --shard $w --num-shards 4 > $O/${TAG}_retry_w${w}.log 2>&1 & sleep 4; done
    sleep 120; while pgrep -f "run_batch_m4.py --tag $TAG " > /dev/null; do sleep 120; done
  fi
  echo "$TAG done $(date +%H:%M): $(ls $O/${TAG}_done | wc -l)/16 cells" >> $O/${TAG}_chain.log; touch $O/${TAG}_ALL_DONE
}
case $STAGE in
  both)  run_stage run${ARM}Full_s${SEED} ""; run_stage run${ARM}Lat04_s${SEED} "--obs-lateral-crop 0.4" ;;
  full)  run_stage run${ARM}Full_s${SEED} "" ;;
  lat04) run_stage run${ARM}Lat04_s${SEED} "--obs-lateral-crop 0.4" ;;
  lat025) run_stage run${ARM}L025_s${SEED} "--obs-lateral-crop 0.25" ;;
esac
