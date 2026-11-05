#!/bin/bash
#
# NOTE: paths are anchored to the original experiment host; adapt PKG/RUNS/CODE
# to your environment. The CUDA pre-init before any Ultralytics call is a
# workaround for select_device() overwriting CUDA_VISIBLE_DEVICES (see README).
# Phase 3-B v3: structural ablations + external baselines over a 4-GPU pool.
# v3 fixes vs v2:
#  - all `local X=$1 ... T="${X}..."` split into two lines (bash 5.2 expands
#    declaration-command words before assignments, so same-line references
#    read stale/caller-scope values -> variant det dirs collided as _f_d)
#  - do_export checks export_done BEFORE latest.pth (baseline chains have no
#    fusion checkpoint and aborted silently in v2)
#  - do_fusion resumes interrupted variant runs (--resume) instead of rm -rf
# All det/eval launches pin CUDA via context pre-init (select_device CVD fix).
export PATH=$HOME/miniconda3/envs/adaptivenn/bin:$PATH
# torch defaults to 64 OMP threads per process here; 4 concurrent jobs drove
# load to 102 on 64 cores (mostly spin). Cap per-process threads; fusion
# training is CPU-side anyway and leaves headroom for 2 chains per GPU.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PKG=/home/stu5/chengyu0918/LESAFusion_TAES_20261028
RUNS=$PKG/runs_mha_followup
CODE=$PKG/code
OUT3=$PKG/runs_mha_round3
M=$PKG/manifest/local_split_v1.json
CFGS=$OUT3/configs
mkdir -p $OUT3/logs
log() { echo "[$(date +%H:%M:%S)] $*" >> $OUT3/phase3b.log; }

make_cfg() {
  local V=$1 S=$2
  local C=$CFGS/${V}_s${S}.json
  [ -f "$C" ] && return 0
  python3 -c "
import json
c=json.load(open('$CFGS/h4_s3410.json'))
c['train']['seed']=$S
json.dump(c,open('$C','w'),indent=2)"
}

do_fusion() {
  local V=$1 S=$2 GPU=$3
  local T="${V}_s${S}"
  if [ "$V" = "cddfuse" ] || [ "$V" = "swinfusion" ]; then
    local O=$RUNS/fusion/$T
    if [ -f $O/export_done ] && [ -f $O/export_trainval/data.yaml ]; then log "SKIP baseline $V"; return 0; fi
    rm -rf $O; log "BASELINE_EXPORT $V gpu=$GPU"
    env CUDA_VISIBLE_DEVICES=$GPU python /tmp/export_baseline.py "$V" \
      "$O/export_trainval" train,val "$M" >> $OUT3/phase3b.log 2>&1
    env CUDA_VISIBLE_DEVICES=$GPU python /tmp/export_baseline.py "$V" \
      "$O/export_test" test "$M" >> $OUT3/phase3b.log 2>&1
    if [ -f $O/export_trainval/data.yaml ] && [ -f $O/export_test/data.yaml ]; then
      touch $O/export_done; log "BASELINE_DONE $V"; return 0
    fi
    log "FAIL baseline $V"; return 1
  fi
  make_cfg $V $S
  local O=$RUNS/fusion/$T
  if [ -f $O/completed.json ] && [ -f $O/variant_manifest.json ]; then log "SKIP fusion $T"; return 0; fi
  if [ -f $O/latest.pth ]; then
    log "RESUME fusion $T gpu=$GPU"
    env CUDA_VISIBLE_DEVICES=$GPU python /tmp/train_variant.py --variant $V \
      --config $CFGS/${V}_s${S}.json --run-dir $O --device cuda --resume > $OUT3/logs/fus_${T}.log 2>&1
  else
    rm -rf $O; log "START fusion $T gpu=$GPU"
    env CUDA_VISIBLE_DEVICES=$GPU python /tmp/train_variant.py --variant $V \
      --config $CFGS/${V}_s${S}.json --run-dir $O --device cuda > $OUT3/logs/fus_${T}.log 2>&1
  fi
  if [ -f $O/completed.json ] && [ -f $O/variant_manifest.json ]; then log "DONE fusion $T"
  else log "FAIL fusion $T"; return 1; fi
}

do_export() {
  local V=$1 S=$2 GPU=$3
  local T="${V}_s${S}"
  [ -f $RUNS/fusion/$T/export_done ] && return 0
  local K=$RUNS/fusion/$T/latest.pth
  [ ! -f "$K" ] && return 1
  log "EXPORT $T"
  env CUDA_VISIBLE_DEVICES=$GPU python /tmp/export_variant.py "$V" "$K" \
    "$RUNS/fusion/$T/export_trainval" train,val "$M" >> $OUT3/phase3b.log 2>&1
  env CUDA_VISIBLE_DEVICES=$GPU python /tmp/export_variant.py "$V" "$K" \
    "$RUNS/fusion/$T/export_test" test "$M" >> $OUT3/phase3b.log 2>&1
  touch $RUNS/fusion/$T/export_done; log "EXPORT_DONE $T"
}

do_det() {
  local V=$1 F=$2 D=$3 GPU=$4
  local T="${V}_f${F}_d${D}"
  local DA=$RUNS/fusion/${V}_s${F}/export_trainval/data.yaml
  [ ! -f "$DA" ] && { log "SKIP det $T (no data)"; return 1; }
  local O=$RUNS/detection/$T
  [ -f $O/completed.json ] && { log "SKIP det $T"; return 0; }
  rm -rf $O; log "START det $T gpu=$GPU"
  env CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys, torch; sys.path.insert(0,'$CODE')
_ = torch.zeros(1, device='cuda')
from pkd.detection import train_detector
train_detector('$DA','$O',afpn=False,
  initial_weights='$PKG/yolov8n.pt',nc=1,
  epochs=50,batch=16,imgsz=640,seed=$D,workers=4,device='cuda')
" > $OUT3/logs/det_${T}.log 2>&1
  [ -f $O/completed.json ] && log "DONE det $T" || log "FAIL det $T"
}

do_eval() {
  local V=$1 F=$2 D=$3 GPU=$4
  local T="${V}_f${F}_d${D}"
  local DE=$RUNS/detection/$T/weights/best.pt
  [ ! -f "$DE" ] && return 1
  [ -f $RUNS/eval/$T.json ] && return 0
  local TV=$RUNS/fusion/${V}_s${F}/export_trainval
  local TE=$RUNS/fusion/${V}_s${F}/export_test
  log "EVAL $T"
  env CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys, torch; sys.path.insert(0,'$CODE'); sys.path.insert(0,'/tmp')
_ = torch.zeros(1, device='cuda')
from adapter_eval import evaluate_run
evaluate_run(run_name='$T',det_ckpt='$DE',
  val_images='$TV/images/val',val_labels='$TV/labels/val',val_yaml='x',
  test_images='$TE/images/test',test_labels='$TE/labels/test',test_yaml='x',
  out_dir='$RUNS/eval',device='cuda')" >> $OUT3/phase3b.log 2>&1
  log "EVAL_DONE $T"
}

chain() {
  local V=$1 S=$2 GPU=$3
  do_fusion $V $S $GPU || return 1
  do_export $V $S $GPU || return 1
  for D in 0 1 2; do
    do_det $V $S $D $GPU
    do_eval $V $S $D $GPU
  done
  log "CHAIN_DONE $V s=$S"
}

JOBS=("cddfuse 0" "swinfusion 0" "noattn 3407" "noattn 3408" "noattn 3409" "nodfu 3407" "nodfu 3408" "nodfu 3409" "coarse 3407" "coarse 3408" "coarse 3409")
# 6 chain slots over 4 GPUs (slot -> gpu): fusion chains are CPU-bound at
# 2-9% GPU, so two chains may share a GPU; co-located detections merely share
# the device (~1.5x slower each, still a net win).
SLOT_GPUS=(0 1 2 3 0 1)
declare -A SLOT
next_slot() {
  for i in "${!SLOT_GPUS[@]}"; do
    if [ -z "${SLOT[$i]}" ] || ! kill -0 "${SLOT[$i]}" 2>/dev/null; then
      echo "$i ${SLOT_GPUS[$i]}"; return
    fi
  done
}
log "PHASE3B_V3_START"
for J in "${JOBS[@]}"; do
  set -- $J; V=$1; S=$2
  while :; do
    read -r SI G < <(next_slot)
    [ -n "$G" ] && break
    sleep 60
  done
  chain $V $S $G & SLOT[$SI]=$!
  log "POOL_START $V s=$S gpu=$G slot=$SI pid=${SLOT[$SI]}"
  sleep 30
done
wait
log "PHASE3B_COMPLETE"
echo "PHASE3B_ALL_DONE $(date)"
