#!/bin/bash
# eval_combined_rr.sh — eval PP combined (UGM + ringroad pseudolabel) di VAL MANUAL 80.
# Metrik sama persis dengan baseline: test.py -> result.pkl -> eval_infov.py (in-FOV).
set -e
ROOT=/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi
EPOCH=${1:-50}
CKPT=$ROOT/OpenPCDet/output/kitti_models/pointpillar_combined_rr/rr_pseudo/ckpt/checkpoint_epoch_$EPOCH.pth

cd "$ROOT/OpenPCDet/tools"
conda run -n openpcdet python test.py \
    --cfg_file cfgs/kitti_models/pointpillar_combined_rr.yaml \
    --ckpt "$CKPT" \
    --batch_size 4 \
    --extra_tag rr_pseudo \
    --save_to_file 2>&1 | grep -viE "it/s\]" | tail -30

RES=$ROOT/OpenPCDet/output/kitti_models/pointpillar_combined_rr/rr_pseudo/eval/epoch_$EPOCH/val/default/result.pkl
echo ""
echo "########## EVAL IN-FOV vs GT MANUAL (pembanding baseline) ##########"
cd "$ROOT"
conda run -n openpcdet python scripts/eval_infov.py "$RES" 2>&1 | grep -viE "it/s\]" | tail -40
