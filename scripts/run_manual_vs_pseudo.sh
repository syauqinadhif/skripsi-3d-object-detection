#!/bin/bash
# Eksperimen: label manual 150 vs pseudolabel, di data train UGM.
#
#  run 1  m150   @50ep  : 150 frame manual        (recipe SAMA dgn baseline pointpillar_ugm)
#  run 2  mix394 @50ep  : 150 manual + 244 pseudo (recipe SAMA)  <-- perbandingan utama
#  run 3  m150   @130ep : 150 manual, ~jumlah iterasi setara run 2 (150*130 ~= 394*50)
#         -> kontrol supaya "m150 kalah" tidak bisa dijelaskan sekadar undertrained.
#
# Semua eval pakai val = 80 frame VAL MANUAL (GT jujur).
set -e
cd /media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/OpenPCDet/tools
LOG=/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/output/logs_manual_vs_pseudo
mkdir -p $LOG

run () {  # $1=cfg  $2=epochs  $3=extra_tag
  echo "=== $(date +%H:%M:%S)  TRAIN $1 ep=$2 tag=$3 ==="
  python train.py --cfg_file cfgs/kitti_models/$1.yaml \
      --batch_size 4 --epochs $2 --workers 4 --extra_tag $3 \
      > $LOG/train_$1_$3.log 2>&1
  echo "=== $(date +%H:%M:%S)  EVAL_ALL $1 tag=$3 ==="
  # --max_waiting_mins 0: tanpa ini test.py --eval_all NUNGGU 30 menit checkpoint baru
  # yang tak akan pernah datang (training sudah selesai) -> 30 menit idle per run.
  python test.py --cfg_file cfgs/kitti_models/$1.yaml \
      --batch_size 4 --workers 4 --extra_tag $3 --eval_all --max_waiting_mins 0 \
      > $LOG/eval_$1_$3.log 2>&1
}

# run 1 (pointpillar_ugm_m150 ep50) SUDAH SELESAI 2026-07-18 15:33 — dilewati.
run pointpillar_ugm_mix394 50  ep50
run pointpillar_ugm_m150   130 ep130_itermatched

echo "=== $(date +%H:%M:%S) SELESAI SEMUA ==="
