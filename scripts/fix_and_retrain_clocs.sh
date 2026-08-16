#!/bin/bash
# fix_and_retrain_clocs.sh
#
# Fixes CLOCs training yang hasilnya tidak meningkat karena 3D candidates
# yang dipakai adalah post-NMS (~10-20 boxes/frame, sigmoid scores).
#
# Penyebab root: clocs_data/3D/ berisi hasil post-NMS PointPillar, bukan
# pre-NMS. CLOCs butuh ~500 kandidat/frame (loose NMS, raw logit scores)
# agar 2D-3D spatial overlap cukup untuk belajar fusion.
#
# Yang dilakukan script ini:
#   Step 1 — Backup log & checkpoint run pertama (rename yolov8m -> yolov8m_run1)
#   Step 2 — Generate pre-NMS candidates untuk train split via OpenPCDet test.py
#   Step 3 — Convert val + train result.pkl -> individual .pt files di clocs_data/3D/
#   Step 4 — Hapus input_data cache (dibangun dari data yang salah)
#   Step 5 — Regenerate input_data tensors via generate_data.py
#   Step 6 — Re-run CLOCs training 50 epoch
#
# Cara pakai:
#   bash scripts/fix_and_retrain_clocs.sh
#
# Pastikan conda environment 'openpcdet' sudah ada dan terinstall dengan benar.

set -e  # exit on error

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT="/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi"
OPENPCDET="$PROJECT_ROOT/OpenPCDet"
CLOCS_LQS="$PROJECT_ROOT/CLOCs_LQS"
SCRIPTS="$PROJECT_ROOT/scripts"

CKPT="$OPENPCDET/pointpillar_7728.pth"

# Pre-NMS val pkl (sudah ada dari run sebelumnya)
VAL_PKL="$OPENPCDET/output/kitti_models/pointpillar_prenms/loose_nms/eval/epoch_7728/val/default/result.pkl"

# Pre-NMS train pkl (akan digenerate di Step 2)
TRAIN_PKL="$OPENPCDET/output/kitti_models/pointpillar_prenms_train/loose_nms/eval/epoch_7728/train/default/result.pkl"

# Target 3D directory di CLOCs_LQS
OUTPUT_3D="$CLOCS_LQS/data/clocs_data/3D"

# CLOCs_LQS internal paths
D2_PATH="./data/clocs_data/2D"
D3_PATH="./data/clocs_data/3D"
INPUT_DATA="./data/clocs_data/input_data"
TRAIN_INDEX="./index/train.txt"
VAL_INDEX="./index/val.txt"
TRAINVAL_INDEX="./index/trainval.txt"
INFO_PATH="./data/clocs_data/info/kitti_infos_trainval.pkl"
CFG_FILE="./tool/cfgs/kitti_models/pointpillar/pointpillar_car.yaml"
LOG_PATH="./log/pointpillar/yolov8m"

# ── Helper ───────────────────────────────────────────────────────────────────

step() { echo; echo "══════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════"; }

# ── Step 1: Backup run pertama ────────────────────────────────────────────────

step "Step 1/6 — Backup log & checkpoint run pertama"

LOG_DIR="$CLOCS_LQS/log/pointpillar/yolov8m"
BACKUP_DIR="$CLOCS_LQS/log/pointpillar/yolov8m_run1"

if [ -d "$LOG_DIR" ]; then
    if [ -d "$BACKUP_DIR" ]; then
        echo "  [SKIP] Backup sudah ada di $BACKUP_DIR"
    else
        mv "$LOG_DIR" "$BACKUP_DIR"
        echo "  [OK] Backup berhasil: yolov8m -> yolov8m_run1"
        echo "  Log run pertama: $BACKUP_DIR/log.txt"
        echo "  Checkpoint run pertama: $BACKUP_DIR/48.pt (best epoch 48)"
    fi
else
    echo "  [SKIP] Log dir tidak ditemukan, tidak perlu backup"
fi

# ── Step 2: Generate train pre-NMS candidates ─────────────────────────────────

step "Step 2/6 — Generate pre-NMS candidates untuk train split"

if [ -f "$TRAIN_PKL" ]; then
    echo "  [SKIP] Train pre-NMS PKL sudah ada: $TRAIN_PKL"
    TRAIN_COUNT=$(python3 -c "import pickle; d=pickle.load(open('$TRAIN_PKL','rb')); print(len(d))")
    echo "  -> $TRAIN_COUNT frames, sample candidacy: $(python3 -c "import pickle; d=pickle.load(open('$TRAIN_PKL','rb')); print(len(d[0]['score']))")/frame"
else
    echo "  Running OpenPCDet test.py dengan pointpillar_prenms_train.yaml..."
    echo "  Output: $TRAIN_PKL"
    echo "  (Ini akan memakan waktu ~5-10 menit)"

    cd "$OPENPCDET/tools"
    conda run -n openpcdet --no-capture-output python test.py \
        --cfg_file cfgs/kitti_models/pointpillar_prenms_train.yaml \
        --ckpt "$CKPT" \
        --extra_tag loose_nms \
        --batch_size 4 \
        --workers 4 \
        --save_to_file

    if [ ! -f "$TRAIN_PKL" ]; then
        echo "ERROR: result.pkl tidak ditemukan di $TRAIN_PKL"
        echo "Cek output OpenPCDet di atas untuk error."
        exit 1
    fi
    echo "  [OK] Train pre-NMS PKL berhasil digenerate"
fi

# ── Step 3: Convert pkl -> individual .pt files ───────────────────────────────

step "Step 3/6 — Convert result.pkl -> individual .pt files di clocs_data/3D/"

echo "  Val pkl : $VAL_PKL"
echo "  Train pkl: $TRAIN_PKL"
echo "  Output  : $OUTPUT_3D"
echo "  (Ini menggantikan semua file di 3D/ dengan pre-NMS candidates)"

if [ ! -f "$VAL_PKL" ]; then
    echo "ERROR: Val PKL tidak ditemukan: $VAL_PKL"
    exit 1
fi

conda run -n openpcdet --no-capture-output python "$SCRIPTS/convert_pkl_to_pt.py" \
    --val_pkl   "$VAL_PKL" \
    --train_pkl "$TRAIN_PKL" \
    --output_dir "$OUTPUT_3D"

echo "  [OK] Konversi selesai"

# ── Step 4: Hapus input_data cache ───────────────────────────────────────────

step "Step 4/6 — Hapus input_data cache (built from wrong post-NMS data)"

INPUT_DATA_DIR="$CLOCS_LQS/data/clocs_data/input_data"
CACHE_COUNT=$(ls "$INPUT_DATA_DIR" 2>/dev/null | wc -l)

if [ "$CACHE_COUNT" -gt 0 ]; then
    echo "  Menghapus $CACHE_COUNT file cache di $INPUT_DATA_DIR..."
    rm -f "$INPUT_DATA_DIR"/*.pt
    echo "  [OK] Cache dihapus"
else
    echo "  [SKIP] input_data sudah kosong"
fi

# ── Step 5: Regenerate input_data tensors ─────────────────────────────────────

step "Step 5/6 — Regenerate input_data tensors (generate_data.py)"

echo "  Ini membangun ulang cached tensors 2D-3D IoU dari pre-NMS data."
echo "  Estimasi waktu: 15-30 menit (7481 frames)"

cd "$CLOCS_LQS"
conda run -n openpcdet --no-capture-output python generate_data.py \
    --cfg_file    "$CFG_FILE" \
    --d2path      "$D2_PATH" \
    --d3path      "$D3_PATH" \
    --inputpath   "$INPUT_DATA" \
    --train-val-indexpath "$TRAINVAL_INDEX" \
    --infopath    "$INFO_PATH"

NEW_CACHE=$(ls "$INPUT_DATA_DIR" 2>/dev/null | wc -l)
echo "  [OK] $NEW_CACHE cache files digenerate"

# ── Step 6: Re-run CLOCs training ────────────────────────────────────────────

step "Step 6/6 — Re-run CLOCs training (50 epochs)"

echo "  Log akan disimpan ke: $CLOCS_LQS/$LOG_PATH"
echo "  Estimasi waktu: ~15-20 menit"

cd "$CLOCS_LQS"
conda run -n openpcdet --no-capture-output python train.py \
    --cfg_file    "$CFG_FILE" \
    --d2path      "$D2_PATH" \
    --d3path      "$D3_PATH" \
    --inputpath   "$INPUT_DATA" \
    --train-indexpath "$TRAIN_INDEX" \
    --val-indexpath   "$VAL_INDEX" \
    --infopath    "$INFO_PATH" \
    --d2method    yolov8m \
    --d3method    pointpillar \
    --log-path    "$LOG_PATH" \
    --epochs      50

step "Selesai!"
echo "  Checkpoint tersimpan di: $CLOCS_LQS/$LOG_PATH/"
echo "  Log training: $CLOCS_LQS/$LOG_PATH/log.txt"
echo "  Run 1 (salah): $CLOCS_LQS/log/pointpillar/yolov8m_run1/"
echo ""
echo "  Untuk melihat hasil eval setiap epoch:"
echo "  grep '3d   AP:' $CLOCS_LQS/$LOG_PATH/log.txt"
