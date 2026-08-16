#!/usr/bin/env python3
"""
Bangun dataset COMBINED KITTI + UGM untuk joint training PointPillars.
- KITTI train (3712 frame, id 000000-007xxx) + UGM m300 train (300, id di-offset '9'+id).
- UGM di-OVERSAMPLE (×OVERSAMPLE) agar tak ditenggelamkan KITTI (7481 vs 300).
- Training PP hanya butuh velodyne + infos (annos sudah punya gt_boxes_lidar) → label/calib
  tak perlu disalin. gt_sampling pakai db UGM (paste objek UGM ke scene KITTI+UGM).
- Eval di UGM val (offset id) → GT pkl offset dibuat utk eval_range (cocokkan by frame_id).

Output: dataset_kitti_ugm/ (velodyne symlink + infos + dbinfos + ImageSets) + symlink OpenPCDet/data.
"""
import os, pickle, copy
from pathlib import Path

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
KITTI = ROOT / 'OpenPCDet/data/kitti'
UGM = ROOT / 'dataset_ugm_m300_v3'
OUT = ROOT / 'dataset_kitti_ugm'
OVERSAMPLE = 10
PREFIX = '9'   # offset id UGM: '000042' -> '9000042' (tak collision dgn KITTI 6-digit)

(OUT / 'training').mkdir(parents=True, exist_ok=True)
(OUT / 'ImageSets').mkdir(parents=True, exist_ok=True)
vel = OUT / 'training/velodyne'; vel.mkdir(exist_ok=True)

# ── 1. velodyne symlinks ────────────────────────────────────────────────
def link(src, dst):
    if not dst.exists(): dst.symlink_to(src)

kitti_vel = (KITTI / 'training/velodyne').resolve()
kitti_ids = [x['point_cloud']['lidar_idx'] for x in pickle.load(open(KITTI / 'kitti_infos_train.pkl','rb'))]
for i in kitti_ids:
    link(kitti_vel / f'{i}.bin', vel / f'{i}.bin')
ugm_vel = (UGM / 'training/velodyne').resolve()
ugm_tr = pickle.load(open(UGM / 'kitti_infos_train.pkl','rb'))
ugm_va = pickle.load(open(UGM / 'kitti_infos_val.pkl','rb'))
for inf in ugm_tr + ugm_va:
    oid = inf['point_cloud']['lidar_idx']
    link(ugm_vel / f'{oid}.bin', vel / f'{PREFIX}{oid}.bin')
print(f"velodyne: {len(kitti_ids)} KITTI + {len(ugm_tr)+len(ugm_va)} UGM(offset) symlink")

# ── 2. infos: offset UGM lidar_idx, oversample, merge ───────────────────
def offset_infos(infos):
    out = []
    for inf in infos:
        c = copy.deepcopy(inf)
        oid = c['point_cloud']['lidar_idx']
        c['point_cloud']['lidar_idx'] = f'{PREFIX}{oid}'
        if 'image' in c: c['image']['image_idx'] = f'{PREFIX}{oid}'
        out.append(c)
    return out

kitti_tr_infos = pickle.load(open(KITTI / 'kitti_infos_train.pkl','rb'))
ugm_tr_off = offset_infos(ugm_tr)
ugm_va_off = offset_infos(ugm_va)

combined_train = kitti_tr_infos + ugm_tr_off * OVERSAMPLE
pickle.dump(combined_train, open(OUT / 'kitti_infos_train.pkl','wb'))
pickle.dump(ugm_va_off, open(OUT / 'kitti_infos_val.pkl','wb'))
pickle.dump(kitti_tr_infos + ugm_tr_off, open(OUT / 'kitti_infos_trainval.pkl','wb'))
print(f"infos train: {len(kitti_tr_infos)} KITTI + {len(ugm_tr)}×{OVERSAMPLE} UGM = {len(combined_train)}")
print(f"infos val (UGM offset): {len(ugm_va_off)}")

# GT pkl utk eval (offset, cocok dgn hasil model combined)
pickle.dump(ugm_va_off, open(OUT / 'gt_val_offset.pkl','wb'))

# ── 3. gt_database: symlink dir + copy dbinfos (paths relatif 'gt_database/..') ──
link((UGM / 'gt_database').resolve(), OUT / 'gt_database')
import shutil
shutil.copy(UGM / 'kitti_dbinfos_train.pkl', OUT / 'kitti_dbinfos_train.pkl')
print("gt_database: symlink ke UGM db (gt_sampling pakai objek UGM)")

# ── 4. ImageSets (id unik, tanpa oversample) ────────────────────────────
(OUT / 'ImageSets/train.txt').write_text('\n'.join(kitti_ids + [f'{PREFIX}{x["point_cloud"]["lidar_idx"]}' for x in ugm_tr]) + '\n')
val_ids = [x['point_cloud']['lidar_idx'] for x in ugm_va_off]
(OUT / 'ImageSets/val.txt').write_text('\n'.join(val_ids) + '\n')
(OUT / 'ImageSets/test.txt').write_text('\n'.join(val_ids) + '\n')
(OUT / 'ImageSets/trainval.txt').write_text('\n'.join(kitti_ids + [f'{PREFIX}{x["point_cloud"]["lidar_idx"]}' for x in ugm_tr] + val_ids) + '\n')

# ── 5. symlink ke OpenPCDet/data ────────────────────────────────────────
opd = ROOT / 'OpenPCDet/data/dataset_kitti_ugm'
if opd.is_symlink() or opd.exists(): opd.unlink()
opd.symlink_to(OUT)
print(f"OpenPCDet/data link: {opd} -> {OUT}")
print("\nSELESAI. Next: buat dataset yaml + model config, lalu train.py")
