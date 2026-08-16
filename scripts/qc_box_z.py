#!/usr/bin/env python3
"""
qc_box_z.py — QC ketinggian cuboid tanpa perlu GT.

Metrik: persen titik di dalam footprint BEV box yang berada DI BAWAH dasar box.
Box yang melayang -> banyak titik body objek terpotong di bawah -> persen naik.
Acuan: label manual val80 user (dianggap benar) diukur dengan metrik yang sama.

Usage:
  python scripts/qc_box_z.py --data_dir dataset_ringroad --label_dir dataset_ringroad/label_2
  python scripts/qc_box_z.py --data_dir dataset_ugm --label_dir dataset_ugm/label_2_val_manual
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
from pcdet.utils import box_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common  # noqa
from pcdet.utils import object3d_kitti

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='dataset_ringroad')
ap.add_argument('--label_dir', default=None)
ap.add_argument('--max_frames', type=int, default=200)
args = ap.parse_args()

data_dir = ROOT / args.data_dir
label_dir = Path(args.label_dir) if args.label_dir else data_dir / 'label_2'
if not label_dir.is_absolute():
    label_dir = ROOT / label_dir

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
frames = sorted(p.stem for p in label_dir.glob('*.txt'))[:args.max_frames]
print(f"QC z: {label_dir}  ({len(frames)} frame)\n")

stats = {c: [] for c in CLASSES}
zc = {c: [] for c in CLASSES}

for fid in tqdm(frames, desc='qc'):
    lf = label_dir / f'{fid}.txt'
    if lf.stat().st_size == 0:
        continue
    objs = object3d_kitti.get_objects_from_label(str(lf))
    objs = [o for o in objs if o.cls_type in CLASSES]
    if not objs:
        continue
    calib = calibration_kitti.Calibration(str(data_dir / 'calib' / f'{fid}.txt'))
    loc = np.array([o.loc for o in objs])
    dims = np.array([[o.l, o.w, o.h] for o in objs])
    rots = np.array([o.ry for o in objs])
    loc_lidar = calib.rect_to_lidar(loc)
    loc_lidar[:, 2] += dims[:, 2] / 2          # bottom-center -> geometric center
    boxes = np.concatenate(
        [loc_lidar, dims, -(np.pi / 2 + rots[:, None])], axis=1)

    pts = np.fromfile(str(data_dir / 'velodyne' / f'{fid}.bin'),
                      dtype=np.float32).reshape(-1, 4)[:, :3]

    for i, o in enumerate(objs):
        b = boxes[i]
        # rotasi titik ke frame box, ambil yang di dalam footprint BEV (x,y)
        dxy = pts[:, :2] - b[:2]
        c, s = np.cos(-b[6]), np.sin(-b[6])
        lx = dxy[:, 0] * c - dxy[:, 1] * s
        ly = dxy[:, 0] * s + dxy[:, 1] * c
        m = (np.abs(lx) <= b[3] / 2) & (np.abs(ly) <= b[4] / 2)
        if m.sum() < 5:
            continue
        z = pts[m, 2]
        bottom = b[2] - b[5] / 2
        stats[o.cls_type].append(float((z < bottom).mean()) * 100)
        zc[o.cls_type].append(float(b[2]))

print(f"\n{'kelas':<11} {'n':>5} {'%titik_bawah_med':>17} {'mean':>7} "
      f"{'p90':>7} {'z_center_med':>13}")
for c in CLASSES:
    v = np.array(stats[c])
    if len(v) == 0:
        print(f"{c:<11} {'0':>5}   (tidak ada box)")
        continue
    print(f"{c:<11} {len(v):>5} {np.median(v):>17.2f} {v.mean():>7.2f} "
          f"{np.percentile(v, 90):>7.2f} {np.median(zc[c]):>13.3f}")
print("\nMakin besar %titik_bawah = box makin melayang di atas objek.")
