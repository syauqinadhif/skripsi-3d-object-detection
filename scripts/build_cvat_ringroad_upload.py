#!/usr/bin/env python3
"""
build_cvat_ringroad_upload.py — paket labeling manual CVAT 3D untuk RINGROAD.

Isi:
  cvat_ringroad_upload/
  ├── pointcloud/000xxx.pcd            <- dari velodyne .bin (binary PCD, x y z intensity)
  ├── related_images/000xxx_pcd/000xxx.png
  ├── label/000xxx.json                <- pre-annotation ft30 (format Sustech POINTS)
  └── PANDUAN_LABELING.txt

Pre-annotation dibaca dari label_2_ft30 (KITTI, koordinat kamera) lalu DIKONVERSI BALIK
ke frame LiDAR supaya selaras dengan .pcd yang diupload (CVAT 3D bekerja di frame LiDAR).

Usage:
  conda run -n openpcdet python scripts/build_cvat_ringroad_upload.py --n 300
"""
import argparse, json, sys, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
from pcdet.utils import object3d_kitti, calibration_kitti
from tqdm import tqdm

RR = ROOT / 'dataset_ringroad'
OUT = ROOT / 'cvat_ringroad_upload'
CLASSES = ['Car', 'Pedestrian', 'Cyclist']

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--data_dir', default=str(RR), help='dataset root (velodyne/, image_2/, calib/)')
ap.add_argument('--ids_file', default=None, help='pilih N frame dari daftar id ini (default: semua .bin)')
ap.add_argument('--label_dir', default=str(RR / 'label_2_ft30'))
ap.add_argument('--out', default=str(OUT))
args = ap.parse_args()

OUT = Path(args.out)
RR = Path(args.data_dir)
label_dir = Path(args.label_dir)
PCD_DIR = OUT / 'pointcloud'
REL_DIR = OUT / 'related_images'
LAB_DIR = OUT / 'label'
for d in (PCD_DIR, REL_DIR, LAB_DIR):
    d.mkdir(parents=True, exist_ok=True)


def bin_to_pcd(bin_path: Path, pcd_path: Path):
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    n = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
        f"COUNT 1 1 1 1\nWIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    with open(pcd_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(pts.tobytes())


def kitti_label_to_lidar_objs(lf: Path, calib):
    """KITTI label (frame kamera) -> list objek di frame LiDAR untuk Sustech POINTS."""
    if not lf.exists() or lf.stat().st_size == 0:
        return []
    objs = [o for o in object3d_kitti.get_objects_from_label(str(lf)) if o.cls_type in CLASSES]
    if not objs:
        return []
    loc = np.array([o.loc for o in objs])
    loc_lidar = calib.rect_to_lidar(loc)
    out = []
    for i, o in enumerate(objs):
        x, y, z = [float(v) for v in loc_lidar[i]]
        z += o.h / 2                       # KITTI location = bottom-center -> pusat box
        yaw = -(o.ry + np.pi / 2)          # kebalikan dari ry = -yaw - pi/2
        yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
        out.append({
            "obj_id": str(i),
            "obj_type": o.cls_type,
            "psr": {
                "position": {"x": x, "y": y, "z": z},
                "rotation": {"x": 0.0, "y": 0.0, "z": yaw},
                "scale": {"x": float(o.l), "y": float(o.w), "z": float(o.h)},
            },
        })
    return out


if args.ids_file:
    all_ids = sorted(Path(args.ids_file).read_text().split())
else:
    all_ids = sorted(p.stem for p in (RR / 'velodyne').glob('*.bin'))
# ambil N frame tersebar merata (bukan N frame berurutan di satu ruas jalan)
idx = np.round(np.linspace(0, len(all_ids) - 1, args.n)).astype(int)
ids = [all_ids[i] for i in sorted(set(idx))]

print(f"\nPaket CVAT: {len(ids)} frame dari {len(all_ids)} kandidat")
print(f"  data           : {RR}")
print(f"  pre-annotation : {label_dir}\n")

total = 0
for fid in tqdm(ids, desc='build'):
    bin_to_pcd(RR / 'velodyne' / f'{fid}.bin', PCD_DIR / f'{fid}.pcd')
    rel = REL_DIR / f'{fid}_pcd'
    rel.mkdir(exist_ok=True)
    shutil.copy(RR / 'image_2' / f'{fid}.png', rel / f'{fid}.png')   # resolve symlink
    calib = calibration_kitti.Calibration(str(RR / 'calib' / f'{fid}.txt'))
    objs = kitti_label_to_lidar_objs(label_dir / f'{fid}.txt', calib)
    total += len(objs)
    (LAB_DIR / f'{fid}.json').write_text(json.dumps(objs, indent=1))

(OUT / 'ImageSets_terpilih.txt').write_text('\n'.join(ids) + '\n')
print(f"\n{len(ids)} frame, {total} pre-annotation box ({total/len(ids):.1f}/frame)")
print(f"Output: {OUT}")
