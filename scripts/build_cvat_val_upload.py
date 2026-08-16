#!/usr/bin/env python3
"""
Susun folder upload CVAT 3D untuk VAL set: point cloud (.pcd) + context image.
Struktur output (format point-cloud CVAT):

  cvat_val_upload/
  ├── pointcloud/
  │   ├── 000001.pcd ...
  └── related_images/
      ├── 000001_pcd/
      │   └── 000001.png ...

Konversi velodyne .bin (N,4 float32 x,y,z,intensity) -> PCD binary (header + raw bytes).
"""
import shutil
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
VEL = ROOT / 'dataset_ugm/velodyne'
IMG = ROOT / 'dataset_ugm/image_2'
OUT = ROOT / 'cvat_val_upload'

PCD_DIR = OUT / 'pointcloud'
REL_DIR = OUT / 'related_images'
PCD_DIR.mkdir(parents=True, exist_ok=True)
REL_DIR.mkdir(parents=True, exist_ok=True)


def bin_to_pcd(bin_path: Path, pcd_path: Path):
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    n = pts.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(pcd_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(pts.tobytes())  # (N,4) float32 = body PCD binary


val_ids = (ROOT / 'dataset_ugm/ImageSets/val.txt').read_text().split()
print(f"Membangun folder upload CVAT untuk {len(val_ids)} frame val...\n")

n_img = 0
for fid in val_ids:
    bin_to_pcd(VEL / f"{fid}.bin", PCD_DIR / f"{fid}.pcd")
    rel = REL_DIR / f"{fid}_pcd"
    rel.mkdir(parents=True, exist_ok=True)
    src_img = IMG / f"{fid}.png"
    if src_img.exists():
        shutil.copy(src_img, rel / f"{fid}.png")
        n_img += 1

print(f"  pointcloud/      : {len(list(PCD_DIR.glob('*.pcd')))} .pcd")
print(f"  related_images/  : {len(list(REL_DIR.glob('*_pcd')))} folder, {n_img} gambar")
print(f"\nFolder siap: {OUT}")
