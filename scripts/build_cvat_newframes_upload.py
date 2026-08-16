#!/usr/bin/env python3
"""
Bungkus paket upload CVAT 3D untuk TOP-N frame baru Car-terpadat.
Struktur (format point-cloud CVAT):
  cvat_newframes_v3_upload/
  ├── pointcloud/<fid>.pcd
  ├── related_images/<fid>_pcd/<fid>.png
  └── annotations/<fid>.json          (pra-anotasi Sustech POINTS, sudah di-snap)
+ selected_ids.txt (daftar fid terpilih)  + .zip

Pra-anotasi diambil dari cvat_newframes_v3/label/ (hasil gen_pseudolabel_newframes_v3.py).
Frame dipilih dari counts.csv, diurutkan Car desc, ambil TOP_N.
"""
import shutil, csv
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
VEL  = ROOT / 'dataset_ugm/velodyne'
IMG  = ROOT / 'dataset_ugm/image_2'
SRC  = ROOT / 'cvat_newframes_v3'          # dari generator: label/, counts.csv
OUT  = ROOT / 'cvat_newframes_v3_upload'
TOP_N = 150

PCD_DIR = OUT / 'pointcloud'
REL_DIR = OUT / 'related_images'
ANN_DIR = OUT / 'annotations'


def bin_to_pcd(bin_path, pcd_path):
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    n = pts.shape[0]
    header = ("# .PCD v0.7 - Point Cloud Data file format\n"
              "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
              "COUNT 1 1 1 1\n" f"WIDTH {n}\nHEIGHT 1\n"
              "VIEWPOINT 0 0 0 1 0 0 0\n" f"POINTS {n}\nDATA binary\n")
    with open(pcd_path, 'wb') as f:
        f.write(header.encode('ascii')); f.write(pts.tobytes())


rows = list(csv.DictReader(open(SRC / 'counts.csv')))
rows.sort(key=lambda r: int(r['car']), reverse=True)
sel = [r['frame_id'] for r in rows[:TOP_N]]

for d in (PCD_DIR, REL_DIR, ANN_DIR):
    d.mkdir(parents=True, exist_ok=True)

n_img = n_ann = tot_car = tot_cyc = 0
for r in rows[:TOP_N]:
    fid = r['frame_id']
    bin_to_pcd(VEL / f"{fid}.bin", PCD_DIR / f"{fid}.pcd")
    rel = REL_DIR / f"{fid}_pcd"; rel.mkdir(parents=True, exist_ok=True)
    img = IMG / f"{fid}.png"
    if img.exists():
        shutil.copy(img, rel / f"{fid}.png"); n_img += 1
    ann = SRC / 'label' / f"{fid}.json"
    if ann.exists():
        shutil.copy(ann, ANN_DIR / f"{fid}.json"); n_ann += 1
    tot_car += int(r['car']); tot_cyc += int(r['cyclist'])

(OUT / 'selected_ids.txt').write_text('\n'.join(sorted(sel)) + '\n')
shutil.make_archive(str(OUT), 'zip', str(OUT))

print(f"TOP-{TOP_N} frame Car-terpadat dibungkus.")
print(f"  pointcloud/     : {len(list(PCD_DIR.glob('*.pcd')))} .pcd")
print(f"  related_images/ : {n_img} context image")
print(f"  annotations/    : {n_ann} pra-anotasi JSON (Car={tot_car} Cyc={tot_cyc})")
print(f"  selected_ids.txt: {len(sel)} id")
print(f"\nPaket : {OUT}")
print(f"Zip   : {OUT}.zip  ({(OUT.with_suffix('.zip')).stat().st_size/1e6:.0f} MB)" if (OUT.with_suffix('.zip')).exists() else "")
