#!/usr/bin/env python3
"""
Bangun dataset YOLO 2-KELAS (Car, Cyclist) dari dataset_ugm_yolo_v3 (3-kelas).
Pedestrian di-EXCLUDE dari studi → buang semua label kelas 1, remap Cyclist 2→1.
Car tetap 0. Image di-symlink (hemat ruang). Output: dataset_ugm_yolo_2cls/.

Apples-to-apples dgn v3: image & bbox SAMA, satu-satunya beda = set kelas.
"""
from pathlib import Path

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
SRC  = ROOT / 'dataset_ugm_yolo_v3'
DST  = ROOT / 'dataset_ugm_yolo_2cls'
REMAP = {0: 0, 2: 1}   # Car→0, Cyclist→1 ; kelas 1 (Ped) dibuang

kept = {'Car': 0, 'Cyclist': 0}
for split in ('train', 'val'):
    (DST / 'images' / split).mkdir(parents=True, exist_ok=True)
    (DST / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for img in sorted((SRC / 'images' / split).glob('*')):
        link = DST / 'images' / split / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
    for lbl in sorted((SRC / 'labels' / split).glob('*.txt')):
        out = []
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            c = int(parts[0])
            if c not in REMAP:
                continue
            parts[0] = str(REMAP[c])
            out.append(' '.join(parts))
            kept['Car' if REMAP[c] == 0 else 'Cyclist'] += 1
        (DST / 'labels' / split / lbl.name).write_text('\n'.join(out) + ('\n' if out else ''))

yaml = f"""# YOLOv8 dataset UGM 2-KELAS (Car, Cyclist) — Pedestrian di-exclude.
# Diturunkan dari dataset_ugm_yolo_v3 (calib v3, bbox dari label). Image = symlink.
path: {DST}
train: images/train
val: images/val

names:
  0: Car
  1: Cyclist
"""
(DST / 'ugm2.yaml').write_text(yaml)
print(f"Dataset 2-kelas dibuat: {DST}")
print(f"  label objek → Car={kept['Car']}  Cyclist={kept['Cyclist']}")
print(f"  yaml: {DST/'ugm2.yaml'}")
