#!/usr/bin/env python3
"""
Generate deteksi 2D untuk CLOCs dari YOLO 2-KELAS (v8m_2cls, Car+Cyclist, Ped di-exclude).
Sama dgn gen_2d_clocs_v3.py tapi model 2-kelas → tak ada FP Pedestrian yang membanjiri fusi.
CLOCs mencocokkan 2D-3D by NAMA kelas, jadi Car/Cyclist selaras; kandidat 3D Ped tak dapat
pasangan 2D (di-exclude dari studi, jadi tak masalah).

Output: 2D_ugm_y26_2cls/<fid>.txt, format KITTI + skor kolom-16 (dibaca CLOCs tool/dataset.py).
conf=0.0 WAJIB: semua kandidat masuk, fusi yang menyaring. conf tinggi = fusi kolaps.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
CLASSES = ['Car', 'Cyclist']                       # indeks model 2-kelas: 0=Car, 1=Cyclist
IMG_DIR = ROOT / 'dataset_ugm/image_2'
WEIGHTS = ROOT / 'output/yolo26_2cls/y26m_2cls/weights/best.pt'
INDEX = ROOT / 'CLOCs_LQS/data/clocs_data/index/ugm_m150_trainval.txt'
OUT = ROOT / 'CLOCs_LQS/data/clocs_data/2D_ugm_y26_2cls'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.0)
    ap.add_argument('--imgsz', type=int, default=1280)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(WEIGHTS))
    ids = [x.strip() for x in open(INDEX) if x.strip()]
    n_det = 0
    per_cls = {c: 0 for c in CLASSES}
    for fid in ids:
        img = IMG_DIR / f'{fid}.png'
        lines = []
        if img.exists():
            r = m(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            for b in r.boxes:
                c = int(b.cls[0])
                if c < 0 or c >= len(CLASSES):
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                s = float(b.conf[0])
                lines.append(f"{CLASSES[c]} -1 -1 -10 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                             f"-1 -1 -1 -1000 -1000 -1000 -10 {s:.4f}")
                per_cls[CLASSES[c]] += 1
                n_det += 1
        (OUT / f'{fid}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print(f"{len(ids)} frame -> {OUT}")
    print(f"total deteksi {n_det} ({n_det/len(ids):.1f}/frame)  {per_cls}")


if __name__ == '__main__':
    main()
