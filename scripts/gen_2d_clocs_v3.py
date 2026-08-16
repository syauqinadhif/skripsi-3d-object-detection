#!/usr/bin/env python3
"""
Generate deteksi 2D untuk CLOCs dari YOLO v3 (v8m_calibv3, dilatih label calib v3).

Output: satu .txt per frame di 2D_ugm_v3/, format KITTI + skor di kolom 16 (index 15):
  {Class} -1 -1 -10 x1 y1 x2 y2 -1 -1 -1 -1000 -1000 -1000 -10 {score}
Ini format yang dibaca CLOCs_LQS/tool/dataset.py (score = kolom ke-16).

conf rendah (0,01) = titik operasi CLOCs: semua kandidat masuk, fusi yang menyaring.
Model v3 sudah mengeluarkan kelas UGM langsung (0=Car,1=Pedestrian,2=Cyclist), jadi tak
perlu pemetaan COCO. Frame diambil dari index trainval m150 (228 frame = 150 train + 78 val).
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
IMG_DIR = ROOT / 'dataset_ugm/image_2'
WEIGHTS = ROOT / 'output/yolo_v3/v8m_calibv3/weights/best.pt'
INDEX = ROOT / 'CLOCs_LQS/data/clocs_data/index/ugm_m150_trainval.txt'
OUT = ROOT / 'CLOCs_LQS/data/clocs_data/2D_ugm_v3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.0)  # CLOCs: semua kandidat masuk
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
