#!/usr/bin/env python3
"""
Bandingkan YOLO lama (v8m_cy399, calib salah) vs baru (v8m_calibv3) di TITIK OPERASI CLOCs.

Kenapa bukan mAP50 bawaan Ultralytics: masing-masing model dievaluasi terhadap dataset val-nya
SENDIRI (yang GT-nya beda), jadi angka mAP-nya tidak sebanding. Di sini keduanya diadu pada
GT YANG SAMA (label v3, 78 frame) supaya adil.

Titik operasi CLOCs = conf sangat rendah (semua kandidat masuk, fusi yang menyaring), jadi
metrik yang penting = RECALL pada conf rendah, bukan mAP.
"""
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
GT_DIR = ROOT / 'dataset_ugm_yolo_v3/labels/val'
IMG_DIR = ROOT / 'dataset_ugm_yolo_v3/images/val'
IMG_W, IMG_H = 1280, 720


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def load_gt(fid):
    out = []
    f = GT_DIR / f'{fid}.txt'
    if not f.exists():
        return out
    for l in f.read_text().split('\n'):
        if not l.strip():
            continue
        t = l.split(); c = int(t[0])
        x, y, w, h = (float(v) for v in t[1:5])
        out.append((c, [(x-w/2)*IMG_W, (y-h/2)*IMG_H, (x+w/2)*IMG_W, (y+h/2)*IMG_H]))
    return out


def evaluate(weights, conf, iou_thr):
    m = YOLO(weights)
    tp = {c: 0 for c in range(3)}
    ngt = {c: 0 for c in range(3)}
    ndet = 0
    for img in sorted(IMG_DIR.glob('*.png')):
        gt = load_gt(img.stem)
        for c, _ in gt:
            ngt[c] += 1
        if not gt:
            continue
        r = m(str(img), conf=conf, verbose=False)[0]
        dets = [(int(b.cls[0]), b.xyxy[0].cpu().numpy()) for b in r.boxes]
        ndet += len(dets)
        used = set()
        for c, g in gt:
            best, bj = 0.0, -1
            for j, (dc, d) in enumerate(dets):
                if j in used or dc != c:
                    continue
                v = iou(g, d)
                if v > best:
                    best, bj = v, j
            if best >= iou_thr:
                tp[c] += 1
                used.add(bj)
    return tp, ngt, ndet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', type=float, default=0.01)
    ap.add_argument('--iou', type=float, default=0.3)
    args = ap.parse_args()

    runs = {
        'LAMA v8m_cy399 (calib salah)': ROOT / 'output/yolo_manual/v8m_cy399/weights/best.pt',
        'BARU v8m_calibv3': ROOT / 'output/yolo_v3/v8m_calibv3/weights/best.pt',
    }
    print(f"GT = label v3 (78 frame val) — SAMA untuk kedua model")
    print(f"conf={args.conf}  IoU>={args.iou}  (titik operasi CLOCs)\n")
    for name, w in runs.items():
        if not w.exists():
            print(f"{name}: weights belum ada ({w})")
            continue
        tp, ngt, nd = evaluate(str(w), args.conf, args.iou)
        print(f"{name}")
        tot_tp = tot_gt = 0
        for c, cn in enumerate(CLASSES):
            r = tp[c] / ngt[c] if ngt[c] else float('nan')
            print(f"   {cn:11s} recall {r:6.3f}  ({tp[c]:3d}/{ngt[c]:3d})")
            tot_tp += tp[c]; tot_gt += ngt[c]
        print(f"   {'TOTAL':11s} recall {tot_tp/tot_gt:6.3f}  ({tot_tp}/{tot_gt})   deteksi={nd}\n")


if __name__ == '__main__':
    main()
