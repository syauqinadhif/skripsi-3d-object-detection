#!/usr/bin/env python3
"""
Peringkat frame berdasarkan KEKAYAAN kelas yang langka, memakai COCO YOLO sbg penaksir.

Kenapa: labeling ACAK terbukti tak menggerakkan recall Car (+50 frame -> 0.333 ke 0.338),
sementara Pedestrian cuma punya ~115 instance latih (recall 0.02). Labeling BERTARGET pada
frame yang padat Pedestrian/Car jauh lebih efisien per jam kerja.

Skor default menimbang Pedestrian jauh lebih tinggi karena itu kelas paling kekurangan.
"""
import argparse
from pathlib import Path
ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
COCO = {0: 'Pedestrian', 2: 'Car', 7: 'Car', 5: 'Car', 3: 'Cyclist', 1: 'Cyclist'}

ap = argparse.ArgumentParser()
ap.add_argument('--frames', required=True)
ap.add_argument('--image-dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--top', type=int, default=100)
ap.add_argument('--w-ped', type=float, default=3.0, help='bobot Pedestrian (kelas paling langka)')
ap.add_argument('--w-car', type=float, default=1.0)
ap.add_argument('--conf', type=float, default=0.25, help='ambang agak tinggi = hitungan lebih dipercaya')
ap.add_argument('--min-height', type=float, default=25, help='abaikan objek lebih kecil dari ini (px)')
a = ap.parse_args()

from ultralytics import YOLO
m = YOLO('yolov8m.pt')
ids = (ROOT / a.frames).read_text().split() if (ROOT / a.frames).exists() else Path(a.frames).read_text().split()
img_dir = ROOT / a.image_dir
rows = []
for fid in ids:
    p = img_dir / f'{fid}.png'
    if not p.exists():
        continue
    r = m(str(p), conf=a.conf, verbose=False)[0]
    n = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    dets = []
    for b in r.boxes:
        c = COCO.get(int(b.cls[0]))
        if not c:
            continue
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        if (y2 - y1) >= a.min_height:
            dets.append((c, (x1, y1, x2, y2)))
    # Pengendara motor: COCO memberi 'person' terpisah, tapi di konvensi KITTI itu bagian dari
    # Cyclist -> user akan menghapusnya. Kalau ikut dihitung, peringkat Pedestrian jadi PALSU
    # (terukur: korelasi person-vs-Pedestrian asli cuma +0.56, person-vs-Cyclist +0.38).
    cyc = [b for c, b in dets if c == 'Cyclist']
    for c, b in dets:
        if c == 'Pedestrian' and cyc:
            pa = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
            ins = max((max(0, min(b[2], cb[2]) - max(b[0], cb[0])) *
                       max(0, min(b[3], cb[3]) - max(b[1], cb[1]))) / pa for cb in cyc)
            if ins >= 0.55:
                continue          # pengendara -> bukan Pedestrian
        n[c] += 1
    rows.append((a.w_ped * n['Pedestrian'] + a.w_car * n['Car'], n, fid))
rows.sort(key=lambda r: (-r[0], r[2]))
top = rows[:a.top]
Path(a.out).write_text('\n'.join(f for _, _, f in sorted(top, key=lambda r: r[2])) + '\n')
tp = sum(n['Pedestrian'] for _, n, _ in top); tc = sum(n['Car'] for _, n, _ in top)
ap_ = sum(n['Pedestrian'] for _, n, _ in rows); ac = sum(n['Car'] for _, n, _ in rows)
print(f"dinilai {len(rows)} frame -> dipilih {len(top)}")
print(f"  Pedestrian terdeteksi: {tp} dari {ap_} total ({tp/max(ap_,1)*100:.0f}% terkumpul di {len(top)/max(len(rows),1)*100:.0f}% frame)")
print(f"  Car terdeteksi       : {tc} dari {ac} total ({tc/max(ac,1)*100:.0f}%)")
print(f"  rata2 per frame terpilih: Ped {tp/max(len(top),1):.1f}, Car {tc/max(len(top),1):.1f}")
print(f"-> {a.out}")
