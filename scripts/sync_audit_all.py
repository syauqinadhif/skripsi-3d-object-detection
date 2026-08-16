#!/usr/bin/env python3
"""
Audit sinkronisasi kamera-LiDAR untuk SEMUA frame — TANPA perlu label 3D.

Cara: cari gugus titik LiDAR berukuran mobil (buang tanah -> DBSCAN -> filter ukuran),
proyeksikan pusatnya ke gambar, lalu bandingkan dgn deteksi mobil COCO YOLO
(COCO independen dari calib & label). Frame sinkron -> gugus & deteksi berimpit;
frame desync -> tidak berhubungan sama sekali.

Dipakai untuk memutuskan apakah ekstraksi ulang dari rosbag sepadan.
"""
import pickle, sys
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
IMG_W, IMG_H = 1280, 720
FX, FY, CX, CY = 699.47418510, 707.80363724, 630.32152421, 360.0
P2 = np.array([[FX, 0, CX, 0], [0, FY, CY, 0], [0, 0, 1, 0]])
TR = np.load(ROOT / 'dataset_ugm/tr_synced.npy')
TR4 = np.vstack([TR, [0, 0, 0, 1]])
COCO_CAR = {2, 5, 7}


def car_clusters(p):
    """gugus titik berukuran mobil di depan, <=30 m"""
    m = (p[:, 0] > 3) & (p[:, 0] < 30) & (np.abs(p[:, 1]) < 20)
    q = p[m]
    if len(q) < 200:
        return []
    ground = np.percentile(q[:, 2], 8)
    q = q[(q[:, 2] > ground + 0.35) & (q[:, 2] < ground + 2.6)]
    if len(q) < 100:
        return []
    lab = DBSCAN(eps=0.7, min_samples=12).fit_predict(q[:, :2])
    out = []
    for l in set(lab) - {-1}:
        c = q[lab == l]
        ext = c.max(0) - c.min(0)
        if 1.5 <= max(ext[0], ext[1]) <= 6.0 and 0.9 <= min(ext[0], ext[1]) <= 3.0 and ext[2] >= 0.7:
            out.append(c.mean(0))
    return out


def project(pt):
    cam = (TR4 @ np.array([pt[0], pt[1], pt[2], 1.0]))[:3]
    if cam[2] <= 0.5:
        return None
    pr = P2 @ np.array([cam[0], cam[1], cam[2], 1.0])
    return pr[0] / pr[2], pr[1] / pr[2]


def main():
    ids = sorted(x.stem for x in (ROOT / 'dataset_ugm/velodyne').glob('*.bin'))
    m = YOLO('yolov8m.pt')
    rows = []
    for i, fid in enumerate(ids):
        p = np.fromfile(ROOT / f'dataset_ugm/velodyne/{fid}.bin', dtype=np.float32).reshape(-1, 4)[:, :3]
        cl = car_clusters(p)
        img = ROOT / f'dataset_ugm/image_2/{fid}.png'
        if not img.exists() or not cl:
            rows.append((fid, np.nan, len(cl), 0)); continue
        r = m(str(img), conf=0.30, verbose=False)[0]
        dets = [b.xyxy[0].cpu().numpy() for b in r.boxes if int(b.cls[0]) in COCO_CAR]
        if not dets:
            rows.append((fid, np.nan, len(cl), 0)); continue
        dc = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in dets])
        ds = []
        for c in cl:
            uv = project(c)
            if uv is None or not (-100 < uv[0] < IMG_W + 100):
                continue
            ds.append(np.sqrt(((dc[:, 0] - uv[0]) ** 2 + (dc[:, 1] - uv[1]) ** 2).min()))
        rows.append((fid, np.median(ds) if ds else np.nan, len(cl), len(dets)))
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(ids)}", flush=True)
    pickle.dump(rows, open(ROOT / 'output/sync_audit.pkl', 'wb'))
    d = np.array([r[1] for r in rows], dtype=float)
    ok = np.nansum(d < 80); bad = np.nansum(d >= 200); mid = np.nansum((d >= 80) & (d < 200))
    unk = int(np.isnan(d).sum())
    n = len(rows)
    print(f"\nAUDIT {n} frame:")
    print(f"  SINKRON  (<80 px)  : {int(ok):3d} ({ok/n*100:.0f}%)")
    print(f"  ragu     (80-200)  : {int(mid):3d} ({mid/n*100:.0f}%)")
    print(f"  DESYNC   (>=200 px): {int(bad):3d} ({bad/n*100:.0f}%)")
    print(f"  tak terukur        : {unk:3d} ({unk/n*100:.0f}%)  (tak ada gugus mobil / tak ada deteksi)")
    judged = int(ok + mid + bad)
    if judged:
        print(f"\nDari {judged} frame yang TERUKUR: sinkron {ok/judged*100:.0f}%, desync {bad/judged*100:.0f}%")
    sync_ids = [r[0] for r in rows if not np.isnan(r[1]) and r[1] < 80]
    (ROOT / 'dataset_ugm/ImageSets/synced_all.txt').write_text('\n'.join(sync_ids) + '\n')
    print(f"daftar frame sinkron -> dataset_ugm/ImageSets/synced_all.txt ({len(sync_ids)} frame)")


if __name__ == '__main__':
    main()
