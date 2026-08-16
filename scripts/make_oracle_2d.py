#!/usr/bin/env python3
"""
Buat deteksi 2D "ORACLE" = proyeksi GT 3D (calib cy=399) sebagai pengganti keluaran YOLO.

TUJUAN: mengukur PLAFON fusi CLOCs. Kalau dengan cabang 2D SEMPURNA pun CLOCs tidak
mengalahkan PointPillars-only, maka menambah label 2D tidak akan menolong dan klaim
"fusi > sensor tunggal" tidak bisa didukung di dataset ini.

⚠️ INI EKSPERIMEN DIAGNOSTIK / UPPER BOUND — ANGKANYA TIDAK BOLEH DILAPORKAN SEBAGAI
HASIL SISTEM. Oracle memakai GT di frame val, jadi bocor secara definisi. Fungsinya hanya
menjawab "apakah mekanisme fusinya sanggup menang kalau diberi 2D yang bagus?".

Opsi --recall & --jitter untuk mensimulasikan detektor 2D REALISTIS tapi bagus
(mis. --recall 0.8 --jitter 5 = 80% objek terdeteksi, box meleset ~5 px).
"""
import argparse
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
IMG_W, IMG_H = 1280, 720
CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def load_calib():
    d = {}
    for line in (ROOT / 'dataset_ugm/calib/000001.txt').read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    return d['P2'].reshape(3, 4)


P2 = load_calib()


def project(x, y, z, h, w, l, ry):
    xs = [l/2, l/2, -l/2, -l/2] * 2
    ys = [0, 0, 0, 0, -h, -h, -h, -h]
    zs = [w/2, -w/2, -w/2, w/2] * 2
    R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    cor = (R @ np.array([xs, ys, zs])).T + np.array([x, y, z])
    if (cor[:, 2] <= 0.1).any():
        return None
    pr = (P2 @ np.hstack([cor, np.ones((8, 1))]).T).T
    uv = pr[:, :2] / pr[:, 2:3]
    x1, y1 = uv.min(0); x2, y2 = uv.max(0)
    if x2 < 0 or x1 > IMG_W or y2 < 0 or y1 > IMG_H:
        return None
    b = (np.clip(x1, 0, IMG_W), np.clip(y1, 0, IMG_H), np.clip(x2, 0, IMG_W), np.clip(y2, 0, IMG_H))
    return b if (b[2] - b[0]) >= 1 and (b[3] - b[1]) >= 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--recall', type=float, default=1.0, help='fraksi objek yang "terdeteksi"')
    ap.add_argument('--jitter', type=float, default=0.0, help='std derau posisi box (px)')
    ap.add_argument('--shrink', type=float, default=0.0,
                    help='susutkan box (px) — meniru box detektor yang lebih KETAT dari proyeksi kuboid')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    srcs = [('dataset_ugm/label_2_train_manual', 'dataset_ugm/ImageSets/train_manual150.txt'),
            ('dataset_ugm/label_2_val_manual_v2', 'dataset_ugm/ImageSets/val_manual_v2_78.txt')]
    n_tot = n_det = 0
    for ld, sf in srcs:
        for fid in (ROOT / sf).read_text().split():
            lines = []
            for l in (ROOT / ld / f'{fid}.txt').read_text().splitlines():
                t = l.split()
                if len(t) < 15 or t[0] not in CLASSES:
                    continue
                n_tot += 1
                if rng.random() > args.recall:
                    continue
                h, w, ll = map(float, t[8:11])
                xc, yc, zc = map(float, t[11:14])
                b = project(xc, yc, zc, h, w, ll, float(t[14]))
                if b is None:
                    continue
                if args.shrink:
                    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                    w = max(b[2]-b[0]-args.shrink, 4); h = max(b[3]-b[1]-args.shrink*0.3, 4)
                    b = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
                if args.jitter:
                    j = rng.normal(0, args.jitter, 2)
                    b = (b[0]+j[0], b[1]+j[1]*0.6, b[2]+j[0], b[3]+j[1]*0.6)
                score = float(np.clip(rng.normal(0.85, 0.1), 0.05, 0.99))
                lines.append(f"{t[0]} -1 -1 -10 {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} "
                             f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}")
                n_det += 1
            (out / f'{fid}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print(f"oracle 2D: {n_det}/{n_tot} objek jadi deteksi (recall={args.recall}, jitter={args.jitter}px) -> {out}")


if __name__ == '__main__':
    main()
