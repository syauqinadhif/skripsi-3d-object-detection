#!/usr/bin/env python3
"""
Koreksi bias-z label VAL manual: sandarkan alas tiap cuboid ke permukaan tanah lokal.

Kenapa: label val 80 frame melayang (alas box - tanah: Car +0.40, Ped +0.77, Cyc +0.33 m)
karena mewarisi penempatan vertikal pseudolabel pra-anotasi. Lihat memory [[gt_val_bias_z]].
Mobil/pejalan kaki/sepeda SELALU menapak tanah -> alas = tanah adalah kendala fisik, bukan tebakan.

Input : annotations/default.json dari export Datumaro 3D val (task1_job1_labels_frame0-79.zip)
Output: default.json terkoreksi (siap di-IMPORT balik ke task CVAT) + ringkasan geseran.
        HANYA position[2] yang diubah. x/y/dimensi/yaw/atribut tidak disentuh.

Estimasi tanah = rata-rata 2 estimator independen (persentil-10 cincin 2-5 m, dan fit bidang
robust IRLS pada titik rendah radius <=7 m). Ketidaksepakatan keduanya ~0.11-0.14 m di <20 m
(dipakai sebagai indikator keandalan per objek).

Setelah user verifikasi di CVAT, export lagi lalu jalankan convert_datumaro_to_kitti.py.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
VEL = ROOT / 'dataset_ugm/velodyne'
TRUST_RANGE = 25.0   # di luar ini estimator TIDAK andal -> ditandai, jangan dipercaya


def plane_z(pts, x, y):
    if len(pts) < 30:
        return None
    P = pts; w = np.ones(len(P))
    for _ in range(3):
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        try:
            c = np.linalg.lstsq(A * w[:, None], P[:, 2] * w, rcond=None)[0]
        except Exception:
            return None
        r = P[:, 2] - A @ c
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
        w = 1 / (1 + (r / (2 * s)) ** 2)
    return float(c[0] * x + c[1] * y + c[2])


def ground_estimates(p, x, y, sz):
    d = np.hypot(p[:, 0] - x, p[:, 1] - y)
    g1 = g2 = None
    r1 = p[(d > 2) & (d < 5)]
    if len(r1) >= 50:
        g1 = float(np.percentile(r1[:, 2], 10))
    r2 = p[(d > sz / 2 + 0.3) & (d < sz / 2 + 7.0)]
    if len(r2) >= 60:
        lo = r2[r2[:, 2] < np.percentile(r2[:, 2], 30)]
        g2 = plane_z(lo, x, y)
    return g1, g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='annotations/default.json val asli')
    ap.add_argument('--out', required=True, help='folder output (berisi annotations/default.json)')
    args = ap.parse_args()

    data = json.load(open(args.json))
    shifts, far, unreliable, disagree = [], 0, 0, []

    for it in data['items']:
        fid = it['id']
        bin_path = VEL / f'{fid}.bin'
        if not bin_path.exists():
            continue
        p = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        for a in it['annotations']:
            if a.get('type') != 'cuboid_3d':
                continue
            x, y, z = a['position']; dx, dy, dz = a['scale']
            rng = float(np.hypot(x, y))
            g1, g2 = ground_estimates(p, x, y, max(dx, dy))
            gs = [g for g in (g1, g2) if g is not None]
            if not gs:
                unreliable += 1
                continue
            if len(gs) == 2:
                disagree.append(abs(gs[0] - gs[1]))
            ground = float(np.mean(gs))
            new_z = ground + dz / 2
            shifts.append(new_z - z)
            a['position'][2] = round(new_z, 3)
            if rng > TRUST_RANGE:
                far += 1

    out = Path(args.out); (out / 'annotations').mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out / 'annotations/default.json', 'w'))
    shutil.make_archive(str(out), 'zip', str(out))

    s = np.array(shifts)
    print(f"{len(s)} cuboid dikoreksi ({unreliable} dilewati: titik tanah tak cukup)")
    print(f"geseran z: median={np.median(s):+.2f} m  mean={s.mean():+.2f}  std={s.std():.2f}")
    print(f"  |geser| <=0.2 m (sudah benar) : {(np.abs(s)<=0.2).mean()*100:.0f}%")
    print(f"  |geser| >0.5 m (jelas melayang): {(np.abs(s)>0.5).mean()*100:.0f}%")
    if disagree:
        print(f"ketidaksepakatan 2 estimator: median={np.median(disagree):.2f} m (indikator keandalan)")
    print(f"{far} objek di luar {TRUST_RANGE:.0f} m -> estimator TIDAK andal, jangan diverifikasi manual;"
          f" ekskusi saat eval (DontCare).")
    print(f"\noutput -> {out}.zip  (import ke task CVAT val, lalu verifikasi objek <={TRUST_RANGE:.0f} m)")


if __name__ == '__main__':
    main()
