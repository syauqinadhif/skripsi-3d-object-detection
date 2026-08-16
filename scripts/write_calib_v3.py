#!/usr/bin/env python3
"""
Tulis calib UGM v3: intrinsik ASLI rosbag + ekstrinsik hasil PnP (tr_pnp_gt_v3).

Kenapa P2 dan Tr_velo_to_cam HARUS diganti bersamaan: tr_pnp_gt_v3 di-fit dengan intrinsik
rosbag dikunci (K di calibrate_pnp_gt.py). Memakai ekstrinsik baru dgn P2 lama (fx=699,
warisan KITTI) = mencampur dua model kamera berbeda → proyeksi salah lagi.

  P2            = [K | 0]  dgn K rosbag (fx=fy=959.155, cx=650.105, cy=362.808)
  R0_rect       = I        (ZED sudah rectified, distorsi = 0 per camera_info)
  Tr_velo_to_cam= tr_pnp_gt_v3.npy (LiDAR -> optical frame kamera kiri, langsung)

P0/P1/P3 ditulis sama dgn P2: dataset UGM monokular (cuma kamera kiri dipakai); field-field
itu ada hanya agar parser KITTI OpenPCDet tidak error.

⚠️ SETELAH menjalankan ini, label_2_* WAJIB di-generate ulang dari JSON Datumaro. Field
`location` di label tersimpan dalam frame KAMERA memakai Tr LAMA — tidak boleh dipatch.
"""
import shutil
import numpy as np
from pathlib import Path

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
CALIB = ROOT / 'dataset_ugm/calib'
BACKUP = ROOT / 'dataset_ugm/calib_pre_v3_backup'
TR_NPY = ROOT / 'dataset_ugm/tr_pnp_gt_v3.npy'

K = np.array([[959.155, 0.0, 650.105],
              [0.0, 959.155, 362.808],
              [0.0, 0.0, 1.0]])


def fmt(m):
    return ' '.join(f'{v:.12e}' for v in np.asarray(m).reshape(-1))


def main():
    Tr = np.load(TR_NPY)
    assert Tr.shape == (3, 4), Tr.shape
    P = np.hstack([K, np.zeros((3, 1))])
    R0 = np.eye(3)

    files = sorted(CALIB.glob('*.txt'))
    assert files, 'calib kosong'

    if BACKUP.exists():
        print(f'backup sudah ada, dilewati: {BACKUP}')
    else:
        shutil.copytree(CALIB, BACKUP)
        print(f'backup {len(files)} file -> {BACKUP}')

    text = (f'P0: {fmt(P)}\n'
            f'P1: {fmt(P)}\n'
            f'P2: {fmt(P)}\n'
            f'P3: {fmt(P)}\n'
            f'R0_rect: {fmt(R0)}\n'
            f'Tr_velo_to_cam: {fmt(Tr)}\n'
            f'Tr_imu_to_velo: {fmt(np.hstack([np.eye(3), np.zeros((3, 1))]))}\n')

    for f in files:
        f.write_text(text)
    print(f'ditulis ulang {len(files)} file calib')
    print(f'  P2 fx=fy={K[0,0]} cx={K[0,2]} cy={K[1,2]}')
    print(f'  Tr translasi = {Tr[:,3].round(3)}  |t|={np.linalg.norm(Tr[:,3]):.3f} m')


if __name__ == '__main__':
    main()
