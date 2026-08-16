#!/usr/bin/env python3
"""
Kalibrasi TARGETLESS ekstrinsik LiDAR->kamera (Tr_velo_to_cam) untuk rig UGM.

Kenapa: convert_ugm_to_kitti.py menulis ekstrinsik KITTI sebagai PLACEHOLDER dan memperingatkan
"WAJIB DIGANTI dengan nilai real UGM sebelum training CLOCs fusion" — tak pernah dijalankan.
Semua proyeksi 3D->2D (label YOLO, cabang 2D CLOCs, bbox infos) karena itu salah tempat.

Metode: maksimalkan kecocokan antara pusat box 3D GT yang diproyeksikan dan pusat deteksi
COCO YOLO (independen calib). Objektif "soft count": sum exp(-d^2/2sigma^2) — mulus, kebal
outlier (objek tak terdeteksi menyumbang ~0), dan tidak punya solusi degenerate (mendorong
proyeksi keluar gambar memberi skor 0).

Intrinsik DIKUNCI (fx,fy,cx dari kamera; cy=360=pusat gambar) supaya kemiringan rig diserap
oleh rotasi ekstrinsik, bukan oleh cy. cy & pitch nyaris degenerate kalau dioptimasi bersama.

Optimasi di frame TRAIN saja; frame VAL dipakai sebagai uji tahan-banting (held-out).
"""
import pickle, sys
import numpy as np
from scipy.optimize import differential_evolution, minimize

SP = sys.argv[1] if len(sys.argv) > 1 else '.'  # direktori berisi corr.pkl
D = pickle.load(open(f'{SP}/corr.pkl', 'rb'))
IMG_W, IMG_H = 1280, 720
SIGMA = 60.0
# INTRINSIK ASLI dari rosbag /zed/zed_node/rgb/camera_info (bukan lagi tebakan/KITTI).
# Nilai lama (fx=699.47, fy=707.80) meleset 37% -> objek makin ke pinggir makin salah tempat,
# yang dulu SALAH saya kira desync kamera-LiDAR.
FX, FY, CX, CY = 959.155, 959.155, 650.105, 362.808
P2 = np.array([[FX, 0, CX, 0], [0, FY, CY, 0], [0, 0, 1, 0]])


def base_tr():
    for l in open('dataset_ugm/calib_cy440_backup/000001.txt'):
        if l.startswith('Tr_velo_to_cam'):
            return np.array([float(x) for x in l.split(':', 1)[1].split()]).reshape(3, 4)


TR0 = base_tr()


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def make_tr(p):
    R = rodrigues(p[:3]) @ TR0[:, :3]
    t = TR0[:, 3] + p[3:6]
    return np.hstack([R, t.reshape(3, 1)])


def corners(b):
    x, y, z, dx, dy, dz, yaw = b[:7]
    c = np.array([[dx/2, dx/2, -dx/2, -dx/2] * 2, [dy/2, -dy/2, -dy/2, dy/2] * 2, [-dz/2]*4 + [dz/2]*4])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R @ c).T + np.array([x, y, z])


CACHE = {fid: [corners(b) for b in v['boxes3d']] for fid, v in D.items()}


def score(p, split):
    Tr = make_tr(p); Tr4 = np.vstack([Tr, [0, 0, 0, 1]])
    tot = 0.0
    for fid, v in D.items():
        if v['split'] != split:
            continue
        dets = v['dets2d']
        dc = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)
        for cor in CACHE[fid]:
            cam = (Tr4 @ np.hstack([cor, np.ones((8, 1))]).T).T[:, :3]
            if (cam[:, 2] <= 0.5).any():
                continue
            pr = (P2 @ np.hstack([cam, np.ones((8, 1))]).T).T
            uv = pr[:, :2] / pr[:, 2:3]
            u = (uv[:, 0].min() + uv[:, 0].max()) / 2; v_ = (uv[:, 1].min() + uv[:, 1].max()) / 2
            if not (-200 < u < IMG_W + 200 and -200 < v_ < IMG_H + 200):
                continue
            d2 = ((dc[:, 0] - u) ** 2 + (dc[:, 1] - v_) ** 2).min()
            tot += np.exp(-d2 / (2 * SIGMA ** 2))
    return tot


def neg(p):
    return -score(p, 'train')


if __name__ == '__main__':
    print(f"skor awal (ekstrinsik KITTI): train={score(np.zeros(6),'train'):.1f}  val={score(np.zeros(6),'val'):.1f}")
    # Batas FISIK: kamera & LiDAR terpasang pada rig yang sama (jarak < ~1 m).
    # Tanpa batas ini optimizer menemukan solusi degenerate (pernah keluar translasi +30 m)
    # yang skornya bagus tapi mustahil secara fisik.
    bounds = [(-0.52, 0.52)] * 3 + [(-1.0, 1.0)] * 3
    res = differential_evolution(neg, bounds, maxiter=60, popsize=18, tol=1e-6, seed=0,
                                 polish=False, disp=False)
    res = minimize(neg, res.x, method='Powell', bounds=bounds)   # polish WAJIB terbatas
    p = res.x
    print(f"skor akhir                  : train={score(p,'train'):.1f}  val={score(p,'val'):.1f}  (held-out)")
    print(f"koreksi rotasi  = {np.degrees(p[:3]).round(2)} derajat")
    print(f"koreksi translasi = {p[3:6].round(3)} m")
    Tr = make_tr(p)
    np.save(f'{SP}/tr_opt.npy', Tr)
    # diagnostik: jarak pusat proyeksi ke deteksi terdekat, sebelum vs sesudah
    def dists(pp, split):
        Tr = make_tr(pp); Tr4 = np.vstack([Tr, [0, 0, 0, 1]]); out = []
        for fid, v in D.items():
            if v['split'] != split: continue
            dets = v['dets2d']
            dc = np.stack([(dets[:,0]+dets[:,2])/2, (dets[:,1]+dets[:,3])/2], 1)
            for cor in CACHE[fid]:
                cam = (Tr4 @ np.hstack([cor, np.ones((8,1))]).T).T[:, :3]
                if (cam[:,2] <= 0.5).any(): continue
                pr = (P2 @ np.hstack([cam, np.ones((8,1))]).T).T
                uv = pr[:,:2]/pr[:,2:3]
                u=(uv[:,0].min()+uv[:,0].max())/2; v_=(uv[:,1].min()+uv[:,1].max())/2
                out.append(np.sqrt(((dc[:,0]-u)**2+(dc[:,1]-v_)**2).min()))
        return np.array(out)
    for sp in ['train','val']:
        a=dists(np.zeros(6),sp); b=dists(p,sp)
        print(f"  {sp}: jarak pusat median {np.median(a):.0f} -> {np.median(b):.0f} px | "
              f"dalam 60 px: {(a<60).mean()*100:.0f}% -> {(b<60).mean()*100:.0f}%")
    print("Tr_velo_to_cam baru:")
    print('  ' + ' '.join(f'{x:.10e}' for x in Tr.ravel()))
