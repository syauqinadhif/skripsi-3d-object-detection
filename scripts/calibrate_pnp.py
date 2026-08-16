#!/usr/bin/env python3
"""
Kalibrasi ekstrinsik LiDAR->kamera dgn solvePnP + RANSAC (asosiasi iteratif ala ICP).

Kenapa bukan optimasi "soft-count" seperti sebelumnya: objektif tetangga-terdekat terjebak
asosiasi salah saat tebakan awal masih meleset ratusan piksel (hasilnya mentok bimodal).
PnP+RANSAC memisahkan dua hal: (1) RANSAC membuang pasangan salah, (2) PnP menyelesaikan pose
secara tertutup dari pasangan yang benar. Diulang: asosiasi -> PnP -> asosiasi lebih baik.

Korespondensi:
  3D = pusat gugus titik berukuran mobil (dari LiDAR, tanpa perlu label)
  2D = pusat deteksi mobil COCO YOLO (independen kalibrasi)
INTRINSIK dikunci ke nilai ASLI dari rosbag camera_info (fx=fy=959.155, cx=650.105, cy=362.808).
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
K = np.array([[959.155, 0, 650.105], [0, 959.155, 362.808], [0, 0, 1]])
DIST = np.zeros(5)
CACHE = ROOT / 'output/pnp_correspondences.pkl'


def car_clusters(p):
    m = (p[:, 0] > 3) & (p[:, 0] < 35) & (np.abs(p[:, 1]) < 22)
    q = p[m]
    if len(q) < 200:
        return []
    g = np.percentile(q[:, 2], 8)
    q = q[(q[:, 2] > g + 0.35) & (q[:, 2] < g + 2.6)]
    if len(q) < 100:
        return []
    lab = DBSCAN(eps=0.7, min_samples=12).fit_predict(q[:, :2])
    out = []
    for l in set(lab) - {-1}:
        c = q[lab == l]
        e = c.max(0) - c.min(0)
        if 1.5 <= max(e[0], e[1]) <= 6.0 and 0.9 <= min(e[0], e[1]) <= 3.0 and e[2] >= 0.7:
            out.append(c.mean(0))
    return out


def build_cache():
    m = YOLO('yolov8m.pt')
    ids = sorted(x.stem for x in (ROOT / 'dataset_ugm/velodyne').glob('*.bin'))
    data = []
    for i, fid in enumerate(ids):
        p = np.fromfile(ROOT / f'dataset_ugm/velodyne/{fid}.bin', dtype=np.float32).reshape(-1, 4)[:, :3]
        cl = car_clusters(p)
        img = ROOT / f'dataset_ugm/image_2/{fid}.png'
        if not cl or not img.exists():
            continue
        r = m(str(img), conf=0.35, verbose=False)[0]
        dets = [b.xyxy[0].cpu().numpy() for b in r.boxes if int(b.cls[0]) in (2, 5, 7)]
        if not dets:
            continue
        dc = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in dets])
        data.append({'fid': fid, 'p3': np.array(cl), 'p2': dc})
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(ids)}", flush=True)
    pickle.dump(data, open(CACHE, 'wb'))
    return data


def project(P3, rvec, tvec):
    uv, _ = cv2.projectPoints(P3.astype(np.float64), rvec, tvec, K, DIST)
    return uv.reshape(-1, 2)


def associate(data, rvec, tvec, thr):
    """pasangkan tiap gugus 3D ke deteksi 2D terdekat (saling terdekat, di bawah ambang)"""
    A3, A2 = [], []
    for d in data:
        uv = project(d['p3'], rvec, tvec)
        for i, q in enumerate(uv):
            j = int(np.argmin(np.linalg.norm(d['p2'] - q, axis=1)))
            if np.linalg.norm(d['p2'][j] - q) < thr:
                # saling-terdekat: gugus i juga harus yg terdekat bagi deteksi j
                k = int(np.argmin(np.linalg.norm(uv - d['p2'][j], axis=1)))
                if k == i:
                    A3.append(d['p3'][i]); A2.append(d['p2'][j])
    return np.array(A3), np.array(A2)


def main():
    data = pickle.load(open(CACHE, 'rb')) if CACHE.exists() else build_cache()
    print(f"{len(data)} frame punya (gugus 3D + deteksi 2D); "
          f"total gugus={sum(len(d['p3']) for d in data)}, deteksi={sum(len(d['p2']) for d in data)}")

    # tebakan awal: sumbu LiDAR (x maju, y kiri, z atas) -> kamera optik (x kanan, y bawah, z maju)
    R0 = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    rvec = cv2.Rodrigues(R0)[0]; tvec = np.zeros((3, 1))

    best = None
    for it, thr in enumerate([400, 250, 150, 90, 60, 45]):
        A3, A2 = associate(data, rvec, tvec, thr)
        if len(A3) < 12:
            print(f"  iterasi {it+1} (thr {thr}px): pasangan cuma {len(A3)} — berhenti")
            break
        ok, rv, tv, inl = cv2.solvePnPRansac(A3.astype(np.float64), A2.astype(np.float64), K, DIST,
                                             rvec=rvec.copy(), tvec=tvec.copy(), useExtrinsicGuess=True,
                                             reprojectionError=thr * 0.5, iterationsCount=8000,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < 10:
            print(f"  iterasi {it+1} (thr {thr}px): PnP gagal"); break
        rvec, tvec = rv, tv
        err = np.linalg.norm(project(A3[inl.ravel()], rvec, tvec) - A2[inl.ravel()], axis=1)
        print(f"  iterasi {it+1} (thr {thr:3d}px): pasangan={len(A3):4d} inlier={len(inl):4d} "
              f"galat median={np.median(err):5.1f} px")
        best = (rvec.copy(), tvec.copy(), len(inl), float(np.median(err)))

    if best is None:
        print("GAGAL — tidak ada solusi"); return
    rvec, tvec, ninl, med = best
    R = cv2.Rodrigues(rvec)[0]
    Tr = np.hstack([R, tvec.reshape(3, 1)])
    np.save(ROOT / 'dataset_ugm/tr_pnp.npy', Tr)
    print(f"\nHASIL: inlier={ninl}, galat reproyeksi median={med:.1f} px")
    print(f"  translasi (m) = {tvec.ravel().round(3)}")
    print(f"  rotasi (deg)  = {np.degrees(cv2.RQDecomp3x3(R)[0]).round(2) if False else np.degrees(rvec.ravel()).round(2)}")
    print(f"  -> dataset_ugm/tr_pnp.npy")


if __name__ == '__main__':
    main()
