#!/usr/bin/env python3
"""
Kalibrasi ekstrinsik LiDAR->kamera dgn solvePnP+RANSAC (asosiasi iteratif ala ICP).

Versi GT (bukan DBSCAN, lihat calibrate_pnp.py): titik 3D = pusat box GT Car MANUAL,
gabungan train_manual (150 frame) + val_manual_v3 (78 frame, GT v3 sudah dibersihkan dari
FP pseudolabel di frame padat). Titik 2D = deteksi
mobil COCO YOLO (independen kalibrasi). INTRINSIK dikunci ke nilai ASLI rosbag camera_info
(fx=fy=959.155, cx=650.105, cy=362.808).

GT box camera-frame -> LiDAR-frame lewat invers Tr_velo_to_cam (matriks sama yg dipakai
convert_datumaro_to_kitti.py utk konversi LiDAR->kamera saat label dibuat -- round trip exact,
lihat 'round-trip z err' di log converter).
"""
import pickle
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
K = np.array([[959.155, 0, 650.105], [0, 959.155, 362.808], [0, 0, 1]])
DIST = np.zeros(5)
CACHE = ROOT / 'output/pnp_gt_v3_correspondences.pkl'

CALIB_DIR = ROOT / 'dataset_ugm/calib'
LABEL_DIRS = [ROOT / 'dataset_ugm/label_2_train_manual', ROOT / 'dataset_ugm/label_2_val_manual_v3']
IMG_DIR = ROOT / 'dataset_ugm/image_2'


def load_tr(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        d[k.strip()] = np.array([float(x) for x in v.split()])
    return d['Tr_velo_to_cam'].reshape(3, 4)


def cam_to_lidar_center(loc, dims_h, Tr):
    Tr4 = np.vstack([Tr, [0., 0., 0., 1.]])
    inv = np.linalg.inv(Tr4)
    p = np.hstack([loc, np.ones((len(loc), 1))])
    lidar_bottom = (inv @ p.T).T[:, :3]
    lidar_bottom[:, 2] += dims_h / 2
    return lidar_bottom


def gt_car_centers():
    Tr = load_tr(sorted(CALIB_DIR.glob('*.txt'))[0])
    out = {}
    for ld in LABEL_DIRS:
        for f in sorted(ld.glob('*.txt')):
            if f.stat().st_size == 0:
                continue
            locs, hs = [], []
            for line in f.read_text().splitlines():
                p = line.split()
                if p[0] != 'Car':
                    continue
                h, w, l = float(p[8]), float(p[9]), float(p[10])
                x, y, z = float(p[11]), float(p[12]), float(p[13])
                locs.append([x, y, z]); hs.append(h)
            if not locs:
                continue
            centers = cam_to_lidar_center(np.array(locs), np.array(hs), Tr)
            out.setdefault(f.stem, []).extend(centers.tolist())
    return out


def build_cache():
    m = YOLO('yolov8m.pt')
    gt = gt_car_centers()
    print(f"{len(gt)} frame punya GT Car; total box={sum(len(v) for v in gt.values())}")
    data = []
    for fid, p3 in sorted(gt.items()):
        img = IMG_DIR / f'{fid}.png'
        if not img.exists():
            continue
        r = m(str(img), conf=0.35, verbose=False)[0]
        dets = [b.xyxy[0].cpu().numpy() for b in r.boxes if int(b.cls[0]) in (2, 5, 7)]
        if not dets:
            continue
        dc = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in dets])
        data.append({'fid': fid, 'p3': np.array(p3), 'p2': dc})
    pickle.dump(data, open(CACHE, 'wb'))
    return data


def project(P3, rvec, tvec):
    uv, _ = cv2.projectPoints(P3.astype(np.float64), rvec, tvec, K, DIST)
    return uv.reshape(-1, 2)


def associate(data, rvec, tvec, thr):
    A3, A2 = [], []
    for d in data:
        uv = project(d['p3'], rvec, tvec)
        for i, q in enumerate(uv):
            j = int(np.argmin(np.linalg.norm(d['p2'] - q, axis=1)))
            if np.linalg.norm(d['p2'][j] - q) < thr:
                k = int(np.argmin(np.linalg.norm(uv - d['p2'][j], axis=1)))
                if k == i:
                    A3.append(d['p3'][i]); A2.append(d['p2'][j])
    return np.array(A3), np.array(A2)


def detection_side_metric(data, rvec, tvec, thr=40.0):
    """% box GT Car yg punya deteksi 2D dlm radius thr px (metrik dari TODO)."""
    n_ok = n_tot = 0
    frame_zero = 0
    for d in data:
        uv = project(d['p3'], rvec, tvec)
        ok = 0
        for q in uv:
            if len(d['p2']) and np.min(np.linalg.norm(d['p2'] - q, axis=1)) < thr:
                ok += 1
        n_ok += ok; n_tot += len(uv)
        if ok == 0:
            frame_zero += 1
    return n_ok / max(n_tot, 1), frame_zero / len(data)


def main():
    data = pickle.load(open(CACHE, 'rb')) if CACHE.exists() else build_cache()
    print(f"{len(data)} frame punya (GT Car 3D + deteksi 2D); "
          f"total GT={sum(len(d['p3']) for d in data)}, deteksi={sum(len(d['p2']) for d in data)}")

    R0 = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    rvec = cv2.Rodrigues(R0)[0]; tvec = np.zeros((3, 1))

    m0, z0 = detection_side_metric(data, rvec, tvec)
    print(f"tebakan awal (KITTI-like): match<40px={m0:.1%}, frame 0-match={z0:.1%}")

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
    np.save(ROOT / 'dataset_ugm/tr_pnp_gt_v3.npy', Tr)

    mf, zf = detection_side_metric(data, rvec, tvec)
    print(f"\nHASIL: inlier={ninl}, galat reproyeksi median={med:.1f} px")
    print(f"  translasi (m) = {tvec.ravel().round(3)}  |t|={np.linalg.norm(tvec):.2f} m")
    print(f"  rotasi (deg, Rodrigues) = {np.degrees(rvec.ravel()).round(2)}")
    print(f"  metrik sisi-deteksi: match<40px {m0:.1%} -> {mf:.1%} | frame 0-match {z0:.1%} -> {zf:.1%}")
    print(f"  -> dataset_ugm/tr_pnp_gt_v3.npy")


if __name__ == '__main__':
    main()
