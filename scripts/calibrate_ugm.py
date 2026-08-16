"""
calibrate_ugm.py — Estimasi calibration matrix UGM via optimization

Karena tidak ada sesi kalibrasi formal (checkerboard), kita pakai:
  - GT label 3D (x_cam, y_cam, z_cam) sebagai "3D truth"
  - YOLOv8 2D detections pada UGM images sebagai "2D truth"
  - scipy.optimize untuk cari nilai terbaik

Stage 1 — P2 intrinsics (fx, fy, cx, cy):
  Minimizes: jarak antara projected GT 3D center vs YOLOv8 2D center

Stage 2 — Tr_velo_to_cam (tx, ty, tz, rx, ry, rz):
  Minimizes: jarak antara projected approx-LiDAR positions vs YOLOv8 2D center
  Approx LiDAR: dihitung dari GT camera-frame via inverse axis transform

R0_rect: di-set Identity (tidak perlu dioptimize)

Output:
  dataset_ugm/calib_opt/*.txt  — calib files baru untuk semua UGM frames

Usage:
  conda run -n openpcdet python scripts/calibrate_ugm.py
  conda run -n openpcdet python scripts/calibrate_ugm.py --stage 1  # P2 only
  conda run -n openpcdet python scripts/calibrate_ugm.py --verbose   # detail per iterasi
"""

import argparse
import json
import numpy as np
import pickle
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
LABEL_DIR     = PROJECT_ROOT / 'dataset_ugm' / 'label_2'
IMAGE_DIR     = PROJECT_ROOT / 'dataset_ugm' / 'image_2'
CALIB_IN_DIR  = PROJECT_ROOT / 'dataset_ugm' / 'calib'
TRAIN_SPLIT   = PROJECT_ROOT / 'dataset_ugm' / 'ImageSets' / 'train.txt'
YOLO_CACHE    = PROJECT_ROOT / 'output' / 'calib_yolo_cache.json'
YOLO_MODEL    = 'yolov8m.pt'   # COCO pretrained, lebih reliable untuk deteksi umum

IMG_W, IMG_H  = 1280, 720

# KITTI values sebagai starting point
KITTI_P2_FX   = 707.0493
KITTI_P2_CY   = 180.5066

KITTI_TR = np.array([
    [ 6.927964e-03, -9.999722e-01, -2.757829e-03, -2.457729e-02],
    [-1.162982e-03,  2.749836e-03, -9.999955e-01, -6.127237e-02],
    [ 9.999753e-01,  6.931141e-03, -1.143899e-03, -3.321029e-01],
])

KITTI_R0 = np.array([
    [9.999128e-01,  1.009263e-02, -8.511932e-03],
    [-1.012729e-02, 9.999406e-01, -4.037671e-03],
    [ 8.470675e-03,  4.123522e-03,  9.999556e-01],
])

# Class mapping: KITTI label names → YOLOv8 class IDs (COCO)
KITTI_TO_COCO = {
    'Car':        [2, 5, 7],   # car, bus, truck
    'Pedestrian': [0],          # person
    'Cyclist':    [1, 3],       # bicycle, motorcycle
    'Motorcycle': [3, 1],
}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_gt_labels(frame_id):
    """
    Load GT labels → list of dict {class, x_cam, y_cam, z_cam}
    x,y,z dalam camera frame (KITTI format).
    """
    path = LABEL_DIR / f'{frame_id}.txt'
    if not path.exists():
        return []
    objects = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 15 or parts[0] in ('DontCare', ''):
            continue
        cls   = parts[0]
        x_cam = float(parts[11])
        y_cam = float(parts[12])
        z_cam = float(parts[13])
        if z_cam < 1.0:           # objek di belakang kamera / terlalu dekat
            continue
        objects.append({'class': cls, 'x': x_cam, 'y': y_cam, 'z': z_cam})
    return objects


def load_yolo_detections_cached(frame_ids):
    """
    Run YOLOv8 (COCO pretrained) pada semua UGM train frames.
    Cache hasilnya ke JSON agar tidak perlu re-run tiap optimize.
    Returns: dict {frame_id: [(cls_id, cx, cy, score), ...]}
    """
    if YOLO_CACHE.exists():
        print(f'  Loading cached YOLOv8 detections from {YOLO_CACHE.name}')
        return json.load(open(YOLO_CACHE))

    print(f'  Running YOLOv8 on {len(frame_ids)} UGM frames (akan di-cache)...')
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)

    cache = {}
    for i, fid in enumerate(frame_ids):
        if i % 50 == 0:
            print(f'    {i}/{len(frame_ids)}')
        img_path = IMAGE_DIR / f'{fid}.png'
        if not img_path.exists():
            cache[fid] = []
            continue
        results = model(str(img_path), conf=0.05, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            score  = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dets.append([cls_id, cx, cy, score])
        cache[fid] = dets

    YOLO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(YOLO_CACHE, 'w'))
    print(f'  Cached to {YOLO_CACHE}')
    return cache


# ── Projection utilities ───────────────────────────────────────────────────────

def make_P2(fx, fy, cx, cy):
    return np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
    ])


def make_Tr(tx, ty, tz, rx, ry, rz):
    """Build 3×4 Tr_velo_to_cam dari 6 DOF (euler XYZ)."""
    R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    return np.hstack([R, [[tx], [ty], [tz]]])


def project_cam_center(x, y, z, fx, fy, cx, cy):
    """Project camera-frame 3D point ke pixel (u, v)."""
    return fx * x / z + cx, fy * y / z + cy


def project_lidar_center(xl, yl, zl, Tr, P2):
    """Project LiDAR-frame 3D point ke pixel via Tr + P2 (R0=I)."""
    pt    = Tr @ np.array([xl, yl, zl, 1.0])   # → camera frame (3,)
    if pt[2] <= 0:
        return None, None
    u = P2[0, 0] * pt[0] / pt[2] + P2[0, 2]
    v = P2[1, 1] * pt[1] / pt[2] + P2[1, 2]
    return u, v


def nearest_yolo_dist(u_proj, v_proj, yolo_dets, allowed_cls_ids=None, dist_thresh=250):
    """
    Cari YOLOv8 detection terdekat dari (u_proj, v_proj).
    Returns jarak minimum, atau None jika tidak ada match dalam threshold.
    """
    best = dist_thresh
    for cls_id, cx, cy, score in yolo_dets:
        if allowed_cls_ids and cls_id not in allowed_cls_ids:
            continue
        d = np.sqrt((cx - u_proj) ** 2 + (cy - v_proj) ** 2)
        if d < best:
            best = d
    return best if best < dist_thresh else None


# ── Stage 1: Optimize P2 intrinsics ──────────────────────────────────────────

def build_stage1_data(frame_ids, yolo_cache):
    """
    Pre-compute semua (x_cam, y_cam, z_cam, yolo_dets) per frame.
    Disimpan sebagai list agar cost function tinggal loop.
    """
    data = []
    for fid in frame_ids:
        gts  = load_gt_labels(fid)
        yolo = yolo_cache.get(fid, [])
        if not gts or not yolo:
            continue
        for obj in gts:
            data.append({
                'x': obj['x'], 'y': obj['y'], 'z': obj['z'],
                'cls': obj['class'],
                'yolo': yolo,
            })
    return data


iter_count = [0]

def cost_stage1(params, data, dist_thresh=200, verbose=False):
    fx, fy, cx, cy = params
    total, n = 0.0, 0

    for d in data:
        x, y, z = d['x'], d['y'], d['z']
        u_proj, v_proj = project_cam_center(x, y, z, fx, fy, cx, cy)

        # Skip jika proyeksi jauh di luar gambar
        if not (-200 <= u_proj <= IMG_W + 200 and -200 <= v_proj <= IMG_H + 200):
            continue

        allowed = KITTI_TO_COCO.get(d['cls'])
        dist = nearest_yolo_dist(u_proj, v_proj, d['yolo'],
                                  allowed_cls_ids=allowed,
                                  dist_thresh=dist_thresh)
        if dist is not None:
            total += dist
            n += 1

    cost = total / n if n > 0 else 1e6
    # Penalize if match rate too low (< 30% of data)
    if n < len(data) * 0.1:
        cost += 500.0

    iter_count[0] += 1
    if verbose and iter_count[0] % 20 == 0:
        print(f'    iter {iter_count[0]:4d}: cost={cost:.2f}px  matched={n}/{len(data)}'
              f'  fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}')
    return cost


# ── Stage 2: Optimize Tr_velo_to_cam ─────────────────────────────────────────

def build_stage2_data(frame_ids, yolo_cache):
    """
    Build LiDAR-frame positions dari GT labels menggunakan inverse axis transform.

    Transformasi dasar LiDAR→Camera (berlaku untuk hampir semua automotive setup):
      x_cam ≈ -y_lidar   (lateral)
      y_cam ≈ -z_lidar   (vertical)
      z_cam ≈  x_lidar   (forward/depth)

    Sehingga: x_lidar ≈ z_cam, y_lidar ≈ -x_cam, z_lidar ≈ -y_cam
    """
    data = []
    for fid in frame_ids:
        gts  = load_gt_labels(fid)
        yolo = yolo_cache.get(fid, [])
        if not gts or not yolo:
            continue
        for obj in gts:
            x_cam, y_cam, z_cam = obj['x'], obj['y'], obj['z']
            # Approximate LiDAR position via inverse axis transform
            x_l =  z_cam    # forward
            y_l = -x_cam    # left
            z_l = -y_cam    # up
            data.append({
                'xl': x_l, 'yl': y_l, 'zl': z_l,
                'cls': obj['class'],
                'yolo': yolo,
            })
    return data


def cost_stage2(params, data, P2, dist_thresh=200, verbose=False):
    tx, ty, tz, rx, ry, rz = params
    Tr = make_Tr(tx, ty, tz, rx, ry, rz)
    total, n = 0.0, 0

    for d in data:
        u_proj, v_proj = project_lidar_center(d['xl'], d['yl'], d['zl'], Tr, P2)
        if u_proj is None:
            continue
        if not (-200 <= u_proj <= IMG_W + 200 and -200 <= v_proj <= IMG_H + 200):
            continue

        allowed = KITTI_TO_COCO.get(d['cls'])
        dist = nearest_yolo_dist(u_proj, v_proj, d['yolo'],
                                  allowed_cls_ids=allowed,
                                  dist_thresh=dist_thresh)
        if dist is not None:
            total += dist
            n += 1

    cost = total / n if n > 0 else 1e6
    if n < len(data) * 0.1:
        cost += 500.0

    iter_count[0] += 1
    if verbose and iter_count[0] % 20 == 0:
        print(f'    iter {iter_count[0]:4d}: cost={cost:.2f}px  matched={n}/{len(data)}'
              f'  tx={tx:.3f} ty={ty:.3f} tz={tz:.3f}')
    return cost


# ── Output: write calib files ─────────────────────────────────────────────────

def write_calib_files(P2_opt, Tr_opt, out_dir):
    """
    Tulis ulang semua calib file UGM dengan nilai P2 dan Tr yang dioptimalkan.
    R0_rect = Identity.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    R0_eye = np.eye(3)
    r0_str = ' '.join(f'{v:.10e}' for v in R0_eye.flatten())
    p2_str = ' '.join(f'{v:.10e}' for v in P2_opt.flatten())
    tr_str = ' '.join(f'{v:.10e}' for v in Tr_opt.flatten())

    src_files = sorted(CALIB_IN_DIR.glob('*.txt'))
    for src in src_files:
        with open(out_dir / src.name, 'w') as f:
            f.write(f'P0: {p2_str}\n')
            f.write(f'P1: {p2_str}\n')
            f.write(f'P2: {p2_str}\n')
            f.write(f'P3: {p2_str}\n')
            f.write(f'R0_rect: {r0_str}\n')
            f.write(f'Tr_velo_to_cam: {tr_str}\n')

    print(f'Written {len(src_files)} calib files → {out_dir}')


# ── Summary helper ────────────────────────────────────────────────────────────

def print_P2_summary(label, fx, fy, cx, cy):
    hfov = 2 * np.degrees(np.arctan(IMG_W / (2 * fx)))
    vfov = 2 * np.degrees(np.arctan(IMG_H / (2 * fy)))
    print(f'  [{label}]  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}'
          f'  → HFoV={hfov:.1f}°  VFoV={vfov:.1f}°')


def print_Tr_summary(label, Tr):
    R = Tr[:, :3]
    t = Tr[:, 3]
    euler = np.degrees(Rotation.from_matrix(R).as_euler('xyz'))
    print(f'  [{label}]  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m'
          f'  r=[{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]')


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, default=0,
                   help='0=both, 1=P2 only, 2=Tr only (skip Stage 1, use init P2)')
    p.add_argument('--out_calib', default='dataset_ugm/calib_opt',
                   help='Output dir untuk calib files teroptimasi')
    p.add_argument('--max_frames', type=int, default=0,
                   help='Batasi jumlah frame (0=semua). Untuk debug.')
    p.add_argument('--verbose', action='store_true',
                   help='Print cost tiap 20 iterasi')
    p.add_argument('--recache', action='store_true',
                   help='Hapus cache YOLOv8 dan re-generate')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.recache and YOLO_CACHE.exists():
        YOLO_CACHE.unlink()
        print('Cache dihapus, akan di-generate ulang.')

    # Load frame IDs
    frame_ids = Path(TRAIN_SPLIT).read_text().splitlines()
    frame_ids = [f.strip() for f in frame_ids if f.strip()]
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]
    print(f'\nFrame: {len(frame_ids)} training frames')

    # Pre-compute YOLOv8 detections (dengan cache)
    print('\nStep 0: YOLOv8 detections')
    yolo_cache = load_yolo_detections_cached(frame_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Optimize P2
    # ─────────────────────────────────────────────────────────────────────────
    P2_opt = None

    if args.stage in (0, 1):
        print('\n' + '─' * 60)
        print('Stage 1: Optimasi P2 (fx, fy, cx, cy)')
        print('─' * 60)

        print('Building correspondences...')
        s1_data = build_stage1_data(frame_ids, yolo_cache)
        print(f'  {len(s1_data)} GT objects dengan YOLOv8 detections tersedia')

        # Dua starting points untuk dibandingkan
        # A: KITTI asli
        x0_kitti = np.array([707.0, 707.0, 604.0, 180.0])
        # B: KITTI fx/fy tapi cx,cy disesuaikan ke image size UGM
        x0_fixed = np.array([707.0, 707.0, 640.0, 360.0])

        cost_kitti = cost_stage1(x0_kitti, s1_data, verbose=False)
        cost_fixed = cost_stage1(x0_fixed, s1_data, verbose=False)
        print(f'\n  Baseline cost (KITTI asli,    cy=180): {cost_kitti:.2f} px')
        print(f'  Baseline cost (KITTI + cx,cy fixed): {cost_fixed:.2f} px')

        # Bounds
        bounds = [
            (300, 1500),    # fx
            (300, 1500),    # fy
            (500, 780),     # cx   (tengah ±140px dari 640)
            (250, 470),     # cy   (tengah ±110px dari 360)
        ]

        iter_count[0] = 0
        print(f'\n  Optimizing dari starting point (fx=707, fy=707, cx=640, cy=360)...')
        result1 = minimize(
            cost_stage1, x0_fixed,
            args=(s1_data, 200, args.verbose),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6},
        )

        fx_opt, fy_opt, cx_opt, cy_opt = result1.x
        P2_opt = make_P2(fx_opt, fy_opt, cx_opt, cy_opt)

        print(f'\n  ── Hasil Stage 1 ──')
        print_P2_summary('KITTI asli', 707.0, 707.0, 604.0, 180.0)
        print_P2_summary('Fixed cx,cy', 707.0, 707.0, 640.0, 360.0)
        print_P2_summary('OPTIMIZED ', fx_opt, fy_opt, cx_opt, cy_opt)
        print(f'\n  Cost: {cost_kitti:.2f} → {cost_fixed:.2f} → {result1.fun:.2f} px')
        print(f'  Improvement vs KITTI: {cost_kitti - result1.fun:.2f} px ({(cost_kitti - result1.fun)/cost_kitti*100:.1f}%)')

    # Jika Stage 1 dilewati, pakai KITTI P2 dengan cx,cy yang diperbaiki
    if P2_opt is None:
        P2_opt = make_P2(707.0, 707.0, 640.0, 360.0)
        print(f'\nStage 1 dilewati → menggunakan P2 dengan cx=640, cy=360')

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Optimize Tr_velo_to_cam
    # ─────────────────────────────────────────────────────────────────────────
    Tr_opt = KITTI_TR.copy()

    if args.stage in (0, 2):
        print('\n' + '─' * 60)
        print('Stage 2: Optimasi Tr_velo_to_cam (tx, ty, tz, rx, ry, rz)')
        print('─' * 60)
        print('Note: LiDAR positions diaproksimasi dari GT camera-frame labels')
        print('      via inverse axis transform (x_l≈z_c, y_l≈-x_c, z_l≈-y_c)')

        print('\nBuilding approximate LiDAR positions...')
        s2_data = build_stage2_data(frame_ids, yolo_cache)
        print(f'  {len(s2_data)} approx LiDAR points dengan YOLOv8 detections')

        # Dekomposisi KITTI Tr ke euler angles sebagai starting point
        R_kitti = KITTI_TR[:, :3]
        t_kitti = KITTI_TR[:, 3]
        euler0  = Rotation.from_matrix(R_kitti).as_euler('xyz')
        x0_tr   = np.concatenate([t_kitti, euler0])

        cost_tr0 = cost_stage2(x0_tr, s2_data, P2_opt, verbose=False)
        print(f'\n  Baseline cost (KITTI Tr): {cost_tr0:.2f} px')

        # Bounds: translasi ±50cm, rotasi ±20° dari KITTI starting point
        dr = np.radians(20)
        bounds_tr = [
            (t_kitti[0] - 0.5, t_kitti[0] + 0.5),
            (t_kitti[1] - 0.5, t_kitti[1] + 0.5),
            (t_kitti[2] - 0.5, t_kitti[2] + 0.5),
            (euler0[0] - dr, euler0[0] + dr),
            (euler0[1] - dr, euler0[1] + dr),
            (euler0[2] - dr, euler0[2] + dr),
        ]

        iter_count[0] = 0
        print(f'  Optimizing...')
        result2 = minimize(
            cost_stage2, x0_tr,
            args=(s2_data, P2_opt, 200, args.verbose),
            method='L-BFGS-B',
            bounds=bounds_tr,
            options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6},
        )

        tx, ty, tz, rx, ry, rz = result2.x
        Tr_opt = make_Tr(tx, ty, tz, rx, ry, rz)

        print(f'\n  ── Hasil Stage 2 ──')
        print_Tr_summary('KITTI asli', KITTI_TR)
        print_Tr_summary('OPTIMIZED ', Tr_opt)
        print(f'\n  Cost: {cost_tr0:.2f} → {result2.fun:.2f} px')
        print(f'  Improvement vs KITTI: {cost_tr0 - result2.fun:.2f} px ({(cost_tr0 - result2.fun)/cost_tr0*100:.1f}%)')

    # ─────────────────────────────────────────────────────────────────────────
    # Tulis output calib files
    # ─────────────────────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / args.out_calib
    print(f'\n' + '─' * 60)
    print(f'Menulis calib files ke: {out_dir}')
    write_calib_files(P2_opt, Tr_opt, out_dir)

    print('\n══ FINAL CALIBRATION ══')
    print(f'P2:\n{P2_opt}')
    print(f'\nTr_velo_to_cam:\n{Tr_opt}')
    print(f'\nR0_rect: Identity (3×3)')
    print(f'\nCalib files: {out_dir}/')
    print(f'\nUntuk demo dengan calib baru:')
    print(f'  conda run -n openpcdet python scripts/demo_clocs_ugm.py \\')
    print(f'    --calib_dir {out_dir} --frame_id 000001 --score_thresh 0.05')
