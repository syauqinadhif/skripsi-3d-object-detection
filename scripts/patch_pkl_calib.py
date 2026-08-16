"""
Step 1: Patch kitti_infos_*.pkl files with new calibration (calib_opt).

Updates per frame:
  - calib['P2'], calib['R0_rect'], calib['Tr_velo_to_cam']
  - annos['bbox']  — recomputed 2D projections using new P2

New P2: fx=699.47, fy=707.80, cx=630.32, cy=440.0  (from dataset_ugm/calib_opt/)
New R0_rect: Identity
New Tr_velo_to_cam: from calib_opt/ (optimized)
"""

import pickle
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
PKL_DIR   = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/OpenPCDet/data/ugm")
CALIB_OPT = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/calib_opt")
PKL_NAMES = ["kitti_infos_train.pkl", "kitti_infos_val.pkl",
             "kitti_infos_trainval.pkl", "kitti_infos_test.pkl"]

# ── read one calib_opt file to get the global calibration ─────────────────────
def parse_calib_file(path):
    data = {}
    with open(path) as f:
        for line in f:
            key, *vals = line.strip().split()
            data[key.rstrip(':')] = np.array([float(v) for v in vals])
    return data

def get_global_calib():
    sample = parse_calib_file(CALIB_OPT / "000000.txt")

    # P2: 3×4 (stored as 12 values, row-major)
    P2_3x4 = sample['P2'].reshape(3, 4)
    # Pad to 4×4
    P2_4x4 = np.eye(4)
    P2_4x4[:3, :] = P2_3x4

    # R0_rect: Identity 4×4
    R0_4x4 = np.eye(4)

    # Tr_velo_to_cam: 3×4 → 4×4
    Tr_3x4 = sample['Tr_velo_to_cam'].reshape(3, 4)
    Tr_4x4 = np.eye(4)
    Tr_4x4[:3, :] = Tr_3x4

    return P2_4x4, R0_4x4, Tr_4x4, P2_3x4

# ── 3D box → 8 corners in camera frame ────────────────────────────────────────
def box3d_corners_camera(loc, dim, rot_y):
    """
    loc:   (x, y, z) bottom-center in camera frame
    dim:   (h, w, l)
    rot_y: rotation around Y-axis in camera frame
    returns: (8, 3)
    """
    h, w, l = dim
    cx, cy, cz = loc

    # 8 corners relative to bottom-center (before rotation)
    # x: forward/back (l), y: up (-h), z: left/right (w)
    corners = np.array([
        [ l/2, 0,    w/2],
        [-l/2, 0,    w/2],
        [-l/2, 0,   -w/2],
        [ l/2, 0,   -w/2],
        [ l/2, -h,   w/2],
        [-l/2, -h,   w/2],
        [-l/2, -h,  -w/2],
        [ l/2, -h,  -w/2],
    ], dtype=np.float32)   # (8, 3)

    R = np.array([
        [ np.cos(rot_y), 0, np.sin(rot_y)],
        [ 0,             1, 0            ],
        [-np.sin(rot_y), 0, np.cos(rot_y)],
    ], dtype=np.float32)

    corners = (R @ corners.T).T + np.array([cx, cy, cz], dtype=np.float32)
    return corners  # (8, 3)

# ── project corners to image ───────────────────────────────────────────────────
def project_corners(corners_3d, P2_3x4):
    """corners_3d: (8,3) → bbox [u1,v1,u2,v2]"""
    n = corners_3d.shape[0]
    pts_h = np.hstack([corners_3d, np.ones((n, 1), dtype=np.float32)])  # (8,4)
    pts_2d = (P2_3x4 @ pts_h.T).T   # (8,3)
    # keep only corners in front of camera (z > 0)
    valid = pts_2d[:, 2] > 0
    if valid.sum() == 0:
        return None
    pts_2d = pts_2d[valid]
    pts_2d[:, :2] /= pts_2d[:, 2:3]
    return pts_2d[:, :2]  # (N,2) u,v

def compute_bbox(locs, dims, rots, P2_3x4, img_shape):
    """Compute 2D bboxes for all objects in a frame."""
    img_h, img_w = img_shape
    bboxes = []
    for loc, dim, rot in zip(locs, dims, rots):
        corners = box3d_corners_camera(loc, dim, rot)
        uv = project_corners(corners, P2_3x4)
        if uv is None:
            bboxes.append([0, 0, 0, 0])
            continue
        u1 = float(np.clip(uv[:, 0].min(), 0, img_w))
        v1 = float(np.clip(uv[:, 1].min(), 0, img_h))
        u2 = float(np.clip(uv[:, 0].max(), 0, img_w))
        v2 = float(np.clip(uv[:, 1].max(), 0, img_h))
        bboxes.append([u1, v1, u2, v2])
    return np.array(bboxes, dtype=np.float32)

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    P2_4x4, R0_4x4, Tr_4x4, P2_3x4 = get_global_calib()
    print("New P2 (3×4):")
    print(P2_3x4)
    print()

    for pkl_name in PKL_NAMES:
        pkl_path = PKL_DIR / pkl_name
        if not pkl_path.exists():
            print(f"Skip (not found): {pkl_name}")
            continue

        with open(pkl_path, 'rb') as f:
            infos = pickle.load(f)

        print(f"Patching {pkl_name}  ({len(infos)} frames)...")
        n_updated = 0

        for info in infos:
            # update calib
            info['calib']['P2'] = P2_4x4.copy()
            info['calib']['R0_rect'] = R0_4x4.copy()
            info['calib']['Tr_velo_to_cam'] = Tr_4x4.copy()

            # recompute 2D bbox from camera-frame 3D annotations
            annos = info.get('annos', {})
            locs = annos.get('location', None)   # (N, 3)
            dims = annos.get('dimensions', None)  # (N, 3) [h,w,l]
            rots = annos.get('rotation_y', None)  # (N,)

            if locs is not None and len(locs) > 0:
                img_shape = info['image']['image_shape']  # (H, W)
                new_bbox = compute_bbox(locs, dims, rots, P2_3x4, img_shape)
                info['annos']['bbox'] = new_bbox
                n_updated += 1

        # save in-place (backup first)
        backup = pkl_path.with_suffix('.pkl.bak')
        if not backup.exists():
            import shutil
            shutil.copy2(pkl_path, backup)
            print(f"  Backup saved: {backup.name}")

        with open(pkl_path, 'wb') as f:
            pickle.dump(infos, f)

        print(f"  Done. {n_updated}/{len(infos)} frames had annotations updated.\n")

    print("All PKL files patched.")

if __name__ == '__main__':
    main()
