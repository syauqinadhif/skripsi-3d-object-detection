#!/usr/bin/env python3
"""
fix_ugm_2d_labels.py

Fixes UGM dataset labels by computing proper:
  - 2D bounding boxes (projected from 3D using camera calibration)
  - alpha (observation angle)

Previously all labels had alpha=-10 and bbox=0,0,0,0 (invalid placeholders),
which caused CLOCs evaluation to report bbox AP=0 and AOS=0, and broke the
fusion training signal.

Usage:
    python fix_ugm_2d_labels.py [--dry-run]
"""

import os
import sys
import shutil
import argparse
import numpy as np
from pathlib import Path

LABEL_DIR = "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/label_2"
CALIB_DIR = "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/calib"
IMG_W = 1280
IMG_H = 720


def read_calib(calib_path):
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.strip().split(':', 1)
            vals = val.strip().split()
            if vals:
                calib[key.strip()] = np.array([float(x) for x in vals])
    P2 = calib['P2'].reshape(3, 4)
    return P2


def get_3d_corners(h, w, l, x, y, z, ry):
    """
    Returns 8 corners of the 3D box in camera coordinates.

    KITTI convention:
      (x, y, z) = bottom-center of box in camera frame (x right, y down, z fwd)
      h = height (extends upward, i.e., -y direction)
      w = width  (x direction)
      l = length (z direction)
      ry = yaw around Y-axis
    """
    # Local corners centred at box geometric centre
    corners_local = np.array([
        [ l/2,  h/2,  w/2],
        [ l/2,  h/2, -w/2],
        [-l/2,  h/2, -w/2],
        [-l/2,  h/2,  w/2],
        [ l/2, -h/2,  w/2],
        [ l/2, -h/2, -w/2],
        [-l/2, -h/2, -w/2],
        [-l/2, -h/2,  w/2],
    ])

    c, s = np.cos(ry), np.sin(ry)
    R = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])

    corners_rotated = corners_local @ R.T          # (8, 3)
    center = np.array([x, y - h / 2.0, z])         # geometric centre
    return corners_rotated + center                 # (8, 3)


def project_to_image(corners_cam, P2):
    """
    Projects 3D corners to 2D image using P2 (3x4).
    Only returns points where Z > 0 (in front of camera).
    Returns (N, 2) array of (u, v) or None if all points behind camera.
    """
    n = corners_cam.shape[0]
    corners_h = np.hstack([corners_cam, np.ones((n, 1))])   # (8, 4)
    proj = (P2 @ corners_h.T).T                              # (8, 3)

    mask = proj[:, 2] > 0
    if not mask.any():
        return None

    pts = proj[mask]
    uvs = pts[:, :2] / pts[:, 2:3]   # (N, 2)
    return uvs


def compute_alpha(x_cam, z_cam, ry):
    """alpha = ry - atan2(x, z), normalised to [-pi, pi]."""
    alpha = ry - np.arctan2(x_cam, z_cam)
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    return float(alpha)


def fix_labels(label_dir, calib_dir, img_w, img_h, dry_run=False):
    label_dir = Path(label_dir)
    calib_dir = Path(calib_dir)

    # Backup original labels (only once)
    backup_dir = label_dir.parent / 'label_2_original_backup'
    if not dry_run and not backup_dir.exists():
        print(f"Backing up original labels → {backup_dir}")
        shutil.copytree(label_dir, backup_dir)

    # All calib files are identical — read any one
    calib_file = sorted(calib_dir.glob('*.txt'))[0]
    P2 = read_calib(calib_file)
    print(f"Using calib: {calib_file}")
    print(f"P2:\n{P2}\n")

    label_files = sorted(label_dir.glob('*.txt'))
    n_files = 0
    n_total_obj = 0
    n_fixed = 0
    n_outside = 0
    n_clipped = 0

    for lf in label_files:
        raw = lf.read_text().strip()
        if not raw:
            continue
        lines = raw.split('\n')
        new_lines = []

        for line in lines:
            parts = line.split()
            if len(parts) < 15:
                new_lines.append(line)
                continue

            cls        = parts[0]
            truncated  = float(parts[1])
            occluded   = int(parts[2])
            # parts[3]  = alpha  (broken)
            # parts[4:8] = 2D bbox (broken)
            h  = float(parts[8])
            w  = float(parts[9])
            l  = float(parts[10])
            x  = float(parts[11])
            y  = float(parts[12])
            z  = float(parts[13])
            ry = float(parts[14])

            n_total_obj += 1

            # --- compute alpha ---
            alpha = compute_alpha(x, z, ry)

            # --- project 3D box → 2D bbox ---
            corners_cam = get_3d_corners(h, w, l, x, y, z, ry)
            uvs = project_to_image(corners_cam, P2)

            if uvs is None:
                # Object entirely behind the camera
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                n_outside += 1
            else:
                rx1 = float(np.min(uvs[:, 0]))
                ry1 = float(np.min(uvs[:, 1]))
                rx2 = float(np.max(uvs[:, 0]))
                ry2 = float(np.max(uvs[:, 1]))

                if rx2 < 0 or ry2 < 0 or rx1 >= img_w or ry1 >= img_h:
                    # Box projected entirely outside image
                    x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                    n_outside += 1
                else:
                    x1 = max(0.0, rx1)
                    y1 = max(0.0, ry1)
                    x2 = min(float(img_w - 1), rx2)
                    y2 = min(float(img_h - 1), ry2)
                    if (rx1, ry1, rx2, ry2) != (x1, y1, x2, y2):
                        n_clipped += 1
                    n_fixed += 1

            new_line = (
                f"{cls} {truncated:.2f} {occluded} {alpha:.2f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{h:.2f} {w:.2f} {l:.2f} "
                f"{x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
            )
            new_lines.append(new_line)

        if not dry_run:
            lf.write_text('\n'.join(new_lines) + '\n')
        n_files += 1

    print(f"{'[DRY RUN] Would update' if dry_run else 'Updated'} {n_files} label files")
    print(f"Total objects  : {n_total_obj}")
    print(f"  2D bbox fixed (visible)  : {n_fixed}")
    print(f"  Clipped to image border  : {n_clipped}")
    print(f"  Behind / outside image   : {n_outside}")


def verify_sample(label_dir, calib_dir, frame_id='000000'):
    """Print before/after for one frame to sanity-check."""
    import copy
    label_file = Path(label_dir) / f'{frame_id}.txt'
    calib_file = Path(calib_dir) / f'{frame_id}.txt'
    P2 = read_calib(calib_file)

    print(f"\n=== Verification: {frame_id} ===")
    for line in label_file.read_text().strip().split('\n')[:3]:
        parts = line.split()
        if len(parts) < 15:
            continue
        cls = parts[0]
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])
        alpha = compute_alpha(x, z, ry)
        corners_cam = get_3d_corners(h, w, l, x, y, z, ry)
        uvs = project_to_image(corners_cam, P2)
        if uvs is not None:
            x1 = max(0.0, float(np.min(uvs[:, 0])))
            y1 = max(0.0, float(np.min(uvs[:, 1])))
            x2 = min(float(IMG_W - 1), float(np.max(uvs[:, 0])))
            y2 = min(float(IMG_H - 1), float(np.max(uvs[:, 1])))
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        bbox_h = y2 - y1
        print(f"  {cls}: alpha={alpha:.2f}  bbox=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]  height={bbox_h:.1f}px")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without writing files')
    args = parser.parse_args()

    fix_labels(LABEL_DIR, CALIB_DIR, IMG_W, IMG_H, dry_run=args.dry_run)

    if not args.dry_run:
        verify_sample(LABEL_DIR, CALIB_DIR, '000000')
        verify_sample(LABEL_DIR, CALIB_DIR, '000010')
