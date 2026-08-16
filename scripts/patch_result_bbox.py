#!/usr/bin/env python3
"""
Proyeksikan ulang field 'bbox' pada result.pkl memakai calib SEKARANG (dataset_ugm/calib).

Kenapa: hasil test.py menyimpan bbox 2D yang dihitung dengan calib saat inferensi. Setelah
cy diperbaiki 440 -> 399, bbox itu basi. Di protokol eval kita bbox HANYA dipakai untuk filter
in-FOV (metrik 3D memakai location/dimensions/rotation_y dari Tr_velo_to_cam, tak tergantung P2),
jadi cukup bbox-nya yang diperbarui — boxes_lidar & field 3D TIDAK disentuh.

Usage: python scripts/patch_result_bbox.py <in.pkl> <out.pkl>
"""
import pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
IMG_W, IMG_H = 1280, 720


def load_calib():
    d = {}
    for line in (ROOT / 'dataset_ugm/calib/000001.txt').read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    return (d['P2'].reshape(3, 4),
            d['R0_rect'].reshape(3, 3) if 'R0_rect' in d else np.eye(3),
            d['Tr_velo_to_cam'].reshape(3, 4))


P2, R0, Tr = load_calib()
Tr4 = np.vstack([Tr, [0, 0, 0, 1]])
R4 = np.eye(4); R4[:3, :3] = R0


def bbox_from_lidar(bl):
    """bl (N,7) [x,y,z,dx,dy,dz,yaw] frame LiDAR -> bbox 2D (N,4)"""
    out = np.zeros((len(bl), 4))
    for i, b in enumerate(bl):
        x, y, z, dx, dy, dz, yaw = b[:7]
        c = np.array([[dx/2, dx/2, -dx/2, -dx/2] * 2,
                      [dy/2, -dy/2, -dy/2, dy/2] * 2,
                      [-dz/2] * 4 + [dz/2] * 4])
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        cor = (R @ c).T + np.array([x, y, z])
        cam = (R4 @ Tr4 @ np.hstack([cor, np.ones((8, 1))]).T).T[:, :3]
        if (cam[:, 2] <= 0.1).any():
            continue                       # di belakang kamera -> bbox 0000 (luar FOV)
        pr = (P2 @ np.hstack([cam, np.ones((8, 1))]).T).T
        uv = pr[:, :2] / pr[:, 2:3]
        x1, y1 = uv.min(0); x2, y2 = uv.max(0)
        if x2 < 0 or x1 > IMG_W or y2 < 0 or y1 > IMG_H:
            continue                       # keluar bingkai -> luar FOV
        out[i] = [np.clip(x1, 0, IMG_W), np.clip(y1, 0, IMG_H),
                  np.clip(x2, 0, IMG_W), np.clip(y2, 0, IMG_H)]
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = pickle.load(open(src, 'rb'))
    n_in = n_out = 0
    for d in data:
        bl = np.asarray(d.get('boxes_lidar', []))
        if len(bl) == 0:
            d['bbox'] = np.zeros((0, 4)); continue
        bb = bbox_from_lidar(bl)
        d['bbox'] = bb
        n_in += int(((bb[:, 2] - bb[:, 0]) >= 1).sum()); n_out += len(bb)
    pickle.dump(data, open(dst, 'wb'))
    print(f"{len(data)} frame, {n_out} deteksi, {n_in} in-FOV setelah reproyeksi -> {Path(dst).name}")


if __name__ == '__main__':
    main()
