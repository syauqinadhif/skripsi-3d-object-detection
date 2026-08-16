#!/usr/bin/env python3
"""
Konversi anotasi VAL manual (export CVAT Datumaro 3D) -> KITTI label_2.

Input : task1_job1_labels_frame0-79.zip (sudah diekstrak) berisi
          annotations/default.json  (cuboid_3d di frame LiDAR: position/rotation/scale)
Output: dataset_ugm/label_2_val_manual/<fid>.txt  (KITTI 15-field, frame kamera)

Konversi LiDAR->kamera pakai calib UGM yang benar (dataset_ugm/calib/ == calib_opt,
cy=440, hasil search_calib_p2). 2D bbox diproyeksikan beneran (P2), occluded dari Datumaro.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
DATUMARO_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path('val_labels/annotations/default.json')
CALIB_DIR = ROOT / 'dataset_ugm/calib'
OUT_DIR   = ROOT / 'dataset_ugm/label_2_val_manual'
IMG_W, IMG_H = 1280, 720
CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def load_calib(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        d[k.strip()] = np.array([float(x) for x in v.split()])
    return (d['P2'].reshape(3, 4),
            d['R0_rect'].reshape(3, 3) if 'R0_rect' in d else np.eye(3),
            d['Tr_velo_to_cam'].reshape(3, 4))


def lidar_to_cam_center(xyz, R0, Tr):
    Tr4 = np.vstack([Tr, [0, 0, 0, 1]])
    R4 = np.eye(4); R4[:3, :3] = R0
    p = np.hstack([xyz, np.ones((len(xyz), 1))])
    return (R4 @ Tr4 @ p.T).T[:, :3]


def box3d_corners_cam(x, y, z, h, w, l, ry):
    # KITTI: (x,y,z)=bottom center kamera, y ke bawah
    xc = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    yc = [0, 0, 0, 0, -h, -h, -h, -h]
    zc = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    c = np.array([xc, yc, zc])
    R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    return (R @ c).T + np.array([x, y, z])


def project(corners_cam, P2):
    valid = corners_cam[:, 2] > 0.1
    if valid.sum() < 1:
        return None
    pts = corners_cam[valid]
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    img = (P2 @ hom.T).T
    img = img[:, :2] / img[:, 2:3]
    x1, y1 = img.min(0); x2, y2 = img.max(0)
    fully_in = (valid.all() and x2 > 0 and x1 < IMG_W and y2 > 0 and y1 < IMG_H)
    x1 = np.clip(x1, 0, IMG_W); x2 = np.clip(x2, 0, IMG_W)
    y1 = np.clip(y1, 0, IMG_H); y2 = np.clip(y2, 0, IMG_H)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2, fully_in


def main():
    P2, R0, Tr = load_calib(sorted(CALIB_DIR.glob('*.txt'))[0])
    data = json.load(open(DATUMARO_JSON))
    cat = [c['name'] for c in data['categories']['label']['labels']]  # id->name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_frame = n_box = n_outfov = 0
    roundtrip_err = []
    frame_ids = []
    for it in data['items']:
        fid = it['id']; frame_ids.append(fid)
        lines = []
        for a in it['annotations']:
            if a.get('type') != 'cuboid_3d':
                continue
            name = cat[a['label_id']]
            x, y, z = a['position']; dx, dy, dz = a['scale']; yaw = a['rotation'][2]
            occ = 1 if a.get('attributes', {}).get('occluded') else 0

            # lidar center -> camera center; lalu turunkan ke bottom-center (KITTI)
            cc = lidar_to_cam_center(np.array([[x, y, z]]), R0, Tr)[0]
            h, w, l = dz, dy, dx
            ry = (-yaw - np.pi/2 + np.pi) % (2*np.pi) - np.pi
            xc, yc, zc = cc[0], cc[1] + h/2, cc[2]   # y kamera ke bawah -> bottom = center + h/2

            # round-trip check (cam->lidar z) utk validasi calib/konvensi
            Tr4 = np.vstack([Tr, [0, 0, 0, 1]]); R4 = np.eye(4); R4[:3, :3] = R0
            inv = np.linalg.inv(R4 @ Tr4)
            back = (inv @ np.array([cc[0], cc[1], cc[2], 1]))[:3]
            roundtrip_err.append(np.abs(back[2] - z))

            alpha = -np.arctan2(-y, x) + ry
            alpha = (alpha + np.pi) % (2*np.pi) - np.pi

            corners = box3d_corners_cam(xc, yc, zc, h, w, l, ry)
            proj = project(corners, P2)
            if proj is None:
                bb = (0.0, 0.0, 0.0, 0.0); trunc = 0.0; n_outfov += 1
            else:
                x1, y1, x2, y2, fully = proj
                bb = (x1, y1, x2, y2)
                trunc = 0.0 if fully else 0.5

            lines.append(
                f"{name} {trunc:.2f} {occ} {alpha:.2f} "
                f"{bb[0]:.2f} {bb[1]:.2f} {bb[2]:.2f} {bb[3]:.2f} "
                f"{h:.2f} {w:.2f} {l:.2f} {xc:.2f} {yc:.2f} {zc:.2f} {ry:.2f}"
            )
            n_box += 1
        (OUT_DIR / f"{fid}.txt").write_text('\n'.join(lines) + ('\n' if lines else ''))
        n_frame += 1

    (ROOT / 'dataset_ugm/ImageSets/val_manual.txt').write_text('\n'.join(sorted(frame_ids)) + '\n')
    print(f"OK: {n_frame} frame, {n_box} box ({n_outfov} luar-FOV bbox 0000)")
    print(f"round-trip z err: max={max(roundtrip_err):.4f}m mean={np.mean(roundtrip_err):.4f}m (harus ~0)")
    print(f"label -> {OUT_DIR}")
    print(f"split -> dataset_ugm/ImageSets/val_manual.txt ({n_frame} frame)")


if __name__ == '__main__':
    main()
