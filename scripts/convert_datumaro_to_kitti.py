#!/usr/bin/env python3
"""
Konversi anotasi manual (export CVAT Datumaro 3D) -> KITTI label_2. Generik (val / train).

Generalisasi dari convert_datumaro_val_to_kitti.py (yang khusus val 80 frame). Logika
konversi IDENTIK; yang dibuat argumen cuma path input/output/split.

⚠️ DIPERBAIKI 2026-07-21 — dua bug geometri yang baru terekspos oleh calib v3:

(1) ALAS BOX digeser di sumbu yang salah. Dulu: pusat box dikonversi ke frame kamera, lalu
    digeser +h/2 sepanjang sumbu **y kamera**. Padahal OpenPCDet membalikkannya (get_infos ->
    rect_to_lidar lalu `loc_lidar[:,2] += h/2`) sepanjang sumbu **z LiDAR**. Keduanya hanya
    setara kalau kamera tegak sempurna terhadap LiDAR. Calib lama pitch-nya 0,24 deg -> galat
    0,3 cm (tak terlihat); calib v3 pitch aslinya 2,18 deg -> galat 2,9 cm sistematis di sumbu x.
    SEKARANG: alas dihitung di frame LiDAR (center - [0,0,h/2]) LALU ditransformasi -> round-trip
    ke gt_boxes_lidar eksak.

(2) BBOX 2D dibangun lewat aproksimasi frame kamera (`box3d_corners_cam`, asumsi box hanya
    berotasi terhadap sumbu y kamera). Asumsi itu juga patah saat kamera ber-pitch.
    SEKARANG: 8 sudut dibangun di frame LiDAR (tempat box memang terdefinisi: rotasi murni
    yaw terhadap z) lalu diproyeksikan. Lebih akurat -> penting untuk label YOLO & CLOCs.

Catatan reproduksibilitas: klaim lama "byte-identical mereproduksi output val" berlaku untuk
versi SEBELUM tanggal ini. Output lama ada di `dataset_ugm/label_2_*_pre_v3_backup/`.

Contoh:
  python scripts/convert_datumaro_to_kitti.py \
      --json /tmp/.../train150/annotations/default.json \
      --out  dataset_ugm/label_2_train_manual \
      --split dataset_ugm/ImageSets/train_manual150.txt
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
CALIB_DIR = ROOT / 'dataset_ugm/calib'
IMG_W, IMG_H = 1280, 720


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


def box3d_corners_lidar(x, y, z, l, w, h, yaw):
    """8 sudut box di frame LiDAR, tempat box memang terdefinisi (rotasi murni yaw thd z).
    (x,y,z) = PUSAT box. Tidak ada aproksimasi kamera-tegak di sini."""
    xc = np.array([l/2, l/2, -l/2, -l/2] * 2)
    yc = np.array([w/2, -w/2, -w/2, w/2] * 2)
    zc = np.array([-h/2] * 4 + [h/2] * 4)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]])
    return (R @ np.vstack([xc, yc, zc])).T + np.array([x, y, z])


def project(corners_lidar, P2, R0, Tr):
    corners_cam = lidar_to_cam_center(corners_lidar, R0, Tr)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='annotations/default.json dari export Datumaro 3D')
    ap.add_argument('--out', required=True, help='folder output label KITTI')
    ap.add_argument('--split', required=True, help='file ImageSets yang ditulis (daftar frame id)')
    args = ap.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT / split_path

    P2, R0, Tr = load_calib(sorted(CALIB_DIR.glob('*.txt'))[0])
    data = json.load(open(args.json))
    cat = [c['name'] for c in data['categories']['label']['labels']]  # id->name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frame = n_box = n_outfov = n_empty = 0
    roundtrip_err = []
    frame_ids = []
    Tr4 = np.vstack([Tr, [0, 0, 0, 1]]); R4 = np.eye(4); R4[:3, :3] = R0
    inv = np.linalg.inv(R4 @ Tr4)

    for it in data['items']:
        fid = it['id']; frame_ids.append(fid)
        lines = []
        for a in it['annotations']:
            if a.get('type') != 'cuboid_3d':
                continue
            name = cat[a['label_id']]
            x, y, z = a['position']; dx, dy, dz = a['scale']; yaw = a['rotation'][2]
            occ = 1 if a.get('attributes', {}).get('occluded') else 0

            h, w, l = dz, dy, dx
            ry = (-yaw - np.pi/2 + np.pi) % (2*np.pi) - np.pi

            # ALAS box dihitung di frame LiDAR dulu (lihat catatan bug (1) di docstring),
            # baru ditransformasi -> inverse eksak dari get_infos() OpenPCDet.
            bottom_lidar = np.array([[x, y, z - h/2]])
            xc, yc, zc = lidar_to_cam_center(bottom_lidar, R0, Tr)[0]

            # round-trip: kamera -> LiDAR -> pusat, harus kembali ke z semula
            back = (inv @ np.array([xc, yc, zc, 1]))[:3]
            roundtrip_err.append(np.abs(back[2] + h/2 - z))

            alpha = -np.arctan2(-y, x) + ry
            alpha = (alpha + np.pi) % (2*np.pi) - np.pi

            proj = project(box3d_corners_lidar(x, y, z, l, w, h, yaw), P2, R0, Tr)
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
        if not lines:
            n_empty += 1
        (out_dir / f"{fid}.txt").write_text('\n'.join(lines) + ('\n' if lines else ''))
        n_frame += 1

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text('\n'.join(sorted(frame_ids)) + '\n')
    print(f"OK: {n_frame} frame, {n_box} box ({n_outfov} luar-FOV bbox 0000, {n_empty} frame kosong)")
    print(f"round-trip z err: max={max(roundtrip_err):.4f}m mean={np.mean(roundtrip_err):.4f}m (harus ~0)")
    print(f"label -> {out_dir}")
    print(f"split -> {split_path} ({n_frame} frame)")


if __name__ == '__main__':
    main()
