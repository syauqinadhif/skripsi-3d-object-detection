"""
Konversi result.pkl dari OpenPCDet ke format .pt per-frame untuk CLOCs_LQS.
CLOCs_LQS expects: torch.load(file)[0] → dict dengan keys:
  boxes_lidar, location, dimensions, rotation_y, bbox, score, frame_id, name, ...
Format: [dict] (list dengan satu elemen)

FIX BUG CALIB-OUTPUT (2026-07-16):
  Dulu field kamera (location/dimensions/rotation_y/bbox) DISALIN mentah dari result.pkl,
  padahal result.pkl dibuat inferensi PP saat info pkl masih calib SALAH → location ngambang
  di frame kamera → eval 3D CLOCs jadi ~0 (padahal boxes_lidar benar).
  Sekarang field kamera DIHITUNG ULANG dari boxes_lidar (calib-independent & benar) memakai
  calib yang benar (default: dataset_ugm/calib == cy440). boxes_lidar tetap apa adanya.
"""
import pickle
import torch
import numpy as np
from pathlib import Path
import argparse

DEFAULT_CALIB = '/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/calib'
IMG_W, IMG_H = 1280, 720


def load_calib(path):
    d = {}
    for ln in open(path):
        if ':' in ln:
            k, v = ln.split(':', 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = d['P2'].reshape(3, 4)
    R0 = d['R0_rect'].reshape(3, 3) if 'R0_rect' in d else np.eye(3)
    Tr = d['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr


def boxes_lidar_to_camera(boxes_lidar, P2, R0, Tr):
    """boxes_lidar (N,7)[x,y,z,dx,dy,dz,yaw] -> (location bottom-center, dimensions[l,h,w], rotation_y, bbox)."""
    n = len(boxes_lidar)
    if n == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32), np.zeros((0, 4), np.float32))
    # DIPERBAIKI 2026-07-21 (BUG GEOMETRI #3, lihat convert_datumaro_to_kitti.py):
    # dulu geser h/2 di sumbu y KAMERA & bangun sudut di frame kamera (asumsi box hanya
    # berotasi thd sumbu y kamera) — patah saat kamera ber-pitch 2,18 deg. Sekarang semua
    # dihitung di frame LiDAR (tempat box terdefinisi: rotasi murni yaw thd z) lalu ditransform.
    Tr4 = np.vstack([Tr, [0, 0, 0, 1]])
    R4 = np.eye(4); R4[:3, :3] = R0
    M = R4 @ Tr4                                          # LiDAR -> kamera (rect)
    dx, dy, dz = boxes_lidar[:, 3], boxes_lidar[:, 4], boxes_lidar[:, 5]
    h, w, l = dz, dy, dx
    yaw = boxes_lidar[:, 6]
    ry = (-yaw - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    dims = np.stack([l, h, w], axis=1)

    # location KITTI = alas-tengah di frame kamera; alas dihitung di LiDAR dulu lalu ditransform
    bottom_lidar = boxes_lidar[:, :3].copy(); bottom_lidar[:, 2] -= h / 2
    bh = np.hstack([bottom_lidar, np.ones((n, 1))])
    loc = (M @ bh.T).T[:, :3]

    bbox = np.zeros((n, 4), np.float32)
    for i in range(n):
        cx, cy, cz = boxes_lidar[i, :3]
        Rz = np.array([[np.cos(yaw[i]), -np.sin(yaw[i]), 0],
                       [np.sin(yaw[i]),  np.cos(yaw[i]), 0], [0, 0, 1]])
        xc = np.array([l[i]/2, l[i]/2, -l[i]/2, -l[i]/2] * 2)
        yc = np.array([w[i]/2, -w[i]/2, -w[i]/2, w[i]/2] * 2)
        zc = np.array([-h[i]/2] * 4 + [h[i]/2] * 4)
        cor_l = (Rz @ np.vstack([xc, yc, zc])).T + [cx, cy, cz]      # sudut di frame LiDAR
        cor = (M @ np.hstack([cor_l, np.ones((8, 1))]).T).T[:, :3]   # -> frame kamera
        v = cor[:, 2] > 0.1
        if v.sum() < 1:
            continue
        hom = np.hstack([cor[v], np.ones((v.sum(), 1))])
        im = (P2 @ hom.T).T
        im = im[:, :2] / im[:, 2:3]
        x1, y1 = im.min(0); x2, y2 = im.max(0)
        bbox[i] = [np.clip(x1, 0, IMG_W), np.clip(y1, 0, IMG_H), np.clip(x2, 0, IMG_W), np.clip(y2, 0, IMG_H)]
    return loc.astype(np.float32), dims.astype(np.float32), ry.astype(np.float32), bbox


def convert(result_pkl, output_dir, calib_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    calib_dir = Path(calib_dir)

    with open(result_pkl, 'rb') as f:
        det_annos = pickle.load(f)
    print(f"Converting {len(det_annos)} frames... (recompute camera fields dari boxes_lidar, calib={calib_dir})")

    for anno in det_annos:
        frame_id = anno['frame_id']
        result = {}
        for key in anno.keys():
            val = anno[key]
            result[key] = (val.astype(np.float32) if isinstance(val, np.ndarray) and val.dtype == np.float64 else val)

        result['score'] = np.array(anno['score'], dtype=np.float32)
        boxes_lidar = np.array(anno['boxes_lidar'], dtype=np.float32)
        result['boxes_lidar'] = boxes_lidar

        # ── FIX: hitung ulang field kamera dari boxes_lidar + calib BENAR (bukan salin mentah) ──
        cf = calib_dir / f"{frame_id}.txt"
        P2, R0, Tr = load_calib(cf if cf.exists() else calib_dir / '000001.txt')
        loc, dims, ry, bbox = boxes_lidar_to_camera(boxes_lidar, P2, R0, Tr)
        result['location'] = loc
        result['dimensions'] = dims
        result['rotation_y'] = ry
        result['bbox'] = bbox
        result['frame_id'] = frame_id

        torch.save([result], Path(output_dir) / f"{frame_id}.pt")

    print(f"Done! {len(det_annos)} .pt files saved to {output_dir}")
    sample = torch.load(Path(output_dir) / f"{det_annos[0]['frame_id']}.pt")
    assert isinstance(sample, list) and len(sample) == 1, "Format error: should be [dict]"
    print(f"Sample frame {det_annos[0]['frame_id']}: {len(sample[0]['score'])} detections; "
          f"location[:1]={sample[0]['location'][:1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_pkl', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--calib_dir', type=str, default=DEFAULT_CALIB,
                        help='dir calib KITTI-format (default cy440 UGM)')
    args = parser.parse_args()
    convert(args.result_pkl, args.output_dir, args.calib_dir)
