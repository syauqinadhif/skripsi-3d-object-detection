"""
generate_pseudolabels_ft.py — Generate pseudo-labels UGM menggunakan
PointPillars fine-tuned KITTI→UGM (checkpoint_epoch_30.pth).

Perbedaan dari generate_ugm_pseudolabels.py (versi lama):
  - Memakai fine-tuned checkpoint (bukan KITTI pretrained)
  - Config pointpillar_ugm.yaml (point range UGM, bukan KITTI)
  - 2D bbox dihitung dari proyeksi 3D box → image pakai P2 UGM asli
  - Output ke label_2_pseudolabel_ft/ (tidak menimpa label_2/)

Output format KITTI label (per baris):
  class truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

Usage:
  conda run -n openpcdet python scripts/generate_pseudolabels_ft.py
  conda run -n openpcdet python scripts/generate_pseudolabels_ft.py --score_thresh 0.25
  conda run -n openpcdet python scripts/generate_pseudolabels_ft.py --out_dir dataset_ugm/label_2
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
OPENPCDET    = PROJECT_ROOT / 'OpenPCDet'
sys.path.insert(0, str(OPENPCDET))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

PP_CFG  = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar_ugm.yaml'
PP_CKPT = OPENPCDET / 'output/kitti_models/pointpillar_ugm/finetune_kitti/ckpt/checkpoint_epoch_30.pth'

UGM_DIR    = PROJECT_ROOT / 'dataset_ugm'
VELODYNE   = UGM_DIR / 'velodyne'
CALIB_DIR  = UGM_DIR / 'calib'
OUT_DIR    = UGM_DIR / 'label_2_pseudolabel_ft'

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
IMG_W, IMG_H = 1280, 720


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class MinimalDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, logger):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=logger,
        )

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return {}


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calib(calib_path):
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line or line.startswith('#'):
                continue
            key, val = line.strip().split(':', 1)
            vals = val.split()
            if vals:
                calib[key.strip()] = np.array([float(x) for x in vals])
    P2             = calib['P2'].reshape(3, 4)
    R0_rect        = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam


# ── Geometry helpers ──────────────────────────────────────────────────────────

def boxes_lidar_to_camera(boxes_lidar, R0_rect, Tr_velo_to_cam):
    """(N,7) LiDAR → (N,7) KITTI camera [x,y,z,h,w,l,ry]."""
    if len(boxes_lidar) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    Tr_ext = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    R0_ext = np.eye(4); R0_ext[:3, :3] = R0_rect

    centers_h = np.hstack([boxes_lidar[:, :3], np.ones((len(boxes_lidar), 1))])
    centers_c = (R0_ext @ Tr_ext @ centers_h.T).T[:, :3]

    dx, dy, dz = boxes_lidar[:, 3], boxes_lidar[:, 4], boxes_lidar[:, 5]
    h, w, l = dz, dy, dx

    heading = boxes_lidar[:, 6]
    ry = ((-heading - np.pi / 2) + np.pi) % (2 * np.pi) - np.pi

    return np.stack([centers_c[:, 0], centers_c[:, 1], centers_c[:, 2],
                     h, w, l, ry], axis=1).astype(np.float32)


def get_3d_corners(h, w, l, x, y, z, ry):
    """8 sudut 3D box dalam koordinat kamera."""
    corners = np.array([
        [ l/2,  0,  w/2], [ l/2,  0, -w/2],
        [-l/2,  0, -w/2], [-l/2,  0,  w/2],
        [ l/2, -h,  w/2], [ l/2, -h, -w/2],
        [-l/2, -h, -w/2], [-l/2, -h,  w/2],
    ])
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return corners @ R.T + np.array([x, y, z])


def project_box_to_image(h, w, l, x, y, z, ry, P2):
    """Project 3D box → 2D bbox (x1,y1,x2,y2) dalam pixel. None jika di belakang kamera."""
    corners = get_3d_corners(h, w, l, x, y, z, ry)
    corners_h = np.hstack([corners, np.ones((8, 1))])
    proj = (P2 @ corners_h.T).T
    mask = proj[:, 2] > 0
    if not mask.any():
        return None
    pts = proj[mask]
    uvs = pts[:, :2] / pts[:, 2:3]
    x1 = max(0.0,          float(uvs[:, 0].min()))
    y1 = max(0.0,          float(uvs[:, 1].min()))
    x2 = min(IMG_W - 1.0, float(uvs[:, 0].max()))
    y2 = min(IMG_H - 1.0, float(uvs[:, 1].max()))
    if x2 < 0 or y2 < 0 or x1 >= IMG_W or y1 >= IMG_H:
        return None
    return x1, y1, x2, y2


def compute_alpha(x_cam, z_cam, ry):
    alpha = ry - np.arctan2(x_cam, z_cam)
    return float((alpha + np.pi) % (2 * np.pi) - np.pi)


# ── Label writer ──────────────────────────────────────────────────────────────

def boxes_to_label_lines(names, scores, boxes_camera, P2):
    lines = []
    for i in range(len(names)):
        x, y, z, h, w, l, ry = boxes_camera[i]
        alpha = compute_alpha(x, z, ry)
        bbox = project_box_to_image(h, w, l, x, y, z, ry, P2)
        if bbox is None:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        else:
            x1, y1, x2, y2 = bbox
        lines.append(
            f"{names[i]} 0.00 0 {alpha:.2f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{x:.2f} {y:.2f} {z:.2f} {ry:.2f}"
        )
    return lines


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(logger):
    os.chdir(str(OPENPCDET / 'tools'))
    cfg_from_yaml_file(str(PP_CFG), cfg)
    dataset = MinimalDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=OPENPCDET / 'data/ugm',
        logger=logger,
    )
    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(filename=str(PP_CKPT), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    return model, dataset


def infer_frame(model, dataset, bin_path):
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    data_dict = dataset.prepare_data({'points': points, 'frame_id': bin_path.stem})
    data_dict = dataset.collate_batch([data_dict])
    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)
    pred   = pred_dicts[0]
    boxes  = pred['pred_boxes'].cpu().numpy()
    scores = pred['pred_scores'].cpu().numpy()
    labels = pred['pred_labels'].cpu().numpy()
    names  = np.array([CLASS_NAMES[l - 1] for l in labels])
    return names, scores, boxes


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--score_thresh', type=float, default=0.3,
                   help='Score threshold (default 0.3). Turunkan ke 0.2 untuk lebih banyak recall.')
    p.add_argument('--out_dir', type=str, default=str(OUT_DIR),
                   help='Direktori output label (default: label_2_pseudolabel_ft/)')
    p.add_argument('--velodyne_dir', type=str, default=str(VELODYNE))
    p.add_argument('--calib_dir',   type=str, default=str(CALIB_DIR))
    return p.parse_args()


if __name__ == '__main__':
    args  = parse_args()
    vel_dir   = Path(args.velodyne_dir)
    calib_dir = Path(args.calib_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(vel_dir.glob('*.bin'))

    print(f"\nGenerate Pseudo-Labels UGM (Fine-tuned Model)")
    print(f"  Model      : {PP_CKPT.name}  ({PP_CKPT.parent.parent.name})")
    print(f"  Config     : {PP_CFG.name}")
    print(f"  Frames     : {len(bin_files)}")
    print(f"  Threshold  : score >= {args.score_thresh}")
    print(f"  Output     : {out_dir}")
    print(f"  Image size : {IMG_W}x{IMG_H} (untuk proyeksi 2D bbox)\n")

    # Load satu calib (semua frame UGM pakai kalibrasi yang sama)
    sample_calib = sorted(calib_dir.glob('*.txt'))[0]
    P2, R0_rect, Tr_velo_to_cam = load_calib(sample_calib)
    print(f"  Calib P2 (fx={P2[0,0]:.1f} fy={P2[1,1]:.1f} "
          f"cx={P2[0,2]:.1f} cy={P2[1,2]:.1f})\n")

    logger = common_utils.create_logger()
    model, dataset = build_model(logger)

    counts   = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    n_empty  = 0

    for bin_path in tqdm(bin_files, desc='  Inferencing'):
        names, scores, boxes_lidar = infer_frame(model, dataset, bin_path)

        mask        = scores >= args.score_thresh
        names       = names[mask]
        scores      = scores[mask]
        boxes_lidar = boxes_lidar[mask]

        boxes_cam = boxes_lidar_to_camera(boxes_lidar, R0_rect, Tr_velo_to_cam)
        lines     = boxes_to_label_lines(names, scores, boxes_cam, P2)

        out_path = out_dir / f"{bin_path.stem}.txt"
        out_path.write_text('\n'.join(lines) + ('\n' if lines else ''))

        if not lines:
            n_empty += 1
        for n in names:
            if n in counts:
                counts[n] += 1

    total = sum(counts.values())
    print(f"\n── Hasil ──────────────────────────────────────")
    print(f"  Label files     : {len(bin_files)}")
    print(f"  Frame kosong    : {n_empty}")
    print(f"  Total boxes     : {total}  (avg {total/max(len(bin_files),1):.1f}/frame)")
    print(f"  Car             : {counts['Car']}  ({counts['Car']/max(total,1)*100:.1f}%)")
    print(f"  Pedestrian      : {counts['Pedestrian']}  ({counts['Pedestrian']/max(total,1)*100:.1f}%)")
    print(f"  Cyclist         : {counts['Cyclist']}  ({counts['Cyclist']/max(total,1)*100:.1f}%)")
    print(f"\n  Output: {out_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review sample output: head {out_dir}/000000.txt")
    print(f"  2. Import ke CVAT sebagai pre-annotation (KITTI 1.0 format)")
    print(f"  3. Setelah review bagus, copy ke label_2/ untuk replace pseudo-label lama:")
    print(f"       cp -r {out_dir}/ {UGM_DIR}/label_2_backup/  # backup dulu")
    print(f"       cp {out_dir}/*.txt {UGM_DIR}/label_2/")
