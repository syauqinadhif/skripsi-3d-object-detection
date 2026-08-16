"""
generate_ugm_pseudolabels.py — Generate pseudo-labels dataset UGM
menggunakan PointPillars pretrained KITTI (pointpillar_7728.pth).

Alur:
  1. Load PointPillars pretrained KITTI
  2. Inference ke setiap .bin di dataset_ugm/velodyne/
  3. Filter by score threshold (default 0.3)
  4. Convert boxes LiDAR frame → KITTI camera frame format
  5. Simpan sebagai .txt di dataset_ugm/label_2/

Output format KITTI label (per baris):
  class truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

Usage:
  conda run -n openpcdet python scripts/generate_ugm_pseudolabels.py
  conda run -n openpcdet python scripts/generate_ugm_pseudolabels.py --score_thresh 0.4
"""

import argparse
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

PP_CFG  = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar.yaml'
PP_CKPT = OPENPCDET / 'pointpillar_7728.pth'

UGM_DIR    = PROJECT_ROOT / 'dataset_ugm'
VELODYNE   = UGM_DIR / 'velodyne'
LABEL2_DIR = UGM_DIR / 'label_2'
CALIB_DIR  = UGM_DIR / 'calib'

# KITTI class names (index 1=Car, 2=Pedestrian, 3=Cyclist)
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


# ── Minimal dataset wrapper ────────────────────────────────────────────────────

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


# ── Calibration helpers ───────────────────────────────────────────────────────

def load_calib_matrices(calib_path):
    """Load P2, R0_rect, Tr_velo_to_cam dari file calib KITTI."""
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line or line.startswith('#'):
                continue
            key, val = line.strip().split(':', 1)
            calib[key.strip()] = np.array([float(x) for x in val.split()])
    P2            = calib['P2'].reshape(3, 4)
    R0_rect       = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam


def boxes_lidar_to_kitti_camera(boxes_lidar, R0_rect, Tr_velo_to_cam):
    """
    Convert boxes dari LiDAR frame ke KITTI camera frame.
    Input:  boxes_lidar (N, 7)  [x, y, z, dx, dy, dz, heading]
    Output: boxes_camera (N, 7) [x, y, z, h, w, l, ry]
    """
    if len(boxes_lidar) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    # Transform center points: lidar → camera
    Tr_ext = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])  # (4, 4)
    R0_ext = np.eye(4)
    R0_ext[:3, :3] = R0_rect

    centers_l = np.hstack([boxes_lidar[:, :3], np.ones((len(boxes_lidar), 1))])  # (N, 4)
    centers_c = (R0_ext @ Tr_ext @ centers_l.T).T[:, :3]  # (N, 3) camera frame

    # Dimensions: lidar [dx, dy, dz] → camera [h, w, l]
    # In LiDAR: dx=length(forward), dy=width(left), dz=height(up)
    # In KITTI camera: h=height, w=width, l=length
    dx = boxes_lidar[:, 3]  # length
    dy = boxes_lidar[:, 4]  # width
    dz = boxes_lidar[:, 5]  # height
    h = dz
    w = dy
    l = dx

    # Rotation: lidar heading → camera rotation_y
    # ry ≈ -heading - pi/2
    heading = boxes_lidar[:, 6]
    ry = -heading - np.pi / 2
    # Normalize to [-pi, pi]
    ry = (ry + np.pi) % (2 * np.pi) - np.pi

    boxes_camera = np.stack([
        centers_c[:, 0],  # x
        centers_c[:, 1],  # y
        centers_c[:, 2],  # z
        h, w, l, ry
    ], axis=1).astype(np.float32)

    return boxes_camera


def boxes_to_kitti_label_lines(names, scores, boxes_camera):
    """
    Tulis boxes ke format KITTI label string.
    Format: class truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
    """
    lines = []
    for i in range(len(names)):
        x, y, z, h, w, l, ry = boxes_camera[i]
        score = scores[i]
        # 2D bbox: dummy values (tidak dipakai untuk LiDAR-only training)
        x1, y1, x2, y2 = 0.0, 0.0 ,0.0, 0.0
        line = (
            f"{names[i]} 0.00 0 -10.00 "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{x:.2f} {y:.2f} {z:.2f} "
            f"{ry:.2f}"
        )
        lines.append(line)
    return lines


# ── Inference ─────────────────────────────────────────────────────────────────

def build_model(logger):
    import os
    os.chdir(str(OPENPCDET / 'tools'))  # _BASE_CONFIG_ paths resolve relative to tools/
    cfg_from_yaml_file(str(PP_CFG), cfg)
    dataset = MinimalDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=OPENPCDET / 'data/kitti',
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


def infer_single_frame(model, dataset, bin_path):
    """Inference satu frame .bin, return (names, scores, boxes_lidar)."""
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    input_dict = {'points': points, 'frame_id': bin_path.stem}
    data_dict  = dataset.prepare_data(data_dict=input_dict)
    data_dict  = dataset.collate_batch([data_dict])

    load_data_to_gpu(data_dict)
    with torch.no_grad():
        pred_dicts, _ = model.forward(data_dict)

    pred = pred_dicts[0]
    boxes  = pred['pred_boxes'].cpu().numpy()    # (N, 7)
    scores = pred['pred_scores'].cpu().numpy()   # (N,)
    labels = pred['pred_labels'].cpu().numpy()   # (N,) 1-indexed
    names  = np.array([CLASS_NAMES[l - 1] for l in labels])

    return names, scores, boxes


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='Score threshold untuk pseudo-label (default 0.3)')
    parser.add_argument('--velodyne_dir', type=str, default=str(VELODYNE))
    parser.add_argument('--label_dir',   type=str, default=str(LABEL2_DIR))
    parser.add_argument('--calib_dir',   type=str, default=str(CALIB_DIR))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    velodyne_dir = Path(args.velodyne_dir)
    label_dir    = Path(args.label_dir)
    calib_dir    = Path(args.calib_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(velodyne_dir.glob('*.bin'))
    print(f"\nGenerate Pseudo-Labels UGM")
    print(f"  Model  : {PP_CKPT.name}")
    print(f"  Frames : {len(bin_files)}")
    print(f"  Thresh : score >= {args.score_thresh}")
    print(f"  Output : {label_dir}\n")

    # Load calib (semua frame pakai KITTI calib yang sama)
    sample_calib = sorted(calib_dir.glob('*.txt'))[0]
    P2, R0_rect, Tr_velo_to_cam = load_calib_matrices(sample_calib)

    # Build model
    logger = common_utils.create_logger()
    model, dataset = build_model(logger)

    # Inference loop
    total_boxes = 0
    for bin_path in tqdm(bin_files, desc="  Inferencing"):
        names, scores, boxes_lidar = infer_single_frame(model, dataset, bin_path)

        # Filter by score
        mask   = scores >= args.score_thresh
        names  = names[mask]
        scores = scores[mask]
        boxes_lidar = boxes_lidar[mask]

        # Convert ke KITTI camera frame
        boxes_camera = boxes_lidar_to_kitti_camera(boxes_lidar, R0_rect, Tr_velo_to_cam)

        # Tulis label
        label_path = label_dir / f"{bin_path.stem}.txt"
        lines = boxes_to_kitti_label_lines(names, scores, boxes_camera)
        label_path.write_text('\n'.join(lines) + ('\n' if lines else ''))

        total_boxes += len(lines)

    n_labels = len(list(label_dir.glob('*.txt')))
    print(f"\n  ── Hasil ──")
    print(f"  Label files   : {n_labels}")
    print(f"  Total boxes   : {total_boxes}")
    print(f"  Avg boxes/frame: {total_boxes / max(n_labels, 1):.1f}")
    print(f"\n  Pseudo-labels tersimpan di: {label_dir}")
    print(f"  Next: setup ugm_dataset.yaml lalu training PointPillars UGM")
