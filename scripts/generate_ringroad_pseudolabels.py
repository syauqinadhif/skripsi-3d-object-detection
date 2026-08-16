#!/usr/bin/env python3
"""
generate_ringroad_pseudolabels.py — Pseudolabel 1000 frame RINGROAD.

Dua hal yang diminta user, dan dasarnya:

1) THRESHOLD TIDAK TERLALU RENDAH
   Diukur di 80 frame GT manual (calibrate_pseudolabel_thresh.py). Di thr 0.20
   ada ~18 box Car/frame dengan presisi 0.27 -> mayoritas sampah, capek dihapus.
   Default di sini 0.40: Car ~4.4 box/frame, presisi 0.54.

2) Z TIDAK KETINGGIAN
   Koreksi z per kelas = median (z_pred - z_gt_manual) di 80 frame GT manual.
   Rig ringroad = rig UGM (kamera+LiDAR sama), jadi offset ini bisa dipakai ulang.
   Car -0.07 m, Pedestrian -0.01 m, Cyclist -0.68 m (Cyclist paling melayang).
   Tinggi box (dh) sudah benar (median ~0.00), jadi HANYA z yang digeser.

Usage:
  conda run -n openpcdet python scripts/generate_ringroad_pseudolabels.py
  conda run -n openpcdet python scripts/generate_ringroad_pseudolabels.py --score_thresh 0.5
  conda run -n openpcdet python scripts/generate_ringroad_pseudolabels.py --no_z_correction
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_pseudolabels_ft as G

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

CLASSES = ['Car', 'Pedestrian', 'Cyclist']

# median(z_pred - z_gt_manual) diukur di val80 GT manual, thr 0.30
Z_OFFSET = {'Car': -0.07, 'Pedestrian': -0.01, 'Cyclist': -0.68}

RR = ROOT / 'dataset_ringroad'
CKPT = ROOT / 'OpenPCDet/output/kitti_models/pointpillar_ugm/default/ckpt/checkpoint_epoch_50.pth'

ap = argparse.ArgumentParser()
ap.add_argument('--score_thresh', type=float, default=0.40)
ap.add_argument('--out_dir', default=str(RR / 'label_2'))
ap.add_argument('--data_dir', default=str(RR))
ap.add_argument('--ckpt', default=str(CKPT))
ap.add_argument('--no_z_correction', action='store_true')
args = ap.parse_args()

data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
zoff = {k: 0.0 for k in CLASSES} if args.no_z_correction else Z_OFFSET

os.chdir(str(G.OPENPCDET / 'tools'))
cfg_from_yaml_file(str(G.PP_CFG), cfg)
cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh

logger = common_utils.create_logger()
dataset = G.MinimalDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                           root_path=data_dir, logger=logger)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
model.cuda().eval()

frame_ids = sorted(p.stem for p in (data_dir / 'velodyne').glob('*.bin'))
print(f"\nPseudolabel RINGROAD\n  ckpt   : {Path(args.ckpt).name}"
      f"\n  thr    : {args.score_thresh}\n  z_off  : {zoff}"
      f"\n  frames : {len(frame_ids)}\n  out    : {out_dir}\n")

count = {c: 0 for c in CLASSES}
n_empty = 0

for fid in tqdm(frame_ids, desc='pseudolabel'):
    pts = np.fromfile(str(data_dir / 'velodyne' / f'{fid}.bin'), dtype=np.float32).reshape(-1, 4)
    dd = dataset.prepare_data({'points': pts, 'frame_id': fid})
    dd = dataset.collate_batch([dd]); load_data_to_gpu(dd)
    with torch.no_grad():
        pred, _ = model.forward(dd)
    p = pred[0]
    boxes = p['pred_boxes'].cpu().numpy()
    scores = p['pred_scores'].cpu().numpy()
    labels = p['pred_labels'].cpu().numpy()

    keep = scores >= args.score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    names = np.array([CLASSES[l - 1] for l in labels])

    # koreksi z per kelas (frame LiDAR, sebelum konversi ke kamera)
    for i, nm in enumerate(names):
        boxes[i, 2] += zoff[nm]
        count[nm] += 1

    P2, R0, Tr = G.load_calib(data_dir / 'calib' / f'{fid}.txt')
    boxes_cam = G.boxes_lidar_to_camera(boxes, R0, Tr)
    lines = G.boxes_to_label_lines(names, scores, boxes_cam, P2)
    if not lines:
        n_empty += 1
    (out_dir / f'{fid}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))

total = sum(count.values())
print(f"\nSelesai: {total} objek di {len(frame_ids)} frame ({total/len(frame_ids):.2f} obj/frame)")
for c in CLASSES:
    print(f"  {c:<11}: {count[c]:>5} ({count[c]/len(frame_ids):.2f}/frame)")
print(f"  frame kosong: {n_empty}")
