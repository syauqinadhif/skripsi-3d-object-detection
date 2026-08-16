#!/usr/bin/env python3
"""
Generate pseudolabel VAL set (98 frame) dengan TTA (flip-Y + merge NMS) @ threshold 0.20.
Output: format KITTI (15 field) sama seperti dataset_ugm/label_2, siap untuk CVAT.

Floor inferensi 0.05 (agar kandidat TTA banyak), final filter score>=0.20 SETELAH merge.
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_pseudolabels_ft as G

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from tqdm import tqdm

CLASSES = ['Car', 'Pedestrian', 'Cyclist']

ap = argparse.ArgumentParser()
ap.add_argument('--score_thresh', type=float, default=0.20)
ap.add_argument('--floor', type=float, default=0.05)
ap.add_argument('--out_dir', type=str, default=str(ROOT / 'dataset_ugm/label_2_val_tta_020'))
ap.add_argument('--nms_iou', type=float, default=0.1)
args = ap.parse_args()

os.chdir(str(G.OPENPCDET / 'tools'))
cfg_from_yaml_file(str(G.PP_CFG), cfg)
cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.floor

logger = common_utils.create_logger()
dataset = G.MinimalDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                           root_path=G.OPENPCDET / 'data/ugm', logger=logger)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=str(G.PP_CKPT), logger=logger, to_cpu=False)
model.cuda().eval()


def infer(points):
    dd = dataset.prepare_data({'points': points, 'frame_id': 'x'})
    dd = dataset.collate_batch([dd]); load_data_to_gpu(dd)
    with torch.no_grad():
        pred, _ = model.forward(dd)
    p = pred[0]
    return (p['pred_boxes'].cpu().numpy(), p['pred_scores'].cpu().numpy(),
            p['pred_labels'].cpu().numpy())


def nms_merge(boxes, scores, labels, iou_thr):
    kb, ks, kl = [], [], []
    for c in range(1, len(CLASSES) + 1):
        m = labels == c
        if m.sum() == 0:
            continue
        b = torch.from_numpy(boxes[m]).float().cuda()
        s = torch.from_numpy(scores[m]).float().cuda()
        idx, _ = iou3d_nms_utils.nms_gpu(b, s, thresh=iou_thr)
        idx = idx.cpu().numpy()
        kb.append(boxes[m][idx]); ks.append(scores[m][idx]); kl.append(labels[m][idx])
    if not kb:
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(kb), np.concatenate(ks), np.concatenate(kl)


out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
val_ids = (ROOT / 'dataset_ugm/ImageSets/val.txt').read_text().split()
vel = ROOT / 'dataset_ugm/velodyne'
calib_dir = ROOT / 'dataset_ugm/calib'
P2, R0_rect, Tr = G.load_calib(sorted(calib_dir.glob('*.txt'))[0])

print(f"Generate VAL pseudolabel + TTA | thr={args.score_thresh} floor={args.floor}")
print(f"  Model: {G.PP_CKPT.name} | frames: {len(val_ids)} | out: {out_dir}\n")

counts = {c: 0 for c in CLASSES}
for fid in tqdm(val_ids, desc='  gen'):
    pts = np.fromfile(str(vel / f"{fid}.bin"), dtype=np.float32).reshape(-1, 4)
    b0, s0, l0 = infer(pts)
    pf = pts.copy(); pf[:, 1] *= -1
    bf, sf, lf = infer(pf)
    bf = bf.copy(); bf[:, 1] *= -1; bf[:, 6] *= -1
    bb, ss, ll = nms_merge(np.concatenate([b0, bf]), np.concatenate([s0, sf]),
                           np.concatenate([l0, lf]), args.nms_iou)
    keep = ss >= args.score_thresh
    bb, ss, ll = bb[keep], ss[keep], ll[keep]
    names = np.array([CLASSES[k - 1] for k in ll]) if len(ll) else np.array([])
    for n in names:
        counts[n] += 1
    boxes_cam = G.boxes_lidar_to_camera(bb, R0_rect, Tr)
    lines = G.boxes_to_label_lines(names, ss, boxes_cam, P2)
    (out_dir / f"{fid}.txt").write_text('\n'.join(lines) + ('\n' if lines else ''))

print(f"\nSelesai. Objek per kelas: {counts} | total {sum(counts.values())}")
print(f"Output: {out_dir}")
