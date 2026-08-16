#!/usr/bin/env python3
"""
Sweep threshold + TTA untuk pseudolabel di VAL set (98 frame).
Tujuan: lihat berapa objek terdeteksi di tiap score threshold (recall vs effort),
dan apakah TTA (flip-Y + merge NMS) menambah recall.

Hanya MENGHITUNG (tidak menulis label). Override POST_PROCESSING.SCORE_THRESH=0.05
supaya skor rendah ikut keluar untuk disaring.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_pseudolabels_ft as G  # reuse konstanta + MinimalDataset

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
THRESHOLDS = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
FLOOR = 0.05

os.chdir(str(G.OPENPCDET / 'tools'))
cfg_from_yaml_file(str(G.PP_CFG), cfg)
cfg.MODEL.POST_PROCESSING.SCORE_THRESH = FLOOR   # turunkan floor

logger = common_utils.create_logger()
dataset = G.MinimalDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                           root_path=G.OPENPCDET / 'data/ugm', logger=logger)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=str(G.PP_CKPT), logger=logger, to_cpu=False)
model.cuda().eval()
print(f"Model: {G.PP_CKPT.name} | floor SCORE_THRESH={FLOOR}\n")


def infer(points):
    dd = dataset.prepare_data({'points': points, 'frame_id': 'x'})
    dd = dataset.collate_batch([dd]); load_data_to_gpu(dd)
    with torch.no_grad():
        pred, _ = model.forward(dd)
    p = pred[0]
    return (p['pred_boxes'].cpu().numpy(),
            p['pred_scores'].cpu().numpy(),
            p['pred_labels'].cpu().numpy())


def nms_merge(boxes, scores, labels, iou_thr=0.1):
    """Class-wise 3D NMS untuk dedup gabungan asli+TTA."""
    keep_b, keep_s, keep_l = [], [], []
    for c in range(1, len(CLASSES) + 1):
        m = labels == c
        if m.sum() == 0:
            continue
        b = torch.from_numpy(boxes[m]).float().cuda()
        s = torch.from_numpy(scores[m]).float().cuda()
        idx, _ = iou3d_nms_utils.nms_gpu(b, s, thresh=iou_thr)
        idx = idx.cpu().numpy()
        keep_b.append(boxes[m][idx]); keep_s.append(scores[m][idx])
        keep_l.append(labels[m][idx])
    if not keep_b:
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(keep_b), np.concatenate(keep_s), np.concatenate(keep_l)


val_ids = (ROOT / 'dataset_ugm/ImageSets/val.txt').read_text().split()
vel = ROOT / 'dataset_ugm/velodyne'
print(f"Val frames: {len(val_ids)}\n")

# akumulasi: counts[mode][thr][class]
def new_acc():
    return {t: {c: 0 for c in CLASSES} for t in THRESHOLDS}
acc_plain, acc_tta = new_acc(), new_acc()

for i, fid in enumerate(val_ids):
    pts = np.fromfile(str(vel / f"{fid}.bin"), dtype=np.float32).reshape(-1, 4)
    b0, s0, l0 = infer(pts)

    # TTA flip-Y: y->-y ; box: y_center->-y, heading->-heading
    pts_f = pts.copy(); pts_f[:, 1] *= -1
    bf, sf, lf = infer(pts_f)
    bf = bf.copy(); bf[:, 1] *= -1; bf[:, 6] *= -1

    bb = np.concatenate([b0, bf]); ss = np.concatenate([s0, sf]); ll = np.concatenate([l0, lf])
    bt, st, lt = nms_merge(bb, ss, ll, iou_thr=0.1)

    for thr in THRESHOLDS:
        for (boxes_s, labs, acc) in [(s0, l0, acc_plain), (st, lt, acc_tta)]:
            m = boxes_s >= thr
            for c_i, c in enumerate(CLASSES, start=1):
                acc[thr][c] += int(((labs == c_i) & m).sum())
    if (i + 1) % 25 == 0:
        print(f"  ...{i+1}/{len(val_ids)}")

def show(title, acc):
    print(f"\n===== {title} =====")
    print(f"{'thr':>5s} | {'Car':>5s} {'Ped':>4s} {'Cyc':>4s} | {'TOTAL':>6s}")
    print("-" * 34)
    for t in THRESHOLDS:
        c, p, y = acc[t]['Car'], acc[t]['Pedestrian'], acc[t]['Cyclist']
        print(f"{t:5.2f} | {c:5d} {p:4d} {y:4d} | {c+p+y:6d}")

show("PLAIN (tanpa TTA)", acc_plain)
show("TTA (flip-Y + merge NMS)", acc_tta)
print("\nPembanding: pseudolabel val saat ini = 745 objek (Car 569, Ped 95, Cyc 81)")
