#!/usr/bin/env python3
"""
Kalibrasi threshold pseudolabel memakai GT MANUAL (80 frame val) sebagai acuan.

Tujuan:
  1. Cari score threshold yang presisinya tinggi  -> user tidak capek hapus objek FP.
  2. Ukur bias z (dan dimensi) prediksi vs GT manual -> tahu perlu koreksi atau tidak.

Matching: BEV IoU (bird's-eye) >= thr per kelas, greedy by score.
Hanya objek IN-FOV yang dihitung (GT luar-FOV tidak dilabel manual dengan andal).
"""
import os, sys, pickle
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
IOU_MATCH = {'Car': 0.5, 'Pedestrian': 0.25, 'Cyclist': 0.25}
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
FLOOR = 0.05
DEFAULT_CKPT = ROOT / 'OpenPCDet/output/kitti_models/pointpillar_ugm/default/ckpt/checkpoint_epoch_50.pth'
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
if not CKPT.is_absolute():
    CKPT = ROOT / CKPT
GT_PKL = ROOT / 'dataset_ugm/kitti_infos_val80_manual.pkl'

os.chdir(str(G.OPENPCDET / 'tools'))
cfg_from_yaml_file(str(G.PP_CFG), cfg)
cfg.MODEL.POST_PROCESSING.SCORE_THRESH = FLOOR

logger = common_utils.create_logger()
dataset = G.MinimalDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                           root_path=G.OPENPCDET / 'data/ugm', logger=logger)
model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
model.load_params_from_file(filename=str(CKPT), logger=logger, to_cpu=False)
model.cuda().eval()
print(f"Model: {CKPT.name} | floor={FLOOR}\n")


def infer(points):
    dd = dataset.prepare_data({'points': points, 'frame_id': 'x'})
    dd = dataset.collate_batch([dd]); load_data_to_gpu(dd)
    with torch.no_grad():
        pred, _ = model.forward(dd)
    p = pred[0]
    return (p['pred_boxes'].cpu().numpy(), p['pred_scores'].cpu().numpy(),
            p['pred_labels'].cpu().numpy())


def valid_bbox(bbox):
    bbox = np.asarray(bbox).reshape(-1, 4)
    return (bbox[:, 2] - bbox[:, 0] >= 1) & (bbox[:, 3] - bbox[:, 1] >= 1)


gt_infos = pickle.load(open(GT_PKL, 'rb'))

# --- kumpulkan prediksi sekali (floor 0.05), GT in-FOV ---
per_frame = []
for info in tqdm(gt_infos, desc='infer'):
    fid = info['point_cloud']['lidar_idx']
    pts = np.fromfile(str(G.VELODYNE / f'{fid}.bin'), dtype=np.float32).reshape(-1, 4)
    boxes, scores, labels = infer(pts)

    # GT PENUH (termasuk luar-FOV kamera): labeling CVAT dilakukan di tampilan 3D,
    # jadi objek luar-FOV tetap dilabel manual dan tetap harus dibandingkan.
    a = info['annos']
    gt_boxes = np.asarray(a['gt_boxes_lidar']).reshape(-1, 7)
    gt_names = np.asarray(a['name'])
    per_frame.append(dict(pred_boxes=boxes, pred_scores=scores, pred_labels=labels,
                          gt_boxes=gt_boxes, gt_names=gt_names))


def bev_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return iou3d_nms_utils.boxes_bev_iou_cpu(
        torch.from_numpy(a[:, :7]).float(), torch.from_numpy(b[:, :7]).float()).numpy()


print("\n" + "=" * 78)
print("PRESISI / RECALL vs GT MANUAL (BEV IoU: Car 0.5, Ped/Cyc 0.25)")
print("=" * 78)
print(f"{'thr':>5} {'kelas':<11} {'pred':>6} {'TP':>5} {'FP':>5} {'GT':>5} "
      f"{'presisi':>8} {'recall':>7} {'obj/frame':>10}")

best_rows = {}
for thr in THRESHOLDS:
    for ci, cname in enumerate(CLASSES, start=1):
        TP = FP = NGT = NPR = 0
        for f in per_frame:
            pm = (f['pred_labels'] == ci) & (f['pred_scores'] >= thr)
            pb = f['pred_boxes'][pm]; ps = f['pred_scores'][pm]
            gb = f['gt_boxes'][f['gt_names'] == cname]
            NPR += len(pb); NGT += len(gb)
            if len(pb) == 0:
                continue
            order = np.argsort(-ps)
            pb = pb[order]
            iou = bev_iou(pb, gb)
            taken = set()
            for i in range(len(pb)):
                if len(gb) == 0:
                    FP += 1; continue
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= IOU_MATCH[cname] and j not in taken:
                    taken.add(j); TP += 1
                else:
                    FP += 1
        prec = TP / NPR if NPR else 0.0
        rec = TP / NGT if NGT else 0.0
        print(f"{thr:>5.2f} {cname:<11} {NPR:>6} {TP:>5} {FP:>5} {NGT:>5} "
              f"{prec:>8.3f} {rec:>7.3f} {NPR/len(per_frame):>10.2f}")
        best_rows[(thr, cname)] = (prec, rec, NPR)
    print("-" * 78)


# --- bias z & dimensi pada match (thr 0.3) ---
print("\n" + "=" * 78)
print("BIAS PREDIKSI vs GT MANUAL (pasangan match, thr=0.30)")
print("=" * 78)
print(f"{'kelas':<11} {'n':>5} {'dz_med':>8} {'dz_mean':>8} {'dh_med':>8} "
      f"{'z_pred_med':>11} {'z_gt_med':>9}")
for ci, cname in enumerate(CLASSES, start=1):
    dz, dh, zp, zg = [], [], [], []
    for f in per_frame:
        pm = (f['pred_labels'] == ci) & (f['pred_scores'] >= 0.30)
        pb = f['pred_boxes'][pm]
        gb = f['gt_boxes'][f['gt_names'] == cname]
        if len(pb) == 0 or len(gb) == 0:
            continue
        iou = bev_iou(pb, gb)
        for i in range(len(pb)):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= IOU_MATCH[cname]:
                dz.append(pb[i, 2] - gb[j, 2])
                dh.append(pb[i, 5] - gb[j, 5])
                zp.append(pb[i, 2]); zg.append(gb[j, 2])
    if dz:
        print(f"{cname:<11} {len(dz):>5} {np.median(dz):>8.3f} {np.mean(dz):>8.3f} "
              f"{np.median(dh):>8.3f} {np.median(zp):>11.3f} {np.median(zg):>9.3f}")
    else:
        print(f"{cname:<11} {'0':>5}  (tidak ada match)")

print("\nCatatan: dz = z_pred - z_gt (meter, pusat box, frame LiDAR).")
print("dz > 0 berarti prediksi TERLALU TINGGI -> perlu koreksi turun.")
