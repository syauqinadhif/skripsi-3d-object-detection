#!/usr/bin/env python3
"""
Buat file anotasi pseudolabel VAL untuk import ke CVAT 3D — format Sustech POINTS.
Box diambil LANGSUNG dari output PointPillars di frame LiDAR (boxes_lidar):
  position = pusat box (x,y,z) ; scale = (dx,dy,dz) ; rotation.z = heading.
Tidak ada konversi ke koordinat kamera → selaras dengan .pcd yang diupload.

TTA flip-Y + merge NMS, final threshold 0.20 (sama seperti label_2_val_tta_020).
Output: cvat_val_annotations/label/<fid>.json  (1 json per frame)
"""
import os, sys, json
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
SCORE_THRESH, FLOOR, NMS_IOU = 0.20, 0.05, 0.1

os.chdir(str(G.OPENPCDET / 'tools'))
cfg_from_yaml_file(str(G.PP_CFG), cfg)
cfg.MODEL.POST_PROCESSING.SCORE_THRESH = FLOOR
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


OUT = ROOT / 'cvat_val_annotations' / 'label'
OUT.mkdir(parents=True, exist_ok=True)
val_ids = (ROOT / 'dataset_ugm/ImageSets/val.txt').read_text().split()
vel = ROOT / 'dataset_ugm/velodyne'
print(f"Generate anotasi Sustech POINTS untuk {len(val_ids)} frame (thr={SCORE_THRESH})\n")

total = 0
for fid in tqdm(val_ids, desc='  ann'):
    pts = np.fromfile(str(vel / f"{fid}.bin"), dtype=np.float32).reshape(-1, 4)
    b0, s0, l0 = infer(pts)
    pf = pts.copy(); pf[:, 1] *= -1
    bf, sf, lf = infer(pf)
    bf = bf.copy(); bf[:, 1] *= -1; bf[:, 6] *= -1
    bb, ss, ll = nms_merge(np.concatenate([b0, bf]), np.concatenate([s0, sf]),
                           np.concatenate([l0, lf]), NMS_IOU)
    keep = ss >= SCORE_THRESH
    bb, ll = bb[keep], ll[keep]
    objs = []
    for j in range(len(bb)):
        x, y, z, dx, dy, dz, yaw = [float(v) for v in bb[j]]
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # normalisasi ke [-pi, pi]
        objs.append({
            "obj_id": str(j),
            "obj_type": CLASSES[ll[j] - 1],
            "psr": {
                "position": {"x": x, "y": y, "z": z},
                "rotation": {"x": 0.0, "y": 0.0, "z": yaw},
                "scale": {"x": dx, "y": dy, "z": dz},
            },
        })
    total += len(objs)
    (OUT / f"{fid}.json").write_text(json.dumps(objs, indent=1))

print(f"\nSelesai. {len(val_ids)} json, total {total} box.")
print(f"Output: {OUT.parent}")
