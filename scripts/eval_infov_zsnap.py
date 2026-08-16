#!/usr/bin/env python3
"""
Eval IN-FOV: hitung AP hanya untuk objek di dalam FOV kamera.
- GT   : buang objek luar-FOV (bbox 0000 / area 0) -> sisakan yang benar-benar in-FOV.
- Pred : buang prediksi yang center-nya di luar image (bbox area 0) -> tak jadi FP paksa.
Lalu pakai evaluator resmi KITTI (kitti_object_eval_python).
Bandingkan dengan eval full (semua) untuk lihat efek artefak FP luar-FOV.
"""
import os, sys, pickle, argparse
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'OpenPCDet', 'tools'))
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

GT_PKL = '../data/ugm/kitti_infos_val80_manual_zsnap.pkl'
# DT result.pkl bisa dioper via argv[1]; default = baseline PP KITTI-anchor epoch_50
DT_PKL = sys.argv[1] if len(sys.argv) > 1 else '../output/kitti_models/pointpillar_ugm/default/eval/epoch_50/val/ab_manual/result.pkl'
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
GT_KEYS = ['name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y']


def valid_bbox_mask(bbox):
    bbox = np.asarray(bbox).reshape(-1, 4)
    w = bbox[:, 2] - bbox[:, 0]; h = bbox[:, 3] - bbox[:, 1]
    return (w >= 1) & (h >= 1)


def subset(anno, mask, keys):
    out = {}
    for k in keys:
        v = np.asarray(anno[k])
        out[k] = v[mask] if v.shape[0] == mask.shape[0] else v
    return out


def build(gt_infos, dt_list, infov):
    dtmap = {d['frame_id']: d for d in dt_list}
    gt_annos, dt_annos = [], []
    for info in gt_infos:
        fid = info['point_cloud']['lidar_idx']
        ga = info['annos']; d = dtmap[fid]
        gmask = np.ones(len(ga['name']), bool)
        dmask = np.ones(len(d['name']), bool)
        if infov:
            gmask = valid_bbox_mask(ga['bbox'])
            dmask = valid_bbox_mask(d['bbox'])
        g = subset(ga, gmask, GT_KEYS); g['score'] = np.zeros(gmask.sum())
        dd = subset(d, dmask, GT_KEYS + ['score'])
        gt_annos.append(g); dt_annos.append(dd)
    return gt_annos, dt_annos


def run(tag, infov):
    gt = pickle.load(open(GT_PKL, 'rb'))
    dt = pickle.load(open(DT_PKL, 'rb'))
    ga, da = build(gt, dt, infov)
    ng = sum(len(x['name']) for x in ga); nd = sum(len(x['name']) for x in da)
    print(f"\n########## {tag}  (GT obj={ng}, pred={nd}) ##########")
    s, _ = kitti_eval.get_official_eval_result(ga, da, CLASSES)
    for line in s.splitlines():
        if 'AP' in line or 'aos' in line or line.strip().endswith(':'):
            print(line)


if __name__ == '__main__':
    run('EVAL FULL (semua objek, termasuk luar-FOV)', infov=False)
    run('EVAL IN-FOV (hanya dalam FOV kamera)', infov=True)
