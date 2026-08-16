#!/usr/bin/env python3
"""
Bangun info PKL untuk GT v3 (78 frame, sama id dgn val_manual_v2_78, exclude 000195/000212).
Sama persis logika build_val_v2_infos.py, cuma ganti sumber label & split.
"""
import os, pickle
from pathlib import Path
import numpy as np

os.chdir(str(Path(__file__).resolve().parent.parent / 'OpenPCDet' / 'tools'))
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils, object3d_kitti, box_utils

cfg_from_yaml_file('cfgs/dataset_configs/ugm_dataset.yaml', cfg)
logger = common_utils.create_logger()
ROOT = Path('../data/ugm')
ids = sorted(x.strip() for x in open(ROOT / 'ImageSets/val_manual_v3_78.txt') if x.strip())
print(f"{len(ids)} frame val_manual_v3")


def process(ds, sample_idx, label_dir):
    info = {'point_cloud': {'num_features': 4, 'lidar_idx': sample_idx}}
    info['image'] = {'image_idx': sample_idx, 'image_shape': ds.get_image_shape(sample_idx)}
    calib = ds.get_calib(sample_idx)
    P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
    R0 = np.zeros([4, 4], dtype=calib.R0.dtype); R0[3, 3] = 1.; R0[:3, :3] = calib.R0
    V2C = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
    info['calib'] = {'P2': P2, 'R0_rect': R0, 'Tr_velo_to_cam': V2C}

    lf = label_dir / f'{sample_idx}.txt'
    obj_list = object3d_kitti.get_objects_from_label(str(lf)) if lf.stat().st_size > 0 else []
    a = {}
    a['name'] = np.array([o.cls_type for o in obj_list])
    a['truncated'] = np.array([o.truncation for o in obj_list])
    a['occluded'] = np.array([o.occlusion for o in obj_list])
    a['alpha'] = np.array([o.alpha for o in obj_list])
    a['bbox'] = np.concatenate([o.box2d.reshape(1, 4) for o in obj_list], axis=0) if obj_list else np.zeros((0, 4))
    a['dimensions'] = np.array([[o.l, o.h, o.w] for o in obj_list]).reshape(-1, 3)
    a['location'] = np.concatenate([o.loc.reshape(1, 3) for o in obj_list], axis=0) if obj_list else np.zeros((0, 3))
    a['rotation_y'] = np.array([o.ry for o in obj_list])
    a['score'] = np.array([o.score for o in obj_list])
    a['difficulty'] = np.array([o.level for o in obj_list], np.int32)
    num_obj = len([o for o in obj_list if o.cls_type != 'DontCare'])
    num_gt = len(a['name'])
    a['index'] = np.array(list(range(num_obj)) + [-1] * (num_gt - num_obj), dtype=np.int32)

    if num_obj > 0:
        loc, dims, rots = a['location'][:num_obj], a['dimensions'][:num_obj], a['rotation_y'][:num_obj]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    else:
        gt_boxes_lidar = np.zeros((0, 7))
    a['gt_boxes_lidar'] = gt_boxes_lidar

    npig = -np.ones(num_gt, dtype=np.int32)
    if num_obj > 0:
        pts = ds.get_lidar(sample_idx)
        pts_rect = calib.lidar_to_rect(pts[:, 0:3])
        fov = ds.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
        pts_fov = pts[fov]
        corners = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
        for k in range(num_obj):
            npig[k] = box_utils.in_hull(pts_fov[:, 0:3], corners[k]).sum()
    a['num_points_in_gt'] = npig
    info['annos'] = a
    return info


def gen(label_subdir, out_pkl):
    ds = KittiDataset(dataset_cfg=cfg, class_names=['Car', 'Pedestrian', 'Cyclist'],
                      root_path=ROOT, training=False, logger=logger)
    ds.set_split('val_manual_v3_78')
    ld = (ROOT / label_subdir).resolve()
    infos = [process(ds, fid, ld) for fid in ids]
    with open(ROOT / out_pkl, 'wb') as f:
        pickle.dump(infos, f)
    from collections import Counter
    c = Counter(n for i in infos for n in i['annos']['name'] if n != 'DontCare')
    print(f"  {out_pkl}: {len(infos)} frame, {sum(c.values())} obj {dict(c)}  [label={label_subdir}]")


gen('label_2_val_manual_v3', 'kitti_infos_val78_manual_v3.pkl')
print("selesai.")
