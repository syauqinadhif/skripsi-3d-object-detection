#!/usr/bin/env python3
"""
create_infos_for.py — create_kitti_infos dengan path EKSPLISIT.

Kenapa perlu: blok __main__ di pcdet/datasets/kitti/kitti_dataset.py meng-hardcode
data_path/save_path ke `data/kitti` dan MENGABAIKAN DATA_PATH di yaml. Jadi
"python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos <yaml>" selalu
memproses KITTI, bukan dataset di yaml itu.

Skrip ini memanggil create_kitti_infos() langsung dengan data_path yang benar,
dan melewati split 'test' (dataset kita tidak punya testing/ terpisah).

Usage:
  conda run -n openpcdet python scripts/create_infos_for.py \
      --cfg OpenPCDet/tools/cfgs/dataset_configs/combined_dataset.yaml \
      --data_dir dataset_combined
"""
import argparse, pickle, sys
from pathlib import Path
import yaml
from easydict import EasyDict

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils import common_utils

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True)
ap.add_argument('--data_dir', required=True)
ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args()

cfg_path = Path(args.cfg)
if not cfg_path.is_absolute():
    cfg_path = ROOT / cfg_path
data_path = Path(args.data_dir)
if not data_path.is_absolute():
    data_path = ROOT / data_path

dataset_cfg = EasyDict(yaml.safe_load(open(cfg_path)))
class_names = ['Car', 'Pedestrian', 'Cyclist']

dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names,
                       root_path=data_path, training=False,
                       logger=common_utils.create_logger())

train_split = dataset_cfg.DATA_SPLIT['train']
val_split = dataset_cfg.DATA_SPLIT['test']

print(f"\ndata_path : {data_path}\ntrain split: {train_split}\nval split  : {val_split}\n")

dataset.set_split(train_split)
infos_train = dataset.get_infos(num_workers=args.workers, has_label=True, count_inside_pts=True)
(data_path / 'kitti_infos_train.pkl').write_bytes(pickle.dumps(infos_train))
print(f"train infos: {len(infos_train)} -> kitti_infos_train.pkl")

dataset.set_split(val_split)
infos_val = dataset.get_infos(num_workers=args.workers, has_label=True, count_inside_pts=True)
(data_path / 'kitti_infos_val.pkl').write_bytes(pickle.dumps(infos_val))
print(f"val infos  : {len(infos_val)} -> kitti_infos_val.pkl")

(data_path / 'kitti_infos_trainval.pkl').write_bytes(pickle.dumps(infos_train + infos_val))
print("trainval infos saved")

print("\n--- gt_database (augmentasi gt_sampling) ---")
dataset.set_split(train_split)
dataset.create_groundtruth_database(str(data_path / 'kitti_infos_train.pkl'), split=train_split)
print("\nSelesai.")
