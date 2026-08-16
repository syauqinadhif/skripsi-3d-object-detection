#!/usr/bin/env python3
"""
prep_demo100.py — Voxelisasi 100 frame UGM di PC untuk demo inferensi Jetson.

Voxelisasi memakai DATA_CONFIG yang SAMA dengan saat ONNX/engine diekspor
(CUDA-PointPillars/cfgs/kitti_models/pointpillar.yaml: range [0,-39.68,-3,69.12,39.68,1],
voxel 0.16, maks 32 titik/pilar) supaya voxel_idxs cocok dengan grid engine di Jetson.

Output:
    deploy/demo100/voxels_demo100.npz   (v_/n_/c_ per frame + ids)
    deploy/demo100/images/<fid>.png     (citra utk YOLO + render di Jetson)

Usage:
    PYTHONPATH=OpenPCDet /home/mf/miniconda3/envs/openpcdet/bin/python deploy/prep_demo100.py \
        --start 0 --count 100
"""
import os, sys, argparse, shutil
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
sys.path.insert(0, str(ROOT / 'deploy'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import load_data_to_gpu
from export_pp_onnx import _DS

UGM = ROOT / 'dataset_ugm'
CFG = ROOT / 'CUDA-PointPillars/cfgs/kitti_models/pointpillar.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--out', type=str, default=str(ROOT / 'deploy/demo100'))
    args = ap.parse_args()

    cwd = os.getcwd(); os.chdir(CFG.parents[2])
    cfg_from_yaml_file(str(CFG), cfg); os.chdir(cwd)
    ds = _DS(cfg.DATA_CONFIG, cfg.CLASS_NAMES)

    out = Path(args.out); (out / 'images').mkdir(parents=True, exist_ok=True)

    fids = []
    for i in range(args.start, args.start + args.count):
        fid = f'{i:06d}'
        if (UGM / 'velodyne' / f'{fid}.bin').exists() and (UGM / 'image_2' / f'{fid}.png').exists():
            fids.append(fid)

    store = {}
    for k, fid in enumerate(fids):
        pts = np.fromfile(str(UGM / 'velodyne' / f'{fid}.bin'), dtype=np.float32).reshape(-1, 4)
        dd = ds.collate_batch([ds.prepare_data({'points': pts, 'frame_id': fid})])
        v  = dd['voxels'].astype(np.float32)              # (nv,32,4)
        n  = dd['voxel_num_points'].astype(np.int32)      # (nv,)
        c  = dd['voxel_coords'][:, 1:4].astype(np.int32)  # (nv,3) = [z,y,x]
        store[f'v_{fid}'] = v; store[f'n_{fid}'] = n; store[f'c_{fid}'] = c
        shutil.copy(UGM / 'image_2' / f'{fid}.png', out / 'images' / f'{fid}.png')
        if k % 20 == 0:
            print(f'  {k+1}/{len(fids)}  {fid}: {v.shape[0]} voxel')
    store['ids'] = np.array(fids)
    np.savez(out / 'voxels_demo100.npz', **store)

    npz_mb = (out / 'voxels_demo100.npz').stat().st_size / 1e6
    img_mb = sum(f.stat().st_size for f in (out / 'images').glob('*.png')) / 1e6
    print(f'\nSelesai: {len(fids)} frame')
    print(f'  voxels_demo100.npz : {npz_mb:.1f} MB')
    print(f'  images/            : {img_mb:.1f} MB ({len(fids)} png)')
    print(f'  TOTAL transfer     : {npz_mb + img_mb:.1f} MB')


if __name__ == '__main__':
    main()
