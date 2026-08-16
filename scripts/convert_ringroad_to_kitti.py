#!/usr/bin/env python3
"""
convert_ringroad_to_kitti.py — Convert dataset RINGROAD ke format KITTI.

Beda dari convert_ugm_to_kitti.py:
  - image_2/*.png dibuat SYMLINK ke sumber (hemat disk; sumber di drive yang sama)
  - calib/*.txt disalin dari dataset_ugm/calib (cy=440, hasil kalibrasi search_calib_p2)
    -> rig kamera+LiDAR ringroad SAMA dengan UGM, jadi calib dipakai ulang
  - velodyne/*.bin tetap file nyata (pcd binary_compressed tidak bisa di-symlink jadi bin)

Usage:
  conda run -n openpcdet python scripts/convert_ringroad_to_kitti.py
"""
import argparse
import shutil
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
SOURCE = Path('/media/mf/AUTODRIVING-4TB/7 NOV RINGROAD/datasetx/2026-04-02_route00')
FRAMES = ROOT / 'output/ringroad_selection/selected_frames.txt'
OUTDIR = ROOT / 'dataset_ringroad'
CALIB_SRC = ROOT / 'dataset_ugm/calib/000000.txt'   # cy=440, calib benar
TRAIN_RATIO = 0.8


def pcd_to_bin(pcd_path: Path, bin_path: Path):
    pcd = o3d.t.io.read_point_cloud(str(pcd_path))
    xyz = pcd.point['positions'].numpy()
    inten = pcd.point['intensity'].numpy()
    inten = (inten / 255.0).astype(np.float32)
    np.hstack([xyz, inten]).astype(np.float32).tofile(str(bin_path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_dir', default=str(SOURCE))
    ap.add_argument('--frames', default=str(FRAMES))
    ap.add_argument('--output_dir', default=str(OUTDIR))
    ap.add_argument('--train_ratio', type=float, default=TRAIN_RATIO)
    args = ap.parse_args()

    src, outdir = Path(args.source_dir), Path(args.output_dir)
    frame_ids = [l.strip() for l in open(args.frames) if l.strip()]
    print(f"\nConvert RINGROAD -> KITTI\n  Source : {src}\n  Frames : {len(frame_ids)}"
          f"\n  Output : {outdir}\n")

    for sub in ['velodyne', 'image_2', 'calib', 'label_2', 'ImageSets']:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    calib_txt = CALIB_SRC.read_text()
    mapping = ["# seq_id  timestamp_id"]

    for i, ts in enumerate(tqdm(frame_ids, desc='convert')):
        sid = f'{i:06d}'
        mapping.append(f'{sid} {ts}')

        pcd_to_bin(src / 'lidar/cld' / f'{ts}.pcd', outdir / 'velodyne' / f'{sid}.bin')

        img_dst = outdir / 'image_2' / f'{sid}.png'
        if img_dst.is_symlink() or img_dst.exists():
            img_dst.unlink()
        img_dst.symlink_to(src / 'camera/rgb' / f'{ts}.png')

        (outdir / 'calib' / f'{sid}.txt').write_text(calib_txt)

    (outdir / 'frame_mapping.txt').write_text('\n'.join(mapping) + '\n')

    # split train/val deterministik
    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(frame_ids))
    rng.shuffle(idx)
    n_train = int(len(frame_ids) * args.train_ratio)
    train = sorted(f'{i:06d}' for i in idx[:n_train])
    val = sorted(f'{i:06d}' for i in idx[n_train:])
    (outdir / 'ImageSets/train.txt').write_text('\n'.join(train) + '\n')
    (outdir / 'ImageSets/val.txt').write_text('\n'.join(val) + '\n')
    (outdir / 'ImageSets/trainval.txt').write_text(
        '\n'.join(sorted(train + val)) + '\n')

    print(f"\nSelesai: {len(train)} train, {len(val)} val")
    print(f"  velodyne/  : {len(frame_ids)} .bin (file nyata)")
    print(f"  image_2/   : {len(frame_ids)} .png (symlink ke sumber)")
    print(f"  calib/     : disalin dari {CALIB_SRC}")
