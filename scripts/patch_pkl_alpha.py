#!/usr/bin/env python3
"""
patch_pkl_alpha.py

Patches alpha values in kitti_infos PKL files from the fixed label_2 txt files.
Run this after fix_ugm_2d_labels.py.
"""

import pickle
import numpy as np
from pathlib import Path

LABEL_DIR = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/label_2")

PKL_FILES = [
    "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/kitti_infos_train.pkl",
    "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/kitti_infos_val.pkl",
    "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm/kitti_infos_trainval.pkl",
    "/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/CLOCs_LQS/data/clocs_data/info/ugm_infos_trainval.pkl",
]


def read_alpha_from_label(frame_id: str) -> np.ndarray:
    """Read alpha values for all objects in a label file."""
    lf = LABEL_DIR / f"{int(frame_id):06d}.txt"
    if not lf.exists():
        return np.array([])
    alphas = []
    for line in lf.read_text().strip().split('\n'):
        parts = line.split()
        if len(parts) >= 15:
            alphas.append(float(parts[3]))
    return np.array(alphas, dtype=np.float32)


def patch_pkl(pkl_path: str):
    pkl_path = Path(pkl_path)
    print(f"\nPatching: {pkl_path.name}")

    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    n_patched = 0
    n_mismatch = 0

    for info in infos:
        frame_id = info['point_cloud']['lidar_idx']
        alphas_from_label = read_alpha_from_label(frame_id)

        ann = info.get('annos', {})
        if 'alpha' not in ann or len(alphas_from_label) == 0:
            continue

        old_alpha = ann['alpha']

        if len(alphas_from_label) != len(old_alpha):
            n_mismatch += 1
            continue

        ann['alpha'] = alphas_from_label
        n_patched += 1

    print(f"  Patched {n_patched} frames, {n_mismatch} mismatches (skipped)")

    # Backup then overwrite
    backup = pkl_path.with_suffix('.pkl.alpha_backup')
    if not backup.exists():
        import shutil
        shutil.copy2(pkl_path, backup)
        print(f"  Backed up → {backup.name}")

    with open(pkl_path, 'wb') as f:
        pickle.dump(infos, f)
    print(f"  Saved.")


def verify(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)
    ann = infos[0]['annos']
    print(f"  alpha sample (first frame): {ann['alpha'][:4]}")


if __name__ == '__main__':
    for p in PKL_FILES:
        patch_pkl(p)

    print("\n=== Verification ===")
    for p in PKL_FILES:
        print(Path(p).name)
        verify(p)
