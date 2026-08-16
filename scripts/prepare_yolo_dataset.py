"""
prepare_yolo_dataset.py — Konversi dataset UGM ke format YOLO untuk YOLOv8

Sumber 2D bbox: kitti_infos_train/val.pkl (sudah di-proyeksi dari 3D via P2 KITTI)
Note: P2 adalah nilai KITTI frame 000000 (bukan kalibrasi UGM real).
      Proyeksi akan meleset dari posisi objek di gambar — ini disengaja untuk
      melihat seberapa jauh melesetnya saat CLOCs fusion.

Output:
  dataset_ugm_yolo/
  ├── images/train/  ← symlink ke dataset_ugm/image_2/*.png
  ├── images/val/    ← symlink ke dataset_ugm/image_2/*.png
  ├── labels/train/  ← YOLO format .txt (cls x_c y_c w h normalized)
  ├── labels/val/    ← YOLO format .txt
  └── ugm.yaml

Usage:
  conda run -n openpcdet python scripts/prepare_yolo_dataset.py
"""

import pickle
import numpy as np
from pathlib import Path

PROJECT  = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
UGM_DIR  = PROJECT / 'dataset_ugm'
YOLO_DIR = PROJECT / 'dataset_ugm_yolo'

IMG_W, IMG_H = 1280, 720

CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}


def kitti_bbox_to_yolo(x1, y1, x2, y2):
    """Convert [x1,y1,x2,y2] pixel coords to YOLO [x_c,y_c,w,h] normalized."""
    x_c = ((x1 + x2) / 2) / IMG_W
    y_c = ((y1 + y2) / 2) / IMG_H
    w   = (x2 - x1) / IMG_W
    h   = (y2 - y1) / IMG_H
    return x_c, y_c, w, h


def process_split(split, pkl_path, img_src_dir, out_img_dir, out_lbl_dir):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    n_imgs = 0
    n_boxes = 0
    n_skipped = 0

    for info in infos:
        sample_idx = info['image']['image_idx']   # e.g. '000007'
        img_src = img_src_dir / f'{sample_idx}.png'
        img_dst = out_img_dir / f'{sample_idx}.png'
        lbl_dst = out_lbl_dir / f'{sample_idx}.txt'

        # Symlink image (avoid copying 500 PNGs)
        if not img_dst.exists():
            img_dst.symlink_to(img_src.resolve())

        annos = info.get('annos', {})
        names = annos.get('name', [])
        bboxes = annos.get('bbox', [])

        lines = []
        for i, name in enumerate(names):
            cls_id = CLASS_MAP.get(name, -1)
            if cls_id == -1:
                n_skipped += 1
                continue
            x1, y1, x2, y2 = bboxes[i]
            w_box = x2 - x1
            h_box = y2 - y1
            # Skip degenerate boxes
            if w_box < 2 or h_box < 2:
                n_skipped += 1
                continue
            x_c, y_c, w_n, h_n = kitti_bbox_to_yolo(x1, y1, x2, y2)
            # Clamp to [0, 1]
            x_c  = np.clip(x_c,  0.0, 1.0)
            y_c  = np.clip(y_c,  0.0, 1.0)
            w_n  = np.clip(w_n,  0.0, 1.0)
            h_n  = np.clip(h_n,  0.0, 1.0)
            lines.append(f'{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}')
            n_boxes += 1

        lbl_dst.write_text('\n'.join(lines) + ('\n' if lines else ''))
        n_imgs += 1

    print(f'  {split}: {n_imgs} images, {n_boxes} boxes, {n_skipped} skipped')
    return n_imgs, n_boxes


def make_yaml(yolo_dir):
    content = f"""# YOLOv8 dataset config — UGM (pseudo-labels via KITTI calib)
# NOTE: 2D bbox dari proyeksi 3D pseudo-label menggunakan P2 KITTI (bukan P2 UGM real)

path: {yolo_dir}
train: images/train
val:   images/val

nc: 3
names:
  0: Car
  1: Pedestrian
  2: Cyclist
"""
    (yolo_dir / 'ugm.yaml').write_text(content)
    print(f'  YAML: {yolo_dir}/ugm.yaml')


if __name__ == '__main__':
    print('\nPrepare YOLO Dataset — UGM')
    print(f'  Source : {UGM_DIR}')
    print(f'  Output : {YOLO_DIR}\n')

    img_src = UGM_DIR / 'image_2'

    process_split(
        'train',
        UGM_DIR / 'kitti_infos_train.pkl',
        img_src,
        YOLO_DIR / 'images' / 'train',
        YOLO_DIR / 'labels' / 'train',
    )
    process_split(
        'val',
        UGM_DIR / 'kitti_infos_val.pkl',
        img_src,
        YOLO_DIR / 'images' / 'val',
        YOLO_DIR / 'labels' / 'val',
    )

    make_yaml(YOLO_DIR)

    print(f'\nDone. Dataset siap di: {YOLO_DIR}')
    print('Next: jalankan train_yolov8.py')
