#!/usr/bin/env python3
"""
Bangun dataset YOLO dari label manual v3 (kalibrasi v3: intrinsik rosbag + ekstrinsik PnP).

Beda dari prepare_yolo_dataset_manual.py (versi lama):
  - **Tidak memproyeksikan ulang.** Versi lama membaca field 3D dari label lalu memproyeksikan
    sendiri dengan P2 + hack `CY_FIX = 399`. Hack itu sekarang USANG: cy=399 dulu ditambal
    untuk mengompensasi kalibrasi yang salah, dan proyeksinya memakai aproksimasi frame kamera
    (asumsi box hanya berotasi thd sumbu y kamera) yang patah saat kamera ber-pitch 2,18 deg.
  - Bbox diambil LANGSUNG dari file label, yang sudah dihitung converter v3 di frame LiDAR
    (8 sudut dibangun di tempat box memang terdefinisi, lalu diproyeksikan). Terukur lebih
    akurat: median IoU thd deteksi COCO YOLO 0,251 -> 0,675.
  - val memakai `label_2_val_manual_v3` (78 frame, GT bersih), bukan v2.

Hanya objek IN-FOV yang ditulis; objek di luar FOV kamera bbox-nya 0000 di label (LiDAR 360 deg
vs kamera ~67 deg, jadi ini mayoritas) dan memang tak terlihat kamera.
"""
import shutil
from pathlib import Path

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
IMG_SRC = ROOT / 'dataset_ugm/image_2'
OUT = ROOT / 'dataset_ugm_yolo_v3'
IMG_W, IMG_H = 1280, 720
CLASSES = ['Car', 'Pedestrian', 'Cyclist']

SPLITS = {
    'train': ('dataset_ugm/label_2_train_manual', 'dataset_ugm/ImageSets/train_manual150.txt'),
    'val':   ('dataset_ugm/label_2_val_manual_v3', 'dataset_ugm/ImageSets/val_manual_v3_78.txt'),
}


def build(split, label_dir, split_file):
    ids = (ROOT / split_file).read_text().split()
    img_dir = OUT / 'images' / split
    lbl_dir = OUT / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_obj = n_skip = 0
    per_cls = {c: 0 for c in CLASSES}
    for fid in ids:
        src_img = IMG_SRC / f'{fid}.png'
        if not src_img.exists():
            continue
        dst = img_dir / f'{fid}.png'
        if not dst.exists():
            shutil.copy(src_img, dst)

        lines = []
        for l in (ROOT / label_dir / f'{fid}.txt').read_text().splitlines():
            t = l.split()
            if len(t) < 15 or t[0] not in CLASSES:
                continue
            x1, y1, x2, y2 = map(float, t[4:8])
            if (x2 - x1) < 1 or (y2 - y1) < 1:      # bbox 0000 = luar FOV kamera
                n_skip += 1
                continue
            lines.append(f"{CLASSES.index(t[0])} "
                         f"{((x1+x2)/2)/IMG_W:.6f} {((y1+y2)/2)/IMG_H:.6f} "
                         f"{(x2-x1)/IMG_W:.6f} {(y2-y1)/IMG_H:.6f}")
            per_cls[t[0]] += 1
            n_obj += 1
        (lbl_dir / f'{fid}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print(f"  {split:5}: {len(ids)} frame, {n_obj} objek in-FOV "
          f"({per_cls}) — {n_skip} dilewati krn luar FOV")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, (ld, sf) in SPLITS.items():
        build(split, ld, sf)
    (OUT / 'ugm.yaml').write_text(
        "# YOLOv8 dataset UGM — label manual v3, bbox 2D diambil langsung dari label\n"
        "# (kalibrasi v3: fx=fy=959.155 cx=650.105 cy=362.808 + ekstrinsik tr_pnp_gt_v3)\n"
        f"path: {OUT}\ntrain: images/train\nval: images/val\n\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"\nDataset -> {OUT}\nyaml    -> {OUT/'ugm.yaml'}")


if __name__ == '__main__':
    main()
