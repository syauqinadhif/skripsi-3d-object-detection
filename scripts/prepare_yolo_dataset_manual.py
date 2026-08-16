#!/usr/bin/env python3
"""
Bangun dataset YOLO dari label MANUAL dgn bbox 2D calib BENAR (cy=440).

Kenapa ada (pengganti prepare_yolo_dataset.py): script lama memakai proyeksi pseudo-label
dengan P2 KITTI (cy=180) — komentarnya mengakui sendiri. Akibatnya YOLO belajar box yang
tergeser ~142 px ke atas dan recall 2D-nya 0.00 terhadap GT benar, sehingga cabang kamera
CLOCs tidak menyumbang apa pun. Lihat memory [[gt_val_bias_z]].

Sumber di sini = file label KITTI hasil scripts/convert_datumaro_to_kitti.py, yang bbox-nya
SUDAH diproyeksikan dengan P2 UGM benar:
  train : dataset_ugm/label_2_train_manual      (150 frame manual)
  val   : dataset_ugm/label_2_val_manual_v2     (78 frame, GT v2 terverifikasi)

Hanya objek IN-FOV (bbox valid) yang ditulis — objek luar FOV tak terlihat kamera.
"""
from pathlib import Path
import shutil
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
IMG_SRC = ROOT / 'dataset_ugm/image_2'
OUT = ROOT / 'dataset_ugm_yolo_manual'
IMG_W, IMG_H = 1280, 720
CLASSES = ['Car', 'Pedestrian', 'Cyclist']

# cy TERKOREKSI. cy=440 (dataset_ugm/calib) membuat proyeksi jatuh +41 px terlalu rendah,
# diukur objektif thd 344 deteksi YOLO COCO pretrained (independen calib) di 228 frame.
# Sebagian karena calib lama dikalibrasi thd label 3D yang MELAYANG; setelah z dibetulkan,
# box turun ~19 px lagi. Offset konstan lintas jarak (28/43/40 px @0-10/10-20/20-30 m) => murni cy.
CY_FIX = 399.0

SPLITS = {
    'train': ('dataset_ugm/label_2_train_manual', 'dataset_ugm/ImageSets/train_manual150.txt'),
    'val':   ('dataset_ugm/label_2_val_manual_v2', 'dataset_ugm/ImageSets/val_manual_v2_78.txt'),
}


def load_P2():
    d = {}
    for line in (ROOT / 'dataset_ugm/calib/000001.txt').read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = d['P2'].reshape(3, 4).copy()
    P2[1, 2] = CY_FIX                      # <-- koreksi
    return P2


P2 = load_P2()


def project_3d(x, y, z, h, w, l, ry):
    """proyeksikan kubus 3D (koordinat kamera KITTI, (x,y,z)=alas tengah) -> bbox 2D"""
    xs = [l/2, l/2, -l/2, -l/2] * 2
    ys = [0, 0, 0, 0, -h, -h, -h, -h]
    zs = [w/2, -w/2, -w/2, w/2] * 2
    c = np.array([xs, ys, zs])
    R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    cor = (R @ c).T + np.array([x, y, z])
    if (cor[:, 2] <= 0.1).any():
        return None
    pr = (P2 @ np.hstack([cor, np.ones((8, 1))]).T).T
    uv = pr[:, :2] / pr[:, 2:3]
    x1, y1 = uv.min(0); x2, y2 = uv.max(0)
    return (max(x1, 0), max(y1, 0), min(x2, IMG_W), min(y2, IMG_H))


def build(split, label_dir, split_file):
    ids = (ROOT / split_file).read_text().split()
    img_dir = OUT / 'images' / split
    lbl_dir = OUT / 'labels' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_obj = n_skip = 0
    for fid in ids:
        src_img = IMG_SRC / f'{fid}.png'
        if not src_img.exists():
            continue
        dst = img_dir / f'{fid}.png'
        if not dst.exists():
            shutil.copy(src_img, dst)

        lines = []
        lf = ROOT / label_dir / f'{fid}.txt'
        for l in lf.read_text().splitlines():
            t = l.split()
            if len(t) < 15 or t[0] not in CLASSES:
                continue
            if float(t[6]) - float(t[4]) < 1:       # luar FOV / bbox 0000 di label
                n_skip += 1
                continue
            h_, w_, l_ = map(float, t[8:11])
            xc_, yc_, zc_ = map(float, t[11:14])
            ry = float(t[14])
            box = project_3d(xc_, yc_, zc_, h_, w_, l_, ry)
            if box is None:
                n_skip += 1
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                n_skip += 1
                continue
            xc = ((x1 + x2) / 2) / IMG_W
            yc = ((y1 + y2) / 2) / IMG_H
            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H
            lines.append(f"{CLASSES.index(t[0])} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            n_obj += 1
        (lbl_dir / f'{fid}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print(f"  {split:5}: {len(ids)} frame, {n_obj} objek in-FOV ({n_skip} dilewati krn luar FOV)")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, (ld, sf) in SPLITS.items():
        build(split, ld, sf)
    (OUT / 'ugm.yaml').write_text(
        f"# YOLOv8 dataset UGM — label MANUAL, bbox 2D proyeksi P2 BENAR (cy=440)\n"
        f"path: {OUT}\ntrain: images/train\nval: images/val\n\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"\nDataset -> {OUT}\nyaml    -> {OUT/'ugm.yaml'}")


if __name__ == '__main__':
    main()
