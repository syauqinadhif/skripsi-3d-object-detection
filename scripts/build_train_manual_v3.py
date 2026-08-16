#!/usr/bin/env python3
"""
Bangun dataset root v3: train = 150 frame label manual, val = 78 frame GT v3.
Semua label & calib memakai kalibrasi v3 (intrinsik rosbag + ekstrinsik tr_pnp_gt_v3).

Beda dari build_train_manual_variants.py (versi lama, sekarang BEKU):
  - val memakai `label_2_val_manual_v3` (78 frame, GT bersih) — bukan `label_2_val_manual`
    (v1, 80 frame, masih bias-z melayang).
  - hanya membangun varian m150. Varian mix394 tidak dibangun ulang: eksperimen sudah
    menunjukkan 150 frame bersih > 394 frame teracuni, dan pseudolabel `label_2` masih
    ber-calib lama (harus diregenerasi dari model dulu kalau mau dipakai).
  - calib di-symlink ke dataset_ugm/calib (v3). Dataset lama sudah dibekukan dgn salinan
    nyata calib_pre_v3_backup supaya tetap konsisten-sendiri & reproducible.
"""
import shutil
from pathlib import Path

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
SRC = ROOT / 'dataset_ugm'
L_TRAIN = SRC / 'label_2_train_manual'
L_VAL = SRC / 'label_2_val_manual_v3'
NAME = 'dataset_ugm_m150_v3'

train150 = sorted(open(SRC / 'ImageSets/train_manual150.txt').read().split())
val78 = sorted(open(SRC / 'ImageSets/val_manual_v3_78.txt').read().split())
assert len(train150) == 150 and len(val78) == 78
assert not (set(train150) & set(val78)), 'bocor train/val!'

# Frame tanpa objek -> file label kosong bikin get_infos() OpenPCDet crash saat concatenate
# bbox. Konvensi KITTI: satu baris DontCare. num_obj tetap 0, gt_boxes_lidar (0,7), dan
# evaluator mengabaikan DontCare -> frame tetap jadi negatif asli.
DONTCARE = 'DontCare -1 -1 -10 0.00 0.00 0.00 0.00 -1 -1 -1 -1000 -1000 -1000 -10\n'


def copy_label(src, dst):
    txt = src.read_text()
    dst.write_text(txt if txt.strip() else DONTCARE)


def main():
    root = ROOT / NAME
    if root.exists():
        shutil.rmtree(root)
    (root / 'ImageSets').mkdir(parents=True)
    (root / 'training').mkdir()
    lab = root / 'training/label_2'
    lab.mkdir()

    for sub in ['velodyne', 'image_2', 'calib']:
        (root / 'training' / sub).symlink_to(SRC / sub)
        (root / sub).symlink_to(SRC / sub)

    for fid in train150:
        copy_label(L_TRAIN / f'{fid}.txt', lab / f'{fid}.txt')
    for fid in val78:
        copy_label(L_VAL / f'{fid}.txt', lab / f'{fid}.txt')

    (root / 'ImageSets/train.txt').write_text('\n'.join(train150) + '\n')
    (root / 'ImageSets/val.txt').write_text('\n'.join(val78) + '\n')
    (root / 'ImageSets/trainval.txt').write_text('\n'.join(train150 + val78) + '\n')
    (root / 'ImageSets/test.txt').write_text('\n'.join(val78) + '\n')

    link = ROOT / 'OpenPCDet/data' / NAME
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(root)

    print(f'{NAME}: train={len(train150)} manual, val={len(val78)} (GT v3)')
    print(f'  -> {root}\n  -> symlink {link}')


if __name__ == '__main__':
    main()
