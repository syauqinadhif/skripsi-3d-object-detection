#!/usr/bin/env python3
"""
Bangun 2 dataset root untuk eksperimen "label manual vs pseudolabel" di data train UGM.

  A) dataset_ugm_m150   : train = 150 frame LABEL MANUAL saja
  B) dataset_ugm_mix394 : train = 150 manual + 244 pseudo (sisa train.txt)

Keduanya pakai val = 80 frame VAL MANUAL (label_2_val_manual) supaya eval saat training
langsung memakai GT jujur. Data mentah (velodyne/image_2/calib) di-symlink ke dataset_ugm;
cuma label_2 + ImageSets yang nyata dibuat. dataset_ugm ASLI tidak disentuh.
"""
import shutil
from pathlib import Path

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
SRC = ROOT / 'dataset_ugm'
L_MANUAL_TR = SRC / 'label_2_train_manual'
L_MANUAL_VAL = SRC / 'label_2_val_manual'
L_PSEUDO = SRC / 'label_2'

train_all = sorted(open(SRC / 'ImageSets/train.txt').read().split())
train150 = sorted(open(SRC / 'ImageSets/train_manual150.txt').read().split())
val80 = sorted(open(SRC / 'ImageSets/val_manual.txt').read().split())
pseudo244 = sorted(set(train_all) - set(train150))

assert len(train150) == 150 and len(pseudo244) == 244 and len(val80) == 80
assert not (set(train_all) & set(val80)), 'bocor train/val!'

# Frame tanpa objek (15 di train manual, 2 di val) = label file kosong -> get_infos OpenPCDet
# crash saat concatenate bbox. Konvensi KITTI untuk ini: satu baris DontCare. num_obj tetap 0,
# gt_boxes_lidar (0,7), dan evaluator mengabaikan DontCare -> frame tetap jadi negatif asli.
DONTCARE = 'DontCare -1 -1 -10 0.00 0.00 0.00 0.00 -1 -1 -1 -1000 -1000 -1000 -10\n'


def copy_label(src, dst):
    txt = src.read_text()
    dst.write_text(txt if txt.strip() else DONTCARE)


def build(name, train_ids):
    root = ROOT / name
    if root.exists():
        shutil.rmtree(root)
    (root / 'ImageSets').mkdir(parents=True)
    (root / 'training').mkdir()
    lab = root / 'training/label_2'
    lab.mkdir()

    for sub in ['velodyne', 'image_2', 'calib']:
        (root / 'training' / sub).symlink_to(SRC / sub)
        (root / sub).symlink_to(SRC / sub)

    n_man = n_pse = 0
    for fid in train_ids:
        if fid in set(train150):
            copy_label(L_MANUAL_TR / f'{fid}.txt', lab / f'{fid}.txt'); n_man += 1
        else:
            copy_label(L_PSEUDO / f'{fid}.txt', lab / f'{fid}.txt'); n_pse += 1
    for fid in val80:
        copy_label(L_MANUAL_VAL / f'{fid}.txt', lab / f'{fid}.txt')

    (root / 'ImageSets/train.txt').write_text('\n'.join(train_ids) + '\n')
    (root / 'ImageSets/val.txt').write_text('\n'.join(val80) + '\n')
    (root / 'ImageSets/trainval.txt').write_text('\n'.join(train_ids + val80) + '\n')
    (root / 'ImageSets/test.txt').write_text('\n'.join(val80) + '\n')

    # symlink ke OpenPCDet/data supaya bisa dipakai config (DATA_PATH: ../data/<name>)
    link = ROOT / 'OpenPCDet/data' / name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(root)

    print(f'{name:22} train={len(train_ids):3d} (manual {n_man}, pseudo {n_pse})  val={len(val80)}')


build('dataset_ugm_m150', train150)
build('dataset_ugm_mix394', sorted(train150 + pseudo244))
print('\nSelesai. Lanjut: create_infos_for.py untuk masing-masing root.')
