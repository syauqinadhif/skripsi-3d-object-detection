#!/usr/bin/env python3
"""
build_combined_dataset.py — Gabungkan UGM (label lama) + RINGROAD (pseudolabel)
jadi satu dataset KITTI lewat symlink per-file. Tidak ada file besar yang disalin.

Peta frame id:
  UGM      000000-000499  -> id TETAP (supaya val80 manual tetap cocok id-nya)
  RINGROAD 000000-000999  -> id + 1000 = 001000-001999

Split:
  train = UGM train (394) + RINGROAD semua (kecuali frame tanpa objek)
  val   = UGM val80 manual (80)  <- label diambil dari label_2_val_manual

Usage:
  python scripts/build_combined_dataset.py
"""
from pathlib import Path
import shutil

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
UGM = ROOT / 'dataset_ugm'
RR = ROOT / 'dataset_ringroad'
OUT = ROOT / 'dataset_combined'
RR_OFFSET = 1000

for sub in ['velodyne', 'image_2', 'calib', 'label_2', 'ImageSets']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)


def link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def add(src_dir: Path, sid: str, did: str, label_src: Path):
    link(src_dir / 'velodyne' / f'{sid}.bin', OUT / 'velodyne' / f'{did}.bin')
    link(src_dir / 'image_2' / f'{sid}.png', OUT / 'image_2' / f'{did}.png')
    link(src_dir / 'calib' / f'{sid}.txt', OUT / 'calib' / f'{did}.txt')
    link(label_src, OUT / 'label_2' / f'{did}.txt')


# ── UGM: train (label_2 lama) ────────────────────────────────────────────────
ugm_train = [l.strip() for l in open(UGM / 'ImageSets/train.txt') if l.strip()]
for sid in ugm_train:
    add(UGM, sid, sid, UGM / 'label_2' / f'{sid}.txt')

# ── UGM: val80 manual (label_2_val_manual) ───────────────────────────────────
ugm_val = [l.strip() for l in open(UGM / 'ImageSets/val_manual.txt') if l.strip()]
for sid in ugm_val:
    add(UGM, sid, sid, UGM / 'label_2_val_manual' / f'{sid}.txt')

# ── RINGROAD: semua, kecuali frame tanpa objek ───────────────────────────────
rr_ids, skipped = [], 0
for sid in sorted(p.stem for p in (RR / 'velodyne').glob('*.bin')):
    lf = RR / 'label_2' / f'{sid}.txt'
    if not lf.exists() or lf.stat().st_size == 0:
        skipped += 1
        continue
    did = f'{int(sid) + RR_OFFSET:06d}'
    add(RR, sid, did, lf)
    rr_ids.append(did)

train_ids = sorted(ugm_train + rr_ids)
val_ids = sorted(ugm_val)
(OUT / 'ImageSets/train.txt').write_text('\n'.join(train_ids) + '\n')
(OUT / 'ImageSets/val.txt').write_text('\n'.join(val_ids) + '\n')
(OUT / 'ImageSets/trainval.txt').write_text('\n'.join(sorted(train_ids + val_ids)) + '\n')
(OUT / 'ImageSets/test.txt').write_text('\n'.join(val_ids) + '\n')

# ── training/ testing/ (pola KittiDataset) ───────────────────────────────────
for split in ['training', 'testing']:
    d = OUT / split
    d.mkdir(exist_ok=True)
    for sub in ['velodyne', 'image_2', 'calib', 'label_2']:
        link(OUT / sub, d / sub)

print(f"dataset_combined siap di {OUT}")
print(f"  train : {len(train_ids)}  (UGM {len(ugm_train)} + ringroad {len(rr_ids)})")
print(f"  val   : {len(val_ids)}  (UGM val80 manual)")
print(f"  ringroad dilewati (tanpa objek): {skipped}")
