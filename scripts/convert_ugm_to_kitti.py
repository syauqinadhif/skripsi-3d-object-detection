"""
convert_ugm_to_kitti.py — Convert dataset UGM ke format KITTI untuk OpenPCDet

Yang dilakukan:
  1. Baca selected_frames.txt (output dari select_ugm_frames.py)
  2. Convert lidar/cld/*.pcd → velodyne/*.bin (x,y,z,intensity float32)
  3. Copy camera/rgb/*.png → image_2/*.png
  4. Generate placeholder calib/*.txt (diisi nanti setelah dapat kalibrasi)
  5. Buat folder label_2/ kosong (siap untuk Supervisely export)
  6. Buat ImageSets/train.txt + val.txt (split 80/20)
  7. Simpan frame_mapping.txt (sequential_id ↔ timestamp)

Struktur output:
  dataset_ugm/
  ├── velodyne/          ← 000000.bin ... 000499.bin
  ├── image_2/           ← 000000.png ... 000499.png
  ├── calib/             ← 000000.txt ... 000499.txt  [PLACEHOLDER]
  ├── label_2/           ← kosong, diisi setelah Supervisely labeling
  ├── ImageSets/
  │   ├── train.txt      ← 000000 ... 000399  (80%)
  │   └── val.txt        ← 000400 ... 000499  (20%)
  └── frame_mapping.txt  ← mapping sequential_id ↔ timestamp

Usage:
  conda run -n openpcdet python scripts/convert_ugm_to_kitti.py
  conda run -n openpcdet python scripts/convert_ugm_to_kitti.py --frames output/ugm_selection/selected_frames.txt
"""

import argparse
import shutil
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

SOURCE_DIR    = Path("/media/mf/AUTODRIVING-4TB/UGM Baru/autoriving-oskarnatan/datasetx/2026-02-09_route00")
FRAMES_TXT    = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/output/ugm_selection/selected_frames.txt")
OUTPUT_DIR    = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm")

TRAIN_RATIO   = 0.8   # 80% train, 20% val

# ── Calib — KITTI frame 000000 (dipakai sementara, saran dosen) ───────────────
# Sumber: KITTI/calib/000000.txt
# Untuk training LiDAR-only (PointPillars), nilai ini TIDAK berpengaruh pada
# hasil training maupun evaluasi 3D AP.
# WAJIB DIGANTI dengan nilai real UGM sebelum training CLOCs fusion!

CALIB_P0  = "7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
CALIB_P1  = "7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"
CALIB_P2  = "7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03"
CALIB_P3  = "7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03"
CALIB_R0  = "9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01"
CALIB_TR  = "6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01"


def make_calib_txt():
    """Generate KITTI calib string (nilai dari KITTI frame 000000, saran dosen)."""
    lines = [
        "# KITTI calib — dataset UGM",
        "# SUMBER: KITTI/calib/000000.txt (saran dosen, sementara untuk LiDAR-only training)",
        "# WAJIB DIGANTI dengan nilai real UGM sebelum training CLOCs fusion!",
        f"P0: {CALIB_P0}",
        f"P1: {CALIB_P1}",
        f"P2: {CALIB_P2}",
        f"P3: {CALIB_P3}",
        f"R0_rect: {CALIB_R0}",
        f"Tr_velo_to_cam: {CALIB_TR}",
    ]
    return '\n'.join(lines) + '\n'


# ── PCD → BIN ─────────────────────────────────────────────────────────────────

def pcd_to_bin(pcd_path: Path, bin_path: Path):
    """Convert VLP-32 .pcd (binary_compressed) ke .bin (x,y,z,intensity float32)."""
    pcd = o3d.t.io.read_point_cloud(str(pcd_path))
    xyz       = pcd.point['positions'].numpy()          # (N, 3)
    intensity = pcd.point['intensity'].numpy()          # (N, 1), range 1-255
    intensity_norm = (intensity / 255.0).astype(np.float32)  # normalize ke [0,1]
    points = np.hstack([xyz, intensity_norm]).astype(np.float32)  # (N, 4)
    points.tofile(str(bin_path))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default=str(SOURCE_DIR))
    parser.add_argument('--frames',     type=str, default=str(FRAMES_TXT),
                        help='Path ke selected_frames.txt')
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO)
    return parser.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    src    = Path(args.source_dir)
    outdir = Path(args.output_dir)

    # ── Baca frame list ──
    with open(args.frames) as f:
        frame_ids = [line.strip() for line in f if line.strip()]
    print(f"\nConvert UGM → KITTI format")
    print(f"  Source    : {src}")
    print(f"  Frames    : {len(frame_ids)}")
    print(f"  Output    : {outdir}\n")

    # ── Buat folder struktur ──
    for sub in ['velodyne', 'image_2', 'calib', 'label_2', 'ImageSets']:
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    # ── Split train / val ──
    n_total  = len(frame_ids)
    n_train  = int(n_total * args.train_ratio)
    n_val    = n_total - n_train
    # Shuffle deterministik (seed fixed) sebelum split
    rng      = np.random.default_rng(seed=42)
    indices  = rng.permutation(n_total)
    train_idx = sorted(indices[:n_train])
    val_idx   = sorted(indices[n_train:])

    # ── Konversi frame ──
    mapping_lines = []
    errors = []

    print("  Converting frames...")
    for seq_id, frame_id in enumerate(tqdm(frame_ids, desc="  Converting")):
        seq_str = f"{seq_id:06d}"

        pcd_path = src / 'lidar' / 'cld' / f"{frame_id}.pcd"
        png_path = src / 'camera' / 'rgb' / f"{frame_id}.png"
        bin_out  = outdir / 'velodyne' / f"{seq_str}.bin"
        img_out  = outdir / 'image_2'  / f"{seq_str}.png"
        cal_out  = outdir / 'calib'    / f"{seq_str}.txt"

        # LiDAR: PCD → BIN
        try:
            pcd_to_bin(pcd_path, bin_out)
        except Exception as e:
            errors.append(f"[LiDAR] {frame_id}: {e}")
            continue

        # Camera: copy PNG
        try:
            shutil.copy2(str(png_path), str(img_out))
        except Exception as e:
            errors.append(f"[Image] {frame_id}: {e}")

        # Calib: placeholder
        cal_out.write_text(make_calib_txt())

        mapping_lines.append(f"{seq_str} {frame_id}")

    # ── Tulis ImageSets ──
    train_ids_str = [f"{i:06d}" for i in train_idx]
    val_ids_str   = [f"{i:06d}" for i in val_idx]

    (outdir / 'ImageSets' / 'train.txt').write_text('\n'.join(train_ids_str) + '\n')
    (outdir / 'ImageSets' / 'val.txt').write_text('\n'.join(val_ids_str) + '\n')
    (outdir / 'ImageSets' / 'trainval.txt').write_text(
        '\n'.join(train_ids_str + val_ids_str) + '\n')

    # ── Tulis frame mapping ──
    (outdir / 'frame_mapping.txt').write_text(
        "# seq_id  timestamp_id\n" + '\n'.join(mapping_lines) + '\n')

    # ── Summary ──
    n_bin = len(list((outdir / 'velodyne').glob('*.bin')))
    n_img = len(list((outdir / 'image_2').glob('*.png')))

    print(f"\n  ── Hasil ──")
    print(f"  velodyne/  : {n_bin} file .bin")
    print(f"  image_2/   : {n_img} file .png")
    print(f"  calib/     : {len(mapping_lines)} file .txt  [PLACEHOLDER]")
    print(f"  label_2/   : kosong (siap untuk Supervisely export)")
    print(f"  ImageSets/ : train={n_train}, val={n_val}")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")

    print(f"""
  ── Langkah Berikutnya ──
  1. Upload image_2/ ke Supervisely untuk labeling 3D bounding box
  2. Export label dari Supervisely ke format KITTI → simpan ke label_2/
  3. Setelah dapat kalibrasi dari dosen → update calib/ files
  4. Generate dataset info:
       conda run -n openpcdet python -m pcdet.datasets.kitti.kitti_dataset \\
           create_kitti_infos tools/cfgs/dataset_configs/ugm_dataset.yaml

  CATATAN: calib/ berisi nilai KITTI frame 000000 (saran dosen, cukup untuk LiDAR-only).
           WAJIB diganti dengan nilai real UGM sebelum training CLOCs fusion!
""")
