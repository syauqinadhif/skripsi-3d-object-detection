"""
Prepare val set frames for Supervisely 3D annotation upload.

Copies val set images + point clouds + calib to a staging folder:
    dataset_ugm/supervisely_upload/
        val/
            images/   ← .png files
            velodyne/ ← .bin files
            calib/    ← .txt files

After labeling in Supervisely and exporting to KITTI format,
replace dataset_ugm/label_2/ files with the Supervisely exports.
"""

import shutil
from pathlib import Path

VAL_INDEX   = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/CLOCs_LQS/data/clocs_data/index/ugm_val.txt")
DATASET_DIR = Path("/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/dataset_ugm")
OUT_DIR     = DATASET_DIR / "supervisely_upload" / "val"

def main():
    # Read val frame IDs
    frame_ids = VAL_INDEX.read_text().strip().splitlines()
    print(f"Val frames: {len(frame_ids)}")

    # Create output dirs
    for sub in ["images", "velodyne", "calib"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    copied = {"images": 0, "velodyne": 0, "calib": 0}
    missing = []

    for fid in frame_ids:
        img_src  = DATASET_DIR / "image_2"  / f"{fid}.png"
        vel_src  = DATASET_DIR / "velodyne" / f"{fid}.bin"
        cal_src  = DATASET_DIR / "calib"    / f"{fid}.txt"

        for src, dst_dir, key in [
            (img_src,  OUT_DIR / "images",   "images"),
            (vel_src,  OUT_DIR / "velodyne", "velodyne"),
            (cal_src,  OUT_DIR / "calib",    "calib"),
        ]:
            if src.exists():
                shutil.copy2(src, dst_dir / src.name)
                copied[key] += 1
            else:
                missing.append(str(src))

    print(f"Copied: {copied}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    else:
        print("All files copied successfully.")
    print(f"\nOutput: {OUT_DIR}")

if __name__ == "__main__":
    main()
