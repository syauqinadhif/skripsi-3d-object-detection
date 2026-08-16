"""
convert_pkl_to_pt.py
Converts OpenPCDet result.pkl (pre-NMS, loose NMS) to individual per-frame .pt files
for CLOCs training.

Why: CLOCs needs pre-NMS candidates (~500/frame, raw logit scores).
     The previous training used post-NMS results (~10-20/frame, sigmoid scores) — wrong.

Usage (called from fix_and_retrain_clocs.sh, or manually):
    python scripts/convert_pkl_to_pt.py \
        --val_pkl  <path to loose_nms val result.pkl> \
        --train_pkl <path to loose_nms train result.pkl> \
        --output_dir <CLOCs_LQS/data/clocs_data/3D>
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_pkl(pkl_path: str) -> list:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_and_save(data: list, output_dir: Path, split_name: str):
    """Save each frame dict as a single-element list .pt file: [dict]."""
    output_dir.mkdir(parents=True, exist_ok=True)

    score_min_all, score_max_all, count_all = [], [], []

    for frame_dict in tqdm(data, desc=f"Converting {split_name}"):
        frame_id = str(frame_dict["frame_id"])  # e.g. '000001'
        scores = frame_dict["score"]
        count_all.append(len(scores))
        score_min_all.append(float(scores.min()))
        score_max_all.append(float(scores.max()))

        out_path = output_dir / f"{frame_id}.pt"
        torch.save([frame_dict], str(out_path))

    print(f"  [{split_name}] {len(data)} frames saved to {output_dir}")
    print(
        f"  [{split_name}] candidates/frame — "
        f"min={min(count_all)}, max={max(count_all)}, "
        f"avg={np.mean(count_all):.1f}"
    )
    print(
        f"  [{split_name}] score range — "
        f"[{min(score_min_all):.4f}, {max(score_max_all):.4f}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_pkl", required=True, help="Path to val loose_nms result.pkl")
    parser.add_argument("--train_pkl", required=True, help="Path to train loose_nms result.pkl")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output dir — CLOCs_LQS/data/clocs_data/3D (will overwrite existing files)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"\nLoading val pkl: {args.val_pkl}")
    val_data = load_pkl(args.val_pkl)
    print(f"  -> {len(val_data)} val frames, sample 0: {len(val_data[0]['score'])} candidates")

    print(f"\nLoading train pkl: {args.train_pkl}")
    train_data = load_pkl(args.train_pkl)
    print(f"  -> {len(train_data)} train frames, sample 0: {len(train_data[0]['score'])} candidates")

    # Check for overlap / duplicates between train and val splits
    val_ids = {str(d["frame_id"]) for d in val_data}
    train_ids = {str(d["frame_id"]) for d in train_data}
    overlap = val_ids & train_ids
    if overlap:
        print(f"\nWARNING: {len(overlap)} frame IDs appear in both splits: {sorted(overlap)[:5]}...")

    total = len(val_ids) + len(train_ids) - len(overlap)
    print(f"\nTotal unique frames to write: {total}")
    print(f"Output dir: {output_dir}\n")

    convert_and_save(val_data, output_dir, split_name="val")
    convert_and_save(train_data, output_dir, split_name="train")

    print("\nDone. Verify a sample:")
    sample_file = output_dir / sorted(val_ids)[0]
    sample_file = output_dir / f"{sorted(val_ids)[0]}.pt"
    if sample_file.exists():
        d = torch.load(str(sample_file), weights_only=False)
        print(f"  {sample_file.name}: {len(d[0]['score'])} candidates, "
              f"score [{d[0]['score'].min():.4f}, {d[0]['score'].max():.4f}]")


if __name__ == "__main__":
    main()
