"""
Generate 2D detection results using YOLO26 for CLOCs fusion.
conf=0.0 agar semua kandidat masuk ke CLOCs (CLOCs yang akan filter).

YOLO26 default mode (end2end=True): max 300 box per frame, sudah NMS-free.
CLOCs hardcoded cap di 200 box/frame → 300 → 200 aman, tidak overflow.

Output format identik dengan generate_2d_detections.py (YOLOv8) — KITTI-style txt:
  class trunc occ alpha x1 y1 x2 y2 h w l x y z ry score
"""
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import argparse

# COCO class ID → KITTI class name (sama dengan YOLOv8 — YOLO26 juga dilatih di COCO)
COCO_TO_KITTI = {
    0: 'Pedestrian',
    1: 'Cyclist',    # bicycle
    2: 'Car',
    3: 'Cyclist',    # motorcycle
    5: 'Car',        # bus
    7: 'Car',        # truck
}

def generate_2d_detections(image_dir, output_dir, conf=0.0, frame_list=None, model_path='yolo26m.pt'):
    model = YOLO(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if frame_list:
        with open(frame_list) as f:
            frame_ids = [line.strip() for line in f if line.strip()]
        img_paths = sorted([Path(image_dir) / f"{fid}.png" for fid in frame_ids])
    else:
        img_paths = sorted(Path(image_dir).glob("*.png"))

    print(f"Processing {len(img_paths)} images...")
    for i, img_path in enumerate(img_paths):
        if i % 500 == 0:
            print(f"  {i}/{len(img_paths)}")

        results = model(str(img_path), conf=conf, verbose=False)[0]
        frame_id = img_path.stem
        lines = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_TO_KITTI:
                continue
            kitti_class = COCO_TO_KITTI[cls_id]
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Format KITTI: class trunc occ alpha x1 y1 x2 y2 h w l x y z ry score
            line = (f"{kitti_class} -1 -1 -10 "
                    f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                    f"-1 -1 -1 -1000 -1000 -1000 -10 {score:.4f}")
            lines.append(line)

        with open(Path(output_dir) / f"{frame_id}.txt", 'w') as f:
            f.write('\n'.join(lines))

    print(f"Done! Saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--conf', type=float, default=0.0)
    parser.add_argument('--frame_list', default=None, help='txt file with frame IDs')
    parser.add_argument('--model_path', default='yolo26m.pt', help='YOLO26 model path')
    args = parser.parse_args()
    generate_2d_detections(args.image_dir, args.output_dir, args.conf, args.frame_list, args.model_path)
