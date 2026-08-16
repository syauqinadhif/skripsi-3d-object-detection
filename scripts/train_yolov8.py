"""
train_yolov8.py — Training YOLOv8 di dataset UGM

Model: YOLOv8m (medium) — balance antara speed dan accuracy
Input: dataset_ugm_yolo/ (output dari prepare_yolo_dataset.py)

Usage:
  conda run -n openpcdet python scripts/train_yolov8.py
  conda run -n openpcdet python scripts/train_yolov8.py --model yolov8s --epochs 50
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
YAML    = PROJECT / 'dataset_ugm_yolo' / 'ugm.yaml'
OUTDIR  = PROJECT / 'output' / 'yolov8'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   default='yolov8m.pt', help='YOLOv8 model variant')
    p.add_argument('--epochs',  type=int, default=50)
    p.add_argument('--imgsz',   type=int, default=640)
    p.add_argument('--batch',   type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--name',    default='ugm_pseudo')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f'\nTrain YOLOv8 — UGM Dataset')
    print(f'  Model  : {args.model}')
    print(f'  Epochs : {args.epochs}')
    print(f'  Dataset: {YAML}')
    print(f'  Output : {OUTDIR / args.name}\n')

    model = YOLO(args.model)

    results = model.train(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(OUTDIR),
        name=args.name,
        device=0,
        exist_ok=True,
        # Augmentation
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        degrees=0.0,
        # Logging
        plots=True,
        save=True,
        save_period=10,
        val=True,
    )

    print(f'\nTraining selesai.')
    print(f'Best checkpoint: {OUTDIR}/{args.name}/weights/best.pt')
    print(f'mAP50: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')
    print(f'mAP50-95: {results.results_dict.get("metrics/mAP50-95(B)", "N/A")}')
