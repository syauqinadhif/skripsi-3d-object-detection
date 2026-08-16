#!/usr/bin/env python3
"""Latih YOLOv8 pada label MANUAL dgn bbox 2D calib benar (cy=399).
Pengganti train_yolov8.py yang memakai dataset pseudo-label ber-calib salah."""
from ultralytics import YOLO
ROOT = '/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi'
m = YOLO('yolov8m.pt')
m.train(data=f'{ROOT}/dataset_ugm_yolo_manual/ugm.yaml',
        epochs=100, imgsz=1280, batch=8, workers=4, patience=30,
        project=f'{ROOT}/output/yolo_manual', name='v8m_cy399',
        exist_ok=True, verbose=True)
