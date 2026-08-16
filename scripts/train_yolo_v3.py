#!/usr/bin/env python3
"""
Latih YOLOv8 pada dataset YOLO v3 (kalibrasi v3, bbox 2D diambil langsung dari label).

Hyperparameter SENGAJA dibuat identik dgn train_yolo_manual.py (run `v8m_cy399`) — yolov8m,
imgsz 1280, 100 epoch, batch 8, patience 30 — supaya perbandingan lama vs baru apples-to-apples
dan satu-satunya variabel yang berubah = kualitas kalibrasi/label.
"""
from ultralytics import YOLO

ROOT = '/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi'

m = YOLO('yolov8m.pt')
m.train(data=f'{ROOT}/dataset_ugm_yolo_v3/ugm.yaml',
        epochs=100, imgsz=1280, batch=8, workers=4, patience=30,
        project=f'{ROOT}/output/yolo_v3', name='v8m_calibv3',
        exist_ok=True, verbose=True)
