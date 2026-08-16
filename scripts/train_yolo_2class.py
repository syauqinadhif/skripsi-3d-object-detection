#!/usr/bin/env python3
"""
Latih YOLOv8 2-KELAS (Car, Cyclist) — Pedestrian di-exclude.
HP IDENTIK train_yolo_v3.py (yolov8m, imgsz 1280, 100 ep, batch 8, patience 30)
supaya perbandingan 3-kelas vs 2-kelas apples-to-apples.
"""
from ultralytics import YOLO

ROOT = '/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'

m = YOLO('yolov8m.pt')
m.train(data=f'{ROOT}/dataset_ugm_yolo_2cls/ugm2.yaml',
        epochs=100, imgsz=1280, batch=8, workers=4, patience=30,
        project=f'{ROOT}/output/yolo_2cls', name='v8m_2cls',
        exist_ok=True, verbose=True)
