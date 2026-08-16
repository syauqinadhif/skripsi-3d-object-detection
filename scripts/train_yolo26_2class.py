#!/usr/bin/env python3
"""
Latih YOLO26m 2-KELAS (Car, Cyclist) — pembanding v8m_2cls untuk skripsi (YOLO-only & fusi).
HP IDENTIK train_yolo_2class.py (imgsz 1280, 100 ep, batch 8, patience 30) → satu-satunya beda =
arsitektur (yolo26m vs yolov8m). Dataset SAMA (dataset_ugm_yolo_2cls).
"""
from ultralytics import YOLO

ROOT = '/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'

m = YOLO(f'{ROOT}/yolo26m.pt')
m.train(data=f'{ROOT}/dataset_ugm_yolo_2cls/ugm2.yaml',
        epochs=100, imgsz=1280, batch=8, workers=4, patience=30,
        project=f'{ROOT}/output/yolo26_2cls', name='y26m_2cls',
        exist_ok=True, verbose=True)
