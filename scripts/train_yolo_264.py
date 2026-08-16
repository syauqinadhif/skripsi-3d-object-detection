from ultralytics import YOLO
R='/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi'
YOLO('yolov8m.pt').train(data=f'{R}/dataset_ugm_yolo_264/ugm.yaml', epochs=100, imgsz=1280,
    batch=8, workers=4, patience=30, project=f'{R}/output/yolo_manual', name='v8m_264', exist_ok=True)
