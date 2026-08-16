# Skripsi S1 — Deteksi Objek 3D LiDAR-Kamera (PointPillars + YOLO26 + CLOCs)

Implementasi deteksi objek 3D berbasis fusi kamera-LiDAR untuk lingkungan kampus UGM,
dengan deployment ke edge device **Jetson AGX Orin**.

## Pipeline

```
LiDAR  → PointPillars (OpenPCDet)  → kandidat 3D
Kamera → YOLO26 (Ultralytics)      → kandidat 2D
                    ↓
              CLOCs Late Fusion (re-scoring)
                    ↓
           3D Bounding Boxes final
```

Kelas: **Car**, **Cyclist** (Pedestrian di-exclude dari studi).

## Isi repositori

| Path | Isi |
|------|-----|
| `scripts/` | Skrip inferensi, demo, evaluasi, pembangkit figur, konversi data |
| `deploy/` | Ekspor ONNX, decoder portabel, fusi standalone, skrip Jetson (TensorRT) |
| `models_final/` | 3 bobot model final (PointPillars, YOLO26, CLOCs) — lihat `models_final/README.md` |
| `DEMO_INFERENCE.md` | Panduan demo inferensi di PC & Jetson |

## Tidak disertakan (lihat `.gitignore`)

Dataset (UGM/KITTI/Ringroad), paket anotasi CVAT, checkpoint selain model final,
engine TensorRT/ONNX, dan repo pihak ketiga (OpenPCDet, CLOCs, CLOCs_LQS, CUDA-PointPillars —
clone terpisah dari sumber aslinya).

## Referensi

OpenPCDet · CLOCs · Ultralytics YOLO · CUDA-PointPillars · KITTI 3D Object Detection Benchmark.
