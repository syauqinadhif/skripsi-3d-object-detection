# Model Final

Tiga bobot model final untuk pipeline fusi **PointPillars + YOLO26 + CLOCs**.

| Berkas | Peran | Path asal di project |
|--------|-------|----------------------|
| `pointpillars_finetune_kitti_ep120.pth` | Detektor 3D LiDAR (PointPillars, fine-tune dari KITTI, epoch 120) | `OpenPCDet/output/kitti_models/pointpillar_ugm_m300_v3/finetune_kitti_m300/ckpt/checkpoint_epoch_120.pth` |
| `yolo26m_2cls_best.pt` | Detektor 2D kamera (YOLO26m, 2 kelas: Car, Cyclist) | `output/yolo26_2cls/y26m_2cls/weights/best.pt` |
| `clocs_ft120_yolo26_ep20.pt` | Jaringan fusi CLOCs (epoch 20) | `CLOCs_LQS/log/pointpillar_ugm/ft120_y26/20.pt` |

Kelas: **Car**, **Cyclist** (Pedestrian di-exclude dari studi).

Untuk menjalankan inferensi, letakkan/rujuk bobot ini sesuai path yang dipakai skrip di `scripts/`
dan `deploy/`. Detail langkah demo ada di `../DEMO_INFERENCE.md`.
