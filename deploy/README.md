# Deployment PointPillars → Jetson AGX Orin

Runbook export & deploy detektor PointPillars (LiDAR) ke Jetson AGX Orin via TensorRT.

## Model yang di-deploy
**PP m300 v0.16 FINETUNE-KITTI (`finetune_kitti_m300/ckpt/checkpoint_epoch_120.pth`)** — MODEL TERBAIK
& deployable: Car 3D@0.7 = 50,98 (vs scratch 45,31), Cyclist 3D@0.5 = 78,94. Voxel 0,16 → **cocok
params CUDA-PointPillars** (RANGE [0,-39.68,-3,69.12,39.68,1], VOXEL 0.16, 3 kelas, anchor KITTI).
ONNX: `models/pointpillar_ft120.onnx` (terverifikasi). (Scratch m300@100 `pointpillar_m300.onnx`
masih ada sbg pembanding, jangan dipakai deploy.)

⚠️ **vox010 (voxel 0.10) TIDAK dipakai untuk deploy** meski akurasi Cyclist lebih baik: range/voxel
custom-nya perlu modifikasi `params.h` CUDA-PointPillars + rebuild C++ (riskan). Trade-off didokumentasikan
pada catatan eksperimen. Kalau mau deploy vox010, sesuaikan `params.h` & anchor di CUDA-PP dulu.

## Status
| Tahap | Status | Di mana |
|---|---|---|
| Train PP (finetune-KITTI ep120) | ✅ | x86 (RTX 4080) |
| Export ONNX | ✅ `models/pointpillar_ft120.onnx` | x86 |
| Verifikasi numerik ONNX vs PyTorch | ✅ (Δ~1e-2, numerik CPU-vs-CUDA) | x86 |
| Build TensorRT engine (FP32/FP16) | ✅ SELESAI | Jetson AGX Orin |
| Benchmark latency/FPS/power | ✅ SELESAI | Jetson AGX Orin |
| Konsistensi akurasi FP32 vs FP16 | ✅ SELESAI (AP eksplisit terdekode) | Jetson AGX Orin |
| **FUSI PENUH (PP+YOLO26+CLOCs) di Jetson** | ✅ SELESAI (fusi jujur 56,85) | Jetson AGX Orin |

## ✅ HASIL DEPLOYMENT (Jetson AGX Orin — 2026-08-15)
Lingkungan: JetPack 6 / L4T R36.3.0, TensorRT 8.6.2, CUDA 12.2, mode daya MAXN. Engine dibangun
langsung di Jetson via trtexec dari `pointpillar_ft120.onnx` (input tetap 10000×32×4).

**Benchmark kecepatan, memori, dan daya:**
| Presisi | Latency GPU (ms) | Throughput (FPS) | Memori GPU (MiB) | Daya GPU_SOC (W) | Daya board total (W) | Ukuran engine |
|---|---|---|---|---|---|---|
| FP32 | 14,30 | 68,4 | ~273 | 24,4 | 35,5 | 82 MB |
| FP16 | 10,72 | 90,6 | ~273 | 19,6 | 29,5 | 44 MB |

FP16 vs FP32: **1,32× lebih cepat, daya board ~17% lebih hemat (35,5→29,5 W)**, ukuran engine separuh.

**Konsistensi akurasi (output jaringan pada frame val nyata):**
| Output | \|nilai\| maks | mean Δ FP16–FP32 | maks Δ (outlier sel kosong) | FP32 vs PyTorch (maks) |
|---|---|---|---|---|
| cls_preds | 41,67 | 0,052 (~0,1%) | 0,82 | 0,155 |
| box_preds | 8,73 | 0,005 (~0,06%) | 1,11 | 0,018 |
| dir_cls_preds | 18,87 | 0,047 (~0,25%) | 0,95 | 0,065 |

**FUSI PENUH di Jetson (PP engine + YOLO26 + CLOCs). AP terdekode, 78 frame, box_utils jujur, 3D/BEV:**
| Kelas & ambang | Workstation | Jetson FP32 | Jetson FP16 |
|---|---|---|---|
| Car @0,70 | 56,85 / 70,89 | **56,85 / 70,89** | **56,60 / 70,57** |
| Car @0,50 | 71,11 / 74,95 | 71,11 / 74,95 | 70,92 / 74,70 |
| Car @0,25 | 78,53 / 80,55 | 78,53 / 80,55 | 78,43 / 80,48 |
| Cyclist @0,70 | 27,91 / 49,73 | 27,91 / 49,73 | 20,91 / 45,95 |
| Cyclist @0,50 | 78,82 / 84,84 | 78,82 / 84,84 | 76,19 / 84,68 |
| Cyclist @0,25 | 84,84 / 84,84 | 84,84 / 84,84 | 84,68 / 84,68 |

**Baseline PP-only (box_utils, kandidat IDENTIK): Car@0,7 = 50,98 → fusi +5,87 (56,85). JUJUR.**

**Kesimpulan konsistensi akurasi:**
- **FP32 fusi Jetson = workstation, PERSIS (Δ ≈ 0,00)** → ONNX→TensorRT lossless, bahkan untuk fusi penuh.
- **FP16: Car nyaris terjaga (Δ −0,25), Cyclist turun** (@0,5 −2,63; @0,7 −7,00) → objek kecil sensitif kuantisasi.

⚠️ **DUA catatan penting:**
1. Konversi eval WAJIB `pcdet.utils.box_utils.boxes3d_lidar_to_kitti_camera`, BUKAN `convert_3d_to_clocs`
   (yg beri inflasi tak cocok GT). Untuk INPUT fusi (bbox 2D), `clocs_fuse.boxes_lidar_to_camera` dipakai
   (konvensi sama saat CLOCs dilatih — benar).
2. Perbandingan PP vs fusi HARUS apple-to-apple: kandidat 3D IDENTIK, hanya skor beda (box terverifikasi sama).
   Fusi murni re-score → +5,87 Car@0,7 nyata.

Alur fusi: voxelisasi 78 frame (x86) → Jetson `run_fusion.py` (engine + `pp_decode.py` + `clocs_fuse.py` +
`clocs_net.pt` + 2D YOLO26 precomputed) → `resfu_{fp32,fp16}.pkl` → x86 NMS + box_utils + eval.
Komponen Jetson: `~/pp_deploy/{run_fusion.py, clocs_fuse.py, clocs_net.pt, tool/fusion.py, d2/}`.

Artefak di Jetson: `~/pp_deploy/{pp_fp32.engine, pp_fp16.engine, trt_infer.py, teg_*.txt, run_*.txt}`.
Skrip inferensi TensorRT (buffer torch, API v3): `~/pp_deploy/trt_infer.py`.

## Langkah 1 — Export ONNX (di x86, SUDAH dijalankan)
```bash
PYTHONPATH=OpenPCDet /home/mf/miniconda3/envs/openpcdet/bin/python deploy/export_pp_onnx.py \
  --ckpt OpenPCDet/output/kitti_models/pointpillar_ugm_m300_v3/ppbaseline_m300/ckpt/checkpoint_epoch_100.pth \
  --out models/pointpillar_m300.onnx
# verifikasi:
PYTHONPATH=OpenPCDet /home/mf/miniconda3/envs/openpcdet/bin/python deploy/verify_onnx.py \
  --onnx models/pointpillar_m300.onnx \
  --ckpt OpenPCDet/output/kitti_models/pointpillar_ugm_m300_v3/ppbaseline_m300/ckpt/checkpoint_epoch_100.pth
```
Wrapper (`PPExport`) membungkus model → forward(voxels, voxel_num, voxel_idxs) → raw head preds
(cls/box/dir), voxelisasi+decode+NMS dikerjakan sisi CUDA/C++. Ini menghindari friction versi
pcdet 846cf3e + torch 1.x yang diminta `tool/export_onnx.py` NVIDIA (kita pcdet 0.6.0 + torch 2.0.1).

Output ONNX: input `voxels[10000,32,4]`, `voxel_num[10000]`, `voxel_idxs[10000,4]`;
output `cls_preds[1,18,248,216]`, `box_preds[1,42,…]`, `dir_cls_preds[1,12,…]`.

## Langkah 2 — Kirim ke Jetson
```bash
scp models/pointpillar_m300.onnx deploy/jetson/*.sh <user>@<jetson-ip>:~/pp_deploy/
```

## Langkah 3 — Build engine (DI JETSON)
```bash
cd ~/pp_deploy
bash build_engine.sh pointpillar_m300.onnx .    # hasilkan _fp32.engine & _fp16.engine
```
Jika trtexec gagal karena **dim spatial dinamis** (dari ScatterND PointPillarScatter): jalankan
`tool/modify_onnx.py` dari CUDA-PointPillars pada ONNX ini dulu (ia mem-fixed-kan I/O sesuai C++),
ATAU tambahkan `--minShapes/--optShapes/--maxShapes` di trtexec. Lihat "Opsi B" di bawah.

## Langkah 4 — Benchmark (DI JETSON)
```bash
bash benchmark.sh pointpillar_m300_fp32.engine pointpillar_m300_fp16.engine 300
```
Set `nvpmodel -m 0` (MAXN) + `jetson_clocks` (script sudah melakukannya). Rekam power/mem via `tegrastats`.

### Tabel metrik skripsi
Sudah terisi — lihat bagian "✅ HASIL DEPLOYMENT" di atas (FP32: 14,30 ms / 68,4 FPS / 35,5 W;
FP16: 10,72 ms / 90,6 FPS / 29,5 W).

## Opsi B — Pipeline C++ CUDA-PointPillars penuh (end-to-end, paling representatif)
ONNX kita mengeluarkan raw head preds dgn NAMA I/O yang sama dgn NVIDIA (voxels/voxel_num/voxel_idxs
→ cls/box/dir), TAPI layout output NCHW mentah. Binary C++ `pointpillar` mengharapkan layout hasil
`tool/modify_onnx.py`. Untuk end-to-end penuh (voxelisasi+engine+NMS dlm CUDA):
```bash
# di Jetson, dalam repo CUDA-PointPillars:
python3 tool/modify_onnx.py   # sesuaikan input onnx kita → layout C++
. tool/environment.sh && mkdir build && cd build && cmake .. && make -j$(nproc)
sh ../tool/build_trt_engine.sh
./pointpillar ../data/ ../data/ --timer   # --timer cetak breakdown latency per tahap
```
⚠️ Kompatibilitas layout C++ perlu diverifikasi di Jetson — bila `modify_onnx.py` menolak graf kita,
alternatif paling aman = re-export via OpenPCDet 846cf3e resmi (Docker CUDA-PP) memakai checkpoint sama.

## Catatan: stack lengkap (PP + YOLO + CLOCs)
Deployment ini = **cabang LiDAR (PointPillars) saja** — milestone deployment inti penelitian ini.
Fusi penuh butuh juga: YOLOv8 2-kelas (Ultralytics `yolo export format=engine` → TensorRT, mudah)
+ CLOCs fusion net (kecil, bisa ONNX/TensorRT). Deploy fusi penuh = milestone terpisah/lanjutan.

## Artefak
- `deploy/export_pp_onnx.py` — exporter wrapper (tahan versi)
- `deploy/verify_onnx.py` — verifikasi numerik ONNX vs PyTorch
- `deploy/jetson/build_engine.sh` — build engine FP32/FP16 (trtexec)
- `deploy/jetson/benchmark.sh` — ukur latency/FPS/power (trtexec + tegrastats)
- `models/pointpillar_m300.onnx` — ONNX terverifikasi
