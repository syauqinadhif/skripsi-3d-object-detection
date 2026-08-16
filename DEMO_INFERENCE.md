# Panduan Demo Inferensi Fusi CLOCs (PC + Jetson AGX Orin)

Panduan langkah demi langkah untuk demo inferensi model fusi **PointPillars + YOLO26 + CLOCs**.
Semua script dan data sudah disiapkan dan **sudah diuji jalan** pada 2026-08-16.

Ada dua demo terpisah:

| Demo | Perangkat | Keluaran | Cakupan |
|------|-----------|----------|---------|
| **A (utama)** | PC (RTX 4080) | **citra per-frame** kamera + BEV | frame pilihan |
| **A (opsional)** | PC | video mp4 sekuens berjalan | frame 142–205 |
| **B** | Jetson AGX Orin | **citra per-frame** + ringkasan latency | 100 frame (000142–000241) |

---

## Demo A (utama) — Inferensi satu frame di PC

Menjalankan pipeline fusi penuh pada **satu frame** lalu menyimpan dua citra: pandangan kamera
dengan 3D bbox, dan bird-eye view LiDAR. Cocok untuk ditampilkan/di-screenshot saat sidang.

```bash
cd /media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi
conda activate openpcdet

python scripts/demo_clocs_ugm_yolo26.py --frame_id 000160 --live --score_thresh 0.20
```

Keluaran:
- `output/demo_ugm_yolo26/000160_camera.jpg` — citra kamera + 3D bbox (hijau = Car, oranye = Cyclist).
- `output/demo_ugm_yolo26/000160_bev.jpg` — bird-eye view titik LiDAR + kotak.

Ganti `--frame_id` untuk frame lain. `--live` wajib (menghitung ulang, bukan hasil pra-komputasi).
Naikkan `--score_thresh` bila masih ada kotak skor rendah yang mengganggu (default 0,20).
Label mengikuti kelas asli PointPillars: Car (hijau), Cyclist termasuk motor (oranye); Pedestrian di-skip.

### Frame terkurasi (sudah di-render, siap pakai di `output/demo_ugm_yolo26/`)

**Deteksi bagus** (objek dekat/menengah terdeteksi rapi):

| Frame | Isi |
|-------|-----|
| **000160** | Parkiran penuh, hampir semua mobil terdeteksi kiri-kanan (paling meyakinkan) |
| **000150** | Deret mobil parkir terdeteksi rapi, pejalan kaki benar tidak dikotaki |
| **000190** | Deret mobil (hijau) + motor (oranye/Cyclist) terdeteksi dengan label benar |

**Objek jauh / kecil tidak terdeteksi** (untuk menunjukkan keterbatasan jangkauan LiDAR):

| Frame | Isi |
|-------|-----|
| **000205** | Motor dekat terdeteksi tajam, tetapi kendaraan di ujung jalan (terlihat kamera) tanpa kotak 3D; BEV memperlihatkan titik sangat jarang di atas ~40 m |
| **000210** | Objek dekat/menengah terdeteksi; BEV titik menipis drastis di atas ~40 m, kendaraan ujung jalan tak terdeteksi |
| **000200** | Cyclist jauh ~40 m hanya sedikit titik; bus dekat (bukan kelas) tidak dikotaki |
| **000158** | Kendaraan jauh di bawah gerbang bangunan tidak terdeteksi |

> **Poin naratif keterbatasan LiDAR.** Pada frame 205/210, bandingkan citra kamera (objek jauh masih
> terlihat jelas) dengan BEV (titik LiDAR habis di ~40–50 m). Objek yang jauh tidak memperoleh kotak
> 3D bukan karena model gagal, melainkan karena kerapatan titik LiDAR pada jarak itu terlalu rendah
> untuk membentuk sebuah objek. Ini bukti visual yang kuat untuk bagian keterbatasan penelitian.

---

## Demo A (opsional) — Video inferensi sekuens di PC

> **Rentang frame video dipilih otomatis** (`scripts/scan_frames.py`) sebagai bagian sekuens dengan
> deteksi paling bersih: **142–205**. Di luar rentang ini (mis. 220+) muncul lebih banyak kotak
> Cyclist palsu pada semak/tiang karena frame ini di luar distribusi data validasi.

> Model: PointPillars fine-tune KITTI ep120 + YOLO26 2-kelas + CLOCs ft120_y26.
> Kelas yang digambar: **Car** (hijau) dan **Cyclist** (oranye). Pedestrian di-skip sesuai batasan studi.

Model dimuat **sekali**, lalu seluruh frame pada rentang diinferensikan berturut-turut dan
dirangkai menjadi satu video. Tiap frame komposit: **kiri** = citra kamera + 3D bbox, **kanan** =
bird-eye view (titik LiDAR berwarna ketinggian + kotak + grid jarak 10 m).

### Langkah

```bash
# 1. masuk folder project + aktifkan environment
cd /media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi
conda activate openpcdet

# 2. jalankan (default sudah rentang terbersih 142–205, 10 FPS, thr 0.15)
python scripts/demo_video_fusion.py

#    atau eksplisit / rentang lain:
# python scripts/demo_video_fusion.py --start 142 --end 205 --fps 10 --thr 0.15 --out output/demo_fusion.mp4

# 3. putar hasilnya
xdg-open output/demo_fusion.mp4
```

### Opsi yang bisa diubah saat demo

| Argumen | Arti | Default |
|---------|------|---------|
| `--start` / `--end` | rentang frame (end eksklusif); sekuens 000000–000499 | 142 / 205 |
| `--fps` | frame rate video keluaran | 10 |
| `--thr` | ambang skor fusi untuk digambar (skala CLOCs terkompresi: 0.09–0.37) | 0.15 |
| `--out` | lokasi file mp4 | output/demo_fusion.mp4 |

Contoh seluruh 500 frame:

```bash
python scripts/demo_video_fusion.py --start 0 --end 500 --fps 12 --out output/demo_full.mp4
```

> **Kecepatan.** Setelah frame pertama (warmup ~0.8 s), inferensi PC berjalan ~20 FPS (≈45 ms/frame),
> jadi praktis realtime. Video keluaran diputar pada FPS yang Anda set (`--fps`).
>
> **Catatan ambang.** Skor CLOCs tidak terkalibrasi (maksimum ~0.37, median ~0.11). Default 0.15
> sudah dipilih agar bersih pada rentang 142–205. Bila masih ada kotak palsu, naikkan `--thr`;
> bila deteksi benar hilang, turunkan.

---

## Demo B — Inferensi 100 frame di Jetson AGX Orin

Pipeline **penuh berjalan di Jetson**: engine TensorRT PointPillars + YOLO26 (ultralytics) +
CLOCs fuse + NMS. Keluaran berupa citra kamera ber-anotasi per frame plus ringkasan latency
per tahap. Voxelisasi LiDAR dilakukan di PC (butuh pcdet) lalu di-transfer; ini langkah
prapemrosesan, bukan bagian inferensi.

**Data sudah di-transfer ke Jetson** (`~/pp_deploy/`): `voxels_demo100.npz` (frame 000142–000241, 342 MB)
+ `images/` (100 png, 84 MB) + `run_demo_jetson.py`. Total **426 MB** (storage Jetson sisa 11 GB, cukup).

### Langkah (di terminal PC, SSH ke Jetson)

```bash
# 1. SSH ke Jetson (sesuaikan IP; password tanya pengelola perangkat)
ssh mf@<IP_JETSON>

# 2. di Jetson: masuk folder deploy
cd ~/pp_deploy

# 3a. jalankan demo FP16 (cepat) — 100 frame, default thr 0.15
python3 run_demo_jetson.py --prec fp16

# 3b. atau FP32 (presisi penuh)
python3 run_demo_jetson.py --prec fp32
```

Keluaran:
- `~/pp_deploy/demo_out/<fid>.jpg` — 100 citra hasil deteksi.
- Ringkasan latency di terminal, contoh (FP16, hasil uji):

```
=== Ringkasan latency Jetson (fp16, 100 frame; rata-rata tanpa warmup) ===
  PointPillars (TensorRT) : ~52 ms   (termasuk transfer host + decode; engine murni ~10.7 ms)
  YOLO26 (ultralytics)    : ~171 ms
  CLOCs fuse + NMS        : ~54 ms
  TOTAL per frame         : ~319 ms  (~3.1 FPS pipeline penuh)
```

> Angka engine murni FP32 14,30 ms / FP16 10,72 ms (Tabel 5.7 naskah) diukur terpisah lewat
> `benchmark.sh`; angka di atas adalah **pipeline penuh** termasuk YOLO, fusi, dan render.

### Opsi

| Argumen | Arti | Default |
|---------|------|---------|
| `--prec` | `fp16` atau `fp32` | fp16 |
| `--thr` | ambang skor fusi untuk digambar | 0.15 |
| `--nms_iou` | IoU NMS di bidang BEV (kecil = lebih agresif membuang duplikat) | 0.1 |
| `--limit` | batasi jumlah frame (0 = semua 100); mis. `--limit 20` untuk uji cepat | 0 |
| `--outdir` | folder citra keluaran | demo_out |

> Frame yang di-transfer (000142–000241) mencakup rentang terbersih 142–205 di awal. Untuk hanya
> menampilkan bagian terbersih, gunakan `--limit 63`.

### (Opsional) Rangkai citra Jetson menjadi video + tarik ke PC

```bash
# di Jetson: rangkai jadi mp4 (bila ffmpeg tersedia)
cd ~/pp_deploy
ffmpeg -framerate 10 -pattern_type glob -i 'demo_out/*.jpg' -c:v libx264 -pix_fmt yuv420p demo_jetson.mp4

# di PC: tarik hasil (citra + video) dari Jetson
cd /media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi
rsync -rz -e ssh mf@<IP_JETSON>:~/pp_deploy/demo_out/ output/demo_jetson_frames/
rsync -z  -e ssh mf@<IP_JETSON>:~/pp_deploy/demo_jetson.mp4 output/
```

Bila `ffmpeg` tidak ada di Jetson, cukup tarik folder `demo_out/` lalu rangkai di PC:

```bash
rsync -rz -e ssh mf@<IP_JETSON>:~/pp_deploy/demo_out/ output/demo_jetson_frames/
python - <<'PY'
import cv2, glob
fs = sorted(glob.glob('output/demo_jetson_frames/*.jpg'))
h, w = cv2.imread(fs[0]).shape[:2]
vw = cv2.VideoWriter('output/demo_jetson.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
for f in fs: vw.write(cv2.imread(f))
vw.release(); print('output/demo_jetson.mp4', len(fs), 'frame')
PY
```

---

## Ringkasan file

| File | Lokasi | Fungsi |
|------|--------|--------|
| `scripts/demo_clocs_ugm_yolo26.py` | PC | **Demo A utama**: inferensi satu frame (kamera + BEV) |
| `output/demo_ugm_yolo26/` | PC | Frame terkurasi yang sudah di-render (siap pakai) |
| `scripts/demo_video_fusion.py` | PC | Demo A opsional: inferensi video sekuens |
| `scripts/scan_frames.py` | PC | Pemilih rentang frame terbersih (sudah dijalankan) |
| `deploy/prep_demo100.py` | PC | Voxelisasi 100 frame (sudah dijalankan) |
| `deploy/demo100/` | PC | voxels_demo100.npz + images/ (sumber transfer) |
| `deploy/jetson/run_demo_jetson.py` | PC (asal) / `~/pp_deploy/` (Jetson) | Demo B: inferensi Jetson |
| `~/pp_deploy/voxels_demo100.npz`, `images/` | Jetson | data 100 frame (sudah di-transfer) |

## Prasyarat yang sudah terpenuhi

- **PC**: conda env `openpcdet`, model PP ep120, YOLO26 best.pt, CLOCs ft120_y26 — semua terverifikasi.
- **Jetson**: engine `pp_fp16.engine`/`pp_fp32.engine`, `best.pt`, `clocs_net.pt`, `anchors.npy`,
  `calib.txt`, `pp_decode.py`, `clocs_fuse.py`, `tool/`, torch 2.3 + TensorRT 8.6.2 + torchvision +
  ultralytics + cv2 — semua ada dan terverifikasi.

## Troubleshooting

- **IP Jetson berubah** (ganti jaringan/hotspot): sesuaikan IP Jetson pada perintah SSH/rsync.
- **Password SSH** Jetson: tanya pengelola perangkat. Ketik langsung saat diminta.
- **Kotak palsu berlebih**: naikkan `--thr`. **Deteksi benar hilang**: turunkan `--thr`.
- **Frame pertama lambat** (YOLO ~3 s, numba kompilasi): itu warmup; sudah dikecualikan dari statistik latency.
- **Demo B ingin cepat dicoba**: tambahkan `--limit 20`.
