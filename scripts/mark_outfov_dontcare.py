#!/usr/bin/env python3
"""
Tandai objek di LUAR FOV kamera sebagai `DontCare` pada info pkl yang dipakai CLOCs.

Alasan: CLOCs = fusion kamera-LiDAR. Objek yang titik pusatnya tidak terproyeksi
ke dalam gambar (z<=0 atau di luar batas gambar) MUSTAHIL difusi karena tidak ada
kandidat 2D dari YOLO. Kalau objek-objek ini tetap dianggap GT valid:
  - CLOCs (yang belajar "3D box tanpa pasangan 2D = kemungkinan FP") menekan skornya
    -> objek nyata hilang -> recall turun -> mAP CLOCs anjlok.
Dengan menandainya DontCare, objek tsb dikeluarkan dari perhitungan recall/precision
eval (lihat clean_data() di kitti eval), tanpa mengubah label LiDAR PointPillars.

Kriteria out-of-FOV (per objek, koordinat kamera):
    z <= 0  ATAU  u=fx*x/z+cx di luar [0,W)  ATAU  v=fy*y/z+cy di luar [0,H)
P2, lebar/tinggi gambar diambil per-frame dari info['calib'] (fallback ke default).
"""
import argparse, pickle, shutil, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="CLOCs_LQS/data/clocs_data/info/ugm_infos_trainval.pkl",
                    help="info pkl yang dipakai CLOCs untuk train/eval")
    ap.add_argument("--img-w", type=int, default=1280)
    ap.add_argument("--img-h", type=int, default=720)
    ap.add_argument("--dry-run", action="store_true", help="hitung saja, jangan tulis")
    args = ap.parse_args()

    infos = pickle.load(open(args.pkl, "rb"))
    print(f"Loaded {len(infos)} frame dari {args.pkl}")

    tot = changed = 0
    per_class = {}
    for inf in infos:
        an = inf["annos"]
        # dtype lebar supaya "DontCare" (8 char) tidak terpotong oleh dtype
        # array asli yang pendek (mis. '<U3' untuk 'Car' -> "Don")
        names = np.array(an["name"], dtype="<U16")
        loc = np.array(an["location"], dtype=float)
        # kalibrasi per-frame jika ada
        P2 = np.array(inf.get("calib", {}).get("P2",
              [[699.4741851, 0, 630.32152421, 0],
               [0, 707.80363724, 440.0, 0],
               [0, 0, 1, 0], [0, 0, 0, 1]]))
        fx, cx, fy, cy = P2[0, 0], P2[0, 2], P2[1, 1], P2[1, 2]
        for i in range(len(names)):
            if names[i] == "DontCare":
                continue
            tot += 1
            x, y, z = loc[i]
            out = False
            if z <= 0:
                out = True
            else:
                u = fx * x / z + cx
                v = fy * y / z + cy
                if u < 0 or u >= args.img_w or v < 0 or v >= args.img_h:
                    out = True
            if out:
                per_class[names[i]] = per_class.get(names[i], 0) + 1
                names[i] = "DontCare"
                changed += 1
        an["name"] = names  # tulis balik array yang sudah diubah

    print(f"\nTotal objek non-DontCare : {tot}")
    print(f"Ditandai DontCare (out-FOV): {changed} ({changed/tot*100:.1f}%)")
    print("Rincian per kelas asli     :", per_class)

    if args.dry_run:
        print("\n[dry-run] tidak menulis file.")
        return

    bak = args.pkl + ".prefov_backup"
    if not os.path.exists(bak):
        shutil.copy(args.pkl, bak)
        print(f"\nBackup dibuat: {bak}")
    else:
        print(f"\nBackup sudah ada (tidak ditimpa): {bak}")
    pickle.dump(infos, open(args.pkl, "wb"))
    print(f"Tersimpan: {args.pkl}")


if __name__ == "__main__":
    main()
