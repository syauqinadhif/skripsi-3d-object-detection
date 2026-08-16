#!/usr/bin/env python3
"""
Siapkan paket labeling 2D untuk CVAT: gambar + PRA-ANOTASI (format YOLO 1.1).

Tujuan: menaikkan recall detektor 2D dari 0,245 -> >=0,65, ambang di mana fusi CLOCs
mengalahkan LiDAR-saja (lihat ablasi di memory gt_val_bias_z.md).

Pra-anotasi dibuat dgn ENSEMBLE dua detektor lalu NMS, supaya RECALL setinggi mungkin —
menghapus box salah jauh lebih cepat daripada menggambar box yang hilang:
  1. YOLOv8m COCO pretrained (kelas COCO dipetakan ke kelas UGM)
  2. YOLOv8m hasil fine-tune 150 frame UGM (output/yolo_manual/v8m_cy399)

⚠️ Frame VAL (val_manual_v2_78) SENGAJA DIKECUALIKAN — melabelinya lalu memakainya untuk
melatih YOLO = kebocoran data, dan semua angka fusi jadi tidak sah.

Output (siap diunggah ke CVAT, format "YOLO 1.1"):
  <out>/obj.names            <- daftar kelas
  <out>/obj_train_data/*.txt <- pra-anotasi (kelas x_c y_c w h ternormalisasi)
  <out>/obj_train_data/*.png <- gambar
  <out>/train.txt            <- daftar file
  <out>.zip
"""
import argparse, shutil
from pathlib import Path
import numpy as np

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
COCO_MAP = {2: 'Car', 7: 'Car', 5: 'Car', 0: 'Pedestrian', 3: 'Cyclist', 1: 'Cyclist'}


def nms(boxes, scores, thr=0.45):
    if not len(boxes):
        return []
    b = np.asarray(boxes, float); s = np.asarray(scores, float)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, help='file berisi daftar frame id')
    ap.add_argument('--image-dir', default='dataset_ugm/image_2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conf', type=float, default=0.15, help='ambang rendah = recall tinggi')
    ap.add_argument('--exclude', default='dataset_ugm/ImageSets/val_manual_v2_78.txt',
                    help='frame yang HARUS dikecualikan (anti-bocor)')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--prefix', default='', help="awalan nama file — WAJIB utk sumber non-UGM "
                    "(id frame ringroad BERTABRAKAN dgn UGM; tanpa ini file saling menimpa)")
    ap.add_argument('--weights', default='output/yolo_manual/v8m_200/weights/best.pt')
    args = ap.parse_args()

    from ultralytics import YOLO
    ids = (ROOT / args.frames).read_text().split()
    # Pengaman anti-bocor berbasis ID. HANYA sah bila frame berasal dari dataset yang SAMA
    # dengan daftar exclude. ID ringroad kebetulan bertabrakan dgn ID UGM (8 frame), jadi kalau
    # --prefix dipakai (= sumber lain), pengaman ini dilewati supaya tak membuang frame sah.
    if args.exclude and not args.prefix:
        ex = set((ROOT / args.exclude).read_text().split())
        n0 = len(ids); ids = [i for i in ids if i not in ex]
        print(f"anti-bocor: {n0 - len(ids)} frame val dikecualikan")
    elif args.prefix:
        print(f"anti-bocor: dilewati (sumber lain, prefix '{args.prefix}') — "
              f"pastikan sendiri sumber ini tidak memuat frame val UGM")
    if args.limit:
        ids = ids[:args.limit]

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    data = out / 'obj_train_data'
    if out.exists():
        shutil.rmtree(out)
    data.mkdir(parents=True)

    coco = YOLO('yolov8m.pt')
    ft_path = ROOT / args.weights
    ft = YOLO(str(ft_path)) if ft_path.exists() else None

    img_dir = ROOT / args.image_dir
    n_box = 0; listing = []
    for fid in ids:
        img = img_dir / f'{fid}.png'
        if not img.exists():
            continue
        W, H = 1280, 720
        boxes, scores, labels = [], [], []
        for r in coco(str(img), conf=args.conf, verbose=False):
            for b in r.boxes:
                c = COCO_MAP.get(int(b.cls[0]))
                if c:
                    boxes.append(b.xyxy[0].cpu().numpy()); scores.append(float(b.conf[0])); labels.append(c)
        if ft is not None:
            for r in ft(str(img), conf=args.conf, verbose=False):
                for b in r.boxes:
                    ci = int(b.cls[0])
                    if ci < len(CLASSES):
                        boxes.append(b.xyxy[0].cpu().numpy()); scores.append(float(b.conf[0])); labels.append(CLASSES[ci])
        kept = {}
        for c in CLASSES:                       # NMS per kelas
            idx = [i for i, l in enumerate(labels) if l == c]
            if not idx:
                continue
            kept[c] = [boxes[idx[k]] for k in nms([boxes[i] for i in idx], [scores[i] for i in idx])]

        # Pengendara motor: COCO memberi 'person' + 'motorcycle' terpisah, tapi konvensi KITTI
        # Cyclist = pengendara + kendaraan JADI SATU box. Buang Pedestrian yang sebagian besar
        # tubuhnya berada di dalam box Cyclist, supaya user tak perlu menghapusnya manual.
        cyc = kept.get('Cyclist', [])
        if cyc and kept.get('Pedestrian'):
            keep_ped = []
            for pb in kept['Pedestrian']:
                pa = max((pb[2] - pb[0]) * (pb[3] - pb[1]), 1e-9)
                inside = 0.0
                for cb in cyc:
                    ix = max(0, min(pb[2], cb[2]) - max(pb[0], cb[0]))
                    iy = max(0, min(pb[3], cb[3]) - max(pb[1], cb[1]))
                    inside = max(inside, ix * iy / pa)
                if inside < 0.55:
                    keep_ped.append(pb)
            kept['Pedestrian'] = keep_ped

        lines = []
        for c in CLASSES:
            for bx in kept.get(c, []):
                x1, y1, x2, y2 = bx
                x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(f"{CLASSES.index(c)} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} "
                             f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                n_box += 1
        name = f'{args.prefix}{fid}'
        (data / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        shutil.copy(img, data / f'{name}.png')
        listing.append(f'obj_train_data/{name}.png')

    (out / 'obj.names').write_text('\n'.join(CLASSES) + '\n')
    (out / 'train.txt').write_text('\n'.join(listing) + '\n')
    (out / 'obj.data').write_text(
        f"classes = {len(CLASSES)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n")
    shutil.make_archive(str(out), 'zip', str(out))
    print(f"{len(listing)} frame, {n_box} pra-anotasi ({n_box/max(len(listing),1):.1f}/frame)")
    print(f"-> {out}.zip  (CVAT: buat task baru, Upload annotations -> format 'YOLO 1.1')")


if __name__ == '__main__':
    main()
