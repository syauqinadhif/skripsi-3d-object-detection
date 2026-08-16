#!/usr/bin/env python3
"""
Eval dengan BATAS JARAK + DontCare — protokol evaluasi final UGM.

Kenapa ada: anotasi vertikal di luar ~25 m tidak bisa dipercaya dari sumber MANA PUN
(noise estimator tanah 0.29 m, scatter label manusia 0.63 m — lihat memory [[gt_val_bias_z]]).
Protokolnya: evaluasi DIBATASI pada objek <=25 m; GT & prediksi di luar itu tidak dinilai.

Tiga penyaring, berurutan:
  1. IN-FOV  : GT & prediksi di luar FOV kamera (bbox area <1) dibuang. Objek luar-FOV tak
               bisa di-match dan cuma jadi FP paksa (artefak eval, bukan kesalahan model).
  2. GT jauh : > --max-range ditandai 'DontCare' (bbox tetap) = dikeluarkan dari penilaian.
  3. Pred jauh: > --max-range dibuang, agar prediksi di zona tak-ternilai tidak dihitung
               sebagai FP di ruang kosong. INI yang melakukan hampir semua pekerjaan.

CATATAN JUJUR (diuji 2026-07-18, scratchpad/test_dontcare.py): di evaluator KITTI OpenPCDet,
menandai GT jauh sebagai 'DontCare' memberi hasil IDENTIK dengan menghapusnya — baik saat
prediksi jauh dibuang maupun dipertahankan. Mekanisme "DontCare menekan FP yang menimpanya"
TIDAK aktif di jalur kode ini. Jadi jangan mengklaim itu di metodologi skripsi; deskripsi yang
benar = "evaluasi dibatasi <=25 m, objek di luar itu dikecualikan dari penilaian".
Dampak protokol ini nyata & besar (m150 ep130, Car 3D@0.7): tanpa penanganan 10.28 -> dgn 16.37.

Contoh:
  python scripts/eval_range.py --dt <result.pkl> --gt <infos.pkl> --max-range 25
  python scripts/eval_range.py --dt <result.pkl> --max-range 25 --no-infov
"""
import argparse, os, pickle, sys
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'OpenPCDet', 'tools'))
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
KEYS = ['name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y']


def valid_bbox_mask(bbox):
    b = np.asarray(bbox).reshape(-1, 4)
    return ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)


def subset(anno, mask, keys):
    out = {}
    for k in keys:
        v = np.asarray(anno[k])
        out[k] = v[mask] if v.shape[0] == mask.shape[0] else v
    return out


def boxes_range(anno, lidar_key):
    """jarak horizontal dari LiDAR; pakai boxes_lidar kalau ada, jika tidak dari location kamera"""
    b = anno.get(lidar_key)
    if b is not None and len(b):
        b = np.asarray(b)
        return np.hypot(b[:, 0], b[:, 1])
    loc = np.asarray(anno['location'])
    return np.hypot(loc[:, 2], loc[:, 0]) if len(loc) else np.zeros(0)


def build(gt_infos, dt_list, max_range, infov):
    dtmap = {d['frame_id']: d for d in dt_list}
    gt_annos, dt_annos = [], []
    n_dc = n_drop = n_gt = n_dt = 0
    for info in gt_infos:
        fid = info['point_cloud']['lidar_idx']
        ga = info['annos']; d = dtmap[fid]

        gmask = valid_bbox_mask(ga['bbox']) if infov else np.ones(len(ga['name']), bool)
        dmask = valid_bbox_mask(d['bbox']) if infov else np.ones(len(d['name']), bool)

        grng = boxes_range(ga, 'gt_boxes_lidar')
        drng = boxes_range(d, 'boxes_lidar')
        if len(drng) == len(dmask):
            dmask = dmask & (drng <= max_range)

        g = subset(ga, gmask, KEYS); g['score'] = np.zeros(gmask.sum())
        g['name'] = np.asarray(g['name'], dtype='<U20')
        # GT jauh -> DontCare (diabaikan evaluator, TIDAK dihapus)
        if len(grng) == len(gmask) and gmask.sum():
            far = grng[gmask] > max_range
            if far.any():
                g['name'] = g['name'].copy()
                g['name'][far] = 'DontCare'
                n_dc += int(far.sum())
        n_drop += int((~dmask).sum()); n_gt += int((g['name'] != 'DontCare').sum())
        dd = subset(d, dmask, KEYS + ['score']); n_dt += len(dd['name'])
        gt_annos.append(g); dt_annos.append(dd)
    print(f"  GT dinilai={n_gt}  GT jadi DontCare (>{max_range:g} m)={n_dc}  "
          f"prediksi dipakai={n_dt}  prediksi dibuang={n_drop}")
    return gt_annos, dt_annos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', required=True, help='result.pkl hasil test.py')
    ap.add_argument('--gt', default='../data/ugm/kitti_infos_val80_manual.pkl')
    ap.add_argument('--max-range', type=float, default=25.0)
    ap.add_argument('--no-infov', action='store_true', help='jangan saring ke FOV kamera')
    args = ap.parse_args()

    gt = pickle.load(open(args.gt, 'rb'))
    dt = pickle.load(open(args.dt, 'rb'))
    infov = not args.no_infov
    print(f"\n########## EVAL <= {args.max_range:g} m"
          f"{' , IN-FOV' if infov else ''}  (GT jauh = DontCare) ##########")
    ga, da = build(gt, dt, args.max_range, infov)
    s, _ = kitti_eval.get_official_eval_result(ga, da, CLASSES)
    for line in s.splitlines():
        if 'AP' in line or line.strip().endswith(':'):
            print(line)


if __name__ == '__main__':
    main()
