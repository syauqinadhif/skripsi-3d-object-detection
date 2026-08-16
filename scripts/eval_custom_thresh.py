#!/usr/bin/env python3
"""
Eval AP di threshold IoU CUSTOM (di luar default KITTI 0.7/0.5/0.25) — protokol sama eval_range
(≤25 m, in-FOV, GT v3). Menampilkan 3D DAN BEV di tiap threshold utk Car & Cyclist.
Default KITTI cuma Car@{0.7,0.5}, Cyc@{0.5,0.25}; skrip ini menambah Car@0.35 & Cyc@0.7.

Usage:
  PYTHONPATH=OpenPCDet python scripts/eval_custom_thresh.py --dt <result.pkl> --gt <infos.pkl>
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
sys.path.insert(0, str(ROOT / 'scripts'))
from eval_range import build   # chdir ke OpenPCDet/tools + import kitti_eval
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

ap = argparse.ArgumentParser()
ap.add_argument('--dt', required=True)
ap.add_argument('--gt', required=True)
ap.add_argument('--max-range', type=float, default=25.0)
args = ap.parse_args()

gt = pickle.load(open(args.gt, 'rb'))
dt = pickle.load(open(args.dt, 'rb'))
ga, da = build(gt, dt, args.max_range, True)

# kelas KITTI eval: 0=Car, 2=Cyclist. Baris = set threshold (Car_thr, Cyc_thr).
current = [0, 2]
rows = [(0.70, 0.70), (0.50, 0.50), (0.25, 0.25)]   # Car: .7/.5/.25 ; Cyc: .7/.5/.25
mo = np.zeros((len(rows), 3, 6))
for i, (c, y) in enumerate(rows):
    mo[i, :, 0] = c   # Car (bbox/bev/3d)
    mo[i, :, 2] = y   # Cyclist
res = kitti_eval.do_eval(ga, da, current, mo[:, :, current])
bev = res[5]; d3 = res[6]   # [num_class(2), num_diff(3), num_set] ; moderate = diff idx 1
MOD = 1

def g(arr, cls_idx, set_idx): return arr[cls_idx, MOD, set_idx]

print(f"\n=== AP_R40 Moderate (≤{args.max_range:g} m, in-FOV) — dt={Path(args.dt).name} ===")
print(f"{'kelas / metrik':22s}{'3D':>9s}{'BEV':>9s}")
labels = [('Car',   0, [('@0.70',0),('@0.50',1),('@0.25 (NEW)',2)]),
          ('Cyclist',1, [('@0.70 (NEW)',0),('@0.50',1),('@0.25',2)])]
for name, ci, sets in labels:
    for lab, si in sets:
        print(f"  {name+' '+lab:20s}{g(d3,ci,si):>9.2f}{g(bev,ci,si):>9.2f}")
