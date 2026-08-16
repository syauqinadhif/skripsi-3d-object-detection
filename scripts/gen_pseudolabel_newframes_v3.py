"""
gen_pseudolabel_newframes_v3.py — Pra-anotasi CVAT untuk FRAME BARU (belum pernah dilabeli manual)
menggunakan model TERBAIK PP m150@130ep (pointpillar_ugm_m150_v3, checkpoint_epoch_130.pth).

Tujuan: menaikkan ceiling recall Car @IoU0.7 (34%) dengan menambah label 3D Car.
- Frame = 500 pool  −  (train_manual150 ∪ val_manual_v3_80)  = frame yang BELUM pernah dilabeli.
- Kelas: HANYA Car + Cyclist (Pedestrian di-DROP — sudah di-exclude dari studi).
- TTA flip-Y + NMS merge (per catatan lama: TTA +20–25% recall untuk pra-anotasi).
- Output: Sustech POINTS JSON (frame LiDAR, position/scale/rotation.z) → import langsung ke CVAT 3D.
- CSV hitung per-frame (Car/Cyc) + ringkasan sebaran-z untuk verifikasi tidak melayang.

CATATAN JUJUR (dari TODO/REKAP):
- Pseudolabel hanya hemat ~6% tenaga; tetap WAJIB scan FN + koreksi FP saat labeling.
- Model m150@130 TIDAK ber-bias-z (Car dz_med +0,036 m) → pseudo z relatif benar,
  tapi tetap dilaporkan di bawah untuk dicek mata.

Usage:
  /home/mf/miniconda3/envs/openpcdet/bin/python scripts/gen_pseudolabel_newframes_v3.py
  ... --score_thresh 0.3   (default; turunkan ke 0.2 untuk recall lebih tinggi)
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch

ROOT      = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
OPENPCDET = ROOT / 'OpenPCDet'
sys.path.insert(0, str(OPENPCDET))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from tqdm import tqdm

PP_CFG  = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar_ugm_m150_v3.yaml'
PP_CKPT = OPENPCDET / 'output/kitti_models/pointpillar_ugm_m150/ep130_itermatched/ckpt/checkpoint_epoch_130.pth'

VEL   = ROOT / 'dataset_ugm/velodyne'
ISETS = ROOT / 'dataset_ugm/ImageSets'
OUT   = ROOT / 'cvat_newframes_v3'          # label/<fid>.json + frames.txt + counts.csv

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']   # urutan label model (1-indexed)
KEEP        = {'Car', 'Cyclist'}                 # Pedestrian di-drop
NMS_IOU     = 0.1
SNAP_RANGE  = 25.0   # >ini estimator tanah tak andal → box TIDAK di-snap (biar tak ngaco)


# ── Estimasi tanah lokal (disalin dari snap_val_datumaro_z.py) ──────────────────
def plane_z(pts, x, y):
    if len(pts) < 30:
        return None
    P = pts; w = np.ones(len(P))
    for _ in range(3):
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        try:
            c = np.linalg.lstsq(A * w[:, None], P[:, 2] * w, rcond=None)[0]
        except Exception:
            return None
        r = P[:, 2] - A @ c
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-3
        w = 1 / (1 + (r / (2 * s)) ** 2)
    return float(c[0] * x + c[1] * y + c[2])


def ground_z(p, x, y, sz):
    """Rata-rata 2 estimator independen; None jika titik tanah tak cukup."""
    d = np.hypot(p[:, 0] - x, p[:, 1] - y)
    g1 = g2 = None
    r1 = p[(d > 2) & (d < 5)]
    if len(r1) >= 50:
        g1 = float(np.percentile(r1[:, 2], 10))
    r2 = p[(d > sz / 2 + 0.3) & (d < sz / 2 + 7.0)]
    if len(r2) >= 60:
        lo = r2[r2[:, 2] < np.percentile(r2[:, 2], 30)]
        g2 = plane_z(lo, x, y)
    gs = [g for g in (g1, g2) if g is not None]
    return float(np.mean(gs)) if gs else None


class MinimalDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, logger):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=root_path, logger=logger)
    def __len__(self):  return 0
    def __getitem__(self, i):  return {}


def build_model(logger):
    os.chdir(str(OPENPCDET / 'tools'))          # _BASE_CONFIG_ resolve relatif ke tools/
    cfg_from_yaml_file(str(PP_CFG), cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = 0.05   # floor rendah; ambang final di bawah
    ds = MinimalDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, OPENPCDET / 'data/ugm', logger)
    m = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    m.load_params_from_file(filename=str(PP_CKPT), logger=logger, to_cpu=False)
    m.cuda().eval()
    return m, ds


def infer(model, ds, points):
    dd = ds.prepare_data({'points': points, 'frame_id': 'x'})
    dd = ds.collate_batch([dd]); load_data_to_gpu(dd)
    with torch.no_grad():
        pred, _ = model.forward(dd)
    p = pred[0]
    return (p['pred_boxes'].cpu().numpy(),
            p['pred_scores'].cpu().numpy(),
            p['pred_labels'].cpu().numpy())


def nms_merge(boxes, scores, labels, iou_thr):
    kb, ks, kl = [], [], []
    for c in range(1, len(CLASS_NAMES) + 1):
        m = labels == c
        if m.sum() == 0:
            continue
        b = torch.from_numpy(boxes[m]).float().cuda()
        s = torch.from_numpy(scores[m]).float().cuda()
        idx, _ = iou3d_nms_utils.nms_gpu(b, s, thresh=iou_thr)
        idx = idx.cpu().numpy()
        kb.append(boxes[m][idx]); ks.append(scores[m][idx]); kl.append(labels[m][idx])
    if not kb:
        return np.zeros((0, 7)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(kb), np.concatenate(ks), np.concatenate(kl)


def tta_infer(model, ds, points):
    """Inference asli + flip-Y, gabung via NMS."""
    b0, s0, l0 = infer(model, ds, points)
    pf = points.copy(); pf[:, 1] *= -1
    bf, sf, lf = infer(model, ds, pf)
    bf = bf.copy(); bf[:, 1] *= -1; bf[:, 6] *= -1
    return nms_merge(np.concatenate([b0, bf]), np.concatenate([s0, sf]),
                     np.concatenate([l0, lf]), NMS_IOU)


def load_ids(name):
    return set((ISETS / name).read_text().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score_thresh', type=float, default=0.3)
    args = ap.parse_args()

    all_ids   = {p.stem for p in VEL.glob('*.bin')}
    labeled   = load_ids('train_manual150.txt') | load_ids('val_manual_v3_80.txt')
    new_ids   = sorted(all_ids - labeled)
    (OUT / 'label').mkdir(parents=True, exist_ok=True)

    print(f"\nPra-anotasi frame BARU (model m150@130, TTA, Car+Cyclist)")
    print(f"  pool={len(all_ids)}  labeled={len(labeled)}  BARU={len(new_ids)}")
    print(f"  thr={args.score_thresh}  out={OUT}\n")

    logger = common_utils.create_logger()
    model, ds = build_model(logger)

    counts = {'Car': 0, 'Cyclist': 0}
    floor_z = {'Car': [], 'Cyclist': []}     # z alas = z_center - dz/2 SETELAH snap
    snaps   = []                              # geseran z akibat snap-ke-tanah
    n_far   = 0                               # box >SNAP_RANGE (tak di-snap)
    csv_rows = ['frame_id,car,cyclist,total']

    for fid in tqdm(new_ids, desc='  infer'):
        pts = np.fromfile(str(VEL / f"{fid}.bin"), dtype=np.float32).reshape(-1, 4)
        pxyz = pts[:, :3]
        bb, ss, ll = tta_infer(model, ds, pts)
        keep = ss >= args.score_thresh
        bb, ll = bb[keep], ll[keep]

        objs, per = [], {'Car': 0, 'Cyclist': 0}
        for j in range(len(bb)):
            name = CLASS_NAMES[ll[j] - 1]
            if name not in KEEP:
                continue
            x, y, z, dx, dy, dz, yaw = [float(v) for v in bb[j]]
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            # snap alas box ke tanah lokal (kendala fisik; kurangi warisan bias melayang)
            rng = float(np.hypot(x, y))
            if rng <= SNAP_RANGE:
                g = ground_z(pxyz, x, y, max(dx, dy))
                if g is not None:
                    new_z = g + dz / 2
                    snaps.append(new_z - z)
                    z = new_z
            else:
                n_far += 1
            objs.append({
                "obj_id": str(len(objs)),
                "obj_type": name,
                "psr": {
                    "position": {"x": x, "y": y, "z": z},
                    "rotation": {"x": 0.0, "y": 0.0, "z": yaw},
                    "scale":    {"x": dx, "y": dy, "z": dz},
                },
            })
            per[name] += 1
            counts[name] += 1
            floor_z[name].append(z - dz / 2)

        (OUT / 'label' / f"{fid}.json").write_text(json.dumps(objs, indent=1))
        csv_rows.append(f"{fid},{per['Car']},{per['Cyclist']},{per['Car']+per['Cyclist']}")

    (OUT / 'frames.txt').write_text('\n'.join(new_ids) + '\n')
    (OUT / 'counts.csv').write_text('\n'.join(csv_rows) + '\n')

    print(f"\n── Hasil ──")
    print(f"  frame          : {len(new_ids)}")
    print(f"  Car boxes      : {counts['Car']}  (avg {counts['Car']/max(len(new_ids),1):.2f}/frame)")
    print(f"  Cyclist boxes  : {counts['Cyclist']}  (avg {counts['Cyclist']/max(len(new_ids),1):.2f}/frame)")
    sn = np.array(snaps)
    if len(sn):
        print(f"\n  snap-ke-tanah  : {len(sn)} box di-snap (≤{SNAP_RANGE:.0f} m), {n_far} box jauh dibiarkan")
        print(f"    geseran z    : median {np.median(sn):+.3f} m  |geser|>0.2m: {(np.abs(sn)>0.2).mean()*100:.0f}%  (seberapa melayang pseudo asli)")
    for c in ('Car', 'Cyclist'):
        z = np.array(floor_z[c])
        if len(z):
            print(f"  z-alas {c:8s}: median {np.median(z):+.3f} m (SETELAH snap; harusnya ≈ tanah lokal)")
    print(f"\n  Sustech JSON : {OUT/'label'}")
    print(f"  Ranking Car  : {OUT/'counts.csv'}  (urutkan desc kolom car untuk pilih frame padat)")
    print(f"\n  Tanah UGM ~ -1,8 m. z-alas Car harus mendekati itu; jauh di atas = melayang.")


if __name__ == '__main__':
    main()
