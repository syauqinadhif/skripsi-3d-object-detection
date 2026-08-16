#!/usr/bin/env python3
"""
Demo inferensi CLOCs v3 (calib benar + YOLO v3) untuk SATU frame.

Memakai pipeline v3 yang sudah jadi (tidak menjalankan PP/YOLO ulang):
  - input_data cache : input_data_ugm_m150_v3/<fid>.pt (fusion_input + tensor_index)
  - kandidat 3D      : 3D_ugm_m150_v3/<fid>.pt (500 pre-NMS PP m150 ep130)
  - model CLOCs      : log/pointpillar_ugm/m150_v3/<epoch>.pt (fusion_layer tersimpan utuh)
Forward CLOCs -> skor fusi baru -> NMS -> box final. Divisualkan:
  <out>/<fid>_camera.jpg  (box 3D diproyeksikan ke gambar, calib v3)
  <out>/<fid>_bev.jpg     (bird-eye view + titik LiDAR)

Usage:
  python scripts/demo_clocs_v3.py --frame_id 000196 [--epoch 49] [--score_thresh 0.1]
"""
import argparse, sys, os
from pathlib import Path
import numpy as np, cv2, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
CLOCS = ROOT / 'CLOCs_LQS'
os.chdir(str(CLOCS))
sys.path.insert(0, str(CLOCS))
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
from tool import nms
from pcdet.config import cfg, cfg_from_yaml_file

CLASSES = ['Car', 'Pedestrian', 'Cyclist']
COLORS = {'Car': (0, 255, 0), 'Pedestrian': (255, 150, 0), 'Cyclist': (0, 200, 255)}
IMG_W, IMG_H = 1280, 720


def load_calib(fid):
    d = {}
    for ln in (ROOT / f'dataset_ugm/calib/{fid}.txt').read_text().splitlines():
        if ':' in ln:
            k, v = ln.split(':', 1); d[k.strip()] = np.array([float(x) for x in v.split()])
    return d['P2'].reshape(3, 4), d['Tr_velo_to_cam'].reshape(3, 4), d.get('R0_rect', np.eye(3).ravel()).reshape(3, 3)


def corners_lidar(b):
    x, y, z, dx, dy, dz, yaw = b[:7]
    xc = np.array([dx/2, dx/2, -dx/2, -dx/2]*2); yc = np.array([dy/2, -dy/2, -dy/2, dy/2]*2)
    zc = np.array([-dz/2]*4 + [dz/2]*4)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return (R @ np.vstack([xc, yc, zc])).T + [x, y, z]


def draw_cam(img, boxes, names, P2, Tr, R0):
    M = np.eye(4); M[:3, :3] = R0; T = np.vstack([Tr, [0, 0, 0, 1]]); M = M @ T
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for b, nm in zip(boxes, names):
        cor = corners_lidar(b)
        cam = (M @ np.hstack([cor, np.ones((8,1))]).T).T[:, :3]
        if (cam[:, 2] > 0.1).sum() < 8:
            continue
        im = (P2 @ np.hstack([cam, np.ones((8,1))]).T).T
        im = im[:, :2] / im[:, 2:3]
        col = COLORS.get(nm, (255,255,255))
        for a, c in E:
            cv2.line(img, tuple(im[a].astype(int)), tuple(im[c].astype(int)), col, 2)
    return img


def draw_bev(pts, boxes, names, scores, out):
    fig, ax = plt.subplots(figsize=(10, 10))
    m = (np.abs(pts[:, 1]) < 40) & (pts[:, 0] > 0) & (pts[:, 0] < 70)
    ax.scatter(pts[m, 1], pts[m, 0], s=0.2, c='gray', alpha=0.5)
    for b, nm, sc in zip(boxes, names, scores):
        cor = corners_lidar(b)[:4]
        col = np.array(COLORS.get(nm, (255,255,255)))[::-1] / 255
        poly = np.vstack([cor[:, 1], cor[:, 0]]).T
        poly = np.vstack([poly, poly[0]])
        ax.plot(poly[:, 0], poly[:, 1], c=col, lw=2)
        ax.text(cor[:, 1].mean(), cor[:, 0].mean(), f'{nm[:3]} {sc:.2f}', color=col, fontsize=8)
    ax.set_xlabel('y (kiri +)'); ax.set_ylabel('x (depan)'); ax.set_title('BEV — CLOCs v3 fusi')
    ax.set_xlim(40, -40); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    fig.savefig(out, dpi=100, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame_id', required=True)
    ap.add_argument('--epoch', type=int, default=49)
    ap.add_argument('--score_thresh', type=float, default=0.1)
    ap.add_argument('--out', default=str(ROOT / 'output/clocs_v3_demo'))
    args = ap.parse_args()
    fid = args.frame_id
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg_from_yaml_file('./tool/cfgs/kitti_models/pointpillar_ugm/pointpillar_ugm_m150.yaml', cfg)
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh

    d2 = './data/clocs_data/2D_ugm_v3'
    d3 = f'./data/clocs_data/3D_ugm_m150_v3/{fid}.pt'
    inp = f'./data/clocs_data/input_data_ugm_m150_v3/{fid}.pt'
    for p in [d3, inp, f'{d2}/{fid}.txt']:
        assert Path(p).exists(), f'tidak ada: {p}'

    net = torch.load(f'./log/pointpillar_ugm/m150_v3/{args.epoch}.pt')
    net.cuda().eval()

    cache = torch.load(inp)['input_data']
    fusion_input = torch.tensor(cache['fusion_input']).reshape(1, 4, 1, -1).cuda()
    tensor_index = torch.tensor(cache['tensor_index']).reshape(-1, 2).cuda()

    r3 = torch.load(d3)[0]
    n3 = len(r3['score'])
    with torch.no_grad():
        fused, _ = net(fusion_input, tensor_index)
    sc = torch.sigmoid(fused.reshape(-1))[:n3].cuda()
    r3['score'] = sc.cpu().numpy()
    box = torch.tensor(np.array(r3['boxes_lidar']), dtype=torch.float32).cuda()
    sel = nms.nms(sc, box, cfg.MODEL.POST_PROCESSING).cpu().numpy()

    boxes = np.array(r3['boxes_lidar'])[sel]
    names = np.array(r3['name'])[sel]
    scores = np.array(r3['score'])[sel]
    order = np.argsort(-scores)
    boxes, names, scores = boxes[order], names[order], scores[order]
    print(f"frame {fid}: {n3} kandidat -> {len(sel)} box lolos NMS (score>={args.score_thresh})")
    from collections import Counter
    print("  per kelas:", dict(Counter(names)))
    for nm, s, b in zip(names[:12], scores[:12], boxes[:12]):
        print(f"    {nm:11s} score {s:.3f}  xyz=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f}) l{b[3]:.1f}w{b[4]:.1f}h{b[5]:.1f}")

    P2, Tr, R0 = load_calib(fid)
    img = cv2.imread(str(ROOT / f'dataset_ugm/image_2/{fid}.png'))
    img = draw_cam(img, boxes, names, P2, Tr, R0)
    cv2.putText(img, f'CLOCs v3 fusi ep{args.epoch} — {fid} ({len(sel)} box)', (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(out / f'{fid}_camera.jpg'), img)

    pts = np.fromfile(ROOT / f'dataset_ugm/velodyne/{fid}.bin', dtype=np.float32).reshape(-1, 4)
    draw_bev(pts, boxes, names, scores, out / f'{fid}_bev.jpg')
    print(f"  -> {out}/{fid}_camera.jpg  &  {fid}_bev.jpg")


if __name__ == '__main__':
    main()
