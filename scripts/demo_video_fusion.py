"""
demo_video_fusion.py — Inferensi VIDEO fusi CLOCs pada sekuens dataset UGM (di PC).

Model dimuat SEKALI, lalu seluruh frame pada rentang sekuens diinferensikan
berturut-turut (PointPillars -> YOLO26 -> CLOCs) dan dirender menjadi satu video mp4.
Tiap frame komposit menampilkan:
    kiri  : citra kamera + 3D bbox hasil fusi (diproyeksikan)
    kanan : bird-eye view (titik LiDAR + kotak hasil fusi)

Contoh:
    conda run -n openpcdet python scripts/demo_video_fusion.py \
        --start 0 --end 200 --fps 10 --out output/demo_fusion.mp4

    # seluruh 500 frame:
    conda run -n openpcdet python scripts/demo_video_fusion.py --start 0 --end 500
"""
import argparse, sys, os, time
import numpy as np
import cv2
import torch
from pathlib import Path

ROOT     = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
OPENPCDET= ROOT / 'OpenPCDet'
CLOCS_LQS= ROOT / 'CLOCs_LQS'
UGM      = ROOT / 'dataset_ugm'

PP_CKPT  = OPENPCDET / 'output/kitti_models/pointpillar_ugm_m300_v3/finetune_kitti_m300/ckpt/checkpoint_epoch_120.pth'
PP_CFG   = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar_ugm_m300_prenms_val.yaml'
YOLO_CKPT= ROOT / 'output/yolo26_2cls/y26m_2cls/weights/best.pt'
CLOCS_CKPT=CLOCS_LQS / 'log/pointpillar_ugm/ft120_y26/20.pt'
CLOCS_CFG =CLOCS_LQS / 'tool/cfgs/kitti_models/pointpillar_ugm/pointpillar_ugm_m150.yaml'

sys.path.insert(0, str(CLOCS_LQS))
sys.path.insert(0, str(OPENPCDET))

COLOR = {'Car': (0, 255, 0), 'Cyclist': (0, 165, 255)}   # BGR
BEV_X = (0.0, 60.0)     # jangkauan maju (m) yang ditampilkan
BEV_Y = (-25.0, 25.0)   # jangkauan melintang (m)
BEV_W, BEV_H = 600, 720 # ukuran kanvas BEV (piksel)

# ── kalibrasi (rig UGM tunggal, semua frame identik) ──────────────────────────
def load_calib():
    calib = {}
    for line in open(UGM / 'calib' / '000000.txt'):
        if ':' in line:
            k, v = line.strip().split(':', 1)
            calib[k] = np.array([float(x) for x in v.split()])
    return (calib['P2'].reshape(3, 4),
            calib['R0_rect'].reshape(3, 3),
            calib['Tr_velo_to_cam'].reshape(3, 4))

def corners_lidar(box):
    x, y, z, dx, dy, dz, yaw = box
    lx, ly, lz = dx/2, dy/2, dz/2
    c = np.array([[lx,ly,-lz],[lx,-ly,-lz],[-lx,-ly,-lz],[-lx,ly,-lz],
                  [lx,ly,lz],[lx,-ly,lz],[-lx,-ly,lz],[-lx,ly,lz]]).T
    cs, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
    c = R @ c
    c[0] += x; c[1] += y; c[2] += z
    return c

def project(pts3d, P2):
    h = np.vstack([pts3d, np.ones((1, pts3d.shape[1]))])
    p = P2 @ h
    p[:2] /= p[2]
    return p[:2].T

# ── render kamera ─────────────────────────────────────────────────────────────
def draw_camera(img, det, P2, R0, Tr, thr):
    R0e = np.eye(4); R0e[:3,:3] = R0
    Tre = np.vstack([Tr, [0,0,0,1]])
    for i in range(len(det['score'])):
        if det['score'][i] < thr or det['name'][i] not in COLOR:
            continue
        cl = corners_lidar(det['boxes_lidar'][i])
        cc = R0 @ (Tr @ np.vstack([cl, np.ones((1,8))]))
        if (cc[2] <= 0.1).any():
            continue
        c2 = project(cc, P2).astype(int)
        col = COLOR[det['name'][i]]
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            cv2.line(img, tuple(c2[a]), tuple(c2[b]), col, 2)
        x1, y1 = int(c2[:,0].min()), int(c2[:,1].min())
        cv2.putText(img, f"{det['name'][i]} {det['score'][i]:.2f}",
                    (x1, max(y1-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img

# ── render BEV (murni OpenCV, cepat untuk video) ──────────────────────────────
def bev_xy_to_px(x, y):
    u = (BEV_Y[1] - y) / (BEV_Y[1] - BEV_Y[0]) * BEV_W
    v = (BEV_X[1] - x) / (BEV_X[1] - BEV_X[0]) * BEV_H
    return int(u), int(v)

def draw_bev(points, det, thr):
    canvas = np.full((BEV_H, BEV_W, 3), 26, dtype=np.uint8)   # abu gelap
    m = ((points[:,0] > BEV_X[0]) & (points[:,0] < BEV_X[1]) &
         (points[:,1] > BEV_Y[0]) & (points[:,1] < BEV_Y[1]))
    pf = points[m]
    if len(pf):
        u = ((BEV_Y[1] - pf[:,1]) / (BEV_Y[1]-BEV_Y[0]) * BEV_W).astype(int)
        v = ((BEV_X[1] - pf[:,0]) / (BEV_X[1]-BEV_X[0]) * BEV_H).astype(int)
        ok = (u>=0)&(u<BEV_W)&(v>=0)&(v<BEV_H)
        hn = np.clip((pf[:,2]+2)/4.0, 0, 1)[ok]
        cols = (cv2.applyColorMap((hn*255).astype(np.uint8).reshape(-1,1), cv2.COLORMAP_PLASMA)
                .reshape(-1,3))
        canvas[v[ok], u[ok]] = cols
    # posisi ego
    cv2.drawMarker(canvas, bev_xy_to_px(0,0), (255,255,255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    for i in range(len(det['score'])):
        if det['score'][i] < thr or det['name'][i] not in COLOR:
            continue
        b = det['boxes_lidar'][i]
        x,y,_,dx,dy,_,yaw = b
        cs,sn = np.cos(yaw), np.sin(yaw)
        crn = np.array([[ dx/2, dy/2],[ dx/2,-dy/2],[-dx/2,-dy/2],[-dx/2, dy/2]])
        R = np.array([[cs,-sn],[sn,cs]])
        crn = (R @ crn.T).T + [x,y]
        pts = np.array([bev_xy_to_px(px,py) for px,py in crn], np.int32)
        col = COLOR[det['name'][i]]
        cv2.polylines(canvas, [pts], True, col, 2)
        # garis arah hadap (ke tengah sisi depan)
        fx,fy = (crn[0]+crn[1])/2
        cv2.line(canvas, bev_xy_to_px(x,y), bev_xy_to_px(fx,fy), col, 1)
    # skala grid tiap 10 m
    for gx in range(int(BEV_X[0]), int(BEV_X[1])+1, 10):
        _, vy = bev_xy_to_px(gx, 0)
        cv2.line(canvas, (0,vy), (BEV_W,vy), (60,60,60), 1)
        cv2.putText(canvas, f"{gx}m", (4, vy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150),1)
    return canvas

# ── pemuatan model (sekali) ───────────────────────────────────────────────────
def build_models():
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from pcdet.datasets import DatasetTemplate

    cwd = os.getcwd(); os.chdir(str(OPENPCDET / 'tools'))
    cfg_from_yaml_file(str(PP_CFG), cfg); os.chdir(cwd)
    logger = common_utils.create_logger()

    class OneFrame(DatasetTemplate):
        def __init__(s): super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                          training=False, root_path=OPENPCDET/'data/ugm', logger=logger)
        def __len__(s): return 1
        def __getitem__(s, i): return {}
    dataset = OneFrame()
    pp = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    pp.load_params_from_file(filename=str(PP_CKPT), logger=logger)
    pp.cuda().eval()

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_CKPT))

    from pcdet.config import cfg as cfg_c, cfg_from_yaml_file as load_c
    cwd = os.getcwd(); os.chdir(str(CLOCS_LQS)); load_c(str(CLOCS_CFG), cfg_c); os.chdir(cwd)
    net = torch.load(str(CLOCS_CKPT), weights_only=False); net.cuda().eval()

    return cfg, dataset, pp, yolo, net, cfg_c

# ── fusi satu frame ───────────────────────────────────────────────────────────
def fuse_frame(fid, cfg, dataset, pp, yolo, net, cfg_c, P2, R0, Tr):
    from pcdet.models import load_data_to_gpu
    from tool.dataset import clocs_data
    from tool import nms as clocs_nms

    points = np.fromfile(str(UGM/'velodyne'/f'{fid}.bin'), dtype=np.float32).reshape(-1,4)
    dd = dataset.collate_batch([dataset.prepare_data({'points': points, 'frame_id': fid})])
    with torch.no_grad():
        load_data_to_gpu(dd)
        pred, _ = pp(dd)
    boxes = pred[0]['pred_boxes'].cpu().numpy()
    scr   = pred[0]['pred_scores'].cpu().numpy()
    lab   = pred[0]['pred_labels'].cpu().numpy()
    names = np.array([cfg.CLASS_NAMES[int(l)-1] for l in lab])

    R0e = np.eye(4); R0e[:3,:3] = R0
    V2C = R0e @ np.vstack([Tr, [0,0,0,1]])
    ctr = (V2C @ np.hstack([boxes[:,:3], np.ones((len(boxes),1))]).T).T[:,:3]
    pimg= project(ctr.T, P2)
    bbox= np.hstack([pimg-20, pimg+20])
    d3 = {'name':names,'truncated':np.full(len(scr),-1.0),'occluded':np.full(len(scr),-1,int),
          'alpha':np.full(len(scr),-10.0),'bbox':bbox.astype(np.float32),'dimensions':boxes[:,3:6],
          'location':ctr.astype(np.float32),'rotation_y':boxes[:,6].astype(np.float32),
          'score':scr.astype(np.float32),'boxes_lidar':boxes.astype(np.float32),'frame_id':np.str_(fid)}

    rj = yolo(str(UGM/'image_2'/f'{fid}.png'), conf=0.0, verbose=False)[0]
    b2 = rj.boxes.xyxy.cpu().numpy(); c2 = rj.boxes.conf.cpu().numpy()
    top = np.hstack([b2, c2.reshape(-1,1)])
    if len(top) > 200:
        top = top[np.argsort(-top[:,4])[:200]]

    dummy = clocs_data.__new__(clocs_data)
    _, iou_test, tidx = dummy.train_stage_2(d3, top)
    fin = iou_test.reshape(4,1,-1).unsqueeze(0).cuda()
    tidx= tidx.reshape(-1,2).cuda()
    with torch.no_grad():
        fcls, _ = net(fin, tidx)
    cls = torch.sigmoid(fcls.reshape(-1).cpu())[:len(d3['score'])]
    d3['score'] = cls.numpy()

    sel = clocs_nms.nms(cls.cuda(), torch.tensor(d3['boxes_lidar']).cuda(),
                        cfg_c.MODEL.POST_PROCESSING).cpu().numpy()
    for k in d3:
        if k != 'frame_id':
            d3[k] = d3[k][sel]
    return d3, points

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=142, help='rentang demo terbersih: 142-205')
    ap.add_argument('--end',   type=int, default=205, help='eksklusif')
    ap.add_argument('--fps',   type=int, default=10)
    ap.add_argument('--thr',   type=float, default=0.15, help='ambang skor fusi utk digambar')
    ap.add_argument('--out',   type=str, default='output/demo_fusion.mp4')
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    P2, R0, Tr = load_calib()

    print('[1/3] Memuat model (PointPillars + YOLO26 + CLOCs)...')
    cfg, dataset, pp, yolo, net, cfg_c = build_models()

    fids = [f'{i:06d}' for i in range(args.start, args.end)
            if (UGM/'velodyne'/f'{i:06d}.bin').exists()]
    print(f'[2/3] Menginferensi {len(fids)} frame -> {out}')

    writer = None
    t_all = time.time(); lat = []
    for k, fid in enumerate(fids):
        t0 = time.time()
        det, points = fuse_frame(fid, cfg, dataset, pp, yolo, net, cfg_c, P2, R0, Tr)
        lat.append(time.time() - t0)
        img = cv2.imread(str(UGM/'image_2'/f'{fid}.png'))
        cam = draw_camera(img, det, P2, R0, Tr, args.thr)
        bev = draw_bev(points, det, args.thr)
        # samakan tinggi, gabung
        h = cam.shape[0]
        bev = cv2.resize(bev, (int(BEV_W * h / BEV_H), h))
        combo = np.hstack([cam, bev])
        n = int((det['score'] >= args.thr).sum())
        cv2.putText(combo, f"frame {fid} | {n} objek | fusi CLOCs (PP+YOLO26)",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'),
                                     args.fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        if k % 20 == 0:
            print(f'  {k+1}/{len(fids)}  frame {fid}  {n} objek  ({lat[-1]*1000:.0f} ms)')
    if writer: writer.release()
    dt = time.time() - t_all
    print(f'[3/3] Selesai. {len(fids)} frame dalam {dt:.1f}s '
          f'(rata-rata {np.mean(lat)*1000:.0f} ms/frame, {1/np.mean(lat):.1f} FPS inferensi)')
    print(f'  Video: {out}')

if __name__ == '__main__':
    main()
