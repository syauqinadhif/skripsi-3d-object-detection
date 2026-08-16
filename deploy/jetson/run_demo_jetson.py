#!/usr/bin/env python3
"""
run_demo_jetson.py — Demo inferensi fusi CLOCs pada Jetson AGX Orin (per-frame, 100 frame).

Pipeline penuh berjalan di Jetson:
    voxel (pra-voxelisasi) -> engine TensorRT PointPillars -> decode
    citra                  -> YOLO26 (ultralytics, live)   -> kandidat 2D
                                                            -> CLOCs fuse -> NMS
Keluaran:
    demo_out/<fid>.jpg   citra kamera + 3D bbox hasil fusi
    ringkasan latency per tahap + FPS di akhir.

Dijalankan dari ~/pp_deploy:
    python3 run_demo_jetson.py --prec fp16 --thr 0.12
"""
import numpy as np, torch, tensorrt as trt, time, argparse, os
import cv2
from torchvision.ops import nms
import pp_decode, clocs_fuse
from ultralytics import YOLO

COLOR = {1: (0, 255, 0), 3: (0, 165, 255)}   # 1=Car hijau, 3=Cyclist oranye (BGR); 2=Ped di-skip
NAME  = {1: 'Car', 3: 'Cyclist'}
TT = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.INT32: torch.int32}


def load_engine(path):
    lg = trt.Logger(trt.Logger.ERROR)
    with open(path, 'rb') as f, trt.Runtime(lg) as rt:
        return rt.deserialize_cuda_engine(f.read())


def bev_aabb(boxes):
    """AABB (di bidang BEV x-y) dari kotak rotasi -> untuk NMS spasial via torchvision.
    boxes(N,7)[x,y,z,dx,dy,dz,yaw] -> (N,4)[x1,y1,x2,y2]."""
    x, y, dx, dy, yaw = boxes[:,0], boxes[:,1], boxes[:,3], boxes[:,4], boxes[:,6]
    ex = np.abs(dx/2*np.cos(yaw)) + np.abs(dy/2*np.sin(yaw))
    ey = np.abs(dx/2*np.sin(yaw)) + np.abs(dy/2*np.cos(yaw))
    return np.stack([x-ex, y-ey, x+ex, y+ey], axis=1).astype(np.float32)


def corners_lidar(box):
    x, y, z, dx, dy, dz, yaw = box
    lx, ly, lz = dx/2, dy/2, dz/2
    c = np.array([[lx,ly,-lz],[lx,-ly,-lz],[-lx,-ly,-lz],[-lx,ly,-lz],
                  [lx,ly,lz],[lx,-ly,lz],[-lx,-ly,lz],[-lx,ly,lz]]).T
    cs, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]])
    c = R @ c; c[0]+=x; c[1]+=y; c[2]+=z
    return c


def draw(img, boxes, scores, labels, P2, R0, Tr, thr):
    n = 0
    for i in range(len(scores)):
        if scores[i] < thr or int(labels[i]) not in COLOR:
            continue
        cl = corners_lidar(boxes[i])
        cc = R0 @ (Tr @ np.vstack([cl, np.ones((1,8))]))
        if (cc[2] <= 0.1).any():
            continue
        h = P2 @ np.vstack([cc, np.ones((1,8))]); h[:2] /= h[2]
        p = h[:2].T.astype(int)
        col = COLOR[int(labels[i])]
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            cv2.line(img, tuple(p[a]), tuple(p[b]), col, 2)
        x1,y1 = int(p[:,0].min()), int(p[:,1].min())
        cv2.putText(img, f"{NAME[int(labels[i])]} {scores[i]:.2f}", (x1, max(y1-5,15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prec', choices=['fp16','fp32'], default='fp16')
    ap.add_argument('--thr', type=float, default=0.15)
    ap.add_argument('--nms_iou', type=float, default=0.1, help='NMS BEV IoU (kecil = agresif)')
    ap.add_argument('--outdir', type=str, default='demo_out')
    ap.add_argument('--limit', type=int, default=0, help='0 = semua frame di npz')
    ap.add_argument('--frames', type=str, default='',
                    help='daftar frame spesifik dipisah koma, mis. 000150,000158,000205')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = np.load('voxels_demo100.npz'); ids = [str(x) for x in data['ids']]
    if args.frames:
        want = [f.strip().zfill(6) for f in args.frames.split(',') if f.strip()]
        missing = [f for f in want if f not in ids]
        if missing:
            print(f'  [peringatan] frame tidak ada di npz (rentang {ids[0]}-{ids[-1]}): {missing}')
        ids = [f for f in want if f in ids]
    elif args.limit:
        ids = ids[:args.limit]
    anchors = np.load('anchors.npy')
    P2, R0, Tr = clocs_fuse.load_calib('calib.txt')
    net = torch.load('clocs_net.pt', weights_only=False); net.cuda().eval()
    yolo = YOLO('best.pt')

    eng = load_engine(f'pp_{args.prec}.engine'); ctx = eng.create_execution_context(); buf = {}
    for i in range(eng.num_bindings):
        nm = eng.get_binding_name(i); dt = eng.get_binding_dtype(i); sh = tuple(ctx.get_binding_shape(i))
        buf[nm] = torch.zeros(sh, dtype=TT[dt], device='cuda'); ctx.set_tensor_address(nm, int(buf[nm].data_ptr()))

    print(f'[Jetson] engine={args.prec}  {len(ids)} frame  thr={args.thr}')
    t_pp, t_yolo, t_fuse, t_tot = [], [], [], []
    for k, fid in enumerate(ids):
        tt0 = time.time()
        # 1) PointPillars TensorRT
        v = data[f'v_{fid}']; n = data[f'n_{fid}']; c = data[f'c_{fid}']; nv = v.shape[0]
        buf['voxels'].zero_(); buf['voxels'][:nv] = torch.from_numpy(v).to(buf['voxels'].dtype).cuda()
        buf['voxel_num'].zero_(); buf['voxel_num'][:nv] = torch.from_numpy(n).to(torch.int32).cuda(); buf['voxel_num'][nv:] = 1
        buf['voxel_idxs'].zero_(); buf['voxel_idxs'][:nv,1:4] = torch.from_numpy(c).to(torch.int32).cuda()
        torch.cuda.synchronize(); t0 = time.time()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s): ctx.execute_async_v3(s.cuda_stream)
        s.synchronize(); torch.cuda.synchronize()
        cls = buf['cls_preds'].float().cpu().numpy(); box = buf['box_preds'].float().cpu().numpy(); dr = buf['dir_cls_preds'].float().cpu().numpy()
        b, sc, l = pp_decode.decode(cls, box, dr, anchors, device='cuda')
        t_pp.append(time.time() - t0)

        # 2) YOLO26 live
        t0 = time.time()
        rj = yolo(f'images/{fid}.png', conf=0.0, verbose=False)[0]
        b2 = rj.boxes.xyxy.cpu().numpy(); s2 = rj.boxes.conf.cpu().numpy()
        top = np.hstack([b2, s2.reshape(-1,1)]).astype(np.float32) if len(b2) else np.zeros((0,5), np.float32)
        t_yolo.append(time.time() - t0)

        # 3) CLOCs fuse + NMS
        t0 = time.time()
        _, _, _, bbox = clocs_fuse.boxes_lidar_to_camera(b, P2, R0, Tr)
        sf = clocs_fuse.fuse(b, sc, bbox, top, net, device='cuda')
        if len(b):
            keep = nms(torch.tensor(bev_aabb(b), dtype=torch.float32),   # NMS spasial di BEV
                       torch.tensor(sf, dtype=torch.float32), args.nms_iou).numpy()
            b, sf, l = b[keep], sf[keep], l[keep]
        t_fuse.append(time.time() - t0)

        # 4) render
        img = cv2.imread(f'images/{fid}.png')
        ndrawn = draw(img, b, sf, l, P2, R0, Tr, args.thr)
        cv2.putText(img, f"Jetson {args.prec.upper()} | frame {fid} | {ndrawn} objek | fusi CLOCs",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(f'{args.outdir}/{fid}.jpg', img)
        t_tot.append(time.time() - tt0)
        if k % 10 == 0:
            print(f'  {k+1}/{len(ids)}  {fid}  {ndrawn} objek  '
                  f'PP={t_pp[-1]*1000:.1f} YOLO={t_yolo[-1]*1000:.1f} fusi={t_fuse[-1]*1000:.1f} ms')

    def ms(a): return np.mean(a) * 1000
    # buang frame-0 (warmup) dari statistik
    print(f'\n=== Ringkasan latency Jetson ({args.prec}, {len(ids)} frame; rata-rata tanpa warmup) ===')
    print(f'  PointPillars (TensorRT) : {ms(t_pp[1:]):.1f} ms')
    print(f'  YOLO26 (ultralytics)    : {ms(t_yolo[1:]):.1f} ms')
    print(f'  CLOCs fuse + NMS        : {ms(t_fuse[1:]):.1f} ms')
    print(f'  TOTAL per frame         : {ms(t_tot[1:]):.1f} ms  ({1000/ms(t_tot[1:]):.1f} FPS pipeline penuh)')
    print(f'  Hasil citra: {args.outdir}/  ({len(ids)} jpg)')


if __name__ == '__main__':
    main()
