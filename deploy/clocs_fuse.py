#!/usr/bin/env python3
"""Modul fusi CLOCs standalone (numpy/torch/numba) — jalan di workstation & Jetson TANPA pcdet.
Berisi: konversi kamera (konvensi convert_3d_to_clocs, sesuai saat CLOCs dilatih — untuk INPUT fusi),
asosiasi 2D-3D (build_stage2_training, numba), dan pemanggilan net fusi.
NMS final + konversi eval (box_utils) dikerjakan terpisah di workstation.
"""
import numpy as np, torch, numba
IMG_W, IMG_H = 1280, 720

def load_calib(path):
    d = {}
    for ln in open(path):
        if ':' in ln:
            k, v = ln.split(':', 1); d[k.strip()] = np.array([float(x) for x in v.split()])
    P2 = d['P2'].reshape(3, 4); R0 = d['R0_rect'].reshape(3, 3); Tr = d['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, Tr

def boxes_lidar_to_camera(boxes_lidar, P2, R0, Tr):
    """(N,7)[x,y,z,dx,dy,dz,yaw] -> (loc, dims[l,h,w], ry, bbox 2D). Konvensi = saat CLOCs dilatih."""
    n = len(boxes_lidar)
    if n == 0:
        return (np.zeros((0,3),np.float32), np.zeros((0,3),np.float32), np.zeros((0,),np.float32), np.zeros((0,4),np.float32))
    Tr4 = np.vstack([Tr, [0,0,0,1]]); R4 = np.eye(4); R4[:3,:3] = R0; M = R4 @ Tr4
    dx, dy, dz = boxes_lidar[:,3], boxes_lidar[:,4], boxes_lidar[:,5]
    h, w, l = dz, dy, dx; yaw = boxes_lidar[:,6]
    ry = (-yaw - np.pi/2 + np.pi) % (2*np.pi) - np.pi
    dims = np.stack([l, h, w], axis=1)
    bottom = boxes_lidar[:,:3].copy(); bottom[:,2] -= h/2
    loc = (M @ np.hstack([bottom, np.ones((n,1))]).T).T[:,:3]
    bbox = np.zeros((n,4), np.float32)
    for i in range(n):
        cx, cy, cz = boxes_lidar[i,:3]
        Rz = np.array([[np.cos(yaw[i]),-np.sin(yaw[i]),0],[np.sin(yaw[i]),np.cos(yaw[i]),0],[0,0,1]])
        xc = np.array([l[i]/2,l[i]/2,-l[i]/2,-l[i]/2]*2); yc = np.array([w[i]/2,-w[i]/2,-w[i]/2,w[i]/2]*2)
        zc = np.array([-h[i]/2]*4 + [h[i]/2]*4)
        cor_l = (Rz @ np.vstack([xc,yc,zc])).T + [cx,cy,cz]
        cor = (M @ np.hstack([cor_l, np.ones((8,1))]).T).T[:,:3]
        v = cor[:,2] > 0.1
        if v.sum() < 1: continue
        im = (P2 @ np.hstack([cor[v], np.ones((v.sum(),1))]).T).T
        im = im[:,:2] / im[:,2:3]
        x1,y1 = im.min(0); x2,y2 = im.max(0)
        bbox[i] = [np.clip(x1,0,IMG_W), np.clip(y1,0,IMG_H), np.clip(x2,0,IMG_W), np.clip(y2,0,IMG_H)]
    return loc.astype(np.float32), dims.astype(np.float32), ry.astype(np.float32), bbox

@numba.jit(nopython=True, parallel=True)
def build_stage2_training(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d, overlaps, tensor_index):
    N = boxes.shape[0]; K = query_boxes.shape[0]; ind = 0
    for k in range(K):
        qbox_area = ((query_boxes[k,2]-query_boxes[k,0])*(query_boxes[k,3]-query_boxes[k,1]))
        for n in range(N):
            iw = min(boxes[n,2],query_boxes[k,2]) - max(boxes[n,0],query_boxes[k,0])
            if iw > 0:
                ih = min(boxes[n,3],query_boxes[k,3]) - max(boxes[n,1],query_boxes[k,1])
                if ih > 0:
                    ua = (boxes[n,2]-boxes[n,0])*(boxes[n,3]-boxes[n,1]) + qbox_area - iw*ih
                    overlaps[ind,0] = iw*ih/ua; overlaps[ind,1] = scores_3d[n]
                    overlaps[ind,2] = scores_2d[k,0]; overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k; tensor_index[ind,1] = n; ind += 1
                elif k == K-1:
                    overlaps[ind,0] = -10; overlaps[ind,1] = scores_3d[n]; overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]; tensor_index[ind,0] = k; tensor_index[ind,1] = n; ind += 1
            elif k == K-1:
                overlaps[ind,0] = -10; overlaps[ind,1] = scores_3d[n]; overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]; tensor_index[ind,0] = k; tensor_index[ind,1] = n; ind += 1
    return overlaps, tensor_index, ind

def fuse(boxes_lidar, scores_3d, bbox_3d, top_predictions, net, device='cuda'):
    """boxes_lidar(N,7), scores_3d(N), bbox_3d(N,4) proyeksi 2D, top_predictions(K,5)=[x1,y1,x2,y2,score].
    Return skor fusi (N,) tersigmoid."""
    if len(boxes_lidar) == 0:
        return np.zeros((0,), np.float32)
    if len(top_predictions) > 200:                       # net out_1 dim = 200 deteksi 2D
        top_predictions = top_predictions[np.argsort(-top_predictions[:,4])[:200]]
    dis = np.linalg.norm(boxes_lidar[:,:2], axis=1, keepdims=True) / 82.0
    box2d_sc = top_predictions[:,4].reshape(-1,1)
    ov = np.full((900000,4), -1.0, np.float32); ti = np.full((900000,2), -1.0, np.float32)
    iou, ti, mx = build_stage2_training(bbox_3d.astype(np.float32), top_predictions[:,:4].astype(np.float32),
                                        -1, scores_3d.astype(np.float32), box2d_sc.astype(np.float32),
                                        dis.astype(np.float32), ov, ti)
    it = torch.FloatTensor(iou).permute(1,0).reshape(4,900000)
    tit = torch.LongTensor(ti).reshape(-1,2)
    if mx == 0:
        return scores_3d.astype(np.float32)
    it = it[:,:mx]; tit = tit[:mx,:]
    fin = it.reshape(4,1,-1).unsqueeze(0).to(device)
    with torch.no_grad():
        cls, _ = net(fin, tit.to(device))
    sc = torch.sigmoid(cls.reshape(-1).cpu()).numpy()[:len(scores_3d)]
    return sc.astype(np.float32)
