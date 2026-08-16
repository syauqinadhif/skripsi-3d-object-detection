#!/usr/bin/env python3
"""Decoder PointPillars portabel (torch murni, TANPA pcdet/CUDA-ext) untuk output engine TensorRT.
Mereplikasi AnchorHeadSingle.generate_predicted_boxes + ResidualCoder.decode_torch + refinement arah.
NMS TIDAK dilakukan di sini (dikerjakan di workstation). Anchor dibaca dari anchors.npy.

Dipakai di Jetson: cls/box/dir engine [1,C,248,216] -> boxes[N,7] + scores[N] + labels[N] (thr 0.1).
"""
import numpy as np, torch

DIR_OFFSET = 0.78539
DIR_LIMIT_OFFSET = 0.0
NUM_DIR_BINS = 2
SCORE_THRESH = 0.1

def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period

def residual_decode(box_enc, anchors):
    xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, dxt, dyt, dzt, rt = torch.split(box_enc, 1, dim=-1)
    diagonal = torch.sqrt(dxa**2 + dya**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za
    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza
    rg = rt + ra
    return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg], dim=-1)

def decode(cls_e, box_e, dir_e, anchors, score_thresh=SCORE_THRESH, device='cuda'):
    """cls_e[1,18,H,W], box_e[1,42,H,W], dir_e[1,12,H,W] (NCHW raw); anchors[1,-1,7]. Return np arrays."""
    cls_e = torch.as_tensor(cls_e, device=device, dtype=torch.float32)
    box_e = torch.as_tensor(box_e, device=device, dtype=torch.float32)
    dir_e = torch.as_tensor(dir_e, device=device, dtype=torch.float32)
    anchors = torch.as_tensor(anchors, device=device, dtype=torch.float32).view(1, -1, 7)
    # NCHW -> NHWC -> [1, num_anchors, C]
    cls = cls_e.permute(0, 2, 3, 1).reshape(1, -1, 3)
    box = box_e.permute(0, 2, 3, 1).reshape(1, -1, 7)
    dr  = dir_e.permute(0, 2, 3, 1).reshape(1, -1, NUM_DIR_BINS)
    box_dec = residual_decode(box, anchors)
    dir_labels = torch.max(dr, dim=-1)[1]
    period = 2 * np.pi / NUM_DIR_BINS
    dir_rot = limit_period(box_dec[..., 6] - DIR_OFFSET, DIR_LIMIT_OFFSET, period)
    box_dec[..., 6] = dir_rot + DIR_OFFSET + period * dir_labels.to(box_dec.dtype)
    scores = torch.sigmoid(cls)
    smax, labels = scores.max(dim=-1)          # [1, num_anchors]
    mask = smax[0] >= score_thresh
    b = box_dec[0][mask].cpu().numpy()
    s = smax[0][mask].cpu().numpy()
    l = (labels[0][mask] + 1).cpu().numpy()    # 1-indexed: 1=Car 2=Ped 3=Cyclist
    return b, s, l
