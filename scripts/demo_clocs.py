"""
demo_clocs.py — Inference & Visualisasi CLOCs (PointPillars + YOLOv8 + CLOCs Fusion)

Mode 1 (default, cepat):
    Menggunakan hasil pre-computed dari result/45.pt (epoch terbaik Run 2).
    Tidak perlu load ulang model PointPillars/YOLOv8.

Mode 2 (--live):
    Menjalankan full inference pipeline dari awal:
    LiDAR .bin -> PointPillars -> 3D candidates
    Image      -> YOLOv8      -> 2D candidates
                              -> CLOCs Fusion -> Final 3D Boxes

Output:
    <output_dir>/<frame_id>_camera.jpg  — 3D boxes diproyeksikan ke gambar kamera
    <output_dir>/<frame_id>_bev.jpg     — Bird's-Eye View dengan LiDAR points + boxes

Usage:
    # Mode cepat (pre-computed):
    conda run -n openpcdet python scripts/demo_clocs.py --frame_id 003219

    # Live inference:
    conda run -n openpcdet python scripts/demo_clocs.py --frame_id 003219 --live

    # Specify output dir:
    conda run -n openpcdet python scripts/demo_clocs.py --frame_id 003219 --output_dir ./output
"""

import argparse
import sys
import os
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
OPENPCDET    = PROJECT_ROOT / 'OpenPCDet'
CLOCS_LQS    = PROJECT_ROOT / 'CLOCs_LQS'
KITTI_ROOT   = Path('/media/mf/AUTODRIVING-4TB/Downloads/Datasets/KITTI')

PP_CKPT      = OPENPCDET / 'pointpillar_7728.pth'
PP_CFG       = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar_prenms.yaml'
YOLO_CKPT    = OPENPCDET / 'tools/yolov8m.pt'
CLOCS_CKPT   = CLOCS_LQS / 'log/pointpillar/yolov8m/45.pt'
CLOCS_CFG    = CLOCS_LQS / 'tool/cfgs/kitti_models/pointpillar/pointpillar_car.yaml'
PRECOMP_RESULT = CLOCS_LQS / 'log/pointpillar/yolov8m/result/45.pt'

sys.path.insert(0, str(OPENPCDET))
sys.path.insert(0, str(CLOCS_LQS))

# ── Calibration helpers ───────────────────────────────────────────────────────

def load_calib(frame_id):
    calib_path = KITTI_ROOT / 'calib' / f'{frame_id}.txt'
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in val.split()])
    P2            = calib['P2'].reshape(3, 4)
    R0_rect       = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam


# ── 3D box corner utilities ───────────────────────────────────────────────────

def corners_from_lidar_box(box_lidar):
    """Compute 8 corners in lidar frame from boxes_lidar [x,y,z,dx,dy,dz,heading].
    Lidar frame: x=forward, y=left, z=up. Center is geometric center.
    Returns (3, 8).
    """
    x, y, z, dx, dy, dz, heading = box_lidar
    lx, ly, lz = dx / 2, dy / 2, dz / 2
    # 8 corners relative to box center (before rotation)
    corners = np.array([
        [ lx,  ly, -lz],  # front-left-bottom
        [ lx, -ly, -lz],  # front-right-bottom
        [-lx, -ly, -lz],  # back-right-bottom
        [-lx,  ly, -lz],  # back-left-bottom
        [ lx,  ly,  lz],  # front-left-top
        [ lx, -ly,  lz],  # front-right-top
        [-lx, -ly,  lz],  # back-right-top
        [-lx,  ly,  lz],  # back-left-top
    ]).T  # (3, 8)
    # Rotate around z-axis by heading
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ corners
    corners[0] += x
    corners[1] += y
    corners[2] += z
    return corners  # (3, 8) in lidar frame


def lidar_corners_to_camera(corners_lidar, Tr_velo_to_cam, R0_rect):
    """Transform (3, 8) lidar corners to (3, 8) rectified camera corners."""
    corners_h = np.vstack([corners_lidar, np.ones((1, corners_lidar.shape[1]))])
    return R0_rect @ (Tr_velo_to_cam @ corners_h)  # (3, 8)


def project_to_image(pts_3d, P2):
    """Project (3, N) 3D points to (N, 2) image coords using P2 (3x4)."""
    pts_h = np.vstack([pts_3d, np.ones((1, pts_3d.shape[1]))])
    pts_2d = P2 @ pts_h
    pts_2d[:2] /= pts_2d[2]
    return pts_2d[:2].T  # (N, 2)


def draw_box3d_on_image(img, corners_2d, color=(0, 255, 0), thickness=2):
    """Draw 3D box wireframe given 8 projected corners.
    Connections: front face 0-1-2-3, back face 4-5-6-7, pillars 0-4,1-5,2-6,3-7
    """
    corners = corners_2d.astype(int)
    # Bottom face
    for i, j in [(0,1),(1,2),(2,3),(3,0)]:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    # Top face
    for i, j in [(4,5),(5,6),(6,7),(7,4)]:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    # Pillars
    for i, j in [(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)


def get_bev_corners(box_lidar):
    """box_lidar: [x, y, z, dx, dy, dz, heading] → (4, 2) BEV corners"""
    x, y, _, dx, dy, _, heading = box_lidar
    l, w = dx, dy
    c, s = np.cos(heading), np.sin(heading)
    corners = np.array([
        [ l/2*c - w/2*s,  l/2*s + w/2*c],
        [ l/2*c + w/2*s,  l/2*s - w/2*c],
        [-l/2*c + w/2*s, -l/2*s - w/2*c],
        [-l/2*c - w/2*s, -l/2*s + w/2*c],
    ])
    corners[:, 0] += x
    corners[:, 1] += y
    return corners  # (4, 2)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_camera(frame_id, detections, output_path, score_thresh=0.2):
    """Draw 3D boxes projected onto camera image."""
    img_path = KITTI_ROOT / 'image_2' / f'{frame_id}.png'
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')

    P2, R0_rect, Tr_velo_to_cam = load_calib(frame_id)

    names  = detections['name']
    scores = detections['score']
    boxes  = detections['boxes_lidar']  # (N, 7) lidar frame [x,y,z,dx,dy,dz,heading]

    color_map = {'Car': (0, 255, 0), 'Pedestrian': (255, 165, 0), 'Cyclist': (0, 165, 255)}

    n_drawn = 0
    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        color = color_map.get(names[i], (200, 200, 200))
        corners_lidar = corners_from_lidar_box(boxes[i])          # (3, 8) lidar frame
        corners_cam   = lidar_corners_to_camera(corners_lidar, Tr_velo_to_cam, R0_rect)  # (3, 8) cam frame
        corners_2d    = project_to_image(corners_cam, P2)         # (8, 2)

        # Skip if all corners behind camera
        if (corners_cam[2] < 0).all():
            continue

        draw_box3d_on_image(img, corners_2d, color=color, thickness=2)

        # Score label
        x1, y1 = int(corners_2d[:, 0].min()), int(corners_2d[:, 1].min())
        label = f"{names[i]} {scores[i]:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        n_drawn += 1

    cv2.putText(img, f'CLOCs Run2 | frame {frame_id} | {n_drawn} detections (score>{score_thresh})',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), img)
    print(f'  Saved camera view: {output_path}  ({n_drawn} boxes drawn)')


def visualize_bev(frame_id, detections, output_path, score_thresh=0.2,
                  x_range=(-40, 40), y_range=(0, 70)):
    """Bird's-Eye View: LiDAR points + 3D boxes."""
    bin_path = KITTI_ROOT / 'velodyne' / f'{frame_id}.bin'
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # Plot LiDAR points (BEV: x=forward, y=left in lidar frame)
    mask = ((pts[:, 0] > y_range[0]) & (pts[:, 0] < y_range[1]) &
            (pts[:, 1] > x_range[0]) & (pts[:, 1] < x_range[1]))
    pts_filt = pts[mask]
    height_norm = np.clip((pts_filt[:, 2] + 2) / 4.0, 0, 1)
    ax.scatter(-pts_filt[:, 1], pts_filt[:, 0], c=height_norm,
               cmap='plasma', s=0.3, alpha=0.6)

    # Draw 3D boxes
    color_map = {'Car': '#00ff00', 'Pedestrian': '#ffa500', 'Cyclist': '#00aaff'}
    names  = detections['name']
    scores = detections['score']
    boxes  = detections['boxes_lidar']  # [x,y,z,dx,dy,dz,heading] in lidar frame

    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        color = color_map.get(names[i], '#cccccc')
        corners = get_bev_corners(boxes[i])  # (4, 2) in lidar x,y
        # BEV plot: horizontal=y (left), vertical=x (forward)
        bev_x = -corners[:, 1]
        bev_y =  corners[:, 0]
        poly = plt.Polygon(np.column_stack([bev_x, bev_y]),
                           fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
        ax.add_patch(poly)
        # Direction indicator (front of box)
        front_x = -(corners[0, 1] + corners[1, 1]) / 2
        front_y =  (corners[0, 0] + corners[1, 0]) / 2
        cx, cy  = -boxes[i][1], boxes[i][0]
        ax.plot([cx, front_x], [cy, front_y], color=color, linewidth=1.5, alpha=0.9)
        # Label
        ax.text(cx, cy, f'{scores[i]:.2f}', color=color,
                fontsize=5, ha='center', va='center')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Y (left/right)', color='white')
    ax.set_ylabel('X (forward)', color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'CLOCs Run2 BEV | frame {frame_id} | score > {score_thresh}',
                 color='white', fontsize=11)
    ax.set_aspect('equal')

    # Ego vehicle marker
    ax.plot(0, 0, 'w^', markersize=8, label='ego')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved BEV view:    {output_path}')


# ── Pre-computed mode ─────────────────────────────────────────────────────────

def load_precomputed(frame_id):
    """Load detection result from pre-computed result/45.pt."""
    all_results = torch.load(str(PRECOMP_RESULT), weights_only=False)
    for det in all_results:
        if str(det['frame_id']) == frame_id:
            return det
    raise ValueError(f'Frame {frame_id} not found in pre-computed results '
                     f'(check that it is in the KITTI val split)')


# ── Live inference mode ───────────────────────────────────────────────────────

def run_live_inference(frame_id):
    """Full pipeline: PointPillars + YOLOv8 + CLOCs."""
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from pcdet.datasets import DatasetTemplate
    from tool.dataset import build_stage2_training
    from tool import fusion, nms as clocs_nms

    print('  [Live] Loading PointPillars...')
    cfg_from_yaml_file(str(PP_CFG), cfg)
    logger = common_utils.create_logger()

    # Build a minimal dataset to handle preprocessing
    class SingleFrameDataset(DatasetTemplate):
        def __init__(self, cfg, class_names, training=False, root_path=None, logger=None):
            super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=class_names,
                             training=training, root_path=root_path, logger=logger)
        def __len__(self): return 1
        def __getitem__(self, idx): return {}

    dataset = SingleFrameDataset(cfg, cfg.CLASS_NAMES, root_path=OPENPCDET/'data/kitti', logger=logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(PP_CKPT), logger=logger)
    model.cuda()
    model.eval()

    print('  [Live] Running PointPillars inference...')
    bin_path = KITTI_ROOT / 'velodyne' / f'{frame_id}.bin'
    calib_path = KITTI_ROOT / 'calib' / f'{frame_id}.txt'
    P2, R0_rect, Tr_velo_to_cam = load_calib(frame_id)

    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    input_dict = {'points': points, 'frame_id': frame_id}
    data_dict  = dataset.prepare_data(input_dict)
    data_dict  = dataset.collate_batch([data_dict])
    with torch.no_grad():
        from pcdet.utils.common_utils import load_data_to_gpu
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model(data_dict)

    pp_result = pred_dicts[0]  # keys: pred_boxes, pred_scores, pred_labels

    # Convert to CLOCs 3D format (same as loose_nms result.pkl format)
    from pcdet.datasets.kitti.kitti_utils import boxes3d_kitti_lidar_to_camera
    boxes_lidar  = pp_result['pred_boxes'].cpu().numpy()
    scores_raw   = pp_result['pred_scores'].cpu().numpy()

    # Project boxes to camera frame for CLOCs
    R0_ext  = np.eye(4); R0_ext[:3, :3] = R0_rect
    Tr_ext  = np.vstack([Tr_velo_to_cam, [0,0,0,1]])
    V2C     = R0_ext @ Tr_ext

    # Transform box centers lidar→camera
    centers_l  = np.hstack([boxes_lidar[:, :3], np.ones((len(boxes_lidar), 1))])
    centers_c  = (V2C @ centers_l.T).T[:, :3]

    # Project 3D boxes to get 2D bbox
    # (Simplified: use center projection + size estimate)
    pts_img = project_to_image(centers_c.T, P2)  # (N, 2)
    bbox_2d = np.hstack([pts_img - 20, pts_img + 20])  # rough bbox

    _3d_result = {
        'name':       np.array(['Car'] * len(scores_raw)),
        'truncated':  np.full(len(scores_raw), -1.0),
        'occluded':   np.full(len(scores_raw), -1, dtype=int),
        'alpha':      np.full(len(scores_raw), -10.0),
        'bbox':       bbox_2d.astype(np.float32),
        'dimensions': boxes_lidar[:, 3:6],
        'location':   centers_c.astype(np.float32),
        'rotation_y': boxes_lidar[:, 6].astype(np.float32),
        'score':      scores_raw.astype(np.float32),
        'boxes_lidar': boxes_lidar.astype(np.float32),
        'frame_id':   np.str_(frame_id),
    }

    print('  [Live] Running YOLOv8 inference...')
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_CKPT))
    img_path = KITTI_ROOT / 'image_2' / f'{frame_id}.png'
    results_yolo = yolo(str(img_path), conf=0.0, verbose=False)
    r = results_yolo[0]
    boxes_2d = r.boxes.xyxy.cpu().numpy()
    confs_2d = r.boxes.conf.cpu().numpy()
    # Convert to KITTI 2D format for CLOCs
    top_predictions = np.hstack([boxes_2d, confs_2d.reshape(-1, 1)])
    if len(top_predictions) > 200:
        sort_idx = np.argsort(-top_predictions[:, 4])
        top_predictions = top_predictions[sort_idx[:200]]

    print('  [Live] Running CLOCs fusion...')
    from tool.dataset import clocs_data
    # Build fusion input using train_stage_2 logic
    _dummy = clocs_data.__new__(clocs_data)
    res, iou_test, tensor_index = _dummy.train_stage_2(_3d_result, top_predictions)
    fusion_input = iou_test.unsqueeze(0).cuda()
    tensor_index = tensor_index.reshape(-1, 2).cuda()

    cfg_clocs = cfg.__class__()
    from pcdet.config import cfg_from_yaml_file
    cfg_from_yaml_file(str(CLOCS_CFG), cfg_clocs)

    net = torch.load(str(CLOCS_CKPT), weights_only=False)
    net.cuda(); net.eval()
    with torch.no_grad():
        fusion_cls_preds, _ = net(fusion_input, tensor_index)
    cls_preds = torch.sigmoid(fusion_cls_preds.reshape(-1).cpu())
    cls_preds = cls_preds[:len(_3d_result['score'])]
    _3d_result['score'] = cls_preds.detach().numpy()

    box_preds = torch.tensor(_3d_result['boxes_lidar']).cuda()
    selected  = clocs_nms.nms(cls_preds, box_preds, cfg_clocs.MODEL.POST_PROCESSING).numpy()
    for key in _3d_result:
        if key == 'frame_id':
            continue
        _3d_result[key] = _3d_result[key][selected]

    print(f'  [Live] Final detections: {len(_3d_result["score"])} boxes')
    return _3d_result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='CLOCs Demo — inference & visualisasi')
    parser.add_argument('--frame_id',   type=str, default='003219',
                        help='KITTI frame ID (6 digits, e.g. 003219)')
    parser.add_argument('--live',       action='store_true',
                        help='Run full pipeline (PointPillars+YOLOv8+CLOCs). '
                             'Default: use pre-computed result/45.pt')
    parser.add_argument('--score_thresh', type=float, default=0.3,
                        help='Score threshold for visualization (default 0.3)')
    parser.add_argument('--output_dir', type=str, default='output/demo',
                        help='Where to save output images')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    frame_id = args.frame_id.zfill(6)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nCLOCs Demo — frame {frame_id}')
    print(f'Mode: {"LIVE inference" if args.live else "pre-computed result/45.pt"}')
    print(f'Score threshold: {args.score_thresh}')
    print(f'Output dir: {out_dir}\n')

    # ── Get detections ──
    if args.live:
        detections = run_live_inference(frame_id)
    else:
        print(f'Loading pre-computed results from {PRECOMP_RESULT.name}...')
        detections = load_precomputed(frame_id)
        n = len(detections['score'])
        above = (detections['score'] >= args.score_thresh).sum()
        print(f'  Frame {frame_id}: {n} total boxes, {above} above score {args.score_thresh}')

    # ── Visualize ──
    cam_out = out_dir / f'{frame_id}_camera.jpg'
    bev_out = out_dir / f'{frame_id}_bev.jpg'

    visualize_camera(frame_id, detections, cam_out, score_thresh=args.score_thresh)
    visualize_bev(frame_id, detections, bev_out, score_thresh=args.score_thresh)

    print(f'\nDone.')
    print(f'  Camera: {cam_out}')
    print(f'  BEV   : {bev_out}')
