"""
demo_clocs_ugm.py — Inference & Visualisasi CLOCs pada Dataset UGM
                     (model dilatih dengan kalibrasi KITTI yang salah)

Mode 1 (default, cepat):
    Menggunakan hasil pre-computed dari result/49.pt (CLOCs UGM epoch 49).

Mode 2 (--live):
    Menjalankan full inference pipeline dari awal:
    LiDAR .bin -> PointPillars (fine-tuned UGM) -> 3D candidates
    Image      -> YOLOv8 (trained UGM)           -> 2D candidates
                                                  -> CLOCs Fusion -> Final 3D Boxes

Output:
    <output_dir>/<frame_id>_camera.jpg  — 3D boxes diproyeksikan ke gambar kamera
    <output_dir>/<frame_id>_bev.jpg     — Bird's-Eye View dengan LiDAR points + boxes

Usage:
    # Mode cepat (pre-computed, epoch 49):
    conda run -n openpcdet python scripts/demo_clocs_ugm.py --frame_id 000136

    # Live inference:
    conda run -n openpcdet python scripts/demo_clocs_ugm.py --frame_id 000136 --live

    # Turunkan threshold untuk melihat lebih banyak box:
    conda run -n openpcdet python scripts/demo_clocs_ugm.py --frame_id 000136 --score_thresh 0.05
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
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi')
OPENPCDET    = PROJECT_ROOT / 'OpenPCDet'
CLOCS_LQS    = PROJECT_ROOT / 'CLOCs_LQS'
UGM_ROOT     = PROJECT_ROOT / 'dataset_ugm'

PP_CKPT      = OPENPCDET / 'output/kitti_models/pointpillar_ugm/finetune_kitti/ckpt/checkpoint_epoch_30.pth'
PP_CFG       = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar_ugm_prenms.yaml'
YOLO_CKPT    = PROJECT_ROOT / 'output/yolov8/ugm_pseudo/weights/best.pt'
CLOCS_CKPT   = CLOCS_LQS / 'log/pointpillar_ugm/yolov8_ugm/49.pt'
CLOCS_CFG    = CLOCS_LQS / 'tool/cfgs/kitti_models/pointpillar_ugm/pointpillar_ugm_all.yaml'
PRECOMP_RESULT = CLOCS_LQS / 'log/pointpillar_ugm/yolov8_ugm/result/49.pt'

# CLOCs_LQS first for 'tool' package, but OpenPCDet must win for 'pcdet'
sys.path.insert(0, str(CLOCS_LQS))
sys.path.insert(0, str(OPENPCDET))

# Will be set from args
CALIB_DIR = UGM_ROOT / 'calib'

# ── Calibration helpers ───────────────────────────────────────────────────────

def load_calib(frame_id):
    """Load UGM calib file (KITTI format)."""
    calib_path = CALIB_DIR / f'{frame_id}.txt'
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in val.split()])
    P2             = calib['P2'].reshape(3, 4)
    R0_rect        = calib['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam


# ── 3D box geometry utilities ─────────────────────────────────────────────────

def corners_from_lidar_box(box_lidar):
    """Compute 8 corners (3,8) in lidar frame from [x,y,z,dx,dy,dz,heading]."""
    x, y, z, dx, dy, dz, heading = box_lidar
    lx, ly, lz = dx / 2, dy / 2, dz / 2
    corners = np.array([
        [ lx,  ly, -lz],
        [ lx, -ly, -lz],
        [-lx, -ly, -lz],
        [-lx,  ly, -lz],
        [ lx,  ly,  lz],
        [ lx, -ly,  lz],
        [-lx, -ly,  lz],
        [-lx,  ly,  lz],
    ]).T  # (3, 8)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = R @ corners
    corners[0] += x; corners[1] += y; corners[2] += z
    return corners


def lidar_corners_to_camera(corners_lidar, Tr_velo_to_cam, R0_rect):
    corners_h = np.vstack([corners_lidar, np.ones((1, corners_lidar.shape[1]))])
    return R0_rect @ (Tr_velo_to_cam @ corners_h)


def project_to_image(pts_3d, P2):
    pts_h = np.vstack([pts_3d, np.ones((1, pts_3d.shape[1]))])
    pts_2d = P2 @ pts_h
    pts_2d[:2] /= pts_2d[2]
    return pts_2d[:2].T  # (N, 2)


def draw_box3d_on_image(img, corners_2d, color=(0, 255, 0), thickness=2):
    corners = corners_2d.astype(int)
    for i, j in [(0,1),(1,2),(2,3),(3,0)]:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    for i, j in [(4,5),(5,6),(6,7),(7,4)]:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)
    for i, j in [(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)


def get_bev_corners(box_lidar):
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

def visualize_camera(frame_id, detections, output_path, score_thresh=0.05):
    img_path = UGM_ROOT / 'image_2' / f'{frame_id}.png'
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')

    P2, R0_rect, Tr_velo_to_cam = load_calib(frame_id)

    names  = detections['name']
    scores = detections['score']
    boxes  = detections['boxes_lidar']

    color_map = {'Car': (0, 255, 0), 'Pedestrian': (255, 165, 0), 'Cyclist': (0, 165, 255)}

    n_drawn = 0
    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        color = color_map.get(names[i], (200, 200, 200))
        corners_lidar = corners_from_lidar_box(boxes[i])
        corners_cam   = lidar_corners_to_camera(corners_lidar, Tr_velo_to_cam, R0_rect)
        corners_2d    = project_to_image(corners_cam, P2)

        if (corners_cam[2] < 0).all():
            continue

        draw_box3d_on_image(img, corners_2d, color=color, thickness=2)

        x1 = int(corners_2d[:, 0].min())
        y1 = int(corners_2d[:, 1].min())
        label = f"{names[i]} {scores[i]:.3f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        n_drawn += 1

    note = f'CLOCs | {CALIB_DIR.name}'
    cv2.putText(img, f'{note} | frame {frame_id} | {n_drawn} det (score>{score_thresh})',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), img)
    print(f'  Saved camera view: {output_path}  ({n_drawn} boxes drawn)')


def visualize_bev(frame_id, detections, output_path, score_thresh=0.05,
                  x_range=(-40, 40), y_range=(0, 70)):
    bin_path = UGM_ROOT / 'velodyne' / f'{frame_id}.bin'
    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)

    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    mask = ((pts[:, 0] > y_range[0]) & (pts[:, 0] < y_range[1]) &
            (pts[:, 1] > x_range[0]) & (pts[:, 1] < x_range[1]))
    pts_filt = pts[mask]
    height_norm = np.clip((pts_filt[:, 2] + 2) / 4.0, 0, 1)
    ax.scatter(-pts_filt[:, 1], pts_filt[:, 0], c=height_norm,
               cmap='plasma', s=0.3, alpha=0.6)

    color_map = {'Car': '#00ff00', 'Pedestrian': '#ffa500', 'Cyclist': '#00aaff'}
    names  = detections['name']
    scores = detections['score']
    boxes  = detections['boxes_lidar']

    for i in range(len(scores)):
        if scores[i] < score_thresh:
            continue
        color = color_map.get(names[i], '#cccccc')
        corners = get_bev_corners(boxes[i])
        bev_x = -corners[:, 1]
        bev_y =  corners[:, 0]
        poly = plt.Polygon(np.column_stack([bev_x, bev_y]),
                           fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
        ax.add_patch(poly)
        front_x = -(corners[0, 1] + corners[1, 1]) / 2
        front_y =  (corners[0, 0] + corners[1, 0]) / 2
        cx, cy  = -boxes[i][1], boxes[i][0]
        ax.plot([cx, front_x], [cy, front_y], color=color, linewidth=1.5, alpha=0.9)
        ax.text(cx, cy, f'{scores[i]:.3f}', color=color,
                fontsize=5, ha='center', va='center')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('Y (left/right)', color='white')
    ax.set_ylabel('X (forward)', color='white')
    ax.tick_params(colors='white')
    ax.set_title(f'CLOCs | {CALIB_DIR.name} | frame {frame_id} | score > {score_thresh}',
                 color='white', fontsize=11)
    ax.set_aspect('equal')
    ax.plot(0, 0, 'w^', markersize=8, label='ego')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Saved BEV view:    {output_path}')


# ── Pre-computed mode ─────────────────────────────────────────────────────────

def load_precomputed(frame_id):
    all_results = torch.load(str(PRECOMP_RESULT), weights_only=False)
    for det in all_results:
        if str(det['frame_id']) == frame_id:
            return det
    raise ValueError(f'Frame {frame_id} not found in pre-computed results '
                     f'({PRECOMP_RESULT}). Check UGM val split.')


# ── Live inference mode ───────────────────────────────────────────────────────

def run_live_inference(frame_id):
    import os
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils
    from pcdet.datasets import DatasetTemplate
    from tool import fusion, nms as clocs_nms

    print('  [Live] Loading PointPillars (UGM fine-tuned)...')
    _orig_dir = os.getcwd()
    os.chdir(str(OPENPCDET / 'tools'))
    cfg_from_yaml_file(str(PP_CFG), cfg)
    os.chdir(_orig_dir)
    logger = common_utils.create_logger()

    class SingleFrameDataset(DatasetTemplate):
        def __init__(self, cfg, class_names, training=False, root_path=None, logger=None):
            super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=class_names,
                             training=training, root_path=root_path, logger=logger)
        def __len__(self): return 1
        def __getitem__(self, idx): return {}

    dataset = SingleFrameDataset(cfg, cfg.CLASS_NAMES,
                                 root_path=OPENPCDET / 'data/ugm', logger=logger)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(PP_CKPT), logger=logger)
    model.cuda()
    model.eval()

    print('  [Live] Running PointPillars inference...')
    P2, R0_rect, Tr_velo_to_cam = load_calib(frame_id)
    points = np.fromfile(str(UGM_ROOT / 'velodyne' / f'{frame_id}.bin'), dtype=np.float32).reshape(-1, 4)

    input_dict = {'points': points, 'frame_id': frame_id}
    data_dict  = dataset.prepare_data(input_dict)
    data_dict  = dataset.collate_batch([data_dict])
    with torch.no_grad():
        from pcdet.models import load_data_to_gpu
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model(data_dict)

    pp_result    = pred_dicts[0]
    boxes_lidar  = pp_result['pred_boxes'].cpu().numpy()
    scores_raw   = pp_result['pred_scores'].cpu().numpy()

    R0_ext = np.eye(4); R0_ext[:3, :3] = R0_rect
    Tr_ext = np.vstack([Tr_velo_to_cam, [0, 0, 0, 1]])
    V2C    = R0_ext @ Tr_ext
    centers_l = np.hstack([boxes_lidar[:, :3], np.ones((len(boxes_lidar), 1))])
    centers_c = (V2C @ centers_l.T).T[:, :3]
    pts_img   = project_to_image(centers_c.T, P2)
    bbox_2d   = np.hstack([pts_img - 20, pts_img + 20])

    _3d_result = {
        'name':        np.array(['Car'] * len(scores_raw)),
        'truncated':   np.full(len(scores_raw), -1.0),
        'occluded':    np.full(len(scores_raw), -1, dtype=int),
        'alpha':       np.full(len(scores_raw), -10.0),
        'bbox':        bbox_2d.astype(np.float32),
        'dimensions':  boxes_lidar[:, 3:6],
        'location':    centers_c.astype(np.float32),
        'rotation_y':  boxes_lidar[:, 6].astype(np.float32),
        'score':       scores_raw.astype(np.float32),
        'boxes_lidar': boxes_lidar.astype(np.float32),
        'frame_id':    np.str_(frame_id),
    }

    print('  [Live] Running YOLOv8 (UGM) inference...')
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_CKPT))
    results_yolo = yolo(str(UGM_ROOT / 'image_2' / f'{frame_id}.png'), conf=0.0, verbose=False)
    r = results_yolo[0]
    boxes_2d = r.boxes.xyxy.cpu().numpy()
    confs_2d = r.boxes.conf.cpu().numpy()
    top_predictions = np.hstack([boxes_2d, confs_2d.reshape(-1, 1)])
    if len(top_predictions) > 200:
        sort_idx = np.argsort(-top_predictions[:, 4])
        top_predictions = top_predictions[sort_idx[:200]]

    print('  [Live] Running CLOCs fusion...')
    from tool.dataset import clocs_data
    _dummy = clocs_data.__new__(clocs_data)
    res, iou_test, tensor_index = _dummy.train_stage_2(_3d_result, top_predictions)
    fusion_input = iou_test.reshape(4, 1, -1).unsqueeze(0).cuda()  # [1, 4, 1, N]
    tensor_index = tensor_index.reshape(-1, 2).cuda()

    from pcdet.config import cfg as cfg_clocs
    from pcdet.config import cfg_from_yaml_file as cfg_from_yaml_file2
    os.chdir(str(CLOCS_LQS))
    cfg_from_yaml_file2(str(CLOCS_CFG), cfg_clocs)
    os.chdir(_orig_dir)

    net = torch.load(str(CLOCS_CKPT), weights_only=False)
    net.cuda(); net.eval()
    with torch.no_grad():
        fusion_cls_preds, _ = net(fusion_input, tensor_index)
    cls_preds = torch.sigmoid(fusion_cls_preds.reshape(-1).cpu())
    cls_preds = cls_preds[:len(_3d_result['score'])]
    _3d_result['score'] = cls_preds.detach().numpy()

    box_preds = torch.tensor(_3d_result['boxes_lidar']).cuda()
    selected  = clocs_nms.nms(cls_preds.cuda(), box_preds, cfg_clocs.MODEL.POST_PROCESSING).cpu().numpy()
    for key in _3d_result:
        if key == 'frame_id':
            continue
        _3d_result[key] = _3d_result[key][selected]

    print(f'  [Live] Final: {len(_3d_result["score"])} boxes after NMS')
    return _3d_result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='CLOCs UGM Demo — KITTI-Calib mismatch experiment')
    parser.add_argument('--frame_id',     type=str,   default='000136',
                        help='UGM frame ID (6 digits). Default: 000136')
    parser.add_argument('--live',         action='store_true',
                        help='Run full live pipeline instead of using pre-computed result/49.pt')
    parser.add_argument('--score_thresh', type=float, default=0.05,
                        help='Score threshold for visualization (default 0.05)')
    parser.add_argument('--output_dir',   type=str,   default='output/demo_ugm',
                        help='Where to save output images')
    parser.add_argument('--calib_dir',    type=str,   default=None,
                        help='Calib directory override (e.g. dataset_ugm/calib_opt). '
                             'Default: dataset_ugm/calib (KITTI values)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    frame_id = args.frame_id.zfill(6)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set calib directory (module-level var, no global needed at module scope)
    if args.calib_dir:
        CALIB_DIR = Path(args.calib_dir)
        if not CALIB_DIR.is_absolute():
            CALIB_DIR = PROJECT_ROOT / CALIB_DIR
        calib_label = f'OPTIMIZED ({CALIB_DIR.name})'
    else:
        CALIB_DIR = UGM_ROOT / 'calib'
        calib_label = 'KITTI (salah untuk UGM)'

    print(f'\nCLOCs Demo UGM — frame {frame_id}')
    print(f'Mode     : {"LIVE inference" if args.live else "pre-computed result/49.pt"}')
    print(f'Calib    : {calib_label}')
    print(f'Threshold: {args.score_thresh}')
    print(f'Output   : {out_dir}\n')

    if args.live:
        detections = run_live_inference(frame_id)
    else:
        print(f'Loading pre-computed results from {PRECOMP_RESULT.name}...')
        detections = load_precomputed(frame_id)
        n     = len(detections['score'])
        above = (detections['score'] >= args.score_thresh).sum()
        print(f'  Frame {frame_id}: {n} boxes total, {above} above score {args.score_thresh}')
        print(f'  Score range: min={detections["score"].min():.4f}, max={detections["score"].max():.4f}')

    cam_out = out_dir / f'{frame_id}_camera.jpg'
    bev_out = out_dir / f'{frame_id}_bev.jpg'

    visualize_camera(frame_id, detections, cam_out, score_thresh=args.score_thresh)
    visualize_bev(frame_id, detections, bev_out, score_thresh=args.score_thresh)

    print(f'\nDone.')
    print(f'  Camera : {cam_out}')
    print(f'  BEV    : {bev_out}')
