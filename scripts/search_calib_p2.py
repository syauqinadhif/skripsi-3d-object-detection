"""
Search for optimal P2 calibration parameters by maximizing 2D IoU between
projected GT 3D boxes and manually-annotated GT 2D bboxes.

Usage:
    python scripts/search_calib_p2.py --pkl dataset_ugm/kitti_infos_val.pkl
    python scripts/search_calib_p2.py --pkl dataset_ugm/kitti_infos_val.pkl --mode 2d  # also search cx,fx,fy
"""

import pickle
import numpy as np
import argparse


def box3d_corners_camera(l, h, w, x, y, z, ry):
    """8 corners of KITTI 3D box in camera space. (x,y,z) = bottom center."""
    # Unrotated corners relative to bottom center
    cx = [l/2, l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    cy = [0,    0,    0,    0,    -h,   -h,   -h,   -h  ]  # 0=bottom, -h=top
    cz = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    corners = np.array([cx, cy, cz], dtype=np.float64)

    R = np.array([[ np.cos(ry), 0, np.sin(ry)],
                  [ 0,          1, 0          ],
                  [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    corners = R @ corners
    corners[0] += x
    corners[1] += y
    corners[2] += z
    return corners  # 3×8


def project_corners(corners, fx, fy, cx, cy):
    """Project 3×8 corners using simple pinhole (no distortion)."""
    z = corners[2]
    # Only keep corners with z > 0.1
    valid = z > 0.1
    if valid.sum() < 4:
        return None
    u = fx * corners[0, valid] / z[valid] + cx
    v = fy * corners[1, valid] / z[valid] + cy
    return np.array([u, v])  # 2×N


def projected_bbox(pts):
    """2D bbox enclosing projected corners."""
    if pts is None:
        return None
    return np.array([pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()])


def iou_2d(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def collect_boxes(frames):
    """Return list of (location, dims, ry, gt_bbox) for all valid GT boxes."""
    pairs = []
    for fr in frames:
        ann = fr['annos']
        names = ann['name']
        locs  = ann['location']   # Nx3
        dims  = ann['dimensions'] # Nx3 : [l, h, w] in OpenPCDet
        rys   = ann['rotation_y']
        bboxes= ann['bbox']       # Nx4 : [x1,y1,x2,y2]
        for i, name in enumerate(names):
            if name == 'DontCare':
                continue
            z = locs[i, 2]
            if z < 0.5:
                continue
            pairs.append((locs[i], dims[i], rys[i], bboxes[i]))
    return pairs


def score_params(pairs, fx, fy, cx, cy, img_w=1280, img_h=720):
    ious = []
    for loc, dim, ry, gt_box in pairs:
        # OpenPCDet annos['dimensions'] order: [l, h, w]
        l, h, w = dim[0], dim[1], dim[2]
        corners = box3d_corners_camera(l, h, w, loc[0], loc[1], loc[2], ry)
        pts = project_corners(corners, fx, fy, cx, cy)
        pb = projected_bbox(pts)
        if pb is None:
            continue
        # Clip to image
        pb = np.array([max(0, pb[0]), max(0, pb[1]),
                       min(img_w, pb[2]), min(img_h, pb[3])])
        if pb[2] <= pb[0] or pb[3] <= pb[1]:
            continue
        ious.append(iou_2d(pb, gt_box))
    return np.mean(ious) if ious else 0.0, len(ious)


def search_cy(pairs, fx, fy, cx, cy_range, img_w=1280, img_h=720):
    results = []
    for cy in cy_range:
        mean_iou, n = score_params(pairs, fx, fy, cx, cy, img_w, img_h)
        results.append((cy, mean_iou, n))
    return results


def search_cx(pairs, fx, fy, cy, cx_range, img_w=1280, img_h=720):
    results = []
    for cx in cx_range:
        mean_iou, n = score_params(pairs, fx, fy, cx, cy, img_w, img_h)
        results.append((cx, mean_iou, n))
    return results


def search_fx(pairs, fy, cx, cy, fx_range, img_w=1280, img_h=720):
    results = []
    for fx in fx_range:
        mean_iou, n = score_params(pairs, fx, fy, cx, cy, img_w, img_h)
        results.append((fx, mean_iou, n))
    return results


def refine(pairs, base_fx, base_fy, base_cx, base_cy, step, n_steps, img_w=1280, img_h=720):
    """Fine-grained search around a good starting point for both cy and cx."""
    best = (base_cy, base_cx, 0.0)
    cy_vals = np.arange(base_cy - n_steps*step, base_cy + n_steps*step + step, step)
    cx_vals = np.arange(base_cx - n_steps*step, base_cx + n_steps*step + step, step)
    for cy in cy_vals:
        for cx in cx_vals:
            iou, _ = score_params(pairs, base_fx, base_fy, cx, cy, img_w, img_h)
            if iou > best[2]:
                best = (cy, cx, iou)
    return best


def print_table(header, results, param_name):
    print(f"\n{'='*55}")
    print(f"  {header}")
    print(f"{'='*55}")
    print(f"  {param_name:>8s}  mean_IoU   n_boxes")
    print(f"  {'-'*35}")
    best_iou = max(r[1] for r in results)
    for val, iou, n in results:
        marker = " <-- best" if abs(iou - best_iou) < 1e-9 else ""
        print(f"  {val:8.1f}  {iou:.6f}   {n}{marker}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='dataset_ugm/kitti_infos_val.pkl')
    parser.add_argument('--img_w', type=int, default=1280)
    parser.add_argument('--img_h', type=int, default=720)
    parser.add_argument('--mode', choices=['cy', '2d'], default='cy',
                        help='cy=search only cy; 2d=search cy+cx+fx')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as f:
        frames = pickle.load(f)
    print(f"Loaded {len(frames)} frames from {args.pkl}")

    # Use base P2 from first frame
    P2 = frames[0]['calib']['P2']
    base_fx = P2[0, 0]
    base_fy = P2[1, 1]
    base_cx = P2[0, 2]
    base_cy = P2[1, 2]
    print(f"Base P2: fx={base_fx:.2f}, fy={base_fy:.2f}, cx={base_cx:.2f}, cy={base_cy:.2f}")

    pairs = collect_boxes(frames)
    print(f"Collected {len(pairs)} GT boxes for scoring\n")

    # --- STEP 1: coarse cy search ---
    print("Step 1: Coarse cy search (step=20) ...")
    cy_coarse = np.arange(0, 721, 20)
    res_cy = search_cy(pairs, base_fx, base_fy, base_cx, cy_coarse, args.img_w, args.img_h)
    print_table("Coarse cy search (cx=base, fx=base, fy=base)", res_cy, "cy")
    best_cy_coarse = max(res_cy, key=lambda x: x[1])[0]

    # --- STEP 2: fine cy search ---
    print(f"\nStep 2: Fine cy search around cy={best_cy_coarse} (step=5) ...")
    cy_fine = np.arange(best_cy_coarse - 40, best_cy_coarse + 41, 5)
    res_cy_fine = search_cy(pairs, base_fx, base_fy, base_cx, cy_fine, args.img_w, args.img_h)
    print_table("Fine cy search", res_cy_fine, "cy")
    best_cy = max(res_cy_fine, key=lambda x: x[1])[0]

    if args.mode == '2d':
        # --- STEP 3: coarse cx search with best cy ---
        print(f"\nStep 3: Coarse cx search with cy={best_cy} (step=20) ...")
        cx_coarse = np.arange(200, 1081, 20)
        res_cx = search_cx(pairs, base_fx, base_fy, best_cy, cx_coarse, args.img_w, args.img_h)
        print_table(f"Coarse cx search (cy={best_cy})", res_cx, "cx")
        best_cx_coarse = max(res_cx, key=lambda x: x[1])[0]

        # --- STEP 4: fine cx search ---
        print(f"\nStep 4: Fine cx search around cx={best_cx_coarse} (step=5) ...")
        cx_fine = np.arange(best_cx_coarse - 40, best_cx_coarse + 41, 5)
        res_cx_fine = search_cx(pairs, base_fx, base_fy, best_cy, cx_fine, args.img_w, args.img_h)
        print_table("Fine cx search", res_cx_fine, "cx")
        best_cx = max(res_cx_fine, key=lambda x: x[1])[0]

        # --- STEP 5: fx search with best cy, cx ---
        print(f"\nStep 5: fx search with cy={best_cy}, cx={best_cx} (step=20) ...")
        fx_range = np.arange(300, 1201, 20)
        res_fx = search_fx(pairs, base_fy, best_cx, best_cy, fx_range, args.img_w, args.img_h)
        print_table(f"fx search (cy={best_cy}, cx={best_cx})", res_fx, "fx")
        best_fx = max(res_fx, key=lambda x: x[1])[0]

        # --- STEP 6: joint refinement ---
        print(f"\nStep 6: Joint cy+cx fine refinement around cy={best_cy}, cx={best_cx} ...")
        best_cy2, best_cx2, best_iou2 = refine(
            pairs, best_fx, base_fy, best_cx, best_cy,
            step=2, n_steps=10, img_w=args.img_w, img_h=args.img_h
        )
        print(f"  Best joint: cy={best_cy2}, cx={best_cx2}, mean_IoU={best_iou2:.6f}")

        print(f"\n{'='*60}")
        print(f"FINAL RECOMMENDATION:")
        print(f"  fx = {best_fx:.2f}  (was {base_fx:.2f})")
        print(f"  fy = {base_fy:.2f}  (unchanged)")
        print(f"  cx = {best_cx2:.2f}  (was {base_cx:.2f})")
        print(f"  cy = {best_cy2:.2f}  (was {base_cy:.2f})")
        print(f"  mean_IoU = {best_iou2:.6f}")
        print(f"\nP2 matrix:")
        print(f"  [[{best_fx:.2f}, 0, {best_cx2:.2f}, 0],")
        print(f"   [0, {base_fy:.2f}, {best_cy2:.2f}, 0],")
        print(f"   [0, 0, 1, 0],")
        print(f"   [0, 0, 0, 1]]")
    else:
        print(f"\n{'='*60}")
        print(f"FINAL RECOMMENDATION (cy only):")
        print(f"  cy = {best_cy:.2f}  (was {base_cy:.2f})")
        base_iou, n = score_params(pairs, base_fx, base_fy, base_cx, base_cy,
                                    args.img_w, args.img_h)
        best_iou, _ = score_params(pairs, base_fx, base_fy, base_cx, best_cy,
                                    args.img_w, args.img_h)
        print(f"  mean_IoU baseline (cy={base_cy:.0f}): {base_iou:.6f}")
        print(f"  mean_IoU optimal  (cy={best_cy:.0f}): {best_iou:.6f}")

    # --- Show per-class breakdown at optimal params ---
    print(f"\n--- Per-class IoU breakdown at optimal cy ---")
    if args.mode == '2d':
        opt_fx, opt_cx, opt_cy = best_fx, best_cx2, best_cy2
    else:
        opt_fx, opt_cx, opt_cy = base_fx, base_cx, best_cy

    for cls_name in ['Car', 'Pedestrian', 'Cyclist', 'Motorcycle']:
        cls_pairs = []
        for fr in frames:
            ann = fr['annos']
            for i, name in enumerate(ann['name']):
                if name != cls_name:
                    continue
                z = ann['location'][i, 2]
                if z < 0.5:
                    continue
                cls_pairs.append((ann['location'][i], ann['dimensions'][i],
                                   ann['rotation_y'][i], ann['bbox'][i]))
        if not cls_pairs:
            continue
        iou, n = score_params(cls_pairs, opt_fx, base_fy, opt_cx, opt_cy,
                               args.img_w, args.img_h)
        print(f"  {cls_name:12s}: mean_IoU={iou:.4f}  n={n}")


if __name__ == '__main__':
    main()
