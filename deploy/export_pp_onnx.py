#!/usr/bin/env python3
"""
Export PointPillars (checkpoint OpenPCDet kita) → ONNX untuk CUDA-PointPillars / TensorRT.

Kenapa wrapper: CUDA-PointPillars `tool/export_onnx.py` mengandalkan pcdet commit 846cf3e +
torch 1.x (dict-arg diperlakukan positional). Kita pakai pcdet 0.6.0 + torch 2.0.1 (dict-arg
jadi kwargs → `forward() missing batch_dict`). Wrapper di bawah membungkus model jadi
forward(voxels, voxel_num, voxel_coords) → (cls_preds, box_preds, dir_preds) = RAW dense-head
output (sebelum decode/NMS), persis yang dibutuhkan graf CUDA-PointPillars. Voxelisasi &
augmentasi pillar 10-channel + decode + NMS dikerjakan CUDA-PP di sisi C++/CUDA.

HANYA untuk model voxel 0.16 (m300 v0.16) yang cocok params CUDA-PointPillars
(POINT_CLOUD_RANGE [0,-39.68,-3,69.12,39.68,1], VOXEL 0.16, 3 kelas). vox010 TIDAK cocok
tanpa modif params.h CUDA-PP.

Usage (dari root project):
  PYTHONPATH=OpenPCDet /home/mf/miniconda3/envs/openpcdet/bin/python deploy/export_pp_onnx.py \
    --ckpt OpenPCDet/output/kitti_models/pointpillar_ugm_m300_v3/ppbaseline_m300/ckpt/checkpoint_epoch_100.pth \
    --cfg CUDA-PointPillars/cfgs/kitti_models/pointpillar.yaml \
    --out models/pointpillar_m300.onnx
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

MAX_VOXELS = 10000   # sama dgn CUDA-PointPillars export_onnx.py


class _DS(DatasetTemplate):
    def __init__(self, dcfg, names):
        super().__init__(dataset_cfg=dcfg, class_names=names, training=False,
                         root_path=Path('.'), logger=common_utils.create_logger())
    def __len__(self): return 0
    def __getitem__(self, i): return {}


class PPExport(nn.Module):
    """Bungkus PointPillar → forward tensor positional, keluarkan raw head preds."""
    def __init__(self, model):
        super().__init__()
        self.vfe, self.scatter, self.backbone, self.head = model.module_list
    def forward(self, voxels, voxel_num_points, voxel_coords):
        bd = {'voxels': voxels, 'voxel_num_points': voxel_num_points,
              'voxel_coords': voxel_coords, 'batch_size': 1}
        bd = self.vfe(bd)
        bd = self.scatter(bd)
        bd = self.backbone(bd)
        x = bd['spatial_features_2d']
        return self.head.conv_cls(x), self.head.conv_box(x), self.head.conv_dir_cls(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default=str(ROOT / 'CUDA-PointPillars/cfgs/kitti_models/pointpillar.yaml'))
    ap.add_argument('--out', default=str(ROOT / 'models/pointpillar_m300.onnx'))
    ap.add_argument('--simplify', action='store_true',
                    help='jalankan onnxsim (default OFF; fusi Gemm bisa merusak graf PFN 3D, TRT optimize sendiri)')
    args = ap.parse_args()

    # _BASE_CONFIG_ di cfg CUDA-PP relatif → chdir ke sana saat load
    cfg_dir = Path(args.cfg).resolve().parents[2]   # .../CUDA-PointPillars
    cwd = os.getcwd(); os.chdir(cfg_dir)
    cfg_from_yaml_file(str(Path(args.cfg).resolve()), cfg)
    os.chdir(cwd)

    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=_DS(cfg.DATA_CONFIG, cfg.CLASS_NAMES))
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()

    wrap = PPExport(model).cuda().eval()
    voxels = torch.zeros((MAX_VOXELS, 32, 4), dtype=torch.float32, device='cuda')
    vnum   = torch.zeros((MAX_VOXELS,), dtype=torch.int32, device='cuda')
    vcoord = torch.zeros((MAX_VOXELS, 4), dtype=torch.int32, device='cuda')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrap, (voxels, vnum, vcoord), args.out,
            export_params=True, opset_version=11, do_constant_folding=True,
            keep_initializers_as_inputs=False,
            input_names=['voxels', 'voxel_num', 'voxel_idxs'],
            output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
        )
    print(f"\nONNX raw tersimpan: {args.out}")
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(args.out)
            ms, ok = simplify(m)
            if ok:
                onnx.save(ms, args.out); print("onnxsim: OK")
            else:
                print("onnxsim: gagal cek, pakai versi raw")
        except Exception as e:
            print(f"onnxsim dilewati: {e}")

    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print("onnx.checker: VALID")
    print("Inputs :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("Outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])


if __name__ == '__main__':
    main()
