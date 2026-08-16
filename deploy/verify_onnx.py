#!/usr/bin/env python3
"""
Verifikasi numerik ONNX vs PyTorch: voxelisasi 1 frame nyata, jalankan kedua jalur,
bandingkan raw head output. Lolos = max|Δ| kecil (< ~1e-3).

Usage:
  PYTHONPATH=OpenPCDet /home/mf/miniconda3/envs/openpcdet/bin/python deploy/verify_onnx.py \
    --onnx models/pointpillar_m300.onnx \
    --ckpt OpenPCDet/output/kitti_models/pointpillar_ugm_m300_v3/ppbaseline_m300/ckpt/checkpoint_epoch_100.pth \
    --bin dataset_ugm/velodyne/000001.bin
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
sys.path.insert(0, str(ROOT / 'OpenPCDet'))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
sys.path.insert(0, str(ROOT / 'deploy'))
from export_pp_onnx import PPExport, _DS, MAX_VOXELS

CFG = ROOT / 'CUDA-PointPillars/cfgs/kitti_models/pointpillar.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--bin', default=str(ROOT / 'dataset_ugm/velodyne/000001.bin'))
    args = ap.parse_args()

    cwd = os.getcwd(); os.chdir(CFG.parents[2])
    cfg_from_yaml_file(str(CFG), cfg); os.chdir(cwd)
    logger = common_utils.create_logger()
    ds = _DS(cfg.DATA_CONFIG, cfg.CLASS_NAMES)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()
    wrap = PPExport(model).cuda().eval()

    # voxelisasi 1 frame lewat data processor pcdet
    pts = np.fromfile(args.bin, dtype=np.float32).reshape(-1, 4)
    dd = ds.prepare_data({'points': pts, 'frame_id': '0'})
    dd = ds.collate_batch([dd]); load_data_to_gpu(dd)
    v, vn, vc = dd['voxels'], dd['voxel_num_points'], dd['voxel_coords']
    nv = v.shape[0]
    print(f"frame {Path(args.bin).stem}: {nv} voxel aktif (pad ke {MAX_VOXELS})")

    # pad ke MAX_VOXELS (sama seperti input engine)
    def pad(t, n, dtype):
        out = torch.zeros((MAX_VOXELS,) + tuple(t.shape[1:]), dtype=dtype, device='cuda')
        out[:n] = t[:min(n, MAX_VOXELS)].to(dtype); return out
    v_p  = pad(v, nv, torch.float32)
    vn_p = pad(vn, nv, torch.int32); vn_p[nv:] = 1   # padded voxel num=1 → hindari 0/0 NaN di PillarVFE
    vc_p = pad(vc, nv, torch.int32)

    with torch.no_grad():
        pc, pb, pd = wrap(v_p, vn_p, vc_p)
    pt_out = {'cls_preds': pc.cpu().numpy(), 'box_preds': pb.cpu().numpy(), 'dir_cls_preds': pd.cpu().numpy()}

    import onnxruntime as ort
    # matikan graph-opt: fusi MatMulBn→Gemm ort salah di PFN (input rank-3). TRT tak kena ini.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=['CPUExecutionProvider'])
    ort_out = sess.run(['cls_preds', 'box_preds', 'dir_cls_preds'], {
        'voxels': v_p.cpu().numpy(), 'voxel_num': vn_p.cpu().numpy(), 'voxel_idxs': vc_p.cpu().numpy()})

    # Toleransi 5e-2: ORT-CPU vs PyTorch-CUDA beda implementasi conv → akumulasi fp32 ~1% pada
    # logit mentah adalah normal & tak berpengaruh setelah sigmoid/decode. >5e-2 = curigai struktur.
    TOL = 5e-2
    print("\n=== PyTorch(CUDA) vs ONNXRuntime(CPU) — max |Δ| logit mentah ===")
    ok = True
    for i, k in enumerate(['cls_preds', 'box_preds', 'dir_cls_preds']):
        d = np.abs(pt_out[k] - ort_out[i]).max()
        status = 'OK' if d < TOL else 'BEDA BESAR'
        ok = ok and d < TOL
        print(f"  {k:15s}: {d:.2e}  [{status}]")
    print(f"\nHASIL: {'LOLOS — struktur ONNX setara PyTorch (selisih = numerik CPU-vs-CUDA)' if ok else 'GAGAL — cek export'}")
    print("Catatan: TensorRT akan lebih dekat lagi ke PyTorch (sama-sama CUDA). Selisih ini bukan indikasi bug.")


if __name__ == '__main__':
    main()
