#!/usr/bin/env python3
"""
Driver EVAL-ONLY CLOCs: jalankan model fusion terlatih (tanpa training) untuk
menghasilkan file result dengan location BENAR (diwarisi dari kandidat d3path cy440).
train.py tidak punya mode eval-only; ini mereplikasi eval() (train.py:103-124) saja.

Output location benar karena eval mewarisi field kamera dari file kandidat d3path
(dataset.py:182 -> path; train.py:107,116-119) = 3D_ugm_cy440 (sudah calib benar).
"""
import os, sys, argparse
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm

CLOCS = Path('/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/CLOCs_LQS')
os.chdir(str(CLOCS)); sys.path.insert(0, str(CLOCS))
from pcdet.config import cfg, cfg_from_yaml_file
from tool.dataset import clocs_data
from tool import nms
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='tool/cfgs/kitti_models/pointpillar_ugm/pointpillar_ugm_all.yaml')
ap.add_argument('--model', default='log/pointpillar_ugm/prenms_FULLgt_v8/49.pt')
ap.add_argument('--d2path', default='data/clocs_data/2D_ugm_cy440')
ap.add_argument('--d3path', default='data/clocs_data/3D_ugm_cy440')   # location BENAR (cy440)
ap.add_argument('--index', default='data/clocs_data/index/ugm_val.txt')
ap.add_argument('--input_data', default='data/clocs_data/input_data_ugm_prenms_v8')
ap.add_argument('--info', default='data/clocs_data/info/ugm_infos_trainval.pkl')
ap.add_argument('--out', default='log/pointpillar_ugm/prenms_FULLgt_v8/result/49_cy440fix.pt')
args = ap.parse_args()

cfg_from_yaml_file(args.cfg, cfg)
net = torch.load(args.model, weights_only=False).cuda().eval()
ds = clocs_data(args.d2path, args.d3path, args.index, args.input_data,
                infopath=args.info, val=True, class_names=cfg.CLASS_NAMES)
loader = DataLoader(ds, batch_size=1, num_workers=4)

det_annos = []
for fusion_input, tensor_index, path in tqdm(loader, desc='eval'):
    fusion_input = fusion_input.cuda()
    tensor_index = tensor_index.reshape(-1, 2).cuda()
    _3d = torch.load(path[0], weights_only=False)[0]              # kandidat: location BENAR
    with torch.no_grad():
        fusion_cls_preds, flag = net(fusion_input, tensor_index)
    cls_preds = torch.sigmoid(fusion_cls_preds.reshape(-1).cpu())
    cls_preds = cls_preds[:len(_3d['score'])]
    _3d['score'] = cls_preds.detach().numpy()
    box_preds = torch.tensor(_3d['boxes_lidar']).cuda()
    selected = nms.nms(cls_preds, box_preds, cfg.MODEL.POST_PROCESSING).numpy()
    for k in list(_3d.keys()):
        if k == 'frame_id':
            continue
        _3d[k] = _3d[k][selected]
    det_annos.append(_3d)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(det_annos, args.out)
n = sum(len(x['score']) for x in det_annos)
print(f"\nSaved {args.out} | {len(det_annos)} frame, {n} deteksi")
