"""Scan sekuens UGM: hitung deteksi PointPillars yakin per frame (Car/Cyclist),
lalu pilih jendela kontigu terbaik untuk video demo."""
import os, sys, json
import numpy as np, torch
from pathlib import Path
ROOT = Path('/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi')
OPENPCDET = ROOT / 'OpenPCDet'
sys.path.insert(0, str(OPENPCDET))
UGM = ROOT / 'dataset_ugm'
PP_CKPT = OPENPCDET / 'output/kitti_models/pointpillar_ugm_m300_v3/finetune_kitti_m300/ckpt/checkpoint_epoch_120.pth'
PP_CFG  = OPENPCDET / 'tools/cfgs/kitti_models/pointpillar_ugm_m300_prenms_val.yaml'
CONF = 0.45   # ambang skor PP mentah (terkalibrasi) utk "objek nyata"

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

cwd = os.getcwd(); os.chdir(str(OPENPCDET/'tools')); cfg_from_yaml_file(str(PP_CFG), cfg); os.chdir(cwd)
logger = common_utils.create_logger()
class OneFrame(DatasetTemplate):
    def __init__(s): super().__init__(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                      training=False, root_path=OPENPCDET/'data/ugm', logger=logger)
    def __len__(s): return 1
    def __getitem__(s,i): return {}
ds = OneFrame()
pp = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
pp.load_params_from_file(filename=str(PP_CKPT), logger=logger); pp.cuda().eval()

N = 500
strong = np.zeros(N); clutter = np.zeros(N); present = np.zeros(N, bool)
for i in range(N):
    fid = f'{i:06d}'
    binp = UGM/'velodyne'/f'{fid}.bin'
    if not binp.exists(): continue
    present[i] = True
    pts = np.fromfile(str(binp), dtype=np.float32).reshape(-1,4)
    dd = ds.collate_batch([ds.prepare_data({'points':pts,'frame_id':fid})])
    with torch.no_grad():
        load_data_to_gpu(dd); pred,_ = pp(dd)
    b = pred[0]['pred_boxes'].cpu().numpy(); s = pred[0]['pred_scores'].cpu().numpy()
    d = np.hypot(b[:,0], b[:,1]); inr = d<=40
    strong[i]  = int(((s>=0.60)&inr).sum())            # deteksi yakin (kemungkinan besar nyata)
    clutter[i] = int(((s>=0.20)&(s<0.45)&inr).sum())   # deteksi medium (kemungkinan FP, lolos thr fusi)
    if i%50==0: print(f'  {i}: strong={strong[i]:.0f} clutter={clutter[i]:.0f}')

# kualitas frame: banyak strong, sedikit clutter; butuh minimal 2 strong utk bukan frame kosong
q = strong - 0.8*clutter
q[strong < 2] = -5
q[~present] = -10

for W in (80,100,120):
    best_s, best_i = -1e9, 0
    for a in range(0, N-W+1):
        w = q[a:a+W]
        if (w<=-10).any(): continue
        sc = w.sum()
        if sc>best_s: best_s, best_i = sc, a
    seg = slice(best_i,best_i+W)
    print(f'\nW={W}: rentang TERBERSIH [{best_i}, {best_i+W}) '
          f'strong/frame={strong[seg].mean():.1f} clutter/frame={clutter[seg].mean():.1f} skor={best_s:.0f}')
json.dump({'strong':strong.tolist(),'clutter':clutter.tolist(),'present':present.tolist()},
          open('/tmp/scan_frames.json','w'))
print('\nSimpan /tmp/scan_frames.json')
