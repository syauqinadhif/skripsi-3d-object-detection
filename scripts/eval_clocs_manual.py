#!/usr/bin/env python3
"""Eval prediksi CLOCs (cy440) vs GT manual (full & in-FOV). Reuse evaluator KITTI."""
import os, pickle, sys
import numpy as np, torch
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'OpenPCDet', 'tools'))
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

GT_PKL='../data/ugm/kitti_infos_val80_manual.pkl'
CLOCS=sys.argv[1] if len(sys.argv)>1 else '../../CLOCs_LQS/log/pointpillar_ugm/yolov8_ugm_cy440/result/49.pt'
CLASSES=['Car','Pedestrian','Cyclist']
GK=['name','truncated','occluded','alpha','bbox','dimensions','location','rotation_y']

def vmask(b):
    b=np.asarray(b).reshape(-1,4); return (b[:,2]-b[:,0]>=1)&(b[:,3]-b[:,1]>=1)
def sub(a,m,keys):
    o={}
    for k in keys:
        v=np.asarray(a[k]); o[k]=v[m] if v.shape[0]==m.shape[0] else v
    return o

gt=pickle.load(open(GT_PKL,'rb'))
dt=torch.load(CLOCS, weights_only=False)
dtmap={d['frame_id']:d for d in dt}

# diagnosa skor
allsc=np.concatenate([np.asarray(d['score']) for d in dt if len(d['score'])])
print(f"CLOCs deteksi: {len(allsc)} total | score min={allsc.min():.3f} max={allsc.max():.3f} med={np.median(allsc):.3f}")
gtcars=sum(1 for i in gt for n in i['annos']['name'] if n=='Car')
print(f"GT manual (80 frame): {gtcars} Car\n")

def run(infov):
    ga,da=[],[]
    for info in gt:
        fid=info['point_cloud']['lidar_idx']; a=info['annos']; d=dtmap.get(fid)
        if d is None: continue
        gm=vmask(a['bbox']) if infov else np.ones(len(a['name']),bool)
        dm=vmask(d['bbox']) if infov else np.ones(len(d['name']),bool)
        g=sub(a,gm,GK); g['score']=np.zeros(gm.sum()); ga.append(g)
        da.append(sub(d,dm,GK+['score']))
    tag='IN-FOV' if infov else 'FULL'
    ng=sum(len(x['name']) for x in ga); nd=sum(len(x['name']) for x in da)
    print(f"===== CLOCs vs GT manual — {tag} (GT={ng}, pred={nd}) =====")
    s,_=kitti_eval.get_official_eval_result(ga,da,CLASSES)
    for ln in s.splitlines():
        if '3d   AP' in ln or 'bev  AP' in ln or ln.strip().endswith(':') and 'AP' in ln:
            print(ln)
    print()

run(False); run(True)
