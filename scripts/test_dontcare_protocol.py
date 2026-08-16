"""Uji: apakah DontCare benar-benar membuat evaluator MENGABAIKAN prediksi yg menimpanya?
Bandingkan 3 perlakuan GT jauh: (a) DontCare, (b) dihapus, (c) dibiarkan dinilai."""
import sys, pickle, numpy as np
sys.argv=['x','--dt','dummy']
sys.path.insert(0,'scripts')
import importlib.util
spec=importlib.util.spec_from_file_location('er','/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi/scripts/eval_range.py')
er=importlib.util.module_from_spec(spec)
import builtins
_argv=sys.argv; sys.argv=['eval_range.py','--dt','x']
try: spec.loader.exec_module(er)
except SystemExit: pass
sys.argv=_argv
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
import numpy as np
GT='../data/ugm/kitti_infos_val80_manual.pkl'
DT='../output/kitti_models/pointpillar_ugm_m150/ep130_itermatched/eval/eval_all_default/default/epoch_130/val/result.pkl'
gt=pickle.load(open(GT,'rb')); dt=pickle.load(open(DT,'rb'))
MAXR=25.0
def build(mode, filter_pred=True):
    dtmap={d['frame_id']:d for d in dt}; GA=[];DA=[]
    for info in gt:
        ga=info['annos']; d=dtmap[info['point_cloud']['lidar_idx']]
        gm=er.valid_bbox_mask(ga['bbox']); dm=er.valid_bbox_mask(d['bbox'])
        grng=er.boxes_range(ga,'gt_boxes_lidar'); drng=er.boxes_range(d,'boxes_lidar')
        if filter_pred and len(drng)==len(dm): dm=dm&(drng<=MAXR)
        if len(grng)==len(gm) and gm.sum():
            far=grng[gm]>MAXR
        else: far=np.zeros(int(gm.sum()),bool)
        g=er.subset(ga,gm,er.KEYS); g['score']=np.zeros(gm.sum())
        g['name']=np.asarray(g['name'],dtype='<U20')
        if mode=='dontcare': g['name']=g['name'].copy(); g['name'][far]='DontCare'
        elif mode=='hapus':
            keep=~far; g={k:(np.asarray(v)[keep] if np.asarray(v).shape[0]==len(keep) else v) for k,v in g.items()}
        GA.append(g); DA.append(er.subset(d,dm,er.KEYS+['score']))
    return GA,DA
print("A. prediksi jauh DIBUANG (protokol eval_range.py):")
cases=[('dontcare',True),('hapus',True),('dinilai',True),
       ('SEP',True),
       ('dontcare',False),('hapus',False),('dinilai',False)]
for mode,fp in cases:
    if mode=='SEP':
        print("B. prediksi jauh DIPERTAHANKAN (uji apakah DontCare benar2 menekan FP):"); continue
    GA,DA=build(mode,fp)
    s,_=kitti_eval.get_official_eval_result(GA,DA,er.CLASSES)
    out={}
    lines=s.splitlines()
    for i,l in enumerate(lines):
        if l.strip().startswith('Car AP_R40@0.70, 0.70'):
            for j in range(i+1,i+5):
                if lines[j].strip().startswith('3d   AP:'): out['Car3d@.7']=float(lines[j].split(':')[1].split(',')[1])
        if l.strip().startswith('Cyclist AP_R40@0.50, 0.50'):
            for j in range(i+1,i+5):
                if lines[j].strip().startswith('3d   AP:'): out['Cyc3d@.5']=float(lines[j].split(':')[1].split(',')[1])
    print(f"  GT jauh '{mode:9}' -> Car 3D@0.7={out.get('Car3d@.7',float('nan')):6.2f}  Cyc 3D@0.5={out.get('Cyc3d@.5',float('nan')):6.2f}")
