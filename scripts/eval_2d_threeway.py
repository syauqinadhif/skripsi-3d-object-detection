#!/usr/bin/env python3
"""Perbandingan 2D tiga pendekatan (kamera-saja / LiDAR-saja / fusi) via KITTI bbox AP (2D image).
PP-only & fusi: kotak 3D diproyeksikan ke 2D (box_utils). YOLO-only: deteksi 2D native.
Protokol & GT identik (≤25 m in-FOV, GT v3, AP_R40 Moderate). Semua di workstation.
"""
import os, sys, pickle, numpy as np, torch
R='/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'
sys.path.insert(0, R+'/OpenPCDet'); sys.path.insert(0, R+'/scripts')
from eval_range import build
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
GT = pickle.load(open(R+'/dataset_ugm_m150_v3/kitti_infos_val.pkl','rb'))
CLS=['Car','Pedestrian','Cyclist']

def bbox_ap(dt_annos, tag):
    ga, da = build(GT, dt_annos, 25.0, True)
    mo = np.zeros((3,3,6))
    for i,(c,y) in enumerate([(0.70,0.70),(0.50,0.50),(0.25,0.25)]):
        mo[i,:,0]=c; mo[i,:,2]=y
    res = kitti_eval.do_eval(ga, da, [0,2], mo[:,:,[0,2]])
    bbox = res[4]  # mAP_bbox_R40 [cls(2), diff(3), set(3)]
    return bbox

# ── YOLO-only: jalankan YOLO26 di 78 frame val, bangun anno ──────────────
def yolo_annos():
    from ultralytics import YOLO
    m = YOLO(R+'/output/yolo26_2cls/y26m_2cls/weights/best.pt')
    N2C=['Car','Cyclist']
    annos=[]
    for info in GT:
        fid=info['point_cloud']['lidar_idx']
        rr=m(R+f'/dataset_ugm/image_2/{fid}.png', conf=0.05, imgsz=1280, verbose=False)[0]
        b=rr.boxes.xyxy.cpu().numpy(); s=rr.boxes.conf.cpu().numpy(); c=rr.boxes.cls.cpu().numpy().astype(int)
        nm=np.array([N2C[x] for x in c])
        n=len(b)
        # boxes_lidar dummy dlm-jangkauan (x=10m) supaya lolos filter ≤25m; deteksi 2D dieval apa adanya
        bl=np.zeros((n,7),np.float32); bl[:,0]=10.0; bl[:,3:6]=1.0
        annos.append({'frame_id':fid,'name':nm,'truncated':np.zeros(n),'occluded':np.zeros(n,int),
            'alpha':-10*np.ones(n),'bbox':b.astype(np.float32),'dimensions':np.ones((n,3)),
            'location':np.tile([10,0,0],(n,1)).astype(np.float32),'rotation_y':np.zeros(n),
            'score':s.astype(np.float32),'boxes_lidar':bl})
    return annos

def load(p): return pickle.load(open(p,'rb'))

sets=[('@0,70',0),('@0,50',1),('@0,25',2)]
print("Menghitung bbox AP (2D) tiga pendekatan...\n")
tables={}
tables['YOLO-only (kamera)']=bbox_ap(yolo_annos(),'yolo')
tables['PointPillars-only (LiDAR)']=bbox_ap(load('/tmp/resultC_wspp.pkl'),'pp')
tables['Fusi (kamera-LiDAR)']=bbox_ap(load('/tmp/resultC_wsfu.pkl'),'fu')

MOD=1
print(f"=== bbox AP 2D (AP_R40 Moderate, ≤25 m in-FOV) ===")
print(f"{'pendekatan':30s}{'Car@0,70':>10s}{'Car@0,50':>10s}{'Cyc@0,50':>10s}{'Cyc@0,25':>10s}")
for k,bb in tables.items():
    car=[bb[0,MOD,0],bb[0,MOD,1]]; cyc=[bb[1,MOD,1],bb[1,MOD,2]]
    print(f"{k:30s}{car[0]:>10.2f}{car[1]:>10.2f}{cyc[0]:>10.2f}{cyc[1]:>10.2f}")
