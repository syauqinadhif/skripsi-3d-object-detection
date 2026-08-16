#!/usr/bin/env python3
"""
Deteksi frame yang gambar & point cloud-nya TIDAK SINKRON.

Temuan 2026-07-20: ~separuh frame UGM punya kamera-LiDAR desync di DATA MENTAH. Bukti: dgn
ekstrinsik yang dipas-kan pada frame sinkron (galat sisa 34 px), frame bermasalah tetap
meleset 629 px — tak ada kalibrasi yang bisa memperbaikinya.

Cara kerja: proyeksikan box 3D GT (Car) lalu ukur jarak ke deteksi COCO YOLO terdekat
(COCO independen dari calib). Frame sinkron -> jarak kecil; desync -> ratusan piksel.

Frame desync HARUS dibuang dari: label 2D YOLO, cabang 2D CLOCs, dan evaluasi fusi.
Tidak memengaruhi PointPillars (LiDAR saja).
"""
import pickle, sys
import numpy as np
from ultralytics import YOLO

ROOT='/media/mf/AUTODRIVING-4TB/Downloads/project_skripsi'
IMG_W,IMG_H=1280,720
FX,FY,CX,CY=699.47418510,707.80363724,630.32152421,360.0
P2=np.array([[FX,0,CX,0],[0,FY,CY,0],[0,0,1,0]])
TR=np.load(sys.argv[1] if len(sys.argv)>1 else f'{ROOT}/dataset_ugm/tr_synced.npy')
TR4=np.vstack([TR,[0,0,0,1]])
COCO={2:'Car',7:'Car',5:'Car'}

def corners(b):
    x,y,z,dx,dy,dz,yaw=b[:7]
    c=np.array([[dx/2,dx/2,-dx/2,-dx/2]*2,[dy/2,-dy/2,-dy/2,dy/2]*2,[-dz/2]*4+[dz/2]*4])
    R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return (R@c).T+np.array([x,y,z])

def classify(pkls, out_txt):
    m=YOLO('yolov8m.pt'); rows=[]
    for pkl in pkls:
        for info in pickle.load(open(pkl,'rb')):
            fid=info['point_cloud']['lidar_idx']
            a=info['annos']; gb=np.asarray(a['gt_boxes_lidar'])
            if len(gb)==0: continue
            nm=np.asarray(a['name'],dtype=str)[:len(gb)]
            keep=[k for k in range(len(gb)) if nm[k]=='Car' and gb[k,0]>2 and np.hypot(gb[k,0],gb[k,1])<=30]
            if len(keep)<2: rows.append((fid,np.nan,len(keep))); continue
            r=m(f'{ROOT}/dataset_ugm/image_2/{fid}.png',conf=0.30,verbose=False)[0]
            dets=[b.xyxy[0].cpu().numpy() for b in r.boxes if COCO.get(int(b.cls[0]))]
            if not dets: rows.append((fid,np.nan,len(keep))); continue
            dc=np.array([[(d[0]+d[2])/2,(d[1]+d[3])/2] for d in dets])
            ds=[]
            for k in keep:
                cor=corners(gb[k])
                cam=(TR4@np.hstack([cor,np.ones((8,1))]).T).T[:,:3]
                if (cam[:,2]<=0.5).any(): continue
                pr=(P2@np.hstack([cam,np.ones((8,1))]).T).T; uv=pr[:,:2]/pr[:,2:3]
                u=(uv[:,0].min()+uv[:,0].max())/2; v=(uv[:,1].min()+uv[:,1].max())/2
                ds.append(np.sqrt(((dc[:,0]-u)**2+(dc[:,1]-v)**2).min()))
            rows.append((fid, np.median(ds) if ds else np.nan, len(keep)))
    ok=[f for f,d,n in rows if not np.isnan(d) and d<80]
    bad=[f for f,d,n in rows if not np.isnan(d) and d>=200]
    unk=[f for f,d,n in rows if np.isnan(d) or (80<=d<200)]
    open(out_txt,'w').write('\n'.join(sorted(ok))+'\n')
    print(f"total dinilai {len(rows)} frame:")
    print(f"  SINKRON  (<80 px) : {len(ok)}  -> ditulis ke {out_txt}")
    print(f"  DESYNC  (>=200 px): {len(bad)}")
    print(f"  ragu/tak terukur  : {len(unk)}")
    return ok,bad,unk

if __name__=='__main__':
    classify([f'{ROOT}/dataset_ugm_m150/kitti_infos_train.pkl'], f'{ROOT}/dataset_ugm/ImageSets/synced_train.txt')
    classify([f'{ROOT}/OpenPCDet/data/ugm/kitti_infos_val78_manual_v2.pkl'], f'{ROOT}/dataset_ugm/ImageSets/synced_val.txt')
