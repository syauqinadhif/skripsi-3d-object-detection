#!/usr/bin/env python3
"""Eval CLOCs vs GT manual, TAPI 'location' dihitung ulang dari boxes_lidar pakai calib cy440.
Menguji apakah 3D~0 disebabkan calib output salah (bukan kandidat 3D float)."""
import os, pickle, sys
import numpy as np, torch
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'OpenPCDet', 'tools'))
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

GT_PKL='../data/ugm/kitti_infos_val80_manual.pkl'
CLOCS=sys.argv[1]
CALIB='../data/ugm/calib/000001.txt'
IMG_W,IMG_H=1280,720
CLASSES=['Car','Pedestrian','Cyclist']

def load_calib(p):
    d={}
    for ln in open(p):
        if ':' in ln:
            k,v=ln.split(':',1); d[k.strip()]=np.array([float(x) for x in v.split()])
    return d['P2'].reshape(3,4), d['R0_rect'].reshape(3,3), d['Tr_velo_to_cam'].reshape(3,4)

P2,R0,Tr=load_calib(CALIB)
Tr4=np.vstack([Tr,[0,0,0,1]]); R4=np.eye(4); R4[:3,:3]=R0

def reproj(bl):  # bl (N,7) lidar [x,y,z,dx,dy,dz,yaw] -> KITTI camera annos
    c=np.hstack([bl[:,:3],np.ones((len(bl),1))]); cc=(R4@Tr4@c.T).T[:,:3]
    dx,dy,dz=bl[:,3],bl[:,4],bl[:,5]; h,w,l=dz,dy,dx
    ry=(-bl[:,6]-np.pi/2+np.pi)%(2*np.pi)-np.pi
    loc=cc.copy(); loc[:,1]+=h/2
    bbox=[]
    for i in range(len(bl)):
        x,y,z=loc[i]; R=np.array([[np.cos(ry[i]),0,np.sin(ry[i])],[0,1,0],[-np.sin(ry[i]),0,np.cos(ry[i])]])
        xc=[l[i]/2,l[i]/2,-l[i]/2,-l[i]/2]*2; yc=[0,0,0,0,-h[i],-h[i],-h[i],-h[i]]
        zc=[w[i]/2,-w[i]/2,-w[i]/2,w[i]/2]*2
        cor=(R@np.array([xc,yc,zc])).T+[x,y,z]
        v=cor[:,2]>0.1
        if v.sum()<1: bbox.append([0,0,0,0]); continue
        hom=np.hstack([cor[v],np.ones((v.sum(),1))]); im=(P2@hom.T).T; im=im[:,:2]/im[:,2:3]
        x1,y1=im.min(0); x2,y2=im.max(0)
        bbox.append([np.clip(x1,0,IMG_W),np.clip(y1,0,IMG_H),np.clip(x2,0,IMG_W),np.clip(y2,0,IMG_H)])
    return loc,np.stack([l,h,w],1),ry,np.array(bbox).reshape(-1,4)

gt=pickle.load(open(GT_PKL,'rb')); dt=torch.load(CLOCS,weights_only=False)
dtmap={d['frame_id']:d for d in dt}
GK=['name','truncated','occluded','alpha','bbox','dimensions','location','rotation_y']
def vmask(b): b=np.asarray(b).reshape(-1,4); return (b[:,2]-b[:,0]>=1)&(b[:,3]-b[:,1]>=1)
def sub(a,m,keys):
    o={};
    for k in keys:
        v=np.asarray(a[k]); o[k]=v[m] if v.shape[0]==m.shape[0] else v
    return o

def run(infov):
    ga,da=[],[]
    for info in gt:
        fid=info['point_cloud']['lidar_idx']; a=info['annos']; d=dtmap.get(fid)
        if d is None: continue
        bl=np.asarray(d['boxes_lidar'])
        if len(bl)==0:
            da.append({k:np.zeros((0,4) if k=='bbox' else (0,3) if k in('dimensions','location') else 0) for k in GK+['score']})
        else:
            loc,dim,ry,bbox=reproj(bl)
            dd={'name':np.asarray(d['name']),'truncated':np.zeros(len(bl)),'occluded':np.zeros(len(bl)),
                'alpha':np.zeros(len(bl)),'bbox':bbox,'dimensions':dim,'location':loc,'rotation_y':ry,
                'score':np.asarray(d['score'])}
            dm=vmask(bbox) if infov else np.ones(len(bl),bool)
            dd=sub(dd,dm,GK+['score']); da.append(dd)
        gm=vmask(a['bbox']) if infov else np.ones(len(a['name']),bool)
        g=sub(a,gm,GK); g['score']=np.zeros(gm.sum()); ga.append(g)
    tag='IN-FOV' if infov else 'FULL'
    print(f"===== CLOCs REPROJ(cy440) vs manual — {tag} =====")
    s,_=kitti_eval.get_official_eval_result(ga,da,CLASSES)
    for ln in s.splitlines():
        if '3d   AP' in ln or 'bev  AP' in ln or (ln.strip().endswith(':') and 'AP' in ln): print(ln)
    print()

run(True)
