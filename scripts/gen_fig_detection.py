#!/usr/bin/env python3
"""Gambar 5.1 (contoh dataset + GT) & Gambar 5.3 (PP-only vs fusi) untuk naskah."""
import sys, pickle, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'
sys.path.insert(0,R+'/OpenPCDet')
OUT=R+'/skripsi_bab/figures'; FID='000157'; FID3='000001'   # 5.1 padat objek; 5.3 bersih
COL={'Car':(0,180,0),'Cyclist':(0,140,255),'Pedestrian':(230,120,0),'GT':(0,210,0)}

def load_calib():
    d={}
    for ln in open(R+'/dataset_ugm/calib/000000.txt'):
        if ':' in ln: k,v=ln.split(':',1); d[k.strip()]=np.array([float(x) for x in v.split()])
    return d['P2'].reshape(3,4), d['R0_rect'].reshape(3,3), d['Tr_velo_to_cam'].reshape(3,4)
P2,R0,Tr=load_calib()
M=np.eye(4); M[:3,:3]=R0; M=M@np.vstack([Tr,[0,0,0,1]])

def corners(b):
    x,y,z,dx,dy,dz,yaw=b; c,s=np.cos(yaw),np.sin(yaw)
    xc=np.array([dx/2,dx/2,-dx/2,-dx/2]*2); yc=np.array([dy/2,-dy/2,-dy/2,dy/2]*2)
    zc=np.array([-dz/2]*4+[dz/2]*4)
    Rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (Rz@np.vstack([xc,yc,zc])).T+[x,y,z]

def draw_cam(img,boxes,names,scores=None,thr=0.3):
    for i,b in enumerate(boxes):
        if names[i]=='Pedestrian': continue          # Pedestrian di-exclude dari studi
        if scores is not None and scores[i]<thr: continue
        cl=corners(b); cc=(M@np.hstack([cl,np.ones((8,1))]).T).T[:,:3]
        if (cc[:,2]<=0.1).any(): continue
        im=(P2@np.hstack([cc,np.ones((8,1))]).T).T; im=im[:,:2]/im[:,2:3]
        col=COL.get(names[i],(200,200,200)); p=im.astype(int)
        for a,c2 in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            cv2.line(img,tuple(p[a]),tuple(p[c2]),col,2)
    return img

# ── Gambar 5.1: contoh dataset (kamera+GT | BEV+GT) ──────────────────────
gt={inf['point_cloud']['lidar_idx']:inf for inf in pickle.load(open(R+'/dataset_ugm_m150_v3/kitti_infos_val.pkl','rb'))}[FID]
gb=np.asarray(gt['annos']['gt_boxes_lidar']); gn=np.asarray(gt['annos']['name'])
m=np.hypot(gb[:,0],gb[:,1])<=25; gb,gn=gb[m],gn[m]
img=cv2.imread(f'{R}/dataset_ugm/image_2/{FID}.png'); img=draw_cam(img.copy(),gb,gn)
pts=np.fromfile(f'{R}/dataset_ugm/velodyne/{FID}.bin',dtype=np.float32).reshape(-1,4)
fig,axs=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.6,1]})
axs[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); axs[0].axis('off'); axs[0].set_title('(a) Citra kamera + ground truth 3D',fontsize=11)
ax=axs[1]; ax.set_facecolor('#111827')
mk=(pts[:,0]>0)&(pts[:,0]<40)&(np.abs(pts[:,1])<20)
ax.scatter(pts[mk,1],pts[mk,0],s=0.25,c=np.clip((pts[mk,2]+2)/4,0,1),cmap='viridis',alpha=0.6)
for b,n in zip(gb,gn):
    cc=corners(b)[:4]; cc=np.vstack([cc,cc[0]])
    ax.plot(cc[:,1],cc[:,0],color=np.array(COL['GT'][::-1])/255,lw=1.3)
ax.set_xlim(20,-20); ax.set_ylim(0,40); ax.set_xlabel('y (m)'); ax.set_ylabel('x (m)',labelpad=-2)
ax.set_title('(b) Bird\'s-eye view + ground truth',fontsize=11); ax.set_aspect('equal')
ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT+'/fig5_1_dataset.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig5_1_dataset.png OK,",len(gb),"GT")

# ── Gambar 5.3: PP-only vs fusi ──────────────────────────────────────────
pp={a['frame_id']:a for a in pickle.load(open('/tmp/resultC_wspp.pkl','rb'))}[FID3]
fu={a['frame_id']:a for a in pickle.load(open('/tmp/resultC_wsfu.pkl','rb'))}[FID3]
base=cv2.imread(f'{R}/dataset_ugm/image_2/{FID3}.png')
ip=draw_cam(base.copy(),np.asarray(pp['boxes_lidar']),np.asarray(pp['name']),np.asarray(pp['score']),0.3)
iff=draw_cam(base.copy(),np.asarray(fu['boxes_lidar']),np.asarray(fu['name']),np.asarray(fu['score']),0.3)
fig,axs=plt.subplots(1,2,figsize=(12,3.8))
for ax,im,t in [(axs[0],ip,'(a) PointPillars-only (LiDAR)'),(axs[1],iff,'(b) Fusi kamera-LiDAR (CLOCs)')]:
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.axis('off'); ax.set_title(t,fontsize=11)
fig.tight_layout(); fig.savefig(OUT+'/fig5_3_ppvsfusi.png',dpi=200,bbox_inches='tight'); plt.close()
np_pp=(np.asarray(pp['score'])>=0.3).sum(); np_fu=(np.asarray(fu['score'])>=0.3).sum()
print(f"fig5_3_ppvsfusi.png OK, PP {np_pp} box, fusi {np_fu} box (score>0.3)")
