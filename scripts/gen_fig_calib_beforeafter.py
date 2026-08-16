#!/usr/bin/env python3
"""Gambar 4.1: perbandingan kalibrasi SEBELUM (lama, calib_pre_v3_backup) vs SESUDAH (v3).
Titik LiDAR diproyeksikan ke citra; kalibrasi benar → titik menempel pada objek."""
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'; OUT=R+'/skripsi_bab/figures'; FID='000157'
plt.rcParams.update({'font.family':'serif'})

def load_calib(path):
    d={}
    for ln in open(path):
        if ':' in ln: k,v=ln.split(':',1); d[k.strip()]=np.array([float(x) for x in v.split()])
    P2=d['P2'].reshape(3,4); R0=d['R0_rect'].reshape(3,3); Tr=d['Tr_velo_to_cam'].reshape(3,4)
    M=np.eye(4); M[:3,:3]=R0; M=M@np.vstack([Tr,[0,0,0,1]])
    return P2,M

img=cv2.cvtColor(cv2.imread(f'{R}/dataset_ugm/image_2/{FID}.png'),cv2.COLOR_BGR2RGB)
H,W=img.shape[:2]
pts=np.fromfile(f'{R}/dataset_ugm/velodyne/{FID}.bin',dtype=np.float32).reshape(-1,4)

def project(P2,M):
    pc=(M@np.hstack([pts[:,:3],np.ones((len(pts),1))]).T).T[:,:3]
    front=pc[:,2]>0.5
    uv=(P2@np.hstack([pc[front],np.ones((front.sum(),1))]).T).T; uv=uv[:,:2]/uv[:,2:3]
    depth=pc[front,2]; ins=(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)&(depth<40)
    return uv[ins],depth[ins]

P2o,Mo=load_calib(R+'/dataset_ugm/calib_pre_v3_backup/'+FID+'.txt')   # SEBELUM (lama)
P2n,Mn=load_calib(R+'/dataset_ugm/calib/'+FID+'.txt')                 # SESUDAH (v3)
uo,do_=project(P2o,Mo); un,dn=project(P2n,Mn)

fig,axs=plt.subplots(1,2,figsize=(13,3.9))
for ax,(u,dep),t in [(axs[0],(uo,do_),'(a) Kalibrasi lama (intrinsik KITTI, ekstrinsik placeholder)'),
                     (axs[1],(un,dn),'(b) Kalibrasi v3 (intrinsik rosbag + ekstrinsik PnP+RANSAC)')]:
    ax.imshow(img); ax.scatter(u[:,0],u[:,1],c=dep,cmap='turbo',s=2.0,alpha=0.55)
    ax.axis('off'); ax.set_title(t,fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT+'/fig4_1_calib.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"fig4_1_calib.png (before/after) OK — lama {len(uo)} titik, v3 {len(un)} titik")
