#!/usr/bin/env python3
"""Gambar 4.1 (kualitas kalibrasi: titik LiDAR di citra) & 4.2 (ilustrasi pelabelan 3D)."""
import pickle, numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
R='/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'; OUT=R+'/skripsi_bab/figures'; FID='000157'
plt.rcParams.update({'font.family':'serif','font.size':11})

def load_calib():
    d={}
    for ln in open(R+'/dataset_ugm/calib/000000.txt'):
        if ':' in ln: k,v=ln.split(':',1); d[k.strip()]=np.array([float(x) for x in v.split()])
    return d['P2'].reshape(3,4), d['R0_rect'].reshape(3,3), d['Tr_velo_to_cam'].reshape(3,4)
P2,R0,Tr=load_calib(); M=np.eye(4); M[:3,:3]=R0; M=M@np.vstack([Tr,[0,0,0,1]])
pts=np.fromfile(f'{R}/dataset_ugm/velodyne/{FID}.bin',dtype=np.float32).reshape(-1,4)

# ── Gambar 4.1: titik LiDAR diproyeksikan ke citra (warnai kedalaman) ────
img=cv2.cvtColor(cv2.imread(f'{R}/dataset_ugm/image_2/{FID}.png'),cv2.COLOR_BGR2RGB)
H,W=img.shape[:2]
pc=(M@np.hstack([pts[:,:3],np.ones((len(pts),1))]).T).T[:,:3]
front=pc[:,2]>0.5
uv=(P2@np.hstack([pc[front],np.ones((front.sum(),1))]).T).T; uv=uv[:,:2]/uv[:,2:3]
depth=pc[front,2]
ins=(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)&(depth<40)
fig,ax=plt.subplots(figsize=(9,5.2)); ax.imshow(img)
sc=ax.scatter(uv[ins,0],uv[ins,1],c=depth[ins],cmap='turbo',s=2.2,alpha=0.55)
cb=fig.colorbar(sc,ax=ax,fraction=0.03,pad=0.01); cb.set_label('jarak (m)',fontsize=9); cb.ax.tick_params(labelsize=8)
ax.axis('off')
fig.tight_layout(); fig.savefig(OUT+'/fig4_1_calib.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"fig4_1_calib.png OK, {ins.sum()} titik terproyeksi")

# ── Gambar 4.2: ilustrasi pelabelan 3D (perspektif point cloud+kuboid | citra acuan) ──
gt={inf['point_cloud']['lidar_idx']:inf for inf in pickle.load(open(R+'/dataset_ugm_m150_v3/kitti_infos_val.pkl','rb'))}[FID]
gb=np.asarray(gt['annos']['gt_boxes_lidar']); gn=np.asarray(gt['annos']['name'])
m=(np.hypot(gb[:,0],gb[:,1])<=25)&(gb[:,0]>0); gb,gn=gb[m],gn[m]
def corners(b):
    x,y,z,dx,dy,dz,yaw=b; c,s=np.cos(yaw),np.sin(yaw)
    xc=np.array([dx/2,dx/2,-dx/2,-dx/2]*2); yc=np.array([dy/2,-dy/2,-dy/2,dy/2]*2); zc=np.array([-dz/2]*4+[dz/2]*4)
    return (np.array([[c,-s,0],[s,c,0],[0,0,1]])@np.vstack([xc,yc,zc])).T+[x,y,z]
COL={'Car':'#00b050','Cyclist':'#ff8c00'}
fig=plt.figure(figsize=(12,4.8))
ax=fig.add_subplot(121,projection='3d'); ax.set_facecolor('white')
sub=pts[(pts[:,0]>0)&(pts[:,0]<30)&(np.abs(pts[:,1])<15)]
sub=sub[::4]
ax.scatter(sub[:,0],sub[:,1],sub[:,2],s=0.3,c=np.clip((sub[:,2]+2)/4,0,1),cmap='viridis',alpha=0.4)
E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
for b,n in zip(gb,gn):
    cc=corners(b)
    for a,c2 in E: ax.plot(*zip(cc[a],cc[c2]),color=COL.get(n,'gray'),lw=1.1)
ax.set_xlabel('x (m)',fontsize=9); ax.set_ylabel('y (m)',fontsize=9); ax.set_zlabel('z (m)',fontsize=9)
ax.set_xlim(0,30); ax.set_ylim(-15,15); ax.set_zlim(-3,3); ax.view_init(elev=22,azim=-70)
ax.set_box_aspect((3,3,0.8)); ax.set_title('(a) Kuboid 3D pada point cloud',fontsize=11); ax.tick_params(labelsize=7)
ax2=fig.add_subplot(122); ax2.imshow(img); ax2.axis('off'); ax2.set_title('(b) Citra kamera acuan',fontsize=11)
fig.tight_layout(); fig.savefig(OUT+'/fig4_2_label.png',dpi=200,bbox_inches='tight'); plt.close()
print(f"fig4_2_label.png OK, {len(gb)} kuboid")
