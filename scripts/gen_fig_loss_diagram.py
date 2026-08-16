#!/usr/bin/env python3
"""Generate Gambar 5.2 (kurva loss) & Gambar 4.3 (diagram pipeline CLOCs) untuk naskah.
Gaya figur akademik cetak: garis tipis, grid resesif, sumbu berlabel, aman grayscale."""
import re, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT='/media/mf/AUTODRIVING-4TB1/Downloads/project_skripsi'
OUT=ROOT+'/skripsi_bab/figures'
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.linewidth':0.8})
INK='#1a1a1a'; ACCENT='#2b6cb0'; MUTED='#888888'

# ── Gambar 5.2: kurva loss fine-tuning ──────────────────────────────────
ep, loss = [], []
for ln in open(ROOT+'/output/train_finetune_m300.log'):
    m = re.search(r'Train:\s+(\d+)/130\s+\([^)]*\)\s+\[\s*(\d+)/(\d+)', ln)
    l = re.search(r'Loss:\s+([\d.]+)', ln)
    if m and l:
        e=int(m.group(1)); it=int(m.group(2)); tot=int(m.group(3))
        ep.append(e-1 + it/max(tot,1)); loss.append(float(l.group(1)))
ep=np.array(ep); loss=np.array(loss)
fig,ax=plt.subplots(figsize=(7,4.2))
ax.plot(ep,loss,color=MUTED,lw=0.6,alpha=0.45)                 # raw (samar)
# smoothing rata-rata bergerak
k=15; sm=np.convolve(loss,np.ones(k)/k,mode='valid'); eps=ep[k-1:]
ax.plot(eps,sm,color=INK,lw=1.8,label='Loss pelatihan (rata-rata bergerak)')
ax.axvline(120,color=ACCENT,lw=1.2,ls='--')
ax.annotate('epoch 120\n(model terpilih)',xy=(120,sm[np.searchsorted(eps,120)]),
            xytext=(95,1.15),fontsize=9.5,color=ACCENT,
            arrowprops=dict(arrowstyle='->',color=ACCENT,lw=1))
ax.set_xlabel('Epoch'); ax.set_ylabel('Total loss')
ax.set_xlim(0,130); ax.set_ylim(0,max(loss[:80].max(),1.6))
ax.grid(True,lw=0.4,color='#dddddd',alpha=0.9); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(frameon=False,fontsize=9.5,loc='upper right')
fig.tight_layout(); fig.savefig(OUT+'/fig5_2_loss.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig5_2_loss.png:",len(loss),"titik")

# ── Gambar 4.3: diagram pipeline fusi CLOCs ─────────────────────────────
fig,ax=plt.subplots(figsize=(8.5,4.6)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,6)
def box(x,y,w,h,txt,fc='#eef2f7',ec=INK,fs=9.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',
                fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=INK)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=INK,lw=1.1))
# sensor
box(0.2,4.4,2.0,0.9,'LiDAR\n(point cloud)',fc='#e8eef5')
box(0.2,0.7,2.0,0.9,'Kamera\n(citra RGB)',fc='#f5efe8')
# detektor
box(3.0,4.4,2.3,0.9,'PointPillars\n(fine-tune KITTI)',fc='#eef2f7')
box(3.0,0.7,2.3,0.9,'YOLO26\n(conf=0)',fc='#eef2f7')
# kandidat
box(6.0,4.4,1.9,0.9,'Kandidat 3D\n(pra-NMS)',fc='#eef7ee')
box(6.0,0.7,1.9,0.9,'Kandidat 2D',fc='#eef7ee')
# fusi
box(6.55,2.35,2.8,0.95,'CLOCs\n(late fusion,\nre-scoring)',fc='#fdf3e7',fs=9)
# output
box(3.2,2.45,2.6,0.8,'3D box final',fc='#eef2f7')
arrow(2.2,4.85,3.0,4.85); arrow(2.2,1.15,3.0,1.15)
arrow(5.3,4.85,6.0,4.85); arrow(5.3,1.15,6.0,1.15)
arrow(7.0,4.4,7.7,3.3); arrow(7.0,1.6,7.7,2.35)
arrow(6.55,2.85,5.8,2.85)
fig.savefig(OUT+'/fig4_3_clocs.png',dpi=200,bbox_inches='tight'); plt.close()
print("fig4_3_clocs.png: OK")
