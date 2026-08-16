import pickle, numpy as np
P='OpenPCDet/output/kitti_models/pointpillar_ugm/default/eval/epoch_50/val/ab_manual/result.pkl'
pred=pickle.load(open(P,'rb'))
gt=pickle.load(open('OpenPCDet/data/ugm/kitti_infos_val80_manual.pkl','rb'))
gtmap={i['point_cloud']['lidar_idx']:i for i in gt}

# GT: in-eval = difficulty 0/1/2 (in-FOV, bbox valid); ignored = -1 (luar-FOV bbox 0000)
gt_valid=gt_ignore=0
for i in gt:
    a=i['annos']
    for n,d in zip(a['name'],a['difficulty']):
        if n=='DontCare': continue
        if d in (0,1,2): gt_valid+=1
        else: gt_ignore+=1

# Prediksi: bbox 0000 (di luar image / belakang kamera) = akan jadi FP tak-terhindar
pred_tot=pred_oob=0
for p in pred:
    bb=np.asarray(p['bbox']); sc=np.asarray(p['score'])
    for k in range(len(bb)):
        if sc[k]<0.1: continue
        pred_tot+=1
        if bb[k].sum()==0 or (bb[k,2]-bb[k,0])<1 or (bb[k,3]-bb[k,1])<1:
            pred_oob+=1
print(f'GT: in-eval(diff0/1/2)={gt_valid}  ignored/luar-FOV={gt_ignore}')
print(f'Prediksi (score>=0.1): total={pred_tot}  out-of-image(bbox~0)={pred_oob} ({pred_oob/max(pred_tot,1)*100:.0f}%)')
print('Prediksi out-of-image tak bisa match GT (2D bbox 0) -> dihitung FP -> jatuhkan precision/AP.')
