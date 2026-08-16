import pickle, numpy as np
P='OpenPCDet/output/kitti_models/pointpillar_ugm/default/eval/epoch_50/val/ab_manual/result.pkl'
pred=pickle.load(open(P,'rb'))
gt=pickle.load(open('OpenPCDet/data/ugm/kitti_infos_val80_manual.pkl','rb'))
gtmap={i['point_cloud']['lidar_idx']:i for i in gt}
print('pred[0] keys:', list(pred[0].keys()))

def match(pb, gb):
    res=[]; used=set()
    for i in range(len(pb)):
        if len(gb)==0: break
        d=np.hypot(gb[:,0]-pb[i,0], gb[:,1]-pb[i,1])
        j=int(np.argmin(d))
        if d[j]<2.0 and j not in used:
            used.add(j); res.append((pb[i],gb[j]))
    return res

for cls in ['Car','Pedestrian','Cyclist']:
    dz=[]; dh=[]
    for p in pred:
        fid=p['frame_id']; g=gtmap.get(fid)
        if g is None: continue
        pb=np.asarray(p['boxes_lidar'])[np.asarray(p['name'])==cls]
        gann=g['annos']; gb=np.asarray(gann['gt_boxes_lidar'])[np.asarray(gann['name'])==cls]
        for pbi,gbi in match(pb,gb):
            dz.append(pbi[2]-gbi[2]); dh.append(pbi[5]-gbi[5])
    if dz:
        dz=np.array(dz); dh=np.array(dh)
        print(f'{cls:11}: matched={len(dz):3d} | dz-center med={np.median(dz):+.3f} mean={np.mean(dz):+.3f} std={np.std(dz):.3f} | d-tinggi med={np.median(dh):+.3f}')
    else:
        print(f'{cls:11}: 0 match')
print('dz>0 = prediksi model DI ATAS gt (ngambang). std kecil = bias sistematis -> bisa dikoreksi.')
