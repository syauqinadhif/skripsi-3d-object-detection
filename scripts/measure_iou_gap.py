import pickle, numpy as np, torch
from pcdet.ops.iou3d_nms import iou3d_nms_utils
P='OpenPCDet/output/kitti_models/pointpillar_ugm/default/eval/epoch_50/val/ab_manual/result.pkl'
pred=pickle.load(open(P,'rb'))
gt=pickle.load(open('OpenPCDet/data/ugm/kitti_infos_val80_manual.pkl','rb'))
gtmap={i['point_cloud']['lidar_idx']:i for i in gt}

for cls in ['Car','Pedestrian','Cyclist']:
    bev=[]; d3=[]
    for p in pred:
        g=gtmap.get(p['frame_id'])
        if g is None: continue
        pb=np.asarray(p['boxes_lidar'])[np.asarray(p['name'])==cls]
        gann=g['annos']; gb=np.asarray(gann['gt_boxes_lidar'])[np.asarray(gann['name'])==cls]
        if len(pb)==0 or len(gb)==0: continue
        pt=torch.from_numpy(pb[:,:7]).float().cuda()
        gtt=torch.from_numpy(gb[:,:7]).float().cuda()
        iou3d=iou3d_nms_utils.boxes_iou3d_gpu(pt,gtt).cpu().numpy()
        ioubev=iou3d_nms_utils.boxes_iou_bev(pt,gtt).cpu().numpy()
        # untuk tiap prediksi, ambil GT terbaik (by bev) lalu catat bev & 3d di pasangan itu
        for i in range(len(pb)):
            j=int(np.argmax(ioubev[i]))
            if ioubev[i,j]>0.1:
                bev.append(ioubev[i,j]); d3.append(iou3d[i,j])
    if bev:
        bev=np.array(bev); d3=np.array(d3)
        print(f'{cls:11}: n={len(bev):3d} | BEV IoU med={np.median(bev):.3f} 3D IoU med={np.median(d3):.3f} '
              f'| pass BEV>=0.7:{(bev>=0.7).mean()*100:4.1f}% 3D>=0.7:{(d3>=0.7).mean()*100:4.1f}% 3D>=0.5:{(d3>=0.5).mean()*100:4.1f}%')
print('Kalau 3D IoU med << BEV IoU med -> vertikal (z/tinggi) yg jatuhkan. Kalau mirip -> XY/BEV sendiri yg lemah.')
