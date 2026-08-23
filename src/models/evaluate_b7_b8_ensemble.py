"""Validation-calibrated B7–B8 probability ensemble evaluator.

B7: pretrained ResNet-34 U-Net with inverse-frequency crops.
B8: pretrained DeepLabV3-ResNet50 with focal Tversky loss.
The B7 probability coefficient is selected only on validation sources, then reported once
on the untouched test sources. No training images, masks or source-level splits are changed.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from generate_b7_native_900dpi_figures import PretrainedResUNet,MEAN,STD,NAMES,C

class DS(Dataset):
    def __init__(self,rows,w=640,h=512):self.rows=rows;self.w=w;self.h=h
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image_path']) as im:im=im.convert('RGB').resize((self.w,self.h),Image.Resampling.BILINEAR)
        with Image.open(r['mask_path']) as ma:ma=ma.convert('L').resize((self.w,self.h),Image.Resampling.NEAREST)
        x=torch.from_numpy(np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.);x=(x-MEAN.squeeze(0))/STD.squeeze(0)
        return x,torch.from_numpy(np.asarray(ma,dtype=np.int64)),r['source_id']
def loader(rows):return DataLoader(DS(rows),1,shuffle=False,num_workers=0,pin_memory=True)
def deeplab():return deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=C,aux_loss=False)
def cm_metric(cm):
    iou=[];dice=[]
    for k in range(C):
        tp=cm[k,k].float();u=cm[k].sum()+cm[:,k].sum()-tp;t=cm[k].sum()+cm[:,k].sum();iou.append(float(tp/u) if u else 0.);dice.append(float(2*tp/t) if t else 0.)
    return {'per_class_iou':dict(zip(NAMES,iou)),'per_class_dice':dict(zip(NAMES,dice)),'mean_iou_all':float(np.mean(iou)),'mean_iou_minerals':float(np.mean(iou[1:])),'mean_dice_minerals':float(np.mean(dice[1:])),'confusion_matrix':cm.tolist()}
def evaluate(a,b,ld,dev,weight):
    cm=torch.zeros((C,C),dtype=torch.long);a.eval();b.eval()
    with torch.no_grad():
        for x,y,_ in ld:
            x=x.to(dev);pa=a(x).softmax(1);pb=b(x)['out'].softmax(1);pr=(weight*pa+(1-weight)*pb).argmax(1).cpu();cm+=torch.bincount(y.reshape(-1)*C+pr.reshape(-1),minlength=C*C).reshape(C,C)
    return cm_metric(cm)
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--b7-checkpoint',required=True);p.add_argument('--b8-checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--weights-grid',default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0');args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
    with Path(args.manifest).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    sets={s:[r for r in rows if r['split']==s] for s in ('val','test')};dev=torch.device('cuda');a=PretrainedResUNet().to(dev);b=deeplab().to(dev)
    a.load_state_dict(torch.load(args.b7_checkpoint,map_location=dev,weights_only=False)['model']);b.load_state_dict(torch.load(args.b8_checkpoint,map_location=dev,weights_only=False)['model'])
    grid=[float(x) for x in args.weights_grid.split(',')];vl=loader(sets['val']);tl=loader(sets['test']);scores=[]
    for w in grid:
        m=evaluate(a,b,vl,dev,w);scores.append({'b7_probability_weight':w,'validation':m});print(json.dumps(scores[-1]))
    best=max(scores,key=lambda x:x['validation']['mean_iou_minerals']);tw=best['b7_probability_weight']
    result={'device':'cuda','selection_protocol':'B7 ensemble coefficient selected on validation sources only; reported once on untouched test sources','b7_checkpoint':str(args.b7_checkpoint),'b8_checkpoint':str(args.b8_checkpoint),'validation_grid':scores,'selected_b7_probability_weight':tw,'selected_b8_probability_weight':1-tw,'test_b7':evaluate(a,b,tl,dev,1.0),'test_b8':evaluate(a,b,tl,dev,0.0),'test_validation_calibrated_ensemble':evaluate(a,b,tl,dev,tw)}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'b7_b8_ensemble_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
