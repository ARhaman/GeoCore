"""Controlled B8 experiment: DeepLabV3-ResNet50 with rare-mineral focal Tversky loss.

Changes from B7 are deliberately limited to: (1) an ImageNet-pretrained ResNet-50
DeepLabV3 segmentation architecture with ASPP context aggregation, and (2) a focal
Tversky mineral loss. It preserves B7's source-level split, normalized masks,
640x512 evaluation geometry, 448x448 inverse-frequency mineral-centred crops,
class weights, optimizer family, seed and metrics.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageEnhance
from torch.utils.data import Dataset,DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

NAMES=['background','olivine','pyroxene','plagioclase','alkali_feldspar','quartz','biotite','muscovite','hornblende']; C=len(NAMES)
MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1);STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)
def seed(v):
    random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def weights_from(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    vals=data['weights'] if isinstance(data,dict) and isinstance(data.get('weights'),list) else [data['ce_weights'][n] for n in NAMES]
    if len(vals)!=C:raise ValueError(f'Expected {C} weights, found {len(vals)}')
    vals=[float(x) for x in vals]
    if all(abs(x-1)<1e-6 for x in vals):raise ValueError('Refusing all-one weights for rare-mineral experiment.')
    return torch.tensor(vals,dtype=torch.float32)

class DS(Dataset):
    def __init__(self,rows,w,h,train,seedv,sampling_weights,rare_prob):self.rows=rows;self.w=w;self.h=h;self.train=train;self.seed=seedv;self.sw=np.asarray(sampling_weights,dtype=float);self.rare_prob=rare_prob
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image_path']) as im:im=im.convert('RGB')
        with Image.open(r['mask_path']) as ma:ma=ma.convert('L')
        im=im.resize((self.w,self.h),Image.Resampling.BILINEAR);ma=ma.resize((self.w,self.h),Image.Resampling.NEAREST)
        rng=random.Random(self.seed+i)
        if self.train:
            if rng.random()<.5:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);ma=ma.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random()<.35:im=ImageEnhance.Contrast(im).enhance(.75+.5*rng.random())
            cw,ch=min(448,self.w),min(448,self.h);a=np.asarray(ma);present=np.unique(a[(a>0)&(a<C)])
            if len(present) and rng.random()<self.rare_prob:
                cls=int(rng.choices(list(present),weights=list(self.sw[present]),k=1)[0]);loc=np.argwhere(a==cls);yy,xx=loc[rng.randrange(len(loc))]
                x=max(0,min(self.w-cw,int(xx)-cw//2));y=max(0,min(self.h-ch,int(yy)-ch//2))
            else:x=rng.randint(0,self.w-cw);y=rng.randint(0,self.h-ch)
            im=im.crop((x,y,x+cw,y+ch));ma=ma.crop((x,y,x+cw,y+ch))
        x=torch.from_numpy(np.asarray(im,dtype=np.float32).transpose(2,0,1)/255.);x=(x-MEAN.squeeze(0))/STD.squeeze(0);y=torch.from_numpy(np.asarray(ma,dtype=np.int64));return x,y,r['source_id']

def loader(rows,args,train,sw):return DataLoader(DS(rows,args.width,args.height,train,args.seed,sw,args.rare_crop_probability),args.batch_size,shuffle=train,num_workers=0,pin_memory=True)

def build_model():
    try:
        m=deeplabv3_resnet50(weights=None,weights_backbone=ResNet50_Weights.IMAGENET1K_V2,num_classes=C,aux_loss=False);pre=True
    except Exception as exc:
        print(f'Warning: pretrained ResNet-50 unavailable ({exc}); fallback to random backbone.');m=deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=C,aux_loss=False);pre=False
    return m,pre

def focal_tversky(logits,y,alpha=.35,beta=.65,gamma=.75):
    p=logits.softmax(1);oh=F.one_hot(y,C).permute(0,3,1,2).float();terms=[]
    for k in range(1,C):
        pp=p[:,k].flatten(1);tt=oh[:,k].flatten(1);tp=(pp*tt).sum(1);fp=(pp*(1-tt)).sum(1);fn=((1-pp)*tt).sum(1);t=(tp+1)/(tp+alpha*fp+beta*fn+1);terms.append((1-t).pow(gamma))
    return torch.stack(terms).mean()

def loss_fn(out,y,w):return .7*F.cross_entropy(out,y,weight=w)+.3*focal_tversky(out,y)

def metric(model,ld,dev):
    cm=torch.zeros((C,C),dtype=torch.long);model.eval()
    with torch.no_grad():
        for x,y,_ in ld:
            pred=model(x.to(dev))['out'].argmax(1).cpu();cm+=torch.bincount(y.reshape(-1)*C+pred.reshape(-1),minlength=C*C).reshape(C,C)
    iou=[];dice=[]
    for k in range(C):
        tp=cm[k,k].float();un=cm[k].sum()+cm[:,k].sum()-tp;to=cm[k].sum()+cm[:,k].sum();iou.append(float(tp/un) if un else 0.);dice.append(float(2*tp/to) if to else 0.)
    return {'per_class_iou':dict(zip(NAMES,iou)),'per_class_dice':dict(zip(NAMES,dice)),'mean_iou_all':float(np.mean(iou)),'mean_iou_minerals':float(np.mean(iou[1:])),'mean_dice_minerals':float(np.mean(dice[1:])),'confusion_matrix':cm.tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--height',type=int,default=512);p.add_argument('--width',type=int,default=640);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=2026);p.add_argument('--rare-crop-probability',type=float,default=.85);args=p.parse_args();seed(args.seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required.');
    dev=torch.device('cuda');out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with Path(args.manifest).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    group={s:[r for r in rows if r['split']==s] for s in ('train','val','test')};w=weights_from(args.weights).to(dev);sw=w.cpu().tolist();train=loader(group['train'],args,True,sw);val=loader(group['val'],args,False,sw);test=loader(group['test'],args,False,sw)
    model,pre=build_model();model=model.to(dev);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs);scaler=torch.amp.GradScaler('cuda',enabled=True);best=-1.;hist=[]
    for ep in range(1,args.epochs+1):
        model.train();total=0.
        for x,y,_ in train:
            x,y=x.to(dev),y.to(dev);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=True):loss=loss_fn(model(x)['out'],y,w)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=float(loss)
        sched.step();va=metric(model,val,dev);r={'epoch':ep,'loss':total/max(1,len(train)),'val':va};hist.append(r);print(json.dumps(r))
        if va['mean_iou_minerals']>best:best=va['mean_iou_minerals'];torch.save({'model':model.state_dict(),'epoch':ep,'weights':w.tolist(),'pretrained':pre},out/'best.pt')
    ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);model.load_state_dict(ck['model']);final={'device':'cuda','architecture':'deeplabv3_resnet50_aspp','pretrained_encoder':bool(ck['pretrained']),'best_epoch':ck['epoch'],'classes':NAMES,'weights':ck['weights'],'weights_source':str(args.weights),'same_split_as':'B7','same_crop_protocol_as':'B7 inverse-frequency mineral-centred crops','loss':'0.7 class-weighted CE + 0.3 focal Tversky (alpha=0.35,beta=0.65,gamma=0.75)','test':metric(model,test,dev)};(out/'history.json').write_text(json.dumps(hist,indent=2),encoding='utf-8');(out/'final_metrics.json').write_text(json.dumps(final,indent=2),encoding='utf-8');print(json.dumps(final,indent=2))
if __name__=='__main__':main()
