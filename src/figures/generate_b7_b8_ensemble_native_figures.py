"""Generate native-resolution 900-DPI B7–B8 ensemble qualitative figures.

B7 and B8 probabilities are mixed with a coefficient selected on the original validation
sources. The script uses native 1280x1024 images and indexed masks, ranks held-out frames
by ensemble mineral mIoU, and exports strong, typical and hard evidence figures.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from generate_b7_native_900dpi_figures import PretrainedResUNet,MEAN,STD,miou,make_group,C

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--b7-checkpoint',required=True);p.add_argument('--b8-checkpoint',required=True);p.add_argument('--b7-weight',type=float,default=.6);p.add_argument('--output',required=True);p.add_argument('--cases-per-group',type=int,default=2);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for native ensemble inference.')
    if not 0<=args.b7_weight<=1:raise ValueError('B7 weight must be in [0,1].')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with Path(args.manifest).open(newline='',encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
    dev=torch.device('cuda');b7=PretrainedResUNet().to(dev);b8=deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=C,aux_loss=False).to(dev)
    b7.load_state_dict(torch.load(args.b7_checkpoint,map_location=dev,weights_only=False)['model']);b8.load_state_dict(torch.load(args.b8_checkpoint,map_location=dev,weights_only=False)['model']);b7.eval();b8.eval();records=[]
    with torch.no_grad():
        for r in rows:
            with Image.open(r['image_path']) as im:image=im.convert('RGB')
            with Image.open(r['mask_path']) as ma:gt=np.asarray(ma.convert('L'),dtype=np.int64)
            x=torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.).unsqueeze(0);x=(x-MEAN)/STD
            x=x.to(dev);pred=(args.b7_weight*b7(x).softmax(1)+(1-args.b7_weight)*b8(x)['out'].softmax(1)).argmax(1).squeeze().cpu().numpy().astype(np.int64)
            records.append({'source_id':r['source_id'],'image_path':r['image_path'],'image':image,'gt':gt,'pred':pred,'miou':miou(gt,pred)})
    records.sort(key=lambda x:x['miou']);n=args.cases_per_group;mid=len(records)//2
    groups=[('strong','B7–B8 ensemble: strong held-out cases','Native 1280 × 1024 frames; B7=0.6 and B8=0.4 selected on validation sources.',(26,126,111),records[-n:]),('typical','B7–B8 ensemble: typical held-out cases','Native 1280 × 1024 frames; good domain recovery with remaining mineral-boundary confusion.',(50,120,168),records[mid-n//2:mid-n//2+n]),('hard','B7–B8 ensemble: difficult held-out cases','Native 1280 × 1024 frames; retained to show failure modes rather than hide them.',(180,92,72),records[:n])]
    summary={'model':'validation-calibrated B7–B8 probability ensemble','b7_weight':args.b7_weight,'b8_weight':1-args.b7_weight,'native_inference_resolution':'1280x1024','dpi':900,'test_frames':len(records),'figures':[]}
    for key,title,sub,accent,items in groups:summary['figures'].append(make_group(items,key,title,sub,accent,out))
    (out/'ensemble_native_900dpi_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
