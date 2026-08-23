"""Run repeated grouped source-level B7–B8 ensemble cross-validation sequentially.

Each fold trains B7 and B8 independently on the exact source-level manifest, calibrates
an ensemble coefficient on the fold validation sources only, then evaluates the fold's
held-out source test data once. Runs resume safely by skipping completed ensemble metrics.
"""
from __future__ import annotations
import argparse,csv,subprocess,sys
from pathlib import Path

def call(cmd,label):
    print(f'\n=== {label} ===\n'+' '.join(f'"{x}"' if ' ' in str(x) else str(x) for x in cmd));subprocess.run([str(x) for x in cmd],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--b7-trainer',required=True);p.add_argument('--b8-trainer',required=True);p.add_argument('--ensemble-evaluator',required=True);p.add_argument('--weights',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    with Path(args.plan).open(newline='',encoding='utf-8') as f:runs=list(csv.DictReader(f))
    for path in (args.b7_trainer,args.b8_trainer,args.ensemble_evaluator,args.weights):
        if not Path(path).exists():raise FileNotFoundError(path)
    print(f'Repeated grouped B7–B8 ensemble CV: {len(runs)} folds, using {sys.executable}')
    for n,r in enumerate(runs,1):
        root=Path(r['output_dir']);b7=root/'b7';b8=root/'b8';ens=root/'ensemble';final=ens/'b7_b8_ensemble_metrics.json';tag=f"repeat {r['repeat']} fold {r['fold']} ({n}/{len(runs)})"
        if final.exists() and not args.overwrite:print(f'Skipping completed {tag}: {final}');continue
        root.mkdir(parents=True,exist_ok=True)
        b7final=b7/'final_metrics.json';b8final=b8/'final_metrics.json'
        if not b7final.exists() or args.overwrite:
            call([sys.executable,args.b7_trainer,'--manifest',r['manifest'],'--weights',args.weights,'--output',b7,'--epochs',args.epochs,'--batch-size',args.batch_size,'--rare-crop-probability','.85'],f'{tag} | B7 training')
        if not b8final.exists() or args.overwrite:
            call([sys.executable,args.b8_trainer,'--manifest',r['manifest'],'--weights',args.weights,'--output',b8,'--epochs',args.epochs,'--batch-size',args.batch_size],f'{tag} | B8 training')
        call([sys.executable,args.ensemble_evaluator,'--manifest',r['manifest'],'--b7-checkpoint',b7/'best.pt','--b8-checkpoint',b8/'best.pt','--output',ens],f'{tag} | validation-calibrated ensemble')
    print('\nAll requested B7–B8 grouped ensemble folds completed. Aggregate the ensemble metrics next.')
if __name__=='__main__':main()
