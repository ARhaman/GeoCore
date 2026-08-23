"""Aggregate completed repeated grouped B7–B8 ensemble cross-validation results.

Reads each fold's validation-calibrated ensemble metrics, verifies all planned folds are
present, and reports mean, SD, percentile range, and approximate 95% CI across folds.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    with Path(args.plan).open(newline='',encoding='utf-8') as f:plan=list(csv.DictReader(f))
    rows=[];perclass={}
    for run in plan:
        fp=Path(run['output_dir'])/'ensemble'/'b7_b8_ensemble_metrics.json'
        if not fp.exists():raise FileNotFoundError(f'Missing completed ensemble fold: {fp}')
        data=json.loads(fp.read_text(encoding='utf-8'));m=data['test_validation_calibrated_ensemble']
        row={'repeat':int(run['repeat']),'fold':int(run['fold']),'b7_probability_weight':data['selected_b7_probability_weight'],'b8_probability_weight':data['selected_b8_probability_weight'],'mean_iou_all':m['mean_iou_all'],'mean_iou_minerals':m['mean_iou_minerals'],'mean_dice_minerals':m['mean_dice_minerals']}
        rows.append(row)
        for c,v in m['per_class_iou'].items():perclass.setdefault(c,[]).append(float(v))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'ensemble_cv_fold_metrics.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def stat(vals):
        a=np.asarray(vals,float);n=len(a);se=float(a.std(ddof=1)/math.sqrt(n)) if n>1 else 0.;return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if n>1 else 0.,'se':se,'approximate_95_ci':[float(a.mean()-1.96*se),float(a.mean()+1.96*se)],'range_across_folds':[float(a.min()),float(a.max())],'percentile_2_5_97_5':[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]}
    summary={'design':'B7–B8 coefficient selected on validation sources within each outer fold; tested once on that fold\'s held-out sources','fold_estimates':len(rows),'repeats':len(set(r['repeat'] for r in rows)),'folds_per_repeat':len(rows)//len(set(r['repeat'] for r in rows)),'ensemble_weights':{'b7':stat([r['b7_probability_weight'] for r in rows]),'b8':stat([r['b8_probability_weight'] for r in rows])},'metrics':{'mean_iou_all':stat([r['mean_iou_all'] for r in rows]),'mean_iou_minerals':stat([r['mean_iou_minerals'] for r in rows]),'mean_dice_minerals':stat([r['mean_dice_minerals'] for r in rows])},'per_class_iou':{c:stat(v) for c,v in perclass.items()},'fold_metrics_csv':str(out/'ensemble_cv_fold_metrics.csv')}
    (out/'ensemble_repeated_cv_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
