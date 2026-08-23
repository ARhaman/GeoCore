"""Compute stable class weights from class_balance.json.

Uses median-frequency balancing with a cap to avoid exploding weights for very
rare classes. Background is downweighted separately. The output is intended for
weighted CE + Dice training, not for changing labels.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

NAMES=["background","olivine","pyroxene","plagioclase","alkali_feldspar","quartz","biotite","muscovite","hornblende"]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',default='class_weights.json'); a=p.parse_args()
    data=json.loads(Path(a.input).read_text(encoding='utf-8'))
    counts={str(i):0 for i in range(len(NAMES))}
    for split, vals in data['by_split'].items():
        for k,v in vals.items(): counts[str(k)] += int(v)
    total=sum(counts.values()); freq=np.array([counts[str(i)]/total for i in range(len(NAMES))],dtype=float)
    nonbg=freq[1:]; median=float(np.median(nonbg)); raw=median/np.maximum(nonbg,1e-12)
    # Conservative caps: prevent rare mineral weights from destabilising training.
    weights=np.concatenate([[0.35], np.clip(raw,0.75,4.0)])
    out={'class_names':NAMES,'pixel_counts':{NAMES[i]:counts[str(i)] for i in range(len(NAMES))},'frequencies':{NAMES[i]:float(freq[i]) for i in range(len(NAMES))},'median_frequency_nonbackground':median,'ce_weights':{NAMES[i]:float(weights[i]) for i in range(len(NAMES))},'method':'background=0.35; non-background median-frequency weights clipped to [0.75,4.0]'}
    Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
