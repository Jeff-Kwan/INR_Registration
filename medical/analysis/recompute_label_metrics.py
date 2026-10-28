"""Recompute MONAI Dice/HD95 from saved warped labels; no model or flow execution."""
import argparse
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from medical.registration.io import LABEL_METRIC_DEFINITION, label_metrics, load_label, write_json
from medical.registration.train import resolve_dataset_path
from medical.registration.configuration import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='New/empty directory')
    args=parser.parse_args()
    if len({p.name for p in args.runs})!=len(args.runs):
        parser.error('Run directory names must be distinct')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output must be empty: {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    result={'metric_definition':LABEL_METRIC_DEFINITION,'training':False,'flow_recomputed':False,'runs':{}}
    for run in args.runs:
        config=load_config(run/'config.json')
        reference=nib.load(str(resolve_dataset_path(config, config['dataset']['fixed'])))
        fixed=load_label(resolve_dataset_path(config, config['dataset']['fixed_label']),reference)
        rows=[]
        for path in sorted(run.glob('S??/warped_label.nii.gz')):
            started=time.perf_counter()
            subject=path.parent.name
            labels=load_label(path,reference)
            scores,per_label=label_metrics(fixed,labels,reference.header.get_zooms()[:3])
            previous=json.loads((path.parent/'metrics.json').read_text())['labels']
            output=args.output/run.name/subject;output.mkdir(parents=True)
            write_json(output/'metrics.json',{'source_run':str(run.resolve()),'subject':subject,
                'historical_labels':previous,'labels':scores})
            write_json(output/'per_label_metrics.json',per_label)
            row={'subject':subject,'historical_dice':previous['dice'],'dice':scores['dice'],
                 'historical_hd95_mm':previous['hd95_mm'],'hd95_mm':scores['hd95_mm'],
                 'missing_labels':scores['missing_labels'],'seconds':time.perf_counter()-started}
            rows.append(row)
            means={key:float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
                   for key in ('historical_dice','dice','historical_hd95_mm','hd95_mm')}
            result['runs'][run.name]={'source':str(run.resolve()),'recovery':config['recovery'],
                                     'completed_subjects':len(rows),'means':means,'subjects':rows}
            write_json(args.output/'comparison.json',result)
            print(run.name,json.dumps(row),flush=True)
        if not rows:
            raise FileNotFoundError(f'No saved Sxx/warped_label.nii.gz in {run}')
    print(json.dumps({k:v['means'] for k,v in result['runs'].items()},indent=2),flush=True)


if __name__=='__main__':
    main()
